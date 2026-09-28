# -*- coding: utf-8 -*-
"""31 结构新增原型验证：锚定基线 + 单调包络跟随（ABDR）。

用户诉求（2026-09-26 追加）
------------------------
* 「尽量在 1~2 s 建立基线，然后就能在一个范围内上下波动，而且**不损失实际的基线波动**」；
* 现有调参「总是让基线值下降向下漂」，不符合传感器原理预期（恒载读数不应持续下降）；
* 但纯欠扣的版本又会「慢慢上漂」，同样不接受。
* 允许**新增**算法结构，但不得改动原有结构；原有参数必须仍能良好适配四指指腹那种**有加载台阶**的形状。

新机制（纯新增，关闭时与现役逐帧完全一致）
--------------------------------------
逐通道维护：
    anchor_     锚定基线：算法启动后 [anchor_t0_s, anchor_t1_s] 窗口内显示值的中位数
    anchor_hi_  锚定之后显示值的**历史最高**（带极慢向当前值松弛，用于很久之后跟随真实载荷变化）
输出叠加：
    extra = clamp(anchor_hi_ − anchor_, 0, max(显示, 0))     # 相对锚定基线的"单向上漂"量
    显示 −= anchor_gain · extra
性质：
    ① 恒载下读数上漂 ⇒ anchor_hi_ 跟着涨 ⇒ extra 全扣掉 ⇒ 显示**钉在 anchor_**，不上漂；
    ② 读数向下波动 ⇒ anchor_hi_ 不动 ⇒ 显示跟着下探，**真实波动原样保留**；
    ③ 扣减量单调不减 ⇒ 显示**绝不会被额外压下去**（不存在向下的伪漂移）；
    ④ 锚定窗口内直通 ⇒ 1~2 s 内完成基线建立。

用法：python 31_anchor.py [传感器...]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402
from sensor_common import LIVE, OUT_DIR, SENSORS, load  # noqa: E402
from sweep_lib import PREP  # noqa: E402

NEW_PARAMS = {
    "anchor_t0_s": 1.0,          # 锚定窗口起点（相对算法启动/调零时刻）
    "anchor_t1_s": 2.0,          # 锚定窗口终点；<= t0 ⇒ 本机制关闭
    "anchor_relax_tau_s": 600.0,  # 锚定上限向当前值慢松弛的 τ（<=0 关闭）；用于超长录制后跟随真实载荷
    "anchor_gain": 1.0,          # 混合系数（0~1）
}


@dataclass
class ParamsA(Params):
    anchor_t0_s: float = 1.0
    anchor_t1_s: float = 2.0
    anchor_relax_tau_s: float = 600.0
    anchor_gain: float = 1.0


class AnchorObserver:
    """现役 K9 观测器（前 27 参数逐帧不变）+ 末尾追加锚定基线漂移抑制。"""

    def __init__(self, p: ParamsA):
        self.p = p
        self._core = CreepObserverK9(replace(Params(**{f.name: getattr(p, f.name)
                                                       for f in fields(Params)})))
        self._core._trace_frame = lambda *a, **k: None
        self._t0 = None
        self._n = 0
        self._anchored = False
        self._samples: list[np.ndarray] = []
        self._reset_anchor()

    def _reset_anchor(self):
        self._anchored = False
        self._samples = []
        self._last = None

    def process(self, ts: float, v: np.ndarray) -> np.ndarray:
        p = self.p
        if self._t0 is None:
            self._t0 = float(ts)
        out = self._core.process(float(ts), v)
        n = out.size
        if n != self._n:
            self._n = n
            self.anchor = np.zeros(n)
            self.hi = np.zeros(n)
            self._reset_anchor()
        dt = 0.0 if self._last is None else min(max(float(ts) - self._last, 0.0), 0.1)
        self._last = float(ts)
        if p.anchor_t1_s <= p.anchor_t0_s:          # 关闭 ⇒ 输出与现役完全一致
            return out
        rel = float(ts) - self._t0
        if rel < p.anchor_t0_s:
            return out
        if rel <= p.anchor_t1_s:
            self._samples.append(np.array(out, copy=True))
            return out
        if not self._anchored:
            self._anchored = True
            self.anchor = (np.median(np.vstack(self._samples), axis=0)
                           if self._samples else np.array(out, copy=True))
            self.hi = np.array(out, copy=True)
            self._samples = []
        if dt > 0 and p.anchor_relax_tau_s > 0:
            low = self.hi > out
            self.hi[low] += (dt / p.anchor_relax_tau_s) * (out[low] - self.hi[low])
        self.hi = np.maximum(self.hi, out)
        extra = np.clip(self.hi - self.anchor, 0.0, np.maximum(out, 0.0))
        return out - p.anchor_gain * extra

    def __init_hist__(self):
        pass


def run(p: ParamsA, t: np.ndarray, V: np.ndarray) -> np.ndarray:
    o = AnchorObserver(p)
    return np.array([o.process(float(t[i]), V[i]).sum() for i in range(len(t))])


def metrics(t: np.ndarray, out: np.ndarray, tin: np.ndarray, t0: float,
            t_end: float) -> dict:
    """绝对 ADC 口径 + 波动保真度。"""
    w = t >= t0 + 3.0
    anchor = float(np.median(out[(t >= t0 + 1.0) & (t <= t0 + 2.0)]))
    hold = t >= t0 + 30.0
    # 慢分量（20 s 滑动中值）与快分量：用来衡量"真实波动有没有被吃掉"
    def slow(x):
        k = 1000                       # ±10 s @100Hz
        y = np.empty_like(x)
        for i in range(x.size):
            a, b = max(0, i - k), min(x.size, i + k + 1)
            y[i] = np.median(x[a:b])
        return y
    so, si = slow(out), slow(tin)
    fo, fi = out - so, tin - si
    m = hold & (t >= t0 + 30.0)
    fo_h, fi_h = fo[m], fi[m]
    sd_ratio = float(np.std(fo_h) / max(np.std(fi_h), 1e-9)) if fi_h.size > 10 else float("nan")
    corr = float(np.corrcoef(fo_h, fi_h)[0, 1]) if fi_h.size > 10 and np.std(fi_h) > 1e-9 \
        else float("nan")
    # 建立时间：显示进入 最终锚定带 ±15 ADC 并保持的时刻
    band = np.abs(out - anchor) <= 15.0
    settle = float("nan")
    for i in range(out.size):
        if t[i] >= t0 and band[i:].all():
            settle = float(t[i] - t0)
            break
    tail = t >= t[-1] - 30.0
    slope = float(np.polyfit(t[tail], out[tail], 1)[0]) if tail.sum() > 5 else 0.0
    return {
        "anchor": anchor,
        "settle_s": settle,
        "end_exc": float(out[-1] - anchor),
        "low_exc": float(max(0.0, anchor - out[w].min())),
        "peak_exc": float(max(0.0, out[w].max() - anchor)),
        "notch": float(np.max(np.maximum.accumulate(out[w]) - out[w])),
        "tail_slope": slope,
        "std_ratio": sd_ratio,
        "corr": corr,
        "ded_pct": float((tin[-1] - out[-1]) / max(tin[-1] - tin[0], 1e-9) * 100),
    }


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    only = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SENSORS)
    base = ParamsA(**{f.name: getattr(LIVE, f.name) for f in fields(Params)})
    VARIANTS = [
        ("现役默认", base, {}),
        ("本轮预设", None, {}),                      # 从 09_presets.json 取
        ("锚定档(观测器全关)", base, {"r_fast": 0.0, "slope_gate_frac": 0.0,
                                     "r_slow_max": 0.001, "slope_cap_frac": 0.0}),
        ("锚定档(保留x1)", base, {"r_fast": 0.06, "slope_gate_frac": 0.0,
                                  "r_slow_max": 0.001, "slope_cap_frac": 0.0}),
    ]
    presets = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))
    rows_all = {}
    print(f"{'传感器':<12s} {'方案':<20s} {'会话':<10s} {'建立s':>6s} {'末偏':>7s} "
          f"{'下探':>7s} {'上探':>7s} {'末30s斜率':>9s} {'波动比':>7s} {'相关':>6s} {'扣除%':>6s}")
    for sensor in only:
        for nm, pbase, over in VARIANTS:
            if nm == "本轮预设":
                p = ParamsA(**{**{f.name: getattr(base, f.name) for f in fields(Params)},
                               **presets[sensor]["params"]})
            else:
                p = replace(pbase, **over)
            for tag in SENSORS[sensor]["sessions"]:
                info = PREP[f"{sensor}/{tag}"]
                d = load(info["path"].split("data\\", 1)[-1].replace("\\", "/"))
                t = d["t"] - d["t"][0]
                V = d["V"] - d["V"][0]
                if sensor == "四指指腹":
                    # 四指指腹必须用调零口径（首帧即零点）——这里 V 已按首帧归零
                    pass
                out = run(p, t, V)
                m = metrics(t, out, V.sum(axis=1), info["t0"], t[-1])
                print(f"{sensor:<12s} {nm:<20s} {tag:<10s} {m['settle_s']:6.1f} "
                      f"{m['end_exc']:7.1f} {m['low_exc']:7.1f} {m['peak_exc']:7.1f} "
                      f"{m['tail_slope']:9.3f} {m['std_ratio']:7.2f} {m['corr']:6.2f} "
                      f"{m['ded_pct']:6.1f}")
                rows_all.setdefault(f"{sensor}|{nm}", {})[tag] = m
        print()
    (OUT_DIR / "20_anchor_proto.json").write_text(
        json.dumps(rows_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {OUT_DIR / '20_anchor_proto.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
