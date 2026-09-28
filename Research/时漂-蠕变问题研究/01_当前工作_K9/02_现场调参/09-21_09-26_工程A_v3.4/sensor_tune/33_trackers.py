# -*- coding: utf-8 -*-
"""33 参考跟踪器选型（优化版）：对比「平滑峰值保持包络」与「一阶低通」两种慢漂移参考。

统一输出式：显示 = anchor + (输出 − 参考)，anchor = [1,2] s 内输出中值。
评价：建立时间 / 末值偏差 / 上探 / 下探 / 末 30 s 斜率 / 波动幅度比 / 波动相关。
（滑动中值参考因 O(n²) 已剔除；峰值保持用快攻慢放包络实现，抗噪靠 0.5 s 中值预滤波。）
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402
from sensor_common import LIVE, OUT_DIR, SENSORS, load  # noqa: E402
from sweep_lib import PREP  # noqa: E402


class Core:
    """现役 K9 核心 + 新增的「锚定 + 慢漂移参考」层（mode='off' 时逐帧与现役一致）。"""

    def __init__(self, p: Params, mode: str, *, w_smooth=0.5, tau_up=0.5, tau_down=600.0,
                 tau_lp=20.0, t_anchor=(1.0, 2.0)):
        self.p = p
        self.mode = mode
        self.core = CreepObserverK9(p)
        self.core._trace_frame = lambda *a, **k: None
        self.k = max(1, int(w_smooth * 100))
        self.tau_up, self.tau_down, self.tau_lp = tau_up, tau_down, tau_lp
        self.t0a, self.t1a = t_anchor
        self.t_start = None
        self.hist: list[np.ndarray] = []
        self.buf: list[np.ndarray] = []
        self.anchor = self.ref = None
        self.anchored = False
        self._ref_init = None
        self._last = None

    def process(self, ts: float, v: np.ndarray) -> np.ndarray:
        if self.t_start is None:
            self.t_start = float(ts)
        out = self.core.process(float(ts), v)
        n = out.size
        if self.anchor is None or self.anchor.size != n:
            self.anchor = np.array(out, copy=True)
            self.ref = np.array(out, copy=True)
            self.anchored, self.hist, self.buf, self._last = False, [], [], None
        if self.mode == "off":
            return out
        dt = 0.0 if self._last is None else min(max(float(ts) - self._last, 0.0), 0.1)
        self._last = float(ts)
        rel = float(ts) - self.t_start
        self.buf.append(np.array(out, copy=True))
        if len(self.buf) > 2 * self.k + 1:
            self.buf.pop(0)
        x = np.median(np.vstack(self.buf), axis=0) if len(self.buf) > 1 else out
        if rel < self.t0a:
            return out
        if self.mode == "peak_early":
            # 锚定值 = [t0a, t1a] 的滚动中值（t1a 之后冻结）；参考包络从 t0a 就开始跟随
            if not self.anchored:
                self.hist.append(np.array(out, copy=True))
                self.anchor = np.median(np.vstack(self.hist), axis=0)
                if self._ref_init is None:
                    self.ref = np.array(x, copy=True)
                    self._ref_init = True
                if rel > self.t1a:
                    self.anchored = True
                    self.hist = []
            if dt > 0:
                up = x > self.ref
                self.ref[up] += (dt / self.tau_up) * (x[up] - self.ref[up])
                dn = ~up
                self.ref[dn] += (dt / self.tau_down) * (x[dn] - self.ref[dn])
            return np.maximum(self.anchor + (out - self.ref), 0.0)
        if rel <= self.t1a:
            self.hist.append(np.array(out, copy=True))
            return out
        if not self.anchored:
            self.anchored = True
            self.anchor = (np.median(np.vstack(self.hist), axis=0)
                           if self.hist else np.array(out, copy=True))
            self.ref = np.array(x, copy=True)
            self.hist = []
        if dt > 0:
            if self.mode == "peak":
                up = x > self.ref
                self.ref[up] += (dt / self.tau_up) * (x[up] - self.ref[up])
                dn = ~up
                self.ref[dn] += (dt / self.tau_down) * (x[dn] - self.ref[dn])
            else:
                self.ref += (dt / self.tau_lp) * (x - self.ref)
        return np.maximum(self.anchor + (out - self.ref), 0.0)


def run(p: Params, t, V, mode: str, **kw):
    """返回 (逐帧总值, 该实例最终使用的锚定基线总值)。"""
    c = Core(p, mode, **kw)
    y = np.array([c.process(float(t[i]), V[i]).sum() for i in range(len(t))])
    anc = float(np.sum(c.anchor)) if c.anchor is not None else float("nan")
    return y, anc


def slow_dec(x: np.ndarray, ds: int = 50, k: int = 20) -> np.ndarray:
    """20 s 滑动中值的快速近似：先抽稀到 ds 分之一，再中值，再线性插值回原长。"""
    n = x.size
    idx = np.arange(0, n, ds)
    y = x[idx]
    m = np.empty_like(y)
    for i in range(y.size):
        a, b = max(0, i - k), min(y.size, i + k + 1)
        m[i] = np.median(y[a:b])
    return np.interp(np.arange(n), idx, m)


def metrics(t, out, tin, t0, anchor: float) -> dict:
    w = t >= t0 + 3.0
    # 建立时间：最后一次显示高出锚定值 150 ADC 之后的时刻（即"上漂停止"）
    above = np.nonzero((t >= t0) & (out > anchor + 150.0))[0]
    settle = float(t[above[-1]] - t0) if above.size else 0.0
    tail = t >= t[-1] - 30.0
    slope = float(np.polyfit(t[tail], out[tail], 1)[0]) if tail.sum() > 5 else 0.0
    hold = t >= t0 + 30.0
    fo, fi = out - slow_dec(out), tin - slow_dec(tin)
    a, b = fo[hold], fi[hold]
    ratio = float(np.std(a) / max(np.std(b), 1e-9)) if b.size > 10 else float("nan")
    corr = float(np.corrcoef(a, b)[0, 1]) if b.size > 10 and np.std(b) > 1e-9 else float("nan")
    return {"anchor": anchor, "settle_s": settle, "end_exc": float(out[-1] - anchor),
            "low_exc": float(max(0.0, anchor - out[w].min())),
            "peak_exc": float(max(0.0, out[w].max() - anchor)),
            "tail_slope": slope, "std_ratio": ratio, "corr": corr}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    base = Params()
    P_OFF = Params(**{**{f.name: getattr(base, f.name) for f in fields(Params)},
                      "r_fast": 0.0, "slope_gate_frac": 0.0, "slope_cap_frac": 0.0,
                      "r_slow_max": 0.001})
    MODES = [
        ("关闭(=现役核心)", "off", {}),
        ("起锚 τup0.3 w0.5", "peak_early", {"tau_up": 0.3}),
        ("起锚 τup1.0 w0.5", "peak_early", {"tau_up": 1.0}),
        ("起锚 τup2.0 w0.5", "peak_early", {"tau_up": 2.0}),
        ("起锚 τup5.0 w0.5", "peak_early", {"tau_up": 5.0}),
        ("起锚 τup10 w0.5", "peak_early", {"tau_up": 10.0}),
        ("起锚 τup2.0 w0.2", "peak_early", {"tau_up": 2.0, "w_smooth": 0.2}),
        ("起锚 τup5.0 w0.2", "peak_early", {"tau_up": 5.0, "w_smooth": 0.2}),
    ]
    out_all = {}
    for sensor in (sys.argv[1:] or ["左拇指指腹", "右手掌"]):
        print(f"\n##### {sensor} #####")
        print(f"  {'方案':<18s} {'会话':<10s} {'建立s':>6s} {'末偏':>7s} {'下探':>7s} "
              f"{'上探':>7s} {'末30s斜':>8s} {'波动比':>7s} {'相关':>6s}")
        for nm, mode, kw in MODES:
            for tag in SENSORS[sensor]["sessions"]:
                info = PREP[f"{sensor}/{tag}"]
                d = load(info["path"].split("data\\", 1)[-1].replace("\\", "/"))
                t = d["t"] - d["t"][0]
                V = d["V"] - d["V"][0]
                y, anc_cls = run(P_OFF, t, V, mode, **kw)
                if mode == "off":
                    anc = float(np.median(y[(t >= info["t0"] + 1.0)
                                            & (t <= info["t0"] + 2.0)]))
                else:
                    anc = anc_cls
                o = y
                m = metrics(t, o, V.sum(axis=1), info["t0"], anc)
                print(f"  {nm:<18s} {tag:<10s} {m['settle_s']:6.1f} {m['end_exc']:7.1f} "
                      f"{m['low_exc']:7.1f} {m['peak_exc']:7.1f} {m['tail_slope']:8.3f} "
                      f"{m['std_ratio']:7.2f} {m['corr']:6.2f}", flush=True)
                out_all.setdefault(f"{sensor}|{nm}", {})[tag] = m
        print()
    (OUT_DIR / "21_tracker.json").write_text(json.dumps(out_all, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"写出 {OUT_DIR / '21_tracker.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
