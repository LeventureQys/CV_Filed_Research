# -*- coding: utf-8 -*-
"""128_force_cases：力值模式测试台（真实会话 + 合成）。

cases：
  hold_1d925c  : 真实力值会话（算法开、pre 流为输入、两段保压，+50%/+45% 爬升）
  in_03225d    : 真实力值会话（算法关 → seg 即输入，35.8 s）
  in_102924    : 真实力值会话（算法关 → seg 即输入，15.5 s）
  synth_flat17 : 合成恒压 17 N、零蠕变 60 s（量「纯电平扣除」的失真代价）
  synth_creep10: 合成 17 N + 10% 慢蠕变（低保真蠕变工况）
  synth_tap2s  : 合成 17 N 保持 2 s 后卸载（量短按被扣多少）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

fl = import_module("122_force_lib")
OUT = TEMP / "palm7" / "out"
M = 71
FPS = 100.0


def ramp(Vout: np.ndarray, t0: float, t1: float, level: float) -> None:
    """把 Vout 在 [t0,t1] 内线性升到 level（逐通道均分）。"""
    i0, i1 = int(t0 * FPS), int(t1 * FPS)
    Vout[i0:i1] = np.linspace(0.0, 1.0, i1 - i0)[:, None] * (level / M)


def synth(kind: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """合成台：加载斜坡 0.5→1.1 s，1.6 s 起为平台；creep 类按占 E 的比例匀速上爬。"""
    dt = 1.0 / FPS
    if kind == "synth_tap2s":
        dur, hold, level = 8.0, 2.0, 17.0
    else:
        dur, hold, level = 60.0, 57.0, 17.0
    t = np.arange(0, dur, dt)
    V = np.zeros((len(t), M))
    ramp(V, 0.5, 1.1, level)
    i1 = int(1.6 * FPS)
    V[i1:] = level / M        # 弹性电平 = 17 N（理想阶跃后立即稳定）
    if kind == "synth_creep10":
        # 蠕变总量 = 10%·E，均摊在 hold 秒内（逐通道同速率）
        rates = np.zeros(len(t))
        rates[i1:] = 0.10 * level / hold * dt      # 每帧增量（N）
        V += np.cumsum(rates)[:, None] / M
    segs = [{"t0": 0.5, "t1": 0.5 + hold + 0.6, "E": level, "E_fast": level,
             "i0": int(1.6 * FPS), "i1": min(len(t) - 1, int((0.5 + hold) * FPS))}]
    return t, V, segs


def load_force_input(path: Path, sess: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    lines = (sess / "device_001_seg000.csv").read_text(encoding="utf-8-sig").splitlines()
    di = next(i for i, ln in enumerate(lines) if ln.startswith("##Data"))
    rows = [ln.split(",") for ln in lines[di + 2:] if ln.strip()]
    a = np.array([[float(x) for x in r] for r in rows], dtype=np.float64)
    t = a[:, 1] - a[0, 1]
    V = a[:, 3:]
    tin = V.sum(1)
    segs = fl.segments(t, tin)
    np.savez(path, t=t, V=V, tot=tin)
    return t, V, segs


def build() -> list[dict]:
    cases = []
    z = np.load(OUT / "force_1d925c.npz")
    t, pre = z["t"], z["pre"]
    cases.append({"name": "hold_1d925c", "t": t, "V": pre,
                  "segs": fl.segments(t, pre.sum(1)),
                  "note": "真实·算法开·两段保压（+50%/+45%）"})
    for key, name in ((r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\手掌数据"
                       r"\20260927_111205_single_device_03225d", "in_03225d"),
                      (r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\archived\手掌数据"
                       r"\20260927_102924_single_device_ebfaca", "in_102924")):
        p = OUT / f"{name}.npz"
        sess = Path(key)
        if p.is_file():
            zz = np.load(p)
            t2, V2 = zz["t"], zz["V"]
        else:
            t2, V2, _ = load_force_input(p, sess)
        cases.append({"name": name, "t": t2, "V": V2,
                      "segs": fl.segments(t2, V2.sum(1)),
                      "note": "真实·算法关（seg 即输入）"})
    for k in ("synth_flat17", "synth_creep10", "synth_tap2s"):
        t3, V3, sg = synth(k)
        cases.append({"name": k, "t": t3, "V": V3, "segs": sg,
                      "note": "合成"})
    return cases


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    for c in build():
        print(f"\n[{c['name']}] {c['note']}  n={len(c['t'])} 时长={c['t'][-1]:.1f}s "
              f"通道={c['V'].shape[1]}")
        for j, s in enumerate(c["segs"]):
            tin = c["V"].sum(1)
            a = int(np.searchsorted(c["t"], s["t0"]))
            pre_lv = float(tin[max(0, a - 30):a].mean()) if a > 0 else 0.0
            print(f"  段{j+1} {s['t0']:.2f}→{s['t1']:.2f}s E={s['E']:8.3f} 台阶前={pre_lv:7.3f} "
                  f"段末输入={float(tin[s['i1']]):8.3f} 爬升={float(tin[s['i1']])-s['E']:+7.3f}")
