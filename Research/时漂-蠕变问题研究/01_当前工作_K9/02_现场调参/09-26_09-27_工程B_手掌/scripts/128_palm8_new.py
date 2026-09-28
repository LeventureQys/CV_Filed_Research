# -*- coding: utf-8 -*-
"""128_palm8_new：新会话 20260927_161747（ADC 显示、算法关、单流 ~495 s）预处理+定档。

单流 = 算法输入（录制时算法关，显示=输入）。台阶检测 + E 口径 + 候选参数复算。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

s90 = import_module("90_palm4_prep")
obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT = TEMP / "palm8" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SESS = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\手掌数据\20260927_161747_single_device_b2d896")

CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
       "slope_cap_frac": 0.025, "r_slow_max": 0.03, "tau_r_fast_s": 6.0,
       "tau_r_slow_idle_s": 0.5}
REC = {**CUR, "r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0,
       "slope_cap_frac": 0.005, "r_slow_max": 0.15}
REC6 = {**REC, "r_slow_max": 0.6}
AGGR = {**CUR, "r_fast": 0.08, "tau_c_fast_s": 2.0, "slow_confirm_s": 0.0,
        "soft_unfreeze_s": 0.1, "slope_cap_frac": 0.05, "r_slow_max": 0.6}


def steps(t, tin):
    k = 30
    d = np.abs(tin[k:] - tin[:-k])
    hot = np.flatnonzero(d > 400)
    if not hot.size:
        return []
    groups = [[int(hot[0])]]
    for i in hot[1:]:
        if i - groups[-1][-1] <= 60:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    st = []
    for g in groups:
        i = g[int(np.argmax(d[g]))]
        st.append(i)
    return st


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, V = s90.load_csv(SESS / "device_001_seg000.csv")
    t = t - t[0]
    fps = (len(t) - 1) / (t[-1] - t[0])
    print(f"帧数={len(t)} 通道数={V.shape[1]} 时长={t[-1]:.1f}s fps={fps:.1f}")
    np.savez(OUT / "streams.npz", t=t, V=V, tot=V.sum(axis=1))
    tin = V.sum(axis=1)
    print(f"总值：首={tin[0]:.0f} 峰={tin.max():.0f}@{t[np.argmax(tin)]:.1f}s 末={tin[-1]:.0f}")
    st = steps(t, tin)
    print(f"\n检测到 {len(st)} 个台阶：")
    segs = []
    for j, i in enumerate(st):
        t_s = t[i]
        m = (t >= t_s + 0.1) & (t <= t_s + 0.6)
        E = float(tin[m].min())
        nxt = t[st[j + 1]] if j + 1 < len(st) else t[-1]
        segs.append({"i": i, "t0": float(t_s), "E": E, "t1": float(nxt)})
        print(f"  台阶{j+1} @{t_s:7.2f}s E≈{E:8.0f}  段长={nxt-t_s:6.1f}s")
    (OUT / "segs.json").write_text(json.dumps(segs), encoding="utf-8")

    cands = [("cur", CUR, "现参数(rf.04 τc1=1 conf1.5 soft1.5 cap.025 rsm.03)"),
             ("rec", REC, "推荐(rf.06 τc1=2 conf2 soft1.5 cap.005 rsm.15)"),
             ("rec6", REC6, "推荐+rsm.6"),
             ("aggr", AGGR, "慢漂优先(rf.08 conf0 soft.1 cap.05 rsm.6)")]
    for name, p, label in cands:
        r = obs.run(t, V, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"], fps)
        print(f"\n== {label}")
        for j, sg in enumerate(segs):
            i0 = int(np.searchsorted(t, sg["t0"] + 0.6))
            i1 = int(np.searchsorted(t, min(sg["t1"] - 0.6, t[-1])))
            if i1 - i0 < 100:
                continue
            yy, tt = y[i0:i1 + 1], t[i0:i1 + 1]
            E = sg["E"]
            # 全卸载检测：段末输入回到近基线
            base = tin[: int(np.searchsorted(t, segs[0]["t0"]))].mean() if segs else tin[0]
            unloaded = tin[i1] < base + 0.25 * E
            end = float(yy[-1])
            mm = tt >= tt[-1] - 5.0
            slope = float(np.polyfit(tt[mm], yy[mm], 1)[0]) if mm.sum() > 10 else float("nan")
            print(f"  段{j+1}({sg['t0']:.0f}~{sg['t1']:.0f}s E={E:.0f}{', 卸载' if unloaded else ''}): "
                  f"落点={end-E:+7.0f}({(end-E)/1000:+.2f}N) 下坠={yy.max()-end:5.0f} "
                  f"过减={max(0.0, E-yy.min()):5.0f} 末斜率={slope:+6.1f}ADC/s "
                  f"x2末={r['x2'][i1]:.0f}")
        np.savez(OUT / f"run_{name}.npz", out_tot=r["out_tot"], x1=r["x1"], x2=r["x2"],
                 applied=r["applied"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
