# -*- coding: utf-8 -*-
"""103_palm6：20260927_152928（ADC 显示、主机算法开、pre/seg 双流）分析。

问题：慢态"过调"了为何显示仍不停上浮？
分解：输入蠕变 vs x1/x2/applied 的时间轨迹；检查 x2 是否被 r_slow_max 饱和。
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
OUT = TEMP / "palm6" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SESS = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\手掌数据\20260927_152928_single_device_3efae9")
P = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
     "slope_cap_frac": 0.025, "r_slow_max": 0.03, "tau_r_fast_s": 6.0,
     "tau_r_slow_idle_s": 0.5}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, pre = s90.load_csv(SESS / "device_001_pre_seg0.csv")
    t2, seg = s90.load_csv(SESS / "device_001_seg000.csv")
    t = t - t[0]
    np.savez(OUT / "streams.npz", t=t, pre=pre, seg=seg,
             tot_pre=pre.sum(axis=1), tot_seg=seg.sum(axis=1))
    r = obs.run(t, pre, obs.default_with(P))
    mine, dev = r["out_tot"], r["out_tot"] - seg.sum(axis=1)
    print(f"parity：均差={np.abs(dev).mean():.1f} 最大={np.abs(dev).max():.1f} ADC")
    tin = pre.sum(axis=1)
    # 台阶
    kk = 30
    d = np.abs(tin[kk:] - tin[:-kk])
    hot = np.flatnonzero(d > 400)
    groups = [[int(hot[0])]]
    for i in hot[1:]:
        if i - groups[-1][-1] <= 60:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    for g in groups:
        i = g[int(np.argmax(d[g]))]
        E = float(tin[(t >= t[i] + 0.1) & (t <= t[i] + 0.6)].min())
        print(f"台阶@{t[i]:6.2f}s Δ={tin[min(i+kk,len(tin)-1)]-tin[i]:+8.0f} E≈{E:.0f}")
    print(f"\n{'t':>6s} {'输入':>9s} {'显示(录)':>9s} {'复算显':>9s} {'x1':>8s} {'x2':>8s} {'applied':>8s} {'E基':>7s}")
    i0 = groups[0][int(np.argmax(d[groups[0]]))]
    E1 = float(tin[(t >= t[i0] + 0.1) & (t <= t[i0] + 0.6)].min())
    for tt in np.arange(2.0, 41.0, 3.0):
        i = int(np.searchsorted(t, tt))
        if i >= len(t):
            break
        print(f"{t[i]:6.2f} {tin[i]:9.1f} {seg.sum(axis=1)[i]:9.1f} {mine[i]:9.1f} "
              f"{r['x1'][i]:8.1f} {r['x2'][i]:8.1f} {r['applied'][i]:8.1f} "
              f"{tin[i]-E1:+7.1f}")
    # x2 饱和检查：x2 总值 vs rsm·e 总值上界
    e_est = (tin - r["x1"] - r["x2"]) * 0.97  # 粗略
    hi_total = 0.03 * np.maximum(tin - r["x1"], 0)
    sat = r["x2"] / np.maximum(hi_total, 1e-9)
    m = t > 5
    print(f"\nx2 / (rsm·e) 比：t>5s 均值={sat[m].mean():.3f} p95={np.percentile(sat[m],95):.3f} "
          f"max={sat[m].max():.3f}（≈1 即饱和）")
    print(f"末段：输入−E={tin[-1]-E1:+.1f}  x1末={r['x1'][-1]:.1f}  x2末={r['x2'][-1]:.1f}  "
          f"applied末={r['applied'][-1]:.1f}")
    sm = s93.medfilt1s(seg.sum(axis=1), 100.6)
    m2 = t > 5
    mm = t[m2] >= t[m2][-1] - 5.0
    print(f"显示末5s斜率={np.polyfit(t[m2][mm], sm[m2][mm], 1)[0]:+.1f} ADC/s "
          f"（={np.polyfit(t[m2][mm], sm[m2][mm], 1)[0]/1000:+.4f} N/s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
