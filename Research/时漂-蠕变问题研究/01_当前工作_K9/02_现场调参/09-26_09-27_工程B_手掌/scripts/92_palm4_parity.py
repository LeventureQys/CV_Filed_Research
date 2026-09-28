# -*- coding: utf-8 -*-
"""92_palm4_parity：Python 移植复算 vs 录制 seg 流的 parity + 过减定因。

录制参数（session.json）：rf=0.3, τc1=40, conf=1, soft=2, cap=0.006, rsm=0.05,
τr1=6, τrsi=0.5（其余=K9 默认）。
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

obs = import_module("91_observer")
OUT = TEMP / "palm4" / "out"

REC = {"r_fast": 0.3, "tau_c_fast_s": 40, "slow_confirm_s": 1, "soft_unfreeze_s": 2,
       "slope_cap_frac": 0.006, "r_slow_max": 0.05, "tau_r_fast_s": 6,
       "tau_r_slow_idle_s": 0.5}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, pre, seg = z["t"], z["pre"], z["seg"]
    r = obs.run(t, pre, obs.default_with(REC))
    mine, dev = r["out_tot"], r["out_tot"] - z["tot_seg"]
    print(f"parity：复算显示总值 vs 录制 seg 总值")
    print(f"  均差={np.abs(dev).mean():.2f} 最大差={np.abs(dev).max():.2f} ADC "
          f"(段末差={dev[-1]:+.2f})")
    tin = z["tot_pre"]
    T0 = 1.45
    me = (t >= T0 + 0.1) & (t <= T0 + 0.6)
    E = float(tin[me].min())
    m = t >= T0 + 0.1
    print(f"\nE(台阶后输入最小)={E:.1f} ADC；输入末值={tin[-1]:.1f} "
          f"(输入蠕变 +{tin[-1]-E:.1f} = +{(tin[-1]-E)/12363*100:.1f}% 台阶)")
    for name, y in (("录制显示(seg)", z["tot_seg"]), ("复算显示", mine)):
        seg_y = y[m]
        seg_t = t[m]
        print(f"{name}: 峰={seg_y.max():.1f}@{seg_t[np.argmax(seg_y)]:.2f}s "
              f"末={seg_y[-1]:.1f} 谷={seg_y.min():.1f}@{seg_t[np.argmin(seg_y)]:.2f}s "
              f"过减(E−谷)={E-seg_y.min():.1f} 落点(末−E)={seg_y[-1]-E:+.1f} "
              f"下坠(峰−末)={seg_y.max()-seg_y[-1]:.1f}")
    # 逐帧轨迹
    print(f"\n{'t':>6s} {'输入':>8s} {'录制显示':>9s} {'复算显示':>9s} {'x1':>8s} {'x2':>8s} {'applied':>8s}")
    for tt in (1.0, 1.8, 2.5, 3.5, 5.0, 7.0, 9.0, 11.0, 13.0, 13.65):
        i = int(np.searchsorted(t, tt))
        if i >= len(t):
            break
        print(f"{t[i]:6.2f} {tin[i]:8.1f} {z['tot_seg'][i]:9.1f} {mine[i]:9.1f} "
              f"{r['x1'][i]:8.1f} {r['x2'][i]:8.1f} {r['applied'][i]:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
