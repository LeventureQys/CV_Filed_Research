# -*- coding: utf-8 -*-
"""138_palm9_localize：过扣/负冲定位 + tau_r_fast_s 对二次加载过扣的作用。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT9 = TEMP / "palm9" / "out"
HOLDS = [(0.9, 77.9, 13222.0), (84.5, 149.6, 16752.0)]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT9 / "streams.npz")
    t, pre = z["t"], z["pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    tin = z["tot_pre"]
    BASE = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
            "slope_cap_frac": 0.01, "r_slow_max": 0.15, "tau_r_fast_s": 6.0,
            "tau_r_slow_idle_s": 0.5}
    for tr1 in [6.0, 3.0, 2.0, 1.0]:
        p = {**BASE, "tau_r_fast_s": tr1}
        r = obs.run(t, pre, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"], fps) / 1000.0
        # 段1内过扣位置
        i0, i1 = int(np.searchsorted(t, 1.5)), int(np.searchsorted(t, 77.3))
        k1 = i0 + int(np.argmin(y[i0:i1]))
        # 段2内过扣位置
        j0, j1 = int(np.searchsorted(t, 85.1)), int(np.searchsorted(t, 149.0))
        k2 = j0 + int(np.argmin(y[j0:j1]))
        ia, ib = int(np.searchsorted(t, 78.5)), int(np.searchsorted(t, 84.0))
        kd = ia + int(np.argmin(y[ia:ib]))
        print(f"τr1={tr1:g}s：段1过扣={1.5-y[k1] if y[k1]<13.222 else 0:6.2f}N@{t[k1]:.1f}s"
              f"（E=13.22，min={y[k1]:.2f}）  段2过扣 min={y[k2]:.2f}N@{t[k2]:.1f}s（E=16.75）"
              f"  卸载负冲 min={y[kd]:+.2f}N@{t[kd]:.1f}s  x1末(段1)={r['x1'][int(np.searchsorted(t,77))]:.0f}"
              f" x2末(段1)={r['x2'][int(np.searchsorted(t,77))]:.0f}")
        if tr1 == 6.0:
            print("  参考轨迹(t, 输入N, 显示N)：")
            for tt in [76, 77, 77.5, 78, 79, 80, 82, 84, 84.5, 85, 86, 88, 92, 100, 120, 149]:
                i = min(int(np.searchsorted(t, tt)), len(t) - 1)
                print(f"    {t[i]:6.1f}s 输入={tin[i]/1000:6.2f} 显示={y[i]:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
