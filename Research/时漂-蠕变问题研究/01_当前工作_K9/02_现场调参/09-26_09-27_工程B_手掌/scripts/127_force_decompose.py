# -*- coding: utf-8 -*-
"""127_force_decompose：力值模式分解——下坠（sag）到底由哪一项造成。

对若干候选参数打印 x1/x2/applied 总值轨迹、逐帧 dx2 是否被 cap 钳位、
以及「冻结关系」D(t) vs v(t_g) 的验证。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
fl = import_module("122_force_lib")
OUT = TEMP / "palm7" / "out"

CANDS = [
    ("基线 rf.03 τc1 40 cap.011", {}),
    ("rf.50 τc1 0.5", {"r_fast": 0.5, "tau_c_fast_s": 0.5}),
    ("cap.006", {"slope_cap_frac": 0.006}),
    ("cap.05", {"slope_cap_frac": 0.05}),
    ("rsm.05", {"r_slow_max": 0.05}),
    ("conf5", {"slow_confirm_s": 5.0}),
]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "force_1d925c.npz")
    t, pre = z["t"], z["pre"]
    tin = pre.sum(1)
    fps = (len(t) - 1) / (t[-1] - t[0])
    segs = fl.segments(t, tin)
    print(f"fps={fps:.2f}  段={[(round(s['t0'],2), round(s['t1'],2), round(s['E'],3)) for s in segs]}")
    for name, over in CANDS:
        p = {**fl.REC8, **over}
        r = obs.run(t, pre, obs.default_with(p))
        rows = fl.evaluate(r["out_tot"], t, segs, fps)
        print(f"\n===== {name} =====")
        for j, rr in enumerate(rows):
            print(f"  段{j+1}: 落点{rr['dev']:+7.3f} 下坠{rr['drop']:6.3f} 过扣{rr['over']:6.3f} "
                  f"峰@{rr['peak_t']:5.2f}s 末斜率{rr['slope_end']:+.4f}")
        # 段1 轨迹
        i0 = segs[0]["i0"]
        print(f"  {'t':>6s} {'输入':>8s} {'显示':>8s} {'x1总':>7s} {'x2总':>7s} {'appl总':>7s} "
              f"{'输入−显示':>8s} {'v_lp±':>7s}")
        for tt in np.arange(segs[0]["t0"], segs[0]["t1"], 1.0):
            i = int(np.searchsorted(t, tt))
            if i >= len(t):
                break
            print(f"  {t[i]:6.2f} {tin[i]:8.3f} {r['out_tot'][i]:8.3f} {r['x1'][i]:7.3f} "
                  f"{r['x2'][i]:7.3f} {r['applied'][i]:7.3f} {tin[i]-r['out_tot'][i]:8.3f} "
                  f"{'':>7s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
