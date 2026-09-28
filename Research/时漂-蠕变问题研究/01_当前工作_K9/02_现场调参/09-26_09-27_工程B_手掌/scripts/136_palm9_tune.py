# -*- coding: utf-8 -*-
"""136_palm9_tune：右手掌 164332——最小档参数是否需要再调。
两段保压（0.9~77.9s、84.5~149.6s），全卸载后重起算 E。候选：录制的 rsm.15 vs 提高 rsm。"""
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
s93 = import_module("93_palm4_sweep")
OUT = TEMP / "palm9" / "out"

P_MIN = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
         "slope_cap_frac": 0.005, "r_slow_max": 0.15, "tau_r_fast_s": 6.0,
         "tau_r_slow_idle_s": 0.5}
HOLDS = [(0.9, 77.9, 13222.0), (84.5, 149.6, 16752.0)]  # (t0, t1, E=加载后输入平台)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, pre, seg = z["t"], z["pre"], z["tot_seg"]
    tin = z["tot_pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])

    print("== 各保压段输入蠕变（相对各自 E）")
    for t0, t1, E in HOLDS:
        for frac in [0.13, 0.5, 1.0]:
            tt = t0 + frac * (t1 - t0)
            i = min(int(np.searchsorted(t, tt)), len(t) - 1)
            print(f"  段{HOLDS.index((t0,t1,E))+1} @{t[i]-t0:5.1f}s 输入−E={tin[i]-E:+6.0f} "
                  f"({(tin[i]-E)/E*100:+5.1f}%)")

    print("\n== 设备录制显示（最小档实拍）与候选复算（N=ADC/1000）")
    print(f"{'参数集':<22s}{'段':>3s} {'落点(N)':>9s}{'10s→末(N)':>10s}{'尾段带宽(N)':>11s}"
          f"{'下坠(N)':>9s}{'过扣(N)':>9s}{'末斜率':>8s}{'x2/(rsm·e)':>11s}")

    def report(name, y, r=None):
        for j, (t0, t1, E) in enumerate(HOLDS):
            i0 = int(np.searchsorted(t, t0 + 0.6))
            i1 = int(np.searchsorted(t, t1 - 0.6))
            yy, tt = y[i0:i1 + 1], t[i0:i1 + 1]
            end = float(yy[-1])
            i10 = min(int(10 * fps), len(yy) - 1)
            i50 = min(int(50 * fps), len(yy) - 1)
            mm = tt >= tt[-1] - 5.0
            sat = ""
            if r is not None:
                hi = P_MIN["r_slow_max"] if name == "rec_rsm.15" else None
            print(f"{name:<22s}{j+1:3d} {end-E:+9.3f}{end-yy[i10]:+10.3f}"
                  f"{yy[i50:].max()-yy[i50:].min():11.3f}{yy.max()-end:9.3f}"
                  f"{max(0.0, E-yy.min()):9.3f}{np.polyfit(tt[mm], yy[mm], 1)[0]:+8.2f}")

    yseg = s93.medfilt1s(seg, fps)
    report("设备录制(最小档)", yseg / 1000.0)

    cands = [("rec_rsm.15", P_MIN)]
    for rsm in [0.25, 0.35, 0.5, 0.6]:
        cands.append((f"rsm{rsm:g}", {**P_MIN, "r_slow_max": rsm}))
    cands.append(("rsm.35_cap.01", {**P_MIN, "r_slow_max": 0.35, "slope_cap_frac": 0.01}))
    rows = []
    for name, p in cands:
        r = obs.run(t, pre, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"], fps) / 1000.0
        # 饱和度：x2 / (rsm·e)，e 用 (in−x1−x2) 总值近似逐通道同分布
        e_tot = np.maximum(tin - r["x1"] - r["x2"], 0.0)
        e_ch = e_tot / pre.shape[1]
        hi_ch = p["r_slow_max"] * np.maximum(e_ch, 1.0)
        x2_ch = r["x2"] / pre.shape[1]
        satv = x2_ch / np.maximum(hi_ch, 1e-9)
        m_end = t > HOLDS[1][0] + 30
        satmx = float(satv[m_end].max())
        report(name, y)
        print(f"{'':22s}   段2尾段 x2/(rsm·e) max={satmx:.2f}（≥0.95 即撞顶）")
        rows.append({"name": name, "params": p, "sat2": satmx})
    (OUT / "136_tune.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
