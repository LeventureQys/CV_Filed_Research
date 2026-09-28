# -*- coding: utf-8 -*-
"""100_palm5_synth：判定 1.29 N 回调能否全由算法产生。

合成输入：完全相同的压上过程（0→14.97 N, 1.35~2.2 s 斜坡），之后**恒定不变**。
若已写入参数在此输入上就能产生 ~1.29 N 下坠，则用户解释成立，且能定位是哪个态；
再测 rf=0.01 等修正档的下坠上限。ADC 尺度（×1000 送入观测器，嵌入式口径）。
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
s93 = import_module("93_palm4_sweep")


def synth(t: np.ndarray, level_n: float = 14.97) -> np.ndarray:
    """恒压合成输入（总值 N）：斜坡 1.35~2.2 s 压到 level，此后恒定。"""
    y = np.where(t < 1.35, 0.0,
                 np.where(t < 2.2, level_n * (t - 1.35) / 0.85, level_n))
    return y


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t = np.arange(0, 15.5, 1.0 / 100.7)
    ytot = synth(t)
    # 展到 71 通道（每通道等比）
    V = np.repeat(ytot[:, None] * 1000.0 / 71.0, 71, axis=1)
    fps = 100.7
    WRITTEN = {"r_fast": 0.10, "tau_c_fast_s": 15, "slow_confirm_s": 1,
               "soft_unfreeze_s": 2, "slope_cap_frac": 0.008, "r_slow_max": 0.10,
               "tau_r_fast_s": 6, "tau_r_slow_idle_s": 0.5}
    cands = [
        ("written", WRITTEN, "已写入 rf.10 τc1=15 cap.008 rsm.10"),
        ("rf01", {**WRITTEN, "r_fast": 0.01}, "rf=0.01（其余同已写入）"),
        ("rf003", {**WRITTEN, "r_fast": 0.03}, "rf=0.03"),
        ("rf006", {**WRITTEN, "r_fast": 0.06}, "rf=0.06"),
        ("rf01_cap0", {**WRITTEN, "r_fast": 0.01, "slope_cap_frac": 0.003,
                       "r_slow_max": 0.05}, "rf.01 cap.003 rsm.05"),
    ]
    print("合成恒压输入（峰 14.97 N 后恒定，无任何手部松弛）：")
    for name, p, label in cands:
        r = obs.run(t, V, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"] / 1000.0, fps)
        m = t >= 2.2
        drop = y[m].max() - y[m][-1]
        print(f"  {label:<34s} 显示: 峰={y[m].max():7.3f} 末={y[m][-1]:7.3f} "
              f"下坠={drop:6.3f} N  落点(末−14.97)={y[m][-1]-14.97:+7.3f} N "
              f"x1末={r['x1'][-1]/1000:.3f} x2末={r['x2'][-1]/1000:.3f}")
    print("\n（对照）实测 102924 显示：峰 14.72 → 末 13.55，下坠 ≈1.17~1.29 N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
