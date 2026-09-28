# -*- coding: utf-8 -*-
"""133_edge_check：验证力值模式下「沿检测 / 缓坡前馈」是否根本不触发。

算法内部 slope(k) = (v(k) − v_lp(k)) / tau_slope_s（tau_slope_s=1 非界面参数）。
  fire    : slope > edge_slope_thres(60)  → 触发沿、t_edge 清零、τc1→2 s
  ramp_on : 0.5 < slope ≤ 60              → 缓坡计时，满 4 s 后 τc1 过渡到 2 s
同时统计 soft_unfreeze_s 门（t_edge > soft·ln2）在保压期是否恒满足。
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


def slope_stats(t: np.ndarray, V: np.ndarray, tag: str) -> None:
    n, m = V.shape
    v_lp = V[0].astype(float).copy()
    last = float(t[0])
    smax = np.zeros(n)
    frac_ramp = 0
    frac_fire = 0
    for i in range(1, n):
        dt = min(max(float(t[i]) - last, 0.0), 0.1)
        last = float(t[i])
        if dt <= 0:
            smax[i] = smax[i - 1]
            continue
        sl = (V[i] - v_lp) / 1.0
        smax[i] = np.abs(sl).max()
        frac_fire += int((sl > 60.0).any())
        frac_ramp += int(((sl > 0.5) & (sl <= 60.0)).any())
        v_lp = v_lp + (dt / 1.0) * (V[i] - v_lp)
    print(f"\n[{tag}] 帧数={n}")
    print(f"  逐帧 max|slope|（N/s，通道最大）：p50={np.percentile(smax,50):.4f} "
          f"p95={np.percentile(smax,95):.4f} p99.9={np.percentile(smax,99.9):.4f} max={smax.max():.4f}")
    print(f"  触发沿（slope>60 N/s）的帧数={frac_fire}（0 才说明沿检测在力值模式失效）")
    print(f"  触发缓坡（0.5<slope≤60）的帧数={frac_ramp}")
    print(f"  结论：τc1 在全程 = tau_c_fast_s（{frac_fire+frac_ramp == 0} 时成立；"
          f"若有缓坡帧，需看是否连续满 4 s）")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "force_1d925c.npz")
    slope_stats(z["t"], z["pre"], "力值 1d925c（pre 流，N）")
    z2 = np.load(OUT / "in_03225d.npz")
    slope_stats(z2["t"], z2["V"], "力值 03225d（seg 即输入，N）")
    z4 = np.load(TEMP / "palm6" / "out" / "streams.npz")
    slope_stats(z4["t"], z4["pre"], "ADC 3efae9（pre 流，ADC）— 对照")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
