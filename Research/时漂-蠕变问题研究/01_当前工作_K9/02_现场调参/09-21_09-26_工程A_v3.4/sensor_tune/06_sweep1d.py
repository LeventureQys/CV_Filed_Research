# -*- coding: utf-8 -*-
"""06 第一阶段：现役默认 + 12 个旋钮的一维扫描（每次只动一个参数），5 类传感器各跑一遍。

产出：out/03_sweep1d.json（全部原始指标）+ out/03_sweep1d.txt（表格快照）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, agg, print_table, run_sensor  # noqa: E402

# 一维扫描：只覆盖该键，其余保持 creep_observer.h 的现役默认
SWEEPS: list[tuple[str, str, list[float]]] = [
    ("快态幅度比 r_fast", "r_fast", [0.0, 0.02, 0.04, 0.06, 0.10, 0.16]),
    ("快态收敛 τ tau_c_fast_s", "tau_c_fast_s", [2.0, 4.0, 6.0, 12.0, 24.0]),
    ("慢态幅度上限 r_slow_max", "r_slow_max", [0.02, 0.04, 0.06, 0.08, 0.12, 0.20, 0.35]),
    ("慢态积分速率上限 slope_cap_frac", "slope_cap_frac", [0.002, 0.005, 0.01, 0.02, 0.04]),
    ("慢态受载确认 slow_confirm_s", "slow_confirm_s", [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]),
    ("慢态沿后软冻结 soft_unfreeze_s", "soft_unfreeze_s", [0.5, 1.0, 2.0, 4.0]),
    ("空载慢态泄放 tau_r_slow_idle_s", "tau_r_slow_idle_s", [0.5, 2.0, 8.0, 30.0]),
    ("空载快态恢复 tau_r_fast_s", "tau_r_fast_s", [0.2, 0.5, 1.0, 2.0, 6.0]),
    ("预留池余量 hold_eps", "hold_eps", [0.0, 0.5, 2.0, 8.0]),
    ("速率低通 τ tau_slope_s", "tau_slope_s", [0.5, 1.0, 2.0, 3.0]),
    ("沿门 slope_gate_frac", "slope_gate_frac", [0.01, 0.02, 0.05, 0.10, 0.20]),
    ("沿阈 edge_slope_thres", "edge_slope_thres", [20.0, 60.0, 150.0, 400.0]),
    ("近零带 idle_frac", "idle_frac", [0.02, 0.05, 0.10, 0.20]),
    ("零点跟踪 τ tau_zero_s", "tau_zero_s", [2.0, 8.0, 30.0]),
    ("去趋势基线 τ y_floor_tau_s", "y_floor_tau_s", [60.0, 300.0, 900.0]),
]


def build_sets() -> list[Set]:
    sets = [Set("A00_现役默认")]
    for label, key, vals in SWEEPS:
        for v in vals:
            sets.append(Set(f"{key}={v:g}", {key: v}))
    return sets


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sets = build_sets()
    print(f"参数集 {len(sets)} 个 × 5 类传感器")
    all_res = {}
    lines = []
    for sensor in SENSORS:
        res = run_sensor(sensor, sets)
        all_res[sensor] = res
        rows = agg(res, sensor)
        print_table(rows, sensor, top=14)
        lines.append(f"\n===== {sensor} =====\n")
        lines.append(f"{'参数集':<38s} {'score':>7s} {'残漂%':>8s} {'过扣%':>8s} {'欠扣%':>8s} "
                     f"{'末30s%':>8s} {'抖动%':>7s} {'扣除%':>7s}")
        for r in rows:
            lines.append(f"{r['set']:<38s} {r['score']:7.2f} {r['resid_pct']:8.2f} "
                         f"{r['over_pct']:8.2f} {r['under_pct']:8.2f} {r['tail_pct']:8.2f} "
                         f"{r['std_pct']:7.2f} {r['ded_pct']:7.1f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "03_sweep1d.json").write_text(
        json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "03_sweep1d.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '03_sweep1d.json'} / 03_sweep1d.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
