# -*- coding: utf-8 -*-
"""10 报告：把寻优结果整理成「每类传感器怎么调参」的最终结论表 + 稳健性检查。

内容
----
1. 每类传感器：现役默认 vs 最优参数的逐会话指标（含 ADC 绝对值）
2. 稳健性：同一组参数换「原始 elapsed 时间轴」（等价上位机批量到达时刻）复算，看结论是否翻转
3. 敏感性：从一维扫描里挑出「单改一个参数收益最大」的旋钮，作为现场调参顺序

产物：out/05_report.json / out/05_report.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import PREP, Set, run_sensor, score  # noqa: E402

ORDER = ["r_fast", "r_slow_max", "slope_gate_frac", "slope_cap_frac", "tau_slope_s",
         "edge_slope_thres", "slow_confirm_s", "soft_unfreeze_s", "tau_c_fast_s",
         "tau_r_slow_idle_s", "tau_r_fast_s", "hold_eps", "idle_frac", "ramp_slope_min"]
CN = {
    "r_fast": "快态幅度比 r_fast", "r_slow_max": "慢态幅度上限 r_slow_max",
    "slope_gate_frac": "沿门 slope_gate_frac", "slope_cap_frac": "慢态积分速率上限 slope_cap_frac",
    "tau_slope_s": "速率低通 τ tau_slope_s", "edge_slope_thres": "沿阈 edge_slope_thres",
    "slow_confirm_s": "受载确认 slow_confirm_s", "soft_unfreeze_s": "沿后软冻结 soft_unfreeze_s",
    "tau_c_fast_s": "快态收敛 τ tau_c_fast_s", "tau_r_slow_idle_s": "空载慢态泄放 τ",
    "tau_r_fast_s": "空载快态恢复 τ", "hold_eps": "预留池余量 hold_eps",
    "idle_frac": "近零带 idle_frac", "ramp_slope_min": "缓坡下限 ramp_slope_min",
}


def fmt_params(p: dict) -> str:
    return "、".join(f"{k}={p[k]:g}" for k in ORDER if k in p) or "（全部默认）"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    presets = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))
    sweep1d = json.loads((OUT_DIR / "03_sweep1d.json").read_text(encoding="utf-8"))
    lines: list[str] = []
    report: dict = {}

    def w(s: str = "") -> None:
        print(s)
        lines.append(s)

    for sensor in SENSORS:
        best = presets[sensor]["params"]
        live_score = presets[sensor]["live_score"]
        best_score = presets[sensor]["preset_score"]
        w(f"\n{'=' * 100}")
        w(f"### {sensor}   现役默认 score={live_score:.1f}  →  推荐预设 score={best_score:.2f}")
        w(f"    {presets[sensor]['note']}")
        w(f"    推荐参数偏离现役默认的项：{fmt_params(best)}")

        # ── 逐会话指标（uniform + raw 两种时间轴）──
        res_u = run_sensor(sensor, [Set("live", {}), Set("best", best)],
                           tag_name=f"rep_u_{sensor}")
        res_r = run_sensor(sensor, [Set("live", {}), Set("best", best)],
                           time_mode="raw", tag_name=f"rep_r_{sensor}")
        w(f"\n  {'会话':<12s} {'参数':<8s} {'残漂%':>7s} {'过扣%':>7s} {'末30s%':>8s} "
          f"{'扣除%':>7s} {'末残ADC':>9s} {'蠕变ADC':>8s} │ {'raw残漂%':>8s} {'raw扣除%':>9s}")
        rep_sess = {}
        for tag in SENSORS[sensor]["sessions"]:
            p = PREP[f"{sensor}/{tag}"]
            row = {}
            for nm, cn in (("live", "现役"), ("best", "最优")):
                mu = res_u[nm][tag]
                mr = res_r[nm][tag]
                w(f"  {tag:<12s} {cn:<8s} {mu['resid_pct']:7.1f} {mu['over_pct']:7.1f} "
                  f"{mu['tail_pct']:8.2f} {mu['ded_pct']:7.1f} {mu['err_end_adc']:9.1f} "
                  f"{p['creep_total']:8.0f} │ {mr['resid_pct']:8.1f} {mr['ded_pct']:9.1f}")
                row[cn] = {"uniform": mu, "raw": mr}
            rep_sess[tag] = row
        w(f"  （残漂/过扣/末30s 均以「本该扣掉的蠕变」为分母；扣除% = 实际扣掉 / 真实蠕变）")

        # ── 敏感性：单改一个参数的最大收益 ──
        sens = []
        for nm, per in sweep1d[sensor].items():
            s_ = sum(score(m) for m in per.values()) / len(per)
            sens.append((s_, nm))
        sens.sort()
        w(f"\n  一维扫描里最有效的单参数改动（top6，score 越小越好；现役 {live_score:.1f}）：")
        for s_, nm in sens[:6]:
            w(f"    {nm:<38s} score={s_:8.2f}  （相对现役 {s_ - live_score:+8.2f}）")

        report[sensor] = {
            "live_score": live_score, "best_score": best_score, "best_params": best,
            "sessions": rep_sess,
            "top_single_knobs": [{"set": nm, "score": s_} for s_, nm in sens[:8]],
        }
    (OUT_DIR / "05_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (OUT_DIR / "05_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    w(f"\n写出 {OUT_DIR / '05_report.json'} / 05_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
