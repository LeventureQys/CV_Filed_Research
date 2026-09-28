# -*- coding: utf-8 -*-
"""28 定稿：从各轮搜索结果里挑出每类传感器的最优参数，写成 09_presets.json（新一轮定稿），
   并打一张逐会话的绝对 ADC 验收表（过减 / 欠减 / 末值偏差 / 绝对过减 / 扣除率）。

候选来源（按优先级，全部用**绝对口径目标函数**重新独立复算排序，避免沿用旧评分）：
    out/16_refine.json      （细网格收敛）
    out/17_focus.json       （预留池定点收敛）
    out/14_search_adc.json  （绝对口径寻优）
    out/figures_archive/09_presets_第1轮.json（第一轮比例制预设，作为兜底对照）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score_adc  # noqa: E402


def load(name: str) -> dict:
    f = OUT_DIR / name
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8"))
    return {k: v["params"] for k, v in d.items() if isinstance(v, dict) and "params" in v}


def rank_score(sensor: str, m: dict) -> float:
    """四指指腹的载荷跨 7.9k~33.7k ADC（差 4 倍），绝对指标会被大会话主导 ⇒ 用相对 ΣE 的百分比；
    其余四类载荷量级接近 ⇒ 用绝对 ADC（用户验收口径）。"""
    if sensor == "四指指腹":
        e = m["Esum"]
        return ((m["peak_exc"] / e * 100) + 0.5 * abs(m["end_exc"] / e * 100)
                + 4.0 * max(0.0, m["low_exc"] / e * 100 - 1.0)
                + 1.0 * max(0.0, m["notch"] / e * 100 - 1.5))
    return score_adc(m)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    srcs = {
        "四指指腹专项": load("18_search_4f.json"),
        "细网格收敛": load("16_refine.json"),
        "预留池定点": load("17_focus.json"),
        "绝对口径寻优": load("14_search_adc.json"),
        "第一轮(比例制)": load("figures_archive/09_presets_第1轮.json"),
    }
    out = {}
    lines = ["# v3.4 观测器 · 五类传感器推荐预设（第二轮：绝对 ADC 口径 + 过减约束）", ""]
    print(f"{'传感器':<12s} {'来源':<14s} {'过减ADC':>8s} {'过减%ΣE':>8s} {'回落max':>8s} "
          f"{'欠减峰':>7s} {'末值均':>8s} {'扣除%':>7s} {'J':>7s}")
    for sensor in SENSORS:
        cands = [("现役默认", {})] + [(nm, src[sensor]) for nm, src in srcs.items()
                                      if sensor in src]
        sets = [Set(f"p{i}", c) for i, (_, c) in enumerate(cands)]
        res = run_sensor(sensor, sets, tag_name=f"final_{sensor}")
        rows = []
        for i, (nm, c) in enumerate(cands):
            per = list(res[f"p{i}"].values())
            if not per:
                continue
            rows.append((sum(rank_score(sensor, m) for m in per) / len(per), nm, c, res[f"p{i}"]))
        rows.sort(key=lambda r: r[0])
        for j, (sc, nm, c, per) in enumerate(rows):
            v = list(per.values())
            mark = "★" if j == 0 else " "
            print(f"{sensor:<12s} {mark}{nm:<13s} {sum(m['low_exc'] for m in v) / len(v):8.1f} "
                  f"{sum(m['low_exc'] / m['Esum'] * 100 for m in v) / len(v):8.2f} "
                  f"{max(m['notch'] for m in v):8.1f} "
                  f"{sum(m['peak_exc'] for m in v) / len(v):7.1f} "
                  f"{sum(m['end_exc'] for m in v) / len(v):8.1f} "
                  f"{sum(m['ded_pct'] for m in v) / len(v):7.1f} {sc:7.2f}")
        sc, nm, c, per = rows[0]
        live = next(r for r in rows if r[1] == "现役默认")
        v = list(per.values())
        lv = list(live[3].values())
        out[sensor] = {
            "note": f"本轮定稿（来源：{nm}）。过减(显示沉到参考线下) "
                    f"{sum(m['low_exc'] for m in v) / len(v):.0f} ADC、"
                    f"欠减峰 {sum(m['peak_exc'] for m in v) / len(v):.0f} ADC、"
                    f"扣除率 {sum(m['ded_pct'] for m in v) / len(v):.0f}%；"
                    f"现役默认为 过减 {sum(m['low_exc'] for m in lv) / len(lv):.0f} / "
                    f"欠减峰 {sum(m['peak_exc'] for m in lv) / len(lv):.0f} / "
                    f"扣除 {sum(m['ded_pct'] for m in lv) / len(lv):.0f}%。",
            "params": c, "source": nm,
            "live_score": live[0], "preset_score": sc,
            "live_ded_pct": sum(m["ded_pct"] for m in lv) / len(lv),
            "preset_ded_pct": sum(m["ded_pct"] for m in v) / len(v),
            "live_over_pct": sum(m["low_exc"] for m in lv) / len(lv),
            "preset_over_pct": sum(m["low_exc"] for m in v) / len(v),
            "sessions": per,
        }
        lines.append(f"\n## {sensor}\n\n来源：{nm}\n\n```json\n"
                     f"{json.dumps(c, ensure_ascii=False)}\n```\n\n{out[sensor]['note']}\n")
        print()
    (OUT_DIR / "09_presets.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    (OUT_DIR / "09_presets.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写出 {OUT_DIR / '09_presets.json'}（新一轮定稿）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
