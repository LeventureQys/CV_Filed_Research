# -*- coding: utf-8 -*-
"""20 参数面缺口核实：把「推荐预设用到的每一项」逐项对照
   (A) `src/domain/drift_v6/creep_observer.h::Params` 的 27 个成员（算法层面能不能表达）
   (B) `src/ui/dialogs/creep_observer_params_dialog.cpp::kSpecs` 的 8 项及其 min/max（今天能不能调）
   并对比三种情形的实测成绩：现役默认 / 只用现有 8 项调到最好 / 全量推荐预设。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score  # noqa: E402

REPO = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp")
HEADER = REPO / "src/domain/drift_v6/creep_observer.h"
DIALOG = REPO / "src/ui/dialogs/creep_observer_params_dialog.cpp"
PRESETS = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))
UI_SEARCH = json.loads((OUT_DIR / "04_search_ui.json").read_text(encoding="utf-8"))


def params_members() -> list[str]:
    txt = HEADER.read_text(encoding="utf-8")
    body = txt.split("struct Params {", 1)[1].split("\n    };", 1)[0]
    return re.findall(r"double\s+(\w+)\s*=", body)


def ui_specs() -> dict[str, tuple[float, float]]:
    txt = DIALOG.read_text(encoding="utf-8")
    body = txt.split("const std::array<ParamSpec, 8> kSpecs = {{", 1)[1].split("}};", 1)[0]
    out = {}
    for m in re.finditer(r'\{"(\w+)",\s*"[^"]*",\s*&Params::\w+,\s*([\d.]+),\s*([\d.]+),', body):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    pm, ui = params_members(), ui_specs()
    print(f"C++ Params 成员 {len(pm)} 个；参数对话框 kSpecs {len(ui)} 项\n")

    used: set[str] = set()
    for spec in PRESETS.values():
        used |= set(spec["params"])
    print(f"推荐预设共用到 {len(used)} 个键，逐项核对：")
    print(f"  {'键':<22s} {'在 Params 里':>12s} {'在对话框里':>10s} {'对话框范围':>18s} {'预设取值越界':>12s}")
    missing, hidden, outrange = [], [], []
    for k in sorted(used):
        in_pm = k in pm
        in_ui = k in ui
        rng = f"{ui[k][0]:g} ~ {ui[k][1]:g}" if in_ui else "-"
        vals = [spec["params"][k] for spec in PRESETS.values() if k in spec["params"]]
        oor = ""
        if in_ui:
            bad = [v for v in vals if v < ui[k][0] - 1e-9 or v > ui[k][1] + 1e-9]
            oor = "是 " + ",".join(f"{v:g}" for v in sorted(set(bad))) if bad else "否"
            if bad:
                outrange.append(k)
        else:
            hidden.append(k)
        if not in_pm:
            missing.append(k)
        print(f"  {k:<22s} {'是' if in_pm else '否 ←':>12s} {'是' if in_ui else '否 ←':>10s} "
              f"{rng:>18s} {oor:>12s}")
    print(f"\n  · 不在 Params 里的键（= 真需要新增算法参数）：{missing or '无'}")
    print(f"  · 在 Params 里但对话框未暴露（= 只需暴露，不需新增）：{hidden or '无'}")
    print(f"  · 已暴露但预设取值超出对话框范围（= 只需放宽 min/max）：{outrange or '无'}")

    # ── 三种情形的实测成绩 ──
    print(f"\n{'传感器':<12s} {'情形':<34s} {'score':>8s} {'扣除%':>7s} {'过扣%':>7s}")
    rows_all = {}
    for sensor in SENSORS:
        cases = [("① 现役默认（8 项全默认）", {}),
                 ("② 只用现有 8 项调到最好", UI_SEARCH[sensor]["runs"][0]["params"]),
                 ("③ 全量推荐预设（含 6 个未暴露项）", PRESETS[sensor]["params"])]
        sets = [Set(f"s{i}", c) for i, (_, c) in enumerate(cases)]
        res = run_sensor(sensor, sets, tag_name=f"gap2_{sensor}")
        rows = []
        for i, (nm, c) in enumerate(cases):
            per = list(res[f"s{i}"].values())
            sc = sum(score(m) for m in per) / len(per)
            ded = sum(m["ded_pct"] for m in per) / len(per)
            over = sum(m["over_pct"] for m in per) / len(per)
            rows.append({"case": nm, "score": sc, "ded_pct": ded, "over_pct": over})
            print(f"{sensor:<12s} {nm:<34s} {sc:8.2f} {ded:7.1f} {over:7.1f}")
        rows_all[sensor] = rows
        print()
    (OUT_DIR / "12_param_gap.json").write_text(
        json.dumps({"params_members": pm, "ui_specs": ui,
                    "used_keys": sorted(used), "need_new": missing,
                    "need_expose": hidden, "need_widen": outrange,
                    "results": rows_all}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {OUT_DIR / '12_param_gap.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
