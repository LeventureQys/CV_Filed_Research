# -*- coding: utf-8 -*-
"""134_force_pick：在细网格结果上按「物理目标」筛参。

目标（按优先级）：
  ① 段末读数不再漂：max|末段斜率| 小（= 显示在保压末尾走平，既不继续上漂也不继续下坠）
  ② 不下穿目标电平：过扣 = E_pl − 段内最小 ≤ 0.15 N
  ③ 稳态误差小：max|落点| 小
  ④ 观感不跳：max 下坠 小
输出：满足①+②的最优若干组，以及全表统计。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
OUT = TEMP / "palm7" / "out"
REAL = ("hold_1d925c", "in_03225d")


def row(r: dict) -> tuple:
    segs = r["c"]["hold_1d925c"] + r["c"]["in_03225d"]
    names = ["段1", "段2", "03225d"]
    return (max(abs(x["slope_end"]) for x in segs),
            max(x["over"] for x in segs),
            max(abs(x["dev"]) for x in segs),
            max(x["drop"] for x in segs), segs, names)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    data = json.loads((OUT / "130_fine.json").read_text(encoding="utf-8"))
    scored = []
    for r in data:
        sl, ov, dv, dp, segs, names = row(r)
        scored.append({**r, "_sl": sl, "_ov": ov, "_dv": dv, "_dp": dp, "_segs": segs})
    base = [r for r in scored if (r["rf"], r["tc1"], r["cap"], r["conf"], r["rsm"])
            == (0.03, 40.0, 0.011, 2.0, 0.35)]
    if not base:
        print("[warn] 现役参数不在细网格内（τc1=40 未覆盖）")

    hdr = (f"{'rf':>5s} {'τc1':>5s} {'cap':>6s} {'conf':>4s} {'rsm':>4s} | "
           f"{'末斜1':>8s} {'末斜2':>8s} {'末斜3':>8s} | {'落点1':>6s} {'落点2':>6s} {'落点3':>6s} | "
           f"{'下坠max':>7s} {'过扣max':>7s} | {'恒压':>7s}")

    def line(r: dict) -> str:
        s1, s2, s3 = r["_segs"]
        f = r["c"]["synth_flat17"][0]
        return (f"{r['rf']:5g} {r['tc1']:5g} {r['cap']:6g} {r['conf']:4g} {r['rsm']:4g} | "
                f"{s1['slope_end']:+8.5f} {s2['slope_end']:+8.5f} {s3['slope_end']:+8.5f} | "
                f"{s1['dev']:+6.2f} {s2['dev']:+6.2f} {s3['dev']:+6.2f} | "
                f"{r['_dp']:7.3f} {r['_ov']:7.3f} | {f['dev']:+7.2f}")

    print("== 全表：末段斜率（+ = 仍在上漂，− = 仍在往下掉）==")
    print(hdr)
    for r in sorted(scored, key=lambda r: r["_sl"])[:0]:
        pass
    # 末斜率最接近 0 且合规的
    ok = [r for r in scored if r["_ov"] <= 0.15]
    print("\n== ①末斜率最接近 0（且过扣 ≤0.15），按 |落点| 次排序 ==")
    print(hdr)
    for r in sorted(ok, key=lambda r: (r["_sl"], r["_dv"]))[:15]:
        print(line(r))

    print("\n== ②在 |末斜率| ≤ 0.006 N/s 的集合里，落点+下坠 最小 ==")
    ok2 = [r for r in ok if r["_sl"] <= 0.006]
    print(f"   共 {len(ok2)} 组")
    print(hdr)
    for r in sorted(ok2, key=lambda r: r["_dv"] + r["_dp"])[:15]:
        print(line(r))

    print("\n== ③现役 vs 若干候选（含全部指标） ==")
    show = [r for r in scored if (r["rf"], r["tc1"], r["cap"], r["conf"], r["rsm"]) in {
        (0.03, 40.0, 0.011, 2.0, 0.35), (0.03, 8.0, 0.010, 0.0, 0.35),
        (0.03, 8.0, 0.008, 0.0, 0.35), (0.03, 1.0, 0.008, 0.0, 0.35),
        (0.05, 4.0, 0.010, 0.0, 0.35), (0.02, 2.0, 0.010, 0.0, 0.35)}]
    print(hdr)
    for r in show:
        print(line(r))
    (OUT / "134_pick.json").write_text(json.dumps(
        [{"rf": r["rf"], "tc1": r["tc1"], "cap": r["cap"], "conf": r["conf"], "rsm": r["rsm"],
          "sl": r["_sl"], "ov": r["_ov"], "dv": r["_dv"], "dp": r["_dp"]}
         for r in sorted(ok2, key=lambda r: r["_dv"] + r["_dp"])[:60]],
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
