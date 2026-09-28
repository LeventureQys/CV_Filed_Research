# -*- coding: utf-8 -*-
"""v3.6 最终对照表：四臂一次跑完（v3.1 / v3.2-R1 / v3.4-已落地 / v3.6-候选）+ 跨数据集副作用。

一键复跑：
  python v36_compare_final.py
产物：results/v36_final_table.txt、results/v36_final_cross.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
import v36_replay as R  # noqa: E402
from v36_probe_internal import R_arm  # noqa: E402
from v36_ab import episode_table, step_metrics  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

ARMS = [
    ("v3.1 现役基线", "v31"),
    ("v3.2 R1(0.7)", "v32:0.7"),
    ("v3.4 creep-mem(已落地)", "v34:120"),
    ("v3.6 本轮候选", "v36:--seed-gain:1.0:--v36-d:400"),
]


def main():
    ds = K.load(K.DS_TARGET)
    rows, gs, gg, extra = {}, {}, {}, {}
    for lab, spec in ARMS:
        st = R.run_arm(ds, *R_arm(spec))
        rows[lab] = episode_table(st)
        sm = step_metrics(st)
        gs[lab] = [g for (_, _, _, g) in sm if np.isfinite(g)]
        extra[lab] = dict(
            viol=float(np.nanmax(np.abs(st["max_clamp_viol"]))),
            hits=float(st["clamp_hits"][-1]),
            pkpk=float(np.percentile(st["sum_out"][(st["t"] >= 70) & (st["t"] <= 232)], 99) -
                       np.percentile(st["sum_out"][(st["t"] >= 70) & (st["t"] <= 232)], 1)),
        )

    L = ["== 目标录制 20260919_160854 · 四臂对照（总量 Σ21，ADC；ded = 输入 − 显示）==",
         "   数据源：working/零基线-反复增减同一负载/20260919_160854_single_device_7b3977", ""]
    order = [lab for lab, _ in ARMS]
    L.append("%10s" % "段" + "".join("%26s" % lab for lab in order))
    for i, r0 in enumerate(rows[order[0]]):
        cells = []
        for lab in order:
            r = rows[lab][i]
            cells.append("%26s" % ("%9.0f / ded%+7.0f" % (r["out_lvl"], r["ded"])))
        L.append("%10s" % r0["lab"][:10] + "".join(cells))
    L.append("")
    L.append("%10s" % "汇总" + "".join("%26s" % lab for lab in order))
    grp = {lab: [x for x in rows[lab] if x["kind"] == "full"][1:] for lab in order}  # E2~E9
    for key, fn in (("同负载组极差", lambda rs: max(x["out_lvl"] for x in rs) -
                                              min(x["out_lvl"] for x in rs)),
                    ("前4中位(E2~E5)", lambda rs: np.median([x["out_lvl"] for x in rs[:4]])),
                    ("后4中位(E6~E9)", lambda rs: np.median([x["out_lvl"] for x in rs[4:]])),
                    ("前后差", lambda rs: np.median([x["out_lvl"] for x in rs[:4]]) -
                                          np.median([x["out_lvl"] for x in rs[4:]])),
                    ("ded极差(E2~E9)", lambda rs: max(x["ded"] for x in rs) -
                                                  min(x["ded"] for x in rs))):
        L.append("%10s" % key + "".join("%26.0f" % fn(grp[lab]) for lab in order))
    L.append("%10s" % "G中位" + "".join("%26.3f" % np.median(gs[lab]) for lab in order))
    L.append("%10s" % "G最差" + "".join("%26.3f" % min(gs[lab]) for lab in order))
    L.append("%10s" % "C限幅越界" + "".join("%26.3g" % extra[lab]["viol"] for lab in order))
    L.append("%10s" % "C限幅命中" + "".join("%26.0f" % extra[lab]["hits"] for lab in order))
    L.append("%10s" % "长保压pk-pk" + "".join("%26.0f" % extra[lab]["pkpk"] for lab in order))
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(RES, "v36_final_table.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print("\n-> %s" % os.path.join(RES, "v36_final_table.txt"))


if __name__ == "__main__":
    main()
