# -*- coding: utf-8 -*-
"""T6-A2：网格步长敏感性（口径分歧的定量溯源）。

发现：项目 00 号文档 §4.2.4 规定"插到 100 Hz 均匀网格（dt=0.01 s）"，
但第一轮 `11-paper-v6/scripts/pv_run.py` 用的 `ad_lib.prep` 网格步长是
`dtm = span/(n-1)`（实测恒载族 ≈ 0.00995 s，即 ≈100.5 Hz）—— **两者不是同一个网格**。
本脚本对 9 份恒载分别在两种网格上跑 v5.1 / v6，量化该差异对 L2 可重复性数字的影响，
用以解释本报告与 `13-v6-assessment/results/b_repeat_*.csv` 的分歧。

产出：results/t6_gridsens.csv、results/t6_gridsens_disp.csv、results/_t6a2_gridsens.log
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import t6_common as C                                              # noqa: E402
from t6_ad_lib import load_rec                                     # noqa: E402
from t6_glm53_v51 import GLM53v51                                  # noqa: E402
from t6_glm53_v6 import GLM53v6                                    # noqa: E402
from t6_glm53_v61 import GLM53v61                                  # noqa: E402

RES = C.RES
LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def grid_g(path, mode):
    """mode='g100' → dt=0.01；mode='gleg' → dtm=span/(n-1)（= 第一轮 ad_lib.prep 口径）。"""
    t, X = load_rec(path)
    span = t[-1] - t[0]
    dtm = 0.01 if mode == "g100" else span / (len(t) - 1)
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu, dtm


def run(Cls, tu, Xu):
    c = Cls(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def main():
    rows = []
    for tag in C.HOLD_TAGS:
        grp = C.GROUP[tag.split("/")[0]]
        for mode in ("g100", "gleg"):
            tu, Xu, dt = grid_g(C.HOLD[tag], mode)
            tot = Xu.sum(axis=1)
            dom = C.onset_geometry(tu, tot, dt)
            for name, Cls in (("v5.1", GLM53v51), ("v6", GLM53v6), ("v6.1", GLM53v61)):
                Y = run(Cls, tu, Xu)
                m = C.display_metrics(tu, Y, dom, dt)
                rows.append(dict(tag=tag, group=grp, grid=mode, dtm=dt, n=len(tu),
                                 arm=name, **{k: v for k, v in m.items()}))
                p(f"  {tag:>18} {mode} dtm={dt:.6f} {name:>5} plat={m['plat_lvl']:.3f} "
                  f"err5={m['err_plat5_pct']:+.3f} errstep={m['err_platstep_pct']:+.3f} "
                  f"T_stable={m['T_stable']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t6_gridsens.csv"), index=False, encoding="utf-8-sig")

    out = []
    for (arm, g), gg in df.groupby(["arm", "grid"]):
        a = gg.set_index("tag")
        for key, lbl in (("err_plat5_pct", "plat_err_std_R5_pp"),
                         ("err_platstep_pct", "plat_err_std_Rstep_pp"),
                         ("plat_lvl", "abs_plat_std"), ("T_stable", "T_stable_std_s")):
            s = a.groupby("group")[key].std(ddof=1)
            out.append(dict(arm=arm, grid=g, metric=lbl, median=float(s.median()),
                            by_group="|".join(f"{v:.3f}" for v in s)))
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(RES, "t6_gridsens_disp.csv"), index=False, encoding="utf-8-sig")
    p("")
    p("=" * 100)
    p("网格步长敏感性：同 9 份恒载、同实现，只换网格步长")
    p("=" * 100)
    piv = d.pivot_table(index=["arm", "metric"], columns="grid", values="median")
    with pd.option_context("display.width", 160):
        p(piv.round(4).to_string())
    p("")
    p("逐录制对照（v6，err_plat5_pct；用于与 b_repeat_records.csv 的同一列对拍）")
    w = df[(df.arm == "v6")].pivot_table(index=["group", "tag"], columns="grid",
                                         values="err_plat5_pct")
    with pd.option_context("display.width", 160):
        p(w.round(3).to_string())
    p("")
    p("逐录制对照（v6，err_platstep_pct）")
    w2 = df[(df.arm == "v6")].pivot_table(index=["group", "tag"], columns="grid",
                                          values="err_platstep_pct")
    with pd.option_context("display.width", 160):
        p(w2.round(3).to_string())
    with io.open(os.path.join(RES, "_t6a2_gridsens.log"), "w", encoding="utf-8") as fh:
        fh.write("T6-A2 gridsens log\n" + "\n".join(LOG) + "\n")
    p("\n-> results/t6_gridsens.csv / t6_gridsens_disp.csv / _t6a2_gridsens.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
