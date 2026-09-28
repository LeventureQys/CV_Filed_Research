# -*- coding: utf-8 -*-
"""T5A-Q6：**非滤波手段**扫描 —— 滑行器速率上限 / 交接时刻 τ_ho / 过充硬限幅 / 反向事件快速撤销。

每条手段都在**同一批事件、同一指标口径**下跑完整链路（13 份录制 × 12 臂），
给出代价与推荐组合。所有手段都是**因果**的（只用 t 及以前的数据）。

臂定义：
  B0_baseline             现状（κ 1.30/1.12，RATE_MAX 0.8，HO_MIN 5.0，REVOKE 0.40）
  G*_ratemax_*            滑行器上行速率上限 RATE_MAX ∈ {0.4, 0.6, 0.8, 1.2}
  H*_homin_*              交接时刻 HO_MIN ∈ {2.5, 3.5, 5.0, 8.0}
  C*_clip_*               过充硬限幅 C ∈ {0.02, 0.05, 0.10}（显示 ≤ 原始 + C·|原始|）
  R*_revoke_*             反向事件撤销窗 REVOKE ∈ {0.20, 0.30, 0.40, 0.60}
  X*_combo_*              组合（按 Q4/Q6 结论取最优两项 / 三项）

产物：`results/t5a_nonfilter_options.csv`（主表）、`t5a_nonfilter_perevent.csv`、
      `_t5a_nonfilter.log`。

用法：`python scripts/t5a_nonfilter.py`（长任务，建议后台）
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402

LOG = []
T0 = time.time()


def rec(m):
    s = f"[{time.time() - T0:7.1f}s] {m}"
    print(s, flush=True)
    LOG.append(s)


def clip_display(Y, Z, frac):
    """过充硬限幅（有界跟随）：显示总量不超过 原始总量 + frac·|原始总量|。
    逐通道按同一比例分摊（保持通道相对结构）。"""
    Zsum = Y.sum(axis=1)
    cap = Z + frac * np.abs(Z)
    over = Zsum > cap
    if not over.any():
        return Y
    out = Y.copy()
    k = (cap[over] / np.maximum(Zsum[over], 1e-9))
    out[over] = out[over] * k[:, None]
    return out


ARMS = [
    dict(id="B0_baseline", rate_max=0.8, ho_min=5.0, clip=None, revoke=0.40),
    dict(id="G1_ratemax_0.4", rate_max=0.4, ho_min=5.0, clip=None, revoke=0.40),
    dict(id="G2_ratemax_0.6", rate_max=0.6, ho_min=5.0, clip=None, revoke=0.40),
    dict(id="G4_ratemax_1.2", rate_max=1.2, ho_min=5.0, clip=None, revoke=0.40),
    dict(id="H1_homin_2.5", rate_max=0.8, ho_min=2.5, clip=None, revoke=0.40),
    dict(id="H2_homin_3.5", rate_max=0.8, ho_min=3.5, clip=None, revoke=0.40),
    dict(id="H4_homin_8.0", rate_max=0.8, ho_min=8.0, clip=None, revoke=0.40),
    dict(id="C1_clip_0.02", rate_max=0.8, ho_min=5.0, clip=0.02, revoke=0.40),
    dict(id="C2_clip_0.05", rate_max=0.8, ho_min=5.0, clip=0.05, revoke=0.40),
    dict(id="C3_clip_0.10", rate_max=0.8, ho_min=5.0, clip=0.10, revoke=0.40),
    dict(id="R1_revoke_0.20", rate_max=0.8, ho_min=5.0, clip=None, revoke=0.20),
    dict(id="R2_revoke_0.30", rate_max=0.8, ho_min=5.0, clip=None, revoke=0.30),
    dict(id="R4_revoke_0.60", rate_max=0.8, ho_min=5.0, clip=None, revoke=0.60),
    dict(id="X1_ratemax0.6_homin3.5", rate_max=0.6, ho_min=3.5, clip=None, revoke=0.40),
    dict(id="X2_ratemax0.6_clip0.05", rate_max=0.6, ho_min=5.0, clip=0.05, revoke=0.40),
    dict(id="X3_ratemax0.6_homin3.5_clip0.05", rate_max=0.6, ho_min=3.5, clip=0.05,
         revoke=0.40),
]


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    load = ev[ev.kind.isin(["onset", "restep"])]
    rec(f"事件 {len(ev)}（加载类 {len(load)}）；非滤波臂 {len(ARMS)}；"
        f"总推理 {len(ARMS) * len(recs)} 次")

    rows = []
    for ai, arm in enumerate(ARMS, 1):
        for k, d in recs.items():
            sub = load[load.key == k]
            if not len(sub):
                continue
            c = C.KV6(d["Xu"].shape[1])
            c.kappa_onset, c.kappa_restep = 1.30, 1.12
            c.RATE_MAX = float(arm["rate_max"])
            c.HO_MIN = float(arm["ho_min"])
            c.REVOKE = float(arm["revoke"])
            out = c.run(d["tu"], d["Xu"])
            Y = out["Y"]
            if arm["clip"] is not None:
                Y = clip_display(Y, d["Z"], float(arm["clip"]))
            Ys = Y.sum(axis=1)
            for _, e in sub.iterrows():
                m = C.metrics_at(d["Z"], Ys, d["tu"], float(e["t_on"]), d["span"])
                m.update(key=k, t_on=float(e["t_on"]), kind=e["kind"], dom=d["dom"],
                         arm=arm["id"])
                rows.append(m)
        rec(f"  [{ai:2d}/{len(ARMS)}] {arm['id']}")
    pe = pd.DataFrame(rows)
    pe.to_csv(os.path.join(C.TASK, "results", "t5a_nonfilter_perevent.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")

    base = pe[pe.arm == "B0_baseline"].set_index(["key", "t_on", "kind"])
    sm = []
    for arm, g in pe.groupby("arm", sort=False):
        on = g[g.kind == "onset"]
        gi = g.set_index(["key", "t_on", "kind"])
        common = gi.index.intersection(base.index)
        a = dict(arm=arm, cfg=str(next(x for x in ARMS if x["id"] == arm)),
                 n=len(g), n_onset=len(on),
                 OS_med=g.OS_pct.median(), OS_p90=g.OS_pct.quantile(.90),
                 OS_max=g.OS_pct.max(), OS_med_onset=on.OS_pct.median(),
                 OS_max_onset=on.OS_pct.max(),
                 US_med=g.US_pct.median(), US_max=g.US_pct.max(),
                 T_med=g.T_stable.median(), T_p90=g.T_stable.quantile(.90),
                 T_nan=int(g.T_stable.isna().sum()),
                 err1_med=g.err_1s_pct.median(), err1_abs_med=g.err_1s_pct.abs().median(),
                 G_med=g.G20.median(), G_min=g.G20.min(),
                 MD_med=g.MD.median(), MD_p90=g.MD.quantile(.90))
        if len(common):
            a.update(dOS_med=(gi.loc[common, "OS_pct"] - base.loc[common, "OS_pct"]).median(),
                     dOS_absmax=(gi.loc[common, "OS_pct"] - base.loc[common, "OS_pct"]).abs().max(),
                     dT_med=(gi.loc[common, "T_stable"] - base.loc[common, "T_stable"]).median(),
                     dT_absmax=(gi.loc[common, "T_stable"] - base.loc[common, "T_stable"]).abs().max(),
                     dG_med=(gi.loc[common, "G20"] - base.loc[common, "G20"]).median(),
                     dE_abs_med=(gi.loc[common, "err_1s_pct"].abs()
                                 - base.loc[common, "err_1s_pct"].abs()).median(),
                     dMD_med=(gi.loc[common, "MD"] - base.loc[common, "MD"]).median())
        sm.append(a)
    S = pd.DataFrame(sm)
    S["上线成本"] = ["一行常量" if x.startswith(("B0", "G", "H", "R")) else
                  ("显示层封顶 +1 行" if x.startswith("X3") else "显示层封顶 +1 行") for x in S.arm]
    S.to_csv(os.path.join(C.TASK, "results", "t5a_nonfilter_options.csv"),
             index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_nonfilter_options.csv")

    lines = ["\n===== T5A-Q6 非滤波手段摘要（加载类 n=41）====="]
    lines.append(f"{'arm':30s} {'OS_med':>8s} {'OS_max':>8s} {'dOS_med':>8s} "
                 f"{'T_med':>7s} {'dT_med':>7s} {'err1|med|':>10s} {'G_med':>6s} "
                 f"{'G_min':>6s} {'MD_med':>9s}")
    for _, r in S.iterrows():
        lines.append(f"{r['arm']:30s} {r['OS_med']:8.3f} {r['OS_max']:8.3f} "
                     f"{r.get('dOS_med', np.nan):8.3f} {r['T_med']:7.3f} "
                     f"{r.get('dT_med', np.nan):7.3f} {r['err1_abs_med']:10.3f} "
                     f"{r['G_med']:6.3f} {r['G_min']:6.3f} {r['MD_med']:9.1f}")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_nonfilter.log"),
                "python scripts/t5a_nonfilter.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
