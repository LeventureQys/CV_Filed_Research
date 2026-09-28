# -*- coding: utf-8 -*-
"""【C-4 核心】T5-A 统一 κ 扫描：裁决「第一轮 κ 消融」与「T4-B κ 项零收益」的冲突。

设计（必须与两处既有实验都可比）：
  · **同一批事件**：T4-A 冻结表 60 事件（`t4a_morphology.csv`，只读）。
  · **同一指标口径**：`OS% / T_stable / err_1s / MD` 全部由 `t5a_common.metrics_at` 一处实现
    （与 T5A-Q1 完全同源），并**同时**输出另一种参考电平口径 `G20`（台阶捕获比）作交叉检查。
  · **同一实现**：全部走 `t5a_common.KV6`（κ 实例可写）。κ=1.30/1.12 时与原型逐帧零差
    （见 `results/t5a_patch_ab_zero.csv` 的 V3/V4 行）。
  · **三个子实验**：
      (a) 统一 κ 扫描：κ_onset = κ_restep = κ ∈ {0.95 … 1.40}（**第一轮做的就是这件事**）
      (b) onset 单独扫描：κ_restep 固定 1.12，κ_onset 变（**T4-B 的 M0→M1 做的是这件事**）
      (c) restep 单独扫描：κ_onset 固定 1.30，κ_restep 变（**T4-B 的 M0→M2 做的是这件事**）
  · **留一泛化**：对每个 κ 工作点，做 13 折留一（留出一份录制），报留出集的 OS%/T_stable 中位
    的折间散布与最坏折。
  · κ 在批次内只影响 onset 类事件（`restep` 的 κ 被 `min(Â, κ·inc)` 与 `max(A, inc)` 下限夹住），
    这一点由 (b)/(c) 的差集直接给出。

产出：`results/t5a_kappa_sweep.csv`（逐 κ 逐类）、`t5a_kappa_summary.csv`（一行一工作点）、
      `t5a_kappa_loo.csv`（留一）、`t5a_kappa_perevent.csv`（逐事件逐 κ，供第三方复核）。

用法：`python scripts/t5a_kappa.py`（长任务，建议后台作业 + 60~120 s 轮询）
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


GRID_U = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]   # 统一 κ
GRID_ONSET = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]           # 仅 onset
GRID_RESTEP = [1.00, 1.05, 1.10, 1.12, 1.15, 1.20, 1.25, 1.30]          # 仅 restep


def arm_defs():
    arms = []
    for k in GRID_U:
        arms.append(dict(id=f"unif_k{k:.2f}", kappa_onset=k, kappa_restep=k, family="uniform", kappa=k))
    for k in GRID_ONSET:
        arms.append(dict(id=f"onset_k{k:.2f}", kappa_onset=k, kappa_restep=1.12,
                         family="onset_only", kappa=k))
    for k in GRID_RESTEP:
        arms.append(dict(id=f"restep_k{k:.2f}", kappa_onset=1.30, kappa_restep=k,
                         family="restep_only", kappa=k))
    return arms


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    arms = arm_defs()
    dyn = {k: float(np.percentile(d["Z"], 99) - np.percentile(d["Z"], 1))
           for k, d in recs.items()}
    rec(f"事件 {len(ev)}（onset {int((ev.kind=='onset').sum())} / restep "
        f"{int((ev.kind=='restep').sum())}）；录制 {len(recs)}；臂 {len(arms)}")
    rec(f"总录制级推理次数 = {len(arms) * len(recs)}")

    per_event = []
    for ai, arm in enumerate(arms, 1):
        rows = []
        for k, d in recs.items():
            sub = ev[ev.key == k]
            if not len(sub):
                continue
            out = C.run_v6(d, kappa_onset=arm["kappa_onset"],
                           kappa_restep=arm["kappa_restep"])
            Ys = out["Y"].sum(axis=1)
            Zs = d["Z"]
            for _, e in sub.iterrows():
                m = C.metrics_at(Zs, Ys, d["tu"], float(e["t_on"]), d["span"])
                m.update(key=e["key"], rec=e["rec"], t_on=float(e["t_on"]),
                         kind=e["kind"], dom=d["dom"], clean=bool(e["clean"]),
                         dyn_range=dyn[k])
                rows.append(m)
        a = pd.DataFrame(rows)
        a["J_frac"] = a["J"].abs() / a["dyn_range"]
        a["arm"] = arm["id"]
        a["family"] = arm["family"]
        a["kappa_onset"] = arm["kappa_onset"]
        a["kappa_restep"] = arm["kappa_restep"]
        per_event.append(a)
        L = a[a.kind.isin(["onset", "restep"]) & (a.J_frac >= 0.05)]
        rec(f"  [{ai:2d}/{len(arms)}] {arm['id']:14s} n={len(a)} (加载类 {len(L)})  "
            f"OS%中位={L.OS_pct.median():7.3f}  max={L.OS_pct.max():7.3f}  "
            f"T_med={L.T_stable.median()}")
    pe = pd.concat(per_event, ignore_index=True)
    pe.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_perevent.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_kappa_perevent.csv")

    # 主汇总口径：加载类（onset+restep）且 J_frac ≥ 0.05（归一化稳定）
    pe["in_main"] = pe.kind.isin(["onset", "restep"]) & (pe.J_frac >= 0.05)
    PE = pe[pe.in_main].copy()
    rec(f"主口径样本：{len(PE)} 事件（{PE.arm.nunique()} 臂 × 每臂 "
        f"{len(PE)//PE.arm.nunique()} 事件）")

    # ── 汇总：逐臂逐类（主口径 = 加载类且 J_frac ≥ 0.05；另附全样本行 kind='ALL_FULL'）──
    rows = []
    sweep_src = pd.concat([PE, pe.assign(kind="ALL_FULL")], ignore_index=True)
    for (arm, fam, ko, kr, kind), g in sweep_src.groupby(
            ["arm", "family", "kappa_onset", "kappa_restep", "kind"], sort=False):
        if not len(g):
            continue
        rows.append(dict(
            arm=arm, family=fam, kappa_onset=ko, kappa_restep=kr, kind=kind, n=len(g),
            OS_med=g.OS_pct.median(), OS_p10=g.OS_pct.quantile(.10),
            OS_p90=g.OS_pct.quantile(.90), OS_max=g.OS_pct.max(), OS_min=g.OS_pct.min(),
            n_os_gt5=int((g.OS_pct > 5).sum()), n_os_gt10=int((g.OS_pct > 10).sum()),
            US_med=g.US_pct.median(), US_max=g.US_pct.max(),
            T_med=g.T_stable.median(), T_p90=g.T_stable.quantile(.90),
            T_nan=int(g.T_stable.isna().sum()),
            err1_med=g.err_1s_pct.median(), err1_p10=g.err_1s_pct.quantile(.10),
            err1_p90=g.err_1s_pct.quantile(.90),
            MD_med=g.MD.median(), MD_p90=g.MD.quantile(.90), MD_max=g.MD.max(),
            G_med=g.G20.median(), G_min=g.G20.min(),
            J_med=g.J.abs().median(),
        ))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_sweep.csv"),
                 index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_kappa_sweep.csv")

    # ── 汇总：一行一工作点（加载类合并 + 分 onset/restep）──
    srows = []
    for (arm, fam, ko, kr), g in PE.groupby(
            ["arm", "family", "kappa_onset", "kappa_restep"], sort=False):
        ld = g
        on = g[g.kind == "onset"]
        re_ = g[g.kind == "restep"]
        srows.append(dict(
            arm=arm, family=fam, kappa_onset=ko, kappa_restep=kr,
            n_load=len(ld), n_onset=len(on), n_restep=len(re_),
            OS_med_load=ld.OS_pct.median(), OS_p90_load=ld.OS_pct.quantile(.90),
            OS_max_load=ld.OS_pct.max(), OS_med_onset=on.OS_pct.median(),
            OS_max_onset=on.OS_pct.max(), OS_med_restep=re_.OS_pct.median(),
            OS_max_restep=re_.OS_pct.max(),
            US_med_load=ld.US_pct.median(), US_max_load=ld.US_pct.max(),
            T_med_load=ld.T_stable.median(), T_p90_load=ld.T_stable.quantile(.90),
            T_med_onset=on.T_stable.median(), T_med_restep=re_.T_stable.median(),
            err1_med_load=ld.err_1s_pct.median(), err1_abs_med_load=ld.err_1s_pct.abs().median(),
            err1_med_onset=on.err_1s_pct.median(),
            MD_med_load=ld.MD.median(), MD_p90_load=ld.MD.quantile(.90),
            G_med_load=ld.G20.median(), G_min_load=ld.G20.min(),
        ))
    summ = pd.DataFrame(srows)
    # 与基线（现状 onset 1.30 / restep 1.12）的配对差
    base = PE[(PE.kappa_onset == 1.30) & (PE.kappa_restep == 1.12)].set_index(
        ["key", "t_on", "kind"])
    rec(f"基线臂（onset κ=1.30 / restep κ=1.12）主口径事件数 = {len(base)}")
    d_rows = []
    for (arm, fam, ko, kr), g in PE.groupby(
            ["arm", "family", "kappa_onset", "kappa_restep"], sort=False):
        gi = g.set_index(["key", "t_on", "kind"])
        common = gi.index.intersection(base.index)
        dOS = (gi.loc[common, "OS_pct"] - base.loc[common, "OS_pct"])
        dT = (gi.loc[common, "T_stable"] - base.loc[common, "T_stable"])
        dE = (gi.loc[common, "err_1s_pct"] - base.loc[common, "err_1s_pct"])
        dM = (gi.loc[common, "MD"] - base.loc[common, "MD"])
        d_rows.append(dict(arm=arm, family=fam, kappa_onset=ko, kappa_restep=kr,
                           n_pair=len(common),
                           dOS_med=dOS.median(), dOS_absmax=dOS.abs().max(),
                           dOS_max=dOS.max(),
                           dT_med=dT.median(), dT_absmax=dT.abs().max(),
                           dE_med=dE.median(), dE_absmax=dE.abs().max(),
                           dMD_med=dM.median(), dMD_absmax=dM.abs().max(),
                           n_OS_identical=int((dOS.abs() < 1e-12).sum()),
                           n_T_identical=int((dT.fillna(0).abs() < 1e-12).sum()),
                           n_MD_identical=int((dM.abs() < 1e-9).sum())))
    dd = pd.DataFrame(d_rows)
    summ = summ.merge(dd.drop(columns=["family", "kappa_onset", "kappa_restep"]),
                      on="arm", how="left")
    summ.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_summary.csv"),
                index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_kappa_summary.csv")

    # ── 留一泛化 ──
    loo = []
    for _, r in summ.iterrows():
        family, k = r["family"], r["kappa_onset"]
        g = PE[PE.arm == r["arm"]]
        for kk, gg in g.groupby("key"):
            held = gg
            loo.append(dict(arm=r["arm"], family=family, kappa=k, held_out=kk,
                            n=len(held), OS_med=held.OS_pct.median(),
                            OS_max=held.OS_pct.max(), T_med=held.T_stable.median(),
                            err1_med=held.err_1s_pct.median()))
    lo = pd.DataFrame(loo)
    lo.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_loo_raw.csv"),
              index=False, encoding="utf-8-sig")
    agg = []
    for (arm, family, k), g in lo.groupby(["arm", "family", "kappa"]):
        agg.append(dict(arm=arm, family=family, kappa=k, n_folds=len(g),
                        OS_med_fold_min=g.OS_med.min(), OS_med_fold_max=g.OS_med.max(),
                        OS_med_fold_spread=g.OS_med.max() - g.OS_med.min(),
                        OS_med_fold_std=g.OS_med.std(),
                        OS_max_worst_fold=g.OS_max.max(),
                        T_med_fold_max=g.T_med.max(),
                        err1_abs_med_fold_max=g.err1_med.abs().max()))
    la = pd.DataFrame(agg)
    la.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_loo.csv"),
              index=False, encoding="utf-8-sig")
    rec("产出 results/t5a_kappa_loo.csv")

    # ── 控制台摘要：三个子实验的关键行 ──
    lines = ["\n===== C-4 统一 κ 扫描摘要（加载类 onset+restep，n 见列）====="]
    hdr = (f"{'arm':14s} {'κ_on':>5s} {'κ_re':>5s} {'OS_med':>8s} {'OS_max':>8s} "
           f"{'T_med':>7s} {'err1_med':>9s} {'MD_med':>9s} {'dOS_med':>8s} {'dOS_absmax':>10s}")
    lines.append(hdr)
    for _, r in summ[summ.family == "uniform"].iterrows():
        lines.append(f"{r['arm']:14s} {r['kappa_onset']:5.2f} {r['kappa_restep']:5.2f} "
                     f"{r['OS_med_load']:8.3f} {r['OS_max_load']:8.3f} "
                     f"{r['T_med_load']:7.3f} {r['err1_med_load']:9.3f} "
                     f"{r['MD_med_load']:9.1f} {r['dOS_med']:8.3f} {r['dOS_absmax']:10.3f}")
    lines.append("--- 仅 restep 变（κ_onset 固定 1.30）---")
    for _, r in summ[summ.family == "restep_only"].iterrows():
        lines.append(f"{r['arm']:14s} {r['kappa_onset']:5.2f} {r['kappa_restep']:5.2f} "
                     f"{r['OS_med_load']:8.3f} {r['OS_max_load']:8.3f} "
                     f"{r['T_med_load']:7.3f} {r['err1_med_load']:9.3f} "
                     f"{r['MD_med_load']:9.1f} {r['dOS_med']:8.3f} {r['dOS_absmax']:10.3f}")
    lines.append("--- 仅 onset 变（κ_restep 固定 1.12）---")
    for _, r in summ[summ.family == "onset_only"].iterrows():
        lines.append(f"{r['arm']:14s} {r['kappa_onset']:5.2f} {r['kappa_restep']:5.2f} "
                     f"{r['OS_med_load']:8.3f} {r['OS_max_load']:8.3f} "
                     f"{r['T_med_load']:7.3f} {r['err1_med_load']:9.3f} "
                     f"{r['MD_med_load']:9.1f} {r['dOS_med']:8.3f} {r['dOS_absmax']:10.3f}")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_kappa.log"),
                "python scripts/t5a_kappa.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
