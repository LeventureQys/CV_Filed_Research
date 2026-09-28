# -*- coding: utf-8 -*-
"""T7-A / step0：数据与事件盘点（不跑算法）。

目的：
  1. 冒烟自检：13 份录制的帧数/通道数/域/时间轴网格/包间隔，以及与既有清单是否一致；
  2. 域检查：原始读数是否存在负值、是否被钳在 0（决定"回零过冲"能否观测）；
  3. 用**冻结定义**检测卸载事件，与 T4-A 的 60 事件冻结表、13-v6-assessment 的
     `b_unload_raw.csv`（21 处）逐一对齐（这是增量起点，禁止重做）。

产出：results/t7_unload_recon.csv、results/_t7a_0_recon.log
用法：python scripts/t7a_0_recon.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402

LOG = C.Log("0_recon")


def main():
    C.log_reconfigure()
    LOG("=" * 118)
    LOG("T7-A step0 盘点：13 份录制（恒载 9 = display/force 域；实录 4 = ADC 域）")
    LOG("口径：timestamp 列、100 Hz 均匀网格(dt=0.01s)、Z=Σch、Z̄=median(Z,0.5s)；"
        "卸载定义=减少量≥80%当前电平(unload)/20~80%(partial_unload)，且前后各 2 s 稳定")
    LOG("=" * 118)
    LOG(f"{'录制':<18}{'家族':<5}{'nch':>4}{'帧数':>7}{'span/s':>8}{'包间隔/ms':>10}"
        f"{'min(X)':>9}{'X≤0占比':>9}{'底':>9}{'峰':>10}")

    rows = []
    for tag, path in C.ALL:
        if not os.path.exists(path):
            LOG(f"[缺文件] {tag} -> {path}")
            continue
        d = C.prep(tag)
        ds = C.zbar(d["tot"], d["dtm"])
        floor, peak = float(np.percentile(ds, 5)), float(np.percentile(ds, 99.5))
        acc, allc, edges, _ = C.detect_unloads(d["tu"], d["tot"], d["dtm"], d["pkt_dt"])
        nf = sum(1 for r in allc if r["kind"] == "unload" and r["stable"])
        np_ = sum(1 for r in allc if r["kind"] == "partial_unload" and r["stable"])
        nfa = sum(1 for r in allc if r["kind_adapt"] == "unload" and r["stable_adapt"])
        LOG(f"{tag:<18}{d['kind']:<5}{d['nch']:>4}{len(d['t']):>7}{d['span']:>8.1f}"
            f"{d['pkt_dt']*1000:>10.2f}{d['xmin']:>9.3f}{d['xzero_frac']:>9.4f}"
            f"{floor:>9.2f}{peak:>10.1f}   严格集 卸载={nf} 部分={np_} / 自适应集 卸载={nfa} "
            f"(边沿候选 {len(edges)})")
        for r in allc:
            rows.append(dict(rec=tag, kind_rec=d["kind"], dom=d["dom"], nch=d["nch"], **r))
    df = pd.DataFrame(rows).drop(columns=["i0", "i_cand"], errors="ignore")
    C.save(df, "t7_unload_recon.csv")

    # ── 与既有事件表对齐 ──
    LOG("")
    LOG("=" * 118)
    LOG("与既有事件表对齐（只比对时刻；本任务的 t0 = 卸载沿起点 = 最后一个仍在旧电平 10% 带内的帧）")
    LOG("=" * 118)
    t4a = C.load_t4a_morph()
    t4u = t4a[t4a["kind"].isin(["unload", "partial_unload"])].copy()
    mine = df[df["kind"].isin(["unload", "partial_unload"]) & df["stable"]].copy()
    LOG(f"T4-A 冻结表：卸载 {int((t4u['kind']=='unload').sum())} 个 + "
        f"部分卸载 {int((t4u['kind']=='partial_unload').sum())} 个")
    LOG(f"本任务（冻结定义 + 稳定判据）：卸载 {int((mine['kind']=='unload').sum())} 个 + "
        f"部分卸载 {int((mine['kind']=='partial_unload').sum())} 个")
    LOG("")
    LOG(f"{'T4-A rec':<18}{'T4-A t_on':>10}{'T4-A kind':>15}{'本任务 t0':>10}{'Δt/s':>8}"
        f"{'本任务 kind':>15}{'step_frac':>10}")
    pairs = []
    for rec, grp in t4u.groupby("rec"):
        m = mine[mine["rec"] == rec]
        for _, r in grp.iterrows():
            if len(m) == 0:
                LOG(f"{rec:<18}{r['t_on']:>10.2f}{r['kind']:>15}{'—':>10}{'—':>8}{'—':>15}{'—':>10}")
                continue
            cands = df[df["rec"] == rec]
            j = int(np.argmin(np.abs(cands["t0"].to_numpy(float) - float(r["t_on"]))))
            mm = cands.iloc[j]
            dt = float(mm["t0"]) - float(r["t_on"])
            if abs(dt) > 3.0:
                LOG(f"{rec:<18}{r['t_on']:>10.2f}{r['kind']:>15}{'—(3 s 内无对应)':>10}"
                    f"{'—':>8}{'—':>15}{'—':>10}")
                continue
            pairs.append((rec, float(r["t_on"]), r["kind"], float(mm["t0"]), dt, mm["kind"]))
            LOG(f"{rec:<18}{r['t_on']:>10.2f}{r['kind']:>15}{mm['t0']:>10.2f}{dt:>8.2f}"
                f"{mm['kind']:>15}{mm['step_frac']:>10.4f}")
    if pairs:
        a = np.array([abs(p[4]) for p in pairs])
        LOG(f"\n配对 n={len(pairs)}；|Δt0| 中位 {np.median(a):.2f} s、最大 {a.max():.2f} s"
            f"（Δ = 本任务 t0 − T4-A t_on）")
        ag = np.array([1 if p[2] == p[5] else 0 for p in pairs])
        LOG(f"kind 一致率 {ag.mean()*100:.0f}%（{int(ag.sum())}/{len(ag)}）")

    # ── 与 b_unload_raw.csv 对齐（第一轮 21 处）──
    bp = os.path.join(C.PROG, "13-v6-assessment", "results", "b_unload_raw.csv")
    if os.path.exists(bp):
        b = pd.read_csv(bp)
        LOG("")
        LOG("=" * 118)
        LOG(f"与第一轮 b_unload_raw.csv（{len(b)} 处）对齐："
            f"第一轮 full={(b['ev_class']=='full').sum()} partial={(b['ev_class']=='partial').sum()}")
        LOG("=" * 118)
        LOG(f"{'rec':<18}{'b t_dn':>9}{'b class':>9}{'本任务 t0':>10}{'Δt/s':>8}"
            f"{'本任务 kind':>15}{'stable':>7}{'stab_pre':>9}{'stab_post':>10}")
        pairs2 = []
        for _, r in b.iterrows():
            m = df[(df["rec"] == r["rec"])]
            if len(m) == 0:
                continue
            j = int(np.argmin(np.abs(m["t0"].to_numpy(float) - float(r["t_dn"]))))
            mm = m.iloc[j]
            dt = float(mm["t0"]) - float(r["t_dn"])
            if abs(dt) > 3.0:
                LOG(f"{r['rec']:<18}{r['t_dn']:>9.2f}{r['ev_class']:>9}"
                    f"{'—(无对应)':>10}{'—':>8}{'—':>15}{'—':>7}{'—':>9}{'—':>10}")
                continue
            pairs2.append((r["rec"], float(r["t_dn"]), r["ev_class"], float(mm["t0"]), dt,
                           mm["kind"], bool(mm["stable"])))
            LOG(f"{r['rec']:<18}{r['t_dn']:>9.2f}{r['ev_class']:>9}{mm['t0']:>10.2f}{dt:>8.2f}"
                f"{mm['kind']:>15}{str(bool(mm['stable'])):>7}{mm['stab_pre']:>9.4f}"
                f"{mm['stab_post']:>10.4f}")
        if pairs2:
            a = np.array([abs(p[4]) for p in pairs2])
            LOG(f"\n配对 n={len(pairs2)}；|Δt0| 中位 {np.median(a):.2f} s、最大 {a.max():.2f} s")
            agree = sum(1 for p in pairs2 if (p[2] == "full") == (p[5] == "unload"))
            LOG(f"full/unload 与 partial/partial_unload 一致：{agree}/{len(pairs2)}")
            ns = sum(1 for p in pairs2 if not p[6])
            LOG(f"其中不满足「前后各 2 s 稳定」的：{ns}/{len(pairs2)}（冻结定义会剔除）")

    LOG("")
    LOG("=" * 118)
    LOG("严格集 vs 自适应后窗集（post 窗被 6 s 内下一次变载污染时，严格判据剔除、自适应集保留）")
    LOG("=" * 118)
    LOG(f"{'rec':<18}{'t0':>8}{'step_frac':>10}{'stab_pre':>9}{'stab_post':>10}"
        f"{'kind(严格)':>14}{'step_f(自适应)':>14}{'stab_p(自适应)':>14}{'kind(自适应)':>15}")
    rec_rows = df[df["step_frac_adapt"] >= 0.20]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for _, r in rec_rows.iterrows():
            LOG(f"{r['rec']:<18}{r['t0']:>8.2f}{r['step_frac']:>10.4f}{r['stab_pre']:>9.4f}"
                f"{r['stab_post']:>10.4f}{r['kind']:>14}{r['step_frac_adapt']:>14.4f}"
                f"{r['stab_post_adapt']:>14.4f}{r['kind_adapt']:>15}")
    LOG("")
    LOG("被严格定义剔除、但自适应集可保留的事件（本任务的敏感性对照集）：")
    extra = df[(df["step_frac"] >= 0.20) & (~df["stable"]) & (df["stable_adapt"])]
    if len(extra):
        LOG(extra[["rec", "t0", "step_frac", "stab_pre", "stab_post", "kind_adapt"]]
            .round(4).to_string(index=False))
    else:
        LOG("  （无）")
    LOG("")
    LOG("幅度不足 20%（不满足指标字典 partial_unload 定义，属 restep/小减重）：")
    small = df[df["step_frac"] < 0.20]
    LOG(small[["rec", "t0", "step_frac", "stab_pre", "stab_post"]].round(4).to_string(index=False))
    LOG("")
    LOG("域检查结论（X = 逐通道原始读数）：")
    for tag, path in C.ALL:
        if not os.path.exists(path):
            continue
        d = C.prep(tag)
        LOG(f"  {tag:<18} min(X)={d['xmin']:>9.3f}  X≤0 占比={d['xzero_frac']*100:>7.3f}%  "
            f"（{'存在负值→可观测"回到零以下"' if d['xmin'] < -1e-9 else '无负值→需判断是否为域钳位'}）")
    LOG.close("python scripts/t7a_0_recon.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
