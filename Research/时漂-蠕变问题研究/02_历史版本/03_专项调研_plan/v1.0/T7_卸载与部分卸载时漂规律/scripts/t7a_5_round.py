# -*- coding: utf-8 -*-
"""T7-A / step5：轮次偏差（T7-Q5）——第一轮的残余是否影响第二/第三轮。

本批数据没有"同一标称载荷重复加载"的样本（手工变载，每轮载荷都不同），
因此**不能**直接算指标字典的 Δ_rep = (两次 Z_final 之差)/载荷；
本脚本给三件能算的东西，并明确标出不能算的部分：

  1. **零点传递 carry-over**：第 k 轮卸载后的零点读数 z_after_k 与第 k+1 轮加载前的
     零点读数 z_before_{k+1} 之差（两者是同一次实测的相邻量，差=空载期内的零点漂移）；
  2. **轮次间的表观幅度**：|J_k| vs |J_{k+1}|（含相对偏差），并把"表观幅度差"拆成
     「零点基线平移」+「操作载荷差」两部分中**可测的那部分**（前者）；
  3. **同载荷近邻对**：|J| 相对差 ≤10% 的轮次对上的轮次偏差（作为 Δ_rep 的近似上界）。

产出：results/t7_round_bias.csv、results/_t7a_5_round.log
用法：python scripts/t7a_5_round.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402

LOG = C.Log("5_round")


def main():
    C.log_reconfigure()
    rd = pd.read_csv(os.path.join(C.RES, "t7_zero_rounds.csv"))
    ev = pd.read_csv(os.path.join(C.RES, "t7_unload_events.csv"))
    fo = pd.read_csv(os.path.join(C.RES, "t7_frozen_offset.csv"))
    LOG("=" * 126)
    LOG("T7-A step5：轮次偏差（T7-Q5）")
    LOG("数据限制：本批 13 份录制中没有任何「同一标称载荷重复加载」的样本（手工变载，"
        "每轮载荷都不同）⇒ 指标字典的 Δ_rep 无法直接计算，只能给下列可测量。")
    LOG("=" * 126)
    rows = []
    for tag, sub in rd.groupby("rec"):
        sub = sub.sort_values("round_idx")
        for k in range(len(sub) - 1):
            a, b = sub.iloc[k], sub.iloc[k + 1]
            if not (np.isfinite(a["z_after_raw"]) and np.isfinite(b["z_before_raw"])):
                continue
            Jk = a["plat_drop"]
            Jk1 = b["plat_drop"]
            evk = ev[(ev["rec"] == tag) & (ev["round_idx"] == int(a["round_idx"]))]
            key = evk["key"].iloc[0] if len(evk) else None
            tj = fo[fo["key"] == key].sort_values("tau") if key else pd.DataFrame()

            def rt(x):
                if len(tj) == 0:
                    return np.nan
                r = tj.iloc[int(np.argmin(np.abs(tj["tau"].to_numpy(float) - x)))]
                return float(r["resid_raw"])
            r2, r4, rend = rt(2.0), rt(4.0), (float(tj["resid_raw"].iloc[-1]) if len(tj) else np.nan)
            rows.append(dict(
                rec=tag, kind_rec=a["kind_rec"], pair=f"{int(a['round_idx'])}->{int(b['round_idx'])}",
                t_gap_start=a["t_end"], t_gap_end=b["t_start"],
                gap_s=float(b["t_start"] - a["t_end"]),
                plat_k=a["plat_level"], plat_k1=b["plat_level"],
                J_k=Jk, J_k1=Jk1, dJ=Jk1 - Jk, dJ_rel=(Jk1 - Jk) / Jk if abs(Jk) > 1e-9 else np.nan,
                z_after_k=a["z_after_raw"], z_before_k1=b["z_before_raw"],
                gap_resid_t2=r2, gap_resid_t4=r4, gap_resid_end=rend,
                gap_recovery_pct=(100.0 * (r4 - rend) / r4 if abs(r4) > 1e-12 else np.nan),
                resid_transfer_frac=(rend / r4 if abs(r4) > 1e-12 else np.nan),
                carry_pct_of_J=(rend / abs(Jk) if abs(Jk) > 1e-9 else np.nan),
                dZ_rel_to_zero=(rend / abs(a["z_after_raw"]) if abs(a["z_after_raw"]) > 1e-9
                                else np.nan),
                zero_shift_vs_first=b["z_before_raw"] - sub.iloc[0]["z_before_raw"],
                zero_shift_vs_first_rel=((b["z_before_raw"] - sub.iloc[0]["z_before_raw"])
                                         / abs(sub.iloc[0]["z_before_raw"])
                                         if abs(sub.iloc[0]["z_before_raw"]) > 1e-9 else np.nan),
                near_same_load=bool(abs((Jk1 - Jk) / Jk) <= 0.10) if abs(Jk) > 1e-9 else False,
                hold_k=a["dur_s"], hold_k1=b["dur_s"]))
    df = pd.DataFrame(rows)
    C.save(df, "t7_round_bias.csv")
    LOG("")
    cols = ["rec", "kind_rec", "pair", "gap_s", "plat_k", "plat_k1", "J_k", "J_k1", "dJ_rel",
            "gap_resid_t2", "gap_resid_t4", "gap_resid_end", "gap_recovery_pct",
            "resid_transfer_frac", "carry_pct_of_J", "dZ_rel_to_zero",
            "zero_shift_vs_first", "zero_shift_vs_first_rel", "near_same_load"]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        LOG(df[cols].round(5).to_string(index=False))
    LOG("")
    LOG("【1】第 k 轮卸载后的残余在**空载期内的回复**（残余轨迹取 t7_frozen_offset.csv，n=%d）：" % len(df))
    for k, nm in ((("gap_resid_t2"), "τ=2 s"), (("gap_resid_t4"), "τ=4 s"),
                  (("gap_resid_end"), "空载期末")):
        x = df[k].to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x):
            LOG(f"     {nm:<8} 中位 {np.median(x):+10.4f}  p10~p90 {np.percentile(x,10):+10.4f} ~ "
                f"{np.percentile(x,90):+10.4f}")
    x = df["gap_recovery_pct"].to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x):
        LOG(f"     空载期内回复比例（τ=4 s→期末）中位 {np.median(x):.1f}%  "
            f"p10~p90 {np.percentile(x,10):.1f}% ~ {np.percentile(x,90):.1f}%")
    LOG("     ⇒ 残余在空载期内只回复一部分；**到下一轮加载时，剩下的部分就是新的起点基线**"
        "（读数不会在加载瞬间被归零）。")
    LOG("")
    LOG("【2】轮次间表观幅度差 dJ = |J_k+1| − |J_k|（相对前一轮幅度）：")
    x = df["dJ_rel"].to_numpy(float)
    x = x[np.isfinite(x)]
    LOG(f"     中位 {np.median(x)*100:+.2f}%  p10~p90 {np.percentile(x,10)*100:+.2f}% ~ "
        f"{np.percentile(x,90)*100:+.2f}%  （n={len(x)}）")
    LOG("     注意：dJ 里既有「操作者这次压得比上次重/轻」，也有「零点基线平移」；"
        "本批无标称载荷标注 ⇒ **无法把两者分开**（这是本任务最主要的数据缺口之一）。")
    LOG("")
    LOG("【3】同载荷近邻对（|dJ/J| ≤10%，可作为 Δ_rep 的近似）：")
    ns = df[df["near_same_load"]]
    if len(ns):
        LOG(ns[["rec", "pair", "J_k", "J_k1", "dJ_rel", "zero_shift_vs_first"]]
            .round(5).to_string(index=False))
    else:
        LOG("     （本批无此类配对：所有相邻两轮的 |J| 相对差都 >10%）")
    LOG("")
    LOG("【4】轮次偏差（零点累计平移 vs 首轮零点，相对零点自身）：")
    for tag, sub in df.groupby("rec"):
        LOG(f"     {tag:<18} " + "  ".join(
            f"{r['pair']}:{r['zero_shift_vs_first']:+.3f}({r['zero_shift_vs_first_rel']*100:+.1f}%)"
            for _, r in sub.iterrows()))
    LOG.close("python scripts/t7a_5_round.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
