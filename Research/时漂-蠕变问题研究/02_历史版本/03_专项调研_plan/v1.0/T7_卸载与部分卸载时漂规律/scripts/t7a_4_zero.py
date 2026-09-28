# -*- coding: utf-8 -*-
"""T7-A / step4：**零基线漂移 P3**（T7-Q9，本任务核心交付）。

问题：多轮加卸载后"零点读数"随轮次怎么漂？是否与累计加载时长/累计载荷相关？
**两条线必须分开看**（否则分不清传感器零点漂移 vs 算法零点处理）：
  * `*_raw`  = 原始读数（不含任何补偿）→ 传感器/采集侧的真实零点行为；
  * `*_e3s` / `*_v6` = 现役 v5.1 与 v6 的补偿后读数；
  * `*_v5`   = v5（含"空载自动归零" b_ 链路，v5.1 已删除）→ 反事实对照。

零点读数定义（每个空载平台 = 一段 Z̄ 低于「底 + 10%·(峰−底)」且长 ≥0.6 s 的区间）：
  取平台内**跳过起始 0.5 s 瞬态后的最后 3 s 中位**作为该平台的零点读数；
另给出"整段中位"与"平台内 p10~p90"作为稳健性。

产出：results/t7_zero_drift.csv（长表：录制 × 平台 × 臂）
      results/t7_zero_corr.csv（相关性：Δ零点 vs 累计加载时长/累计载荷/轮次/总时长）
      results/t7_zero_rounds.csv（每轮汇总）
      results/_t7a_4_zero.log
用法：python scripts/t7a_4_zero.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402

LOG = C.Log("4_zero")


def main():
    C.log_reconfigure()
    LOG("=" * 130)
    LOG("T7-A step4：零基线漂移 P3（T7-Q9）——原始读数 vs 补偿后读数**两条线**")
    LOG("口径：零点读数 = 空载平台内（跳过起始 0.5 s）最后 3 s 的 Z̄ 中位；"
        "Δ零点 = 本轮零点 − 首轮零点（同录制内）")
    LOG("=" * 130)
    rows, rounds_corr = [], []
    cum = {}                                        # 每份录制的累计量
    for tag, path in C.ALL:
        if not os.path.exists(path):
            LOG(f"[缺文件] {tag}")
            continue
        d = C.prep(tag)
        z = C.load_rec_cache(tag)
        tu, dtm = d["tu"], d["dtm"]
        n = len(tu)
        Zb = {"raw": C.med_smooth(d["tot"], int(round(0.5 / dtm)))}
        for a in ("e3s", "v6", "v5"):
            Zb[a] = C.med_smooth(z["Y_" + a].astype(float), int(round(0.5 / dtm)))
        ds = Zb["raw"]
        gaps, eps, floor, peak = C.parse_plateaus(ds, tu, dtm)
        # 轮次：受载段序号 + 该轮起点前的累计加载时长/载荷
        cum_t = 0.0
        cum_i = 0.0
        ep_meta = []
        for k, e in enumerate(eps):
            lvl, pk = C.episode_level(ds, e, dtm)
            z0 = np.nan
            gp = [g for g in gaps if g["i1"] <= e["i0"]]
            if gp:
                z0 = C.zero_level(ds, gp[-1], dtm, 0.5, 3.0)[0]
            dur = (e["i1"] - e["i0"]) * dtm
            integ = float(np.sum(np.clip(ds[e["i0"]:e["i1"]] - (z0 if np.isfinite(z0) else floor),
                                         0, None)) * dtm)
            ep_meta.append(dict(round_idx=k + 1, ep=e, lvl=lvl, peak=pk, z0_pre=z0,
                                dur=dur, integ=integ,
                                cum_t_prev=cum_t, cum_i_prev=cum_i))
            cum_t += dur
            cum_i += integ
        key0 = {}
        for j, g in enumerate(gaps):
            zl = 3.0
            zref, nsel = C.zero_level(ds, g, dtm, 0.5, zl)
            zall = float(np.median(ds[g["i0"]:g["i1"]]))
            p10, p90 = C.p10p90(ds[g["i0"]:g["i1"]])
            nxt = next((m for m in ep_meta if m["ep"]["i0"] >= g["i1"]), None)
            prev = next((m for m in reversed(ep_meta) if m["ep"]["i1"] <= g["i0"]), None)
            rr = dict(rec=tag, kind_rec=("恒载" if "指尖" in tag else "实采"), dom=d["dom"],
                      gap_idx=j, t0=g["t0"], t1=g["t1"], dur_s=g["t1"] - g["t0"],
                      level_raw=zl, p10=float(p10), p90=float(p90),
                      z_med_all=zall, z_select_s=zl, n_sel=int(nsel),
                      after_round=(prev["round_idx"] if prev else 0),
                      before_round=(nxt["round_idx"] if nxt else 0),
                      cum_load_t_s=(prev["cum_t_prev"] + prev["dur"] if prev else 0.0),
                      cum_load_integ=(prev["cum_i_prev"] + prev["integ"] if prev else 0.0),
                      n_prior_unloads=(prev["round_idx"] if prev else 0))
            for a in ("raw", "e3s", "v6", "v5"):
                zv, _ = C.zero_level(Zb[a], g, dtm, 0.5, zl)
                rr[f"z_{a}"] = zv
                rr[f"zall_{a}"] = float(np.median(Zb[a][g["i0"]:g["i1"]]))
                rr[f"z_{a}_minus_raw"] = zv - zref
            rows.append(rr)
        # 首轮零点参考 + 逐平台 Δ
        base = next((r for r in rows if r["rec"] == tag and r["after_round"] >= 0), None)
        z1 = base["z_raw"] if base else np.nan
        for r in rows:
            if r["rec"] != tag:
                continue
            for a in ("raw", "e3s", "v6", "v5"):
                r[f"dz_{a}"] = r[f"z_{a}"] - z1
        # 每轮汇总（该轮前后的零点）
        for m in ep_meta:
            g_after = [r for r in rows if r["rec"] == tag and r["after_round"] == m["round_idx"]]
            g_before = [r for r in rows if r["rec"] == tag and r["before_round"] == m["round_idx"]
                        and r["after_round"] != m["round_idx"]]
            zb = g_before[0] if g_before else None      # 本轮**前**那个空载平台
            za = g_after[0] if g_after else None        # 本轮**后**那个空载平台
            for nm, obj in (("z_before", zb), ("z_after", za)):
                pass
            rounds_corr.append(dict(
                rec=tag, kind_rec=("恒载" if "指尖" in tag else "实采"),
                round_idx=m["round_idx"], t_start=m["ep"]["t0"], t_end=m["ep"]["t1"],
                dur_s=m["dur"], load_integ=m["integ"], plat_level=m["lvl"], peak=m["peak"],
                cum_load_t_s=m["cum_t_prev"], cum_load_integ=m["cum_i_prev"],
                z_before_raw=(zb["z_raw"] if zb is not None else np.nan),
                z_after_raw=(za["z_raw"] if za is not None else np.nan),
                z_before_e3s=(zb["z_e3s"] if zb is not None else np.nan),
                z_after_e3s=(za["z_e3s"] if za is not None else np.nan),
                z_before_v6=(zb["z_v6"] if zb is not None else np.nan),
                z_after_v6=(za["z_v6"] if za is not None else np.nan),
                z_before_v5=(zb["z_v5"] if zb is not None else np.nan),
                z_after_v5=(za["z_v5"] if za is not None else np.nan),
                dz_raw=((za["z_raw"] - zb["z_raw"]) if (za is not None and zb is not None) else np.nan),
                dz_e3s=((za["z_e3s"] - zb["z_e3s"]) if (za is not None and zb is not None) else np.nan),
                dz_v6=((za["z_v6"] - zb["z_v6"]) if (za is not None and zb is not None) else np.nan),
                dz_v5=((za["z_v5"] - zb["z_v5"]) if (za is not None and zb is not None) else np.nan),
                dz_raw_rel=((za["z_raw"] - zb["z_raw"]) / zb["z_raw"]
                            if (za is not None and zb is not None and abs(zb["z_raw"]) > 1e-9)
                            else np.nan),
                hold_s=m["dur"], plat_drop=(m["lvl"] - (zb["z_raw"] if zb is not None else np.nan))))
    df = pd.DataFrame(rows)
    rr_ = pd.DataFrame(rounds_corr)
    C.save(df, "t7_zero_drift.csv")
    C.save(rr_, "t7_zero_rounds.csv")

    LOG("")
    LOG("─" * 130)
    LOG("【零点读数库存】逐空载平台（零点读数为 raw 柱；同录制的 '臂−raw' 用于区分传感器/算法）")
    LOG("─" * 130)
    cols = ["rec", "kind_rec", "gap_idx", "t0", "t1", "dur_s", "z_raw", "z_e3s", "z_v6", "z_v5",
            "z_e3s_minus_raw", "z_v6_minus_raw", "z_v5_minus_raw", "dz_raw",
            "after_round", "cum_load_t_s", "cum_load_integ", "n_prior_unloads"]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        LOG(df[cols].round(4).to_string(index=False))

    LOG("")
    LOG("─" * 130)
    LOG("【两条线的差】每个空载平台上「补偿后 − 原始」的零点差（原语：算法到底有没有动零点）")
    LOG("─" * 130)
    for a in ("e3s", "v6", "v5"):
        x = df[f"z_{a}_minus_raw"].to_numpy(float)
        x = x[np.isfinite(x)]
        LOG(f"   {a:<4} n={len(x):>3}  中位 {np.median(x):>+10.4f}  p10~p90 "
            f"{np.percentile(x,10):>+10.4f} ~ {np.percentile(x,90):>+10.4f}  "
            f"max|·| {np.max(np.abs(x)):>10.4f}")
    LOG("   ⇒ 若某臂的中位≈0 且 max|·| ≈ 0，则该臂在空载段**完全不改读数**（零点原样透传）。")

    LOG("")
    LOG("─" * 130)
    LOG("【逐轮】零点漂移与累计量的关系（dz = 本轮后零点 − 本轮前零点；两条线并报）")
    LOG("─" * 130)
    c2 = ["rec", "kind_rec", "round_idx", "t_start", "t_end", "dur_s", "plat_level",
          "z_before_raw", "z_after_raw", "dz_raw", "dz_e3s", "dz_v6", "dz_v5",
          "cum_load_t_s", "cum_load_integ"]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        LOG(rr_[c2].round(4).to_string(index=False))

    rr_["dz_raw_abs"] = rr_["dz_raw"].abs()
    rr_["dz_v5_abs"] = rr_["dz_v5"].abs()
    # 相关性
    corr = []
    feats = ("cum_load_t_s", "cum_load_integ", "round_idx", "dur_s", "plat_drop", "t_start")
    for tgt in ("dz_raw", "dz_raw_abs", "dz_raw_rel", "dz_e3s", "dz_v6", "dz_v5", "dz_v5_abs"):
        if tgt not in rr_.columns:
            continue
        for feat in feats:
            for nm, sub in (("全部", rr_), ("实录", rr_[rr_["kind_rec"] == "实采"]),
                            ("恒载", rr_[rr_["kind_rec"] == "恒载"])):
                st = C.spearman(sub[feat], sub[tgt])
                corr.append(dict(subset=nm, target=tgt, feature=feat, **st))
    cf = pd.DataFrame(corr)
    C.save(cf, "t7_zero_corr.csv")
    LOG("")
    LOG("─" * 130)
    LOG("【相关性】Δ零点 vs 累计加载时长 / 累计载荷 / 轮次 / 本轮时长 / 本轮幅度（只列 |ρ|>0.5 或 p<0.10）")
    LOG("─" * 130)
    sel = cf[(cf["p_rho"] < 0.10) | (cf["rho"].abs() > 0.5)]
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        LOG(sel.round(4).to_string(index=False) if len(sel) else "（无）")
    LOG("")
    LOG("★ 判据（区分「传感器零点漂移」与「算法伪零点」）——同一特征下的两条线对比：")
    for feat in ("plat_drop", "cum_load_integ", "round_idx", "t_start"):
        for sub_nm in ("全部", "实录"):
            a = cf[(cf["target"] == "dz_raw") & (cf["feature"] == feat) & (cf["subset"] == sub_nm)]
            b = cf[(cf["target"] == "dz_v5") & (cf["feature"] == feat) & (cf["subset"] == sub_nm)]
            if len(a) and len(b):
                LOG(f"   [{sub_nm}] {feat:<16} dz_raw(传感器): ρ={a.iloc[0]['rho']:+.3f} "
                    f"(p={a.iloc[0]['p_rho']:.3f}, n={int(a.iloc[0]['n'])})   |   "
                    f"dz_v5(算法伪零点): ρ={b.iloc[0]['rho']:+.3f} (p={b.iloc[0]['p_rho']:.3f})")
    LOG("")
    LOG("【按录制看趋势】同一录制内零点随时间单调性（raw 柱，Spearman vs t0）：")
    for tag, sub in df.groupby("rec"):
        sx = sub.sort_values("t0")
        st = C.spearman(sx["t0"], sx["z_raw"])
        st2 = C.spearman(sx["cum_load_integ"], sx["z_raw"])
        LOG(f"   {tag:<18} 平台数 {len(sx):>2}  零点范围 "
            f"{sx['z_raw'].min():>10.3f}~{sx['z_raw'].max():>10.3f}  "
            f"ρ(零点,时间)={st['rho']:>+6.3f} (p={st['p_rho']:.3f})  "
            f"ρ(零点,累计载荷)={st2['rho']:>+6.3f} (p={st2['p_rho']:.3f})")
    LOG("")
    LOG("【零点漂移的总量级】")
    LOG("  (a) 相对本轮载荷幅度 |Δ零点|/本轮幅度（跨域只能比相对量）：")
    rr_["dz_raw_pct"] = rr_["dz_raw"] / rr_["plat_drop"].abs()
    for nm, sub in (("全部", rr_), ("实录", rr_[rr_["kind_rec"] == "实采"]),
                    ("恒载", rr_[rr_["kind_rec"] == "恒载"])):
        x = sub["dz_raw_pct"].to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            LOG(f"   {nm}: 无有效值")
            continue
        LOG(f"   {nm:<4} n={len(x):>3}  中位 {np.median(x)*100:>+8.3f}%  p10~p90 "
            f"{np.percentile(x,10)*100:>+8.3f}% ~ {np.percentile(x,90)*100:>+8.3f}%  "
            f"max|·| {np.max(np.abs(x))*100:.3f}%")
    LOG("  (b) 相对零点自身（Δ零点/本轮前零点）——用户直接看到的就是这个量：")
    x = rr_["dz_raw_rel"].to_numpy(float)
    x = x[np.isfinite(x)]
    with pd.option_context("display.width", 200):
        LOG(rr_[["rec", "round_idx", "z_before_raw", "z_after_raw", "dz_raw", "dz_raw_rel"]]
            .round(4).to_string(index=False))
    if len(x):
        LOG(f"   n={len(x)}  中位 {np.median(x)*100:>+8.1f}%  p10~p90 "
            f"{np.percentile(x,10)*100:>+8.1f}% ~ {np.percentile(x,90)*100:>+8.1f}%  "
            f"max|·| {np.max(np.abs(x))*100:.1f}%")
    LOG.close("python scripts/t7a_4_zero.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
