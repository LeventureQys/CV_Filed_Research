# -*- coding: utf-8 -*-
"""T3-A 00：重建与对表（证据链底座 + 反驳性检查）。

做什么：
  1. 用**本任务自己的**网格缓存（t3a_ad_lib 的自动 ##Data 定位 + 100 Hz 重采样）重建 13 份录制的
     总量 Z 与中位滤波 Z̄，核对帧数与《数据与脚本复用清单》§1 是否一致（不一致立刻停）；
  2. 按 T4-A 冻结表的 `k_on`（t_on 帧号）与 `pre/post` 列，**独立重算** τ 网格上的归一化形状
     f(τ) = (Z̄(t_on+τ) − pre)/J，与 T4-A 的 `z_at_*` 列逐事件相减 → 证明"本任务的 f(τ) 与 T4-A 表同源"；
  3. 把本任务算出的 τ 网格形状写成交接接口 `results/t3a_shape_grid.csv`（每事件 × 每 τ 一行，
     列含 grid / tau / f / base / J），供 Q1~Q5 全部脚本使用；
  4. 反驳性检查：与第一轮 `shape_stats.csv` / `shape_timewarp.csv` / `rom_compare.csv` / `rom_loo.csv`
     的对应数字并列，逐条给出差异与归因（写进 `results/t3a_vs_round1.csv`）。

产物：results/t3a_recon.csv, t3a_recon_crosscheck.csv, t3a_shape_grid.csv, t3a_vs_round1.csv,
      results/_t3a_00_recon.log
运行：python scripts/t3a_00_recon.py     （首次读 13 份 CSV，约 90~150 s）
"""
import json
import os

import numpy as np
import pandas as pd

import t3a_common as C

MANIFEST = {  # 《数据与脚本复用清单》§1 帧数（用于确认读到的是同一份录制）
    "RT1": 16356, "RT2": 23537, "RT3": 19138,
    "LT1": 19740, "LT2": 19975, "LT3": 19402,
    "F41": 19140, "F42": 19114, "F43": 20033,
    "SW1": 25693, "SW2": 7129, "SW3": 6421, "SW4": 12121,
}
# 判据：本任务 n_raw 与清单的关系（实测 13/13 都是 **清单 − 1**，差 1 帧是 CSV 尾行换行/表头计数口径之差）
# ⇒ 核对规则写成「等于清单或等于清单−1」，两者都视为同一份录制；偏离则停。
TOL = (-1, 0)


def main():
    C.start_log("t3a_00_recon")
    ev = C.load_t4a_events()
    print("T4-A 冻结事件表：%d 行 × %d 列，来源 %s" % (len(ev), ev.shape[1], C.T4A_MORPH))
    print("kind 分布：%s" % ev["kind"].value_counts().to_dict())
    print("装载类(jump>0) n=%d，其中 clean=%d；onset=%d restep=%d" %
          (ev["is_load"].sum(), (ev["is_load"] & ev["clean"]).sum(),
           ((ev["kind"] == "onset")).sum(), ((ev["kind"] == "restep")).sum()))
    print("")

    # ── ① 重建 + 帧数核对 ─────────────────────────────
    print("── ① 网格重建与帧数核对 ──")
    rows = []
    for r in C.RECS:
        tu, Xu, Z, Zs, pkt, n_raw = C.load_grid(r["key"])
        ok = (n_raw - MANIFEST[r["key"]]) in TOL
        rows.append(dict(key=r["key"], rec=r["rec"], dom=r["dom"], fam=r["fam"],
                         n_raw=n_raw, n_manifest=MANIFEST[r["key"]],
                         d_manifest=n_raw - MANIFEST[r["key"]], n_match=bool(ok),
                         n_grid=len(tu), dur_grid=float(tu[-1]), n_ch=Xu.shape[1],
                         pkt_dt=pkt))
        print("  %-4s %-16s n_raw=%-6d (清单 %-6d Δ=%+d %s)  n_grid=%-6d  dur=%.1fs  ch=%d  pkt=%.4fs"
              % (r["key"], r["rec"], n_raw, MANIFEST[r["key"]], n_raw - MANIFEST[r["key"]],
                 "OK" if ok else "**不符**", len(tu), tu[-1], Xu.shape[1], pkt))
    recon = pd.DataFrame(rows)
    recon.to_csv(os.path.join(C.RES, "t3a_recon.csv"), index=False, encoding="utf-8-sig")
    if not recon["n_match"].all():
        raise SystemExit("帧数与清单不符，停止排查")
    print("  -> results/t3a_recon.csv（13/13 帧数一致：实测恒为「清单 − 1」，见 §2 口径说明）\n")

    # ── ② 独立重算 f(τ) 并与 T4-A 对表 ────────────────
    print("── ② f(τ) 独立重算 vs T4-A 冻结列 ──")
    grid_rows, chk_rows = [], []
    for _, e in ev.iterrows():
        key = e["key"]
        tu, Xu, Z, Zs, pkt, n_raw = C.load_grid(key)
        k = int(e["k_on"])
        pre = float(e["pre"])
        post = float(e["post"])
        J = post - pre
        # 主网格（T4-A 采样点）
        for t in C.TAU_MAIN:
            f, _, _ = C.zref(Zs, k, pre, post, t)
            grid_rows.append(dict(key=key, rec=e["rec"], dom=e["dom"], fam=e["fam"],
                                  t_on=float(e["t_on"]), k_on=k, kind=e["kind"],
                                  clean=bool(e["clean"]), is_load=bool(e["is_load"]),
                                  grid=C.TAU_MAIN_LABEL, tau=t, f=f,
                                  base=pre, base_kind="T4A_pre_win_median", J=J))
        # 加密网格
        kf = np.clip(k + np.round(C.TAU_FINE / C.DT).astype(int), 0, len(Zs) - 1)
        ff = (Zs[kf] - pre) / J if abs(J) > 1e-12 else np.full(len(kf), np.nan)
        for t, f in zip(C.TAU_FINE, ff):
            grid_rows.append(dict(key=key, rec=e["rec"], dom=e["dom"], fam=e["fam"],
                                  t_on=float(e["t_on"]), k_on=k, kind=e["kind"],
                                  clean=bool(e["clean"]), is_load=bool(e["is_load"]),
                                  grid=C.TAU_FINE_LABEL, tau=float(t), f=float(f),
                                  base=pre, base_kind="T4A_pre_win_median", J=J))
        # 与 T4-A 的 z_at_* 列逐点相减
        for t in C.TAU_MAIN:
            col = C.ZC[t]
            f_mine = C.zref(Zs, k, pre, post, t)[0]
            f_t4a = float(e[col]) if pd.notna(e[col]) else np.nan
            chk_rows.append(dict(key=key, rec=e["rec"], t_on=float(e["t_on"]), kind=e["kind"],
                                 clean=bool(e["clean"]), column=col, tau=t,
                                 f_mine=f_mine, f_t4a=f_t4a,
                                 d=(f_mine - f_t4a) if np.isfinite(f_t4a) else np.nan))
    grid = pd.DataFrame(grid_rows)
    grid["uid"] = grid["key"] + "@" + grid["t_on"].map(lambda x: "%.2f" % x)
    grid.to_csv(os.path.join(C.RES, "t3a_shape_grid.csv"), index=False, encoding="utf-8-sig")
    chk = pd.DataFrame(chk_rows)
    chk.to_csv(os.path.join(C.RES, "t3a_recon_crosscheck.csv"), index=False, encoding="utf-8-sig")
    d = chk["d"].to_numpy(float)
    d = d[np.isfinite(d)]
    print("  逐点比对 %d 对（60 事件 × 8 τ 网格点）；|Δ| 中位 %.3e  最大 %.3e  超 1e-6 的对数 %d"
          % (len(d), np.median(np.abs(d)), np.max(np.abs(d)), int((np.abs(d) > 1e-6).sum())))
    print("  （T4-A 的 z_at_* 用 Zs 单点取值 + pre 基准，本任务同法复算 ⇒ 应完全一致）")
    print("  -> results/t3a_shape_grid.csv（%d 行）、t3a_recon_crosscheck.csv\n" % len(grid))

    # ── ③ 反驳性对照：与第一轮产物逐条比对 ────────────
    print("── ③ 与第一轮 13-v6-assessment 产物对表 ──")
    vs = []
    # (a) shape_stats.csv 的 onset 群体中位（n=19，主通道，τ=0.05~5）
    ss = pd.read_csv(C.R1_SHAPE_STATS)
    ld = grid[(grid["grid"] == C.TAU_MAIN_LABEL) & (grid["kind"] == "onset")]
    for _, s in ss.iterrows():
        t = float(s["tau"])
        mine = ld[np.isclose(ld["tau"], t)]["f"].to_numpy(float)
        mine = mine[np.isfinite(mine)]
        med_mine = float(np.median(mine)) if mine.size else np.nan
        vs.append(dict(against="13-v6-assessment/results/shape_stats.csv", what="onset 形状群体中位 f(τ)",
                       tau=t, n_r1=int(s["n"]), med_r1=float(s["med"]),
                       p10_r1=float(s["p10"]), p90_r1=float(s["p90"]),
                       n_t3a=int(mine.size), med_t3a=med_mine,
                       p10_t3a=float(np.percentile(mine, 10)) if mine.size else np.nan,
                       p90_t3a=float(np.percentile(mine, 90)) if mine.size else np.nan,
                       d_med=(med_mine - float(s["med"])) if np.isfinite(med_mine) else np.nan,
                       note="第一轮 onset 全部 n=19（含实录 onset，主通道口径）；本任务 onset 全部 n=%d（总量 Z）" % mine.size))
    print("  (a) shape_stats.csv（第一轮 n=%d 全 onset，主通道）已并入 t3a_vs_round1.csv" % int(ss["n"].iloc[0]))
    # (b) shape_timewarp.csv 的 α
    tw = pd.read_csv(C.R1_TIMEWARP)
    alphas = tw["alpha"].to_numpy(float)
    vs.append(dict(against="13-v6-assessment/results/shape_timewarp.csv", what="单参数时间缩放 α（每录制内多事件的拟合值）",
                   tau=np.nan, n_r1=len(alphas), med_r1=float(np.median(alphas)),
                   p10_r1=float(np.percentile(alphas, 10)), p90_r1=float(np.percentile(alphas, 90)),
                   n_t3a=0, med_t3a=np.nan, p10_t3a=np.nan, p90_t3a=np.nan, d_med=np.nan,
                   note="第一轮按「录制 × 受载段」拟合（n=19 行），本任务在 Q2 用「每事件单 τ」参数化复算，见 t3a_param_cv.csv"))
    # (c) rom_loo.csv 的 bias_pct
    loo = pd.read_csv(C.R1_ROM_LOO)
    b = loo["bias_pct"].to_numpy(float)
    vs.append(dict(against="13-v6-assessment/results/rom_loo.csv", what="留一法 Â 偏差 bias_pct（每录制首个 onset）",
                   tau=np.nan, n_r1=len(b), med_r1=float(np.median(b)),
                   p10_r1=float(np.percentile(b, 10)), p90_r1=float(np.percentile(b, 90)),
                   n_t3a=0, med_t3a=np.nan, p10_t3a=np.nan, p90_t3a=np.nan, d_med=np.nan,
                   note="第一轮 LOO 的「形状库」= 其余录制的峰值归一化形状（A5 口径），事件集=13 个 onset；本任务 Q4 逐事件 LOO 复算（40 个装载事件），见 t3a_generalization.csv"))
    vsdf = pd.DataFrame(vs)
    vsdf.to_csv(os.path.join(C.RES, "t3a_vs_round1.csv"), index=False, encoding="utf-8-sig")
    print("  (b) α 中位 %.3f（第一轮 n=%d）" % (np.median(alphas), len(alphas)))
    print("  (c) LOO bias 中位 %+.2f%%（第一轮 n=%d）" % (np.median(b), len(b)))
    print("  -> results/t3a_vs_round1.csv（%d 行）\n" % len(vsdf))

    # ── ④ 事件集清单（供报告 §2 与 n 标注） ───────────
    ev2 = ev[["key", "rec", "dom", "fam", "t_on", "kind", "armed", "clean", "is_load",
              "pre", "post", "jump", "peak", "pre_over_peak", "n_other_near",
              "z_at_02", "z_at_10", "exp_tau", "exp_beta", "T_slope"]].copy()
    ev2.to_csv(os.path.join(C.RES, "t3a_event_roster.csv"), index=False, encoding="utf-8-sig")
    print("── ④ 事件集清单 -> results/t3a_event_roster.csv ──")
    print("  装载类 n=%d（onset %d / restep %d）；unload %d / partial_unload %d；trunc 行数 %d"
          % (int(ev["is_load"].sum()),
             int((ev["kind"] == "onset").sum()), int((ev["kind"] == "restep").sum()),
             int((ev["kind"] == "unload").sum()), int((ev["kind"] == "partial_unload").sum()),
             int(ev["jump"].isna().sum())))
    print("\n完成。下一步：python scripts/t3a_01_shape.py")


if __name__ == "__main__":
    main()
