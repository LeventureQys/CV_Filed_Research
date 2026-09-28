# -*- coding: utf-8 -*-
"""T3-A 02：分层稳定性（Q3：族 / 载荷量级 / 批次 / 输入形态）与形状的时不变性（Q5）。

Q3 分层维度（每层都报 n）：
  fam      右拇指 / 左拇指 / 四指 / 实录（录制族）
  magbin   相对载荷量级 |J|/记录峰值 四档 —— **绝对载荷量级无标定数据**（见报告 §5 缺口 G1）
  batch    同族内的第 1/2/3 次录制（数据1/2/3）—— 批次/个体差
  onsetT   输入形态：快速阶跃 onset（T_ramp<0.15 s）/ 慢压 onset（≥0.15 s）/ restep
           （T_ramp 取 T4-A 反卷积列，只读引用）

Q5 时不变性（同一录制内早/中/晚）：
  1. 对每个事件算其在**自身录制内的相对位置** pos = t_on / (录制末尾 − t_on)；
  2. 按 pos 三分位分层，比较形状描述量（f(0.3/0.5/1.0 s)、τ1、β、w_fast）的中位与 p10~p90；
  3. 按录制族分组，只保留**单录制内 ≥2 个同 kind 事件**的录制做组内 trend（Spearman ρ）；
  4. 附加"加载历史"对照：距上一次检出事件的间隔（前 5 s / 5~30 s / >30 s）。

产物：results/t3a_strata.csv, t3a_time_stability.csv, t3a_time_trend.csv, t3a_vs_round1.csv,
      results/_t3a_02_strata.log
运行：python scripts/t3a_02_strata.py   （秒级~30 s）
"""
import os

import numpy as np
import pandas as pd

import t3a_common as C

METRICS = [("z_at_03", "f(0.30s)"), ("z_at_05", "f(0.50s)"), ("z_at_10", "f(1.00s)"),
           ("tau1", "tau1"), ("beta", "beta"), ("w_fast", "w_fast"), ("two_tau_rms", "two_tau_rms")]


def layer_stats(df, dim, metric_cols):
    """给一个分层的 DataFrame，按 dim 分组算各指标的 n/中位/p10~p90/CQV。"""
    rows = []
    for name, g in df.groupby(dim, dropna=False):
        for col, lab in metric_cols:
            v = g[col].to_numpy(float)
            v = v[np.isfinite(v)]
            med, p10, p90, n = C.band(v)
            if n >= 4:
                p75, p25 = np.percentile(v, 75), np.percentile(v, 25)
                cqv = (p75 - p25) / (p75 + p25) if abs(p75 + p25) > 1e-9 else np.nan
            else:
                cqv = np.nan
            rows.append(dict(dim=dim, layer=str(name), metric=lab, col=col, n=n, med=med,
                             p10=p10, p90=p90, spread=p90 - p10, cqv=cqv,
                             n_events_layer=int(len(g))))
    return rows


def main():
    C.start_log("t3a_02_strata")
    ev = C.load_t4a_events()
    ev["uid"] = ev["key"] + "@" + ev["t_on"].map(lambda x: "%.2f" % x)
    aud = pd.read_csv(os.path.join(C.RES, "t3a_inlier_audit.csv"))
    prm = pd.read_csv(os.path.join(C.RES, "t3a_param_events.csv"))
    inp = C.load_t4a_input()
    inp["uid"] = inp["key"] + "@" + inp["t_on"].map(lambda x: "%.2f" % x)
    tr = pd.read_csv(os.path.join(C.RES, "t3a_shape_trajectory.csv"))

    L = ev[ev["is_load"]].merge(aud[["uid", "J_ok", "posdef", "usable", "J_over_peak"]], on="uid")
    L = L.merge(prm[["uid", "tau1", "beta", "w_fast", "two_tau_rms", "tau_e"]], on="uid")
    L = L.merge(inp[["uid", "T_ramp", "rmse_ramp_pct", "rmse_step_pct"]], on="uid", how="left")
    # 录制末尾时间（做时不变性用）
    ends = {}
    for r in C.RECS:
        g = C.load_grid(r["key"])
        ends[r["key"]] = float(g[0][-1])
    L["rec_end"] = L["key"].map(ends)
    L["pos"] = L["t_on"] / L["rec_end"].clip(lower=1.0)
    # 距上一事件的间隔（加载历史）
    prev = []
    for _, e in L.sort_values(["key", "t_on"]).iterrows():
        s = ev[(ev["key"] == e["key"]) & (ev["t_on"] < e["t_on"] - 1e-9)]["t_on"]
        prev.append(float(e["t_on"] - s.max()) if len(s) else np.nan)
    L = L.sort_values(["key", "t_on"]).reset_index(drop=True)
    L["gap_prev"] = prev

    # 分层列
    bins = [0.0, 0.05, 0.15, 0.40, 1e9]
    labs = ["J/peak<5%", "5~15%", "15~40%", ">=40%"]
    L["magbin"] = pd.cut(L["J_over_peak"].abs(), bins=bins, labels=labs, right=False)
    L["batch"] = L["key"].str[-1].map(lambda x: "数据%s" % x if x.isdigit() else "实录")
    def otype(r):
        if r["kind"] == "restep":
            return "restep"
        t = r["T_ramp"]
        if not np.isfinite(t):
            return "onset(T_ramp缺)"
        return "onset 快速阶跃" if t < 0.15 else "onset 慢压"
    L["otype"] = L.apply(otype, axis=1)
    L.to_csv(os.path.join(C.RES, "t3a_strata_events.csv"), index=False, encoding="utf-8-sig")

    print("══ Q3 分层稳定性 ══")
    print("装载事件 n=%d（usable %d）；绝对载荷量级**无标定数据**，只用相对量 |J|/记录峰值（缺口 G1）\n"
          % (len(L), int(L["usable"].sum())))
    rows = []
    for dim in ["fam", "magbin", "batch", "otype"]:
        for tag, sub in [("all", L), ("usable", L[L["usable"]])]:
            s = layer_stats(sub.assign(_t=tag), dim, METRICS)
            for r in s:
                r["sample"] = tag
            rows.extend(s)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(C.RES, "t3a_strata.csv"), index=False, encoding="utf-8-sig")
    for dim in ["fam", "magbin", "batch", "otype"]:
        print("  ── 维度 %s（usable 样本，n 为各层事件数）──" % dim)
        sub = st[(st["dim"] == dim) & (st["sample"] == "usable")]
        for layer, g in sub.groupby("layer", sort=False):
            ne = int(g["n_events_layer"].iloc[0])
            r = g[g["col"] == "z_at_10"].iloc[0]
            r3 = g[g["col"] == "z_at_03"].iloc[0]
            note = "（n≤3 仅定性参考）" if ne <= 3 else ""
            print("    %-16s n=%-3d f(0.3s)中位=%-7s p10~p90=%.3f~%.3f | f(1.0s)中位=%-7s "
                  "p10~p90=%.3f~%.3f CQV=%.3f %s"
                  % (layer, ne, ("%.3f" % r3["med"]) if np.isfinite(r3["med"]) else "n/a",
                     r3["p10"], r3["p90"], ("%.3f" % r["med"]) if np.isfinite(r["med"]) else "n/a",
                     r["p10"], r["p90"], r["cqv"] if np.isfinite(r["cqv"]) else np.nan, note))
        print("")
    print("  -> results/t3a_strata.csv (%d 行)、t3a_strata_events.csv\n" % len(st))

    # ── Kruskal-Wallis / Mann-Whitney：层间是否有显著差异 ──
    print("  ── 层间差异检验（Kruskal-Wallis，H 与 p；对 f(1.0s) 与 tau1）──")
    from scipy.stats import kruskal, mannwhitneyu
    test_rows = []
    for dim in ["fam", "magbin", "batch", "otype"]:
        for col, lab in [("z_at_10", "f(1.00s)"), ("z_at_03", "f(0.30s)"), ("tau1", "tau1")]:
            groups = [g[col].to_numpy(float) for _, g in L[L["usable"]].groupby(dim, dropna=False)
                      if np.isfinite(g[col].to_numpy(float)).sum() >= 2]
            if len(groups) >= 2:
                try:
                    h, p = kruskal(*groups)
                except Exception:
                    h, p = np.nan, np.nan
                test_rows.append(dict(dim=dim, metric=lab, col=col, test="kruskal",
                                      k=len(groups), H=h, p=p,
                                      n_each=";".join(str(np.isfinite(g).sum()) for g in groups)))
                print("    %-8s %-9s k=%d H=%7.3f p=%.4f  (各层 n = %s)"
                      % (dim, lab, len(groups), h, p,
                         ",".join(str(int(np.isfinite(g).sum())) for g in groups)))
    # 家族两两对照（右拇指 vs 左拇指 vs 四指）
    us = L[L["usable"] & L["fam"].isin(["右拇指", "左拇指", "四指"])]
    if us["fam"].nunique() >= 2:
        fams = sorted(us["fam"].unique())
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                a = us[us["fam"] == fams[i]]["z_at_10"].to_numpy(float)
                b = us[us["fam"] == fams[j]]["z_at_10"].to_numpy(float)
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if len(a) >= 2 and len(b) >= 2:
                    u, p = mannwhitneyu(a, b, alternative="two-sided")
                    d = float(np.median(a) - np.median(b))
                    test_rows.append(dict(dim="fam_pair", metric="f(1.00s)", col="z_at_10",
                                          test="mannwhitney", k=2, H=u, p=p,
                                          n_each="%d;%d" % (len(a), len(b))))
                    print("    fam 对照 %s(n=%d) vs %s(n=%d): Δ中位=%+.3f p=%.4f"
                          % (fams[i], len(a), fams[j], len(b), d, p))
    pd.DataFrame(test_rows).to_csv(os.path.join(C.RES, "t3a_strata_tests.csv"), index=False,
                                   encoding="utf-8-sig")
    print("")

    # ══════════════ Q5 时不变性 ══════════════
    print("══ Q5 形状的时不变性（同一录制内早/中/晚 + 加载历史）══")
    disp = L[L["usable"]].copy()
    disp["pos_bin"] = pd.cut(disp["pos"], [0, 1 / 3, 2 / 3, 1.0],
                             labels=["早期(前1/3)", "中期(中1/3)", "晚期(后1/3)"], include_lowest=True)
    rows = []
    sub0 = disp
    for dim in ["pos_bin"]:
        for name, g in sub0.groupby(dim, dropna=False):
            for col, lab in METRICS:
                v = g[col].to_numpy(float)
                v = v[np.isfinite(v)]
                med, p10, p90, n = C.band(v)
                rows.append(dict(strat="pos_bin(全部可用事件)", layer=str(name), metric=lab, col=col,
                                 n=n, med=med,
                                 p10=p10, p90=p90, spread=p90 - p10, n_events=int(len(g))))
    # 按 kind 分开（onset / restep 各自三分位）
    for kind in ["onset", "restep"]:
        sub = disp[disp["kind"] == kind].copy()
        if len(sub) < 4:
            continue
        sub["pos_bin_k"] = pd.cut(sub["pos"], [0, 1 / 3, 2 / 3, 1.0],
                                  labels=["早期", "中期", "晚期"], include_lowest=True)
        for name, g in sub.groupby("pos_bin_k", dropna=False):
            for col, lab in METRICS:
                v = g[col].to_numpy(float)
                v = v[np.isfinite(v)]
                med, p10, p90, n = C.band(v)
                rows.append(dict(strat="pos_bin(%s)" % kind, layer=str(name), metric=lab, col=col,
                                 n=n, med=med, p10=p10, p90=p90, spread=p90 - p10,
                                 n_events=int(len(g))))
    # 加载历史分层
    disp["gap_bin"] = pd.cut(disp["gap_prev"], [0, 5, 30, 1e9],
                             labels=["≤5s", "5~30s", ">30s"], include_lowest=True)
    for name, g in disp.groupby("gap_bin", dropna=False):
        for col, lab in METRICS:
            v = g[col].to_numpy(float)
            v = v[np.isfinite(v)]
            med, p10, p90, n = C.band(v)
            rows.append(dict(strat="gap_prev(距上一事件)", layer=str(name), metric=lab, col=col, n=n,
                             med=med, p10=p10, p90=p90, spread=p90 - p10, n_events=int(len(g))))
    tstab = pd.DataFrame(rows)
    tstab.to_csv(os.path.join(C.RES, "t3a_time_stability.csv"), index=False, encoding="utf-8-sig")
    for strat in ["pos_bin(全部可用事件)", "pos_bin(onset)", "gap_prev(距上一事件)"]:
        s = tstab[tstab["strat"] == strat]
        if s.empty:
            continue
        print("  ── %s ──" % strat)
        for layer, g in s.groupby("layer", sort=False):
            ne = int(g["n_events"].iloc[0])
            r = g[g["col"] == "z_at_10"].iloc[0]
            r3 = g[g["col"] == "z_at_03"].iloc[0]
            note = "（n≤3 仅定性参考）" if ne <= 3 else ""
            print("    %-14s n=%-3d f(0.3s)=%-6s f(1.0s)中位=%-6s p10~p90=%.3f~%.3f %s"
                  % (layer, ne, ("%.3f" % r3["med"]) if np.isfinite(r3["med"]) else "n/a",
                     ("%.3f" % r["med"]) if np.isfinite(r["med"]) else "n/a",
                     r["p10"], r["p90"], note))
        print("")

    # 组内 trend（同录制内 ≥2 个同 kind 事件 → Spearman ρ vs t_on）
    print("  ── 组内趋势（同录制内 ≥2 个同 kind 事件：Spearman ρ(f(1.0s), t_on)）──")
    tro = []
    for (key, kind), g in disp.groupby(["key", "kind"]):
        if len(g) < 2:
            continue
        from scipy.stats import spearmanr
        for col, lab in [("z_at_10", "f(1.00s)"), ("z_at_03", "f(0.30s)"), ("tau1", "tau1")]:
            x = g["t_on"].to_numpy(float)
            y = g[col].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 2:
                continue
            if len(set(y[m])) < 2:
                rho, p = 0.0, 1.0
            else:
                rho, p = spearmanr(x[m], y[m])
            tro.append(dict(key=key, kind=kind, metric=lab, col=col, n=int(m.sum()),
                            rho=float(rho), p=float(p),
                            t_span=float(x[m].max() - x[m].min())))
    tdf = pd.DataFrame(tro)
    tdf.to_csv(os.path.join(C.RES, "t3a_time_trend.csv"), index=False, encoding="utf-8-sig")
    if len(tdf):
        s = tdf[tdf["metric"] == "f(1.00s)"]
        print("    f(1.00s) 组内 ρ：n=%d 组，中位 ρ=%+.3f，|ρ|≥0.8 的组 %d 个，p<0.05 的组 %d 个"
              % (len(s), float(np.median(s["rho"])), int((s["rho"].abs() >= 0.8).sum()),
                 int((s["p"] < 0.05).sum())))
        for _, r in s.iterrows():
            flag = "←显著" if r["p"] < 0.05 else ""
            print("      %-4s %-6s n=%d 时间跨度%6.1fs ρ=%+0.3f p=%.3f %s"
                  % (r["key"], r["kind"], int(r["n"]), r["t_span"], r["rho"], r["p"], flag))
    print("  -> results/t3a_time_stability.csv (%d 行)、t3a_time_trend.csv (%d 行)\n"
          % (len(tstab), len(tdf)))

    # ══════════════ 反驳性对照表（合并 Q4 数字） ══════════════
    print("══ 与第一轮 13-v6-assessment 的逐条对照（写入 t3a_vs_round1.csv 的补充行）══")
    extra = []
    ss = pd.read_csv(C.R1_SHAPE_STATS)
    grid = pd.read_csv(os.path.join(C.RES, "t3a_shape_grid.csv"))
    grid_m = grid[grid["grid"] == C.TAU_MAIN_LABEL]
    for _, s in ss.iterrows():
        t = float(s["tau"])
        for tag, uids in [("usable_onset", set(L[L["usable"] & (L["kind"] == "onset")]["uid"])),
                          ("all_onset", set(L[L["kind"] == "onset"]["uid"]))]:
            v = grid_m[np.isclose(grid_m["tau"], t) & grid_m["uid"].isin(uids)]["f"].to_numpy(float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            extra.append(dict(what="onset 形状中位 f(τ)", tau=t, variant=tag, n=int(v.size),
                              med=float(np.median(v)), p10=float(np.percentile(v, 10)),
                              p90=float(np.percentile(v, 90)), source="t3a_shape_grid.csv"))
    ex = pd.DataFrame(extra)
    ex.to_csv(os.path.join(C.RES, "t3a_shape_vs_round1.csv"), index=False, encoding="utf-8-sig")
    print("   本任务 onset 中位形状（usable / all）对 τ:")
    for tag in ["usable_onset", "all_onset"]:
        s = ex[ex["variant"] == tag]
        if s.empty:
            continue
        print("     %-13s n=%d  " % (tag, int(s["n"].iloc[0]))
              + " ".join("%.2fs=%.3f" % (r["tau"], r["med"]) for _, r in s.iterrows()))
    r1 = ss[["tau", "med"]].rename(columns={"med": "r1_med"})
    m = ex[ex["variant"] == "usable_onset"].merge(r1, on="tau", how="outer")
    m["d_med"] = m["med"] - m["r1_med"]
    m.to_csv(os.path.join(C.RES, "t3a_shape_vs_round1.csv"), index=False, encoding="utf-8-sig")
    print("   -> results/t3a_shape_vs_round1.csv")
    print("\n完成。下一步：python scripts/t3a_03_figures.py")


if __name__ == "__main__":
    main()
