# -*- coding: utf-8 -*-
"""T2-A：三阶段边界判据 + 阶段时长（T2-Q1 / T2-Q2 / T2-Q5）。

事件集：**复用 T4-A 冻结事件表**（`T4_*/results/t4a_morphology.csv`，60 行），不重新检测。
形状轴：**原始读数**（100 Hz 网格上的 Z），另给 3 帧中值（0.03 s）与 0.5 s 中值（Z̄）对照。
J 与 pre/post：指标字典 §2.1（pre=[t0−2,t0)、post=[t0+4,t0+6]，落在 Z̄ 上）⇒ 与 T4-A 逐事件可比。

阶段定义（报告 §2.3 给理由）：
  S1 阶跃    : t_on → t25（首次达到 0.25·|J|）
  S2 快相    : t25 → t90
  S3 慢相    : t90 → t_end（= min(下一事件−0.5 s, t_on+60 s, 录制末−0.5 s)）
  「阶跃成立」判据：t25 ≤ 0.05 s（=5 帧 = 3 个指尖包）；阈值敏感性见 t2_phase_boundary_rules.csv

产物：results/t2_phase_times.csv（主表）、results/t2_phase_durations.csv、results/t2_phase_boundary_rules.csv
"""
import numpy as np
import pandas as pd

import t2_common as C


def main():
    C.start_log("t2a_phases")
    print("== T2-A 三阶段边界与时长 ==")
    all_d = C.load_all()
    ev = C.load_events()
    print("事件表：%d 行（%s）" % (len(ev), ev["kind"].value_counts().to_dict()))

    # ── 自检 1：网格与 J 是否与 T4-A 完全对齐 ──
    rows, chk_pre, chk_j, chk_jrel = [], [], [], []
    for _, e in ev.iterrows():
        d = all_d[e["key"]]
        k = int(e["k_on"])
        t0 = float(d["tu"][k])
        t_ax, y_raw, ybar, _ = C.axis_series(d, "grid")
        pre, post, J = C.j_of_idx(ybar, k)
        chk_pre.append(abs(pre - float(e["pre"])))
        chk_j.append(abs(J - float(e["jump"])))
        chk_jrel.append(abs(J - float(e["jump"])) / max(abs(J), 1e-9))
        n = len(t_ax)
        t_rec = float(t_ax[-1])
        # 下一事件（同录制）
        nxt = ev[(ev.key == e["key"]) & (ev.t_on > t0 + 0.1)].t_on
        t_next = float(nxt.min()) if len(nxt) else np.inf
        r = dict(key=e["key"], rec=e["rec"], fam=e["fam"], dom=e["dom"], main_ch=int(e["main_ch"]),
                 t_on=round(t0, 3), k_on=k, kind=e["kind"], armed=bool(e["armed"]),
                 clean=bool(e["clean"]), trunc=bool(e.get("trunc", False)),
                 pre=pre, post=post, jump=J, ratio=abs(J) / max(pre, 1e-9),
                 peak=float(e["peak"]), T_ramp=e.get("T_ramp", np.nan),
                 t_on5=e.get("t_on5", np.nan), d_t_on5=e.get("d_t_on5", np.nan),
                 pkt_dt=float(d["pkt"]))
        # ── 比例穿越（主信号 = Z3，原始读数 3 帧中值 = 0.03 s）──
        #  tXX      = 指标字典 §2.3 字面口径（首次穿越）
        #  tXX_sus  = **持续穿越**口径（此后 0.10 s 内持续 ≥ 阈值）→ 阶段边界用它
        y3 = d["Z3"]
        SUS = 0.10
        for fr in (0.10, 0.25, 0.50, 0.80, 0.90, 0.95):
            r["t%02d" % int(fr * 100)] = C.cross_time(t_ax, y3, t0, pre, J, fr)
            r["t%02d_sus" % int(fr * 100)] = C.cross_time(t_ax, y3, t0, pre, J, fr, sustain=SUS)
        r["t25_sus03"] = C.cross_time(t_ax, y3, t0, pre, J, 0.25, sustain=0.03)
        r["t25_sus20"] = C.cross_time(t_ax, y3, t0, pre, J, 0.25, sustain=0.20)
        # 变体口径（对照）：原始 Z / Z̄0.5 / 包轴（均用首次穿越，便于与既有数字对齐）
        for tag, yv, ax in (("raw", y_raw, d["tu"]), ("bar", d["Zbar"], d["tu"]),
                            ("pkt", d["Zp3"], d["tp"]), ("pktraw", d["Zp"], d["tp"])):
            for fr in (0.25, 0.50, 0.80, 0.90, 0.95):
                r["t%02d_%s" % (int(fr * 100), tag)] = C.cross_time(ax, yv, t0, pre, J, fr)
        r["t25_raw_sus"] = C.cross_time(t_ax, y_raw, t0, pre, J, 0.25, sustain=SUS)
        r["t90_bar_sus"] = C.cross_time(t_ax, d["Zbar"], t0, pre, J, 0.90, sustain=SUS)
        # ── 完成度 f(τ)（主口径 = Z3，J 用指标字典 post 窗）──
        for tau in (0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0):
            r[C.TAU_COL[tau]] = C.f_at(t_ax, y3, t0, pre, J, tau)
        r["z_at_02"] = r["f_020"]
        r["z_at_10"] = r["f_100"]
        r["z_at_005"] = r["f_005"]
        for tau in (0.05, 0.10, 0.20, 1.0):
            r["f_%03d_raw" % int(tau * 1000)] = C.f_at(t_ax, y_raw, t0, pre, J, tau)
        # ── 加载撞击瞬态诊断（真实信号特征：0.02~0.05 s 过冲 → 0.1 s 内回落）──
        tw = np.round(np.arange(0.0, 0.101, C.DT), 3)
        fw = np.array([C.f_at(t_ax, y3, t0, pre, J, x) for x in tw], float)
        fwr = np.array([C.f_at(t_ax, y_raw, t0, pre, J, x) for x in tw], float)
        r["fmax_early"] = float(np.nanmax(fw)) if np.isfinite(fw).any() else np.nan
        r["fmax_early_raw"] = float(np.nanmax(fwr)) if np.isfinite(fwr).any() else np.nan
        r["f_at_10"] = r["f_010"]
        fa = r["fmax_early"]
        r["transient_decay"] = ((fa - r["f_010"]) / fa if np.isfinite(fa) and fa > 1e-9 else np.nan)
        r["transient_flag"] = bool(np.isfinite(fa) and fa >= 0.25 and np.isfinite(r["f_010"])
                                   and r["f_010"] <= 0.6 * fa)
        # 单位时间交付率（%/s，前 0.05 s 与 0.2 s）
        r["rate_005_pct_s"] = 100 * r["f_005"] / 0.05
        r["rate_02_pct_s"] = 100 * r["f_020"] / 0.20
        # ── 尖峰诊断：原始 Z 上的单帧尖峰是否伪造了早期穿越 ──
        sgn = 1.0 if J > 0 else -1.0
        i0, i1 = k, min(n - 1, k + int(2.0 / C.DT))
        d1 = np.diff(y_raw[i0:i1 + 1])
        r["max_dfrm_frac"] = float(np.max(np.abs(d1)) / abs(J)) if d1.size else np.nan
        seg_r = y_raw[i0:i1 + 1] - pre
        tr = np.where(sgn * seg_r >= 0.25 * abs(J))[0]
        t3 = r["t25"]
        r["n_raw_spike_frames"] = int((tr * C.DT < t3).sum()) if (len(tr) and np.isfinite(t3)) else np.nan
        # ── 阶跃承载帧数 ──
        seg3 = y3[i0:i1 + 1] - pre
        cs3 = np.cumsum(np.clip(np.diff(sgn * seg3), 0.0, None))
        seg = y_raw[i0:i1 + 1] - pre
        cs = np.cumsum(np.clip(np.diff(sgn * seg), 0.0, None))
        need, need9 = 0.5 * abs(J), 0.9 * abs(J)
        r["frames_to_50_z3"] = int(np.argmax(cs3 >= need) + 1) if cs3.size and cs3[-1] >= need else np.nan
        r["frames_to_90_z3"] = int(np.argmax(cs3 >= need9) + 1) if cs3.size and cs3[-1] >= need9 else np.nan
        r["frames_to_50_raw"] = int(np.argmax(cs >= need) + 1) if cs.size and cs[-1] >= need else np.nan
        # 「跃变帧」结构口径：单帧承载 ≥10%·|J| 的帧（10 ms 内交付 ≥10% 台阶 ⇒ 速率 ≥10·|J|/s；
        #  0.5 s 线性斜坡的上限只有 2·|J|/s ⇒ 该判据能把「跃变」与「斜坡」分开）
        i2 = min(n - 1, k + int(0.5 / C.DT))
        dj = np.abs(np.diff(y3[k:i2 + 1]))
        jf = np.where(dj >= 0.10 * abs(J))[0] if dj.size else np.array([], int)
        r["jump_frames"] = int(len(jf))
        r["t_step_end"] = float(jf[-1] + 1) * C.DT if len(jf) else np.nan
        r["frame_rate_max"] = float(dj.max() / C.DT / abs(J)) if dj.size else np.nan   # 单位 |J|/s
        # ── 包级承载（包轴，Z3）──
        tp = d["tp"]
        j0 = int(np.searchsorted(tp, t0))
        j1 = min(len(tp) - 1, j0 + int(1.0 / max(d["pkt"], 1e-3)))
        csp = np.cumsum(np.clip(np.diff(sgn * (d["Zp3"][j0:j1 + 1] - pre)), 0.0, None))
        cspr = np.cumsum(np.clip(np.diff(sgn * (d["Zp"][j0:j1 + 1] - pre)), 0.0, None))
        r["npkts_to_50"] = int(np.argmax(csp >= need) + 1) if csp.size and csp[-1] >= need else np.nan
        r["npkts_to_50_raw"] = int(np.argmax(cspr >= need) + 1) if cspr.size and cspr[-1] >= need else np.nan
        # ── 阶段时长（边界用**持续穿越**口径）──
        t25, t90 = r["t25_sus"], r["t90_sus"]
        t_end_nat = min(t_next - C.GUARD_NEXT, t_rec - C.GUARD_NEXT)
        t_end = min(t_end_nat, t0 + C.SLOW_CAP)
        r["t_next_ev"], r["t_end_nat"], r["t_end"] = t_next, t_end_nat, t_end
        r["dur_s1"] = t25
        r["dur_s2"] = (t90 - t25) if np.isfinite(t90) and np.isfinite(t25) else np.nan
        r["dur_s3"] = ((t_end - t0 - t90) if (np.isfinite(t90) and (t_end - t0) > t90) else np.nan)
        r["t_span"] = t_end - t0
        # 字面口径（首次穿越）的时长，供与既有数字对齐
        r["dur_s1_naive"] = r["t25"]
        r["dur_s2_naive"] = (r["t90"] - r["t25"]
                             if np.isfinite(r["t90"]) and np.isfinite(r["t25"]) else np.nan)
        r["dur_s3_naive"] = ((t_end - t0 - r["t90"])
                             if (np.isfinite(r["t90"]) and (t_end - t0) > r["t90"]) else np.nan)
        r["dur_s1_raw"] = r["t25_raw_sus"]
        r["dur_s2_raw"] = (r["t90_raw"] - r["t25_raw_sus"]
                           if np.isfinite(r["t90_raw"]) and np.isfinite(r["t25_raw_sus"]) else np.nan)
        r["dur_s3_raw"] = ((t_end - t0 - r["t90_raw"])
                           if (np.isfinite(r["t90_raw"]) and (t_end - t0) > r["t90_raw"]) else np.nan)
        r["dur_s1_bar"] = r["t25_bar"]
        r["dur_s2_bar"] = (r["t90_bar_sus"] - r["t25_bar"]
                           if np.isfinite(r["t90_bar_sus"]) and np.isfinite(r["t25_bar"]) else np.nan)
        # 「阶跃成立」判据与敏感性（主判：持续穿越 t25_sus ≤ 0.05 s）
        r["step_ok"] = bool(np.isfinite(t25) and t25 <= C.T_S1_MAX)
        r["step_ok_naive"] = bool(np.isfinite(r["t25"]) and r["t25"] <= C.T_S1_MAX)
        r["step_ok_sus20"] = bool(np.isfinite(r["t25_sus20"]) and r["t25_sus20"] <= C.T_S1_MAX)
        for f in C.STEP_SENS_FRAC:
            for tm in C.STEP_SENS_T:
                tv = C.cross_time(t_ax, y3, t0, pre, J, f, sustain=SUS)
                r["ok_f%02d_t%02d" % (int(f * 100), int(tm * 100))] = bool(
                    np.isfinite(tv) and tv <= tm)
        # ── 慢相速率（%/s）：Z̄ 上从 t90 到 t_end ──
        if np.isfinite(r["dur_s3"]) and r["dur_s3"] > 1.0:
            z90 = np.interp(t0 + t90, t_ax, ybar)
            zend = np.interp(t_end, t_ax, ybar)
            r["slow_rate_pct_s"] = 100 * (zend - z90) / J / r["dur_s3"]
            r["slow_amp_pct"] = 100 * (zend - z90) / J
            r["slow_win_s"] = r["dur_s3"]
        else:
            r["slow_rate_pct_s"] = r["slow_amp_pct"] = r["slow_win_s"] = np.nan
        rows.append(r)

    print("\n[自检 1] 网格/J 对齐（对 T4-A）：max|Δpre| = %.2e，max|ΔJ| = %.2e（相对 %.2e）" %
          (max(chk_pre), max(chk_j), max(chk_jrel)))
    df = pd.DataFrame(rows)
    # 合并 T4-A 的既有列（只读引用），便于逐事件比对
    keep = ["key", "t_on", "t90", "t95", "t50", "t80", "t10", "z_at_005", "z_at_01",
            "exp_tau", "exp_beta", "exp_rms", "pow_p", "pow_tau", "pow_rms",
            "step_frame_frac", "top3_frame_frac", "n_frames_rise", "rise_frames_10_90",
            "frames_to_50", "T_slope", "pre_over_peak", "n_other_near"]
    t4 = ev[keep].rename(columns={c: c + "_t4a" for c in keep if c not in ("key",)})
    t4["_t2"] = ev["t_on"].round(2)
    df["_t2"] = df["t_on"].round(2)
    df = df.merge(t4, on=["key", "_t2"], how="left").drop(columns=["_t2"])

    order = (["key", "rec", "fam", "dom", "main_ch", "t_on", "k_on", "kind", "armed", "clean",
              "trunc", "pre", "post", "jump", "ratio", "peak", "T_ramp"]
             + ["t10", "t25", "t50", "t80", "t90", "t95"]
             + ["t10_sus", "t25_sus", "t50_sus", "t80_sus", "t90_sus", "t95_sus",
                "t25_sus03", "t25_sus20"]
             + ["t25_raw", "t50_raw", "t80_raw", "t90_raw", "t95_raw", "t25_raw_sus"]
             + ["t25_bar", "t50_bar", "t80_bar", "t90_bar", "t95_bar", "t90_bar_sus"]
             + ["t25_pkt", "t50_pkt", "t80_pkt", "t90_pkt", "t95_pkt"]
             + ["t25_pktraw", "t50_pktraw", "t90_pktraw", "t95_pktraw"]
             + ["f_002", "f_005", "f_010", "f_020", "f_050", "f_100", "f_200", "f_500",
                "f_050_raw", "f_100_raw", "f_200_raw", "f_1000_raw",
                "fmax_early", "fmax_early_raw", "f_at_10", "transient_decay", "transient_flag",
                "z_at_005", "z_at_02", "z_at_10", "rate_005_pct_s", "rate_02_pct_s"]
             + ["frames_to_50_z3", "frames_to_90_z3", "frames_to_50_raw", "npkts_to_50",
                "npkts_to_50_raw", "max_dfrm_frac", "n_raw_spike_frames", "jump_frames", "t_step_end", "frame_rate_max",
                "step_ok", "step_ok_naive", "step_ok_sus20",
                "t_next_ev", "t_end_nat", "t_end", "dur_s1", "dur_s2", "dur_s3",
                "dur_s1_naive", "dur_s2_naive", "dur_s3_naive",
                "dur_s1_raw", "dur_s2_raw", "dur_s3_raw", "dur_s1_bar", "dur_s2_bar",
                "t_span", "slow_win_s", "slow_amp_pct", "slow_rate_pct_s"]
             + ["d_t_on5", "t_on5", "pkt_dt"]
             + [c for c in df.columns if c.endswith("_t4a")]
             + [c for c in df.columns if c.startswith("ok_f")])
    df = df[order]
    df.to_csv(C.os.path.join(C.RES, "t2_phase_times.csv"), index=False, encoding="utf-8-sig")
    print("  -> results/t2_phase_times.csv  (%d 行 × %d 列)" % df.shape)

    # ── Q2：分 kind 的阶段时长 ──
    srows = []
    for kind in ("onset", "restep", "unload", "partial_unload"):
        s = df[df.kind == kind]
        for col in ("dur_s1", "dur_s2", "dur_s3", "t25_sus", "t50_sus", "t90_sus", "t95_sus",
                    "t_step_end", "jump_frames", "t_span",
                    "f_005", "f_020", "f_100", "frames_to_50_z3", "npkts_to_50",
                    "fmax_early", "transient_decay",
                    "slow_rate_pct_s", "slow_amp_pct", "T_ramp"):
            m, p10, p90, n = C.band(s[col])
            mx = float(np.nanmax(s[col])) if n else np.nan
            srows.append(dict(kind=kind, metric=col, n=n, med=m, p10=p10, p90=p90, max=mx))
    sdf = pd.DataFrame(srows)
    sdf.to_csv(C.os.path.join(C.RES, "t2_phase_durations.csv"), index=False, encoding="utf-8-sig")
    print("  -> results/t2_phase_durations.csv")

    # ── Q1/Q5：判据定义 + 阈值敏感性 ──
    rrows = []
    for k, v in [
        ("主信号", "Z3 = 总量 Z 的 3 帧中值（0.03 s）；100 Hz 网格；禁用 τ=2 s 平滑"),
        ("S1 起点", "t_on = 候选锚点 ±0.20 s 内最大单帧跳变帧（T4-A/第一轮同法）"),
        ("S1 末 / S2 起", "t25_sus：**持续穿越**——Z3 首次达到 pre + 0.25·|J| 且此后 0.10 s 内持续 ≥ 阈值。理由：restep 在 0.02~0.05 s 有 25~45%·J 的加载撞击过冲、0.1 s 内回落，字面首次穿越会把瞬态误判成阶跃"),
        ("S2 末 / S3 起", "t90_sus：同上，阈值 0.90·|J|（指标字典 §2.2 的 0.90 比例，加持续窗）；备选 t95 与双指数速率交汇点见 t2_phase_boundary_conventions.csv"),
        ("S3 末", "t_end = min(下一事件 t_on − 0.5 s, t_on + 60 s, 录制末 − 0.5 s)"),
        ("J / pre / post", "指标字典 §2.1：pre=[t0−2,t0) 中位、post=[t0+4,t0+6] 中位（Z̄ 上、索引窗），J=post−pre；与 T4-A 逐事件一致（max|ΔJ|/|J| < 1e-6）"),
        ("阶跃成立判据", "t25_sus ≤ 0.05 s（5 帧 / 3 个指尖包）；等价于「0.05 s 内持续交付 ≥25%·J」"),
        ("判据阈值选择的理由", "0.25 是「0.5 s 斜坡在 0.05 s 内最多交付 10%」的 2.5 倍余量；0.05 s 下限受包周期（16.7/40 ms）与时间轴畸变区约束，不取更小"),
        ("持续窗长度", "0.10 s（10 帧）：实测撞击瞬态回落时间中位见 t2_phase_times.csv 的 transient_decay；敏感性 0.03/0.20 s 见 t25_sus03 / t25_sus20"),
    ]:
        rrows.append(dict(item=k, detail=v, n_onset=np.nan, n_restep=np.nan))
    for f in C.STEP_SENS_FRAC:
        for tm in C.STEP_SENS_T:
            c = "ok_f%02d_t%02d" % (int(f * 100), int(tm * 100))
            on = df[df.kind == "onset"]
            re = df[df.kind == "restep"]
            rrows.append(dict(item="敏感性：frac=%.2f, t≤%.2f s" % (f, tm), detail=c,
                              n_onset=int(on[c].sum()), n_restep=int(re[c].sum())))
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(C.os.path.join(C.RES, "t2_phase_boundary_rules.csv"), index=False,
               encoding="utf-8-sig")
    print("  -> results/t2_phase_boundary_rules.csv")

    # ── 控制台汇总 ──
    print("\n== Q2 阶段时长（主口径：t25 / t90 / t_end）==")
    for kind in ("onset", "restep", "unload", "partial_unload"):
        s = df[df.kind == kind]
        if not len(s):
            continue
        print("[%s] n=%d" % (kind, len(s)))
        print("   S1 阶跃(dur_s1)  " + C.fmt_band(s.dur_s1, 3, " s"))
        print("   S2 快相(dur_s2)  " + C.fmt_band(s.dur_s2, 3, " s"))
        print("   S3 慢相(dur_s3)  " + C.fmt_band(s.dur_s3, 3, " s"))
        print("   整段 t_span      " + C.fmt_band(s.t_span, 3, " s"))
        print("   f(0.05)=%.3f f(0.2)=%.3f f(1s)=%.3f" %
              (C.band(s.f_005)[0], C.band(s.f_020)[0], C.band(s.f_100)[0]))
        print("   原始 Z 口径：dur_s1=%.3f dur_s2=%.3f dur_s3=%.3f s；f(0.05)=%.3f" %
              (C.band(s.dur_s1_raw)[0], C.band(s.dur_s2_raw)[0], C.band(s.dur_s3_raw)[0],
               C.band(s.f_050_raw)[0]))
        print("   字面口径(首次穿越)：dur_s1=%.3f dur_s2=%.3f dur_s3=%.3f s" %
              (C.band(s.dur_s1_naive)[0], C.band(s.dur_s2_naive)[0], C.band(s.dur_s3_naive)[0]))
        print("   帧数：frames_to_50 Z3 中位 %.1f / 原始 %.1f 帧；包数中位 %.1f 包；"
              "单帧最大跃变占比中位 %.3f；跃变帧数中位 %.1f（跃变结束 %.3f s）；"
              "撞击瞬态 flag %d/%d（fmax_early 中位 %.3f，0.1 s 回落中位 %.2f）" %
              (C.band(s.frames_to_50_z3)[0], C.band(s.frames_to_50_raw)[0],
               C.band(s.npkts_to_50)[0], C.band(s.max_dfrm_frac)[0],
               C.band(s.jump_frames)[0], C.band(s.t_step_end)[0],
               int(s.transient_flag.sum()), len(s), C.band(s.fmax_early)[0],
               C.band(s.transient_decay)[0]))
    print("\n== Q5 阶跃成立判据（主判：持续穿越 t25_sus ≤ 0.05 s；对照：字面首次穿越）==")
    for kind in ("onset", "restep", "unload", "partial_unload"):
        s = df[df.kind == kind]
        if not len(s):
            continue
        print("   %-15s 主判 %2d/%2d 成立 (t25_sus 中位 %s) | 字面 %2d/%2d (t25 中位 %s)" %
              (kind, int(s.step_ok.sum()), len(s), C.fmt_band(s.t25_sus, 3, " s"),
               int(s.step_ok_naive.sum()), len(s), C.fmt_band(s.t25, 3, " s")))
    print("\n完成。")


if __name__ == "__main__":
    main()
