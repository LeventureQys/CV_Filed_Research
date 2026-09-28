# -*- coding: utf-8 -*-
"""T7-A / step2：卸载事件的**纯数据**时间特征与类型分类（T7-Q1 / Q4）。

只用原始读数（raw），不叠加任何算法（§A 纪律）。
- 事件表列名与 T4-A 的加载事件冻结表 `t4a_morphology.csv` **逐列对齐**（便于对照），
  另加卸载专属列（残差、过冲、类型、±1 包敏感性、轮次…）。
- 归一化：`J_time = z0_ref − Z̄(pre)`（卸载到"加载前空载电平"的幅度，模拟 T4-A 口径）；
  完成度 `z_at_*` = (Z̄(pre) − Z̄(t0+τ)) / |J_frozen|（**卸载方向取镜像**，见报告 §2）。
- 时间指标一律从 `t0`（真实卸载沿 = 最后一个仍在旧电平 10% 带内的帧）起算。

产出：results/t7_unload_events.csv、results/t7_unload_shape.csv、results/_t7a_2_events.log
用法：python scripts/t7a_2_events.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402

LOG = C.Log("2_events")
TAU_GRID = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0,
                     3.0, 5.0, 8.0, 15.0, 30.0, 60.0])
MAIN_CH = {"右拇指指尖": 17, "左拇指指尖": 18, "四指指尖": 11}


def q(x, lo=10, hi=90):
    x = np.asarray([v for v in np.asarray(x, float) if np.isfinite(v)])
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (float(np.median(x)), float(np.percentile(x, lo)),
            float(np.percentile(x, hi)), int(len(x)))


def build(rec_row, ds, d, gaps, episodes, tol_frac=(0.005, 0.01)):
    """把一个候选行的几何 + 指标算全。

    两条平滑线（**关键口径声明**）：
      ds  = 中心 0.5 s 中值（= 指标字典的 Z̄）→ 只用于电平类量（pre/post/J/判稳/残余/过冲判据）；
      dst = **因果（拖尾）0.1 s 中值** → 用于时间类量（t10~t95、u_at_*、谷值、单帧跃变）。
    理由：中心 0.5 s 中值在边沿处 ±0.25 s 双向取数，会把"边沿完成度"整体前移，
    使 t50 塌到 1~2 帧（实测 T4-A 的卸载 t50=0.01~0.02 s 就是这个量级）。本表两套数并报。
    """
    tu, dtm = d["tu"], d["dtm"]
    t0 = rec_row["t0"]
    i0 = rec_row["i0"]
    n = len(ds)
    dst = C.med_smooth(d["tot"], max(1, int(round(0.1 / dtm))))
    # ── 受载段与前置空载段 ──
    ep = next((e for e in episodes if e["i0"] <= i0 <= e["i1"]), None)
    if ep is None:
        return None
    gap_pre = None
    for g in gaps:
        if g["i1"] <= ep["i0"]:
            gap_pre = g
        else:
            break
    if gap_pre is not None:
        z0, z0_n = C.zero_level(ds, gap_pre, dtm, skip_in_s=0.5, use_last_s=3.0)
        z0_src = "local"
        idle_win = ds[max(gap_pre["i0"], gap_pre["i1"] - int(3.0 / dtm)):gap_pre["i1"]]
    else:
        z0, z0_n = float(np.percentile(ds, 5)), 0
        z0_src = "global_p5"
        idle_win = ds[:int(3.0 / dtm)]
    noise = float(1.4826 * np.median(np.abs(idle_win - np.median(idle_win)))) \
        if len(idle_win) > 5 else float("nan")
    gap_post = next((g for g in gaps if g["i0"] >= i0), None)
    ep_next = next((e for e in episodes if e["i0"] > i0), None)
    t_end = float(tu[-1])
    if ep_next is not None:
        t_end = min(t_end, float(tu[ep_next["i0"]]) - 0.5)
    obs_end = min(t_end, t0 + 60.0)
    obs_win = obs_end - t0

    pre, post = float(rec_row["pre"]), float(rec_row["post"])
    Jf = float(rec_row["J"])                     # 指标字典口径 J（可 NaN）
    Jt = z0 - pre                                # 时间归一化用（负）
    if not (Jt < 0):
        return None
    absJt = abs(Jt)
    tau = tu - t0
    m = tau >= 0.0
    prog_f = np.where(m, (pre - ds) / abs(Jf) if np.isfinite(Jf) and abs(Jf) > 1e-12 else np.nan,
                      np.nan)
    prog_t = np.where(m, (pre - dst) / absJt, np.nan)          # 因果线（时间指标用）
    prog_c = np.where(m, (pre - ds) / absJt, np.nan)           # 中心线（与 T4-A 口径对照）
    cross = np.where(m, (dst[i0] - dst) / absJt, np.nan)       # 以 t0 那一帧为基准（T4-A 式）

    def first_tau(prog, lev):
        idx = np.where(m & np.isfinite(prog) & (prog >= lev))[0]
        return float(tu[idx[0]] - t0) if len(idx) else float("nan")

    def at_tau(prog, x):
        if x > obs_win:
            return float("nan")
        j = i0 + int(round(x / dtm))
        return float(prog[j]) if j < n else float("nan")

    t10, t50, t80, t90, t95 = (first_tau(prog_t, 0.10), first_tau(prog_t, 0.50),
                               first_tau(prog_t, 0.80), first_tau(prog_t, 0.90),
                               first_tau(prog_t, 0.95))
    t50_c, t90_c = first_tau(prog_c, 0.50), first_tau(prog_c, 0.90)
    frames_90_10 = (t90 - t10) / dtm if np.isfinite(t90) and np.isfinite(t10) else float("nan")
    # 单帧跃变占比（±1 包敏感性另算；用因果线，避免中心线把跃变摊平）
    j1 = min(n, i0 + int(1.0 / dtm))
    dif = np.abs(np.diff(dst[i0:j1])) if j1 > i0 + 1 else np.array([np.nan])
    step_frame_frac = float(np.nanmax(dif)) / absJt if len(dif) and np.isfinite(np.nanmax(dif)) else float("nan")
    top3 = float(np.nansum(np.sort(dif)[-3:])) / absJt if len(dif) >= 3 else float("nan")
    # 谷值 / 过冲 / 残余
    j8 = min(n, i0 + int(8.0 / dtm))
    kk = i0 + int(np.argmin(dst[i0:j8]))
    trough_s = float(tu[kk] - t0)
    trough_depth_post = (post - float(dst[kk])) if np.isfinite(post) else float("nan")
    min_raw = float(np.min(dst[i0:j8]))
    dip_zero = z0 - min_raw
    tol_z = max(3.0 * noise, tol_frac[0] * absJt) if np.isfinite(noise) else tol_frac[0] * absJt
    tol_r = max(3.0 * noise, tol_frac[1] * absJt) if np.isfinite(noise) else tol_frac[1] * absJt
    below = dst[i0:j8] < (z0 - tol_z)
    n_below = int(np.sum(below))
    # 残余（观测窗末 10 s / 整窗）
    late_a = i0 + int(max(0.0, obs_win - 10.0) / dtm)
    late_b = min(n, i0 + int(obs_win / dtm))
    resid_late = float(np.median(ds[late_a:late_b])) - z0 if late_b > late_a else float("nan")
    resid = (post - z0) if np.isfinite(post) else float("nan")
    # 三类型（T7-Q4）：**先看"不回来"**（稳态残余），再看"过冲回零"（跌破加载前空载电平），
    # 最后才是"单调回零"。阈值：tol_r = max(3σ_idle, 1%·|J|)；tol_z = max(3σ_idle, 0.5%·|J|)。
    if np.isfinite(resid_late) and abs(resid_late) > tol_r:
        typ = "C_no_return_pos" if resid_late > 0 else "C_no_return_neg"
    elif n_below >= max(3, int(0.1 / dtm)):
        typ = "B_overshoot_zero"                # 过冲回零：先跌破加载前空载电平再回升
    else:
        typ = "A_monotone_zero"                 # 单调回零
    dip_dur_s = float(n_below) * dtm
    # 「贴到绝对零」的时间占比（卸载后 1.5 s 内）：阈值取 max(3σ_idle, 0.2%·|J|)
    tol0 = max(3.0 * noise, 0.002 * absJt) if np.isfinite(noise) else 0.002 * absJt
    j15 = min(n, i0 + int(1.5 / dtm))
    seg0 = dst[i0:j15]
    at_abs0_frac = float(np.mean(seg0 <= tol0)) if len(seg0) else float("nan")
    # 峰值 / 前置比例（与 T4-A 列对齐）
    peak = float(np.max(ds[:i0 + 1]))
    # ±1 包敏感性（包间隔 = timestamp 正差分 p90；实录 40 ms、恒载 17.6 ms）
    step_pkt = max(1, int(round(rec_row["pkt_dt"] / dtm))) if rec_row["pkt_dt"] > dtm else 0
    sens = {}
    for sgn, nm in ((-1, "m1pkt"), (+1, "p1pkt")):
        i0s = int(np.clip(i0 + sgn * step_pkt, 0, n - 1))
        pr = np.where((tu - tu[i0s]) >= 0.0, (dst[i0s] - dst) / absJt, np.nan)
        idx = np.where(np.isfinite(pr) & (pr >= 0.5))[0]
        sens[f"t50_{nm}"] = float(tu[idx[0]] - tu[i0s]) if len(idx) else float("nan")
        idx = np.where(np.isfinite(pr) & (pr >= 0.90))[0]
        sens[f"t90_{nm}"] = float(tu[idx[0]] - tu[i0s]) if len(idx) else float("nan")
        jj = i0s + int(round(0.2 / dtm))
        sens[f"u_at_02_{nm}"] = float(pr[jj]) if jj < n else float("nan")
    # 设备速率（原始采样，未插值）上的最大单包跳变：这才是"单帧跳变"的真实分辨率
    traw, Xraw = d["t"], d["tot_raw"]
    mr = (traw >= t0) & (traw <= t0 + 1.0)
    if mr.sum() >= 2:
        step_pkt_frac = float(np.max(np.abs(np.diff(Xraw[mr])))) / absJt
    else:
        step_pkt_frac = float("nan")
    row = dict(
        key="U-" + rec_row["key_src"], rec=rec_row["rec"], dom=d["dom"],
        main_ch=MAIN_CH.get(rec_row["rec"].split("/")[0], -1),
        t_on=t0, kind=rec_row["kind"], armed=np.nan, armed_10=np.nan, armed_20=np.nan,
        armed_30=np.nan, kind_prej="", clean=bool(rec_row["stable"]),
        n_other_near=int(0 if rec_row["stable"] else 1),
        pre=pre, post=post, jump=Jf, ratio=rec_row["step_frac"], peak=peak,
        pre_over_peak=pre / max(peak, 1e-12), pre_over_jump=(pre / abs(Jf) if np.isfinite(Jf)
                                                            and abs(Jf) > 1e-12 else np.nan),
        t10=t10, t50=t50, t80=t80, t90=t90, t95=t95,
        rise_frames_10_90=frames_90_10, n_frames_rise=frames_90_10,
        frames_to_50=(t50 / dtm if np.isfinite(t50) else np.nan),
        frames_to_90=(t90 / dtm if np.isfinite(t90) else np.nan),
        step_frame_frac=step_frame_frac, top3_frame_frac=top3,
        **{f"z_at_{k}": at_tau(prog_f, v) for k, v in
           (("005", 0.05), ("01", 0.10), ("02", 0.20), ("03", 0.30), ("05", 0.50),
            ("10", 1.0), ("20", 2.0), ("50", 5.0))},
        **{f"u_at_{k}": at_tau(prog_t, v) for k, v in
           (("005", 0.05), ("01", 0.10), ("02", 0.20), ("05", 0.50), ("10", 1.0),
            ("20", 2.0), ("50", 5.0))},
        t50_c=t50_c, t90_c=t90_c, u_at_02_c=at_tau(prog_c, 0.2),
        t50_cross=first_tau(cross, 0.5), z_at_02_cross=at_tau(cross, 0.2),
        min_raw=min_raw, min_abs0=(min_raw if Jt < 0 else np.nan),
        step_pkt_frac=step_pkt_frac, pkt_dt_p90=rec_row["pkt_dt"],
        exp_tau=np.nan, exp_beta=np.nan, exp_rms=np.nan, pow_p=np.nan, pow_tau=np.nan,
        pow_rms=np.nan, T_slope=np.nan, t_on5=np.nan, d_t_on5=np.nan,
        src_ev="t7a", pkt_dt=rec_row["pkt_dt"],
        # ── 卸载专属 ──
        dir="unload", J_frozen=Jf, J_time=Jt, z0_ref=z0, z0_src=z0_src, z0_n=int(z0_n),
        round_idx=int(rec_row["round_idx"]), hold_s=float(rec_row["hold_s"]),
        stable=bool(rec_row["stable"]), stab_pre=rec_row["stab_pre"],
        stab_post=rec_row["stab_post"], post_win_s=rec_row["post_win_s"],
        noise_idle=noise, tol_zero=tol_z, tol_resid=tol_r,
        trough_s=trough_s, trough_depth_post=trough_depth_post,
        dip_zero=dip_zero, dip_zero_pct=(dip_zero / absJt),
        n_frames_below_zero=n_below, dip_dur_s=dip_dur_s, at_abs0_frac=at_abs0_frac,
        resid=resid, resid_pct=(resid / absJt if np.isfinite(resid) else np.nan),
        resid_late=resid_late, resid_late_pct=(resid_late / absJt if np.isfinite(resid_late) else np.nan),
        obs_win_s=obs_win, type=typ, gap_post_idx=(int(gaps.index(gap_post)) if gap_post else -1),
        **sens)
    return row, prog_t


def main():
    C.log_reconfigure()
    LOG("=" * 118)
    LOG("T7-A step2：卸载事件时间特征（T7-Q1）与类型分类（T7-Q4）——**只用原始读数**")
    LOG(f"时间网格 100 Hz；τ 网格 {list(TAU_GRID)}；方向约定：J<0，z_at_*/u_at_* 为「卸载完成度」"
        "（越大＝越接近加载前空载电平）；t0 = 真实卸载沿（最后一个仍在旧电平 10% 带内的帧）")
    LOG("=" * 118)
    t4a = C.load_t4a_morph()
    t4a_u = t4a[t4a["kind"] == "unload"]
    rows, shapes, recon = [], [], []
    for tag, path in C.ALL:
        if not os.path.exists(path):
            LOG(f"[缺文件] {tag}")
            continue
        d = C.prep(tag)
        acc, allc, edges, ds = C.detect_unloads(d["tu"], d["tot"], d["dtm"], d["pkt_dt"])
        gaps, eps, floor, peak = C.parse_plateaus(ds, d["tu"], d["dtm"])
        # 轮次编号：受载段序号
        for k, r in enumerate(allc):
            r["rec"] = tag
            r["key_src"] = f"{tag.replace('/', '_')}_{r['t0']:.2f}"
            ep = next((e for e in eps if e["i0"] <= r["i0"] <= e["i1"]), None)
            r["round_idx"] = (eps.index(ep) + 1) if ep is not None else -1
            r["hold_s"] = (float(d["tu"][r["i0"]] - d["tu"][ep["i0"]]) if ep is not None else np.nan)
            recon.append(r)
            if not (r["kind"] == "unload" and r["stable"]):
                continue
            out = build(r, ds, d, gaps, eps)
            if out is None:
                LOG(f"[跳过] {tag} @{r['t0']:.2f}：无法定位前置空载电平或 J_time≥0")
                continue
            row, prog = out
            rows.append(row)
            for x in TAU_GRID:
                j = r["i0"] + int(round(x / d["dtm"]))
                if j < len(ds) and x <= row["obs_win_s"]:
                    shapes.append(dict(key=row["key"], rec=tag, kind_rec=d["kind"], tau=x,
                                       prog=float(prog[j]), J_time=row["J_time"],
                                       J_frozen=row["J_frozen"]))
    ev = pd.DataFrame(rows)
    sh = pd.DataFrame(shapes)
    C.save(ev, "t7_unload_events.csv")
    C.save(sh, "t7_unload_shape.csv")

    LOG("")
    LOG("─" * 118)
    LOG("逐事件表（n=%d；Δ 相对 z0 一律负向为正）" % len(ev))
    LOG("─" * 118)
    cols = ["rec", "round_idx", "t_on", "hold_s", "pre", "post", "J_time", "z0_ref",
            "t10", "t50", "t80", "t90", "t95", "u_at_01", "u_at_02", "step_frame_frac",
            "trough_s", "min_raw", "dip_zero", "dip_zero_pct", "dip_dur_s",
            "resid", "resid_pct", "resid_late", "resid_late_pct",
            "obs_win_s", "type"]
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        LOG(ev[cols].round(4).to_string(index=False))
    LOG("")
    LOG("─" * 118)
    LOG("平滑口径对照：中心 0.5 s 中值（Z̄，指标字典默认）vs 因果 0.1 s 中值（本任务用于时间指标）")
    LOG("─" * 118)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        LOG(ev[["rec", "t_on", "t50_c", "t50", "t90_c", "t90", "u_at_02_c", "u_at_02",
                "step_frame_frac"]].round(4).to_string(index=False))
    dd50 = (ev["t50_c"] - ev["t50"]).abs()
    dd90 = (ev["t90_c"] - ev["t90"]).abs()
    LOG(f"   → 中心线比因果线把 t50 前移 中位 {np.nanmedian(ev['t50']-ev['t50_c']):+.3f} s"
        f"（|差| 中位 {dd50.median():.3f} s、最大 {dd50.max():.3f} s）；"
        f"t90 前移 中位 {np.nanmedian(ev['t90']-ev['t90_c']):+.3f} s（|差| 最大 {dd90.max():.3f} s）")
    LOG("   含义：**用中心 0.5 s 中值量卸载沿会把瞬态时间整体压缩 ±0.25 s**，"
        "t50 会被压到 1~2 帧（这正是既有 T4-A 卸载 t50=0.01~0.02 s 的量级来源之一）。")
    LOG("")
    LOG("归一化基对照：J_time（到前置空载电平）vs J_frozen（指标字典 post 窗）：")
    rt = (ev["J_time"] / ev["J_frozen"]).replace([np.inf, -np.inf], np.nan)
    LOG(f"   比值 J_time/J_frozen 中位 {rt.median():.4f}、p10~p90 "
        f"{np.nanpercentile(rt,10):.4f}~{np.nanpercentile(rt,90):.4f}（越接近 1 说明"
        f"卸载确实停在加载前空载电平上）")

    def agg(sub, name):
        LOG("")
        LOG(f"【{name}】n={len(sub)}" + ("（n≤3，仅作定性参考）" if len(sub) <= 3 else ""))
        for c in ("t10", "t50", "t80", "t90", "t95", "step_frame_frac", "u_at_01", "u_at_02",
                  "trough_s", "dip_zero_pct", "resid_pct", "resid_late_pct"):
            md, p10, p90, nn = q(sub[c])
            LOG(f"   {c:<18} 中位 {md:>10.4f}   p10~p90 {p10:>10.4f} ~ {p90:>10.4f}   n={nn}")

    agg(ev, "全部 unload（严格冻结集）")
    agg(ev[ev["rec"].str.contains("指尖")], "恒载 9 组（力域）")
    agg(ev[~ev["rec"].str.contains("指尖")], "实录（ADC 域）")
    LOG("")
    LOG("按录制家族的时长口径提示：包间隔（timestamp 正差分 p90）恒载 17.6 ms（帧内中位 15.4 ms）、"
        "实录 40.2 ms（每包含 ~4 行、行间时间戳仅差 1~2 µs ⇒ 有效采样率 ≈25 Hz）")
    LOG("")
    LOG("卸载沿的'单包性'检查：")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        LOG(ev[["rec", "t_on", "pkt_dt_p90", "t10", "t50", "t90", "step_frame_frac",
                "step_pkt_frac", "min_raw"]].round(4).to_string(index=False))
    LOG(f"   网格级最大单帧降幅占 |J| 的比例 中位 {np.nanmedian(ev['step_frame_frac']):.3f}；"
        f"设备速率（原始采样）级最大单包降幅占 |J| 的比例 中位 {np.nanmedian(ev['step_pkt_frac']):.3f}")
    LOG("   ⇒ 卸载沿在**1~2 包内走完全部幅度**：t90 中位 %.3f s，与包间隔同量级，"
        "属于分辨率极限，不能再往下拆（τ<0.2 s 结论均处于时间轴畸变区）。"
        % np.nanmedian(ev["t90"]))
    LOG("")
    LOG("±1 包敏感性（τ<0.2 s 与 40 ms 包间隔同量级，必须看这一节）：")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        LOG(ev[["rec", "t_on", "pkt_dt", "t50", "t50_m1pkt", "t50_p1pkt",
                "t90", "t90_m1pkt", "t90_p1pkt", "u_at_02", "u_at_02_m1pkt",
                "u_at_02_p1pkt"]].round(4).to_string(index=False))
    d50 = np.nanmax(np.abs(ev[["t50_m1pkt", "t50_p1pkt"]].to_numpy(float) - ev[["t50"]].to_numpy(float)))
    d90 = np.nanmax(np.abs(ev[["t90_m1pkt", "t90_p1pkt"]].to_numpy(float) - ev[["t90"]].to_numpy(float)))
    LOG(f"   → ±1 包引起的 t50 变化 ≤ {d50:.3f} s、t90 变化 ≤ {d90:.3f} s")
    LOG("")
    LOG("类型分类（T7-Q4）：")
    for k, v in ev["type"].value_counts().items():
        LOG(f"   {k:<18} {v:>3} 个   ({v/len(ev)*100:.0f}%)")
    LOG("   规则（先判'不回来'，再判'过冲'）：")
    LOG("     |resid_late| > max(3σ_idle, 1%·|J|)          → C_no_return_pos / C_no_return_neg（不回来）")
    LOG("     否则 dip_zero > max(3σ_idle, 0.5%·|J|) 且持续 ≥0.1 s → B_overshoot_zero（过冲回零）")
    LOG("     否则                                          → A_monotone_zero（单调回零）")
    LOG("   σ_idle = 1.4826·MAD(前置空载段)；dip_zero = z0_ref − min(读数[t0,t0+8 s])")
    LOG("")
    LOG("阈值敏感性（同一批事件换阈值时的类型比例）：")
    for fz, fr in ((0.002, 0.005), (0.005, 0.01), (0.01, 0.02)):
        t = ev.apply(lambda r: ("C_no_return" if abs(r["resid_late"]) >
                                max(3 * r["noise_idle"], fr * abs(r["J_time"]))
                                else ("B_overshoot" if r["dip_zero"] >
                                      max(3 * r["noise_idle"], fz * abs(r["J_time"]))
                                      else "A_monotone")), axis=1)
        LOG(f"   dip>{fz*100:.1f}%·|J| / resid>{fr*100:.1f}%·|J|：" +
            "  ".join(f"{k}={v}" for k, v in t.value_counts().items()))
    LOG("")
    LOG("分录制类型分布：")
    LOG(ev.groupby(["rec", "type"]).size().to_string())
    LOG("")
    LOG("─" * 118)
    LOG("与 T4-A 冻结表的口径交叉核对（同一批 14 个事件）")
    LOG("─" * 118)
    chk = []
    for _, r in t4a_u.iterrows():
        m = ev[(ev["rec"] == r["rec"]) & (np.abs(ev["t_on"] - float(r["t_on"])) < 3.0)]
        if len(m) == 0:
            chk.append((r["rec"], float(r["t_on"]), np.nan, np.nan, np.nan, np.nan))
            continue
        mm = m.iloc[0]
        chk.append((r["rec"], float(r["t_on"]), float(r["t50"]), float(mm["t50"]),
                    float(r["z_at_02"]), float(mm["z_at_02"])))
    ck = pd.DataFrame(chk, columns=["rec", "t4a_t_on", "t4a_t50", "t7a_t50",
                                    "t4a_z_at_02", "t7a_z_at_02"])
    with pd.option_context("display.width", 200):
        LOG(ck.round(4).to_string(index=False))
    d = (ck["t7a_t50"] - ck["t4a_t50"]).abs()
    LOG(f"   t50 差异：中位 {d.median():.3f} s、最大 {d.max():.3f} s（口径差异 = "
        f"T4-A 以 Z(t_on) 为基准、本任务以 pre 窗中位为基准；两者在同一帧几乎相等）")
    dz = (ck["t7a_z_at_02"] - ck["t4a_z_at_02"]).abs()
    LOG(f"   z_at_02 差异：中位 {dz.median():.4f}、最大 {dz.max():.4f}")
    LOG("")
    LOG("时间轴畸变区提示：τ<0.2 s 的结论落在 40 ms 包量化与 T2 的 0.2 s 畸变区内，"
        "本报告只在 u_at_02 / t50 上引用，t10 及更早只作参考。")
    # ─────────────────────────────────────────────────────────────────
    LOG("")
    LOG("─" * 118)
    LOG("【C-5 支撑】同一份录制内「加载沿 vs 卸载沿」时间常数配对（加载侧取 T4-A 冻结表的")
    LOG("kind=onset 事件；卸载侧取本任务严格集。两侧口径不同，只作数量级对照，")
    LOG("完整的对称性专项属 T7-B）")
    LOG("─" * 118)
    t4o = t4a[t4a["kind"] == "onset"]
    sym = []
    for tag, sub in ev.groupby("rec"):
        o = t4o[t4o["rec"] == tag]
        if len(o) == 0:
            continue
        sym.append(dict(rec=tag, n_load=len(o), n_unload=len(sub),
                        load_t50_med=float(np.median(o["t50"])),
                        load_t90_med=float(np.median(o["t90"])),
                        unload_t50_med=float(np.median(sub["t50"])),
                        unload_t90_med=float(np.median(sub["t90"])),
                        ratio_t90=(float(np.median(o["t90"])) / float(np.median(sub["t90"]))
                                   if np.median(sub["t90"]) > 1e-9 else float("nan")),
                        ratio_t50=(float(np.median(o["t50"])) / float(np.median(sub["t50"]))
                                   if np.median(sub["t50"]) > 1e-9 else float("nan"))))
    sy = pd.DataFrame(sym)
    with pd.option_context("display.width", 220):
        LOG(sy.round(4).to_string(index=False))
    if len(sy):
        LOG(f"   同录制配对 n={len(sy)}：t90 比值（加载/卸载）中位 "
            f"{np.nanmedian(sy['ratio_t90']):.2f}、p10~p90 "
            f"{np.nanpercentile(sy['ratio_t90'],10):.2f}~{np.nanpercentile(sy['ratio_t90'],90):.2f}；"
            f"t50 比值中位 {np.nanmedian(sy['ratio_t50']):.2f}")
        LOG("   ⇒ 加载沿（零基线 onset）比卸载沿慢 1~2 个数量级 ⇒ **正负方向的时间常数明显不对称**；")
        LOG("     T4-A 的 E2 证据因此只能读作「卸载沿本身极快」这个方向性事实，"
            "不能反推「传感器在带载状态的反应与零基线加载一样快」。")
    LOG.close("python scripts/t7a_2_events.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
