# -*- coding: utf-8 -*-
"""t4a_02_mechanism：T4-Q3 机制证据（判定「差异在输入而不在传感器」）。

四条**互相独立**的证据（详见报告 §3）：
  E1 单帧跃变/上升沿帧结构：onset 的上升集中在极少数帧，restep 分散（逐帧微分形状）。
  E2 卸载对称性：**同一录制、同一带载状态**下的减载是瞬时完成的 → 该状态下传感器本身不慢，
     线性粘弹响应正负方向对称 ⇒ 加载沿的慢只能来自输入。
  E3 通道间一致性与同步性：同一事件内各通道的归一化形状/上升沿时刻是否一致
     （一致 ⇒ 共模输入调制；不一致且有序 ⇒ 局部接触扩展）。
  E4 输入反卷积：用 onset 实测的阶跃响应当核，反演每个事件的**等效输入上升时间 T_ramp**；
     若 restep 的 T_ramp 落在 0.3~1.2 s 而 onset 落在 ≤0.1 s，则既有论断被独立验证。

产出：results/t4a_input_recover.csv、results/t4a_channel_consistency.csv、
      results/t4a_mechanism.csv
日志：results/_t4a_02_mechanism.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t4a_common as C
import t4a_ad_lib as L

FIT_S = 3.0            # 反卷积拟合窗长（相对拟合原点）
PRE_OFF = 0.30         # 拟合原点取在 t_on 之前 0.30 s（容纳「斜坡起点早于 t_on」）
NG = int(FIT_S / C.DT)                                  # 核网格 0..3 s
NTR = int((FIT_S + PRE_OFF) / C.DT) + 1                 # 轨迹网格 0..3.3 s
T_GRID = np.concatenate([[0.0], np.arange(0.01, 0.5001, 0.01), np.arange(0.55, 4.0001, 0.05)])
TAU0_GRID = np.arange(0.0, PRE_OFF + 1e-9, 0.05)
CH_TAU = np.array([0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0])


def cum_kernel(g):
    cg = np.zeros(len(g) + 1)
    cg[1:] = np.cumsum(g) * C.DT
    return cg


def ramp_model(T, tau0, cg, g):
    """输入 = 从 τ0 起、时长 T 的线性斜坡（幅值 1），输出（归一化阶跃响应 g 的卷积）。

    y(τ) = (1/T)·[cg(τ−τ0) − cg(τ−τ0−T)]；T = 0 ⇒ 纯阶跃，直接取 g。"""
    i = np.arange(NTR)
    i0 = int(round(tau0 / C.DT))
    if T <= 0:
        return g[np.clip(i - i0, 0, NG)]
    iT = max(1, int(round(T / C.DT)))
    hi = np.clip(i - i0, 0, NG)
    lo = np.clip(i - i0 - iT, 0, NG)
    return (cg[hi] - cg[lo]) / T


def fit_event(y, g, cg, jnorm, tgrid=T_GRID, tau0_grid=TAU0_GRID):
    """网格搜 (τ0, T)（每个组合用最小二乘解幅值 A），返回最优参数与残差（以 |J| 归一）。

    对照组 = **同样自由搜 τ0 的最优阶跃模型**（T=0），保证「斜坡 vs 阶跃」是公平比较。"""
    m = np.isfinite(y)
    yy = y[m]
    if jnorm <= 1e-12 or yy.size < 20:
        return dict(T_ramp=np.nan, tau0=np.nan, A_ramp=np.nan, rmse_ramp_pct=np.nan,
                    rmse_step_pct=np.nan, gain_ramp=np.nan)
    best, step_rms = None, np.nan
    for t0 in tau0_grid:
        for T in tgrid:
            mm = ramp_model(T, t0, cg, g)[m]
            den = float((mm * mm).sum())
            if den <= 1e-12:
                continue
            A = float((yy * mm).sum() / den)
            rms = float(np.sqrt(np.mean((yy - A * mm) ** 2)))
            if T <= 0 and (not np.isfinite(step_rms) or rms < step_rms):
                step_rms = rms
            if best is None or rms < best[3]:
                best = (T, t0, A, rms)
    if best is None:
        return dict(T_ramp=np.nan, tau0=np.nan, A_ramp=np.nan, rmse_ramp_pct=np.nan,
                    rmse_step_pct=np.nan, gain_ramp=np.nan)
    T, t0, A, rms = best
    return dict(T_ramp=float(T), tau0=float(t0), A_ramp=float(A),
                rmse_ramp_pct=100 * rms / jnorm,
                rmse_step_pct=(100 * step_rms / jnorm if np.isfinite(step_rms) else np.nan),
                gain_ramp=(float(step_rms / rms) if rms > 0 else np.nan))


def main():
    C.start_log()
    mp = os.path.join(C.RES, "t4a_morphology.csv")
    if not os.path.isfile(mp):
        print("!! 先运行 t4a_01_morphology.py")
        return 1
    mor = pd.read_csv(mp)
    mor = mor[mor.jump.notna()].copy()

    # ── 逐录制重建 Zs / 通道网格 ──
    recs = {}
    for r in C.RECS:
        tu, Xu, pkt, nraw = C.load_grid(r)
        Z = Xu.sum(axis=1)
        Zs = L.med_smooth(Z, C.KSM)
        recs[r["key"]] = dict(tu=tu, Xu=Xu, Z=Z, Zs=Zs, amp=float(np.percentile(Z, 99.5)
                                                                  - np.percentile(Z, 0.5)),
                              pkt=pkt, main=r["ch"], rec=r["rec"], dom=r["dom"])

    # ── ① 传感器阶跃响应核 g_step(u)：每域取「最快的 onset」中位形状（u 以 t_on 为 0） ──
    gtau = np.arange(NG + 1) * C.DT
    kernels, kern_src = {}, {}
    for dom in ("显示域", "ADC域"):
        sub = mor[(mor.dom == dom) & (mor.jump > 0) & (mor.kind == "onset") & mor.clean
                  & mor.z_at_02.notna()]
        sub = sub.sort_values("z_at_02", ascending=False)
        use = sub.head(max(3, int(np.ceil(len(sub) / 2.0))))
        if len(use) < 3:
            use = sub
        curves = []
        for _, q in use.iterrows():
            R = recs[q.key]
            k = int(q.k_on)
            seg = R["Zs"][k:k + NG + 1]
            if len(seg) < NG + 1:
                seg = np.pad(seg, (0, NG + 1 - len(seg)), mode="edge")
            curves.append((seg - q.pre) / q.jump)
        g = np.median(np.array(curves), axis=0)
        g = np.maximum.accumulate(np.clip(g, 0.0, None))
        if g[-1] > 1e-9:
            g = g / g[-1]
        kernels[dom] = g
        kern_src[dom] = (len(use), list(use.rec + "@" + use.t_on.astype(str)))
        print("阶跃响应核 g_step[%s]：n=%d 事件（%s）" % (dom, len(use), "; ".join(kern_src[dom][1])))
        for tt in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0):
            print("    g_step(%.2f s) = %.3f" % (tt, g[int(round(tt / C.DT))]))
        print("    g_step(5 s 外推=>1.0)；核单调化后 g(0)=%.3f" % g[0])

    # ── ② 反卷积：每个加载事件的等效输入上升时间 T_ramp 与输入起点 ──
    rows = []
    for _, q in mor[mor.jump > 0].iterrows():
        R = recs[q.key]
        k = int(q.k_on)
        i0 = max(0, k - int(PRE_OFF / C.DT))
        seg = R["Zs"][i0:i0 + NTR]
        if len(seg) < NTR:
            seg = np.pad(seg, (0, NTR - len(seg)), mode="edge")
        y = seg - q.pre
        g = kernels[R["dom"]]
        cg = cum_kernel(g)
        f = fit_event(y, g, cg, abs(q.jump))
        f["t_in_start"] = round(float(q.t_on) - PRE_OFF + f.get("tau0", np.nan), 3)
        rows.append(dict(key=q.key, rec=q.rec, dom=q.dom, t_on=q.t_on, kind=q.kind, clean=q.clean,
                         jump=q.jump, pre_over_peak=q.pre_over_peak, z_at_02=q.z_at_02,
                         T_slope=q.T_slope, frames_to_50=q.frames_to_50,
                         step_frame_frac=q.step_frame_frac, top3_frame_frac=q.top3_frame_frac,
                         d_t_on5=q.d_t_on5, **f))
    ir = pd.DataFrame(rows)
    ir.round(4).to_csv(os.path.join(C.RES, "t4a_input_recover.csv"), index=False,
                       encoding="utf-8-sig")
    print("\n== 等效输入上升时间 T_ramp（反卷积，n=%d）==" % len(ir))
    print(ir.groupby("kind")[["T_ramp", "rmse_ramp_pct", "rmse_step_pct", "gain_ramp", "A_ramp"]
                              ].median().round(3).to_string())
    print("\n== 干净事件 T_ramp 明细 ==")
    cc = ir[ir.clean]
    print(cc[["rec", "kind", "t_on", "jump", "pre_over_peak", "z_at_02", "T_slope", "T_ramp",
              "rmse_ramp_pct", "rmse_step_pct", "gain_ramp"]].round(3).to_string(index=False))

    # ── ③ 通道间一致性 ──
    chrows = []
    for _, q in mor[mor.jump > 0].iterrows():
        R = recs[q.key]
        k = int(q.k_on)
        Xu, n = R["Xu"], len(R["Zs"])
        Zc = np.column_stack([L.med_smooth(Xu[:, c], C.KSM) for c in range(Xu.shape[1])])
        pre_i = slice(max(0, k - int(2.0 / C.DT)), k)
        post_i = slice(min(n, k + int(4.0 / C.DT)), min(n, k + int(6.0 / C.DT)))
        Jc = np.median(Zc[post_i, :], axis=0) - np.median(Zc[pre_i, :], axis=0)
        sgn = 1.0 if q.jump > 0 else -1.0
        Jc = Jc * sgn                                # 统一为「同向增量」
        sigma = 1.4826 * np.median(np.abs(np.diff(Xu[max(0, k - int(2.0 / C.DT)):k, :], axis=0)),
                                   axis=0) / np.sqrt(2)
        mx = np.nanmax(np.abs(Jc)) if Jc.size else np.nan
        act = np.where((np.abs(Jc) >= np.maximum(0.15 * mx, 5 * sigma)) & (np.abs(Jc) > 0))[0]
        t50c, sfc, curves = [], [], []
        for c in act:
            jc = Jc[c]
            z = Zc[k:min(n, k + int(5.0 / C.DT)), c] - np.median(Zc[pre_i, c])
            hit = np.where(z >= 0.5 * jc)[0]
            t50c.append(float(hit[0]) * C.DT if len(hit) else np.nan)
            a2 = max(0, k - 1)
            b2 = min(n - 1, k + int(1.5 / C.DT))
            dd = np.abs(np.diff(Xu[a2:b2, c]))
            sfc.append(float(dd.max() / abs(jc)) if dd.size and abs(jc) > 0 else np.nan)
            curves.append([(Zc[min(n - 1, k + int(round(t / C.DT))), c]
                            - np.median(Zc[pre_i, c])) / jc for t in CH_TAU])
        t50c = np.array(t50c, float)
        sfc = np.array(sfc, float)
        cv = np.array(curves, float) if curves else np.zeros((0, len(CH_TAU)))
        cc_r = np.nan
        if cv.shape[0] >= 3:
            cs = np.corrcoef(cv)
            iu = np.triu_indices(cv.shape[0], 1)
            cc_r = float(np.nanmedian(cs[iu]))
        chrows.append(dict(key=q.key, rec=q.rec, kind=q.kind, clean=q.clean, t_on=q.t_on,
                           n_ch=Xu.shape[1], n_active=int(len(act)),
                           active_frac=round(len(act) / Xu.shape[1], 3),
                           t50_ch_iqr=float(np.nanpercentile(t50c, 75) - np.nanpercentile(t50c, 25))
                           if len(t50c) >= 3 else np.nan,
                           stepfrac_ch_iqr=float(np.nanpercentile(sfc, 75) - np.nanpercentile(sfc, 25))
                           if len(sfc) >= 3 else np.nan,
                           t50_ch_med=float(np.nanmedian(t50c)) if len(t50c) else np.nan,
                           shape_corr_med=cc_r))
    ch = pd.DataFrame(chrows)
    ch.round(4).to_csv(os.path.join(C.RES, "t4a_channel_consistency.csv"), index=False,
                       encoding="utf-8-sig")
    print("\n== 通道间一致性（干净事件中位）==")
    print(ch[ch.clean].groupby("kind")[["n_active", "active_frac", "t50_ch_med", "t50_ch_iqr",
                                        "stepfrac_ch_iqr", "shape_corr_med"]].median()
          .round(3).to_string())

    # ── ④ 汇总证据表 ──
    ev = []
    ld = mor[(mor.jump > 0)]
    clean = ld[ld.clean]

    def add(eid, name, metric, ga, gb, contrast="中位差"):
        va, vb = ga[metric].to_numpy(float), gb[metric].to_numpy(float)
        ma, pa0, pa1, na = C.band(va)
        mb, pb0, pb1, nb = C.band(vb)
        ev.append(dict(evidence_id=eid, evidence=name, metric=metric,
                       group_a="onset", n_a=na, med_a=ma, p10_a=pa0, p90_a=pa1,
                       group_b="restep", n_b=nb, med_b=mb, p10_b=pb0, p90_b=pb1,
                       contrast=contrast, diff=(ma - mb), ratio=(ma / mb if mb else np.nan),
                       auc=C.auc(va, vb),
                       band_overlap=C.overlap_ratio((pa0, pa1), (pb0, pb1))))
    o, rs = clean[clean.kind == "onset"], clean[clean.kind == "restep"]
    ir_clean = ir[ir.clean]
    oi = ir_clean[ir_clean.kind == "onset"]
    ri = ir_clean[ir_clean.kind == "restep"]
    add("E1", "逐帧微分：单帧跃变占比 step_frame_frac", "step_frame_frac", o, rs)
    add("E1", "逐帧微分：前 3 帧承载比例 top3_frame_frac", "top3_frame_frac", o, rs)
    add("E1", "逐帧微分：承载 50% 台阶所需帧数", "frames_to_50", o, rs)
    add("E1", "逐帧微分：0.05 s 完成度 z_at_005", "z_at_005", o, rs)
    add("E1", "逐帧微分：0.10 s 完成度 z_at_01", "z_at_01", o, rs)
    add("E1", "逐帧微分：0.20 s 完成度 z_at_02", "z_at_02", o, rs)
    add("E4", "反卷积等效输入上升时间 T_ramp (s)", "T_ramp", oi, ri)
    add("E4", "反卷积：斜坡模型残差 (%)", "rmse_ramp_pct", oi, ri)
    add("E4", "反卷积：阶跃模型残差 (%)", "rmse_step_pct", oi, ri)
    add("E4", "反卷积：斜坡/阶跃残差改善倍数", "gain_ramp", oi, ri)
    add("E4", "辅助斜率估计 T_slope (s)", "T_slope", o, rs)
    add("E5", "加载沿与读数沿错位 d_t_on5 (s)", "d_t_on5", o, rs)
    chc = ch[ch.clean]
    add("E3", "通道间 t50 四分位距 (s)", "t50_ch_iqr", chc[chc.kind == "onset"],
        chc[chc.kind == "restep"])
    add("E3", "通道间形状相关中位", "shape_corr_med", chc[chc.kind == "onset"],
        chc[chc.kind == "restep"])
    add("E3", "活跃通道数", "n_active", chc[chc.kind == "onset"], chc[chc.kind == "restep"])

    # E2 卸载对称性：同一录制、同一带载状态
    un = mor[(mor.jump < 0) & (mor.clean) & (mor.jump.abs() >= 0.5 * mor.pre)]
    e2 = []
    for _, q in un.iterrows():
        same = clean[(clean.key == q.key) & (clean.kind == "restep")]
        e2.append(dict(rec=q.rec, t_unload=q.t_on, jump_unload=q.jump,
                       unload_step_frac=q.step_frame_frac, unload_z_at_02=q.z_at_02,
                       unload_t50=q.t50,
                       rec_restep_n=len(same),
                       restep_step_frac_med=(same.step_frame_frac.median() if len(same) else np.nan),
                       restep_z_at_02_med=(same.z_at_02.median() if len(same) else np.nan)))
    e2d = pd.DataFrame(e2)
    e2d.round(4).to_csv(os.path.join(C.RES, "t4a_unload_symmetry.csv"), index=False,
                        encoding="utf-8-sig")
    print("\n== E2 卸载对称性（同录制同带载状态，读数是 Z 总量）==")
    print(e2d.round(3).to_string(index=False))
    for m in ("step_frame_frac", "z_at_02", "t50"):
        va, vb = un[m].to_numpy(float), rs[m].to_numpy(float)
        ma, pa0, pa1, na = C.band(va)
        mb, pb0, pb1, nb = C.band(vb)
        ev.append(dict(evidence_id="E2", evidence="卸载 vs 带载加载（同一带载状态）", metric=m,
                       group_a="unload", n_a=na, med_a=ma, p10_a=pa0, p90_a=pa1,
                       group_b="restep", n_b=nb, med_b=mb, p10_b=pb0, p90_b=pb1,
                       contrast="中位差", diff=(ma - mb), ratio=(ma / mb if mb else np.nan),
                       auc=C.auc(va, vb),
                       band_overlap=C.overlap_ratio((pa0, pa1), (pb0, pb1))))
    df = pd.DataFrame(ev)
    df.round(4).to_csv(os.path.join(C.RES, "t4a_mechanism.csv"), index=False,
                       encoding="utf-8-sig")
    print("\n== 机制证据汇总（t4a_mechanism.csv）==")
    print(df[["evidence_id", "metric", "n_a", "med_a", "n_b", "med_b", "ratio", "auc",
              "band_overlap"]].round(3).to_string(index=False))
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
