# -*- coding: utf-8 -*-
"""t4b_q6_branch.py —— T4-Q6 / T4-Q7：误判代价（T1 口径）与"该不该硬分支"的裁决。

思路：**同一份算法（t4b_glm53_v6 补丁版，已用 t4b_verify_patch.py 证明补丁不改变行为）、
同一批实测事件，只改形状库与 κ**，逐帧跑完再按真值分组比较 T1 口径指标
⇒ 「误判代价」= 同一事件在错分支下的指标差（不是抽象损失）。

臂（全部作用在同一 13 份录制 / 100 Hz 网格）：
  M0_now       : 现状——ROM_onset + κ=1.30（对 restep 也用它 ⇒ 隐式把 restep 当 onset）
  M1_flat112   : 只把 κ 统一成 1.12（孤立 κ 这一项的贡献）
  M2_correctA  : ROM_onset + (onset κ=1.30 / restep κ=1.12)（= 现行代码真实配置）
  M2_correct   : 真值分支：onset→(ROM_onset,1.30)、restep→(ROM_restep,1.12)（**上界，不可实现**）
  M2_inverse   : **完全反着切**：onset→(ROM_restep,1.12)、restep→(ROM_onset,1.30)（最坏情形）
  M3_causal_P1 : 用 T4-Q5 判据 P1 做分支（pre 因果电平/因果滚动峰值 > 0.30 ⇒ 带载支）
  M4_warped    : 连续参数化（不分支）：统一 ROM_onset + 按前 0.6 s 实测增量自适应 α

产物：results/t4b_arm_metrics.csv、results/t4b_branch_confusion.csv、
      results/t4b_decision_agreement.csv、results/t4b_rom_restep.csv
运行：python scripts/t4b_q6_branch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_ad_lib as L          # noqa: E402
import t4b_common as C          # noqa: E402
import t4b_glm53_v6 as G6       # noqa: E402

O_TAU, O_G = G6.ROM_TAU.copy(), G6.ROM_G.copy()
P1_THR = 0.30        # 因果判据 P1 的门限（Q5：p10~p90 完全分离，0.20~0.42 之间的任何值都行）
RUN_MAX_INIT = 1e-9


def rom_scaled(alpha):
    """时间缩放形状库：g_α(τ) = g_onset(τ/α)。α<1 ⇒ 更慢（实测 restep 就是这种）。"""
    return O_TAU.copy(), np.interp(O_TAU / alpha, O_TAU, O_G, left=0.0, right=1.0)


def event_metrics(y, r, k, i_next, dt, J, A_abs):
    n = len(y)
    m = C.t1_metrics(y, r, k, dt, n, i_next, J)
    out = dict(T_stable=m["T_stable"], T_stable_drift=m["T_stable_drift"],
               os_pct=m["os_pct"], us_pct=m["us_pct"], md_adc=m["md_adc"],
               d1=m["d1"], d2=m["d2"], win_s=m["win_s"])
    a, b = k + int(8 / dt), min(n - 1, k + int(20 / dt))
    out["plat_dev_pct"] = (round(100 * float(np.median(y[a:b] - r[a:b])) / J, 3)
                           if b - a > 10 else np.nan)
    # 更稳的口径：平台段（事件后 8~20 s）显示相对原始的中位绝对偏差 ÷ 事件绝对幅度 A_abs
    #   为什么不用 J：restep 的 J 相对其平台电平很小，用 J 归一会把指标放大到几百 %，
    #   掩盖掉"到底偏了多少 ADC"这个用户真正看得见的量。
    A_abs = max(abs(A_abs), 1e-9)
    if b - a > 10:
        dv = y[a:b] - r[a:b]
        out["plat_dev_abs_pct"] = round(100 * float(np.median(dv)) / A_abs, 3)
        out["plat_mae_abs_pct"] = round(100 * float(np.mean(np.abs(dv))) / A_abs, 3)
        out["plat_dev_adc"] = round(float(np.median(dv)), 1)
    else:
        out["plat_dev_abs_pct"] = np.nan
        out["plat_mae_abs_pct"] = np.nan
        out["plat_dev_adc"] = np.nan
    return out


def make_branching_class(strategy, loc, r_tau, r_g, thr=P1_THR):
    """按事件切形状库的子类；strategy ∈ {correct, inverse, causal_p1}。

    `thr` 只对 causal_p1 生效（用于门限扫描：判据自己的误判代价）。
    """
    tbl = {round(float(r.t_on), 3): r for _, r in loc.iterrows()}
    times = np.array(sorted(tbl.keys())) if tbl else np.array([])

    class Branch(G6.GLM53v6):
        def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
            super()._new_event(t0, base, v0, y0, kind, hist, prev_state)
            row = None
            if len(times):
                j = int(np.argmin(np.abs(times - t0)))
                if abs(times[j] - t0) < 1.5:
                    row = tbl[float(times[j])]
            gt_onset = (row.kind == "onset") if row is not None else (kind == "onset")
            run_max = self.max_tot if self.max_tot > 1e-9 else 1.0
            loaded_pred = 1 if (base / run_max) > thr else 0     # 因果可得
            if strategy == "correct":
                is_onset = gt_onset
            elif strategy == "inverse":
                is_onset = not gt_onset
            else:
                is_onset = (loaded_pred == 0)
            self.G_TAU, self.G_G = (O_TAU, O_G) if is_onset else (r_tau, r_g)
            self._branch_is_onset = bool(is_onset)
    return Branch


def make_warped_class():
    """连续参数化臂：不分支，用实测早期增量拟合时间缩放 α 后再做形状反演。"""

    class Warped(G6.GLM53v6):
        def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
            super()._new_event(t0, base, v0, y0, kind, hist, prev_state)
            self._alpha = None
            self.G_TAU, self.G_G = O_TAU, O_G

        def _inv_est(self, hist, tau, kappa):
            if self._alpha is None:
                tt = np.array([h[0] for h in hist]) if len(hist) else np.array([])
                yy = np.array([h[1] for h in hist]) if len(hist) else np.array([])
                m = (tt >= 0.10) & (tt <= 0.60)
                self._alpha = 1.0
                if m.sum() >= 4:
                    yo, y1 = float(yy[m][0]), float(yy[m][-1])
                    to, t1 = float(tt[m][0]), float(tt[m][-1])
                    if y1 > 1e-12 and y1 > yo:
                        g = np.interp(np.array([to, t1]), O_TAU, O_G, left=0.0, right=1.0)
                        r_obs = (y1 - yo) / y1
                        r_mod = (g[1] - g[0]) / max(g[1], 1e-9)
                        if r_obs > 1e-6 and r_mod > 1e-6:
                            # 模型增长比实测慢 ⇒ r_mod<r_obs ⇒ α<1（形状要走得更慢）
                            self._alpha = float(np.clip(r_mod / r_obs, 0.3, 2.5))
                self.G_TAU, self.G_G = rom_scaled(self._alpha)
            return super()._inv_est(hist, tau, kappa)

        def _handoff(self, ts, v, ev):
            super()._handoff(ts, v, ev)
            self._alpha = None
    return Warped


def load_t4a_clean():
    """T4-A 接口（只读）：返回 {(rec, round(t_on,1)) : clean_bool} 与其 `t_on`。

    用于把本任务的臂实验限制到 **T4-A 的 clean 主样本**（质量门 G3 的口径统一要求）。
    """
    p = os.path.join(C.TASK, "results", "t4a_morphology.csv")
    if not os.path.isfile(p):
        return None
    a = pd.read_csv(p)
    a = a[a.kind.isin(["onset", "restep"])]
    return a[["rec", "t_on", "clean", "kind", "armed"]].copy()


def filter_clean(loc, t4a):
    """按 T4-A 的 clean 子集筛选本任务事件（±0.6 s 配对）。"""
    if t4a is None:
        return loc, 0
    keep = []
    for _, r in loc.iterrows():
        c = t4a[(t4a.rec == r.rec) & (np.abs(t4a.t_on - r.t_on) <= 0.6)]
        if len(c):
            j = c.iloc[(c.t_on - r.t_on).abs().argmin()]
            if bool(j.clean):
                keep.append(r.name)
    return loc.loc[keep], len(loc) - len(keep)


def main():
    ev, cache = C.build_events(verbose=False)
    rise = ev[ev.kind.isin(["onset", "restep"])].copy()
    cols = [c for c in rise.columns if c.startswith("sh_") and c.endswith("b")]
    taus = np.array([int(c[3:6]) / 100 for c in cols])
    on = rise[rise.kind == "onset"]
    re = rise[rise.kind == "restep"]
    mon = np.array([on[c].median() for c in cols])
    mre = np.array([re[c].median() for c in cols])
    a_hat = mre * taus / np.maximum(mon, 1e-9)
    win = (taus >= 0.10) & (taus <= 1.50)
    a_use = float(np.median(a_hat[win]))
    r_tau, r_g = rom_scaled(a_use)
    pd.DataFrame(dict(tau=taus, f_onset=mon, f_restep=mre, alpha=a_hat)).to_csv(
        os.path.join(C.RES, "t4b_rom_restep.csv"), index=False, encoding="utf-8-sig")
    print("实测定 α：单一时间缩放 α(τ)=τ·f_restep/f_onset 随 τ 变化很大 ⇒ 形状差不只是'快慢'")
    print("  τ =%s" % np.round(taus, 2))
    print("  f_on=%s" % np.round(mon, 3))
    print("  f_re=%s" % np.round(mre, 3))
    print("  α   =%s" % np.round(a_hat, 3))
    print("  取窗口 0.10~1.50 s 的中位 **α=%.3f**（更短的 0.20~0.80 s 窗口给 %.3f）"
          % (a_use, float(np.median(a_hat[(taus >= 0.20) & (taus <= 0.80)]))))
    print("  ROM_restep = %s（= ROM_onset(τ/α)，本任务自建、非既有产物）" % np.round(r_g, 3))

    ARMS = [
        ("M0_now", dict(G_TAU=O_TAU, G_G=O_G, KAPPA_ONSET=1.30, KAPPA_RESTEP=1.30), None),
        ("M1_flat112", dict(G_TAU=O_TAU, G_G=O_G, KAPPA_ONSET=1.12, KAPPA_RESTEP=1.12), None),
        ("M2_correctA", dict(G_TAU=O_TAU, G_G=O_G, KAPPA_ONSET=1.30, KAPPA_RESTEP=1.12), None),
        ("M2_correct", dict(KAPPA_ONSET=1.30, KAPPA_RESTEP=1.12), "correct"),
        ("M2_inverse", dict(KAPPA_ONSET=1.12, KAPPA_RESTEP=1.30), "inverse"),
        ("M3_causal_P1", dict(KAPPA_ONSET=1.30, KAPPA_RESTEP=1.12), "causal_p1"),
        ("M3_causal_thr0.20", dict(KAPPA_ONSET=1.30, KAPPA_RESTEP=1.12), "causal_p1_t020"),
        ("M3_causal_thr0.40", dict(KAPPA_ONSET=1.30, KAPPA_RESTEP=1.12), "causal_p1_t040"),
        ("M4_warped", dict(KAPPA_ONSET=1.30, KAPPA_RESTEP=1.30), "warp"),
    ]

    rows, agree_rows = [], []
    t_all = time.time()
    RECS_RUN = C.RECS
    if "--fast" in sys.argv:                      # 调试用：只跑 2 份 ADC 录制（最短路径）
        RECS_RUN = [r for r in C.RECS if r[0] in ("再切换负载", "中途切换-1d9493")]
    # 口径统一：可只跑 T4-A 的 clean 主样本（本任务事件 ∩ T4-A clean=True）
    CLEAN_ONLY = "--clean-t4a" in sys.argv
    SUF = "_clean" if CLEAN_ONLY else ""
    t4a = load_t4a_clean() if CLEAN_ONLY else None
    if CLEAN_ONLY:
        print("== --clean-t4a：只保留与 T4-A clean=True 配对的本任务事件 ==", flush=True)
    for name, path, dom in RECS_RUN:
        if name not in cache:
            continue
        cc = cache[name]
        tu, Xu, dt, tot, peak = cc["tu"], cc["Xu"], cc["dt"], cc["tot"], cc["peak"]
        n = len(tu)
        loc = ev[ev.rec == name].sort_values("t_on")
        if CLEAN_ONLY:
            loc, dropped = filter_clean(loc, t4a)
            if dropped:
                print("    %s：剔除非 clean 事件 %d 个，保留 %d 个" % (name, dropped, len(loc)),
                      flush=True)
            if not len(loc):
                continue
        idxs = [int(np.searchsorted(tu, t)) for t in loc.t_on]
        for arm, kw, strat in ARMS:
            t_arm = time.time()
            if strat == "warp":
                Cls = make_warped_class()
            elif strat is None:
                Cls = G6.GLM53v6
            elif strat == "causal_p1_t020":
                Cls = make_branching_class("causal_p1", loc, r_tau, r_g, thr=0.20)
            elif strat == "causal_p1_t040":
                Cls = make_branching_class("causal_p1", loc, r_tau, r_g, thr=0.40)
            else:
                Cls = make_branching_class(strat, loc, r_tau, r_g)
            comp = Cls(Xu.shape[1])
            for k, v in kw.items():
                setattr(comp, k, v)
            Y = np.empty_like(Xu)
            for i in range(n):
                Y[i] = comp.process(tu[i], Xu[i])
            yd = L.med_smooth(Y.sum(axis=1), max(1, int(C.SMOOTH_S / dt)))
            yr = L.med_smooth(tot, max(1, int(C.SMOOTH_S / dt)))
            for j, (_, row) in enumerate(loc.iterrows()):
                k = idxs[j]
                nx = idxs[j + 1] if j + 1 < len(idxs) else None
                mm = event_metrics(yd, yr, k, nx, dt, row.J_nc, row.A5_self)
                kap = (kw["KAPPA_ONSET"] if row.kind == "onset" else kw["KAPPA_RESTEP"])
                # 该臂在该事件上"实际用了哪支"
                if strat == "correct":
                    used = row.kind
                elif strat == "inverse":
                    used = "restep" if row.kind == "onset" else "onset"
                elif strat in ("causal_p1", "warp"):
                    used = "onset" if row.p1_onset else "restep"
                elif strat == "causal_p1_t020":
                    used = "onset" if row.p1_ratio <= 0.20 else "restep"
                elif strat == "causal_p1_t040":
                    used = "onset" if row.p1_ratio <= 0.40 else "restep"
                else:                                   # 单库臂：统一用 onset 库
                    used = "onset"
                mm.update(arm=arm, rec=name, t_on=row.t_on, kind=row.kind, arm_gt=row.arm,
                          J_nc=row.J_nc, pre_frac=row.pre_frac, kappa=kap,
                          label_confident=row.label_confident, branch_used=used,
                          p1_onset=row.p1_onset)
                rows.append(mm)
                agree_rows.append(dict(arm=arm, rec=name, t_on=row.t_on, kind=row.kind,
                                       branch_used=used, branch_gt=row.kind,
                                       misbranched=int(used != row.kind)))
            print("  [%6.1fs] %-14s %-20s n=%d" % (time.time() - t_all, name, arm, n),
                  flush=True)
    print("总耗时 %.1f s" % (time.time() - t_all), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RES, "t4b_arm_metrics%s.csv" % SUF), index=False, encoding="utf-8-sig")
    ag = pd.DataFrame(agree_rows)
    ag.to_csv(os.path.join(C.RES, "t4b_decision_agreement%s.csv" % SUF), index=False,
              encoding="utf-8-sig")

    # ── 汇总表（混淆 / 代价）──
    conf = []
    for arm in df.arm.unique():
        for kind in ("onset", "restep"):
            s = df[(df.arm == arm) & (df.kind == kind)]
            if not len(s):
                continue
            conf.append(dict(arm=arm, true_kind=kind, n=len(s),
                             kappa=s.kappa.iloc[0],
                             T_stable_med=s.T_stable.median(),
                             T_stable_p90=s.T_stable.quantile(.9),
                             T_stable_drift_med=s.T_stable_drift.median(),
                             os_med=s.os_pct.median(), os_max=s.os_pct.max(),
                             us_med=s.us_pct.median(), us_min=s.us_pct.min(),
                             md_med=s.md_adc.median(), md_max=s.md_adc.max(),
                             plat_dev_med=s.plat_dev_pct.median(),
                             plat_dev_absmax=s.plat_dev_pct.abs().max(),
                             plat_dev_abs_med=s.plat_dev_abs_pct.median(),
                             plat_dev_abs_maxabs=s.plat_dev_abs_pct.abs().max(),
                             plat_mae_abs_med=s.plat_mae_abs_pct.median(),
                             plat_dev_adc_med=s.plat_dev_adc.median(),
                             plat_dev_adc_absmax=s.plat_dev_adc.abs().max(),
                             d1_med=s.d1.median(), d2_med=s.d2.median()))
    conf = pd.DataFrame(conf)
    conf.to_csv(os.path.join(C.RES, "t4b_branch_confusion%s.csv" % SUF), index=False,
                encoding="utf-8-sig")

    pd.set_option("display.width", 260)
    print("\n== 各臂 × 真值类别的 T1 口径指标（中位/p90）==")
    print(conf.to_string(index=False))
    print("\n== 误判代价（相对真值分支 M2_correct，同类别同事件配对的差值）==")
    base = df[df.arm == "M2_correct"].set_index(["rec", "t_on"])
    cost = []
    for arm in ("M0_now", "M1_flat112", "M2_correctA", "M2_inverse", "M3_causal_P1",
                "M3_causal_thr0.20", "M3_causal_thr0.40", "M4_warped"):
        s = df[df.arm == arm].set_index(["rec", "t_on"])
        common = s.index.intersection(base.index)
        for kind in ("onset", "restep"):
            m = [i for i in common if base.loc[i, "kind"] == kind]
            if not len(m):
                continue
            dT = (s.loc[m, "T_stable"] - base.loc[m, "T_stable"]).dropna()
            dO = (s.loc[m, "os_pct"] - base.loc[m, "os_pct"]).dropna()
            dP = (s.loc[m, "plat_dev_pct"] - base.loc[m, "plat_dev_pct"]).dropna()
            dM = (s.loc[m, "md_adc"] - base.loc[m, "md_adc"]).dropna()
            dA = (s.loc[m, "plat_dev_abs_pct"] - base.loc[m, "plat_dev_abs_pct"]).dropna()
            dMA = (s.loc[m, "plat_mae_abs_pct"] - base.loc[m, "plat_mae_abs_pct"]).dropna()
            dADC = (s.loc[m, "plat_dev_adc"] - base.loc[m, "plat_dev_adc"]).dropna()
            cost.append(dict(arm=arm, true_kind=kind, n=len(m),
                             dT_med=dT.median(), dT_absmax=dT.abs().max(),
                             dOS_med=dO.median(), dOS_absmax=dO.abs().max(),
                             dPlat_med=dP.median(), dPlat_absmax=dP.abs().max(),
                             dPlatAbs_med=dA.median(), dPlatAbs_absmax=dA.abs().max(),
                             dMAE_abs_med=dMA.abs().median(), dMAE_abs_max=dMA.abs().max(),
                             dADC_med=dADC.median(), dADC_absmax=dADC.abs().max(),
                             dMD_med=dM.median(), dMD_absmax=dM.abs().max()))
    cost = pd.DataFrame(cost)
    cost.to_csv(os.path.join(C.RES, "t4b_branch_cost%s.csv" % SUF), index=False, encoding="utf-8-sig")
    print(cost.to_string(index=False))

    q = ag[ag.arm.isin(["M2_correct", "M2_inverse", "M3_causal_P1", "M3_causal_thr0.20",
                        "M3_causal_thr0.40", "M4_warped", "M0_now", "M1_flat112",
                        "M2_correctA"])]
    print("\n== 分支判定混淆（每臂：真值 vs 实际使用支；misbranched = 误判事件数）==")
    tb = []
    for arm in q.arm.unique():
        s = q[q.arm == arm]
        for kind in ("onset", "restep"):
            t = s[s.kind == kind]
            if not len(t):
                continue
            tb.append(dict(arm=arm, true_kind=kind, n=len(t),
                           used_onset=int((t.branch_used == "onset").sum()),
                           used_restep=int((t.branch_used == "restep").sum()),
                           misbranched=int(t.misbranched.sum())))
    tb = pd.DataFrame(tb)
    print(tb.to_string(index=False))
    tb.to_csv(os.path.join(C.RES, "t4b_branch_confmat%s.csv" % SUF), index=False, encoding="utf-8-sig")
    print("\n== 判据在全体事件上的二分表现（P1 门限 0.30）==")
    rr = rise.copy()
    for thr in (0.10, 0.20, 0.30):
        pred_on = rr.pre_frac <= thr
        tp = int(((rr.kind == "onset") & pred_on).sum())
        fn = int(((rr.kind == "onset") & ~pred_on).sum())
        fp = int(((rr.kind == "restep") & pred_on).sum())
        tn = int(((rr.kind == "restep") & ~pred_on).sum())
        print("  门限 %.2f：TP(onset→onset)=%d FN=%d FP=%d TN=%d  准确率=%.3f"
              % (thr, tp, fn, fp, tn, (tp + tn) / (tp + fn + fp + tn)))
    print("\n-> results/t4b_arm_metrics.csv / t4b_branch_confusion.csv / "
          "t4b_branch_cost.csv / t4b_branch_confmat.csv / t4b_decision_agreement.csv / t4b_rom_restep.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
