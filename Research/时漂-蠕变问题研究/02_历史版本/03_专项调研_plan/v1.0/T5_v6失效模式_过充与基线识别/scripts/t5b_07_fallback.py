# -*- coding: utf-8 -*-
"""T5-B / 07：T5B-Q6 「找不到基线」时的保底行为（置信度不足时该做什么）

置信度信号（**因果、C++ 侧可直接实现**，只用算法已有的两个量）：
    low_conf  ⇔  5·σ_d  >  LOWCONF_FRAC × (0.05·|lv_ref|)
即"抖动把 σ 门限抬到超过 5% 电平门槛"的那一刻起，算法自己就知道"这时的电平变化判据不可信"。

四种保底策略（默认关 = 与原型逐帧零差，见 `t5b_patch_ab_zero.csv`）
    P0 none       现状（无保底）
    P1 hold       低置信期**冻结基线锚定**：`_handoff` / `_reanchor` 不回写 A（快相滑行照常）
    P2 raw        低置信期**免责直通**：输出 = 原始读数，置信恢复后回到补偿输出
    P3 v5         低置信期整段切到 v5.1 的输出（v5.1 自带免责期）
    P4 freeze     低置信期**冻结输出**（保持最后可信输出）并置 `recal_required` 标志
另外对 P1 扫 LOWCONF_FRAC ∈ {0.5, 0.75, 1.0, 1.5}（"多保守才合适"）。

产物
    results/t5b_fallback_policies.csv  每策略的收益/副作用量化
    results/t5b_fallback_trials.csv    逐 trial 明细
    results/_t5b_07.log
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t5b_core import (TraceV6, load_window, run_full, perturb_window,   # noqa: E402
                      gt_in_window, load_gt, WIN, DT)
from t5b_glm53_v51 import GLM53v51                                     # noqa: E402


class FBV6(TraceV6):
    POLICY = "none"
    LOWCONF_FRAC = 1.0
    RECOVER_S = 1.0

    def __init__(self, n):
        self._lowconf = False
        self.n_hold = 0
        self.n_free = 0
        self.n_freeze = 0
        self.recal_required = False
        self.frozen_out = None
        self._v5 = GLM53v51(n)
        self._sum_dev_fallback = 0.0
        self._n_fallback_frames = 0
        super().__init__(n)

    @property
    def sig_d(self):
        return self._sig_d

    @sig_d.setter
    def sig_d(self, v):
        if self.CAPF is not None and self._cur_lv_ref is not None:
            lvl = max(self.DET_REL * abs(self._cur_lv_ref), self.DET_ABS_FRAC * self.max_tot)
            v = min(v, self.CAPF * lvl / self.DET_K)
        self._sig_d = float(v)
        lvl2 = self.DET_REL * abs(self._cur_lv_ref if self._cur_lv_ref is not None else 0.0)
        self._lowconf = bool(lvl2 > 0 and (self.DET_K * self._sig_d) > self.LOWCONF_FRAC * lvl2)

    def _handoff(self, ts, v, ev=None):
        if self.POLICY in ("hold", "freeze") and self._lowconf:
            ev = ev if ev is not None else self.ev
            self.tr_handoff.append((float(ts), "SKIP:" + str(ev["kind"]), 0.0, 0.0,
                                    float(np.sum(self.A)), float(self.g)))
            self.n_hold += 1
            self.recal_required = True
            self.ev = None
            self.state = "slow"
            self.ev_end = ts
            self.trim_target_sum = None
            return
        return super()._handoff(ts, v, ev)

    def _reanchor(self, ts, v):
        if self.POLICY in ("hold", "freeze") and self._lowconf:
            self.hold_comp = None
            self.n_hold += 1
            self.recal_required = True
            return
        return super()._reanchor(ts, v)

    def process(self, ts, v):
        out = super().process(ts, v)
        v = np.asarray(v, float)
        if self.POLICY == "none":
            return out
        if self.POLICY == "v5":
            o5 = self._v5.process(float(ts), v)
            if self._lowconf:
                self.n_free += 1
                self._n_fallback_frames += 1
                self._sum_dev_fallback += float(np.abs(o5 - out).sum())
                return o5
            return out
        if self.POLICY == "raw":
            if self._lowconf:
                self.n_free += 1
                self._n_fallback_frames += 1
                self._sum_dev_fallback += float(np.abs(v - out).sum())
                return v
            return out
        if self.POLICY == "freeze":
            if self._lowconf:
                self.n_freeze += 1
                self.recal_required = True
                self._n_fallback_frames += 1
                if self.frozen_out is not None:
                    self._sum_dev_fallback += float(np.abs(self.frozen_out - out).sum())
                    return self.frozen_out.copy()
                self.frozen_out = out.copy()
                return out
            self.frozen_out = out.copy()
            return out
        return out


POLICIES = [("none", 1.0), ("hold", 0.5), ("hold", 0.75), ("hold", 1.0), ("hold", 1.5),
            ("raw", 1.0), ("v5", 1.0), ("freeze", 1.0)]


def trial_metrics(tu, Xp, ref, base, gt_w, W):
    """跑一次策略/原型的对照：返回关键指标。"""
    r = run_full(tu, Xp, FBV6, POLICY=cur_policy, LOWCONF_FRAC=cur_lcf)
    tr = r["tr"]
    dev = r["Ysum"] - ref["Ysum"]
    lvl = float(np.percentile(np.abs(dev), 50)) if len(dev) else 0.0
    Zs = pd.Series(r["Z"]).rolling(30, center=True, min_periods=1).median().to_numpy()
    Zsc = pd.Series(ref["Z"]).rolling(30, center=True, min_periods=1).median().to_numpy()
    n = min(len(Zs), len(Zsc))
    excess = (r["Ysum"][:n] - Zs[:n]) - (ref["Ysum"][:n] - Zsc[:n])
    lv_max = float(np.percentile(np.abs(Zs), 95))
    # 相对"无策略"基线的输出差异（= 保底策略自己引入的副作用）
    dev_vs_base = r["Ysum"] - base["Ysum"]
    c = r["c"]
    return dict(n_epoch=len(r["epoch"]), n_revoke=len(r["revoke"]),
                n_handoff=len(r["handoff"]), n_hold=c.n_hold, n_free=c.n_free,
                n_freeze=c.n_freeze, recal=int(c.recal_required),
                lowconf_frac=float((tr["sig_d"] * 5.0 >
                                    cur_lcf * 0.05 * np.abs(tr["lv_ref"])).mean()),
                fallback_frames=c._n_fallback_frames,
                fb_dev_sum=float(c._sum_dev_fallback),
                excess_rel=float(np.abs(excess).max() / max(lv_max, 1.0)),
                dev_base_rel=float(np.abs(dev_vs_base).max() / max(lv_max, 1.0)),
                dA_end=float(tr["sumA"][-1] - ref["tr"]["sumA"][-1]))


cur_policy = "none"
cur_lcf = 1.0


def main():
    global cur_policy, cur_lcf
    t00 = time.time()
    gt = load_gt()
    tu, Xu, W = load_window(WIN["W1_1w"])
    gt_w = gt_in_window(gt, W["rec"], W["t0"], W["t1"])
    ref = run_full(tu, Xu, TraceV6)
    lvl = float(np.median(Xu.sum(axis=1)))
    print(f"[{WIN['W1_1w']['note']}] 帧={len(tu)} 电平={lvl:.0f} "
          f"干净 epoch={len(ref['epoch'])} 真值={len(gt_w)}", flush=True)
    rows = []
    base = run_full(tu, Xu, TraceV6)              # 无策略基线（只跑一次）
    for pol, lcf in POLICIES:
        cur_policy, cur_lcf = pol, lcf
        t0 = time.time()
        cases = []
        # 干净
        cases.append(("clean", Xu))
        # 拍击（2 组合 × 7 点）
        for pct, dur in ((50, 100), (100, 500)):
            for site in (6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0):
                Xp, _ = perturb_window(tu, Xu, "tap", pct / 100.0 * lvl, 7, t_inj=site,
                                       rise_ms=max(10, dur // 4), hold_ms=max(10, dur // 2),
                                       fall_ms=max(10, dur // 4))
                cases.append((f"tap{pct}_{dur}ms@{site:.0f}", Xp))
        # 抖动
        for amp in (1000.0, 2000.0, 4000.0):
            for sd in (0, 1):
                Xp, _ = perturb_window(tu, Xu, "white", amp, sd * 1000 + 7)
                cases.append((f"white{amp:.0f}#{sd}", Xp))
        # 真实阶跃（延迟代价）
        for sd in (0, 1):
            Xp, _ = perturb_window(tu, Xu, "step", 0.2 * lvl, sd * 1000 + 7, t_inj=36.0)
            cases.append((f"step20%#{sd}", Xp))
        for cname, Xp in cases:
            m = trial_metrics(tu, Xp, ref, base, gt_w, W)
            rows.append(dict(policy=pol, lowconf_frac_thr=lcf, case=cname, level=lvl, **m))
        sub = pd.DataFrame([r for r in rows if r["policy"] == pol and r["lowconf_frac_thr"] == lcf])
        print(f"  [{pol} lcf={lcf}] 干净 side-effect={sub[sub.case=='clean']['dev_base_rel'].iloc[0]:.4f} "
              f"拍击 dA_end_med={sub[sub.case.str.startswith('tap')]['dA_end'].abs().median():.0f} "
              f"抖动 excess_p90={sub[sub.case.str.startswith('white')]['excess_rel'].quantile(.9):.3f} "
              f"低置信帧占比={sub[sub.case.str.startswith('white')]['lowconf_frac'].median():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t5b_fallback_trials.csv"), index=False, encoding="utf-8-sig")
    # 汇总
    agg = []
    for (pol, lcf), g in df.groupby(["policy", "lowconf_frac_thr"]):
        cl = g[g["case"] == "clean"]
        tp = g[g["case"].str.startswith("tap")]
        nz = g[g["case"].str.startswith("white")]
        st = g[g["case"].str.startswith("step")]
        agg.append(dict(policy=pol, lowconf_thr=lcf,
                        clean_sideeffect_rel=round(float(cl["dev_base_rel"].median()), 4),
                        tap_dA_end_med=round(float(tp["dA_end"].abs().median()), 1),
                        tap_dA_end_rel_med=round(float(tp["dA_end"].abs().median()) / lvl, 4),
                        tap_excess_p90=round(float(tp["excess_rel"].quantile(0.9)), 4),
                        tap_survived_handoff_med=float(tp["n_handoff"].median()),
                        tap_hold_mean=round(float(tp["n_hold"].mean()), 2),
                        noise_lowconf_frac_med=round(float(nz["lowconf_frac"].median()), 3),
                        noise_excess_p90=round(float(nz["excess_rel"].quantile(0.9)), 4),
                        noise_excess_med=round(float(nz["excess_rel"].median()), 4),
                        noise_dA_end_med=round(float(nz["dA_end"].abs().median()), 1),
                        step_excess_rel=round(float(st["excess_rel"].median()), 4),
                        fallback_frames_med=float(nz["fallback_frames"].median()),
                        fb_extra_dev_rel_med=(round(float(nz["fb_dev_sum"].median()) /
                                                    (lvl * max(1, int(nz["fallback_frames"].median()))),
                                                    6)
                                              if float(nz["fallback_frames"].median()) > 0 else 0.0),
                        recal_flagged_frac=round(float(nz["recal"].mean()), 3),
                        n=len(g)))
    sm = pd.DataFrame(agg)
    sm.to_csv(os.path.join(RES, "t5b_fallback_policies.csv"), index=False, encoding="utf-8-sig")
    print(f"\n总耗时 {time.time()-t00:.0f}s")
    print(sm.to_string())


if __name__ == "__main__":
    main()
