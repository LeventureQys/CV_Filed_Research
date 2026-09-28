# -*- coding: utf-8 -*-
"""t3b_06_sens.py —— T3-B 的敏感性检验（两项，都是 R2/口径要求的）：

1. **平滑口径敏感性**：指标字典 §1 规定 `Z̄ = median(Z, 0.5 s)` 用于指标评估，而第一轮
   `r4_filter_tradeoff.stable_time` 用的是**未平滑**序列。实测两者能差 2 倍以上
   （脚本 t3b_dbg_stable.py：同一事件 未平滑 2.670 s vs 平滑 1.100 s），
   故本表对 v6 / v51_f3 两条关键臂**同时给两列**，供 T8 记账。
2. **±1 包（R2）事件起点敏感性**：变载实录的包时间戳成批突发（本表同时报 `dup_frac` 与
   `pkt_p90`），事件起点有 ±40 ms 不确定性 ⇒ 对 k0 做 ±4 帧（±40 ms）平移，看 T_stable 变化。

产出：results/t3b_smoothing_sensitivity.csv、results/t3b_jitter_pm1pkt.csv、
      results/_t3b_06_sens.log

运行：python scripts/t3b_06_sens.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_ad_lib as L     # noqa: E402
import t3b_arms as A       # noqa: E402
import t3b_common as C     # noqa: E402
import t3b_settle as ST    # noqa: E402

ARMS = ("v6", "v51_f3")


def main():
    ST.start_log(C.RES, "06_sens")
    t0 = time.time()
    ev, cache = A.load_all(verbose=False)
    rows, jit = [], []
    for arm_name in ARMS:
        arm = A.ARM_BY_NAME[arm_name]
        for name, path, dom in C.RECS:
            cc = cache[name]
            tu, Xu, dt, tot, peak = cc["tu"], cc["Xu"], cc["dt"], cc["tot"], cc["peak"]
            comp = A._make(arm, Xu.shape[1])
            Y = np.empty_like(Xu)
            for i in range(len(tu)):
                Y[i] = comp.process(tu[i], Xu[i])
            yt = Y.sum(axis=1)
            k5 = max(1, int(round(0.5 / dt)))
            yt_s = L.med_smooth(yt, k5)
            loc = ev[ev.rec == name].sort_values("t_on")
            tlist = [(float(r.t_on), float(r.J_nc)) for _, r in loc.iterrows()
                     if np.isfinite(r.J_nc) and abs(r.J_nc) >= 0.02 * peak
                     and r.kind in ("onset", "restep")]
            for _, r in loc.iterrows():
                k0 = int(np.searchsorted(tu, float(r.t_on)))
                if k0 >= len(tu) - 5:
                    continue
                times = [(t, j) for t, j in tlist if abs(t - float(r.t_on)) > 1e-9]
                cut = ST.next_event_cut(times, float(r.t_on), peak)
                cut_k = None if cut is None else int(round(cut / dt))
                out = dict(arm=arm_name, rec=name, dom=dom, kind=r.kind,
                           t_on=round(float(r.t_on), 3))
                for tag, Ys in (("sm", yt_s), ("ns", yt)):
                    J, _, _ = ST.amp_5s(Ys, k0)
                    t, cens, _ = ST.t_stable_ev(tu, Ys, k0, J, cut_k)
                    out["T_" + tag] = t
                    out["cens_" + tag] = cens
                rows.append(out)
                # ±1 包平移（只对 v6，且只对 40 ms 包周期的实录域更有意义，但两类都做）
                if arm_name == "v6":
                    for d in (-4, 0, 4):
                        kk = max(0, k0 + d)
                        J, _, _ = ST.amp_5s(yt_s, kk)
                        t, cens, _ = ST.t_stable_ev(tu, yt_s, kk, J, cut_k)
                        jit.append(dict(rec=name, dom=dom, kind=r.kind, t_on=round(float(r.t_on), 3),
                                        shift_frames=d, shift_ms=d * 10.0, T_stable_ev_tot5=t,
                                        censored=cens))
    sm = pd.DataFrame(rows)
    sm["dT_sm_minus_ns"] = sm.T_sm - sm.T_ns
    sm.to_csv(os.path.join(C.RES, "t3b_smoothing_sensitivity.csv"), index=False,
              encoding="utf-8-sig")
    jd = pd.DataFrame(jit)
    jd.to_csv(os.path.join(C.RES, "t3b_jitter_pm1pkt.csv"), index=False, encoding="utf-8-sig")

    print("== 1. 平滑口径敏感性（T_stable_ev_tot5；sm = 0.5 s 中值平滑，ns = 未平滑）==")
    for arm_name in ARMS:
        s = sm[sm.arm == arm_name]
        d = s.dropna(subset=["T_sm", "T_ns"])
        print("  %-8s n=%-3d  sm 中位 %6.3f  ns 中位 %6.3f  配对差中位 %+6.3f s  (ns/sm 比值中位 %.2f)"
              % (arm_name, len(d), s.T_sm.median(), s.T_ns.median(),
                 (d.T_sm - d.T_ns).median(),
                 float(np.median((d.T_ns / d.T_sm.replace(0, np.nan)).dropna())) if len(d) else np.nan))
    print("\n== 2. 帧时间戳统计（变载实录为何有 ±1 包不确定性）==")
    print("%-22s %8s %8s %8s %8s" % ("rec", "dt_med", "dt_p90", "dup_frac", "eff_hz"))
    for name, path, dom in C.RECS:
        t, X = L.load_rec(path)
        d = np.diff(t)
        dup = float(np.mean(d <= 0))
        eff = len(t) / float(t[-1] - t[0])
        print("%-22s %8.4f %8.4f %8.3f %8.1f" % (name, float(np.median(d)),
                                                 float(np.percentile(d, 90)), dup, eff))
    print("\n== 3. ±1 包（±40 ms）事件起点敏感性（v6，D1-ev 总通道）==")
    for dom in ("显示域", "ADC域"):
        s = jd[jd.dom == dom]
        piv = s.pivot_table(index=["rec", "t_on"], columns="shift_frames",
                            values="T_stable_ev_tot5")
        if not len(piv):
            continue
        dm = (piv[4] - piv[-4]).dropna()
        m0 = piv[0].median()
        print("  %-6s n=%-3d  中位 −40ms %6.3f / 0 %6.3f / +40ms %6.3f  |Δ(±40ms)| 中位 %6.3f s 最大 %6.3f s"
              % (dom, len(piv), piv[-4].median(), m0, piv[4].median(),
                 float(dm.abs().median()) if len(dm) else np.nan,
                 float(dm.abs().max()) if len(dm) else np.nan))
    print("\n总耗时 %.1f s" % (time.time() - t0))
    print("-> results/t3b_smoothing_sensitivity.csv / t3b_jitter_pm1pkt.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
