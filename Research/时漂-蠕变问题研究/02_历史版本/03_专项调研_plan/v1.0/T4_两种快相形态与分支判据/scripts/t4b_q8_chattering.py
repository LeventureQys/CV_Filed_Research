# -*- coding: utf-8 -*-
"""t4b_q8_chattering.py —— T4-Q8：分支门限附近的切换抖动（chattering）与迟滞参数建议。

判据（T4-Q5 的 P1）：`s = 因果电平(t_on) / 因果滚动峰值`，s ≤ 门限 θ ⇒ 零基线支（onset）。
本脚本在**同一批实测事件**上做两件独立的事：

① **真值扰动敏感性**：给总量注入带限噪声（0.3~40 Hz，总量 RMS = A ADC，规范见指标字典 §5），
   A ∈ {0, 50, 100, 200, 400}，并在 t_on 上叠加 ±1 包（±40 ms）时序抖动；
   每事件 × 每 A × 每抖动各 20 次实现 → 统计
     θ 附近事件的"整场误判率"以及"事件后 0~1.0 s 内决策翻转率"（chattering 的定义）。
② **迟滞设计**：双门限（进零基线支 θ_hi，回带载支 θ_lo = θ_hi − h），撒 h ∈ 0~0.15，
   找"翻转率 = 0"所需的最小迟滞。

产物：results/t4b_chattering.csv（每 A × 每 θ × 每 h 一行）、
      results/t4b_chattering_per_event.csv（逐事件原始 s 与噪声下的 s 分布）、
      figures/T4B_03_chattering.png
运行：python scripts/t4b_q8_chattering.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_ad_lib as L          # noqa: E402
import t4b_common as C          # noqa: E402

RNG = np.random.default_rng(20260919)
N_REAL = 20
AMPS = [0, 50, 100, 200, 400]            # 总量扰动 RMS（ADC）
THETAS = [0.20, 0.25, 0.30, 0.35, 0.40]
HYS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]
WIN_S = 0.20                             # 决策观察窗：分支判据必须在 Δ 内定下（Q5 要求）


def bandlimited(n, fs, f_lo, f_hi):
    w = RNG.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    W[(f < f_lo) | (f > f_hi)] = 0.0
    x = np.fft.irfft(W, n) if n % 2 == 0 else np.fft.irfft(W, n)
    s = x.std()
    return x / s if s > 1e-12 else x


def add_noise(tot, A, fs):
    if A <= 0:
        return tot.copy()
    return tot + A * bandlimited(len(tot), fs, 0.3, 40.0)


def decision_series(zc, k, dt, n, theta, h=0.0):
    """事件后 [k, k+WIN] 的逐帧分支决策：1 = 零基线支。

    **因果**实现（也是工程上正确的实现）：门限值在**检测到事件的那一帧冻结** ——
    V = 因果滚动峰值（事件前历史最大电平，v6 内部本来就有 `max_tot`）。
    ⚠️ 若不冻结 V 而让它随斜坡一起涨，判据会自我指涉（s(i)=z(i)/max_{≤i}z），
    在斜坡上必然反复穿越门限 —— 那是判据设计错误，不是真实抖动（见报告 §T4-Q8）。

    返回 (d, n_cross)：d = 逐帧状态；n_cross = 门限穿越次数。
    判据本质上就是"电平是否越过了 θ·V"这一个固定电平门限，
    因此**时间维度的抖动**（同一门限被反复穿越）才是要抑制的东西。
    """
    i1 = min(n - 1, k + int(WIN_S / dt))
    v = float(np.max(zc[:k + 1])) * theta          # 冻结的绝对门限电平
    d = np.zeros(i1 - k + 1, np.int8)
    st = 1 if zc[k - 1] <= v else 0
    cross = 0
    for j, i in enumerate(range(k, i1 + 1)):
        z = zc[i]
        if st == 1:
            if z > v:
                st, cross = 0, cross + 1
        else:
            if z <= v - h * max(float(np.max(zc[:k + 1])), 1e-9):
                st, cross = 1, cross + 1
        d[j] = st
    return d, cross


def main():
    ev, cache = C.build_events(verbose=False)
    rise = ev[ev.kind.isin(["onset", "restep"])].copy()
    print("事件 n=%d（onset %d / restep %d）" % (len(rise), (rise.kind == "onset").sum(),
                                              (rise.kind == "restep").sum()))

    per_ev, agg = [], []
    for name, cc in cache.items():
        loc = rise[rise.rec == name]
        if not len(loc):
            continue
        ts = cc["tu"]
        dt = cc["dt"]
        n = len(ts)
        blk = np.median(np.diff(ts)) if len(ts) > 1 else 0.04
        pkt = max(0.04, float(np.median(np.diff(cc["X"][:, 0])) if False else 0.04))
        for _, row in loc.iterrows():
            k0 = int(np.searchsorted(ts, row.t_on))
            # 只用 t ≤ t_on 的帧：滚动峰值也必须因果（截到 k0）
            rec = dict(rec=name, t_on=row.t_on, kind=row.kind, arm=row.arm,
                       s_nom=row.pre_frac, s_peak_nc=row.pre_frac_nc)
            # 无噪声、无抖动的因果口径（用于把"时序抖动"与"噪声"分开看）
            zc0 = L.med_smooth(cc["tot"], C.DET_SMOOTH, causal=True)
            i0 = max(1, k0 - int(0.50 / dt))
            b0 = float(zc0[i0:k0].mean()) if k0 > i0 else float(zc0[k0])
            rec["s_short_nom"] = round(b0 / max(float(np.max(zc0[:k0 + 1])), 1e-9), 4)
            for A in AMPS:
                ss = []
                for _ in range(N_REAL):
                    tot = add_noise(cc["tot"], A, 1.0 / dt)
                    zc = L.med_smooth(tot, C.DET_SMOOTH, causal=True)
                    # ±1 包时序抖动：观察点整体挪 ±4 帧（±40 ms），覆盖变载实录的包不确定性
                    jit = int(RNG.integers(-4, 5))
                    k = min(max(1, k0 + jit), n - 1)
                    # 因果滚动峰值：只看 t ≤ 观察点
                    rm = float(np.max(zc[:k + 1]))
                    ss.append(float(zc[k] / max(rm, 1e-9)))
                ss = np.array(ss)
                rec["s_%d_med" % A] = round(float(np.median(ss)), 4)
                rec["s_%d_p90" % A] = round(float(np.percentile(ss, 90)), 4)
                rec["s_%d_p10" % A] = round(float(np.percentile(ss, 10)), 4)
                rec["s_%d_std" % A] = round(float(ss.std()), 4)
            per_ev.append(rec)

    pe = pd.DataFrame(per_ev)
    pe.to_csv(os.path.join(C.RES, "t4b_chattering_per_event.csv"), index=False,
              encoding="utf-8-sig")
    print("\n== 判据统计量 s 的噪声敏感性（逐事件，中位）；"
          "s_nom = Q5 口径（0.5 s 因果电平/全历史滚动峰值），"
          "s_short_nom = 冻结 V 的因果口径、无噪声无抖动 ==")
    print(pe.groupby("kind")[["s_nom", "s_short_nom", "s_0_med", "s_100_med", "s_400_med"]]
          .median().to_string())
    print("  s_0_med 与 s_short_nom 之差 = **纯 ±40 ms 时序抖动 + 5 次实现** 带来的漂移")
    err_short = (((pe.kind == "onset") & (pe.s_short_nom > 0.30))
                 | ((pe.kind == "restep") & (pe.s_short_nom <= 0.30))).sum()
    print("  用 s_short_nom、门限 0.30：误判 %d / %d" % (err_short, len(pe)))

    # ── 门限 × 噪声 → 误判率与翻转率 ──
    print("\n== 误判率（事件级整场判错，含 ±40 ms 时序抖动与噪声）==")
    for A in AMPS:
        col = "s_%d_med" % A
        for th in THETAS:
            pred_on = pe[col] <= th
            err = ((pe.kind == "onset") & ~pred_on) | ((pe.kind == "restep") & pred_on)
            # 门限裕度：落在 |s−θ| ≤ 0.5·s_std 的事件算"抖动敏感"
            marg = (pe[col] - th).abs() <= 0.5 * pe["s_%d_std" % A]
            agg.append(dict(A=A, theta=th, n=len(pe), n_misbranch=int(err.sum()),
                            misbranch_rate=round(float(err.mean()), 4),
                            n_near_thr=int(marg.sum()),
                            near_thr_rate=round(float(marg.mean()), 4)))
    ag = pd.DataFrame(agg)
    ag.to_csv(os.path.join(C.RES, "t4b_chattering.csv"), index=False, encoding="utf-8-sig")
    print(ag[ag.A.isin([0, 100, 400])].to_string(index=False))

    # ── 迟滞：门限穿越次数（噪声引起的额外穿越 = 抖动）──
    print("\n== 迟滞 h 对『门限穿越次数』的抑制（每事件平均；穿越 1 次 = 正常的支切换）==")
    rows = []
    for A in (0, 50, 100, 200, 400):
        for th in (0.20, 0.30, 0.40):
            for h in HYS:
                cross_sum, wrong, tot_ev = 0, 0, 0
                for name, cc in cache.items():
                    loc = rise[rise.rec == name]
                    if not len(loc):
                        continue
                    ts, dt, n = cc["tu"], cc["dt"], len(cc["tu"])
                    for _, row in loc.iterrows():
                        k0 = int(np.searchsorted(ts, row.t_on))
                        for _ in range(5):
                            tot = add_noise(cc["tot"], A, 1.0 / dt)
                            zc = L.med_smooth(tot, C.DET_SMOOTH, causal=True)
                            jit = int(RNG.integers(-4, 5))
                            k = min(max(2, k0 + jit), n - 1)
                            d, cr = decision_series(zc, k, dt, n, th, h)
                            cross_sum += cr
                            final = int(d[-1])
                            wrong += int(final != (1 if row.kind == "onset" else 0))
                            tot_ev += 1
                rows.append(dict(A=A, theta=th, hyst=h, n_seq=tot_ev,
                                 cross_per_ev=round(cross_sum / max(tot_ev, 1), 4),
                                 final_err_rate=round(wrong / max(tot_ev, 1), 4)))
    ch = pd.DataFrame(rows)
    ch.to_csv(os.path.join(C.RES, "t4b_chattering_hyst.csv"), index=False, encoding="utf-8-sig")
    piv = ch[ch.A == 400].pivot_table(index="theta", columns="hyst", values="cross_per_ev")
    print("（A = 400 ADC，行 = 门限 θ，列 = 迟滞 h，值 = 每事件穿越次数）")
    print(piv.to_string())
    piv2 = ch[ch.A == 400].pivot_table(index="theta", columns="hyst", values="final_err_rate")
    print("（同表，值 = 观察窗末端的整场误判率）")
    print(piv2.to_string())
    print("\n== 噪声引起的额外穿越（相对 A=0 的增量）==")
    ex = []
    for th in (0.20, 0.30, 0.40):
        for h in HYS:
            c0 = ch[(ch.A == 0) & (ch.theta == th) & (ch.hyst == h)].cross_per_ev.iloc[0]
            c4 = ch[(ch.A == 400) & (ch.theta == th) & (ch.hyst == h)].cross_per_ev.iloc[0]
            ex.append(dict(theta=th, hyst=h, cross_A0=c0, cross_A400=c4,
                           excess=round(c4 - c0, 4)))
    ex = pd.DataFrame(ex)
    print(ex.pivot_table(index="theta", columns="hyst", values="excess").to_string())
    print("\n-> results/t4b_chattering.csv / t4b_chattering_per_event.csv / t4b_chattering_hyst.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
