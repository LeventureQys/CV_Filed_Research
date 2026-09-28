# -*- coding: utf-8 -*-
"""r2：事件级三阶段量化 + 快相形状稳定性 + v6 逆模型 Â 的过充量化 + onset/restep 对比。

关键：**逐字复刻 v6 的逆模型**（`glm53_v6.py::_inv_est`）：
    Â = Σ_{τ∈[0.20, min(τ_now,0.80)]} y(τ)·g(τ) / Σ g(τ)²,  再 max(A, inc)、min(A, κ·inc)
    g = ROM（13 份 onset 合并标定的中位形状，g(0.2)=0.790, g(0.8)=0.886, g(5)=1.0）
于是可以拿真实事件回答：ROM 与现场形状的失配会带来多少过充/欠充。

产出：results/events.csv、results/shape_stats.csv、results/rom_loo.csv、results/overcharge.csv
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FLASH = os.path.dirname(os.path.dirname(OUT))
TEMP = os.path.dirname(FLASH)
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, os.path.join(FLASH, "progress", "04-v5", "scripts"))
import ad_lib as L  # noqa: E402

# ── v6 的形状 ROM 与逆模型（逐字取自 glm53_v6.py）─────────────────
ROM_TAU = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80,
                    1.00, 1.50, 2.00, 3.00, 4.00, 5.00])
ROM_G = np.array([0.0, 0.680, 0.740, 0.790, 0.824, 0.854, 0.873, 0.886,
                  0.904, 0.922, 0.941, 0.970, 0.989, 1.000])
TAU_REF, AWIN = 0.20, 0.60
KAPPA_ONSET, KAPPA_RESTEP = 1.30, 1.12
GRID = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00])


def g_shape(tau, tau_grid=ROM_TAU, g_grid=ROM_G):
    return np.interp(np.asarray(tau, float), tau_grid, g_grid, left=0.0, right=1.0)


def inv_est(tau_q, y_q, tau_now, kappa, tau_grid=ROM_TAU, g_grid=ROM_G):
    """v6 的 Â：电平域最小二乘 + 下限/上限。tau_q/y_q 为事件原点起的采样。"""
    m = (tau_q >= TAU_REF) & (tau_q <= min(tau_now, TAU_REF + AWIN))
    if m.sum() < 5:
        return np.nan
    g = g_shape(tau_q[m], tau_grid, g_grid)
    den = float((g * g).sum())
    if den < 1e-12:
        return np.nan
    inc = float(y_q[-1])
    if inc <= 0:
        return np.nan
    A = float((y_q[m] * g).sum()) / den
    return float(min(max(A, inc), kappa * inc))


B = os.path.join(TEMP, "变化负载")
ALL = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"), "显示域")
       for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)] + [
    ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                       "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"), "ADC域"),
    ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv"), "ADC域"),
    ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv"), "ADC域"),
    ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv"), "ADC域"),
]
MAIN_CH = {"右拇指指尖": 17, "左拇指指尖": 18, "四指指尖": 11}


def find_edges(y, tu, dt, amp):
    """找加载/卸载沿：0.2 s 中心差分 + 相对幅度门限 + 最小间隔 1.0 s。"""
    w = max(1, int(0.10 / dt))
    ys = L.med_smooth(y, max(1, int(0.10 / dt)))
    d = np.zeros_like(ys)
    d[w:-w] = ys[2 * w:] - ys[:-2 * w]
    thr = max(0.08 * amp, 1e-9)
    cand = np.where(np.abs(d) > thr)[0]
    edges, last = [], -10 ** 9
    for i in cand:
        if i - last < int(1.0 / dt):
            continue
        j0, j1 = max(0, i - int(0.30 / dt)), min(len(d), i + int(0.30 / dt))
        j = j0 + int(np.argmax(np.abs(d[j0:j1])))
        edges.append(j)
        last = j
    # 用单帧最大跳变精修真沿
    out = []
    for j in edges:
        a, b = max(1, j - int(0.20 / dt)), min(len(y) - 1, j + int(0.20 / dt))
        k = a + int(np.argmax(np.abs(np.diff(y[a:b]))))
        out.append(k)
    return sorted(set(out))


def event_metrics(tu, y, k, dt, amp):
    def lvl(t):
        i = k + int(t / dt)
        if i < 0 or i >= len(y):
            return np.nan
        h = max(1, int(0.03 / dt))
        return float(np.median(y[max(0, i - h):min(len(y), i + h + 1)]))
    pre = float(np.median(y[max(0, k - int(1.0 / dt)):max(1, k - int(0.15 / dt))]))
    lv = {t: lvl(t) for t in GRID}
    y_q = np.array([lvl(t) - pre for t in (0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80)])
    tau_q = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80])
    A5 = lv[5.00] - pre
    A1 = lv[1.00] - pre
    A02 = lv[0.20] - pre
    kind = "onset" if pre < 0.20 * max(abs(A5), 1e-12) else ("restep" if A5 > 0 else "decrement")
    kap = KAPPA_ONSET if kind == "onset" else KAPPA_RESTEP
    # v6 在 τ=0.8 s 时算出的 Â（此时窗口已满）
    Ah = inv_est(tau_q, y_q, 0.80, kap) if kind != "decrement" else np.nan
    r = dict(t_edge=round(float(tu[k]), 3), pre=round(pre, 3), A02=round(A02, 3), A1=round(A1, 3),
             A5=round(A5, 3), kind=kind,
             step_frac=round(100 * A02 / A5, 2) if A5 else np.nan,
             frac1=round(100 * A1 / A5, 2) if A5 else np.nan,
             Ahat=round(float(Ah), 3) if Ah == Ah else np.nan,
             over_vs_A5=round(100 * (Ah / A5 - 1), 2) if (Ah == Ah and A5) else np.nan,
             over_vs_A1=round(100 * (Ah / A1 - 1), 2) if (Ah == Ah and A1) else np.nan,
             kappa=kap, amp=round(float(amp), 2))
    for t in GRID:
        r["sh_%02d" % int(t * 100)] = round((lv[t] - pre) / A5, 4) if A5 else np.nan
    # 阶跃段与快相段的 10%→90% 时间
    for tag, lo, hi in (("step", pre, pre + A02), ("fast", pre + A02, pre + A5)):
        if hi - lo <= 0:
            r["%s_t90" % tag] = np.nan
            continue
        seg = y[k:min(len(y), k + int(5.0 / dt))]
        kk = np.where(seg >= lo + 0.9 * (hi - lo))[0]
        r["%s_t90" % tag] = round(float(kk[0] * dt), 3) if len(kk) else np.nan
    return r


def main():
    rows, recs = [], {}
    for name, path, dom in ALL:
        if not os.path.isfile(path):
            print("!! 缺:", path)
            continue
        d = L.prep(path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        loc = name.split("/")[0]
        ch = MAIN_CH.get(loc)
        if ch is None or ch >= Xu.shape[1]:
            ch = int(np.argmax(Xu[int(len(tu) * 0.2):int(len(tu) * 0.8)].std(axis=0)))
        y = Xu[:, ch]
        amp = float(np.percentile(y, 99.5) - np.percentile(y, 0.5))
        recs[name] = dict(y=y, tu=tu, dt=dt, amp=amp, dom=dom, ch=ch)
        for k in find_edges(y, tu, dt, amp):
            r = event_metrics(tu, y, k, dt, amp)
            r.update(rec=name, dom=dom, ch=ch)
            rows.append(r)
    ev = pd.DataFrame(rows)
    ev = ev[["rec", "dom", "ch", "kind", "t_edge", "pre", "A02", "A1", "A5", "amp",
             "step_frac", "frac1", "Ahat", "over_vs_A5", "over_vs_A1", "kappa",
             "step_t90", "fast_t90"] + [c for c in ev.columns if c.startswith("sh_")]]
    ev.to_csv(os.path.join(RES, "events.csv"), index=False, encoding="utf-8-sig")

    sh_cols = [c for c in ev.columns if c.startswith("sh_")]
    # ① 形状稳定性（仅 onset，按 v6 的 ROM 用法：归一到 5 s 增量）
    onset = ev[(ev.kind == "onset") & ev.A5.gt(0)].copy()
    st = []
    for c in sh_cols:
        v = onset[c].dropna()
        if len(v) < 3:
            continue
        st.append(dict(tau=int(c[3:]) / 100, n=len(v), med=v.median(), p10=v.quantile(.1),
                       p90=v.quantile(.9), lo=v.min(), hi=v.max(), spread=v.max() - v.min(),
                       rom=float(g_shape(int(c[3:]) / 100))))
    st = pd.DataFrame(st)
    st.to_csv(os.path.join(RES, "shape_stats.csv"), index=False, encoding="utf-8-sig")

    # ② 留一法：用其余 onset 的中位形状当 ROM，看对留出事件的高估
    loo = []
    ons = onset[onset.A02.notna()]
    for _, r in ons.iterrows():
        others = ons.drop(index=r.name)
        tg = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80])
        gg = np.array([others[c].median() for c in sh_cols[:7]])
        if np.isnan(gg).any():
            continue
        y_q = np.array([r[c] for c in sh_cols[:7]]) * r.A5     # 还原到电平域增量
        m = tg >= TAU_REF
        A = float((y_q[m] * gg[m]).sum()) / float((gg[m] ** 2).sum())
        A = min(max(A, y_q[-1]), r.kappa * y_q[-1])
        loo.append(dict(rec=r.rec, t_edge=r.t_edge, A5=r.A5, Ahat_loo=A,
                        bias_pct=100 * (A / r.A5 - 1)))
    loo = pd.DataFrame(loo)
    loo.to_csv(os.path.join(RES, "rom_loo.csv"), index=False, encoding="utf-8-sig")

    # ③ 过充汇总
    oc = []
    for kind in ("onset", "restep", "decrement"):
        s = ev[ev.kind == kind]
        if len(s) == 0:
            continue
        oc.append(dict(kind=kind, n=len(s),
                       over_med=s.over_vs_A5.median(), over_p90=s.over_vs_A5.quantile(.9),
                       over_min=s.over_vs_A5.min(), over_max=s.over_vs_A5.max(),
                       step_frac_med=s.step_frac.median(), frac1_med=s.frac1.median(),
                       step_t90_med=s.step_t90.median(), fast_t90_med=s.fast_t90.median()))
    oc = pd.DataFrame(oc)
    oc.to_csv(os.path.join(RES, "overcharge.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("== 事件统计 ==")
    print(ev.groupby(["kind"]).size().to_string())
    print("\n== 形状稳定性（onset，归一到 5 s 增量；rom=当前 ROM 值）==")
    print(st.to_string(index=False))
    print("\n== 留一法（用其余样本中位形状做 ROM，对留出事件的高估）==")
    print("   n=%d  中位 %+.2f%%  p10 %+.2f%%  p90 %+.2f%%  max %+.2f%%"
          % (len(loo), loo.bias_pct.median(), loo.bias_pct.quantile(.1),
             loo.bias_pct.quantile(.9), loo.bias_pct.max()))
    print("\n== 各类事件的过充/形态 ==")
    print(oc.to_string(index=False))
    print("\n== 逐事件（前 24）==")
    print(ev.head(24).to_string(index=False))
    print("\n-> results/events.csv(%d)、shape_stats.csv、rom_loo.csv、overcharge.csv" % len(ev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
