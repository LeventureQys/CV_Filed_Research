# -*- coding: utf-8 -*-
"""v5 实测验证：4 份变化负载/切换负载录制上的事件增益、慢相时漂、自检。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

B = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
RECS = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "最终测试目标", "device_001_seg000.csv"))]

ALGOS = [("raw", None, {}),
         ("v3", GLM53v3, {}),
         ("v4", L.GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
         ("v5", GLM53v5, {})]
ORDER = [k for k, _, _ in ALGOS]

summary, allev = [], []
for tag, path in RECS:
    if not os.path.exists(path):
        print(f"[skip] 缺文件 {path}")
        continue
    d = L.prep(path)
    d["periods"] = L.find_periods(d["tot"], d["dtm"])
    d["events"] = [e for e, _ in L.detect_events(d["tot"], d["dtm"])]
    Ys, ck, ep = {}, {}, {}
    print("=" * 100)
    print(f"[{tag}] {d['span']:.1f}s / {len(d['t'])} 帧 / 负载段 {len(d['periods'])} / 事件 {len(d['events'])}")
    for k, cls, kw in ALGOS:
        if cls is None:
            Ys[k] = d["Xu"].copy()
            continue
        Ys[k], c = L.run_algo(d["tu"], d["Xu"], L.make_traced(cls), **kw)
        ck[k], ep[k] = c, c.epoch_t
        print(f"   [{k}] 自检 A_max={c.A.max():8.3f} loaded={int(c.loaded.sum()):3d} "
              f"g_end={c.g:+.4f}  epoch={len(c.epoch_t)}  γ∈[{c.gamma.min():.2f},{c.gamma.max():.2f}]")
    d["Ys"] = Ys
    ev = L.event_table(d, d["events"], Ys, ep["v3"], gain_s=6.0, algos=["v3", "v4", "v5"])
    ev["valid_gain"] = ev["raw_gain"].abs() > 0.5 * ev["jump"].abs()
    st = L.slow_windows(d, algos=["raw", "v3", "v4", "v5"])
    # 只保留"真正稳定"的窗：窗内短滞后电平变化 < 3%（排除含真实变载的窗，否则会奖励"把真实加载扣掉"）
    tu_, tot_ = d["tu"], d["tot"]
    dtm_ = d["dtm"]
    tot_s_ = L.med_smooth(tot_, 0.5 / dtm_)
    for w0, w1 in st[["window_s", "end_s"]].drop_duplicates().itertuples(index=False):
        i0, i1 = int(w0 / dtm_), int(w1 / dtm_)
        seg = tot_s_[i0:i1]
        n = len(seg)
        step = max(1, int(0.3 / dtm_))
        lag = max(1, int(0.5 / dtm_))
        if n > step + lag:
            dmax = 0.0
            for i in range(step + lag, n):
                a_now = seg[i - step:i].mean()
                a_ref = seg[i - step - lag:i - step].mean()
                dmax = max(dmax, abs(a_now - a_ref))
            st.loc[(st.window_s == w0) & (st.end_s == w1), "stable"] = \
                dmax < 0.03 * max(seg.mean(), 1e-9)
        else:
            st.loc[(st.window_s == w0) & (st.end_s == w1), "stable"] = False
    ev.insert(0, "rec", tag)
    st.insert(0, "rec", tag)
    allev.append(ev)
    all_ev = ev
    ml = L.mid_load_events(ev).copy()
    if len(ml):
        ml["detected"] = ml.restep.notna()

        def cap(v, ok):
            return f"{v:5.2f}" if ok else "  n/a"

        print("   负载内变载（比值 / 是否重捕获 / 台阶捕获比 / 最大偏差）:")
        for _, r in ml.iterrows():
            ratio_v5 = (r["gain_v5"] / r["raw_gain"]) if abs(r["raw_gain"]) > 1e-9 else float("nan")
            print(f"     t={r.t:6.2f}s 比值={r.ratio:5.3f} "
                  f"{'识别' if r.detected else '漏检'}  "
                  f"台阶捕获 v3={cap(r.gain_ratio_v3, r.valid_gain)} "
                  f"v5={cap(ratio_v5, r.valid_gain)}  "
                  f"最大偏差 v3={r.gap_v3:6.0f} v5={r.gap_v5:6.0f}")
    if len(st):
        piv = st.pivot_table(index=["window_s", "end_s", "dur_s"], columns="algo",
                             values="drift_pct").reindex(columns=ORDER)
        print("   慢相稳定窗时漂残余 (%)（全部窗）:")
        print(piv.round(2).to_string().replace("\n", "\n   "))
        stq = st[st.stable.astype(bool)]
        if len(stq):
            print("   —— 只含「窗内无真实变载」的窗 ——")
            piv2 = stq.pivot_table(index=["window_s", "end_s", "dur_s"], columns="algo",
                                   values="drift_pct").reindex(columns=ORDER)
            print(piv2.round(2).to_string().replace("\n", "\n   "))
            print("   |均值|: " + "  ".join(
                f"{k}={stq[stq.algo == k].drift_pct.abs().mean():.2f}%" for k in ORDER))
        st.to_csv(os.path.join(RES, f"v5_{tag}_windows.csv"), index=False, encoding="utf-8-sig")
    # 全程最大偏差
    dtm = d["dtm"]
    tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
    for k in ORDER[1:]:
        y = L.med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)
        g = np.abs(y - tot_s)
        print(f"   [{k}] 全程 max|显示−原始| = {g.max():7.0f} ADC @ {d['tu'][int(np.argmax(g))]:.1f}s"
              f"（占峰值 {100*g.max()/d['tot'].max():.1f}%）")
        summary.append(dict(rec=tag, algo=k, max_abs_gap=float(g.max()),
                            pct_of_peak=100 * float(g.max()) / d["tot"].max()))
    np.savez_compressed(os.path.join(RES, f"v5_{tag}.npz"), tu=d["tu"], Xu=d["Xu"],
                        periods=np.array(d["periods"]), events=np.array(d["events"]),
                        **{f"Y_{k}": v for k, v in Ys.items()})

if allev:
    ev = pd.concat(allev, ignore_index=True)
    ml = L.mid_load_events(ev).copy()
    ml["detected"] = ml.restep.notna()
    ml.to_csv(os.path.join(RES, "v5_midload_events.csv"), index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("汇总：负载内变载的「台阶捕获比 = 显示增量 ÷ 原始增量」（1.0 = 台阶被完整透传；<1 = 被吃掉）")
    print("=" * 100)
    print(f"  {'算法':>6}{'有效事件':>9}{'捕获比中位':>11}{'捕获比均值':>11}{'捕获比最小':>11}"
          f"{'最大偏差中位':>13}")
    mm = ml[ml.valid_gain.astype(bool)]
    for k in ORDER[1:]:
        s = mm[f"gain_{k}"].div(mm["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {k:>6}{len(s):9d}{s.median():11.2f}{s.mean():11.2f}{s.min():11.2f}"
              f"{mm[f'gap_{k}'].median():13.0f}")
    sv = pd.DataFrame(summary)
    sv.to_csv(os.path.join(RES, "v5_maxgap.csv"), index=False, encoding="utf-8-sig")
    print("\n全程最大偏差汇总：")
    print(sv.pivot_table(index="rec", columns="algo", values="max_abs_gap").round(0).to_string())
    print("\n（占峰值 %）")
    print(sv.pivot_table(index="rec", columns="algo", values="pct_of_peak").round(1).to_string())
