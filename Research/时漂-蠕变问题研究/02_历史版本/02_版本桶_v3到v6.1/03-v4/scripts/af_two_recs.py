# -*- coding: utf-8 -*-
"""跑 temp/变化负载/零负载-中途切换负载-零负载-切换负载 下的两份录制（含 最终测试目标/）。

产出：results/rec_<tag>.npz、results/rec_<tag>_events.csv、results/rec_<tag>_metrics.csv，
并在控制台打印事件比值↔是否重捕获的对照表（回答「为什么图上看不到 15N 被拉回」）。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402

BASE = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
        r"\零负载-中途切换负载-零负载-切换负载")
RECS = [("1d9493", os.path.join(BASE, "device_001_seg000.csv")),
        ("13ffca", os.path.join(BASE, "最终测试目标", "device_001_seg000.csv"))]

all_ev = []
for tag, path in RECS:
    if not os.path.exists(path):
        print(f"[skip] 不存在: {path}")
        continue
    d = L.prep(path)
    d["periods"] = L.find_periods(d["tot"], d["dtm"])
    d["events"] = [e for e, _ in L.detect_events(d["tot"], d["dtm"])]
    print("=" * 104)
    print(f"[{tag}] {os.path.relpath(path, os.path.dirname(BASE))}")
    print(f"  帧数={len(d['t'])} 时长={d['span']:.2f}s 采样={1/d['dtm']:.2f}Hz 通道={d['X'].shape[1]} "
          f"重复时间戳={d['dup']} 峰值总量={d['tot'].max():.0f}")
    print(f"  负载段 {len(d['periods'])} 个: " + ", ".join(
        f"[{d['tu'][a]:.2f},{d['tu'][b]:.2f}]={np.median(d['tot'][a:b]):.0f}" for a, b in d["periods"]))
    print(f"  变载事件 {len(d['events'])} 个: " + ", ".join(f"{d['tu'][e]:.2f}" for e in d["events"]))

    Ys, ck, epochs = {}, {}, {}
    for k, cls, kw in L.ALGOS:
        if cls is None:
            Ys[k] = d["Xu"].copy()
            continue
        Ys[k], c = L.run_algo(d["tu"], d["Xu"], L.make_traced(cls), **kw)
        ck[k], epochs[k] = c, c.epoch_t
        print(f"  [{k}] 自检 A_max={c.A.max():.3f} loaded={int(c.loaded.sum())} g_end={c.g:+.4f} "
              f"epoch={len(c.epoch_t)}")
    d["Ys"] = Ys
    d["epochs"] = epochs

    ev = L.event_table(d, d["events"], Ys, epochs["v3"])
    st = L.slow_windows(d)
    ev.insert(0, "rec", tag)
    st.insert(0, "rec", tag)
    ev.to_csv(os.path.join(RES, f"rec_{tag}_events.csv"), index=False, encoding="utf-8-sig")
    st.to_csv(os.path.join(RES, f"rec_{tag}_metrics.csv"), index=False, encoding="utf-8-sig")
    all_ev.append(ev)

    print(f"\n  事件台账（比值 = |ADC台阶| / 变载前读数；'重捕获' = v3 在该事件后 6s 内是否 restep）:")
    print(f"    {'t(s)':>7}{'变载前读数':>11}{'台阶Δ':>9}{'比值':>7}{'重捕获':>10}"
          f"{'v3最大偏差':>11}{'v4最大偏差':>11}{'显示增益/原始':>13}   类型")
    for _, r in ev.iterrows():
        kind = "空载→负载" if r["pre"] <= 0.3 * ev["pre"].max() else \
               ("卸载" if r["post"] <= 0.3 * ev["pre"].max() else "负载内变载")
        gr = r.get("gain_ratio_v3")
        grs = f"{gr:.2f}" if pd.notna(gr) else "-"
        print(f"    {r['t']:7.2f}{r['pre']:11.0f}{r['jump']:+9.0f}{r['ratio']:7.3f}"
              f"{('是 @+%.1fs' % r['restep']) if pd.notna(r['restep']) else '否':>10}"
              f"{r['gap_v3']:11.0f}{r['gap_v4_fast5']:11.0f}{grs:>13}   [{kind}]")
    if len(st):
        piv = st.pivot_table(index=["window_s", "end_s", "dur_s"], columns="algo",
                             values="drift_pct").reindex(columns=L.ORDER)
        print(f"\n  慢相稳定窗时漂残余（事件后 8s 保护带，占本窗电平 %）:")
        print(piv.round(2).to_string())
        print("   |均值|: " + "  ".join(
            f"{L.LBL[k]}={st[st.algo == k]['drift_pct'].abs().mean():.2f}%" for k in L.ORDER))

    np.savez_compressed(os.path.join(RES, f"rec_{tag}.npz"),
                        tu=d["tu"], Xu=d["Xu"], periods=np.array(d["periods"]),
                        events=np.array(d["events"]),
                        **{f"Y_{k}": v for k, v in Ys.items()})
    print(f"  saved: results/rec_{tag}.npz / rec_{tag}_events.csv / rec_{tag}_metrics.csv")

# ============ 汇总：比值 ↔ 是否重捕获（回答用户的疑问）============
if all_ev:
    ev = pd.concat(all_ev, ignore_index=True)
    ml = L.mid_load_events(ev)                  # 只看负载内变载
    ml["detected"] = ml["restep"].notna()
    print("\n" + "=" * 104)
    print("汇总：ADC 域「台阶 / 变载前读数」比值  ↔  v3 是否识别为变载（只算负载内变载）")
    print("=" * 104)
    print(f"  {'比值区间':>12}{'事件数':>7}{'被识别':>8}{'被漏检':>8}{'v3 最大偏差中位':>17}"
          f"{'显示增益/原始 中位':>20}")
    for lo, hi in ((0.0, 0.20), (0.20, 0.29), (0.29, 0.40), (0.40, 1.0), (1.0, 99.0)):
        s = ml[(ml.ratio >= lo) & (ml.ratio < hi)]
        if not len(s):
            continue
        print(f"  {f'{lo:.2f}~{hi:.2f}':>12}{len(s):7d}{int(s.detected.sum()):8d}"
              f"{int((~s.detected).sum()):8d}{s.gap_v3.median():17.0f}"
              f"{s.gain_ratio_v3.median():20.2f}")
    hi = ml[ml.ratio >= 0.29]
    lo = ml[ml.ratio < 0.29]
    print(f"\n  比值 ≥0.29（理论门限以上）: {int(hi.detected.sum())}/{len(hi)} 被识别")
    print(f"  比值 <0.29（理论门限以下）: {int(lo.detected.sum())}/{len(lo)} 被识别")
    ml.to_csv(os.path.join(RES, "rec_midload_events.csv"), index=False, encoding="utf-8-sig")
    print("  saved: results/rec_midload_events.csv")
