# -*- coding: utf-8 -*-
"""图 3：短滞后电平差判据的实测分离度。

统计量（与 §5.1 的表一致）：
  d_lev(t) = mean{total : (t-0.3, t]} − mean{total : (t-0.8, t-0.3]}
  相对电平 = |d_lev| / |参考窗电平|  ，门限取 5%
按「稳定窗（负载段内、且距任何真实变载沿 >3 s）」与「变载沿 ±3 s」分组统计其
分位数与最大值，看两侧是否可分。

事件沿本身由 scripts/ad_lib.detect_events 的独立口径给出（与判据无关），
因此这里的分离度不是自证。

产出：figures/F3_step_detect.png、results/f3_detect_stats.csv、results/f3_detect_frames.csv
"""
import numpy as np
import pandas as pd

import pd as P
from figstyle import (C_ALG, C_ALT, C_FAST, C_GRAY, plt, save_figure)

LEV_FAST, LEV_LAG = 0.3, 0.5
EV_HALF = 3.0


def dlev_frames(tu, tot):
    csum = np.concatenate([[0.0], np.cumsum(tot)])
    lo_now = np.searchsorted(tu, tu - LEV_FAST, side="right")
    hi_now = np.searchsorted(tu, tu, side="right")
    lo_ref = np.searchsorted(tu, tu - (LEV_FAST + LEV_LAG), side="right")
    hi_ref = np.searchsorted(tu, tu - LEV_FAST, side="right")
    n = len(tu)
    lv_now = np.full(n, np.nan)
    lv_ref = np.full(n, np.nan)
    ok = hi_now > lo_now
    lv_now[ok] = (csum[hi_now[ok]] - csum[lo_now[ok]]) / (hi_now[ok] - lo_now[ok])
    ok = hi_ref > lo_ref
    lv_ref[ok] = (csum[hi_ref[ok]] - csum[lo_ref[ok]]) / (hi_ref[ok] - lo_ref[ok])
    return lv_now, lv_ref


def rolling_std(x, w):
    """等间隔序列上的 w 点滑动标准差。"""
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    lo = np.clip(np.arange(n) - w + 1, 0, n)
    hi = np.arange(n) + 1
    cnt = (hi - lo).astype(float)
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]
    var = np.maximum(s2 / cnt - (s / cnt) ** 2, 0.0) * cnt / np.maximum(cnt - 1.0, 1.0)
    return np.sqrt(var)


stats, frames = [], []
for tag, path in P.RECS:
    d = P.load_rec(path)
    tu, dtm, tot, tot_s = d["tu"], d["dtm"], d["tot"], d["tot_s"]
    d["periods"] = P.L.find_periods(tot, dtm)
    ev_idx = [e for e, _ in P.L.detect_events(tot, dtm)]
    evtbl = P.L.event_table(d, ev_idx, {"a": tot}, [], gain_s=6.0, algos=[])
    evtbl["mid"] = False
    if len(evtbl):
        thr_e = 0.30 * evtbl["pre"].max()
        evtbl["mid"] = (evtbl["pre"] > thr_e) & (evtbl["post"] > thr_e)
    ev_t = sorted(float(x) for x in evtbl.loc[evtbl.mid, "t"].to_numpy()) if len(evtbl) else []

    lv_now, lv_ref = dlev_frames(tu, tot)
    dlev_abs = np.abs(lv_now - lv_ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlev_pct = 100.0 * dlev_abs / np.abs(lv_ref)

    # 负载段
    in_period = np.zeros(len(tu), bool)
    for a, b in d["periods"]:
        in_period |= (tu >= float(tu[a]) + 2.0) & (tu <= float(tu[min(b, len(tu) - 1)]))

    # 「稳定」的定义：前后 ±3 s 内总量平滑电平的滑动标准差 < 0.5%×该帧电平。
    # 这样快相尾巴、加载/卸载沿、手指调整、未检出的轻微变载都被排除在稳定窗之外，
    # 剩下的帧上 |d_lev| 才真正只反映传感噪声与慢相斜坡。
    w = max(3, int(round(3.0 / dtm)))
    sd = rolling_std(tot_s, w)
    lvl = np.maximum(tot_s, 1.0)
    quiet = sd < 0.005 * lvl
    near_ev = np.zeros(len(tu), bool)
    for te in ev_t:
        near_ev |= (tu >= te - EV_HALF) & (tu <= te + EV_HALF)

    valid = np.isfinite(dlev_pct) & (lv_ref > 0.02 * np.max(tot))
    stable = valid & in_period & quiet & (~near_ev)
    evmask = valid & near_ev & (dlev_abs > 0)
    s, e = dlev_pct[stable], dlev_pct[evmask]
    if s.size == 0 or e.size == 0:
        print(f"    [warn] {tag}: 稳定 {s.size}、事件 {e.size}")
    stats.append(dict(rec=tag, n_frames=len(tu), n_stable=int(s.size), n_event=int(e.size),
                      stable_p50=float(np.median(s)), stable_p99=float(np.percentile(s, 99)),
                      stable_max=float(s.max()),
                      event_max=float(e.max()), event_p90=float(np.percentile(e, 90)),
                      sep=float(e.max() / np.percentile(s, 99)),
                      n_events=len(ev_t),
                      stable_max_abs=float(dlev_abs[stable].max()),
                      event_max_abs=float(dlev_abs[evmask].max())))
    print(f"[F3] {tag}: 稳定窗 p99 {np.percentile(s,99):.2f}%、max {s.max():.2f}%"
          f"（{dlev_abs[stable].max():.0f} ADC，{s.size} 帧）；"
          f"变载沿 max {e.max():.0f}%；分离 {e.max()/np.percentile(s,99):.0f}×（{len(ev_t)} 次变载）")

    sub = np.zeros(len(tu), bool)
    sub[::2] = True
    for i in np.where(stable & sub)[0][:8000]:
        frames.append(dict(rec=tag, kind="稳定窗", pct=float(dlev_pct[i])))
    for i in np.where(evmask)[0]:
        frames.append(dict(rec=tag, kind="变载沿邻域", pct=float(dlev_pct[i])))

P.save_table(pd.DataFrame(stats), "f3_detect_stats.csv")
P.save_table(pd.DataFrame(frames), "f3_detect_frames.csv")

# ── 绘图 ──────────────────────────────────────────────────────────
st = pd.DataFrame(stats)
fr = pd.DataFrame(frames)
fig = plt.figure(figsize=(12.6, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22,
                      left=0.062, right=0.985, top=0.90, bottom=0.215)

ax = fig.add_subplot(gs[0, 0])
bins = np.logspace(-2, 2.7, 70)
ax.hist(fr.loc[fr.kind == "稳定窗", "pct"].clip(1e-2, 10 ** 2.7), bins=bins, color=C_GRAY,
        alpha=0.78, label="稳定窗（距真实变载 > 3 s）")
ax.hist(fr.loc[fr.kind == "变载沿邻域", "pct"].clip(1e-2, 10 ** 2.7), bins=bins, color=C_ALG,
        alpha=0.70, label="真实变载沿 ±3 s")
ax.axvline(5.0, color=C_FAST, lw=2.0, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-2, 10 ** 2.7)
ax.set_ylim(0.6, 2.2e4)
ax.set_xlabel("0.5 s 滞后窗电平差 $|d_{\\mathrm{lev}}|$ ÷ 参考窗电平（%）")
ax.set_ylabel("帧数")
ax.set_title("(a) 台阶与蠕变斜坡在同一差分上的分布（实采 4 份）", fontsize=11)
ax.grid(alpha=0.25, which="both", lw=0.6)
ax.legend(fontsize=8.4, loc="upper left", framealpha=0.95, borderaxespad=0.8)
ax.text(6.2, 3.0, "5% 门限", fontsize=9.2, color="#8A4B00",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.8))
fig.text(0.062, 0.035,
         "稳定窗最大 %.2f%%（含静默期抖动）；门限取 5%% 时下侧余量 %.1f×；真实变载沿最大 %.0f%%"
         "（四份里最小的一次为 %.0f%%，上侧余量 %.1f×）"
         % (st.stable_max.max(), 5.0 / st.stable_max.max(), st.event_max.max(),
            st.event_max.min(), st.event_max.min() / 5.0),
         fontsize=9.0, color=C_GRAY, ha="left", va="bottom")

ax2 = fig.add_subplot(gs[0, 1])
ypos = np.arange(len(st))[::-1]
ax2.barh(ypos + 0.28, st.stable_p99, height=0.26, color=C_ALT, alpha=0.9, label="稳定窗 p99")
ax2.barh(ypos, st.stable_max, height=0.26, color=C_GRAY, alpha=0.9, label="稳定窗 最大")
ax2.barh(ypos - 0.30, st.event_max, height=0.26, color=C_ALG, alpha=0.9,
         label="真实变载沿 最大")
ax2.axvline(5.0, color=C_FAST, lw=2.0, ls="--")
ax2.set_xscale("log")
ax2.set_xlim(0.05, 2000)
ax2.set_yticks(ypos)
ax2.set_yticklabels(["%s\n(%d 次负载内变载)" % (r.rec, int(r.n_events)) for r in st.itertuples()],
                    fontsize=8.6)
ax2.set_xlabel("$|d_{\\mathrm{lev}}|$ ÷ 参考窗电平（%，对数）")
ax2.set_title("(b) 四份录制：门限两侧的余量", fontsize=11)
ax2.grid(alpha=0.25, axis="x", which="both", lw=0.6)
ax2.legend(fontsize=8.3, loc="lower right", framealpha=0.95, borderaxespad=0.7)
for y, r in zip(ypos, st.itertuples()):
    ax2.text(r.event_max * 1.45, y - 0.30, "%.0f%%" % r.event_max, va="center",
             fontsize=8.3, color=C_ALG)
    ax2.text(max(r.stable_max, r.stable_p99) * 1.45, y - 0.02, "%.1f%%" % r.stable_max,
             va="center", fontsize=8.3, color=C_GRAY)

save_figure(fig, "F3_step_detect.png")
