# -*- coding: utf-8 -*-
"""GLM53 探针：变化负载下 v1/v2 的逐帧行为量化（独立复算，不覆盖 f_ 脚本产物）

目的：回答"切换负载时显示是否出现不合理的大跳变"。
做法：
  1) 复用 f_varying_load.py 的 CompV1/CompV2 实现（import 其类，避免复制走样）；
  2) 逐帧记录 v2 内部状态（level/fast/slow/A 覆盖帧数/g/creep/b/armed/in_load）；
  3) 事件化分析：对每个 v2 事件计算前后真实电平、显示电平、跃变幅度、
     跃变耗时、稳态误差、稳态平坦度；
  4) 输出 figures/g1_*.png 与 results/g_*.csv（不触碰 e_* 产物）。
"""
import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                      # GLM53
BASE = os.path.dirname(OUT)                      # temp
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")

_spec = importlib.util.spec_from_file_location("fv", os.path.join(HERE, "f_varying_load.py"))
fv = importlib.util.module_from_spec(_spec)
sys.modules["fv"] = fv
_spec.loader.exec_module(fv)                     # 该模块顶层会重算 e_* 产物（参数一致，结果不变）

CompV1, CompV2, load_csv, find_segment = fv.CompV1, fv.CompV2, fv.load_csv, fv.find_segment


class ProbeV2(CompV2):
    """带逐帧状态记录的 v2"""

    def __init__(self):
        super().__init__()
        self.trace = []

    def process(self, ts, v):
        out = super().process(ts, v)
        self.trace.append(dict(
            ts=ts,
            total=float(np.sum(v)),
            out_total=float(np.sum(out)),
            level=self.level, fast=self.fast, slow=self.slow,
            ts_sm=self.ts, min_ts=self.min_ts, max_ts=self.max_ts,
            A_mean=float(np.mean(self.A)) if self.A is not None else 0.0,
            n_loaded=int(np.sum(self.loaded)) if self.loaded is not None else 0,
            g=self.g, in_load=self.in_load, armed=self.armed,
            a_captured=self.a_captured,
            creep_total=float(np.sum(np.clip(self.gamma * self.A * self.g,
                                             fv.P2["creep_lo"] * self.A,
                                             fv.P2["creep_hi"] * self.A))) if self.A is not None else 0.0,
            b_total=float(np.sum(self.b)) if self.b is not None else 0.0,
            pend=self.pend_t is not None,
        ))
        return out


def dedup(t):
    """时间戳重复帧（本传感器约 3/4）会掩盖真实跃变速率，取唯一时间戳子集"""
    return np.r_[True, np.diff(t) > 1e-6]


def analyse(loc_path, tag, events):
    t, X, tc = load_csv(loc_path)
    total = X.sum(axis=1)
    main = int(np.argmax(X.max(axis=0)))

    comp = ProbeV2()
    Y2 = np.empty_like(X, dtype=float)
    for i in range(len(t)):
        Y2[i] = comp.process(t[i], X[i].astype(float))
    Y1, c1 = fv.run_case(CompV1, X, t)

    tr = pd.DataFrame(comp.trace)
    tr["tot1"] = Y1.sum(1)
    tr["main_raw"] = X[:, main]
    tr["main_v1"] = Y1[:, main]
    tr["main_v2"] = Y2[:, main]
    keep = dedup(t.to_numpy() if hasattr(t, "to_numpy") else t)
    print(f"\n{'='*100}\n[{tag}] dur={t[-1]:.1f}s frames={len(t)} uniq_ts={int(keep.sum())} main=ch{main}")
    print(f"  v2 events: {[(round(a,1), b) for a, b in comp.events]}")
    print(f"  v1 events: {[(round(a,1), b) for a, b in c1.events]}")

    # ── 每个 v2 事件的前后量化（用唯一时间戳帧） ──
    rows = []
    for ets, etag in comp.events:
        i_ev = int(np.argmin(np.abs(tr.ts.values - ets)))
        pre = tr[(tr.ts > ets - 3.0) & (tr.ts < ets - 0.5)]
        post_on = tr[(tr.ts > ets + 0.2) & (tr.ts < ets + 1.2)]
        post_st = tr[(tr.ts > ets + 4.0) & (tr.ts < ets + 8.0)]
        if len(pre) < 5 or len(post_st) < 20:
            continue
        raw_pre, raw_post = pre.total.median(), post_st.total.median()
        disp_pre, disp_post = pre.out_total.median(), post_st.out_total.median()
        d_true = raw_post - raw_pre
        d_disp = disp_post - disp_pre
        win = tr[(tr.ts > ets - 1.5) & (tr.ts < ets + 4.0)]
        win_u = win[np.r_[True, np.diff(win.ts.values) > 1e-6]]
        d_disp_series = np.diff(win_u.out_total.values)
        d_true_series = np.diff(win_u.total.values)
        imax_disp = int(np.argmax(np.abs(d_disp_series))) if len(d_disp_series) else 0
        # 阶跃速率：把 0.2s 窗口内的显示变化折算成 ADC/s
        rate_disp = np.abs(d_disp_series) / np.maximum(np.diff(win_u.ts.values), 1e-9)
        rows.append(dict(
            dataset=tag, event=etag, t=round(ets, 2),
            raw_pre=round(raw_pre), raw_post=round(raw_post), d_true=round(d_true),
            disp_pre=round(disp_pre), disp_post=round(disp_post), d_disp=round(d_disp),
            gain=round(d_disp / d_true, 3) if abs(d_true) > 1 else np.nan,
            peak_dev_from_pre=round(float(np.max(np.abs(win_u.out_total.values - disp_pre)))),
            max_rate_ADC_s=round(float(np.max(rate_disp))) if len(rate_disp) else 0,
            raw_max_rate_ADC_s=round(float(np.max(np.abs(d_true_series) / np.maximum(np.diff(win_u.ts.values), 1e-9)))) if len(d_true_series) else 0,
            post_flat_std=round(float(post_st.out_total.std())),
            post_flat_pct=round(100 * float(post_st.out_total.std()) / max(abs(disp_post), 1e-9), 2),
            err_after=round(disp_post - raw_post),
            err_after_pct=round(100 * (disp_post - raw_post) / max(abs(raw_post), 1e-9), 2),
        ))
    ev = pd.DataFrame(rows)
    print("\n  ── 事件跃变量化（disp=显示总量, raw=原始总量） ──")
    print(ev.to_string(index=False) if len(ev) else "  (无事件)")

    # ── 逐帧最大跃变率（唯一时间戳帧） ──
    t_u = tr.ts.values[keep]
    print("\n  ── 逐帧跃变率（唯一时间戳帧, ADC/s） ──")
    for name, col in [("raw", "total"), ("v1", "tot1"), ("v2", "out_total")]:
        y = tr[col].values[keep]
        r = np.abs(np.diff(y)) / np.maximum(np.diff(t_u), 1e-9)
        j = int(np.argmax(r))
        print(f"    {name:4s} max={r.max():>12,.0f}  @ t={t_u[j+1]:>6.2f}s   p99.9={np.percentile(r,99.9):>10,.0f}")

    # ── 检测到的电平段：显示 vs 真实（稳态误差） ──
    segs = find_segment(total)
    rows2 = []
    for si, (s0, s1) in enumerate(segs):
        if t[s1 - 1] - t[s0] < 5:
            continue
        sl = slice(s0, s1)
        lvl = float(total[sl].mean())
        for name, Y in [("raw", X.sum(1)), ("v1", Y1.sum(1)), ("v2", Y2.sum(1))]:
            y = Y[sl]
            core = y[int(0.15 * len(y)): int(0.9 * len(y))]
            rows2.append(dict(dataset=tag, seg=si, t0=round(t[s0], 1),
                              dur=round(t[s1 - 1] - t[s0], 1),
                              raw_level=round(lvl), algo=name,
                              mean=round(float(core.mean())),
                              err_vs_raw_pct=round(100 * (core.mean() - lvl) / max(abs(lvl), 1e-9), 1),
                              std_pct=round(100 * core.std() / max(abs(lvl), 1e-9), 1)))
    seg = pd.DataFrame(rows2)
    print("\n  ── 分段稳态（相对原始电平的误差%） ──")
    print(seg.pivot_table(index=["seg", "t0", "dur", "raw_level"], columns="algo",
                          values="err_vs_raw_pct").to_string())

    tr.to_csv(os.path.join(RES, f"g_trace_{tag}.csv"), index=False, encoding="utf-8-sig")

    # ── 图：叠加事件标注 ──
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), constrained_layout=True, sharex=True)
    axes[0].plot(t, total, lw=0.5, color="k")
    axes[0].set_title(f"{tag} · 原始总量")
    axes[1].plot(t, total, lw=0.5, color="k", alpha=0.35, label="原始")
    axes[1].plot(t, Y1.sum(1), lw=0.7, color="tab:red", label="v1")
    axes[1].plot(t, Y2.sum(1), lw=0.7, color="tab:blue", label="v2")
    axes[1].set_title("显示总量对比"); axes[1].legend(fontsize=8)
    axes[2].plot(t, tr.creep_total, lw=0.7, color="tab:green", label="creep 总量")
    axes[2].plot(t, tr.b_total, lw=0.7, color="tab:orange", label="基线 b 总量")
    axes[2].set_title("v2 内部量"); axes[2].legend(fontsize=8)
    axes[3].plot(t, tr.g, lw=0.7, color="tab:purple", label="g")
    axes[3].plot(t, tr.n_loaded / max(tr.n_loaded.max(), 1), lw=0.7, color="gray", label="n_loaded(归一)")
    axes[3].plot(t, tr.a_captured.astype(int), lw=0.7, color="brown", label="a_captured")
    axes[3].set_title("v2 蠕变状态"); axes[3].legend(fontsize=8); axes[3].set_xlabel("时间 (s)")
    for ax in axes:
        for ets, etag in comp.events:
            ax.axvline(ets, color="gray", ls=":", lw=0.8)
    fig.savefig(os.path.join(FIG, f"g1_probe_{tag}.png"), dpi=130)
    plt.close(fig)
    return ev, seg, tr


VARY = {"A": "零负载-切换负载-零负载-再切换负载",
        "B": "零负载-中途切换负载-零负载-切换负载"}
all_ev, all_seg = [], []
for tag, loc in VARY.items():
    ev, seg, tr = analyse(os.path.join(BASE, "变化负载", loc, "device_001_seg000.csv"), tag, None)
    all_ev.append(ev); all_seg.append(seg)

pd.concat(all_ev).to_csv(os.path.join(RES, "g_events.csv"), index=False, encoding="utf-8-sig")
pd.concat(all_seg).to_csv(os.path.join(RES, "g_segments.csv"), index=False, encoding="utf-8-sig")
print("\nsaved: figures/g1_probe_A.png, g1_probe_B.png; results/g_events.csv, g_segments.csv, g_trace_*.csv")
