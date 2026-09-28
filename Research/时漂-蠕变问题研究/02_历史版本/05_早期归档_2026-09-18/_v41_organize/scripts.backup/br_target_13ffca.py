# -*- coding: utf-8 -*-
"""以「真实加载沿」为原点的免责期三档对比：temp/变化负载/零负载-中途切换负载-零负载-切换负载/最终测试目标

用户要求：不只比"无责期"这个参数，而要**从真实加载沿的角度**考量 ——
即把 1s / 3s / 5s 三档的行为，全部换算到「距真实加载沿多少秒」这条时间轴上，
再把三档的实践结果画在同一张图里。时间划分由本脚本给出（见 §划分）。

一、时间划分（全部相对**真实加载沿**，单位 s）
    阶跃确认   0      → 2.5            `kStepPersistS`，与档位无关
    免责期     2.5    → 2.5 + FAST     `FAST_S` = 1 / 3 / 5
    A 采集窗   2.5+FAST−FAST/3 → 2.5+FAST   免责期后 1/3（长度 = FAST/3，与档位联动）
    首扣       2.5 + FAST              ← 从这一刻起按新幅度扣（1s→3.5s / 3s→5.5s / 5s→7.5s）
    扣除建立   首扣 + 3τ / + 9τ         g 从 0 起一阶平滑 τ=3s ⇒ 63% 在 +3s、95% 在 +9s

    | 档位 | 确认 | 免责期 | A 窗 | **首扣** | 扣除 63% | 扣除 95% |
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    | 1s | 0~2.5 | 2.5~3.5 | 3.17~3.5 | **3.5** | 6.5 | 12.5 |
    | 3s | 0~2.5 | 2.5~5.5 | 4.50~5.5 | **5.5** | 8.5 | 14.5 |
    | 5s | 0~2.5 | 2.5~7.5 | 5.87~7.5 | **7.5** | 10.5 | 16.5 |

参数口径与 C++ `DriftCompensator::SetFastPhase` 完全一致（FAST_S / EXEMPT_AWIN=FAST_S/3 /
LEV_ARM_S=FAST_S 三处联动），其余参数一律不动；算法本体 = `glm53_v51.py`（v5.1，与 C++
逐帧对拍过）；`v3` 仅作现役基线参考。**未改 src/ 任何代码。**

产出：figures/H3_13ffca_loadedge.png（2×3，全部以真实加载沿为原点）
      results/target_13ffca_{events,metrics,edges,phases}.csv、target_13ffca.npz
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v3 import GLM53v3                           # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CSV = os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                   "最终测试目标", "device_001_seg000.csv")
CONFIRM_S = 2.5                                        # kStepPersistS
TAU_G = 3.0                                            # kTauCreepSmoothS
KS = ["1s", "3s", "5s"]
FASTOF = {"1s": 1.0, "3s": 3.0, "5s": 5.0}
STYLE = {"raw": dict(color="0.55", lw=1.1, ls="-"),
         "1s": dict(color="#1f77b4", lw=1.6, ls="-"),
         "3s": dict(color="#ff7f0e", lw=1.6, ls="-"),
         "5s": dict(color="#d62728", lw=1.6, ls="-"),
         "v3": dict(color="0.35", lw=1.1, ls="--")}
CFG = [(k, GLM53v51, dict(FAST_S=FASTOF[k], EXEMPT_AWIN=FASTOF[k] / 3.0, LEV_ARM_S=FASTOF[k]))
       for k in KS] + [("v3", GLM53v3, {})]


def phases(fast):
    """返回该档在「距真实加载沿」坐标下的阶段时刻（s）。"""
    return dict(confirm_end=CONFIRM_S, exempt_end=CONFIRM_S + fast,
                awin_start=CONFIRM_S + fast - fast / 3.0, awin_end=CONFIRM_S + fast,
                first_ded=CONFIRM_S + fast, ded63=CONFIRM_S + fast + TAU_G,
                ded95=CONFIRM_S + fast + 3 * TAU_G)


def make_traced(cls):
    class _T(cls):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []
            self.st = []

        def process(self, ts, v):
            y = super().process(ts, v)
            self.st.append((float(ts), int(self.in_load), int(self.pending), int(self.hold),
                            float(self.g)))
            return y

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    return _T


# ═══════════════════════════ 数据与运行 ═══════════════════════════
d = L.prep(CSV)
tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
tot_s = L.med_smooth(tot, 0.5 / dtm)
peak = float(tot.max())
periods = [(int(a), min(int(b), len(tu) - 1)) for a, b in L.find_periods(tot, dtm)]
events = [e for e, _ in L.detect_events(tot, dtm)]
print(f"[最终测试目标] {d['span']:.1f}s / {len(tu)} 帧 / {Xu.shape[1]}ch / dt={1000*dtm:.2f}ms / "
      f"峰值 {peak:.0f} ADC / 负载段 {len(periods)} / 检出事件 {len(events)}")

print("\n时间划分（相对真实加载沿，s）")
print(f"{'档位':>5} {'确认结束':>8} {'免责期':>14} {'A 采集窗':>16} {'首扣':>6} "
      f"{'扣除63%':>8} {'扣除95%':>8}")
for k in KS:
    p = phases(FASTOF[k])
    print(f"{k:>5} {p['confirm_end']:8.1f} {CONFIRM_S:>6.1f}~{p['exempt_end']:<6.1f} "
          f"{p['awin_start']:>7.2f}~{p['awin_end']:<7.2f} {p['first_ded']:6.1f} "
          f"{p['ded63']:8.1f} {p['ded95']:8.1f}")

Ys, eps, sts = {}, {}, {}
for k, cls, kw in CFG:
    c = make_traced(cls)(Xu.shape[1])
    for kk, vv in kw.items():
        setattr(c, kk, vv)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    Ys[k], eps[k], sts[k] = Y, c.epoch_t, np.array(c.st)
    print(f"   [{k:3s}] A_max={c.A.max():8.1f} loaded={int(c.loaded.sum()):2d} "
          f"g_end={c.g:+.4f} γ∈[{c.gamma.min():.2f},{c.gamma.max():.2f}] epoch={len(c.epoch_t)}")
Ys["raw"] = Xu.copy()
d["Ys"] = Ys

# ═══════════════════════════ 真实加载沿（事件）表 ═══════════════════════════
ev = L.event_table(d, events, Ys, eps["3s"], gain_s=6.0, algos=KS + ["v3"])
ev["big_event"] = ev["jump"].abs() >= 2000.0
thr = 0.30 * ev["pre"].max()
ev["mid_load"] = (ev["pre"] > thr) & (ev["post"] > thr)
ev["kind"] = np.where(ev["mid_load"], "负载内变载",
                      np.where(ev["jump"] < 0, "卸载", "空载→负载"))
keep = ["t", "pre", "post", "jump", "ratio", "kind", "mid_load", "big_event", "restep", "raw_gain"] + \
       [f"gain_{k}" for k in KS + ["v3"]] + [f"gap_{k}" for k in KS + ["v3"]]
ev = ev[keep]
ev.to_csv(os.path.join(RES, "target_13ffca_events.csv"), index=False, encoding="utf-8-sig")

# 真实「加载沿」= 上升沿（jump > 0 且 |jump| ≥ 2000 ADC）
edges = ev[(ev.jump > 2000)].copy()
edges["idx"] = [int(np.searchsorted(tu, t)) for t in edges.t]
print("\n真实加载沿（上升沿）")
print(f"{'t(s)':>7} {'台阶(ADC)':>10} {'台阶比':>7} {'类型':>10} | "
      + " ".join(f"{k:>16}" for k in KS + ["v3"]))
print("-" * 118)
for _, r in edges.iterrows():
    rr = " ".join(f"{r[f'gain_{k}'] / r['raw_gain']:7.2f}({r[f'gain_{k}']:6.0f})" for k in KS + ["v3"])
    print(f"{r.t:7.2f} {r.jump:10.0f} {r.ratio:7.3f} {r.kind:>10} | {rr}")

erows = []
for _, r in edges.iterrows():
    i = int(r.idx)
    row = dict(t=float(r.t), jump=float(r.jump), kind=r.kind)
    for k in KS + ["v3"]:
        y = L.med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)
        for d6 in (6.0, 12.0):
            j = min(len(tu) - 1, i + int(d6 / dtm))
            row[f"gain6_{k}"] = float(y[j] - y[i]) if d6 == 6.0 else row.get(f"gain6_{k}", np.nan)
            if d6 == 6.0:
                row[f"raw6"] = float(tot_s[j] - tot_s[i])
                row[f"dev12_{k}"] = np.nan
            else:
                row[f"dev12_{k}"] = float(y[j] - tot_s[j])
        row[f"rate6_{k}"] = row[f"gain6_{k}"] / row["raw6"] if abs(row.get("raw6", 0)) > 1e-9 else np.nan
    erows.append(row)
de = pd.DataFrame(erows)
de.to_csv(os.path.join(RES, "target_13ffca_edges.csv"), index=False, encoding="utf-8-sig")

print("\n以真实加载沿为原点的实测口径")
print(f"{'加载沿':>9} {'档位':>5} {'沿后6s台阶捕获':>14} {'沿后12s偏差(ADC)':>17}")
print("-" * 52)
for _, r in de.iterrows():
    for k in KS + ["v3"]:
        print(f"{r.t:9.2f} {k:>5} {r[f'rate6_{k}']:14.2f} {r[f'dev12_{k}']:17.0f}")
print("\n均值（全部真实加载沿）")
for k in KS + ["v3"]:
    print(f"  {k:>5}: 沿后6s捕获 {de[f'rate6_{k}'].mean():.2f}   沿后12s偏差 {de[f'dev12_{k}'].mean():+7.0f} ADC")

# 汇总指标（全程）
rows = []
ml = ev[ev.mid_load & ev.big_event]
for k in KS + ["v3"]:
    y = L.med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)
    gap = np.abs(y - tot_s)
    cap = (ml[f"gain_{k}"] / ml["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna()
    rows.append(dict(algo=k, max_abs_gap=float(gap.max()), pct_of_peak=100 * float(gap.max()) / peak,
                     gap_med=float(ml[f"gap_{k}"].median()), cap_med=float(cap.median()),
                     cap_min=float(cap.min()), epoch=len(eps[k]),
                     rate6=float(de[f"rate6_{k}"].mean()), dev12=float(de[f"dev12_{k}"].mean())))
dm = pd.DataFrame(rows)
dm.to_csv(os.path.join(RES, "target_13ffca_metrics.csv"), index=False, encoding="utf-8-sig")

# 首个加载沿的实测首扣（只有从零起扣时才可测）
e0 = edges.iloc[0]
base0 = float(np.median(tot_s[max(0, int(e0.idx) - int(2 / dtm)):int(e0.idx)]))
amp0 = float(tot_s[int(e0.idx):int(e0.idx) + int(8 / dtm)].max() - base0)
ph_rows, firstded = [], {}
for k in KS:
    ded = (Xu - Ys[k]).sum(axis=1)
    seg = ded[int(e0.idx):int(e0.idx) + int(10 / dtm)]
    fd = float(np.where(seg > 0.005 * amp0)[0][0] * dtm) if (seg > 0.005 * amp0).any() else np.nan
    firstded[k] = fd
    ep = [x for x in eps[k] if -0.5 <= x - float(e0.t) <= 8.0]
    ph_rows.append(dict(档位=k, 确认结束=CONFIRM_S, 免责期结束=phases(FASTOF[k])["exempt_end"],
                        A窗起=phases(FASTOF[k])["awin_start"], 首扣理论=phases(FASTOF[k])["first_ded"],
                        扣除63=phases(FASTOF[k])["ded63"], 扣除95=phases(FASTOF[k])["ded95"],
                        epoch实测=(ep[0] - float(e0.t)) if ep else np.nan, 首扣实测=fd))
dp = pd.DataFrame(ph_rows)
dp.to_csv(os.path.join(RES, "target_13ffca_phases.csv"), index=False, encoding="utf-8-sig")
print(f"\n首个加载沿 @{e0.t:.2f}s（台阶 {e0.jump:+.0f} ADC）的实测首扣：" +
      "  ".join(f"{k} {firstded[k]:.1f}s" for k in KS))

np.savez_compressed(os.path.join(RES, "target_13ffca.npz"), tu=tu, Xu=Xu, tot=tot, tot_s=tot_s,
                    dtm=np.array([dtm]), peak=np.array([peak]), periods=np.array(periods),
                    events=np.array(events), edges=edges[["t", "jump", "idx"]].to_numpy(float),
                    **{f"Y_{k}": v for k, v in Ys.items()},
                    **{f"st_{k}": sts[k] for k in KS})

# ═══════════════════════════ 图：全部以真实加载沿为原点 ═══════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(19.5, 9.6))
fig.suptitle("免责期 1s / 3s / 5s —— 以「真实加载沿」为原点的实践结果"
             f"（中途切换-13ffca / 最终测试目标：{len(tu)} 帧 / {d['span']:.1f}s / "
             f"{Xu.shape[1]}ch / 峰值 {peak:.0f} ADC）", fontsize=14)

# (0,0) 时间划分甘特图
ax = axes[0, 0]
for i, k in enumerate(KS):
    p, col = phases(FASTOF[k]), STYLE[k]["color"]
    y = 2 - i
    ax.add_patch(Rectangle((0, y - 0.28), CONFIRM_S, 0.56, fc="0.8", ec="0.45"))
    ax.add_patch(Rectangle((p["confirm_end"], y - 0.28), FASTOF[k], 0.56,
                           fc=col, alpha=0.28, ec=col))
    ax.add_patch(Rectangle((p["awin_start"], y - 0.28), FASTOF[k] / 3.0, 0.56, fc=col, alpha=0.95))
    ax.add_patch(Rectangle((p["first_ded"], y - 0.28), 9.0, 0.56, fc=col, alpha=0.12,
                           ec=col, ls=":", hatch="///"))
    ax.plot([p["first_ded"]], [y], marker="v", color=col, ms=12, zorder=5)
    ax.text(p["first_ded"] + 0.15, y + 0.34, f"首扣 {p['first_ded']:.1f}s", fontsize=8.5, color=col)
    ax.text(p["first_ded"] + 9.15, y, f"扣除建立 +3s/+9s → {p['ded63']:.1f}/{p['ded95']:.1f}s",
            fontsize=8, color=col, va="center")
    ax.text(-2.15, y, f"免责 {k}", fontsize=10, color=col, va="center", ha="left", weight="bold")
ax.set_xlim(-2.3, 20.5); ax.set_ylim(-0.7, 2.75)
ax.set_yticks([]); ax.set_xlabel("距真实加载沿的时间 (s)")
ax.set_title("(1) 时间划分：0~2.5s 阶跃确认（与档位无关，灰色）→ 免责期（浅色）→ "
             "A 采集窗（深色，末端即首扣）→ 扣除建立（斜纹）")
ax.axvline(0, color="k", lw=1.2)
ax.text(0.05, 2.62, "真实加载沿", fontsize=9, va="top")
ax.grid(axis="x", alpha=0.3)

# (0,1)~(1,0) 三个真实加载沿各自放大（x = 距加载沿）
PANELS = [((0, 1), edges.iloc[0], 18.0, "①空载→负载"), ((0, 2), edges.iloc[1], 16.0, "②负载内加重"),
          ((1, 0), edges.iloc[2], 16.0, "③负载内加重（v3 漏检那一次）")]
for (pos, e, xw, tag) in PANELS:
    ax = axes[pos]
    i0 = int(e.idx)
    w = slice(max(0, i0 - int(2 / dtm)), min(len(tu), i0 + int(xw / dtm)))
    xx = tu[w] - float(e.t)
    ax.plot(xx, tot_s[w], label="原始", **STYLE["raw"])
    ax.plot(xx, L.med_smooth(Ys["v3"].sum(axis=1), 0.5 / dtm)[w], label="v3（现役，参考）",
            **STYLE["v3"])
    for k in KS:
        ax.plot(xx, L.med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)[w], label=f"免责 {k}", **STYLE[k])
        p = phases(FASTOF[k])
        ax.axvline(p["first_ded"], color=STYLE[k]["color"], ls=":", lw=1.0)
        ax.axvspan(p["awin_start"], p["awin_end"], color=STYLE[k]["color"], alpha=0.12, lw=0)
    ax.axvline(CONFIRM_S, color="0.4", ls="--", lw=0.9)
    ax.text(CONFIRM_S + 0.06, 0.02, "确认结束\n2.5s", fontsize=7.5, color="0.35",
            transform=ax.get_xaxis_transform())
    j12 = min(len(tu) - 1, i0 + int(12 / dtm))
    ax.axhline(tot_s[j12], color="0.4", ls="-.", lw=0.9)
    ax.text(xw - 0.2, tot_s[j12], f"原始 +12s = {tot_s[j12]:.0f}", fontsize=8, color="0.3",
            va="bottom", ha="right")
    for k in KS:
        dev = L.med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)[j12] - tot_s[j12]
        ax.annotate(f"{k} {dev:+.0f}", (xw, L.med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)[j12]),
                    fontsize=8, color=STYLE[k]["color"], xytext=(-2, 0),
                    textcoords="offset points", ha="right", va="center")
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlim(-2, xw)
    ax.set_title(f"({['①', '②', '③'][[p[3] for p in PANELS].index(tag)]}) 真实加载沿 {tag} "
                 f"@{float(e.t):.2f}s（台阶 {float(e.jump):+.0f} ADC，占电平 {float(e.ratio):.2f}）"
                 f"：虚线 = 各档首扣时刻")
    ax.set_xlabel("距真实加载沿的时间 (s)"); ax.set_ylabel("总量 (ADC)")
    ax.legend(fontsize=7.5, loc="lower right")

# (1,1) 扣除量 vs 距加载沿（三个加载沿叠加 + 均值）
ax = axes[1, 1]
tt = np.arange(-2.0, 18.0, 0.05)
for k in KS:
    acc = []
    for _, e in edges.iloc[:3].iterrows():
        i0 = int(e.idx)
        ded = L.med_smooth((Xu - Ys[k]).sum(axis=1), 0.5 / dtm)
        idx = np.clip((i0 + (tt / dtm)).astype(int), 0, len(tu) - 1)
        y = ded[idx]
        ax.plot(tt, y, color=STYLE[k]["color"], lw=0.9, alpha=0.35)
        acc.append(y / max(abs(y).max(), 1e-9))
    ax.plot(tt, np.mean(acc, axis=0), color=STYLE[k]["color"], lw=2.2,
            label=f"免责 {k}（3 条沿均值，归一化）")
ax.axvline(0, color="k", lw=1.0)
for k in KS:
    p = phases(FASTOF[k])
    ax.plot([p["first_ded"]], [-0.03], marker="v", color=STYLE[k]["color"], ms=10, clip_on=False)
ax.set_xlabel("距真实加载沿的时间 (s)"); ax.set_ylabel("已生效扣除（各沿除以自身峰值）")
ax.set_title("(4) 扣除量的建立过程（3 个真实加载沿各自细线 + 均值粗线）")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

# (1,2) 表：阶段时刻 + 实测结果
ax = axes[1, 2]
ax.axis("off")
cell = [["档位", "首扣\n(理论)", "扣除 63%/95%\n(理论, s)", "首扣\n(实测)", "沿后 6s\n台阶捕获",
         "沿后 12s\n偏差(ADC)", "全程最大\n偏差(ADC)", "占\n峰值"]]
for k in KS:
    p = phases(FASTOF[k])
    cell.append([f"免责 {k}", f"{p['first_ded']:.1f}", f"{p['ded63']:.1f} / {p['ded95']:.1f}",
                 f"{firstded[k]:.1f}", f"{de[f'rate6_{k}'].mean():.2f}",
                 f"{de[f'dev12_{k}'].mean():+.0f}",
                 f"{float(dm[dm.algo == k].max_abs_gap.iloc[0]):.0f}",
                 f"{float(dm[dm.algo == k].pct_of_peak.iloc[0]):.1f}%"])
cell.append(["v3（参考）", "—", "—", "—", f"{de['rate6_v3'].mean():.2f}",
             f"{de['dev12_v3'].mean():+.0f}",
             f"{float(dm[dm.algo == 'v3'].max_abs_gap.iloc[0]):.0f}",
             f"{float(dm[dm.algo == 'v3'].pct_of_peak.iloc[0]):.1f}%"])
tb = ax.table(cellText=cell, cellLoc="center", bbox=[0.0, 0.06, 1.0, 0.86])
tb.auto_set_font_size(False); tb.set_fontsize(9)
for j in range(8):
    tb[0, j].set_facecolor("#e8e8e8"); tb[0, j].set_text_props(weight="bold")
for i2, k in enumerate(KS + ["v3"], start=1):
    tb[i2, 0].set_facecolor(STYLE[k]["color"] if k in KS else "0.85")
    tb[i2, 0].set_text_props(color="white" if k in KS else "black", weight="bold")
ax.set_title(f"(5) 实测（{len(edges)} 个真实加载沿；沿后 6s 捕获 = 该沿后 6s 显示增量 ÷ 原始增量）",
             pad=16)

fig.tight_layout(rect=(0, 0, 1, 0.93))
fp = os.path.join(FIG, "H3_13ffca_loadedge.png")
fig.savefig(fp, dpi=115)
plt.close(fig)
print(f"\n图已保存：{fp}")
