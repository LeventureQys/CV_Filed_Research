# -*- coding: utf-8 -*-
"""步骤 C3：因果算法对比图组（最终版）。"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import savefig, dump_json, DATASETS, FIG, RES
from causal_harness import prepare, run_stream, online_load_detect, causal_smooth
import causal_algorithms as CA
from causal_algorithms import Shape, FROZEN
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESF = os.path.join(RES, "C2_causal_sweep.json")
if not os.path.exists(RESF):
    raise SystemExit("请先运行 c2_causal_sweep.py")
with open(RESF, encoding="utf-8") as f:
    SWEEP = json.load(f)
with open(os.path.join(RES, "C0_creep_shape.json"), encoding="utf-8") as f:
    SHAPE = json.load(f)

SC = Shape(**FROZEN["common"])
S1 = Shape(**FROZEN["d1"])

DATA = {}
for name in DATASETS:
    D = prepare(name)
    X, t, fs = D["X"], D["t"], D["fs"]
    a0, a1 = D["pre"]; c0, d0 = D["load"]
    w = lambda s: int(round(s * fs))
    Xn = X - X[a0:a1].mean(axis=0)[None, :]
    n_on, _, _ = online_load_detect(Xn.sum(axis=1), fs, a1)
    REF = Xn[n_on + w(1.0):n_on + w(3.0)].mean(axis=0)
    act = np.where(Xn[c0:d0].max(axis=0) > 0.05)[0]
    j = int(act[np.argmax(REF[act])])
    DATA[name] = dict(D=D, Xn=Xn, t=t, fs=fs, n_on=n_on, REF=REF, act=act,
                      j=j, c0=c0, d0=d0, a0=a0, a1=a1, w=w,
                      on_true=D["on_true"])

CAND = [
    ("原始（不补偿）", CA.Raw(), "k", 1.5),
    ("因果 EWMA τ=1s", CA.CausalEWMA(1.0), "#7f7f7f", 1.0),
    ("因果 MA 1s", CA.CausalMA(1.0), "#bcbd22", 1.0),
    ("因果 Kalman q=1e-11", CA.CausalKalmanCV(1e-11, 1e-5), "#2ca02c", 1.2),
    ("形态先验+水平追踪 τs=1s", CA.ShapeDivide(SC, 1.0), "#d62728", 1.9),
    ("形态先验+水平追踪 τs=3s", CA.ShapeDivide(SC, 3.0), "#ff7f0e", 1.5),
    ("阵列共模因子", CA.ArrayCommonMode(3.0, 2.0, "sum"), "#1f77b4", 1.7),
    ("自适应 RLS", CA.AdaptiveShapeRLS(SC.p, 0.99995), "#9467bd", 1.4),
]

# ============================ 图 1：总览与漂移形态
fig, axes = plt.subplots(3, 3, figsize=(16.5, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    I = DATA[name]; D = I["D"]; t = I["t"]; fs = I["fs"]
    Xn = I["Xn"]; w = I["w"]; j = I["j"]
    ax = axes[r, 0]
    ax.plot(t, Xn.sum(axis=1), lw=0.5, color="#1f77b4")
    ax.axvline(t[I["n_on"]], color="g", lw=1.1, label="在线检测的加载时刻")
    ax.axvspan(t[I["c0"]], t[I["d0"]], color="orange", alpha=0.14, label="负载段")
    ax.set_title(f"{name}  Σ受载通道")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Σ Force (N)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[r, 1]
    ax.plot(t, Xn[:, j], lw=0.55, color="tab:red")
    ax.axhline(I["REF"][j], color="k", ls="--", lw=1.2,
               label=f"1~3s 参考水平 {I['REF'][j]*1000:.0f}mN")
    late = Xn[I["d0"] - w(5):I["d0"], j].mean()
    ax.axhline(late, color="tab:purple", ls=":", lw=1.2,
               label=f"末5s {late*1000:.0f}mN (+{(late/I['REF'][j]-1)*100:.0f}%)")
    ax.axvspan(t[I["c0"]], t[I["d0"]], color="orange", alpha=0.14)
    ax.set_title(f"{name} 主通道 {D['ch'][j]}")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Force (N)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[r, 2]
    tau = t[I["c0"]:I["d0"]] - t[I["c0"]]
    ref = Xn[I["c0"] + w(1.0):I["c0"] + w(3.0), I["act"]].mean(axis=0)
    ok = np.abs(ref) > 1e-6
    acc = I["act"][ok]
    wts = np.abs(ref[ok]); wts = wts / wts.sum()
    hs = (Xn[I["c0"]:I["d0"], acc] / ref[ok][None, :]) @ wts
    ax.plot(tau, causal_smooth(hs, w(0.5)), lw=1.0, color="gray", label="实测形态")
    ax.plot(tau, SC(tau), lw=1.5, color="tab:red",
            label=f"冻结形态 c={SC.c:.3f},p={SC.p:.2f},τf={SC.tauf:.2f}s")
    ax.set_title(f"{name} 归一化漂移形态 h(τ)")
    ax.set_xlabel("负载持续 τ (s)"); ax.set_ylabel("h(τ)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("图 1  数据总览与漂移形态", fontsize=13)
savefig(fig, "F1_overview.png")

# ============================ 图 2：主通道因果补偿对比
fig, axes = plt.subplots(3, 2, figsize=(16, 11.5), constrained_layout=True)
for r, name in enumerate(DATASETS):
    I = DATA[name]; D = I["D"]; t = I["t"]; fs = I["fs"]
    Xn = I["Xn"]; w = I["w"]; j = I["j"]; n_on = I["n_on"]
    ax = axes[r, 0]
    for label, algo, col, lw in CAND:
        Y = run_stream(algo, Xn, t, fs, n_on)
        seg = slice(max(0, n_on - w(2)), min(I["d0"] + w(10), len(t)))
        ax.plot(t[seg], Y[seg, j], lw=lw, color=col, label=label)
    ax.axhline(I["REF"][j], color="k", ls="--", lw=1.3)
    ax.axvline(t[n_on], color="g", lw=1)
    ax.set_title(f"{name} 主通道 {D['ch'][j]}：各因果算法输出")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Force (N)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[r, 1]
    for label, algo, col, lw in CAND:
        Y = run_stream(algo, Xn, t, fs, n_on)
        tt = t[n_on:I["d0"]] - t[n_on]
        dev = (Y[n_on:I["d0"], j] - I["REF"][j]) * 1000
        ax.plot(tt, causal_smooth(dev, w(1.0)), lw=lw, color=col, label=label)
    ax.axhline(0, color="k", ls="--", lw=1.2)
    ax.set_title(f"{name} 输出相对 1~3s 参考水平的偏差（1s 平滑，mN）")
    ax.set_xlabel("负载持续 (s)"); ax.set_ylabel("Δ (mN)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
fig.suptitle("图 2  主通道：因果算法时域补偿效果", fontsize=13)
savefig(fig, "F2_causal_main.png")

# ============================ 图 3：阵列级热图
PICK = [("原始（不补偿）", CA.Raw()),
        ("因果 MA 1s", CA.CausalMA(1.0)),
        ("因果 Kalman q=1e-11", CA.CausalKalmanCV(1e-11, 1e-5)),
        ("形态先验+水平追踪 τs=3s", CA.ShapeDivide(SC, 3.0)),
        ("阵列共模因子", CA.ArrayCommonMode(3.0, 2.0, "sum")),
        ("自适应 RLS", CA.AdaptiveShapeRLS(SC.p, 0.99995))]
fig, axes = plt.subplots(3, len(PICK) + 1, figsize=(2.9 * (len(PICK) + 1), 9.5),
                         constrained_layout=True)
im = None
for r, name in enumerate(DATASETS):
    I = DATA[name]; Xn = I["Xn"]; t = I["t"]; fs = I["fs"]
    w = I["w"]; act = I["act"]; n_on = I["n_on"]
    for cidx, (label, algo) in enumerate(PICK):
        ax = axes[r, cidx]
        Y = run_stream(algo, Xn, t, fs, n_on)
        dev = (Y[I["d0"] - w(5):I["d0"]].mean(axis=0) - I["REF"]) * 1000
        im = ax.imshow(dev[None, :], cmap="coolwarm", vmin=-300, vmax=300, aspect="auto")
        ax.set_yticks([]); ax.set_xticks(range(0, 31, 5)); ax.tick_params(labelsize=6)
        ax.set_title(f"{label}\n中位 {np.median(dev[act]):+.0f}mN", fontsize=8)
    ax = axes[r, len(PICK)]
    ax.bar(np.arange(31), I["REF"] * 1000, color="tab:blue")
    ax.set_title(f"{name} 各通道 1~3s 参考水平 (mN)", fontsize=8)
    ax.set_xticks(range(0, 31, 5)); ax.tick_params(labelsize=6)
fig.colorbar(im, ax=axes[:, -1], shrink=0.45, label="末5s 相对 1~3s 的偏差 (mN)")
fig.suptitle("图 3  阵列级对比：负载末段各通道输出偏差（每行一组数据）", fontsize=13)
savefig(fig, "F3_array.png")

# ============================ 图 4：指标散点与汇总
rows = []
for name in DATASETS:
    for rr in SWEEP[name]["rows"]:
        rows.append(dict(name=name, **rr))
cmap = {"0. 参照": "k", "1. 通用因果滤波": "#7f7f7f",
        "2. 因果漂移模型": "#d62728", "3. 阵列共模因果补偿": "#1f77b4"}
fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
ax = axes[0, 0]
for g in cmap:
    rs = [r for r in rows if r["group"] == g]
    if not rs:
        continue
    ax.scatter([abs(r["drift_algo_mN"]) for r in rs], [r["gain"] for r in rs],
               s=44, color=cmap[g], label=f"{g}（{len(rs)//3}个配置×3组）", alpha=0.85)
    for r in rs:
        if abs(r["drift_algo_mN"]) < 60 and 0.9 < r["gain"] < 1.1:
            ax.annotate(r["algo"][:16], (abs(r["drift_algo_mN"]), r["gain"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("|残余时漂| (mN，越小越好)")
ax.set_ylabel("gain（1.0 = 完整保留真实力）")
ax.set_title("(a) 时漂抑制 与 真实力保留 的权衡")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[0, 1]
for g in cmap:
    rs = [r for r in rows if r["group"] == g]
    if not rs:
        continue
    ax.scatter([r["cv_algo"] * 100 for r in rs], [abs(r["drift_algo_mN"]) for r in rs],
               s=44, color=cmap[g], label=g, alpha=0.85)
ax.axvline(SWEEP["数据1"]["rows"][0]["cv_raw"] * 100, color="k", ls="--", lw=1,
           label="原始 CV")
ax.set_xlabel("平台期 CV (%)")
ax.set_ylabel("|残余时漂| (mN)")
ax.set_title("(b) 持载全程读数恒定度 与 末端残余")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[1, 0]
gl = [g for g in cmap if any(r["group"] == g for r in rows)]
data = [[abs(r["drift_algo_mN"]) for r in rows if r["group"] == g] for g in gl]
bp = ax.boxplot(data, tick_labels=[g.split(". ")[1] for g in gl], patch_artist=True)
for p, g in zip(bp["boxes"], gl):
    p.set_facecolor(cmap[g]); p.set_alpha(0.55)
ax.set_ylabel("|残余时漂| (mN)")
ax.set_title("(c) 各类算法的残余时漂分布")
ax.grid(alpha=0.3, axis="y")

ax = axes[1, 1]
I = DATA["数据1"]; Xn = I["Xn"]; t = I["t"]; fs = I["fs"]; w = I["w"]
for label, algo, col, lw in CAND[:6]:
    Y = run_stream(algo, Xn, t, fs, I["n_on"])
    dev = (Y[I["n_on"]:I["d0"], I["j"]] - I["REF"][I["j"]]) * 1000
    ax.plot(t[I["n_on"]:I["d0"]] - t[I["n_on"]], causal_smooth(dev, w(2.0)),
            lw=lw, color=col, label=label)
ax.axhline(0, color="k", ls="--", lw=1.2)
ax.set_xlabel("负载持续 (s)"); ax.set_ylabel("输出偏差 (mN)")
ax.set_title(f"(d) 数据1 主通道 {DATA['数据1']['D']['ch'][I['j']]}：偏差随时间（2s 平滑）")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle("图 4  因果算法横向指标对比", fontsize=13)
savefig(fig, "F4_metrics.png")

# ============================ 图 5：因果性验证 + 在线检测 + 敏感性
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
I = DATA["数据1"]; Xn = I["Xn"]; t = I["t"]; fs = I["fs"]; w = I["w"]
ax = axes[0]
algo = CA.ShapeDivide(SC, 3.0)
for frac, ls in ((0.30, "--"), (0.55, ":"), (0.80, "-."), (1.0, "-")):
    m = int(len(t) * frac)
    Y = run_stream(algo, Xn[:m], t[:m], fs, I["n_on"])
    ax.plot(t[:m] - t[I["n_on"]], Y[:, I["j"]], ls=ls, lw=1.2,
            label=f"仅用前 {frac*100:.0f}% 数据")
ax.axhline(I["REF"][I["j"]], color="k", ls="--", lw=1)
ax.set_xlim(-2, 70)
ax.set_title("因果性校验：不同截断长度输出完全重合")
ax.set_xlabel("负载持续 (s)"); ax.set_ylabel("Force (N)"); ax.grid(alpha=0.3)
ax.legend(fontsize=7)

ax = axes[1]
tot = Xn.sum(axis=1)
seg = slice(max(0, I["n_on"] - w(1.5)), I["n_on"] + w(3))
ax.plot(t[seg] - t[I["on_true"]], tot[seg], lw=1.0, color="tab:blue")
ax.axvline(0, color="k", lw=1.2, label="离线标注加载沿")
ax.axvline(t[I["n_on"]] - t[I["on_true"]], color="g", lw=1.2,
           label=f"在线检测（偏差 {(I['n_on']-I['on_true'])/fs*1000:+.0f}ms）")
ax.axhline(0, color="gray", lw=0.6)
ax.set_title("在线加载检测（仅用历史）")
ax.set_xlabel("相对真实加载沿 (s)"); ax.set_ylabel("Σ Force (N)")
ax.grid(alpha=0.3); ax.legend(fontsize=7)

ax = axes[2]
try:
    with open(os.path.join(RES, "C2_onset_sensitivity.json"), encoding="utf-8") as f:
        ons = json.load(f)
    xs = sorted(int(k) for k in ons)
    for lbl in ons[str(xs[0])]:
        ax.plot(xs, [ons[str(x)][lbl]["drift"] for x in xs], "o-", ms=4, lw=1.2,
                label=lbl.split("(")[0][:16])
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("加载时刻偏差 (ms)")
    ax.set_ylabel("残余时漂 (mN)")
    ax.set_title("对加载时刻检测误差的敏感性（数据1）")
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
except Exception as e:
    ax.text(0.5, 0.5, f"敏感性数据缺失\n{e}", ha="center", transform=ax.transAxes)
fig.suptitle("图 5  因果性保证、在线加载检测与鲁棒性", fontsize=13)
savefig(fig, "F5_causality.png")

print("[ok] C3 图组完成")
