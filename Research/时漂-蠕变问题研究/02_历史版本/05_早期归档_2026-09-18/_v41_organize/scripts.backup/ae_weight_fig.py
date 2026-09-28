# -*- coding: utf-8 -*-
"""「10N 保压后加 5N」场景图：合成扫描 + 实测录制对照。

figures/weight_scenario.png
 ① 合成：不同 ADC 台阶比例下，加码后显示的走向（归一化到 10N）
 ② 合成：末端显示 / 加码前显示  vs  ADC 台阶比例（门限 0.29）
 ③ 实测 13ffca @27.66s（比值 0.24 被漏检）显示被拉回
 ④ 实测 13ffca @18.68s（比值 0.40 被识别）显示正确跟随
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.facecolor"] = "white"
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402

REC0 = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
        r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
        r"\device_001_seg000.csv")

# ---- 蠕变律（实测拟合）----
d0 = L.prep(REC0)
a, b = int(133.84 / d0["dtm"]), int(182.0 / d0["dtm"])
uu = d0["tu"][a:b] - d0["tu"][a]
yy = d0["tot"][a:b] / d0["tot"][a + int(5.0 / d0["dtm"])] - 1.0
fit = lambda u2, a1, t1, a2, t2: a1 * (1 - np.exp(-u2 / t1)) + a2 * (1 - np.exp(-u2 / t2))
P, _ = optimize.curve_fit(fit, uu, yy, p0=[0.03, 3.0, 0.05, 60.0],
                          bounds=([0, 0.2, 0, 5], [1, 60, 1, 3000]))
cr = lambda x: np.where(x > 0, fit(np.maximum(x, 0), *P), 0.0)

FS, DUR = 100.0, 380.0
tt = np.arange(0.0, DUR, 1 / FS)
L10, T_LOAD, T_ADD = 10000.0, 20.0, 300.0
RATIOS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def synth(ratio):
    s = np.zeros_like(tt)
    m1 = tt > T_LOAD
    s[m1] += L10 * (1 + cr(tt[m1] - T_LOAD))
    m2 = tt > T_ADD
    s[m2] += ratio * L10 * (1 + cr(tt[m2] - T_ADD))
    return s


def run(sig):
    c = GLM53v3(1)
    Y = np.empty(len(tt))
    for i in range(len(tt)):
        Y[i] = c.process(tt[i], np.array([sig[i]]))[0]
    return Y


iA = int(T_ADD * FS)


def verdict(end, pre):
    r = end / pre
    if r < 1.10:
        return f"漏检→被拉回 {r:.2f}×10N"
    if r <= 1.55:
        return f"识别→正常 {r:.2f}×10N"
    return f"识别→偏高 {r:.2f}×10N"

# ---------------- 图 ----------------
fig = plt.figure(figsize=(19, 11), constrained_layout=True)
gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.20)

# ① 归一化轨迹
ax = fig.add_subplot(gs[0, 0])
res = {}
for r in RATIOS:
    sig = synth(r)
    Y = run(sig)
    pre = Y[iA - 200:iA].mean()
    res[r] = (sig, Y, pre)
    sl = slice(int(295 / (1 / FS)), int(365 / (1 / FS)))
    mark = "漏检" if verdict(Y[-int(30 / (1 / FS)):].mean(), pre).startswith("漏检") else "识别"
    ax.plot(tt[sl] - T_ADD, Y[sl] / pre, lw=1.6 if mark != "漏检" else 2.2,
            ls="-" if mark != "漏检" else "--",
            color=plt.cm.viridis(RATIOS.index(r) / (len(RATIOS) - 1)),
            label=f"ADC 台阶 {r:.0%}×10N → {verdict(Y[-int(30/(1/FS)):].mean(), pre)}")
sig0 = res[RATIOS[0]][0]
sl = slice(int(295 / (1 / FS)), int(365 / (1 / FS)))
ax.plot(tt[sl] - T_ADD, sig0[sl] / res[RATIOS[0]][2], color="#9e9e9e", lw=1.0, ls=":",
        label="原始读数（无补偿，参考）")
ax.axhline(1.5, color="k", lw=1.0, ls="--", alpha=0.6)
ax.text(62, 1.52, "真值 15N（=1.5×10N）", fontsize=10, ha="right", color="#333333")
ax.axvline(0, color="#d62728", lw=1.2)
ax.text(0.6, 1.02, "加 5N", color="#a01010", fontsize=11)
ax.set_xlim(-5, 65)
ax.set_ylim(0.9, 1.85)
ax.set_xlabel("加码后时间 (s)", fontsize=11)
ax.set_ylabel("显示值 ÷ 加码前显示（10N 基准）", fontsize=11)
ax.set_title("① 合成场景：10N 保压 280s → 叠加 5N（真值 15N）的显示走向", fontsize=12.5, loc="left")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.15)

# ② 末端 vs 比值
ax = fig.add_subplot(gs[0, 1])
xs = np.linspace(0.05, 1.0, 40)
endv, trigv = [], []
for r in xs:
    sig = synth(r)
    Y = run(sig)
    pre = Y[iA - 200:iA].mean()
    endv.append(Y[-int(30 / (1 / FS)):].mean() / pre)
ax.plot(xs, endv, "-", color="#1f77b4", lw=2.4, label="v3 末端显示 ÷ 加码前显示")
ax.axhline(1.5, color="k", ls="--", lw=1.4, label="理想值 1.5（真值 15N）")
ax.axvline(0.29, color="#d62728", ls="--", lw=1.6, label="有效识别门限 ≈0.29")
ax.axvspan(0.05, 0.29, color="#d62728", alpha=0.08)
ax.text(0.16, 1.20, "漏检区：\n+5N 被当成蠕变\n显示被拉回 ≈10N", fontsize=11.5, ha="center",
        color="#a01010")
ax.text(0.62, 1.20, "识别区：台阶被透传\n（略高于 15N，因旧载蠕变被重置为基线）",
        fontsize=11, ha="center", color="#1a5a1a")
ax.set_xlim(0.05, 1.0)
ax.set_ylim(0.95, 1.75)
ax.set_xlabel("ADC 台阶 ÷ 10N 读数（力上固定为 +5N）", fontsize=11)
ax.set_ylabel("末端显示 ÷ 加码前显示", fontsize=11)
ax.set_title("② 加 5N 后稳定值 vs 台阶相对大小（力不变、只变 ADC 占比）", fontsize=12.5, loc="left")
ax.legend(fontsize=9.5, loc="lower right")
ax.grid(alpha=0.15)

# ③④ 实测对照
z = np.load(os.path.join(RES, "rec_13ffca.npz"))
tu, Xu = z["tu"], z["Xu"]
Ys = {k[2:]: z[k] for k in z.files if k.startswith("Y_")}
dtm = tu[1] - tu[0]
tot = Xu.sum(axis=1)
sm = lambda x: pd.Series(x).rolling(int(0.5 / dtm), center=True, min_periods=1).median().to_numpy()
raw_s = sm(tot)
COL = {"v3": "#1f77b4", "v4_fast5": "#d62728"}
for j, (lo, hi, t_ev, ratio, tag) in enumerate(
        [(22.0, 40.0, 27.66, 0.240, "漏检"), (13.0, 32.0, 18.68, 0.398, "识别")]):
    ax = fig.add_subplot(gs[1, j])
    sl = slice(int(lo / dtm), int(hi / dtm))
    ax.plot(tu[sl], raw_s[sl], color="#9e9e9e", lw=1.4, label="原始（无补偿）")
    for k in ("v3", "v4_fast5"):
        ax.plot(tu[sl], sm(Ys[k].sum(axis=1))[sl], color=COL[k], lw=2.0,
                label=L.LBL[k] if k == "v3" else "v4 免责5s")
    ax.axvline(t_ev, color="#d62728", ls="--", lw=1.3)
    pre_i = int((t_ev - 1.5) / dtm)
    end_i = int((t_ev + 8) / dtm)
    d_raw = raw_s[end_i] - raw_s[pre_i]
    d_v3 = sm(Ys["v3"].sum(axis=1))[end_i] - sm(Ys["v3"].sum(axis=1))[pre_i]
    ax.annotate(f"台阶 +{d_raw:.0f}（比值 {ratio:.2f}，{tag}）\n"
                f"显示只走了 +{d_v3:.0f}（{100*d_v3/d_raw:.0f}%）",
                xy=(t_ev, raw_s[pre_i]), xytext=(lo + 1.0, raw_s[sl].min() + 0.06 * (raw_s[sl].max() - raw_s[sl].min())),
                fontsize=10.5, color="#a01010" if tag == "漏检" else "#1a5a1a",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff0f0" if tag == "漏检" else "#f0fff0",
                          ec="#d62728" if tag == "漏检" else "#2ca02c", lw=0.9),
                arrowprops=dict(arrowstyle="->", lw=1.2,
                                color="#d62728" if tag == "漏检" else "#2ca02c"))
    ax.set_xlim(lo, hi)
    ax.set_xlabel("时间 (s)", fontsize=10.5)
    ax.set_ylabel("整阵显示总量 (ADC)", fontsize=10.5)
    ax.set_title(f"③{'ab'[j]} 实测 13ffca @{t_ev:.1f}s：{tag} → "
                 + ("显示被拉回（只吃到台阶的一小部分）" if tag == "漏检" else "显示正确跟随台阶"),
                 fontsize=12, loc="left")
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(alpha=0.15)

fig.suptitle("为什么「10N 稳定后加 5N」可能被拉回 10N：变载识别的相对门限 ≈29% 电平"
             f"（蠕变律取自实测：{P[1]:.0f}s/{100*P[0]:.1f}% + {P[3]:.0f}s/{100*P[2]:.1f}%）",
             fontsize=15.5)
fig.savefig(os.path.join(FIG, "weight_scenario.png"), dpi=118)
plt.close(fig)
print("saved: figures/weight_scenario.png")

print("\n合成扫描表：")
print(f"  {'ADC台阶比例':>12}{'识别':>6}{'加码前显示':>11}{'末端显示':>10}{'末端/加码前':>12}")
for r in RATIOS:
    sig, Y, pre = res[r]
    end = Y[-int(30 / (1 / FS)):].mean()
    hit = abs(end / pre - 1.5) < 0.10
    print(f"  {r:12.2f}{('是' if hit else '否'):>6}{pre:11.0f}{end:10.0f}{end/pre:12.3f}")
