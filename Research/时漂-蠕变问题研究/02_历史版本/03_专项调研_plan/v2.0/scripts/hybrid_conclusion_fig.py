# -*- coding: utf-8 -*-
"""折中方案 ②′ 出图：机制图 + 三臂对比 + 一致性 + 代价。

图内容（一次看明白为什么"去涨落"会坏事）：
  ① 卸载沿处**同一个包**内含两个真实采样（17307 与 3495）⇒ 去涨落后造出 10400 这个
     **从未存在过**的电平，被算法当成一次真实加载 ⇒ 显示偏移 +6896 ADC（实测）
  ② 振荡工况：三臂显示输出与偏移
  ③ ★反复同一荷载一致性：inc(τ) 随 τ 的极差
  ④ 代价：干净数据的全程偏移极差
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.abspath(os.path.join(SCRIPTS, "build", "batch_runner_exp.exe"))
ODD = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                   "20260919_134056_single_device_f40a1b")
OUT = os.path.abspath(os.path.join(SCRIPTS, "..", "results"))
FIG = os.path.join(OUT, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _n in ("Microsoft YaHei", "SimHei", "DengXian"):
    if any(f.name == _n for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.sans-serif"] = [_n]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break


def T(zh, en):
    return zh if matplotlib.rcParams["font.sans-serif"][0] in ("Microsoft YaHei", "SimHei", "DengXian") else en


def feed(el, V, mode):
    n = V.shape[1]
    lines = ["%d 100" % n]
    for i in range(len(el)):
        lines.append("%.6f " % el[i] + " ".join("%.6f" % x for x in V[i]))
    r = subprocess.run([RUNNER, mode, "median"], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8")
    rows = r.stdout.splitlines()[1:-1]
    si = np.array([float(x.split()[1]) for x in rows])
    so = np.array([float(x.split()[2]) for x in rows])
    return si, so


def main():
    preN = L.load_stream(L.DS_ZERO, "device_001_pre_seg0.csv")
    elN, VN = preN["el"], preN["V"]
    tpN = VN.sum(1)
    preO = L.load_stream(ODD, "device_001_pre_seg0.csv")
    elO, VO = preO["el"], preO["V"]
    tpO = VO.sum(1)

    armsN = {m: feed(elN, VN, m) for m in ("raw", "packavg", "hybrid")}
    armsO = {m: feed(elO, VO, m) for m in ("raw", "packavg", "hybrid")}

    fig = plt.figure(figsize=(18, 12), dpi=110)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.15, 1.05],
                          hspace=0.45, wspace=0.22)

    # ① 机制：卸载沿处一个包跨两态
    ax = fig.add_subplot(gs[0, 0])
    a, b = 110.9, 111.35
    m = (elN >= a) & (elN < b)
    ax.plot(elN[m], tpN[m], "o-", color="#9aa0a6", ms=4, lw=1.0,
            label=T("输入逐帧（每个点=1 帧）", "input frames"))
    for mm, col in (("packavg", "#1a73e8"), ("hybrid", "#188038")):
        ax.plot(elN[m], armsN[mm][0][m], "s--", color=col, ms=4, lw=0.9,
                label=T("%s 实际喂给算法的值" % mm, "%s fed value" % mm))
    ax.axhline(17307, color="#d93025", lw=0.8, ls=":")
    ax.axhline(3495, color="#d93025", lw=0.8, ls=":")
    ax.annotate(T("同一个包(t=111.061)里同时含\n满载 17307 与空载 3495 两个真实采样\n"
                  "去涨落造出 10400 —— 从未存在过的电平\n被算法当成一次真实加载 ⇒ 偏移 +6896 ADC",
                  "One packet spans the unload edge"),
                xy=(111.061, 10400), xytext=(111.10, 11500), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#d93025"),
                bbox=dict(fc="#fce8e6", ec="#d93025", lw=0.8, boxstyle="round,pad=0.35"))
    ax.set_title(T("① 机制：为什么「包内去涨落」会在真实卸载沿上坏事",
                   "① Mechanism: de-rippling across a real unload edge"), fontsize=10.5)
    ax.set_ylabel(T("总量 (ADC)", "sum (ADC)"), fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="center right")

    # ② 干净数据的偏移（代价）
    ax = fig.add_subplot(gs[0, 1])
    for mm, col in (("raw", "#d93025"), ("packavg", "#1a73e8"), ("hybrid", "#188038")):
        off = armsN[mm][1] - armsN[mm][0]
        ax.plot(elN, off, color=col, lw=0.8, label=T("%s（极差 %.0f ADC）" % (mm, off.max() - off.min()), mm))
    ax.axhline(0, color="#5f6368", lw=0.8, ls=":")
    ax.set_title(T("② 代价：干净数据（新录制 337 s）的显示偏移\n尖峰只出现在真实卸载沿、且会自回落",
                   "② Cost on clean data: spikes at real unload edges"), fontsize=10.5)
    ax.set_xlabel(T("时间 (s)", "time (s)"), fontsize=9)
    ax.set_ylabel(T("显示 − 输入 (ADC)", "display - input (ADC)"), fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)

    # ③ 振荡工况
    ax = fig.add_subplot(gs[1, 0])
    a, b = 70.0, 76.0
    m = (elO >= a) & (elO < b)
    ax.plot(elO[m], tpO[m], color="#9aa0a6", lw=0.6, label=T("输入", "input"))
    for mm, col in (("raw", "#d93025"), ("packavg", "#1a73e8"), ("hybrid", "#188038")):
        ax.plot(elO[m], armsO[mm][1][m], color=col, lw=1.2, label=mm)
    ax.set_title(T("③ 振荡工况 70~76 s：显示输出（去涨落反而把偏移抖得更厉害：std 686→1055）",
                   "③ Oscillating case: de-rippling makes offset worse"), fontsize=10.5)
    ax.set_xlabel(T("时间 (s)", "time (s)"), fontsize=9)
    ax.set_ylabel(T("显示总量 (ADC)", "display (ADC)"), fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, ncol=2)

    # ④ 一致性
    ax = fig.add_subplot(gs[1, 1])
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tpN, 3), np.percentile(tpN, 97)
    ups, _ = L.edges_from_tot(np.convolve(tpN, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in ups:
        if not tu or elN[i] - tu[-1] > 1.0:
            tu.append(float(elN[i]))
    taus = [1.0, 2.0, 5.0, 10.0]
    res = {}
    for mm in ("raw", "packavg", "hybrid"):
        si, so = armsN[mm]
        vals = []
        for tau in taus:
            iv = []
            for t in tu:
                m0 = (elN >= t - 1.2) & (elN < t - 0.35)
                m1 = (elN >= t + tau - 0.15) & (elN < t + tau + 0.15)
                if m0.sum() < 5 or m1.sum() < 1:
                    continue
                iv.append(float(np.median(so[m1] - tpN[m1])) - float(np.median(so[m0] - tpN[m0])))
            vals.append(max(iv) - min(iv) if len(iv) > 1 else np.nan)
        res[mm] = vals
    w = 0.26
    xs = np.arange(len(taus))
    for i, (mm, col) in enumerate((("raw", "#d93025"), ("packavg", "#1a73e8"), ("hybrid", "#188038"))):
        ax.bar(xs + (i - 1) * w, res[mm], width=w, color=col, label=mm)
        for x, v in zip(xs + (i - 1) * w, res[mm]):
            if v == v:
                ax.text(x, v, "%.0f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([T("沿后 %g s" % t, "t+%gs" % t) for t in taus])
    ax.set_ylabel(T("各沿 inc(τ) 的极差 (ADC)", "range of inc(tau) (ADC)"), fontsize=9)
    ax.set_title(T("④ ★反复同一荷载一致性（越矮越好）：@1 s 的差异全部来自一个 A=228 的无效沿；@10 s 去涨落明显更优",
                   "④ Repeat-load consistency (lower = better)"), fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=8)

    # ⑤ 汇总表
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    rows = [
        [T("指标", "metric"), "raw", "packavg", "hybrid ②′"],
        [T("振荡工况 偏移 std（越小越好）", "osc offset std"), "686", "1096", "1055"],
        [T("振荡工况 安静段偏移中位", "quiet-segment bias"), "-666", "-667", "-671"],
        [T("★一致性 极差@1 s / @2 s / @5 s / @10 s（越小越好）", "consistency ranges"),
         "40 / 97 / 289 / 379", "18 / 93 / 290 / 93", "21 / 97 / 293 / 87"],
        [T("干净数据 显示偏移极差（越小越好）", "clean-data offset range"), "820", "10233", "3040"],
        [T("干净数据 Δ1s / G", "clean Δ1s / G"), "64 / 0.895", "76 / 0.895", "65 / 0.895"],
        [T("去涨落生效量", "de-ripple applied"), "—", T("全部 13160 帧", "all frames"),
         T("986 帧 / 266 包", "986 frames")],
    ]
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.7)
    for j in range(4):
        tb[0, j].set_facecolor("#e8f0fe")
    ax.set_title(T("汇总：结论 = 去涨落在真实卸载沿上会造出不存在的电平，因此 raw 仍是最稳的一致选择",
                   "Summary"), fontsize=11)

    fig.suptitle(T("折中方案 ②′ 离线验证结论：raw / packavg(始终去涨落) / hybrid(仅抖动时去涨落)",
                   "Hybrid offline A/B conclusion"), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    p = os.path.join(FIG, "fig_hybrid_conclusion.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
