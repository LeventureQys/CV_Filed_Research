# -*- coding: utf-8 -*-
"""plan-v2.0 离线验证（振荡工况优化候选）：三臂 A/B + 出图。

臂：
  raw      现役交付算法（逐帧原始喂入）                          —— 基线
  packavg  同一 elapsed 的 1~4 帧先取平均再喂                    —— 候选②
  jitter   检出抖动时冻结补偿器锚定、显示退回输入的滑动中位等比平移 —— 候选③
            （实现见 scripts/batch_runner_exp.cpp，模式判据：0.3 s 窗内 ±1 过零 ≥6 且极差 ≥800 ADC）

评据：
  ① 抗抖：振荡段 显示偏移 std（越小越好）、输入 std 作对照
  ② 无偏：**安静段偏移中位**（是否被前段振荡带偏）——这是"基线有没有被搞坏"的判据
  ③ 干净数据回归：新录制 337 s / 8 个加载沿的 Δ1s（沿后 0.8~1.2 s 显示−输入）与 G（台阶捕获比）
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.abspath(os.path.join(SCRIPTS, "build", "batch_runner.exe"))
RUNNER_EXP = os.path.abspath(os.path.join(SCRIPTS, "build", "batch_runner_exp.exe"))
ODD = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                   "20260919_134056_single_device_f40a1b")
OUT = os.path.abspath(os.path.join(SCRIPTS, "..", "results"))
FIG = os.path.join(OUT, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def pick_font():
    for name in ("Microsoft YaHei", "SimHei", "DengXian"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK = pick_font()


def T(zh, en):
    return zh if CJK else en


def feed(runner, el, V, mode=None):
    n = V.shape[1]
    lines = ["%d 100" % n]
    for i in range(len(el)):
        lines.append("%.6f " % el[i] + " ".join("%.6f" % x for x in V[i]))
    argv = [runner] + ([mode] if mode else [])
    r = subprocess.run(argv, input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:400])
    rows = r.stdout.splitlines()
    body = rows[1:-1]
    si = np.array([float(x.split()[1]) for x in body])
    so = np.array([float(x.split()[2]) for x in body])
    c = [float(x) for x in rows[-1].split()[1:]]
    return si, so, c


def packavg(el, V):
    keys = np.round(el, 3)
    bounds = np.flatnonzero(np.concatenate([[True], np.diff(keys) != 0]))
    ends = np.append(bounds[1:], len(el))
    elB = el[bounds]
    VB = np.vstack([V[a:b].mean(0) for a, b in zip(bounds, ends)])
    return elB, VB


def windows(el):
    return [(5, 8), (27, 36), (55, 58), (69.7, 71.7), (72, 74), (79.6, 89.6),
            (108, 118), (120, 125)]


def main():
    os.makedirs(FIG, exist_ok=True)
    pre = L.load_stream(ODD, "device_001_pre_seg0.csv")
    el, V = pre["el"], pre["V"]
    tp = V.sum(1)
    elB, VB = packavg(el, V)

    arms = {}
    si, so, c = feed(RUNNER, el, V)
    arms["raw"] = (el, si, so, c)
    siB, soB, cB = feed(RUNNER, elB, VB)
    # 映射回原帧位置便于同位置比较
    arms["packavg"] = (el, np.interp(el, elB, siB), np.interp(el, elB, soB), cB)
    siJ, soJ, cJ = feed(RUNNER_EXP, el, V, "jitter")
    arms["jitter"] = (el, siJ, soJ, cJ)

    print("=== ① 内部量（振荡数据）===")
    print("%-9s %10s %10s %10s %10s %10s" %
          ("臂", "shape_hits", "n_clamp", "g_floor", "冻结帧", "进入冻结次数"))
    for k, (_e, _si, _so, c) in arms.items():
        extra = (int(c[10]), int(c[11])) if len(c) > 11 else (0, 0)
        print("%-9s %10d %10d %10d %10d %10d" %
              (k, int(c[0]), int(c[5]), int(c[3]), extra[0], extra[1]))

    print("\n=== ② 显示偏移（显示−输入）按区间 ===")
    print("%-16s %10s %12s %12s %12s" % ("区间", "输入std", "raw", "packavg", "jitter"))
    for a, b in windows(el):
        m = (el >= a) & (el < b)
        if not m.any():
            continue
        vals = []
        for k in ("raw", "packavg", "jitter"):
            _e, si_, so_, _c = arms[k]
            vals.append(np.median(so_[m] - si_[m]))
        print("%-16s %10.0f %12.0f %12.0f %12.0f"
              % ("%g~%gs" % (a, b), tp[m].std(), vals[0], vals[1], vals[2]))

    print("\n=== ③ 抗抖与无偏 ===")
    print("%-9s %14s %16s %18s %16s" %
          ("臂", "振荡段偏移std", "振荡段偏移极差", "安静段偏移中位", "全程位移范围"))
    for k in ("raw", "packavg", "jitter"):
        _e, si_, so_, _c = arms[k]
        off = so_ - si_
        mo = (el >= 69.7) & (el < 71.7)
        mq = (el >= 108) & (el < 118)
        print("%-9s %14.0f %16.0f %18.0f %16.0f"
              % (k, off[mo].std(), off[mo].max() - off[mo].min(),
                 np.median(off[mq]), off.max() - off.min()))

    # ── 干净数据回归（新录制）──
    pre2 = L.load_stream(L.DS_ZERO, "device_001_pre_seg0.csv")
    el2, V2 = pre2["el"], pre2["V"]
    tp2 = V2.sum(1)
    el2B, V2B = packavg(el2, V2)
    si2, so2, _ = feed(RUNNER, el2, V2)
    si2B, so2B, _ = feed(RUNNER, el2B, V2B)
    si2J, so2J, c2J = feed(RUNNER_EXP, el2, V2, "jitter")
    A2 = {"raw": (el2, si2, so2), "packavg": (el2, si2, so2),
          "jitter": (el2, si2J, so2J)}
    A2["packavg"] = (el2, np.interp(el2, el2B, si2B), np.interp(el2, el2B, so2B))

    k9 = np.ones(9) / 9.0
    lo, hi = np.percentile(tp2, 3), np.percentile(tp2, 97)
    ups, _ = L.edges_from_tot(np.convolve(tp2, k9, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in ups:
        if not tu or el2[i] - tu[-1] > 1.0:
            tu.append(float(el2[i]))

    print("\n=== ④ 干净数据（新录制 337 s / %d 个加载沿）回归 ===" % len(tu))
    print("%-9s %12s %12s %12s %12s" % ("臂", "Δ1s中位", "G中位", "受载段偏移中位", "全程极差"))
    clean = {}
    for k in ("raw", "packavg", "jitter"):
        _e, si_, so_ = A2[k][0], A2[k][1], A2[k][2]
        d1s, gs = [], []
        for t in tu:
            m0 = (el2 >= t - 1.2) & (el2 < t - 0.35)
            if m0.sum() < 5:
                continue
            bp = float(np.median(tp2[m0]))
            te = min(t + 30.0, el2[-1] - 0.2)
            m1 = (el2 >= te - 1.5) & (el2 <= te)
            A_ = float(np.median(tp2[m1])) - bp
            m2 = (el2 >= t + 0.8) & (el2 <= t + 1.2)
            d1s.append(float(np.median(so_[m2] - tp2[m2])) if m2.any() else np.nan)
            m3 = (el2 >= t + 4.0) & (el2 <= t + 5.0)
            if m3.any() and abs(A_) > 500:
                gs.append((float(np.median(so_[m3])) - float(np.median(so_[m0]))) / A_)
        loaded = tp2 > np.percentile(tp2, 60)
        off = so_ - si_
        clean[k] = (np.nanmedian(d1s), np.nanmedian(gs), np.median(off[loaded]),
                    off.max() - off.min())
        print("%-9s %12.0f %12.3f %12.0f %12.0f"
              % (k, clean[k][0], clean[k][1], clean[k][2], clean[k][3]))

    np.save(os.path.join(OUT, "osc_arms.npy"),
            {"el": el, "tp": tp,
             "raw": arms["raw"][2], "packavg": arms["packavg"][2],
             "jitter": arms["jitter"][2]}, allow_pickle=True)
    make_fig(el, tp, arms)
    return arms, clean


def make_fig(el, tp, arms):
    a, b = 66.0, 84.0
    m = (el >= a) & (el < b)
    t = el[m]
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), dpi=110, sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1.3, 1.1]})
    ax = axes[0]
    ax.plot(t, tp[m], color="#9aa0a6", lw=0.6, label=T("输入逐帧", "input per-frame"))
    ax.plot(t, arms["raw"][2][m], color="#d93025", lw=1.3, label=T("raw 现役", "raw"))
    ax.plot(t, arms["packavg"][2][m], color="#1a73e8", lw=1.1, label=T("packavg ②包内平均", "packavg"))
    ax.plot(t, arms["jitter"][2][m], color="#188038", lw=1.1, label=T("jitter ③抖动冻结", "jitter-freeze"))
    ax.set_ylabel(T("显示总量 (ADC)", "display sum (ADC)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.95)
    ax.set_title(T("离线验证 A/B · 随机震荡工况 66~84 s：三臂显示输出对比",
                   "Offline A/B - oscillating condition 66~84 s"), fontsize=12)

    ax = axes[1]
    for k, col in (("raw", "#d93025"), ("packavg", "#1a73e8"), ("jitter", "#188038")):
        _e, si_, so_ = arms[k][0], arms[k][1], arms[k][2]
        off = so_ - si_
        kk = 20
        ax.plot(t, np.convolve(off, np.ones(kk) / kk, mode="same")[m], color=col, lw=1.5,
                label=T("%s 偏移(0.2 s 平滑)" % k, "%s offset" % k))
    ax.axhline(0, color="#5f6368", lw=0.8, ls=":")
    ax.set_ylabel(T("显示 − 输入 (ADC)", "display - input (ADC)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    ax = axes[2]
    win = 30
    for k, col in (("raw", "#d93025"), ("packavg", "#1a73e8"), ("jitter", "#188038")):
        _e, si_, so_ = arms[k][0], arms[k][1], arms[k][2]
        off = so_ - si_
        roll = np.array([off[max(0, i - win):i + win].std() for i in range(len(off))])
        ax.plot(t, roll[m], color=col, lw=1.2, label=T("%s 偏移滚动std" % k, k))
    ax.set_xlabel(T("时间 (s)", "time (s)"), fontsize=10)
    ax.set_ylabel(T("偏移 std (ADC)", "offset std (ADC)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.set_title(T("偏移的抖动强度（越小越稳）", "offset jitter (lower = steadier)"), fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIG, "fig_oscillation_ab.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
