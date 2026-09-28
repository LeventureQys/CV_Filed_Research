# -*- coding: utf-8 -*-
"""v3.4 约定对比图：同一负载反复增减时，两种显示约定各自意味着什么。

约定 A（当前状态参考）：每次加载后显示 = 本次加载的快相落点（输入在沿后
  [2,4]s 的中位），保压期钉平 —— "这个传感器现在压这个负载就是这么多"。
约定 B（新鲜参考）：同一负载永远显示第一次加载时的落点 —— "没受过载历史的
  传感器压这个负载是这么多"；粘弹性残余（零点漂移+快相落点上移）全部扣除。

数据：working/零基线-反复增减同一负载 7b3977（pre 流为输入）。
输出：../figure/v34_约定对比_A当前状态参考_vs_B新鲜参考.png
"""
import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "build", "v34_runner.exe")
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")
OUT = os.path.join(HERE, "..", "figure",
                   "v34_约定对比_A当前状态参考_vs_B新鲜参考.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# (类别, 沿完成时刻, 段末)——沿完成 = 楼梯最后一级 + ~1s
HOLDS = [
    ("满载", 17.0, 35.5), ("半载", 38.0, 40.5),
    ("满载", 43.0, 55.5), ("半载", 58.0, 62.0),
    ("满载", 64.5, 215.0), ("满载", 234.0, 238.5),
    ("满载", 247.0, 250.5), ("半载", 253.3, 254.4),
    ("满载", 263.0, 274.0), ("满载", 279.5, 281.2),
    ("半载", 283.0, 288.0), ("满载", 290.5, 294.2),
    ("半载", 295.5, 296.2), ("满载", 304.5, 307.5),
]


def replay(extra):
    pre = L.load_stream(DS, "device_001_pre_seg0.csv")
    n = pre["V"].shape[1]
    lines = [str(n)]
    for i in range(pre["n"]):
        lines.append("%.6f " % pre["ts"][i] +
                     " ".join("%.1f" % x for x in pre["V"][i]))
    p = subprocess.run([RUNNER] + list(extra), input="\n".join(lines),
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=HERE)
    rows = []
    for ln in p.stdout.splitlines():
        f = ln.split()
        if not f or f[0] in ("t", "OK", "END", "CH"):
            continue
        try:
            rows.append([float(x) for x in f])
        except ValueError:
            pass
    arr = np.array(rows)
    return pre["el"], arr[:, 1], arr[:, 2]


def build_ideal(el, sin, levels):
    """levels: [(t_settle, t_end, level)] → 分段平坦、1s 斜坡过渡的理想显示。"""
    y = np.zeros_like(sin)
    y[:] = np.nan
    prev_end_level = None
    for t0, t1, lv in levels:
        i0 = int(np.searchsorted(el, t0 - 1.0))
        i1 = int(np.searchsorted(el, t0))
        i2 = int(np.searchsorted(el, t1))
        y[i1:i2] = lv
        if prev_end_level is not None and i1 > i0:
            w = np.linspace(0, 1, i1 - i0)
            y[i0:i1] = prev_end_level + w * (lv - prev_end_level)
        elif i1 > i0:
            y[i0:i1] = lv
        prev_end_level = lv
    # 空载/未覆盖段跟随输入（低于 4000 的空载带）
    m = np.isnan(y) | (y < 4000)
    y[m] = sin[m]
    return y


def main():
    el, sin, _ = replay(["--mem", "0"])      # 输入臂（sin 与 OFF 臂相同）
    _, _, o_off = replay(["--mem", "0"])
    _, _, o_on = replay(["--mem", "120"])

    # 每段快相落点（约定 A 电平）
    holds = []
    for cls, t0, t1 in HOLDS:
        m = (el >= t0) & (el <= min(t0 + 2.0, t1))
        lv = float(np.median(sin[m]))
        holds.append([cls, t0, t1, lv])
    # 约定 B 电平：每类别用第一次出现的落点
    first = {}
    for h in holds:
        if h[0] not in first:
            first[h[0]] = h[3]
    for h in holds:
        h.append(first[h[0]])

    print("%-4s %8s %8s | A落点(本次快相) B落点(首次) 输入中位" % ("类", "t0", "t1"))
    for cls, t0, t1, lv_a, lv_b in holds:
        m = (el >= t0) & (el <= t1)
        print("%-4s %8.1f %8.1f | %9.0f %9.0f %9.0f"
              % (cls, t0, t1, lv_a, lv_b, float(np.median(sin[m]))))
    for cls in ("满载", "半载"):
        a = [h[3] for h in holds if h[0] == cls]
        b = [h[4] for h in holds if h[0] == cls]
        print("%s族: A 极差 %6.0f std %5.0f | B 极差 %6.0f std %5.0f"
              % (cls, max(a) - min(a), float(np.std(a)),
                 max(b) - min(b), float(np.std(b))))

    ideal_a = build_ideal(el, sin, [(h[1], h[2], h[3]) for h in holds])
    ideal_b = build_ideal(el, sin, [(h[1], h[2], h[4]) for h in holds])

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 2]})

    ax = axes[0]
    ax.plot(el, sin, color="#aaa", lw=0.6, label="输入（补偿前）")
    ax.plot(el, ideal_a, color="#ff7f0e", lw=1.6,
            label="理想显示 · 约定A（当前状态参考：每次加载自己的快相落点）")
    ax.plot(el, ideal_b, color="#2ca02c", lw=1.6,
            label="理想显示 · 约定B（新鲜参考：永远用第一次的落点）")
    ax.set_ylabel("21 通道总量 (ADC)")
    ax.set_title("两种显示约定的含义对比（虚影区 = 两约定的分歧带，"
                 "即粘弹性残余的大小）", fontsize=11)
    ax.fill_between(el, ideal_a, ideal_b, color="#2ca02c", alpha=0.12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(el, sin, color="#aaa", lw=0.5, label="输入")
    ax.plot(el, ideal_b, color="#2ca02c", lw=1.2, label="理想·约定B")
    ax.plot(el, o_off, color="#d62728", lw=0.7, label="实际回放 mem OFF（v3.1）")
    ax.plot(el, o_on, color="#1f77b4", lw=0.7, label="实际回放 mem ON（记忆补丁）")
    ax.set_ylabel("21 通道总量 (ADC)")
    ax.set_title("现有实现与两约定的关系：mem OFF ≈ 约定A（重载段）/ 家族血统（其余）；"
                 "mem ON ≈ 逼近约定B（事件账本式）", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[2]
    labels, aa, bb, oo_off, oo_on = [], [], [], [], []
    for k, (cls, t0, t1, lv_a, lv_b) in enumerate(holds):
        m = (el >= t0 + 1.5) & (el <= t1)
        if m.sum() < 30:
            continue
        labels.append("%s\n%.0fs" % (cls, t0))
        aa.append(lv_a)
        bb.append(lv_b)
        oo_off.append(float(np.median(o_off[m])))
        oo_on.append(float(np.median(o_on[m])))
    x = np.arange(len(labels))
    w = 0.2
    ax.bar(x - 2 * w, aa, w, color="#ff7f0e", label="理想A")
    ax.bar(x - w, bb, w, color="#2ca02c", label="理想B")
    ax.bar(x, oo_off, w, color="#d62728", label="mem OFF")
    ax.bar(x + w, oo_on, w, color="#1f77b4", label="mem ON")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("保压段显示中位 (ADC)")
    ax.set_title("逐保压段对比：理想A / 理想B / 现有两实现", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("零基线-反复增减同一负载 · 显示约定 A vs B · "
                 "（数据 20260919_160854_single_device_7b3977）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("saved:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
