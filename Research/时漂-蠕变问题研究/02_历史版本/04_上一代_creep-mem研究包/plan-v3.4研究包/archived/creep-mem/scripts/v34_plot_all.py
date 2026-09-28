# -*- coding: utf-8 -*-
"""v3.4 全数据集出图：temp 下全部会话（算法数据集 + B 组原始数据），
每个数据一张图（输入 + mem OFF 显示 + mem ON 显示），保存到 ../figure/all/。
"""
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_probe_oos import run_stream, load_input
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "figure", "all")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def safe(name):
    return re.sub(r"[^\w\-.]+", "_", name)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    roots = [("算法数据", L.DATA_ROOT), ("原始数据only",
            os.path.join(L.ROOT, "temp", "原始数据only"))]
    n = 0
    for rtag, root in roots:
        for label, d in L.discover_sessions(root, 5):
            name, s = load_input(d)
            el = s["el"]
            sin0, o0 = run_stream(s["ts"], s["V"], ["--mem", "0"])
            _, o1 = run_stream(s["ts"], s["V"], ["--mem", "120"])
            if len(o0) != len(el):
                el = el[:len(o0)]

            fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
            ax = axes[0]
            ax.plot(el, sin0, color="#aaa", lw=0.6, label="输入（补偿前）")
            ax.plot(el, o0, color="#d62728", lw=0.8,
                    label="显示 mem OFF（plan-v3.1）")
            ax.plot(el, o1, color="#1f77b4", lw=0.8,
                    label="显示 mem ON（plan-v3.4）")
            ax.set_ylabel("通道总量 (ADC)")
            ax.set_title("%s · %s  [输入流: %s]  %d 帧" %
                         (label.replace("\\", "/"), rtag, name, len(o0)),
                         fontsize=10)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.25)

            ax = axes[1]
            ax.plot(el, o1 - o0, color="#9467bd", lw=0.5)
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("Δ(ON−OFF)")
            ax.grid(alpha=0.25)

            fn = safe("%s_%s_%s" % (rtag, label.replace("\\", "/"), name)
                      ).replace("device_001_", "") + ".png"
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, fn), dpi=110)
            plt.close(fig)
            n += 1
            print("saved", fn)
    print("total:", n)


if __name__ == "__main__":
    main()
