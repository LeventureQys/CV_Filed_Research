# -*- coding: utf-8 -*-
"""v3.4 观测器全量出图：temp 下全部会话，每数据一张（输入/显示/非弹性状态），
存到 ../figure/observer/；并输出全量指标表到 ../results/v34_observer_all.txt。"""
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_observer_core2 import observe2 as observe
from v34_law_identify import segments, classify
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "figure", "observer")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def safe(name):
    return re.sub(r"[^\w\-.]+", "_", name)


def load_input(ds_dir):
    pre = os.path.join(ds_dir, "device_001_pre_seg0.csv")
    name = "device_001_pre_seg0.csv" if os.path.isfile(pre) \
        else "device_001_seg000.csv"
    return name, L.load_stream(ds_dir, name)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    report = []
    roots = [("算法数据", L.DATA_ROOT),
             ("原始数据only", os.path.join(L.ROOT, "temp", "原始数据only"))]
    for rtag, root in roots:
        for label, d in L.discover_sessions(root, 5):
            name, s = load_input(d)
            el, ts, V = s["el"], s["ts"], s["V"]
            D, X = observe(ts, V)
            din = V.sum(axis=1)
            dout = D.sum(axis=1)
            dx = X.sum(axis=1)
            n = min(len(din), len(el))
            el, din, dout, dx = el[:n], din[:n], dout[:n], dx[:n]

            # 指标：平台段 in/out std 中位 + 显示无负值帧数
            segs = classify(segments(el, din))
            si_s, so_s = [], []
            for t0, t1, lv, c in segs:
                m = (el >= t0 + 2.0) & (el <= t1 - 1.0)
                if m.sum() < 100:
                    continue
                si_s.append(float(np.std(din[m])))
                so_s.append(float(np.std(dout[m])))
            med_i = float(np.median(si_s)) if si_s else 0.0
            med_o = float(np.median(so_s)) if so_s else 0.0
            report.append("%-58s 平台数%2d  保压std: in=%6.0f → out=%6.0f"
                          % (label.replace("\\", "/") + " [" + name + "]",
                             len(segs), med_i, med_o))

            fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.4]})
            ax = axes[0]
            ax.plot(el, din, color="#aaa", lw=0.6, label="输入（补偿前）")
            ax.plot(el, dout, color="#1f77b4", lw=0.8,
                    label="显示 = 输入 − 非弹性状态（双态观测器）")
            ax.set_ylabel("通道总量 (ADC)")
            ax.set_title("%s · %s  [输入: %s]  %d 帧" %
                         (label.replace("\\", "/"), rtag, name, n), fontsize=10)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(alpha=0.25)
            ax = axes[1]
            ax.plot(el, dx, color="#9467bd", lw=0.7, label="非弹性状态 x1+x2（总量）")
            ax.fill_between(el, 0, dx, color="#9467bd", alpha=0.15)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("x (ADC)")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.25)
            fn = safe("%s_%s_%s" % (rtag, label.replace("\\", "/"), name)
                      ).replace("device_001_", "") + ".png"
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, fn), dpi=110)
            plt.close(fig)
            print("saved", fn)
    txt = "\n".join(report)
    with open(os.path.join(HERE, "..", "results", "v34_observer_all.txt"),
              "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
