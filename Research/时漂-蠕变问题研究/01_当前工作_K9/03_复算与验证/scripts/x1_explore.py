# -*- coding: utf-8 -*-
"""步骤1：看数据 —— working 四个会话的输入/录制显示/观测器显示 + x1/x2 分解。"""
import os
import sys

import numpy as np

import x1_lab_lib as X
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_observer_core3 import observe3  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figure", "x1lab")
os.makedirs(OUT, exist_ok=True)


def decompose(ts, V, p=None):
    """跑 core3，但同时导出 x1/x2 逐帧轨迹。"""
    import v34_observer_core3 as C
    pp = p if p is not None else C.P3
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    zero = V[0].copy()
    y_max = np.maximum(V[0] - zero, 0.0)
    v_lp = V[0].copy()
    X1 = np.empty(n)
    X2 = np.empty(n)
    D = np.empty(n)
    t_prev = ts[0]
    for i in range(n):
        dt = min(max(ts[i] - t_prev, 0.0), 0.1)
        t_prev = ts[i]
        v = V[i]
        if dt > 0.0:
            y = v - zero
            y_max = np.maximum(y_max * np.exp(-dt / pp["y_max_tau"]),
                               np.maximum(y, 0.0))
            idle = y < pp["idle_frac"] * np.maximum(y_max, 1.0)
            zero = np.where(idle, zero + (dt / pp["tau_zero"]) * (v - zero), zero)
            y = v - zero
            e_now = np.maximum(y - x1 - x2, 0.0)
            dx1_rate = np.where(e_now > 0.0,
                                (pp["r1"] * e_now - x1) / pp["tc1"],
                                -x1 / pp["tr1"])
            x1 = np.maximum(x1 + dt * dx1_rate, 0.0)
            slope = (v - v_lp) / pp["tau_slope"]
            v_lp = v_lp + (dt / pp["tau_slope"]) * (v - v_lp)
            e = np.maximum(y - x1 - x2, 0.0)
            rate_cap = pp["slope_cap"] * np.maximum(e, 1.0)
            gate = pp["slope_gate"] * np.maximum(e, 1.0)
            dx2 = np.clip(slope - dx1_rate, -rate_cap, rate_cap)
            dx2 = np.where((e > 0.0) & (np.abs(slope) < gate), dx2, 0.0) * dt
            x2 = x2 + dx2
            x2 = np.where(e > 0.0,
                          np.clip(x2, 0.0, pp["r2max"] * np.maximum(e, 1.0)),
                          np.maximum(x2 - dt * x2 / pp["tr2"], 0.0))
        X1[i] = x1.sum()
        X2[i] = x2.sum()
        D[i] = (v - x1 - x2).sum()
    return D, X1, X2


def main():
    for label, d in X.sessions(X.WORKING, 5):
        name, pre = X.load_pre(d)
        rec = X.load_recorded(d)
        el, ts, V = pre["el"], pre["ts"], pre["V"]
        din = X.total(pre)
        drec = X.total(rec)
        D, X1, X2 = decompose(ts, V)
        ups, downs = X.edges(el, din)
        print("=" * 100)
        print("%s  [%s]  %d 帧 %.0fs  加载沿 %d 个"
              % (label.replace("\\", "/"), name, len(el), el[-1], len(ups)))
        print("  电平范围 输入 %.0f~%.0f  录制显示 %.0f~%.0f  观测器 %.0f~%.0f"
              % (din.min(), din.max(), drec.min(), drec.max(), D.min(), D.max()))
        print("  x1 峰值 %.0f   x2 峰值 %.0f   输入末端%.0f 录制末端%.0f 观测器末端%.0f"
              % (X1.max(), X2.max(), din[-1], drec[-1], D[-1]))
        # 事件明细
        steps = X.step_levels(el, din, ups, downs)
        print("  沿明细（输入台阶 → x1 段内涨幅 / x2 段内涨幅 / 显示段内变化）:")
        for k, s in enumerate(steps[:12]):
            i, j = s["i"], s["j"]
            x1g = X1[j - 1] - X1[i]
            x2g = X2[j - 1] - X2[i]
            dg = D[j - 1] - D[i]
            print("    #%2d t=%7.1f  输入台阶=%+7.0f  段末输入-x1-x2: x1+%6.0f x2+%6.0f "
                  "显示%+7.0f  显示-输入=%+7.0f"
                  % (k, s["t"], s["step"], x1g, x2g, dg, D[j - 1] - din[j - 1]))
        # 图
        fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.6, 1.6]})
        ax = axes[0]
        ax.plot(el, din, color="#bbb", lw=0.6, label="输入 pre（补偿前）")
        ax.plot(el, drec, color="#ff7f0e", lw=0.7, label="录制显示 seg（v3.4 实机）")
        ax.plot(el, D, color="#1f77b4", lw=0.8, label="原型观察器 显示 = v−x1−x2")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylabel("通道总量 ADC")
        ax.set_title(label.replace("\\", "/"), fontsize=10)
        ax.grid(alpha=0.25)
        ax = axes[1]
        ax.plot(el, X1, color="#d62728", lw=0.8, label="x1 快态")
        ax.plot(el, X2, color="#2ca02c", lw=0.8, label="x2 慢态")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_ylabel("x ADC")
        ax = axes[2]
        ax.plot(el, D - din, color="#9467bd", lw=0.6)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylabel("显示−输入")
        ax.set_xlabel("时间 s")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fn = label.replace("\\", "/").replace("/", "__") + ".png"
        fig.savefig(os.path.join(OUT, fn), dpi=100)
        plt.close(fig)
        np.savez(os.path.join(OUT, label.replace("\\", "/").replace("/", "__") + ".npz"),
                 el=el, ts=ts, V=V, din=din, drec=drec, D=D, X1=X1, X2=X2)
        print("  图 -> figure/x1lab/%s" % fn)


if __name__ == "__main__":
    main()
