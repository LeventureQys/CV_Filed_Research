# -*- coding: utf-8 -*-
"""长期数据 × K9 新参数（r_fast=0.06）：跑最新算法并标注 x2（慢态）阶段。

数据：长期数据/ 下 4 段录制（2026-09-22，未开补偿的 processed_display = 原始输入）。

每段标注：
  - 加载沿 / x1 阶段（红底）：扣除以快态 x1 为主
  - x2 启动时刻（竖线）：首次 x2 越过载荷台阶 0.5% 且持续
  - x2 阶段（绿底）：慢态积分门（dwell≥10s + 软冻结解除 + 斜率门）开启
  - 下图：x1/x2 合计曲线 + 积分门开启通道数

用法：python longterm_test.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_v34_on_csv import read_session_csv  # noqa: E402
from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402

ROOT = HERE / "长期数据"
LN2 = 0.69314718056


def run_annotated(p: Params, t, V):
    """跑 K9 并逐帧记录 x1/x2/积分门开启通道数。"""
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    x1s = np.empty(n); x2s = np.empty(n); outs = np.empty(n)
    gate = np.empty(n, dtype=int)          # 本帧 x2 三重门全开的通道数
    for i in range(n):
        outs[i] = c.process(float(t[i]), V[i]).sum()
        x1s[i] = c.x_fast.sum()
        x2s[i] = c.x_slow.sum()
        pr = c.p
        slope = getattr(c, "_slope", np.zeros(c._n))
        e = np.maximum(c.x_fast * 0 + getattr(c, "_e_now", np.zeros(c._n)), 0)
        # 三重门：dwell ≥ slow_confirm / 软冻结解除 / 斜率门（口径同 ⑧ 段）
        soft_ok = 1.0 - np.exp(-c.t_edge / pr.soft_unfreeze_s) > 0.5
        gate[i] = int(np.count_nonzero(
            (c.load_dwell >= pr.slow_confirm_s) & soft_ok &
            (np.abs(slope) < pr.slope_gate_frac * np.maximum(e, 1.0)) & (e > 0)))
    return outs, x1s, x2s, gate


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sessions = sorted(ROOT.glob("*/device_001_seg000.csv"))
    if not sessions:
        print("[错] 长期数据/ 下没有会话", file=sys.stderr)
        return 3
    p_new = replace(Params(), r_fast=0.06)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    n_seg = len(sessions)
    fig, axes = plt.subplots(n_seg, 2, figsize=(17, 3.6 * n_seg),
                             sharex="col",
                             gridspec_kw={"width_ratios": [2.4, 1.6]})
    if n_seg == 1:
        axes = axes.reshape(1, 2)

    for row, csv in enumerate(sessions):
        d = read_session_csv(csv)
        t, V = d["t"], d["values"]
        dur = t[-1] - t[0]
        tin = V.sum(axis=1)
        out, x1s, x2s, gate = run_annotated(p_new, t, V)

        peak = tin.max()
        base5 = float(np.percentile(tin, 5))
        step = peak - base5
        i_on = int(np.argmax(tin > 0.2 * step + base5))
        # x2 启动：恒 >0.5% 台阶 的首个时刻
        thr_x2 = 0.005 * step
        above = x2s > thr_x2
        i_x2 = int(np.argmax(above)) if above.any() else -1

        a1, a2 = axes[row]
        # x1 阶段 / x2 阶段着色
        if i_x2 > 0:
            a1.axvspan(t[i_on], t[i_x2], color="#e74c3c", alpha=0.06)
            a1.axvspan(t[i_x2], t[-1], color="#27ae60", alpha=0.07)
        else:
            a1.axvspan(t[i_on], t[-1], color="#e74c3c", alpha=0.06)

        a1.plot(t, tin, color="#7f8c8d", lw=1.0, label="输入（未补偿）")
        a1.plot(t, out, color="#2980b9", lw=1.3, label="K9 新参（r_fast=0.06）显示")
        a1.axvline(t[i_on], color="#e67e22", lw=1.2, ls="--")
        if i_x2 > 0:
            a1.axvline(t[i_x2], color="#27ae60", lw=1.6)
            a1.annotate(f"x2 启动 t={t[i_x2]:.0f}s\n(距加载 {t[i_x2]-t[i_on]:.0f}s)",
                        xy=(t[i_x2], out[i_x2]), xytext=(8, -34),
                        textcoords="offset points", fontsize=9, color="#1e8449",
                        arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1))
        a1.annotate(f"加载沿 t={t[i_on]:.0f}s", xy=(t[i_on], tin[i_on]),
                    xytext=(6, 18), textcoords="offset points", fontsize=9,
                    color="#b9770e")
        a1.set_title(f"{csv.parent.name}（{dur:.0f} s，{V.shape[1]} 通道，"
                     f"载荷台阶 {step:.0f} ADC）", loc="left", fontsize=10.5)
        a1.set_ylabel("总量 (ADC)")
        a1.legend(loc="upper right", fontsize=9)
        a1.grid(alpha=0.25)

        a2.axvspan(0, 1, color="#e74c3c", alpha=0.0)
        a2.plot(t, x1s, color="#e67e22", lw=1.4, label="x1 快态合计")
        a2.plot(t, x2s, color="#27ae60", lw=1.6, label="x2 慢态合计")
        a2.plot(t, x1s + x2s, color="#8e44ad", lw=1.0, ls="--", label="总扣除 x1+x2")
        ag = a2.twinx()
        ag.plot(t, gate, color="#34495e", lw=0.9, alpha=0.55)
        ag.set_ylabel("x2 积分门开启通道数", fontsize=8.5, color="#34495e")
        ag.tick_params(labelsize=8)
        if i_x2 > 0:
            a2.axvline(t[i_x2], color="#27ae60", lw=1.4)
        a2.set_ylabel("扣除量 (ADC)")
        a2.set_xlabel("时间 (s)")
        a2.set_title("扣除分解：x1（快态）/ x2（慢态，三重门：dwell≥10s+软冻结+斜率门）",
                     loc="left", fontsize=9.5)
        a2.legend(loc="upper left", fontsize=8.5)
        a2.grid(alpha=0.25)

        # 简报
        print(f"{csv.parent.name}: {dur:.0f}s, 台阶 {step:.0f}")
        print(f"  加载 t={t[i_on]:.1f}s  " +
              (f"x2 启动 t={t[i_x2]:.1f}s（距加载 {t[i_x2]-t[i_on]:.1f}s）"
               if i_x2 > 0 else "x2 未启动") +
              f"  末帧 x1={x1s[-1]:.0f} x2={x2s[-1]:.0f} "
              f"(x2 占 {x2s[-1]/max(x1s[-1]+x2s[-1],1e-9)*100:.0f}%)  "
              f"输入蠕变 {tin[i_on:].max()-tin[i_on]:+.0f}")

    fig.suptitle("长期数据 × K9 新参数（r_fast=0.06）：x1 / x2 阶段标注", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = HERE / "out" / "长期数据_K9新参_x2标注.png"
    fig.savefig(out, dpi=125)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
