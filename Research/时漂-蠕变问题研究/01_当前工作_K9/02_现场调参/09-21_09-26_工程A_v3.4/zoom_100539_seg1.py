# -*- coding: utf-8 -*-
"""放大：长期/100539 第一段受载窗（2.3~78s）的调参效果。

三联图（同一时间窗放大）：
  ① 总量：输入 / 最初参数 / 最终5参（标出各自 x2 起效时刻）
  ② 显示斜率（2s 窗）：>0 上漂 / <0 下漂
  ③ 扣除分解：x1 / x2（最终5参）
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

P0 = Params()
P5 = replace(P0, r_fast=0.06, slow_confirm_s=4.0, soft_unfreeze_s=2.0,
             tau_r_slow_idle_s=2.0, tau_r_fast_s=0.5)


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    out = np.empty(n); x1 = np.empty(n); x2 = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
        x1[i] = c.x_fast.sum(); x2[i] = c.x_slow.sum()
    return out, x1, x2


def slope_of(t, x, win=2.0, fs=100.0):
    k = max(1, int(win * fs))
    n = len(x)
    s = np.zeros(n)
    for i in range(n):
        j = min(i + k, n - 1)
        i0 = max(i - k, 0)
        dt = t[j] - t[i0]
        s[i] = (x[j] - x[i0]) / dt if dt > 1e-6 else 0.0
    return s


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    csv = HERE / "长期数据" / "20260922_100539_single_device_d95573" / "device_001_seg000.csv"
    d = read_session_csv(csv)
    t, V = d["t"], d["values"]
    tin = V.sum(axis=1)

    out0, x10, x20 = run(P0, t, V)
    out5, x15, x25 = run(P5, t, V)

    step = tin.max() - np.percentile(tin, 5)
    base5 = float(np.percentile(tin, 5))
    i_on = int(np.argmax(tin > 0.2 * step + base5))
    t_on = t[i_on]
    t_stop = 78.0                      # 第一段受载窗结束（原分析窗口）

    def x2_start(x2):
        above = x2 > 0.005 * step
        return t[int(np.argmax(above))] if above.any() else None

    tx0, tx5 = x2_start(x20), x2_start(x25)
    print(f"加载沿 t={t_on:.1f}s；x2 起效：最初 {tx0:.1f}s（+{tx0-t_on:.1f}s）"
          f" / 最终5参 {tx5:.1f}s（+{tx5-t_on:.1f}s）")

    def stats(out):
        m = (t >= t_on) & (t <= t_stop)
        k10 = max(1, int(m.sum() * 0.1))
        idx = np.where(m)[0]
        drift = float(np.mean(out[idx[-k10:]]) - np.mean(out[idx[:k10]]))
        return drift, float(np.min(out[m])), float(np.max(out[m]))

    for nm, out in (("最初参数", out0), ("最终5参", out5)):
        dr, mn, mx = stats(out)
        print(f"  {nm}: 段内显示变化 {dr:+7.0f} ADC  区间 [{mn:.0f}, {mx:.0f}]  "
              f"输入同段 {np.mean(tin[(t>=t_on)&(t<=t_stop)][-100:]):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    m = (t >= max(0.0, t_on - 3)) & (t <= t_stop)
    tw = t[m]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                                     gridspec_kw={"height_ratios": [2.0, 1.1, 1.1]})

    a1.plot(tw, tin[m], color="#7f8c8d", lw=1.1, label="输入")
    a1.plot(tw, out0[m], color="#c0392b", lw=1.4, label="最初参数（现役）")
    a1.plot(tw, out5[m], color="#27ae60", lw=1.6, label="最终5参")
    a1.axvline(t_on, color="#e67e22", lw=1.3, ls="--")
    a1.annotate(f"加载沿 {t_on:.1f}s", xy=(t_on, tin[i_on]), xytext=(6, 14),
                textcoords="offset points", fontsize=9, color="#b9770e")
    for tx, col, nm in ((tx0, "#c0392b", "最初 x2 起效"), (tx5, "#27ae60", "最终 x2 起效")):
        if tx is not None:
            a1.axvline(tx, color=col, lw=1.1, ls=":")
            a1.annotate(f"{nm} {tx:.1f}s\n(+{tx - t_on:.1f}s)", xy=(tx, a1.get_ylim()[1]),
                        xytext=(6, -14), textcoords="offset points", fontsize=8.5, color=col,
                        va="top")
    a1.set_ylabel("总量 (ADC)")
    a1.set_title("长期/100539 第一段受载窗（2~78s）放大：最初 vs 最终5参", loc="left", fontsize=11)
    a1.legend(loc="lower right", fontsize=9)
    a1.grid(alpha=0.25)

    a2.axhline(0, color="k", lw=0.7)
    a2.plot(tw, slope_of(t, tin)[m], color="#7f8c8d", lw=1.0, label="输入斜率")
    a2.plot(tw, slope_of(t, out0)[m], color="#c0392b", lw=1.3, label="最初 显示斜率")
    a2.plot(tw, slope_of(t, out5)[m], color="#27ae60", lw=1.4, label="最终5参 显示斜率")
    a2.set_ylabel("斜率 (ADC/s)")
    a2.set_title("显示斜率：>0 上漂 / <0 下漂（持载理想 = 0）", loc="left", fontsize=10)
    a2.legend(loc="lower right", fontsize=9, ncol=3)
    a2.grid(alpha=0.25)

    a3.plot(tw, x15[m], color="#e67e22", lw=1.3, label="x1 快态（最终5参）")
    a3.plot(tw, x25[m], color="#27ae60", lw=1.5, label="x2 慢态（最终5参）")
    a3.plot(tw, x15[m] + x25[m], color="#8e44ad", lw=1.0, ls="--", label="总扣除")
    a3.plot(tw, x10[m] + x20[m], color="#c0392b", lw=1.0, alpha=0.7,
            label="总扣除（最初参数）")
    a3.set_ylabel("扣除量 (ADC)")
    a3.set_xlabel("时间 (s)")
    a3.set_title("扣除分解（最终5参）与最初参数总扣除对照", loc="left", fontsize=10)
    a3.legend(loc="lower right", fontsize=9)
    a3.grid(alpha=0.25)

    fig.tight_layout()
    png = HERE / "out" / "100539_第一段放大.png"
    fig.savefig(png, dpi=130)
    print(f"图已保存：{png}")


if __name__ == "__main__":
    main()
