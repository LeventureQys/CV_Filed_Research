# -*- coding: utf-8 -*-
"""长期/100539 第一段放大 + 加载沿后 1/3/5/15s 时间区间与 Δ 标注。

三联图：
  ① 总量（输入 / 最初参数 / 最终5参）＋ 稳态平台线 ＋ 1/3/5/15s 竖线与 Δ 标注
  ② 显示斜率（最终5参 vs 输入）
  ③ 扣除分解（x1/x2 + 已扣量检查点）
Δ 口径：
  已扣   = 该时刻（输入升幅 − 显示升幅）= 已施加的补偿量
  距平台 = 平台均值 − 该时刻显示（>0 此刻偏低→后续上漂；<0 此刻偏高→后续下漂）
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
CKPTS = (1.0, 3.0, 5.0, 15.0)


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


def at(t, arr, tq):
    i = min(int(np.searchsorted(t, tq)), len(t) - 1)
    return float(arr[i])


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
    i_pk = next(i for i in range(i_on, len(t)) if tin[i] > 0.8 * step + base5)
    i_un = next((i for i in range(i_pk + 1, len(t)) if tin[i] < 0.3 * step + base5), len(t) - 1)
    t_end = min(t[i_un], 78.0)

    m_plat = (t >= t_on + 0.6 * (t_end - t_on)) & (t <= t_end) & (tin > 0.5 * step + base5)
    plat = float(np.mean(out5[m_plat]))
    print(f"加载 t={t_on:.1f}s  段末 {t_end:.1f}s  稳态平台 {plat:.0f} ADC")

    rows = []
    for dt_ in CKPTS:
        tq = min(t_on + dt_, t_end)
        o, i_ = at(t, out5, tq), at(t, tin, tq)
        o0, i0 = at(t, out5, t_on), at(t, tin, t_on)
        ded = (i_ - i0) - (o - o0)          # 已扣
        dplat = plat - o                    # 距平台
        rows.append((dt_, tq, o, i_, ded, dplat, at(t, x15, tq), at(t, x25, tq)))
        print(f"  Δ={dt_:4.0f}s 显示 {o:7.0f} 输入 {i_:7.0f} 已扣 {ded:7.0f} "
              f"距平台 {dplat:+7.0f} x1 {at(t, x15, tq):6.0f} x2 {at(t, x25, tq):6.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    m = (t >= max(0.0, t_on - 3)) & (t <= t_end + 2)
    tw = t[m]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(16, 12.5), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]})

    # ── ① 总量 + 区间/Δ 标注 ──
    a1.plot(tw, tin[m], color="#7f8c8d", lw=1.1, label="输入")
    a1.plot(tw, out0[m], color="#c0392b", lw=1.3, alpha=0.85, label="最初参数（现役）")
    a1.plot(tw, out5[m], color="#27ae60", lw=1.7, label="最终5参")
    a1.axhline(plat, color="#2c3e50", lw=1.1, ls="-.", alpha=0.8,
               label=f"稳态平台 {plat:.0f} ADC（Δ 的参照线）")
    a1.axvline(t_on, color="#e67e22", lw=1.4, ls="--")
    a1.annotate(f"加载沿 {t_on:.1f}s", xy=(t_on, tin[i_on]), xytext=(6, 16),
                textcoords="offset points", fontsize=9.5, color="#b9770e")

    # 区间着色 + 区间标签
    edges = [t_on] + [t_on + c for c in CKPTS] + [t_end + 2]
    labels = ["0~1s", "1~3s", "3~5s", "5~15s", "15s 以后"]
    for k in range(len(edges) - 1):
        a1.axvspan(edges[k], edges[k + 1], color="#3498db" if k % 2 == 0 else "#8e44ad",
                   alpha=0.05, lw=0)
        a1.text((edges[k] + edges[k + 1]) / 2, 0.995, labels[k], transform=a1.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="#34495e")

    # 检查点竖线 + Δ 文本框
    for j, (dt_, tq, o, i_, ded, dplat, x1v, x2v) in enumerate(rows):
        a1.axvline(tq, color="#2c3e50", lw=0.9, ls=":", alpha=0.7)
        # 从平台线到显示值的箭头 + Δ 文本
        ytxt = plat - 1500 if j % 2 == 0 else plat + 1500
        a1.annotate(f"Δ={dt_:.0f}s\n已扣 {ded:.0f}\n距平台 {dplat:+.0f}",
                    xy=(tq, o), xytext=(tq + 1.2, ytxt), fontsize=8.5, color="#2c3e50",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#2c3e50", alpha=0.85),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=0.9))
    a1.set_ylabel("总量 (ADC)")
    a1.set_title("长期/100539 第一段放大：加载沿后 1/3/5/15s 区间与 Δ 标注"
                 "（最初参数 vs 最终5参）", loc="left", fontsize=11.5)
    a1.legend(loc="lower right", fontsize=9)
    a1.grid(alpha=0.25)

    # ── ② 斜率 ──
    a2.axhline(0, color="k", lw=0.7)
    a2.plot(tw, slope_of(t, tin)[m], color="#7f8c8d", lw=1.0, label="输入斜率")
    a2.plot(tw, slope_of(t, out0)[m], color="#c0392b", lw=1.2, alpha=0.85, label="最初参数 斜率")
    a2.plot(tw, slope_of(t, out5)[m], color="#27ae60", lw=1.4, label="最终5参 斜率")
    for (dt_, tq, *_rest) in rows:
        a2.axvline(tq, color="#2c3e50", lw=0.8, ls=":", alpha=0.6)
        a2.text(tq, 0.98, f"{dt_:.0f}s", transform=a2.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color="#2c3e50")
    a2.set_ylabel("斜率 (ADC/s)")
    a2.set_title("显示斜率：>0 上漂 / <0 下漂（最终5参在 5s 后基本贴 0）", loc="left", fontsize=10)
    a2.legend(loc="lower right", fontsize=9, ncol=3)
    a2.grid(alpha=0.25)

    # ── ③ 扣除分解 ──
    a3.plot(tw, x15[m], color="#e67e22", lw=1.3, label="x1 快态（最终5参）")
    a3.plot(tw, x25[m], color="#27ae60", lw=1.5, label="x2 慢态（最终5参）")
    a3.plot(tw, (x15 + x25)[m], color="#8e44ad", lw=1.1, ls="--", label="总扣除（最终5参）")
    a3.plot(tw, (x10 + x20)[m], color="#c0392b", lw=1.0, alpha=0.7, label="总扣除（最初参数）")
    for (dt_, tq, o, i_, ded, dplat, x1v, x2v) in rows:
        a3.axvline(tq, color="#2c3e50", lw=0.8, ls=":", alpha=0.6)
        a3.annotate(f"{ded:.0f}", xy=(tq, ded), xytext=(tq + 0.8, ded + 250),
                    fontsize=8.5, color="#2c3e50",
                    arrowprops=dict(arrowstyle="-", color="#2c3e50", lw=0.7))
    a3.set_ylabel("扣除量 (ADC)")
    a3.set_xlabel("时间 (s)")
    a3.set_title("扣除分解与各检查点的已扣量", loc="left", fontsize=10)
    a3.legend(loc="lower right", fontsize=9)
    a3.grid(alpha=0.25)

    fig.tight_layout()
    png = HERE / "out" / "100539_第一段放大_带Δ标注.png"
    fig.savefig(png, dpi=130)
    print(f"图已保存：{png}")


if __name__ == "__main__":
    main()
