# -*- coding: utf-8 -*-
"""x2 阶段上漂对比：最初参数 vs 最终3参 vs 最终5参（+x2 提速候选）。

上漂定义：持载段（避开加载瞬态）显示从段内锚点到段末的抬升量（正=上漂）。
理想显示 = 恒定（蠕变被 x1+x2 完全吃掉）。

候选杠杆（若仍有上漂）：
  slope_cap_frac 0.01→0.02   x2 积分限速上限放宽
  slope_gate_frac 0.05→0.10  斜率门放宽（较陡的蠕变也放行）

用法：python x2_upward_drift.py
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

P0 = Params()                                                                     # 最初
P3 = replace(P0, r_fast=0.06, slow_confirm_s=4.0, soft_unfreeze_s=2.0)            # 最终3参
P5 = replace(P3, tau_r_slow_idle_s=2.0, tau_r_fast_s=0.5)                         # 最终5参
PD = replace(P5, slope_cap_frac=0.02, slope_gate_frac=0.10)                       # +x2提速

CANDS = [("最初参数", P0), ("最终3参", P3), ("最终5参", P5), ("+x2提速(限速2%/门10%)", PD)]

# 干净持载段的数据（大载荷两段含二次施压，单列说明）
DATASETS = [
    ("193320 长保压",
     Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\算法数据&原始数据\working"
          r"\零基线-同一荷载测试\20260919_193320_single_device_3f32c5\device_001_pre_seg0.csv"),
     (15, 9999)),
    ("095849 长期", HERE / "长期数据" / "20260922_095849_single_device_6cca99" / "device_001_seg000.csv",
     (20, 9999)),
    ("100347 长期", HERE / "长期数据" / "20260922_100347_single_device_151a72" / "device_001_seg000.csv",
     (15, 9999)),
    ("100118 长期(含二次施压)",
     HERE / "长期数据" / "20260922_100118_single_device_2113fb" / "device_001_seg000.csv",
     (20, 75)),
    ("100539 长期(含卸载重载)",
     HERE / "长期数据" / "20260922_100539_single_device_d95573" / "device_001_seg000.csv",
     (20, 78)),
]


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    out = np.empty(n); x2 = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
        x2[i] = c.x_slow.sum()
    return out, x2


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(15, 2.9 * len(DATASETS)))
    print("上漂 = 持载段内显示（锚点=段开始后10s均值 → 段末5s均值）的抬升；"
          "蠕变量 = 同窗口输入抬升\n")
    for r, (name, csv, (t_a, t_b)) in enumerate(DATASETS):
        d = read_session_csv(csv)
        t, V = d["t"], d["values"]
        tin = V.sum(axis=1)
        ax = axes[r]
        ax.plot(t, tin, color="#7f8c8d", lw=0.8, label="输入")
        m = (t >= t[0] + t_a) & (t <= t[0] + t_b)
        creep = float(np.mean(tin[t >= t[0] + t_b - 5] if t_b < 9999 else tin[-5:]) -
                      np.mean(tin[(t >= t[0] + t_a) & (t < t[0] + t_a + 5)]))
        print(f"{name}: 窗口 [{t_a}, {t_b if t_b < 9999 else '末尾'}]s，输入抬升 {creep:+.0f} ADC")
        for (cname, p), col in zip(CANDS, ("#c0392b", "#e67e22", "#2980b9", "#27ae60")):
            out, x2 = run(p, t, V)
            a = float(np.mean(out[(t >= t[0] + t_a) & (t < t[0] + t_a + 5)]))
            if t_b < 9999:
                b = float(np.mean(out[(t >= t[0] + t_b - 5) & (t < t[0] + t_b)]))
            else:
                b = float(np.mean(out[-50:]))
            print(f"    {cname:22s} 显示上漂 {b - a:+8.0f}  (输入抬升的 {(b-a)/creep*100 if creep else 0:5.0f}%)  "
                  f"x2末值 {x2[-1]:6.0f}")
            ax.plot(t, out, lw=1.1, color=col, label=cname)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("总量 (ADC)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
        print()

    fig.suptitle("x2 阶段上漂：最初 → 最终3参 → 最终5参 → +x2提速", y=0.998, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    png = HERE / "out" / "x2上漂对比.png"
    fig.savefig(png, dpi=120)
    print(f"图已保存：{png}")


if __name__ == "__main__":
    main()
