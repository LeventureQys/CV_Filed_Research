# -*- coding: utf-8 -*-
"""当前调参结果（最终5参）单独体检：上漂/下漂 + 斜率，两份数据。

数据：抖动/20260921（10Hz→100Hz 插值）、长期/100539（原生 100Hz）。
最终5参：r_fast=0.06, slow_confirm_s=4, soft_unfreeze_s=2,
          tau_r_slow_idle_s=2, tau_r_fast_s=0.5
上图：输入 vs 调参后显示（黄底=受载窗，白底=卸载窗）
下图：调参后显示斜率 vs 输入斜率（2s 窗差分）——>0 上漂 / <0 下漂，持载理想 ≈0。
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

P_NOW = replace(Params(), r_fast=0.06, slow_confirm_s=4.0, soft_unfreeze_s=2.0,
                tau_r_slow_idle_s=2.0, tau_r_fast_s=0.5)


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


def load_dataset(csv: Path, need_up: bool):
    d = read_session_csv(csv)
    t0, V0 = d["t"], d["values"]
    if need_up:
        nu = int(round((t0[-1] - t0[0]) * 100.0)) + 1
        t = t0[0] + np.arange(nu) / 100.0
        V = np.stack([np.interp(t, t0, V0[:, k]) for k in range(V0.shape[1])], axis=1)
    else:
        t, V = t0, V0
    return t, V


def analyze(name, t, V, ax1, ax2):
    tin = V.sum(axis=1)
    c = CreepObserverK9(P_NOW)            # 当前调参（最终5参）
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()

    s_in = slope_of(t, tin)
    s_out = slope_of(t, out)

    step = tin.max() - np.percentile(tin, 5)
    base5 = float(np.percentile(tin, 5))
    loaded = tin > 0.3 * step + base5

    # 分段漂移简报
    print(f"\n=== {name}：调参后（最终5参）分段漂移（正=上漂，负=下漂）===")
    segs = []
    i = 0
    n = len(t)
    while i < n:
        if loaded[i]:
            j = i
            while j + 1 < n and loaded[j + 1]:
                j += 1
            if j - i > 50:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    for (i, j) in segs:
        k10 = max(1, (j - i) // 10)
        drift = float(np.mean(out[j - k10:j]) - np.mean(out[i:i + k10]))
        d_in = float(np.mean(tin[j - k10:j]) - np.mean(tin[i:i + k10]))
        print(f"  持载 {t[i]:5.1f}~{t[j]:5.1f}s: 显示 {drift:+8.0f} ADC "
              f"（输入 {d_in:+8.0f}，相对输入 {drift - d_in:+8.0f}）  "
              f"段内斜率均值 {np.mean(s_out[i:j]):+7.1f} ADC/s")

    ax1.fill_between(t, 0, 1, where=loaded, transform=ax1.get_xaxis_transform(),
                     color="#f1c40f", alpha=0.10)
    ax1.plot(t, tin, color="#7f8c8d", lw=0.9, label="输入")
    ax1.plot(t, out, color="#27ae60", lw=1.4, label="当前调参（最终5参）")
    ax1.set_ylabel("总量 (ADC)")
    ax1.set_title(f"{name}（黄底=受载窗，白底=卸载窗）", loc="left", fontsize=10.5)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.fill_between(t, 0, 1, where=loaded, transform=ax2.get_xaxis_transform(),
                     color="#f1c40f", alpha=0.10)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.plot(t, s_in, color="#7f8c8d", lw=0.9, label="输入斜率（2s 窗）")
    ax2.plot(t, s_out, color="#27ae60", lw=1.3,
             label="调参后显示斜率（2s 窗）：>0 上漂 / <0 下漂")
    ax2.set_ylabel("斜率 (ADC/s)")
    ax2.set_xlabel("时间 (s)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.25)


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

    jobs = [
        ("抖动/20260921", HERE / "device_001_seg000.csv", True),
        ("长期/100539", HERE / "长期数据" / "20260922_100539_single_device_d95573"
         / "device_001_seg000.csv", False),
    ]

    fig, axes = plt.subplots(len(jobs) * 2, 1, figsize=(15, 4.2 * len(jobs) * 1.1))
    for r, (name, csv, up) in enumerate(jobs):
        t, V = load_dataset(csv, up)
        analyze(name, t, V, axes[2 * r], axes[2 * r + 1])
        print(f"[{r + 1}/{len(jobs)}] {name} 完成")

    fig.suptitle("当前调参（最终5参：r.06/sc4/soft2/idle2/fast0.5）体检：上漂/下漂与斜率",
                 y=0.998, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    png = HERE / "out" / "调参后_漂移与斜率.png"
    fig.savefig(png, dpi=130)
    print(f"\n图已保存：{png}")


if __name__ == "__main__":
    main()
