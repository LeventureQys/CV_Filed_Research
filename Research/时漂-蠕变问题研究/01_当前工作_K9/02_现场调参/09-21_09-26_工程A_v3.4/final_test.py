# -*- coding: utf-8 -*-
"""最终参数集 vs 现役参数：全部 6 份数据回归对比。

最终参数集（相对现役 3 处）：
  r_fast 0.10→0.06 / slow_confirm_s 10→4 / soft_unfreeze_s 4→2

数据：长期数据 4 段（2026-09-22）+ 长保压 193320（pre_seg0）+ 抖动数据（2026-09-21）。
指标：x2 起效、x2 末值、持载后半程显示斜率、受载段显示最低点、末帧显示。

用法：python final_test.py
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

P_BASE = Params()
P_NEW = replace(Params(), r_fast=0.06, slow_confirm_s=4.0, soft_unfreeze_s=2.0)

DATASETS = [
    ("长期数据/095849", HERE / "长期数据" / "20260922_095849_single_device_6cca99" / "device_001_seg000.csv", False),
    ("长期数据/100118", HERE / "长期数据" / "20260922_100118_single_device_2113fb" / "device_001_seg000.csv", False),
    ("长期数据/100347", HERE / "长期数据" / "20260922_100347_single_device_151a72" / "device_001_seg000.csv", False),
    ("长期数据/100539", HERE / "长期数据" / "20260922_100539_single_device_d95573" / "device_001_seg000.csv", False),
    ("长保压/193320",
     Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\算法数据&原始数据\working"
          r"\零基线-同一荷载测试\20260919_193320_single_device_3f32c5\device_001_pre_seg0.csv"), False),
    ("抖动/20260921", HERE / "device_001_seg000.csv", True),      # 10 Hz → 插值 100 Hz
]


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    out = np.empty(n); x1 = np.empty(n); x2 = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
        x1[i] = c.x_fast.sum(); x2[i] = c.x_slow.sum()
    return out, x1, x2


def analyze(name, t, tin, out, x2, p):
    peak = tin.max(); base5 = float(np.percentile(tin, 5)); step = peak - base5
    i_on = int(np.argmax(tin > 0.2 * step + base5))
    above = x2 > 0.005 * step
    tx2 = float(t[int(np.argmax(above))] - t[i_on]) if above.any() else float("nan")
    loaded = tin > 0.2 * step + base5
    hold = t > max(t[i_on] + 20, t[-1] * 0.5)
    if hold.sum() < 50:
        hold = t > t[i_on] + 5
    slope_hold = float(np.polyfit(t[hold], out[hold], 1)[0])
    disp_min = float(out[loaded].min())
    return {"x2_start": tx2, "x2_end": float(x2[-1]), "slope_hold": slope_hold,
            "disp_min": disp_min, "out_end": float(out[-1]), "ded_max": float(np.max(tin - out))}


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

    n = len(DATASETS)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.9 * n), sharex=False)
    results = []

    for r, (name, csv, need_up) in enumerate(DATASETS):
        d = read_session_csv(csv)
        t0, V0 = d["t"], d["values"]
        if need_up:
            nu = int(round((t0[-1] - t0[0]) * 100.0)) + 1
            t = t0[0] + np.arange(nu) / 100.0
            V = np.stack([np.interp(t, t0, V0[:, k]) for k in range(V0.shape[1])], axis=1)
            note = "（10Hz→100Hz 插值）"
        else:
            t, V = t0, V0
            note = ""
        tin = V.sum(axis=1)

        ob, _, x2b = run(P_BASE, t, V)
        on_, _, x2n = run(P_NEW, t, V)
        mb = analyze(name, t, tin, ob, x2b, P_BASE)
        mn = analyze(name, t, tin, on_, x2n, P_NEW)
        results.append((name, mb, mn))

        ax = axes[r]
        ax.plot(t, tin, color="#7f8c8d", lw=0.8, label="输入")
        ax.plot(t, ob, color="#c0392b", lw=1.0, label="现役参数")
        ax.plot(t, on_, color="#27ae60", lw=1.2, label="最终参数")
        ax.set_title(f"{name} {note}：x2 起效 {mb['x2_start']:+.1f}s → {mn['x2_start']:+.1f}s，"
                     f"末帧扣除 {tin[-1]-ob[-1]:.0f} → {tin[-1]-on_[-1]:.0f} ADC",
                     loc="left", fontsize=9.5)
        ax.set_ylabel("总量 (ADC)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.25)
        if r == n - 1:
            ax.set_xlabel("时间 (s)")
        print(f"[{r+1}/{n}] {name} 完成")

    fig.suptitle("最终参数（r_fast=0.06, slow_confirm=4, soft_unfreeze=2）vs 现役 —— 全部 6 份数据",
                 y=0.998, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    png = HERE / "out" / "最终参数_全数据对比.png"
    fig.savefig(png, dpi=120)

    print("\n=== 指标对比（现役 → 最终）===")
    print(f"{'数据集':16s} {'x2起效':>14s} {'x2末值':>14s} {'持载斜率(ADC/s)':>20s} "
          f"{'受载显示最低':>16s} {'末帧显示':>18s}")
    for name, mb, mn in results:
        print(f"{name:16s} {mb['x2_start']:+6.1f}s→{mn['x2_start']:+5.1f}s "
              f"{mb['x2_end']:6.0f}→{mn['x2_end']:6.0f} "
              f"{mb['slope_hold']:+7.1f}→{mn['slope_hold']:+7.1f} "
              f"{mb['disp_min']:8.0f}→{mn['disp_min']:8.0f} "
              f"{mb['out_end']:8.0f}→{mn['out_end']:8.0f}")
    print(f"\n图已保存：{png}")


if __name__ == "__main__":
    main()
