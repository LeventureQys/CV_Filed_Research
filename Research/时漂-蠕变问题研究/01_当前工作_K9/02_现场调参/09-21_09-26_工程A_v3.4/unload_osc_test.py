# -*- coding: utf-8 -*-
"""卸载期震荡优化：在最终参数之上，加快 x1/x2 的泄放。

假设：绿线（最终参数）卸载期震荡大 = x2 积累变多后，泄放 τ 太长
      （tau_r_slow_s=150s 非空载 / tau_r_slow_idle_s=8s 空载），残留扣除与
      来回跳的输入不同步。

候选（在最终参数 r_fast=0.06 / sc=4 / soft=2 之上叠加）：
  C0  仅最终参数（对照 = 现在的绿线）
  C1  + tau_r_slow_s 150→30        （非空载残留泄放，回撤后下漂收敛也受益）
  C2  C1 + tau_r_slow_idle_s 8→3   （空载 x2 快泄放）
  C3  C2 + tau_r_fast_s 2→1        （x1 卸载恢复加快）

指标：卸载/低位窗（输入 < 30% 台阶）内显示的 min 与 std（震荡幅度）；
      以及全程持载斜率（确认不伤钉平）。

用法：python unload_osc_test.py
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

P_FINAL = replace(Params(), r_fast=0.06, slow_confirm_s=4.0, soft_unfreeze_s=2.0)
CANDS = [
    ("C0 最终参数", P_FINAL),
    ("C2 idle8→3+fast1", replace(P_FINAL, tau_r_slow_idle_s=3.0, tau_r_fast_s=1.0)),
    ("C3b idle4+fast1.5", replace(P_FINAL, tau_r_slow_idle_s=4.0, tau_r_fast_s=1.5)),
    ("C4 idle2+fast0.5", replace(P_FINAL, tau_r_slow_idle_s=2.0, tau_r_fast_s=0.5)),
]

DATASETS = [
    ("长期/095849", HERE / "长期数据" / "20260922_095849_single_device_6cca99" / "device_001_seg000.csv", False),
    ("长期/100118", HERE / "长期数据" / "20260922_100118_single_device_2113fb" / "device_001_seg000.csv", False),
    ("长期/100539", HERE / "长期数据" / "20260922_100539_single_device_d95573" / "device_001_seg000.csv", False),
    ("抖动/20260921", HERE / "device_001_seg000.csv", True),
]


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    out = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


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

    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(16, 3.2 * len(DATASETS)))
    summary = []
    for r, (name, csv, up) in enumerate(DATASETS):
        d = read_session_csv(csv)
        t0, V0 = d["t"], d["values"]
        if up:
            nu = int(round((t0[-1] - t0[0]) * 100.0)) + 1
            t = t0[0] + np.arange(nu) / 100.0
            V = np.stack([np.interp(t, t0, V0[:, k]) for k in range(V0.shape[1])], axis=1)
        else:
            t, V = t0, V0
        tin = V.sum(axis=1)
        peak = tin.max(); base5 = float(np.percentile(tin, 5)); step = peak - base5
        low = tin < 0.3 * step + base5        # 卸载/低位窗
        ax = axes[r]
        ax.plot(t, tin, color="#7f8c8d", lw=0.8)
        res = {}
        for (cname, p), col, ls in zip(CANDS, ("#c0392b", "#e67e22", "#2980b9", "#27ae60"),
                                       ("-", "--", "--", "-")):
            out = run(p, t, V)
            res[cname] = out
            osc_std = float(np.std(out[low])) if low.any() else float("nan")
            osc_min = float(out[low].min()) if low.any() else float("nan")
            hold = t > max(t[0] + 20, t[-1] * 0.5)
            slope = float(np.polyfit(t[hold], out[hold], 1)[0]) if hold.sum() > 50 else float("nan")
            summary.append((name, cname, osc_std, osc_min, slope, float(out[-1])))
            ax.plot(t, out, lw=1.1, color=col, ls=ls, label=cname)
        if low.any():
            ax.fill_between(t, *ax.get_ylim(), where=low, color="#95a5a6", alpha=0.12,
                            label="卸载/低位窗")
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("总量 (ADC)")
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        print(f"[{r+1}/{len(DATASETS)}] {name} 完成")

    print("\n=== 卸载/低位窗震荡指标（std 越小越好；min 越高越好）===")
    print(f"{'数据集':12s} {'方案':16s} {'低位std':>9s} {'低位min':>9s} {'持载斜率':>10s} {'末帧':>8s}")
    for name, cname, sd, mn, sl, end in summary:
        print(f"{name:12s} {cname:16s} {sd:9.0f} {mn:9.0f} {sl:+9.1f}/s {end:8.0f}")

    fig.suptitle("卸载期震荡优化：泄放 τ 提速（在最终参数之上叠加）", y=0.998, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    png = HERE / "out" / "卸载震荡优化.png"
    fig.savefig(png, dpi=120)
    print(f"\n图已保存：{png}")


if __name__ == "__main__":
    main()
