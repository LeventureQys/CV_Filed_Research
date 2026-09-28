# -*- coding: utf-8 -*-
"""测量当前长期数据的快相(x1阶段)时长，并验证 x2 门提前的可行性。

步骤：
  1) 每段录制：加载沿后输入总量平滑斜率从峰值衰减到 20% 的时刻 = 快相结束；
     对比 x1 收敛点与 slow_confirm_s=10s 的相对位置；
  2) 候选参数（提前 x2）：(10,4)现役 / (6,2.5) / (4,2)，跑长期数据看 x2 起效与持载钉平；
  3) 安全复核：抖动数据（20260921 device_001_seg000.csv）上确认 x2 不会在
     短促触碰里误积（对比 x2 峰值与显示最低点）。

用法：python shift_x2_earlier.py
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


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    out = np.empty(n); x1 = np.empty(n); x2 = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
        x1[i] = c.x_fast.sum(); x2[i] = c.x_slow.sum()
    return out, x1, x2


def smooth(x, fs=100.0, win=1.0):
    k = max(1, int(win * fs))
    return np.convolve(x, np.ones(k) / k, mode="same")


def load_edge(tin, step, base5):
    return int(np.argmax(tin > 0.2 * step + base5))


def fast_phase_end(t, tin, i_on):
    """加载沿后：平滑输入斜率从峰值降到 20% 峰值的时刻（快相结束）。
    时间戳存在同包重复值，用窗口首尾差分（2s 窗）求斜率，避免除零。"""
    y = smooth(tin, 2.0)
    w = 200                                   # ~2 s @100 Hz
    n = len(t)
    sl = np.zeros(n)
    for i in range(n):
        j = min(i + w, n - 1)
        dt = t[j] - t[max(i - w, 0)]
        sl[i] = (y[j] - y[max(i - w, 0)]) / dt if dt > 1e-6 else 0.0
    seg = sl[i_on:]
    pk = float(np.max(seg))
    below = np.where(seg < 0.2 * pk)[0]
    return float(t[below[0] + i_on] - t[i_on]) if len(below) else float("nan")


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    sessions = sorted((HERE / "长期数据").glob("*/device_001_seg000.csv"))
    print("=== 1) 当前数据的快相(x1阶段)时长 ===")
    rows = []
    for csv in sessions:
        d = read_session_csv(csv)
        t, V = d["t"], d["values"]
        tin = V.sum(axis=1)
        peak = tin.max(); base5 = float(np.percentile(tin, 5)); step = peak - base5
        i_on = load_edge(tin, step, base5)
        tf = fast_phase_end(t, tin, i_on)
        rows.append((csv.parent.name, t, V, tin, i_on, step, tf))
        print(f"  {csv.parent.name}: 加载 t={t[i_on]:6.1f}s  快相结束 ≈ +{tf:5.1f}s  "
              f"（现役 x2 确认窗 10s ⇒ 空窗 ≈ {10 - tf:+5.1f}s）")

    print("\n=== 2) x2 提前候选（r_fast=0.06 不变）在长期数据上的效果 ===")
    cands = [("现役门 (sc=10, soft=4)", Params()),
             ("候选A (sc=6, soft=2.5)", replace(Params(), slow_confirm_s=6.0, soft_unfreeze_s=2.5)),
             ("候选B (sc=4, soft=2)", replace(Params(), slow_confirm_s=4.0, soft_unfreeze_s=2.0))]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    n_seg = len(rows)
    fig, axes = plt.subplots(n_seg, 2, figsize=(16, 3.4 * n_seg), sharex="col",
                             gridspec_kw={"width_ratios": [2.2, 1.8]})
    if n_seg == 1:
        axes = axes.reshape(1, 2)

    for r, (name, t, V, tin, i_on, step, tf) in enumerate(rows):
        a1, a2 = axes[r]
        a1.plot(t, tin, color="#7f8c8d", lw=0.9, label="输入")
        results = {}
        for (cname, p0) in cands:
            p = replace(p0, r_fast=0.06)
            out, x1, x2 = run(p, t, V)
            results[cname] = (out, x1, x2)
            above = x2 > 0.005 * step
            tx2 = t[int(np.argmax(above))] if above.any() else None
            # 持载钉平：持载段后半程显示斜率（理想≈0）
            hold = t > max(t[i_on] + 20, t[-1] * 0.5)
            slope_hold = float(np.polyfit(t[hold], out[hold], 1)[0])
            print(f"  {name[-22:]}  {cname}: x2起效 "
                  f"{(f'+{tx2 - t[i_on]:.1f}s' if tx2 else '未启动'):>8s}  "
                  f"x2末值 {x2[-1]:6.0f}  持载后半程显示斜率 {slope_hold:+7.1f} ADC/s")
            a1.plot(t, out, lw=1.1,
                    label=f"{cname.split(' ')[0]} 显示（x2 {f'{tx2-t[i_on]:+.0f}s' if tx2 else '—'}）")
        # 标注快相结束与 x2 门
        a1.axvline(t[i_on] + tf, color="#e67e22", lw=1.2, ls="--")
        a1.text(t[i_on] + tf, a1.get_ylim()[1] * 0.02 + tin.max() * 0.98,
                f" 快相结束 +{tf:.0f}s", fontsize=8.5, color="#b9770e", va="top")
        a1.set_title(name, loc="left", fontsize=9.5)
        a1.set_ylabel("总量 (ADC)")
        a1.legend(loc="lower right", fontsize=8)
        a1.grid(alpha=0.25)

        for (cname, _), col in zip(cands, ("#27ae60", "#2980b9", "#c0392b")):
            a2.plot(t, results[cname][2], lw=1.2, color=col, label=f"x2 {cname.split(' ')[0]}")
        a2.axvline(t[i_on] + tf, color="#e67e22", lw=1.2, ls="--")
        a2.set_ylabel("x2 (ADC)")
        a2.set_xlabel("时间 (s)")
        a2.legend(loc="upper left", fontsize=8)
        a2.grid(alpha=0.25)

    print("\n=== 3) 安全复核：抖动数据（20260921 device_001_seg000.csv）===")
    csv2 = HERE / "device_001_seg000.csv"
    d = read_session_csv(csv2)
    t0r, V0r = d["t"], d["values"]
    n_up = int(round((t0r[-1] - t0r[0]) * 100.0)) + 1
    tb = t0r[0] + np.arange(n_up) / 100.0
    Vb = np.stack([np.interp(tb, t0r, V0r[:, k]) for k in range(V0r.shape[1])], axis=1)
    for (cname, p0) in cands:
        p = replace(p0, r_fast=0.06)
        out, x1, x2 = run(p, tb, Vb)
        loaded = Vb.sum(axis=1) > 0.2 * (Vb.sum(axis=1).max() - np.percentile(Vb.sum(axis=1), 5)) \
            + np.percentile(Vb.sum(axis=1), 5)
        disp_min_in_load = float(out[loaded].min())
        print(f"  {cname}: x2 峰值 {x2.max():6.0f}  x2 末值 {x2[-1]:5.0f}  "
              f"受载段显示最低 {disp_min_in_load:8.0f}（越低=越过度扣除）")

    fig.suptitle("x2 门提前（快相已变短 => 确认窗 10→6/4s，软冻结 4→2.5/2s）", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_png = HERE / "out" / "x2提前_对比.png"
    fig.savefig(out_png, dpi=125)
    print(f"\n图已保存：{out_png}")


if __name__ == "__main__":
    main()
