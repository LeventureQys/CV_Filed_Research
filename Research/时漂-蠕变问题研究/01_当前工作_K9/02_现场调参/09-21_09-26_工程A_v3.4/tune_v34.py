# -*- coding: utf-8 -*-
"""K9 观测器抗下漂调参实验（v2：指标按本录制的实际工况划分）。

机理定位（本数据实测）：
  * 慢态 x2 因负载抖动（load_dwell 到不了 10 s）几乎不参与，扣除 ≈ 全部 x1；
  * 下漂主体 = x1 以 τc_fast=12 s 慢爬向 r_fast·e（=10% 载荷）的长尾：
    加载后 11 s 里显示又沉了 ~800 ADC，且 35 s 才沉完 —— 既慢又过扣；
  * 回撤后 x2/残留以 τr_slow=150 s 泄放，太慢。

参数杠杆：
  tau_c_fast_s ↓    x1 更快到位 ⇒ 显示更早钉平（下漂收敛更快，收敛速度不受损反而更快）
  r_fast ↓          稳态扣除水位从 10% 降下来（过收敛幅度变小）
  tau_r_slow_s / tau_r_slow_idle_s ↓  回撤后残留补偿释放更快

指标（在本录制的事件结构上计算）：
  conv_s    首个持载段（onset→第一次卸载）内，显示斜率最后一次超 25 ADC/s 的时刻
            距 onset 的秒数 —— 显示"钉平"用时，越小越好
  sink_hold 首个持载段内显示的额外下沉（onset+2s → 卸载前 0.5s，ADC，正=下沉）
  slope_tail 末段 43.5~46 s 显示线性斜率（ADC/s，理想 ≈0）
  ded_level 稳态扣除水位 = 持载段 12~16 s 平均扣除 / 载荷台阶（%，过收敛程度）
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

FS = 100.0


def load_input():
    d = read_session_csv(HERE / "device_001_seg000.csv")
    t0, V0 = d["t"], d["values"]
    n_up = int(round((t0[-1] - t0[0]) * FS)) + 1
    t = t0[0] + np.arange(n_up) / FS
    V = np.stack([np.interp(t, t0, V0[:, k]) for k in range(V0.shape[1])], axis=1)
    return t, V


def run(p: Params, t, V):
    c = CreepObserverK9(p)
    Y = np.empty_like(V)
    for i in range(len(t)):
        Y[i] = c.process(float(t[i]), V[i])
    return Y


def smooth(x, win=1.0, fs=FS):
    k = max(1, int(win * fs))
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def metrics(t, tot_in, tot_out):
    peak = tot_in.max()
    loaded = tot_in > 0.2 * peak
    onset = int(np.argmax(loaded))

    def win_mean(a, t0, t1):
        m = (t >= t0) & (t < t1)
        return float(np.mean(a[m]))

    # surge：加载沿过后 0.5s（boost 冲击起点）→ 9~11s 的额外下沉（x1 过激的直接度量）
    surge = win_mean(tot_out, t[onset] + 0.5, t[onset] + 1.0) - \
            win_mean(tot_out, t[onset] + 9.0, t[onset] + 11.0)
    # settle：持载段内显示斜率（0.5s 平滑）最后一次超 20 ADC/s 的时刻 − onset
    y = smooth(tot_out, 0.5)
    dy = np.gradient(y, t)
    seg = (t >= t[onset] + 1.0) & (t <= 15.5)
    bad = np.where(np.abs(dy[seg]) > 20.0)[0]
    settle_s = float(t[bad[-1]] - t[onset]) if len(bad) else 0.0
    m_lv = (t >= 12) & (t <= 16)
    step = float(np.mean(tot_in[m_lv])) - float(np.percentile(tot_in, 5))
    ded_level = float(np.mean((tot_in - tot_out)[m_lv])) / step * 100.0
    return {"settle_s": settle_s, "surge": surge, "ded_level": ded_level}


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, V = load_input()
    tot_in = V.sum(axis=1)
    print(f"输入：{len(t)} 帧 @100 Hz，时长 {t[-1]:.1f} s，载荷台阶 ≈"
          f"{np.mean(tot_in[(t>=12)&(t<=16)])-np.percentile(tot_in,5):.0f} ADC\n")

    base = Params()
    cases: list[tuple[str, Params]] = [
        ("baseline（K9 现役）", base),
        # —— x1 降火：幅度 ——
        ("r_fast .10→.08", replace(base, r_fast=0.08)),
        ("r_fast .10→.06", replace(base, r_fast=0.06)),
        ("r_fast .10→.05", replace(base, r_fast=0.05)),
        # —— x1 降火：沿后 boost 削弱（τc_boost 2→3/4，窗 2→1 s）——
        ("r_fast.06 + τc_boost 3", replace(base, r_fast=0.06, tau_c_fast_boost_s=3.0)),
        ("r_fast.06 + boost窗 2→1s", replace(base, r_fast=0.06, edge_boost_s=1.0)),
        ("r_fast.06 + τc_boost4 + 窗1s", replace(base, r_fast=0.06, tau_c_fast_boost_s=4.0,
                                                 edge_boost_s=1.0)),
        # —— x1 长尾收短（下漂收敛更快），与降火组合 ——
        ("r_fast.06 + τc 12→8", replace(base, r_fast=0.06, tau_c_fast_s=8.0)),
        ("推荐: r.06 τc_boost3 窗1s τc8",
         replace(base, r_fast=0.06, tau_c_fast_boost_s=3.0, edge_boost_s=1.0,
                 tau_c_fast_s=8.0)),
        ("保守推荐: r.07 τc_boost3 窗1s τc8 τr40/idle4",
         replace(base, r_fast=0.07, tau_c_fast_boost_s=3.0, edge_boost_s=1.0,
                 tau_c_fast_s=8.0, tau_r_slow_s=40.0, tau_r_slow_idle_s=4.0)),
    ]

    print(f"{'方案':34s} {'settle_s':>8s} {'surge(过激下沉)':>14s} {'ded_%':>7s}")
    runs = {}
    for name, p in cases:
        Y = run(p, t, V)
        runs[name] = Y.sum(axis=1)
        m = metrics(t, tot_in, runs[name])
        print(f"{name:36s} {m['settle_s']:7.1f}s {m['surge']:+12.0f} {m['ded_level']:6.1f}%")

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    y_base = runs["baseline（K9 现役）"]
    y_r06 = runs["r_fast .10→.06"]
    y_soft = runs["r_fast.06 + boost窗 2→1s"]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(15, 13), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    a1.plot(t, tot_in, color="#7f8c8d", lw=1.2, label="输入总量（CSV 插值 100 Hz）")
    a1.plot(t, y_base, color="#c0392b", lw=1.5, label="K9 现役（r_fast=0.10）")
    a1.plot(t, y_r06, color="#27ae60", lw=1.6, label="推荐：r_fast=0.06（其余不动）")
    a1.plot(t, y_soft, color="#2980b9", lw=1.4, ls="--",
            label="更软：r_fast=0.06 + boost窗 2→1s")
    a1.set_ylabel("52 通道总量 (ADC)")
    a1.set_title("K9 x1 降火调参：现役 vs 推荐（同一输入，加载沿后 0.5~10s 的额外下沉"
                 " +803 → −16 ADC）", loc="left")
    a1.legend(loc="upper right")
    a1.grid(alpha=0.25)

    a2.plot(t, tot_in - y_base, color="#c0392b", lw=1.4, label="现役扣除量")
    a2.plot(t, tot_in - y_r06, color="#27ae60", lw=1.4, label="推荐扣除量")
    a2.plot(t, tot_in - y_soft, color="#2980b9", lw=1.2, ls="--", label="更软扣除量")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel("扣除量 (ADC)")
    a2.legend(loc="upper right")
    a2.grid(alpha=0.25)

    a3.plot(t, y_base - y_r06, color="#8e44ad", lw=1.4)
    a3.axhline(0, color="k", lw=0.6)
    a3.set_ylabel("显示差（现役−推荐）")
    a3.set_xlabel("时间 (s)")
    a3.set_title(">0 = 推荐方案显示更高（少扣）；长尾即现役 x1 的下漂长尾", loc="left")
    a3.grid(alpha=0.25)

    fig.tight_layout()
    out = HERE / "out" / "k9_调参对比.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
