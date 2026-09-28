# -*- coding: utf-8 -*-
"""在实机录制的算法前流（pre_seg0）上对比 K9 现役参数 vs 新参数（r_fast=0.06）。

数据：零基线-同一荷载测试 20260919_193320（355.8 s，原生 ~100 Hz，52 通道）。
曲线：输入（算法前） / K9 现役(r=0.10) / K9 新参(r=0.06) / 当天 C++ 录制结果(observer-v3)。

用法：python compare_params_on.py
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

SESSION = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\算法数据&原始数据"
               r"\working\零基线-同一荷载测试\20260919_193320_single_device_3f32c5")


def run_k9(p: Params, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None     # 35.7k 帧 × 14 键的 trace 不留，省 ~200MB
    Y = np.empty_like(V)
    for i in range(len(t)):
        Y[i] = c.process(float(t[i]), V[i])
    return Y


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    d_pre = read_session_csv(SESSION / "device_001_pre_seg0.csv")
    t, V = d_pre["t"], d_pre["values"]
    d_cpp = read_session_csv(SESSION / "device_001_seg000.csv")
    tc, Ycpp = d_cpp["t"], d_cpp["values"]
    n, dur = len(t), t[-1] - t[0]
    print(f"输入 pre_seg0：{n} 帧 × {V.shape[1]} 通道，时长 {dur:.1f} s，"
          f"原生帧率 {n/dur:.1f} Hz（不插值，与 C++ 链路同口径）")

    print("[1/2] K9 现役参数（r_fast=0.10）…")
    Y_base = run_k9(Params(), t, V)
    print("[2/2] K9 新参数（r_fast=0.06，其余不动）…")
    Y_new = run_k9(replace(Params(), r_fast=0.06), t, V)

    tin = V.sum(axis=1)
    tb = Y_base.sum(axis=1)
    tn = Y_new.sum(axis=1)
    tc_ = Ycpp.sum(axis=1)

    print(f"\n简报：")
    print(f"  输入      峰 {tin.max():9.0f}  谷 {tin.min():7.0f}")
    print(f"  K9现役    末 {tb[-1]:9.0f}  最大扣除 {np.max(tin - tb):8.0f}")
    print(f"  K9新参    末 {tn[-1]:9.0f}  最大扣除 {np.max(tin - tn):8.0f}")
    print(f"  C++当天   末 {tc_[-1]:9.0f}  最大扣除 {np.max(tin[:len(tc_)] - tc_):8.0f}")

    # 持载段（>20% 峰值）里每段的显示下沉：现役 vs 新参
    loaded = tin > 0.2 * tin.max()
    segs = []
    i = 0
    while i < n:
        if loaded[i]:
            j = i
            while j + 1 < n and loaded[j + 1]:
                j += 1
            if j - i > 300:          # >3 s 的受载段
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    print(f"\n受载段 {len(segs)} 个（>3s）：下沉量 = 段内显示(首10%) − 显示(末10%)")
    print(f"  {'段起止 (s)':>18s} {'现役下沉':>9s} {'新参下沉':>9s} {'C++当天下沉':>10s}")
    for (i, j) in segs:
        k10 = max(1, (j - i) // 10)
        def sink(x):
            return float(np.mean(x[i:i + k10]) - np.mean(x[j - k10:j]))
        print(f"  {t[i]:7.1f}~{t[j]:7.1f} {sink(tb):+9.0f} {sink(tn):+9.0f} "
              f"{sink(tc_[:n]):+10.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 10.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.1, 1.0]})
    a1.plot(t, tin, color="#7f8c8d", lw=1.0, label="输入（算法前 pre_seg0，原生 100 Hz）")
    a1.plot(tc, tc_, color="#8e44ad", lw=1.0, alpha=0.8,
            label="当天 C++ 录制结果（observer-v3，2026-09-19）")
    a1.plot(t, tb, color="#c0392b", lw=1.3, label="K9 现役参数（r_fast=0.10）")
    a1.plot(t, tn, color="#27ae60", lw=1.3, label="K9 新参数（r_fast=0.06）")
    a1.set_ylabel("52 通道总量 (ADC)")
    a1.set_title("零基线-同一荷载测试 20260919_193320：K9 现役 vs 新参数（同一算法前输入）",
                 loc="left")
    a1.legend(loc="upper right", fontsize=10)
    a1.grid(alpha=0.25)

    a2.plot(t, tin - tb, color="#c0392b", lw=1.1, label="现役扣除量")
    a2.plot(t, tin - tn, color="#27ae60", lw=1.1, label="新参扣除量")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel("扣除量 (ADC)")
    a2.set_xlabel("时间 (s)")
    a2.legend(loc="upper right", fontsize=10)
    a2.grid(alpha=0.25)

    fig.tight_layout()
    out = HERE / "out" / "193320_参数对比.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\n图已保存：{out}")


if __name__ == "__main__":
    main()
