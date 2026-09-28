# -*- coding: utf-8 -*-
"""同一份录制数据上对比 C++ 两档算法的 Python 抽取版：v3.4 观测器(K9) vs v6 快速稳定。

用法（在本目录下）：
    python compare_v34_v6.py [device_001_seg000.csv]

产物（写到 out/）：
    v34_vs_v6_对比.png   一张图：总量对比（输入 / v3.4-K9 / v6）+ 各自扣除量
    v34_vs_v6_结果.csv   逐帧三条曲线的总量与扣除量

口径：
    输入     = CSV 的逐通道显示值（本文件是 C++ 开 v3.4(K9) 时录的 processed_display，
               作为两条算法的公共测试输入喂进去）
    v3.4-K9  = creep_observer_k9.py（= C++ creep_observer.cpp，plan-v4 observer-k9）
    v6       = drift_v6_py.py（= C++ drift_v6_compensator.cpp，菜单实跑口径 mem_tau_s=0）
    注意：v6 的检测窗/迟滞按 ~100 Hz 标定（C++ 头注释已知边界），本录制 10 Hz，
          v6 曲线按同参数直接运行，不重新标定 —— 与 C++ 软件在此帧率下的行为一致。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_v34_on_csv import read_session_csv  # noqa: E402
from creep_observer_k9 import CreepObserverK9, Params as K9Params  # noqa: E402
from drift_v6_py import DriftV6, DriftV6Params  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    csv_name = sys.argv[1] if len(sys.argv) > 1 else "device_001_seg000.csv"
    csv_path = Path(csv_name) if os.path.isabs(csv_name) else HERE / csv_name
    if not csv_path.exists():
        print(f"[错] 找不到 {csv_path}", file=sys.stderr)
        return 3
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)

    d = read_session_csv(csv_path)
    t0, V0 = d["t"], d["values"]
    n_frames, n_ch = V0.shape
    print(f"读取 {csv_path.name}：{n_frames} 帧 × {n_ch} 通道，"
          f"时长 {t0[-1]-t0[0]:.1f} s，录制帧率 {n_frames/max(t0[-1]-t0[0],1e-9):.2f} Hz")

    # 录制是 10 Hz 抽帧，C++ 软件里两档算法都跑在 ~100 Hz 设备全帧流上。
    # v6 检测窗（0.2/0.15/0.3 s + 回溯 0.6 s）按 100 Hz 标定，10 Hz 下帧数不足
    # 结构性无法建事件（与 C++ 同参数行为一致）。为复现 C++ 链路口径，
    # 这里把录制线性插值回 100 Hz 再喂给两档算法（两算法喂同一插值流，公平对拍）。
    FS = 100.0
    n_up = int(round((t0[-1] - t0[0]) * FS)) + 1
    t = t0[0] + np.arange(n_up) / FS
    V = np.empty((n_up, n_ch))
    for k in range(n_ch):
        V[:, k] = np.interp(t, t0, V0[:, k])
    print(f"插值回 {FS:.0f} Hz：{n_up} 帧（两算法公共输入）")

    print("[1/3] 跑 v3.4 观测器（K9）…")
    k9 = CreepObserverK9(K9Params())
    Y34 = np.empty_like(V)
    for i in range(n_up):
        Y34[i] = k9.process(float(t[i]), V[i])

    print("[2/3] 跑 v6 快速稳定（PCT-fix，菜单实跑口径）…")
    v6 = DriftV6(DriftV6Params.as_run())
    Y6 = np.empty_like(V)
    for i in range(n_up):
        Y6[i] = v6.process(float(t[i]), V[i])
    print(f"      v6 诊断：形状ROM命中 {v6.shape_hits}  撤销 {v6.n_revoke}  "
          f"限幅格点 {v6.n_clamp}  PCT命中 {v6.n_pct_hits}")

    tot_in = V.sum(axis=1)
    tot_34 = Y34.sum(axis=1)
    tot_6 = Y6.sum(axis=1)
    ded_34 = tot_in - tot_34
    ded_6 = tot_in - tot_6

    # 逐帧结果 CSV
    res = out_dir / "v34_vs_v6_结果.csv"
    with res.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("t,frame,total_in,total_v34_k9,total_v6,ded_v34_k9,ded_v6\n")
        for i in range(n_up):
            f.write(f"{t[i]:.6f},{i},{tot_in[i]:.6f},{tot_34[i]:.6f},{tot_6[i]:.6f},"
                    f"{ded_34[i]:.6f},{ded_6[i]:.6f}\n")

    # ---- 一张对比图 ----
    print("[3/3] 出图…")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax1.plot(t, tot_in, lw=1.4, color="#7f8c8d", label="输入（CSV 显示总量）")
    ax1.plot(t, tot_34, lw=1.6, color="#c0392b", label="v3.4 观测器 K9（creep_observer）")
    ax1.plot(t, tot_6, lw=1.6, color="#2980b9", label="v6 快速稳定 PCT-fix（drift_v6）")
    ax1.set_ylabel("52 通道总量 (ADC)")
    ax1.set_title(f"v3.4 观测器(K9) vs v6 快速稳定 —— 同一输入逐帧对拍"
                  f"（{csv_path.name} 录制 10 Hz → 插值 100 Hz，{n_up} 帧）", loc="left")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(alpha=0.25)

    ax2.plot(t, ded_34, lw=1.5, color="#c0392b", label="K9 扣除量（输入−显示）")
    ax2.plot(t, ded_6, lw=1.5, color="#2980b9", label="v6 扣除量（输入−显示）")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("扣除量 (ADC)")
    ax2.set_xlabel("时间 (s)")
    ax2.set_title("各算法实际施加的补偿量（>0 = 显示被压低）", loc="left")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(alpha=0.25)

    note = ("口径：两算法均为 C++ 源码的 Python 逐行抽取（参数=现役默认值）；"
            "输入为 10 Hz 录制线性插值到 100 Hz 的公共流（复现 C++ 全帧链路口径）。")
    fig.text(0.01, 0.005, note, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    png = out_dir / "v34_vs_v6_对比.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    # 简报
    print(f"\n简报（时长 {t[-1]-t[0]:.1f} s）：")
    print(f"  输入总量   首 {tot_in[0]:8.1f}  末 {tot_in[-1]:8.1f}  "
          f"峰 {tot_in.max():8.1f}  谷 {tot_in.min():8.1f}")
    print(f"  v3.4-K9    首 {tot_34[0]:8.1f}  末 {tot_34[-1]:8.1f}  "
          f"峰 {tot_34.max():8.1f}  谷 {tot_34.min():8.1f}  最大扣除 {ded_34.max():7.1f}")
    print(f"  v6         首 {tot_6[0]:8.1f}  末 {tot_6[-1]:8.1f}  "
          f"峰 {tot_6.max():8.1f}  谷 {tot_6.min():8.1f}  最大扣除 {ded_6.max():7.1f}")
    print(f"\n完成：{png}")
    print(f"      {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
