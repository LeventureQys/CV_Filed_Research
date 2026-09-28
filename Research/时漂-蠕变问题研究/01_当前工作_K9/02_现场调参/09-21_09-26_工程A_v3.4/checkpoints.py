# -*- coding: utf-8 -*-
"""最终5参：加载沿后 1/3/5/15s 的漂移体检。

三个口径（都算「最终5参」的输出）：
  ① 相对输入的净漂移  = (显示升幅 − 输入升幅)：<0 = 扣多了（下漂）/>0 = 扣少了（上漂）
  ② 相对稳态平台的偏离 = 平台均值 − 该时刻显示：>0 = 该时刻低于稳态（还会往上补 = 当时下漂）
                                              <0 = 该时刻高于稳态（还会往下走 = 当时上漂）
  ③ 瞬时斜率（2s 窗中心）= 此刻显示的移动方向与速率

平台参考 = 该持载段 [t_on+60s, 段末] 的显示均值（不足则取后 40% 段）。

用法：python checkpoints.py
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

DATASETS = [
    ("长期/100539 首段",
     HERE / "长期数据" / "20260922_100539_single_device_d95573" / "device_001_seg000.csv",
     False, 78.0),
    ("抖动/20260921 首段", HERE / "device_001_seg000.csv", True, 15.8),
    ("长期/095849", HERE / "长期数据" / "20260922_095849_single_device_6cca99"
     / "device_001_seg000.csv", False, 1e9),
    ("长期/100347", HERE / "长期数据" / "20260922_100347_single_device_151a72"
     / "device_001_seg000.csv", False, 1e9),
    ("长保压/193320",
     Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\算法数据&原始数据\working"
          r"\零基线-同一荷载测试\20260919_193320_single_device_3f32c5\device_001_pre_seg0.csv"),
     False, 1e9),
]

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


def at(t, arr, tq):
    i = int(np.searchsorted(t, tq))
    i = min(max(i, 0), len(t) - 1)
    return float(arr[i])


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    rows = []
    for name, csv, up, t_stop in DATASETS:
        d = read_session_csv(csv)
        t0, V0 = d["t"], d["values"]
        if up:
            nu = int(round((t0[-1] - t0[0]) * 100.0)) + 1
            t = t0[0] + np.arange(nu) / 100.0
            V = np.stack([np.interp(t, t0, V0[:, k]) for k in range(V0.shape[1])], axis=1)
        else:
            t, V = t0, V0
        tin = V.sum(axis=1)
        out, x1, x2 = run(P5, t, V)

        step = tin.max() - np.percentile(tin, 5)
        base5 = float(np.percentile(tin, 5))
        i_on = int(np.argmax(tin > 0.2 * step + base5))
        t_on = t[i_on]
        # 该持载段的真实结束：先确认已真正受载（>80% 台阶），再找回落点（卸载）
        i_pk = next((i for i in range(i_on, len(t))
                     if tin[i] > 0.8 * step + base5), i_on)
        i_un = next((i for i in range(i_pk + 1, len(t))
                     if tin[i] < 0.3 * step + base5), len(t) - 1)
        t_end = min(t[i_un], t_on + t_stop)

        # 平台参考：段内后 40%（且必须仍在受载态）
        m_plat = (t >= t_on + 0.6 * (t_end - t_on)) & (t <= t_end) & \
                 (tin > 0.5 * step + base5)
        if m_plat.sum() < 50:
            m_plat = (tin > 0.5 * step + base5) & (t >= t_on + 20) & (t <= t_end)
        plat = float(np.mean(out[m_plat]))
        plat_in = float(np.mean(tin[m_plat]))

        print(f"\n=== {name}（加载 t={t_on:.1f}s，段末 {t_end:.1f}s，"
              f"平台参考 {plat:.0f} ADC（受载帧均值））===")
        print(f"{'Δ':>5s} {'显示':>8s} {'输入':>8s} {'已扣(升幅差)':>12s}"
              f" {'②该时刻距平台':>14s} {'③斜率':>10s} {'x1':>7s} {'x2':>7s}  方向")
        for dt_ in CKPTS:
            tq = min(t_on + dt_, t_end)
            o, i_ = at(t, out, tq), at(t, tin, tq)
            o0, i0 = at(t, out, t_on), at(t, tin, t_on)
            rise_o, rise_i = o - o0, i_ - i0
            d1 = rise_o - rise_i                 # 相对输入净漂移
            d2 = plat - o                        # 相对平台
            # 瞬时斜率（±1s 中心窗）
            ia = int(np.searchsorted(t, max(tq - 1.0, t[0])))
            ib = int(np.searchsorted(t, min(tq + 1.0, t[-1])))
            sl = (out[ib] - out[ia]) / (t[ib] - t[ia]) if t[ib] > t[ia] else 0.0
            # ② = 平台 − 该时刻显示
            #   >0 此刻低于稳态 ⇒ 后续显示会上升（上漂）
            #   <0 此刻高于稳态 ⇒ 后续显示会下降（下漂）
            direction = "此刻偏低→后续上漂" if d2 > 0 else "此刻偏高→后续下漂"
            print(f"{dt_:4.0f}s {o:8.0f} {i_:8.0f} {d1:+12.0f}"
                  f" {d2:+14.0f} {sl:+9.0f}/s "
                  f"{at(t, x1, tq):7.0f} {at(t, x2, tq):7.0f}  {direction}")
            rows.append((name, dt_, d1, d2, sl))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    names = [n for n, *_ in DATASETS]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9))
    w = 0.2
    xs = np.arange(len(names))
    for j, dt_ in enumerate(CKPTS):
        d1 = [r[2] for r in rows if r[1] == dt_]
        d2 = [r[3] for r in rows if r[1] == dt_]
        a1.bar(xs + (j - 1.5) * w, d1, w, label=f"Δ={dt_:.0f}s")
        a2.bar(xs + (j - 1.5) * w, d2, w, label=f"Δ={dt_:.0f}s")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xticks(xs); a1.set_xticklabels(names, fontsize=9)
    a1.set_ylabel("① 相对输入的净漂移 (ADC)\n<0 = 下漂（扣多了）")
    a1.set_title("最终5参：加载沿后 1/3/5/15s 的净漂移", loc="left")
    a1.legend(fontsize=9); a1.grid(alpha=0.25, axis="y")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(xs); a2.set_xticklabels(names, fontsize=9)
    a2.set_ylabel("② 相对稳态平台的偏离 (ADC)\n>0 = 当时偏低(下漂) / <0 = 当时偏高(上漂)")
    a2.set_title("最终5参：各时刻相对最终平台的位置", loc="left")
    a2.legend(fontsize=9); a2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    png = HERE / "out" / "最终5参_加载后漂移检查点.png"
    fig.savefig(png, dpi=130)
    print(f"\n图已保存：{png}")


if __name__ == "__main__":
    main()
