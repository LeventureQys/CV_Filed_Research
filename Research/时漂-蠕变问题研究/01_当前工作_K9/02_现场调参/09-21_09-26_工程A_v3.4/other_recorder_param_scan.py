# -*- coding: utf-8 -*-
"""other_recorder 参数扫描 + 归因：v3.4 观测器「后续段收敛幅度」到底被哪个参数卡住。

结论口径（数据 = other_recorder/device_001_seg000.csv，显示单位 N）：
  * 段1 = 1.06~21.4 s（阶跃到 11.7 N 后慢爬 +1.9 N）；
  * 段2 = 25.3~50.6 s（部分卸载到 7.8 N 后慢爬 +2.3 N）⇒ 本文件的「后续段」。
  * 现役默认下段2 显示仍涨 +0.296 N（输入涨 +2.310 N），即后续段收敛幅度不足。

用法：python other_recorder_param_scan.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv, CSV_PATH  # noqa: E402


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


def metrics(p, t, V, tot_in):
    out = run(p, t, V)
    m1 = (t >= 2.0) & (t <= 21.0)
    m2 = t >= 25.5
    return (float((tot_in - out)[-1]), float(out[m1][-1] - out[m1][0]),
            float(out[m2][-1] - out[m2][0]))


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    d = read_any_session_csv(CSV_PATH)
    t, V = d["t"], d["V"]
    tot_in = V.sum(axis=1)
    m1 = (t >= 2.0) & (t <= 21.0)
    m2 = t >= 25.5
    print(f"段1 输入涨 {tot_in[m1][-1] - tot_in[m1][0]:+.3f} N   "
          f"段2 输入涨 {tot_in[m2][-1] - tot_in[m2][0]:+.3f} N\n")
    print(f"{'参数':30s} {'末帧扣除':>9s} {'段1显示涨':>10s} {'段2显示涨':>10s}")

    def show(name, p):
        a, b, cc = metrics(p, t, V, tot_in)
        print(f"{name:30s} {a:9.3f} {b:+10.3f} {cc:+10.3f}")

    show("现役默认", LIVE)
    for v in (8.0, 6.0, 4.0, 2.0):
        show(f"tau_c_fast_s={v}", replace(LIVE, tau_c_fast_s=v))
    for v in (0.015, 0.02, 0.03, 0.05):
        show(f"slope_cap_frac={v}", replace(LIVE, slope_cap_frac=v))
    for v in (0.08, 0.10, 0.15):
        show(f"r_fast={v}", replace(LIVE, r_fast=v))
    for v in (0.0, 1.0, 0.5):
        show(f"slow_confirm_s={v}", replace(LIVE, slow_confirm_s=v))
    for v in (0.5, 0.0):
        show(f"ramp_slope_min={v}", replace(LIVE, ramp_slope_min=v))
    for v in (1.0, 2.0, 3.0):
        show(f"tau_slope_s={v}", replace(LIVE, tau_slope_s=v))
    for v in (50.0, 0.5, 0.1):
        show(f"edge_slope_thres={v}", replace(LIVE, edge_slope_thres=v))
    show("r_slow_max=0.60（幅度上限）", replace(LIVE, r_slow_max=0.60))
    show("soft_unfreeze_s=0.3", replace(LIVE, soft_unfreeze_s=0.3))

    # ── 归因：x2 增量被谁截掉 ──
    c = CreepObserverK9(LIVE)
    for i in range(len(t)):
        c.process(float(t[i]), V[i])
    tr = c.traces()
    p = LIVE
    e_now, xf, sl = tr["e_now"], tr["x_fast"], tr["slope"]
    tc1 = np.where(tr["t_edge"] < p.edge_boost_s, p.tau_c_fast_boost_s, p.tau_c_fast_s)
    w = np.minimum(tr["ramp_dwell"] / p.ramp_full_s, 1.0)
    tc1 = tc1 * (1.0 - w) + p.tau_c_fast_boost_s * w
    xf_prev = np.vstack([xf[:1], xf[:-1]])
    e_prev = np.maximum(np.vstack([e_now[:1], e_now[:-1]]), 0.0)
    dx1 = np.where(e_prev > 0, (p.r_fast * e_prev - xf_prev) / tc1, -xf_prev / p.tau_r_fast_s)
    raw, cap = sl - dx1, p.slope_cap_frac * np.maximum(e_prev, 1.0)
    on = e_prev > 0
    print("\n=== x2 增量归因（现役默认；数值 = Σrate×dt 的代理，单位 显示单位）===")
    print(f"  未截断 Σraw×0.01 = {raw[on].sum() * 0.01:+.3f}   "
          f"限幅后 Σclip×0.01 = {np.clip(raw, -cap, cap)[on].sum() * 0.01:+.3f}")
    print(f"  被 slope_cap_frac 截掉的量 = "
          f"{(np.clip(raw, -cap, cap) - raw)[on].sum() * 0.01:+.3f}"
          f"（命中限幅 {int((np.abs(raw[on]) > cap[on]).sum())}/{int(on.sum())} 通道帧）")
    print(f"  r_slow_max 上限命中次数 = "
          f"{int((tr['x_slow'] > p.r_slow_max * np.maximum(e_now, 1.0) * 0.999).sum())}"
          f" / {tr['x_slow'].size}（0 ⇒ 幅度上限完全不起作用）")
    hit = (np.abs(raw) > cap) & (e_prev > 0)
    pos = hit & (raw > 0)
    print(f"  命中限幅的 {int(hit.sum())} 个通道帧里 raw>0（想增 x2）占 "
          f"{float((hit & (raw > 0)).sum()) / max(int(hit.sum()), 1) * 100:.1f}%，"
          f"其中 slope>0 占 {float((sl[pos] > 0).mean()) * 100:.1f}%、"
          f"x1 回撤贡献占 {float((dx1[pos] < 0).mean()) * 100:.1f}%"
          f" ⇒ 卡住的是输入自身的上升速率，不是 x1")
    ded0 = float((tot_in - run(replace(LIVE, r_fast=0.0), t, V))[-1])
    print(f"  关掉 x1（r_fast=0）末帧扣除 = {ded0:.3f} vs 现役 "
          f"{float((tot_in - run(LIVE, t, V))[-1]):.3f}"
          f" ⇒ 本数据补偿几乎全部来自 x2，x1 只贡献 ~0.1 N")
    print(f"  e 最大值 = {e_now.max():.3f} ⇒ max(e,1.0) 恒为 1.0，"
          f"slope_gate/cap 退化成绝对阈值 0.05 / 0.01（按 N 计）")
    print(f"  逐通道 |slope| 最大 = {np.abs(sl).max():.3f} ≪ edge_slope_thres=60 "
          f"⇒ H3 沿检测/前馈（τc1→2 s）本会话一次未触发")


if __name__ == "__main__":
    main()
