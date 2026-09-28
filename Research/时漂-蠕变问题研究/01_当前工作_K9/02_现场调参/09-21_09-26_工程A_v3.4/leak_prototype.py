# -*- coding: utf-8 -*-
"""候选新参数 `tau_leak_s`（慢态泄漏时间常数）原型验证。

动机（实测）：x2 是纯积分器，平台期斜率估计的残余正偏（本例 ≈ +0.9 ADC/s）被无限积分，
300 s 时总扣除已达真实蠕变的 115~118 %，显示停在弹性电平下方 1.5~1.8 % 且仍在缓慢下沉。
把 x2 改成**带泄漏的积分器**后：
    dx2 = clamp(slope − dx1_rate − x2/τ_leak, ±cap)
平台期残余偏置 b 只能把 x2 抬到 b·τ_leak（有界）；对饱和型蠕变（总量 C、时间常数 τ_creep）
稳态只损失 τ_creep/(τ_leak + τ_creep)：τ_leak=120 s、τ_creep=11.2 s ⇒ 少 8.5 %。

本文件的 `CreepObserverLeak` 与 `creep_observer_k9.py`（= C++ creep_observer.cpp）逐帧一致，
只多一个泄漏项，用于验证是否值得落地到 C++。

用法：python leak_prototype.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402

NEW = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法"
           r"\20260924_155543_single_device_1201c1\device_001_seg000.csv")
LONG = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\01_当前工作_K9\02_现场调参\09-21_09-26_工程A_v3.4\长期数据")


@dataclass
class LeakParams(Params):
    tau_leak_s: float = 0.0


class CreepObserverLeak(CreepObserverK9):
    """K9 + 泄漏项；其余逐帧完全一致。"""

    def __init__(self, params: LeakParams | None = None) -> None:
        self.p = params or LeakParams()
        super().__init__(params)

    def process(self, timestamp_s: float, values: np.ndarray) -> np.ndarray:
        p = self.p
        tau_leak = float(getattr(p, "tau_leak_s", 0.0))
        if tau_leak <= 0.0:
            return super().process(timestamp_s, values)

        v = np.array(values, dtype=np.float64, copy=True)
        n = v.size
        if n <= 0:
            return v
        if self._n != n:
            self._reset_for(n)

        if self._first:
            return super().process(timestamp_s, values)

        dt = float(timestamp_s) - self._last_ts
        self._last_ts = float(timestamp_s)
        dt = min(dt, 0.1) if dt > 0.0 else 0.0

        bypass = False
        if dt > 0.0:
            total = float(np.sum(v))
            if np.isfinite(total):
                alpha = min(dt / p.bypass_base_tau_s, 1.0)
                release = max(p.bypass_release_frac * self.total_baseline,
                              self.total_baseline + p.bypass_noise_sigma * self.total_noise)
                engage = max(p.bypass_engage_frac * self.total_baseline,
                             self.total_baseline + (p.bypass_noise_sigma + 1.0) * self.total_noise)
                if total < engage:
                    self.total_baseline += alpha * (total - self.total_baseline)
                    self.total_noise += alpha * (abs(total - self.total_baseline) - self.total_noise)
                    if self.total_noise < 0.0:
                        self.total_noise = 0.0
                bypass = bool(total < release)

        if dt > 0.0:
            y_decay = float(np.exp(-dt / p.y_max_tau_s))
            y = v - self.zero
            slope = (v - self.v_lp) / p.tau_slope_s
            e_now = np.maximum(y - self.x_fast - self.x_slow, 0.0)
            self._slope = slope
            self._e_now = e_now

            for k in range(n):
                y0 = v[k] - self.zero[k]
                if y0 < self.y_floor[k]:
                    self.y_floor[k] = y0
                else:
                    self.y_floor[k] += (dt / p.y_floor_tau_s) * (y0 - self.y_floor[k])
                self.y_max[k] = max(self.y_max[k] * y_decay, max(y0, 0.0))
                span = max(self.y_max[k] - self.y_floor[k], 1.0)
                idle_k = (y[k] - self.y_floor[k]) < p.idle_frac * span
                self.load_dwell[k] = 0.0 if idle_k else self.load_dwell[k] + dt
                if idle_k:
                    self.zero[k] += (dt / p.tau_zero_s) * (v[k] - self.zero[k])

                self.t_edge[k] += dt
                if slope[k] > p.edge_slope_thres and \
                        self.t_edge[k] > max(p.edge_refract_s, p.edge_boost_s):
                    self.t_edge[k] = 0.0

                ramp_on = (slope[k] > p.ramp_slope_min) and (slope[k] <= p.edge_slope_thres)
                self.ramp_dwell[k] = self.ramp_dwell[k] + dt if ramp_on else 0.0

                if abs(slope[k]) > p.edge_slope_thres:
                    self.load_dwell[k] = 0.0

                tc1 = p.tau_c_fast_boost_s \
                    if (p.edge_boost_s > 0.0 and self.t_edge[k] < p.edge_boost_s) \
                    else p.tau_c_fast_s
                if p.ramp_full_s > 0.0:
                    w = min(self.ramp_dwell[k] / p.ramp_full_s, 1.0)
                    tc1 = tc1 * (1.0 - w) + p.tau_c_fast_boost_s * w

                dx1_rate = (p.r_fast * e_now[k] - self.x_fast[k]) / tc1 \
                    if e_now[k] > 0.0 else -self.x_fast[k] / p.tau_r_fast_s
                self.x_fast[k] = max(self.x_fast[k] + dx1_rate * dt, 0.0)
                self.v_lp[k] += (dt / p.tau_slope_s) * (v[k] - self.v_lp[k])

                e = max(y[k] - self.x_fast[k] - self.x_slow[k], 0.0)
                if e > 0.0:
                    base = max(e, 1.0)
                    cap = p.slope_cap_frac * base
                    soft_ok = (1.0 - float(np.exp(-self.t_edge[k] / p.soft_unfreeze_s))) > 0.5
                    dwell_ok = self.load_dwell[k] >= p.slow_confirm_s
                    if dwell_ok and soft_ok and abs(slope[k]) < p.slope_gate_frac * base:
                        dx2 = slope[k] - dx1_rate - self.x_slow[k] / tau_leak   # ← 唯一差异
                        self.x_slow[k] += dt * min(max(dx2, -cap), cap)
                    hi = p.r_slow_max * base
                    if self.x_slow[k] > hi:
                        self.x_slow[k] = hi
                    if self.x_slow[k] < 0.0:
                        self.x_slow[k] = 0.0
                if e <= 0.0 or idle_k:
                    tau_r2 = p.tau_r_slow_idle_s \
                        if (idle_k and p.tau_r_slow_idle_s > 0.0) else p.tau_r_slow_s
                    self.x_slow[k] -= (dt / tau_r2) * self.x_slow[k]
                    if self.x_slow[k] < 0.0:
                        self.x_slow[k] = 0.0

                if p.hold_eps > 0.0:
                    slope_a = (v[k] - self.v_fast_lp[k]) / p.hold_tau_s
                    self.v_fast_lp[k] += (dt / p.hold_tau_s) * (v[k] - self.v_fast_lp[k])
                    d_des = self.x_fast[k] + self.x_slow[k]
                    a_up = self.applied[k] + dt * max(slope_a + p.hold_eps, 0.0)
                    a_dn = self.applied[k] + dt * (slope_a - p.hold_eps)
                    a = min(d_des, a_up) if d_des > self.applied[k] else max(d_des, a_dn)
                    if e > 0.0:
                        a = min(max(a, 0.0), max(d_des, 0.0))
                    else:
                        a = d_des
                    self.applied[k] = a

        if bypass:
            self.applied = self.x_fast + self.x_slow
            out = v.copy()
        elif p.hold_eps > 0.0:
            out = v - self.applied
        else:
            out = v - self.x_fast - self.x_slow
        self._trace_frame(v, out, bypass)
        return out


def run(cls, p, t, V):
    c = cls(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


def base_params(**kw):
    d = dict(r_fast=LIVE.r_fast, tau_c_fast_s=LIVE.tau_c_fast_s,
             slow_confirm_s=LIVE.slow_confirm_s, soft_unfreeze_s=LIVE.soft_unfreeze_s,
             slope_cap_frac=LIVE.slope_cap_frac, tau_r_slow_idle_s=LIVE.tau_r_slow_idle_s,
             tau_r_fast_s=LIVE.tau_r_fast_s)
    d.update(kw)
    return LeakParams(**d)


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # ── A) 合成 300 s（这份传感器的蠕变形态，真实蠕变 2962 ADC）──
    syn = np.load(HERE / "out" / "synth_converge.npz")
    tt, tin_s = syn["t"], syn["tin"]
    creep = syn["creep"]
    E = float(syn["E"])
    # 用通道数重建合成矩阵：这里只做总量口径的评估，故按"单通道承载总值"重放不合适，
    # 改用 52 通道均分（总量等价；算法为逐通道，均分只影响门限的逐通道量级）
    n_ch = 52
    Vs = np.repeat((tin_s / n_ch)[:, None], n_ch, axis=1)
    print("=== A) 合成 300 s（总蠕变 2962 ADC；载荷 %.0f）===" % E)
    print(f"{'方案':30s} {'err@30s':>9s} {'err@60s':>9s} {'err@120s':>9s} {'err@300s':>9s} "
          f"{'末段斜率':>8s} {'末扣除/蠕变':>12s}")
    arms = [("τc1=2 cap=0.010（无泄漏）", base_params(tau_c_fast_s=2.0)),
            ("τc1=2 cap=0.010 leak60", base_params(tau_c_fast_s=2.0, tau_leak_s=60.0)),
            ("τc1=2 cap=0.010 leak120", base_params(tau_c_fast_s=2.0, tau_leak_s=120.0)),
            ("τc1=2 cap=0.010 leak300", base_params(tau_c_fast_s=2.0, tau_leak_s=300.0)),
            ("τc1=2 cap=0.010 r_slow_max=0.06",
             base_params(tau_c_fast_s=2.0, r_slow_max=0.06) if False else None)]
    for tag, p in arms:
        if p is None:
            continue
        out = run(CreepObserverLeak, p, tt, Vs)
        err = out - E
        m = [int(np.searchsorted(tt, x)) for x in (30.0, 60.0, 120.0, len(tt) * 0.01 - 0.01)]
        m[-1] = len(tt) - 1
        tail = slice(len(tt) - 3000, len(tt))
        k = float(np.polyfit(tt[tail], out[tail], 1)[0])
        print(f"{tag:30s} " + "".join(f"{err[i]:+9.0f}" for i in m)
              + f" {k:+8.2f} {(tin_s - out)[-1]:6.0f}/{creep[-1]:.0f}")

    # ── B) 两份 142 s 长保压：泄漏能否止住持续下漂 ──
    for name in ("20260922_095849_single_device_6cca99", "20260922_100118_single_device_2113fb"):
        d = read_any_session_csv(LONG / name / "device_001_seg000.csv")
        t, V = d["t"], d["V"]
        tin = V.sum(axis=1)
        print(f"\n=== B) {name}（142 s 长保压；末 30 s 显示斜率，负 = 仍在下漂）===")
        for tag, p in (("现役 cap0.010", base_params()),
                       ("cap0.010 leak60", base_params(tau_leak_s=60.0)),
                       ("cap0.010 leak120", base_params(tau_leak_s=120.0)),
                       ("cap0.010 leak300", base_params(tau_leak_s=300.0))):
            out = run(CreepObserverLeak, p, t, V)
            m = t >= t[-1] - 30.0
            k = float(np.polyfit(t[m], out[m], 1)[0])
            print(f"  {tag:20s} 末30s斜率 {k:+7.2f} ADC/s   末扣除 {float((tin - out)[-1]):+7.0f}")

    # ── C) 18 s 实测会话：泄漏对"跟踪能力"的代价 ──
    d = read_any_session_csv(NEW)
    t, V = d["t"], d["V"]
    tin = V.sum(axis=1)
    print("\n=== C) 20260924_155543（18 s 实测；末值相对拟合 E=29278）===")
    for tag, p in (("现役", base_params()),
                   ("leak60", base_params(tau_leak_s=60.0)),
                   ("leak120", base_params(tau_leak_s=120.0)),
                   ("leak300", base_params(tau_leak_s=300.0))):
        out = run(CreepObserverLeak, p, t, V)
        print(f"  {tag:10s} 末值−E {out[-1] - 29278:+8.1f}   末扣除 {float((tin - out)[-1]):+7.0f}")


if __name__ == "__main__":
    main()
