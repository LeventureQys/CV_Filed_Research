# -*- coding: utf-8 -*-
"""12 新参数验证：把「慢态门限/上限里的硬编码下限 max(e,1)」换成可配的 e_floor，
   看它对「无弹性台阶型」传感器（拇指/手掌）有多大收益。

动机（来自 11_ceiling 的实机证据）
--------------------------------
1. x2 被钳在 r_slow_max·base，base = max(e,1)；而 e = y − x1 − x2。
   当弹性电平≈0（指纹/掌纹从加载起读数就一路爬升）时 e ≈ 累计蠕变，
   ⇒ 可扣除比例上限 = r/(1+r)：默认 r=0.35 只能扣 26%，实测正好 16~27%。
   要把蠕变扣到 90% 需要 r ≈ 9，远超声画面上限 0.600。
2. 速率门/积分上限同样乘在 base 上：早期 e≈0 时 base=1，
   门限退化成 0.05 ADC/s、上限 0.01 ADC/s 的绝对值，
   于是「读数一开始就在爬」的那一段（占蠕变的 60~80%）根本进不了 x2。

本脚本给观测器加一个参数 e_floor（默认 1.0 = 与现役 C++ 完全等价），
把 base = max(e, 1.0) 改成 base = max(e, e_floor)，逐帧比对收益。

用法：python 12_variant.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace, fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from creep_observer_k9 import Params  # noqa: E402
from sensor_common import LIVE, OUT_DIR, SENSORS, load  # noqa: E402
from sweep_lib import PREP  # noqa: E402


@dataclass
class ParamsV(Params):
    e_floor: float = 1.0          # ★ 新参数：慢态门限/上限/幅度的绝对下限（ADC，逐通道）


class ObserverV:
    """creep_observer_k9 的等价实现 + e_floor（其余逐帧顺序完全一致）。"""

    def __init__(self, p: ParamsV):
        self.p = p
        self._n = 0
        self._first = True
        self._last_ts = 0.0

    def process(self, ts: float, values: np.ndarray) -> np.ndarray:
        v = np.array(values, dtype=np.float64, copy=True)
        n = v.size
        p = self.p
        if self._n != n:
            self._n = n
            self._first = True
            z = np.zeros(n)
            self.x_fast, self.x_slow, self.applied = z.copy(), z.copy(), z.copy()
            self.load_dwell, self.ramp_dwell = z.copy(), z.copy()
            self.t_edge = np.full(n, 1e6)
            self.v_lp, self.zero, self.y_max = z.copy(), z.copy(), z.copy()
            self.v_fast_lp, self.y_floor = z.copy(), z.copy()
            self.total_baseline = self.total_noise = 0.0
        if self._first:
            self._first = False
            self._last_ts = float(ts)
            self.zero = v.copy()
            self.y_max = np.zeros(n)
            self.v_lp, self.v_fast_lp, self.y_floor = v.copy(), v.copy(), v.copy()
            self.load_dwell = np.zeros(n)
            self.ramp_dwell = np.zeros(n)
            self.t_edge = np.full(n, 1e6)
            self.applied = np.zeros(n)
            tot = float(np.sum(v))
            self.total_baseline = tot if np.isfinite(tot) else 0.0
            self.total_noise = 0.0
            return v - self.applied
        dt = float(ts) - self._last_ts
        self._last_ts = float(ts)
        dt = min(dt, 0.1) if dt > 0 else 0.0

        bypass = False
        if dt > 0:
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
                    self.total_noise = max(self.total_noise, 0.0)
                bypass = bool(total < release)

        if dt > 0:
            y_decay = float(np.exp(-dt / p.y_max_tau_s))
            y = v - self.zero
            slope = (v - self.v_lp) / p.tau_slope_s
            e_now = np.maximum(y - self.x_fast - self.x_slow, 0.0)
            for k in range(n):
                y0 = v[k] - self.zero[k]
                if y0 < self.y_floor[k]:
                    self.y_floor[k] = y0
                else:
                    self.y_floor[k] += (dt / p.y_floor_tau_s) * (y0 - self.y_floor[k])
                self.y_max[k] = max(self.y_max[k] * y_decay, max(y0, 0.0))
                span = max(self.y_max[k] - self.y_floor[k], p.e_floor)
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
                    if (p.edge_boost_s > 0 and self.t_edge[k] < p.edge_boost_s) \
                    else p.tau_c_fast_s
                if p.ramp_full_s > 0:
                    w = min(self.ramp_dwell[k] / p.ramp_full_s, 1.0)
                    tc1 = tc1 * (1.0 - w) + p.tau_c_fast_boost_s * w
                dx1_rate = (p.r_fast * e_now[k] - self.x_fast[k]) / tc1 \
                    if e_now[k] > 0 else -self.x_fast[k] / p.tau_r_fast_s
                self.x_fast[k] = max(self.x_fast[k] + dx1_rate * dt, 0.0)
                self.v_lp[k] += (dt / p.tau_slope_s) * (v[k] - self.v_lp[k])
                e = max(y[k] - self.x_fast[k] - self.x_slow[k], 0.0)
                span_k = max(self.y_max[k] - self.y_floor[k], p.e_floor)
                idle_k2 = (y[k] - self.y_floor[k]) < p.idle_frac * span_k
                if e > 0:
                    base = max(e, p.e_floor)              # ★ 唯一改动点
                    cap = p.slope_cap_frac * base
                    soft_ok = (1.0 - float(np.exp(-self.t_edge[k] / p.soft_unfreeze_s))) > 0.5
                    dwell_ok = self.load_dwell[k] >= p.slow_confirm_s
                    if dwell_ok and soft_ok and abs(slope[k]) < p.slope_gate_frac * base:
                        dx2 = min(max(slope[k] - dx1_rate, -cap), cap)
                        self.x_slow[k] += dt * dx2
                    hi = p.r_slow_max * base
                    self.x_slow[k] = min(self.x_slow[k], hi)
                    self.x_slow[k] = max(self.x_slow[k], 0.0)
                if e <= 0 or idle_k2:
                    tau_r2 = p.tau_r_slow_idle_s \
                        if (idle_k2 and p.tau_r_slow_idle_s > 0) else p.tau_r_slow_s
                    self.x_slow[k] -= (dt / tau_r2) * self.x_slow[k]
                    self.x_slow[k] = max(self.x_slow[k], 0.0)
                if p.hold_eps > 0:
                    slope_a = (v[k] - self.v_fast_lp[k]) / p.hold_tau_s
                    self.v_fast_lp[k] += (dt / p.hold_tau_s) * (v[k] - self.v_fast_lp[k])
                    d_des = self.x_fast[k] + self.x_slow[k]
                    a_up = self.applied[k] + dt * max(slope_a + p.hold_eps, 0.0)
                    a_dn = self.applied[k] + dt * (slope_a - p.hold_eps)
                    a = min(d_des, a_up) if d_des > self.applied[k] else max(d_des, a_dn)
                    a = min(max(a, 0.0), max(d_des, 0.0)) if e > 0 else d_des
                    self.applied[k] = a
        if bypass:
            self.applied = self.x_fast + self.x_slow
            return v
        if p.hold_eps > 0:
            return v - self.applied
        return v - self.x_fast - self.x_slow


def run(p: ParamsV, t, V):
    o = ObserverV(p)
    n = len(t)
    return np.array([o.process(float(t[i]), V[i]).sum() for i in range(n)])


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    live = ParamsV(**{k: getattr(LIVE, k) for k in Params.names()})
    # 先确认 e_floor=1 时与 k9 完全一致
    from creep_observer_k9 import CreepObserverK9
    d = load(PREP["左拇指指腹/d1"]["path"].split("data\\", 1)[-1].replace("\\", "/"))
    t = d["t"] - d["t"][0]
    V = d["V"] - d["V"][0]
    a = run(live, t, V)
    c = CreepObserverK9(LIVE)
    c._trace_frame = lambda *x, **y: None
    b = np.array([c.process(float(t[i]), V[i]).sum() for i in range(len(t))])
    print(f"等价性检查（e_floor=1）：max|Δ| = {np.max(np.abs(a - b)):.3e} ADC\n")

    variants = [
        ("现役默认", live),
        ("e_floor=1 + 放宽 r_slow_max=5 (旧口径)", replace(live, r_slow_max=5.0)),
        ("仅 e_floor=20 (r 仍 0.35)", replace(live, e_floor=20.0)),
        ("仅 e_floor=100", replace(live, e_floor=100.0)),
        ("e_floor=100 + r_slow_max=1", replace(live, e_floor=100.0, r_slow_max=1.0)),
        ("e_floor=100 + r_slow_max=3", replace(live, e_floor=100.0, r_slow_max=3.0)),
        ("e_floor=50 + r_slow_max=3 + cap=0.05",
         replace(live, e_floor=50.0, r_slow_max=3.0, slope_cap_frac=0.05)),
    ]
    out = {}
    sensors = sys.argv[1:] or ["左拇指指腹", "右手掌"]
    print(f"{'传感器':<12s} {'变体':<34s} {'扣除%':>7s} {'残漂%':>7s} {'过扣%':>7s}")
    for sensor in sensors:
        for nm, p in variants:
            deds, resids, overs = [], [], []
            for tag in SENSORS[sensor]["sessions"]:
                info = PREP[f"{sensor}/{tag}"]
                dd = load(info["path"].split("data\\", 1)[-1].replace("\\", "/"))
                tt = dd["t"] - dd["t"][0]
                VV = dd["V"] - dd["V"][0]
                tin = VV.sum(axis=1)
                o = run(p, tt, VV)
                Esum, creep = info["Esum_channels"], max(info["creep_total"], 1.0)
                deds.append(float((tin - o)[-1]) / creep * 100)
                resids.append(float(o[-1] - Esum) / creep * 100)
                overs.append(-float(np.min(o[tt >= info["t0"] + 5] - Esum)) / creep * 100)
            m = (np.mean(deds), np.mean(resids), np.mean(overs))
            print(f"{sensor:<12s} {nm:<34s} {m[0]:7.1f} {m[1]:7.1f} {m[2]:7.1f}")
            out.setdefault(sensor, {})[nm] = {"ded_pct": m[0], "resid_pct": m[1], "over_pct": m[2]}
        print()
    (OUT_DIR / "07_variant.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"写出 {OUT_DIR / '07_variant.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
