# -*- coding: utf-8 -*-
"""v2.0 阶段二 · F2 卸载出口候选的可行性 + 收益 A/B。

背景（design_failure_census.txt 实测）：本录制里 `idle_now` 在**事件态内只真 1 帧、慢相态内真 0 帧**，
而现有卸载出口在事件→退出的迁移中触发 63 次（都在下探极深处）⇒「事件期内的卸载」拿不到现有出口。

候选出口判据（三者都只依赖输入/显示自身，不依赖 idle_now）：
  V1 谷底：ts_smooth 回到近 W 秒窗谷底（w_min + frac·(w_max−w_min)）且窗内幅度够大
  V2 掉幅：事件内 inc < drop_frac · inc_max 持续 hold 秒
  V3 回落：显示已回到事件前电平附近（|显示 − base_y| < back_frac · Â）

对每个候选给出：触发次数、以及"长保压段偏移/瞬态超前/T5%"三项指标的变化。
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

W_S, VALLEY_FRAC, RANGE_FRAC = 40.0, 0.15, 0.05
CLAMP_ALPHA = 0.005


class Arm(P.GLM53v6):
    def __init__(self, n, clamp=None, exit_mode=None, drop_frac=0.5, drop_hold=0.30,
                 back_frac=0.25, w_s=W_S):
        super().__init__(n)
        self.KAPPA_ONSET, self.KAPPA_RESTEP, self.HO_MIN = 1.05, 1.12, 3.5
        self.clamp = clamp
        self.exit_mode = exit_mode
        self.drop_frac, self.drop_hold, self.back_frac, self.w_s = \
            drop_frac, drop_hold, back_frac, w_s
        self._win = []
        self.n_exit = 0
        self.drop_run = 0.0

    def _win_update(self, ts):
        self._win.append((ts, self.ts_smooth))
        cut = ts - self.w_s
        while self._win and self._win[0][0] < cut:
            self._win.pop(0)
        arr = [v for _t, v in self._win]
        mn, mx = min(arr), max(arr)
        eps = 1e-6 * (1 + abs(mx))
        return (self.ts_smooth < mn + VALLEY_FRAC * max(mx - mn, eps)
                and (mx - mn) > RANGE_FRAC * max(abs(mx), eps))

    def process(self, ts, v):
        raw = np.asarray(v, float).copy()
        n_frame = np.asarray(super().process(ts, raw.copy()), float).copy()
        self.valley = self._win_update(ts)
        if self.clamp is not None:
            n_frame = np.minimum(n_frame, raw + self.clamp * np.abs(raw))
        return n_frame

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        if self.exit_mode and ev is not None:
            tau = ts - ev["t0"]
            fire = False
            if tau > 0.60:
                if self.exit_mode == "V1":
                    fire = self.valley
                elif self.exit_mode == "V2":
                    inc_s = self._win_mean(ts - 0.10, ts)
                    inc_s = (inc_s - ev["base"]) if inc_s is not None else (total - ev["base"])
                    if ev["inc_max"] > eps and inc_s < self.drop_frac * ev["inc_max"]:
                        self.drop_run += dt
                    else:
                        self.drop_run = 0.0
                    fire = self.drop_run >= self.drop_hold
                elif self.exit_mode == "V1V2":
                    inc_s = self._win_mean(ts - 0.10, ts)
                    inc_s = (inc_s - ev["base"]) if inc_s is not None else (total - ev["base"])
                    if ev["inc_max"] > eps and inc_s < self.drop_frac * ev["inc_max"]:
                        self.drop_run += dt
                    else:
                        self.drop_run = 0.0
                    fire = self.valley or (self.drop_run >= self.drop_hold)
            if fire:
                self.n_exit += 1
                self.ev = None
                self.state = "idle"
                self.last_ev_ts = ts
                self.idle_since = ts
                self.ev_block_until = ts + self.UNLOAD_BLOCK
                self.hold_comp = None
                self.loaded[:] = False
                self.g = 0.0
                self.last_out = v.copy()
                return v
        return super()._run_event(ts, v, total, dt, eps, idle_now)


def run(el, V, **kw):
    c = Arm(V.shape[1], **kw)
    out = np.empty((len(el), V.shape[1]))
    for i in range(len(el)):
        out[i] = c.process(float(el[i]), V[i])
    return c, out.sum(1)


def edges(el, tot):
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tot, 3), np.percentile(tot, 97)
    up, _ = L.edges_from_tot(np.convolve(tot, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in up:
        if not tu or el[i] - tu[-1] > 1.0:
            tu.append(float(el[i]))
    return tu


def metrics(el, pre, out, tu, segs):
    lead, tail, t5, bias = [], [], [], []
    for t in tu:
        m0 = (el >= t - 1.2) & (el < t - 0.35)
        if m0.sum() < 5:
            continue
        bp = float(np.median(pre[m0]))
        te = min(t + 30.0, el[-1] - 0.2)
        m1 = (el >= te - 1.5) & (el <= te)
        ps, os_ = float(np.median(pre[m1])), float(np.median(out[m1]))
        A = ps - bp
        if abs(A) < 500:
            continue
        idx = np.where((el >= t - 0.05) & (el <= te))[0]
        lead.append(float(np.max((out[idx] - pre[idx]) / abs(A))))
        tail.append(float(np.max((out[idx] - os_) / abs(A))))
        bias.append((os_ - ps) / abs(A))
        for k in range(len(idx)):
            if abs(out[idx[k]] - os_) <= 0.05 * abs(A):
                t5.append(el[idx[k]] - t)
                break
    hold_offs = []
    for kind, a, b in segs:
        if kind == "loaded" and b - a >= 200:
            hold_offs.append(float(np.median((out - pre)[a:b + 1])))
    hold_offs = np.asarray(hold_offs) if hold_offs else np.asarray([0.0])
    return dict(lead=np.median(lead) if lead else float("nan"),
                tail=np.median(tail) if tail else float("nan"),
                t5=np.median(t5) if t5 else float("nan"),
                bias=np.median(bias) if bias else float("nan"),
                hold_abs=np.median(np.abs(hold_offs)),
                hold_max=np.max(np.abs(hold_offs)))


def main():
    d = L.load_dataset(L.DS_ZERO)
    el = d["pre"]["el"]
    V = d["pre"]["V"]
    pre = V.sum(1)
    tu = edges(el, pre)
    segs = L.plateau_segments(pre, el, min_dur=5.0)

    arms = [
        ("base", dict()),
        ("C", dict(clamp=CLAMP_ALPHA)),
        ("C+V1", dict(clamp=CLAMP_ALPHA, exit_mode="V1")),
        ("C+V2", dict(clamp=CLAMP_ALPHA, exit_mode="V2")),
        ("C+V1V2", dict(clamp=CLAMP_ALPHA, exit_mode="V1V2")),
        ("C+V2(hold0.6)", dict(clamp=CLAMP_ALPHA, exit_mode="V2", drop_hold=0.60)),
    ]
    print(f"{'臂':16s}{'出口次数':>8}{'超前峰中位':>10}{'尾峰中位':>9}"
          f"{'T5%中位':>8}{'稳态偏差':>9}{'保压|off|中位':>12}{'保压|off|最大':>12}")
    for tag, kw in arms:
        c, osum = run(el, V, **kw)
        m = metrics(el, pre, osum, tu, segs)
        print(f"{tag:16s}{c.n_exit:8d}{100*m['lead']:9.2f}%{100*m['tail']:8.2f}%"
              f"{m['t5']:8.2f}{100*m['bias']:8.2f}%{m['hold_abs']:12.0f}{m['hold_max']:12.0f}")


if __name__ == "__main__":
    main()
