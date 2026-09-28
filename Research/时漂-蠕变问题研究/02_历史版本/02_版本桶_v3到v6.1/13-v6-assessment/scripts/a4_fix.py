# -*- coding: utf-8 -*-
"""a4 改进验证：在原型上做可开关的 4 项改造（消融实验），用同一批扰动/拍击场景对比前后数字。

改造项（每一项可单独开关）：
  ① 单帧可信跳变门 |Δ_jump| ≥ max(kJumpRel·平台电平, kJumpAbs·max_tot)
     —— 拍击/瞬态伪 epoch 的"幅度上限"约束（原型原来只要 |jump| > 1e-12 就放行）；
  ② 驻留确认：d 超门限必须连续保持 ≥ HOLD 秒（原型 3 帧 ≈30 ms）；
  ③ σ_d 只用 |d| ≤ σWIN·thr_level 的帧估计（排除事件/瞬态响应，避免 σ 被自己抬高）；
  ④ 空载态额外门限从 10%·max_tot 提到 IDLE_FLOOR（对"零负载起步"的平台最弱的档位）。

输出：每个场景 × 变体的 epoch 数、漏/误触发、ΣA 偏差、稳态显示偏差、最大显示偏差。
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import (REC, load_uniform, make_perturb, add_tap, add_step,   # noqa: E402
                      TRACED_V6, TRACED_V6 as _T, run_traced, ROOT)
from glm53_v6 import GLM53v6, _cap                                 # noqa: E402

RES = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
os.makedirs(RES, exist_ok=True)


class GLM53v6fix(GLM53v6):
    """带开关的改造版（默认全关 = 与原型等价，用于自检）。"""
    F_JUMP = False
    F_HOLD = False
    F_SIGWIN = False
    F_IDLE = False
    K_JUMP_REL = 0.30      # 单帧可信跳变下限（相对平台电平）
    K_JUMP_ABS = 0.02      # 单帧可信跳变下限（相对历史最大总量）
    HOLD_S = 0.25          # 驻留确认时长
    SIGWIN = 2.5           # σ 估计只用 |d| ≤ SIGWIN·thr_level 的帧
    IDLE_FLOOR = 0.25      # 空载态额外门限（相对 max_tot）

    def __init__(self, n):
        super().__init__(n)
        self._hold_t = 0.0
        self._hold_on = False
        self._jump = 0.0     # 最近一次回溯到的单帧/近窗跳变量（供门槛用）

    # ── 只改检测器段，其余逐行照抄原 process ──
    def process(self, ts, v):
        v = np.asarray(v, float).copy()
        total = float(v.sum())
        if self.first:
            self.last_out = v.copy()
        self._push(ts, total, v, self.last_out[:self.n])

        dt = 0.0
        if self.first:
            self.first = False
            self.last_ts = self.t0 = ts
            self.ts_smooth = self.level_ref = total
            self.min_ts = self.max_ts = total
        else:
            dt = ts - self.last_ts
            self.last_ts = ts
            dt = 0.0 if not (dt > 0.0) else min(dt, 0.1)
            if dt > 0.0:
                self.ts_smooth += (dt / self.TAU_TOTAL) * (total - self.ts_smooth)
                self.level_ref += (dt / self.TAU_LEVEL) * (self.ts_smooth - self.level_ref)
        self.min_ts = min(self.min_ts, self.ts_smooth)
        self.max_ts = max(self.max_ts, self.ts_smooth)
        self.max_tot = max(self.max_tot, total)
        eps = 1e-6 * (1.0 + abs(self.max_tot))

        # ── 检测器（改造版）──
        lv_now = self._win_mean(ts - self.DET_FAST, ts)
        lv_ref = self._win_mean(ts - self.DET_FAST - self.DET_GAP - self.DET_LAG,
                                ts - self.DET_FAST - self.DET_GAP)
        d = (lv_now - lv_ref) if (lv_now is not None and lv_ref is not None) else 0.0
        lvr = abs(lv_ref) if lv_ref is not None else 0.0
        thr_level = max(self.DET_REL * lvr, self.DET_ABS_FRAC * self.max_tot)
        i = self._dn % self.DCAP
        self._d[i] = d
        self._dn += 1
        if self._dn > 40:
            h = self._d[:min(self._dn, self.DCAP)]
            if self.F_SIGWIN:
                # 门限先把「事件/瞬态响应」的大 |d| 排除（否则 σ 被自己的响应抬高）
                thr_g = max(self.DET_K, self.SIGWIN) * thr_level
                h2 = h[np.abs(h) <= thr_g]
                if len(h2) >= 30:
                    h = h2
            self.sig_d = float(1.4826 * np.median(np.abs(h - np.median(h))))
        thr_d = max(self.DET_K * self.sig_d, thr_level)
        tau_ep = (ts - self.ev["t0"]) if self.ev is not None else (
            (ts - self.end_ref()) if self.state == "slow" else 1e9)
        tail_gate = (self.TAIL_GATE_FRAC * lvr
                     if (self.state != "idle" and tau_ep < self.TAIL_GATE_S) else 0.0)
        raw_hit = abs(d) > max(thr_d, tail_gate)
        self._hit_run = (self._hit_run + 1) if raw_hit else 0
        self._quiet_run = 0 if raw_hit else (self._quiet_run + 1)
        if self._quiet_run >= self.DET_PERSIST:
            self._armed = True
        hit = (self._hit_run >= self.DET_PERSIST) and self._armed
        if self.F_HOLD:
            if raw_hit:
                self._hold_t += dt
            else:
                self._hold_t = 0.0
            if self._hold_t < self.HOLD_S:
                hit = False

        idle_now = (self.ts_smooth < self.IDLE_FRAC * max(self.level_ref, eps) or
                    self.ts_smooth < self.UNLOAD_MIN_RATIO * self.min_ts + eps)

        if hit and d > 0 and self.state == "idle":
            if (ts - self.idle_since) < self.IDLE_SETTLE:
                hit = False
            else:
                floor = self.IDLE_FLOOR if self.F_IDLE else self.DET_IDLE_FRAC
                if d <= max(thr_d, floor * self.max_tot):
                    hit = False
        if hit and ts < self.ev_block_until:
            hit = False
        if hit and d > 0:
            new_ev = None
            if self.ev is None:
                if (ts - self.last_ev_ts) > self.REVOKE_COOLDOWN:
                    new_ev = "onset_or_restep"
            elif tau_ep > self.REVOKE:
                new_ev = "restep_reload"
            if new_ev is not None:
                t0, base, hist = self._backdate(ts)
                self._jump = self._win_range(ts)
                ok_jump = True
                if self.F_JUMP:
                    need = max(self.K_JUMP_REL * max(abs(base), 1.0),
                               self.K_JUMP_ABS * self.max_tot)
                    ok_jump = self._jump >= need
                v0 = self._mat_mean(self._bvx, t0 - 0.30, t0 - 0.05)
                y0 = self._mat_mean(self._byx, t0 - 0.30, t0 - 0.05)
                if ok_jump and v0 is not None and y0 is not None and len(hist) >= 4:
                    idle_pre = (base < self.IDLE_FRAC * max(self.level_ref, eps) or
                                base < self.UNLOAD_MIN_RATIO * self.min_ts + eps)
                    prev = self.state
                    if new_ev == "restep_reload":
                        kind = "restep_reload"
                    else:
                        kind = "onset" if (prev == "idle" and idle_pre) else "restep"
                    self._new_event(t0, base, v0, y0, kind, hist, prev)
                    self._armed = False
                self.last_ev_ts = ts
        elif hit and d < 0 and self.state != "idle" and self.ev is None:
            if (ts - self.last_ev_ts) > self.REVOKE_COOLDOWN:
                t0, base, hist = self._backdate(ts)
                v0 = self._mat_mean(self._bvx, t0 - 0.30, t0 - 0.05)
                y0 = self._mat_mean(self._byx, t0 - 0.30, t0 - 0.05)
                if v0 is not None and y0 is not None and len(hist) >= 4:
                    self._new_event(t0, base, v0, y0, "decrease", hist, self.state)
                self.last_ev_ts = ts

        if self.ev is not None:
            out = self._run_event(ts, v, total, dt, eps, idle_now)
            self.last_out = out
            return out
        if self.state == "slow":
            if idle_now and (ts - self.ev_end) > self.UNLOAD_FAST:
                self._to_idle()
                self.last_out = v.copy()
                return v
            if self.hold_comp is not None:
                if ts >= self.hold_until:
                    self._reanchor(ts, v)
                else:
                    m = self.loaded & (self.A > 1e-9)
                    out = v.copy()
                    if m.any():
                        out[m] = [v[k] - _cap(v[k], self.hold_comp[k])
                                  for k in range(self.n) if m[k]]
                    self.last_out = out
                    return out
            out = self._slow_step(ts, v, dt)
            self.last_out = out
            return out
        self.last_out = v.copy()
        return v

    def end_ref(self):
        return self.ev_end

    def _win_range(self, ts):
        """回溯窗内的电平跨度（ptp）——单帧可信跳变的判据。"""
        m = self._mask(ts - self.BACKDATE_S, ts)
        n = min(self._bn, self.CAP)
        if not m.any():
            return 0.0
        vv = self._bvm[:n][m]
        return float(vv.max() - vv.min())


def make_traced(cls):
    from a_common import make_traced as mt
    return mt(cls)


VARIANTS = [
    ("原型", dict()),
    ("+跳变门", dict(F_JUMP=True)),
    ("+驻留0.25s", dict(F_HOLD=True)),
    ("+σ窗", dict(F_SIGWIN=True)),
    ("+空载门0.25", dict(F_IDLE=True)),
    ("+跳变门+σ窗+空载门", dict(F_JUMP=True, F_SIGWIN=True, F_IDLE=True)),
]

JIT = [("白噪2000", "white", dict(f_hi=40.0), 2000),
       ("带限2000", "band", dict(f_lo=0.3, f_hi=5.0), 2000),
       ("白噪4000", "white", dict(f_hi=40.0), 4000)]
RECS = ["零负载-切换负载-零负载-再切换负载", "中途切换-最终测试目标"]
TAP_AT = {"零负载-切换负载-零负载-再切换负载": (21.5, 2610),
          "中途切换-最终测试目标": (53.1, 3354)}


def perturb_on_mask(X, A, kind, rng, kw):
    E = make_perturb(X, A, kind, rng, **kw)
    t = E.sum(axis=1)
    s = t.std()
    return E * (A / s) if s > 1e-12 else E


def main():
    rows = []
    # ── 自检：改造版"全关"必须与原型逐帧一致 ──
    d0 = load_uniform(REC["零负载-切换负载-零负载-再切换负载"], tmax=40.0)
    Yref, cref = _plain(GLM53v6, d0["tu"], d0["Xu"])
    Yfix, cfix = _plain(GLM53v6fix, d0["tu"], d0["Xu"])
    same = np.allclose(Yref, Yfix, atol=1e-9) and len(cref.epoch_t) == len(cfix.epoch_t)
    print(f"[自检] 改造版(全关) vs 原型: 逐帧一致={same} "
          f"epoch {len(cref.epoch_t)} vs {len(cfix.epoch_t)} "
          f"maxdiff={np.abs(Yref - Yfix).max():.3g}", flush=True)
    if not same:
        print("!! 自检失败：后续对比不可信", flush=True)
    for name in RECS:
        d = load_uniform(REC[name])
        tu, X = d["tu"], d["Xu"]
        clean = {}
        for lbl, kw in VARIANTS:
            T = _cls(lbl)
            r = run_traced(T, tu, X)
            clean[lbl] = r
            rows.append(dict(rec=name, scene="干净基线", variant=lbl, n_epoch=len(r["epoch"]),
                             n_miss=0, n_extra=0, dA=0.0, maxdev=0.0, slow_dev=0.0,
                             max_abs_disp=round(float(np.abs(r["Y"].sum(axis=1)).max()), 1)))
            print(f"[{name}] 基线 {lbl:18s} epoch={len(r['epoch']):3d} "
                  f"ΣA={np.sum(r['comp'].A):9.0f} "
                  f"eps={[round(e[0], 2) for e in r['epoch']][:8]}", flush=True)
        scenes = []
        for lbl, kind, kw, amp in JIT:
            scenes.append(("抖动" + lbl,
                           X + perturb_on_mask(X, amp, kind, np.random.default_rng(7), kw)))
        t_inj, lvl = TAP_AT[name]
        i0 = int(round(t_inj / 0.01))
        for amp in [4000, 6000]:
            Xp, _ = add_tap(X, i0, amp, rise_ms=50, hold_ms=100)
            scenes.append((f"拍击{amp}", Xp))
        scenes.append(("真实阶跃4000", add_step(X, i0, 4000, rise_ms=50)))

        for scene, Xp in scenes:
            for lbl, kw in VARIANTS:
                rc, r = clean[lbl], run_traced(_cls(lbl), tu, Xp)
                Y0 = rc["Y"].sum(axis=1)
                disp = r["Y"].sum(axis=1)
                dev = disp - Y0
                epl = sorted(set(e[0] for e in rc["epoch"]))
                ep = sorted(set(e[0] for e in r["epoch"]))
                miss = [e for e in epl if not any(abs(e - x) <= 0.8 for x in ep)]
                extra = [e for e in ep if not any(abs(e - x) <= 0.8 for x in epl)]
                slow = (np.asarray(r["st"]) == 2) & (np.asarray(rc["st"]) == 2)
                rows.append(dict(
                    rec=name, scene=scene, variant=lbl, n_epoch=len(ep),
                    n_miss=len(miss), n_extra=len(extra),
                    dA=round(float(np.sum(r["comp"].A) - np.sum(rc["comp"].A)), 1),
                    maxdev=round(float(np.abs(dev).max()), 1),
                    slow_dev=round(float(np.median(dev[slow])), 1) if slow.sum() > 50 else 0.0,
                    max_abs_disp=round(float(np.abs(disp).max()), 1)))
            sel = [x for x in rows if x["rec"] == name and x["scene"] == scene]
            print(f"[{name}] {scene:10s} " + " | ".join(
                f"{x['variant']}: ep{x['n_epoch']}(漏{x['n_miss']}/误{x['n_extra']}) "
                f"maxdev{x['maxdev']:.0f} slow{x['slow_dev']:.0f}" for x in sel), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a4_fix_ablation.csv"), index=False, encoding="utf-8-sig")
    print("\n-- 汇总（跨记录平均，仅扰动/瞬态场景）--")
    sub = df[df.scene != "干净基线"]
    print(sub.groupby(["variant", "scene"])[["n_epoch", "n_miss", "n_extra", "maxdev", "slow_dev"]]
          .mean().round(1).to_string())


def _plain(cls, tu, X):
    c = cls(X.shape[1])
    Y = np.empty_like(X)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], X[i])
    return Y, c


_CLS_CACHE = {}


def _cls(lbl):
    if lbl not in _CLS_CACHE:
        kw = _V(lbl)
        _CLS_CACHE[lbl] = make_traced(
            type("T_" + str(abs(hash(lbl)) % 100000), (GLM53v6fix,), kw))
    return _CLS_CACHE[lbl]


def _V(lbl):
    for l, kw in VARIANTS:
        if l == lbl:
            return kw
    return {}


if __name__ == "__main__":
    main()
