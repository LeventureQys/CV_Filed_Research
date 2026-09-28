# -*- coding: utf-8 -*-
"""a6 改造验证 v2：针对"抖动抬升 σ_d ⇒ 门限抬高 ⇒ 漏事件/基线错位"的可开关改造。

与原型的差异（全部可开关，默认关时与原型逐帧一致，脚本首行自检）：
  ① CAP   ：thr_d = max(min(5σ_d, CAP·thr_level), thr_level) —— 抖动不得把门限抬到 5% 电平之上
  ② IDLE  ：空载态额外门限 IDLE_FLOOR（0.25 / 0.50 · max_tot，原型 0.10）—— 零负载平台抗小拍
  ③ DWELL ：d 超门限必须**连续保持** DWELL 秒（原型仅 3 帧 ≈30 ms）—— 驻留确认
  ④ MAG   ：建事件前要求已实测增量/回溯跨度 ≥ MAG·thr_level —— 幅度确认（幅度门）
输出：epoch 的**检测延迟**（相对真值沿）、漏/误事件、ΣA 偏差、稳态显示偏差、最大显示偏差。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import (REC, load_uniform, make_perturb, add_tap, add_step,   # noqa: E402
                      run_traced, make_traced, ROOT)
from glm53_v6 import GLM53v6, _cap                                          # noqa: E402

RES = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
os.makedirs(RES, exist_ok=True)

# 真值加载沿（由无扰动基线的 epoch 时刻给出，见 a0/a4 日志）
GT = {
    "零负载-切换负载-零负载-再切换负载": [3.34, 11.76, 31.20, 35.93, 50.58, 57.28],
    "中途切换-最终测试目标": [8.31, 18.69, 26.33, 27.61, 45.27, 68.90, 87.81, 98.88, 113.81],
}


class V6fix(GLM53v6):
    F_CAP = False
    F_IDLE = False
    F_DWELL = False
    F_MAG = False
    CAPF = 1.0           # min(5σ_d, CAPF·thr_level)
    IDLE_FLOOR = 0.25
    DWELL_S = 0.25
    MAG = 0.5            # 已实测增量 ≥ MAG·thr_level

    def __init__(self, n):
        super().__init__(n)
        self._dwell = 0.0
        self._rej = 0

    def _win_range(self, ts):
        m = self._mask(ts - self.BACKDATE_S, ts)
        n = min(self._bn, self.CAP)
        if not m.any():
            return 0.0
        vv = self._bvm[:n][m]
        return float(vv.max() - vv.min())

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

        # ── 检测器（改造点 ①②③）──
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
            self.sig_d = float(1.4826 * np.median(np.abs(h - np.median(h))))
        sig_term = self.DET_K * self.sig_d
        if self.F_CAP:
            sig_term = min(sig_term, self.CAPF * thr_level)
        thr_d = max(sig_term, thr_level)
        tau_ep = ((ts - self.ev["t0"]) if self.ev is not None
                  else ((ts - self.ev_end) if self.state == "slow" else 1e9))
        tail_gate = (self.TAIL_GATE_FRAC * lvr
                     if (self.state != "idle" and tau_ep < self.TAIL_GATE_S) else 0.0)
        raw_hit = abs(d) > max(thr_d, tail_gate)
        self._hit_run = (self._hit_run + 1) if raw_hit else 0
        self._quiet_run = 0 if raw_hit else (self._quiet_run + 1)
        if self._quiet_run >= self.DET_PERSIST:
            self._armed = True
        hit = (self._hit_run >= self.DET_PERSIST) and self._armed
        if self.F_DWELL:
            self._dwell = (self._dwell + dt) if raw_hit else 0.0
            if self._dwell < self.DWELL_S:
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
                ok_mag = True
                if self.F_MAG and hist:
                    span = max(h[1] for h in hist) - min(h[1] for h in hist)
                    ok_mag = span >= self.MAG * thr_level
                v0 = self._mat_mean(self._bvx, t0 - 0.30, t0 - 0.05)
                y0 = self._mat_mean(self._byx, t0 - 0.30, t0 - 0.05)
                if ok_mag and v0 is not None and y0 is not None and len(hist) >= 4:
                    idle_pre = (base < self.IDLE_FRAC * max(self.level_ref, eps) or
                                base < self.UNLOAD_MIN_RATIO * self.min_ts + eps)
                    prev = self.state
                    kind = ("restep_reload" if new_ev == "restep_reload"
                            else ("onset" if (prev == "idle" and idle_pre) else "restep"))
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


VARIANTS = [
    ("原型", {}),
    ("+CAPF1.0", dict(F_CAP=True, CAPF=1.0)),
    ("+CAPF0.5", dict(F_CAP=True, CAPF=0.5)),
    ("+IDLE0.25", dict(F_IDLE=True, IDLE_FLOOR=0.25)),
    ("+DWELL0.25", dict(F_DWELL=True, DWELL_S=0.25)),
    ("+MAG0.5", dict(F_MAG=True, MAG=0.5)),
    ("+CAPF1.0+IDLE0.25+MAG0.5", dict(F_CAP=True, CAPF=1.0, F_IDLE=True, F_MAG=True)),
]
JIT = [("抖动白噪2000", "white", dict(f_hi=40.0), 2000),
       ("抖动带限2000", "band", dict(f_lo=0.3, f_hi=5.0), 2000),
       ("抖动白噪4000", "white", dict(f_hi=40.0), 4000)]
RECS = ["零负载-切换负载-零负载-再切换负载", "中途切换-最终测试目标"]
TAP_AT = {"零负载-切换负载-零负载-再切换负载": 21.5,
          "中途切换-最终测试目标": 53.1}
_CACHE = {}


def cls_for(lbl):
    if lbl not in _CACHE:
        _CACHE[lbl] = make_traced(
            type("V_" + str(abs(hash(lbl)) % 100000), (V6fix,), dict(VAR(lbl))))
    return _CACHE[lbl]


def VAR(lbl):
    for l, kw in VARIANTS:
        if l == lbl:
            return kw
    return {}


def perturb_on_mask(X, A, kind, rng, kw):
    E = make_perturb(X, A, kind, rng, **kw)
    t = E.sum(axis=1)
    s = t.std()
    return E * (A / s) if s > 1e-12 else E


def latencies(epochs, gt):
    """每个真值沿的检测延迟（s）；未检出记 nan。"""
    out = []
    for g in gt:
        cand = [e - g for e in epochs if 0.0 <= e - g <= 1.5]
        out.append(min(cand) if cand else np.nan)
    return out


def main():
    rows = []
    # 自检
    d0 = load_uniform(REC[RECS[0]], tmax=40.0)
    Ya = _plain(GLM53v6, d0["tu"], d0["Xu"])
    Yb = _plain(V6fix, d0["tu"], d0["Xu"])
    print(f"[自检] 全关改造版 vs 原型逐帧一致={np.allclose(Ya, Yb, atol=1e-9)} "
          f"maxdiff={np.abs(Ya - Yb).max():.3g}", flush=True)
    for name in RECS:
        d = load_uniform(REC[name])
        tu, X = d["tu"], d["Xu"]
        gt = GT[name]
        clean, rowsum = {}, []
        for lbl, kw in VARIANTS:
            r = run_traced(cls_for(lbl), tu, X)
            clean[lbl] = r
            lat = latencies(sorted(e[0] for e in r["epoch"]), gt)
            rows.append(dict(rec=name, scene="干净基线", variant=lbl,
                             n_epoch=len(r["epoch"]), n_miss=0, n_extra=0, dA=0.0,
                             maxdev=0.0, slow_dev=0.0,
                             lat_max=(np.nanmax(lat) * 1000 if np.any(~np.isnan(lat)) else np.nan),
                             lat_med=(np.nanmedian(lat) * 1000 if np.any(~np.isnan(lat)) else np.nan)))
            rowsum.append(f"{lbl}:ep{len(r['epoch'])}/延迟中位{np.nanmedian(lat)*1000:.0f}ms"
                          if np.any(~np.isnan(lat)) else f"{lbl}:ep{len(r['epoch'])}")
        print(f"[{name}] 干净基线 " + " | ".join(rowsum), flush=True)

        scenes = [("抖动" + l, X + perturb_on_mask(X, a, k, np.random.default_rng(7), kw))
                  for l, k, kw, a in JIT]
        i0 = int(round(TAP_AT[name] / 0.01))
        for a in [2000, 4000, 6000]:
            scenes.append((f"拍击{a}", add_tap(X, i0, a, rise_ms=50, hold_ms=100)[0]))
        scenes.append(("真实阶跃4000", add_step(X, i0, 4000, rise_ms=50)))

        for scene, Xp in scenes:
            out = []
            for lbl, kw in VARIANTS:
                rc, r = clean[lbl], run_traced(cls_for(lbl), tu, Xp)
                tottp = (np.abs(Xp.sum(axis=1) - X.sum(axis=1)).max() > 1e-6)
                epl = sorted(set(e[0] for e in rc["epoch"]))
                ep = sorted(set(e[0] for e in r["epoch"]))
                miss = [e for e in epl if not any(abs(e - x) <= 0.8 for x in ep)]
                extra = [e for e in ep if not any(abs(e - x) <= 0.8 for x in epl)]
                dev = r["Y"].sum(axis=1) - rc["Y"].sum(axis=1)
                slow = (np.asarray(r["st"]) == 2) & (np.asarray(rc["st"]) == 2)
                rows.append(dict(
                    rec=name, scene=scene, variant=lbl, n_epoch=len(ep),
                    n_miss=len(miss), n_extra=len(extra),
                    dA=round(float(np.sum(r["comp"].A) - np.sum(rc["comp"].A)), 1),
                    maxdev=round(float(np.abs(dev).max()), 1),
                    slow_dev=round(float(np.median(dev[slow])), 1) if slow.sum() > 50 else 0.0,
                    lat_max=np.nan, lat_med=np.nan))
                out.append(f"{lbl}:ep{len(ep)}(漏{len(miss)}/误{len(extra)}) "
                           f"ΔA{np.sum(r['comp'].A)-np.sum(rc['comp'].A):.0f} "
                           f"maxdev{np.abs(dev).max():.0f} slow{np.median(dev[slow]) if slow.sum()>50 else 0:.0f}")
            print(f"[{name}] {scene:12s} " + " | ".join(out), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a6_fix_ablation.csv"), index=False, encoding="utf-8-sig")
    sub = df[df.scene != "干净基线"]
    print("\n-- 汇总（跨记录平均）--")
    print(sub.groupby(["variant", "scene"])[["n_epoch", "n_miss", "n_extra", "dA",
                                             "maxdev", "slow_dev"]].mean().round(1).to_string())
    print("\n-- 干净基线的检测延迟（ms）--")
    print(df[df.scene == "干净基线"].groupby("variant")[["lat_med", "lat_max", "n_epoch"]]
          .mean().round(1).to_string())


def _plain(cls, tu, X):
    c = cls(X.shape[1])
    Y = np.empty_like(X)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], X[i])
    return Y


if __name__ == "__main__":
    main()
