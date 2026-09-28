# -*- coding: utf-8 -*-
"""GLM53 时间线核验：restep 前后逐 0.25s 打印 raw/v2 显示/内部量，确认跳变机制。"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))  # temp
OUT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("fv", os.path.join(HERE, "f_varying_load.py"))
fv = importlib.util.module_from_spec(spec)
sys.modules["fv"] = fv
spec.loader.exec_module(fv)


class Rec(fv.CompV2):
    def __init__(self):
        super().__init__()
        self.tr = []

    def process(self, ts, v):
        out = super().process(ts, v)
        creep = 0.0
        if self.in_load and self.a_captured:
            ld = self.loaded & (self.A > 1e-9)
            creep = float(np.sum(np.clip(self.gamma[ld] * self.A[ld] * self.g,
                                         self.p["creep_lo"] * self.A[ld],
                                         self.p["creep_hi"] * self.A[ld])))
        self.tr.append(dict(ts=ts, raw=float(np.sum(v)), out=float(np.sum(out)),
                            g=float(self.g), creep=creep, fast=float(self.fast),
                            slow=float(self.slow), divergence=float(abs(self.fast - self.slow)),
                            pend=float("nan") if self.pend_t is None else float(self.pend_t),
                            load=int(self.in_load), cap=int(self.a_captured)))
        return out


def run(loc):
    t, X, tc = fv.load_csv(os.path.join(BASE, "变化负载", loc, "device_001_seg000.csv"))
    c = Rec()
    Y = np.empty_like(X, dtype=float)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i].astype(float))
    return t, pd.DataFrame(c.tr), [(float(a), b) for a, b in c.events]


VARY = {"A": "零负载-切换负载-零负载-再切换负载",
        "B": "零负载-中途切换负载-零负载-切换负载"}
for tag, loc in VARY.items():
    t, tr, evs = run(loc)
    print(f"\n===== 数据{tag}: {loc} =====")
    print("events:", [(round(a, 1), b) for a, b in evs])
    # 每个 restep / unload 前后打印
    for ets, etag in evs:
        if etag not in ("restep",):
            continue
        t0 = float(ets)
        w = tr[(tr.ts >= t0 - 4) & (tr.ts <= t0 + 5)]
        w = w[np.r_[True, np.diff(w.ts.values) > 1e-6]]
        print(f"\n--- {etag}@{t0:.2f}s (每 0.25s 采样) ---")
        print(f"  {'t':>6} {'raw':>7} {'v2显示':>7} {'fast':>7} {'slow':>7} {'div':>6} {'g':>6} {'creep':>6} {'pend':>6} L C")
        idx = 0
        tt = t0 - 4
        while tt <= t0 + 5:
            i = int(np.searchsorted(w.ts.values, tt))
            i = min(i, len(w) - 1)
            r = w.iloc[i]
            print(f"  {r.ts:6.1f} {r.raw:7.0f} {r.out:7.0f} {r.fast:7.0f} {r.slow:7.0f} "
                  f"{r.divergence:6.0f} {r.g:6.3f} {r.creep:6.0f} {r.pend:6.1f} {r.load} {r.cap}")
            tt += 0.25
