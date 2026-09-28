# -*- coding: utf-8 -*-
"""GLM53 v3 残差跳变定位: restep 前后逐帧打印 v3 显示/raw/hold/A/g。"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))

spec = importlib.util.spec_from_file_location("fv", os.path.join(HERE, "f_varying_load.py"))
fv = importlib.util.module_from_spec(spec)
sys.modules["fv"] = fv
spec.loader.exec_module(fv)


class Rec(fv.CompV3):
    def __init__(self):
        super().__init__()
        self.tr = []

    def process(self, ts, v):
        out = super().process(ts, v)
        comp = 0.0
        if self.hold and self.hold_comp is not None:
            comp = float(self.hold_comp.sum())
        elif self.a_captured:
            ld = self.loaded & (self.A > 1e-9)
            comp = float(np.sum(np.clip(self.gamma[ld] * self.A[ld] * self.g,
                                        self.p["creep_lo"] * self.A[ld],
                                        self.p["creep_hi"] * self.A[ld])))
        self.tr.append(dict(ts=ts, raw=float(np.sum(v)), out=float(np.sum(out)),
                            g=float(self.g), comp=comp, hold=int(self.hold),
                            pend=float("nan") if self.pend_t is None else float(self.pend_t),
                            Amax=float(self.A.max()) if self.a_captured else float("nan"),
                            Asum=float(self.A.sum()) if self.a_captured else float("nan"),
                            gAsum=float((self.gamma * self.A).sum()) if self.a_captured else float("nan"),
                            nld=int(np.sum(self.loaded)) if self.loaded is not None else 0,
                            nframes=self.a_new_frames))
        return out


t, X, tc = fv.load_csv(os.path.join(BASE, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                                    "device_001_seg000.csv"))
c = Rec()
Y = np.empty_like(X, dtype=float)
for i in range(len(t)):
    Y[i] = c.process(t[i], X[i].astype(float))
tr = pd.DataFrame(c.tr)
t0 = 23.83
w = tr[(tr.ts >= t0 - 3) & (tr.ts <= t0 + 1.5)]
w = w[np.r_[True, np.diff(w.ts.values) > 1e-6]]
print(f"  {'t':>6} {'raw':>7} {'显示':>7} {'comp':>6} {'g':>6} {'hold':>4} {'pend':>6} "
      f"{'Amax':>7} {'Asum':>7} {'gAsum':>8} {'nLd':>4} {'nF':>4}")
tt = t0 - 3
while tt <= t0 + 1.5:
    i = int(np.searchsorted(w.ts.values, tt))
    i = min(i, len(w) - 1)
    r = w.iloc[i]
    print(f"  {r.ts:6.2f} {r.raw:7.0f} {r.out:7.0f} {r.comp:6.0f} {r.g:6.3f} "
          f"{int(r.hold):4d} {r.pend:6.2f} {r.Amax:7.0f} {r.Asum:7.0f} {r.gAsum:8.0f} "
          f"{int(r.nld):4d} {int(r.nframes):4d}")
    tt += 0.1
