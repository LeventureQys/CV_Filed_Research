# -*- coding: utf-8 -*-
"""探查单组新实采数据的负载结构（不产图，只打印诊断）。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3          # noqa: E402
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4   # noqa: E402

REC = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
       r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
       r"\device_001_seg000.csv")


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def segs(total, frac):
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


t, X = load_rec(REC)
span = t[-1] - t[0]
dtm = span / (len(t) - 1)
print(f"frames={len(t)}  span={span:.2f}s  dt={dtm*1000:.3f}ms  fps={1/dtm:.2f}")
print(f"channels={X.shape[1]}  dup_timestamp={(np.diff(t) <= 0).sum()}")

tu = np.arange(0.0, span, dtm)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
tot = Xu.sum(axis=1)
print(f"total: min={tot.min():.0f} max={tot.max():.0f} "
      f"p1={np.percentile(tot,1):.0f} p50={np.percentile(tot,50):.0f} p99={np.percentile(tot,99):.0f}")

for frac in (0.05, 0.15, 0.30):
    ss = segs(Xu.sum(axis=1), frac)
    print(f"\n-- frac={frac}: {len(ss)} segments (top 8) --")
    for a, b in ss[:8]:
        print(f"   [{tu[a]:7.2f}, {tu[b]:7.2f}]  dur={tu[b]-tu[a]:6.2f}s  "
              f"mean_tot={Xu[a:b].sum(axis=1).mean():9.0f}")

# 主通道与幅度
ss = segs(Xu.sum(axis=1), 0.15)
s0, s1 = ss[0]
base = Xu[:max(1, s0)].mean(axis=0)
amp = Xu[s0:s1].mean(axis=0) - base
m = int(np.argmax(amp))
print(f"\nmain channel ch{m}  amp={amp[m]:.1f}  loaded chans={int((amp > 0.1*amp.max()).sum())}")
print("amp per ch:", np.round(amp, 1))

for tag, cls, kw in [("v3", GLM53v3, {}),
                     ("v4_fast5", GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0))]:
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    print(f"\n[{tag}] A_max={c.A.max():.4f}  loaded={int(c.loaded.sum())}  "
          f"g_end={c.g:+.4f}  gamma[min,max]=[{c.gamma.min():.3f},{c.gamma.max():.3f}]")
    print(f"   out[s0..s1] main ch: start={Y[s0,m]:.1f} end={Y[s1-1,m]:.1f} "
          f"drift={100*(Y[s1-1,m]-Y[s0,m])/amp[m]:+.2f}%   "
          f"raw drift={100*(Xu[s1-1,m]-Xu[s0,m])/amp[m]:+.2f}%")
