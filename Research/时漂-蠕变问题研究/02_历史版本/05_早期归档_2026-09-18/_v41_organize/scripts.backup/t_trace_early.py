# -*- coding: utf-8 -*-
"""看 v3 与 v4_fast5 在加载后前 20s 的逐帧行为：A 何时被捕获、补偿何时开始、基线抬到哪里。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3  # noqa: E402
from r_fastphase import GLM53v4  # noqa: E402


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def segs(total, frac=0.15):
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


for loc, name in [("右拇指指尖", "数据1"), ("四指指尖", "数据1")]:
    t, X = load_rec(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    s0r, s1r = segs(X.sum(axis=1))[0]
    s0, s1 = int(np.searchsorted(tu, t[s0r])), int(np.searchsorted(tu, t[s1r]))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    tot = Xu.sum(axis=1)
    base = tot[:s0].mean()
    thr = base + 0.05 * (tot[s0:s1].max() - base)
    n_on = next(i for i in range(s0, s0 + 300) if tot[i] > thr)

    # 逐帧记录内部状态
    logs = {}
    for lab, cls, kw in [("v3", GLM53v3, {}), ("v4", GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0))]:
        c = cls(Xu.shape[1])
        for k, v in kw.items():
            setattr(c, k, v)
        Y = np.empty_like(Xu)
        Acap, loaded_n = [], []
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
            Acap.append(float(c.A[m]) if getattr(c, "a_captured", False) else np.nan)
            loaded_n.append(int(c.loaded.sum()) if getattr(c, "a_captured", False) else 0)
        logs[lab] = (Y, np.array(Acap), np.array(loaded_n))

    print(f"\n=== {loc}/{name} ch{m}  onset t={tu[n_on]:.2f}s  空载基线={Xu[:s0, m].mean():.4f} ===")
    print(f"{'t(s)':>7} {'原始X':>8} | {'v3 Y':>8} {'v3补偿':>8} {'v3 A':>8} {'n_ld':>5} | "
          f"{'v4 Y':>8} {'v4补偿':>8} {'v4 A':>8} {'n_ld':>5}")
    for tt in [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0]:
        k = n_on + int(round(tt / dt))
        if k >= s1:
            break
        Yv3, Av3, Lv3 = logs["v3"]
        Yv4, Av4, Lv4 = logs["v4"]
        print(f"{tt:7.2f} {Xu[k,m]:8.4f} | {Yv3[k,m]:8.4f} {Yv3[k,m]-Xu[k,m]:8.4f} "
              f"{Av3[k]:8.4f} {Lv3[k]:5d} | {Yv4[k,m]:8.4f} {Yv4[k,m]-Xu[k,m]:8.4f} "
              f"{Av4[k]:8.4f} {Lv4[k]:5d}")
    print(f"  加载瞬时读数 X(0+)={Xu[n_on,m]:.4f}   5s 时 X={Xu[n_on+int(5/dt),m]:.4f}"
          f"   快相增量={Xu[n_on+int(5/dt),m]-Xu[n_on,m]:.4f}"
          f"   (占总增量 {100*(Xu[n_on+int(5/dt),m]-Xu[n_on,m])/(Xu[s1-1,m]-Xu[n_on,m]):.1f}%)")
