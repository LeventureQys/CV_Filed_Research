# -*- coding: utf-8 -*-
"""核查「快相 0~4s 归一化轮廓」是物理形状还是平滑伪影。

对照：
  raw   ：仅 3 帧中值（≈45ms）——接近显示链路真实可见的读数
  sm2s  ：w2_repeat.py 用的 τ=2.0s 因果指数平滑（Document/06 §1.2 口径）
若两者形状差一个数量级，说明文档里的「机械加载段 0~1s / 快相尾巴 1~4s」主要由平滑器决定。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(os.path.dirname(HERE), "results")

DATA = [
    ("右拇指_1", "右拇指", r"temp\右拇指指尖\数据1\device_001_seg000.csv"),
    ("右拇指_2", "右拇指", r"temp\右拇指指尖\数据2\device_001_seg000.csv"),
    ("右拇指_3", "右拇指", r"temp\右拇指指尖\数据3\device_001_seg000.csv"),
    ("左拇指_1", "左拇指", r"temp\左拇指指尖\数据1\device_001_seg000.csv"),
    ("左拇指_2", "左拇指", r"temp\左拇指指尖\数据2\device_001_seg000.csv"),
    ("左拇指_3", "左拇指", r"temp\左拇指指尖\数据3\device_001_seg000.csv"),
    ("四指_1", "四指", r"temp\四指指尖\数据1\device_001_seg000.csv"),
    ("四指_2", "四指", r"temp\四指指尖\数据2\device_001_seg000.csv"),
    ("四指_3", "四指", r"temp\四指指尖\数据3\device_001_seg000.csv"),
    ("切换负载", "变化", r"temp\变化负载\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc\device_001_seg000.csv"),
    ("再切换", "变化", r"temp\变化负载\零负载-切换负载-零负载-再切换负载\device_001_seg000.csv"),
    ("中途1d9493", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\device_001_seg000.csv"),
    ("中途13ffca", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\最终测试目标\device_001_seg000.csv"),
]

OFF = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
       0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]


def load(path):
    df = pd.read_csv(path, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    t = df["timestamp"].to_numpy(float)
    return t - t[0], df[ch].to_numpy(float)


def smooth_tau(x, dt, tau):
    a = dt / tau
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


rows = []
print("=" * 104)
print("A. 各录制帧间隔质量（决定时间轴可信度）")
print("=" * 104)
print(f"{'录制':<12}{'帧数':>7}{'中位dt(ms)':>11}{'p95(ms)':>9}{'max(ms)':>9}{'>2×中位占比':>12}")
for name, pos, rel in DATA:
    t, X = load(os.path.join(ROOT, rel))
    dt = np.diff(t)
    dt = dt[dt > 0]
    md = np.median(dt)
    print(f"{name:<12}{len(t):>7}{1000*md:>11.2f}{1000*np.percentile(dt,95):>9.2f}"
          f"{1000*dt.max():>9.1f}{100*np.mean(dt > 2*md):>11.1f}%")

print()
print("=" * 104)
print("B. 加载沿的原始形状（相对 5s 增量归一化）")
print("=" * 104)
for name, pos, rel in DATA:
    t, X = load(os.path.join(ROOT, rel))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    tot = pd.Series(tot).rolling(3, center=True, min_periods=1).median().to_numpy()
    # 负载段
    thr = 0.15 * tot.max()
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    segs = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)
    if not segs:
        continue
    s0, s1 = segs[0]
    base = float(np.median(tot[max(0, s0 - int(2 / dt)):s0])) if s0 > 0 else float(tot[0])
    thr2 = base + 0.05 * (tot[s0:s1].max() - base)
    n_on = next((i for i in range(max(0, s0 - int(1 / dt)), s0 + 300) if tot[i] > thr2), s0)
    i5 = min(len(tot) - 1, n_on + int(round(5.0 / dt)))
    a5 = tot[i5] - base
    prof_raw = [(tot[min(len(tot) - 1, n_on + int(round(g / dt)))] - base) / a5 for g in OFF]
    sm = smooth_tau(Xu[:, int(np.argmax(Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)))], dt, 2.0)
    sm = smooth_tau(tot, dt, 2.0)
    b = float(np.median(sm[max(0, s0 - int(2 / dt)):s0])) if s0 > 0 else sm[0]
    prof_sm = [(sm[min(len(sm) - 1, n_on + int(round(g / dt)))] - b) / a5 for g in OFF]
    rows.append(dict(ds=name, pos=pos, on_t=float(tu[n_on]), base=base, jump=a5,
                     **{f"raw_{g:.2f}": v for g, v in zip(OFF, prof_raw)},
                     **{f"sm2_{g:.2f}": v for g, v in zip(OFF, prof_sm)}))
    print(f"\n[{name}] 加载沿 t={tu[n_on]:.2f}s  空载基线={base:.3f}  5s 增量={a5:.3f}")
    print("  τ(s)   :" + "".join(f"{g:>7.2f}" for g in OFF))
    print("  原始   :" + "".join(f"{v:>7.3f}" for v in prof_raw))
    print("  τ=2s   :" + "".join(f"{v:>7.3f}" for v in prof_sm))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "v6_raw_vs_smoothed_profile.csv"), index=False, encoding="utf-8-sig")

print()
print("=" * 104)
print("C. 汇总（9 组恒载 onset）")
print("=" * 104)
sub = df[df.pos != "变化"]
for tag in ("raw", "sm2"):
    cols = [f"{tag}_{g:.2f}" for g in OFF]
    m = sub[cols].median().to_numpy()
    print(f"{tag:5s} 中位:" + "".join(f"{v:>7.3f}" for v in m))
print("τ(s)     :" + "".join(f"{g:>7.2f}" for g in OFF))
sub2 = df[df.pos == "变化"]
print("\n4 份实录 onset:")
for tag in ("raw", "sm2"):
    cols = [f"{tag}_{g:.2f}" for g in OFF]
    m = sub2[cols].median().to_numpy()
    print(f"{tag:5s} 中位:" + "".join(f"{v:>7.3f}" for v in m))
print("\nsaved: results/v6_raw_vs_smoothed_profile.csv")
