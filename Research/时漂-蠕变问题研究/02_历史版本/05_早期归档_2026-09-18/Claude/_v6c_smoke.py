# -*- coding: utf-8 -*-
"""v6c 冒烟/回归测试（临时脚本，不属于交付物）。

A. v6c(SHAPE_CORRECT=False, ACCEL_G_INIT=False) 必须与 GLM53v51 逐帧数值等价
B. 交接连续性：|扣除(交接帧) − 扣除(前一帧)| 必须 ~0（无跳变）
C. 恒载场景：稳定时间 vs v5.1、蠕变残余
D. 两级加载（输入中途停住）：不得过充
E. 单帧掉点：不得触发假 epoch / 巨大跳变
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, os.pardir, "v4.1flash", "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)
sys.stdout.reconfigure(encoding="utf-8")

from glm53_v51 import GLM53v51                      # noqa: E402
from glm53_v6c import GLM53v6c, g_shape             # noqa: E402

DT, N = 0.01, 5
rng = np.random.default_rng(7)


def synth(load_sched, creep=0.35, tau_c=25.0, T=150.0, noise=3.0, base=40.0, drop=None):
    """load_sched: list[(t_start, level)] 阶梯载荷；形状 ROM 驱动快相，指数驱动慢相。"""
    t = np.arange(0.0, T, DT)
    X = np.zeros((len(t), N))
    for i in range(len(t)):
        X[i] = base
        for ts, lv in load_sched:
            if t[i] < ts:
                continue
            uu = t[i] - ts
            fast = float(g_shape(uu)) if uu < 5.0 else 1.0
            slow = 1.0 + creep * (1.0 - np.exp(-max(uu, 0.0) / tau_c))
            X[i] += lv * fast * slow / N
    X *= (1.0 + 0.05 * np.arange(N) / N)
    X += rng.normal(0.0, noise, X.shape)
    if drop is not None:
        j = int(drop / DT)
        X[j] *= 0.3
    return t, X


def run(cls, t, X, **kw):
    c = cls(X.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(X)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i])
    return Y, c


def ded(Y, X):
    return (X - Y).sum(axis=1)


def nep(c):
    return len(getattr(c, "epoch_t", []))


ok = True
print("=" * 100)
print("A. 回归等价性：v6c(两个开关全关) vs GLM53v51")
print("=" * 100)
for tag, sched, dr in [("恒载 3000", [(5.0, 3000.0)], None),
                       ("两级加载", [(5.0, 2000.0), (40.0, 3000.0)], None),
                       ("掉点", [(5.0, 3000.0)], 12.0)]:
    t, X = synth(sched, drop=dr)
    ya, ca = run(GLM53v51, t, X)
    yb, cb = run(GLM53v6c, t, X, SHAPE_CORRECT=False, ACCEL_G_INIT=False)
    d = float(np.max(np.abs(ya - yb)))
    same = d < 1e-9
    ok &= same
    print(f"  {tag:10s} 最大逐帧差 = {d:.3e}   {'OK' if same else 'FAIL'}")

print()
print("=" * 100)
print("B/C. 恒载场景：交接连续性 + 稳定速度 + 蠕变残余")
print("=" * 100)
t, X = synth([(5.0, 3000.0)])
for name, cls in [("v5.1", GLM53v51), ("v6c", GLM53v6c)]:
    Y, c = run(cls, t, X)
    D = ded(Y, X)
    hs = getattr(c, "handoff_ts", 0.0)
    if hs > 0:
        j = int(np.searchsorted(t, hs))
        jump = float(abs(D[j] - D[j - 1]))
    else:
        j, jump = 0, float("nan")
    fin = float(np.median(D[t > 120.0]))
    t2pct = next((tt for tt, dd in zip(t, D) if dd > 20.0), float("nan"))
    t90 = next((tt for tt, dd in zip(t, D) if dd > 0.9 * fin), float("nan"))
    print(f"  {name:5s} epoch={nep(c):2d} 首扣>20ADC @{t2pct:6.2f}s  达终值90% @{t90:6.2f}s  "
          f"A_max={c.A.max():7.1f} g_end={c.g:+.4f} 末期扣除={fin:7.1f} 交接跳变={jump:.2f} ADC")
    if name == "v6c":
        print(f"        ratio={c.shape_ratio:.4f} real_step={[round(x,2) for x in c.real_step_t]} "
              f"epoch_ts={[round(x,2) for x in c.epoch_t]} 限速命中={c._ramp_limited}")
        print(f"        ramp(A_obs,A_corr,b0,b1,t0,dur)={tuple(round(x,3) for x in c.ramp_a)}")
        print(f"        真值末期扣除 ≈ {0.35 * 3000 * (1 - np.exp(-1)):.1f} ADC")

print()
print("=" * 100)
print("D. 两级加载（输入中途停住 @30s）：不得过充")
print("=" * 100)
t, X = synth([(5.0, 2000.0), (30.0, 3000.0)])
for name, cls in [("v5.1", GLM53v51), ("v6c", GLM53v6c)]:
    Y, c = run(cls, t, X)
    D = ded(Y, X)
    over = float(np.max(D[t > 33.0] / X[t > 33.0].sum(axis=1)))
    print(f"  {name:5s} epoch={nep(c):2d} 变载后最大扣除/读数 = {over:.3f}  "
          f"(>0.6 视为过充)  A_max={c.A.max():7.1f}")

print()
print("=" * 100)
print("E. 单帧掉点 @12s（−70%）：不得触发假 epoch / 巨大跳变")
print("=" * 100)
t, X = synth([(5.0, 3000.0)], drop=12.0)
for name, cls in [("v5.1", GLM53v51), ("v6c", GLM53v6c)]:
    Y, c = run(cls, t, X)
    D = ded(Y, X)
    j = int(12.0 / DT)
    step = float(np.max(np.abs(np.diff(D[j - 100:j + 100]))))
    print(f"  {name:5s} epoch={nep(c):2d} 掉点附近扣除单帧最大变化 = {step:8.1f} ADC")

print()
print("=" * 100)
print("F. 变载场景的逐通道交接连续性")
print("=" * 100)
t, X = synth([(5.0, 3000.0), (45.0, 3200.0)])
for nm, cls in [("v5.1", GLM53v51), ("v6c", GLM53v6c)]:
    Y, c = run(cls, t, X)
    hs = getattr(c, "handoff_ts", 0.0)
    if hs <= 0:
        print(f"  {nm}: 无交接记录（末次 epoch 未走完快相）")
        continue
    j = int(np.searchsorted(t, hs))
    dd = ded(Y, X)
    print(f"  {nm}: 交接 @{hs:.2f}s  总扣除跳变 = {abs(dd[j] - dd[j - 1]):.3f} ADC  "
          f"epoch={nep(c)}")

print()
print("A 段结果：", "OK" if ok else "FAIL")

