# -*- coding: utf-8 -*-
"""合成场景：v5 的「零点被压到 0（负值→显示层钳 0）」到底会在什么情况下发生、持续多久。

蠕变律取自实测（temp/v4.1flash/Document/03）：c(t) = 15.64% · (1 − e^(−t/199s))（对数/指数等价近似）。
通道模型：3 通道权重 0.5/0.3/0.2，阵列满量程 10000 ADC(=10N)。
输出：每个场景的零点（卸载后）显示轨迹与「被钳 0」时长，v3 vs v5 对照。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

FS = 100.0
W = np.array([0.5, 0.3, 0.2])
CREEP = 0.1564
TAU_C = 199.0


def build(loads):
    """loads: [(t_start, t_end, level_adc)]；逐通道生成 raw（含蠕变）"""
    t_end = max(e for _, e, _ in loads) + 20.0
    t = np.arange(0.0, t_end, 1.0 / FS)
    X = np.zeros((len(t), len(W)))
    rng = np.random.default_rng(7)
    for ts, te, lv in loads:
        m = (t >= ts) & (t < te)
        f = 1.0 + CREEP * (1.0 - np.exp(-(t[m] - ts) / TAU_C))
        X[m] += lv * f[:, None] * W[None, :]
    X += rng.normal(0.0, 2.0, X.shape)      # ~2 ADC/通道噪声
    return t, X


def run(cls, t, X):
    c = cls(X.shape[1])
    Y = np.empty_like(X)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i])
    return Y


SCEN = {
    "S1 10N 长保压 180s → 直接卸载":        [(20, 200, 10000)],
    "S2 10N→(100s)+5N→1.5s 后卸载":         [(20, 121.5, 10000), (120, 121.5, 5000)],
    "S3 10N→(100s)+5N→4s 后卸载":           [(20, 124.0, 10000), (120, 124.0, 5000)],
    "S4 10N 保压 15s → 卸载":               [(20, 35, 10000)],
    "S5 10N→(60s) 部分卸载到 2.5N（留载）":  [(20, 80, 10000), (60, 80, -7500)],
}
UNLOAD = {"S1 10N 长保压 180s → 直接卸载": 200.0,
          "S2 10N→(100s)+5N→1.5s 后卸载": 121.5,
          "S3 10N→(100s)+5N→4s 后卸载": 124.0,
          "S4 10N 保压 15s → 卸载": 35.0,
          "S5 10N→(60s) 部分卸载到 2.5N（留载）": 60.0}

print(f"{'场景':38s} {'算法':4s} {'卸载后 0~1s 显示':>16s} {'0~4s 显示均值':>13s} "
      f"{'最低显示':>9s} {'被钳0时长':>9s} {'恢复>0 耗时':>11s}")
print("-" * 118)
for name, loads in SCEN.items():
    t, X = build(loads)
    e = int(UNLOAD[name] * FS)
    raw = X.sum(axis=1)
    for aname, cls in (("v3", GLM53v3), ("v5", GLM53v5)):
        Y = run(cls, t, X)
        s = Y.sum(axis=1)
        w1 = s[e:e + int(1.0 * FS)]
        w4 = s[e:e + int(4.0 * FS)]
        clamp = s[e:] <= 1.0
        k = np.where(~clamp)[0]
        back = (k[0] / FS) if len(k) else float("nan")
        print(f"{name:38s} {aname:4s} {w1.min():8.0f}~{w1.max():<7.0f} {w4.mean():13.0f} "
              f"{s[e:e+int(6*FS)].min():9.0f} {clamp[:int(6*FS)].sum()/FS:8.2f}s {back:10.2f}s")
    print(f"{'':38s} raw  {raw[e:e+int(1.0*FS)].min():8.0f}~{raw[e:e+int(1.0*FS)].max():<7.0f} "
          f"{raw[e:e+int(4.0*FS)].mean():13.0f}")

# S2 / S3 细节轨迹
for name in ("S2 10N→(100s)+5N→1.5s 后卸载", "S3 10N→(100s)+5N→4s 后卸载"):
    t, X = build(SCEN[name])
    e = int(UNLOAD[name] * FS)
    print(f"\n── {name}：卸载沿附近逐 0.25s（原始 / v3 / v5；显示层把 ≤0 钳成 0）")
    for k in range(-4, 25):
        i = e + int(k * 0.25 * FS)
        if i < 0 or i >= len(t):
            continue
        print(f"   t={t[i]:7.2f}s  raw={X[i].sum():8.0f}")
    for aname, cls in (("v3", GLM53v3), ("v5", GLM53v5)):
        Y = run(cls, t, X)
        s = Y.sum(axis=1)
        vals = " ".join(f"{s[e+int(k*0.25*FS)]:7.0f}" for k in range(-4, 25)
                        if 0 <= e + int(k * 0.25 * FS) < len(t))
        print(f"   {aname}: {vals}")
