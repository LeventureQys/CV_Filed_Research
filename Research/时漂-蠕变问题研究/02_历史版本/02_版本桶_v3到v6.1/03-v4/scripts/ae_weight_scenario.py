# -*- coding: utf-8 -*-
"""检查「10N 保压稳定后加 5N（真值 15N）会不会被现役 v3 拉回 10N」。

用实测录制拟合出真实蠕变律，再合成该场景扫描「5N 在 ADC 域占 10N 读数的比例」，
看现役算法（GLM53v3，与 src/domain/drift/drift_compensator.cpp 一一对应）把显示稳到哪。
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402

REC = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
       r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
       r"\device_001_seg000.csv")

# ---------- 1) 从实测保压段拟合蠕变律 ----------
d = L.prep(REC)
a, b = int(133.84 / d["dtm"]), int(182.0 / d["dtm"])
u = d["tu"][a:b] - d["tu"][a]
y = d["tot"][a:b] / d["tot"][a + int(5.0 / d["dtm"])] - 1.0
fit = lambda uu, a1, t1, a2, t2: a1 * (1 - np.exp(-uu / t1)) + a2 * (1 - np.exp(-uu / t2))
p, _ = optimize.curve_fit(fit, u, y, p0=[0.03, 3.0, 0.05, 60.0],
                          bounds=([0, 0.2, 0, 5], [1, 60, 1, 3000]))
print(f"实测蠕变律拟合（138.8~182s 保压段）: {p[0]*100:.2f}%·(1-e^(-t/{p[1]:.2f}s))"
      f" + {p[2]*100:.2f}%·(1-e^(-t/{p[3]:.0f}s))"
      f"   → 保压 60s 蠕变 {100*fit(np.array([60.0]), *p)[0]:.2f}%，280s {100*fit(np.array([280.0]), *p)[0]:.2f}%")

# ---------- 2) 合成「10N → 保压 → 叠加 5N」 ----------
FS, DUR = 100.0, 360.0
dt = 1.0 / FS
tt = np.arange(0.0, DUR, dt)
L10, T_LOAD, T_ADD = 10000.0, 20.0, 300.0
cr = lambda uu: np.where(uu > 0, fit(np.maximum(uu, 0), *p), 0.0)


def synth(ratio, nch=1):
    """ratio = 5N 在 ADC 域占 10N 读数的比例；nch=1 单通道等价于整阵总量口径"""
    base = np.zeros_like(tt)
    m1 = tt > T_LOAD
    base[m1] += L10 * (1 + cr(tt[m1] - T_LOAD))
    m2 = tt > T_ADD
    base[m2] += ratio * L10 * (1 + cr(tt[m2] - T_ADD))
    return np.repeat(base[:, None], nch, axis=1) / nch


def run(cls, sig, **kw):
    c = cls(sig.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty(sig.shape[0])
    flag = np.zeros((sig.shape[0], 2), bool)
    for i in range(len(tt)):
        Y[i] = c.process(tt[i], sig[i])[0] * sig.shape[1]
        flag[i] = (c.hold, c.pending)
    return Y, flag


iA = int(T_ADD / dt)
raw = synth(0.0)[:, 0] * 1.0
print("\n" + "=" * 100)
print("合成场景：加载 10N → 保压 280s → 再叠加 5N（真值 15N）。现役 v3 的显示去向")
print("=" * 100)
print(f"{'ADC台阶/10N读数':>16}{'识别':>6}{'加码前显示':>11}{'加码后最低':>11}{'末端显示':>10}"
      f"{'末端/加码前':>12}{'  结论':>16}")
for ratio in (0.15, 0.20, 0.25, 0.28, 0.30, 0.40, 0.50, 1.00):
    sig = synth(ratio)
    Y, flag = run(GLM53v3, sig)
    pre = Y[iA - 200:iA].mean()
    lo = Y[iA:iA + int(60 / dt)].min()
    end = Y[-int(30 / dt):].mean()
    trig = bool(flag[iA:iA + int(10 / dt)].any())
    ok = abs(end / pre - 1.5) < 0.10
    print(f"{ratio:16.2f}{('是' if trig else '否'):>6}{pre:11.0f}{lo:11.0f}{end:10.0f}"
          f"{end/pre:12.3f}{('  正常稳到 15N' if ok else f'  被拉回({end/pre:.2f}×10N)'):>16}")

print("\n[轨迹对照] ADC 台阶 25%（漏检） vs 50%（识别）")
for ratio in (0.25, 0.50):
    sig = synth(ratio)
    Y, flag = run(GLM53v3, sig)
    print(f"\n  -- ADC 台阶 = {ratio:.0%} × 10N 读数 --")
    print(f"  {'t(s)':>7}{'真实总量':>11}{'v3显示':>10}{'扣除量':>10}{'hold':>6}{'pend':>6}")
    for w in (298, 300, 301, 303, 305, 310, 320, 340, 360, 359.9):
        i = min(len(tt) - 1, int(w / dt))
        print(f"  {w:7.1f}{sig[i, 0]:11.0f}{Y[i]:10.0f}{sig[i, 0]-Y[i]:10.0f}"
              f"{int(flag[i, 0]):6d}{int(flag[i, 1]):6d}")

print("\n[同场景 v4 / v4r 对照]")
for ratio in (0.25, 0.50):
    sig = synth(ratio)
    line = f"  台阶 {ratio:.0%}: "
    for tag, cls, kw in (("v3 ", GLM53v3, {}),
                         ("v4 ", L.GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
                         ("v4r", L.GLM53v4r, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0))):
        Y, _ = run(cls, sig, **kw)
        pre = Y[iA - 200:iA].mean()
        end = Y[-int(30 / dt):].mean()
        line += f"{tag}→{end:7.0f}({end/pre:.2f}×10N)   "
    print(line)

# ---------- 3) 边界：加码距上次变载不足 6s（kStepSuppressS）----------
print("\n" + "=" * 100)
print("附加边界：kStepSuppressS=6s —— 时隔 <6s 的第二次变化不允许 restep")
print("=" * 100)
FS2, DUR2 = 100.0, 400.0
tt2 = np.arange(0.0, DUR2, 1 / FS2)
for gap_s in (20.0, 8.0, 5.0, 3.0):
    sig = np.zeros_like(tt2)
    m1 = tt2 > 20.0
    sig[m1] += L10 * (1 + cr(tt2[m1] - 20.0))
    t2 = 320.0
    m2 = tt2 > t2
    sig[m2] += 0.5 * L10 * (1 + cr(tt2[m2] - t2))
    t3 = t2 + gap_s
    m3 = tt2 > t3
    sig[m3] += 0.5 * L10 * (1 + cr(tt2[m3] - t3))
    c = GLM53v3(1)
    Y = np.empty(len(tt2))
    for i in range(len(tt2)):
        Y[i] = c.process(tt2[i], np.array([sig[i]]))[0]
    i3 = int(t3 / dt)
    end = Y[-int(30 / 100):].mean()
    true_end = sig[-1]
    print(f"  10N→(+5N @320s)→(+5N @{t3:.0f}s, 间隔 {gap_s:.0f}s): "
          f"真值 {true_end:6.0f}  末端显示 {end:6.0f}  偏差 {100*(end-true_end)/true_end:+6.1f}%")
