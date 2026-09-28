# -*- coding: utf-8 -*-
"""补测：多档连续加载（10N → +5N → +5N）与 kStepSuppressS=6s 抑制窗的影响。"""
import os
import sys
import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402

REC0 = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
        r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
        r"\device_001_seg000.csv")
d0 = L.prep(REC0)
a, b = int(133.84 / d0["dtm"]), int(182.0 / d0["dtm"])
uu = d0["tu"][a:b] - d0["tu"][a]
yy = d0["tot"][a:b] / d0["tot"][a + int(5.0 / d0["dtm"])] - 1.0
fit = lambda x, a1, t1, a2, t2: a1 * (1 - np.exp(-x / t1)) + a2 * (1 - np.exp(-x / t2))
P, _ = optimize.curve_fit(fit, uu, yy, p0=[0.03, 3.0, 0.05, 60.0],
                          bounds=([0, 0.2, 0, 5], [1, 60, 1, 3000]))
cr = lambda x: np.where(x > 0, fit(np.maximum(x, 0), *P), 0.0)

FS = 100.0
dt = 1.0 / FS
DUR = 600.0
tt = np.arange(0.0, DUR, dt)
N5 = 5000.0      # 5N 对应的 ADC 增量（10N 读数 = 10000）


def build(steps, ratio5=0.5):
    """steps = [(时刻, 增量倍数)]，基准 10N。ratio5 仅用于把「5N」换算成 ADC 增量"""
    s = np.zeros_like(tt)
    m = tt > 20.0
    s[m] += 10000.0 * (1 + cr(tt[m] - 20.0))
    for t0, k in steps:
        mm = tt > t0
        s[mm] += k * ratio5 * 10000.0 * (1 + cr(tt[mm] - t0))
    return s


def run(cls, sig, **kw):
    c = cls(sig.shape[1] if sig.ndim > 1 else 1)
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty(sig.shape[0])
    fl = np.zeros((sig.shape[0], 2), bool)
    for i in range(len(tt)):
        v = sig[i] if sig.ndim > 1 else np.array([sig[i]])
        Y[i] = c.process(tt[i], v)[0] * (1 if sig.ndim == 1 else sig.shape[1])
        fl[i] = (c.hold, c.pending)
    return Y, fl


def report(title, steps, ratio5=0.5):
    sig = build(steps, ratio5)
    Y, fl = run(GLM53v3, sig)
    t_last = steps[-1][0]
    nominal = 10000.0 + sum(k * ratio5 * 10000.0 for _, k in steps)
    raw_end = sig[-int(30 / dt):].mean()
    disp = Y[-int(30 / dt):].mean()
    print(f"\n{title}")
    print(f"  {'t(s)':>8}{'真实读数':>12}{'v3显示':>10}{'扣除量':>10}{'hold':>6}{'pend':>6}")
    for w in [20.0] + [s[0] for s in steps] + [DUR - 5]:
        i = min(len(tt) - 1, int(w / dt) + 20)
        print(f"  {tt[i]:8.1f}{sig[i]:12.0f}{Y[i]:10.0f}{sig[i]-Y[i]:10.0f}"
              f"{int(fl[i,0]):6d}{int(fl[i,1]):6d}")
    print(f"  → 稳态 理想 {nominal:.0f}（{nominal/10000:.2f}×10N）｜原始读数 {raw_end:.0f}"
          f"（含蠕变 +{100*(raw_end/nominal-1):.1f}%）｜v3 显示 {disp:.0f}（{disp/10000:.2f}×10N）"
          f"｜相对理想 {100*(disp-nominal)/nominal:+.1f}%")


print("=" * 100)
print("多档连续加载：真值按 10N 基准换算，蠕变律取自实测（保压 280s 蠕变 %.2f%%）" % (100 * fit(np.array([280.0]), *P)[0]))
print("=" * 100)
report("A. 10N → 保压 280s → +5N（单档）", [(300.0, 1.0)])
report("B. 10N → 保压 200s → +5N → 保压 80s → +5N（两档，间隔 80s）",
       [(220.0, 1.0), (300.0, 1.0)])
report("C. 10N → 保压 200s → +5N → 8s 后 +5N（间隔 8s，落在 6s 抑制窗附近）",
       [(220.0, 1.0), (228.0, 1.0)])
report("D. 10N → 保压 200s → +5N → 3s 后 +5N（间隔 3s，落在抑制窗内）",
       [(220.0, 1.0), (223.0, 1.0)])
report("E. 10N → 保压 280s → +2.5N（台阶仅为电平的 25%）", [(300.0, 0.5)])
report("F. 20N → 保压 280s → +5N（台阶仅为电平的 25%）", [(300.0, 1.0)], ratio5=1.0)
