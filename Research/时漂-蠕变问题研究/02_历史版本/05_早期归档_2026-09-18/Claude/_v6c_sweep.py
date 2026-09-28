# -*- coding: utf-8 -*-
"""扫描 g_obs 的各种定标方式（临时脚本），看哪个既保住加速又不过充。

候选：
  raw      : g_obs = obs / A_corr                       （当前实现）
  clampHI  : 同上，但把 γ·A·g_obs 封顶到 CREEP_HI·A 之外再加 min(·, carry·K)
  capK     : ded_target = min(ded_target, K · carry)     （K=1.2/1.5/2.0）
  obsRef   : obs_rel = max(obs − carry·(1−1/ratio), 0)
  blend    : ded_target = carry + (ded_obs − carry) 截断到 [0, K·carry]
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, os.pardir, "v4.1flash", "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402
import glm53_v6c as M                                  # noqa: E402

B = os.path.join(os.path.dirname(os.path.dirname(SCRIPTS)), "变化负载")
RECS = [("快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                  "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                "最终测试目标", "device_001_seg000.csv"))]


def run(cls, tu, Xu, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def metrics(Y, c, tu, Xu, dtm, tot_s):
    gap = np.abs(L.med_smooth(Y.sum(axis=1), 0.5 / dtm) - tot_s)
    return dict(gapmax=float(gap.max()), gapmaxpct=100 * float(gap.max()) / float(tot_s.max()),
                amax=float(c.A.max()), gend=float(c.g))


print(f"{'录制':>8} {'变体':>9} {'gap最大':>9} {'占峰值%':>8} {'A_max':>8} {'g_end':>9}  "
      f"{'过充ADC':>9} {'过充%':>7}")
base = {}
for tag, path in RECS:
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    Y5, c5 = run(GLM53v51, tu, Xu)
    m5 = metrics(Y5, c5, tu, Xu, dtm, tot_s)
    print(f"{tag:>8} {'v5.1':>9} {m5['gapmax']:9.0f} {m5['gapmaxpct']:8.2f} "
          f"{m5['amax']:8.0f} {m5['gend']:+9.4f}")
    base[tag] = m5
    for k in (1.0, 1.1, 1.2, 1.5):
        Y, c = run(M.GLM53v6c, tu, Xu, SHAPE_RAMP_CAP=float(k))
        m = metrics(Y, c, tu, Xu, dtm, tot_s)
        dd = np.abs(Y.sum(axis=1) - Y5.sum(axis=1))
        print(f"{'':>8} {'capK=%.1f' % k:>9} {m['gapmax']:9.0f} {m['gapmaxpct']:8.2f} "
              f"{m['amax']:8.0f} {m['gend']:+9.4f}  |Δ显示|中位={np.median(dd):7.0f} max={dd.max():7.0f}")
