# -*- coding: utf-8 -*-
"""探针：为什么 1s 档在实录上「首扣时延」反而离奇地长（10~24s）？

动机：bp_exempt_sweep.py 测到 1s 档的实测首扣时刻在几份实录上远晚于理论（3.5s），
而 3s/5s 档基本贴着理论值。假设：1s 档把 `LEV_ARM_S` 也压到 1s，而 epoch 起点后 1s
（= 真实台阶后 3.5s）**快相尾巴还在爬**，短滞后电平判据会把它当成变载 → pending → hold
冻结补偿（hold_comp 在免责期内被设成 carry=0）→ 直到 u>6s 才被当成 restep 重锚。
本探针逐帧打印状态，验证/否定该假设。

输出：控制台表格（results/_exempt_1s_probe.log 由调用方 tee）。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

B = os.path.join(TEMP, "变化负载")
REC = ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                         "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"))
CFG = {"1s": dict(FAST_S=1.0, EXEMPT_AWIN=1.0 / 3.0, LEV_ARM_S=1.0),
       "3s": dict(FAST_S=3.0, EXEMPT_AWIN=1.0, LEV_ARM_S=3.0),
       "5s": dict(FAST_S=5.0, EXEMPT_AWIN=5.0 / 3.0, LEV_ARM_S=5.0)}


class Probe(GLM53v51):
    def __init__(self, n):
        super().__init__(n)
        self.log = []

    def process(self, ts, v):
        y = super().process(ts, v)
        self.log.append((float(ts), bool(self.in_load), bool(self.pending), bool(self.hold),
                         float(self.A.max()), float(self.g), int(self.ex_n),
                         float(self.carry.max()), float(self.onset_ts)))
        return y


d = L.prep(REC[1])
tu, Xu = d["tu"], d["Xu"]
tot_s = L.med_smooth(d["tot"], 0.5 / d["dtm"])
tag, path = REC
print(f"[{tag}] {d['span']:.1f}s / {len(tu)} 帧")
# 第一个负载沿（与主脚本同口径）
base = float(np.median(tot_s[:int(8 / d["dtm"])]))
on = next(i for i in range(int(5 / d["dtm"]), len(tu))
          if tot_s[i] > base + 0.05 * (tot_s.max() - base))
print(f"首个负载沿 @{tu[on]:.2f}s（平滑总量 {base:.0f} → {tot_s[on]:.0f}）")

for k, kw in CFG.items():
    c = Probe(Xu.shape[1])
    for kk, vv in kw.items():
        setattr(c, kk, vv)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    log = np.array([(a, b, cc, dd, e, f, g, h, i2) for a, b, cc, dd, e, f, g, h, i2 in c.log])
    ded = L.med_smooth((Xu - Y).sum(axis=1), 0.5 / d["dtm"])
    ts = log[:, 0]
    print(f"\n===== 免责 {k}（FAST_S={kw['FAST_S']:.2f} AWIN={kw['EXEMPT_AWIN']:.3f} "
          f"ARM={kw['LEV_ARM_S']:.2f}）=====")
    # 状态迁移点（在首个负载沿之后 40s 内）
    m = (ts >= tu[on] - 1) & (ts <= tu[on] + 45)
    idx = np.where(m)[0]
    prev = None
    for i in idx:
        st = (log[i, 1], log[i, 2], log[i, 3])
        if st != prev:
            rel = ts[i] - tu[on]
            print(f"   t=+{rel:6.2f}s  in_load={int(st[0])} pending={int(st[1])} hold={int(st[2])}"
                  f"  epoch起点@+{log[i, 8] - tu[on]:6.2f}s  A_max={log[i, 4]:7.0f}"
                  f"  g={log[i, 5]:+.4f}  扣除={ded[i]:7.0f}")
            prev = st
    # 每 2s 采样
    print("   采样（+2s 间隔）: t / pending / hold / A_max / g / 扣除")
    samp = [i for i in idx if abs((ts[i] - tu[on]) % 2.0) < 0.006]
    for i in samp[:23]:
        print(f"     +{ts[i] - tu[on]:6.2f}s  P={int(log[i, 2])} H={int(log[i, 3])}"
              f"  A={log[i, 4]:7.0f}  g={log[i, 5]:+.4f}  ded={ded[i]:7.0f}")
    # 统计：pending / hold 占空比（首个负载沿后 40s）
    mm = (ts >= tu[on]) & (ts <= tu[on] + 40)
    print(f"   [+0~40s] pending 占比 {100 * log[mm, 2].mean():5.1f}%   "
          f"hold 占比 {100 * log[mm, 3].mean():5.1f}%   "
          f"该窗内 epoch 重启 {int((np.diff(log[mm, 8]) != 0).sum())} 次   "
          f"扣除峰值 {ded[mm].max():.0f}")
