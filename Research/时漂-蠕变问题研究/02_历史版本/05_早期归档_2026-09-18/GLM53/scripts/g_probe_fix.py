# -*- coding: utf-8 -*-
"""GLM53 探针 4：切换瞬间跳变的来源拆解与可选修法对比

对比三种在"负载内切换(restep)"时刻的幅度重捕获策略：
  base   = f_ 脚本 v2（restep 后 A 清零，用 a_w0..a_w1 (1..3s) 窗口重捕获，
           期间补偿直通；同时 g 之前已把偏移算成蠕变，切换瞬间补偿被整体清零）
  snap   = restep 当帧直接用当前 Z 的逐通道中值快照 A（不经过 1..3s 窗口）
  blend  = A 向快照目标做指数逼近(tau=0.1s，约 0.3s 内到位)
指标：切换瞬间纯算法跳变、切换后 2s 净漂移、后段平坦度。
"""
import os
import sys
import importlib.util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
BASE = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")

_spec = importlib.util.spec_from_file_location("fv", os.path.join(HERE, "f_varying_load.py"))
fv = importlib.util.module_from_spec(_spec)
sys.modules["fv"] = fv
_spec.loader.exec_module(fv)

from g_probe_varying import ProbeV2, dedup, load_csv  # noqa: E402


class SnapV2(ProbeV2):
    """restep 当帧快照 A（去掉 1..3s 直通窗口）"""
    mode = "snap"
    tau_a = 0.1

    def process(self, ts, v):
        n_before = len(self.events)
        was_unload = self.in_load
        out = super().process(ts, v)
        if len(self.events) > n_before and self.events[-1][1] == "restep" and self.in_load:
            Z = v - (self.b if self.b is not None else 0.0)
            tgt = np.array(Z, dtype=float)
            if self.mode == "snap":
                self.A = tgt
            else:
                self.A = self.A + (tgt - self.A) * (1 - np.exp(-0.02 / self.tau_a))
            amax = float(np.max(self.A))
            self.loaded = self.A > fv.P2["loaded_frac"] * amax if amax > 1e-9 else np.zeros(self.n, bool)
            self.a_captured = True
            self.g = 0.0
            self.g2 = 0.0
            self.g_rel = np.zeros(self.n)
            self.gamma = np.ones(self.n)
            # 本帧输出随之更新（原实现本帧仍返回旧偏移结果）
            ld = self.loaded & (self.A > 1e-9)
            out = np.array(Z, dtype=float)
        return out


class BlendV2(SnapV2):
    mode = "blend"


def run(cls, X, t):
    c = cls()
    Y = np.empty_like(X, dtype=float)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i].astype(float))
    return Y, c


CASES = {"A": ("零负载-切换负载-零负载-再切换负载", [(40.98, "restep"), (60.33, "restep")]),
         "B": ("零负载-中途切换负载-零负载-切换负载", [(23.83, "restep")])}

print(f"{'数据':<4}{'策略':<7}{'t':>7}{'纯算法跳变':>12}{'显示跳变':>11}{'原始跳变':>10}{'前2s漂移':>11}{'后2s漂移':>11}{'后段std':>10}")
print("-" * 86)
rows = []
for tag, (loc, evs) in CASES.items():
    t, X, _ = load_csv(os.path.join(BASE, "变化负载", loc, "device_001_seg000.csv"))
    for name, cls in [("base", ProbeV2), ("snap", SnapV2), ("blend", BlendV2)]:
        Y, c = run(cls, X, t)
        tot = Y.sum(1)
        keep = dedup(t)
        tu = t[keep]; to = tot[keep]
        for t0, etag in evs:
            i = int(np.argmin(np.abs(tu - t0)))
            d_disp = to[i] - to[i - 1]
            d_raw = float(X.sum(1)[keep][i] - X.sum(1)[keep][i - 1])
            pre = to[(tu > t0 - 2) & (tu < t0)]
            post = to[(tu > t0) & (tu < t0 + 2)]
            later = to[(tu > t0 + 2) & (tu < t0 + 6)]
            dv = (pre[-1] - pre[0]) if len(pre) > 1 else 0
            dp = (post[-1] - post[0]) if len(post) > 1 else 0
            print(f"{tag:<4}{name:<7}{tu[i]:>7.2f}{d_disp - d_raw:>12,.0f}{d_disp:>11,.0f}{d_raw:>10,.0f}{dv:>11,.0f}{dp:>11,.0f}{later.std():>10,.0f}")
            rows.append(dict(dataset=tag, policy=name, t=round(float(tu[i]), 2),
                             algo_jump=round(d_disp - d_raw), d_disp=round(d_disp),
                             d_raw=round(d_raw), pre2s=round(dv), post2s=round(dp),
                             late_std=round(float(later.std()))))
pd.DataFrame(rows).to_csv(os.path.join(RES, "g_restep_policies.csv"), index=False, encoding="utf-8-sig")
print("\n(纯算法跳变 = 显示跳变 - 原始跳变；应尽量接近 0)")
