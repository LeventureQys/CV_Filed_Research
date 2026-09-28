# -*- coding: utf-8 -*-
"""论文配图共用的数据装载与口径（全部来自 temp/ 实采录制，与 glm53_v51.py 共用同一份数据）。

不产生任何合成数据；合成场景仅由 pf4 通过 glm53_v51.py 现算。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # temp/v4.1flash/paper
FLASH = os.path.dirname(os.path.dirname(ROOT))                                          # temp/v4.1flash
TEMP = os.path.dirname(FLASH)                                          # temp
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_lib as L                                                      # noqa: E402

# ── 恒载 9 组（显示域，力值 N；主通道 ch17 / ch18 / ch11）─────────────
STATIC = [(loc, f"数据{i}") for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
STATIC_MAIN = {"右拇指指尖": 17, "左拇指指尖": 18, "四指指尖": 11}

# ── 4 份实采录制（ADC 域，21 ch）──────────────────────────────────
B = os.path.join(TEMP, "变化负载")
RECS = [
    ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                       "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
    ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
    ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
    ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv")),
]
# 论文正文引用目标录制（与 §5 触发机制台账对应）
TARGET_REC = "中途切换-13ffca"
TARGET_CSV = dict(RECS)[TARGET_REC]


def static_path(loc, name):
    return os.path.join(TEMP, loc, name, "device_001_seg000.csv")


def load_static():
    """返回 [(标签, dict(tu, dtm, x, tot, tot_s, s0, s1, amp)), ...]（已重采样到等间隔网格，
    与 scripts/w2_repeat.py 同口径）。s1 已退掉负载段末端的卸载沿（0.29 s），
    画曲线时不会在末尾出现一条与算法无关的掉零竖线。"""
    out = []
    for loc, name in STATIC:
        p = static_path(loc, name)
        if not os.path.exists(p):
            print(f"[skip] 缺失 {p}")
            continue
        d = L.prep(p)
        tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
        x = Xu[:, STATIC_MAIN[loc]]
        tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
        segs = L.find_periods(d["tot"], dtm)
        s0, s1 = segs[0]
        s0 = int(np.searchsorted(tu, tu[min(s0, len(tu) - 1)]))
        s1 = int(np.searchsorted(tu, min(d["t"][s1], d["span"]))) - int(round(0.29 / dtm))
        amp = float(np.mean(x[s0:s1]) - np.mean(x[:max(1, s0)]))
        out.append((f"{loc}/{name}", dict(tu=tu, dtm=dtm, x=x, tot=d["tot"], tot_s=tot_s,
                                          s0=s0, s1=s1, amp=amp)))
    return out


def load_rec(path, main_ch=None):
    d = L.prep(path)
    d["tot_s"] = L.med_smooth(d["tot"], 0.5 / d["dtm"])
    if main_ch is None:
        amp = d["Xu"].max(axis=0) - d["Xu"].min(axis=0)
        main_ch = int(np.argmax(amp))
    d["main_ch"] = main_ch
    d["x"] = d["Xu"][:, main_ch]
    return d


def ema(x, dt, tau):
    a = dt / tau
    y = np.empty_like(x)
    acc = float(x[0])
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


PROFILE_GRID = [0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def fast_profile(d):
    """快相归一化轮廓：amp_ref = 平滑原始(onset+5s) − 空载基线，f(s) = (平滑(onset+s) − 基线)/amp_ref。

    与 temp/v4.1flash/scripts/w2_repeat.py 同口径（平滑 τ=2.0 s，onset 为越过
    基线上方 5% 台阶高度的首帧；曲线取该数据集的受载主通道）。
    """
    tu, dtm = d["tu"], d["dtm"]
    tot = d["tot"]
    s0, s1 = L.find_periods(tot, dtm)[0]
    base_tot = tot[:s0].mean()
    thr2 = base_tot + 0.05 * (tot[s0:s1].max() - base_tot)
    n_on = next((i for i in range(s0, min(s0 + 300, len(tu))) if tot[i] > thr2), s0)
    Xs = ema(d["x"], dtm, 2.0)
    b = Xs[:s0].mean()
    a5 = Xs[n_on + int(round(5.0 / dtm))] - b
    prof = np.array([(Xs[n_on + int(round(g / dtm))] - b) / a5 for g in PROFILE_GRID])
    x = d["x"]
    amp = float(x[s0:s1].mean() - x[:s0].mean())
    return prof, dict(onset_s=float(tu[n_on]), start_s=float(tu[s0]), end_s=float(tu[s1]),
                      amp_main=amp, dtm=dtm)


def onset_index(d, frac=0.05):
    """负载起点：总量平滑曲线越过 基线上方 5% 台阶高度 的第一帧。"""
    tot_s = d["tot_s"]
    base = float(np.median(tot_s[:max(1, int(3.0 / d["dtm"]))]))
    hi = float(tot_s.max())
    thr = base + frac * (hi - base)
    idx = np.where(tot_s > thr)[0]
    return int(idx[0]) if len(idx) else 0, base


def make_traced(base):
    """给参考原型加 epoch 起点记录（_begin / _restep 各记一次）。"""
    class _Traced(base):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    return _Traced


def run_algo(tu, Xu, traced=True, **kw):
    """跑参考原型 glm53_v51（与 C++ 逐帧一致的实现）；traced=True 时同时记录 epoch 起点。"""
    from glm53_v51 import GLM53v51
    cls = make_traced(GLM53v51) if traced else GLM53v51
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def save_table(df, name):
    p = os.path.join(RES, name)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  -> {os.path.relpath(p, FLASH)}")
    return p
