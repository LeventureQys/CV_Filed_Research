# -*- coding: utf-8 -*-
"""r6：κ（Â 的上限系数）消融 —— 「宁可欠报」到底要付多少代价？

v6 里 Â 被夹在 [inc, κ·inc]（κ = 1.30 onset / 1.12 restep），所以"预测的那部分前置量"
最多是当前实测增量的 30%。把 κ 压到 1.00 就等于关掉预判（显示只跟实测走，不再冲高）。
本脚本扫 κ_onset ∈ {1.00,1.10,1.20,1.30}（κ_restep 同步 −0.18），测三个指标：
  · 加载后 10 s 内最大正偏差（过充，%×阶跃）
  · T_stable（总通道口径，中位）
  · 1 s / 2 s 时刻误差（%×阶跃）
产出 results/kappa_ablation.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FLASH = os.path.dirname(os.path.dirname(OUT))
TEMP = os.path.dirname(FLASH)
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, os.path.join(FLASH, "progress", "04-v5", "scripts"))
import ad_lib as L  # noqa: E402
sys.path.insert(0, os.path.join(FLASH, "progress", "07-v6", "scripts"))
from glm53_v6 import GLM53v6  # noqa: E402

CASES = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"),
          {"右拇指指尖": 17, "左拇指指尖": 18, "四指指尖": 11}[loc])
         for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]


def stable_time(tu, Y, step, lead_s=1.0, hold_s=30.0, tol_frac=0.05):
    """返回「自 tu[lead_s] 起算」的稳定时长（s）。"""
    n = len(Y)
    H = int(hold_s / (tu[1] - tu[0]))
    tol = tol_frac * step
    for k in range(n):
        e = min(n, k + H)
        if e - k < min(H, n):
            break
        if np.max(np.abs(Y[k:e] - Y[k])) <= tol:
            return float(tu[k] - tu[0]) - lead_s
    return np.nan


def main():
    rows = []
    for name, path, ch in CASES:
        d = L.prep(path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        y = Xu[:, ch]
        w = max(1, int(0.10 / dt))
        ys = L.med_smooth(y, w)
        dd = np.zeros_like(ys)
        dd[w:-w] = ys[2 * w:] - ys[:-2 * w]
        k = int(np.argmax(dd))
        pre = float(np.median(y[max(0, k - int(2.0 / dt)):k]))
        tgt = float(np.median(y[k + int(5.0 / dt):k + int(7.0 / dt)]))
        step = tgt - pre
        if step <= 0:
            continue
        for kap in (1.00, 1.10, 1.20, 1.30):
            kw = {"KAPPA_ONSET": kap, "KAPPA_RESTEP": max(kap - 0.18, 0.9)}
            Y, _c = L.run_algo(tu, Xu, GLM53v6, **kw)
            Yc = Y[:, ch]
            Yt = Y.sum(axis=1)
            step_tot = (float(np.median(Yt[k + int(5.0 / dt):k + int(7.0 / dt)])
                              - np.median(Yt[max(0, k - int(2.0 / dt)):k])))
            tS = stable_time(tu[k - int(1.0 / dt):], Yc[k - int(1.0 / dt):], step)
            seg = Yc[k:k + int(10.0 / dt)]
            rows.append(dict(rec=name, kappa=kap,
                             over_pct=100 * float(np.max(seg - (pre + step))) / step,
                             T_stable=tS,
                             T_stable_tot=stable_time(tu[k - int(1.0 / dt):], Yt[k - int(1.0 / dt):], step_tot)
                             if step_tot > 0 else np.nan,
                             err_1s=100 * float(np.median(Yc[k + int(0.9 / dt):k + int(1.1 / dt)])
                                                - (pre + step)) / step,
                             err_2s=100 * float(np.median(Yc[k + int(1.9 / dt):k + int(2.1 / dt)])
                                                - (pre + step)) / step))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(RES, "kappa_ablation.csv"), index=False, encoding="utf-8-sig")
    g = r.groupby("kappa").agg(过充中位=("over_pct", "median"), 过充最大=("over_pct", "max"),
                               过充最小=("over_pct", "min"),
                               T_stable主_中位=("T_stable", "median"),
                               T_stable总_中位=("T_stable_tot", "median"),
                               err1s中位=("err_1s", "median"), err2s中位=("err_2s", "median")).round(2)
    print("== κ 消融（恒载 9 组，13 份 onset 中的 9 组；κ_restep = κ−0.18）==")
    print(g.to_string())
    print("\n-> results/kappa_ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
