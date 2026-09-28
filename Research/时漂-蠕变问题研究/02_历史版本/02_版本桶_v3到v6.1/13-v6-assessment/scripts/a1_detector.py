# -*- coding: utf-8 -*-
"""a1 检测器诊断：逐帧复刻 v6 检测统计量 d、σ_d、门限与命中，定位误触发来源。

只读取原型常量，用独立实现复刻检测器（与 glm53_v6.GLM53v6.process 的 D 段逐行一致），
不改动原型。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import REC, load_uniform, ROOT, TRACED_V6  # noqa: E402
from glm53_v6 import GLM53v6                             # noqa: E402

OUT = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")


def detector_trace(tu, X, P=GLM53v6):
    """复刻检测器 D 段（无事件建仓逻辑），返回逐帧 d / sig / thr / raw_hit。"""
    n = len(tu)
    tot = X.sum(axis=1)
    bvm = np.empty(n)
    med = []
    for i in range(n):
        med.append(tot[i])
        if len(med) > 3:
            med.pop(0)
        bvm[i] = float(np.median(med))
    dt_arr = np.diff(np.concatenate([[tu[0]], tu]))
    dt_arr[0] = 0.0
    dt_arr = np.clip(dt_arr, 0.0, 0.1)
    ts_s = np.empty(n)
    lev = np.empty(n)
    s = lev[0] = ts_s[0] = tot[0]
    for i in range(1, n):
        s += (dt_arr[i] / P.TAU_TOTAL) * (tot[i] - s)
        ts_s[i] = s
        lev[i] = lev[i - 1] + (dt_arr[i] / P.TAU_LEVEL) * (s - lev[i - 1])
    max_tot = np.maximum.accumulate(tot)
    eps = 1e-6 * (1.0 + np.abs(max_tot))

    def wmean(t0, t1):
        out = np.full(n, np.nan)
        j = 0
        for i in range(n):
            a = np.searchsorted(tu, t0[i], "right")
            b = np.searchsorted(tu, t1[i], "right")
            if b > a:
                out[i] = bvm[a:b].mean()
        return out

    lv_now = wmean(tu - P.DET_FAST, tu)
    lv_ref = wmean(tu - P.DET_FAST - P.DET_GAP - P.DET_LAG, tu - P.DET_FAST - P.DET_GAP)
    d = np.where(np.isnan(lv_now) | np.isnan(lv_ref), 0.0, lv_now - np.nan_to_num(lv_ref))
    sig = np.zeros(n)
    for i in range(n):
        if i + 1 > 40:
            h = d[max(0, i + 1 - P.DCAP): i + 1]
            sig[i] = 1.4826 * np.median(np.abs(h - np.median(h)))
    thr = np.maximum.reduce([P.DET_K * sig, P.DET_REL * np.abs(np.nan_to_num(lv_ref)),
                             P.DET_ABS_FRAC * max_tot])
    return dict(tu=tu, tot=tot, bvm=bvm, ts_s=ts_s, lev=lev, lv_now=lv_now,
                lv_ref=lv_ref, d=d, sig=sig, thr=thr, max_tot=max_tot)


def main():
    rows = []
    for name in ["切换负载-快相无责", "零负载-切换负载-零负载-再切换负载"]:
        d = load_uniform(REC[name])
        tr = detector_trace(d["tu"], d["Xu"])
        raw_hit = np.abs(tr["d"]) > np.maximum(tr["thr"], 0.0)
        np.save(os.path.join(OUT, f"a1_dettrace_{name}.npy"),
                np.vstack([tr["tu"], tr["tot"], tr["bvm"], tr["lv_now"], tr["lv_ref"],
                           tr["d"], tr["sig"], tr["thr"]]))
        idle = tr["ts_s"] < 0.10 * tr["lev"]
        # d 的分布：空闲 vs 受载
        for tag, m in (("idle", idle), ("load", ~idle)):
            if m.sum() < 100:
                continue
            dd = tr["d"][m]
            rows.append(dict(rec=name, seg=tag, n=int(m.sum()),
                             d_std=round(float(dd.std()), 2),
                             d_p99=round(float(np.percentile(np.abs(dd), 99)), 1),
                             d_max=round(float(np.abs(dd).max()), 1),
                             thr_med=round(float(np.median(tr["thr"][m])), 1),
                             sig_med=round(float(np.median(tr["sig"][m])), 2),
                             lvref_med=round(float(np.nanmedian(tr["lv_ref"][m])), 1),
                             hit_frac=round(float(raw_hit[m].mean()), 4)))
        print(f"[{name}] d_std(idle)={tr['d'][idle].std():.2f} thr_med(idle)={np.median(tr['thr'][idle]):.1f} "
              f"d_std(load)={tr['d'][~idle].std():.2f} thr_med(load)={np.median(tr['thr'][~idle]):.1f} "
              f"max|d|={np.abs(tr['d']).max():.0f} sig_med={np.median(tr['sig']):.2f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "a1_detector_stats.csv"),
                              index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
