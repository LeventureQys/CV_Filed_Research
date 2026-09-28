# -*- coding: utf-8 -*-
"""t3b_dbg_stable.py（调试件）—— 定位 T_stable 口径差异：本任务 t3b_settle vs 第一轮 r4。

对同一臂（v6）、同一录制、同一序列，并排计算：
  * t3b_settle.t_stable / t_stable_ev（T1-A 冻结口径实现）
  * r4_filter_tradeoff.stable_time（第一轮口径，逐字复制在下方）
  * 以及 max|Y[k:e] − Y[k]| 在若干 k 处（诊断用）

运行：python scripts/t3b_dbg_stable.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_ad_lib as L      # noqa: E402
import t3b_arms as A        # noqa: E402
import t3b_common as C      # noqa: E402
import t3b_settle as ST     # noqa: E402
import t3b_glm53_v6 as G6   # noqa: E402


def stable_time_r4(tu, Y, i0, step, hold_s=30.0, tol_frac=0.05):
    """逐字复制 `13-v6-assessment/scripts/r4_filter_tradeoff.py::stable_time`。"""
    n = len(Y)
    H = int(hold_s / (tu[1] - tu[0]))
    tol = tol_frac * step
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):
            break
        if np.max(np.abs(Y[k:e] - Y[k])) <= tol:
            return float(tu[k] - tu[i0])
    return np.nan


def onset_index_r4(y, dt):
    """逐字复制 r4 的真沿定位。"""
    w = max(1, int(0.10 / dt))
    ys = L.med_smooth(y, w)
    d = np.zeros_like(ys)
    d[w:-w] = ys[2 * w:] - ys[:-2 * w]
    return int(np.argmax(d))


def main():
    ev, cache = A.load_all(verbose=False)
    for name in ("左拇指指尖/数据2", "四指指尖/数据1", "右拇指指尖/数据1"):
        cc = cache[name]
        tu, Xu, dt, tot = cc["tu"], cc["Xu"], cc["dt"], cc["tot"]
        ch = A.MAIN_CH[name]
        comp = G6.GLM53v6(Xu.shape[1])
        Y = np.empty_like(Xu)
        for i in range(len(tu)):
            Y[i] = comp.process(tu[i], Xu[i])
        ytot = Y.sum(axis=1)
        k5 = max(1, int(round(0.5 / dt)))
        ytot_s = L.med_smooth(ytot, k5)
        r = ev[(ev.rec == name) & (ev.kind == "onset")].iloc[0]
        k0 = int(np.searchsorted(tu, float(r.t_on)))
        print("\n=== %s  onset t_on=%.2f  k0=%d ===" % (name, r.t_on, k0))
        for tag, Ys in (("未平滑", ytot), ("0.5s 中值", ytot_s)):
            J1, pre1, post1 = ST.amp_5s(Ys, k0)
            Jr4 = float(np.median(Ys[k0 + int(5 / dt):k0 + int(7 / dt)])
                        - np.median(Ys[max(0, k0 - int(2 / dt)):k0]))
            t1 = ST.t_stable(tu, Ys, k0, J1)
            t2 = ST.t_stable_ev(tu, Ys, k0, J1, None)
            tr4 = stable_time_r4(tu[k0 - int(1 / dt):], Ys[k0 - int(1 / dt):],
                                 int(1 / dt), Jr4)
            print("  [%s] J(4~6s)=%.3f J(r4 5~7s)=%.3f  tol=%.3f" % (tag, J1, Jr4, 0.05 * J1))
            print("     本任务 t_stable=%s  t_stable_ev=%s   r4 stable_time=%.3f"
                  % (_f(t1[0]), _f(t2[0]), tr4))
        # 诊断：几个 k 处的窗口最大偏差（用未平滑序列）
        for kk in (10, 50, 100, 200, 500, 1000, 2000, 3000, 3050):
            k = k0 + kk
            e = min(len(ytot), k + int(30 / dt))
            if k >= len(ytot) - 10:
                continue
            mx = float(np.max(np.abs(ytot[k:e] - ytot[k])))
            print("     k=t_on+%5.2fs  win=%5.2fs  max|Y-Y[k]|=%.4f  (tol=%.4f)"
                  % (kk / 100.0, (e - k) * dt, mx, 0.05 * ST.amp_5s(ytot, k0)[0]))


def _f(x):
    return "NaN" if not np.isfinite(x) else "%.3f" % x


if __name__ == "__main__":
    sys.exit(main())
