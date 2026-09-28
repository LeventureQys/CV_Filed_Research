# -*- coding: utf-8 -*-
"""v2.0 PROBE-A：新录制「从零基线开始」数据集的总体表征。

目的（只读探针，不产出交付物）：
  1. 数据规模/采样率/显示域与参数集；
  2. pre / main / raw 三条流的电平范围与空载基线；
  3. 自动切段得到「空载/受载」平台序列，给出每个平台的偏移；
  4. 定位 40~48 s 区间（用户报告的基线错误估计）的逐帧明细。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402


def main():
    d = L.load_dataset(L.DS_ZERO)
    pre, main, raw = d["pre"], d["main"], d["raw"]
    el = pre["el"]
    tp, tm, tr = pre["V"].sum(1), main["V"].sum(1), raw["V"].sum(1)
    n = pre["n"]
    dt = np.diff(el)
    print(f"frames={n}  span={el[-1]-el[0]:.3f}s  dt_med={np.median(dt)*1000:.2f}ms "
          f"dt_p95={np.percentile(dt,95)*1000:.2f}ms  fps={n/(el[-1]-el[0]):.2f}")
    print("cfg(pre) =", {k: v for k, v in pre["cfg"].items() if k != "_hdr_has_ch"})
    print("cfg(main)=", {k: v for k, v in main["cfg"].items() if k != "_hdr_has_ch"})
    print(f"tot_pre  min={tp.min():.0f} max={tp.max():.0f} first={tp[0]:.0f} "
          f"med_idle_low={np.percentile(tp,5):.0f}")
    print(f"tot_main min={tm.min():.0f} max={tm.max():.0f} first={tm[0]:.0f}")
    print(f"tot_raw  min={tr.min():.0f} max={tr.max():.0f}")

    segs = L.plateau_segments(tp, el)
    print(f"\n平台数={len(segs)}")
    print(f"{'#':>3} {'kind':7s}{'t0':>8s}{'t1':>8s}{'dur':>7s}"
          f"{'pre':>9s}{'main':>9s}{'off':>9s}{'off/amp':>9s}")
    lo = float(np.percentile(tp, 5))
    amp = float(np.percentile(tp, 95)) - lo
    print(f"# 参考: 空载≈{lo:.0f}  受载≈{lo+amp:.0f}  幅度≈{amp:.0f}")
    for i, (kind, a, b) in enumerate(segs):
        p = float(np.median(tp[a:b + 1]))
        m = float(np.median(tm[a:b + 1]))
        print(f"{i:>3} {kind:7s}{el[a]:8.2f}{el[b]:8.2f}{el[b]-el[a]:7.2f}"
              f"{p:9.0f}{m:9.0f}{m-p:9.0f}{100*(m-p)/amp:9.1f}%")

    # 事件定位（用平滑总量的 40% 门限）
    k = np.ones(5) / 5.0
    tps = np.convolve(tp, k, mode="same")
    thr = lo + 0.4 * amp
    up, dn = L.edges_from_tot(tps, thr)
    print(f"\n上升沿(s): {[round(float(el[i]),2) for i in up]}")
    print(f"下降沿(s): {[round(float(el[i]),2) for i in dn]}")

    # 40~48 s 明细（0.1 s 桶）
    print("\n── t=39~50 s 明细（0.1 s 均值）──")
    print(f"{'t':>7}{'pre':>9}{'main':>9}{'off':>8}{'off/amp':>9}")
    m = (el >= 39.0) & (el <= 50.0)
    idx = np.where(m)[0]
    for i in range(0, len(idx), 10):
        j = idx[i:i + 10]
        a, b = float(np.mean(tp[j])), float(np.mean(tm[j]))
        print(f"{el[j[0]]:7.2f}{a:9.0f}{b:9.0f}{b-a:8.0f}{100*(b-a)/amp:9.1f}%")

    # 首 6 s 明细（确认「从零基线开始」）
    print("\n── t=0~6 s 明细（0.1 s 均值）──")
    m = el <= 6.0
    idx = np.where(m)[0]
    for i in range(0, len(idx), 10):
        j = idx[i:i + 10]
        a, b = float(np.mean(tp[j])), float(np.mean(tm[j]))
        print(f"{el[j[0]]:7.2f}{a:9.0f}{b:9.0f}{b-a:8.0f}")


if __name__ == "__main__":
    main()
