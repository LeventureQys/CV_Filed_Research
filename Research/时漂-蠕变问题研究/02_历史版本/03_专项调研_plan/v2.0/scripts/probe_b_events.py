# -*- coding: utf-8 -*-
"""v2.0 PROBE-B：事件级表征（负载沿 / 过充 / 台阶捕获 / 偏移时间线）。

只读探针：不产出交付物，用于回答需求文档问题 1（加载过充能否优化）
与问题 2（44 s 起基线为何越飘越远）的"现象量化"部分。

定义（口径必须显式，避免歧义）：
  total(t)      = 21 通道求和（显示域，ADC）
  负载台阶 A_e   = 该次加载在 pre 流上的稳态增量（事件后 3~5 s 的 pre 平台中位 − 事件前 0.3 s 平台中位）
  过充 OS(t)    = (main(t) − main_idle_ref) / A_e − 1，其中 main_idle_ref 取该次加载前的 main 空载平台中位
  台阶捕获比 G  = (main 事件后 4~5 s 平台中位 − 事件前 main 平台中位) / A_e
  偏移 off(t)   = main(t) − pre(t)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(OUT, exist_ok=True)


def med(tot, el, t0, t1):
    m = (el >= t0) & (el <= t1)
    return float(np.median(tot[m])) if m.any() else float("nan")


def main():
    d = L.load_dataset(L.DS_ZERO)
    pre, main = d["pre"], d["main"]
    el = pre["el"]
    tp = pre["V"].sum(1)
    tm = main["V"].sum(1)
    n = pre["n"]

    # 慢门限沿检测（用于定位，不用于算法）
    k = np.ones(9) / 9.0
    tps = np.convolve(tp, k, mode="same")
    lo = float(np.percentile(tp, 3))
    hi = float(np.percentile(tp, 97))
    thr = lo + 0.5 * (hi - lo)
    up, dn = L.edges_from_tot(tps, thr)

    print(f"frames={n} span={el[-1]:.1f}s  空载≈{lo:.0f} 受载≈{hi:.0f} 门限={thr:.0f}")
    print(f"上升沿={[round(float(el[i]),2) for i in up]}")
    print(f"下降沿={[round(float(el[i]),2) for i in dn]}")

    # 逐事件分析（以上升沿为准）
    print("\n── 逐次加载事件 ──")
    print(f"{'i':>2}{'t_up':>8}{'t_down':>8}{'hold':>7}"
          f"{'pre_prev':>9}{'A_e':>8}{'pre_post':>9}"
          f"{'OS_max@1s':>11}{'OS_max@3s':>11}{'OS_max@10s':>12}"
          f"{'G(4-5s)':>9}{'off_pre':>9}{'off_post':>9}")
    rows = []
    for i, iu in enumerate(up):
        t_up = float(el[iu])
        # 对应的下降沿
        t_dn = None
        for jd in dn:
            if el[jd] > t_up + 0.5:
                t_dn = float(el[jd])
                break
        pre_prev = med(tp, el, t_up - 1.2, t_up - 0.35)
        t_dn_v = t_dn if t_dn is not None else float(el[-1])
        pre_post = med(tp, el, min(t_dn_v - 0.6, t_up + 3.0), t_dn_v - 0.15)
        A = pre_post - pre_prev
        main_prev = med(tm, el, t_up - 1.2, t_up - 0.35)
        os1 = (med(tm, el, t_up + 0.8, t_up + 1.2) - main_prev) / A - 1.0
        os3 = (med(tm, el, t_up + 2.8, t_up + 3.2) - main_prev) / A - 1.0
        os10 = (med(tm, el, t_up + 9.5, t_up + 10.5) - main_prev) / A - 1.0 \
            if (t_dn is None or t_dn - t_up > 11.0) else float("nan")
        G = (med(tm, el, t_up + 4.0, t_up + 5.0) - main_prev) / A
        off_pre = med(tm - tp, el, t_up - 1.2, t_up - 0.35)
        off_post = med(tm - tp, el, t_up + 4.0, t_up + 5.0)
        print(f"{i:>2}{t_up:8.2f}{(t_dn if t_dn else float('nan')):8.2f}"
              f"{(t_dn_v-t_up):7.1f}{pre_prev:9.0f}{A:8.0f}{pre_post:9.0f}"
              f"{100*os1:10.1f}%{100*os3:10.1f}%{100*os10:11.1f}%"
              f"{G:9.3f}{off_pre:9.0f}{off_post:9.0f}")
        rows.append((i, t_up, t_dn, pre_prev, A, os1, os3, os10, G, off_pre, off_post))

    # 偏移时间线（1 s 桶）
    print("\n── 偏移时间线（1 s 桶；off = main − pre）──")
    step = 1.0
    t = el[0]
    line = ""
    print(f"{'t':>7}{'pre':>8}{'main':>8}{'off':>8}{'off/A1':>9}")
    A1 = rows[0][4] if rows else float("nan")
    while t < el[-1]:
        m = (el >= t) & (el < t + step)
        if m.any():
            p = float(np.mean(tp[m]))
            q = float(np.mean(tm[m]))
            print(f"{t:7.1f}{p:8.0f}{q:8.0f}{q-p:8.0f}{100*(q-p)/A1:9.1f}%")
        t += step

    # 落盘逐帧降采样序列
    tb, acc = L.downsample(el, pre_tot=tp, main_tot=tm, off=tm - tp)
    import csv
    with open(os.path.join(OUT, "probeb_timeline.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "pre_tot", "main_tot", "off"])
        for i, tt in enumerate(tb):
            w.writerow([f"{tt:.3f}", f"{acc['pre_tot'][i]:.1f}",
                        f"{acc['main_tot'][i]:.1f}", f"{acc['off'][i]:.1f}"])
    print("\nwrote results/probeb_timeline.csv")


if __name__ == "__main__":
    main()
