# -*- coding: utf-8 -*-
"""v2.0 PROBE-C：加载沿的细粒度形态（回答需求问题 1「加上负载的过充」）。

对每次加载沿，给出：
  · pre / main / offset 在沿后 0~3 s 的 20 ms 分辨率轨迹；
  · main 是否**超过自身稳态值**（= 真过充，显示冲到比最终值更高的地方再回落），
    以及超过多少、在什么时刻、持续多久；
  · main 到达稳态 ±2% / ±5% 带的时间（沿前基线为 0、A_e 为满量程）；
  · pre 本身上升 10%~90% 的耗时（真值爬升速度，判断"能不能更快"）。
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


def rise_time(el, tot, i0, base, target, lo_frac=0.1, hi_frac=0.9):
    """返回 (t10, t90, dur)，以 base 为 0、target 为满量程。"""
    lo = base + lo_frac * (target - base)
    hi = base + hi_frac * (target - base)
    t10 = t90 = None
    for i in range(i0, len(el)):
        if t10 is None and tot[i] >= lo:
            t10 = float(el[i])
        if tot[i] >= hi:
            t90 = float(el[i])
            break
    if t10 is None or t90 is None:
        return None, None, float("nan")
    return t10, t90, t90 - t10


def main():
    d = L.load_dataset(L.DS_ZERO)
    pre, main = d["pre"], d["main"]
    el = pre["el"]
    tp = pre["V"].sum(1)
    tm = main["V"].sum(1)
    n = pre["n"]

    k = np.ones(9) / 9.0
    tps = np.convolve(tp, k, mode="same")
    lo = float(np.percentile(tp, 3))
    hi = float(np.percentile(tp, 97))
    thr = lo + 0.5 * (hi - lo)
    up, dn = L.edges_from_tot(tps, thr)
    # 去重（相邻帧的重复沿）
    ups = []
    for i in up:
        if not ups or el[i] - el[ups[-1]] > 1.0:
            ups.append(i)

    print(f"n={n} span={el[-1]:.1f}s  空载≈{lo:.0f} 受载≈{hi:.0f}")
    print(f"加载沿(s) = {[round(float(el[i]),2) for i in ups]}")
    print(f"卸载沿(s) = {[round(float(el[i]),2) for i in dn]}")

    # ── 逐沿：过充 / 稳定时间 / 真值爬升 ──
    print("\n── 逐次加载：真过充（main 超过自身稳态值）与到带时间 ──")
    print(f"{'i':>2}{'t_up':>8}{'base':>7}{'A_e':>7}{'main_ss':>8}"
          f"{'main_peak':>10}{'OS_true':>9}{'t_peak':>8}"
          f"{'T2%':>7}{'T5%':>7}{'pre10-90':>9}")
    rec = []
    for ii, iu in enumerate(ups):
        t_up = float(el[iu])
        base_pre = med(tp, el, t_up - 1.2, t_up - 0.35)
        base_main = med(tm, el, t_up - 1.2, t_up - 0.35)
        # 该次加载的卸载时刻
        t_dn = None
        for jd in dn:
            if el[jd] > t_up + 1.0:
                t_dn = float(el[jd])
                break
        if t_dn is None:
            t_dn = float(el[-1])
        pre_ss = med(tp, el, t_dn - 1.0, t_dn - 0.2)
        main_ss = med(tm, el, t_dn - 1.0, t_dn - 0.2)
        A = pre_ss - base_pre
        m = (el >= t_up - 0.05) & (el <= min(t_dn, t_up + 30.0))
        seg = np.where(m)[0]
        peak_i = seg[int(np.argmax(tm[seg]))]
        main_peak = float(tm[peak_i])
        os_true = (main_peak - main_ss) / A
        # 到带时间（以沿前 main 基线为 0）
        t2 = t5 = None
        for i in range(iu, min(n, iu + 4000)):
            if t2 is None and abs(tm[i] - main_ss) <= 0.02 * A:
                t2 = float(el[i]) - t_up
            if t5 is None and abs(tm[i] - main_ss) <= 0.05 * A:
                t5 = float(el[i]) - t_up
        pre_ss_abs = pre_ss
        _, _, dpre = rise_time(el, tp, iu, base_pre, pre_ss_abs)
        print(f"{ii:>2}{t_up:8.2f}{base_pre:7.0f}{A:7.0f}{main_ss:8.0f}"
              f"{main_peak:10.0f}{100*os_true:8.1f}%{float(el[peak_i])-t_up:8.2f}"
              f"{(t2 if t2 is not None else float('nan')):7.2f}"
              f"{(t5 if t5 is not None else float('nan')):7.2f}{dpre:9.2f}")
        rec.append((t_up, base_pre, A, base_main, main_ss, main_peak, os_true, t2, t5, dpre))

    # ── 细节轨迹：逐沿 20 ms 输出 ──
    lines = []
    for ii, iu in enumerate(ups):
        t_up = float(el[iu])
        t_dn = None
        for jd in dn:
            if el[jd] > t_up + 1.0:
                t_dn = float(el[jd])
                break
        if t_dn is None:
            t_dn = float(el[-1])
        base_pre = med(tp, el, t_up - 1.2, t_up - 0.35)
        pre_ss = med(tp, el, t_dn - 1.0, t_dn - 0.2)
        A = max(pre_ss - base_pre, 1.0)
        lines.append(f"\n### 沿 #{ii}  t_up={t_up:.2f}s  t_down={t_dn:.2f}s  "
                     f"base={base_pre:.0f}  A_e={A:.0f}")
        lines.append(f"{'t':>8}{'pre':>8}{'main':>8}{'pre-b':>8}{'main-b':>8}"
                     f"{'(main-pre)/A':>14}")
        t = t_up - 0.20
        while t <= min(t_up + 3.0, el[-1]):
            m = (el >= t) & (el < t + 0.02)
            if m.any():
                p = float(np.mean(tp[m]))
                q = float(np.mean(tm[m]))
                lines.append(f"{t-t_up:8.2f}{p:8.0f}{q:8.0f}{p-base_pre:8.0f}"
                             f"{q-base_pre:8.0f}{100*(q-p)/A:13.1f}%")
            t += 0.02
    with open(os.path.join(OUT, "probec_traces.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote results/probec_traces.txt ({len(lines)} lines)")

    # ── 控制台打印前 5 个沿的轨迹 ──
    for ii in range(min(5, len(ups))):
        t_up = rec[ii][0]
        sel = [ln for ln in lines if ln.startswith(f"### 沿 #{ii}")]
        idx = lines.index(sel[0]) if sel else None
        if idx is not None:
            block = []
            j = idx
            while j < len(lines) and not lines[j].startswith("###") and j > 0:
                j += 1
            # 取到下一个 ### 之前
            j2 = idx + 1
            while j2 < len(lines) and not lines[j2].startswith("###"):
                j2 += 1
            block = lines[idx:j2]
            print("\n".join(block[::5]))


if __name__ == "__main__":
    main()
