# -*- coding: utf-8 -*-
"""探针 L（临时诊断）：每次加载沿的「起跳环境」—— base/min_ts/level_ref/state。

定位判据：加载沿之前，补偿器是否处在「真的干净的空载态」。若 base 不是真正的空载电平
（例如仍是上一次卸载前的电平、或 min_ts 被污染），新事件的 Â 与接力就会算错。

输出：results/v31_onset_env_<tag>.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")
COLS = ("t sum_in sum_out state ev_valid ev_kind g A_sum n_loaded "
        "gam_med gam_min gam_max r_med r_wmean aggA_loaded num_loaded "
        "ded_unclamped ded_clamped ded_capped ideal_ded comp_total "
        "level_ref ts_smooth min_ts max_ts max_tot idle_now valley_now valley_run "
        "tau tau_g0 tglide A_hat inc_max c_applied stalled clamp_hits shape_hits "
        "trim_sum").split()
ONSETS = [3.34, 15.18, 41.75, 63.41, 232.23, 245.31, 258.34, 261.72, 278.63,
          289.33, 300.61, 303.42]


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def main():
    z = np.load(os.path.join(RES, "v31_baseline_%s.npz" % os.path.basename(DS)))
    el, tin, tmain, X = z["el"], z["tin"], z["tmain"], z["X"]
    D = {c: X[:, i] for i, c in enumerate(COLS)}
    tout = D["sum_out"]
    lines = []
    p = lines.append
    p("=== 每次加载沿的起跳环境（沿前 2.5~0.5 s 的窗口）===")
    p("%8s %9s %8s %8s %8s %9s %9s %8s %9s %9s %9s %9s" %
      ("沿t", "in前", "st", "ev", "idle", "state@+1s", "A_sum前", "base估",
       "level_ref", "min_ts", "ts_smooth", "A_hat@+1s"))
    for t in ONSETS:
        w0, w1 = t - 2.5, t - 0.5
        i_in = wmed(el, tin, w0, w1)
        m = (el > t) & (el <= t + 1.0)
        iA = float(np.median(D["A_sum"][m])) if m.any() else float("nan")
        iH = float(np.median(D["A_hat"][m])) if m.any() else float("nan")
        p("%8.2f %9.0f %8.0f %8.0f %8.0f %9.0f %9.0f %8.0f %9.0f %9.0f %9.0f %9.0f" %
          (t, i_in,
           wmed(el, D["state"], t - 0.5, t),
           wmed(el, D["ev_valid"], t - 0.5, t),
           1.0 if wmed(el, D["idle_now"], t - 0.5, t) > 0.5 else 0.0,
           wmed(el, D["state"], t + 1, t + 1.6),
           wmed(el, D["A_sum"], t - 0.5, t),
           wmed(el, D["sum_out"], t, t + 0.3) - 0.0,
           wmed(el, D["level_ref"], t - 0.3, t + 0.1),
           wmed(el, D["min_ts"], t - 0.3, t + 0.1),
           wmed(el, D["ts_smooth"], t - 0.3, t + 0.1), iH))
    p("")
    p("（base估 = 沿后 0~0.3 s 的显示值；A_sum前 = 沿前 0.5 s 的锚定无蠕变电平）")
    p("")
    p("=== 卸载沿的性质（是否真回到空载带）===")
    p("%8s %9s %10s %10s %9s %9s" % ("沿t", "in前", "in后@+3", "回到空载?", "idle@+1s", "state@+1s"))
    for t in (36.99, 56.94, 216.66, 240.11, 252.15, 255.67, 275.75, 282.30, 295.46):
        i_pre = wmed(el, tin, t - 2.5, t - 0.5)
        i_post = wmed(el, tin, t + 3, t + 5)
        p("%8.2f %9.0f %10.0f %10s %9.0f %9.0f" %
          (t, i_pre, i_post, "是" if i_post < 0.2 * i_pre else "否",
           wmed(el, D["idle_now"], t + 0.5, t + 1.5),
           wmed(el, D["state"], t + 0.5, t + 1.5)))
    p("")
    p("=== 第二次 onset(245.31) 与首次 onset(3.34) 的显示轨迹对照（沿后每秒）===")
    p("%6s %9s %9s %9s %9s %9s %9s" %
      ("dt(s)", "in@3.34", "out@3.34", "ded@3.34", "in@245.31", "out@245.31", "ded@245.31"))
    for dt in (0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 60):
        a = wmed(el, tin, 3.34 + dt, 3.34 + dt + 0.5)
        b = wmed(el, tmain, 3.34 + dt, 3.34 + dt + 0.5)
        c = wmed(el, tin, 245.31 + dt, 245.31 + dt + 0.5)
        d = wmed(el, tmain, 245.31 + dt, 245.31 + dt + 0.5)
        p("%6.1f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f" %
          (dt, a, b, a - b, c, d, c - d))
    out = os.path.join(RES, "v31_onset_env_%s.txt" % os.path.basename(DS))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
