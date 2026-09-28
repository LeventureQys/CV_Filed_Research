# -*- coding: utf-8 -*-
"""探针 K（临时诊断）：按「加载/卸载事件」对照显示补偿量，定位基线丢失。

对每个沿取一个「稳态窗」= 沿后 [20,40] s（避开快相与慢相过渡），比较：
  in  = pre 总量中位        （算法输入 = 无补偿时应显示的值）
  out = main 总量中位       （现场显示）
  rep = 当前 C++ 本体回放总量中位
  ded = in − out            （算法实际扣掉的量）
把「同一个负载反复加载」的各段并排，就能看出哪一段丢了基线。

输出：results/v31_cycle_<tag>.txt
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


def movavg(x, win):
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    out = np.empty(n)
    half = win // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def main():
    z = np.load(os.path.join(RES, "v31_baseline_%s.npz" % os.path.basename(DS)))
    el, tin, tmain, traw, X = z["el"], z["tin"], z["tmain"], z["traw"], z["X"]
    D = {c: X[:, i] for i, c in enumerate(COLS)}
    tout = D["sum_out"]
    t_end = float(el[-1])

    sm = movavg(tin, 21)
    d = np.nan_to_num(np.array(
        [wmed(el, sm, t, t + 1.2) - wmed(el, sm, t - 0.4, t) for t in el]))
    hot = np.abs(d) > 900.0
    edges, i, n = [], 0, len(el)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        k = i + int(np.argmax(np.abs(d[i:j])))
        edges.append((float(el[k]), 1 if d[k] > 0 else -1, k))
        i = j

    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("帧 %d 时长 %.1f s" % (len(el), t_end - el[0]))
    p("")
    p("=== 沿 + 沿后稳态窗 [20,40] s 的补偿量 ===")
    p("（ded = 输入 − 显示；ded≈0 ⇒ 该段完全没有补偿 = 基线丢失）")
    p("%8s %3s %9s %10s %10s %9s %8s %10s %9s %8s %10s %9s" %
      ("沿t", "向", "Δin", "in@+20/40", "out现场", "ded现场", "ded场%",
       "out回放", "ded回放", "ded回%", "A_sum场", "levelref"))
    for (t, dirn, k) in edges:
        a = wmed(el, tin, t - 2.5, t - 0.5)
        b = wmed(el, tin, t + 3, t + 5)
        w0, w1 = t + 20, min(t + 40, t_end - 0.1)
        if w1 - w0 < 5:
            w0, w1 = t + 10, min(t + 30, t_end - 0.1)
        if w1 - w0 < 5:
            continue
        i_in = wmed(el, tin, w0, w1)
        i_out = wmed(el, tmain, w0, w1)
        i_rep = wmed(el, tout, w0, w1)
        A = wmed(el, D["A_sum"], w0, w1)
        p("%8.2f %3s %9.0f %10.0f %10.0f %9.0f %8.1f %10.0f %9.0f %8.1f %10.0f %9.0f" %
          (t, "UP" if dirn > 0 else "DN", b - a, i_in, i_out, i_in - i_out,
           0.0 if abs(i_in) < 1 else 100.0 * (i_in - i_out) / abs(i_in),
           i_rep, i_in - i_rep,
           0.0 if abs(i_in) < 1 else 100.0 * (i_in - i_rep) / abs(i_in),
           A, wmed(el, D["level_ref"], w1 - 5, w1)))
    p("")
    p("=== 每次「重新加载」后 补偿量 vs 时间（与首次 onset 对照）===")
    p("（每行为一个加载沿；列为沿后 5/10/20/40/60/90/120 s 的 ded）")
    onsets = [(t, k) for (t, dirn, k) in edges if dirn > 0]
    p("%8s %9s %9s %9s %9s %9s %9s %9s" %
      ("沿t", "+5s", "+10s", "+20s", "+40s", "+60s", "+90s", "+120s"))
    for (t, k) in onsets:
        vals = []
        for dt in (5, 10, 20, 40, 60, 90, 120):
            if t + dt + 2 > t_end:
                vals.append(float("nan"))
                continue
            vals.append(wmed(el, tin, t + dt, t + dt + 2) -
                        wmed(el, tmain, t + dt, t + dt + 2))
        p("%8.2f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f" % (t, *vals))
    p("")
    p("=== 首次 onset(3.34 s) 后 逐通道补偿量 vs 第二次 onset(245.31 s) 后 ===")
    v = z["edges"] if "edges" in z.files else None
    pre_v = None
    ds = L.load_dataset(DS)
    pre_v = ds["pre"]["V"]
    main_v = ds["main"]["V"]
    for tag, t in (("首次 onset 3.34s", 3.34), ("重载 onset 245.31s", 245.31)):
        win = (t + 40, min(t + 60, t_end - 0.1))
        din = np.array([wmed(el, pre_v[:, c], *win) for c in range(21)])
        dout = np.array([wmed(el, main_v[:, c], *win) for c in range(21)])
        p("  --- %s  稳态窗 [%.1f,%.1f] ---" % (tag, win[0], win[1]))
        p("      Δin %+.0f  Δout %+.0f  ded %+.0f" %
          (din.sum(), dout.sum(), (din - dout).sum()))
        o = np.argsort(-np.abs(din))
        p("      " + " ".join("ch%d:%+.0f/%+.0f" % (c, din[c], din[c] - dout[c])
                              for c in o[:7]))
    out = os.path.join(RES, "v31_cycle_%s.txt" % os.path.basename(DS))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
