# -*- coding: utf-8 -*-
"""探针 J（临时诊断）：240 s 附近逐帧内部状态 + 重载事件是否建立。

用法：python v31_probe_window.py <t0> <t1> [skip]
输出：results/v31_window_<t0>_<t1>.txt
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


def main():
    t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 235.0
    t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 265.0
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    npz = os.path.join(RES, "v31_baseline_%s.npz" % os.path.basename(DS))
    z = np.load(npz)
    el = z["el"]
    X = z["X"]
    tin = z["tin"]
    tmain = z["tmain"]
    m = (el >= t0) & (el <= t1)
    idx = np.where(m)[0]
    lines = []
    p = lines.append
    p("窗口 [%.1f, %.1f] 共 %d 帧（每 %d 帧打印一行）" % (t0, t1, len(idx), skip))
    p("%9s %8s %8s %5s %4s %4s %8s %8s %8s %8s %8s %8s %8s %4s %5s %6s %7s %8s %8s %8s" %
      ("t", "sum_in", "sum_out", "st", "ev", "kd", "A_sum", "comp_tot", "ded_unc",
       "ded_cap", "levelref", "tssmoo", "min_ts", "idl", "vlly", "vlrun", "tau",
       "A_hat", "c_appl", "g"))
    for i in idx[::skip]:
        d = dict(zip(COLS, X[i]))
        p("%9.3f %8.0f %8.0f %5.0f %4.0f %4.0f %8.0f %8.0f %8.0f %8.0f %8.1f %8.1f %8.1f %4.0f %5.0f %6.2f %7.3f %8.1f %8.1f %8.4f" %
          (el[i], tin[i], d["sum_out"], d["state"], d["ev_valid"], d["ev_kind"],
           d["A_sum"], d["comp_total"], d["ded_unclamped"], d["ded_clamped"],
           d["level_ref"], d["ts_smooth"], d["min_ts"], d["idle_now"],
           d["valley_now"], d["valley_run"], d["tau"], d["A_hat"],
           d["c_applied"], d["g"]))
    p("")
    p("窗口内 state 变化点（state/ev_valid/ev_kind/idle_now/valley_now 任一变化）：")
    prev = None
    for i in idx:
        d = dict(zip(COLS, X[i]))
        cur = (d["state"], d["ev_valid"], d["ev_kind"], d["idle_now"], d["valley_now"])
        if cur != prev:
            p("  t=%9.3f state=%5.0f ev_valid=%4.0f ev_kind=%4.0f idle=%4.0f valley=%4.0f "
              "tin=%7.0f out=%7.0f ded=%8.0f A=%8.0f levelref=%8.1f min_ts=%8.1f max_ts=%9.1f" %
              (el[i], d["state"], d["ev_valid"], d["ev_kind"], d["idle_now"],
               d["valley_now"], tin[i], d["sum_out"], d["comp_total"], d["A_sum"],
               d["level_ref"], d["min_ts"], d["max_ts"]))
            prev = cur
    out = os.path.join(RES, "v31_window_%.0f_%.0f.txt" % (t0, t1))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
