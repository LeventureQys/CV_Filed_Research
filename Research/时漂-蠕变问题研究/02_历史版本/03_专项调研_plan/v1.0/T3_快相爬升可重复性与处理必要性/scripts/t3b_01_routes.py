# -*- coding: utf-8 -*-
"""t3b_01_routes.py —— T3-Q6 / T3-Q7：两条现状路线在 13 份录制上的稳定时间与代价。

臂（全部在 13 份录制 / 100 Hz 网格 / 在环跑完整算法）：
  raw        : 完全不处理（显示 = 原始总量）
  v51_f1/f3/f5 : v5 免责期路线，免责期 1 s / 3 s / 5 s（A 窗 = 免责期/3、ARM = 免责期，同 06 号文档）
  v6         : 形状约束反演（现役默认：ROM_onset + κ 1.30/1.12 + 滑行器 0.8/s）
  v61_rs106  : 形状库上包络重标 1.06（v6.1 的主修复）

产出：results/t3b_arm_metrics.csv（追加）、results/t3b_route_recording.csv（追加）、
      results/t3b_route_roi.csv、results/t3b_route_q6q7.csv、results/t3b_arm_kind.csv、
      results/_t3b_01_routes.log

运行：python scripts/t3b_01_routes.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_arms as A     # noqa: E402
import t3b_tables as T   # noqa: E402
import t3b_settle as ST  # noqa: E402
import t3b_common as C   # noqa: E402

pd.set_option("display.width", 250)


def show(roi, dd, arms):
    print("\n" + "=" * 110)
    print("== Q6/Q7：两条路线在 13 份录制上的 T_stable 分布（口径见列名；n=录制数=13）==")
    print("=" * 110)
    for a in arms:
        for tag, nm in (("tot_ev", "总通道·D1-ev（窗在下一事件处截断）"),
                        ("ch_ev", "主通道·D1-ev"),
                        ("tot_D1", "总通道·D1（完整 30 s 窗、不截断）")):
            r = dd[(dd.arm == a) & (dd.tag == tag)]
            if not len(r):
                continue
            r = r.iloc[0]
            print("  %-12s %-34s n=%2d 中位 %6.2f s  p10~p90 %6.2f~%6.2f s  ≤2 s 的录制比 %.0f%%"
                  % (a, nm, r.n_rec, r.med, r.p10, r.p90, 100 * r.frac_le_2s))
    print("\n-- 与 1~2 s 上限的距离（主口径 = 总通道 D1-ev）--")
    for a in arms:
        r = dd[(dd.arm == a) & (dd.tag == "tot_ev")].iloc[0]
        d2 = r.med - 2.0
        print("  %-12s 中位 %6.2f s  ⇒ 距 2 s %+6.2f s；距 1 s %+6.2f s；配对数（相对 v6）见下"
              % (a, r.med, d2, r.med - 1.0))
    cols = ["arm", "group", "Tstab_tot_ev_n", "Tstab_tot_ev_med", "Tstab_tot_ev_p10",
            "Tstab_tot_ev_p90", "Tstab_ch_ev_med", "Tstab_ch_ev_p90", "Tstab_tot_D1_med",
            "cens_tot_D1_frac", "OS_ch_med", "OS_ch_p90", "OS_tot_med", "err1_ch_med",
            "err1_tot_med", "MD_tot_med", "MD_tot_max", "G_n", "G_med", "G_min",
            "epoch_per100s", "trigger_rate"]
    print("\n-- 代价-收益对表（四联指标：T_stable / OS% / 1 s 误差 / MD）--")
    print(roi[roi.arm.isin(arms)][cols].round(3).to_string(index=False))
    print("\n-- onset / restep 分开（总通道 D1-ev 中位）--")
    print(roi[roi.arm.isin(arms)][["arm", "n_onset", "Tstab_tot_ev_med_onset",
                                   "n_restep", "Tstab_tot_ev_med_restep",
                                   "OS_ch_med_onset", "OS_ch_med_restep",
                                   "err1_ch_med_onset", "err1_ch_med_restep"]]
          .round(3).to_string(index=False))


def main():
    ST.start_log(C.RES, "01_routes")
    t0 = time.time()
    ev, cache = A.load_all()
    print("口径（C-1 处置）：T_stable 一律用 T1-A 冻结口径的代码实现（t3b_settle.py 逐行复制自")
    print("  t1a_common.py）：D1 = 完整 30 s 窗、不因后续事件截断、漂移 ≤5%·|J_ref|；")
    print("  D1-ev = 同 D1 但窗在下一真实事件处截断（可用窗 ≥5 s）⇒ 实录类唯一可测口径。")
    print("  指标序列 = 总量/主通道的 0.5 s 中值（指标字典 Z̄），J_ref 取 [t_on+4, t_on+6] − 前 2 s 中位。")
    print("  本任务不与 T1 做绝对横比：T1-A 的 t1a_settle_metrics.csv 截稿时未产出（见报告 §5）。\n")
    print("事件集：13 份录制、%d 个事件（onset %d / restep %d / decrement %d / partial_unload %d），"
          % (len(ev), (ev.kind == "onset").sum(), (ev.kind == "restep").sum(),
             (ev.kind == "decrement").sum(), (ev.kind == "partial_unload").sum()))
    print("  其中 T4-A clean=True 的加载事件：onset %d / restep %d\n"
          % (((ev.kind == "onset") & ev.clean_t4a).sum(),
             ((ev.kind == "restep") & ev.clean_t4a).sum()))
    A.run_group("routes", ev, cache, force=("--force" in sys.argv))
    roi, dd, g = T.build_all()
    arms = ["raw", "v51_f1", "v51_f3", "v51_f5", "v6", "v61_rs106"]
    arms = [a for a in arms if a in set(roi.arm)]
    show(roi, dd, arms)
    print("\n总耗时 %.1f s" % (time.time() - t0))
    print("-> results/t3b_arm_metrics.csv / t3b_route_recording.csv / t3b_route_roi.csv / t3b_route_q6q7.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
