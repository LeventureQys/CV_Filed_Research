# -*- coding: utf-8 -*-
"""t3b_03_third.py —— T3-Q9：第三条路候选与数值可行性估计。

候选（每条都在 13 份录制上跑完整算法，不是静态估计）：
  C1 只对 onset 做反演、restep 不预判（≈「restep 走免责」）      → 复用 ROI 臂 `onset_only`
  C2 形状反演 + 输出端轻度滤波（τ=0.1 / 0.3 s EMA）              → `rt_filter010` / `rt_filter030`
  C3 形状反演与慢相"解耦"：稳态锚点不取 Â，改取交接时刻实测平台   → `rt_measured`（+`rt_meas_trim`）
  C4 单侧门（宁可欠报不许过报）：rom_scale 1.08 + κ 1.00          → `rt_onesided`
  C5 把滑行器放慢、把过充"摊平"（RATE_MAX 0.5 / 0.3）             → 复用 ROI 臂 `gl050` / `gl030`

产出：results/t3b_third_routes.csv、results/_t3b_03_third.log

运行：python scripts/t3b_03_third.py
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

CAND = [
    dict(cand="C1", name="C1_onset_only", arm="onset_only",
         mechanism="只对 onset 做形状反演；restep/减载类不做形状预判（目标 = 当前实测增量），"
                   "滑行器与慢相模块不变",
         why="T4-A 实测 restep 的等效输入斜坡 T_ramp 中位 0.55 s ⇒ 1 s 目标物理不可达；"
             "T4-B 实测 restep 是过充/下冲最大来源",
         cost="restep 段不再「提前到位」，事件后 1~2 s 只跟随实测电平（台阶捕获比略降）",
         risk="restep 占实录事件约 1/3；若现场以'负载内加重'为主，则等于放弃对它的快速稳定承诺",
         calib="否（不依赖任何新增标定）",
         reuses_arm=True),
    dict(cand="C2", name="C2_inv_plus_filter", arm="rt_filter030",
         mechanism="形状反演照旧，output 端串一级因果 EMA（τ=0.3 s；另有 0.1 s 对照）",
         why="第一轮实测 τ=0.3 s 是「双赢」：过充 9.11→5.14% 且 T_stable 2.72→1.89 s",
         cost="显示引入 0.3 s 时间常数：1 s 时刻的误差被 EMA 改写（本任务实测见 err1 列）",
         risk="滤波对「方向一致的系统性偏差」只在过渡期有效；τ 越大越吃 1~2 s 预算",
         calib="否（τ 是全局常数）",
         reuses_arm=False),
    dict(cand="C3", name="C3_decouple_slow", arm="rt_measured",
         mechanism="交接时 A 取「交接时刻实测平台」而不是钉住 Â（ANCHOR_MODE=measured），"
                   "使慢相模块的稳态不再继承形状库误差；`rt_meas_trim` 另外打开限速慢修正(0.2%/s)",
         why="第一轮 §8.2 定位：v6 的显示稳态 ≡ Â，形状误差 1:1 搬成平台高度（组内 std 4.83 pp）",
         cost="交接瞬间显示必须先滑到实测平台（多一次回落/抬升），慢相段精度退到 v5 量级",
         risk="实测（第一轮 ③④ 负结果）无明显改善；本任务复核见下表",
         calib="否",
         reuses_arm=False),
    dict(cand="C4", name="C4_one_sided", arm="rt_onesided",
         mechanism="单侧门：形状库上包络重标 1.08 + κ 压到 1.00 ⇒ 结构上「宁可欠报不许过报」",
         why="第一轮 §9.4：过充必须靠回落来还，最刺眼；欠报会被随后的慢相自然补上",
         cost="明显欠报（欠报在'负载内加重'上会压低台阶捕获比）",
         risk="欠报在验收口径'1 s 时刻误差'上变负；需要与'台阶捕获比 ≥0.8'的验收线联判",
         calib="否（但 1.08 这个数取自 v6.1 的实测分布，现场换传感器需复核）",
         reuses_arm=False),
    dict(cand="C5", name="C5_slow_glide", arm="gl050",
         mechanism="把滑行器速率上限从 0.8/s 压到 0.5/s（对照 0.3/s），把过充峰值摊成缓升",
         why="过充的'峰值'由滑行器把 Â 推到位的时间决定；放慢可削峰而不改 Â 本身",
         cost="到位时间 ∝ 幅度/RATE_MAX，直接吃 1~2 s 预算",
         risk="本质是「用时间换幅度」，与 C2 的滤波是同一类交换，二者不应叠加",
         calib="否",
         reuses_arm=True),
]


def main():
    ST.start_log(C.RES, "03_third")
    t0 = time.time()
    ev, cache = A.load_all()
    print("Q9：第三条路候选的数值可行性（全部在 13 份录制上跑完整算法）\n")
    A.run_group("third", ev, cache, force=("--force" in sys.argv))
    roi, dd, g = T.build_all()

    base = roi[roi.arm == "v6"].iloc[0]
    rows = []
    for c in CAND:
        r = roi[roi.arm == c["arm"]]
        if not len(r):
            print("!! 缺臂 %s（%s）" % (c["arm"], c["name"]))
            continue
        r = r.iloc[0]
        rows.append(dict(
            cand=c["cand"], route=c["name"], arm=c["arm"],
            mechanism=c["mechanism"], why=c["why"], cost=c["cost"], risk=c["risk"],
            needs_field_calib=c["calib"],
            n_events=int(r.n_events), n_onset=int(r.n_onset), n_restep=int(r.n_restep),
            Tstab_tot_ev_med=r.Tstab_tot_ev_med, Tstab_tot_ev_p10=r.Tstab_tot_ev_p10,
            Tstab_tot_ev_p90=r.Tstab_tot_ev_p90,
            Tstab_tot_ev_med_onset=r.Tstab_tot_ev_med_onset,
            Tstab_tot_ev_med_restep=r.Tstab_tot_ev_med_restep,
            Tstab_ch_ev_med=r.Tstab_ch_ev_med,
            dT_vs_v6=float(r.Tstab_tot_ev_med - base.Tstab_tot_ev_med),
            OS_ch_med=r.OS_ch_med, OS_ch_p90=r.OS_ch_p90,
            dOS_vs_v6=float(r.OS_ch_med - base.OS_ch_med),
            err1_ch_med=r.err1_ch_med, derr1_vs_v6=float(r.err1_ch_med - base.err1_ch_med),
            MD_tot_med=r.MD_tot_med, MD_tot_max=r.MD_tot_max,
            dMD_vs_v6=float(r.MD_tot_med - base.MD_tot_med),
            epoch_per100s=r.epoch_per100s, trigger_rate=r.trigger_rate,
            G_n=r.G_n, G_med=r.G_med,
            rom_scale=r.rom_scale, kappa_onset=r.kappa_onset, glide_rate=r.glide_rate,
            ema_tau=r.ema_tau, anchor_mode=r.anchor_mode, onset_only=r.onset_only))
    t3 = pd.DataFrame(rows)
    p = os.path.join(C.RES, "t3b_third_routes.csv")
    t3.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n-- 第三条路候选：预期指标（T1 口径；n = 事件数）--")
    cols = ["cand", "route", "n_events", "Tstab_tot_ev_med", "Tstab_tot_ev_p10",
            "Tstab_tot_ev_p90", "Tstab_tot_ev_med_onset", "Tstab_tot_ev_med_restep",
            "dT_vs_v6", "OS_ch_med", "dOS_vs_v6", "err1_ch_med", "derr1_vs_v6",
            "MD_tot_med", "dMD_vs_v6", "epoch_per100s", "G_n", "G_med"]
    print(t3[cols].round(3).to_string(index=False))
    print("\n-- 机制 / 代价 / 风险 / 是否需现场标定 --")
    for _, r in t3.iterrows():
        print("\n[%s] %s（臂 %s）" % (r.cand, r.route, r.arm))
        print("   机制：%s" % r.mechanism)
        print("   理由：%s" % r.why)
        print("   代价：%s" % r.cost)
        print("   风险：%s" % r.risk)
        print("   现场标定：%s" % r.needs_field_calib)
    print("\n总耗时 %.1f s" % (time.time() - t0))
    print("-> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
