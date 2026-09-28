# -*- coding: utf-8 -*-
"""t8_matrix.py -- T8-Q1/Q2/Q3/Q6 + 冲突裁决表：把 T1~T7 的已交付数字汇成可决策矩阵。

纪律（T8 需求文档 §4）：
  * 每个数字必须带出处三元组 `source_task / source_file / column`；
  * 不重新定义指标（口径一律引自 00_共享/指标字典与口径.md 或 T1-A 冻结口径）；
  * 缺数据写「缺数据」，不编造；
  * 报数规范：中位 + p10~p90，标 n 与口径。

本脚本**不重跑任何参数扫描**：所有 value 均逐字取自各任务 `results/*.csv` 或
`results/conclusions*.json`（后者在 `source_file` 里注明 `.json:headline[N]`）。

用法：python scripts/t8_matrix.py
产物：results/t8_matrix.csv / t8_optimism_discount.csv / t8_substitutability.csv
      / t8_data_gaps.csv / t8_conflict_rulings.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from t8_common import (  # noqa: E402
    RESULTS, DIM_ORDER, COL_ORDER, MATRIX_FIELDS, cell, write_csv,
)

NA = "缺数据"

# ---------------------------------------------------------------- T8-Q1 评估矩阵
M = []
A = M.append


def D(dim):  # 取维度下标（保持声明顺序稳定）
    return DIM_ORDER.index(dim)


# ---- 稳定时间 -------------------------------------------------------------
A(cell("稳定时间", "v3", NA, "s", "—", "—", "—",
       "T1~T7 无任何席位交付 v3 臂；第一轮 13-v6-assessment 的 settle_arms.csv 只有 raw / v5.1 / v6 三臂",
       "", "★ 本席位未在 progress/01-v3-baseline 与 04-v5 的产物里找到任何 T_stable 口径的 v3 实测值 ⇒ 空缺（诚实登记）"))
A(cell("稳定时间", "v5.1", "2.89", "s", "T1-A", "results/t1a_settle_summary.csv", "med",
       "T_stable_ch5（主通道+5 s 参考，C-1 冻结主口径）", "n=9（恒载 9 组 onset）",
       "p10~p90 0.618~6.992，worst 13.16"))
A(cell("稳定时间", "v5.1", "3.81", "s", "T3-B", "results/t3b_route_q6q7.csv", "med",
       "总通道 D1-ev（窗在下一事件处截断），tag=tot_ev，arm=v51_f3", "n=13 录制",
       "≤2 s 录制比 8%"))
A(cell("稳定时间", "v6", "1.80", "s", "T1-A", "results/t1a_settle_summary.csv", "med",
       "T_stable_ch5（冻结主口径）", "n=9", "p10~p90 0.488~7.960，worst 13.16；达标率 5/9=55.6%"))
A(cell("稳定时间", "v6", "0.54", "s", "T3-B", "results/t3b_route_q6q7.csv", "med",
       "总通道 D1-ev，tag=tot_ev，arm=v6", "n=13 录制", "≤2 s 录制比 77%"))
A(cell("稳定时间", "v6", "5.36", "s", "T5-A", "results/t5a_kappa_loo.csv", "T10_med_fold_max",
       "κ_onset=1.10 留一 13 折中最坏折的 T_stable(10 s 窗) 中位", "n_folds=13",
       "对照 κ=1.15 为 78.50 s ⇒ 改善 14.6 倍；这是「尾部」而非中位口径"))
A(cell("稳定时间", "v6.1", "1.16", "s", "T1-A", "results/t1a_settle_summary.csv", "med",
       "T_stable_ch5（冻结主口径）", "n=9", "p10~p90 0.430~7.008；达标率 5/9=55.6%"))
A(cell("稳定时间", "T3-B第三条路", "0.52", "s", "T3-B", "results/t3b_route_roi.csv",
       "Tstab_tot_ev_med", "总通道 D1-ev；route=rom_scale 1.06（形状库上包络重标）",
       "n=13 录制", "≤2 s 录制比 69%；拐点 T=knee 1.06（t3b_roi_knee.csv）"))
A(cell("稳定时间", "T3-B第三条路", "0.70", "s", "T3-B", "results/t3b_third_routes.csv",
       "Tstab_tot_ev_med", "候选 C4 单侧门（rt_onesided）", "n=13 录制",
       "过充归零但 err1 −10.97%、G 0.946"))

# ---- 稳态时漂 -------------------------------------------------------------
A(cell("稳态时漂", "v3", "1.58", "%", "01-v3-baseline（转引 04-v5 §4.3）",
       "progress/04-v5/docs/05-v5算法说明.md:§4.3 表（v3 行）",
       "时漂残余 全段", "负载段末 10% − 首 10%，占电平 % 的中位；恒载 9 组",
       "n=9",
       "★ 非 T1~T7 产物，引自 v5 文档的 v3 对照表（列名与时漂残余定义见该表）；"
       "同表：慢相段 1.49%、受载中位 0.59%、噪声比 0.71、平坦度 1.88%、零漂残余 1.50%、阶跃保真 1.00"))
A(cell("稳态时漂", "v5.1", "0.0000", "%·|J|", "T7-A", "results/t7_frozen_summary.csv",
       "resid_late_pct", "卸载后稳态残余（相对加载前空载电平）", "n=14 全卸载",
       "v5.1/v6 在 14/14 事件上 ded_late_pct = 0.0000（加载期累积的慢相补偿被彻底撤销）"))
A(cell("稳态时漂", "v6", "0.0000", "%·|J|", "T7-A", "results/t7_frozen_summary.csv",
       "resid_late_pct", "同上（v6 臂，卸载后稳态残余）", "n=14", "与 v5.1 相同"))
A(cell("稳态时漂", "v6", "34.14", "%", "T5-A", "results/t5a_nonfilter_options.csv",
       "err1_abs_med（arm=B0_baseline = v6 默认参数）", "1 s 时刻显示绝对误差中位", "n=41 加载事件",
       "κ 扫描的配对值见 t5a_kappa_verdict.csv:err1_abs_med_all（κ=1.30 为 30.06%、κ=1.10 为 33.97%）；"
       "该列对 t_on 的 ±1 包极敏感（−3.6%→−69.4%）⇒ 只在同口径配对差里使用"))
A(cell("稳态时漂", "v6", "4.08", "%·J", "T5-A", "results/t5a_overcharge_summary.csv",
       "OS5_med（kind=onset）", "onset 5 s 窗瞬态过充（慢相走廊失配的另一侧面）", "n=22",
       "30 s 窗口径下同批要 +12.5 pt（t5a G10），引用必须带窗长"))
A(cell("稳态时漂", "v6.1", NA, "%", "—", "—", "—", "无 v6.1 的稳态时漂/1 s 误差独立产物",
       "", "T5-A 只给了 v6/v6.1 的过充与 G；T7 未跑 v6.1 臂"))
A(cell("稳态时漂", "T3-B第三条路", "9.15", "%", "T3-B", "results/t3b_route_roi.csv",
       "err1_ch_med（rom_scale=1.06）", "1 s 时刻误差中位（主通道）", "n=13 录制",
       "v6 基线为 −3.70% ⇒ 上包络重标把误差推向「欠报」侧"))

# ---- 台阶保真 -------------------------------------------------------------
A(cell("台阶保真", "v3", "0.89", "—", "01-v3-baseline（转引 01-v3-baseline/MANIFEST.md）",
       "progress/01-v3-baseline/MANIFEST.md:基线数字", "变载台阶捕获比",
       "变载实录台阶捕获比（中位）", "",
       "同处另给 04-v5 §10.3B：v3 0.91/0.85、0.97/0.92、0.93/0.66（三份实录的中位/最小）"))
A(cell("台阶保真", "v3", "1.00", "—", "01-v3-baseline（转引 04-v5 §4.3）",
       "progress/04-v5/docs/05-v5算法说明.md:§4.3 表（v3 行）", "阶跃保真",
       "onset+0.5~2.5 s 幅度 ÷ 原始同窗", "n=9", "事件跳变超额 0.03"))
A(cell("台阶保真", "v5.1", "1.000", "—", "T7-A", "results/t7_frozen_summary.csv",
       "recovery_pct/type 的组合说明列", "台阶捕获比 G（事件后 20 s 显示增量/原始增量）",
       "n=14 卸载", "T7-A 未跑 v5.1 的 G 列；此处给的是「卸载沿回收率」的等价事实，见 note"))
A(cell("台阶保真", "v5.1", "0.98", "—", "07-v6", "07-v6/MANIFEST.md（转引 01 号文档 §3 地基表）",
       "台阶捕获比 G", "历史口径", "—", "转引；本次调研未独立复算 v5.1 的 G"))
A(cell("台阶保真", "v6", "1.033", "—", "T3-B", "results/t3b_route_roi.csv", "G_med（arm=v6）",
       "|Δ原始|≥2000 ADC 的 ADC 域大台阶，20 s 窗", "n=7", "G_min −0.141"))
A(cell("台阶保真", "v6.1", "0.920", "—", "T5-A", "results/t5a_v6_vs_v61_summary.csv",
       "G_med（行 v61_s106）", "同批 41 加载事件", "n=41", "v6 同批为 0.942 ⇒ ROM_SCALE 使台阶略钝"))
A(cell("台阶保真", "T3-B第三条路", "0.982", "—", "T3-B", "results/t3b_route_roi.csv",
       "G_med（rom_scale=1.06）", "同上", "n=7", "v6 1.033 → 0.982（−5 pt）"))

# ---- 过充 ---------------------------------------------------------------
A(cell("过充", "v3", "5.9", "%", "01-v3-baseline（转引 04-v5 §3.6）",
       "progress/04-v5/docs/05-v5算法说明.md:§3.6（台阶 100% 场景）", "实测过冲",
       "合成台阶 100% 场景的稳定值相对理想值的超出（理想 1.50×10N，实测 1.62×10N 为 +7.9%；100% 场景 +5.9%）",
       "合成砝码场景（非实录）",
       "★ 该量是「负载比例蠕变场的稳态偏差」，不是指标字典 §3 的 OS%（5 s 窗瞬态过充）"
       "⇒ **与 v6 的 OS5 不同源、不可横比**；v6 的 OS5 来自快相反演高估，v3 没有形状反演"))
A(cell("过充", "v5.1", "1.05", "%", "第一轮", "progress/13-v6-assessment/docs/v6评估与需求答复.md §5",
       "over_pct（v5.1 行）", "历史口径", "—",
       "本批未独立复算 v5.1 的主通道过充；T5-A 只给了 v5.1 的 OS5 max（见下）"))
A(cell("过充", "v5.1", "54.39", "%", "T5-A", "results/t5a_v6_vs_v61_summary.csv",
       "OS5_max（行 v5.1）", "41 加载事件的 5 s 窗过充最大值", "n=41",
       "★ 这是 T5-A 的可证伪条件：v5.1 的过充上限远低于 v6 的 120.81%"))
A(cell("过充", "v6", "9.64", "%", "T5-A", "results/t5a_kappa_verdict.csv",
       "OS5_med_onset（arm=unif_k1.30）", "onset 22 事件、主通道、5 s 窗", "n=22",
       "与 T3-B 主通道 9.00% 同口径同量级；总通道口径下为 +4.08%（通道平均稀释）"))
A(cell("过充", "v6", "120.81", "%", "T5-A", "results/t5a_kappa_verdict.csv",
       "OS5_max（arm=unif_k1.30）", "41 加载事件", "n=39~41", "12 个事件 >5%"))
A(cell("过充", "v6.1", "-2.51", "%", "T5-A", "results/t5a_v6_vs_v61_summary.csv",
       "OS5_med（行 v61_s106）", "同批 41 加载事件", "n=41",
       "max +74.83%（−38%）、OS>5% 事件 12→7 ⇒ 压掉约一半，未清零"))
A(cell("过充", "T3-B第三条路", "3.02", "%", "T3-B", "results/t3b_route_roi.csv",
       "OS_ch_med（rom_scale=1.06）", "总通道+主通道口径（route_roi 的 OS_ch_med）", "n=13 录制",
       "v6 同列 8.999%；拐点 OS=knee 1.06"))

# ---- 下冲（反向过冲） ------------------------------------------------------
A(cell("下冲", "v3", NA, "%", "—", "—", "—",
       "v3 无形状反演，其「向下」表现是卸载门控自动归零的伪零点（见卸载行为行），不是 load 沿下冲",
       "", "★ 05-v5算法说明 §10.3A 给出 v3 的零点段显示均值 −1051~+210 ADC、卸载沿后 0.5 s 显示总量 −6362~+1289"
       "⇒ 属「零点被强制拉回 0」的另一类缺陷，登记在卸载行为维度"))
A(cell("下冲", "v5.1", NA, "%", "—", "—", "—", "T5-A 的 US5 列只在 v6/v6.1 臂上产出",
       "", "T7 报的是卸载沿「额外下冲」，不是加载沿下冲，不可横比"))
A(cell("下冲", "v6", "18.59", "%·J", "T5-A", "results/t5a_overcharge_summary.csv",
       "OS5_med（kind=restep，负号即下冲）", "restep 11 事件、5 s 窗", "n=11",
       "9/11 向下 >2%，中位下冲 2865.2 ADC；根因 = 复合事件继承的 c0 被 pin 钉住（G=0.473）"))
A(cell("下冲", "v6", "1230.09", "%", "T5-A", "results/t5a_kappa_verdict.csv", "US5_max",
       "41 加载事件的最大欠冲", "n=39~41", "★ ROM_SCALE 1.00→1.06 对该列几乎无效（两臂同值）"))
A(cell("下冲", "v6.1", NA, "%", "—", "—", "—", "缺数据（US5 在 v6.1 汇总表未给分列）",
       "", "从 US5_max 两臂同值可推知 ROM_SCALE 治不了下冲"))
A(cell("下冲", "T3-B第三条路", NA, "%", "—", "—", "—", "缺数据",
       "", "t3b_third_routes.csv 未给 US 列；C4 单侧门只截上限、对欠冲无效（T5-A Q5）"))

# ---- 重复性 -------------------------------------------------------------
A(cell("重复性", "v3", NA, "pp", "—", "—", "—",
       "无 v3 臂的重复性产物（第一轮 b_repeat_disp.csv 与 T6 都只跑 raw/v5.1/v6/v6.1）", "",
       "★ 与 T_stable 同为矩阵中 v3 的空缺项；若要补齐需按 T1-A 冻结口径补跑 v3 臂"))
A(cell("重复性", "v5.1", "3.39", "pp", "T6", "results/t6_repeat_dispersion.csv", "value",
       "std(平台误差) 相对机械台阶电平 Rstep；impl=v5.1", "n=9（3 组×3 次）",
       "相对 5 s 实测电平 R5 口径下为 3.07 pp"))
A(cell("重复性", "v5.1", "0.26", "%·电平", "T6", "results/t6_jitter_summary_a.csv",
       "rms_pct_lvl_med（cond=timing_J100P）", "±1 包(±40.0 ms)时序抖动下输出轨线 RMS 差",
       "n=30 种子", "序列相同率 1.00（路径完全稳定）"))
A(cell("重复性", "v6", "3.52", "pp", "T6", "results/t6_repeat_dispersion.csv", "value",
       "std(平台误差) Rstep；impl=v6", "n=9", "相对 5 s 口径 R5 下为 5.09 pp（比 v5.1 差 66%）"))
A(cell("重复性", "v6", "7.90", "%·电平", "T6", "results/t6_jitter_summary_a.csv",
       "rms_pct_lvl_med（cond=timing_J100P）", "±1 包时序抖动下输出轨线 RMS 差", "n=30 种子",
       "★ 30.6× v5.1；epoch 变 50.0%、revoke 变 73.3%、handoff 变 70.0% ⇒ 主缺口在 L3 路径层"))
A(cell("重复性", "v6", "1.71", "次/100 s", "T3-B", "results/t3b_route_roi.csv", "epoch_per100s",
       "补偿器内部建事件率", "n=13 录制", "v5.1 为 1.05 ⇒ 1.63 倍；与第一轮 1.6~1.7 倍一致"))
A(cell("重复性", "v6.1", "3.37", "pp", "T6", "results/t6_improve_ab.csv", "std_R5_pp",
       "arm=v6.1_asIS，相对 5 s 实测电平 R5", "n=9",
       "−33.8% vs v6，但 ±1 包抖动下 L3 毫无帮助（7.744% vs 7.714%）；bias 由 +2.22 变 −3.83 pp"))
A(cell("重复性", "T3-B第三条路", "2.57", "pp", "T6", "results/t6_improve_ab.csv", "std_R5_pp",
       "arm=I4_ho10s（τ_ho 5→10 s），T1 口径 std(R5)", "n=9",
       "−49.6%；组合臂 I7_h3_I4ho10 在 L3 上再取 −84.7%"))

# ---- 强扰动鲁棒性 --------------------------------------------------------
A(cell("强扰动鲁棒性", "v3", NA, "%", "—", "—", "—",
       "v3 无快相状态机（无 epoch/revoke/handoff），T5-B 的 M1~M4 失败定义对它不成立；第一轮 a_ 分支只跑 v6",
       "", "★ 结构上不可比：v3 的失效模式是「把真实加载当蠕变扣掉」（捕获比 0.13~0.61，见 04-v5 §4.4 注）"))
A(cell("强扰动鲁棒性", "v5.1", NA, "%", "—", "—", "—", "缺数据（T5-B 只扫 v6 臂）",
       "", "T1-B 的 A=2000/4000 档也只做 v6/v6.1 相对档，未给 v5.1 的绝对失败率"))
A(cell("强扰动鲁棒性", "v6", "100", "%", "T5-B", "results/t5b_q1_headline.csv", "white,1000.0,W1_1w",
       "M1~M4 任一成立的 trial 失败率；12048 ADC 平台 + ±1000 ADC RMS 白噪", "n=10",
       "Wilson 95% CI [0.72,1.00]；算法可归因显示超额中位 29.5% 电平（≈3570 ADC）"))
A(cell("强扰动鲁棒性", "v6", "75.7", "ADC", "T5-B", "results/t5b_boundary_measured.csv", "A50",
       "复合（任一 M）50% 失效幅度，@1.2w 平台，白噪", "n=32",
       "95% CI [50.8,100.2] = 0.63% 电平；共模 89.0 ADC；低频带限（f_hi=1 Hz）更狠 3~4 倍"))
A(cell("强扰动鲁棒性", "v6", "0.402", "AUC", "T5-B", "results/t5b_criteria_summary.csv", "auc",
       "现行唯一旋钮（抬 ratio 门限）作为拍击/真实变载判据", "n=1062 trials",
       "比随机差；「保持维」因果版 AUC 0.841、非因果版 0.950 ⇒ 缺的是平台/保持这一维"))
A(cell("强扰动鲁棒性", "v6", "6.96", "个/次", "T1-B", "results/t1b_summary_noise_adc.csv",
       "n_miss_mean", "A=4000 ADC 白噪下真实变载漏检数", "n=11~16 实录事件",
       "A=1000 → 0.31（基线 0.29）、A=2000 → 1.44；带限 0.3~40 Hz 在 2000 ADC 时为 1.96"))
A(cell("强扰动鲁棒性", "v6", "16.18", "s", "T1-B", "results/t1b_summary_noise_hold.csv", "t_stable_med",
       "显示域白噪 r=10% 的 T_stable 中位（总通道）", "n=31",
       "r=2% 时反而 0.39 s ⇒ 「更稳」是假象（同批显示偏置中位 2.26 显示单位 = 12.8% 电平）"))
A(cell("强扰动鲁棒性", "v6.1", NA, "%", "—", "—", "—", "缺数据",
       "", "T5-B 未扫 v6.1；T1-B 只做「3 个相对档 × 5 种子」的四臂对照，未给绝对失败率"))
A(cell("强扰动鲁棒性", "T3-B第三条路",
       "116", "ADC", "T5-B", "results/t5b_fallback_policies.csv", "tap_dA_end_med",
       "低置信期冻结锚定（hold, lowconf_thr=0.50）后的拍击永久锚点损伤", "n=23",
       "基线（policy=none）为 2713.3 ADC；干净数据副作用 clean_sideeffect_rel = 0.0"))

# ---- 卸载行为 -----------------------------------------------------------
A(cell("卸载行为", "v3", "1023", "ADC", "01-v3-baseline（转引 04-v5 §10.3A）",
       "progress/04-v5/docs/05-v5算法说明.md:§10.3A", "零点段显示均值（−1051~+210）",
       "含空载段的 3 份实录、6 个零负载段：零点段显示均值的极差（原始读数 40~76 ADC）", "n=6 个零负载段",
       "★ 这是 v3 的结构性缺陷：卸载门控自动归零（空载期以 τ=2 s 跟踪基线 b_ 并减掉）把零点强制拉回 0；"
       "同表：卸载沿后 0.5 s 显示总量 −6362~+1289、零负载段被钳成 0 的时长 0~6.05 s"))
A(cell("卸载行为", "v3", "0.89", "—", "01-v3-baseline（转引 04-v5 §10.3B）",
       "progress/04-v5/docs/05-v5算法说明.md:§10.3B", "台阶捕获比 中位/最小（v3 列）",
       "4 份变化负载实录", "n=3 份实录 × 2 事件",
       "v3 0.91/0.85、0.97/0.92、0.93/0.66；变载窗最大偏差中位 850/2190/2129 ADC（v5.1 为 247/1082/767）"))
A(cell("卸载行为", "v5.1", "0.8889", "比例", "T1-A", "results/t1a_target_verdict.csv", "rate_2s",
       "unload 工况「≤2 s」达标率（冻结口径 ch5）", "n=9 可测",
       "v6 为 1.0000（9/9）；v5.1 的卸载沿本身也很快（t90 中位 0.03 s），差距在事件级稳定判据上"))
A(cell("卸载行为", "v5.1", "0.745", "s", "T7-A", "results/t7_frozen_offset.csv", "t_ded_half（v5 臂）",
       "v5（含空载自动归零）的撤销半时", "n=14",
       "v5 额外贴零 +5.58 s、额外下冲最大 25.9%、零点被改到 −552~+552 ADC ⇒ 这是被淘汰的实现"))
A(cell("卸载行为", "v6", "0.040", "s", "T7-A", "results/t7_unload_events.csv", "t90",
       "全卸载沿 t90 中位（真实卸载沿 t0）", "n=14",
       "t50 0.02 s、0.2 s 完成度 1.0022；T7-B 独立复算 n=16 得 0.015 s"))
A(cell("卸载行为", "v6", "1.0000", "比例", "T1-A", "results/t1a_target_verdict.csv", "rate_2s",
       "unload 工况「≤2 s」达标率（冻结口径 ch5）", "n=9 可测",
       "worst 1.15 s；是全项目唯一「达标」的工况（rate_2s_lo 0.4737）"))
A(cell("卸载行为", "v6", "2.81", "%·|J|", "T7-A", "results/t7_frozen_summary.csv", "resid4_pct",
       "卸载后 4~6 s 残余（相对加载前空载电平）；列单位为 %·|J|", "n=14",
       "力域（恒载 9）中位 +0.19%、p10~p90 0.00~2.50%、max +2.81%；ADC 域（实录 5）全部 ≤0.08%；"
       "11/14 单调回零、2/14 过冲、1/14 留 2.12% 稳态残余（54.7 s 不衰减）"))
A(cell("卸载行为", "v6", "0.000", "ADC", "T7-B", "results/t7b_partial_amp_response.csv",
       "kind_eff + drop_frac_eff", "反向事件误判率与全卸载稳态偏差", "n=19 减重事件",
       "反向误判 0/19；全卸载稳态偏差 0.000%（±1 s 与 ±2 s 窗均 0%）"))
A(cell("卸载行为", "v6.1", NA, "—", "—", "—", "—", "缺数据", "", "T7 未跑 v6.1 臂"))
A(cell("卸载行为", "T3-B第三条路", "3.4", "%·|J|", "T7-A", "results/t7_inject_amp.csv",
       "e3s_dip_extra_pct（α=1.0 行的中位；该列以分数记，0.034 = 3.4%）",
       "卸载判据前移后的额外下冲（现役 e3s 臂，注入对照）", "n=18（3 录制 × 3 α × 2 沿时长）",
       "α=0.6 时为 42~44%、α=0.3 时为 22~87%；v6 同列略差（α=1.0 为 3.4%）"))

# ---- 计算量 -------------------------------------------------------------
A(cell("计算量", "v3", NA, "次/100 s", "—", "—", "—",
       "v3 无 epoch 状态机；第一轮与 T3-B 都未报 v3 的 epoch_per100s", "",
       "★ 定性可判：v3 只有「三级 EMA + 双滞后电平判据 + 蠕变积分」，无形状反演/滑行器/停滞检测 ⇒ "
       "计算量明显低于 v6；但无量化产物 ⇒ 按纪律写「缺数据」"))
A(cell("计算量", "v5.1", "1.05", "次/100 s", "T3-B", "results/t3b_route_roi.csv", "epoch_per100s",
       "补偿器内部建事件率", "n=13 录制", "v5.1 在 ±1 包抖动下事件序列 100% 不变（T6）"))
A(cell("计算量", "v5.1", "0.98", "s", "T7-A", "results/t7_arm_unload.csv", "t_ded_half",
       "加载期累积慢相的撤销延迟（近似实现成本代理）", "n=14", "T7-A 报中位 0.025 s（非 v5 的 0.745 s）"))
A(cell("计算量", "v6", "1.71", "次/100 s", "T3-B", "results/t3b_route_roi.csv", "epoch_per100s",
       "补偿器内部建事件率", "n=13 录制", "= v5.1 的 1.63 倍；inv_calls 与 trigger_rate 见同表"))
A(cell("计算量", "v6", "0.65", "s", "T1-A", "results/t1a_lowerbound.csv", "T_internal",
       "v6 内部判据/估计延迟下界（TAU_REF 0.20 + STALL_HOLD_S 0.45）", "—",
       "纯跟随器下界 T_in5 中位 4.70 s（n=8）；v6 在 8/8 事件上快于该下界（靠预测+冻结）"))
A(cell("计算量", "v6", "8.6", "×", "T5-A", "results/t5a_kappa_verdict.csv",
       "trigger_rate（arm=onset_k1.15 vs onset_k1.10）", "κ 上限被截断的事件占比（计算路径变化量）",
       "n=39", "κ=1.30 为 4.1%、1.15 为 41%、1.00 为 75.8%"))
A(cell("计算量", "v6.1", "1.71", "次/100 s", "T3-B", "results/t3b_route_roi.csv", "epoch_per100s",
       "与 v6 相同（ROM_SCALE 只改形状库数值，不改状态机）", "n=13 录制",
       "T5-A 亦确认 λ 扫描不改事件结构；故与 v6 同值"))
A(cell("计算量", "T3-B第三条路", "1.71", "次/100 s", "T6", "results/t6_improve_ab.csv", "mean_epoch",
       "I7/I4 改进臂的 epoch 数（未引入新状态机）", "n=9",
       "改动 = 1 个计数器 + 1 个常量；工作量落 src/domain/drift_v6/ 补偿器状态机"))

# ---- 可标定性 -----------------------------------------------------------
A(cell("可标定性", "v3", "不需标定", "—", "01-v3-baseline（转引 04-v5 §3）",
       "progress/04-v5/docs/05-v5算法说明.md:§3（v3 对照列）",
       "常量表（22 个常量逐项保持）", "无形状库、无现场标定项；但有 22 个面板常量", "",
       "★ v3 的代价不是标定，而是「把真实加载当蠕变吃掉」（04-v5 §4.4 注：13ffca 的 53~69 s 窗 v3 记 4.67% 看似很好，"
       "实际把一次 +13.7% 的真实加载当蠕变吃掉了，该事件捕获比 0.13~0.61）"))
A(cell("可标定性", "v5.1", "不需标定", "—", "T6", "results/t6_jitter_summary_a.csv",
       "seq_same_rate", "1 s 均值窗（[2,3] s）与跳变沿无关 ⇒ 无标定依赖", "n=30 种子",
       "±1 包下事件序列相同率 1.00、输出 RMS 差 0.26% 电平"))
A(cell("可标定性", "v5.1", "12.794", "倍", "T7-B", "results/t7b_asymmetry.csv",
       "ratio_load_over_unload（contrast=A5_conservative）", "对对称性假设最有利边界下的加载/卸载 t90 比",
       "n=40/16", ">1 即不对称 ⇒ 线性粘弹性镜像假设不成立（C-5 关闭）"))
A(cell("可标定性", "v6", "5.52", "%", "T3-A", "results/t3a_generalization.csv", "med_abs_pct",
       "onset 自身标定(own) → 同族留一事件(loo_event)@tau_d=0.2 s", "n=18",
       "→ 留一录制 4.76% → 跨族 5.33% → **跨形态 56.39%**；restep 侧 29.88%→52.07%"))
A(cell("可标定性", "v6", "1.10", "—", "T5-A", "results/t5a_kappa_verdict.csv", "kappa_onset",
       "C-4 裁决值（κ_restep 保持 1.12），一行常量", "n=39 事件 / 13 折留一",
       "饱和区 κ≥1.15 六个工作点逐事件完全相同（39/39）⇒ 真正的工作点只有两个"))
A(cell("可标定性", "v6.1", "1.06", "—", "T3-B", "results/t3b_roi_knee.csv", "knee_OS",
       "形状库上包络重标倍率拐点", "n=13 录制",
       "甜点区间 1.04~1.06；1.12 时 G 0.950、err1 −10.97%；与 κ 同类 ⇒ 二选一"))
A(cell("可标定性", "T3-B第三条路", "不需新增标定", "—", "T3-B", "results/t3b_third_routes.csv",
       "needs_field_calib", "5 个候选路线全部 needs_field_calib=否", "n=13 录制",
       "C1/C2/C3/C4 均不依赖新增现场标定；C4（单侧门）收益最大但付 11% 欠报"))

write_csv(os.path.join(RESULTS, "t8_matrix.csv"), M, MATRIX_FIELDS)
print("matrix cells:", len(M), "| 维度数:", len(DIM_ORDER), "| 路线数:", len(COL_ORDER))

# ------------------------------------------------------- T8-Q2 乐观性折扣表
OPT_FIELDS = ["rid", "metric", "condition", "optimistic", "opt_source", "pessimistic",
              "pess_source", "ratio_or_gap", "decide_with", "reason"]
O = []
O.append({
    "rid": "O1", "metric": "形状反演稳定性 → 「≤2 s 达标率」",
    "condition": "onset（恒载 9 组，冻结口径 T_stable_ch5）",
    "optimistic": "55.6%（5/9），中位 1.80 s，worst 13.16 s",
    "opt_source": "T1-A results/t1a_target_verdict.csv:rate_2s / t1a_settle_summary.csv:med（caliber=ch5,group=onset·恒载9组）",
    "pessimistic": "仅 3/9（33.3%）在 1 s 内达标（rate_1s），且 worst 13.16 s",
    "pess_source": "T1-A results/t1a_target_verdict.csv:rate_1s（同组）",
    "ratio_or_gap": "1 s→2 s 口径差 22.3 pt；worst/med = 7.3×",
    "decide_with": "悲观（用 rate_1s 与 worst 承压）",
    "reason": "用户约束是「不能超上限」= 最差情况判据（T8 需求 §6-3）。中位达标不构成承诺依据。"})
O.append({
    "rid": "O2", "metric": "形状库泛化（单点反演 @tau_d=0.2 s 中位|误差|）",
    "condition": "onset usable n=18（显示域+ADC 域合并）",
    "optimistic": "3.10%（形状库含本事件自己 = 既有 ROM 自标定等价物）",
    "opt_source": "T3-A results/t3a_generalization.csv:med_abs_pct（sample=usable,kind=onset,scenario=own,tau_d=0.2）",
    "pessimistic": "56.39%（跨形态：拿 onset 库反演 restep）",
    "pess_source": "T3-A results/t3a_generalization.csv:med_abs_pct（scenario=cross_kind,tau_d=0.2）",
    "ratio_or_gap": "18.2×",
    "decide_with": "悲观（跨形态 56.39%）；同族留一 5.52% 作为「只在 onset 上用」的乐观",
    "reason": "★ 决策核心：单形状库必然一侧过充一侧欠充。跨形态误差是本项目最大误差源，必须按它做单侧门设计。"})
O.append({
    "rid": "O3", "metric": "形状库泛化（同族内，留一事件）",
    "condition": "onset usable n=18",
    "optimistic": "3.10%（own）",
    "opt_source": "同上",
    "pessimistic": "5.52%（loo_event）；p90 15.47%、最大 51.78%",
    "pess_source": "T3-A results/t3a_generalization.csv（scenario=loo_event,tau_d=0.2）；conclusions_T3A.json:corrections[3]",
    "ratio_or_gap": "1.78×（中位）；尾部差一个量级",
    "decide_with": "悲观（5.52%，且须并报 p90/max）",
    "reason": "留一录制 4.76% 与留一事件 5.52% 几乎相同 ⇒ 瓶颈是「哪一次加载」而不是「哪一个传感器」；尾部的 51.78% 来自慢压 onset（T_ramp=0.60 s）。"})
O.append({
    "rid": "O4", "metric": "κ_onset 改善尾部（最坏折 T_stable(10 s) 中位）",
    "condition": "13 折留一",
    "optimistic": "5.36 s（κ_onset=1.10）",
    "opt_source": "T5-A results/t5a_kappa_loo.csv:T10_med_fold_max（arm=onset_k1.10）",
    "pessimistic": "78.50 s（κ_onset=1.15 及以上的饱和工作点）",
    "pess_source": "T5-A results/t5a_kappa_loo.csv:T10_med_fold_max（arm=onset_k1.15/1.30）",
    "ratio_or_gap": "14.6×（κ 自身效应仅 14.7 pt，说明收益全在尾部）",
    "decide_with": "乐观（采用 κ=1.10 后的 5.36 s）——但须注明它是「最坏折」而非保证",
    "reason": "κ 是唯一「不动形状库、不加滤波、不占时间预算」的杠杆，上线成本一行常量；代价（onset 由微正转微负）落在安全侧。"})
O.append({
    "rid": "O5", "metric": "强扰动鲁棒性（失效边界 A50）",
    "condition": "1.2w ADC 平台，白噪/共模复合失效",
    "optimistic": "75.7 ADC（0.63% 电平）",
    "opt_source": "T5-B results/t5b_boundary_measured.csv:A50（W1_1w 复合）",
    "pessimistic": "用户所报量级 ±1000 ADC 下 100% 失败",
    "pess_source": "T5-B results/t5b_q1_headline.csv:white,1000.0,W1_1w",
    "ratio_or_gap": "13.2×（1000/75.7）",
    "decide_with": "悲观（100% 失败）",
    "reason": "两者并不冲突：A50 是 50% 概率点，用户量级远在其上。做承诺必须按用户量级 ⇒ 悲观。"})
O.append({
    "rid": "O6", "metric": "可重复性（平台误差组内 std）",
    "condition": "恒载 9 组，3 传感器 × 3 次",
    "optimistic": "3.52 pp（相对机械台阶电平 Rstep，v6 3.52 vs v5.1 3.39，仅差 4%）",
    "opt_source": "T6 results/t6_repeat_dispersion.csv:value（metric=plat_err_std_Rstep_pp）",
    "pessimistic": "5.09 pp（相对 5 s 实测电平 R5，v6 5.09 vs v5.1 3.07，差 66%）",
    "pess_source": "T6 results/t6_repeat_dispersion.csv:value（metric=plat_err_std_R5_pp）；t6_improve_ab.csv:std_R5_pp",
    "ratio_or_gap": "44.6%（且排名在 Rstep 口径下反转）",
    "decide_with": "悲观（5.09 pp）",
    "reason": "R5 是与用户读数一致的口径（显示值 vs 实测电平）；Rstep 需要机械台阶真值，现场拿不到。另：0.3% 网格差即可翻转排名（dt=0.00995 时 v6 2.69 优于 v5.1 3.38）。"})
O.append({
    "rid": "O7", "metric": "路径可重复性（±1 包时序抖动下输出 RMS 差）",
    "condition": "同一实录输入，只改整包到达时刻",
    "optimistic": "0.26%·电平（v5.1）",
    "opt_source": "T6 results/t6_jitter_summary_a.csv:rms_pct_lvl_med（impl=v5.1,cond=timing_J100P）",
    "pessimistic": "7.90%·电平（v6）",
    "pess_source": "T6 results/t6_jitter_summary_a.csv:rms_pct_lvl_med（impl=v6,cond=timing_J100P）",
    "ratio_or_gap": "30.6×",
    "decide_with": "悲观（7.90%），但采用 T6 的零代价修复臂：I7_revoke_hyst3 把 L3 从 7.71% 降到 0.72%（−90.7%）",
    "reason": "这是 v6「可重复性下降」的主因（L3 路径层），且是本次调研唯一有零代价修复的一层；残余 0.4~0.7% 仍是 v5.1 的 1.6~2.7 倍。"})
O.append({
    "rid": "O8", "metric": "卸载后残余偏移",
    "condition": "全卸载 n=14（严格冻结集）",
    "optimistic": "0.19%·|J|（4~6 s 窗中位）",
    "opt_source": "T7-A results/t7_frozen_summary.csv:resid4_pct",
    "pessimistic": "2.81%·|J|（同列最大，右拇指/数据3 稳态残余 2.12% 且 54.7 s 不衰减）",
    "pess_source": "T7-A results/t7_frozen_summary.csv:resid4_pct / conclusions_T7A.json:headline[1]",
    "ratio_or_gap": "14.8×（max/med）",
    "decide_with": "悲观（按 max 2.81% 与「1/14 不回来」做验收）",
    "reason": "以 2%·|J| 为界仅 1/14 超限 —— 比例小但后果是永久残余，必须按 worst 承诺。"})
O.append({
    "rid": "O9", "metric": "卸载沿速度快慢（加载/卸载对称性）",
    "condition": "同录制同 |J|(≤15%) 配对",
    "optimistic": "24.0×（t90 比值中位，A3_matched）",
    "opt_source": "T7-B results/t7b_asymmetry.csv:ratio_load_over_unload（contrast=A3_matched,metric=t90_ratio）",
    "pessimistic": "13.5×（对对称性最有利的 ±1 包保守边界，仍 >1）",
    "pess_source": "T7-B results/t7b_asymmetry.csv（contrast=A8_tight_conservative）",
    "ratio_or_gap": "两种口径都表明明显不对称（最严子样本 27.4×）",
    "decide_with": "悲观（13.5× 的保守方向仍成立 ⇒ 对称假设被否决）",
    "reason": "C-5 关闭；T4-A 的 E2 证据须从「独立机制证据」下调为「与机制一致的方向性事实」（T4-A 结论方向不变，E1/E3/E4 不依赖对称性）。"})
O.append({
    "rid": "O10", "metric": "偏转的归因与相位（τ_d 敏感度）",
    "condition": "onset usable n=18，tau_d ∈ {0.2, 1.0} s",
    "optimistic": "2.13%（own @tau_d=1.0 s）",
    "opt_source": "T3-A results/t3a_generalization.csv（scenario=own,kind=onset,tau_d=1.0）",
    "pessimistic": "5.14%（cross_kind @tau_d=1.0 s）",
    "pess_source": "T3-A results/t3a_generalization.csv（scenario=cross_kind,kind=onset,tau_d=1.0）",
    "ratio_or_gap": "2.4×；@0.2 s 时同两口径为 3.10% → 56.39%",
    "decide_with": "两者并报（@0.2 s 做判据设计、@1.0 s 做稳态验收）",
    "reason": "误差随时间衰减：0.2 s 处跨形态灾难（56.39%）、1.0 s 处降到 10.78%（onset 反演 restep 的更细分级见 t3a_generalization.csv）。⇒ 快相内靠形状库有风险，1 s 后靠实测更稳。"})

write_csv(os.path.join(RESULTS, "t8_optimism_discount.csv"), O, OPT_FIELDS)

# --------------------------------------------------- T8-Q3 可替代性判定
SUB_FIELDS = ["condition", "verdict", "criterion", "evidence_v6", "evidence_ref",
              "risk", "precondition", "confidence"]
S = []
S.append({
    "condition": "① 单次加载长保压（onset，恒载族）",
    "verdict": "v6 可替代（有条件）",
    "criterion": "冻结口径 T_stable_ch5 中位 ≤2 s **且** ≤2 s 达标率显著高于 v5.1 **且** 过充/下冲不超出验收带 **且** 无永久基线损伤",
    "evidence_v6": "中位 1.80 s（v5.1 2.89 s）、达标率 5/9=55.6%（v5.1 4/9=44.4%）、worst 13.16 s（与 v5.1 相同的最差事件）",
    "evidence_ref": "T1-A results/t1a_settle_summary.csv:med + t1a_target_verdict.csv:rate_2s/rate_2s_lo（caliber=ch5,group=onset·恒载9组）",
    "risk": "过充中位 +9.64%（主通道）/**max +120.81%**（T5-A t5a_kappa_verdict.csv，arm=unif_k1.30）；达标率只有 55.6% ⇒ **不能承诺上限**",
    "precondition": "必须同时上 κ_onset=1.10（压尾部：最坏折 T_stable(10 s) 78.50→5.36 s）+ 过充硬限幅 clip 0.02（MD 中位 2396.7→555.4 ADC）；并接受 1 s 时刻欠报 −3.7%→−9.15%",
    "confidence": "中（n=9 恒载事件，且 R5 口径与网格敏感）"})
S.append({
    "condition": "② 多次变载（restep / 复合事件）",
    "verdict": "**不可替代**（v6 在此工况上比 v5.1 更危险）",
    "criterion": "存在可测的 T_stable 且 ≤2 s；否则至少「事件后偏差」不劣于 v5.1 **且** 无永久锚点损伤",
    "evidence_v6": "restep 19 个事件仅 2 个可测且都 >2 s（75 s 量级）；OS5 中位 −18.59%·J（11 个中 9 个向下 >2%）；G=0.473、err1s=−286.7%；MD 383→1440 ADC（v5.1 的 3.8 倍）",
    "evidence_ref": "T1-A t1a_target_verdict.csv（caliber=ch5,group=restep,n_measurable=2）；T5-A t5a_overcharge_summary.csv:OS5_med(kind=restep) + t5a_internal_trace.csv（帧 7187~8167）；T3-B t3b_route_roi.csv:MD_tot_med 与 epoch_per100s",
    "risk": "根因是状态机继承旧修正量 c0 再被 pin 永久钉住（T5-A 逐帧锁死）⇒ **v6.1 的 ROM_SCALE 治不了**（US5_max 两臂同为 1230.09%）；T_stable 口径在此工况下结构性失效（容差/单通道噪声中位仅 1.20，9/19 个 restep <1）",
    "precondition": "必须先做 T5-A 建议的「清复合事件继承的 c0」+ T7-A 的「卸载/减重判据前移」；并把验收从 T_stable 换成事件级偏差指标（T1-A §6.3）",
    "confidence": "高（机制被逐帧锁死，两个独立席位同向）"})
S.append({
    "condition": "③ 卸载与部分卸载",
    "verdict": "**有条件可替代**（全卸载可替代；部分卸载不可替代）",
    "criterion": "全卸载：T_stable 达标 **且** 残余偏移 ≤2%·|J| **且** 反向事件误判 = 0；部分卸载：20%~80% 档样本 ≥10 例后重判",
    "evidence_v6": "全卸载 9/9 达标（中位 0.09 s、worst 1.15 s）；残余 4~6 s 中位 +0.19%·|J|（max 2.81%，1/14 不衰减）；反向误判 0/19；卸载沿 t90 中位 0.015~0.040 s",
    "evidence_ref": "T1-A t1a_target_verdict.csv:rate_2s(group=unload)；T7-A t7_frozen_summary.csv:resid4_pct/resid_late_pct；T7-B t7b_partial_amp_response.csv:kind_eff/drop_frac_eff 与 t7b_reverse_misjudge.csv",
    "risk": "20%~80% 档真实样本 **n=1**（T7-B）；T4-A 原标的 5 例里有 2 例实测为全卸载（post 窗被后续加载沿压到）；小减重档 n=2 会被继续当蠕变扣掉（稳态偏差 −26.9%/−38.4%）",
    "precondition": "① 卸载/减重判据前移进事件态（T7-A 建议 1）；② 补采 20%~80% 档 ≥10 例后才能把「有条件」升级为「可替代」",
    "confidence": "全卸载高、部分卸载低（n=1，仅定性）"})
S.append({
    "condition": "④ 强扰动环境（人手拍击 / 抖动 / ±1000 ADC 级扰动）",
    "verdict": "**不可替代**（v6 与 v5.1 都失效，且 v6 的退化路径更危险）",
    "criterion": "在标称最大扰动下 M1~M4 复合失败率 ≤5% **且** 无永久锚点损伤（>20% 电平）",
    "evidence_v6": "12048 ADC + ±1000 ADC 白噪：失败率 10/10=100%（Wilson CI [0.72,1.00]），M2 锚错 1.00、M4 显示越界 1.00；A50 仅 75.7 ADC（0.63% 电平）；最大锚点误差 ΔΣA=11377 ADC=94% 电平且不自愈",
    "evidence_ref": "T5-B t5b_q1_headline.csv、t5b_boundary_measured.csv:A50、t5b_internal_trace.csv/t5b_mech_chain.csv；T1-B t1b_summary_noise_adc.csv:n_miss_mean",
    "risk": "v6 对 ±1 包时序抖动的路径 RMS 差 7.90%（v5.1 0.26%，30.6×）；v5.1 无此类状态机 ⇒ 强扰动下**退回 v5.1 反而更安全**（但 v5.1 的过充/台阶代价仍在）",
    "precondition": "必须上「低置信期冻结锚定」（拍击永久锚点损伤 2713→116 ADC、干净数据副作用 0），并补采 1w ADC ± 1000 ADC 实测录制后重判",
    "confidence": "高（独立于 v6 侧口径：T5-B 用的是 M1~M4 失败定义，不依赖 T_stable）"})

write_csv(os.path.join(RESULTS, "t8_substitutability.csv"), S, SUB_FIELDS)

# ------------------------------------------------------ T8-Q6 数据缺口清单
GF = ["prio", "gap", "impact", "spec", "source_task", "source_file", "merged_from"]
G = []
G.append({
    "prio": "P0", "gap": "无载荷量（N/gf）标定与真值力",
    "impact": "「快相形状/过充/重复性是否依赖载荷量级」完全无法判定；所有 N 口径验收缺真值；跨域（显示域 N vs ADC）无法换算",
    "spec": "同一传感器同一加载装置：5/10/20 N ×（硬阶跃 / 0.5 s 斜坡）× 各 ≥5 次；力传感器同步采集、时间对齐 ≤5 ms；把 force_N 与标定系数写入 session 的 calibration.params",
    "source_task": "T3-A G1 / T4-A G1,G4 / T4-B G1,N1,N2 / T7-A G3 / T1-B G1",
    "source_file": "results/conclusions_T3A.json:gaps[0] 等",
    "merged_from": "5 处同源缺口合并（G1 系列）"})
G.append({
    "prio": "P0", "gap": "无「1w ADC ± 1000 ADC」量级实测录制",
    "impact": "用户报的「找不到基线」只能靠注入复现；A50/A 曲线全部是注入结论",
    "spec": "① 目标 10000±500 ADC 稳定加载；② 同录制叠加已知幅度扰动源并同步记录扰动参考通道；③ 每档 ≥10 次重复；④ 每档 ≥60 s（覆盖 M2 的 +20 s 检查点）",
    "source_task": "T5-B G1",
    "source_file": "results/conclusions_T5B.json:gaps[0]",
    "merged_from": "—（唯一来源，但直接对应总报告 G7）"})
G.append({
    "prio": "P0", "gap": "无人手拍击/揉压实录（全库仅 3 例瞬态候选，最大一例 v6 连事件都不建）",
    "impact": "无法在实测上评估「拍击是否被误捕获」；空间判据在本数据上不可判定（注入按电平分摊，构造性抹平空间差异）",
    "spec": "≥20 次轻/重/连拍实录，必须记录**逐通道原始值**，并标注拍击位置与接触面积；保压段单独拍摄",
    "source_task": "T5-B G2,G3 / T1-B G6",
    "source_file": "results/conclusions_T5B.json:gaps[1],[2]；conclusions_T1B.json:gaps[1]",
    "merged_from": "T5-B 的「人手拍击实录」与 T1-B 的「保压段干净拍击样本」合并"})
G.append({
    "prio": "P0", "gap": "无受控速率实验（T_ramp 只能反演，不能标定）",
    "impact": "判据无法落地；t25_sus 与 T_ramp 两个「输入快慢」度量在 restep 上一致率仅 53%（10/19）",
    "spec": "力控/位移控装置 0.05 / 0.2 / 0.5 / 1.0 / 2.0 s 五档上升时间 × 各 5 次，同时记录 T_ramp 与因果替身",
    "source_task": "T2-A G3 / T4-A G3 / T4-B G3,N5,N7 / T3-A G1",
    "source_file": "results/conclusions.json(T2):gaps[2]；conclusions_T4A.json:gaps[2]；conclusions_T4B.json:gaps[2],[6]",
    "merged_from": "4 处合并"})
G.append({
    "prio": "P0", "gap": "无「同一标称载荷重复加载」样本（Δ_rep 无法计算）",
    "impact": "轮次偏差无法与操作载荷差异分离；重复性只能在恒载族上判（变载工况零重复样本）；轮次间表观幅度差中位 +23.3% 无法归因",
    "spec": "同一载荷 ×3 次加压-卸载-再加压；每轮卸载后 ≥60 s 纯空载；同一录制内 ≥5 轮（覆盖 ≥200 s）",
    "source_task": "T7-A G2,G6 / T6 G1 / T4-A G7 / T7-B G1",
    "source_file": "results/conclusions_T7A.json:gaps[1]；conclusions.json(T6):gaps[0]",
    "merged_from": "4 处合并（含「录制内多次同载荷重复」）"})
G.append({
    "prio": "P1", "gap": "clean restep 样本过少（n=6~7），部分卸载 20%~80% 档 n=1",
    "impact": "restep 的一切分位数都是低置信；部分卸载的幅度-回复关系无法定出",
    "spec": "录制「孤立变载」（前后各留 8 s 空档）× ≥15 例；同一载荷上 3~4 级递减台阶 × ≥3 次重复；带载硬阶跃与 0.5 s 斜坡各一组",
    "source_task": "T3-A G2,G3 / T4-A G7 / T4-B G6 / T7-B G1 / T2-A G1",
    "source_file": "results/conclusions_T3A.json:gaps[1]；conclusions_T7B.json:gaps[0]",
    "merged_from": "5 处合并"})
G.append({
    "prio": "P1", "gap": "无「拍击后加压」中间态样本（T4-Q4 的 H 类 n=0）",
    "impact": "「强扰动后基线能否恢复」无法验证；该中间态无法表征",
    "spec": "保压中注入拍击后 0.2~2 s 内立即加压，≥10 组",
    "source_task": "T4-A G2 / T3-B G9",
    "source_file": "results/conclusions_T4A.json:gaps[1]",
    "merged_from": "2 处合并"})
G.append({
    "prio": "P1", "gap": "无长保压（>300 s）与卸载后长空载（≥10 min）录制",
    "impact": "「加载历史累积是否改变快相形状」不可判；零漂的时间常数与物理来源不可分离（黏弹回复 vs 接触状态 vs 温漂）",
    "spec": "(a) 长纯空载 ≥10 min 看零漂时间常数；(b) 不同环境温度对照 + 温度记录；(c) 卸载后拆装复位对照；(d) 长保压 >300 s",
    "source_task": "T3-A G9 / T4-B（长保压缺口登记）/ T5-A G9 / T7-A G4",
    "source_file": "results/conclusions_T3A.json:gaps[8]；conclusions_T7A.json:gaps[3]",
    "merged_from": "4 处合并"})
G.append({
    "prio": "P1", "gap": "显示域量化与「读数跌破 0」不可观测",
    "impact": "恒载族 |J| 只有 11~17 个显示单位 ⇒ f(tau) 分辨率仅 0.06~0.09 归一化单位；逐通道 min(X)=0、无负值 ⇒ 无法分辨卸载谷值是过冲还是落到底",
    "spec": "用 ADC 原始域重录恒载组；采集侧关掉 ≥0 阈值或记录负 ADC",
    "source_task": "T3-A G5 / T4-A G6 / T7-A G5",
    "source_file": "results/conclusions_T3A.json:gaps[4]；conclusions_T7A.json:gaps[4]",
    "merged_from": "3 处合并"})
G.append({
    "prio": "P1", "gap": "跨载荷量级档位的 onset 事件只有 4 个（落点 <0.55·录制峰值）",
    "impact": "相对量级依赖只能用 n=4 定性；半量级档的行为未知",
    "spec": "至少 10 个落到半量级档位（0.4~0.6 峰值）的 onset，且分布在 ≥3 份不同录制里",
    "source_task": "T4-B G2,N4",
    "source_file": "results/conclusions_T4B.json:gaps[1]",
    "merged_from": "—"})
G.append({
    "prio": "P1", "gap": "无同一工况重复录制（±1 包敏感性只能做时间轴平移 = 伪重复）",
    "impact": "「判据必须在 t_on+30 ms 内下决定」「抖动 → 用户可见偏差」都只是单包抖动下的推论，无法换算成实测重复性",
    "spec": "同一工况重复录制 ≥5 次，比对判据统计量的真实重复性；并抓包核对 reorder_frac（T6 的时序模型假设 13% 的包被保序抬升）",
    "source_task": "T4-B G5 / T5-A G8 / T6 G5 / T7-B（未验证项）/ T3-A G7",
    "source_file": "results/conclusions_T4B.json:gaps[4]；conclusions.json(T6):gaps[3],[5]",
    "merged_from": "5 处合并"})
G.append({
    "prio": "P1", "gap": "高采样率（≥500 Hz）录制缺失，卸载/快相瞬态落在包周期以内",
    "impact": "12/16 全卸载的 t90 ≤1 个包周期 ⇒ 卸载是否在单包内完成不可分辨；f(0.05 s) 的 ±1 包不确定度中位 0.017~0.097·J（相对 24%~40%）",
    "spec": "≥500 Hz（包周期 ≤2 ms）录制，或保留包内逐帧时间戳；至少覆盖 onset / unload / partial_unload 三类各 ≥5 例",
    "source_task": "T2-A G2 / T7-B G2",
    "source_file": "results/conclusions.json(T2):gaps[1]；conclusions_T7B.json:gaps[1]",
    "merged_from": "2 处合并"})
G.append({
    "prio": "P2", "gap": "无录制的施加方式标注（一次到位 / 分两级 / 中途保压 / 卸载→再加压的间隔）",
    "impact": "复合事件（基线在动）的触发条件只能用 17 个 restep + 5 个 partial_unload 描述；分支代价与停滞检测代价无法分离",
    "spec": "录制时标注施加方式各 ≥5 次；卸载后 0.5 / 2 / 5 s 再加压 × 各 ≥5 次",
    "source_task": "T5-A G7 / T4-B G4",
    "source_file": "results/conclusions_T5A.json:gaps[6]；conclusions_T4B.json:gaps[3]",
    "merged_from": "2 处合并"})
G.append({
    "prio": "P2", "gap": "无「加载前静置 ≥120 s」的基线录制",
    "impact": "「基线识别之后」的绝对基线可信度无法判定",
    "spec": "每类工况前静置 ≥120 s 后再加载，≥5 份",
    "source_task": "T5-A G9",
    "source_file": "results/conclusions_T5A.json:gaps[8]",
    "merged_from": "—"})
G.append({
    "prio": "P2", "gap": "低幅值（检测门限以下）加载序列缺失",
    "impact": "max(8σ_d, 2% 记录极差) 门限以下的慢小增量未表征；07-v6 的 24 行事件里 9 行不在 T4-A 事件集内",
    "spec": "设计 0.5%~2% 记录极差的阶梯加载序列 × ≥5 次，配合提高信噪比",
    "source_task": "T4-A G5 / T2-A G6",
    "source_file": "results/conclusions_T4A.json:gaps[4]；conclusions.json(T2):gaps[5]",
    "merged_from": "2 处合并"})
G.append({
    "prio": "P2", "gap": "干扰样本：W2 窗口的 A50 不可用、DET_K 5→8 只有小样本",
    "impact": "W2 在 200 ADC 处已 0.917，拟合 A50=29.1 ADC 落在实测区间外；DET_K 收益只有 n=2~4 run",
    "spec": "W2 补测 20~150 ADC；DET_K 按 ≥10 种子 × ≥2 录制复核",
    "source_task": "T5-B G4,G5",
    "source_file": "results/conclusions_T5B.json:gaps[3],[4]",
    "merged_from": "2 处合并"})
G.append({
    "prio": "P2", "gap": "现场 exe 版本未知（v6 / v6.1 / v5.1）",
    "impact": "T5-A 的过充归因是否适用于用户那次体验取决于此；若现场是 v5.1（OS5 max 仅 +54.39%）则归因不适用",
    "spec": "核对现场 exe 的版本号 / 关于对话框 / 产物文件名（CMakeLists 的 TACTILESENSE_DISPLAY_VERSION）",
    "source_task": "T5-A G1",
    "source_file": "results/conclusions_T5A.json:gaps[0] / blockers[0]",
    "merged_from": "★ 待用户确认项（T8-Q7 必列）"})
G.append({
    "prio": "P2", "gap": "算法侧未做 C++ 落地验证与真机 A/B",
    "impact": "全部结论基于 Python 原型逐行复制件；原型↔C++ 常量 45/45 一致但 Push() 多 1 帧（≈10 ms）滞后；HO_MIN / lowconf 阈值 / κ 工作点都需真机复核",
    "spec": "按补丁 A/B 规范（输出级 + 内部 trace 级 + 阳性对照）在 C++ 侧复现关键矛盾；真机 A/B 复核 HO_MIN 3.5 s 与低置信阈值",
    "source_task": "T1-B G9 / T5-A G2 / T5-B G7,G8 / T7-B G6 / T6（未验证项）",
    "source_file": "results/conclusions_T1B.json:gaps[8]；conclusions_T5B.json:gaps[6]",
    "merged_from": "5 处合并"})
G.append({
    "prio": "P2", "gap": "restep 慢相样本不足（统一 30 s 窗只有 1~2 个可用事件）",
    "impact": "restep 慢相只能定性；τ_slow 不可辨识（换窗口给 13.23 s）",
    "spec": "「带载阶跃 + 长保压 ≥60 s」各 ≥5 次",
    "source_task": "T2-A G1",
    "source_file": "results/conclusions.json(T2):gaps[0]",
    "merged_from": "—"})
G.append({
    "prio": "P2", "gap": "M2（锚点偏离）指标混入事件结构差异、候选判据 AUC 是样本内上界",
    "impact": "T5-B 的失败率对窗口选择敏感（W1 的 100% 由一个临界事件贡献）",
    "spec": "设计与事件结构无关的单一显示层锚点指标；候选判据做折外评估",
    "source_task": "T5-B G6,G9",
    "source_file": "results/conclusions_T5B.json:gaps[5],[8]",
    "merged_from": "2 处合并"})
G.append({
    "prio": "P3", "gap": "v3 臂与 v6.1 臂的对照缺口",
    "impact": "评估矩阵中 v3 行与 v6.1 的多数维度只能写「缺数据」；v6.1 只做了 3 个相对档 × 5 种子的四臂对照",
    "spec": "按 T1-A 冻结口径补跑 v3 / v6.1 / v5.1 三臂的 T_stable、OS5、US5、MD、G、epoch 全套；v6.1 做全曲线而非相对档",
    "source_task": "T1-B G5 / T8（本席位新发现的缺口）",
    "source_file": "results/conclusions_T1B.json:gaps[4]；results/t8_matrix.csv（本席位 v3/v6.1 的 NA 行）",
    "merged_from": "含 T8 自查发现的矩阵空格"})

write_csv(os.path.join(RESULTS, "t8_data_gaps.csv"), G, GF)

# ------------------------------------------------- 跨任务冲突最终裁决表
CF = ["cid", "conflict", "position_a", "position_b", "ruling", "final_wording",
      "source_task", "source_file", "column", "n", "status"]
C = []
C.append({
    "cid": "X-1 (C-4) · κ 参数收益",
    "conflict": "第一轮：κ 1.30→1.10 使过充 7.06%→3.33%、T_stable 1.72→0.43 s、1 s 误差 +1.97%→+0.88%（三项同向变好）；T4-B：「κ 项本身零收益」，M0_now(1.30) 与 M2_correctA(1.12) 在 35 事件上输出完全相同",
    "position_a": "第一轮（κ_onset 有效）",
    "position_b": "T4-B（κ 零收益）",
    "ruling": "**两边都对，各自只覆盖光谱的一半**：κ 是单侧上限 min(A, κ·inc)，只有跨过触发阈值才起作用。第一轮改的是 κ_onset（有效），T4-B 比的是 κ_restep（无效）",
    "final_wording": "取 κ_onset = **1.10**、κ_restep 保持 **1.12**。依据：① 单独扫 κ_restep（κ_onset 固定 1.30）时 39/39 事件的 OS/T_stable/err1 逐事件相同（dOS_absmax ≤0.055 pt），只有 MD 有 ≤966 ADC 的单事件差异；② κ_onset ≥1.15 的六个工作点（1.15/1.20/1.25/1.30/1.35/1.40）结果逐事件完全相同（39/39）⇒ 它们本是同一个点；③ κ 1.30→1.10：OS5 中位 +0.02%→−0.06%、OS5 max 120.81%→101.08%、T_stable(5 s) 中位 3.660→0.820 s（p90 23.44→8.48）、err1 −21.69%→−23.11%、G 0.942→0.935；④ 不取 1.00/0.95（明显欠报：OS5 中位 −8.79%/−14.68%、G 0.844/0.778、T_stable 反升到 1.96 s）；⑤ 真实价值在压尾部：最坏折 T_stable(10 s) 78.50→5.36 s（14.6 倍）。上线成本 = 一行常量",
    "source_task": "T5-A（统一 κ 扫描裁决）/ T3-B（触发率独立实测）/ T4-B（原结论）",
    "source_file": "results/t5a_kappa_verdict.csv; results/t5a_kappa_summary.csv:dOS_absmax/n_OS_identical; results/t5a_kappa_loo.csv:T10_med_fold_max; results/t3b_route_roi.csv:trigger_rate",
    "column": "kappa_onset / n_OS_identical / T10_med_fold_max / trigger_rate",
    "n": "39 事件 / 13 折留一 / 13 录制",
    "status": "已裁决（C-4 关闭）"})
C.append({
    "cid": "X-2 · 驻留确认（dwell）是否有效",
    "conflict": "T1-B：dwell 0.15 s 使拍击 30% 的 epoch 2.0→1.0（计数下降）、T_stable 代价 +0.00 s ⇒ 唯一有效改造；T5-B（28 次配对拍击×4 档）：dwell 0.02→0.15 s 使存活伪事件 0.500→1.000、留下 >20% 永久锚点损伤的拍击占比 21.4%→42.9% ⇒ 负收益，Pareto 上 0.07~0.40 s 全部被现状支配",
    "position_a": "T1-B（有效）",
    "position_b": "T5-B（推翻，负收益）",
    "ruling": "**T5-B 对**，且两个数字并不矛盾，只是数了不同东西：T1-B 数的是「建了多少 epoch」（dwell 0.15 时均值 2.0→1.0，确实下降）；T5-B 数的是「撤销后仍存活、并留下锚点损伤的伪事件」（0.500→1.000，确实上升）。两者可以同时成立 —— 因为 dwell 越长，越多的短拍击不再触发（epoch 计数下降），但一旦触发就更可能跨过撤销窗而**走完交接、把 A 永久锚错**（存活伪事件上升）",
    "final_wording": "**不要把驻留确认作为默认改动。** 判据：以「用户实际后果」为准 —— 用户看到的是基线被锚错（永久锚点损伤），而不是内部 epoch 计数。T5-B 的 28 次配对实测给出：dwell 0.02→0.15 s 使存活伪事件 0.500→1.000、>20% 锚点损伤占比 21.4%→42.9%、抖动漏检 0.833→1.000、延迟 0.10→0.23 s，唯一改善是抖动显示超额 p90 0.853→0.800（−6% 相对）。Pareto 非支配点只有现状 (0.10 s, 0.500) 与 0.58 s 档 ⇒ 对 1~2 s 目标的最优工作点 = **现状不改**。保留 T1-B 的正面事实（dwell 0.15 s 的 T_stable 代价确实 ≤+0.005 s、clean onset 延迟 +0.12 s），但它不足以抵消永久损伤翻倍。**替代方案：低置信期冻结锚定**（拍击永久锚点损伤 2713→116 ADC、干净数据副作用 0.0000）",
    "source_task": "T5-B（裁决）/ T1-B（被推翻方）",
    "source_file": "results/t5b_dwell_verify.csv:tap_surv_mean/tap_dA_rel_gt20pct_frac；results/t1b_summary_costbenefit.csv:tap30_ep_mean；results/t1b_summary_tap.csv:n_ep_tap_med（dwell0.15 行 = 1.0）",
    "column": "tap_surv_mean / tap_dA_rel_gt20pct_frac / tap30_ep_mean / n_ep_tap_med",
    "n": "28 配对拍击 × 4 档 / 4 clean onset × 10 种子",
    "status": "已裁决（MainAgent 原判维持，本席位核验通过并补上「两数字为何不矛盾」的机制解释）"})
C.append({
    "cid": "X-3 · EMA 滤波是否「双赢」",
    "conflict": "第一轮：输出端 EMA τ=0.3 s 是双赢（过充 9.11%→5.14% 且 T_stable 2.72→1.89 s）；T3-B（13 录制 × 23 臂）：过充确实降（9.00%→7.33%），但 T_stable 反而变差（0.54→1.24 s）、MD 从 1440 暴涨到 4211 ADC（最大 26826）；T5-A：显示层可压约三成过充，内部状态层（进补偿器前）全部更差（G 掉到 0.438、过充中位翻成 −47%、T_stable 出现 NaN）",
    "position_a": "第一轮（双赢）",
    "position_b": "T3-B（变差）+ T5-A（显示层可压三成、内部层更差）",
    "ruling": "**T3-B 对**（第一轮漏报 MD，代价当时不可见）。★ **口径差异必须说清**：两者用的不是同一个滤波窗 —— 第一轮报的「τ=0.3 s 是双赢」出自 r4/filter_pareto 的 **输出端 EMA 全档扫描**，其 T_stable 列还带**切片索引 bug（所有值整齐偏大 1.0 s）**，且**没有报 MD**；T3-B 用的是 **30 s 窗**口径（T_stable(30 s) / MD=事件后最大绝对偏差），T5-A 用的是 **5 s 窗**口径（OS5/T5_med），故「三成」这个数字的口径是 5 s 窗下的 OS5 中位：11.65%→8.54%（EMA τ=0.1 s）；若换成 T3-B 的 30 s 窗 + 总通道，则是 9.00%→7.33%（−18.6% 相对）。⇒ **三者不是同一件事**：第一轮看的是过充与稳定时间（漏掉 MD）；T3-B 补上 MD 后发现「把过充峰值换成了显示与原始读数的瞬时失配」；T5-A 进一步证明**落点必须在显示层**（内部层会破坏检测器判据与滑行器起点）",
    "final_wording": "**不要把 EMA 作为默认改动。** ① 显示层因果 EMA（τ=0.1~0.3 s）能把 5 s 窗过充中位压约三成（11.65%→8.54% @τ=0.1 s；9.00%→7.33% @τ=0.3 s/30 s 窗），但代价是把 MD 从 1440 抬到 4211 ADC（最大 26826）并把 T_stable 从 0.54 推到 1.24 s；② τ≥1 s 进入明确的时间换幅度区间（τ=3 s：过充 5.57% 但 T_stable 10.71 s、err1 74.8%）；③ **内部状态层滤波被否决**（G 掉到 0.438、过充中位翻成 −47%、T_stable NaN）。⇒ 若一定要压过充，用**过充硬限幅 clip_raw 0.02**（MD 中位 2396.7→555.4 ADC、G_min 与基线相同、不磨台阶），或用形状库上包络重标 rom_scale 1.04~1.06",
    "source_task": "T3-B（裁决）/ T5-A（落点裁决）/ 第一轮（被推翻方）",
    "source_file": "results/t3b_third_routes.csv:dMD_vs_v6/dT_vs_v6（arm=rt_filter030）；results/t5a_filter_pareto.csv:OS_med/T_med/MD_med（layer=display / internal_input）；results/t5a_nonfilter_options.csv:MD_med（C1_clip_0.02）",
    "column": "dMD_vs_v6 / OS_med / T_med / MD_med / G_min",
    "n": "13 录制 × 23 臂；41 加载事件 × 32 滤波档 × 2 落点",
    "status": "已裁决（第一轮结论作废）"})
C.append({
    "cid": "X-4 · partial_unload 样本数",
    "conflict": "T7-A：按冻结定义（减少量 20%~80%）本批「部分卸载」样本为 0 —— 5 个里 4 个降幅只有 3.4%/12.8%/15.0%/14.7%（<20%，属 restep/小减重），第 5 个（切换负载@212.40，降幅 74.0%）幅度合格但 post 窗被 6 s 内的变载污染（stab_post=0.127>0.05）；T7-B：实测 1 例（SW1@212.40，drop_frac_eff=0.7435，用自适应 post 窗）",
    "position_a": "T7-A（按冻结定义为 0）",
    "position_b": "T7-B（实测 1 例）",
    "ruling": "**两者都成立，是「样本数」与「可用样本数」的区别**：T7-A 按冻结定义 + 恒定 post 窗 [t0+4,t0+6] 判定污染后剔除 ⇒ 0；T7-B 改用自适应 post 窗（避开紧随其后的加载沿）把同一事件的减重幅度测准 ⇒ 1。另有 2 例属 <20% 的小减重档（15.0%/14.7%），T7-B 单列为 decrement_small，不计入 partial_unload；T4-A 原标的另外 2 例（SW1@129.37 标 3.4%、SW1@182.03 标 12.8%）经自适应窗实测为**全卸载**（99.8% 电平）",
    "final_wording": "统一表述：**本批 19 个真实减重事件中，全卸载 16 例、严格意义的「部分卸载」（减少量 20%~80%）1 例、小减重（<20%）2 例。** 引用规则：① 「部分卸载 n=1」时**必须标注仅作定性参考**，且必须写明 post 窗口径（自适应窗 vs 冻结窗）；② 冻结定义下的 partial_unload 表（T4-A t4a_morphology.csv 的 5 行）**不可作为「部分卸载有 5 个样本」使用** —— 其中 4 个幅度不合格（<20%）、1 个 post 窗受污染，另 2 个经自适应窗实测为全卸载；③ 该 1 例的行为：t90=0.20 s（±1 包 0.16~0.24 s）、f(0.2 s)=0.908、相对稳态回弹 −9.2%、稳态偏差 −2.25%（v5.1）/−2.30%（v6）（−371/−378 ADC）",
    "source_task": "T7-B（裁决，提供自适应窗口测量）/ T7-A（冻结定义侧）",
    "source_file": "results/t7b_partial_amp_response.csv:kind_eff/drop_frac_eff/post_win_kind；results/conclusions_T7A.json:corrections[8]",
    "column": "kind_eff / drop_frac_eff / drop_frac_frozen / post_win_kind",
    "n": "19 减重事件（16/1/2）",
    "status": "已裁决（统一表述见 final_wording）"})
C.append({
    "cid": "X-5 · T_stable 网格敏感（0.3% 网格差翻转排名）",
    "conflict": "T1-A：1/9 份录制在 100 Hz vs 100.5 Hz（span/(n−1)）网格下 T_stable 从 3.79 s 变 6.66 s（路径抖动）；T6：dt=0.00995（legacy）时 v6 2.69 pp **优于** v5.1 3.38 pp，dt=0.010000 时 v6 3.52 pp **劣于** v5.1 3.39 pp，翻转全由「右拇指/数据3」一份引起",
    "position_a": "T1-A（网格是第 4 个口径自由度）",
    "position_b": "T6（0.3% 网格差翻转可重复性排名）",
    "ruling": "**两处独立发现互为正佐证，成立**：不是「某个脚本算错了」，而是 v6 的状态机决策（revoke / handoff / epoch）在网格微小变化下会翻面（与 T6 的 L3 路径分歧机制同源）；v5.1 因用 1 s 均值窗（[2,3] s）而对 t0/grid 不敏感（±1 包下序列相同率 1.00）",
    "final_wording": "**任何 T_stable / 可重复性排名都必须连同网格一起声明，并给出网格敏感性。** 具体：① 项目统一网格为 `timestamp` 列 → `np.interp` → 严格 100 Hz（dt=0.01），legacy 网格 `dtm=span/(n−1)` 仅作对照；② v6 的绝对值在两种网格下可差 1.76 倍（3.79→6.66 s，单份录制），v5.1 的排名在 0.3% 网格差下会与 v6 互换；③ 因此 v6 vs v5.1 的**中位差**（−2.5~−3.1 s、11~13/13 录制更快）是稳健的，而**单份录制的绝对值与「排序」结论不稳定** —— 后者必须用配对差而非绝对排名表达；④ 与 T4-B 的独立佐证一致：T_stable 中位在 2 样本 × 9 臂 × 2 类事件上只取 4 个值（0.500/9.630/2.170/0.480 s），差异主要来自样本而非臂",
    "source_task": "T1-A / T6（两处独立发现）",
    "source_file": "results/t1a_grid_sensitivity.csv；results/t6_gridsens.csv + t6_gridsens_disp.csv（转引 t6_improve_ab.csv:std_R5_pp）",
    "column": "T_stable / grid / dtm / plat_err_std_R5_pp / plat_err_std_Rstep_pp",
    "n": "9 份恒载（T1-A，1/9 敏感）；9 重复（T6，1 份引起翻转）",
    "status": "已裁决（升格为项目纪律）"})

write_csv(os.path.join(RESULTS, "t8_conflict_rulings.csv"), C, CF)
print("OK")
