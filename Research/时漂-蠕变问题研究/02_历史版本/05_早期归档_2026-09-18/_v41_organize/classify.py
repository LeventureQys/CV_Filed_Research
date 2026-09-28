# -*- coding: utf-8 -*-
"""temp/v4.1flash 版本化归档：分类与计划生成（dry-run 工具，不属于归档产物）。

用法：
    python classify.py            # 生成 plan.json + 报告
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "v4.1flash")
OUT_PLAN = os.path.join(HERE, "plan.json")

# ---------------------------------------------------------------- 版本桶定义
BUCKETS = [
    "01-v3-baseline",
    "02-dsp-route",
    "03-v4",
    "04-v5",
    "05-v5.1",
    "06-v5.1-exempt-1s-3s-5s",
    "07-v6",
    "08-v6.1",
    "09-v7",
    "10-paper-v5-route",
    "11-paper-v6",
    "12-fig-audit",
]

# ------------------------------------------------- 主 scripts/ 的初版归属（按证据）
B_DSR, B_V4, B_V5, B_V51, B_EX, B_V6, B_V61, B_V7 = (
    "02-dsp-route", "03-v4", "04-v5", "05-v5.1",
    "06-v5.1-exempt-1s-3s-5s", "07-v6", "08-v6.1", "09-v7")

BASE_SCRIPT = {
    # 01 —— v3 基线
    "glm53_v3.py": "01-v3-baseline",

    # 02 —— dsp.md 逆滤波路线评审（09-17 10:01~10:31）
    "v_probe_p4_diag.py": B_DSR, "v_probe_scale_diag.py": B_DSR,
    "v_dsp_review_probe.py": B_DSR, "v_probe_decisive.py": B_DSR,
    "v_shape_diag.py": B_DSR, "inventory.py": B_DSR, "w_dsp_vs_v3.py": B_DSR,
    "v_iir_vs_parallel.py": B_DSR, "v_iir_diag2.py": B_DSR,
    "v_iir_accuracy.py": B_DSR, "v_verdict.py": B_DSR,
    "v_transient_cost.py": B_DSR, "v_final_check.py": B_DSR,
    "x_model_id.py": B_DSR, "w2_dsp_vs_v3.py": B_DSR, "y_spatial.py": B_DSR,
    "z_shape_look.py": B_DSR, "v_iir_equiv.py": B_DSR,
    "e_dsp_fail.py": B_DSR, "f_review_figs.py": B_DSR,

    # 03 —— v4（快相免责期、快慢相、变载可辨识性、五算法对比）
    "q_two_phase.py": B_V4, "q2_slope_look.py": B_V4, "s_residual_nature.py": B_V4,
    "t_trace_early.py": B_V4, "u_varying_fast.py": B_V4, "v2_track.py": B_V4,
    "w2_repeat.py": B_V4, "x2_fastfig.py": B_V4, "y2_varying_steps.py": B_V4,
    "y3_varying_steps2.py": B_V4, "r_fastphase.py": B_V4, "z8_gap.py": B_V4,
    "z9_varying_fig.py": B_V4, "aa_final_compare.py": B_V4, "ab_final_figs.py": B_V4,
    "ac_final5.py": B_V4, "ad_probe_new.py": B_V4, "ad_probe_profile.py": B_V4,
    "ad_check_v4_restep.py": B_V4, "ad_trace_v3.py": B_V4, "ad_v4.py": B_V4,
    "ad_single_run.py": B_V4, "ad_trace_tail.py": B_V4, "ad_trace_step.py": B_V4,
    "ad_single_fig.py": B_V4, "ae_weight_scenario.py": B_V4, "af_two_recs.py": B_V4,
    "ae_weight_fig.py": B_V4, "ae_multi_step.py": B_V4, "af_two_figs.py": B_V4,
    "ad_lib.py": B_V4,
    "_tmp_prof.py": B_V4, "_eq.py": B_V4, "_rob.py": B_V4, "_font.py": B_V4,
    "_gap.py": B_V4, "_ema.py": B_V4, "_png.py": B_V4, "_png2.py": B_V4,
    "_datab.py": B_V4, "_trace13.py": B_V4,

    # 04 —— v5（P1~P4 定点修复、C++ 落地与对拍）
    "z2_v5.py": B_V5, "z4_v5_trace.py": B_V5, "z5_diag_v5.py": B_V5,
    "z6_diag_v5b.py": B_V5,
    "ba_v5_scenarios.py": B_V5, "bb_v5_debug.py": B_V5, "bb_v5_debug2.py": B_V5,
    "bc_v5_recordings.py": B_V5, "bd_v5_static9.py": B_V5, "be_v5_epochs.py": B_V5,
    "be_v5_threshold.py": B_V5, "bf_v5_fig.py": B_V5, "bg_v5_cpp_parity.py": B_V5,
    "bh_v5_why_fig.py": B_V5, "bi_v5_5s_awin.py": B_V5,
    "glm53_v5.py": B_V5,
    "_gapcalc.py": B_V5, "_none.py": B_V5, "_diff.py": B_V5, "_runbh.py": B_V5,
    "_png3.py": B_V5,

    # 05 —— v5.1（零点诊断 -> 取消空载强制归零 -> 复核）
    "bj_probe_zero.py": B_V51, "bk_trace_unload.py": B_V51,
    "bl_decompose_zero.py": B_V51, "bm_synth_zero_pin.py": B_V51,
    "bn_trace_baseline.py": B_V51, "bo_v51_verify.py": B_V51,
    "bv_scenarios_v51.py": B_V51, "bw_metrics_current.py": B_V51,
    "glm53_v51.py": B_V51,

    # 06 —— 免责期 1s / 3s / 5s 专项
    "bp_exempt_sweep.py": B_EX, "bq_probe_1s.py": B_EX, "br_target_13ffca.py": B_EX,
    "bs_loadedge_latency.py": B_EX, "bt_constant_load_timeline.py": B_EX,
    "bu_pending_vs_exempt.py": B_EX, "bx_nofreeze_ab.py": B_EX,
    "by_timeline_segments_13ffca.py": B_EX, "bz_zoom_1868.py": B_EX,
    "ca_all_datasets_1s.py": B_EX,

    # 07 —— v6（形状约束反演 + 滑行器）
    "cb_v6_stepshape.py": B_V6, "cc_v6_rawrise.py": B_V6, "cd_v6_tail.py": B_V6,
    "ce_v6_estimator.py": B_V6, "cf_v6_generalize.py": B_V6,
    "cg_v6_selfshape.py": B_V6, "ch_v6_singlepoint.py": B_V6,
    "ci_v6_vs_1s_3s.py": B_V6, "cj_v6_smoke.py": B_V6, "ck_v6_trace.py": B_V6,
    "cl_v6_holddiag.py": B_V6, "cm_v6_outlier.py": B_V6, "cn_v6_userreport.py": B_V6,
    "cp_v6_detwin_sweep.py": B_V6, "cq_v6_userreport2.py": B_V6,
    "cr_v6_unload.py": B_V6, "cs_v6_stateevo.py": B_V6, "ct_v6_verify_figs.py": B_V6,
    "glm53_v6.py": B_V6,

    # 08 —— v6.1（阶跃尖峰定点修复）
    "cu_v61_spike_diag.py": B_V61, "cv_v61_overshoot.py": B_V61,
    "cw_v61_ab.py": B_V61, "cx_v61_smoke.py": B_V61, "cy_v61_shapecal.py": B_V61,
    "cz_v61_check.py": B_V61, "da_v61_window.py": B_V61, "db_v61_overest.py": B_V61,
    "dc_v61_figs.py": B_V61, "dd_v61_equiv.py": B_V61, "de_v61_ledger.py": B_V61,
    "df_v61_ablate.py": B_V61, "dg_v61_panelcheck.py": B_V61,
    "dh_v61_13ffca.py": B_V61, "di_v61_13ffca_ablate.py": B_V61,
    "glm53_v61.py": B_V61,

    # 09 —— v7（已否决的变体，仅存档）
    #   z3_v6.py 名字写 v6，语义实为后来的 v7（首次 onset 固定冻结、变载不冻结）
    "glm53_v7.py": B_V7, "z7_v7_test.py": B_V7, "ad_check_v7.py": B_V7,
    "z3_v6.py": B_V7,
}

PAPER_DIRS = {
    "paper": "10-paper-v5-route",
    "paper_v6": "11-paper-v6",
}

# ------------------------------------------- 非脚本、非图片、非结果的文档归属
BASE_DOC = {
    "dsp方案评审报告.md": B_DSR,
    "快相与慢相分离分析.md": B_V4,
    "五算法实测对比.md": B_V4,
    "免责期1s-3s-5s对比.md": B_EX,
    "v6与免责1s3s对比.md": B_V6,
    os.path.join("Document", "02-v4算法说明.md"): B_V4,
    os.path.join("Document", "03-验证与实测结果.md"): B_V4,
    os.path.join("Document", "04-C++实现骨架.md"): B_V4,
    os.path.join("Document", "05-v5算法说明.md"): B_V5,      # 含 §10 v5.1 -> 后续并入 05
    os.path.join("Document", "06-抗蠕变漂移补偿算法说明.md"): B_V51,
    os.path.join("Document", "07-v6算法说明.md"): B_V6,
    os.path.join("Document", "08-v6.1算法说明.md"): B_V61,
    # 3 个跨版本索引入 legacy（重组织前的入口，不归任一算法版本）
    "README.md": "legacy",
    os.path.join("Document", "README.md"): "legacy",
}

# 跨版本共享文档（同样复制到多桶后删除原件）
SHARED_DOC = [
    (os.path.join("Document", "05-v5算法说明.md"), [B_V5, B_V51]),   # v5 主文档含 §10 v5.1 修订
]

# ------------------------------------------------ 结果文件（results/）归属规则
RESULT_EXACT = {
    # 02 dsp 路线
    "review_probe.txt": B_DSR, "final_check.txt": B_DSR, "_final_check.txt": B_DSR,
    "_verdict.txt": B_DSR, "_iir_acc.txt": B_DSR, "_iir_diag2.txt": B_DSR,
    "_iir_equiv.txt": B_DSR, "_transient.txt": B_DSR, "spatial_check.txt": B_DSR,
    "spatial_deconv.csv": B_DSR, "model_identification.csv": B_DSR,
    "model_shape_check.csv": B_DSR, "dsp_failure.txt": B_DSR,
    "dsp_fail_metrics.csv": B_DSR, "dsp_fail_summary.csv": B_DSR,
    "dsp_fit_params.csv": B_DSR, "dsp_vs_v3_metrics.csv": B_DSR,
    "dsp_vs_v3_summary.csv": B_DSR, "dataset_inventory.csv": B_DSR,

    # 03 v4
    "two_phase_structure.csv": B_V4, "two_phase_structure.txt": B_V4,
    "fastphase_metrics.csv": B_V4, "fastphase_summary.csv": B_V4,
    "fastphase_repeatability.txt": B_V4, "residual_nature.csv": B_V4,
    "track_accuracy.csv": B_V4, "varying_fastphase.csv": B_V4,
    "varying_identifiability.csv": B_V4, "varying_steps.csv": B_V4,
    "final5_metrics.csv": B_V4, "final5_summary.csv": B_V4,
    "final5_drift_by_dataset.csv": B_V4, "final_4algo_metrics.csv": B_V4,
    "final_4algo_summary.csv": B_V4, "final_4algo_drift_by_dataset.csv": B_V4,
    "_slope_look.txt": B_V4,
    "new_switch_load_events.csv": B_V4, "new_switch_load_metrics.csv": B_V4,
    "new_switch_load.npz": B_V4, "rec_13ffca_events.csv": B_V4,
    "rec_13ffca_metrics.csv": B_V4, "rec_13ffca.npz": B_V4,
    "rec_1d9493_events.csv": B_V4, "rec_1d9493_metrics.csv": B_V4,
    "rec_1d9493.npz": B_V4, "rec_midload_events.csv": B_V4,

    # 04/05 v5 家族
    "v5_static9_metrics.csv": B_V5, "v5_static9_summary.csv": B_V5,
    "v5_maxgap.csv": B_V5, "v5_midload_events.csv": B_V5,
    "v5_切换负载-快相无责_windows.csv": B_V5, "v5_切换负载-快相无责.npz": B_V5,
    "v5_再切换负载_windows.csv": B_V5, "v5_再切换负载.npz": B_V5,
    "v5_中途切换-13ffca_windows.csv": B_V5, "v5_中途切换-13ffca.npz": B_V5,
    "v5_中途切换-1d9493_windows.csv": B_V5, "v5_中途切换-1d9493.npz": B_V5,
    "_v5_rec.log": B_V5, "_v5_rec3s.log": B_V5, "_v5_static.log": B_V5,
    "_v5_static3s.log": B_V5,
    "scenarios_v51.csv": B_V51, "metrics_current.csv": B_V51,
    "metrics_current_slowwin.csv": B_V51,
    "_scenarios_v51.log": B_V51, "_metrics_current.log": B_V51,

    # 06 免责期三档
    "exempt_sweep.npz": B_EX, "exempt_sweep_ablation.csv": B_EX,
    "exempt_sweep_awin.csv": B_EX, "exempt_sweep_deddelay.csv": B_EX,
    "exempt_sweep_events.csv": B_EX, "exempt_sweep_mech.csv": B_EX,
    "exempt_sweep_recs.csv": B_EX, "exempt_sweep_static9.csv": B_EX,
    "exempt_sweep_static9_summary.csv": B_EX, "all_datasets_1s.csv": B_EX,
    "constant_load_timeline.csv": B_EX, "loadedge_latency.csv": B_EX,
    "pending_vs_exempt.csv": B_EX, "nofreeze_ab.csv": B_EX,
    "target_13ffca_deddelay.csv": B_EX, "target_13ffca_edges.csv": B_EX,
    "target_13ffca_events.csv": B_EX, "target_13ffca_metrics.csv": B_EX,
    "target_13ffca_phases.csv": B_EX, "target_13ffca.npz": B_EX,
    "timeline_segments_13ffca.csv": B_EX,
    "_exempt_sweep.log": B_EX, "_exempt_1s_probe.log": B_EX,
    "_all_datasets_1s.log": B_EX, "_constant_load_timeline.log": B_EX,
    "_loadedge_latency.log": B_EX, "_pending_vs_exempt.log": B_EX,
    "_nofreeze_ab.log": B_EX, "_target_13ffca.log": B_EX,
    "_timeline_segments.log": B_EX, "_zoom_1868.log": B_EX,
}

RESULT_PREFIX = [
    ("v6_", B_V6), ("_v6_", B_V6),
    ("v61_", B_V61), ("_v61_", B_V61),
    ("varying_steps_v5.csv", B_V5), ("varying_steps_v6.csv", B_V6),
    ("varying_steps_v7.csv", B_V7),
]

# --------------------------------------------------------- 图片文件的来源归属
# 根 figures/ 下的图：由 savefig 反查脚本 -> 脚本所属桶
FIGURE_SCRIPT = {
    "f1_timeseries_grid.png": "w_dsp_vs_v3.py", "f2_load_zoom.png": "w_dsp_vs_v3.py",
    "f3_step_edge.png": "w_dsp_vs_v3.py",
    "f_review_1.png": "f_review_figs.py", "f_review_2.png": "f_review_figs.py",
    "f_review_3.png": "f_review_figs.py", "f_review_4.png": "f_review_figs.py",
    "f_fastphase_1.png": "x2_fastfig.py", "f_varying_change.png": "z9_varying_fig.py",
    "F1_overview.png": "ab_final_figs.py", "F2_zoom.png": "ab_final_figs.py",
    "F3_metrics.png": "ab_final_figs.py",
    "G1_overview.png": "ac_final5.py", "G2_varying.png": "ac_final5.py",
    "G3_zoom.png": "ac_final5.py", "G4_metrics.png": "ac_final5.py",
    "new_switch_load_result.png": "ad_single_fig.py",
    "new_switch_load_metrics.png": "ad_single_fig.py",
    "weight_scenario.png": "ae_weight_fig.py",
    "rec_13ffca_result.png": "af_two_figs.py", "rec_1d9493_result.png": "af_two_figs.py",
    "rec_two_metrics.png": "af_two_figs.py",
    "v5_compare.png": "bf_v5_fig.py", "v5_diff_worst.png": "bh_v5_why_fig.py",
    "H1_exempt_1s_3s_5s.png": "bp_exempt_sweep.py",
    "H2_exempt_1s_ablation.png": "bp_exempt_sweep.py",
    "H3_13ffca_loadedge.png": "br_target_13ffca.py",
    "H4_13ffca_time_segments.png": "by_timeline_segments_13ffca.py",
    "H5_zoom_1868_pending_freeze.png": "bz_zoom_1868.py",
    "H6_1s_overview_all.png": "ca_all_datasets_1s.py",
    "H7_1s_metrics_all.png": "ca_all_datasets_1s.py",
    "I1_v6_overview_all.png": "ci_v6_vs_1s_3s.py",
    "I2_v6_metrics_all.png": "ci_v6_vs_1s_3s.py",
    "J1_v6_detwin_sweep.png": "cp_v6_detwin_sweep.py",
    "K1_v6_fix_verify.png": "ct_v6_verify_figs.py",
    "L1_v61_spike_fix.png": "dc_v61_figs.py", "L2_v61_metrics.png": "dc_v61_figs.py",
    "M1_v61_13ffca.png": "dh_v61_13ffca.py",
}

FIG_AUDIT_DIRS = ["_audit", "_crop", "_review_crops", "_zoom"]

# ----------------------------------------------- 其它整目录/整文件的归属
DIR_BUCKET = {
    "backup": ("01-v3-baseline", "source"),
    "cpp_snippet_check": ("03-v4", "cpp_check"),
    "cpp_v5_check": ("04-v5", "cpp_check"),
}


def read(p):
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def main():
    plan = defaultdict(lambda: defaultdict(list))     # bucket -> kind -> [relpaths]
    shared = defaultdict(set)                          # relpath -> {buckets}
    notes = []

    # ---- 1. 脚本：先按初版归属，再用 import 闭包扩散共享模块
    #       三个脚本目录都参与依赖图：scripts/、paper/scripts/、paper_v6/scripts/
    script_dirs = [os.path.join("scripts", ""),
                   os.path.join("paper", "scripts", ""),
                   os.path.join("paper_v6", "scripts", "")]
    scripts = {}
    for d in script_dirs:
        p = os.path.join(SRC, d)
        if not os.path.isdir(p):
            continue
        for name in os.listdir(p):
            if os.path.isfile(os.path.join(p, name)) and name.endswith(".py"):
                scripts[d + name] = os.path.join(p, name)
    missing = set(BASE_SCRIPT) - set(os.path.basename(r) for r in scripts)
    extra = set(os.path.basename(r) for r in scripts) - set(BASE_SCRIPT) - set(
        n for n in os.listdir(os.path.join(SRC, "paper", "scripts")) if n.endswith(".py")) - set(
        n for n in os.listdir(os.path.join(SRC, "paper_v6", "scripts")) if n.endswith(".py"))
    if missing:
        notes.append("!! BASE_SCRIPT 中不存在: %s" % sorted(missing))
    if extra:
        notes.append("!! 未归类的脚本: %s" % sorted(extra))

    imports = {}
    for name, p in scripts.items():
        txt = read(p)
        mods = set()
        for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", txt, re.M):
            mods.add(m.group(1))
        imports[name] = mods

    # 模块名 -> 脚本 relpath（跨三个脚本目录解析；同名优先同目录）
    mod2file = {}
    for rel in scripts:
        base = os.path.basename(rel)
        mod2file.setdefault(base[:-3], rel)

    base_of = {}
    for rel in scripts:
        base = os.path.basename(rel)
        top = rel.split(os.sep)[0]
        if top in PAPER_DIRS:
            base_of[rel] = PAPER_DIRS[top]
        else:
            base_of[rel] = BASE_SCRIPT.get(base)
    unassigned = [r for r, b in base_of.items() if not b]
    if unassigned:
        notes.append("!! 脚本无初版归属: %s" % sorted(unassigned))
    assign = {r: {b} for r, b in base_of.items() if b}
    for _ in range(10):
        changed = False
        for name, mods in imports.items():
            for m in mods:
                dep = mod2file.get(m)
                if dep and dep in assign and name in assign:
                    if not assign[dep] <= assign[name]:
                        assign[dep] |= assign[name]
                        changed = True
        if not changed:
            break

    # 反向：被引用的模块把桶扩散给引用者（保证每个桶自带全部依赖）
    for _ in range(20):
        changed = False
        for name, mods in imports.items():
            for m in mods:
                dep = mod2file.get(m)
                if dep and dep in assign and name in assign:
                    if not assign[name] <= assign[dep]:
                        # 引用者所在桶也需要该模块 -> 只记录“需要”，不回改引用者
                        pass
        # 计算每个桶需要的模块集合（闭包）
        need = defaultdict(set)
        for name, bs in assign.items():
            for b in bs:
                need[b].add(name)
        for b, names in list(need.items()):
            stack = list(names)
            while stack:
                cur = stack.pop()
                for m in imports.get(cur, ()):
                    dep = mod2file.get(m)
                    if dep and dep not in names:
                        names.add(dep)
                        assign.setdefault(dep, set()).add(b)
                        stack.append(dep)
                        changed = True
        if not changed:
            break

    for rel in scripts:
        bs = sorted(assign.get(rel, {"UNASSIGNED"}))
        for b in bs:
            plan[b]["scripts"].append(rel)
        if len(bs) > 1:
            shared[rel] = set(bs)

    # __pycache__ 跟随同名模块
    base2rel = {os.path.basename(r): r for r in scripts}
    for pyc_rel_dir in (os.path.join("scripts", "__pycache__"),
                        os.path.join("paper", "scripts", "__pycache__"),
                        os.path.join("paper_v6", "scripts", "__pycache__")):
        pyc_dir = os.path.join(SRC, pyc_rel_dir)
        if not os.path.isdir(pyc_dir):
            continue
        for name in os.listdir(pyc_dir):
            stem = name.split(".")[0] + ".py"
            owner = base2rel.get(stem)
            bs = sorted(assign.get(owner, set())) if owner else []
            if not bs:
                notes.append("!! __pycache__ 无归属: %s" % os.path.join(pyc_rel_dir, name))
                continue
            rel = os.path.join(pyc_rel_dir, name)
            for b in bs:
                plan[b]["scripts"].append(rel)
            if len(bs) > 1:
                shared[rel] = set(bs)

    # ---- 2. 结果：先用精确表，再用“脚本引用反查”，最后用前缀规则
    res_dir = os.path.join(SRC, "results")
    refs = defaultdict(set)          # 结果文件名 -> {脚本名}
    for name, p in scripts.items():
        txt = read(p)
        for fn in os.listdir(res_dir):
            if fn in RESULT_EXACT:
                pass
            if fn in txt:
                refs[fn].add(name)

    def result_buckets(fn):
        if fn in RESULT_EXACT:
            return {RESULT_EXACT[fn]}
        for pre, b in RESULT_PREFIX:
            if fn.startswith(pre):
                return {b}
        buckets = set()
        for s in refs.get(fn, ()):
            buckets |= assign.get(s, set())
        if len(buckets) == 1:
            return buckets
        if len(buckets) > 1:
            return buckets
        return set()

    unresolved = []
    for fn in os.listdir(res_dir):
        full = os.path.join(res_dir, fn)
        if os.path.isdir(full):
            continue
        bs = result_buckets(fn)
        if not bs:
            unresolved.append(fn)
            continue
        for b in sorted(bs):
            plan[b]["results"].append(os.path.join("results", fn))
        if len(bs) > 1:
            shared[os.path.join("results", fn)] = bs
    # results/superseded/
    sup = os.path.join(res_dir, "superseded")
    if os.path.isdir(sup):
        for fn in os.listdir(sup):
            bs = result_buckets(fn) or {B_DSR}
            for b in sorted(bs):
                plan[b]["results"].append(os.path.join("results", "superseded", fn))
            if len(bs) > 1:
                shared[os.path.join("results", "superseded", fn)] = bs

    # ---- 3. 图片（按源码里的 savefig 反查产出脚本，再取该脚本所属桶）
    figdir = os.path.join(SRC, "figures")
    for fn in os.listdir(figdir):
        s = FIGURE_SCRIPT.get(fn)
        owner = base2rel.get(s) if s else None
        if not owner or owner not in assign:
            notes.append("!! 图片无归属: %s" % fn)
            continue
        bs = sorted(assign[owner])
        for b in bs:
            plan[b]["figures"].append(os.path.join("figures", fn))
        if len(bs) > 1:
            shared[os.path.join("figures", fn)] = set(bs)

    for d in FIG_AUDIT_DIRS:
        p = os.path.join(SRC, d)
        for root, _dirs, files in os.walk(p):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), SRC)
                plan["12-fig-audit"]["figures"].append(rel)

    # ---- 4. 文档
    for rel, b in BASE_DOC.items():
        if os.path.isfile(os.path.join(SRC, rel)):
            plan[b]["docs"].append(rel)
        else:
            notes.append("!! 文档不存在: %s" % rel)

    # 跨版本共享文档：同时归入多个桶（复制后删除原件）
    for rel, bs in SHARED_DOC:
        if not os.path.isfile(os.path.join(SRC, rel)):
            notes.append("!! 共享文档不存在: %s" % rel)
            continue
        for b in bs:
            plan[b]["docs"].append(rel)
        shared[rel] = set(bs)

    # paper 与 paper_v6 整目录
    for d, b in PAPER_DIRS.items():
        for root, _dirs, files in os.walk(os.path.join(SRC, d)):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), SRC)
                if "__pycache__" in rel:
                    kind = "scripts"
                elif os.sep + "figures" + os.sep in rel:
                    kind = "figures"
                elif os.sep + "results" + os.sep in rel:
                    kind = "results"
                elif os.sep + "scripts" + os.sep in rel:
                    kind = "scripts"
                else:
                    kind = "docs"
                plan[b][kind].append(rel)

    # ---- 5. 其它整目录
    for d, (b, sub) in DIR_BUCKET.items():
        p = os.path.join(SRC, d)
        for root, _dirs, files in os.walk(p):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), SRC)
                plan[b][sub].append(rel)

    # ---- 汇总输出（先去重：paper 目录树与脚本依赖两条路径可能命中同一文件）
    for b in list(plan.keys()):
        for kind in list(plan[b].keys()):
            plan[b][kind] = sorted(set(plan[b][kind]))
    report = {
        "buckets": {},
        "shared": {k: sorted(v) for k, v in sorted(shared.items())},
        "unresolved_results": sorted(unresolved),
        "notes": notes,
    }
    total = 0
    for b in BUCKETS + ["legacy"]:
        d = plan.get(b, {})
        entry = {k: sorted(v) for k, v in sorted(d.items())}
        n = sum(len(v) for v in entry.values())
        total += n
        report["buckets"][b] = {"count": n, **{k: len(v) for k, v in entry.items()}}
    report["_detail"] = {b: {k: sorted(v) for k, v in sorted(plan.get(b, {}).items())}
                         for b in BUCKETS + ["legacy"]}
    report["_assign"] = {k: sorted(v) for k, v in sorted(assign.items())}
    report["_total_planned"] = total

    # ---- 6. 守恒校验：源树里每个文件都必须恰好出现在计划中
    on_disk = set()
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d != "progress"]
        for fn in files:
            on_disk.add(os.path.relpath(os.path.join(root, fn), SRC))
    planned = set()
    dup = []
    for b in BUCKETS + ["legacy"]:
        for kind, rels in plan.get(b, {}).items():
            for rel in rels:
                planned.add(rel)
    for rel in on_disk:
        n = sum(1 for b in BUCKETS + ["legacy"]
                for kind, rels in plan.get(b, {}).items() if rel in rels)
        if n == 0:
            notes.append("!! 未计划（会丢）: %s" % rel)
        elif n > 1 and rel not in shared:
            dup.append(rel)
    ghost = sorted(planned - on_disk)
    if ghost:
        notes.append("!! 计划中存在但磁盘上没有: %s" % ghost)
    if dup:
        notes.append("!! 被多桶计划但未标记共享: %s" % dup)
    report["_on_disk"] = len(on_disk)
    report["_unique_planned"] = len(planned)
    print("源树文件总数: %d ；计划覆盖唯一文件: %d ；多桶共享文件: %d"
          % (len(on_disk), len(planned), len(shared)))

    with open(OUT_PLAN, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("planned entries:", total)
    for b in BUCKETS + ["legacy"]:
        e = report["buckets"][b]
        print("  %-26s %4d  %s" % (b, e["count"],
                                   {k: v for k, v in e.items() if k != "count"}))
    print("shared files (%d):" % len(report["shared"]))
    for k, v in report["shared"].items():
        print("   ", k, v)
    print("unresolved results (%d): %s" % (len(unresolved), sorted(unresolved)))
    for n in notes:
        print(n)


if __name__ == "__main__":
    sys.exit(main())
