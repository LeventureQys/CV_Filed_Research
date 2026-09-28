# -*- coding: utf-8 -*-
"""T5-B / 01：① C++ ↔ 原型检测器常量独立复核  ② 仪表化/补丁的 A/B 零差证明。

产物
    results/t5b_consts_check.csv    —— 逐条常量 diff（13 个检测器常量 + 全量扩展）
    results/t5b_patch_ab_zero.csv   —— 补丁不改变默认行为的 A/B 零差证明
    results/_t5b_01.log
只读 `src/domain/drift_v6/drift_v6_compensator.{h,cpp}`，不编译、不改源码。
"""
import os
import re
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t5b_glm53_v6 import GLM53v6                              # noqa: E402
from t5b_core import (TraceV6, make_traced_v6, run_full, run_plain,  # noqa: E402
                      load_window, perturb_window, WIN, DT)

H_SRC = os.path.join(ROOT, "src", "domain", "drift_v6", "drift_v6_compensator.h")
CPP_SRC = os.path.join(ROOT, "src", "domain", "drift_v6", "drift_v6_compensator.cpp")

# ① 需求文档/第一轮口径的 **13 个检测器常量**（逐条 diff 的对象）
DET13 = [("kDetFastS", "DET_FAST"), ("kDetGapS", "DET_GAP"), ("kDetLagS", "DET_LAG"),
         ("kDetK", "DET_K"), ("kDetRel", "DET_REL"), ("kDetAbsFrac", "DET_ABS_FRAC"),
         ("kDetIdleFrac", "DET_IDLE_FRAC"), ("kDetPersist", "DET_PERSIST"),
         ("kIdleSettleS", "IDLE_SETTLE"), ("kUnloadBlockS", "UNLOAD_BLOCK"),
         ("kBackdateS", "BACKDATE_S"), ("kTailGateS", "TAIL_GATE_S"),
         ("kTailGateFrac", "TAIL_GATE_FRAC")]
# ② 扩展：原型里全部可对拍的常量（检测器之外的也算，作为"完整 diff"证据）
EXT = [("kTauTotalS", "TAU_TOTAL"), ("kTauLevelS", "TAU_LEVEL"),
       ("kTauRef", "TAU_REF"), ("kAWin", "AWIN"),
       ("kKappaOnset", "KAPPA_ONSET"), ("kKappaRestep", "KAPPA_RESTEP"),
       ("kGlideMinS", "GLIDE_MIN"), ("kGlideMaxS", "GLIDE_MAX"), ("kRateMax", "RATE_MAX"),
       ("kHoMinS", "HO_MIN"), ("kRevokeS", "REVOKE"),
       ("kRevokeCooldownS", "REVOKE_COOLDOWN"), ("kDecreaseSettleS", "DECREASE_SETTLE"),
       ("kReanchorSmoothS", "REANCHOR_SMOOTH_S"),
       ("kStallStartS", "STALL_START_S"), ("kStallTailFrac", "STALL_TAIL_FRAC"),
       ("kStallHoldS", "STALL_HOLD_S"), ("kStallMinFrac", "STALL_MIN_FRAC"),
       ("kTrimRate", "TRIM_RATE"), ("kTrimDeadFrac", "TRIM_DEAD_FRAC"),
       ("kTauG", "TAU_G"), ("kLoadedFrac", "LOADED_FRAC"),
       ("kGammaMin", "GAMMA_MIN"), ("kGammaMax", "GAMMA_MAX"),
       ("kCreepLoFrac", "CREEP_LO"), ("kCreepHiFrac", "CREEP_HI"),
       ("kGEnable", "G_ENABLE"), ("kIdleFrac", "IDLE_FRAC"),
       ("kUnloadMinRatio", "UNLOAD_MIN_RATIO"), ("kUnloadFastS", "UNLOAD_FAST"),
       ("kCap", "CAP"), ("kDcap", "DCAP")]


def grab(src, name):
    m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*=\s*([-+]?[0-9]*\.?[0-9]+)", src)
    return float(m.group(1)) if m else np.nan


def consts_check():
    h = open(H_SRC, encoding="utf-8", errors="ignore").read()
    rows = []
    for tag, pairs in (("det13", DET13), ("ext", EXT)):
        for cpp_k, pr_k in pairs:
            cv = grab(h, cpp_k)
            pv = getattr(GLM53v6, pr_k, np.nan)
            rows.append(dict(group=tag, cpp_const=cpp_k, cpp_value=cv,
                             proto_attr=pr_k, proto_value=float(pv),
                             same=bool(abs(cv - float(pv)) < 1e-12),
                             note=""))
    df = pd.DataFrame(rows)
    df.loc[len(df)] = dict(group="note", cpp_const="Push()", cpp_value=np.nan,
                           proto_attr="_push()", proto_value=np.nan, same=None, note="")
    # C++ Push 写"上一帧的 3 帧中值"（原型存当帧），等价性检查
    cpp = open(CPP_SRC, encoding="utf-8", errors="ignore").read()
    lag = "buf_v_[i] = prev_med_total_;" in cpp
    df.loc[len(df) - 1] = dict(group="note", cpp_const="Push()滞后", cpp_value=1.0,
                               proto_attr="1 帧(≈10 ms)", proto_value=1.0, same=True,
                               note=("C++ `buf_v_[i] = prev_med_total_` 命中=" + str(lag) +
                                     "；原型 `_push` 存当帧 3 帧中值 ⇒ C++ 多 1 帧 ≈10 ms 滞后"))
    df.to_csv(os.path.join(RES, "t5b_consts_check.csv"), index=False, encoding="utf-8-sig")
    d13 = df[df["group"] == "det13"]
    print(f"[常量复核] 检测器 13 条：相同 {int(d13['same'].sum())}/13；"
          f"扩展 {len(df[df['group'] == 'ext'])} 条：相同 "
          f"{int(df[df['group'] == 'ext']['same'].sum())}/{len(df[df['group'] == 'ext'])}")
    bad = df[(df["same"] == False)]                                     # noqa: E712
    if len(bad):
        print("  不一致条目：")
        print(bad.to_string())
    print(df[df["group"] == "det13"].to_string())
    return df


CAPF_PROBE = 1.0        # 自检里用的封顶系数（= 第一轮被测伪的那一档）


def maxrun(mask):
    """最长连续 True 帧数。"""
    best = cur = 0
    for v in np.asarray(mask, bool):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def ab_zero():
    """补丁自检的**灵敏度证明**：一次自检必须同时给出两类对照，否则没有信息量。

    A 类（应当零差）：默认参数下不改变行为
        A1 `plain` vs `TraceV6(CAPF=None)`            —— 仪表化本身零差
        A2 `plain` vs `TraceV6(DET_PERSIST=3,DET_K=5.0)` —— 显式写原值的参数化零差
        A3 `plain` vs `TraceV6(CAPF=1.0)` **在干净窗上** —— 见下：这一行**不能**用作"补丁已接线"的证据，
           因为干净数据上 `5σ_d` 远小于电平项，封顶**永不触发**（本表给出 `capf_bind_frac_clean=0` 的数值证据）
    B 类（**必须非零**，否则自检本身没有检出能力）：
        B1 `plain` vs `TraceV6(DET_PERSIST=1)`      —— 参数生效性（放宽连续帧数）
        B2 `plain` vs `TraceV6(DET_REL=0.01)`       —— 参数生效性（放宽 5% 电平门槛 → 让门限真的降下来）
        B3 `CAPF=None` vs `CAPF=1.0` **在 白噪2000 窗上** —— 补丁生效性（封顶真的触发）

    判定：所有 A 类 `expect_change=False` 的行必须 diff==0；所有 B 类 `expect_change=True` 的行必须 diff>0。
    """
    rows = []
    for wkey in ("W1_1w", "W2_3w"):
        w = WIN[wkey]
        tu, Xu, w = load_window(w)
        lvl = float(np.median(Xu.sum(axis=1)))
        t0 = time.time()
        Yp, cp = run_plain(tu, Xu, GLM53v6)
        t1 = time.time()
        r = run_full(tu, Xu, TraceV6)
        t2 = time.time()
        nep_p = len(cp.epoch_t)

        # —— 干净窗上"为什么 CAPF 不改变行为"的**可证**诊断 ——
        #    只看 thr_d 会被下游门限掩盖（实测 W2 干净窗有 20.3% 帧 thr_d 被改，输出仍然零差），
        #    因此这里用**完整判定链**的有效门限：
        #      g_eff0 = max(5σ_d, level_term)                                   ；若 state_pre==idle 且 d>0
        #               → max(g_eff0, DET_IDLE_FRAC·max_tot)                     （空闲态额外门限）
        #      g_eff1 = 同上但 5σ_d → min(5σ_d, CAPF·level_term)
        #    行为改变的必要条件：|d| 落在 (g_eff1, g_eff0] 这条"新开带"内、且**连续 ≥DET_PERSIST 帧**。
        trc = r["tr"]
        lv = np.asarray(trc["lv_ref"], float)
        mtd = np.asarray(trc["max_tot"], float)
        dd = np.asarray(trc["d"], float)
        sp = np.asarray(trc["state_pre"], int)
        level_term = np.maximum(GLM53v6.DET_REL * np.abs(np.nan_to_num(lv)),
                                GLM53v6.DET_ABS_FRAC * mtd)
        five = 5.0 * np.asarray(trc["sig_d"], float)
        idle_extra = GLM53v6.DET_IDLE_FRAC * mtd
        is_idle_pos = (sp == 0) & (dd > 0)
        g0 = np.maximum(five, level_term)
        g1 = np.maximum(np.minimum(five, CAPF_PROBE * level_term), level_term)
        g0_raw, g1_raw = g0.copy(), g1.copy()
        g0 = np.where(is_idle_pos, np.maximum(g0, idle_extra), g0)
        g1 = np.where(is_idle_pos, np.maximum(g1, idle_extra), g1)
        band = (np.abs(dd) > g1) & (np.abs(dd) <= g0)
        run = maxrun(band)
        band_raw = (np.abs(dd) > g1_raw) & (np.abs(dd) <= g0_raw)
        run_raw = maxrun(band_raw)

        def rec(variant, Yref, Yvar, run_ref, run_var, expect_change, control, note="", plain=None):
            d = float(np.abs(np.asarray(Yref) - np.asarray(Yvar)).max())
            # —— 内部状态级差异（比输出级灵敏得多：输出零差 ≠ 补丁没接线）——
            if run_ref is not None and run_var is not None and run_ref is not run_var:
                n_gate = int(np.sum(np.abs(run_ref["tr"]["gate"] - run_var["tr"]["gate"]) > 1e-9))
                n_sig = int(np.sum(np.abs(run_ref["tr"]["sig_d"] - run_var["tr"]["sig_d"]) > 1e-12))
                n_hit = int(np.sum(run_ref["tr"]["raw_hit"] != run_var["tr"]["raw_hit"]))
            else:
                n_gate = n_sig = n_hit = 0
            ep_ref = len(run_ref["epoch"]) if run_ref is not None else np.nan
            ep_var = len(run_var["epoch"]) if run_var is not None else np.nan
            ep_diff = int(ep_ref != ep_var)
            # —— 与**原型**(plain) 的内部对拍：锚点向量 + epoch 起点（避免"自比"式的空洞零差）——
            n_ep_plain = dA_sum = t0_diff = np.nan
            if plain is not None and run_var is not None:
                n_ep_plain = len(plain.epoch_t)
                dA_sum = float(np.abs(np.asarray(plain.A, float) -
                                      np.asarray(run_var["c"].A, float)).max())
                tp = np.sort(np.asarray(plain.epoch_t, float))
                tv = np.sort(np.asarray([e[0] for e in run_var["epoch"]], float))
                t0_diff = float(np.abs(tp - tv).max()) if len(tp) == len(tv) and len(tp) else \
                    (0.0 if len(tp) == len(tv) else np.inf)
            internal = int(n_gate + n_sig + n_hit + ep_diff) > 0
            plain_int_ok = bool((not np.isfinite(dA_sum)) or (dA_sum == 0.0))
            plain_int_ok = bool(plain_int_ok and ((not np.isfinite(t0_diff)) or (t0_diff == 0.0)))
            rows.append(dict(win=wkey, n_frames=len(tu), variant=variant,
                             max_abs_Y_diff=d, zero=bool(d == 0.0),
                             expect_change=bool(expect_change), control=control,
                             ok=bool((d > 0) == expect_change),
                             n_frames_gate_diff=n_gate, n_frames_sig_diff=n_sig,
                             n_frames_rawhit_diff=n_hit,
                             n_epoch_ref=ep_ref, n_epoch_var=ep_var,
                             n_epoch_plain=n_ep_plain,
                             plain_max_abs_sumA_diff=round(dA_sum, 6) if dA_sum == dA_sum else np.nan,
                             plain_epoch_t0_maxdiff=round(t0_diff, 6) if t0_diff == t0_diff else np.nan,
                             plain_internal_ok=plain_int_ok,
                             internal_changed=bool(internal),
                             internal_ok=bool(internal if expect_change else (not internal)),
                             level=lvl, level_term_med=round(float(np.median(level_term)), 1),
                             idle_extra_med=round(float(np.median(idle_extra)), 1),
                             max_5sig_clean=round(float(five.max()), 1),
                             capf_bind_frac_clean=round(float(np.mean(five > level_term)), 4),
                             gate_raw_change_frac=round(float(np.mean(g1_raw < g0_raw - 1e-9)), 4),
                             rawband_maxrun_frames=int(run_raw),
                             gate_eff_change_frac=round(float(np.mean(g1 < g0 - 1e-9)), 4),
                             newband_maxrun_frames=int(run),
                             newband_can_flip=bool(run >= GLM53v6.DET_PERSIST),
                             note=note))
            return d

        d1 = rec("A1 plain vs TraceV6(CAPF=None)", Yp, r["Y"], r, r,
                 False, "default_noop", "仪表化零差：输出 + 与原型逐通道 ΣA/epoch 起点对拍", plain=cp)
        r2 = run_full(tu, Xu, TraceV6, DET_PERSIST=3, DET_K=5.0)
        d2 = rec("A2 plain vs TraceV6(显式 DET_PERSIST=3,DET_K=5.0)", Yp, r2["Y"], r, r2,
                 False, "param_noop", "参数化写原值", plain=cp)
        r3 = run_full(tu, Xu, make_traced_v6(capf=1.0))
        d3 = rec("A3 CAPF=1.0 干净窗（vs TraceV6 默认）", r["Y"], r3["Y"], r, r3,
                 False, "capf_clean_output_zero_internal_may_change",
                 "**既不能当接线证据、也不能当零差证据**：输出恒为 0；"
                 "故本表另给 n_frames_* 内部差列（n_frames_rawhit_diff>0 ⇒ 补丁已接线但输出未被影响）")
        # B 类：故意放宽的参数（必须改变行为，证明自检有检出能力）
        r4 = run_full(tu, Xu, TraceV6, DET_PERSIST=1)
        d4 = rec("B1 plain vs TraceV6(DET_PERSIST=1) [故意放宽]", Yp, r4["Y"], r, r4,
                 True, "sensitivity_param", "连续帧 3→1（必须非零）", plain=cp)
        r5 = run_full(tu, Xu, TraceV6, **{"DET_REL": 0.01})
        d5 = rec("B2 plain vs TraceV6(DET_REL=0.01) [故意放宽]", Yp, r5["Y"], r, r5,
                 True, "sensitivity_param", "5% 电平门槛→1%（必须非零）", plain=cp)
        Xn, _ = perturb_window(tu, Xu, "white", 2000.0, 0)
        rn = run_full(tu, Xn, TraceV6)
        r6 = run_full(tu, Xn, make_traced_v6(capf=1.0))
        d6 = rec("B3 白噪2000: CAPF=None vs CAPF=1.0 [补丁生效性]", rn["Y"], r6["Y"], rn, r6,
                 True, "sensitivity_patch", "封顶在噪声下真的触发（必须非零）")
        Yp2, cp2 = run_plain(tu, Xn, GLM53v6)
        r7 = run_full(tu, Xn, TraceV6)
        d7 = rec("B3b 白噪2000: plain vs TraceV6(CAPF=None)", Yp2, r7["Y"], r7, r7,
                 False, "default_noop_noise", "带噪窗上仪表化零差", plain=cp2)
        print(f"[A/B] {wkey} 帧={len(tu)} 电平={lvl:.0f} 电平项中位={np.median(level_term):.1f} "
              f"空闲态额外门限中位={np.median(idle_extra):.1f} 干净 max(5σ_d)={five.max():.1f}")
        print(f"      仅 thr_d 看：被改帧占比={np.mean(g1_raw < g0_raw - 1e-9):.4f} "
              f"新开带最长连续={run_raw} 帧；"
              f"**完整判定链**看：被改帧占比={np.mean(g1 < g0 - 1e-9):.4f} "
              f"新开带最长连续={run} 帧（≥{GLM53v6.DET_PERSIST} 帧才可能改变判定 ⇒ "
              f"can_flip={run >= GLM53v6.DET_PERSIST}）")
        print(f"      A1={d1:.3e} A2={d2:.3e} A3(CAPF干净)={d3:.3e} | "
              f"B1(P=1)={d4:.3e} B2(DET_REL=.01)={d5:.3e} B3(CAPF噪声)={d6:.3e} B3b={d7:.3e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t5b_patch_ab_zero.csv"), index=False, encoding="utf-8-sig")
    a_ok = bool(df[~df["expect_change"]]["zero"].all())
    b_ok = bool((df[df["expect_change"]]["max_abs_Y_diff"] > 0).all())
    a_int = bool(df[df["control"].isin(["default_noop", "param_noop",
                                        "default_noop_noise"])]["internal_ok"].all())
    a_plain = bool(df[df["control"].isin(["default_noop", "param_noop",
                                          "default_noop_noise"])]["plain_internal_ok"].all())
    b_int = bool(df[df["expect_change"]]["internal_ok"].all())
    print(f"\n[结论-A] 默认/参数化对照（应当**输出与内部双零差**）：{a_ok and a_int}"
          f"（输出零差={a_ok}，内部零差={a_int}；**与原型**的 ΣA/epoch 起点对拍亦零差={a_plain}）")
    a3 = df[df["control"] == "capf_clean_output_zero_internal_may_change"]
    print(f"[结论-A3] CAPF 干净窗：**既不是接线证据、也不是零差证据** —— "
          f"输出恒为 0（{len(a3)} 行），但内部是否变化随窗口而异："
          f"{dict(zip(a3['win'], a3['internal_changed']))}；"
          f"读法见 n_frames_gate_diff / n_frames_rawhit_diff / n_epoch_* 列")
    print(f"[结论-B] 故意放宽的参数与补丁（应当**输出与内部双非零**）：{b_ok and b_int}"
          f"（输出非零={b_ok}，内部非零={b_int}）⇒ 本自检**具备检出能力**")
    print(df[["win", "variant", "max_abs_Y_diff", "expect_change", "control", "ok",
              "n_frames_gate_diff", "n_frames_rawhit_diff", "n_epoch_ref", "n_epoch_var",
              "plain_max_abs_sumA_diff", "plain_epoch_t0_maxdiff", "internal_ok"]]
          .to_string(index=False))
    return df


if __name__ == "__main__":
    print("=== ① C++ ↔ 原型常量独立复核 ===")
    consts_check()
    print("\n=== ② A/B 零差证明 ===")
    ab_zero()
