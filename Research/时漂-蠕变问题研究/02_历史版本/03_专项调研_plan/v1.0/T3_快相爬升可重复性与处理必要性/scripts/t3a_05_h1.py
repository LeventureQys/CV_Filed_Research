# -*- coding: utf-8 -*-
"""T3-A 05：分族/分幅度分层表（Q3 的补充层）+ H1「快相形状稳定」三态裁决表。

H1（需求文档 §1 的用户假设）：快相爬升的形状「总是差不多稳定的」，所以可以把它当作已知形状用掉。
本任务对 H1 的裁决**分准则**给出，避免把「同类内稳定」与「跨类可用」混为一谈：

  C1 同类内可重复性（onset / restep 各自）：用主网格 0.3/0.5/1.0 s 的 CQV 与 spread(p90-p10) 判定
      支持      ：全部 CQV <= 0.15 且 spread <= 0.20
      反驳      ：存在 CQV > 0.40 或 spread > 0.35
      有条件支持：介于两者之间
  C2 跨类（跨形态）可移植性 —— **硬门**：用 onset 库反演 restep（与反向）的 τ_d=0.2 s 中位|误差|
      > 20%  ⇒ 整体不得为「支持」
  C3 形状库泛化（同类内留一）—— 软门：usable 全样本 τ_d=0.5 s 的 LOO 中位|误差| <= 5% 为佳
  C4 稳定 ≠ 正确（08-v6.1）：本任务的库由本批事件构造，不含 ROM 偏慢项 ⇒ 只作方向性标注，不参与裁决

整体三态 = C1（取较差的类）× C2 硬门 × C3 软门。

产物：results/t3a_shape_spread_fam.csv、results/t3a_h1_verdict.csv、results/_t3a_05_h1.log
运行：python scripts/t3a_05_h1.py   （秒级）
"""
import json
import os

import numpy as np
import pandas as pd

import t3a_common as C

GRID = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00, 5.00]
KEY_TAU = [0.30, 0.50, 1.00]          # 裁决用的关键 τ（0.2 s 以下处时间轴畸变区，单列）
C1_SUP_CQV, C1_SUP_SPREAD = 0.15, 0.20
C1_REF_CQV, C1_REF_SPREAD = 0.40, 0.35
C2_HARD_GATE = 20.0                    # 跨形态 τ_d=0.2 s 中位|误差| 的硬门（%）


def qstats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    med, p10, p90, n = C.band(v)
    if n >= 4:
        p75, p25 = np.percentile(v, 75), np.percentile(v, 25)
        cqv = (p75 - p25) / (p75 + p25) if abs(p75 + p25) > 1e-9 else np.nan
        cv, mu, sd, _ = C.cv_of(v)
    else:
        cqv, cv, mu, sd = np.nan, np.nan, (float(np.mean(v)) if n else np.nan), np.nan
    return dict(n=n, med=med, p10=p10, p90=p90, p25=(float(np.percentile(v, 25)) if n >= 4 else np.nan),
                p75=(float(np.percentile(v, 75)) if n >= 4 else np.nan),
                spread=p90 - p10, cqv=cqv, cv=cv, mean=mu, std=sd,
                lo=(float(np.min(v)) if n else np.nan), hi=(float(np.max(v)) if n else np.nan),
                rng=(float(np.max(v) - np.min(v)) if n else np.nan))


def main():
    C.start_log("t3a_05_h1")
    tra = pd.read_csv(os.path.join(C.RES, "t3a_shape_trajectory.csv"))
    aud = pd.read_csv(os.path.join(C.RES, "t3a_inlier_audit.csv"))
    gen = pd.read_csv(os.path.join(C.RES, "t3a_generalization.csv"))
    pcv = pd.read_csv(os.path.join(C.RES, "t3a_param_cv.csv"))
    ts = pd.read_csv(os.path.join(C.RES, "t3a_time_stability.csv"))
    tests = pd.read_csv(os.path.join(C.RES, "t3a_strata_tests.csv"))
    us = set(aud[aud["usable"]]["uid"])
    al = set(aud[aud["is_load"]]["uid"])

    # ── 分族 / 分幅度 的形状散布（Q3 补充层） ──
    rows = []
    L = aud[aud["is_load"]].copy()
    L["magbin"] = pd.cut(L["J_over_peak"].abs(), [0, 0.05, 0.15, 0.40, 1e9],
                         labels=["J/peak<5%", "5~15%", "15~40%", ">=40%"], right=False)
    for sample, uids in [("usable", us), ("all", al)]:
        for kind in ["onset", "restep"]:
            sub = L[(L["uid"].isin(uids)) & (L["kind"] == kind)]
            for dim in ["fam", "magbin"]:
                for name, g in sub.groupby(dim, observed=True):
                    t = tra[tra["uid"].isin(set(g["uid"]))]
                    for tau in GRID:
                        v = t[np.isclose(t["tau"], tau)]["f"].to_numpy(float)
                        r = qstats(v)
                        r.update(sample=sample, kind=kind, dim=dim, layer=str(name),
                                 tau=tau, n_events=int(g["uid"].nunique()))
                        rows.append(r)
    fam = pd.DataFrame(rows)
    fam = fam[["sample", "kind", "dim", "layer", "tau", "n", "n_events", "mean", "std", "cv",
               "med", "p10", "p90", "p25", "p75", "spread", "cqv", "lo", "hi", "rng"]]
    fam.to_csv(os.path.join(C.RES, "t3a_shape_spread_fam.csv"), index=False, encoding="utf-8-sig")
    print("分族/分幅度形状散布 -> results/t3a_shape_spread_fam.csv（%d 行）" % len(fam))
    for sample in ["usable", "all"]:
        for kind in ["onset", "restep"]:
            for dim in ["fam", "magbin"]:
                s = fam[(fam["sample"] == sample) & (fam["kind"] == kind) & (fam["dim"] == dim)
                        & np.isclose(fam["tau"], 0.30)]
                if s.empty:
                    continue
                print("  %s/%s/%.0fs(%s): " % (sample, kind, 0.30, dim)
                      + " | ".join("%s n=%d 中位%.3f p10~p90 %.3f~%.3f CQV=%s"
                                   % (r["layer"], int(r["n_events"]), r["med"], r["p10"], r["p90"],
                                      ("%.3f" % r["cqv"]) if np.isfinite(r["cqv"]) else "n/a")
                                   for _, r in s.iterrows()))
    print("")

    # ── H1 裁决 ──
    def sp_get(kind, tau, col):
        """usable 样本、kind 类的整体统计（不按 dim 分）—— 用 t3a_shape_spread.csv。"""
        d = pd.read_csv(os.path.join(C.RES, "t3a_shape_spread.csv"))
        r = d[(d["kind"] == kind) & (d["sample"] == "usable") & np.isclose(d["tau"], tau)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def gg(scen, kind, tau_d, col, sample="usable", inv="single@tau", ref="post"):
        r = gen[(gen["sample"] == sample) & (gen["scenario"] == scen) & (gen["kind"] == kind)
                & np.isclose(gen["tau_d"], tau_d) & (gen["inv"] == inv) & (gen["ref"] == ref)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    # C1：分 kind
    c1 = {}
    for kind in ["onset", "restep"]:
        cqv = max(sp_get(kind, t, "cqv") for t in KEY_TAU if np.isfinite(sp_get(kind, t, "cqv")))
        spr = max(sp_get(kind, t, "spread") for t in KEY_TAU)
        if cqv <= C1_SUP_CQV and spr <= C1_SUP_SPREAD:
            v = "支持"
        elif cqv > C1_REF_CQV or spr > C1_REF_SPREAD:
            v = "反驳"
        else:
            v = "有条件支持"
        c1[kind] = dict(cqv=cqv, spread=spr, verdict=v,
                        n=int(sp_get(kind, 0.30, "n")),
                        med30=sp_get(kind, 0.30, "med"), med10=sp_get(kind, 1.00, "med"))
    # C2：跨形态（硬门）
    c2 = dict(err_onset_to_restep=gg("cross_kind", "restep", 0.20, "med_abs_pct"),
              err_restep_to_onset=gg("cross_kind", "onset", 0.20, "med_abs_pct"),
              gate=C2_HARD_GATE)
    c2["pass"] = bool(max(c2["err_onset_to_restep"], c2["err_restep_to_onset"]) <= C2_HARD_GATE)
    # C3：同类 LOO
    c3 = dict(onset_loo_05=gg("loo_event", "onset", 0.50, "med_abs_pct"),
              restep_loo_05=gg("loo_event", "restep", 0.50, "med_abs_pct"),
              onset_loo_02=gg("loo_event", "onset", 0.20, "med_abs_pct"),
              restep_loo_02=gg("loo_event", "restep", 0.20, "med_abs_pct"))
    c3["ok_onset"] = bool(c3["onset_loo_05"] <= 5.0)
    c3["ok_restep"] = bool(c3["restep_loo_05"] <= 5.0)
    # 综合
    worst_c1 = "反驳" if any(d["verdict"] == "反驳" for d in c1.values()) else (
        "有条件支持" if any(d["verdict"] == "有条件支持" for d in c1.values()) else "支持")
    if not c2["pass"] and worst_c1 == "支持":
        overall = "有条件支持"
        why = ("同类内稳定（C1 支持）但**跨形态不可移植**（C2 硬门不通过：拿 onset 库反演 restep 的 "
               "τ_d=0.2 s 中位|误差| %.1f%%）⇒ 只能有条件支持" % c2["err_onset_to_restep"])
    elif not c2["pass"] and worst_c1 != "支持":
        overall = "反驳"
        why = ("①同类内 restep 自身不稳定（C1 %s）；②跨形态不可移植（C2：onset→restep τ_d=0.2 s 中位|误差|"
               " %.1f%%）⇒ 作为普遍假设被反驳，仅在 onset+硬阶跃+τ>=0.3 s 的子域内成立"
               % (worst_c1, c2["err_onset_to_restep"]))
    else:
        overall = worst_c1
        why = "C2 硬门通过，按 C1 的较差类取裁决"

    vrows = []
    for kind in ["onset", "restep"]:
        vrows.append(dict(criterion="C1 同类内可重复性", scope=kind, value=(
            "CQV<=%.3f、spread<=%.3f（τ=0.3/0.5/1.0 s 取最差）；f(0.3 s) 中位 %.3f、f(1.0 s) 中位 %.3f；n=%d"
            % (c1[kind]["cqv"], c1[kind]["spread"], c1[kind]["med30"], c1[kind]["med10"],
               c1[kind]["n"])),
            three_state=c1[kind]["verdict"],
            note="阈值：支持 CQV<=%.2f 且 spread<=%.2f；反驳 CQV>%.2f 或 spread>%.2f"
                 % (C1_SUP_CQV, C1_SUP_SPREAD, C1_REF_CQV, C1_REF_SPREAD)))
    vrows.append(dict(criterion="C2 跨形态可移植性（硬门）", scope="onset<->restep",
                      value="onset 库→restep：τ_d=0.2 s 中位|误差| %.1f%%；restep 库→onset：%.1f%%；门限 %.0f%%"
                            % (c2["err_onset_to_restep"], c2["err_restep_to_onset"], C2_HARD_GATE),
                      three_state=("通过" if c2["pass"] else "不通过"),
                      note="跨形态 = 用另一类事件的形状库反演本事件的最终电平（usable 样本）"))
    vrows.append(dict(criterion="C3 同类内泛化（留一事件）", scope="onset / restep",
                      value="τ_d=0.5 s 中位|误差| onset %.2f%%、restep %.2f%%；τ_d=0.2 s：%.2f%% / %.2f%%"
                            % (c3["onset_loo_05"], c3["restep_loo_05"],
                               c3["onset_loo_02"], c3["restep_loo_02"]),
                      three_state=("支持" if (c3["ok_onset"] and c3["ok_restep"]) else
                                   ("有条件支持" if c3["ok_onset"] else "反驳")),
                      note="软门 5%（τ_d=0.5 s）；onset 单独看通过"))
    vrows.append(dict(criterion="C4 稳定 vs 正确（不参与裁决）", scope="—",
                      value="本任务形状库由本批事件自身构造，不含 08-v6.1 的 ROM 偏慢项；"
                            "故只能标注「稳定≠正确」，方向性结论引用 08-v6.1（onset 中位 +3.2%）",
                      three_state="不适用", note="需求文档 §5-3 要求分开报"))
    vrows.append(dict(criterion="H1 整体裁决", scope="H1「快相形状总是差不多稳定」",
                      value="worst(C1)=%s；C2 硬门=%s；C3 onset=%s"
                            % (worst_c1, "通过" if c2["pass"] else "不通过",
                               "通过" if c3["ok_onset"] else "不通过"),
                      three_state=overall, note=why))
    vd = pd.DataFrame(vrows)
    vd.to_csv(os.path.join(C.RES, "t3a_h1_verdict.csv"), index=False, encoding="utf-8-sig")

    print("══ H1 三态裁决 ══")
    for _, r in vd.iterrows():
        print("  [%s] %s\n     value  : %s\n     state  : %s\n     note   : %s"
              % (r["criterion"], r["scope"], r["value"], r["three_state"], r["note"]))
    print("\n-> results/t3a_h1_verdict.csv（%d 行）\n" % len(vd))

    # 供 JSON 交接件直接读的摘要
    summ = dict(c1=c1, c2=c2, c3=c3, overall=overall, why=why)
    with open(os.path.join(C.RES, "t3a_h1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print("-> results/t3a_h1_summary.json")


if __name__ == "__main__":
    main()
