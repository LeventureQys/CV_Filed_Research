# -*- coding: utf-8 -*-
"""T6-C：归因（消融表 T6-Q4）+ 补丁零差证明（t6_patch_ab_zero.csv）。

方法：
  ① **补丁零差证明**：每个补丁类把旋钮设回原型默认值后在参考录制上跑一遍，
     与「未打补丁的仪表化原型」逐帧比对，必须 **完全为 0**；
  ② **消融表**：在 9 份恒载（截到 80 s，覆盖 onset + 平台 40–60 s 窗）上，
     基线 v6 与 9 条"退回去"的臂各跑一遍，报 L2 组内离散度（组内 std，3 组取中位）
     与其余指标，给出每条改动对可重复性的 Δ；
  ③ **起扣时刻实测**：量化"v6 的扣除起点比 v5.1 早多少秒"（用于校核"提前约 4.5 s"）。

真值口径两套同报（R5 = pre+inc5；Rstep = pre+a_step）——排名会反转，必须并列。
T_stable 用 legacy 口径（tol 5%·step / hold 30 s）；**T1-A 未交付，不可与 T1 绝对横比**。

产出：results/t6_patch_ab_zero.csv、t6_attribution.csv、t6_attribution_records.csv、
      t6_deduct_onset.csv、_t6c_attribution.log
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import t6_common as C                                              # noqa: E402
import t6_ad_lib as AL                                             # noqa: E402
import t6_a_common as AC                                            # noqa: E402
import t6_patches as PA                                            # noqa: E402
from t6_glm53_v51 import GLM53v51                                  # noqa: E402

RES = C.RES
TMAX = 80.0                       # 恒载统一截到 80 s（覆盖 t_on≤14.9 s + 平台 40–60 s 窗）
LOG = []
TRACED_V51 = AL.make_traced(GLM53v51)


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def run_traced_arm(Cls, tu, Xu, kw=None):
    r = AC.run_traced(Cls, tu, Xu, **(kw or {}))
    return r["Y"], r


def run_v51(tu, Xu):
    c = TRACED_V51(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def main():
    # ═══════════ ① 零差证明 ═══════════
    ref_tag = "四指指尖/数据1"
    d = C.grid(C.HOLD[ref_tag], tmax=TMAX)
    tu, Xu = d["tu"], d["Xu"]
    Y0, _ = run_traced_arm(PA.BASE, tu, Xu)
    zero = []
    for name, Cls, dflt in PA.ZERO_AB:
        Y1, _ = run_traced_arm(Cls, tu, Xu, dflt)
        dd = np.abs(Y1 - Y0)
        zero.append(dict(arm=name, ref_tag=ref_tag, n_frames=len(tu),
                         max_abs_diff=float(dd.max()), rms_diff=float(np.sqrt(np.mean(dd ** 2))),
                         bit_identical=bool(dd.max() == 0.0)))
        p(f"  patch A/B zero: {name:>15} max|Δ|={dd.max():.3e}  identical={dd.max() == 0.0}")
    z = pd.DataFrame(zero)
    z.to_csv(os.path.join(RES, "t6_patch_ab_zero.csv"), index=False, encoding="utf-8-sig")
    assert z.bit_identical.all(), "补丁默认值下未做到零差，消融归因不成立"
    p("  → 全部补丁在默认值下与原型逐帧完全一致（bit-identical）")

    # ═══════════ ③ 起扣时刻实测（先做，便宜） ═══════════
    ded = []
    for tag in C.HOLD_TAGS:
        dd_ = C.grid(C.HOLD[tag], tmax=TMAX)
        tu2, Xu2, dt2 = dd_["tu"], dd_["Xu"], dd_["dt"]
        dom = C.onset_geometry(tu2, Xu2.sum(axis=1), dt2)
        i_e = dom["i_e"]
        raw = Xu2.sum(axis=1)
        thr = 0.01 * abs(dom["a_step"])
        rec = dict(tag=tag, group=C.GROUP[tag.split("/")[0]], t_e=float(tu2[i_e]),
                   a_step=dom["a_step"])
        for name, Cls, kw in (("v5.1", None, None), ("v6", PA.BASE, None)):
            Y = run_v51(tu2, Xu2)[0] if Cls is None else run_traced_arm(Cls, tu2, Xu2)[0]
            dv = np.abs(Y.sum(axis=1) - raw)
            w = np.where((tu2 >= tu2[i_e]) & (dv > thr))[0]
            rec[f"t_deduct_{name}"] = float(tu2[w[0]] - tu2[i_e]) if len(w) else np.nan
        rec["lead_v6_vs_v51_s"] = rec["t_deduct_v5.1"] - rec["t_deduct_v6"]
        ded.append(rec)
        p(f"  deduct onset {tag:>18} v5.1={rec['t_deduct_v5.1']:.2f}s "
          f"v6={rec['t_deduct_v6']:.2f}s → v6 提前 {rec['lead_v6_vs_v51_s']:.2f}s")
    ddf = pd.DataFrame(ded)
    ddf.to_csv(os.path.join(RES, "t6_deduct_onset.csv"), index=False, encoding="utf-8-sig")
    p(f"  起扣提前量（v6 相对 v5.1）：中位 {ddf.lead_v6_vs_v51_s.median():.2f}s "
      f"(p10~p90 {ddf.lead_v6_vs_v51_s.quantile(.1):.2f}~{ddf.lead_v6_vs_v51_s.quantile(.9):.2f}, "
      f"n={len(ddf)})")

    # ═══════════ ② 消融表 ═══════════
    rows = []
    for arm, Cls, kw in PA.ABLATION:
        for tag in C.HOLD_TAGS:
            dd_ = C.grid(C.HOLD[tag], tmax=TMAX)
            tu2, Xu2, dt2 = dd_["tu"], dd_["Xu"], dd_["dt"]
            dom = C.onset_geometry(tu2, Xu2.sum(axis=1), dt2)
            Y, r = run_traced_arm(Cls, tu2, Xu2, kw)
            m = C.display_metrics(tu2, Y, dom, dt2)
            m.update(arm=arm, tag=tag, group=C.GROUP[tag.split("/")[0]],
                     epoch_total=len(r["epoch"]), n_revoke=len(r["revoke"]),
                     n_handoff=len(r["handoff"]), n_unload=len(r["unload"]))
            rows.append(m)
        p(f"  ablation {arm:>15} done")
    # 参考臂：v5.1（现役）
    for tag in C.HOLD_TAGS:
        dd_ = C.grid(C.HOLD[tag], tmax=TMAX)
        tu2, Xu2, dt2 = dd_["tu"], dd_["Xu"], dd_["dt"]
        dom = C.onset_geometry(tu2, Xu2.sum(axis=1), dt2)
        Y, c = run_v51(tu2, Xu2)
        m = C.display_metrics(tu2, Y, dom, dt2)
        m.update(arm="v5.1_reference", tag=tag, group=C.GROUP[tag.split("/")[0]],
                 epoch_total=len(c.epoch_t), n_revoke=0, n_handoff=0, n_unload=0)
        rows.append(m)
    rec = pd.DataFrame(rows)
    rec.to_csv(os.path.join(RES, "t6_attribution_records.csv"), index=False,
               encoding="utf-8-sig")

    MET = [("err_plat5_pct", "std_R5_pp"), ("err_platstep_pct", "std_Rstep_pp"),
           ("plat_lvl", "std_abs_lvl"), ("T_stable", "std_Tstable_s"),
           ("OS_pct", "std_OS_pp"), ("flat5_pct", "std_flat_pp"),
           ("move3060_pct", "std_move_pp"), ("epoch_total", "std_epoch"),
           ("n_revoke", "std_revoke"), ("n_handoff", "std_handoff")]
    out = []
    for arm, gg in rec.groupby("arm"):
        a = dict(arm=arm, n_rec=len(gg))
        for key, lbl in MET:
            s = gg.groupby("group")[key].std(ddof=1)
            a[lbl] = float(s.median())
        for key, lbl in (("err_plat5_pct", "bias_R5_pp"), ("err_platstep_pct", "bias_Rstep_pp"),
                         ("plat_lvl", "bias_abs_lvl"), ("T_stable", "bias_Tstable_s"),
                         ("epoch_total", "mean_epoch"), ("n_revoke", "mean_revoke"),
                         ("n_handoff", "mean_handoff")):
            a[lbl] = float(gg.groupby("group")[key].mean().median())
        out.append(a)
    ad = pd.DataFrame(out).set_index("arm")
    base = ad.loc["v6_baseline"]
    for lbl in [l for _, l in MET]:
        ad["d_" + lbl] = ad[lbl] - base[lbl]
    for lbl in [l for _, l in MET]:
        ad["pct_" + lbl] = 100.0 * ad["d_" + lbl] / base[lbl] if abs(base[lbl]) > 1e-12 else np.nan
    ad = ad.reset_index()
    ad.to_csv(os.path.join(RES, "t6_attribution.csv"), index=False, encoding="utf-8-sig")

    p("")
    p("=" * 120)
    p(f"表 T6-Q4  消融表（9 份恒载 × 截 {TMAX:.0f} s；L2 组内离散度，3 组取中位；"
      f"Δ 相对 v6 基线；负 Δ = 重复性变好）")
    p("=" * 120)
    cols = ["arm", "std_R5_pp", "d_std_R5_pp", "std_Rstep_pp", "d_std_Rstep_pp",
            "std_Tstable_s", "d_std_Tstable_s", "std_epoch", "d_std_epoch",
            "bias_R5_pp", "mean_epoch", "mean_revoke", "mean_handoff"]
    with pd.option_context("display.width", 240):
        p(ad[cols].round(4).to_string(index=False))
    with io.open(os.path.join(RES, "_t6c_attribution.log"), "w", encoding="utf-8") as fh:
        fh.write("T6-C attribution log\n" + "\n".join(LOG) + "\n")
    p("\n-> results/t6_patch_ab_zero.csv / t6_attribution.csv / t6_attribution_records.csv / "
      "t6_deduct_onset.csv / _t6c_attribution.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
