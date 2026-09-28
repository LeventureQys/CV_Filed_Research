# -*- coding: utf-8 -*-
"""T6-D：改进空间的可量化可行性估计（T6-Q5）+ 可重复性/稳定时间权衡（T6-Q6）。

方法：**沿用 v6 原型 + 单旋钮补丁**（见 `t6_patches.py` 的 I1/I2/I3 与组合臂），
在 9 份恒载（截 80 s）上与 v6 基线同批对照，取
  · `repeat_std_R5` / `repeat_std_Rstep`（L2 组内离散，3 组取中位）
  · `T_stable` 中位（legacy 口径，tol 5%·step / hold 30 s；**T1-A 未交付，不可横比绝对值**）
  · `L3 proxy`：把**同一段数据的时序抖动**再跑一遍（每臂 0/±0.25/±1 包周期，5 个种子），
    得该臂的路径抖动 RMS —— 改进如果只压 L2 却让 L3 变差，不算真改进。

产出：results/t6_improve_ab.csv、t6_tradeoff.csv、results/_t6d_improve.log
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
from t6_glm53_v61 import GLM53v61                                  # noqa: E402

RES = C.RES
TMAX = 80.0
N_SEED_L3 = 5
LOG = []
# ── 臂集合：v6 基线 + 单旋钮改进臂 + **v6.1（既有已实现的"保守形状反演"，作为"改进已存在"的对照）** ──
ARMS_ALL = [(n, c, kw) for n, c, kw, _ in PA.IMPROVE] + \
           [("v6.1_asIS", AC.make_traced(GLM53v61), {})]
# L3 代理场景：恒载 四指/数据1 前 80 s（有 onset + 平台 + 卸载）+ 实录 中途切换-1d9493
L3_SCENES = [("四指指尖/数据1", 80.0), ("中途切换-1d9493", None)]


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def run(Cls, tu, Xu, kw=None):
    r = AC.run_traced(Cls, tu, Xu, **(kw or {}))
    return r["Y"], r


def main():
    # ── ① 补丁零差证明：把 t6c 未覆盖的改进臂补进同一张表（默认值下必须 bit-identical）──
    dref = C.grid(C.HOLD["四指指尖/数据1"], tmax=TMAX)
    tuf, Xuf = dref["tu"], dref["Xu"]
    y0 = run(PA.BASE, tuf, Xuf)[0]
    zp = os.path.join(RES, "t6_patch_ab_zero.csv")
    prev = pd.read_csv(zp) if os.path.exists(zp) else pd.DataFrame(
        columns=["arm", "ref_tag", "n_frames", "max_abs_diff", "rms_diff", "bit_identical"])
    have = set(prev.arm)
    add = []
    for name, Cls, _kw, dflt in PA.IMPROVE:
        if name in have or not dflt:
            continue
        y1 = run(Cls, tuf, Xuf, dflt)[0]
        dd = np.abs(y1 - y0)
        add.append(dict(arm=name, ref_tag="四指指尖/数据1", n_frames=len(tuf),
                        max_abs_diff=float(dd.max()), rms_diff=float(np.sqrt(np.mean(dd ** 2))),
                        bit_identical=bool(dd.max() == 0.0)))
        p(f"  patch A/B zero (t6d): {name:>15} max|Δ|={dd.max():.3e}")
    if add:
        pd.concat([prev, pd.DataFrame(add)], ignore_index=True).to_csv(
            zp, index=False, encoding="utf-8-sig")
    chk = pd.read_csv(zp)
    assert chk.bit_identical.all(), "补丁默认值下未做到零差"
    p(f"  → t6_patch_ab_zero.csv 共 {len(chk)} 条，全部 bit-identical")

    rows = []
    for arm, Cls, kw in ARMS_ALL:
        for tag in C.HOLD_TAGS:
            d = C.grid(C.HOLD[tag], tmax=TMAX)
            tu, Xu, dt = d["tu"], d["Xu"], d["dt"]
            dom = C.onset_geometry(tu, Xu.sum(axis=1), dt)
            Y, r = run(Cls, tu, Xu, kw)
            m = C.display_metrics(tu, Y, dom, dt)
            m.update(arm=arm, tag=tag, group=C.GROUP[tag.split("/")[0]],
                     epoch_total=len(r["epoch"]), n_revoke=len(r["revoke"]),
                     n_handoff=len(r["handoff"]), n_unload=len(r["unload"]))
            rows.append(m)
        p(f"  improve {arm:>16} done")
    rec = pd.DataFrame(rows)
    rec.to_csv(os.path.join(RES, "t6_improve_records.csv"), index=False, encoding="utf-8-sig")

    # ── L3 代理：每臂在 2 个场景上做 ±0.25P / ±1.0P 时序抖动（各 N_SEED_L3 个种子）──
    l3 = []
    for tag, tmax in L3_SCENES:
        d = C.grid(C.ALL[tag], tmax=tmax)
        tu, Xu = d["tu"], d["Xu"]
        pid_raw, nk, P, fpp = C.packet_layout(d["t"])
        pkt = C.grid_packet_id(tu, d["t"], pid_raw)
        for arm, Cls, kw in ARMS_ALL:
            yb = run(Cls, tu, Xu, kw)[0].sum(axis=1)
            for Jrel in (0.25, 1.00):
                rr = []
                for s in range(N_SEED_L3):
                    rng = np.random.default_rng(7000 + s)
                    tj, ro = C.jitter_axis(tu, pkt, P, Jrel * P, rng)
                    yj = run(Cls, tj, Xu, kw)[0].sum(axis=1)
                    dd = yj - yb
                    rr.append(float(np.sqrt(np.mean(dd * dd))))
                l3.append(dict(arm=arm, tag=tag, J_rel=Jrel, n=N_SEED_L3,
                               rms_med=float(np.median(rr)), rms_p90=float(np.percentile(rr, 90)),
                               lvl_med=float(np.median(Xu.sum(axis=1)))))
            p(f"  L3 {tag:>16} {arm:>16} done")
    l3d = pd.DataFrame(l3)
    l3d["rms_pct_lvl"] = 100.0 * l3d.rms_med / l3d.lvl_med.abs()
    l3d.to_csv(os.path.join(RES, "t6_improve_l3.csv"), index=False, encoding="utf-8-sig")

    # ── 汇总 ──
    out = []
    for arm, gg in rec.groupby("arm"):
        a = dict(arm=arm)
        for key, lbl in (("err_plat5_pct", "std_R5_pp"), ("err_platstep_pct", "std_Rstep_pp"),
                         ("T_stable", "T_stable_med_s"), ("OS_pct", "std_OS_pp"),
                         ("flat5_pct", "std_flat_pp"), ("move3060_pct", "std_move_pp"),
                         ("epoch_total", "std_epoch")):
            if key == "T_stable":
                a[lbl] = float(gg.groupby("group")[key].median().median())
            else:
                a[lbl] = float(gg.groupby("group")[key].std(ddof=1).median())
        for key, lbl in (("err_plat5_pct", "bias_R5_pp"), ("plat_lvl", "std_abs_lvl"),
                         ("T_stable", "T_stable_p90_s"), ("epoch_total", "mean_epoch"),
                         ("n_revoke", "mean_revoke")):
            if key == "err_plat5_pct":
                a[lbl] = float(gg.groupby("group")[key].mean().median())
            elif key == "plat_lvl":
                a[lbl] = float(gg.groupby("group")[key].std(ddof=1).median())
            elif key == "T_stable":
                a[lbl] = float(gg.T_stable.quantile(0.9))
            else:
                a[lbl] = float(gg[key].mean())
        sub = l3d[(l3d.arm == arm) & (l3d.J_rel == 1.00)]
        a["L3_rms_pct_lvl_med"] = float(sub.rms_pct_lvl.median()) if len(sub) else np.nan
        a["L3_scenes"] = len(sub)
        out.append(a)
    ad = pd.DataFrame(out).set_index("arm")
    b = ad.loc["v6_baseline"]
    for lbl in ("std_R5_pp", "std_Rstep_pp", "std_abs_lvl", "std_OS_pp", "std_flat_pp",
                "std_move_pp", "std_epoch", "L3_rms_pct_lvl_med"):
        ad["d_" + lbl] = ad[lbl] - b[lbl]
        ad["pct_" + lbl] = 100.0 * ad["d_" + lbl] / b[lbl] if abs(b[lbl]) > 1e-12 else np.nan
    ad["d_T_stable_med_s"] = ad["T_stable_med_s"] - b["T_stable_med_s"]
    ad = ad.reset_index()
    ad.to_csv(os.path.join(RES, "t6_improve_ab.csv"), index=False, encoding="utf-8-sig")
    ad.to_csv(os.path.join(RES, "t6_tradeoff.csv"), index=False, encoding="utf-8-sig")

    p("")
    p("=" * 122)
    p(f"表 T6-Q5/Q6  改进方向 A/B（9 份恒载 × 截 {TMAX:.0f}s；L3 代理 = {len(L3_SCENES)} 场景 "
      f"× ±1 包时序抖动 × {N_SEED_L3} 种子）")
    p("=" * 122)
    cols = ["arm", "std_R5_pp", "pct_std_R5_pp", "std_Rstep_pp", "pct_std_Rstep_pp",
            "L3_rms_pct_lvl_med", "pct_L3_rms_pct_lvl_med", "T_stable_med_s",
            "d_T_stable_med_s", "bias_R5_pp", "mean_revoke"]
    with pd.option_context("display.width", 240):
        p(ad[cols].round(4).to_string(index=False))
    with io.open(os.path.join(RES, "_t6d_improve.log"), "w", encoding="utf-8") as fh:
        fh.write("T6-D improve log\n" + "\n".join(LOG) + "\n")
    p("\n-> results/t6_improve_ab.csv / t6_tradeoff.csv / t6_improve_records.csv / "
      "t6_improve_l3.csv / _t6d_improve.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
