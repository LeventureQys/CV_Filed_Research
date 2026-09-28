# -*- coding: utf-8 -*-
"""T6-A：可重复性三层分解（L1 输入层 / L2 输出层 / L3 见 t6b）—— T6-Q1。

样本集：恒载 9 组 = 3 个传感器组（右拇指/左拇指/四指）× 3 次重复录制（数据1/2/3），
        **同一传感器同一标称负载的 3 次重复**是本次调研里唯一真正的"重复加载"样本集；
        变载实录 4 份是不同加载序列，**不构成重复样本**（故 L1/L2 不出现在它上面）。
域：恒载 9 组为力域（N，量化 0.001）；时间轴 100 Hz 均匀网格（timestamp 列）。
真值口径（两套同时报，因为排名会反转）：
  R5   = 原始读数在 t_on+5 s 的电平（含快相蠕变）
  Rstep= 原始读数在 t_on+0.2 s 的机械台阶电平
  Rabs = 平台绝对电平本身（被"人压得一样不一样"污染，仅作对照）
T_stable：T1-A 未交付 ⇒ 用 legacy 口径（tol 5%·step / hold 30 s）复刻，
        **不可与 T1 绝对横比**，横向结论一律用同一运行内的配对差。

产出：
  results/t6_l1_records.csv         每录制一行的 L1（原始数据层）几何与形状
  results/t6_l2_records.csv         每（录制 × 臂）一行的 L2（显示层）指标
  results/t6_repeat_dispersion.csv  **主交付**：三层 × v5.1/v6/v6.1（+raw）的组内离散度
  figures/T6_01_three_layers.png    L1/L2/L3 三层分解（L3 由 t6b 追加绘制）
  results/_t6a_layers.log
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
from t6_glm53_v51 import GLM53v51                                  # noqa: E402
from t6_glm53_v6 import GLM53v6, ROM_TAU, ROM_G                    # noqa: E402
from t6_glm53_v61 import GLM53v61                                  # noqa: E402

RES, FIG = C.RES, C.FIG
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
LOG = []

TRACED_V6 = AC.make_traced(GLM53v6)
TRACED_V61 = AC.make_traced(GLM53v61)
TRACED_V51 = AL.make_traced(GLM53v51)
ARMS = ["raw", "v5.1", "v6", "v6.1"]


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def run_arm(arm, tu, Xu):
    """跑一条臂，返回 (Y, ledger)。ledger 记录状态机事件（仪表化，不改行为）。"""
    if arm == "raw":
        return Xu.copy(), dict(epoch=[], revoke=[], handoff=[], unload=[])
    if arm == "v5.1":
        c = TRACED_V51(Xu.shape[1])
        Y = np.empty_like(Xu)
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
        return Y, dict(epoch=[float(t) for t in c.epoch_t], revoke=[], handoff=[], unload=[])
    Cls = TRACED_V6 if arm == "v6" else TRACED_V61
    r = AC.run_traced(Cls, tu, Xu)
    return r["Y"], dict(epoch=r["epoch"], revoke=r["revoke"], handoff=r["handoff"],
                        unload=r["unload"], gevent=r["gevent"], comp=r["comp"])


def ahat_legacy(tu, tot, i_e, dt):
    """复刻 `b_common.shape_mismatch`：Â = Σy·g/Σg²（窗 [0.2,0.8] s，0.1 s 中位）。"""
    ts1 = AL.med_smooth(tot, max(1, int(round(0.1 / dt))))
    base = float(np.median(ts1[max(0, i_e - int(0.30 / dt)):max(1, i_e - int(0.05 / dt))]))
    tau = tu - tu[i_e]
    inc = ts1 - base
    sel = (tau >= 0.20) & (tau <= 0.80)
    if sel.sum() < 5:
        return np.nan, np.nan, np.nan
    gg = np.interp(tau[sel], ROM_TAU, ROM_G, left=0.0, right=1.0)
    den = float((gg * gg).sum())
    Ahat = float((inc[sel] * gg).sum()) / den
    inc5 = float(np.interp(5.0, tau, inc))
    return Ahat, inc5, (Ahat / inc5 if abs(inc5) > 1e-12 else np.nan)


def main():
    l1_rows, l2_rows = [], []
    for tag in C.HOLD_TAGS:
        path = C.HOLD[tag]
        grp = C.GROUP[tag.split("/")[0]]
        d = C.grid(path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dt"]
        tot = Xu.sum(axis=1)
        dom = C.onset_geometry(tu, tot, dt)
        f = C.raw_shape(tu, tot, dom, dt)
        Ahat, inc5c, msh = ahat_legacy(tu, tot, dom["i_e"], dt)
        l1_rows.append(dict(
            tag=tag, group=grp, n_frames=len(tu), span_s=round(d["span"], 2),
            t_e=round(dom["t_e"], 3), pre=dom["pre"], a_step=dom["a_step"],
            inc5=dom["inc5"], creep_frac=dom["inc5"] / dom["a_step"] - 1.0,
            Ahat_legacy=Ahat, m_shape=msh,
            **{f"f_{t:.2f}": v for t, v in zip(C.TAU_GRID, f)}))
        # 原始读数臂（无补偿）的 L2
        m = C.display_metrics(tu, Xu, dom, dt)
        m.update(tag=tag, group=grp, arm="raw", epoch_total=0, epoch_pre=0, A_hat=Ahat)
        l2_rows.append(m)
        for arm in ARMS[1:]:
            Y, led = run_arm(arm, tu, Xu)
            mm = C.display_metrics(tu, Y, dom, dt)
            ep = led["epoch"]
            t0 = dom["t_e"]
            ep_pairs = [(float(e[0]) if isinstance(e, (tuple, list)) else float(e),
                         (str(e[1]) if isinstance(e, (tuple, list)) else "")) for e in ep]
            mm.update(tag=tag, group=grp, arm=arm,
                      epoch_total=len(ep_pairs),
                      epoch_pre=sum(1 for t, _ in ep_pairs if t < t0 - 0.3),
                      n_revoke=len(led["revoke"]), n_handoff=len(led["handoff"]),
                      n_unload=len(led["unload"]), A_hat=Ahat)
            l2_rows.append(mm)
            p(f"  {tag:>18} {arm:>5} plat={mm['plat_lvl']:.3f} "
              f"err5={mm['err_plat5_pct']:+.2f}% errstep={mm['err_platstep_pct']:+.2f}% "
              f"T_stable={mm['T_stable']:.2f}s ep={mm['epoch_total']} ep_pre={mm['epoch_pre']}")
        p(f"{tag} [{grp}] t_e={dom['t_e']:.2f}s a_step={dom['a_step']:.3f} "
          f"inc5={dom['inc5']:.3f} m={msh:.4f}")

    l1 = pd.DataFrame(l1_rows)
    l2 = pd.DataFrame(l2_rows)
    l1.to_csv(os.path.join(RES, "t6_l1_records.csv"), index=False, encoding="utf-8-sig")
    l2.to_csv(os.path.join(RES, "t6_l2_records.csv"), index=False, encoding="utf-8-sig")

    # ───────── L1 离散度（输入层，与实现无关） ─────────
    fcols = [f"f_{t:.2f}" for t in C.TAU_GRID]
    rows = []
    for grp, gg in l1.groupby("group"):
        rows.append(dict(layer="L1_input", impl="(input, impl-independent)", group=grp, n=len(gg),
                         metric="jump_a_step_pct", value=100.0 * gg.a_step.std(ddof=1) / gg.a_step.mean()))
        rows.append(dict(layer="L1_input", impl="(input, impl-independent)", group=grp, n=len(gg),
                         metric="inc5_pct", value=100.0 * gg.inc5.std(ddof=1) / gg.inc5.mean()))
        rows.append(dict(layer="L1_input", impl="(input, impl-independent)", group=grp, n=len(gg),
                         metric="creep_frac_pp", value=100.0 * gg.creep_frac.std(ddof=1)))
        rows.append(dict(layer="L1_input", impl="(input, impl-independent)", group=grp, n=len(gg),
                         metric="CV_shape_med", value=float(np.median(
                             [100.0 * gg[c].std(ddof=1) / abs(gg[c].mean()) for c in fcols]))))
        rows.append(dict(layer="L1_input", impl="(input, impl-independent)", group=grp, n=len(gg),
                         metric="Spread_shape_med_pp", value=float(np.median(
                             [100.0 * (gg[c].max() - gg[c].min()) for c in fcols]))))
        rows.append(dict(layer="L1_input", impl="(input, impl-independent)", group=grp, n=len(gg),
                         metric="abs_plat_raw_std", value=float(gg.a_step.std(ddof=1))))
    # τ 网格逐点 CV_shape / Spread
    for grp, gg in l1.groupby("group"):
        for c in fcols:
            rows.append(dict(layer="L1_input", impl="(input, impl-independent)", group=grp, n=len(gg),
                             metric=f"CV_shape@{c[2:]}",
                             value=100.0 * gg[c].std(ddof=1) / abs(gg[c].mean())))
    for grp, gg in l1.groupby("group"):
        for c in fcols:
            rows.append(dict(layer="L1_input", impl="(input, impl-independent)", group=grp, n=len(gg),
                             metric=f"Spread_shape@{c[2:]}",
                             value=100.0 * (gg[c].max() - gg[c].min())))

    # ───────── L2 离散度（输出层，逐实现） ─────────
    L2M = [("err_plat5_pct", "plat_err_std_R5_pp"),
           ("err_platstep_pct", "plat_err_std_Rstep_pp"),
           ("plat_lvl", "abs_plat_std"),
           ("T_stable", "T_stable_std_s"),
           ("OS_pct", "OS_std_pp"),
           ("US_pct", "US_std_pp"),
           ("err5s", "err5s_std_pp"),
           ("flat5_pct", "flat_in_plat_std_pp"),
           ("move3060_pct", "move3060_std_pp")]
    for (arm, grp), gg in l2.groupby(["arm", "group"]):
        for key, lbl in L2M:
            v = gg[key].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) < 2:
                continue
            rows.append(dict(layer="L2_output", impl=arm, group=grp, n=len(v), metric=lbl,
                             value=float(np.std(v, ddof=1))))
            rows.append(dict(layer="L2_output", impl=arm, group=grp, n=len(v),
                             metric=lbl.replace("_std", "_range"), value=float(v.max() - v.min())))
            rows.append(dict(layer="L2_output", impl=arm, group=grp, n=len(v),
                             metric=lbl.replace("_std", "_bias"), value=float(np.mean(v))))
        for key, lbl in (("epoch_total", "epoch_total_std"), ("epoch_pre", "epoch_pre_std")):
            v = gg[key].to_numpy(float)
            rows.append(dict(layer="L2_output", impl=arm, group=grp, n=len(v), metric=lbl,
                             value=float(np.std(v, ddof=1))))
    disp = pd.DataFrame(rows)
    disp.to_csv(os.path.join(RES, "t6_repeat_dispersion.csv"), index=False,
                encoding="utf-8-sig")

    # ───────── 打印：跨 3 组取中位的主表 ─────────
    key = ["plat_err_std_R5_pp", "plat_err_std_Rstep_pp", "abs_plat_std", "T_stable_std_s",
           "OS_std_pp", "err5s_std_pp", "flat_in_plat_std_pp", "move3060_std_pp",
           "epoch_total_std", "epoch_pre_std",
           "plat_err_bias_R5_pp", "plat_err_bias_Rstep_pp", "abs_plat_bias",
           "T_stable_bias_s"]
    L2only = disp[disp.layer == "L2_output"]
    p("")
    p("=" * 118)
    p("表 1  L2 输出层组内离散度/偏差（3 组取中位；n=3/组；口径 R5/pre+inc5、Rstep/pre+a_step）")
    p("=" * 118)
    piv = L2only.pivot_table(index="metric", columns="impl", values="value", aggfunc="median")
    shown = [k for k in key if k in piv.index]
    with pd.option_context("display.width", 200):
        p(piv.loc[shown, [a for a in ARMS if a in piv.columns]].round(4).to_string())
    p("")
    p("表 2  L1 输入层（原始数据本身）组内离散度（3 组取中位；**与实现无关**）")
    L1only = disp[disp.layer == "L1_input"]
    piv1 = L1only.pivot_table(index="metric", columns="impl", values="value", aggfunc="median")
    with pd.option_context("display.width", 200):
        p(piv1.round(4).to_string())
    p("")
    p("表 3  τ 网格逐点 CV_shape / Spread（中位·三组）")
    for pre in ("CV_shape@", "Spread_shape@"):
        sub = L1only[L1only.metric.str.startswith(pre)]
        piv3 = sub.pivot_table(index="metric", columns="group", values="value")
        with pd.option_context("display.width", 200):
            p(f"  [{pre}]"); p(piv3.round(3).to_string())

    with io.open(os.path.join(RES, "_t6a_layers.log"), "w", encoding="utf-8") as fh:
        fh.write("T6-A layers log\n" + "\n".join(LOG) + "\n")
    p("\n-> results/t6_l1_records.csv / t6_l2_records.csv / t6_repeat_dispersion.csv / "
      "_t6a_layers.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
