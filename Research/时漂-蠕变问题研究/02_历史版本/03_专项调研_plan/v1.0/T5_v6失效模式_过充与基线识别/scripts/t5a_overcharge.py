# -*- coding: utf-8 -*-
"""T5A-Q1：基线识别后对抗快相 ⇒ 向上过充 / 向下过冲 的帧段清单。

事件集：**T4-A 冻结表** `T4_*/results/t4a_morphology.csv`（60 事件，含 `t_on`），只读引用。

过充定义（与《指标字典与口径》§3 一致，**参考电平口径见下**）：
    OS% = max_t (Z_disp(t) − Z_final) / J ， t ∈ [t_on, t_on + 30 s]；负值 = 下冲
    Z_final 采用 **`08-v6.1` 的 P 口径**：原始总量在 `[t_on+4.6, t_on+5.4] s` 的中位。

> ⚠ **口径偏离已登记**（报告 §④）：任务书字面规则（`Z_final` = `t_on+60 s` 后中位）
> 在恒载 9 组上会落到**卸载之后**，`Z_final ≈ 0` ⇒ `OS% ≈ +100%` 的伪影。
> 字面口径结果保留在 `OS_pct_task_literal` 列（逐事件可查），两组数字都在 CSV 里。
> 对 `unload / partial_unload`（台阶为负）OS% 定义不适用，改用
> `dev_pre_pct_min/max`（显示相对**事件前电平**的有符号偏离 ÷ |J|）。

稳定性标记：`J_frac = |J| / (P99(Z) − P01(Z))`；`J_frac < 0.05` 的事件在 OS% 口径下
归一化不稳（分母 < 5% 动态范围），单独列出并标 `norm_unstable=True`。

产出：`results/t5a_overcharge_events.csv`（主交付）、`t5a_event_metrics_by_arm.csv`、
      `t5a_overshoot_metric_diff.csv`（与 08-v6.1 的口径差异清单）、
      `t5a_t0_sensitivity.csv`（±1 包 ±40 ms）。

用法：`python scripts/t5a_overcharge.py`
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402

LOG = []
T0 = time.time()


def rec(m):
    s = f"[{time.time() - T0:7.1f}s] {m}"
    print(s, flush=True)
    LOG.append(s)


ARMS = [
    ("v6_now", dict(name="v6_now", cls=C.KV6, kappa_onset=1.30, kappa_restep=1.12)),
    ("v5.1", dict(name="v5.1", cls=None, v51=True)),
    ("v6.1", dict(name="v6.1", cls=C.KV61, kappa_onset=1.30, kappa_restep=1.12)),
    ("raw", dict(name="raw", raw=True)),
]


def main():
    ev = C.ev_load_frozen()
    rec(f"T4-A 冻结事件表：{len(ev)} 事件（onset {int((ev.kind=='onset').sum())} / "
        f"restep {int((ev.kind=='restep').sum())} / "
        f"partial_unload {int((ev.kind=='partial_unload').sum())} / "
        f"unload {int((ev.kind=='unload').sum())}）")
    recs = C.recordings()
    rec(f"读入 {len(recs)} 份录制（100 Hz 网格，timestamp 轴）")

    dyn = {}
    for k, d in recs.items():
        Z = d["Z"]
        dyn[k] = float(np.percentile(Z, 99) - np.percentile(Z, 1))

    rows = []
    for arm_id, arm in ARMS:
        for k, d in recs.items():
            sub = ev[ev.key == k]
            if not len(sub):
                continue
            if arm.get("v51"):
                from t5a_a_common import run_plain, GLM53v51
                Y, _ = run_plain(GLM53v51, d["tu"], d["Xu"])
                out = dict(Y=Y)
            elif arm.get("raw"):
                out = dict(Y=d["Xu"].copy())
            else:
                out = C.run_v6_arm(d, arm)
            Ys = out["Y"].sum(axis=1)
            for _, e in sub.iterrows():
                m = C.metrics_at(d["Z"], Ys, d["tu"], float(e["t_on"]), d["span"])
                m.update(key=k, rec=e["rec"], t_on=float(e["t_on"]), kind=e["kind"],
                         dom=d["dom"], clean=bool(e["clean"]), arm=arm_id,
                         dyn_range=dyn[k])
                rows.append(m)
        rec(f"arm {arm_id} 完成")
    allm = pd.DataFrame(rows)
    allm["J_frac"] = allm["J"].abs() / allm["dyn_range"]
    allm["norm_unstable"] = allm["J_frac"] < 0.05
    allm.to_csv(os.path.join(C.TASK, "results", "t5a_event_metrics_by_arm.csv"),
                index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_event_metrics_by_arm.csv")

    keys = ["key", "rec", "dom", "clean", "kind", "t_on", "J", "J_frac", "dyn_range",
            "norm_unstable"]
    vals = ["OS_pct", "OS_pct_5s", "OS_pct_30s", "US_pct", "US_pct_5s", "OS_pct_task_literal",
            "dev_mid_max_adc", "dev_mid_min_adc", "dev_pre_pct_max",
            "dev_pre_pct_min", "peak_dev_adc", "trough_dev_adc", "MD", "G20",
            "err_1s_pct", "T_stable", "T_stable30", "T_stable10", "T_stable_trunc",
            "Z_final", "pre"]
    sub = allm.copy()
    sub["T_stable_trunc"] = sub["T_stable_trunc"].astype(float)
    piv = sub.pivot_table(index=keys, columns="arm", values=vals, aggfunc="first")
    piv.columns = [f"{a}__{b}" for b, a in piv.columns]
    piv = piv.reset_index()
    for c in ["OS_pct", "OS_pct_5s", "OS_pct_30s", "US_pct", "US_pct_5s",
              "OS_pct_task_literal", "dev_mid_max_adc", "dev_mid_min_adc",
              "dev_pre_pct_max", "dev_pre_pct_min", "peak_dev_adc", "trough_dev_adc",
              "MD", "G20", "err_1s_pct", "T_stable", "T_stable30", "T_stable10",
              "Z_final", "pre"]:
        if f"v6_now__{c}" in piv:
            piv[c] = piv[f"v6_now__{c}"]
    piv["OS_callback"] = (piv["OS_pct"] > 0) & (piv["G20"] < 0.98)
    piv["exceeds_5pct"] = piv["OS_pct"] > 5.0
    piv["exceeds_10pct"] = piv["OS_pct"] > 10.0
    piv["undershoot_5pct"] = piv["US_pct"] > 5.0
    piv["und_gt_os"] = piv["US_pct"] > piv["OS_pct"]
    piv["is_load_class"] = piv["kind"].isin(["onset", "restep"])
    out = piv.sort_values(["key", "t_on"]).reset_index(drop=True)
    p = os.path.join(C.TASK, "results", "t5a_overcharge_events.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"产出 {p}（{len(out)} 行 × {len(out.columns)} 列）")

    # ── 汇总 ──
    lines = ["\n===== T5A-Q1 汇总（v6_now，Z_final = t_on+4.6~5.4 s 原始中位）====="]
    for kind in ["onset", "restep", "unload", "partial_unload"]:
        s = out[out.kind == kind]
        if not len(s):
            continue
        lines.append(
            f"{kind:15s} n={len(s):2d}  OS% 中位 {s.OS_pct.median():8.2f}  "
            f"p10~p90 {s.OS_pct.quantile(.1):8.2f}~{s.OS_pct.quantile(.9):8.2f}  "
            f"max {s.OS_pct.max():8.2f}  |  US% 中位 {s.US_pct.median():7.2f}  "
            f"max {s.US_pct.max():8.2f}  |  norm_unstable n={int(s.norm_unstable.sum())}")
    ld = out[out.is_load_class & ~out.norm_unstable]
    lines.append(f"\n加载类（onset+restep，J_frac≥0.05）n={len(ld)}：")
    lines.append(f"  OS% 中位 {ld.OS_pct.median():.2f}  p10~p90 "
                 f"{ld.OS_pct.quantile(.1):.2f}~{ld.OS_pct.quantile(.9):.2f}  "
                 f"max {ld.OS_pct.max():.2f}；>5% 有 {int(ld.exceeds_5pct.sum())}/{len(ld)}，"
                 f">10% 有 {int(ld.exceeds_10pct.sum())}/{len(ld)}")
    lines.append(f"  US% 中位 {ld.US_pct.median():.2f}  max {ld.US_pct.max():.2f}；"
                 f">5% 有 {int(ld.undershoot_5pct.sum())}/{len(ld)}")
    lines.append(f"  字面口径 OS%（task_literal）中位 {ld.OS_pct_task_literal.median():.2f}"
                 f" ← 与主口径对比可见伪影")
    top = out[out.is_load_class].reindex(
        out[out.is_load_class].OS_pct.abs().sort_values(ascending=False).index).head(10)
    lines.append("\n加载类 OS% 绝对值最大的 10 个事件：")
    for _, r in top.iterrows():
        lines.append(f"  {r['key']:5s} @{r['t_on']:7.2f}s {r['kind']:7s} "
                     f"OS%={r.OS_pct:8.2f}  US%={r.US_pct:7.2f}  J={r.J:9.1f}  "
                     f"J_frac={r.J_frac:5.3f}  MD={r.MD:8.1f}  G20={r.G20:6.3f}  "
                     f"T_stable={r.T_stable}")
    txt = "\n".join(lines)
    rec(txt)

    # ── 与 08-v6.1 口径对照 ──
    cmp_rows = [
        dict(item="参考电平 Z_final / P", ours="原始总量 [t_on+4.6, t_on+5.4] s 中位",
             v61="同一事件后 4.6~5.4 s 的原始总量中位（同）", diff="相同"),
        dict(item="超调窗", ours="max over [t_on, t_on+30 s]",
             v61="max over [t_on, t_on+5 s]", diff="本任务窗长 6 倍（更易捕捉后续回冲）"),
        dict(item="归一化分母", ours="J = 原始 [t_on+4, t_on+6] 中位 − [t_on−2, t_on) 中位",
             v61="step（同一事件的原始台阶）", diff="基本相同"),
        dict(item="事件集", ours="T4-A 冻结表 60 事件", v61="13 份 × 38 个真阶跃（自建检测）",
             diff="事件原点与筛选门限不同（T4-A 含 2% 门限的小台阶）"),
        dict(item="显示量", ours="ΣY（总通道）", v61="ΣY（总通道）", diff="相同"),
        dict(item="任务书字面规则", ours="额外保留在 OS_pct_task_literal 列",
             v61="—", diff="恒载组该列有 +100% 级伪影（Z_final 落在卸载后）"),
    ]
    pd.DataFrame(cmp_rows).to_csv(
        os.path.join(C.TASK, "results", "t5a_overshoot_metric_diff.csv"),
        index=False, encoding="utf-8-sig")

    # ── ±1 包 t_on 敏感性（只跑 v6_now）──
    sen = []
    for shift_ms in (-40, -20, 0, 20, 40):
        rws = []
        arm = ARMS[0][1]
        for k, d in recs.items():
            s = ev[ev.key == k].copy()
            if not len(s):
                continue
            s["t_on"] = s["t_on"] + shift_ms / 1000.0
            out2 = C.run_v6_arm(d, arm)
            Ys = out2["Y"].sum(axis=1)
            for _, e in s.iterrows():
                m = C.metrics_at(d["Z"], Ys, d["tu"], float(e["t_on"]), d["span"])
                m.update(key=k, kind=e["kind"])
                rws.append(m)
        a = pd.DataFrame(rws)
        a["J_frac"] = a.J.abs() / a.key.map(dyn)
        L = a[a.kind.isin(["onset", "restep"]) & (a.J_frac >= 0.05)]
        sen.append(dict(shift_ms=shift_ms, n_load=len(L),
                        OS_med_load=L.OS_pct.median(), OS_max_load=L.OS_pct.max(),
                        US_med_load=L.US_pct.median(),
                        T_med_load=L.T_stable.median(),
                        err1_med_load=L.err_1s_pct.median(),
                        MD_med_load=L.MD.median()))
        rec(f"±t_on 敏感性 shift={shift_ms:+4d} ms：OS% 中位 {sen[-1]['OS_med_load']:.2f}"
            f" / max {sen[-1]['OS_max_load']:.2f} / T_med {sen[-1]['T_med_load']}")
    pd.DataFrame(sen).to_csv(os.path.join(C.TASK, "results", "t5a_t0_sensitivity.csv"),
                             index=False, encoding="utf-8-sig", float_format="%.6g")

    # ── 与 08-v6.1 台账逐事件对表（同窗长 5 s，可直接比）──
    p61 = os.path.join(C.PROG, "08-v6.1", "results", "v61_overshoot.csv")
    if os.path.exists(p61):
        v61 = pd.read_csv(p61)
        mine = allm[(allm.arm == "v6_now") & (allm.kind.isin(["onset", "restep"]))]
        rows61 = []
        for _, r in v61.iterrows():
            ds = r["dataset"]
            key = C.T4A_KEY.get(ds)
            if key is None:
                for k, v in C.T4A_KEY.items():
                    if ds[:3] == k[:3]:
                        key = v
                        break
            if key is None:
                continue
            cand = mine[(mine.key == key) & (np.abs(mine.t_on - r["t0"]) < 0.35)]
            base = dict(v61_dataset=ds, v61_t0=r["t0"], v61_kind=r["ev"],
                        v61_over_peak=r["over_peak"], v61_transient=r["transient"],
                        key=key)
            if not len(cand):
                rows61.append(dict(base, matched=False))
                continue
            c = cand.iloc[0]
            rows61.append(dict(base, matched=True, t_on=c.t_on, kind=c.kind, J=c.J,
                               J_frac=c.J_frac, OS5_mine=c.OS_pct_5s, OS30_mine=c.OS_pct_30s,
                               US5_mine=c.US_pct_5s, MD_mine=c.MD, G20_mine=c.G20,
                               d_t0=c.t_on - r["t0"]))
        X = pd.DataFrame(rows61)
        X.to_csv(os.path.join(C.TASK, "results", "t5a_crosscheck_v61.csv"),
                 index=False, encoding="utf-8-sig", float_format="%.6g")
        Xm = X[X.matched]
        if len(Xm):
            rec(f"\n与 08-v6.1 台账对表：匹配 {len(Xm)}/{len(X)}；"
                f"|Δt_on| 中位 {Xm.d_t0.abs().median():.3f} s")
            rec(f"  08-v6.1 over_peak 中位 {Xm.v61_over_peak.median():.3f}% "
                f"max {Xm.v61_over_peak.max():.3f}%")
            rec(f"  本任务 OS5 中位 {Xm.OS5_mine.median():.3f}% max {Xm.OS5_mine.max():.3f}%"
                f"  ⇒ 差中位 {(Xm.OS5_mine - Xm.v61_over_peak).median():.3f} pt")
            rec(f"  本任务 OS30 中位 {Xm.OS30_mine.median():.3f}%"
                f"  ⇒ 差中位 {(Xm.OS30_mine - Xm.v61_over_peak).median():.3f} pt"
                f"（30 s 窗纳入慢相段，系统性偏大）")
            rec(f"  产出 results/t5a_crosscheck_v61.csv")
    C.write_log(os.path.join(C.TASK, "results", "_t5a_overcharge.log"),
                "python scripts/t5a_overcharge.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
