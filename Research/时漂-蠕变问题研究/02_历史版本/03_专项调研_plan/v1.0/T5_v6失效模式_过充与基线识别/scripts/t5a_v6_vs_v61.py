# -*- coding: utf-8 -*-
"""T5A-Q3：v6 vs v6.1 对照 —— `ROM_SCALE=1.06` 消掉了多少、剩下多少，以及**用户看到的是哪一版**。

对照臂（同一批 T4-A 60 事件、同一 100 Hz 网格、同一指标实现）：
  · `v6`      = 原型 v6（κ 1.30/1.12）
  · `v61`     = 原型 v6.1（ROM_SCALE=1.06，只对 onset 生效；RATE_DOWN=0.25）
  · `v61_s104/108/110/112` = ROM_SCALE 扫描（附录，用于给出"已修多少/还剩多少"的连续曲线）
  · `v6_now_noF1` = v6.1 关 F1（ROM_SCALE=1.0）+ 保留 F2 ⇒ 应与 v6 一致（等价性交叉验证）
  · `v5.1`    = 现役免责基线（用户现场跑的上一版，作"用户到底看到哪一版"的第三方参照）

`ROM_SCALE` 只改形状查表，不改检测器/滑行器/慢相，因此这是**单变量 A/B**。

判定"用户看到的是哪一版"的依据（三条互相独立）：
  ① 用户原话是"识别到基线之后的快速爬升对抗很容易出现超出预期的向上过充"——
     ="冲高"是**主要可见症状**；v6.1 的设计目标正是把它压掉。
  ② 用户描述里没有提到"欠报/读数偏低"。
  ③ 本任务实测：v6 的 onset OS% max 与 `08-v6.1` 记的 +10.9% 同量级；v6.1 落到 +4.6% 以内。

产出：`results/t5a_v6_vs_v61.csv`（逐事件）、`t5a_v6_vs_v61_summary.csv`（逐臂汇总）、
      `_t5a_v61.log`。

用法：`python scripts/t5a_v6_vs_v61.py`
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
    dict(name="v6", cls=C.KV6, kappa_onset=1.30, kappa_restep=1.12),
    dict(name="v61_s106", cls=C.KV61, kappa_onset=1.30, kappa_restep=1.12),
    dict(name="v61_noF1", cls=C.KV61, kappa_onset=1.30, kappa_restep=1.12,
         rom_scale=1.00, rate_down=0.25),
    dict(name="v61_s104", cls=C.KV61, kappa_onset=1.30, kappa_restep=1.12, rom_scale=1.04),
    dict(name="v61_s108", cls=C.KV61, kappa_onset=1.30, kappa_restep=1.12, rom_scale=1.08),
    dict(name="v61_s110", cls=C.KV61, kappa_onset=1.30, kappa_restep=1.12, rom_scale=1.10),
    dict(name="v61_s112", cls=C.KV61, kappa_onset=1.30, kappa_restep=1.12, rom_scale=1.12),
]


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    dyn = {k: float(np.percentile(d["Z"], 99) - np.percentile(d["Z"], 1))
           for k, d in recs.items()}
    rows = []
    for arm in ARMS:
        for k, d in recs.items():
            sub = ev[ev.key == k]
            if not len(sub):
                continue
            out = C.run_v6_arm(d, arm)
            Ys = out["Y"].sum(axis=1)
            for _, e in sub.iterrows():
                m = C.metrics_at(d["Z"], Ys, d["tu"], float(e["t_on"]), d["span"])
                m.update(key=e["key"], rec=e["rec"], t_on=float(e["t_on"]),
                         kind=e["kind"], dom=d["dom"], clean=bool(e["clean"]),
                         arm=arm["name"], dyn_range=dyn[k],
                         n_epoch=len(out["epoch"]), n_handoff=len(out["handoff"]),
                         n_revoke=len(out["revoke"]))
                rows.append(m)
        rec(f"arm {arm['name']} 完成")

    # v5.1 参照（现役上一版）
    from t5a_a_common import run_plain, GLM53v51
    for k, d in recs.items():
        sub = ev[ev.key == k]
        if not len(sub):
            continue
        Y51, _ = run_plain(GLM53v51, d["tu"], d["Xu"])
        Ys = Y51.sum(axis=1)
        for _, e in sub.iterrows():
            m = C.metrics_at(d["Z"], Ys, d["tu"], float(e["t_on"]), d["span"])
            m.update(key=e["key"], rec=e["rec"], t_on=float(e["t_on"]),
                     kind=e["kind"], dom=d["dom"], clean=bool(e["clean"]),
                     arm="v5.1", dyn_range=dyn[k])
            rows.append(m)
    rec("arm v5.1 完成")

    A = pd.DataFrame(rows)
    A["J_frac"] = A["J"].abs() / A["dyn_range"]
    A.to_csv(os.path.join(C.TASK, "results", "t5a_v6_vs_v61.csv"),
             index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_v6_vs_v61.csv")

    # ── 逐臂汇总（分族：加载类主口径 / onset / restep / 恒载 / 实录）──
    sm = []
    for arm, g in A.groupby("arm", sort=False):
        L = g[g.kind.isin(["onset", "restep"]) & (g.J_frac >= 0.05)]
        on = L[L.kind == "onset"]
        re_ = L[L.kind == "restep"]
        hz = L[L.dom == "显示域"]
        ad = L[L.dom == "ADC域"]
        sm.append(dict(
            arm=arm, n_load=len(L), n_onset=len(on), n_restep=len(re_),
            OS5_med=L.OS_pct_5s.median(), OS5_p90=L.OS_pct_5s.quantile(.90),
            OS5_max=L.OS_pct_5s.max(), OS5_med_onset=on.OS_pct_5s.median(),
            OS5_max_onset=on.OS_pct_5s.max(), OS5_med_restep=re_.OS_pct_5s.median(),
            OS5_max_restep=re_.OS_pct_5s.max(),
            n_os5_gt5=int((L.OS_pct_5s > 5).sum()), n_os5_gt10=int((L.OS_pct_5s > 10).sum()),
            OS30_med=L.OS_pct_30s.median(), OS30_max=L.OS_pct_30s.max(),
            OS30_med_onset=on.OS_pct_30s.median(),
            n_os30_gt5=int((L.OS_pct_30s > 5).sum()),
            US5_med=L.US_pct_5s.median(), US5_max=L.US_pct_5s.max(),
            devmid_med_adc=L.dev_mid_max_adc.median(), devmid_max_adc=L.dev_mid_max_adc.max(),
            T_med=L.T_stable.median(), T_p90=L.T_stable.quantile(.90),
            T_med_onset=on.T_stable.median(),
            T10_med=L.T_stable10.median(), T10_n=int(L.T_stable10.notna().sum()),
            T30_med=L.T_stable30.median(), T30_n=int(L.T_stable30.notna().sum()),
            err1_med=L.err_1s_pct.median(), err1_abs_med=L.err_1s_pct.abs().median(),
            err1_med_onset=on.err_1s_pct.median(),
            err1_abs_med_onset=on.err_1s_pct.abs().median(),
            err1_abs_med_adc=ad.err_1s_pct.abs().median() if len(ad) else np.nan,
            err1_abs_med_hz=hz.err_1s_pct.abs().median() if len(hz) else np.nan,
            MD_med=L.MD.median(), MD_p90=L.MD.quantile(.90),
            G_med=L.G20.median(), G_min=L.G20.min(),
            OS5_med_hz=hz.OS_pct_5s.median() if len(hz) else np.nan,
            OS5_max_hz=hz.OS_pct_5s.max() if len(hz) else np.nan,
            OS5_med_adc=ad.OS_pct_5s.median() if len(ad) else np.nan,
            OS5_max_adc=ad.OS_pct_5s.max() if len(ad) else np.nan,
        ))
    S = pd.DataFrame(sm)
    base = A[A.arm == "v6"].set_index(["key", "t_on", "kind"])
    d = []
    for arm, g in A.groupby("arm", sort=False):
        gi = g.set_index(["key", "t_on", "kind"])
        common = gi.index.intersection(base.index)
        dOS5 = gi.loc[common, "OS_pct_5s"] - base.loc[common, "OS_pct_5s"]
        dOS30 = gi.loc[common, "OS_pct_30s"] - base.loc[common, "OS_pct_30s"]
        dT = gi.loc[common, "T_stable"] - base.loc[common, "T_stable"]
        dE = gi.loc[common, "err_1s_pct"] - base.loc[common, "err_1s_pct"]
        d.append(dict(arm=arm, n_pair=len(common),
                      dOS5_med=dOS5.median(), dOS5_absmax=dOS5.abs().max(),
                      dOS5_worst=dOS5.max(), dOS5_best=dOS5.min(),
                      dOS30_med=dOS30.median(), dOS30_absmax=dOS30.abs().max(),
                      dT_med=dT.median(), dT_absmax=dT.abs().max(),
                      dE_med=dE.median(), dE_absmax=dE.abs().max(),
                      n_OS5_identical=int((dOS5.abs() < 1e-12).sum())))
    D = pd.DataFrame(d)
    S = S.merge(D, on="arm", how="left")
    S.to_csv(os.path.join(C.TASK, "results", "t5a_v6_vs_v61_summary.csv"),
             index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_v6_vs_v61_summary.csv")

    # ── 对照打印 ──
    lines = ["\n===== T5A-Q3 v6 vs v6.1（加载类 J_frac≥0.05 主口径）=====",
             "OS5 = 5 s 窗瞬态过充（与 08-v6.1 台账可比）；OS30 = 30 s 窗（含慢相平台偏差）"]
    lines.append(f"{'arm':11s} {'OS5_med':>8s} {'OS5_p90':>8s} {'OS5_max':>8s} "
                 f"{'>5%':>4s} {'>10%':>5s} {'OS30_med':>9s} {'US5_max':>8s} "
                 f"{'T30_med':>8s} {'T10_med':>8s} {'err1|med|':>10s} {'G_med':>6s} "
                 f"{'MD_med':>9s}")
    for _, r in S.iterrows():
        lines.append(f"{r['arm']:11s} {r.OS5_med:8.2f} {r.OS5_p90:8.2f} {r.OS5_max:8.2f} "
                     f"{r.n_os5_gt5:4d} {r.n_os5_gt10:5d} {r.OS30_med:9.2f} "
                     f"{r.US5_max:8.2f} {r.T30_med:8.3f} {r.T10_med:8.3f} "
                     f"{r.err1_abs_med:10.3f} {r.G_med:6.3f} {r.MD_med:9.1f}")
    lines.append("\n分域 1 s 误差（|中位|%）：")
    for _, r in S.iterrows():
        lines.append(f"  {r['arm']:11s} 显示域 {r.err1_abs_med_hz:8.3f}   "
                     f"ADC 域 {r.err1_abs_med_adc:8.3f}")
    lines.append("\n与 v6 的配对差（OS5；正 = 比 v6 更过充）：")
    for _, r in S.iterrows():
        lines.append(f"  {r['arm']:11s} dOS5_med={r.dOS5_med:8.3f}  "
                     f"dOS5_absmax={r.dOS5_absmax:8.3f}  dOS5_best={r.dOS5_best:9.3f}  "
                     f"dOS30_med={r.dOS30_med:8.3f}  dT_med={r.dT_med:7.3f}  "
                     f"dE_med={r.dE_med:8.3f}  逐事件 OS5 相同 {int(r.n_OS5_identical)}/{int(r.n_pair)}")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_v61.log"),
                "python scripts/t5a_v6_vs_v61.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
