# -*- coding: utf-8 -*-
"""T1-B / 03b：口径对齐自检 —— 与 T1-A 冻结口径（T_stable_ch5，主通道）的差异。

T1-B 全部扫描用**总通道口径**（`00_共享/指标字典与口径.md` §1「默认使用总量做主结论」）；
T1-A 冻结的是**主通道口径**。本脚本给出两者在**无扰动基线**上的偏移，
说明 T1-B 的结论是对同一口径的**配对比较**（噪声/抖动/拍击 vs 无扰动），不受该偏移影响。

产物：results/t1b_caliber_check.csv、results/_t1b_03b_caliber.log
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

import t1b_lib as L                      # noqa: E402
from t1_common import med_smooth, t_stable, z_final_level, overshoot   # noqa: E402


def main():
    logp = os.path.join(RES, "_t1b_03b_caliber.log")
    rows = []
    ev = L.event_table()
    for _, e in ev[ev.dom == "显示域"].iterrows():
        d = L.get_grid(e["key"])
        tu, Xu = L.crop(d["tu"], d["Xu"], 0.0, float(e["t_on"]) + 80.0)
        t_on, J = float(e["t_on"]), float(e["J"])
        ch = int(e["main_ch"])
        for arm in ("raw", "v51", "v6", "v61"):
            r = L.run_arm(arm, tu, Xu)
            # J 必须按**同一信号**算（T1-A 冻结口径：J_ref 取自主通道自己）
            for cal, Z, Jc in (("total", r["Z"], J),
                               ("main_ch", r["Y"][:, ch],
                                L.step_amp(tu, Xu[:, ch], t_on)),
                               # 去平滑变体：T1-A 冻结口径**不做**评估前平滑；本任务按
                               # 指标字典 §1 用 Z̄ = median(Z, 0.5 s)。加这两档只为定位差异。
                               ("total_raw", r["Z"], J),
                               ("main_ch_raw", r["Y"][:, ch],
                                L.step_amp(tu, Xu[:, ch], t_on))):
                sm = 0.001 if cal.endswith("_raw") else 0.5
                ts = t_stable(tu, Z, t_on, Jc, smooth_s=sm)
                zf, src = z_final_level(tu, Z, t_on)
                osd = overshoot(tu, Z, t_on, Jc, z_final=zf)
                rows.append(dict(ev=e["ev"], key=e["key"], arm=arm, caliber=cal, ch=ch,
                                 J_used=Jc, t_stable=ts["tau_v1"], ok=int(ts["ok"]),
                                 os_pct=osd["os_pct"], zfin_src=src))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t1b_caliber_check.csv"), index=False, encoding="utf-8-sig")

    # 追加：把 T4-A 表里**另两个 onset**（RT1@8.27、F41@10.94，因"onset 后 5 s 内又有 restep"
    # 在本任务被判为 non-clean 而排除）也算上，检验"9 事件 vs 7 事件"是否解释与 T1-A 的差。
    extra = []
    t4a = pd.read_csv(L.T4A_EVENTS)
    for _, r in t4a[t4a.kind == "onset"].iterrows():
        if bool(r["clean"]):
            continue
        key, _ = L.REC_MAP[r["rec"]]
        d = L.get_grid(key)
        t_on = float(r["t_on"])
        tu, Xu = L.crop(d["tu"], d["Xu"], 0.0, t_on + 80.0)
        ch = int(r["main_ch"])
        for arm in ("raw", "v51", "v6", "v61"):
            rr = L.run_arm(arm, tu, Xu)
            for cal, Z, Jc in (("total", rr["Z"], L.step_amp(tu, Xu.sum(axis=1), t_on)),
                               ("main_ch", rr["Y"][:, ch], L.step_amp(tu, Xu[:, ch], t_on))):
                ts = t_stable(tu, Z, t_on, Jc)
                extra.append(dict(ev=f"{key}@{t_on:.2f}(non-clean)", key=key, arm=arm,
                                  caliber=cal, ch=ch, J_used=Jc, t_stable=ts["tau_v1"],
                                  ok=int(ts["ok"]), os_pct=np.nan, zfin_src="-"))
    dfx = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    dfx.to_csv(os.path.join(RES, "t1b_caliber_check.csv"), index=False, encoding="utf-8-sig")
    logp = os.path.join(RES, "_t1b_03b_caliber.log")
    with open(logp, "a", encoding="utf-8") as f:
        def w(s):
            print(s)
            f.write(s + "\n")
        w("=== T1-B / 03b 口径对齐（总通道 vs 主通道）===")
        w(f"{'arm':5s} {'caliber':8s} {'n_meas':>6s} {'T_med':>7s} {'T_p90':>7s} {'OS%med':>7s}")
        for (arm, cal), g in df.groupby(["arm", "caliber"]):
            m = g[g.ok == 1]
            w(f"{arm:5s} {cal:8s} {len(m):6d} {m.t_stable.median() if len(m) else float('nan'):7.2f} "
              f"{np.nanpercentile(m.t_stable, 90) if len(m) else float('nan'):7.2f} "
              f"{g.os_pct.median():7.2f}")
        piv = df[df.ok == 1].pivot_table(index=["ev", "arm"], columns="caliber",
                                         values="t_stable")
        if {"total", "main_ch"}.issubset(piv.columns):
            w("--- 逐事件（可测）总通道 vs 主通道 ---")
            w(piv.round(2).to_string())
            w(f"配对中位差(主通道−总通道) = "
              f"{(piv['main_ch'] - piv['total']).median():.2f} s；"
              f"n={len(piv)}")
        w("说明：T1-A 冻结口径 = 主通道 + 字典 §2.1 的 J_ref（`T_stable_ch5`）；"
          "T1-B 全部扫描用总通道（字典 §1 默认），二者绝对值不可横比，"
          "但 T1-B 的每条结论都是同口径下的**配对比较**（扰动 vs 无扰动）。")
        w("--- 加上 T4-A 的另 2 个 non-clean onset 后（= T1-A 的『恒载 9 组 onset』口径）---")
        for (arm, cal), g in dfx.groupby(["arm", "caliber"]):
            m = g[g.ok == 1]
            if len(m) == 0:
                continue
            w(f"  {arm:5s} {cal:8s} n={len(m):2d} T_med={m.t_stable.median():7.2f} "
              f"T_p90={np.nanpercentile(m.t_stable, 90):7.2f}")
        w("  对照 T1-A（t1a_settle_summary.csv）：v6 主通道 1.80 s / 总通道 0.55 s")


if __name__ == "__main__":
    main()
