# -*- coding: utf-8 -*-
"""T1-B / 03：事件集与**无扰动基线**（产物：results/t1b_events.csv、t1b_baseline_arms.csv）。

事件集来源：`T4_两种快相形态与分支判据/results/t4a_morphology.csv`（T4-A 冻结的 60 事件表，
只读引用）中 `kind=onset, clean=True` 且事件后 ≥35 s 无其它事件的行 —— **不重做事件检测**。
基线：raw / v5.1 / v6 / v6.1 四条臂在**无扰动**输入上的 T_stable、超调、T_settle。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

import t1b_lib as L          # noqa: E402
from t1_common import med_smooth, overshoot, t_stable, z_final_level   # noqa: E402


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


CROP_S = 80.0        # 事件后分析窗（t_on + 80 s；含 Z_final 的 t_on+60~70 s）


def window(d, ev):
    """返回 (tu, Xu)：显示域裁到 [0, t_on+80]（≈ 全前缀，warm-up 与全长一致）；ADC 域用全长。"""
    t_on = float(ev["t_on"])
    if ev["dom"] == "显示域":
        return L.crop(d["tu"], d["Xu"], 0.0, t_on + CROP_S)
    return d["tu"], d["Xu"]


def main():
    log = Tee(os.path.join(RES, "_t1b_03_events.log"))
    log("=== T1-B 事件集与无扰动基线 ===")
    ev = L.event_table()
    ev.to_csv(os.path.join(RES, "t1b_events.csv"), index=False, encoding="utf-8-sig")
    log(f"事件表 {len(ev)} 行；可测 T_stable 的 {int(ev.measurable.sum())} 个")
    log(ev[["ev", "dom", "t_on", "gap_next", "span_after", "J", "L_hold", "measurable"]]
        .to_string(index=False))

    rows = []
    for _, e in ev.iterrows():
        d = L.get_grid(e["key"])
        tu, Xu = window(d, e)
        t_on = float(e["t_on"])
        Zraw = Xu.sum(axis=1)
        J = float(e["J"])
        t_next = float(e["t_next"])
        for arm in ("raw", "v51", "v6", "v61"):
            r = L.run_arm(arm, tu, Xu)
            Z = r["Z"]
            m = L.eval_event(tu, Z, t_on, J, t_next)
            zf, src = z_final_level(tu, Z, t_on)
            osd = overshoot(tu, Z, t_on, J, z_final=zf)
            tb = med_smooth(Z, 50)[int(round((t_on + 5) / 0.01))]
            rows.append(dict(ev=e["ev"], key=e["key"], dom=e["dom"], t_on=t_on, arm=arm,
                             J=J, L_hold=e["L_hold"], gap_next=e["gap_next"],
                             t_stable_v1=m["t_stable_v1"], t_stable_v2=m["t_stable_v2"],
                             t_stable_ok=m["t_stable_ok"], os_pct=osd["os_pct"],
                             us_pct=osd["us_pct"], ts2=m["ts2"], ts5=m["ts5"],
                             z_at_1=m["z_at_1"], z_at_2=m["z_at_2"],
                             z_final=zf, zfin_src=src, z_display_5s=tb - zf,
                             n_epoch=len(r["epoch"]), n_revoke=len(r["revoke"])))
            del r
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t1b_baseline_arms.csv"), index=False, encoding="utf-8-sig")

    log("--- 无扰动基线（按臂汇总，全部事件 n=%d；中位 [p10~p90]）---" % len(ev))
    for arm in ("raw", "v51", "v6", "v61"):
        s = df[df.arm == arm]
        me = s[s.t_stable_ok == 1]
        q = lambda c: np.nanpercentile(me[c].dropna(), [10, 50, 90]) if len(me) else [np.nan] * 3
        t1, t2, t3 = q("t_stable_v1")
        o1, o2, o3 = np.nanpercentile(s.os_pct.dropna(), [10, 50, 90]) if len(s) else [np.nan] * 3
        log(f"  {arm:4s} n_meas={len(me):2d}/{len(s)}  T_stable_v1 med={t2:.2f}s [{t1:.2f}~{t3:.2f}]"
            f"  OS% med={o2:.2f} [{o1:.2f}~{o3:.2f}]  epoch med={s.n_epoch.median():.1f}")
    log("  —— 达标率（T_stable_v1 ≤ 2 s，仅可测事件）：")
    bad = df[(df.t_stable_ok == 1) & (df.arm.isin(["v6", "v61", "v51", "raw"]))]
    for arm, s in bad.groupby("arm"):
        log(f"     {arm:4s}: {int((s.t_stable_v1 <= 2.0).sum())}/{len(s)}")
    log.f.close()


if __name__ == "__main__":
    main()
