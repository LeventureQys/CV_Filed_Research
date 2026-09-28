# -*- coding: utf-8 -*-
"""T1-B / 06b：拍击的**可归因恢复时间**（对照组差分口径）。

t1b_06 的 `t_recover` 是"显示回到拍击前电平 ±2%"——在慢相还在爬的保压段上，
电平本身就在动，这个判据会大面积返回 NaN（无法归因）。本脚本改用**对照差分**：
  Δ(t) = Z_有拍击(t) − Z_无拍击同段(t)
  t_recover_ctrl = Δ 首次回到 |Δ| ≤ 2%·L_hold 并保持 3 s 的时间
这彻底去掉了慢相漂移的干扰，是"拍击后多久恢复"的干净口径。

产物：results/t1b_tap_recovery_ctrl.csv、results/t1b_tap_traj.npz、results/_t1b_06b.log
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

import t1b_lib as L                                   # noqa: E402
from t1_common import add_tap, med_smooth              # noqa: E402

FRACS = [0.10, 0.30, 0.50]
ARMS = ["v6", "v61"]
TAP = dict(rise_ms=50.0, hold_ms=100.0, fall_ms=50.0)


def main():
    logp = os.path.join(RES, "_t1b_06b.log")
    f = open(logp, "a", encoding="utf-8")

    def w(s):
        print(s)
        f.write(s + "\n")

    w("=== T1-B / 06b 拍击-恢复（对照差分口径）===")
    ev = L.event_table()
    hold = ev[ev.dom == "显示域"]
    rows, traj = [], {}
    for _, e in hold.iterrows():
        key, t_on, Lh = e["key"], float(e["t_on"]), float(e["L_hold"])
        d = L.get_grid(key)
        tu, Xu = L.crop(d["tu"], d["Xu"], 0.0, t_on + 80.0)
        t_tap = t_on + 25.0
        i_tap = int(round(t_tap / 0.01))
        Zc = {}
        for arm in ARMS:
            Zc[arm] = L.run_arm(arm, tu, Xu)["Z"]
        for frac in FRACS:
            Xs, _ = add_tap(Xu, i_tap, frac * Lh, **TAP)
            for arm in ARMS:
                Z = L.run_arm(arm, tu, Xs)["Z"]
                dZ = Z - Zc[arm]
                band = 0.02 * abs(Lh)
                a = i_tap + int(0.3 / 0.01)
                inside = np.abs(dZ) <= band
                t_rec = np.nan
                for i in range(a, min(len(Z), i_tap + int(40 / 0.01))):
                    j = min(len(Z), i + int(3.0 / 0.01))
                    if inside[i:j].all():
                        t_rec = float(tu[i] - t_tap)
                        break
                seg = slice(i_tap, min(len(Z), i_tap + int(10 / 0.01)))
                rows.append(dict(key=key, ev=e["ev"], arm=arm, frac=frac, L_hold=Lh,
                                 amp=frac * Lh,
                                 peak_d=float(dZ[seg].max()), peak_pct=100 * float(dZ[seg].max()) / Lh,
                                 min_d=float(dZ[seg].min()),
                                 t_recover_ctrl=t_rec,
                                 area_pct_s=float(np.trapezoid(np.abs(dZ[seg]), tu[seg]) / Lh),
                                 resid_med=float(np.median(dZ[i_tap + int(30 / 0.01):
                                                              min(len(Z), i_tap + int(40 / 0.01))]))
                                 if len(Z) > i_tap + int(40 / 0.01) else np.nan))
                if abs(frac - 0.30) < 1e-9:
                    traj[f"{key}|{arm}"] = dZ
        if key == "RT2" or key == "LT1":
            traj[f"{key}|tu"] = tu
            traj[f"{key}|ref"] = Zc["v6"]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t1b_tap_recovery_ctrl.csv"), index=False,
              encoding="utf-8-sig")
    np.savez_compressed(os.path.join(RES, "t1b_tap_traj.npz"), **traj)
    w(df.groupby(["arm", "frac"]).agg(
        peak_pct_med=("peak_pct", "median"), peak_pct_p90=("peak_pct", lambda s: np.percentile(s, 90)),
        t_rec_med=("t_recover_ctrl", "median"),
        t_rec_p90=("t_recover_ctrl", lambda s: np.nanpercentile(s.dropna(), 90) if s.notna().any() else np.nan),
        n_rec=("t_recover_ctrl", lambda s: int(s.notna().sum())),
        resid_med=("resid_med", "median"), area_med=("area_pct_s", "median")).round(3).to_string())
    w(f"行数 {len(df)} -> t1b_tap_recovery_ctrl.csv; 轨迹 -> t1b_tap_traj.npz")
    f.close()


if __name__ == "__main__":
    main()
