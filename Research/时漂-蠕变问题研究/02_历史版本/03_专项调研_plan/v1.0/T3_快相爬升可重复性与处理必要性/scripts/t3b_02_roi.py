# -*- coding: utf-8 -*-
"""t3b_02_roi.py —— T3-Q8：反事实「只做部分处理」的 ROI 曲线与拐点。

扫描维度（**每条臂相对 v6 默认只改一个参数**，见 `t3b_arms.ARMS` 的 group='roi'）：
  ① 形状库快慢缩放 `rom_scale`（v6.1 的 1.06 是其中一点）     → rs100/rs104/rs108/rs112
  ② κ 上限（**单侧上限 min(A, κ·inc)**，只有跨过触发阈值才起作用）→ kap125/kap115/kap100/kap095
  ③ 滑行器速率上限 `RATE_MAX`（越小 = 把过充摊得越平）          → gl030/gl050/gl160/gl300
  ④ 只对 onset 生效（事件过滤模拟）                             → onset_only
另含 v61_rs106（= ①的 1.06 点，来自 Q7 的路线臂）作对照。

产出：results/t3b_route_roi.csv（含 rom_scale/kappa/trigger_rate/glide_rate/onset_only 各维
      + 四联指标）、results/_t3b_02_roi.log；并把拐点写入 results/t3b_roi_knee.csv

运行：python scripts/t3b_02_roi.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_arms as A     # noqa: E402
import t3b_tables as T   # noqa: E402
import t3b_settle as ST  # noqa: E402
import t3b_common as C   # noqa: E402

pd.set_option("display.width", 250)


def knee(x, y):
    """Kneedle 式拐点：把 (x, y) 归一化后取「离首末连线最远」的点，返回其 x（原量纲）。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return np.nan
    o = np.argsort(x)
    x, y = x[o], y[o]
    rx = float(np.ptp(x))
    ry = float(np.ptp(y))
    if rx <= 1e-12 or ry <= 1e-12:
        return float(x[len(x) // 2])          # 曲线本来就平 ⇒ 拐点无意义，返回中点
    xn = (x - x.min()) / rx
    yn = (y - y.min()) / ry
    d = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
    d /= max(np.hypot(yn[-1] - yn[0], xn[-1] - xn[0]), 1e-12)
    return float(x[int(np.argmax(d))])


def dim_table(roi, arm_names, strength, label, unit):
    """按"处理强度"排序输出一个维度的四联指标表，并给拐点。"""
    s = roi[roi.arm.isin(arm_names)].copy()
    s["strength"] = [strength[a] for a in s.arm]
    s = s.sort_values("strength")
    cols = ["arm", "strength", "rom_scale", "kappa_onset", "trigger_rate", "glide_rate",
            "onset_only", "Tstab_tot_ev_med", "Tstab_ch_ev_med", "OS_ch_med", "OS_ch_p90",
            "err1_ch_med", "MD_tot_med", "MD_tot_max", "epoch_per100s", "G_med"]
    print("\n-- 维度：%s（强度 = %s）--" % (label, unit))
    print(s[cols].round(3).to_string(index=False))
    kt = knee(s.strength, s.Tstab_tot_ev_med)
    km = knee(s.strength, s.MD_tot_med)
    ko = knee(s.strength, s.OS_ch_med)
    print("   拐点：T_stable(强度)=%.3f   MD(强度)=%.3f   OS(强度)=%.3f" % (kt, km, ko))
    return dict(dim=label, n_arms=len(s),
                strength_range="%.3f~%.3f" % (s.strength.min(), s.strength.max()),
                knee_Tstable=kt, knee_MD=km, knee_OS=ko,
                Tstab_at_knee=float(np.interp(kt, s.strength, s.Tstab_tot_ev_med)),
                MD_at_knee=float(np.interp(kt, s.strength, s.MD_tot_med)),
                OS_at_knee=float(np.interp(kt, s.strength, s.OS_ch_med)))


def main():
    ST.start_log(C.RES, "02_roi")
    t0 = time.time()
    ev, cache = A.load_all()
    print("Q8 反事实：只做部分处理会怎样（处理强度 vs T_stable vs 代价）")
    print("每臂相对 v6 默认只改一个参数；κ 为**单侧上限** min(A, κ·inc)，故必须同时报「触发率」")
    print("（= Â 被上限截断的事件占比，C-4 的关键量）。\n")
    A.run_group("roi", ev, cache, force=("--force" in sys.argv))
    roi, dd, g = T.build_all()

    rs = ["v6", "rs104", "rs108", "rs112", "v61_rs106"]
    kap = ["v6", "kap125", "kap115", "kap100", "kap095"]
    gl = ["gl030", "gl050", "v6", "gl160", "gl300"]
    on = ["v6", "onset_only"]
    rows = []
    rows.append(dim_table(roi, rs, {a: float(roi[roi.arm == a].rom_scale.iloc[0]) for a in rs},
                          "① 形状库快慢缩放（rom_scale，>1 = 更早到位 = 更保守）", "g(0.2 s)/g_v6(0.2 s)"))
    rows.append(dim_table(roi, kap, {a: 1.30 - float(roi[roi.arm == a].kappa_onset.iloc[0])
                                     for a in kap},
                          "② κ 单侧上限（强度 = 1.30 − κ_onset；越大 = 封顶越紧）", "1.30 − κ"))
    rows.append(dim_table(roi, gl, {a: -float(roi[roi.arm == a].glide_rate.iloc[0]) for a in gl},
                          "③ 滑行器速率上限（强度 = −RATE_MAX；越大 = 摊得越平、到位越慢）",
                          "−RATE_MAX [/s]"))
    rows.append(dim_table(roi, on, {a: float(roi[roi.arm == a].onset_only.iloc[0]) for a in on},
                          "④ 只对 onset 生效（0 = 全部事件反演，1 = 仅 onset）", "onset_only"))
    knee_df = pd.DataFrame(rows)
    knee_df.to_csv(os.path.join(C.RES, "t3b_roi_knee.csv"), index=False, encoding="utf-8-sig")

    print("\n-- 触发率明细（κ 维；触发率 = 被 κ·inc 截断的 Â 次数 / 成功反演次数）--")
    print(roi[roi.arm.isin(kap)][["arm", "kappa_onset", "kappa_restep", "inv_calls", "n_cap",
                                  "trigger_rate"]].round(4).to_string(index=False))
    print("\n-- 全维一览（含 Q7 路线臂）--")
    allc = ["arm", "group", "dim", "dim_value", "Tstab_tot_ev_med", "Tstab_ch_ev_med",
            "OS_ch_med", "OS_ch_p90", "err1_ch_med", "MD_tot_med", "MD_tot_max",
            "epoch_per100s", "trigger_rate"]
    print(roi[allc].round(3).to_string(index=False))
    print("\n总耗时 %.1f s" % (time.time() - t0))
    print("-> results/t3b_route_roi.csv / t3b_roi_knee.csv / t3b_route_q6q7.csv / t3b_arm_kind.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
