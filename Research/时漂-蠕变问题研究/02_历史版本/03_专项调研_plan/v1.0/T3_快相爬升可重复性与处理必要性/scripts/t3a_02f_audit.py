# -*- coding: utf-8 -*-
"""T3-A 事件级"可用性"审计（Q1/Q4 的样本边界，先于统计运行）。

问题：归一化形状 f(τ) = (Z̄(t_on+τ) − pre)/J 在少数事件上是**病态的**——
      J 极小（低于指标字典 §2.1 的 2% 峰值门限）、或 f(τ) 出现负值/>1.5
      （前事件复合、基线在动、"unload 后加压"、多级加载、截尾）。
      这些事件一旦进入 std/mean 或单点反演 Â=Z(τ)/f̄(τ)，会把统计量整体带崩。

本脚本产出 `results/t3a_inlier_audit.csv`，给出每个装载事件的三个布尔判据与统计用样本标签：
    J_ok    : |J| ≥ 2% × 记录峰值（指标字典 §2.1 的幅度项）
    posdef  : 主 τ 网格上 0 ≤ f(τ) ≤ 1.5（形状可用于比值反演）
    flags   : 三者组合出的样本标签（all / J_ok / posdef / usable）
    usable  = J_ok & posdef & clean  ← Q1/Q2/Q4 的**主样本**

运行：python scripts/t3a_02f_audit.py   （秒级，读两个 csv）
"""
import os

import numpy as np
import pandas as pd

import t3a_common as C


def build_audit():
    grid = pd.read_csv(os.path.join(C.RES, "t3a_shape_grid.csv"))
    ev = C.load_t4a_events()
    ev["uid"] = ev["key"] + "@" + ev["t_on"].map(lambda x: "%.2f" % x)
    m = (grid[grid["grid"] == C.TAU_MAIN_LABEL]
         .pivot_table(index=["key", "t_on"], columns="tau", values="f"))
    rows = []
    for _, e in ev.iterrows():
        k = (e["key"], e["t_on"])
        if k in m.index:
            r = m.loc[k]
            fmin, fmax = float(np.min(r)), float(np.max(r))
            neg = bool((r < 0).any())
        else:
            fmin = fmax = np.nan
            neg = False
        J = float(e["jump"]) if np.isfinite(e["jump"]) else np.nan
        P = float(e["peak"])
        Jok = bool(np.isfinite(J) and abs(J) >= 0.02 * abs(P))
        posdef = bool(np.isfinite(fmin) and not neg and fmax <= 1.5)
        rows.append(dict(uid=e["uid"], key=e["key"], rec=e["rec"], dom=e["dom"], fam=e["fam"],
                         kind=e["kind"], is_load=bool(e["is_load"]), clean=bool(e["clean"]),
                         t_on=float(e["t_on"]), pre=float(e["pre"]),
                         post=float(e["post"]) if np.isfinite(e["post"]) else np.nan,
                         jump=J, peak=P, J_over_peak=(abs(J) / abs(P) if np.isfinite(J) and P else np.nan),
                         J_ok=Jok, posdef=posdef, neg=neg, f_min=fmin, f_max=fmax,
                         usable=bool(Jok and posdef and bool(e["clean"]) and bool(e["is_load"])),
                         sample_usable_incl_dirty=bool(Jok and posdef and bool(e["is_load"]))))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(C.RES, "t3a_inlier_audit.csv"), index=False, encoding="utf-8-sig")
    return d


def main():
    C.start_log("t3a_02f_audit")
    d = build_audit()
    L = d[d["is_load"]]
    print("装载类事件 n=%d（onset %d / restep %d）" %
          (len(L), (L["kind"] == "onset").sum(), (L["kind"] == "restep").sum()))
    print("")
    print("判据汇总（按类）：")
    for kind in ["onset", "restep"]:
        s = L[L["kind"] == kind]
        print("  %-7s n=%-3d  J_ok=%-3d posdef=%-3d clean=%-3d usable=%-3d"
              % (kind, len(s), int(s["J_ok"].sum()), int(s["posdef"].sum()),
                 int(s["clean"].sum()), int(s["usable"].sum())))
    print("")
    bad = L[~L["posdef"]]
    print("非正定事件（f(τ) 含负值或 >1.5）n=%d —— 这些事件**不可用于比值反演**：" % len(bad))
    print(bad[["uid", "kind", "clean", "fam", "jump", "J_over_peak", "f_min", "f_max"]]
          .to_string(index=False))
    print("")
    print("J 低于 2%% 峰值的事件：%s" % list(L[~L["J_ok"]]["uid"]))
    print("")
    u = L[L["usable"]]
    print("主样本 usable（J_ok & posdef & clean & is_load）n=%d"
          "（onset %d / restep %d；域：显示 %d / ADC %d）"
          % (len(u), (u["kind"] == "onset").sum(), (u["kind"] == "restep").sum(),
             (u["dom"] == "显示域").sum(), (u["dom"] == "ADC域").sum()))
    print("-> results/t3a_inlier_audit.csv（%d 行）" % len(d))


if __name__ == "__main__":
    main()
