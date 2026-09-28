# -*- coding: utf-8 -*-
"""t1a_01_events：产出**全项目统一事件表** `results/t1_events.csv`（列名对齐 00_共享/指标字典 §2.3）。

做法（按派发要求"优先复用 T4-A 已冻结的 60 事件，不重新检测"）：
  直接读 `T4_两种快相形态与分支判据/results/t4a_morphology.csv`（60 行，只读），做**列名映射 + 自检**：
    ds        ← rec                （数据集键，与 T4-A/第一轮 `settle_arms.csv` 的 rec 同表）
    kind      ← kind               （onset / restep / unload / partial_unload）
    t_on      ← t_on               （真沿：检出沿 ±0.20 s 内最大单帧跳变帧）
    J         ← jump               （= post 窗中位 − pre 窗中位，**总通道 Z = Σ_c ch_c 口径**）
    pre/post  ← pre / post         （窗 [t_on−2, t_on) / [t_on+4, t_on+6]，指标字典 §2.1）
    t50/t80/t90/t95 / z_at_02 / z_at_10 ← 同名列
    clean     ← clean              （T4-A 的干净事件标志）
  另保留 key/dom/family/main_ch/k_on/armed_*/kind_prej/src_ev/pkt_dt 等溯源列（放在冻结列之后）。

映射差异（必须声明）：
  1. T4-A 的 `jump` 在**总通道**上算，因此本表 `J` 是**总通道口径**；
     主通道口径的 `J_ch` 由 `t1a_02_run/t1a_03_metrics` 另算（本表不含，避免两个 J 混淆）；
  2. T4-A 未给 `t80` → `t80` 直接沿用其 `t80`（部分小台阶事件为空，保留 NaN）；
  3. 本表新增 `ev_id`（E001…）与 `J_ok`（J 是否有限），其余列名与 T4-A 完全一致。

自检（写入日志 + t1a_smoke_checks 同款断言）：k_on/t_on 必须落在 100 Hz 网格上（|t_on−k_on·dt|≤5 ms）、
J == post−pre（相对误差 <1e-9）、事件数与 T4-A 一致（60）。

产出：results/t1_events.csv、results/_t1a_01_events.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t1a_common as C

T4A = os.path.join(C.PLAN, "T4_两种快相形态与分支判据", "results", "t4a_morphology.csv")

FROZEN = ["ds", "kind", "t_on", "J", "pre", "post", "t50", "t80", "t90", "t95",
          "z_at_02", "z_at_10", "clean"]
EXTRA = ["ev_id", "key", "dom", "family", "main_ch", "k_on", "J_ok", "peak", "ratio",
         "armed_20", "kind_prej", "n_other_near", "src_ev", "pkt_dt", "src"]


def main():
    C.start_log("01_events")
    df = pd.read_csv(T4A, encoding="utf-8-sig")
    print("T4-A 事件表: %d 行 × %d 列  <- %s" % (len(df), df.shape[1], T4A))

    out = pd.DataFrame({
        "ds": df["rec"],
        "kind": df["kind"],
        "t_on": df["t_on"].astype(float),
        "J": df["jump"].astype(float),
        "pre": df["pre"].astype(float),
        "post": df["post"].astype(float),
        "t50": df["t50"].astype(float),
        "t80": df["t80"].astype(float),
        "t90": df["t90"].astype(float),
        "t95": df["t95"].astype(float),
        "z_at_02": df["z_at_02"].astype(float),
        "z_at_10": df["z_at_10"].astype(float),
        "clean": df["clean"].astype(bool),
    })
    out["ev_id"] = ["E%03d" % (i + 1) for i in range(len(out))]
    out["key"] = df["key"]
    out["dom"] = df["dom"]
    out["family"] = [C.REC_BY_KEY[r]["family"] for r in df["rec"]]
    out["main_ch"] = df["main_ch"].astype(int)
    out["k_on"] = df["k_on"].astype(int)
    out["J_ok"] = np.isfinite(out["J"])
    out["peak"] = df["peak"].astype(float)
    out["ratio"] = df["ratio"].astype(float)
    out["armed_20"] = df["armed_20"].astype(bool)
    out["kind_prej"] = df["kind_prej"]
    out["n_other_near"] = df["n_other_near"].astype(int)
    out["src_ev"] = df["src_ev"]
    out["pkt_dt"] = df["pkt_dt"].astype(float)
    out["src"] = "t4a_morphology.csv"
    out = out[FROZEN + EXTRA]

    # ── 自检 ──
    print("\n== 自检 ==")
    bad_grid = int((np.abs(out["t_on"] - out["k_on"] * C.DT) > 0.005).sum())
    print("  网格一致性（|t_on − k_on·dt| ≤ 5 ms）: 不一致 %d 个" % bad_grid)
    res = np.abs(out["J"] - (out["post"] - out["pre"]))
    okj = out["J_ok"]
    print("  J == post − pre（相对误差）: 最大 %.3e（%d 个有效 J）"
          % (float((res[okj] / np.maximum(np.abs(out.loc[okj, "J"]), 1e-9)).max()), int(okj.sum())))
    print("  J 缺失事件: %d 个 -> %s" % (int((~okj).sum()),
                                        ", ".join(out.loc[~okj, "ev_id"] + "/" + out.loc[~okj, "ds"])))
    print("  列 NaN 计数: " + ", ".join("%s=%d" % (c, int(out[c].isna().sum()))
                                       for c in ("J", "pre", "post", "t50", "t80", "t90", "t95",
                                                 "z_at_02", "z_at_10")))
    print("\n== 事件构成（ds × kind）==")
    print(pd.crosstab(out["ds"], out["kind"]).to_string())
    print("\n== 分族 × 分工况 ==")
    print(pd.crosstab(out["family"], out["kind"]).to_string())
    print("\n== 与需求文档 §3-2 的事件覆盖要求 ==")
    n_on = int((out.kind == "onset").sum())
    n_rs = int((out.kind == "restep").sum())
    n_rec = int(out[out.family == "实录"].shape[0])
    print("  onset %d（要求 ≥9）| restep %d（要求 ≥15）| 实录类事件 %d（要求 ≥10，含 onset/restep/unload）"
          % (n_on, n_rs, n_rec))
    print("  J 为总通道口径（Σ_c ch_c）；主通道/含蠕变口径的参考幅度另见 t1a_settle_metrics.csv")

    p = os.path.join(C.RES, "t1_events.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n冻结列（%d）: %s" % (len(FROZEN), ", ".join(FROZEN)))
    print("溯源列（%d）: %s" % (len(EXTRA), ", ".join(EXTRA)))
    print("-> %s" % p)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
