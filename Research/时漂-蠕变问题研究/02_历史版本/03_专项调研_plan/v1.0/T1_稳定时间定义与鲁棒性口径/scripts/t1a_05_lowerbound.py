# -*- coding: utf-8 -*-
"""t1a_05_lowerbound：T1-Q5——稳定时间的**理论下界**（"输入本身还没走完"）。

思路（三条，全部落到数字）：
  1. **等效输入斜坡 `T_ramp`**（T4-A 反卷积给出，只读引用 `t4a_input_recover.csv`）：
     输入自己走完所需时间。onset 是单帧跃变、restep 是"缓慢叠加" —— 两者的 `T_ramp` 分布就是
     "输入还没走完"的直接度量；
  2. **输入侧稳定时间 `T_in5`**：在**原始信号**上用与算法完全相同的冻结口径（D1-ev，5%×J、窗在
     下一事件处截断）算出的"输入首次进入自身 5%·J 带并保持"的时刻。对任何因果跟随器，
     `T_stable ≳ T_in5`；若某臂 `T_stable < T_in5`，说明它靠**预测/冻结**提前停下（代价是过冲）；
  3. **判据侧最小延迟**：v6 系逆模型的最早可用时刻 `TAU_REF=0.20 s` + 停滞确认 `STALL_HOLD_S=0.45 s`
     （常数取自 `t1a_glm53_v6.py`，逐字引用），给出"就算输入瞬时到位"的算法自延迟下界。

输入：results/t1a_settle_metrics.csv、results/t1_events.csv、results/cache/*.npz、
      T4-A/results/t4a_input_recover.csv（只读）
产出：results/t1a_lowerbound.csv、results/_t1a_05_lowerbound.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t1a_common as C

T4A_IN = os.path.join(C.PLAN, "T4_两种快相形态与分支判据", "results", "t4a_input_recover.csv")
T4B_JOIN = os.path.join(C.PLAN, "T4_两种快相形态与分支判据", "results", "t4b_tramp_join.csv")


def desc(v, name):
    s = C.fill_rate_summary(v)
    print("  %-28s n=%2d  中位 %7.3f   p10~p90 %7.3f~%7.3f   最差 %7.3f  %s"
          % (name, s["n"], s["med"], s["p10"], s["p90"], s["worst"], s["note"]))
    return s


def main():
    C.start_log("05_lowerbound")
    m = pd.read_csv(os.path.join(C.RES, "t1a_settle_metrics.csv"), encoding="utf-8-sig")
    ev = pd.read_csv(os.path.join(C.RES, "t1_events.csv"), encoding="utf-8-sig")
    m["kind3"] = m["kind"].replace({"partial_unload": "unload"})

    print("== 0. T4-A 输入反卷积表（只读引用） ==")
    ins = pd.read_csv(T4A_IN, encoding="utf-8-sig")
    print("  %s：%d 行，列=%s" % (os.path.basename(T4A_IN), len(ins),
                                  ",".join(ins.columns[:8]) + ",…"))
    ins = ins.rename(columns={"rec": "ds"})
    ins["t_on_r"] = ins["t_on"].round(2)

    print("\n== 1. Q5 事实基线：输入侧时间尺度（分工况，n=39 有反卷积的事件） ==")
    for kd in ("onset", "restep"):
        s = ins[ins.kind == kd]
        print(" %s（n=%d）" % (kd, len(s)))
        desc(s["T_ramp"], "等效输入斜坡 T_ramp (s)")
        desc(s["A_ramp"], "等效输入幅度 A_ramp")
    print("\n  [口径核对] T4-A 报告 §② 的「restep 中位 0.550 s」= **clean 子集**的中位；")
    for kd in ("onset", "restep"):
        s = ins[ins.kind == kd]
        sc = s[s.clean]
        print("    %-6s 全样本 n=%2d 中位 %.3f | clean n=%2d 中位 %.3f | T_ramp>0 的 n=%2d 中位 %.3f"
              % (kd, len(s), s.T_ramp.median(), len(sc), sc.T_ramp.median(),
                 int((s.T_ramp > 1e-9).sum()), s.loc[s.T_ramp > 1e-9, "T_ramp"].median()))
    print("  全部事件 T_ramp 中位 %.3f s（onset %.3f vs restep %.3f）"
          % (ins.T_ramp.median(), ins[ins.kind == "onset"].T_ramp.median(),
             ins[ins.kind == "restep"].T_ramp.median()))
    if os.path.isfile(T4B_JOIN):
        j = pd.read_csv(T4B_JOIN, encoding="utf-8-sig")
        cols = [c for c in j.columns if "tramp" in c.lower() or "T_ramp" in c]
        print("  [交叉核对] T4-B t4b_tramp_join.csv：%d 行，含 T_ramp 的列=%s" % (len(j), cols))
        for c in cols:
            if pd.api.types.is_numeric_dtype(j[c]):
                print("    %-14s 中位 %8.4f  n=%d" % (c, float(j[c].median()), int(j[c].notna().sum())))

    # ── 2. 逐事件：输入下界 vs 各臂实测 ──
    rows = []
    for ds, g in m.groupby("ds", sort=False):
        z = np.load(os.path.join(C.CACHE, "t1a_%s.npz" % ds.replace("/", "_")))
        tu, ytot, ych = z["tu"], z["yraw_tot"], z["yraw_ch"]
        evd = ev[ev.ds == ds]
        times = [(float(t), float(j)) for t, j in zip(evd.t_on, evd.J) if np.isfinite(j)]
        peak = float(np.max(ytot))
        ex = evd[evd.kind.isin(["onset", "restep"])]
        for _, e in ex.iterrows():
            k0 = int(e["k_on"])
            t_on = float(e["t_on"])
            cut = C.next_event_cut(times, t_on, peak)
            cut_k = None if cut is None else int(round(cut / C.DT))
            J_ch, pre_ch, _ = C.amp_5s(ych, k0)
            J_tot, _, _ = C.amp_5s(ytot, k0)
            t_in_ch, cen_ch, w_ch = C.t_stable_ev(tu, ych, k0, J_ch, cut_k)
            t_in_tot, cen_tot, w_tot = C.t_stable_ev(tu, ytot, k0, J_tot, cut_k)
            r = dict(ds=ds, key=e["key"], ev_id=e["ev_id"], family=e["family"], dom=e["dom"],
                     kind=e["kind"], clean=bool(e["clean"]), t_on=t_on, J_tot=e["J"],
                     t50=e["t50"], t90=e["t90"], t95=e["t95"], z_at_02=e["z_at_02"],
                     z_at_10=e["z_at_10"], win_avail_s=((cut_k if cut_k else len(tu)) * C.DT - t_on),
                     T_in5_raw_ch=t_in_ch, cens_in_ch=cen_ch,
                     T_in5_raw_tot=t_in_tot, cens_in_tot=cen_tot)
            r["T_ramp"] = np.nan
            hit = ins[(ins.ds == ds) & (np.abs(ins.t_on - t_on) < 0.011)]
            if len(hit):
                r["T_ramp"] = float(hit["T_ramp"].iloc[0])
                r["A_ramp"] = float(hit["A_ramp"].iloc[0])
                r["gain_ramp"] = float(hit["gain_ramp"].iloc[0])
            for arm in C.ARMS:
                sub = m[(m.ds == ds) & (m.ev_id == e["ev_id"]) & (m.arm == arm)]
                if len(sub):
                    r["T_stable_ev_ch5_" + arm] = float(sub["T_stable_ev_ch5"].iloc[0])
            r["in_moving_at_1s"] = bool((r["T_ramp"] == r["T_ramp"]) and r["T_ramp"] > 1.0)
            r["input_limited"] = bool((t_in_ch == t_in_ch) and np.isfinite(r.get("T_stable_ev_ch5_v6", np.nan))
                                      and r["T_stable_ev_ch5_v6"] >= t_in_ch - 0.05)
            rows.append(r)
        del z
    lb = pd.DataFrame(rows)
    p = os.path.join(C.RES, "t1a_lowerbound.csv")
    lb.round(4).to_csv(p, index=False, encoding="utf-8-sig")

    print("\n== 2. 输入侧稳定时间 T_in5（原始信号，与算法同口径 D1-ev） ==")
    for kd in ("onset", "restep"):
        s = lb[lb.kind == kd]
        print(" %s（n=%d，可测 %d）" % (kd, len(s), int(s.cens_in_ch.eq(False).sum())))
        desc(s.loc[s.cens_in_ch == False, "T_in5_raw_ch"], "主通道 T_in5 (s)")
        desc(s.loc[s.cens_in_tot == False, "T_in5_raw_tot"], "总通道 T_in5 (s)")
        desc(s["T_ramp"], "等效输入斜坡 T_ramp (s)")

    print("\n== 3. 实测 vs 下界（restep 与 onset 分开；冻结口径 D1-ev，主通道） ==")
    for kd in ("onset", "restep"):
        s = lb[(lb.kind == kd) & (lb.cens_in_ch == False)]
        for arm in C.ARMS:
            c = "T_stable_ev_ch5_" + arm
            if c not in s.columns:
                continue
            v = pd.to_numeric(s[c], errors="coerce").to_numpy(float)
            ok = np.isfinite(v)
            if not ok.any():
                continue
            below = int((v[ok] < s["T_in5_raw_ch"].to_numpy(float)[ok] - 0.05).sum())
            print("  %-6s %-5s n=%2d  中位 %6.2f s  T_in5 中位 %6.2f s  短于下界的事件 %d/%d"
                  % (kd, arm, int(ok.sum()), float(np.median(v[ok])),
                     float(np.median(s["T_in5_raw_ch"].to_numpy(float)[ok])), below, int(ok.sum())))

    print("\n== 4. 「输入本身还没走完」的事件数（1 s / 2 s 判据） ==")
    for kd in ("onset", "restep"):
        s = lb[lb.kind == kd]
        n = int(s["T_ramp"].notna().sum())
        if not n:
            print("  %s：T_ramp 缺数据" % kd)
            continue
        print("  %-6s n=%2d  T_ramp>1 s: %2d/%d (%.1f%%)   T_ramp>2 s: %d/%d   t90>1 s: %d/%d"
              % (kd, n, int((s.T_ramp > 1).sum()), n, 100 * float((s.T_ramp > 1).sum()) / n,
                 int((s.T_ramp > 2).sum()), n,
                 int((pd.to_numeric(s.t90, errors="coerce") > 1).sum()),
                 int(pd.to_numeric(s.t90, errors="coerce").notna().sum())))
    print("  1 s 时刻输入完成度 z_at_10 中位：onset %.3f（n=%d） restep %.3f（n=%d）"
          % (lb[lb.kind == "onset"].z_at_10.median(), int(lb[lb.kind == "onset"].z_at_10.notna().sum()),
             lb[lb.kind == "restep"].z_at_10.median(), int(lb[lb.kind == "restep"].z_at_10.notna().sum())))
    print("\n== 5. 判据侧最小延迟（常数逐字取自 t1a_glm53_v6.py） ==")
    import t1a_glm53_v6 as V6
    print("  TAU_REF=%.2f s（逆模型最早可用）  AWIN=%.2f s（拟合窗）  STALL_START_S=%.2f s"
          % (V6.GLM53v6.TAU_REF, V6.GLM53v6.AWIN, V6.GLM53v6.STALL_START_S))
    print("  STALL_HOLD_S=%.2f s（停滞确认）  HO_MIN=%.2f s（最早交接）  REVOKE=%.2f s（撤销窗）"
          % (V6.GLM53v6.STALL_HOLD_S, V6.GLM53v6.HO_MIN, V6.GLM53v6.REVOKE))
    print("  ⇒ 算法自延迟下界 ≈ TAU_REF + STALL_HOLD_S = %.2f s（输入瞬时到位时也不可能更快「停下」）"
          % (V6.GLM53v6.TAU_REF + V6.GLM53v6.STALL_HOLD_S))
    print("\n-> %s（%d 行）" % (p, len(lb)))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
