# -*- coding: utf-8 -*-
"""T7-A / step3：卸载后的残余偏移随时间（T7-Q2）+ 蠕变扣除是否被正确撤销（T7-Q3）。

§A 部分（纯数据）：raw 柱的残差轨迹与"回复比例"；
§B 部分（算法耦合）：同一条轨迹上叠加 v5.1 / v6 / v5 三臂，给出
  * 冻结偏差 Δ_frozen = resid(arm) − resid(raw)（指标字典 §4）；
  * 内部扣除量 ded_eff = Σ(原始) − Σ(输出) 的撤销过程（是否被冻结/被当成新蠕变）；
  * 贴零时长、额外下冲、单帧跳变（对应既有"7561 ADC 单帧跳变"那条记录）。

产出：results/t7_frozen_offset.csv（长表：事件 × 臂 × τ 的残余偏移-时间曲线）
      results/t7_arm_unload.csv（每事件 × 臂的标量：撤销/冻结/贴零/跳变）
      results/t7_frozen_summary.csv（每事件 raw 柱标量 + 类型）
      results/_t7a_3_offset.log
用法：python scripts/t7a_3_offset.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402

LOG = C.Log("3_offset")
TAU = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0])


def med_at(x, tu, t0, tau, win=1.0):
    m = (tu >= t0 + tau) & (tu <= t0 + tau + win)
    return float(np.median(x[m])) if m.any() else float("nan")


def main():
    C.log_reconfigure()
    ep = pd.read_csv(os.path.join(C.RES, "t7_unload_events.csv"))
    LOG("=" * 118)
    LOG("T7-A step3：卸载后残余偏移随时间（T7-Q2）+ 扣除撤销（T7-Q3）")
    LOG("定义：resid(τ) = Z̄_arm(t0+τ..t0+τ+1s) − z0_ref（z0_ref = 该轮加载前的空载电平中位）")
    LOG("      Δ_frozen = resid(arm) − resid(raw)（指标字典 §4 的「冻结偏差」）")
    LOG("      ded_eff  = 原始总量 − 补偿后总量（实际生效扣除，含输出封顶影响）")
    LOG("=" * 118)
    traj, arm_rows, sum_rows = [], [], []
    for tag, path in C.ALL:
        sub = ep[ep["rec"] == tag]
        if len(sub) == 0 or not os.path.exists(path):
            continue
        d = C.prep(tag)
        z = C.load_rec_cache(tag)
        tu = d["tu"]
        dtm = d["dtm"]
        tot = d["tot"]
        Zb = {a: C.med_smooth(z["Y_" + a].astype(float), int(round(0.5 / dtm))) for a in C.ARMS}
        Db = {a: z["dedeff_" + a].astype(float) for a in C.ARMS}
        St = {a: z["state_" + a].astype(int) for a in C.ARMS}
        for _, e in sub.iterrows():
            t0 = float(e["t_on"])
            i0 = int(round(t0 / dtm))
            z0 = float(e["z0_ref"])
            Jt = float(e["J_time"])
            absJ = abs(Jt)
            obs = float(e["obs_win_s"])
            noise = float(e["noise_idle"]) if np.isfinite(e["noise_idle"]) else 0.0
            tol = max(3.0 * noise, 0.005 * absJ)
            rec = dict(key=e["key"], rec=tag, kind_rec=("恒载" if "指尖" in tag else "实采"),
                       round_idx=int(e["round_idx"]), t0=t0, z0_ref=z0, J_time=Jt,
                       obs_win_s=obs, type=e["type"], noise_idle=noise)
            scal = {}
            for a in C.ARMS:
                yb = Zb[a]
                ded = Db[a]
                ded_pre = float(np.median(ded[max(0, i0 - int(3.0 / dtm)):i0])) if i0 > 3 else 0.0
                # 撤销过程
                j8 = min(len(tu) - 1, i0 + int(8.0 / dtm))
                seg = ded[i0:j8]
                below_half = np.where(np.abs(seg) <= 0.5 * abs(ded_pre) if abs(ded_pre) > 1e-12
                                      else seg == 0)[0]
                t_ded_half = float(tu[i0 + below_half[0]] - t0) if len(below_half) else float("nan")
                late_a = i0 + int(max(0.0, obs - 10.0) / dtm)
                late_b = min(len(tu), i0 + int(obs / dtm))
                ded_late = float(np.median(ded[late_a:late_b])) if late_b > late_a else float("nan")
                ded_late_pct = ded_late / absJ if np.isfinite(ded_late) else float("nan")
                # 贴零时长 / 额外下冲 / 单帧跳变
                j8b = min(len(tu), i0 + int(8.0 / dtm))
                yseg = yb[i0:j8b]
                rseg = Zb["raw"][i0:j8b]
                t_pin = float(np.sum(yseg <= tol) * dtm)
                t_pin_raw = float(np.sum(rseg <= tol) * dtm)
                min_arm = float(np.min(yseg)) if len(yseg) else float("nan")
                min_raw = float(np.min(rseg)) if len(rseg) else float("nan")
                j1 = min(len(tu) - 1, i0 + int(1.0 / dtm))
                jdiff = np.abs(np.diff(yb[i0:j1 + 1])) if j1 > i0 else np.array([np.nan])
                jump_frame = float(np.nanmax(jdiff)) if len(jdiff) else float("nan")
                scal[a] = dict(ded_pre=ded_pre, t_ded_half=t_ded_half, ded_late=ded_late,
                               ded_late_pct=ded_late_pct, t_pin=t_pin, extra_pin=t_pin - t_pin_raw,
                               min_arm=min_arm, dip_extra=min_raw - min_arm,
                               dip_extra_pct=(min_raw - min_arm) / absJ,
                               jump_frame=jump_frame,
                               resid_late=(med_at(yb, tu, t0, max(0.0, obs - 10.0), min(10.0, obs)) - z0
                                           if obs >= 2 else float("nan")))
            # 轨迹长表
            for tau in TAU:
                if tau > obs:
                    continue
                row = dict(rec=tag, kind_rec=rec["kind_rec"], key=e["key"], t0=t0,
                           round_idx=int(e["round_idx"]), tau=tau, z0_ref=z0, J_time=Jt)
                for a in C.ARMS:
                    y = med_at(Zb[a], tu, t0, tau)
                    row[f"y_{a}"] = y
                    row[f"resid_{a}"] = y - z0
                    row[f"resid_{a}_pct"] = (y - z0) / absJ
                    row[f"ded_{a}"] = med_at(Db[a], tu, t0, tau)
                    row[f"ded_{a}_pct"] = med_at(Db[a], tu, t0, tau) / absJ
                    row[f"state_{a}"] = (int(np.round(np.median(St[a][
                        (tu >= t0 + tau) & (tu <= t0 + tau + 1.0)])))
                        if ((tu >= t0 + tau) & (tu <= t0 + tau + 1.0)).any() else -1)
                for a in ("e3s", "v6", "v5"):
                    row[f"dfrozen_{a}"] = row[f"resid_{a}"] - row["resid_raw"]
                    row[f"dfrozen_{a}_pct"] = row[f"dfrozen_{a}"] / absJ
                traj.append(row)
            # per-arm scalars
            for a in C.ARMS:
                arm_rows.append(dict(key=e["key"], rec=tag, kind_rec=rec["kind_rec"],
                                     t0=t0, arm=a, J_time=Jt, **scal[a],
                                     resid_late_raw=scal["raw"]["resid_late"],
                                     dfrozen_late=(scal[a]["resid_late"] - scal["raw"]["resid_late"]),
                                     dfrozen_late_pct=((scal[a]["resid_late"]
                                                        - scal["raw"]["resid_late"]) / absJ)))
            # raw 标量 + 回复比例
            yb = Zb["raw"]
            r4 = med_at(yb, tu, t0, 4.0) - z0
            r6 = med_at(yb, tu, t0, 6.0) - z0
            rL = scal["raw"]["resid_late"]
            tpeak = med_at(yb, tu, t0, 0.0) - z0
            sum_rows.append(dict(key=e["key"], rec=tag, kind_rec=rec["kind_rec"],
                                 round_idx=int(e["round_idx"]), t0=t0, hold_s=float(e["hold_s"]),
                                 pre=float(e["pre"]), post=float(e["post"]), z0_ref=z0,
                                 J_time=Jt, J_frozen=float(e["J_frozen"]),
                                 resid0=tpeak, resid4=r4, resid6=r6, resid_late=rL,
                                 resid4_pct=r4 / absJ, resid_late_pct=rL / absJ,
                                 recovery_pct=(100.0 * (r4 - rL) / r4 if abs(r4) > 1e-9 else np.nan),
                                 slope_late=((rL - r4) / (obs - 14.0) if obs >= 20.0 else np.nan),
                                 n_tau_late=int(max(0, int((obs - 14.0) / 1.0))),
                                 obs_win_s=obs, type=e["type"], stable=bool(e["stable"]),
                                 noise_idle=noise))
    tr = pd.DataFrame(traj)
    ar = pd.DataFrame(arm_rows)
    sr = pd.DataFrame(sum_rows)
    C.save(tr, "t7_frozen_offset.csv")
    C.save(ar, "t7_arm_unload.csv")
    C.save(sr, "t7_frozen_summary.csv")

    LOG("")
    LOG("─" * 118)
    LOG("【T7-Q2】raw 柱残余偏移（绝对值 ADC / 力域 N；相对量按 |J_time|；n=%d）" % len(sr))
    LOG("─" * 118)
    cols = ["rec", "round_idx", "t0", "hold_s", "pre", "z0_ref", "J_time", "post",
            "resid4", "resid4_pct", "resid_late", "resid_late_pct", "recovery_pct",
            "obs_win_s", "type"]
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        LOG(sr[cols].round(4).to_string(index=False))
    for nm, sub in (("全部", sr), ("恒载 9 组（力域）", sr[sr["kind_rec"] == "恒载"]),
                    ("实录（ADC 域）", sr[sr["kind_rec"] == "实采"])):
        LOG("")
        LOG(f"【{nm}】n={len(sub)}" + ("（n≤3，仅作定性参考）" if len(sub) <= 3 else ""))
        for c in ("resid4", "resid4_pct", "resid_late", "resid_late_pct", "recovery_pct",
                  "slope_late"):
            x = sub[c].to_numpy(float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                LOG(f"   {c:<16} 无有效值")
                continue
            LOG(f"   {c:<16} 中位 {np.median(x):>12.5f}   p10~p90 "
                f"{np.percentile(x,10):>12.5f} ~ {np.percentile(x,90):>12.5f}   n={len(x)}")
    LOG("")
    LOG("【回到零没有？按阈值统计】|resid_late| 相对 |J_time| 的超限事件数：")
    for f in (0.002, 0.005, 0.01, 0.02):
        for nm, sub in (("全部", sr), ("恒载", sr[sr["kind_rec"] == "恒载"]),
                        ("实录", sr[sr["kind_rec"] == "实采"])):
            x = sub["resid_late_pct"].abs().to_numpy(float)
            x = x[np.isfinite(x)]
            LOG(f"   tol={f*100:>4.1f}%·|J|   {nm:<4} 超限 {int((x > f).sum())}/{len(x)}"
                f"   最大 {x.max() if len(x) else float('nan'):.4f}")
    LOG("")
    LOG("【T7-Q3】扣除撤销（按臂；ded_pre = 卸载前 3 s 的生效扣除中位）")
    LOG("─" * 118)
    agg = ar.groupby("arm").agg(
        n=("key", "count"),
        ded_pre_med=("ded_pre", "median"),
        t_ded_half_med=("t_ded_half", "median"),
        t_ded_half_max=("t_ded_half", "max"),
        ded_late_med=("ded_late", "median"),
        ded_late_pct_med=("ded_late_pct", "median"),
        ded_late_pct_max=("ded_late_pct", "max"),
        extra_pin_med=("extra_pin", "median"),
        extra_pin_max=("extra_pin", "max"),
        dip_extra_pct_med=("dip_extra_pct", "median"),
        dip_extra_pct_max=("dip_extra_pct", "max"),
        jump_med=("jump_frame", "median"),
        jump_max=("jump_frame", "max"),
        dfrozen_pct_med=("dfrozen_late_pct", "median"),
        dfrozen_pct_max=("dfrozen_late_pct", "max"),
    ).reindex(C.ARMS).round(4)
    LOG(agg.to_string())
    LOG("")
    LOG("逐事件（实录 ADC 域，按 |J| 归一化，%）")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        sub = ar[ar["kind_rec"] == "实采"][["rec", "t0", "arm", "ded_pre", "t_ded_half",
                                            "ded_late", "ded_late_pct", "extra_pin",
                                            "dip_extra_pct", "jump_frame", "dfrozen_late_pct"]]
        LOG(sub.round(4).to_string(index=False))
    LOG("")
    LOG("逐事件（恒载力域；ded/rel 用力的单位 N）")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        sub = ar[ar["kind_rec"] == "恒载"][["rec", "t0", "arm", "ded_pre", "t_ded_half",
                                            "ded_late", "ded_late_pct", "extra_pin",
                                            "dip_extra_pct", "jump_frame", "dfrozen_late_pct"]]
        LOG(sub.round(4).to_string(index=False))
    LOG("")
    LOG("【扣除轨迹是否'被当成新蠕变'】ded_late_pct 显著 > 0 且不随时衰减 ⇒ 扣除被冻结；"
        "≈0 ⇒ 已撤销。逐臂中位见上表 ded_late_pct_med。")
    LOG.close("python scripts/t7a_3_offset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
