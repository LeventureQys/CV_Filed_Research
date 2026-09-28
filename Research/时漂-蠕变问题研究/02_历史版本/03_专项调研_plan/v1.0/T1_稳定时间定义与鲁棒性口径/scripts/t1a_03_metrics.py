# -*- coding: utf-8 -*-
"""t1a_03_metrics：逐事件 × 逐实现算稳定时间指标（三口径 + 修订口径 + 两候选定义）。

输入：results/t1_events.csv（60 事件）、results/cache/t1a_*.npz（4 臂输出）、results/t1a_arm_epochs.csv
产出：results/t1a_settle_metrics.csv（每事件每实现一行，40 列）

口径（**本任务冻结**，逐条对齐 `00_共享/指标字典与口径.md` §2~§3；偏离处均显式标注）：
  * 时间轴 = `timestamp` 重采样到 100 Hz 网格（dt=0.01 s），**禁用 elapsed**；
  * 参考幅度三个口径：
      主通道口径 `J_ch`   = 指定主通道的 [t_on+4, t_on+6] 中位 − [t_on−2, t_on) 中位（与第一轮同）；
      总通道口径 `J_tot`  = 同一窗在**该臂总通道输出**上（Z = Σ_c Y_c）；
      含蠕变口径 `J_creep`= 主通道在"卸载沿前 5 s 中位 − pre"（r4_filter_tradeoff 同法）；
      另有 `J_chalt` = "该录制最大 |J| 事件的台阶最大通道"（主通道定义的敏感性口径）。
  * `T_stable`【D1，冻结主口径】= 首次"停下"：存在 τ 使 [t_on+τ, t_on+τ+30 s] 内自身漂移 ≤5%·|J_ref|；
    **要求完整 30 s 窗**（不足即删失），**不因后续事件截断**（与指标字典 §3 逐字一致）；
  * `T_stable_ev`【D1-ev，修订口径】= 同 D1，但窗在**下一个真实事件**处截断（要求可用窗 ≥5 s）；
  * `T_settle(ε)`【D2】= 首次进入并保持 |Y − Z_final| ≤ ε·|J_ref|（保持到 cut）；ε ∈ {2%, 5%}；
    `Z_final` = t_on+60 s 后的中位（**指标字典 §3 口径**）；实录类因事件密，退化为 t_on+10 s 后中位
    （记在 `zfinal_from_s` 列）；
  * `T_band`【D3】= 首次进入真值带 ±5%·J 并保持 30 s（`11-paper-v6/pv_common.settle_time` 口径）。
"""
import os
import sys

import numpy as np
import pandas as pd

import t1a_common as C


def z_final_of(Y, k0, cut_k, dt=C.DT):
    """Z_final：t_on+60 s 后中位（指标字典口径）；不足则退回 t_on+10 s 后中位。返回 (值, 起点偏移)。"""
    n = len(Y)
    end = n if cut_k is None else min(n, int(cut_k))
    a = k0 + int(60.0 / dt)
    from_s = 60.0
    if end - a < int(3.0 / dt):
        a = k0 + int(10.0 / dt)
        from_s = 10.0
    if end - a < int(1.0 / dt):
        return np.nan, np.nan
    return float(np.median(Y[a:end])), from_s


def os_dir(Y, k0, target, ref_amp, dt=C.DT, hold=30.0):
    """阶跃方向上的最大过冲（%×|J|）：max_t [(Y(t) − target)·sign(J)] / |J| × 100。"""
    n = len(Y)
    a, b = k0, min(n, k0 + int(hold / dt))
    if not np.isfinite(ref_amp) or abs(ref_amp) < 1e-12 or b <= a or not np.isfinite(target):
        return np.nan
    s = np.sign(ref_amp)
    return float(np.max((Y[a:b] - target) * s)) / abs(ref_amp) * 100.0


def main():
    C.start_log("03_metrics")
    ev = pd.read_csv(os.path.join(C.RES, "t1_events.csv"), encoding="utf-8-sig")
    ep = pd.read_csv(os.path.join(C.RES, "t1a_arm_epochs.csv"), encoding="utf-8-sig")
    epmap = {}
    for _, r in ep.iterrows():
        ts = [float(x) for x in str(r["epoch_t"]).split(";") if x.strip()] if isinstance(
            r["epoch_t"], str) else []
        epmap[(r["rec"], r["arm"])] = ts

    print("事件表 %d 行；每行 × %d 臂 = %d 行输出" % (len(ev), len(C.ARMS), len(ev) * len(C.ARMS)))
    rows = []
    for ds, grp in ev.groupby("ds", sort=False):
        z = np.load(os.path.join(C.CACHE, "t1a_%s.npz" % ds.replace("/", "_")))
        tu = z["tu"]
        dt = float(tu[1] - tu[0])
        yraw_ch, yraw_tot = z["yraw_ch"], z["yraw_tot"]
        chalt = int(z["chalt"])
        peak_rec = float(np.max(yraw_tot))
        times = [(float(t), float(j)) for t, j in zip(grp["t_on"], grp["J"]) if np.isfinite(j)]
        for _, e in grp.iterrows():
            k0 = int(e["k_on"])
            t_on = float(e["t_on"])
            ds_arm = {}
            for arm in C.ARMS:
                ds_arm[arm] = dict(ch=z["%s_ch" % arm], tot=z["%s_tot" % arm],
                                   chalt=z["%s_chalt" % arm])
            cut_s = C.next_event_cut(times, t_on, peak_rec)
            cut_k = None if cut_s is None else int(round(cut_s / dt))
            n = len(tu)
            win_avail = ((cut_k if cut_k is not None else n) * dt - t_on)

            J_ch, pre_ch, post_ch = C.amp_5s(yraw_ch, k0)
            J_creep = np.nan
            ju = C.unload_index(yraw_ch, k0)
            if ju is not None:
                a = max(k0, ju - int(5.0 / dt))
                if ju > a:
                    J_creep = float(np.median(yraw_ch[a:ju]) - pre_ch)
            J_chalt, pre_chalt, _ = C.amp_5s(z["yraw_chalt"], k0)

            for arm in C.ARMS:
                Yc, Yt, Yalt = ds_arm[arm]["ch"], ds_arm[arm]["tot"], ds_arm[arm]["chalt"]
                J_tot, pre_tot, _ = C.amp_5s(Yt, k0)
                d = dict(ds=ds, ev_id=e["ev_id"], key=e["key"], family=e["family"], dom=e["dom"],
                         kind=e["kind"], clean=bool(e["clean"]), t_on=round(t_on, 3), arm=arm,
                         J_ch=J_ch, J_tot=J_tot, J_creep=J_creep, J_chalt=J_chalt,
                         J_tot_ev=float(e["J"]),
                         win_avail_s=win_avail, cut_s=(np.nan if cut_s is None else cut_s),
                         main_ch=int(e["main_ch"]), chalt=chalt, pkt_dt=float(e["pkt_dt"]))
                # ── D1（冻结主口径，完整 30 s 窗，不截断）──
                d["T_stable_ch5"], d["cens_ch5"] = C.t_stable(tu, Yc, k0, J_ch)
                d["T_stable_tot5"], d["cens_tot5"] = C.t_stable(tu, Yt, k0, J_tot)
                d["T_stable_chcreep"], d["cens_chcreep"] = C.t_stable(tu, Yc, k0, J_creep)
                d["T_stable_chalt5"], d["cens_chalt5"] = C.t_stable(tu, Yalt, k0, J_chalt)
                # ── D1-ev（事件截断修订口径）──
                (d["T_stable_ev_ch5"], d["cens_ev_ch5"], d["win_ev_ch5"]) = C.t_stable_ev(
                    tu, Yc, k0, J_ch, cut_k)
                (d["T_stable_ev_tot5"], d["cens_ev_tot5"], d["win_ev_tot5"]) = C.t_stable_ev(
                    tu, Yt, k0, J_tot, cut_k)
                (d["T_stable_ev_chcreep"], d["cens_ev_chcreep"], d["win_ev_chcreep"]) = C.t_stable_ev(
                    tu, Yc, k0, J_creep, cut_k)
                # ── D2 / D3 ──
                zc, from_c = z_final_of(Yc, k0, cut_k)
                zt, _ = z_final_of(Yt, k0, cut_k)
                d["zfinal_from_s"] = from_c
                d["T_settle_ch2"], _ = C.t_settle(tu, Yc, k0, J_ch, 0.02, zc, cut_k)
                d["T_settle_ch5"], _ = C.t_settle(tu, Yc, k0, J_ch, 0.05, zc, cut_k)
                d["T_settle_tot2"], _ = C.t_settle(tu, Yt, k0, J_tot, 0.02, zt, cut_k)
                d["T_settle_tot5"], _ = C.t_settle(tu, Yt, k0, J_tot, 0.05, zt, cut_k)
                d["T_band_ch5"], _ = C.t_band(tu, Yc, k0, J_ch, 0.05, pre_ch + J_ch)
                # ── 幅度类指标（给 T5/T8 用，非本任务主轴）──
                d["OS_pct"] = os_dir(Yc, k0, pre_ch + J_ch, J_ch)
                for nm, off in (("err_1s_pct", 1.0), ("err_2s_pct", 2.0)):
                    i0 = k0 + int(round((off - 0.1) / dt))
                    i1 = k0 + int(round((off + 0.1) / dt))
                    d[nm] = (float(np.median(Yc[max(0, i0):i1])) - (pre_ch + J_ch)) / J_ch * 100.0 \
                        if np.isfinite(J_ch) and abs(J_ch) > 1e-12 else np.nan
                # ── 状态机活动（鲁棒性诊断）──
                ets = epmap.get((ds, arm), [])
                d["n_epoch_before"] = int(sum(1 for t in ets if t <= t_on))
                d["n_epoch_in_win"] = int(sum(1 for t in ets if t_on < t <= t_on + win_avail))
                d["epoch_per100s"] = (100.0 * d["n_epoch_in_win"] / win_avail
                                      if win_avail > 1.0 else np.nan)
                if not np.isfinite(e["J"]):
                    d["note"] = "J 缺失（T4-A 未给出 post 窗），指标不可算"
                else:
                    d["note"] = ""
                rows.append(d)
            print("  %-22s %-14s t_on=%7.2f kind=%-14s J=%9.3f J_ch=%8.3f J_creep=%8.3f "
                  "cut=%-7s win=%6.1f s" % (ds, e["ev_id"], t_on, e["kind"], e["J"], J_ch, J_creep,
                                            ("%.1f" % cut_s) if cut_s else "记录末", win_avail))
        del z

    df = pd.DataFrame(rows)
    cols = ["ds", "ev_id", "key", "family", "dom", "kind", "clean", "t_on", "arm",
            "main_ch", "chalt", "J_ch", "J_tot", "J_creep", "J_chalt", "J_tot_ev",
            "T_stable_ch5", "cens_ch5", "T_stable_tot5", "cens_tot5",
            "T_stable_chcreep", "cens_chcreep", "T_stable_chalt5", "cens_chalt5",
            "T_stable_ev_ch5", "cens_ev_ch5", "win_ev_ch5",
            "T_stable_ev_tot5", "cens_ev_tot5", "win_ev_tot5",
            "T_stable_ev_chcreep", "cens_ev_chcreep", "win_ev_chcreep",
            "T_settle_ch2", "T_settle_ch5", "T_settle_tot2", "T_settle_tot5", "zfinal_from_s",
            "T_band_ch5", "OS_pct", "err_1s_pct", "err_2s_pct",
            "win_avail_s", "cut_s", "pkt_dt", "n_epoch_before", "n_epoch_in_win",
            "epoch_per100s", "note"]
    df = df[cols]
    p = os.path.join(C.RES, "t1a_settle_metrics.csv")
    df.to_csv(p, index=False, encoding="utf-8-sig")

    print("\n== 冻结主口径 T_stable_ch5 的中位（按臂 × 事件类别，全部事件）==")
    print(df.pivot_table(index="kind", columns="arm", values="T_stable_ch5",
                         aggfunc="median").round(2).to_string())
    print("\n== 删失比例（cens_*，真=未在窗内停下）==")
    for c in ("cens_ch5", "cens_tot5", "cens_chcreep", "cens_ev_ch5"):
        print("  %-16s 删失 %3d/%3d" % (c, int(df[c].sum()), len(df)))
    print("\n-> %s（%d 行 × %d 列）" % (p, df.shape[0], df.shape[1]))
    print("冻结列: " + ", ".join(cols))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
