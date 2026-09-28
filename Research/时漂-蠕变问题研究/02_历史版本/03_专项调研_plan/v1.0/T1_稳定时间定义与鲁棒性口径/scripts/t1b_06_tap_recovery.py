# -*- coding: utf-8 -*-
"""T1-B / 06：**拍击叠加在保压段**（T1-Q9）+ **真实扰动对照**（T1-Q9 的实测证据）。

产物：results/t1b_tap_recovery.csv、results/t1b_real_tap.csv、results/_t1b_06_tap.log

拍击形态与幅度（逐字按需求文档）：上升/保持/回落 = **50/100/50 ms**，
峰值 = 保压电平 `L_hold` 的 **10% / 30% / 50%**；注入时刻 `t_tap = t_on + 25 s`（显示域，
保压段中段）或 `t_on + 20 s`（ADC 域，需 `gap_next ≥ 25 s`）。
叠加背景噪声档：`none`（确定性 1 次）+ 白噪 `r=0.005` / `r=0.02`（各 10 个种子，总量 RMS）。

指标（回答"会不会破坏已建立的基线、导致 T_stable 重算"）：
  n_ep_tap      拍击 ±1 s 内新建 epoch 数
  n_revoke      拍击窗口内撤销次数
  n_handoff     拍击窗口内交接次数（= 是否真的改写了基线 A）
  dA_sum        拍击前后 ΣA 的变化（ADC / 显示单位）→ **永久基线偏移**
  peak_raw_dev  显示（未平滑）相对拍击前电平的峰值偏差（绝对值）
  peak_dev_pct  同上 ÷ L_hold
  t_recover     显示回到"拍击前电平 ±2%·L_hold"并保持 5 s 所需时间
  T_stable_from_tap  从 t_tap 起重新计时的 T_stable（是否"重算"）
  level_shift   拍击后 [t_tap+30, t_tap+40] 显示中位 − 拍击前中位（永久台阶）
真实扰动对照：在变载实录里自动找"**单侧脉冲后回落**"的真实人手扰动段（真值轨线本身），
  与同幅度同形态的**合成拍击**对照（两条曲线叠加出图 T1B_03）。
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

SEEDS = list(range(10))
FRACS = [0.10, 0.30, 0.50]
NOISE_R = [0.02]                 # 叠加白噪（相对电平；0.02 = 2% 电平 RMS）
ARMS = ["v6", "v61"]
ADC_RECS = ["SW1", "SW4"]
TAP_RISE, TAP_HOLD, TAP_FALL = 50.0, 100.0, 50.0

_CLEAN = {}


def tap_metrics(L, tu, Z, t_tap, L_hold, r_epoch, r_handoff, r_revoke):
    """拍击窗口内的指标。"""
    from t1_common import med_smooth, t_stable
    dtm = 0.01
    Zb = med_smooth(Z, 50)
    i_tap = int(round(t_tap / dtm))
    i_pre0, i_pre1 = i_tap - int(10 / dtm), i_tap - int(1 / dtm)
    pre = float(np.median(Zb[i_pre0:i_pre1])) if i_pre1 > i_pre0 else np.nan
    # 峰值偏差（未平滑，含拍击自身的透传）
    j1 = min(len(Z), i_tap + int(3 / dtm))
    peak_raw = float(np.max(np.abs(Z[i_tap - 5:j1] - pre))) if j1 > i_tap else np.nan
    # 恢复时间：回到 |Z - pre| ≤ 2%·L_hold 且保持 5 s
    band = 0.02 * abs(L_hold)
    inside = np.abs(Z - pre) <= band
    t_rec = np.nan
    for i in range(i_tap, min(len(Z), i_tap + int(40 / dtm))):
        j = min(len(Z), i + int(5 / dtm))
        if inside[i:j].all():
            t_rec = float(tu[i] - t_tap)
            break
    # 永久台阶：拍击后 [t_tap+30, t_tap+40] 中位 − 拍击前中位
    a = min(len(Z) - 1, i_tap + int(30 / dtm))
    b = min(len(Z), i_tap + int(40 / dtm))
    lvl_shift = float(np.median(Zb[a:b]) - pre) if b - a > 100 else np.nan
    # 从拍击起重新计时
    Jt = max(peak_raw, 1e-9)
    ts = t_stable(tu, Z, t_tap, Jt, dtm=dtm, horizon=30.0)
    win = lambda lst, tol=1.0: int(np.sum(np.abs(np.array([e[0] for e in lst], float) - t_tap) <= tol)) if len(lst) else 0
    dA = np.nan
    hs = [h for h in r_handoff]
    before = [h[4] for h in hs if h[0] <= t_tap]
    after = [h[4] for h in hs if h[0] > t_tap and h[0] <= t_tap + 3.0]
    if before and after:
        dA = float(after[0] - before[-1])
    return dict(pre_level=pre, peak_raw_dev=peak_raw,
                peak_dev_pct=100.0 * peak_raw / abs(L_hold) if L_hold else np.nan,
                t_recover=t_rec, level_shift=lvl_shift,
                T_from_tap=ts["tau_v1"], T_from_tap_ok=int(ts["ok"]),
                n_ep_tap=win(r_epoch), n_revoke_tap=win(r_revoke), n_ho_tap=win(r_handoff),
                dA_sum=dA)


def worker(spec):
    L = __import__("t1b_lib")
    from t1_common import add_tap, make_perturb
    key = spec["key"]
    d = L.get_grid(key)
    tu0, Xu0 = L.window_of(d, spec["crop"])
    rng = np.random.default_rng(20000 + spec["seed"])
    Xp = Xu0
    if spec["noise_r"] > 0:
        Xp = Xu0 + make_perturb(Xu0, spec["noise_r"] * spec["L_hold"], "white", rng)
    i_tap = int(round(spec["t_tap"] / 0.01))
    amp = spec["frac"] * spec["L_hold"]
    Xp2, _ = add_tap(Xp, i_tap, amp, rise_ms=TAP_RISE, hold_ms=TAP_HOLD, fall_ms=TAP_FALL)
    rows = []
    for arm in ARMS:
        r = L.run_arm(arm, tu0, Xp2)
        Z = r["Z"]
        ck = (key, json.dumps(spec["crop"], sort_keys=True), arm)
        if ck not in _CLEAN:
            _CLEAN[ck] = L.run_arm(arm, tu0, Xu0)["Z"]
        m = tap_metrics(L, tu0, Z, spec["t_tap"], spec["L_hold"], r["epoch"], r["handoff"],
                        r["revoke"])
        # 同一段的无拍击对照（同噪声实现）：noise=0 时对照就是 clean 跑，直接复用缓存
        if spec["noise_r"] == 0.0:
            m0 = tap_metrics(L, tu0, _CLEAN[ck], spec["t_tap"], spec["L_hold"],
                             [], [], [])
        else:
            r0 = L.run_arm(arm, tu0, Xp)
            m0 = tap_metrics(L, tu0, r0["Z"], spec["t_tap"], spec["L_hold"], r0["epoch"],
                             r0["handoff"], r0["revoke"])
            del r0
        rows.append(dict(scope=spec["scope"], key=key, ev=spec["ev"], dom=spec["dom"],
                         t_on=spec["t_on"], t_tap=spec["t_tap"], arm=arm,
                         tap_ms=f"{TAP_RISE:.0f}/{TAP_HOLD:.0f}/{TAP_FALL:.0f}",
                         frac=spec["frac"], amp=amp, L_hold=spec["L_hold"],
                         noise_r=spec["noise_r"], seed=spec["seed"],
                         **m, ctrl_n_ep=m0["n_ep_tap"], ctrl_n_revoke=m0["n_revoke_tap"],
                         ctrl_n_ho=m0["n_ho_tap"], ctrl_peak=m0["peak_raw_dev"],
                         ctrl_shift=m0["level_shift"]))
        del r
    return rows


def build_tasks():
    import t1b_lib as L
    ev = L.event_table()
    tasks = []
    for _, e in ev[ev.dom == "显示域"].iterrows():
        crop = dict(kind="hold", t_on=float(e["t_on"]), span=80.0)
        for noise_r in [0.0] + NOISE_R:
            seeds = [0] if noise_r == 0.0 else SEEDS
            for frac in FRACS:
                for s in seeds:
                    tasks.append(dict(scope="tap_hold", key=e["key"], dom=e["dom"], crop=crop,
                                      ev=e["ev"], t_on=float(e["t_on"]), t_tap=float(e["t_on"]) + 25.0,
                                      L_hold=float(e["L_hold"]), frac=frac, noise_r=noise_r, seed=s))
    for key in ADC_RECS:
        sub = ev[(ev.key == key) & (ev.gap_next >= 25.0)]
        if not len(sub):
            continue
        for _, e in sub.iterrows():
            crop = dict(kind="full")
            for noise_r in [0.0] + NOISE_R:
                seeds = [0] if noise_r == 0.0 else SEEDS
                for frac in FRACS:
                    for s in seeds:
                        tasks.append(dict(scope="tap_adc", key=key, dom=e["dom"], crop=crop,
                                          ev=e["ev"], t_on=float(e["t_on"]),
                                          t_tap=float(e["t_on"]) + 20.0,
                                          L_hold=float(e["L_hold"]), frac=frac,
                                          noise_r=noise_r, seed=s))
    return tasks


def real_tap_scan():
    """在真实轨线里找"**孤立、双相、净电平不变**"的人手扰动段（真值数据自带，非注入）。

    判据：
      1. `[i, i+2.0 s]` 内出现 |Z − pre| > max(2%·|pre|, 阈值) 的偏离
         （阈值：实录 500 ADC / 恒载 0.3 显示单位）；
      2. `[i+2.5, i+5.0]` 中位回到 pre 的 ±5% 内（**净电平不变** ⇒ 不是一次真实变载）；
      3. 与 T4-A 事件表里任一检出事件相隔 > 1.0 s（排除变载沿本身）；
      4. 3 s 内不重复计数（孤立）。
    对 13 份录制（含 9 组恒载保压段）都扫；按 |dev| 排序取前若干做对照。
    """
    import t1b_lib as L
    from t1_common import med_smooth
    rows = []
    keys = ["SW1", "SW2", "SW3", "SW4", "RT1", "RT2", "RT3", "LT1", "LT2", "LT3",
            "F41", "F42", "F43"]
    truth_all = L.truth_events_all()
    for key in keys:
        d = L.get_grid(key)
        tu, Xu = d["tu"], d["Xu"]
        span_med = float(np.median(Xu.sum(axis=1)))
        big = 500.0 if key.startswith("SW") else 0.3
        Z = med_smooth(Xu.sum(axis=1), 20)          # 0.2 s 中值
        n = len(Z)
        i = int(3.0 / 0.01)
        tr = truth_all.get(key, np.zeros(0))
        while i < n - int(6.0 / 0.01):
            pre = float(np.median(Z[i - int(2.0 / 0.01):i]))
            w = Z[i:i + int(2.0 / 0.01)]
            k = int(np.argmax(np.abs(w - pre)))
            dev = float(w[k] - pre)
            back = float(np.median(Z[i + int(2.5 / 0.01):i + int(5.0 / 0.01)]))
            near_truth = float(np.min(np.abs(tr - tu[i + k]))) if len(tr) else 99.0
            if (abs(dev) > max(0.02 * abs(pre), big)
                    and abs(back - pre) <= 0.05 * abs(pre)
                    and near_truth > 1.0):
                # 基线是否真的已建立：扰动前 12 s 内电平稳定（±1%），且分析窗内无真实变载
                pre12 = float(np.median(Z[max(0, i - int(12.0 / 0.01)):max(1, i - int(2.0 / 0.01))]))
                stable_pre = abs(pre - pre12) <= 0.01 * abs(pre)
                tp = float(tu[i + k])
                n_in_wide = int(np.sum((tr >= tp - 15.0) & (tr <= tp + 35.0))) if len(tr) else 0
                n_in_narrow = int(np.sum((tr >= tp - 8.0) & (tr <= tp + 20.0))) if len(tr) else 0
                rows.append(dict(key=key, t_peak=round(tp, 2), dev=round(dev, 2),
                                 pre=round(pre, 2), dev_pct=round(100 * dev / max(abs(pre), 1e-9), 2),
                                 back=round(back, 2), nearest_truth_s=round(near_truth, 2),
                                 stable_pre=bool(stable_pre), truth_pm15_35=n_in_wide,
                                 truth_pm8_20=n_in_narrow,
                                 rel_to_span=round(dev / max(abs(span_med), 1e-9), 4)))
                i += int(5.0 / 0.01)
            else:
                i += int(0.5 / 0.01)
    df = pd.DataFrame(rows)
    if len(df):
        df["clean"] = ((df.truth_pm8_20 == 0) & (df.nearest_truth_s >= 8.0))
        df = df.sort_values(["clean", "key", "dev"], ascending=[False, True, False])
    return df


REAL_TOP = {"F41": 1, "LT1": 1, "LT2": 1, "RT1": 1, "RT2": 1, "RT3": 1}


def real_tap_compare(cand):
    """真实扰动 vs「同幅度同形态的合成拍击」对照（同一段真实轨线，同一臂）。

    (a) real      : 原轨线（真实人手扰动就在里面）
    (b) real+syn  : 原轨线上再叠加一个 peak=真实 dev、50/100/50 ms 的合成拍击
    (c) ctrl      : 原轨线（= real，用于给出"无拍击时的 epoch/偏差"基线）
    返回 (rows, 轨迹字典)。
    """
    import t1b_lib as L
    from t1_common import add_tap, med_smooth
    rows, traj = [], {}
    for _, e in cand.iterrows():
        key, t_peak = e["key"], float(e["t_peak"])
        d = L.get_grid(key)
        t0 = max(0.0, t_peak - 15.0)
        tu, Xu = L.crop(d["tu"], d["Xu"], t0, t_peak + 35.0)
        tp = t_peak - t0
        i_tap = int(round(tp / 0.01))
        Xs, _ = add_tap(Xu, i_tap, abs(float(e["dev"])), rise_ms=TAP_RISE,
                        hold_ms=TAP_HOLD, fall_ms=TAP_FALL)
        out = {}
        for tag, X in (("real", Xu), ("real+syn", Xs)):
            r = L.run_arm("v6", tu, X)
            Z = r["Z"]
            m = tap_metrics(L, tu, Z, tp, abs(float(e["pre"])), r["epoch"], r["handoff"],
                            r["revoke"])
            out[tag] = Z
            rows.append(dict(key=key, t_peak=t_peak, dev=float(e["dev"]),
                             dev_pct=float(e["dev_pct"]), case=tag, arm="v6",
                             nearest_truth_s=float(e.get("nearest_truth_s", np.nan)),
                             contaminated=(not bool(e.get("clean", False))),
                             n_ep_tap=m["n_ep_tap"], n_revoke_tap=m["n_revoke_tap"],
                             n_ho_tap=m["n_ho_tap"], peak_raw_dev=m["peak_raw_dev"],
                             peak_dev_pct=m["peak_dev_pct"], t_recover=m["t_recover"],
                             level_shift=m["level_shift"], dA_sum=m["dA_sum"]))
        traj[f"{key}@{t_peak:.2f}"] = dict(tu=tu, real=out["real"], syn=out["real+syn"],
                                           t_peak=tp, pre=float(e["pre"]))
    return pd.DataFrame(rows), traj


def log_write(path, s):
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")
    print(s, flush=True)


def main():
    t0 = time.time()
    out = os.path.join(RES, "t1b_tap_recovery.csv")
    logp = os.path.join(RES, "_t1b_06_tap.log")
    log_write(logp, "=== T1-B / 06 拍击-恢复 ===")
    log_write(logp, "cmd: python scripts/t1b_06_tap_recovery.py")

    rt = real_tap_scan()
    rt.to_csv(os.path.join(RES, "t1b_real_tap.csv"), index=False, encoding="utf-8-sig")
    log_write(logp, f"真实扰动候选 {len(rt)} 个（孤立、双相、净电平不变）")
    if len(rt):
        log_write(logp, rt.head(16).to_string(index=False))
    # 真实扰动 vs 同幅度同形态合成拍击（只用 clean 候选：基线已建立且分析窗内无真实变载）
    clean = rt[rt.clean] if len(rt) else rt
    sel = []
    for k, ntop in REAL_TOP.items():
        s = clean[clean.key == k]
        if len(s):
            sel.append(s.head(ntop))
    if not sel and len(rt):
        log_write(logp, "!! 无 clean 候选，退回全部候选（结论需标注基线已被真实变载污染）")
        sel = [rt.head(4)]
    if sel and len(rt):
        # 额外加一例**实录（ADC 域）**的真实扰动：实录里所有孤立双相扰动都紧邻真实变载，
        # 取"离真实事件最远"的一个并标 contaminated=True（报告里如实说明）。
        sw = rt[rt.key.str.startswith("SW")]
        if len(sw):
            sel.append(sw.sort_values("nearest_truth_s", ascending=False).head(1))
    if sel:
        cand = pd.concat(sel).reset_index(drop=True)
        cmp_df, traj = real_tap_compare(cand)
        cmp_df.to_csv(os.path.join(RES, "t1b_real_tap_compare.csv"), index=False,
                      encoding="utf-8-sig")
        np.savez_compressed(os.path.join(RES, "t1b_real_tap_traj.npz"),
                            **{f"{k}|{f}": v[f] for k, v in traj.items()
                               for f in ("tu", "real", "syn")},
                            **{f"{k}|t_peak": np.array([v["t_peak"]]) for k, v in traj.items()},
                            **{f"{k}|pre": np.array([v["pre"]]) for k, v in traj.items()})
        log_write(logp, "--- 真实扰动 vs 同幅度合成拍击（v6）---")
        log_write(logp, cmp_df.to_string(index=False))
    else:
        log_write(logp, "!! 未找到可用的真实扰动候选")

    tasks = build_tasks()
    nw = max(1, min(3, (os.cpu_count() or 4) - 6))
    log_write(logp, f"任务数 {len(tasks)}；workers={nw}")
    done, rows = 0, []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 40 == 0 or done == len(tasks):
                el = time.time() - t0
                log_write(logp, f"  {done}/{len(tasks)} 任务，{len(rows)} 行，{el:.0f}s"
                                f"（剩约 {(len(tasks)-done)*el/max(done,1)/60:.1f} min）")
                pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log_write(logp, f"完成 {len(df)} 行，{(time.time()-t0)/60:.1f} min")

    log_write(logp, "--- 拍击-恢复（v6，无噪声档；中位 [p10~p90]）---")
    s = df[(df.arm == "v6") & (df.noise_r == 0.0)]
    for (scope, frac), g in s.groupby(["scope", "frac"]):
        log_write(logp, f"  {scope:9s} 峰值={100*frac:3.0f}%L  n={len(g):2d}  "
                        f"触发 epoch {g.n_ep_tap.mean():.2f}  撤销 {g.n_revoke_tap.mean():.2f}  "
                        f"交接 {g.n_ho_tap.mean():.2f}  ΣA 变化 med={g.dA_sum.median()}  "
                        f"峰值偏差 {g.peak_dev_pct.median():.1f}%L  恢复 {g.t_recover.median()} s  "
                        f"永久台阶 med={g.level_shift.median()}")
    log_write(logp, "ALL DONE")
    with open(os.path.join(RES, "_t1b_06_done.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=len(df), minutes=round((time.time() - t0) / 60, 1)), f)


if __name__ == "__main__":
    main()
