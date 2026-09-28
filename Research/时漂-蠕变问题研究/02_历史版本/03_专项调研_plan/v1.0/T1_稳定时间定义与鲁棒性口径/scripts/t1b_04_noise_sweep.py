# -*- coding: utf-8 -*-
"""T1-B / 04：**噪声扫描**（T1-Q6 白噪声 / T1-Q7 带限 + 同相共模）。

产物：
  results/t1b_robustness_sweep.csv   噪声类型 × 幅度 × 指标（一行 = 事件×扰动档×种子×臂）
  results/_t1b_04_noise.log          运行日志（含命令、每档进度）

口径要点：
  * 扰动**全部注入在真实轨线上**（恒载 9 组的 clean onset / 变载实录全长），
    不用合成阶跃产生主结论；合成只用于 `t1b_02_selftest.py` 的机制自检。
  * 幅度一律是「**总量扰动的 RMS**」：显示域单位 = N（未标定显示值），ADC 域 = ADC；
    两者**不可直接换算**（13 份录制 `has_raw_adc=false`，无 ADC↔显示标定），
    因此除绝对幅度外同时报 **相对幅度 amp_rel = A / L_hold**（跨域可比的唯一量）。
  * 每档 ≥10 个随机种子；报分布（中位 / p10~p90），n<20 时不报 mean±std。
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

# 显示域：相对幅度档（A = r × L_hold）
R_HOLD = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
KIND_HOLD_FULL = ["white"]
KIND_HOLD_SUB = ["band", "common"]
R_HOLD_SUB = [0.005, 0.01, 0.02, 0.05, 0.10]

# ADC 域：绝对幅度档（总量 RMS，ADC）
A_ADC_LOW = [1, 2, 5, 10, 20, 30, 40, 50]              # T1-Q6 要求的最低档位（逐字）
A_ADC_EXT = [100, 300, 500, 1000, 2000, 4000]          # 扩展档（找拐点）
ADC_RECS = ["SW1", "SW4"]

ARMS_HOLD = ["raw", "v6"]          # 主扫描（v5.1/v6.1 只在 ARM_COMPARE 档上扫，见 build_tasks）
ARMS_ARMCOMP = ["v51", "v61"]      # 四臂对照的额外两臂（与 ARMS_HOLD 合并成 4 臂）
ARMS_ADC = ["raw", "v6"]
R_ARMCOMP = [0.005, 0.02, 0.05]    # 四臂对照用的 3 个相对档

_CLEAN = {}


def _load_lib():
    import t1b_lib as L
    return L


def clean_Z(L, key, crop_spec, arm):
    """无扰动基线的**显示总量**（同裁剪窗、同臂），用于算 maxdev / slow_dev。"""
    ck = (key, json.dumps(crop_spec, sort_keys=True), arm)
    if ck not in _CLEAN:
        d = L.get_grid(key)
        tu, Xu = L.window_of(d, crop_spec)
        r = L.run_arm(arm, tu, Xu)
        _CLEAN[ck] = (tu, r["Z"])
    return _CLEAN[ck]


def worker(spec):
    """一个任务 = (作用域, 事件/录制, 扰动类型, 幅度, 种子, 臂集合)。返回行列表。"""
    L = _load_lib()
    from t1_common import make_perturb, med_smooth
    rows = []
    key = spec["key"]
    d = L.get_grid(key)
    tu, Xu = L.window_of(d, spec["crop"])
    rng = np.random.default_rng(spec["seed"] * 1000 + int(abs(spec["amp"]) * 100) % 997)
    E = make_perturb(Xu, spec["amp"], spec["kind"], rng)
    Xp = Xu + E
    events = spec["events"]
    _t = spec.get("truth")
    truth = [] if _t is None else _t
    has_truth = len(truth) > 0
    for arm in spec["arms"]:
        r = L.run_arm(arm, tu, Xp)
        Z = r["Z"]
        Zc = clean_Z(L, key, spec["crop"], arm)[1]
        for e in events:
            t_on = float(e["t_on"])
            J = float(e["J"])
            m = L.eval_event(tu, Z, t_on, J, float(e["t_next"]))
            md, sd = L.maxdev(tu, Zc, Z, t_on)
            hit, pre, post = L.n_epoch_near(r["epoch"], t_on)
            mt = L.match_truth(truth, r["epoch"]) if has_truth else (0, [], [])
            rows.append(dict(
                scope=spec["scope"], key=key, dom=spec["dom"], ev=e["ev"], t_on=t_on,
                kind=spec["kind"], amp=spec["amp"], amp_rel=spec["amp"] / float(e["L_hold"]),
                seed=spec["seed"], arm=arm, J=J, L_hold=float(e["L_hold"]),
                t_stable_v1=m["t_stable_v1"], t_stable_v2=m["t_stable_v2"],
                t_stable_ok=m["t_stable_ok"], os_pct=m["os_pct"], us_pct=m["us_pct"],
                ts2=m["ts2"], ts5=m["ts5"], z_at_1=m["z_at_1"], z_at_2=m["z_at_2"],
                maxdev=md, slow_dev=sd, n_epoch=len(r["epoch"]), ep_hit=hit,
                ep_pre=pre, ep_post=post, n_revoke=len(r["revoke"]),
                n_handoff=len(r["handoff"]),
                n_miss=len(mt[1]) if has_truth else -1,
                n_extra=len(mt[2]) if has_truth else -1,
            ))
        del r
    return rows


def build_tasks():
    import t1b_lib as L
    ev = L.event_table()
    tasks = []
    # ── 显示域：每个 clean onset 一个任务（裁剪到 t_on+80 s）──
    for _, e in ev[ev.dom == "显示域"].iterrows():
        crop = dict(kind="hold", t_on=float(e["t_on"]), span=80.0)
        evl = [dict(ev=e["ev"], t_on=float(e["t_on"]), J=float(e["J"]),
                    L_hold=float(e["L_hold"]), t_next=float(e["t_next"]))]
        for kind in KIND_HOLD_FULL:
            for r in R_HOLD:
                for s in SEEDS:
                    tasks.append(dict(scope="hold", key=e["key"], dom=e["dom"], crop=crop,
                                      events=evl, kind=kind, amp=float(r * e["L_hold"]),
                                      seed=s, arms=ARMS_HOLD, truth=None))
        for kind in KIND_HOLD_SUB:
            for r in R_HOLD_SUB:
                for s in SEEDS:
                    tasks.append(dict(scope="hold", key=e["key"], dom=e["dom"], crop=crop,
                                      events=evl, kind=kind, amp=float(r * e["L_hold"]),
                                      seed=s, arms=["raw", "v6"], truth=None))
        # 四臂对照（v5.1 / v6.1 只在 3 个档位上扫，用于"哪条实现更抗噪"的对照）
        for kind in ("white", "band", "common"):
            for r in R_ARMCOMP:
                for s in SEEDS:
                    tasks.append(dict(scope="hold_armcomp", key=e["key"], dom=e["dom"], crop=crop,
                                      events=evl, kind=kind, amp=float(r * e["L_hold"]),
                                      seed=s, arms=list(ARMS_ARMCOMP), truth=None))
    # ── ADC 域：每份实录一个任务（全长，一次跑完评估其全部 onset）──
    truth_all = L.truth_events_all()
    for key in ADC_RECS:
        sub = ev[ev.key == key]
        crop = dict(kind="full")
        evl = [dict(ev=r["ev"], t_on=float(r["t_on"]), J=float(r["J"]),
                    L_hold=float(r["L_hold"]), t_next=float(r["t_next"]))
               for _, r in sub.iterrows()]
        if not evl:
            continue
        for a in A_ADC_LOW:
            for s in SEEDS:
                tasks.append(dict(scope="adc", key=key, dom="ADC域", crop=crop, events=evl,
                                  kind="white", amp=float(a), seed=s, arms=ARMS_ADC,
                                  truth=truth_all[key]))
        for a in A_ADC_EXT:
            for kind in ("white", "band", "common"):
                for s in SEEDS:
                    tasks.append(dict(scope="adc", key=key, dom="ADC域", crop=crop, events=evl,
                                      kind=kind, amp=float(a), seed=s, arms=ARMS_ADC,
                                      truth=truth_all[key]))
    return tasks


def log_write(path, s):
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")
    print(s, flush=True)


def main():
    t_start = time.time()
    out = os.path.join(RES, "t1b_robustness_sweep.csv")
    logp = os.path.join(RES, "_t1b_04_noise.log")
    log_write(logp, "=== T1-B / 04 噪声扫描 ===")
    log_write(logp, "cmd: python scripts/t1b_04_noise_sweep.py")
    tasks = build_tasks()
    nw = max(1, min(6, (os.cpu_count() or 4) - 2))
    log_write(logp, f"任务数 {len(tasks)}；workers={nw}；seeds={len(SEEDS)}")

    done, rows, t0 = 0, [], time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 40 == 0 or done == len(tasks):
                el = time.time() - t0
                log_write(logp, f"  {done}/{len(tasks)} 任务完成，累计 {len(rows)} 行，"
                                f"{el:.0f}s（{el/done:.1f}s/任务，剩约 "
                                f"{(len(tasks)-done)*el/done/60:.1f} min）")
                pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log_write(logp, f"完成：{len(df)} 行 -> {os.path.basename(out)}；总耗时 "
                    f"{(time.time()-t_start)/60:.1f} min")

    # ── 汇总打印（中位 [p10~p90]）──
    log_write(logp, "--- 显示域 v6：T_stable 中位 / 达标率（≤2 s） / OS% 中位 随相对噪声 ---")
    h = df[(df.scope.isin(["hold", "hold_armcomp"])) & (df.arm == "v6")]
    for kind in ("white", "band", "common"):
        s = h[h.kind == kind]
        for r, g in s.groupby("amp_rel"):
            gm = g[g.t_stable_ok == 1]
            log_write(logp, f"  {kind:7s} r={r:<7.4f} n={len(g):3d} "
                            f"T_stable med={gm.t_stable_v1.median():6.2f}s "
                            f"[{np.nanpercentile(gm.t_stable_v1, 10) if len(gm) else np.nan:.2f}~"
                            f"{np.nanpercentile(gm.t_stable_v1, 90) if len(gm) else np.nan:.2f}] "
                            f"达标 {int((gm.t_stable_v1 <= 2).sum())}/{len(gm)} "
                            f"OS% med={g.os_pct.median():6.2f} max={g.os_pct.max():6.2f}")
    log_write(logp, "--- ADC 域 v6：漏/误/最大偏差 随绝对噪声 (ADC RMS) ---")
    a = df[(df.scope == "adc") & (df.arm == "v6")]
    for kind in ("white", "band", "common"):
        s = a[a.kind == kind]
        for amp, g in s.groupby("amp"):
            log_write(logp, f"  {kind:7s} A={amp:<7.0f} n={len(g):3d} "
                            f"漏={g.n_miss.mean():5.2f}/次 误={g.n_extra.mean():5.2f}/次 "
                            f"maxdev med={g.maxdev.median():9.1f} OS% med={g.os_pct.median():6.2f}")
    log_write(logp, "ALL DONE")
    with open(os.path.join(RES, "_t1b_04_done.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=len(df), minutes=round((time.time() - t_start) / 60, 1)), f)


if __name__ == "__main__":
    main()
