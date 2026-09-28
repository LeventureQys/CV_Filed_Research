# -*- coding: utf-8 -*-
"""T1-B / 05：**时序抖动与丢包敏感性**（T1-Q8）。

产物：results/t1b_jitter_sensitivity.csv、results/_t1b_05_jitter.log

扰动配置（全部在**真实帧时间戳**上注入，先注入后重采样到 100 Hz）：
  origin_pmK   事件起点不确定性：`t_on' = t_on ± K·T_pkt`（确定性，无种子；K=1,2）
  jitterK      包到达时刻抖动 ±K 包（对称），10 个随机种子
  delay1       包到达只延迟 0~1 包（缓冲延迟模型），10 个随机种子
  drop_fX      单帧丢包率 X，10 个随机种子
  droppktX     整包丢失率 X，10 个随机种子
  aggX         包聚合：每 X 帧合并为 1 个时间戳（确定性；用起始相位 0..9 造 10 个实现）
包周期 T_pkt 用**实测值**（指尖 15.35 ms、实录 39.9 ms；见 results/t1b_probe.csv）。
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
PKT_HOLD_MS = 15.35      # 指尖实测包周期（t1b_probe.csv: dt_med 15.11~15.45 ms）
PKT_ADC_MS = 39.9        # 变载实录实测包周期

CFG_HOLD = [
    ("origin_pm1", dict(op="origin", k=1, det=True), 1),
    ("origin_pm2", dict(op="origin", k=2, det=True), 1),
    ("jitter1", dict(op="jitter", k=1, mode="symmetric"), len(SEEDS)),
    ("jitter2", dict(op="jitter", k=2, mode="symmetric"), len(SEEDS)),
    ("delay1", dict(op="jitter", k=1, mode="delay"), len(SEEDS)),
    ("drop_f0.005", dict(op="drop", frac=0.005), len(SEEDS)),
    ("drop_f0.01", dict(op="drop", frac=0.01), len(SEEDS)),
    ("drop_f0.02", dict(op="drop", frac=0.02), len(SEEDS)),
    ("drop_f0.05", dict(op="drop", frac=0.05), len(SEEDS)),
    ("droppkt0.01", dict(op="drop", frac=0.01, whole=True), len(SEEDS)),
    ("droppkt0.05", dict(op="drop", frac=0.05, whole=True), len(SEEDS)),
    ("agg3", dict(op="agg", k=3), 10),
    ("agg6", dict(op="agg", k=6), 10),
]
CFG_ADC = [
    ("origin_pm1", dict(op="origin", k=1, det=True), 1),
    ("origin_pm2", dict(op="origin", k=2, det=True), 1),
    ("jitter1", dict(op="jitter", k=1, mode="symmetric"), len(SEEDS)),
    ("jitter2", dict(op="jitter", k=2, mode="symmetric"), len(SEEDS)),
    ("delay1", dict(op="jitter", k=1, mode="delay"), len(SEEDS)),
    ("drop_f0.01", dict(op="drop", frac=0.01), len(SEEDS)),
    ("drop_f0.05", dict(op="drop", frac=0.05), len(SEEDS)),
    ("droppkt0.01", dict(op="drop", frac=0.01, whole=True), len(SEEDS)),
    ("droppkt0.05", dict(op="drop", frac=0.05, whole=True), len(SEEDS)),
    ("agg8", dict(op="agg", k=8), 10),
    ("agg16", dict(op="agg", k=16), 10),
]
ARMS = ["raw", "v6"]
ADC_RECS = ["SW1", "SW4"]

_CLEAN = {}


def perturb_frames(t, X, cfg, seed, pkt_s):
    """在帧级时间戳上注入时序扰动，返回 (t2, X2)。"""
    from t1_common import add_packet_jitter, add_dropout, add_packet_aggregate
    op = cfg["op"]
    rng = np.random.default_rng(10000 + seed)
    if op == "jitter":
        return add_packet_jitter(t, X, k_pkt=cfg["k"], pkt_s=pkt_s, rng=rng, mode=cfg["mode"])
    if op == "drop":
        return add_dropout(t, X, frac=cfg["frac"], rng=rng, whole_packet=cfg.get("whole", False))
    if op == "agg":
        return add_packet_aggregate(t, X, k=cfg["k"])
    raise ValueError(op)


def worker(spec):
    L = __import__("t1b_lib")
    from t1_common import to_grid
    key = spec["key"]
    d = L.get_grid(key)
    tu0, Xu0 = L.window_of(d, spec["crop"])          # 无扰动窗（= clean 基线）
    t, X = d["t"], d["X"]
    # 帧级扰动（整段录制上做，再按窗裁剪）
    cfg, seed = spec["cfg"], spec["seed"]
    if cfg.get("det") and cfg["op"] == "origin":
        t2, X2 = t, X                                 # 起点平移在指标层处理
    elif cfg["op"] == "agg":
        from t1_common import add_packet_aggregate
        t2, X2 = add_packet_aggregate(t[seed:], X[seed:], k=cfg["k"])
    else:
        t2, X2 = perturb_frames(t, X, cfg, seed, spec["pkt_s"])
    tug, Xug = to_grid(t2, X2)
    tu, Xu = L.window_of(dict(tu=tug, Xu=Xug), spec["crop"])
    rows = []
    for arm in ARMS:
        r = L.run_arm(arm, tu, Xu)
        Z = r["Z"]
        if (key, json.dumps(spec["crop"], sort_keys=True), arm) not in _CLEAN:
            rc = L.run_arm(arm, tu0, Xu0)
            _CLEAN[(key, json.dumps(spec["crop"], sort_keys=True), arm)] = rc["Z"]
        Zc = _CLEAN[(key, json.dumps(spec["crop"], sort_keys=True), arm)]
        for e in spec["events"]:
            t_on = float(e["t_on"])
            if cfg.get("det") and cfg["op"] == "origin":
                t_on = t_on + cfg["k"] * spec["pkt_s"] * spec["sign"]
            J = float(e["J"])
            m = L.eval_event(tu, Z, t_on, J, float(e["t_next"]))
            md, sd = L.maxdev(tu, Zc, Z, max(0.0, min(float(e["t_on"]), t_on)))
            hit, pre, post = L.n_epoch_near(r["epoch"], float(e["t_on"]))
            _t = spec.get("truth")
            truth = [] if _t is None else _t
            mt = L.match_truth(truth, r["epoch"]) if len(truth) else (0, [], [])
            rows.append(dict(
                scope=spec["scope"], key=key, ev=e["ev"], t_on_ref=float(e["t_on"]),
                t_on_used=t_on, pert=spec["name"], op=cfg["op"], param=spec["param"],
                seed=seed, arm=arm, J=J, L_hold=float(e["L_hold"]),
                t_stable_v1=m["t_stable_v1"], t_stable_v2=m["t_stable_v2"],
                t_stable_ok=m["t_stable_ok"], os_pct=m["os_pct"], us_pct=m["us_pct"],
                ts5=m["ts5"], z_at_1=m["z_at_1"], z_at_2=m["z_at_2"],
                maxdev=md, slow_dev=sd, n_epoch=len(r["epoch"]), ep_hit=hit,
                n_revoke=len(r["revoke"]), n_handoff=len(r["handoff"]),
                n_miss=len(mt[1]) if len(truth) else -1,
                n_extra=len(mt[2]) if len(truth) else -1,
                n_frames=len(tu), grid_span=float(tu[-1]),
            ))
        del r
    return rows


def build_tasks(only=None):
    import t1b_lib as L
    ev = L.event_table()
    truth_all = L.truth_events_all()
    tasks = []
    if only != "adc":
        for _, e in ev[ev.dom == "显示域"].iterrows():
            crop = dict(kind="hold", t_on=float(e["t_on"]), span=80.0)
            evl = [dict(ev=e["ev"], t_on=float(e["t_on"]), J=float(e["J"]),
                        L_hold=float(e["L_hold"]), t_next=float(e["t_next"]))]
            for name, cfg, nseed in CFG_HOLD:
                param = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
                if cfg.get("det") and cfg["op"] == "origin":
                    for sign in (+1, -1):
                        tasks.append(dict(scope="jitter_hold", key=e["key"], crop=crop,
                                          events=evl, cfg=cfg, name=name,
                                          param=f"sign={sign:+d}", seed=0, sign=sign,
                                          pkt_s=PKT_HOLD_MS / 1000.0, truth=None))
                else:
                    for s in range(nseed):
                        tasks.append(dict(scope="jitter_hold", key=e["key"], crop=crop,
                                          events=evl, cfg=cfg, name=name, param=param,
                                          seed=s, sign=+1, pkt_s=PKT_HOLD_MS / 1000.0,
                                          truth=None))
    for key in ADC_RECS:
        if only == "hold":
            break
        sub = ev[ev.key == key]
        if not len(sub):
            continue
        crop = dict(kind="full")
        evl = [dict(ev=r["ev"], t_on=float(r["t_on"]), J=float(r["J"]),
                    L_hold=float(r["L_hold"]), t_next=float(r["t_next"]))
               for _, r in sub.iterrows()]
        for name, cfg, nseed in CFG_ADC:
            param = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
            if cfg.get("det") and cfg["op"] == "origin":
                for sign in (+1, -1):
                    tasks.append(dict(scope="jitter_adc", key=key, crop=crop, events=evl,
                                      cfg=cfg, name=name, param=f"sign={sign:+d}", seed=0,
                                      sign=sign, pkt_s=PKT_ADC_MS / 1000.0,
                                      truth=truth_all[key]))
            else:
                for s in range(nseed):
                    tasks.append(dict(scope="jitter_adc", key=key, crop=crop, events=evl,
                                      cfg=cfg, name=name, param=param, seed=s, sign=+1,
                                      pkt_s=PKT_ADC_MS / 1000.0, truth=truth_all[key]))
    return tasks


def log_write(path, s):
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")
    print(s, flush=True)


def main():
    t0 = time.time()
    only = sys.argv[1] if len(sys.argv) > 1 else None      # 'adc' / 'hold' / None
    out = os.path.join(RES, "t1b_jitter_sensitivity.csv")
    logp = os.path.join(RES, "_t1b_05_jitter.log")
    log_write(logp, "=== T1-B / 05 时序抖动与丢包敏感性 ===")
    log_write(logp, f"cmd: python scripts/t1b_05_jitter_sweep.py {only or ''}"
                    f"（maxdev 已按公共长度对齐，修 2026-09-19 首轮 adc 段广播崩溃）")
    tasks = build_tasks(only=only)
    old = None
    if only == "adc" and os.path.exists(out):
        old = pd.read_csv(out)
        log_write(logp, f"已有结果 {len(old)} 行：本次只补 ADC 段，结束后按主键合并")
    nw = max(1, min(4, (os.cpu_count() or 4) - 4))
    log_write(logp, f"任务数 {len(tasks)}；workers={nw}；包周期 指尖{PKT_HOLD_MS}ms 实录{PKT_ADC_MS}ms")
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
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
        keys = ["scope", "key", "ev", "pert", "param", "seed", "arm"]
        df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log_write(logp, f"完成 {len(df)} 行，{(time.time()-t0)/60:.1f} min")

    log_write(logp, "--- 显示域 v6：各扰动下的 T_stable / maxdev / 达标（中位 [p10~p90]）---")
    h = df[(df.scope == "jitter_hold") & (df.arm == "v6")]
    base = h[(h.pert == "origin_pm1") & (h.param == "sign=+1")]      # 仅作占位，见下 clean 基线
    for name, g in h.groupby("pert"):
        gm = g[g.t_stable_ok == 1]
        q = np.nanpercentile(gm.t_stable_v1, [10, 50, 90]) if len(gm) else [np.nan] * 3
        log_write(logp, f"  {name:12s} n={len(g):4d} T_stable med={q[1]:6.2f}s [{q[0]:.2f}~{q[2]:.2f}] "
                        f"达标 {int((gm.t_stable_v1 <= 2).sum())}/{len(gm)} "
                        f"maxdev med={g.maxdev.median():8.4f} ep+{g.n_extra.median():.1f}")
    log_write(logp, "--- ADC 域 v6：漏/误/最大偏差 ---")
    a = df[(df.scope == "jitter_adc") & (df.arm == "v6")]
    for name, g in a.groupby("pert"):
        log_write(logp, f"  {name:12s} n={len(g):4d} 漏={g.n_miss.mean():5.2f} 误={g.n_extra.mean():5.2f} "
                        f"maxdev med={g.maxdev.median():9.1f} OS% med={g.os_pct.median():6.2f}")
    log_write(logp, "ALL DONE")
    with open(os.path.join(RES, "_t1b_05_done.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=len(df), minutes=round((time.time() - t0) / 60, 1)), f)


if __name__ == "__main__":
    main()
