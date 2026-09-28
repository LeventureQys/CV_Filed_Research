# -*- coding: utf-8 -*-
"""T3 步骤6：参数敏感性（沿门限 / α / τ_boost）。

在代表会话子集上回放，输出 results/t3_sensitivity.csv + results/t3_sensitivity.txt。
每个组合记录：回落 fall 中位、自回弹下冲 中位/最大、T5 中位、电平偏置 中位、
触发次数（含「事件窗内/外」）、事件窗外显示-基线偏差 p95。

用法：python t3_sensitivity.py [会话关键字...]
"""
import csv
import os
import sys

import numpy as np

import t3_lib as T

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

SUBSET_KEYS = ["73032d", "9c3ca5", "f9740b", "f40a1b", "0cb8b6", "7b3977",
               "31b7f9", "602c03"]

# (标签, 参数) —— 首行基线臂
GRID = [("base", dict())]
for thr in (5.0, 10.0, 15.0):
    GRID.append(("thr%.0f_A_lag_a1.0" % thr,
                 dict(mode="A", pred="lag", alpha=1.0, thr=thr, rel_on=0.5)))
for rel in (0.25, 0.5, 1.0):
    GRID.append(("rel%.2f_A_lag_a1.0" % rel,
                 dict(mode="A", pred="lag", alpha=1.0, thr=10.0, rel_on=rel)))
for a in (0.5, 0.8, 1.0, 1.25):
    GRID.append(("A_lag_a%.2f" % a,
                 dict(mode="A", pred="lag", alpha=a, thr=10.0, rel_on=0.5)))
for a in (0.8, 1.0):
    GRID.append(("C_a%.1f_lag_tb2.0" % a,
                 dict(mode="C", pred="lag", alpha=a, tau_boost=2.0, boost_s=1.5,
                      thr=10.0, rel_on=0.5)))
for tb in (1.0, 2.0, 3.0):
    GRID.append(("B_tb%.1f_1.5s" % tb,
                 dict(mode="B", tau_boost=tb, boost_s=1.5, thr=10.0, rel_on=0.5)))
for bs in (1.0, 2.0, 3.0):
    GRID.append(("C_a0.8_lag_tb2.0_bs%.1f" % bs,
                 dict(mode="C", pred="lag", alpha=0.8, tau_boost=2.0,
                      boost_s=bs, thr=10.0, rel_on=0.5)))


def simple_events(el, tot, min_step_abs=500.0, min_step_frac=0.05, gap=2.0):
    rng = float(tot.max() - tot.min())
    ms = max(min_step_abs, min_step_frac * rng)
    rise = tot - np.concatenate([[tot[0]] * 100, tot[:-100]])
    ev, last = [], -1e9
    for i in range(len(el)):
        if rise[i] > ms and el[i] - last > gap:
            ev.append(el[i])
            last = el[i]
    return np.array(ev), ms


def metrics(el, din, disp, edges, base_disp, ev):
    fall, dip, t5s = [], [], []
    for e in edges:
        i, j = e["i"], e["j"]
        y = disp[i:j]
        if len(y) < 20:
            continue
        ys = T_smooth(y)
        peak = float(ys.max())
        ip = int(np.argmax(ys))
        w = min(len(ys), 200)
        settle = float(np.median(ys[-w:]))
        step = e["step"]
        band = 0.05 * abs(step)
        oi = np.nonzero(np.abs(ys - settle) > band)[0]
        t5 = None if (len(oi) and oi[-1] >= len(ys) - 1) else (
            0.0 if len(oi) == 0 else float(el[i + oi[-1] + 1] - el[i]))
        tail = ys[ip:]
        rmin = np.minimum.accumulate(tail)
        fall.append(peak - settle)
        dip.append(float(np.max(tail - rmin)))
        if t5 is not None:
            t5s.append(t5)
    return dict(fall=(np.median(fall) if fall else np.nan),
                dip=(np.median(dip) if dip else np.nan),
                dip_max=(np.max(dip) if dip else np.nan),
                t5=(np.median(t5s) if t5s else np.nan), n=len(fall))


def T_smooth(y, n=30):
    y = np.asarray(y, dtype=float)
    if len(y) <= n:
        return y.copy()
    pad = n // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, np.ones(n) / n, mode="valid")[:len(y)]


def main():
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or SUBSET_KEYS
    subs = [(tag, lb, d) for tag, lb, d in T.all_sessions()
            if any(k in lb for k in keys)]
    print("子集会话 %d 个：%s" % (len(subs), ", ".join(lb[-24:] for _, lb, _ in subs)))
    # 先算基线与事件
    cache = []
    for tag, label, d in subs:
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        din = V.sum(axis=1)
        edges, ms = T.find_edges(el, din)
        ev, _ = simple_events(el, din)
        base = T.observe(ts, V, ff=None)
        cache.append((label, el, ts, V, din, edges, ev, base["D"].sum(axis=1)))
        print("   %-40s 帧%6d 沿%3d 事件%3d" % (label[-40:], len(el), len(edges), len(ev)))
    rows = []
    for name, spec in GRID:
        agg = dict(fall=[], dip=[], dip_max=[], t5=[], lvl=[], n_trig=0,
                   n_trig_off=0, dev_p95=0.0)
        for (label, el, ts, V, din, edges, ev, base_disp) in cache:
            ff = dict(T.DET)
            ff.update(spec)
            r = T.observe(ts, V, ff=ff)
            disp = r["D"].sum(axis=1)
            m = metrics(el, din, disp, edges, None, ev)
            trig_t = np.array([float(el[i]) for c, i, sl, ep, xb in r["trig"]])
            on = np.zeros(len(el), dtype=bool)
            for t0 in ev:
                on |= (el >= t0 - 0.5) & (el <= t0 + 2.0)
            n_off = int(np.sum(~on[np.clip(np.searchsorted(el, trig_t), 0, len(el) - 1)])) if len(trig_t) else 0
            far = ~T.event_mask(el, ev, 0.5, 20.0)
            idle = T.idle_mask(din)
            dev = np.abs(disp - base_disp)
            devfar = dev[far] if far.sum() > 10 else np.array([0.0])
            devidle = dev[idle] if idle.sum() > 10 else np.array([0.0])
            lv = []
            for e in edges:
                i, j = e["i"], e["j"]
                if j - i < 20:
                    continue
                w = min(j - i, 200)
                lv.append(float(np.median(disp[j - w:j]) - np.median(base_disp[j - w:j])))
            agg["fall"].append(m["fall"])
            agg["dip"].append(m["dip"])
            agg["dip_max"].append(m["dip_max"])
            if not np.isnan(m["t5"]):
                agg["t5"].append(m["t5"])
            agg["lvl"].extend(lv)
            agg["n_trig"] += len(trig_t)
            agg["n_trig_off"] += n_off
            if len(devfar):
                agg["dev_p95"] = max(agg["dev_p95"], float(np.percentile(devfar, 95)))
            agg["dev_idle"] = max(agg.get("dev_idle", 0.0), float(np.max(devidle)))
        row = dict(arm=name,
                   thr=spec.get("thr", 10.0), rel=spec.get("rel_on", 0.5),
                   alpha=spec.get("alpha", float("nan")),
                   tau_boost=spec.get("tau_boost", float("nan")),
                   boost_s=spec.get("boost_s", float("nan")),
                   fall=float(np.nanmedian(agg["fall"])),
                   dip=float(np.nanmedian(agg["dip"])),
                   dip_max=float(np.nanmax(agg["dip_max"])),
                   t5=float(np.nanmedian(agg["t5"])) if agg["t5"] else float("nan"),
                   lvl=float(np.nanmedian(agg["lvl"])) if agg["lvl"] else float("nan"),
                   n_trig=agg["n_trig"], n_trig_off=agg["n_trig_off"],
                   dev_p95=agg["dev_p95"], dev_idle=agg.get("dev_idle", 0.0))
        rows.append(row)
        print("   %-28s 回落%7.0f 下冲%6.0f(最大%7.0f) T5%6.1f 偏置%+7.0f 触发%5d(窗外%5d) 窗外偏差p95%7.0f 空载偏差%7.0f"
              % (name, row["fall"], row["dip"], row["dip_max"], row["t5"], row["lvl"],
                 row["n_trig"], row["n_trig_off"], row["dev_p95"], row["dev_idle"]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "t3_sensitivity.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n-> results/t3_sensitivity.csv（%d 组合）" % len(rows))


if __name__ == "__main__":
    main()
