# -*- coding: utf-8 -*-
"""T9-B：恒定负载反复加减 —— 逐周期读数与基线偏移。

对每个「空载平台」与「受载平台」给出：
  pre(真值口径) / main(算法结果) 的平台电平，偏移 = main − pre，
  以及用户实际读到的「本次加载幅度」= 受载平台 − 其前一个空载平台（main 与 pre 各算一份）。
切段用 pre 的 1 s 滚动中位 + 幅度阈值（滞回），避免固定阈值被传感器零漂带偏。
输出 results/t9b_cycles.csv、results/t9b_series_full.csv
"""
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 6)))
DATA = os.path.join(ROOT, "temp", "算法数据&原始数据", "恒定负载下反复加减同一个负载",
                    "20260919_092417_single_device_602c03")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
NCH = 21


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    ch = [i for i, h in enumerate(hdr) if h.startswith("ch") or h.startswith("raw_ch")]
    if not ch:
        ncol = len(rows[di + 2].split(","))
        ch = list(range(ncol - NCH, ncol))
    el, vals = [], [[] for _ in range(NCH)]
    for r in rows[di + 2:]:
        if not r:
            continue
        f = r.split(",")
        el.append(float(f[1]))
        for k, ci in enumerate(ch):
            vals[k].append(float(f[ci]))
    return el, vals


el, pre = load("device_001_pre_seg0.csv")
_, main = load("device_001_seg000.csv")
_, raw = load("device_001_raw_seg000.csv")
n = len(el)
tp = [sum(pre[k][i] for k in range(NCH)) for i in range(n)]
tm = [sum(main[k][i] for k in range(NCH)) for i in range(n)]
tr = [sum(raw[k][i] for k in range(NCH)) for i in range(n)]

# 1 s 滚动中位（因果，用于切段的「真值」参考）
def rolling_med(x, half_s, el):
    out = [0.0] * len(x)
    for i in range(len(x)):
        a = i
        while a > 0 and el[i] - el[a - 1] < half_s:
            a -= 1
        seg = sorted(x[a:i + 1])
        out[i] = seg[len(seg) // 2]
    return out


med = rolling_med(tp, 0.5, el)
lo, hi = min(med), max(med)
thr_hi = lo + 0.45 * (hi - lo)
thr_lo = lo + 0.30 * (hi - lo)

segs = []
state = "idle" if med[0] < thr_hi else "loaded"
start = 0
for i in range(1, n):
    s = state
    if state == "idle" and med[i] > thr_hi:
        s = "loaded"
    elif state == "loaded" and med[i] < thr_lo:
        s = "idle"
    if s != state:
        segs.append((state, start, i - 1))
        state = s
        start = i
segs.append((state, start, n - 1))

# 只保留 ≥0.4 s 的平台
plat = [(k, a, b) for k, a, b in segs if el[b] - el[a] >= 0.4]

with open(os.path.join(OUT, "t9b_cycles.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["idx", "kind", "t_start", "t_end", "dur_s",
                "pre_lvl", "main_lvl", "raw_lvl", "off_main_pre",
                "off_pct_of_amp", "read_amp_main", "read_amp_pre", "read_amp_err_pct"])
    AMP = hi - lo
    prev_idle = None
    for i, (kind, a, b) in enumerate(plat):
        idx = range(a, b + 1)
        p = sorted(tp[k] for k in idx)[len(idx) // 2]
        m = sorted(tm[k] for k in idx)[len(idx) // 2]
        r = sorted(tr[k] for k in idx)[len(idx) // 2]
        ram = rampre = ""
        errp = ""
        if kind == "loaded" and prev_idle is not None:
            ram = f"{m - prev_idle[1]:.0f}"
            rampre = f"{p - prev_idle[0]:.0f}"
            den = max(p - prev_idle[0], 1e-9)
            errp = f"{100.0 * ((m - prev_idle[1]) - (p - prev_idle[0])) / den:+.1f}"
        w.writerow([i, kind, f"{el[a]:.2f}", f"{el[b]:.2f}", f"{el[b]-el[a]:.2f}",
                    f"{p:.0f}", f"{m:.0f}", f"{r:.0f}", f"{m-p:.0f}",
                    f"{100.0*(m-p)/AMP:+.1f}", ram, rampre, errp])
        if kind == "idle":
            prev_idle = (p, m)

print(f"load swing (1s-median) = {AMP:.0f}   平台数 = {len(plat)}")
print("\n idx  kind     t_start   t_end   dur   pre_lvl  main_lvl   off     off/amp | 本次读数(main)  真值(pre)  误差")
for i, (kind, a, b) in enumerate(plat):
    idx = range(a, b + 1)
    p = sorted(tp[k] for k in idx)[len(idx) // 2]
    m = sorted(tm[k] for k in idx)[len(idx) // 2]
    line = (f"{i:3d}  {kind:6s} {el[a]:8.2f} {el[b]:7.2f} {el[b]-el[a]:5.2f} "
            f"{p:8.0f} {m:9.0f} {m-p:7.0f} {100.0*(m-p)/AMP:7.1f}%")
    if kind == "loaded":
        pi = None
        for j in range(i - 1, -1, -1):
            if plat[j][0] == "idle":
                pi = plat[j]
                break
        if pi:
            idxp = range(pi[1], pi[2] + 1)
            p0 = sorted(tp[k] for k in idxp)[len(idxp) // 2]
            m0 = sorted(tm[k] for k in idxp)[len(idxp) // 2]
            line += (f" | {m-m0:8.0f} {p-p0:9.0f} "
                     f"{100.0*((m-m0)-(p-p0))/max(p-p0,1e-9):+7.1f}%")
    print(line)

with open(os.path.join(OUT, "t9b_series_full.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t_s", "pre_tot", "main_tot", "raw_tot", "off"])
    for i in range(0, n, 25):
        w.writerow([f"{el[i]:.3f}", f"{tp[i]:.0f}", f"{tm[i]:.0f}", f"{tr[i]:.0f}", f"{tm[i]-tp[i]:.0f}"])
print("\nwrote results/t9b_cycles.csv, results/t9b_series_full.csv")
