# -*- coding: utf-8 -*-
"""T9-E：把现场录制的「扣除量」ded = pre − main 逐通道拆开，判断它属于哪条路径。
- 若 ded 在空载/受载下都非零且形状固定 ⇒ 慢相模块的 gamma·A·g（A 已固定、g 常数）；
- 若 ded 只在事件期出现 ⇒ 滑行器/事件路径。
同时输出 ded 与「当前读数」「受载电平」的相关性。
"""
import csv
import os

import numpy as np

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
    el, vals = [], []
    for r in rows[di + 2:]:
        if not r:
            continue
        f = r.split(",")
        el.append(float(f[1]))
        vals.append([float(f[ci]) for ci in ch])
    return np.array(el), np.array(vals)


el, PRE = load("device_001_pre_seg0.csv")
_, MAIN = load("device_001_seg000.csv")
DED = PRE - MAIN

picks = [("空载 0.5~1.7s", 0.5, 1.7), ("受载 3.5~5.5s", 3.5, 5.5),
         ("空载 13~15s", 13.0, 15.0), ("受载 17~20s", 17.0, 20.0),
         ("受载 44~51s(长平台)", 44.0, 51.0), ("空载 53~56s", 53.0, 56.0),
         ("受载 58~62s", 58.0, 62.0)]

print("逐通道扣除量 ded = pre − main（取窗中位）")
hdr = f"{'通道':>4s}" + "".join(f"{n:>22s}" for n, _, _ in picks)
print(hdr)
for k in range(NCH):
    row = f"ch{k:<2d}"
    for _, a, b in picks:
        m = (el >= a) & (el <= b)
        row += f"{np.median(DED[m, k]):22.1f}"
    print(row)

print("\n总量：")
for n, a, b in picks:
    m = (el >= a) & (el <= b)
    p = np.median(PRE[m].sum(axis=1))
    d = np.median(DED[m].sum(axis=1))
    print(f"  {n:22s} pre_tot={p:8.0f}  ded_tot={d:8.0f}  ded/pre={100*d/p:+6.2f}%  "
          f"ded/load_amp={100*d/2999:+6.1f}%  ded>0 通道数={int((np.median(DED[m],axis=0)>1).sum())}")

# ded 是否与当前读数成比例（同相位内）
print("\nded_i 与 pre_i 的关系（受载 44~51 s 窗）")
m = (el >= 44) & (el <= 51)
pp = np.median(PRE[m], axis=0)
dd = np.median(DED[m], axis=0)
for k in range(NCH):
    if abs(dd[k]) > 1:
        print(f"  ch{k:<2d} pre={pp[k]:8.1f} ded={dd[k]:8.1f} ded/pre={100*dd[k]/max(pp[k],1e-9):+7.2f}%")

with open(os.path.join(OUT, "t9e_deduction.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t_s"] + [f"pre_ch{k}" for k in range(NCH)] +
               [f"ded_ch{k}" for k in range(NCH)] + ["pre_tot", "main_tot", "ded_tot"])
    for i in range(0, len(el), 10):
        w.writerow([f"{el[i]:.3f}"] + [f"{PRE[i,k]:.1f}" for k in range(NCH)] +
                   [f"{DED[i,k]:.1f}" for k in range(NCH)] +
                   [f"{PRE[i].sum():.0f}", f"{MAIN[i].sum():.0f}", f"{DED[i].sum():.0f}"])
print("\nwrote results/t9e_deduction.csv")
