# -*- coding: utf-8 -*-
"""T3 步骤8：稳定/回落时间专项（补充主表口径之外的时间量）。

对 4 个代表臂在全数据集上计算逐事件：
  t_trough     从沿后峰值到显示最低点的时长（「显示还在往下掉多久」）
  fall_trough  峰值到最低点的落差
  t5 / t2      进入并保持 settle±5% / ±2%·台阶 的时刻（段末仍在漂移则记 -1）
  in_band_frac 峰值后落在 ±5%·台阶 带内的帧占比（恒定义，便于跨事件比较）
  t_set        峰值后最后一次离开 ±5% 带的时刻（= t5，另有定义见下）

输出 results/t3_settle.csv 与 results/t3_settle.txt。
"""
import csv
import os

import numpy as np

import t3_lib as T
import t3_replay as R

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

ARMS = [("base", None),
        ("A_now_a1.0", dict(mode="A", pred="now", alpha=1.0)),
        ("A_lag_a1.0", dict(mode="A", pred="lag", alpha=1.0)),
        ("C_a0.8_tb2.0_1.5s", dict(mode="C", pred="now", alpha=0.8,
                                   tau_boost=2.0, boost_s=1.5))]


def metrics(el, disp, edges):
    out = []
    for e in edges:
        i, j = e["i"], e["j"]
        y = disp[i:j]
        if len(y) < 30:
            continue
        ys = R.smooth(y, 30)
        ee = el[i:j]
        peak = float(ys.max())
        ip = int(np.argmax(ys))
        w = min(len(ys), 200)
        settle = float(np.median(ys[-w:]))
        step = e["step"]
        rec = dict(t=float(ee[0]), step=step, hold=float(ee[-1] - ee[0]), peak=peak,
                   settle=settle)
        # 最低点与回落持续时长
        tail = ys[ip:]
        it = int(np.argmin(tail))
        rec["t_trough"] = float(ee[ip + it] - ee[ip])
        rec["fall_trough"] = float(peak - tail[it])
        for frac, key in ((0.05, "t5"), (0.02, "t2")):
            band = frac * abs(step)
            oi = np.nonzero(np.abs(ys - settle) > band)[0]
            if len(oi) == 0:
                rec[key] = 0.0
            elif oi[-1] >= len(ys) - 1:
                rec[key] = -1.0
            else:
                rec[key] = float(ee[oi[-1] + 1] - ee[0])
        inb = np.abs(ys[ip:] - settle) <= 0.05 * abs(step)
        rec["in_band_frac"] = float(np.mean(inb))
        out.append(rec)
    return out


def main():
    rows = []
    for tag, label, d in T.all_sessions():
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        din = V.sum(axis=1)
        rng = float(din.max() - din.min())
        edges, _ = T.find_edges(el, din)
        for name, spec in ARMS:
            ff = None
            if spec:
                ff = dict(R.DET_BASE)
                ff.update(spec)
            disp = T.observe(ts, V, ff=ff)["D"].sum(axis=1)
            for e, m in zip(edges, metrics(el, disp, edges)):
                rows.append(dict(session=label, arm=name, cls=R.classify(e, rng), **m))
        print("   %-46s 沿%3d" % (label[-46:], len(edges)), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "t3_settle.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["T3 稳定/回落时间专项（口径见脚本 docstring）",
             "",
             "臂                  类           N | t_trough 中位(s) | 回落trough ADC | T5 中位(定义数) | T2 中位(定义数) | 带内占比 中位",
             ""]
    cls_list = ["全部", "首次大台阶", "受载态小台阶", "其他"]
    for name, spec in ARMS:
        for cls in cls_list:
            rr = [r for r in rows if r["arm"] == name and
                  (cls == "全部" or r["cls"] == cls)]
            if not rr:
                continue
            t5 = [r["t5"] for r in rr if r["t5"] >= 0]
            t2 = [r["t2"] for r in rr if r["t2"] >= 0]
            lines.append("%-20s %-10s %3d | %7.1f | %8.0f | %6.1f (%3d) | %6.1f (%3d) | %6.2f"
                         % (name, cls, len(rr),
                            float(np.median([r["t_trough"] for r in rr])),
                            float(np.median([r["fall_trough"] for r in rr])),
                            float(np.median(t5)) if t5 else -1.0, len(t5),
                            float(np.median(t2)) if t2 else -1.0, len(t2),
                            float(np.median([r["in_band_frac"] for r in rr]))))
        lines.append("")
    with open(os.path.join(OUT, "t3_settle.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
