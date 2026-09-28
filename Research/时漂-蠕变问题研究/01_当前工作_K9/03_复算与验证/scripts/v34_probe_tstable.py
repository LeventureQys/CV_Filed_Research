# -*- coding: utf-8 -*-
"""v3.4 稳定时间对比：阶跃沿 → 显示进入并保持 ±5%（及 ±2%）带内的时长。

对象：目标录制 7b3977 的 9 次满载加载沿（楼梯式，含多级）。
三臂：观测器原型 / v6 mem OFF（v3.1）/ v6 mem ON（v3.4 记忆，已归档）。
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_observer_core import observe
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "build", "v34_runner.exe")
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")

# (沿起点, 段末)——加载事件（楼梯首级），含首次加载
EDGES = [(2.38, 14.0), (14.4, 35.5), (41.1, 55.5), (62.8, 215.0), (231.6, 238.5),
         (244.3, 250.5), (257.35, 260.4), (277.61, 281.2),
         (288.6, 294.2), (299.6, 307.5)]


def tstable(el, y, t0, t1, frac):
    """从 t0 起，显示进入 ±frac·台阶 带内并保持到段末的首个时刻。"""
    m = (el >= t0) & (el <= t1)
    if m.sum() < 50:
        return None
    ee, yy = el[m], y[m]
    final = float(np.median(yy[-max(len(yy) // 3, 30):]))
    base = float(np.median(y[(el >= t0 - 2.0) & (el < t0)]))
    step = max(final - base, 1.0)
    band = frac * step
    inside = np.abs(yy - final) <= band
    # 从后往前找最后一段连续 inside 的起点
    i = len(inside) - 1
    while i > 0 and inside[i - 1]:
        i -= 1
    # 要求从 i 起全部 inside 且之后无离带
    if not inside[i:].all():
        return None
    # 再往前探允许短暂离带：取“最后离带时刻+1帧”
    j = i
    while j > 0 and inside[j - 1]:
        j -= 1
    return ee[j] - t0, step


def main():
    s = L.load_stream(DS, "device_001_pre_seg0.csv")
    el, ts, V = s["el"], s["ts"], s["V"]
    din = V.sum(axis=1)
    D, _ = observe(ts, V)
    obs = D.sum(axis=1)

    n = V.shape[1]
    lines = [str(n)]
    for i in range(len(ts)):
        lines.append("%.6f " % ts[i] + " ".join("%.1f" % x for x in V[i]))

    def replay(extra):
        p = subprocess.run([RUNNER] + extra, input="\n".join(lines),
                           capture_output=True, text=True, encoding="utf-8",
                           cwd=HERE)
        rows = []
        for ln in p.stdout.splitlines():
            f = ln.split()
            if not f or f[0] in ("t", "OK", "END", "CH"):
                continue
            try:
                rows.append([float(x) for x in f])
            except ValueError:
                pass
        arr = np.array(rows)
        return arr[:, 2]

    off = replay(["--mem", "0"])
    on = replay(["--mem", "120"])

    print("沿起点   台阶   | T±5%%: 观测器 / v6OFF / v6ON | T±2%%: 观测器 / v6OFF / v6ON")
    agg = {"obs5": [], "off5": [], "on5": [], "obs2": [], "off2": [], "on2": []}
    for t0, t1 in EDGES:
        row = ["%6.1f" % t0]
        r = tstable(el, obs, t0, t1, 0.05)
        row.append("%5.0f" % r[1])
        if r:
            agg["obs5"].append(r[0])
        for name, y in (("obs", obs), ("off", off), ("on", on)):
            for frac, key in ((0.05, "5"), (0.02, "2")):
                rr = tstable(el, y, t0, t1, frac)
                if rr:
                    agg[name + key].append(rr[0])
                    if frac == 0.05:
                        pass
        cell5 = " / ".join(("%5.1f" % agg[k + "5"][-1]) if agg[k + "5"] else "  -  "
                           for k in ("obs", "off", "on"))
        cell2 = " / ".join(("%5.1f" % agg[k + "2"][-1]) if agg[k + "2"] else "  -  "
                           for k in ("obs", "off", "on"))
        print("%s %s | %s | %s" % (row[0], row[1], cell5, cell2))
    print("\n中位汇总:")
    for key in ("obs", "off", "on"):
        v5 = agg[key + "5"]
        v2 = agg[key + "2"]
        print("  %-4s T±5%% 中位 %6.1fs (n=%d)   T±2%% 中位 %6.1fs (n=%d)"
              % (key, np.median(v5) if v5 else -1, len(v5),
                 np.median(v2) if v2 else -1, len(v2)))


if __name__ == "__main__":
    main()
