# -*- coding: utf-8 -*-
"""T3 步骤1：基础事实核查。

1) 基线自检：本模块 observe(ff=None) 与 v34_observer_core3.observe3 逐帧一致；
2) 数据集清单（working / archived 全部会话）；
3) slope 统计：沿期 vs 保压期 vs 空载期，用于标定沿检测门限；
4) 事件台阶幅度分布（评估侧沿切分）。
"""
import os
import sys

import numpy as np

import t3_lib as T

np.seterr(all="ignore")   # 重复时间戳会让 np.gradient 产生除零警告（无影响）

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(T.ROOT, "Document", "Update", "Dev-Version",
                                "v2.7 - 抗蠕变补偿算法", "v4.1flash",
                                "progress", "currentworking", "scripts"))
from v34_observer_core3 import observe3  # noqa: E402


def parity():
    print("== 1) 基线自检（本模块 vs v34_observer_core3.observe3）")
    sess = T.all_sessions()[:6]
    for tag, label, d in sess:
        s = T.load_input(d)
        D0, _ = observe3(s["ts"], s["V"])
        out = T.observe(s["ts"], s["V"], ff=None)
        err = float(np.max(np.abs(D0 - out["D"])))
        print("   %-60s max|Δ| = %.2e %s"
              % (label[-58:], err, "OK" if err < 1e-6 else "**不一致**"))


def inventory():
    print("\n== 2) 数据集清单")
    rows = []
    for tag, label, d in T.all_sessions():
        s = T.load_input(d)
        tot = s["V"].sum(axis=1)
        rows.append((label, s["el"][-1], len(s["el"]),
                     float(tot.min()), float(tot.max())))
    print("   共 %d 个会话（working %d / archived %d）"
          % (len(rows),
             sum(1 for r in rows if r[0].startswith("working")),
             sum(1 for r in rows if r[0].startswith("archived"))))
    for label, dur, n, lo, hi in rows:
        print("   %-70s %6.0fs %6d帧 输入总量 %7.0f~%7.0f"
              % (label[-68:], dur, n, lo, hi))
    return rows


def slope_stats():
    print("\n== 3) slope 统计（逐通道，(v−v_lp)/3 ADC/s）")
    print("   会话                                 空载 p50/p99 | 保压 p50/p99 | 沿期 p50/p99/max")
    allq = []
    for tag, label, d in T.all_sessions():
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        n = len(ts)
        v_lp = V[0].copy()
        S = np.empty((n, V.shape[1]))
        t_prev = ts[0]
        for i in range(n):
            dt = min(max(ts[i] - t_prev, 0.0), 0.1)
            t_prev = ts[i]
            if dt > 0:
                S[i] = (V[i] - v_lp) / 3.0
                v_lp = v_lp + (dt / 3.0) * (V[i] - v_lp)
            else:
                S[i] = 0.0
        tot = V.sum(axis=1)
        a, b = float(tot.min()), float(tot.max())
        if b - a < 200:
            print("   %-40s 电平范围过小，跳过" % label[-38:])
            continue
        idle_m = tot < a + 0.15 * (b - a)
        load_m = tot > a + 0.25 * (b - a)
        # 沿期：|d(tot)/dt| 大
        dtot = np.gradient(tot, el)
        edge_m = np.abs(dtot) > 0.35 * (b - a)
        q = lambda m: (np.percentile(S[m], 50), np.percentile(S[m], 99)) if m.sum() > 10 else (0, 0)
        qs = q(edge_m)
        print("   %-40s %6.1f/%7.1f | %6.1f/%7.1f | %7.1f/%8.1f (max %8.1f)"
              % (label[-38:], q(idle_m)[0], q(idle_m)[1], q(load_m)[0], q(load_m)[1],
                 qs[0], qs[1], float(S.max())))
        allq.append((label, S, load_m, edge_m))
    return allq


def edge_inventory():
    print("\n== 4) 输入台阶分布（评估侧沿切分，min_step = 100 ADC 总量）")
    allrows = []
    for tag, label, d in T.all_sessions():
        s = T.load_input(d)
        el, tot = s["el"], s["V"].sum(axis=1)
        rng = float(tot.max() - tot.min())
        ms = max(100.0, 0.03 * rng)
        ed = T.find_edges(el, tot, ms)
        steps = [e["step"] for e in ed]
        allrows.append((label, rng, ms, len(ed), steps))
        print("   %-46s 量程 %6.0f  min_step %5.0f  沿 %2d 个 台阶 %s"
              % (label[-44:], rng, ms, len(ed),
                 " ".join("%.0f" % x for x in sorted(steps, reverse=True)[:10])))
    small = sum(1 for r in allrows for x in r[4] if x < 1500)
    big = sum(1 for r in allrows for x in r[4] if x >= 1500)
    print("\n   合计：台阶 <1500 ADC 的沿 %d 个（小台阶，用户主诉场景）；≥1500 ADC 的沿 %d 个"
          % (small, big))
    return allrows


if __name__ == "__main__":
    parity()
    inventory()
    slope_stats()
    edge_inventory()
