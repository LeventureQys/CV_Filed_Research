# -*- coding: utf-8 -*-
"""T3 步骤2：沿检测器标定。

问题：门限取多少才能「沿必触发、静置不触发、慢速加载不抖」。
做法（纯统计，不改算法）：
  A. 用输入总量的分段斜率 + 滞回切出「沿窗」（|dT/dt| 超过自适应门限的时段）；
  B. 沿窗外 = 静置/慢变帧，统计逐通道 slope 的 p50/p99/p99.9/max（=> 门限下界）；
  C. 沿窗内：每个沿的逐通道 slope 峰值、对应逐通道台阶 Δc（=> 门限上界与可检性）；
  D. 给出候选门限的「静置帧误触发率」与「沿覆盖数」。
"""
import os
import sys

import numpy as np

import t3_lib as T

np.seterr(all="ignore")


def edge_windows(el, tot, thr_frac=0.004, min_gap=1.0):
    """沿窗：|dT/dt| 超过 thr_frac·量程 的帧并集（前后各留 0.3 s）。"""
    rng = float(tot.max() - tot.min())
    dtot = np.gradient(tot, el)
    thr = thr_frac * rng
    m = np.abs(dtot) > thr
    # 合并间隔 < 0.3 s 的段
    idx = np.nonzero(m)[0]
    if len(idx) == 0:
        return [], thr
    segs = []
    s = p = idx[0]
    for i in idx[1:]:
        if el[i] - el[p] > 0.3:
            segs.append((s, p))
            s = i
        p = i
    segs.append((s, p))
    # 去重（间隔 < min_gap 的合并）
    merged = []
    for s, p in segs:
        if merged and el[s] - el[merged[-1][1]] < min_gap:
            merged[-1] = (merged[-1][0], p)
        else:
            merged.append((s, p))
    # 扩张到「沿起点前一帧 ~ 沿后 0.5 s」
    out = []
    for s, p in merged:
        i0 = max(0, s - 2)
        i1 = min(len(el) - 1, int(np.searchsorted(el, el[p] + 0.5)))
        out.append((i0, i1, s, p))
    return out, thr


def slope_track(ts, V):
    n = V.shape[0]
    S = np.zeros_like(V)
    v_lp = V[0].copy()
    t_prev = ts[0]
    for i in range(n):
        dt = min(max(ts[i] - t_prev, 0.0), 0.1)
        t_prev = ts[i]
        if dt > 0:
            S[i] = (V[i] - v_lp) / 3.0
            v_lp = v_lp + (dt / 3.0) * (V[i] - v_lp)
    return S


def main():
    quiet_all = []
    edges_all = []
    print("== A/B/C 逐会话沿窗与 slope 统计（逐通道 ADC/s）")
    print("   会话                    沿窗数 | 静置 slope p50/p99/p99.9/max | 沿内逐通道峰值 slope 中位/最大")
    for tag, label, d in T.all_sessions():
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        tot = V.sum(axis=1)
        S = slope_track(ts, V)
        wins, thr = edge_windows(el, tot)
        inwin = np.zeros(len(el), dtype=bool)
        for i0, i1, _, _ in wins:
            inwin[i0:i1 + 1] = True
        # 静置帧排除沿窗；另外排除近零带外的低电平？不做，保持保守
        qm = ~inwin
        if qm.sum() < 50:
            continue
        q = S[qm]
        quiet_all.append(q)
        pk, stp = [], []
        for i0, i1, a, b in wins:
            if i1 - i0 < 3:
                continue
            w = S[i0:i1 + 1]
            ch_pk = w.max(axis=0)
            dch = V[b] - V[max(0, a - 3)]
            dch = np.where(np.abs(dch) > 0, np.abs(dch), 1.0)
            pk.append(np.max(ch_pk))
            stp.append(float(np.median(np.abs(dch))))
            edges_all.append((label, float(el[a]), float(np.median(np.abs(dch))),
                              float(np.max(ch_pk)), float(np.percentile(ch_pk, 50)),
                              float(np.sum(np.abs(V[b] - V[max(0, a - 3)])))))
        if pk:
            print("   %-24s %3d | %5.1f/%6.1f/%7.1f/%8.1f | %8.1f/%8.1f"
                  % (label[-22:], len(wins),
                     np.percentile(q, 50), np.percentile(q, 99),
                     np.percentile(q, 99.9), q.max(),
                     np.median(pk), np.max(pk)))
        else:
            print("   %-24s %3d | %5.1f/%6.1f/%7.1f/%8.1f | (无有效沿窗)"
                  % (label[-22:], len(wins),
                     np.percentile(q, 50), np.percentile(q, 99),
                     np.percentile(q, 99.9), q.max()))

    Q = np.concatenate(quiet_all)
    print("\n== 汇总：全部会话静置帧 slope 分位（N=%.2e 个通道帧）" % Q.size)
    for p in (50, 90, 99, 99.9, 99.99):
        print("   p%-6s %8.2f" % (p, np.percentile(Q, p)))
    print("   max     %8.2f" % Q.max())

    print("\n== 候选门限的静置误触发率（逐通道帧占比）")
    for thr in (10, 15, 20, 25, 30, 40, 60, 80, 100):
        rate = float(np.mean(Q > thr))
        print("   thr=%5.0f ADC/s -> 静置误触发率 %.4f%%  (每 100 万通道帧 %d 次)"
              % (thr, rate * 100, int(rate * 1e6)))

    print("\n== D 沿可检性：逐通道台阶 Δc 与沿内 slope 峰值")
    print("   逐通道台阶中位 → 沿内峰值 slope 中位/最大（共 %d 个沿窗）" % len(edges_all))
    bys = sorted(edges_all, key=lambda r: r[2])
    for label, t, dch, pk_max, pk_med, dtot in bys:
        print("   %-20s t=%6.1f 总台阶 %7.0f 逐通道Δ中位 %6.1f  slope峰值 中位 %6.1f 最大 %7.1f"
              % (label[-18:], t, dtot, dch, pk_med, pk_max))

    # 门限覆盖：沿内「至少一个通道」超过 thr 的沿占比；以及超过的通道数占比
    print("\n== 沿覆盖（沿窗内逐通道 slope 峰值）")
    lvls = sorted(set(int(x) for x in np.logspace(np.log10(8), np.log10(200), 18)))
    print("   thr  | 沿窗内通道峰值超门限的通道占比 | 至少 1 通道超门限")
    pk_list = [np.array([e[3]]) for e in edges_all]
    for thr in lvls:
        cov = [1.0 if e[3] > thr else 0.0 for e in edges_all]
        print("   %5d | 沿窗数 %d，覆盖 %d (%.0f%%)"
              % (thr, len(cov), int(np.sum(cov)), 100.0 * np.mean(cov)))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "results", "t3_quiet_slope.npy"), Q)


if __name__ == "__main__":
    main()
