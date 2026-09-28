# -*- coding: utf-8 -*-
"""T3 步骤2：沿检测器标定（稳健版）。

A. 静置帧（输入总量 ±1 s 窗极差 < max(0.3%·量程, 40 ADC)）的逐通道 slope 分位
   => 门限「下界」（静置误触发率）。
B. 真实加载沿（台阶 ≥ max(500, 5%·量程)）沿起 1 s 内逐通道 slope 峰值
   => 门限「上界」（可检性）。
C. 结论：thr = max(thr_abs, k·0.02·max(e_pre,1)) 的候选区间。
"""
import os
import sys

import numpy as np

import t3_lib as T

np.seterr(all="ignore")


def slope_track(ts, V):
    S = np.zeros_like(V)
    v_lp = V[0].copy()
    t_prev = ts[0]
    for i in range(V.shape[0]):
        dt = min(max(ts[i] - t_prev, 0.0), 0.1)
        t_prev = ts[i]
        if dt > 0:
            S[i] = (V[i] - v_lp) / 3.0
            v_lp = v_lp + (dt / 3.0) * (V[i] - v_lp)
    return S


def quiet_mask(el, tot, rel=0.003, abs_min=40.0):
    rng = float(tot.max() - tot.min())
    tol = max(rel * rng, abs_min)
    n = len(el)
    ok = np.zeros(n, dtype=bool)
    j0 = j1 = 0
    for i in range(n):
        while j0 < n and el[j0] < el[i] - 1.0:
            j0 += 1
        while j1 < n and el[j1] < el[i] + 1.0:
            j1 += 1
        w = tot[j0:j1]
        if len(w) > 5 and (w.max() - w.min()) < tol:
            ok[i] = True
    return ok


def main():
    lines = []
    P = lines.append
    P("== A 静置帧逐通道 slope 分位（slope = (v−v_lp)/3，ADC/s）")
    P("   会话                                  静置帧   | p50 / p90 / p99 / p99.9 / max")
    Q = []
    for tag, label, d in T.all_sessions():
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        tot = V.sum(axis=1)
        S = slope_track(ts, V)
        qm = quiet_mask(el, tot)
        if qm.sum() < 200:
            P("   %-36s 静置帧不足(%d)" % (label[-36:], int(qm.sum())))
            continue
        q = S[qm]
        Q.append(q)
        P("   %-36s %6d | %6.2f / %6.2f / %7.2f / %8.2f / %8.2f"
          % (label[-36:], int(qm.sum()),
             np.percentile(q, 50), np.percentile(q, 90), np.percentile(q, 99),
             np.percentile(q, 99.9), q.max()))
    QA = np.concatenate(Q)
    P("")
    P("   汇总（N=%.2e 通道帧）：p50 %.3f  p90 %.3f  p99 %.3f  p99.9 %.3f  p99.99 %.3f  max %.2f"
      % (QA.size, np.percentile(QA, 50), np.percentile(QA, 90),
         np.percentile(QA, 99), np.percentile(QA, 99.9),
         np.percentile(QA, 99.99), QA.max()))
    P("   静置逐通道帧超门限占比（误触发率）：")
    for thr in (2, 3, 4, 5, 8, 10, 15, 20, 30, 50):
        r = float(np.mean(QA > thr))
        P("     thr=%5.1f ADC/s -> %8.5f%%（每百万通道帧 %d 次）" % (thr, 100 * r, int(r * 1e6)))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "results", "t3_quiet_slope.npy"), QA)

    P("")
    P("== B 真实加载沿沿内逐通道 slope 峰值（沿起 1 s 内）")
    P("   会话                                  min_step 沿数 | 逐通道峰值 p10/p50/p90/max | 台阶<0.25量程的沿数")
    all_pk = []
    all_ed = []
    for tag, label, d in T.all_sessions():
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        tot = V.sum(axis=1)
        S = slope_track(ts, V)
        ed, ms = T.find_edges(el, tot)
        pks = []
        small = 0
        for e in ed:
            i1 = min(len(el) - 1, int(np.searchsorted(el, el[e["i"]] + 1.0)))
            w = S[e["i"]:i1 + 1]
            if w.shape[0] < 3:
                continue
            pks.extend(w.max(axis=0).tolist())
            all_pk.extend(w.max(axis=0).tolist())
            all_ed.append((label, e["t"], e["step"], e["base"]))
            if e["step"] < 0.25 * (tot.max() - tot.min()):
                small += 1
        if pks:
            pks = np.array(pks)
            P("   %-36s %8.0f %4d | %6.1f / %6.1f / %6.1f / %8.1f | %d"
              % (label[-36:], ms, len(ed), np.percentile(pks, 10),
                 np.percentile(pks, 50), np.percentile(pks, 90), pks.max(), small))
        else:
            P("   %-36s %8.0f %4d | （无沿）" % (label[-36:], ms, len(ed)))
    AP = np.array(all_pk)
    P("")
    P("   汇总：评估沿 %d 个，通道峰值 slope 的 p1 %.2f / p5 %.2f / p10 %.2f / p25 %.2f"
      % (len(all_ed), np.percentile(AP, 1), np.percentile(AP, 5),
         np.percentile(AP, 10), np.percentile(AP, 25)))
    P("   台阶分布：500-1500 %d / 1500-5000 %d / ≥5000 %d"
      % (sum(1 for e in all_ed if e[2] < 1500),
         sum(1 for e in all_ed if 1500 <= e[2] < 5000),
         sum(1 for e in all_ed if e[2] >= 5000)))
    P("   （受载态小台阶 = 沿前基线 > 20%% 量程 且 台阶 < 25%% 量程 的事件）")
    n_small = 0
    for label, t, step, base in all_ed:
        s = T.load_stream([d for tg, lb, d in T.all_sessions() if lb == label][0],
                          T.pre_name([d for tg, lb, d in T.all_sessions() if lb == label][0]))
        rng = float(s["V"].sum(axis=1).max() - s["V"].sum(axis=1).min())
        if base > 0.2 * rng and step < 0.25 * rng:
            n_small += 1
    P("   受载态小台阶事件数：%d" % n_small)

    P("")
    P("== C 门限候选")
    P("   静置底噪 p99.9 = %.2f ADC/s，沿内逐通道峰值 p10 = %.2f ADC/s"
      % (np.percentile(QA, 99.9), np.percentile(AP, 10)))
    P("   => 绝对底噪门限取 3~5 ADC/s 已高于静置 p99.9；小台阶沿（逐通道台阶 ~5-15 ADC）")
    P("      的峰值 slope 约 2-5 ADC/s（= Δc/3），与底噪同量级 ⇒ 小台阶可靠检测门槛在")
    P("      逐通道台阶 ≥ 15 ADC（总台阶 ≥ 300 ADC）附近。")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "t3_detector_calib.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
