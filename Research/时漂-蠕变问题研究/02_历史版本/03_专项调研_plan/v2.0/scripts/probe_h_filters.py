# -*- coding: utf-8 -*-
"""v2.0 PROBE-H：过充候选手段的三角代价表（离线 A/B，原型驱动）。

三臂：
  base      现役原型（κ_onset=1.05 / κ_restep=1.12 / HO_MIN=3.5 / REVOKE_HOLD_N=3，与 C++ 一致）
  clamp     输出侧超前量限幅：显示 ≤ 原始 + α·|原始|（T5-A 的 A2 路线；α 扫 0.005/0.02/0.05）
  ema       输出侧因果 EMA（τ 扫 0.1/0.2 s，逐通道、含直通帧）

指标（口径先写死）：
  lead(t)   = out_total(t) − pre_total(t)        # 显示相对原始读数的超前量（"过充"的直接观感量）
  A_e       = 该次加载的台阶幅度（pre 事件后 25~30 s 平台 − 事件前 0.35~1.2 s 平台）
  OS_peak   = max(lead − lead_ss) / A_e          # 事件内相对稳态偏移的瞬时过充
  hold      = lead 超过 lead_ss + 2%·A_e 的最长连续时长（驻留）
  T5%       = 显示进入 |out − pre_ss| ≤ 5%·A_e 的首个 τ
  G         = (事件后 4~5 s 显示中位 − 事件前显示中位) / A_e
  lead_rms  = 全段 RMS(lead − lead_ss)（只统计受载段）
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
PROTO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTO)


def replay(el, V, kappa_onset=1.05, kappa_restep=1.12, ho=3.5, hold_n=3,
           clamp_alpha=None, ema_tau=None):
    """注意：过滤/限幅一律作用在**显示输出**上，且指标用显示自身的稳态做参考，
    不能拿"显示−原始"的稳态当参考（本录制该量本身长期为负 ≈ −900，会把瞬态放大成假过充）。"""
    n = len(el)
    c = PROTO.GLM53v6(L.NCH)
    c.KAPPA_ONSET = kappa_onset
    c.KAPPA_RESTEP = kappa_restep
    c.HO_MIN = ho
    out = np.empty((n, L.NCH))
    ema_state = None
    for i in range(n):
        v = V[i].copy()
        y = np.asarray(c.process(float(el[i]), v), float)
        if ema_tau is not None:
            dt = (el[i] - el[i - 1]) if i else 0.0
            dt = min(max(dt, 0.0), 0.1)
            a = min(dt / ema_tau, 1.0) if ema_tau > 0 else 1.0
            ema_state = y.copy() if ema_state is None else ema_state + a * (y - ema_state)
            y = ema_state
        if clamp_alpha is not None:
            # 单侧限幅：显示不得超过「原始 + α·|原始|」（T5-A A2 路线）
            y = np.minimum(y, V[i] + clamp_alpha * np.abs(V[i]))
        out[i] = y
    return c, out.sum(1)


def load_edges(el, tot):
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tot, 3), np.percentile(tot, 97)
    up, _ = L.edges_from_tot(np.convolve(tot, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in up:
        if not tu or el[i] - tu[-1] > 1.0:
            tu.append(float(el[i]))
    return tu


def metrics(el, pre_tot, out_tot, t_ups):
    """过充口径（**参考量是显示自身的稳态，不是"显示−原始"的稳态**）：

      base_out = 事件前 0.35~1.2 s 的显示中位
      out_ss   = 事件后 25~30 s 的显示中位（该次加载的"最终显示电平"）
      A_e      = 同一时刻原始读数的台阶幅度
      OS_tail(τ) = (out(t0+τ) − out_ss) / A_e        # 显示高于自身最终值的相对量 = 真过充
      OS_lead(τ) = (out(t0+τ) − pre(t0+τ)) / A_e     # 显示相对传感器当前的超前量 = 瞬间超前
      hold     = OS_tail > +2% 的最长连续时长（高位驻留）
    """
    rows = []
    for t_up in t_ups:
        m0 = (el >= t_up - 1.2) & (el < t_up - 0.35)
        if m0.sum() < 5:
            continue
        base_pre = float(np.median(pre_tot[m0]))
        base_out = float(np.median(out_tot[m0]))
        t_end = min(t_up + 30.0, el[-1] - 0.2)
        m1 = (el >= t_end - 1.5) & (el <= t_end)
        pre_ss = float(np.median(pre_tot[m1]))
        out_ss = float(np.median(out_tot[m1]))
        A = pre_ss - base_pre
        if abs(A) < 500:
            continue
        idx = np.where((el >= t_up - 0.05) & (el <= t_end))[0]
        tail = (out_tot[idx] - out_ss) / abs(A)
        lead = (out_tot[idx] - pre_tot[idx]) / abs(A)
        tt = el[idx]
        over = tail > 0.02
        best = run = 0.0
        for k in range(len(over)):
            if over[k]:
                run = run + (tt[k] - tt[k - 1] if k else 0.0)
                best = max(best, run)
            else:
                run = 0.0
        # 到带：显示进入 |out − out_ss| ≤ 5%·A 的首个 τ
        t5 = float("nan")
        for k in range(len(idx)):
            if abs(out_tot[idx[k]] - out_ss) <= 0.05 * abs(A):
                t5 = el[idx[k]] - t_up
                break
        G = (float(np.median(out_tot[(el >= t_up + 4.0) & (el <= t_up + 5.0)])) - base_out) / A
        rows.append(dict(t_up=t_up, A=A, tail_peak=float(np.max(tail)),
                         tail_neg=float(np.min(tail)), lead_peak=float(np.max(lead)),
                         hold=best, t5=t5, G=G,
                         bias_ss=(out_ss - pre_ss) / abs(A)))
    return rows


def summarize(tag, rows, el, out_tot, pre_tot, t_ups):
    if not rows:
        print(f"  {tag}: 无有效事件")
        return
    hold = [r["hold"] for r in rows]
    tail = [r["tail_peak"] for r in rows]
    lead = [r["lead_peak"] for r in rows]
    t5 = [r["t5"] for r in rows if r["t5"] == r["t5"]]
    g = [r["G"] for r in rows]
    bias = [r["bias_ss"] for r in rows]
    print(f"  {tag:20s} n={len(rows)}  驻留中位={np.median(hold):5.2f}s 最大={max(hold):5.2f}s  "
          f"真过充中位={100*np.median(tail):6.2f}% 最大={100*max(tail):6.2f}%  "
          f"超前量中位={100*np.median(lead):6.2f}%  "
          f"T5%中位={(np.median(t5) if t5 else float('nan')):5.2f}s  G中位={np.median(g):6.3f}  "
          f"稳态偏移中位={100*np.median(bias):6.2f}%")


def main():
    d = L.load_dataset(L.DS_ZERO)
    el = d["pre"]["el"]
    V = d["pre"]["V"]
    pre_tot = V.sum(1)
    t_ups = load_edges(el, pre_tot)
    print(f"数据集: 新录制  帧数={len(el)}  加载沿={[round(t,2) for t in t_ups]}")
    print("口径: lead = 显示总量 − 原始总量;  OS_peak = (lead − 受载稳态 lead)/台阶 ;  "
          "hold = lead 超 (lead_ss + 2%·台阶) 的最长连续时长\n")

    arms = [
        ("base(现役)", dict()),
        ("clamp α=0.005", dict(clamp_alpha=0.005)),
        ("clamp α=0.02", dict(clamp_alpha=0.02)),
        ("clamp α=0.05", dict(clamp_alpha=0.05)),
        ("ema τ=0.1s", dict(ema_tau=0.1)),
        ("ema τ=0.2s", dict(ema_tau=0.2)),
        ("clamp0.02+ema0.1", dict(clamp_alpha=0.02, ema_tau=0.1)),
    ]
    for tag, kw in arms:
        c, osum = replay(el, V, **kw)
        rows = metrics(el, pre_tot, osum, t_ups)
        summarize(tag, rows, el, osum, pre_tot, t_ups)

    # 逐事件明细：base vs clamp0.02 vs ema0.1
    print("\n逐事件（base / clamp0.02 / ema0.1；列 = 真过充% / 驻留s / T5%s / G）：")
    detail = {}
    for tag, kw in (("base", dict()), ("c02", dict(clamp_alpha=0.02)),
                    ("ema1", dict(ema_tau=0.1))):
        _c, osum = replay(el, V, **kw)
        detail[tag] = metrics(el, pre_tot, osum, t_ups)
    print(f"{'t_up':>8}{'A_e':>8} | " + " | ".join(
        f"{k:^30s}" for k in detail))
    print(f"{'':8}{'':8} | " + " | ".join(
        f"{'OSpk%':>8}{'hold':>8}{'T5%':>7}{'G':>7}" for _ in detail))
    for i in range(min(len(v) for v in detail.values())):
        row = f"{detail['base'][i]['t_up']:8.2f}{detail['base'][i]['A']:8.0f} | "
        for k in detail:
            r = detail[k][i]
            row += (f"{100*r['tail_peak']:8.2f}{r['hold']:8.2f}"
                    f"{(r['t5'] if r['t5'] == r['t5'] else float('nan')):7.2f}{r['G']:7.3f} | ")
        print(row)


if __name__ == "__main__":
    main()
