# -*- coding: utf-8 -*-
"""v3.4 规律辨识 v3：双 Voigt 元线性粘弹性叠加律 + 全平台轨迹拟合。

模型（线性粘弹性标准形式，蠕变幅度 ∝ 载荷）：
  y(t) = E(class) + x_fast(t) + x_slow(t)
  受载(class≠0): x_k 朝 r_k·(E[class]−E[0]) 以 τk_c 增长；
  空载:          x_k 朝 0 以 τk_r 恢复。
  p = [E0,E1,E2, r_f, r_s, τf_c, τs_c, τf_r, τs_r]  （9 参数）

观测：每个平台段内部（去头尾 1.5 s）全部采样点 → 同时约束落点、保压蠕变
（快/慢两个时间常数）、零点漂移与重载落点（恢复）。
检验：① 拟合残差；② 留一法（去掉一个段的观测重拟合）预测该段落点，
      与两个朴素预测器（上一次同类落点 / 全程平均）对比。
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))

SESSIONS = [
    ("目标working_7b3977", os.path.join(
        L.DATA_ROOT, "working", "零基线-反复增减同一负载",
        "20260919_160854_single_device_7b3977")),
    ("归档_0cb8b6", os.path.join(
        L.DATA_ROOT, "archived", "零基线-反复增减同一负载",
        "20260919_152749_single_device_0cb8b6")),
    ("归档_f9740b_恒载反复加减", os.path.join(
        L.DATA_ROOT, "archived", "从零基线开始 - 恒定负载 - 反复加减同一个负载",
        "20260919_100351_single_device_f9740b")),
]


def segments(el, s, win=0.5, frac=0.05):
    rng = s.max() - s.min()
    thr = frac * rng
    n = len(s)
    k = int(win * 100)
    edges = [0]
    t_prev = el[0]
    for i in range(k, n - k, 10):
        if el[i] - t_prev < win:
            continue
        a = float(np.mean(s[i - k:i]))
        b = float(np.mean(s[i:i + k]))
        if abs(b - a) > thr:
            edges.append(i)
            t_prev = el[i]
    edges.append(n)
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 60:
            continue
        segs.append([el[a], el[b - 1], float(np.median(s[a:b]))])
    return segs


def classify(segs):
    lvs = sorted(g[2] for g in segs)
    uniq = []
    for v in lvs:
        if not uniq or v - uniq[-1] > 0.02 * lvs[-1]:
            uniq.append(v)
    ncls = min(3, len(uniq))
    cuts = []
    if len(uniq) > ncls >= 2:
        gaps = sorted(((uniq[i + 1] - uniq[i], i) for i in range(len(uniq) - 1)),
                      reverse=True)
        cuts = sorted(uniq[i] + (uniq[i + 1] - uniq[i]) / 2 for _, i in gaps[:ncls - 1])
    return [(t0, t1, lv, sum(1 for cut in cuts if lv > cut))
            for t0, t1, lv in segs]


def model_curve2(p, segs, t_grid, dt=0.1):
    E = {c: p[c] for c in (0, 1, 2)}
    rf, rs = p[3], p[4]
    tfc, tsc, tfr, tsr = p[5], p[6], p[7], p[8]
    x1 = x2 = 0.0
    res_y = np.full(len(t_grid), np.nan)
    k = 0
    for t0, t1, lv, c in segs:
        amp = max(E[c] - E[0], 0.0)
        a1, a2 = ((rf * amp, rs * amp) if c > 0 else (0.0, 0.0))
        tc1, tc2 = (tfc, tsc) if c > 0 else (tfr, tsr)
        tt = t0
        while tt < t1:
            h = min(dt, t1 - tt)
            x1 += (h / tc1) * (a1 - x1)
            x2 += (h / tc2) * (a2 - x2)
            tt += h
            while k < len(t_grid) and t_grid[k] <= tt:
                res_y[k] = E[c] + x1 + x2
                k += 1
    return res_y


def make_obs(el, s, segs):
    ts, ys, sid = [], [], []
    for j, (t0, t1, lv, c) in enumerate(segs):
        m = (el >= t0 + 1.5) & (el <= t1 - 1.0)
        if m.sum() < 15:
            continue
        for i in np.where(m)[0][::50]:
            ts.append(el[i])
            ys.append(s[i])
            sid.append(j)
    return np.array(ts), np.array(ys), np.array(sid)


def fit(el, s, segs, obs, drop_seg=None):
    ts, ys, sid = obs
    keep = sid != drop_seg if drop_seg is not None else np.ones(len(sid), bool)
    E0 = []
    for c in range(3):
        lvs = [g[2] for g in segs if g[3] == c]
        E0.append(float(np.median(lvs)) if lvs else np.nan)
    base = np.nanmin([v for v in E0 if not np.isnan(v)])
    E0 = [base if np.isnan(v) else v for v in E0]
    top = max(g[2] for g in segs)
    span = max(top - base, 1.0)

    def resid(p):
        return model_curve2(p, segs, ts[keep]) - ys[keep]

    p0 = [E0[0], E0[1], E0[2], 0.10, 0.10, 12.0, 200.0, 8.0, 120.0]
    lo_b = [base - 0.2 * span] * 3 + [0.0, 0.0, 2.0, 30.0, 2.0, 10.0]
    hi_b = [base + 1.2 * span] * 3 + [0.5, 0.5, 60.0, 900.0, 60.0, 900.0]
    p0 = [float(np.clip(v, l + 1e-9, h - 1e-9))
          for v, l, h in zip(p0, lo_b, hi_b)]
    sol = least_squares(resid, p0, bounds=(lo_b, hi_b), max_nfev=4000)
    return sol


def main():
    out = []
    fits = {}
    for tag, d in SESSIONS:
        s = L.load_stream(d, "device_001_pre_seg0.csv")
        el = s["el"]
        tot = s["V"].sum(axis=1)
        segs = classify(segments(el, tot))
        obs = make_obs(el, tot, segs)
        sol = fit(el, tot, segs, obs)
        p = sol.x
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        out.append("== %s  (%d 段, %d 观测点)" % (tag, len(segs), len(obs[0])))
        out.append("   E_zero=%7.0f E_half=%7.0f E_full=%7.0f" % (p[0], p[1], p[2]))
        out.append("   r_fast=%.2f r_slow=%.2f  τf_c=%5.1fs τs_c=%6.1fs "
                   "τf_r=%5.1fs τs_r=%6.1fs" % (p[3], p[4], p[5], p[6], p[7], p[8]))
        out.append("   全轨迹残差 RMS = %.0f ADC（量程的 %.1f%%）"
                   % (rms, 100.0 * rms / max(max(g[2] for g in segs), 1)))

        # 留一法：逐段剔除重拟合，预测该段中部电平，对比朴素预测器
        errs_model, errs_prev, errs_mean = [], [], []
        by_class = {}
        for j, (t0, t1, lv, c) in enumerate(segs):
            m = (el >= t0 + 1.5) & (el <= t1 - 1.0)
            if m.sum() < 15:
                continue
            actual = float(np.median(tot[m]))
            sol_o = fit(el, tot, segs, obs, drop_seg=j)
            tc = np.array([t0 + 1.5 + (t1 - t0 - 2.5) / 2.0])
            pred = float(model_curve2(sol_o.x, segs, tc)[0])
            errs_model.append(abs(pred - actual))
            if c in by_class:
                errs_prev.append(abs(by_class[c][-1] - actual))
            by_class.setdefault(c, []).append(actual)
            errs_mean.append(abs(np.median(by_class[c][:-1] if len(by_class[c]) > 1
                                           else by_class[c]) - actual) if len(by_class[c]) > 1 else abs(pred - actual))
        if errs_model:
            out.append("   留一预测 |误差| 中位: 模型 %4.0f | 朴素(上次同类) %4.0f"
                       % (np.median(errs_model), np.median(errs_prev) if errs_prev else -1))
        fits[tag] = (el, tot, segs, obs, sol)
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "..", "results", "v34_law_identify.txt"),
              "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    tag = "目标working_7b3977"
    el, tot, segs, obs, sol = fits[tag]
    grid = np.arange(el[0], el[-1], 0.2)
    mc = model_curve2(sol.x, segs, grid)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(el, tot, color="#aaa", lw=0.6, label="输入（补偿前）")
    ax.plot(grid, mc, color="#d62728", lw=1.2,
            label="双 Voigt 粘弹性模型（τf=%.0fs τs=%.0fs τr=%.0f/%.0fs）"
                  % (sol.x[5], sol.x[6], sol.x[7], sol.x[8]))
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("21 通道总量 (ADC)")
    ax.set_title("%s：受载蠕变 + 静置恢复的可预测性检验" % tag)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "figure",
                             "v34_粘弹性规律辨识_7b3977.png"), dpi=140)
    print("figure saved")


if __name__ == "__main__":
    main()
