# -*- coding: utf-8 -*-
"""r3：快相形状的稳定性归因 —— 是「同一形状的时间缩放」还是「形状本身在变」？

做法：对每个 onset 事件，用一参数时间缩放 α 拟合 v6 的 ROM（g(τ/α) 最小二乘），
  · 若残差远小于直接把 ROM 套上去的残差 ⇒ 形状稳定，只是"快慢"不同（可用 1 个参数自适应）；
  · 若残差仍大 ⇒ 形状真的在变（换形状库/多库/在线辨识）。
另外给出：
  · 按传感器 / 按 onset-restep 分组的形态统计；
  · 「最快形状上包络 ROM」（v6.1 的做法）下的过充；
  · 幅度相关性与 ROM 假设（形状与幅度无关）的检验。
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(OUT)), "progress", "04-v5", "scripts"))

ROM_TAU = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80,
                    1.00, 1.50, 2.00, 3.00, 4.00, 5.00])
ROM_G = np.array([0.0, 0.680, 0.740, 0.790, 0.824, 0.854, 0.873, 0.886,
                  0.904, 0.922, 0.941, 0.970, 0.989, 1.000])
TG = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80])


def g_of(tau, tg=ROM_TAU, gg=ROM_G):
    return np.interp(np.asarray(tau, float), tg, gg, left=0.0, right=1.0)


def main():
    ev = pd.read_csv(os.path.join(RES, "events.csv"))
    sh = [c for c in ev.columns if c.startswith("sh_")]
    ev["y"] = ev[sh].apply(lambda r: list(r.values)[:7], axis=1)   # 前 7 点 = TG[0.05..0.80]

    # ── ① 分组形态 ────────────────────────────────────────────────
    ev["sensor"] = np.where(ev.rec.str.startswith("右"), "右拇指",
                            np.where(ev.rec.str.startswith("左"), "左拇指",
                                     np.where(ev.rec.str.startswith("四"), "四指", "实录")))
    g1 = ev[ev.kind == "onset"].groupby(["sensor", "kind"]).agg(
        n=("A5", "size"), step_frac=("step_frac", "median"), s02=("sh_20", "median"),
        s05=("sh_50", "median"), s80=("sh_80", "median"),
        over=("over_vs_A5", "median"), over_max=("over_vs_A5", "max"),
        A5_med=("A5", "median")).round(2)
    g2 = ev[ev.kind == "restep"].groupby(["sensor", "kind"]).agg(
        n=("A5", "size"), step_frac=("step_frac", "median"), s02=("sh_20", "median"),
        s05=("sh_50", "median"), s80=("sh_80", "median"),
        over=("over_vs_A5", "median"), over_min=("over_vs_A5", "min"),
        A5_med=("A5", "median")).round(2)
    print("== onset 分组 ==");  print(g1.to_string())
    print("\n== restep 分组 ==");  print(g2.to_string())

    # ── ② 时间缩放拟合 ───────────────────────────────────────────
    rows = []
    ons = ev[(ev.kind == "onset") & ev.A5.gt(0)]
    for _, r in ons.iterrows():
        y = np.array(r["y"], float)
        if np.isnan(y).any():
            continue
        best, bres = None, None
        for a in np.arange(0.3, 3.01, 0.02):
            gg = g_of(TG / a)
            res = float(((y - gg) ** 2).mean())
            if bres is None or res < bres:
                best, bres = a, res
        gg0 = g_of(TG)
        res0 = float(((y - gg0) ** 2).mean())
        rows.append(dict(rec=r.rec, sensor=r.sensor, A5=r.A5, alpha=round(best, 2),
                         rms0=round(np.sqrt(res0), 4), rms_alpha=round(np.sqrt(bres), 4),
                         over0=r.over_vs_A5))
    fit = pd.DataFrame(rows)
    fit.to_csv(os.path.join(RES, "shape_timewarp.csv"), index=False, encoding="utf-8-sig")
    print("\n== 时间缩放拟合（onset，n=%d）==" % len(fit))
    print("  α：中位 %.2f  p10 %.2f  p90 %.2f  极差 %.2f" %
          (fit.alpha.median(), fit.alpha.quantile(.1), fit.alpha.quantile(.9),
           fit.alpha.max() - fit.alpha.min()))
    print("  直接用 ROM 的 RMS 残差 中位 %.4f  →  允许时间缩放后 中位 %.4f（降 %.0f%%）"
          % (fit.rms0.median(), fit.rms_alpha.median(),
             100 * (1 - fit.rms_alpha.median() / max(fit.rms0.median(), 1e-9))))

    # ── ③ 最快形状上包络 ROM（v6.1 思路）与「中位 ROM」的过充对比 ──
    med = np.array([ons[c].median() for c in sh[:7]])
    fast = np.array([ons[c].quantile(.95) for c in sh[:7]])
    def Ahat(y, gg, kappa):
        m = TG >= 0.20
        inc = y[-1]
        if inc <= 0 or (gg[m] ** 2).sum() <= 0:
            return np.nan
        A = float((y[m] * gg[m]).sum()) / float((gg[m] ** 2).sum())
        return float(min(max(A, inc), kappa * inc))
    cmp_rows = []
    for _, r in ons.iterrows():
        y = np.array(r["y"], float)
        if np.isnan(y).any():
            continue
        cmp_rows.append(dict(rec=r.rec, sensor=r.sensor,
                             over_rom=100 * (Ahat(y, g_of(TG), r.kappa) - 1),
                             over_med=100 * (Ahat(y, med, r.kappa) - 1),
                             over_fast=100 * (Ahat(y, fast, r.kappa) - 1),
                             over_timewarp=100 * (Ahat(y, g_of(TG / (fit[fit.rec == r.rec].alpha.iloc[0]
                                                                     if len(fit[fit.rec == r.rec]) else 1.0)),
                                                       r.kappa) - 1)))
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(os.path.join(RES, "rom_compare.csv"), index=False, encoding="utf-8-sig")
    print("\n== 不同 ROM 下的过充（onset，%%；正=超真值，负=欠）==")
    print(cmp[["over_rom", "over_med", "over_fast", "over_timewarp"]].describe()
          .loc[["mean", "50%", "min", "max"]].round(2).to_string())

    # ── ④ 幅度相关性（ROM 假设：形状与幅度无关）────────────────
    print("\n== 幅度相关性（onset）==")
    for dom, sub in ons.groupby("dom"):
        if len(sub) < 4:
            continue
        c1 = np.corrcoef(sub.A5, sub.step_frac)[0, 1]
        c2 = np.corrcoef(sub.A5, sub.over_vs_A5)[0, 1]
        print("  %-6s n=%2d  corr(A5, 0.2s占比)=%+.2f   corr(A5, 过充)=%+.2f"
              % (dom, len(sub), c1, c2))
    print("  同一传感器内 3 次重复的 0.2s 占比极差：")
    for s, sub in ons[ons.sensor != "实录"].groupby("sensor"):
        print("    %-6s %s（极差 %.1f 个百分点）"
              % (s, np.round(sub.step_frac.values, 1), sub.step_frac.max() - sub.step_frac.min()))
    print("\n-> results/shape_timewarp.csv、rom_compare.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
