# -*- coding: utf-8 -*-
"""步骤 C2：因果算法横评（最终版，抗作弊口径）。

主指标
------
* drift_raw_mN  : 原始信号 (末5s均值 - 1~3s均值)，mN        —— 时漂绝对量
* drift_algo_mN : 算法输出 同口径，mN                        —— 残余时漂（越小越好）
* drift_reduc   : 1 - |drift_algo|/|drift_raw|               —— 时漂抑制率
* cv_raw/cv_algo: 平台期（1~3s 之后到卸载）输出变异系数      —— 读数恒定度
* gain          : 算法在 1~3s 的输出 / 原始水平（<<1 = 把真实力也扣了）
* step_fid      : 加载建立过程的净上升保真度
* noise_hi/lo   : 平台期 0.01~1Hz 带内噪声 / <0.01Hz 慢变起伏 之比
"""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_harness import (prepare, run_stream, verify_causality, evaluate,
                            evaluate_multi, online_load_detect)
from tac_common import dump_json, DATASETS, RES
import causal_algorithms as CA
from causal_algorithms import Shape, FROZEN

np.set_printoptions(precision=4, suppress=True, linewidth=200)
lines = []


def P(s=""):
    print(s, flush=True)
    lines.append(s)


def build():
    S1 = Shape(**FROZEN["d1"])
    SC = Shape(**FROZEN["common"])
    a = [CA.Raw()]
    for tau in (1.0, 3.0, 10.0, 30.0):
        a.append(CA.CausalEWMA(tau))
    for w_ in (1.0, 5.0, 20.0):
        a.append(CA.CausalMA(w_))
    for w_ in (1.0, 5.0):
        a.append(CA.CausalMedian(w_))
    for fc in (0.005, 0.01, 0.02, 0.05):
        a.append(CA.CausalHighPass(fc, 2))
    for lam in (0.9999, 0.99999, 0.999999):
        a.append(CA.CausalRLSTrend(lam, 1))
    for q in (1e-11, 1e-10, 1e-9, 1e-8):
        a.append(CA.CausalKalmanCV(q, 1e-5))
    for ts in (0.3, 1.0, 3.0, 10.0):
        a.append(CA.ShapeDivide(SC, ts))
    a.append(CA.ShapeDivide(S1, 3.0))
    a.append(CA.ShapeBackCalc(SC, 3.0, 1.0))
    a.append(CA.TwoPointExtrapolate(SC, 2.0, 20.0, 3.0))
    a.append(CA.AdaptiveShapeRLS(SC.p, 0.99995))
    for g in (0.5, 1.0):
        a.append(CA.DualEMATrend(30.0, 3.0, g))
    for use in ("sum", "mean", "median"):
        a.append(CA.ArrayCommonMode(3.0, 2.0, use))
    for rs in (1.0, 5.0):
        a.append(CA.ArrayCommonMode(rs, 2.0, "sum"))
    a.append(CA.ArrayCommonModePerChannel(3.0, 2.0))
    a.append(CA.ArrayShapeShared(SC.p))
    return a


HDR = (f"    {'算法':<38s} {'漂移原始mN':>10s} {'漂移残余mN':>10s} {'抑制率':>7s} "
       f"{'CV原始%':>8s} {'CV算法%':>8s} {'gain':>6s} {'阶跃保真':>8s} "
       f"{'噪带%':>6s} {'慢变%':>6s} {'卸载后mN':>9s} {'因果':>5s} {'ms':>7s}")

ALL = {}
for name in DATASETS:
    D = prepare(name)
    X, t, fs = D["X"], D["t"], D["fs"]
    a0, a1 = D["pre"]; c0, d0 = D["load"]
    w = lambda s: int(round(s * fs))
    Xn = X - X[a0:a1].mean(axis=0)[None, :]
    tot = Xn.sum(axis=1)
    n_on, base, thr = online_load_detect(tot, fs, a1)
    on_true = D["on_true"]
    act = np.where(Xn[c0:d0].max(axis=0) > 0.05)[0]
    P("=" * 150)
    P(f"### {name}  fs={fs:.3f}Hz  N={len(t)}")
    P(f"    离线沿: 加载起 {on_true}(t={t[on_true]:.3f}s) 卸载止 {D['off_true']}"
      f"(t={t[D['off_true']]:.3f}s)   在线检测 n_on={n_on}(t={t[n_on]:.3f}s) "
      f"偏差 {(n_on-on_true)/fs*1000:+.0f}ms")
    P(f"    受载通道 {len(act)} 个  负载段 {(d0-c0)/fs:.1f}s")
    P(HDR)
    rows = []
    for c in build():
        t0 = time.time()
        try:
            rr = evaluate(c, D, act=act, n_on=n_on)
            vc = verify_causality(c, Xn, t, fs, n_on, n_check=6)
            el = (time.time() - t0) * 1000
        except Exception as ex:
            P(f"    {c.name:<38s}  FAILED: {type(ex).__name__}: {ex}")
            continue
        rec = dict(algo=c.name, group=c.group,
                   **{k: v for k, v in rr.items()
                      if k not in ("Y", "base", "act", "Fref", "n_on")},
                   causal_ok=bool(vc < 1e-9), ms=el)
        rows.append(rec)
        P(f"    {c.name:<38s} {rr['drift_raw_mN']:>10.1f} {rr['drift_algo_mN']:>10.1f} "
          f"{rr['drift_reduction']*100:>6.1f}% {rr['cv_raw']*100:>7.2f}% "
          f"{rr['cv_algo']*100:>7.2f}% {rr['gain']:>6.3f} {rr['step_fid']:>8.3f} "
          f"{rr['noise_hi']:>6.1f} {rr['noise_lo']:>6.1f} {rr['zero_post']:>9.3f} "
          f"{'OK' if vc < 1e-9 else 'BAD':>5s} {el:>7.0f}")
    ALL[name] = dict(rows=rows, n_on=int(n_on), c0=int(c0), on_true=int(on_true),
                     act=[int(i) for i in act],
                     det_delay_ms=float((n_on - on_true) / fs * 1000))

dump_json(ALL, "C2_causal_sweep.json")

# ---------------- 汇总 ----------------
P("")
P("=" * 150)
P("### 三组数据汇总（各指标取三组均值）")
agg = {}
for name in DATASETS:
    for r in ALL[name]["rows"]:
        a = agg.setdefault(r["algo"], dict(group=r["group"], **{
            k: [] for k in ("drift_algo_mN", "drift_reduction", "cv_algo", "cv_raw",
                            "gain", "step_fid", "noise_hi", "noise_lo",
                            "zero_post", "rough_ratio")}))
        for k in ("drift_algo_mN", "drift_reduction", "cv_algo", "cv_raw", "gain",
                  "step_fid", "noise_hi", "noise_lo", "zero_post", "rough_ratio"):
            a[k].append(abs(r[k]) if k == "zero_post" else r[k])
        a["ok"] = a.get("ok", True) and r["causal_ok"]
summary = []
for k, a in agg.items():
    summary.append(dict(algo=k, group=a["group"], causal_ok=bool(a["ok"]), **{
        kk: float(np.mean(vv)) for kk, vv in a.items()
        if kk not in ("group", "ok")}))
P(f"  {'算法':<38s} {'漂移残余mN':>10s} {'抑制率':>7s} {'CV原始%':>8s} {'CV算法%':>8s} "
  f"{'gain':>6s} {'阶跃保真':>8s} {'噪带%':>6s} {'慢变%':>6s} {'|卸载后|mN':>10s} "
  f"{'分组':<20s}")
for r in sorted(summary, key=lambda x: x["drift_algo_mN"]):
    P(f"  {r['algo']:<38s} {r['drift_algo_mN']:>10.1f} {r['drift_reduction']*100:>6.1f}% "
      f"{r['cv_raw']*100:>7.2f}% {r['cv_algo']*100:>7.2f}% {r['gain']:>6.3f} "
      f"{r['step_fid']:>8.3f} {r['noise_hi']:>6.1f} {r['noise_lo']:>6.1f} "
      f"{r['zero_post']:>10.3f} {r['group']:<20s}")
dump_json(summary, "C2_summary.json")

# ---------------- 敏感性 ----------------
P("")
P("### 参考窗敏感性（数据1：漂移残余 mN / gain / 阶跃保真）")
D = prepare("数据1")
SC = Shape(**FROZEN["common"])
key = [CA.Raw(), CA.ShapeDivide(SC, 3.0), CA.ArrayCommonMode(3.0, 2.0, "sum"),
       CA.CausalKalmanCV(1e-10, 1e-5), CA.AdaptiveShapeRLS(SC.p, 0.99995),
       CA.CausalHighPass(0.01, 2)]
sens = {}
for c in key:
    m = evaluate_multi(c, D)
    sens[c.name] = {k: dict(drift=float(v["drift_algo_mN"]), gain=float(v["gain"]),
                            fid=float(v["step_fid"]))
                    for k, v in m.items()}
    P(f"  {c.name:<38s} " + "  ".join(
        f"{k}:{v['drift_algo_mN']:+.0f}mN/{v['gain']:.2f}" for k, v in m.items()))
dump_json(sens, "C2_sensitivity.json")

P("")
P("### 加载时刻偏差敏感性（数据1：漂移残余 mN / gain）")
D = prepare("数据1")
X, t, fs = D["X"], D["t"], D["fs"]
a0, a1 = D["pre"]; c0, d0 = D["load"]
Xn = X - X[a0:a1].mean(axis=0)[None, :]
n_on0, _, _ = online_load_detect(Xn.sum(axis=1), fs, a1)
act = np.where(Xn[c0:d0].max(axis=0) > 0.05)[0]
onsens = {}
for shift_ms in (-1000, -500, -250, 0, 250, 500, 1000):
    n_on = n_on0 + int(round(shift_ms / 1000 * fs))
    row = {}
    for c in [CA.ShapeDivide(SC, 3.0), CA.ArrayCommonMode(3.0, 2.0, "sum"),
              CA.AdaptiveShapeRLS(SC.p, 0.99995)]:
        rr = evaluate(c, D, act=act, n_on=n_on)
        row[c.name] = dict(drift=float(rr["drift_algo_mN"]), gain=float(rr["gain"]),
                           fid=float(rr["step_fid"]))
    onsens[shift_ms] = row
    P(f"  偏移 {shift_ms:+5d}ms: " + "  ".join(
        f"{k.split('(')[0][:14]}:{v['drift']:+.0f}mN/g{v['gain']:.2f}"
        for k, v in row.items()))
dump_json(onsens, "C2_onset_sensitivity.json")

with open(os.path.join(RES, "C2_causal_sweep.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[ok] C2 完成")
