# -*- coding: utf-8 -*-
"""步骤 C1：漂移形态标定（只在 数据1 上做，随后冻结用于 数据2/3 盲测）。

模型：h(τ) = 1 + c·τ^p     （τ = 加载后秒数，h(0)=1）
      ⇒ 因果 ODE 形式：ds/dτ = c·p·F̂·τ^(p-1) - λ·s，  F̂ = y/(1+s)

标定量：p、c（= 形态）、λ（衰减）
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, dump_json, DATASETS, RES

np.set_printoptions(precision=5, suppress=True)
lines = []


def P(s=""):
    print(s)
    lines.append(s)


def fit_power_shape(tau, h):
    """h = 1 + c·tau^p，网格搜 p、线性解 c（含截距，最后归一化到 h(0)=1）。"""
    best = None
    for p in np.arange(0.02, 1.001, 0.005):
        phi = np.power(np.maximum(tau, 0.0), p)
        A = np.vstack([phi, np.ones_like(phi)]).T
        coef, *_ = np.linalg.lstsq(A, h, rcond=None)
        res = h - A @ coef
        ss = float(res @ res)
        if best is None or ss < best[0]:
            best = (ss, float(p), float(coef[0]), float(coef[1]))
    ss, p, c, b = best
    tot = float(np.sum((h - h.mean()) ** 2))
    return p, c, b, (1 - ss / tot if tot > 0 else np.nan)


CAL = {}
for name in DATASETS:
    D = load_dataset(name)
    fi = D["frame_index"]
    k, _ = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), t, fs=fs)
    a0, a = seg["pre"]; c0, d0 = seg["load"]
    Xn = X - X[a0:a].mean(axis=0)[None, :]
    w = lambda s: int(round(s * fs))
    tau = t[c0:d0] - t[c0]
    Fref = Xn[c0 + w(1.0):c0 + w(3.0)].mean(axis=0)
    act = np.where(Fref > 0.05)[0]
    h = Xn[c0:d0, act] / Fref[act][None, :]
    # 用高幅通道加权平均，得到共享形态
    wts = Fref[act] / Fref[act].sum()
    h_sh = h @ wts
    p, c, b, r2 = fit_power_shape(tau, h_sh)
    # 只在高 SNR 区间评估
    m = tau > 1.0
    P("=" * 88)
    P(f"### {name}  共享形态拟合 h(τ)=1+c·τ^p")
    P(f"    p={p:.4f}  c={c:.5f}  截距 b={b:.5f}  R²={r2:.5f}  (拟合全段)")
    p2, c2, b2, r22 = fit_power_shape(tau[m], h_sh[m])
    P(f"    仅 τ>1s: p={p2:.4f} c={c2:.5f} b={b2:.5f} R²={r22:.5f}")
    P(f"    形态抽样: " + "  ".join(
        f"τ={tt:6.1f}s→{1 + c*tt**p:.4f}" for tt in (1, 5, 10, 30, 60, 100, 150)))
    P(f"    实测形态: " + "  ".join(
        f"τ={tt:6.1f}s→{h_sh[min(int(tt*fs), len(h_sh)-1)]:.4f}"
        for tt in (1, 5, 10, 30, 60, 100, 150)))
    # 相对漂移（末端 vs 参考窗口）
    drift = (Xn[d0 - w(2):d0, act].mean(axis=0) / Fref[act] - 1) * 100
    P(f"    实测相对漂移(末2s): 中位={np.median(drift):.2f}%  "
      f"拟合预测={((1 + c*(tau[-1]-1)**p) - 1)*100:.2f}%")
    CAL[name] = dict(fs=float(fs), p=float(p), c=float(c), r2=float(r2),
                     p_gt1=float(p2), c_gt1=float(c2), r2_gt1=float(r22),
                     t_load=float(t[d0] - t[c0]), n_on=int(c0), n_off=int(d0),
                     n_pre=int(a), drift_med=float(np.median(drift)))
    dump_json(CAL[name], f"C1_shape_{DATASETS.index(name)}.json") if False else None

P("")
P("标定结论：")
P(f"  数据1（标定集）: p={CAL['数据1']['p']:.4f}, c={CAL['数据1']['c']:.5f}, "
  f"R²={CAL['数据1']['r2']:.5f}")
P(f"  数据2（盲测）  : p={CAL['数据2']['p']:.4f}, c={CAL['数据2']['c']:.5f}, "
  f"R²={CAL['数据2']['r2']:.5f}  实测漂移={CAL['数据2']['drift_med']:.1f}%")
P(f"  数据3（盲测）  : p={CAL['数据3']['p']:.4f}, c={CAL['数据3']['c']:.5f}, "
  f"R²={CAL['数据3']['r2']:.5f}  实测漂移={CAL['数据3']['drift_med']:.1f}%")

# 冻结参数（来自数据1）
p_frozen = CAL["数据1"]["p"]
c_frozen = CAL["数据1"]["c"]
CAL["frozen"] = dict(
    p=p_frozen, c=c_frozen,
    # ODE：ds/dn = a·F̂·k^(p-1) - (1-λ)·s，k 为帧计数
    # 由 c·p·(1/fs)^p 换算得到每帧系数
    a_ode_frames={n: float(CAL[n]["c"] * CAL[n]["p"] * (1.0 / CAL[n]["fs"]) ** CAL[n]["p"])
                  for n in DATASETS},
)
P("")
P("冻结参数（用于因果在线算法）：")
P(f"  p = {p_frozen:.4f}   c = {c_frozen:.5f} (秒制)")
P(f"  换算成每帧 ODE 系数 a（各数据集 fs 略有差异）:")
for n in DATASETS:
    P(f"    {n}: fs={CAL[n]['fs']:.3f}Hz → a = {CAL['frozen']['a_ode_frames'][n]:.4e} /frame^{p_frozen:.3f}")

dump_json(CAL, "C1_shape_calibration.json")
with open(os.path.join(RES, "C1_shape_calibration.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\n[ok] C1 完成")
