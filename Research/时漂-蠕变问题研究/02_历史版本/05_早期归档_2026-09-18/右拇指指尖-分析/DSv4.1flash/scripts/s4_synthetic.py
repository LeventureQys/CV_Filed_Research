# -*- coding: utf-8 -*-
"""步骤4：合成真值基准（受控实验）。

真实数据没有"真值"（不知道真实力到底是多少），只能做"自洽性"评估。
这里用从真实数据刻画出的参数构造合成数据，注入**已知真值** F_true，
然后精确测量各算法恢复到 F_true 的能力。

合成模型（对照真实数据参数）
---------------------------
y_j(t) = F_j · s(t)                     瞬时阶跃（s 为上升过程）
       + F_j · α1 · (1-exp(-t/τ1))      快蠕变  τ1 ≈ 1.5s，α1 ≈ 0.25
       + F_j · α2 · (1-exp(-t/τ2))      慢蠕变  τ2 ≈ 70s， α2 ≈ 0.45
       + z_j(t)                          零漂：卸载后缓慢回零 + 加载前漂移
       + n_j(t)                          白噪 + 1mN 量化
其中 α1, α2 从真实数据双指数拟合结果标定。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import dump_json, RES, load_dataset, detect_segments, DATASETS
import tac_algorithms as AL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

lines = []


def P(s=""):
    print(s)
    lines.append(s)


# ---------------- 从真实数据标定蠕变参数 ----------------
def calibrate(name):
    D = load_dataset(name)
    fi = D["frame_index"]
    k, b = np.polyfit(fi, D["t"], 1)
    t = fi * k
    fs = 1.0 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), D["t"], fs=fs)
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    base = X[a:bb].mean(axis=0)
    Xn = X - base
    j = int(np.argmax(Xn[c0:d0].max(axis=0)))
    y = Xn[c0:d0, j]
    tt = t[c0:d0] - t[c0]
    F = y[c0:c0 + int(0.15 * fs)].mean()
    # 双指数拟合 (1-exp) 形态
    from scipy.optimize import curve_fit

    def f(t, a1, t1, a2, t2, c):
        return a1 * (1 - np.exp(-t / t1)) + a2 * (1 - np.exp(-t / t2)) + c

    p0 = [0.2 * F, 1.5, 0.4 * F, 70.0, F]
    try:
        popt, _ = curve_fit(f, tt, y, p0=p0, maxfev=200000,
                            bounds=([-np.inf, 0.2, -np.inf, 5, -np.inf],
                                    [np.inf, 1e4, np.inf, 1e4, np.inf]))
    except Exception:
        popt = None
    # 零漂：后空载段
    zp = Xn[d0:, j]
    zt = t[d0:] - t[d0]
    # 后空载起始偏移 + 回零时间常数
    z0 = zp[:int(1 * fs)].mean()
    def fz(t, a, tau, c):
        return a * np.exp(-t / tau) + c
    try:
        pz, _ = curve_fit(fz, zt, zp, p0=[z0, 20.0, 0.0], maxfev=100000,
                          bounds=([-np.inf, 0.5, -np.inf], [np.inf, 1e5, np.inf]))
    except Exception:
        pz = [z0, 20.0, 0.0]
    # 噪声
    dt_ = np.diff(x_pre := Xn[a:bb, j])
    noise = dt_.std() / np.sqrt(2)
    return dict(popt=popt, pz=pz, F=F, noise=noise, fs=fs, j=j,
                dur_load=tt[-1], dur_pre=t[bb] - t[0], dur_post=t[-1] - t[d0],
                X=Xn, t=t, c0=c0, d0=d0, a=a, bb=bb)


CAL = {n: calibrate(n) for n in DATASETS}
P("从真实数据标定的蠕变/零漂参数：")
for n, c in CAL.items():
    if c["popt"] is not None:
        a1, t1, a2, t2, cc = c["popt"]
        P(f"  {n}: F={c['F']:.4f}N  快蠕变 a1={a1:.4f}N τ1={t1:.2f}s ({100*a1/c['F']:.1f}%F)  "
          f"慢蠕变 a2={a2:.4f}N τ2={t2:.1f}s ({100*a2/c['F']:.1f}%F)")
    P(f"        后空载零漂: z0={c['pz'][0]*1000:.2f}mN τz={c['pz'][1]:.1f}s "
      f"残留={c['pz'][2]*1000:.2f}mN   噪声σ={c['noise']*1000:.3f}mN")

# ---------------- 构造合成阵列 ----------------
RNG = np.random.default_rng(20260916)
C = 31
ROWS, COLS = 9, 7
MASK = "110110011011101111110111111011111100000001000000100000010000001"
ON = [i for i, b in enumerate(MASK) if b == "1"]


def make_synthetic(ref, load_dur=120.0, pre_dur=10.0, post_dur=60.0,
                   fs=100.51, n_active_ratio=0.8, noise_mn=2.0,
                   quant_mn=1.0, seed=1):
    """生成合成数据：返回 t, Y_raw(含漂移), F_true(每通道真值), mask_active"""
    rng = np.random.default_rng(seed)
    a1, t1, a2, t2, _ = ref["popt"]
    F = ref["F"]
    n_pre = int(pre_dur * fs); n_load = int(load_dur * fs); n_post = int(post_dur * fs)
    n = n_pre + n_load + n_post
    t = np.arange(n) / fs
    t_load = np.clip(t - pre_dur, 0, None)
    t_post = np.clip(t - (pre_dur + load_dur), 0, None)
    # 真值力（恒定负载）
    true_step = np.where((t >= pre_dur) & (t < pre_dur + load_dur), 1.0, 0.0)
    # 阵列表面的载荷分布：中心高、边缘低（模拟指腹压在阵列上）
    fmap = np.zeros((ROWS, COLS))
    cy, cx = 4.0, 3.0
    for i in range(ROWS):
        for j in range(COLS):
            fmap[i, j] = np.exp(-(((i - cy) / 2.6) ** 2 + ((j - cx) / 2.0) ** 2))
    amp = np.zeros(C)
    for k, pos in enumerate(ON):
        amp[k] = fmap.ravel()[pos]
    active = amp > 0.12
    amp = amp / amp.max() * F * 1.05
    # 每通道蠕变比例（个体差异 ±20%）
    r1 = 1.0 + 0.2 * rng.standard_normal(C)
    r2 = 1.0 + 0.2 * rng.standard_normal(C)
    rise_tau = 0.05
    s = 1 - np.exp(-np.clip(t - pre_dur, 0, None) / rise_tau)
    s = np.where(t < pre_dur, 0.0, s)
    Y = np.zeros((n, C))
    for k in range(C):
        if not active[k]:
            continue
        c1 = a1 / F * np.clip(r1[k], 0.5, 1.5)
        c2 = a2 / F * np.clip(r2[k], 0.5, 1.5)
        creep = (c1 * (1 - np.exp(-t_load / t1)) + c2 * (1 - np.exp(-t_load / t2)))
        creep = creep * amp[k] * true_step
        # 卸载后的零漂：与积累的蠕变同源，按比例残留后指数回零
        z0 = 0.06 * amp[k] * (rng.standard_normal())
        zrec = z0 * np.exp(-t_post / 20.0) * (t_post > 0)
        Y[:, k] = amp[k] * true_step + creep + zrec
    # 共模干扰（温漂/供电漂移）：所有通道同向，幅度与响应弱相关
    common = 0.02 * F * (1 - np.exp(-t_load / 90.0)) * true_step
    Y += common[:, None]
    # 噪声 + 量化（1 mN）
    Yn = Y + noise_mn / 1000.0 * rng.standard_normal((n, C))
    Yq = np.round(Yn / (quant_mn / 1000.0)) * (quant_mn / 1000.0)
    F_true = np.where(active, amp, 0.0) * true_step[:, None]
    return dict(t=t, Y=Yq, F_true=F_true, active=active, pre=(0, n_pre),
                load=(n_pre, n_pre + n_load), post=(n_pre + n_load, n),
                fs=fs, F=F, ref=ref)


SYN = {}
for name, ref in CAL.items():
    if ref["popt"] is None:
        continue
    syn = make_synthetic(ref, load_dur=120.0, pre_dur=10.0, post_dur=60.0,
                         seed=hash(name) % 1000)
    SYN[name] = syn
    P(f"  合成[{name}]: 受载通道 {int(syn['active'].sum())} 个, "
      f"真值峰值={syn['F_true'][syn['load'][0]:syn['load'][1]].max():.4f}N, "
      f"末段含漂移峰值={syn['Y'][syn['load'][0]:syn['load'][1]].max():.4f}N")


def eval_synthetic(syn, cases):
    X = syn["Y"]; t = syn["t"]; fs = syn["fs"]
    a, bb = syn["pre"]; c0, d0 = syn["load"]; n = len(t)
    base = X[a:bb].mean(axis=0)
    Xn = X - base
    Ftrue = syn["F_true"]
    act = np.where(syn["active"])[0]
    w = lambda s: int(round(s * fs))
    rows = []
    for c in cases:
        try:
            Y = c.transform(Xn, t, fs, c0)
            ok = True
        except Exception as e:
            Y = Xn; ok = False; P(f"    !! {c.name}: {e}")
        # 指标：负载末 5s 的 RMSE vs 真值（只在受载通道）
        est = Y[max(d0 - w(5.0), c0):d0][:, act].mean(axis=0)
        tgt = Ftrue[max(d0 - w(5.0), c0):d0][:, act].mean(axis=0)
        rmse = float(np.sqrt(np.mean((est - tgt) ** 2)))
        rel = float(rmse / np.mean(tgt) * 100)
        bias = float(np.mean(est - tgt))
        # 阶跃保真：0~0.15s
        e0 = Y[c0:c0 + w(0.15)][:, act].mean()
        t0 = Ftrue[c0:c0 + w(0.15)][:, act].mean()
        fid = float(e0 / t0 * 100) if t0 else np.nan
        # 卸载后零漂误差（后空载末 3s）
        ze = float(Y[n - w(3.0):][:, act].mean())
        rows.append(dict(algo=c.name, group=getattr(c, "group", "?"),
                         rmse=rmse, rel=rel, bias=bias, fid=fid, zero_err=ze))
    return rows


CASES = [
    AL.Raw(),
    AL.MovingAverage(1.0), AL.MovingAverage(2.0), AL.MovingAverage(5.0), AL.MovingAverage(10.0),
    AL.MedianFilter(1.0), AL.MedianFilter(5.0),
    AL.EWMA(1.0), AL.EWMA(5.0), AL.EWMA(20.0),
    AL.ButterHighPass(0.005), AL.ButterHighPass(0.01), AL.ButterHighPass(0.02),
    AL.ButterHighPass(0.05), AL.ButterHighPass(0.1),
    AL.SavitzkyGolay(1.0), AL.SavitzkyGolay(3.0),
    AL.WaveletDenoise(),
    AL.ModelFitCompensator("exp1"), AL.ModelFitCompensator("exp2"),
    AL.ModelFitCompensator("power"),
    AL.RLSDetrend(0.9995), AL.RLSDetrend(0.9999), AL.RLSDetrend(0.99998),
    AL.KalmanDrift(1e-8, 1e-5), AL.KalmanDrift(1e-7, 1e-5), AL.KalmanDrift(1e-6, 1e-5),
    AL.ModelFitShapeShared(0.2, "exp1"), AL.ModelFitShapeShared(0.2, "exp2"),
    AL.CommonModeRemoval("mean", 5.0, 0.2, True),
    AL.CommonModeRemoval("mean", 20.0, 0.2, True),
    AL.PCASubspace(1), AL.PCASubspace(2), AL.PCASubspace(1, True),
    AL.CommonModeRefFit("mean", 1), AL.CommonModeRefFit("mean", 2),
    AL.AdaptiveNoiseCanceller(0.5, 64), AL.AdaptiveNoiseCanceller(1.0, 128),
]

P("\n合成真值基准结果（误差 = 负载末 5s 估计值与真值之差）：")
allrows = {}
for name, syn in SYN.items():
    rows = eval_synthetic(syn, CASES)
    allrows[name] = rows
    P(f"\n  ==== {name} ====")
    P(f"  {'算法':<34s} {'RMSE[N]':>9s} {'相对%':>7s} {'偏置[N]':>9s} {'阶跃保真%':>9s} {'卸载后零漂[mN]':>13s}")
    for r in sorted(rows, key=lambda x: x["rmse"]):
        P(f"  {r['algo']:<34s} {r['rmse']:>9.4f} {r['rel']:>7.2f} {r['bias']:>+9.4f} "
          f"{r['fid']:>9.1f} {r['zero_err']*1000:>13.3f}")

# 平均排名
from collections import defaultdict
agg = defaultdict(list)
for name, rows in allrows.items():
    for r in rows:
        agg[r["algo"]].append(r)
P("\n三组合成数据平均：")
P(f"  {'算法':<34s} {'RMSE[N]':>9s} {'相对%':>7s} {'阶跃保真%':>9s} {'卸载后零漂[mN]':>13s}")
final = []
for k, v in agg.items():
    rm = np.mean([x["rmse"] for x in v]); rl = np.mean([x["rel"] for x in v])
    fd = np.mean([x["fid"] for x in v]); ze = np.mean([x["zero_err"] for x in v])
    final.append(dict(algo=k, group=v[0]["group"], rmse=float(rm), rel=float(rl),
                      fid=float(fd), zero_err=float(ze)))
for r in sorted(final, key=lambda x: x["rmse"]):
    P(f"  {r['algo']:<34s} {r['rmse']:>9.4f} {r['rel']:>7.2f} {r['fid']:>9.1f} "
      f"{r['zero_err']*1000:>13.3f}")

dump_json(dict(per_dataset=allrows, avg=final), "D_synthetic.json")
with open(os.path.join(RES, "D_synthetic.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
P("\n[ok] 步骤4 完成")
