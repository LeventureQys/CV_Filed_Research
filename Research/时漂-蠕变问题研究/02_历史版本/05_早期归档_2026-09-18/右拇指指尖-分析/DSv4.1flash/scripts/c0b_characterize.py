# -*- coding: utf-8 -*-
"""步骤 C0b：数据特征刻画（修正分段后的最终版本）。

产出图：
  G1 总览（分段 + 在线检测点 + 主通道放大）
  G2 漂移形态与速率（含幂律/双时间尺度）
  G3 噪声与频谱（含 Allan/窗口 σ 判别 1/f 与白噪）
  G4 空间结构（响应图 / 漂移比 / 死通道 / 物理值分布）
  G5 零漂（卸载后恢复、前空载残留、量化死区）
"""
import os
import sys
import numpy as np
from scipy import signal as sps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import (load_dataset, detect_segments, spatial_map, savefig,
                        dump_json, DATASETS, FIG, RES, ROWS, COLS, MASK_ON)
from causal_harness import online_load_detect
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
np.set_printoptions(precision=4, suppress=True, linewidth=150)

lines = []


def P(s=""):
    print(s)
    lines.append(s)


def cs(y, w):
    if w <= 1:
        return y.copy()
    c = np.concatenate([[0.0], np.cumsum(y)])
    n = len(y); idx = np.arange(n); lo = np.maximum(0, idx - w + 1)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)


INFO = {}
for name in DATASETS:
    D = load_dataset(name)
    fi = D["frame_index"]
    k, _ = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1.0 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), t, fs=fs)
    a0, a1 = seg["pre"]; c0, d0 = seg["load"]; p0, p1 = seg["post"]
    w = lambda s: int(round(s * fs))
    base = X[a0:a1].mean(axis=0)
    Xn = X - base[None, :]
    tot = Xn.sum(axis=1)
    n_on, _, thr = online_load_detect(tot, fs, a1)
    resp = Xn[c0:d0].max(axis=0)
    act = np.where(resp > 0.05)[0]
    dead = np.where(X.std(axis=0) == 0)[0]
    REF = Xn[c0 + w(1.0):c0 + w(3.0)].mean(axis=0)
    late = Xn[d0 - w(5):d0].mean(axis=0)
    drift_rel = (late - REF) / np.where(np.abs(REF) < 1e-9, np.nan, REF)

    P("=" * 100)
    P(f"### {name}  fs={fs:.4f}Hz  N={len(t)}  时长={t[-1]:.2f}s")
    P(f"    分段: 前空载 0~{t[a1]:.2f}s({t[a1]:.1f}s) | "
      f"负载 {t[c0]:.2f}~{t[d0]:.2f}s({t[d0]-t[c0]:.1f}s) | "
      f"后空载 {t[p0]:.2f}~{t[p1-1]:.2f}s({t[p1-1]-t[p0]:.1f}s)")
    P(f"    在线检测: n_on={n_on} t={t[n_on]:.3f}s  离线={c0}  偏差={n_on-c0}帧="
      f"{(n_on-c0)/fs*1000:.0f}ms  阈值={thr:.3f}N")
    P(f"    通道: 总数31  受载 {len(act)}  近零 {31-len(act)}  恒零(死) {len(dead)}: "
      f"{[D['ch_cols'][i] for i in dead]}")
    P(f"    响应: max={resp[act].max()*1000:.1f}mN  中位={np.median(resp[act])*1000:.1f}mN  "
      f"总和峰值={Xn[c0:d0].sum(axis=1).max()*1000:.1f}mN")
    P(f"    时漂(末5s vs 1~3s): 中位={np.nanmedian(drift_rel[act])*100:+.2f}%  "
      f"范围={np.nanmin(drift_rel[act])*100:+.2f}%~{np.nanmax(drift_rel[act])*100:+.2f}%  "
      f"绝对量 中位={np.nanmedian((late-REF)[act])*1000:+.2f}mN")
    # 噪声
    nz_ch = []
    for j in act:
        y = Xn[d0 - w(20):d0, j]
        z = y - cs(y, w(2.0))       # 去掉 2s 以上的慢变
        nz_ch.append(z.std())
    nz_ch = np.array(nz_ch)
    P(f"    噪声(末20s 去2s慢变): 中位σ={np.median(nz_ch)*1000:.3f}mN  "
      f"最大σ={nz_ch.max()*1000:.3f}mN")
    P(f"    量化: 最小非零台阶={np.min(np.diff(np.unique(X))[np.diff(np.unique(X))>0])*1000:.3f}mN")
    # 零漂
    zpre = Xn[a1 - w(1):a1].mean(axis=0)[act]
    zpost = Xn[p1 - w(3):p1].mean(axis=0)[act]
    zjump = Xn[d0:d0 + w(0.2)].mean(axis=0)[act] - Xn[d0 - w(0.2):d0].mean(axis=0)[act]
    P(f"    零漂: 前空载末1s中位={np.median(zpre)*1000:+.3f}mN  "
      f"卸载瞬跳中位={np.median(zjump)*1000:+.3f}mN  "
      f"卸载后末3s中位={np.median(zpost)*1000:+.3f}mN  "
      f"|残余|/时漂={np.nanmedian(np.abs(zpost)/np.maximum(np.abs(late[act]-base[act]),1e-9))*100:.1f}%")

    INFO[name] = dict(D=D, t=t, fs=fs, X=X, Xn=Xn, seg=seg, base=base, tot=tot,
                      n_on=n_on, act=act, dead=dead, REF=REF, late=late,
                      drift_rel=drift_rel, c0=c0, d0=d0, p0=p0, p1=p1,
                      a0=a0, a1=a1, resp=resp, nz=nz_ch, thr=thr)
    dump_json({k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in INFO[name].items()
               if k in ("n_on", "c0", "d0", "p0", "p1", "a1", "act", "dead",
                        "drift_rel", "nz")}, f"C0b_{DATASETS.index(name)}.json")

# ============================================== G1 总览
fig, axes = plt.subplots(3, 3, figsize=(16.5, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    I = INFO[name]; D = I["D"]; t = I["t"]; fs = I["fs"]
    w = lambda s: int(round(s * fs))
    ax = axes[r, 0]
    ax.plot(t, I["tot"], lw=0.5, color="#1f77b4")
    ax.axvspan(t[I["c0"]], t[I["d0"]], color="orange", alpha=0.16)
    ax.axvline(t[I["n_on"]], color="g", lw=1.1)
    ax.set_title(f"{name} Σ受载通道（橙=负载段，绿=在线检测）")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Σ Force (N)"); ax.grid(alpha=0.3)

    ax = axes[r, 1]
    j = int(I["act"][np.argmax(I["REF"][I["act"]])])
    ax.plot(t, I["Xn"][:, j], lw=0.5, color="tab:red")
    ax.axhline(I["REF"][j], color="k", ls="--", lw=1.1,
               label=f"1~3s 水平={I['REF'][j]*1000:.0f}mN")
    ax.axhline(I["late"][j], color="tab:purple", ls=":", lw=1.1,
               label=f"末5s 水平={I['late'][j]*1000:.0f}mN (+{I['drift_rel'][j]*100:.0f}%)")
    ax.axvspan(t[I["c0"]], t[I["d0"]], color="orange", alpha=0.16)
    ax.set_title(f"{name} 主通道 {D['ch_cols'][j]}")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Force (N)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[r, 2]
    s0 = max(0, I["n_on"] - w(0.3)); s1 = I["n_on"] + w(3.0)
    ax.plot(t[s0:s1] - t[I["n_on"]], I["Xn"][s0:s1, j], lw=1.0, color="tab:red")
    ax.axvline(0, color="g", lw=1.1)
    ax.set_xlim(-0.3, 3.0)
    ax.set_title(f"{name} 加载后 3s 放大（看快过程）")
    ax.set_xlabel("相对加载 (s)"); ax.set_ylabel("Force (N)"); ax.grid(alpha=0.3)
fig.suptitle("图 G1  三组数据总览（修正分段）", fontsize=13)
savefig(fig, "G1_overview.png")

# ============================================== G2 形态与速率
fig, axes = plt.subplots(3, 3, figsize=(16.5, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    I = INFO[name]; D = I["D"]; t = I["t"]; fs = I["fs"]
    w = lambda s: int(round(s * fs))
    tau = t[I["c0"]:I["d0"]] - t[I["c0"]]
    ok = np.abs(I["REF"][I["act"]]) > 1e-6
    acc = I["act"][ok]
    wts = np.abs(I["REF"][acc]); wts = wts / wts.sum()
    hs = (I["Xn"][I["c0"]:I["d0"], acc] / I["REF"][acc][None, :]) @ wts
    hs = cs(hs, w(0.5))
    ax = axes[r, 0]
    ax.plot(tau, hs, lw=0.9, color="gray", label="实测形态 h(τ)")
    for tt, c in ((1, "tab:red"), (10, "tab:blue"), (30, "tab:green")):
        ax.axvline(tt, color=c, ls=":", lw=0.9)
    ax.set_title(f"{name} 归一化形态（以 1~3s 为 1.0）")
    ax.set_xlabel("负载持续 τ (s)"); ax.set_ylabel("h(τ)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[r, 1]
    ax.loglog(tau[1:], np.maximum(hs[1:] - 1, 1e-5) * 100, lw=0.9, color="tab:purple")
    m = (tau > 2) & (hs > 1.01)
    if m.sum() > 20:
        pp = np.polyfit(np.log(tau[m]), np.log((hs[m] - 1)), 1)
        ax.loglog(tau[m], np.exp(np.polyval(pp, np.log(tau[m]))) * 100, "k--", lw=1.2,
                  label=f"幂律 p={pp[0]:.3f}")
    ax.set_title(f"{name} 相对漂移(%) 双对数（幂律检验）")
    ax.set_xlabel("τ (s)"); ax.set_ylabel("(h-1)·100 (%)"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[r, 2]
    dr = I["drift_rel"][I["act"]] * 100
    ax.hist(dr, bins=16, color="tab:orange")
    ax.axvline(np.nanmedian(dr), color="r", ls="--",
               label=f"中位={np.nanmedian(dr):+.1f}%")
    ax.set_title(f"{name} 各通道时漂分布（末5s vs 1~3s）")
    ax.set_xlabel("相对漂移 (%)"); ax.set_ylabel("通道数"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("图 G2  漂移形态与速率", fontsize=13)
savefig(fig, "G2_shape.png")

# ============================================== G3 噪声与频谱
fig, axes = plt.subplots(3, 3, figsize=(16.5, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    I = INFO[name]; D = I["D"]; t = I["t"]; fs = I["fs"]
    w = lambda s: int(round(s * fs))
    act = I["act"]; Xn = I["Xn"]
    j = int(act[np.argmax(I["REF"][act])])
    ax = axes[r, 0]
    ax.bar(act, I["nz"] * 1000, color="tab:cyan")
    ax.set_xticks(range(0, 31, 5))
    ax.set_title(f"{name} 各受载通道噪声 σ（末20s，去2s慢变）")
    ax.set_xlabel("通道序号"); ax.set_ylabel("σ (mN)"); ax.grid(alpha=0.3, axis="y")

    ax = axes[r, 1]
    segs = {"前空载": Xn[I["a0"]:I["a1"], j], "负载初段": Xn[I["n_on"]:I["n_on"] + w(10), j],
            "负载末段": Xn[I["d0"] - w(10):I["d0"], j]}
    for lab, s in segs.items():
        f, Pxx = sps.welch(s - s.mean(), fs=fs, nperseg=min(2048, len(s)))
        ax.loglog(f[1:], Pxx[1:], lw=0.8, label=lab)
    ax.set_title(f"{name} 主通道 PSD（{D['ch_cols'][j]}）")
    ax.set_xlabel("Hz"); ax.set_ylabel("N²/Hz"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[r, 2]
    taus = np.logspace(np.log10(0.02), np.log10(20), 22)
    ws = np.maximum((taus * fs).astype(int), 2)
    y = Xn[I["d0"] - w(60):I["d0"], j]
    sig = [cs(y, kk)[len(y) // 2:].std() for kk in ws]
    ax.loglog(taus, np.array(sig) * 1000, "o-", ms=3, lw=0.9)
    ax.set_title(f"{name} 窗口 σ(τ)（斜率≈-0.5 白噪，≈0 直流漂移）")
    ax.set_xlabel("窗口长度 τ (s)"); ax.set_ylabel("σ (mN)"); ax.grid(alpha=0.3, which="both")
fig.suptitle("图 G3  噪声与频谱特征", fontsize=13)
savefig(fig, "G3_noise.png")

# ============================================== G4 空间结构
fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    I = INFO[name]; D = I["D"]; act = I["act"]
    nz_map = np.full(31, np.nan); nz_map[act] = I["nz"] * 1000
    dr_map = np.array(I["drift_rel"], dtype=float) * 100
    maps = [
        (spatial_map(I["resp"]), "负载响应 (N)", "hot"),
        (spatial_map(dr_map), "时漂 (%)", "coolwarm"),
        (spatial_map(nz_map), "噪声 σ (mN)", "viridis"),
        (spatial_map(np.abs(I["REF"])), "|参考水平| (N)", "magma"),
    ]
    for cidx, (mat, title, cmap) in enumerate(maps):
        ax = axes[r, cidx]
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(f"{name} {title}", fontsize=9)
        ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("图 G4  空间结构（9×7 阵列；空白=无效位，const-0=死通道）", fontsize=13)
savefig(fig, "G4_spatial.png")

# ============================================== G5 零漂
fig, axes = plt.subplots(3, 3, figsize=(16.5, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    I = INFO[name]; D = I["D"]; t = I["t"]; fs = I["fs"]
    w = lambda s: int(round(s * fs))
    act = I["act"]; Xn = I["Xn"]
    ax = axes[r, 0]
    for j in act[:8]:
        ax.plot(t[I["d0"]:] - t[I["d0"]], Xn[I["d0"]:, j] * 1000, lw=0.8,
                label=D["ch_cols"][j])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{name} 卸载后零漂恢复（前8个受载通道）")
    ax.set_xlabel("卸载后时间 (s)"); ax.set_ylabel("残余 (mN)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

    ax = axes[r, 1]
    ax.plot(t[I["a0"]:I["a1"]] - t[I["a0"]], Xn[I["a0"]:I["a1"]].mean(axis=1) * 1000,
            lw=0.7, label="前空载 全通道均值")
    ax.plot(t[I["p0"]:I["p1"]] - t[I["p0"]], Xn[I["p0"]:I["p1"]].mean(axis=1) * 1000,
            lw=0.7, label="后空载 全通道均值")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{name} 空载基线（毫牛级）")
    ax.set_xlabel("段内时间 (s)"); ax.set_ylabel("均值 (mN)"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[r, 2]
    z = Xn[I["d0"]:I["d0"] + w(30), int(act[np.argmax(I["REF"][act])])]
    ax.plot(t[I["d0"]:I["d0"] + w(30)] - t[I["d0"]], z * 1000, lw=1.0, color="tab:red",
            label="主通道残余（原始）")
    ax.plot(t[I["d0"]:I["d0"] + w(30)] - t[I["d0"]], z * 1000 - cs(z, w(0.5)) * 1000,
            lw=0.7, color="gray", label="去0.5s慢变后")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{name} 卸载瞬态细节（0~30s）")
    ax.set_xlabel("卸载后时间 (s)"); ax.set_ylabel("残余 (mN)"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle("图 G5  零漂（卸载后恢复 / 空载基线 / 量化死区）", fontsize=13)
savefig(fig, "G5_zero.png")

with open(os.path.join(RES, "C0b_characterization.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[ok] C0b 完成")
