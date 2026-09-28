# -*- coding: utf-8 -*-
"""两个关键问题：

Q1 快相形状可复现吗？（决定它是「必须放过的不可补偿量」还是「可标定扣除的确定性量」）
    对同一位置的 3 次录制，取每组的受载通道幅度归一化快相曲线，比较组间一致性。
    归一化口径：amp_ref = 平滑原始在 onset+5s 的值 − 空载基线。

Q2 抖动来源：display 的量化刻度是多少？（决定能不能做亚刻度补偿）
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
LINES = []


def say(s=""):
    LINES.append(str(s))
    print(s)


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def smooth(x, dt, tau=2.0):
    a = dt / tau
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


LOCS = ["右拇指指尖", "左拇指指尖", "四指指尖"]
say("=" * 100)
say("Q1 快相形状可复现性  ·  temp/v4.1flash/scripts/w2_repeat.py")
say("=" * 100)
say("口径：amp_ref = 平滑原始(onset+5s) − 空载基线；f(s) = [平滑原始(onset+s) − 基线] / amp_ref")
say("")
profiles = {}
grid = [0.15, 0.3, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
say(f"{'位置':<12}{'数据':<8}" + "".join(f"{g:>7}s" for g in grid))
say("-" * 100)
for loc in LOCS:
    for name in ["数据1", "数据2", "数据3"]:
        t, X = load_rec(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
        span = t[-1] - t[0]
        dt = span / (len(t) - 1)
        tu = np.arange(0.0, span, dt)
        Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
        tot = Xu.sum(axis=1)
        thr = 0.15 * tot.max()
        ld = tot > thr
        d = np.diff(ld.astype(int))
        s = np.where(d == 1)[0] + 1
        e = np.where(d == -1)[0] + 1
        if ld[0]:
            s = np.r_[0, s]
        if ld[-1]:
            e = np.r_[e, len(ld)]
        s0, s1 = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]
        amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
        m = int(np.argmax(amp))
        base = tot[:s0].mean()
        thr2 = base + 0.05 * (tot[s0:s1].max() - base)
        n_on = next((i for i in range(s0, s0 + 300) if tot[i] > thr2), s0)
        Xs = smooth(Xu[:, m], dt, 2.0)
        b = Xs[:s0].mean()
        a5 = Xs[n_on + int(round(5.0 / dt))] - b
        prof = []
        for g in grid:
            k = n_on + int(round(g / dt))
            prof.append((Xs[k] - b) / a5 if a5 > 1e-9 else np.nan)
        profiles[(loc, name)] = np.array(prof)
        say(f"{loc:<12}{name:<8}" + "".join(f"{v:7.3f}" for v in prof))

say("")
say("组间一致性（同一位置 3 次录制的逐点标准差，越小越可复现）：")
say(f"{'位置':<14}" + "".join(f"{g:>8}s" for g in grid) + f"{'最大标准差':>11}")
say("-" * 100)
for loc in LOCS:
    P = np.vstack([profiles[(loc, n)] for n in ["数据1", "数据2", "数据3"]])
    sd = P.std(axis=0)
    say(f"{loc:<14}" + "".join(f"{v:8.3f}" for v in sd) + f"{sd.max():11.3f}")
say("")
allP = np.vstack(list(profiles.values()))
say(f"全部 9 组逐点标准差：最大 {allP.std(axis=0).max():.3f}，均值 {allP.std(axis=0).mean():.3f}")
say(f"快相 5s 前形状（f(1), f(2), f(3)）的 9 组中位：{np.median(allP[:, grid.index(1.0)]):.3f}, "
    f"{np.median(allP[:, grid.index(2.0)]):.3f}, {np.median(allP[:, grid.index(3.0)]):.3f}")

say("")
say("=" * 100)
say("Q2 显示量化刻度（决定补偿能否做到亚刻度精度）")
say("=" * 100)
for loc in LOCS:
    t, X = load_rec(os.path.join(TEMP, loc, "数据1", "device_001_seg000.csv"))
    allvals = np.unique(np.round(X[:3000].ravel(), 6))
    d2 = np.diff(allvals)
    d2 = d2[d2 > 1e-9]
    step = np.min(d2) if len(d2) else np.nan
    say(f"{loc}: 前 3000 帧全通道取值 {len(allvals)} 个，最小间隔 = {step:.6f}"
        f"（≈ 1/{1/step:.0f}，即约 {abs(np.log10(step)):.1f} 位小数）")
    # 主通道在空载段的取值
    tot = X.sum(axis=1)
    thr = 0.15 * tot.max()
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    s0, s1 = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]
    amp = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    v = np.unique(np.round(X[:s0, m], 6))
    dv = np.diff(v)
    dv = dv[dv > 1e-9]
    say(f"    主通道 ch{m} 空载段取值 {len(v)} 个，相邻间隔 = "
        f"{np.array2string(np.unique(np.round(dv,6))[:6], precision=6)}")

with open(os.path.join(RES, "fastphase_repeatability.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print("\nsaved: results/fastphase_repeatability.txt")
