# -*- coding: utf-8 -*-
"""量化「加载瞬间的快速爬升」与「随后的缓慢蠕变」两段结构。

对每组数据的主通道（以及受载通道中位）做：
  1. 检测加载沿 n_on；
  2. 计算 0.2s 平滑后的相对斜率 dX/dt；
  3. 找出斜率衰减到「慢相平均斜率」某倍数以下的时刻，即快/慢相分界；
  4. 分别统计两段的幅度增量与时长。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
LINES = []


def say(s=""):
    LINES.append(str(s))
    print(s)


def load(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def seg(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]


say("=" * 108)
say("加载后「快速爬升」与「缓慢蠕变」两段结构的量化  ·  temp/v4.1flash/scripts/q_two_phase.py")
say("=" * 108)

rows = []
for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"]:
    for name in ["数据1", "数据2", "数据3"]:
        t, X = load(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
        s0, s1 = seg(X.sum(axis=1))
        tot = X.sum(axis=1)
        base = tot[:s0].mean()
        thr = base + 0.05 * (tot[s0:s1].max() - base)
        n_on = next(i for i in range(s0, s0 + 300) if tot[i] > thr)
        amp = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
        m = int(np.argmax(amp))
        b = X[:s0, m].mean()
        # 在 uniform 网格上做平滑与斜率
        span = t[-1] - t[0]
        dt = span / (len(t) - 1)
        tu = np.arange(0.0, span, dt)
        y = np.interp(tu, t, X[:, m]) - b
        i0 = int(np.searchsorted(tu, t[n_on]))
        # 受载通道的稳健幅度（中位），减少单通道噪声
        loaded = amp > 0.3 * amp.max()
        ymed = np.median(np.vstack([np.interp(tu, t, X[:, c]) - X[:s0, c].mean()
                                    for c in np.where(loaded)[0]]), axis=0)
        # 指数平滑 τ=0.2s
        a = dt / 0.2
        ys = np.empty_like(y)
        acc = y[i0]
        for i in range(len(y)):
            acc = acc + a * (y[i] - acc)
            ys[i] = acc
        slope = np.gradient(ys, dt)
        # 慢相斜率：取加载后 0.5~0.8 段的中位斜率
        nL = s1 - s0
        slow_region = slice(i0 + int(0.55 * (nL * dt) / dt), i0 + int(0.85 * (nL * dt) / dt))
        k_slow = np.median(slope[slow_region])
        # 快相结束：斜率首次降到 2×k_slow 以下（且此后再不超）
        kthr = 2.0 * k_slow
        cand = np.where(slope[i0:i0 + int(20 / dt)] < kthr)[0]
        t_split = np.nan
        for c in cand:
            if np.all(slope[i0 + c:i0 + c + int(1.0 / dt)] < kthr):
                t_split = c * dt
                break
        y_fast = ys[i0 + int(t_split / dt)] - y[i0] if np.isfinite(t_split) else np.nan
        y_slow = y[s1 - 1] - (ys[i0 + int(t_split / dt)] if np.isfinite(t_split) else y[i0])
        rows.append(dict(location=loc, dataset=name, main_ch=m, dt_ms=1000 * dt,
                         fast_dur_s=t_split, y_onset=y[i0], y_fast_end=ys[i0 + int(t_split / dt)] if np.isfinite(t_split) else np.nan,
                         fast_amp=y_fast, slow_amp=y_slow, total_amp=y[s1 - 1],
                         fast_share_pct=100 * y_fast / (y_fast + y_slow) if np.isfinite(t_split) else np.nan,
                         slow_rate_per_s=k_slow))
        say(f"\n[{loc}/{name}] ch{m}  帧间隔={1000*dt:.1f}ms  加载沿 t={t[n_on]:.2f}s")
        say(f"   加载瞬时读数 y(0⁺) = {y[i0]:+.4f}")
        say(f"   快相时长 ≈ {t_split:.2f}s，快相末读数 = {ys[i0+int(t_split/dt)]:+.4f}（快相增量 {y_fast:+.4f}）"
            if np.isfinite(t_split) else "   快相分界未检出")
        say(f"   慢相增量 = {y_slow:+.4f}（到负载段结束），慢相平均斜率 = {k_slow*60:+.4f}/min")
        say(f"   快相占总增量 = {100*y_fast/(y_fast+y_slow):.1f}%")

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "two_phase_structure.csv"), index=False, encoding="utf-8-sig")
say("\n" + "=" * 108)
say("汇总")
say("=" * 108)
say(f"快相时长：中位 {dfm.fast_dur_s.median():.2f}s   范围 {dfm.fast_dur_s.min():.2f}~{dfm.fast_dur_s.max():.2f}s")
say(f"快相占总增量：中位 {dfm.fast_share_pct.median():.1f}%   范围 {dfm.fast_share_pct.min():.1f}~{dfm.fast_share_pct.max():.1f}%")
say(f"慢相斜率：中位 {dfm.slow_rate_per_s.median()*60:+.4f} 读数/min")
say("")
say("按位置分组：")
say(dfm.groupby("location").agg(快相时长_中位=("fast_dur_s", "median"),
                                快相占比_中位=("fast_share_pct", "median"),
                                慢相斜率每分钟=("slow_rate_per_s", lambda s: s.median() * 60)).round(3).to_string())

with open(os.path.join(RES, "two_phase_structure.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print("\nsaved: results/two_phase_structure.txt / two_phase_structure.csv")
