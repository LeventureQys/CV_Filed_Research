# -*- coding: utf-8 -*-
"""变化负载专项：加载/卸载/中途切换的「阶跃保真」与「快相免责期」是否互相冲突。

数据：temp/变化负载/ 两组（8×5/21ch，ADC 域，帧到达成批突发）。
被检算法：raw / v3（现有）/ v4_fast5（快相免责 5s，上一步的改造）
         / v4_dyn3（本轮提出的动态免责期：按阶梯完成检测收尾，上限 3s）

评估口径（每次阶跃用「事件前 2s 稳健线性外推」与「事件后 2s 稳健水平」估计两侧电平）：
  delay     检测延迟（s）：在 delay 处的显示跳变最接近原始跳变
  gain      delay 处增益 = Δ显示/Δ原始（1.0 = 阶跃幅度完全保留）
  excess    |Δ显示−Δ原始| 在事件窗内的最大值（ADC）
  rec95     显示进入「原始 ±10%×本次跳变」所需时间（s）
  rec99     同上 ±2%
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3  # noqa: E402
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4

CASES = [("零负载-切换负载-零负载-再切换负载", "数据A"),
         ("零负载-中途切换负载-零负载-切换负载", "数据B")]


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def robust_level(y, i0, i1):
    """区间稳健水平：中位数"""
    return float(np.median(y[i0:i1]))


def robust_slope_level(y, dt, i_at, back=2.0):
    """用 [i_at-back, i_at] 做稳健线性拟合，外推到 i_at"""
    n = max(3, int(back / dt))
    a = max(0, i_at - n)
    x = np.arange(i_at - a) * dt
    yy = y[a:i_at]
    if len(x) < 3:
        return float(y[i_at])
    k, b = np.polyfit(x, yy, 1)
    return float(k * x[-1] + b)


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


ALGOS = [("raw", None), ("v3", (GLM53v3, {})),
         ("v4_fast5", (GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)))]

rows = []
print("=" * 110)
print("变化负载专项：阶跃保真 vs 快相免责期  ·  temp/v4.1flash/scripts/y2_varying_steps.py")
print("=" * 110)

for name, tag in CASES:
    t, X = load_rec(os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    # 事件检测：总量的一阶差分，取显著跳变（>8% 当前电平 且 >10% 最大电平）
    sm = pd.Series(tot).rolling(max(3, int(0.2 / dt)), center=True, min_periods=1).median().to_numpy()
    d = np.diff(sm)
    thr = np.maximum(0.08 * np.abs(sm[:-1]), 0.10 * tot.max())
    cand = np.where(np.abs(d) > thr)[0]
    # 合并相邻候选
    events = []
    for c in cand:
        if events and c - events[-1][-1] <= int(0.5 / dt):
            events[-1].append(c)
        else:
            events.append([c])
    ev = [int(np.mean(e)) for e in events if len(e) >= 1]
    # 保留相互间隔 > 3s 的
    keep = []
    for e in ev:
        if not keep or e - keep[-1] > int(3.0 / dt):
            keep.append(e)
    ev = keep
    print(f"\n=== {tag}（{name}）  时长 {span:.1f}s  总量 min={tot.min():.0f} max={tot.max():.0f} ===")
    print(f"检出显著跳变事件 {len(ev)} 个，时刻: " + ", ".join(f"{tu[e]:.2f}s" for e in ev))

    Ys = {}
    for lab, cfg in ALGOS:
        Ys[lab] = Xu.copy() if cfg is None else run(tu, Xu, cfg[0], **cfg[1])

    # 用总量判断每个算法自己对阶跃的「判据参考」——这里用最重的通道作图便于观察
    amp = np.zeros(Xu.shape[1])
    for a, b in [(int(tu.searchsorted(tu[0])), len(tu) - 1)]:
        pass
    # 主通道：总量最大区间的通道
    top = np.argsort(tot)[-int(5 / dt):]
    ch = int(np.argmax(Xu[top].mean(axis=0) - Xu[:int(5 / dt)].mean(axis=0)))
    print(f"观察通道 ch{ch}（顶载区间均值最大者）")

    for e in ev:
        j0 = max(0, e - int(3.0 / dt))
        j1 = min(len(tu), e + int(15.0 / dt))
        lvl_pre = robust_slope_level(tot, dt, e, 2.0)
        lvl_post = robust_level(tot, min(len(tot) - 1, e + int(1.0 / dt)),
                                min(len(tot), e + int(3.0 / dt)))
        raw_jump_tot = lvl_post - lvl_pre
        print(f"\n  --- 事件 @ {tu[e]:.2f}s（总量 {lvl_pre:.0f} → {lvl_post:.0f}，Δ={raw_jump_tot:+.0f}）---")
        print(f"    {'算法':<10}{'delay(s)':>9}{'gain':>8}{'excess':>9}{'rec95(s)':>9}{'rec99(s)':>9}")
        for lab, _ in ALGOS:
            Y = Ys[lab]
            pre = robust_slope_level(Y[:, ch], dt, e, 2.0)
            # 在 e+delay 处的显示水平
            best = None
            for k in range(0, int(2.5 / dt)):
                j = e + k
                if j + int(1.0 / dt) >= len(tu):
                    break
                post = robust_level(Y[:, ch], j, j + int(1.0 / dt))
                xpre = robust_slope_level(Xu[:, ch], dt, e, 2.0)
                xpost = robust_level(Xu[:, ch], j, j + int(1.0 / dt))
                g = (post - pre) / (xpost - xpre) if abs(xpost - xpre) > 1e-9 else np.nan
                err = abs(g - 1.0) if np.isfinite(g) else 9e9
                if best is None or err < best[0]:
                    best = (err, k * dt, g, post)
            _, delay, gain, post = best
            # excess：事件窗内单帧 |ΔY−ΔX|
            seg = slice(max(0, e - int(2.0 / dt)), min(len(tu) - 1, e + int(6.0 / dt)))
            excess = float(np.abs(np.diff(Y[:, ch]) - np.diff(Xu[:, ch]))[seg].max())
            # 恢复时间：显示进入 原始 ±10% / ±2% 本次跳变
            rec = {}
            for tol in (0.10, 0.02):
                thrv = tol * abs(lvl_post) if abs(lvl_post) > 1 else tol
                k = e
                while k < min(len(tu), e + int(20.0 / dt)):
                    dev = abs((Y[k, ch] - Xu[k, ch]))
                    if dev > thrv:
                        k += 1
                    else:
                        break
                rec[tol] = (k - e) * dt
            rows.append(dict(dataset=tag, event_s=float(tu[e]), algo=lab,
                             raw_jump_tot=float(raw_jump_tot), delay_s=delay, gain=gain,
                             excess=excess, rec95=rec[0.10], rec99=rec[0.02]))
            print(f"    {lab:<10}{delay:9.2f}{gain:8.3f}{excess:9.1f}{rec[0.10]:9.2f}{rec[0.02]:9.2f}")

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "varying_steps.csv"), index=False, encoding="utf-8-sig")
print("\n" + "=" * 110)
print("汇总（按算法；delay/gain 越接近 0/1 越好，excess/rec 越小越好）")
print("=" * 110)
agg = dfm.groupby("algo").agg(
    事件数=("event_s", "count"),
    检测延迟_s=("delay_s", "mean"),
    阶跃增益=("gain", "mean"),
    跳变超额_ADC=("excess", "mean"),
    恢复95_s=("rec95", "mean"),
    恢复99_s=("rec99", "mean"),
).round(3)
print(agg.to_string())
print("\nsaved: results/varying_steps.csv")
