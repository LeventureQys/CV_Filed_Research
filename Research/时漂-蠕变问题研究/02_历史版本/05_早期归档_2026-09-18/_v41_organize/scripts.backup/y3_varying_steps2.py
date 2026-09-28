# -*- coding: utf-8 -*-
"""变化负载专项 v2：把「中途变载」（负载内加重/减轻）也检出来，并用整阵指标评估。

事件检测（不依赖单帧斜率，故对 1~2s 的缓变台阶也有效）：
  在平滑总量上逐点扫描，比较「前 2s 稳健水平」与「后 1~3s 稳健水平」，
  相对变化 > 15% 且绝对变化 > 8% 满幅者视为变载事件；再按 1.5s 间隔合并去重。

指标（整阵 total，与 GLM53 既有口径一致）：
  Δraw / Δdisp     事件处原始/显示总量跳变
  excess           |Δdisp−Δraw| 在事件窗内单帧最大值（ADC，整阵）
  lag              显示跳变达到原始 90% 的时刻 − 事件时刻（s）
  rec95            显示回到「与原始同电平 ±10%」所需时间（s）
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


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def detect_events(tot, dt, rel=0.15, absfrac=0.08):
    """返回 [(idx, dlevel), ...]"""
    n = len(tot)
    pre_n, post_n = int(2.0 / dt), int(2.0 / dt)
    cand = []
    step = max(1, int(0.1 / dt))
    for i in range(pre_n, n - max(post_n, int(3.0 / dt)), step):
        pre = np.median(tot[i - pre_n:i])
        post = np.median(tot[i:i + post_n])
        dl = post - pre
        if abs(dl) > max(rel * abs(pre), absfrac * tot.max()):
            cand.append((i, dl))
    # 合并 1.5s 内的候选，取 |dlevel| 最大者
    events = []
    for i, dl in cand:
        if events and i - events[-1][0] <= int(1.5 / dt):
            if abs(dl) > abs(events[-1][1]):
                events[-1] = (i, dl)
        else:
            events.append((i, dl))
    # 用「两电平中点」精修事件时刻：找跳变最快的点
    refined = []
    for i, dl in events:
        a = max(0, i - int(2.0 / dt))
        b = min(n - 1, i + int(2.0 / dt))
        sm = pd.Series(tot[a:b]).rolling(max(3, int(0.15 / dt)), min_periods=1).median().to_numpy()
        k = int(np.argmax(np.abs(np.diff(sm))))
        refined.append((a + k, dl))
    return refined


rows = []
print("=" * 112)
print("变化负载专项 v2（整阵口径）  ·  temp/v4.1flash/scripts/y3_varying_steps2.py")
print("=" * 112)

for name, tag in CASES:
    t, X = load_rec(os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    ev = detect_events(tot, dt)
    print(f"\n=== {tag}（{name}）  时长 {span:.1f}s  满幅 {tot.max():.0f} ===")
    print(f"检出变载事件 {len(ev)} 个: " + ", ".join(f"{tu[i]:.2f}s(Δ{dl:+.0f})" for i, dl in ev))

    Yv3 = run(tu, Xu, GLM53v3)
    Yv4 = run(tu, Xu, GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)
    Ys = {"raw": Xu, "v3": Yv3, "v4_fast5": Yv4}

    print(f"\n  {'事件':>8}{'原始跳变':>10}| " + "".join(
        f"{lab+'延迟':>10}{lab+'增益':>9}{lab+'超额':>9}" for lab in ["v3", "v4_fast5"]))
    for e, dl in ev:
        xp = np.median(tot[max(0, e - int(2 / dt)):e])
        xq = np.median(tot[e:e + int(2 / dt)])
        raw_jump = xq - xp
        line = f"  {tu[e]:8.2f}{raw_jump:10.0f}| "
        for lab in ["v3", "v4_fast5"]:
            Y = Ys[lab]
            # 显示跳变：取事件后「显示相对原始不再变化」的稳定差
            # 用 e+2s..e+3s 的 (disp-raw) 作为事件后残余，e-2..e-0.3s 作为事件前残余
            pre_off = np.median((Y[max(0, e - int(2 / dt)):e - int(0.3 / dt)].sum(axis=1)
                                 - tot[max(0, e - int(2 / dt)):e - int(0.3 / dt)]))
            post_off = np.median((Y[e + int(2 / dt):e + int(3 / dt)].sum(axis=1)
                                  - tot[e + int(2 / dt):e + int(3 / dt)]))
            disp_jump = raw_jump + (post_off - pre_off)
            gain = disp_jump / raw_jump if abs(raw_jump) > 1e-9 else np.nan
            # 延迟：显示相对原始的电平差首次达到事件后稳定差 90% 的时刻
            target = 0.9 * (post_off - pre_off)
            lag = np.nan
            for k in range(e - int(0.5 / dt), min(len(tu), e + int(8 / dt))):
                off = Y[k].sum() - tot[k]
                if abs(post_off - pre_off) > 1e-9 and \
                   (off - pre_off) / (post_off - pre_off) >= 0.9:
                    lag = (k - e) * dt
                    break
            # 超额：事件窗内整阵单帧 |Δdisp−Δraw|
            a, b = max(0, e - int(1.0 / dt)), min(len(tu) - 1, e + int(8.0 / dt))
            dYt = np.diff(Y[:, :].sum(axis=1))[a:b]
            dXt = np.diff(tot)[a:b]
            excess = float(np.abs(dYt - dXt).max())
            line += f"{lag:10.2f}{gain:9.3f}{excess:9.0f}"
            rows.append(dict(dataset=tag, event_s=float(tu[e]), raw_jump=float(raw_jump),
                             algo=lab, pre_off=float(pre_off), post_off=float(post_off),
                             lag_s=lag, gain=gain, excess=excess,
                             level_err=float(post_off - pre_off)))
        print(line)

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "varying_steps.csv"), index=False, encoding="utf-8-sig")
print("\n" + "=" * 112)
agg = dfm.groupby("algo").agg(
    事件数=("event_s", "count"),
    阶跃增益=("gain", "mean"),
    跳变超额_整阵ADC=("excess", "mean"),
    跳变超额_最大=("excess", "max"),
    事件后电平误差=("level_err", "mean"),
    延迟_s=("lag_s", "mean"),
).round(3)
print(agg.to_string())
print("\n【读法】gain 越接近 1 越好（1.0 = 负载变化完全保留）；")
print("        超额 = 事件处整阵单帧 |Δ显示−Δ原始|，反映「阶跃被打散/回补」的程度；")
print("        事件后电平误差 = 事件 2~3s 后显示与原始的总量差（负 = 显示偏低）。")
print("saved: results/varying_steps.csv")
