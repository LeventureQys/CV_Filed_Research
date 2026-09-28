# -*- coding: utf-8 -*-
"""v6 核心估计器验证：把固定形状库换成「自识别单参数形状族」。
  tail(t) = [1 − exp(−(t−τref)/τc)] / [1 − exp(−(5−τref)/τc)]   （t ≥ τref，末端=1）
  在窗 [τref, τd] 上联合最小二乘 (A, τc)；τc 由数据自己识别（= 用户的"K 值定相位"）。
对照：固定形状库（跨工况误差 13~31%）。
"""
import numpy as np
import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TAU_REF = 0.20
T5 = 5.0

CL = [("右拇指_1", "右拇指", r"temp\右拇指指尖\数据1\device_001_seg000.csv"),
      ("右拇指_2", "右拇指", r"temp\右拇指指尖\数据2\device_001_seg000.csv"),
      ("右拇指_3", "右拇指", r"temp\右拇指指尖\数据3\device_001_seg000.csv"),
      ("左拇指_1", "左拇指", r"temp\左拇指指尖\数据1\device_001_seg000.csv"),
      ("左拇指_2", "左拇指", r"temp\左拇指指尖\数据2\device_001_seg000.csv"),
      ("左拇指_3", "左拇指", r"temp\左拇指指尖\数据3\device_001_seg000.csv"),
      ("四指_1", "四指", r"temp\四指指尖\数据1\device_001_seg000.csv"),
      ("四指_2", "四指", r"temp\四指指尖\数据2\device_001_seg000.csv"),
      ("四指_3", "四指", r"temp\四指指尖\数据3\device_001_seg000.csv")]
VL = [("切换负载", "变化", r"temp\变化负载\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc\device_001_seg000.csv"),
      ("再切换", "变化", r"temp\变化负载\零负载-切换负载-零负载-再切换负载\device_001_seg000.csv"),
      ("中途1d9493", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\device_001_seg000.csv"),
      ("中途13ffca", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\最终测试目标\device_001_seg000.csv")]


def events_of(path):
    df = pd.read_csv(path, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    t = df["timestamp"].to_numpy(float)
    X = df[ch].to_numpy(float)
    t = t - t[0]
    n = len(t)
    fs = n / t[-1]
    tu = np.arange(n) / fs
    tot = pd.Series(X.sum(axis=1)).rolling(3, center=True, min_periods=1).median().to_numpy()
    w = max(2, int(round(0.08 * fs)))
    slope = np.zeros_like(tot)
    slope[w:] = (tot[w:] - tot[:-w]) / (w / fs)
    med = np.median(slope)
    sig = 1.4826 * np.median(np.abs(slope - med)) + 1e-12
    cand = np.where(np.abs(slope - med) > 10 * sig)[0]
    if not len(cand):
        return tu, tot, fs, []
    groups, cur = [], [cand[0]]
    for c in cand[1:]:
        if (c - cur[-1]) / fs <= 2.0:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    peak = float(np.percentile(tot, 99.5))
    out = []
    for g in groups:
        pk = g[int(np.argmax(np.abs(slope[g])))]
        if pk < int(1.5 * fs) or pk > n - int(7.0 * fs):
            continue
        pre = float(np.median(tot[pk - int(1.5 * fs):pk - int(0.3 * fs)]))
        post = float(np.median(tot[pk + int(4.5 * fs):pk + int(5.5 * fs)]))
        jump = post - pre
        if jump <= 0.02 * peak:
            continue
        i0 = max(pk - int(0.4 * fs), 1)
        for i in range(pk, max(pk - int(0.4 * fs), 1), -1):
            if abs(tot[i] - pre) > 0.04 * jump:
                i0 = i - 1
            else:
                break
        i0 = max(i0, 1)
        ts = [tu[e[0]] for e in []]
        out.append((i0, jump, pre, post, tu[i0]))
    return tu, tot, fs, out


def tail_model(t, tc):
    t = np.asarray(t, float)
    num = 1.0 - np.exp(-np.maximum(t - TAU_REF, 0.0) / tc)
    den = 1.0 - np.exp(-(T5 - TAU_REF) / tc)
    return num / den


def fit(tu, tot, i0, fs, td, tcs):
    grid = np.arange(TAU_REF, td + 1e-9, 1.0 / fs)
    if len(grid) < 3:
        return np.nan, np.nan
    z = np.array([tot[min(len(tot) - 1, i0 + int(round(g * fs)))] for g in grid])
    zr = float(np.interp(TAU_REF, grid, z))
    if abs(TAU_REF - grid[0]) > 1e-9:
        zr = tot[i0 + int(round(TAU_REF * fs))]
    y = z - zr
    best = None
    for tc in tcs:
        f = tail_model(grid, tc)
        if (f * f).sum() <= 0:
            continue
        A = float((y * f).sum() / (f * f).sum())
        res = float(np.mean((y - A * f) ** 2))
        if best is None or res < best[0]:
            best = (res, A, tc, zr)
    return best


TDS = [0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
TCS = np.exp(np.linspace(np.log(0.25), np.log(3.0), 80))

rows = []
for name, pos, rel in CL + VL:
    tu, tot, fs, evs = events_of(os.path.join(ROOT, rel))
    peak = float(np.percentile(tot, 99.5))
    for i0, jump, pre, post, t0 in evs:
        kind = "onset" if pre < 0.15 * peak else "restep"
        r = dict(ds=name, pos=pos, kind=kind, t=t0, pre=pre, post=post, jump=jump)
        for td in TDS:
            b = fit(tu, tot, i0, fs, td, TCS)
            r[f"e_{td:.2f}"] = 100 * (b[1] + b[3]) / post - 100 if b else np.nan
            r[f"tc_{td:.2f}"] = b[2] if b else np.nan
        # 5s 后重锚（窗口 [2,5]）
        b5 = fit(tu, tot, i0, fs, T5, TCS)
        r["e_5.00"] = 100 * (b5[1] + b5[3]) / post - 100 if b5 else np.nan
        r["tc_5.00"] = b5[2] if b5 else np.nan
        rows.append(r)
ev = pd.DataFrame(rows)
ev.to_csv(os.path.join(os.path.dirname(HERE), "results", "v6_selfshape.csv"),
          index=False, encoding="utf-8-sig")

print("=" * 104)
print("自识别单参数形状族（τc 由窗内数据自己定） + 窗内最小二乘定 A")
print("误差 = Â / Z(5s) − 1（%）   格式：中位|误差| / p90|误差| / 最大|误差|")
print("=" * 104)
for kind in ("onset", "restep"):
    s = ev[ev.kind == kind]
    if not len(s):
        continue
    print(f"\n[{kind}] n={len(s)}")
    print(f"{'τ_d(s)':<10}" + "".join(f"{td:>18.2f}" for td in TDS + [5.0]))
    for tag, fn in (("中位|误差|", lambda x: np.median(np.abs(x))),
                    ("p90|误差|", lambda x: np.percentile(np.abs(x), 90)),
                    ("最大|误差|", lambda x: np.max(np.abs(x)))):
        line = f"{tag:<10}"
        for td in TDS + [5.0]:
            v = s[f"e_{td:.2f}"].dropna().to_numpy()
            line += f"{fn(v):>18.1f}" if len(v) else f"{'n/a':>18}"
        print(line)
    line = f"{'τc 中位':<10}"
    for td in TDS + [5.0]:
        line += f"{s[f'tc_{td:.2f}'].median():>18.2f}"
    print(line)
    line = f"{'τc p10~p90':<10}"
    for td in TDS + [5.0]:
        v = s[f"tc_{td:.2f}"].dropna()
        line += f"{v.quantile(0.1):>8.2f}~{v.quantile(0.9):<9.2f}"
    print(line)

print("\n逐事件（onset，τd=1.0s）：")
s = ev[ev.kind == "onset"]
with pd.option_context("display.width", 200):
    print(s[["ds", "pos", "t", "jump", "e_0.50", "e_1.00", "e_2.00", "e_5.00",
             "tc_1.00", "tc_5.00"]].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print("\n逐事件（restep，τd=1.0s）：")
s = ev[ev.kind == "restep"]
if len(s):
    with pd.option_context("display.width", 200):
        print(s[["ds", "pos", "t", "pre", "jump", "e_1.00", "e_2.00", "tc_1.00"]].to_string(
            index=False, float_format=lambda x: f"{x:,.2f}"))
