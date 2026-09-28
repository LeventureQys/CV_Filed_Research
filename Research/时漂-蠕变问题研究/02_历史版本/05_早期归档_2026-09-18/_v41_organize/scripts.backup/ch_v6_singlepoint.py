# -*- coding: utf-8 -*-
"""v6 最简估计器验证：Â(τ_d) = [Z(τ_d) − Z(0)] / f̄(τ_d)，f̄ = 恒载 9 组 onset 形状中位（逐点）。
回答：单点形状反演在 τ_d = 0.2~2 s 的误差分布；跨域（实录 onset）是否同样可用。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00])

CL_PAT = [r"temp\右拇指指尖\数据%d\device_001_seg000.csv",
          r"temp\左拇指指尖\数据%d\device_001_seg000.csv",
          r"temp\四指指尖\数据%d\device_001_seg000.csv"]
VL_PAT = [r"temp\变化负载\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc\device_001_seg000.csv",
          r"temp\变化负载\零负载-切换负载-零负载-再切换负载\device_001_seg000.csv",
          r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\device_001_seg000.csv",
          r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\最终测试目标\device_001_seg000.csv"]


def events(path):
    df = pd.read_csv(os.path.join(ROOT, path), skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    t = df["timestamp"].to_numpy(float)
    X = df[ch].to_numpy(float)
    t = t - t[0]
    n = len(t)
    fs = n / t[-1]
    tot = pd.Series(X.sum(axis=1)).rolling(3, center=True, min_periods=1).median().to_numpy()
    w = max(2, int(round(0.08 * fs)))
    sl = np.zeros_like(tot)
    sl[w:] = (tot[w:] - tot[:-w]) / (w / fs)
    md = np.median(sl)
    sg = 1.4826 * np.median(np.abs(sl - md)) + 1e-12
    cand = np.where(np.abs(sl - md) > 10 * sg)[0]
    if not len(cand):
        return []
    gs, cur = [], [cand[0]]
    for x in cand[1:]:
        if (x - cur[-1]) / fs <= 2.0:
            cur.append(x)
        else:
            gs.append(cur)
            cur = [x]
    gs.append(cur)
    peak = float(np.percentile(tot, 99.5))
    out = []
    for g in gs:
        pk = g[int(np.argmax(np.abs(sl[g])))]
        if pk < int(1.5 * fs) or pk > n - int(7.0 * fs):
            continue
        pre = float(np.median(tot[pk - int(1.5 * fs):pk - int(0.3 * fs)]))
        post = float(np.median(tot[pk + int(4.5 * fs):pk + int(5.5 * fs)]))
        j = post - pre
        if j <= 0.02 * peak:
            continue
        i0 = max(pk - int(0.4 * fs), 1)
        for i in range(pk, max(pk - int(0.4 * fs), 1), -1):
            if abs(tot[i] - pre) > 0.04 * j:
                i0 = i - 1
            else:
                break
        i0 = max(i0, 1)
        out.append(dict(i0=i0, pre=pre, post=post, jump=j, tot=tot, fs=fs,
                        kind="onset" if pre < 0.15 * peak else "restep"))
    return out


def prof(e):
    return np.array([(e["tot"][min(len(e["tot"]) - 1, e["i0"] + int(round(g * e["fs"])))] - e["pre"])
                     / e["jump"] for g in GRID])


cl, vl = [], []
for pat in CL_PAT:
    for k in (1, 2, 3):
        cl += events(pat % k)
for p in VL_PAT:
    vl += events(p)

on_cl = [e for e in cl if e["kind"] == "onset"]
on_vl = [e for e in vl if e["kind"] == "onset"]
print("样本：恒载 onset %d 组，实录 onset %d 个" % (len(on_cl), len(on_vl)))
print()
print("τ(s)      :" + "".join(f"{g:>7.2f}" for g in GRID))
P = np.vstack([prof(e) for e in on_cl])
print("恒载 逐组 :")
for i, r in enumerate(P):
    print(f"   #{i+1:<2d}    :" + "".join(f"{v:>7.3f}" for v in r))
Fbar = np.median(P, axis=0)
print("恒载 中位 :" + "".join(f"{v:>7.3f}" for v in Fbar))
print("恒载 极差 :" + "".join(f"{v:>7.3f}" for v in (P.max(axis=0) - P.min(axis=0))))
if on_vl:
    Q = np.vstack([prof(e) for e in on_vl])
    print("实录 中位 :" + "".join(f"{v:>7.3f}" for v in np.median(Q, axis=0)))
    print("实录 极差 :" + "".join(f"{v:>7.3f}" for v in (Q.max(axis=0) - Q.min(axis=0))))

print()
print("=" * 100)
print("单点反演 Â = Z(τ_d)/f̄(τ_d)（f̄ 用恒载 9 组中位形状）  误差 = Â/Z(5s) − 1（%）")
print("=" * 100)
sel = [i for i, g in enumerate(GRID) if 0.15 <= g <= 2.01]


def table(name, evs_):
    if not evs_:
        return
    P2 = np.vstack([prof(e) for e in evs_])
    print(f"\n[{name}] n={len(evs_)}")
    print(f"{'τ_d(s)':<8}" + "".join(f"{GRID[i]:>9.2f}" for i in sel))
    for tag, fn in (("中位", np.median), ("p10", lambda x: np.percentile(x, 10)),
                    ("p90", lambda x: np.percentile(x, 90)),
                    ("max|·|", lambda x: np.max(np.abs(x)))):
        line = f"{tag:<8}"
        for i in sel:
            err = 100 * (P2[:, i] / Fbar[i] - 1)
            line += f"{fn(err):>9.1f}"
        print(line)


table("恒载 onset（形状库自域，含自身→乐观上界）", on_cl)
table("实录 onset（跨域）", on_vl)


def loo(evs_):
    P2 = np.vstack([prof(e) for e in evs_])
    print(f"\n[恒载 onset 留一] n={len(evs_)}")
    print(f"{'τ_d(s)':<8}" + "".join(f"{GRID[i]:>9.2f}" for i in sel))
    for tag, fn in (("中位", np.median), ("p10", lambda x: np.percentile(x, 10)),
                    ("p90", lambda x: np.percentile(x, 90)),
                    ("max|·|", lambda x: np.max(np.abs(x)))):
        line = f"{tag:<8}"
        for i in sel:
            es = []
            for k in range(len(P2)):
                others = np.delete(P2, k, axis=0)
                fb = np.median(others, axis=0)[i]
                if fb < 1e-6:
                    continue
                es.append(100 * (P2[k, i] / fb - 1))
            line += f"{fn(np.array(es)):>9.1f}"
        print(line)


loo(on_cl)
