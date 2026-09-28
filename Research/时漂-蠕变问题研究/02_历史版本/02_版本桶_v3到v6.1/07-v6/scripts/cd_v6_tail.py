# -*- coding: utf-8 -*-
"""v6 核心量化：把加载沿拆成「瞬时弹性跳变 + 可复现尾巴」，并回答两件事：
  Q2 负载内加重（restep）的尾巴与空载→负载（onset）是否同形；
  Q3 在 τ_d ∈ {0.3,0.5,0.75,1.0,1.5,2.0} s 处用「已知尾巴形状」反演 5s 电平，误差有多大（留一法）。
数据用「包聚合」重建时间轴：时间戳存在成对/成批同刻（同包内两个样本），
先把 dt<1ms 的样本聚成一个显示帧（取均值），网格 = 包周期（≈16.5ms / 60Hz）。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
RES = os.path.join(os.path.dirname(HERE), "results")

DATA = [
    ("右拇指_1", "右拇指", r"temp\右拇指指尖\数据1\device_001_seg000.csv"),
    ("右拇指_2", "右拇指", r"temp\右拇指指尖\数据2\device_001_seg000.csv"),
    ("右拇指_3", "右拇指", r"temp\右拇指指尖\数据3\device_001_seg000.csv"),
    ("左拇指_1", "左拇指", r"temp\左拇指指尖\数据1\device_001_seg000.csv"),
    ("左拇指_2", "左拇指", r"temp\左拇指指尖\数据2\device_001_seg000.csv"),
    ("左拇指_3", "左拇指", r"temp\左拇指指尖\数据3\device_001_seg000.csv"),
    ("四指_1", "四指", r"temp\四指指尖\数据1\device_001_seg000.csv"),
    ("四指_2", "四指", r"temp\四指指尖\数据2\device_001_seg000.csv"),
    ("四指_3", "四指", r"temp\四指指尖\数据3\device_001_seg000.csv"),
    ("切换负载", "变化", r"temp\变化负载\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc\device_001_seg000.csv"),
    ("再切换", "变化", r"temp\变化负载\零负载-切换负载-零负载-再切换负载\device_001_seg000.csv"),
    ("中途1d9493", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\device_001_seg000.csv"),
    ("中途13ffca", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\最终测试目标\device_001_seg000.csv"),
]
TAU_REF = 0.20
TAIL = np.array([0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.25,
                 1.50, 2.00, 2.50, 3.00, 4.00, 5.00])


def load_frames(path):
    """按包聚合：dt<1ms 视为同一显示帧，取该包内各通道均值。"""
    df = pd.read_csv(path, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    t = df["timestamp"].to_numpy(float)
    X = df[ch].to_numpy(float)
    t = t - t[0]
    grp = np.concatenate([[0], np.cumsum(np.diff(t) >= 1e-3)])
    n = grp[-1] + 1
    tp = np.zeros(n)
    Xp = np.zeros((n, X.shape[1]))
    idx = np.arange(len(t))
    cnt = np.bincount(grp, minlength=n)
    for c in range(X.shape[1]):
        Xp[:, c] = np.bincount(grp, weights=X[:, c], minlength=n) / cnt
    tp = np.bincount(grp, weights=t, minlength=n) / cnt
    return tp, Xp, float(np.median(np.diff(tp)))


def find_steps(tot, dt):
    k = 3
    sm = pd.Series(tot).rolling(k, center=True, min_periods=1).median().to_numpy()
    w = max(1, int(round(0.05 / dt)))
    slope = np.zeros_like(sm)
    slope[w:] = (sm[w:] - sm[:-w]) / (w * dt)
    med = np.median(slope)
    sig = 1.4826 * np.median(np.abs(slope - med)) + 1e-12
    cand = np.where(np.abs(slope - med) > 15 * sig)[0]
    if not len(cand):
        return []
    groups, cur = [], [cand[0]]
    for c in cand[1:]:
        if (c - cur[-1]) * dt <= 1.0:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    out = []
    for g in groups:
        pk = g[int(np.argmax(np.abs(slope[g])))]
        if pk < int(1.0 / dt) or pk > len(tot) - int(6.5 / dt):
            continue
        pre = float(np.median(sm[pk - int(1.8 / dt):pk - int(0.25 / dt)]))
        post = float(np.median(sm[pk + int(4.6 / dt):pk + int(5.4 / dt)]))
        jump = post - pre
        if abs(jump) < 1e-9:
            continue
        t0 = pk
        for i in range(pk, max(pk - int(0.4 / dt), 1), -1):
            if abs(sm[i] - pre) > 0.05 * abs(jump):
                t0 = i - 1
            else:
                break
        t0 = max(t0, 1)
        quiet = sm[t0 - int(1.2 / dt):t0 - int(0.25 / dt)]
        if len(quiet) < 10 or np.std(quiet) > 0.04 * abs(jump):
            continue
        out.append((t0, jump, pre, post))
    return out


def at(tot, t0, tau, dt):
    return float(tot[t0 + int(round(tau / dt))])


rows = []
store = {}
print("=" * 112)
print("A. 显示帧网格与加载沿分解（t0 = 阶跃前最后一帧；τ_ref = %.2fs）" % TAU_REF)
print("=" * 112)
for name, pos, rel in DATA:
    tp, Xp, dsp = load_frames(os.path.join(ROOT, rel))
    tot = Xp.sum(axis=1)
    n = len(tp)
    store[name] = (tp, Xp, tot, dsp)
    peak = float(np.percentile(tot, 99.5))
    print(f"\n[{name}] 显示帧 {n} 帧，包周期 {1000*dsp:.2f}ms  ({1/dsp:.1f} Hz)  "
          f"时长 {tp[-1]:.1f}s  峰值 {peak:,.0f}")
    for t0, jump, pre, post in find_steps(tot, dsp):
        if jump <= 0:
            kind = "unload"
        elif pre < 0.15 * peak:
            kind = "onset"
        else:
            kind = "restep"
        z_ref = at(tot, t0, TAU_REF, dsp)
        tail = post - z_ref
        prof = {f"g_{g:.2f}": (at(tot, t0, g, dsp) - z_ref) / tail if abs(tail) > 1e-9 else np.nan
                for g in TAIL}
        rows.append(dict(ds=name, pos=pos, t=float(tp[t0]), kind=kind, pre=pre, post=post,
                         jump=jump, inst_frac=(z_ref - pre) / jump if jump else np.nan,
                         tail_frac=tail / jump if jump else np.nan,
                         **prof))
        if kind in ("onset", "restep"):
            print(f"   {kind:6s} t={tp[t0]:7.2f}s  前级={pre:10.1f}  5s级={post:10.1f}  "
                  f"阶跃={jump:10.1f}  瞬时占比={(z_ref-pre)/jump:5.3f}  尾巴占比={tail/jump:5.3f}")
ev = pd.DataFrame(rows)
ev.to_csv(os.path.join(RES, "v6_tail_events.csv"), index=False, encoding="utf-8-sig")

gcols = [f"g_{g:.2f}" for g in TAIL]
print()
print("=" * 112)
print("B. 尾巴形状（相对 τ_ref=0.2s 归一化，1.0 = 5s 电平）")
print("=" * 112)


def show(sub, title):
    if not len(sub):
        print(f"\n{title}: 无样本")
        return None
    A = sub[gcols].to_numpy(float)
    ok = np.isfinite(A).all(axis=1)
    A = A[ok]
    if not len(A):
        print(f"\n{title}: 无有效样本")
        return None
    med = np.median(A, axis=0)
    print(f"\n{title}（n={len(A)}）")
    print("τ(s)   :" + "".join(f"{g:>7.2f}" for g in TAIL))
    print("中位   :" + "".join(f"{v:>7.3f}" for v in med))
    print("p10    :" + "".join(f"{v:>7.3f}" for v in np.percentile(A, 10, axis=0)))
    print("p90    :" + "".join(f"{v:>7.3f}" for v in np.percentile(A, 90, axis=0)))
    print("半极差 :" + "".join(f"{v:>7.3f}" for v in (np.percentile(A, 90, axis=0) - np.percentile(A, 10, axis=0)) / 2))
    return med


ons = ev[ev.kind == "onset"].copy()
res_ = ev[ev.kind == "restep"].copy()
med_on = show(ons, "onset（空载→负载）")
med_re = show(res_, "restep（负载内加重）")
show(res_[res_.jump / res_.pre > 0.15], "restep 且台阶/前级 > 0.15")

if med_on is not None and med_re is not None:
    print("\nonset 与 restep 尾巴之差（中位曲线）：")
    print("        " + "".join(f"{v:>7.3f}" for v in (med_re - med_on)))

print()
print("=" * 112)
print("C. 在 τ_d 处的反演误差（留一法：用「其余事件」的中位尾巴形状反演自身 5s 电平）")
print("=" * 112)
print("口径：Â = Z(0.2s) + Σ_{τ∈[τ_d,τ_d+0.6]}(Z(τ)−Z(0.2s))·f(τ) / Σ f(τ)² ，f 归一到 τ=5s 处为 1")
print("误差 = Â / Z(5s) − 1")


def loo(sub, taus_d, label, win=0.6):
    A = sub[gcols].to_numpy(float)
    Z9 = sub["post"].to_numpy(float)
    zref = sub["post"].to_numpy(float) - sub["jump"].to_numpy(float) * sub["tail_frac"].to_numpy(float)
    ok = np.isfinite(A).all(axis=1)
    sub2 = sub[ok]
    A, Z9, zref = A[ok], Z9[ok], zref[ok]
    dss = sub2["ds"].to_numpy()
    print(f"\n[{label}] n={len(sub2)}   （单位 %）")
    print("τ_d(s)       " + "".join(f"{t:>16.2f}" for t in taus_d))
    for tag in ("中位", "p10", "p90"):
        pass
    res = {}
    for td in taus_d:
        m = np.array([td <= g <= td + win for g in TAIL])
        if not m.any():
            continue
        errs = []
        for i in range(len(sub2)):
            others = (dss != dss[i]) if len(set(dss)) > 1 else np.ones(len(sub2), bool)
            others = others & (np.arange(len(sub2)) != i)
            f = np.median(A[others][:, m], axis=0)
            zz = zref[i] + (A[i][m] * (Z9[i] - zref[i]) / 1.0)
            # Â = zref + (tail_samples · f)/(f·f)
            tail_samples = A[i][m] * (Z9[i] - zref[i])
            Ahat = zref[i] + float((tail_samples * f).sum() / (f * f).sum())
            errs.append(100 * (Ahat / Z9[i] - 1))
        errs = np.array(errs)
        res[td] = errs
    for name_, fn in (("中位", np.median), ("|误差|中位", lambda x: np.median(np.abs(x))),
                      ("p10", lambda x: np.percentile(x, 10)), ("p90", lambda x: np.percentile(x, 90)),
                      ("max|误差|", lambda x: np.max(np.abs(x)))):
        print(f"{name_:<12}" + "".join(f"{fn(res[t]):>16.2f}" for t in res))
    return res


if med_on is not None:
    loo(ons, [0.30, 0.50, 0.75, 1.00, 1.50, 2.00], "onset")
if len(res_):
    loo(res_, [0.30, 0.50, 0.75, 1.00, 1.50, 2.00], "restep")

print()
print("=" * 112)
print("D. 从 5s 电平到段末的蠕变增量（决定「1s 稳住」之后还会不会被慢相拉走）")
print("=" * 112)
for name, pos, rel in DATA:
    tp, Xp, tot, dsp = store[name]
    peak = float(np.percentile(tot, 99.5))
    ld = tot > 0.5 * peak
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    if not len(s):
        continue
    i0, i1 = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]
    if (i1 - i0) * dsp < 20:
        continue
    n_on = int(np.searchsorted(tp, tp[i0] + 6.0 / 1.0)) if False else i0
    z5 = float(np.median(tot[i0 + int(4.6 / dsp):i0 + int(5.4 / dsp)]))
    z_end = float(np.median(tot[max(i1 - 80, i0):i1]))
    print(f"{name:<12} 负载段 {tp[i0]:7.1f}~{tp[i1]:7.1f}s  5s级={z5:10.1f}  段末={z_end:10.1f}  "
          f"5s→段末 蠕变 +{100*(z_end/z5-1):5.1f}%  （时长 {(i1-i0)*dsp:5.1f}s）")
print(f"\nsaved: {os.path.join(RES, 'v6_tail_events.csv')}")
