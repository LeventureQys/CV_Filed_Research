# -*- coding: utf-8 -*-
"""v6 决策用实验（只读）：
  E1 9 组恒载 onset 的原始形状（逐组 + 组间离散度）——回答「瞬时跳变 + 尾巴」是否可复现；
  E2 restep vs onset 的爬升时间常数 t50/t80/t90/t95（干净事件，剔除相邻 3s/8s 内有其它事件的）；
  E3 两种反演估计器在 τ_d 处的误差（留一法）：
      E1p：单参数，尾巴形状固定、τ_ref=0.2s，窗内最小二乘
      E2p：双参数 (A, δ)，用完整形状 F(τ+δ) 在 [0,τ_d] 上最小二乘（δ∈[-2帧,+4帧]）
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
RES = os.path.join(os.path.dirname(HERE), "results")

CL = [
    ("右拇指_1", "右拇指", r"temp\右拇指指尖\数据1\device_001_seg000.csv"),
    ("右拇指_2", "右拇指", r"temp\右拇指指尖\数据2\device_001_seg000.csv"),
    ("右拇指_3", "右拇指", r"temp\右拇指指尖\数据3\device_001_seg000.csv"),
    ("左拇指_1", "左拇指", r"temp\左拇指指尖\数据1\device_001_seg000.csv"),
    ("左拇指_2", "左拇指", r"temp\左拇指指尖\数据2\device_001_seg000.csv"),
    ("左拇指_3", "左拇指", r"temp\左拇指指尖\数据3\device_001_seg000.csv"),
    ("四指_1", "四指", r"temp\四指指尖\数据1\device_001_seg000.csv"),
    ("四指_2", "四指", r"temp\四指指尖\数据2\device_001_seg000.csv"),
    ("四指_3", "四指", r"temp\四指指尖\数据3\device_001_seg000.csv"),
]
VL = [
    ("切换负载", "变化", r"temp\变化负载\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc\device_001_seg000.csv"),
    ("再切换", "变化", r"temp\变化负载\零负载-切换负载-零负载-再切换负载\device_001_seg000.csv"),
    ("中途1d9493", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\device_001_seg000.csv"),
    ("中途13ffca", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\最终测试目标\device_001_seg000.csv"),
]

GRID = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00])


def load_frames(path):
    df = pd.read_csv(path, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    t = df["timestamp"].to_numpy(float)
    X = df[ch].to_numpy(float)
    t = t - t[0]
    grp = np.concatenate([[0], np.cumsum(np.diff(t) >= 1e-3)])
    n = grp[-1] + 1
    cnt = np.bincount(grp, minlength=n)
    Xp = np.column_stack([np.bincount(grp, weights=X[:, c], minlength=n) / cnt
                          for c in range(X.shape[1])])
    tp = np.bincount(grp, weights=t, minlength=n) / cnt
    return tp, Xp, float(np.median(np.diff(tp)))


def events(tot, dt):
    sm = pd.Series(tot).rolling(3, center=True, min_periods=1).median().to_numpy()
    w = max(2, int(round(0.08 / dt)))
    slope = np.zeros_like(sm)
    slope[w:] = (sm[w:] - sm[:-w]) / (w * dt)
    med = np.median(slope)
    sig = 1.4826 * np.median(np.abs(slope - med)) + 1e-12
    cand = np.where(np.abs(slope - med) > 10 * sig)[0]
    if not len(cand):
        return []
    groups, cur = [], [cand[0]]
    for c in cand[1:]:
        if (c - cur[-1]) * dt <= 1.5:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    out = []
    for g in groups:
        pk = g[int(np.argmax(np.abs(slope[g])))]
        if pk < int(1.5 / dt) or pk > len(tot) - int(7.0 / dt):
            continue
        pre = float(np.median(sm[pk - int(1.5 / dt):pk - int(0.3 / dt)]))
        post = float(np.median(sm[pk + int(4.5 / dt):pk + int(5.5 / dt)]))
        jump = post - pre
        if abs(jump) < 1e-9:
            continue
        t0 = max(pk - int(0.4 / dt), 1)
        for i in range(pk, max(pk - int(0.4 / dt), 1), -1):
            if abs(sm[i] - pre) > 0.04 * abs(jump):
                t0 = i - 1
            else:
                break
        out.append((max(t0, 1), jump, pre, post))
    return out


def at(tot, i0, tau, dt):
    k = min(len(tot) - 1, i0 + int(round(tau / dt)))
    return float(tot[k])


def first_time(tot, i0, pre, level, frac, dt, tmax=10.0):
    tgt = pre + frac * (level - pre)
    n = min(len(tot) - 1, i0 + int(tmax / dt))
    seg = tot[i0:n]
    idx = np.where(seg >= tgt)[0] if level > pre else np.where(seg <= tgt)[0]
    return float(idx[0] * dt) if len(idx) else np.nan


print("=" * 108)
print("E1 恒载 9 组 onset 的原始形状（相对 5s 电平归一化）")
print("=" * 108)
rows = []
for name, pos, rel in CL:
    tp, Xp, dt = load_frames(os.path.join(ROOT, rel))
    tot = Xp.sum(axis=1)
    ev = [e for e in events(tot, dt) if e[1] > 0]
    if not ev:
        print(f"[{name}] 未检出加载沿")
        continue
    i0, jump, pre, post = ev[0]
    prof = [(at(tot, i0, g, dt) - pre) / jump for g in GRID]
    rows.append(dict(ds=name, pos=pos, dt_ms=1000 * dt, t=float(tp[i0]), pre=pre, post=post,
                     **{f"f_{g:.2f}": v for g, v in zip(GRID, prof)}))
    print(f"[{name:<8}] t={tp[i0]:6.2f}s  空载={pre:6.2f}  5s={post:6.2f}  " +
          "".join(f"{v:>6.3f}" for v in prof))
df = pd.DataFrame(rows)
gcols = [f"f_{g:.2f}" for g in GRID]
print("\nτ(s)   :" + "".join(f"{g:>6.2f}" for g in GRID))
print("中位   :" + "".join(f"{v:>6.3f}" for v in df[gcols].median()))
print("极差   :" + "".join(f"{v:>6.3f}" for v in (df[gcols].max() - df[gcols].min())))
print("半极差 :" + "".join(f"{v:>6.3f}" for v in (df[gcols].max() - df[gcols].min()) / 2))
F_on = df[gcols].median().to_numpy()
df.to_csv(os.path.join(RES, "v6_onset_profile9.csv"), index=False, encoding="utf-8-sig")

print()
print("=" * 108)
print("E2 爬升时间常数：t50/t80/t90/t95（干净事件；t0 后 8s、前 3s 内无其它事件）")
print("=" * 108)
rise = []
for name, pos, rel in CL + VL:
    tp, Xp, dt = load_frames(os.path.join(ROOT, rel))
    tot = Xp.sum(axis=1)
    peak = float(np.percentile(tot, 99.5))
    ev = events(tot, dt)
    ts = [tp[e[0]] for e in ev]
    for i0, jump, pre, post in ev:
        t0 = tp[i0]
        if any(0 < abs(x - t0) <= (3.0 if x < t0 else 8.0) for x in ts if x != t0):
            continue
        if jump <= 0.02 * peak:
            continue
        kind = "onset" if pre < 0.15 * peak else "restep"
        r = dict(ds=name, pos=pos, kind=kind, t=t0, dt_ms=1000 * dt, pre=pre, post=post,
                 jump=jump, ratio=jump / max(pre, 1e-9))
        for f in (0.5, 0.8, 0.9, 0.95):
            r[f"t{int(f*100)}"] = first_time(tot, i0, pre, post, f, dt)
        r["z_at_02"] = (at(tot, i0, 0.2, dt) - pre) / jump
        r["z_at_10"] = (at(tot, i0, 1.0, dt) - pre) / jump
        rise.append(r)
rr = pd.DataFrame(rise)
rr.to_csv(os.path.join(RES, "v6_rise_times.csv"), index=False, encoding="utf-8-sig")
for kind in ("onset", "restep"):
    s = rr[rr.kind == kind]
    if not len(s):
        continue
    print(f"\n[{kind}] n={len(s)}")
    print("   t50 中位 {:.2f}s   t80 {:.2f}s   t90 {:.2f}s   t95 {:.2f}s".format(
        s.t50.median(), s.t80.median(), s.t90.median(), s.t95.median()))
    print("   Z(0.2s)/阶跃 中位 {:.3f}   Z(1s)/阶跃 中位 {:.3f}".format(
        s.z_at_02.median(), s.z_at_10.median()))
    with pd.option_context("display.width", 200):
        print(s[["ds", "t", "dt_ms", "kind", "pre", "jump", "ratio", "z_at_02", "z_at_10",
                 "t50", "t80", "t90", "t95"]].to_string(
            index=False, float_format=lambda x: f"{x:,.3f}"))

print()
print("=" * 108)
print("E3 反演估计器误差（留一法，用其余恒载组的形状）")
print("=" * 108)


def estimate(tot, i0, dt, Fgrid, taus, delta_max, two_param, tmax):
    """返回 Â。Fgrid=(τ, F) 完整形状，τ=0 时 F=0，τ=5s 时 F=1。"""
    grid = np.arange(0.0, tmax + 1e-9, dt)
    z = np.array([at(tot, i0, g, dt) for g in grid])
    z0 = z[0]
    y = z - z0
    if not two_param:
        m = (grid >= taus[0]) & (grid <= taus[1])
        f = np.interp(grid[m], Fgrid[0], Fgrid[1])
        f = f / max(Fgrid[1][-1], 1e-9)
        if (f * f).sum() <= 0:
            return np.nan
        return z0 + float((y[m] * f).sum() / (f * f).sum()) * 1.0
    # 双参数：y ≈ A·F(τ+δ)，τ=0 处 F=0
    best = None
    for dd in np.arange(-delta_max, delta_max + 1e-9, dt):
        f = np.interp(np.clip(grid + dd, 0, 1e6), Fgrid[0], Fgrid[1],
                      left=0.0, right=1.0)
        if (f * f).sum() <= 0:
            continue
        A = float((y * f).sum() / (f * f).sum())
        res = float(np.mean((y - A * f) ** 2))
        if best is None or res < best[0]:
            best = (res, A)
    return z0 + best[1] if best else np.nan


def run(sub_events, Fgrid, label, two_deltas):
    print(f"\n[{label}]  误差 = Â / Z(5s) − 1（%）")
    hdr = ["0.20", "0.30", "0.50", "0.75", "1.00", "1.50"]
    print(f"{'估计器':<22}" + "".join(f"{h:>17}" for h in hdr))
    for tag, two_param, dmax in (("单参数(尾巴LSQ)", False, 0.0),
                                 ("双参数(A,δ) 全形状", True, two_deltas)):
        line = f"{tag:<22}"
        for td in (0.20, 0.30, 0.50, 0.75, 1.00, 1.50):
            errs = []
            for j, (ds, rel, i0, dt, pre, post, jump) in enumerate(sub_events):
                tp, Xp, _ = load_frames(os.path.join(ROOT, rel))
                tot = Xp.sum(axis=1)
                others = [k for k in range(len(sub_events)) if k != j]
                Fs = np.vstack([sub_events[k][6] for k in others])
                Fm = np.median(Fs, axis=0)
                if two_param:
                    Ah = estimate(tot, i0, dt, (Fgrid, Fm), (0.0, td), dmax, True, td)
                else:
                    Ah = estimate(tot, i0, dt, (Fgrid, Fm), (td, min(td + 0.6, 5.0)), 0, False, 5.0)
                if np.isfinite(Ah):
                    errs.append(100 * (Ah / post - 1))
            errs = np.array(errs)
            if len(errs):
                line += f"{np.median(np.abs(errs)):>7.1f}/{np.max(np.abs(errs)):>7.1f}  "
            else:
                line += f"{'n/a':>17}"
        print(line + "   ← 中位|误差|/最大|误差|")


# 用 9 组恒载构造形状库
FG = np.arange(0.0, 5.0 + 1e-9, 1.0 / 60.0)
lib = []
for name, pos, rel in CL:
    tp, Xp, dt = load_frames(os.path.join(ROOT, rel))
    tot = Xp.sum(axis=1)
    ev = [e for e in events(tot, dt) if e[1] > 0]
    i0, jump, pre, post = ev[0]
    F = np.array([at(tot, i0, g, dt) for g in FG]) - pre
    F = F / F[-1]
    lib.append((name, rel, i0, dt, pre, post, F))
run(lib, FG, "恒载 9 组 onset", 5.0 / 60.0)
