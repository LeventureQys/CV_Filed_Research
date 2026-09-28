# -*- coding: utf-8 -*-
"""v6 设计前置探测（只读数据，不改算法）：
   Q1 快相形状的"可反演性"：用已知形状反演幅度 A 的时延-精度极限（1/f 噪声增益）。
   Q2 负载内加重（restep）的爬升形状是否与空载→负载（onset）相同；若不同，等效"已走过"时间 δ 是多少。
   Q3 因果可用的检测时延：斜率/电平判据在多少秒内能命中，随台阶大小如何变化。
输出：results/v6_stepshape_*.csv + 控制台报告。
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))  # 仓库根
RES = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RES, exist_ok=True)

FS = 100.0
TAUS = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80,
                 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00])

DATA = [
    # (组名, 位置, 路径)
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


def load_uniform(path):
    df = pd.read_csv(path, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    t = df["timestamp"].to_numpy(float)
    X = df[ch].to_numpy(float)
    t = t - t[0]
    n = int(t[-1] * FS) + 1
    tu = np.arange(n) / FS
    Xu = np.empty((n, X.shape[1]))
    for c in range(X.shape[1]):
        Xu[:, c] = np.interp(tu, t, X[:, c])
    return tu, Xu


def find_steps(tot, dt):
    """返回 [(t0_idx, jump, pre_level)]，覆盖 onset 与 restep、加重与减重。"""
    k = max(1, int(0.03 / dt))
    sm = uniform_filter1d(tot, k)
    w = max(1, int(0.05 / dt))
    slope = np.zeros_like(sm)
    slope[w:] = (sm[w:] - sm[:-w]) / (w * dt)
    sl = slope[w:]
    mad = np.median(np.abs(sl - np.median(sl)))
    sig = 1.4826 * mad + 1e-12
    cand = np.where(np.abs(sl) > 12.0 * sig)[0] + w
    if not len(cand):
        return []
    # 聚类：1.0 s 内只留斜率绝对值最大的点
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
        pre_lvl = float(np.median(sm[pk - int(2.0 / dt):pk - int(0.3 / dt)]))
        post_lvl = float(np.median(sm[pk + int(4.8 / dt):pk + int(5.2 / dt)]))
        jump = post_lvl - pre_lvl
        if abs(jump) < 1e-6:
            continue
        # 回退找 t0：台阶达到 3% 的点
        t0 = pk
        lim = pk - int(0.6 / dt)
        for i in range(pk, max(lim, 1), -1):
            if abs(sm[i] - pre_lvl) > 0.03 * abs(jump):
                t0 = i
            else:
                break
        # 前置静默检查（1.5 s 内不得有其它事件）
        quiet = sm[t0 - int(1.5 / dt):t0 - int(0.25 / dt)]
        if len(quiet) < 20:
            continue
        if np.std(quiet) > 0.05 * abs(jump):
            continue
        if abs(np.median(quiet) - pre_lvl) > 0.05 * abs(jump):
            continue
        out.append((t0, jump, pre_lvl, float(sm[t0 + int(6.0 / dt)])))
    return out


def frame_sigma(tot, dt):
    d = np.diff(tot)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def shape_of(tot, t0, pre_lvl, jump, taus=TAUS):
    idx = t0 + (taus * FS).astype(int)
    idx = np.clip(idx, 0, len(tot) - 1)
    k = max(1, int(0.02 / (1.0 / FS)))
    sm = uniform_filter1d(tot, k)
    return (sm[idx] - pre_lvl) / jump


def detect_latency(tot, t0, pre_lvl, jump, sig_tot, ks=(3.0, 5.0, 8.0)):
    """因果"短滞后电平差"判据：d(τ)=mean[τ-0.1,τ] − mean[τ-0.35,τ-0.15]。
       参考窗/近窗各 0.1 s、0.2 s，噪声 σ_d = σ_tot·sqrt(1/10+1/20)。"""
    dt = 1.0 / FS
    n1, n2 = int(0.10 / dt), int(0.20 / dt)
    sig_d = sig_tot * np.sqrt(1.0 / n1 + 1.0 / n2)
    out = {}
    taus = np.arange(0.05, 1.20, 0.01)
    dvals = []
    for tau in taus:
        i = t0 + int(tau / dt)
        rec = np.mean(tot[i - n1:i])
        ref = np.mean(tot[i - n1 - n2:i - n1])
        dvals.append(rec - ref)
    dvals = np.array(dvals)
    for kk in ks:
        hit = np.where(dvals > kk * sig_d)[0]
        out[kk] = float(taus[hit[0]]) if len(hit) else np.nan
    return out, sig_d


def main():
    rows, curves, latency_rows = [], [], []
    store = {}
    for name, pos, rel in DATA:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"[skip] {p}")
            continue
        tu, Xu = load_uniform(p)
        tot = Xu.sum(axis=1)
        dt = 1.0 / FS
        sig_tot = frame_sigma(tot, dt)
        peak = float(np.percentile(uniform_filter1d(tot, 50), 99))
        evs = find_steps(tot, dt)
        store[name] = (tu, Xu, tot)
        for t0, jump, pre_lvl, lvl6 in evs:
            if jump <= 0:
                kind = "unload"
            elif pre_lvl < 0.15 * peak:
                kind = "onset"
            else:
                kind = "restep"
            sh = shape_of(tot, t0, pre_lvl, jump)
            lat, sig_d = detect_latency(tot, t0, pre_lvl, jump, sig_tot)
            rows.append(dict(ds=name, pos=pos, t=float(tu[t0]), kind=kind,
                             pre=pre_lvl, jump=jump, lvl6=lvl6,
                             ratio=jump / max(pre_lvl, 1e-9),
                             ratio_new=jump / max(lvl6, 1e-9),
                             sig_tot=sig_tot, sig_d=sig_d,
                             **{f"tau{k}": lat[k] for k in lat}))
            curves.append(dict(ds=name, pos=pos, t=float(tu[t0]), kind=kind,
                               pre=pre_lvl, jump=jump,
                               **{f"g_{tau:.2f}": v for tau, v in zip(TAUS, sh)}))
            latency_rows.append(dict(ds=name, kind=kind, jump=jump, sig_d=sig_d,
                                     **{f"lat{k}": lat[k] for k in lat}))
        print(f"[ok] {name:12s} frames={len(tu):6d} sigma_tot={sig_tot:8.2f} events={len(evs)}")

    ev = pd.DataFrame(rows)
    cv = pd.DataFrame(curves)
    ev.to_csv(os.path.join(RES, "v6_stepshape_events.csv"), index=False, encoding="utf-8-sig")
    cv.to_csv(os.path.join(RES, "v6_stepshape_curves.csv"), index=False, encoding="utf-8-sig")

    print("\n================ 事件台账 ================")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(ev[["ds", "t", "kind", "pre", "jump", "ratio", "ratio_new"]].to_string(
            index=False, float_format=lambda x: f"{x:,.3f}"))

    gcols = [c for c in cv.columns if c.startswith("g_")]

    def report(sub, title):
        if not len(sub):
            print(f"\n---- {title}: 无样本 ----")
            return None
        arr = sub[gcols].to_numpy(float)
        med = np.median(arr, axis=0)
        lo = np.percentile(arr, 10, axis=0)
        hi = np.percentile(arr, 90, axis=0)
        print(f"\n---- {title}（n={len(sub)}）----")
        print("tau(s)      :" + "".join(f"{t:>8.2f}" for t in TAUS))
        print("f(τ) 中位   :" + "".join(f"{v:>8.3f}" for v in med))
        print("p10~p90     :" + "".join(f"{a:>4.2f}~{b:<4.2f}" for a, b in zip(lo, hi)))
        print("反演相对误差: " + "".join(
            f"{100*(hi[i]/med[i]-1):>7.1f}%" if med[i] > 1e-3 else "     n/a" for i in range(len(med))))
        return med

    print("\n================ 归一化快相形状 f(τ) ================")
    onset = cv[cv.kind == "onset"]
    restep = cv[cv.kind == "restep"]
    med_on = report(onset, "onset（空载→负载）")
    med_re = report(restep, "restep（负载内变重）")

    print("\n---- 分位置 onset 形状（跨位置离散度）----")
    pos_med = {}
    for pos in ["右拇指", "左拇指", "四指"]:
        s = onset[onset.pos == pos]
        if len(s):
            m = np.median(s[gcols].to_numpy(float), axis=0)
            pos_med[pos] = m
            print(f"{pos} (n={len(s)}): " + "".join(f"{v:>8.3f}" for v in m))
    if len(pos_med) >= 2:
        M = np.vstack(list(pos_med.values()))
        spread = (M.max(axis=0) - M.min(axis=0)) / np.maximum(M.mean(axis=0), 1e-9)
        print("跨位置极差/均值:" + "".join(f"{100*v:>7.1f}%" for v in spread))

    print("\n================ 反演精度（用上述中位形状）=================")
    wins = [(0.10, 0.30), (0.10, 0.50), (0.15, 1.00), (0.20, 1.00),
            (0.30, 1.00), (0.15, 1.50), (0.30, 1.50), (0.50, 2.00), (0.15, 5.00), (0.10, 5.00)]
    for label, sub, med in (("onset", onset, med_on), ("restep", restep, med_re)):
        if med is None:
            continue
        print(f"\n[{label}]  单点反演 / 窗内最小二乘反演  相对误差（%）")
        print("窗口(s)        " + "".join(f"{f'[{a},{b}]':>18}" for a, b in wins))
        for tag, mode in (("单点", "pt"), ("最小二乘", "ls")):
            line = f"{tag:8s}  p10/中位/p90  "
            errs = []
            for a, b in wins:
                m = np.array([a <= t <= b for t in TAUS])
                if not m.any():
                    errs.append((np.nan,) * 3)
                    continue
                f = med[m]
                A = sub[gcols].to_numpy(float)[:, m]
                if mode == "pt":
                    est = A[:, -1] / f[-1]
                else:
                    est = (A * f).sum(axis=1) / (f * f).sum()
                rel = 100 * (est / sub["jump"].to_numpy(float) - 1)
                errs.append((np.percentile(rel, 10), np.median(rel), np.percentile(rel, 90)))
            print(line + "".join(f"{a:>6.1f}/{b:>5.1f}/{c:>5.1f}" for a, b, c in errs))

    print("\n================ restep 等效已走过时间 δ ================")
    print("模型：Z(τ) = L_pre + A·[f(τ+δ) − f(δ)]  （f 在 τ>5s 后取 1）")


    def ext_f(taus):
        tau = np.concatenate([TAUS, [6, 8, 10, 20, 40]])
        f = np.concatenate([med_on, np.ones(5)])
        return np.interp(taus, tau, f)


    delta_rows = []
    for _, r in restep.iterrows():
        ds, t0s, pre, jump = r["ds"], r["t"], r["pre"], r["jump"]
        tu, Xu, tot = store[ds]
        t0 = int(round(t0s * FS))
        k = max(1, int(0.02 * FS))
        sm = uniform_filter1d(tot, k)
        grid = np.arange(0.05, 2.01, 0.01)
        y = sm[t0 + (grid * FS).astype(int)] - pre
        best = None
        for d in np.arange(0.0, 2.01, 0.05):
            f = ext_f(grid + d) - ext_f(np.array([d]))[0]
            A = float((y * f).sum() / (f * f).sum())
            res = float(np.sqrt(np.mean((y - A * f) ** 2)))
            if best is None or res < best[0]:
                best = (res, d, A)
        res0 = None
        f0 = ext_f(grid)
        A0 = float((y * f0).sum() / (f0 * f0).sum())
        res0 = float(np.sqrt(np.mean((y - A0 * f0) ** 2)))
        delta_rows.append(dict(ds=ds, t=t0s, jump=jump, ratio=jump / pre,
                               delta=best[1], res_fit=best[0], res_delta0=res0,
                               A_fit=best[2], A0=A0))
    if delta_rows:
        dd = pd.DataFrame(delta_rows)
        with pd.option_context("display.width", 200):
            print(dd.to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
        dd.to_csv(os.path.join(RES, "v6_restep_delta.csv"), index=False, encoding="utf-8-sig")

    print("\n================ 检测时延（因果短滞后电平差判据）=================")
    lat = pd.DataFrame(latency_rows)
    for kk in (3.0, 5.0, 8.0):
        c = f"lat{kk}"
        sub = lat[np.isfinite(lat[c])]
        print(f"\nk={kk}σ  （σ_d 由各录制噪声在线估计）  命中 {len(sub)}/{len(lat)}")
        if len(sub):
            with pd.option_context("display.width", 200):
                print(sub[["ds", "kind", "jump", "sig_d", c]].to_string(
                    index=False, float_format=lambda x: f"{x:,.4g}"))
    lat.to_csv(os.path.join(RES, "v6_detect_latency.csv"), index=False, encoding="utf-8-sig")
    print(f"\n产物写入 {RES}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
