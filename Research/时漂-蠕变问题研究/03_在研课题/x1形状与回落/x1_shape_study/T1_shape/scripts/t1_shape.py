# -*- coding: utf-8 -*-
"""步骤 2：提取实测快相蠕变形状并拟合四种形态（v2，含窗口起点稳健性与相图法 τ）。

对每个「加载沿 + 长保压」事件：
  t1    = 沿末（|dv/dt| 首次跌回门限）
  t_settle = 沿后导数衰减到峰值 5% 并保持 0.5 s 的时刻（机械加载尾的保守终点）
  两个窗口起点：A = t1（可能含机械尾，τ 偏小），B = t_settle（漏掉早期快相，τ 偏大）
  窗口长度 W = min(30 s, 保压 − 0.5 s)
  拟合 y(t) = P − C·f(t−t1)：
    fixed8 / single / stretch / double（见 README 或报告正文）

  另外给出不依赖"落点"假设的相图法：若 dz/dt = (P−z)/τ，则 (z, dz/dt) 为直线，
  斜率 −1/τ、截距 P/τ ⇒ τ_phase、P_phase 由线性回归直接得到。

主信号两种：tot = 21 通道和（信噪比最好）；逐通道（用于形状参数与幅度/位置的相关性）。
输出 results/t1_shape_events.csv
用法： python t1_shape.py [working|archived|both]
"""
import csv
import os
import sys

import numpy as np

import t1_lib as T

MAX_RAMP = 3.5        # 沿宽上限 s
MIN_PLATEAU = 18.0    # 保压时长下限 s
WIN = 30.0            # 快相窗口长度上限 s
MIN_CH_STEP = 25.0    # 通道自身台阶下限 ADC
MIN_CH_RANGE = 125.0  # 通道量程下限 ADC
NOISE_WIN = 3.0       # 静默噪声估计窗长 s
SETTLE_FRAC = 0.05    # （保留：早期自适应判据用的常数，现未启用）
SKIP_B = 3.0          # 窗口 B 的额外跳过时长 s（固定）

COLS = ["ses", "sig", "ch", "t0", "t1", "t_settle", "ramp", "plateau",
        "noise", "lv_pre", "lv_post", "dch", "rate",
        "c30_A", "c2_30_A", "c5_30_A", "c10_30_A", "c20_30_A",
        "c30_B", "c2_30_B", "c5_30_B", "c10_30_B", "c20_30_B",
        "A_fixed8_rms", "A_single_tau", "A_single_rms", "A_single_aic",
        "A_stretch_tau", "A_stretch_beta", "A_stretch_rms", "A_stretch_aic",
        "A_double_tau1", "A_double_tau2", "A_double_C1", "A_double_C2",
        "A_double_rms", "A_double_aic",
        "B_fixed8_rms", "B_single_tau", "B_single_rms", "B_single_aic",
        "B_stretch_tau", "B_stretch_beta", "B_stretch_rms", "B_stretch_aic",
        "B_double_tau1", "B_double_tau2", "B_double_C1", "B_double_C2",
        "B_double_rms", "B_double_aic",
        "tau_phase", "P_phase", "phase_r2", "tau_phase_B", "tau_eff_5_25",
        "beta_avrami",
        "best_A", "best_B",
        "mono_frac_A", "mono_frac_B", "c30_over_noise", "rms_rel_A", "rms_rel_B",
        "c_frac_A", "c_frac_B"]


def settle_time(t, tot, t0, t1, dts):
    """窗口 B 的起点 = 沿末 + SKIP_B 秒（固定跳过瞬态）。

    曾尝试"沿后 |dv/dt| 最后一次超过沿峰值 5%"的自适应判据：实测不稳定
    （13 个事件给出 −2.8 ~ +20 s，其中 4 个顶到搜索上界），因为沿后数秒的导数
    会出现零星尖峰。改用固定偏移，语义明确：把"机械加载尾 + 最早段快相"整段跳过，
    留下的是"剩下的尾巴"，其 τ 估计对窗口起点极敏感（信噪比低），
    因此只作为**稳健性对照**，结论以窗口 A 与相图法为准。
    """
    return float(t1 + SKIP_B)


def shape_metrics(t, y, tref, W):
    """相对 tref 的实测蠕变在 2/5/10/20/30 s 的完成比例（归一化到 c(30)）。"""
    out = {}
    k0 = int(np.searchsorted(t, tref))
    ref = float(np.median(y[k0:k0 + 50]))
    kT = int(np.searchsorted(t, tref + W))
    cT = float(np.median(y[max(kT - 50, k0):kT + 1]) - ref)
    out["c30"] = round(cT, 2)
    for tt in (2, 5, 10, 20):
        if tt >= W:
            continue
        k = int(np.searchsorted(t, tref + tt))
        c = float(np.median(y[max(k - 30, k0):k + 1]) - ref)
        out["c%d_30" % tt] = round(c / cT, 4) if abs(cT) > 1e-9 else ""
    return out


def phase_tau(t, y, tref, W, noise):
    """相图法：dz/dt = P/τ − z/τ。返回 (τ, P, R²)。"""
    k0 = int(np.searchsorted(t, tref + 1.0))
    k1 = int(np.searchsorted(t, tref + W))
    if k1 - k0 < 200:
        return "", "", ""
    tw = t[k0:k1]
    zh = T.smooth_ma(y, 51)[k0:k1]
    dz = np.gradient(zh, tw)
    A = np.vstack([np.ones_like(zh), zh]).T
    coef, res, rank, sv = np.linalg.lstsq(A, dz, rcond=None)
    b, m = float(coef[0]), float(coef[1])
    if m >= -1e-9:
        return "", "", ""
    tau = -1.0 / m
    P = -b / m
    pred = A @ coef
    ss = float(np.sum((dz - pred) ** 2))
    st = float(np.sum((dz - dz.mean()) ** 2))
    r2 = 1.0 - ss / st if st > 0 else 0.0
    return round(tau, 3), round(P, 2), round(r2, 4)


def fit_one(t, y, tref, W, noise, tag, row):
    k0 = int(np.searchsorted(t, tref))
    k1 = int(np.searchsorted(t, tref + W))
    if k1 - k0 < 200:
        return
    tw, yw = t[k0:k1], y[k0:k1]
    fits = T.fit_shape(tw, yw, tref,
                       models=("fixed8", "single", "stretch", "double"))
    row["%s_fixed8_rms" % tag] = round(fits["fixed8"]["rms"], 2)
    f1 = fits["single"]
    row["%s_single_tau" % tag] = round(f1["popt"][2], 3)
    row["%s_single_rms" % tag] = round(f1["rms"], 2)
    row["%s_single_aic" % tag] = round(f1["aic"], 0)
    fs = fits["stretch"]
    row["%s_stretch_tau" % tag] = round(fs["popt"][2], 3)
    row["%s_stretch_beta" % tag] = round(fs["popt"][3], 3)
    row["%s_stretch_rms" % tag] = round(fs["rms"], 2)
    row["%s_stretch_aic" % tag] = round(fs["aic"], 0)
    fd = fits["double"]
    p = fd["popt"]
    row["%s_double_tau1" % tag] = round(p[2], 3)
    row["%s_double_tau2" % tag] = round(p[4], 2)
    row["%s_double_C1" % tag] = round(p[1], 1)
    row["%s_double_C2" % tag] = round(p[3], 1)
    row["%s_double_rms" % tag] = round(fd["rms"], 2)
    row["%s_double_aic" % tag] = round(fd["aic"], 0)
    # 最优形态（AIC 最小）
    aics = {k: fits[k]["aic"] for k in ("fixed8", "single", "stretch", "double")}
    row["best_%s" % tag] = min(aics, key=aics.get)
    # 相图法 + Avrami β（相图法只写窗口 A，窗口 B 另列，避免互相覆盖）
    tp, pp, r2 = phase_tau(t, y, tref, W, noise)
    if tag == "A":
        row["tau_phase"], row["P_phase"], row["phase_r2"] = tp, pp, r2
    else:
        row["tau_phase_B"] = tp
    u = t[k0:k1] - tref
    P = f1["popt"][0]
    C = f1["popt"][1]
    yh = T.smooth_ma(y, 51)[k0:k1]
    if np.isfinite(P):
        remain = P - yh
        dy = np.gradient(yh, t[k0:k1])
        ok = (u >= 5) & (u <= min(25.0, W - 2.0)) & (remain > 3 * noise) & (dy > 0.02)
        if np.sum(ok) > 50:
            row["tau_eff_5_25"] = round(float(np.median(remain[ok] / dy[ok])), 3)
    if C > 3 * noise:
        ref = float(np.median(y[int(np.searchsorted(t, tref)):][:50]))
        cc = (yh - ref) / C
        sel = (u >= 3) & (u <= W - 2) & (cc > 0.02) & (cc < 0.95)
        if np.sum(sel) > 50:
            X = np.log(np.maximum(u[sel], 1e-3))
            Y = np.log(-np.log(np.clip(1 - cc[sel], 1e-6, 1 - 1e-6)))
            row["beta_avrami"] = round(float(np.polyfit(X, Y, 1)[0]), 3)


def fit_shape_selftest():
    """自检：合成单指数 τ=8 s + 5 ADC 噪声，检验各估计量的偏差。"""
    dts = 0.01
    t = np.arange(0, 30.0, dts)
    rng = np.random.default_rng(7)
    lines = []
    for C in (100.0, 1000.0):
        y = 5000.0 - C * np.exp(-t / 8.0) + rng.normal(0, 5.0, len(t))
        fits = T.fit_shape(t, y, 0.0, models=("fixed8", "single", "stretch",
                                              "double"))
        tp, pp, r2 = phase_tau(t, y, 0.0, 30.0, 5.0)
        lines.append("  合成 τ=8 C=%.0f: single τ=%.2f β=%.2f (拉伸) "
                     "double(τ1=%.2f,τ2=%.0f) 相图 τ=%.2f R²=%.3f"
                     % (C, fits["single"]["popt"][2], fits["stretch"]["popt"][3],
                        fits["double"]["popt"][2], fits["double"]["popt"][4],
                        tp if tp != "" else float("nan"), r2 if r2 != "" else 0))
    return lines


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    roots = []
    if which in ("working", "both"):
        roots.append(("working", T.WORKING))
    if which in ("archived", "both"):
        roots.append(("archived", T.ARCHIVED))
    rows = []
    log = ["形态估计量自检（合成数据，用于确认估计无系统性偏差）："]
    log += fit_shape_selftest()
    for tag, root in roots:
        print("扫描 %s ..." % tag)
        for label, d in T.discover_sessions(root, 5):
            pre, _ = T.load_session(d)
            t, V = pre["t"], pre["V"]
            tot = T.total(pre)
            dts = pre["dt_mean"]
            steps, _ = T.detect_steps(t, tot)
            ups = [s for s in steps if s["sign"] > 0]
            gaps = t[pre["gap_idx"]] if len(pre["gap_idx"]) else np.zeros(0)
            rngc = V.max(axis=0) - V.min(axis=0)
            n_ev = 0
            for k, s in enumerate(steps):
                if s["sign"] <= 0:
                    continue
                nxt = steps[k + 1]["t0"] if k + 1 < len(steps) else t[-1]
                plateau = nxt - s["t1"]
                if s["dur"] > MAX_RAMP or plateau < MIN_PLATEAU:
                    continue
                W = float(min(WIN, plateau - 0.5))
                ts_set = settle_time(t, tot, s["t0"], s["t1"], dts)
                if not T.no_gap(t, s["t0"], s["t1"] + W, gaps):
                    continue
                n_ev += 1
                a = int(np.searchsorted(t, s["t0"]))
                b = int(np.searchsorted(t, s["t1"]))
                sigs = [(-1, tot)]
                for c in range(T.NCH):
                    if rngc[c] < MIN_CH_RANGE:
                        continue
                    lv_pre = float(np.median(V[max(0, a - 60):a + 1, c]))
                    lv_post = float(np.median(V[b:b + 60, c]))
                    if lv_post - lv_pre >= MIN_CH_STEP:
                        sigs.append((c, V[:, c]))
                for c, y in sigs:
                    noise = float(np.std(y[max(0, a - int(NOISE_WIN / dts)):
                                           max(1, a - int(0.3 / dts))]))
                    if not np.isfinite(noise) or noise <= 0:
                        noise = 0.5
                    r = {k2: "" for k2 in COLS}
                    r.update(ses=label, sig="tot" if c < 0 else "ch", ch=c,
                             t0=round(s["t0"], 3), t1=round(s["t1"], 3),
                             t_settle=round(ts_set, 3),
                             ramp=round(s["dur"], 3), plateau=round(plateau, 2),
                             noise=round(noise, 3),
                             lv_pre=round(float(np.median(y[max(0, a - 60):a + 1])), 2),
                             lv_post=round(float(np.median(y[b:b + 60])), 2),
                             dch=round(float(np.median(y[b:b + 60])
                                             - np.median(y[max(0, a - 60):a + 1])), 2),
                             rate=round(s["slope"], 1))
                    r["dch"] = round(r["lv_post"] - r["lv_pre"], 2)
                    mA = shape_metrics(t, y, s["t1"], W)
                    # 窗口 B 与 A 用同一个终点 t1+W（否则短保压事件会越过保压末）
                    mB = shape_metrics(t, y, ts_set, W - SKIP_B)
                    for kk, vv in mA.items():
                        r["%s_A" % kk] = vv
                    for kk, vv in mB.items():
                        r["%s_B" % kk] = vv
                    fit_one(t, y, s["t1"], W, noise, "A", r)
                    fit_one(t, y, ts_set, W - SKIP_B, noise, "B", r)
                    rows.append(r)
            log.append("%s  可用事件 %d" % (label, n_ev))
    os.makedirs(T.RESULTS, exist_ok=True)
    with open(os.path.join(T.RESULTS, "t1_shape_events.csv"), "w",
              encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    with open(os.path.join(T.RESULTS, "t1_shape_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print("written results/t1_shape_events.csv (%d 行)" % len(rows))
    for line in log:
        print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
