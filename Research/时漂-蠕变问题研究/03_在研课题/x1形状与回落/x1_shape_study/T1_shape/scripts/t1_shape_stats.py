# -*- coding: utf-8 -*-
"""步骤 4：形状统计与出图（T1 形状研究 A/B/C 部分）。

输入 results/t1_shape_events.csv（t1_shape.py 产出）
输出 results/t1_shape_stats.txt + figure/T1_*.png
用法： python t1_shape_stats.py
"""
import csv
import io
import os
import sys

import numpy as np
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import t1_lib as T  # noqa: E402
from t1_shape import SKIP_B  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_TAU8 = 1.0 - np.exp(-np.arange(0.0, 31.0, 0.1) / 8.0)


def load():
    with io.open(os.path.join(T.RESULTS, "t1_shape_events.csv"),
                 encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("ses", "sig", "best_A", "best_B"):
                continue
            try:
                r[k] = float(v) if v != "" else float("nan")
            except ValueError:
                r[k] = float("nan")
    return rows


def gate(rows, min_c_over_noise=4.0, max_rms_rel=0.5, key="A", min_dch=25.0):
    out = []
    for r in rows:
        if not (r["dch"] >= min_dch):
            continue
        c30 = r["c30_%s" % key]
        if not np.isfinite(c30) or abs(c30) < min_c_over_noise * r["noise"]:
            continue
        rms = r["%s_single_rms" % key]
        if np.isfinite(rms) and abs(c30) > 0 and rms > max_rms_rel * abs(c30):
            continue
        out.append(r)
    return out


def summ(w, name, a, fmt="%.2f"):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    if len(a) == 0:
        w("  %-34s 无有效样本" % name)
        return
    q = np.percentile(a, [5, 25, 50, 75, 95])
    w(("  %-34s n=%3d 中位 " + fmt + "  p5/p25/p75/p95 = "
       + " / ".join([fmt] * 4) + "  均值 " + fmt)
      % (name, len(a), q[2], q[0], q[1], q[3], q[4], float(a.mean())))


class _W(list):
    """既能 w.append(x) 也能 w(x) 的日志缓冲。"""

    def __call__(self, s):
        self.append(s)


def main():
    rows = load()
    w = _W()
    w.append("=" * 100)
    w.append("T1 形状统计：实测快相蠕变形状 vs 模型 x1 = r1·e·(1−exp(−t/8))")
    w.append("数据：%d 条事件-信号记录，其中 tot（21 通道和）%d 条、逐通道 %d 条"
             % (len(rows), sum(1 for r in rows if r["sig"] == "tot"),
                sum(1 for r in rows if r["sig"] == "ch")))
    tot = [r for r in rows if r["sig"] == "tot"]
    ch = [r for r in rows if r["sig"] == "ch"]

    w.append("")
    w.append("-" * 100)
    w.append("[1] 归一化形状（相对沿末 t1 的实测蠕变完成度；模型 τ=8 的对应值写在括号里）")
    w.append("    完成度 = c(t)/c(30)，c 为相对沿末电平的通道/总量爬升")
    for key in ("A", "B"):
        skip = 0.0 if key == "A" else SKIP_B
        w.append("  窗口 %s（%s）：" % (key, "自沿末 t1 起" if key == "A"
                                       else "自 t1+%.0f s（跳过瞬态）起，终点同 A" % skip))
        for tt, mval in ((2, 0.221 / 0.9765), (5, 0.4647 / 0.9765),
                         (10, 0.7135 / 0.9765), (20, 0.9179 / 0.9765)):
            # 终点固定在 t1+30，参考点后移 skip ⇒ 模型值随之改变（指数无记忆，
            # 但归一化区间缩短，需重算）
            T0 = 30.0 - skip
            mv = ((np.exp(-skip / 8.0) - np.exp(-(tt + skip) / 8.0))
                  / (np.exp(-skip / 8.0) - np.exp(-30.0 / 8.0)))
            k = "c%d_30_%s" % (tt, key)
            a = [r[k] for r in tot if np.isfinite(r.get(k, np.nan))]
            b = [r[k] for r in ch if np.isfinite(r.get(k, np.nan))]
            if a:
                w("    tot  c(%2ds)/c(%.0f) 中位 %.3f  [模型 %.3f]   实测>模型 %d/%d"
                  % (tt, T0, np.median(a), mv, int(np.sum(np.array(a) > mv)), len(a)))
            if b:
                w("    逐通道 c(%2ds)/c(%.0f) 中位 %.3f  [模型 %.3f]   实测>模型 %d/%d"
                  % (tt, T0, np.median(b), mv, int(np.sum(np.array(b) > mv)), len(b)))

    w.append("")
    w.append("-" * 100)
    w.append("[2] 拟合参数分布（窗口 A = 自沿末；窗口 B = 自 t_settle）")
    w.append("    样本门：dch ≥ 25 ADC、|c(30)| ≥ 4×噪声、单指数残差 RMS ≤ 0.5·|c(30)|")
    for tag, grp in (("tot", tot), ("逐通道", ch)):
        w.append("  -- %s --" % tag)
        for key in ("A", "B"):
            g = gate(grp, key=key)
            w.append("   窗口%s 通过门的样本 n=%d / %d"
                     % (key, len(g), len(grp)))
            summ(w, "%s 单指数 τ (s)" % key, [r["%s_single_tau" % key] for r in g])
            summ(w, "%s 拉伸指数 β" % key, [r["%s_stretch_beta" % key] for r in g])
            summ(w, "%s 拉伸指数 τ (s)" % key, [r["%s_stretch_tau" % key] for r in g])
            summ(w, "%s 双指数 τ1 (s)" % key, [r["%s_double_tau1" % key] for r in g])
            summ(w, "%s 双指数 τ2 (s)" % key, [r["%s_double_tau2" % key] for r in g])
            summ(w, "%s 相图法 τ_phase (s)" % key,
                 [r["tau_phase" if key == "A" else "tau_phase_B"] for r in g])
            summ(w, "%s 双指数快分量占比 C1/(C1+C2)" % key,
                 [r["%s_double_C1" % key] / (r["%s_double_C1" % key]
                                             + r["%s_double_C2" % key])
                  if (r["%s_double_C1" % key] + r["%s_double_C2" % key]) > 0
                  else np.nan for r in g])

    w.append("")
    w.append("-" * 100)
    w.append("[3] 形态选择（AIC 最小者胜；残差为窗口内 RMS，单位 ADC）")
    for tag, grp in (("tot", tot), ("逐通道", ch)):
        for key in ("A", "B"):
            cnt = {}
            for r in grp:
                b = r["best_%s" % key]
                cnt[b] = cnt.get(b, 0) + 1
            w("  %s 窗口%s：%s" % (tag, key,
                                   "  ".join("%s=%d" % kv for kv in sorted(cnt.items()))))
            for m in ("fixed8", "single", "stretch", "double"):
                c = [r["%s_%s_rms" % (key, m)] for r in grp
                     if np.isfinite(r["%s_%s_rms" % (key, m)])]
                if c:
                    w("      %-8s RMS 中位 %6.2f ADC" % (m, float(np.median(c))))
    # AIC 差值：相对 single
    w.append("  ΔAIC（负 = 比单指数更好）中位：")
    for tag, grp in (("tot", tot), ("逐通道", ch)):
        for m in ("fixed8", "stretch", "double"):
            k = "A_%s_aic" % m
            d = [r[k] - r["A_single_aic"] for r in grp
                 if np.isfinite(r.get(k, np.nan)) and np.isfinite(r["A_single_aic"])]
            if d:
                w("      %s %-8s ΔAIC 中位 %+8.1f  更好 %d/%d"
                  % (tag, m, float(np.median(d)), int(np.sum(np.array(d) < 0)), len(d)))

    w.append("")
    w.append("-" * 100)
    w.append("[4] 形状参数与台阶幅度/加载速率/通道位置的相关性（Spearman ρ, p）")
    for tag, grp in (("tot", tot), ("逐通道", ch)):
        w.append("  -- %s --" % tag)
        for pk in ("A_single_tau", "A_stretch_beta", "tau_phase"):
            for xk in ("dch", "rate", "ch", "ramp", "c_frac30", "t_settle",
                       "c30_A", "noise"):
                x = np.array([r[xk] for r in grp if np.isfinite(r.get(xk, np.nan))])
                y = np.array([r[pk] for r in grp if np.isfinite(r.get(xk, np.nan))])
                m = np.isfinite(x) & np.isfinite(y)
                if np.sum(m) < 6:
                    continue
                rho, p = sps.spearmanr(x[m], y[m])
                star = " *" if p < 0.05 else ""
                w("    ρ(%s, %s) = %+.3f  p=%.3f  n=%d%s"
                  % (pk, xk, rho, p, int(np.sum(m)), star))

    w.append("")
    w.append("-" * 100)
    w.append("[5] 与固定 τ=8 的偏差（用拟合的 P、C 折算成 ADC）")
    w.append("    Δ(t) = 实测蠕变(t) − C·(1−exp(−t/8))，正 = 实测爬得更快（模型滞后）")
    for tag, grp in (("tot", tot), ("逐通道", ch)):
        for tt in (2, 5, 10, 20):
            k = "c%d_30_A" % tt
            C = "A_single_C" if False else None
            vals = []
            for r in grp:
                # 用窗口 A 的实测完成度 × 单指数拟合的 C 作为实测幅度
                c30 = r["c30_A"]
                f = r.get(k, np.nan)
                if not np.isfinite(f) or not np.isfinite(c30):
                    continue
                model = (1 - np.exp(-tt / 8.0)) / (1 - np.exp(-30.0 / 8.0))
                vals.append((f - model) * abs(c30))
            if vals:
                v = np.array(vals)
                w("  %s t=%2ds  Δ 中位 %+7.1f ADC  |Δ| 中位 %6.1f  p5/p95 %+7.1f/%+7.1f  n=%d"
                  % (tag, tt, float(np.median(v)), float(np.median(np.abs(v))),
                     float(np.percentile(v, 5)), float(np.percentile(v, 95)), len(v)))

    w.append("")
    w.append("-" * 100)
    w.append("[6] 实测蠕变幅度（相对台阶的比例）vs 模型幅度先验 r1=0.12 / r2max=0.35")
    w.append("    c30 = 沿后 30 s 内实测蠕变；比例 = c30 / 通道自身台阶")
    for tag, grp in (("tot", tot), ("逐通道", ch)):
        g = gate(grp, key="A", min_dch=25.0)
        fr = np.array([r["c30_A"] / r["dch"] for r in g
                       if np.isfinite(r["c30_A"]) and r["dch"] != 0])
        summ(w, "%s 30 s 蠕变/台阶" % tag, fr, fmt="%.4f")
        if len(fr):
            w("      %.0f%% 的样本 30 s 蠕变比例 < 0.12（模型 r1）"
              % (100 * float(np.mean(fr < 0.12))))
            w("      比例 > 0.35（模型 r2max）的占比 %.0f%%"
              % (100 * float(np.mean(fr > 0.35))))
        # 与台阶幅度的关系
        d = np.array([r["dch"] for r in g])
        m = np.isfinite(d) & np.isfinite(fr) & (d > 0)
        if np.sum(m) >= 8:
            rho, p = sps.spearmanr(d[m], fr[m])
            w("      ρ(台阶幅度, 蠕变比例) = %+.3f  p=%.3f  n=%d%s"
              % (rho, p, int(np.sum(m)), " *" if p < 0.05 else ""))
        # 分组
        for lo, hi in ((25, 100), (100, 300), (300, 1000), (1000, 1e9)):
            mm = m & (d >= lo) & (d < hi)
            if np.sum(mm) >= 5:
                c30 = np.array([r["c30_A"] for r in g])
                w("      台阶 %5.0f~%-6.0f ADC: n=%3d  蠕变比例中位 %.3f  "
                  "(c30 中位 %.0f ADC)"
                  % (lo, min(hi, 99999), int(np.sum(mm)),
                     float(np.median(fr[mm])), float(np.median(c30[mm]))))

    w.append("")
    w.append("-" * 100)
    w.append("[7] 形状参数按台阶幅度分组（Mann-Whitney U 检验，检验 形状绑定幅度）")
    for tag, grp in (("tot", tot), ("逐通道", ch)):
        g = gate(grp, key="A")
        if not g:
            continue
        d = np.array([r["dch"] for r in g])
        for pk in ("A_single_tau", "A_stretch_beta", "tau_phase"):
            y = np.array([r[pk] for r in g])
            m = np.isfinite(d) & np.isfinite(y)
            if np.sum(m) < 10:
                continue
            lo = y[m & (d < 300)]
            hi = y[m & (d >= 300)]
            if len(lo) < 4 or len(hi) < 4:
                continue
            u, p = sps.mannwhitneyu(lo, hi)
            w.append("  %s %-16s 小台阶(<300ADC) n=%d 中位 %.2f | 大台阶(≥300) n=%d "
                     "中位 %.2f | p=%.3f%s"
                     % (tag, pk, len(lo), float(np.median(lo)), len(hi),
                        float(np.median(hi)), p, " *" if p < 0.05 else ""))

    txt = "\n".join(w)
    with open(os.path.join(T.RESULTS, "t1_shape_stats.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt + "\n")
    plt.close("all")
    make_figures(rows, tot, ch)
    sys.stdout.write("written results/t1_shape_stats.txt + figures\n")


# ---------------------------------------------------------------- 图

def find_dir(label):
    for root in (T.WORKING, T.ARCHIVED):
        for lb, d in T.discover_sessions(root, 5):
            if lb == label:
                return d
    return None


def raw_curves(rows, grid=None, max_curves=60, min_snr=4.0):
    """重新装载事件窗口，返回 [(t_rel, c_norm, 标签)] 实测归一化曲线族。

    归一化：c(t) = 平滑 y(t) − y(沿末 0.5 s 中位)；再除以 c(30)。
    """
    if grid is None:
        grid = np.concatenate([[0.5], np.arange(1.0, 30.01, 0.5)])
    cache = {}
    out = []
    for r in rows:
        if len(out) >= max_curves:
            break
        if not np.isfinite(r["c30_A"]) or r["c30_A"] <= 0:
            continue
        if r["c30_A"] < min_snr * r["noise"]:
            continue
        lab = r["ses"]
        if lab not in cache:
            d = find_dir(lab)
            cache[lab] = T.load_session(d)[0] if d else None
        pre = cache[lab]
        if pre is None:
            continue
        t, V = pre["t"], pre["V"]
        y = T.total(pre) if r["sig"] == "tot" else V[:, int(r["ch"])]
        y = T.smooth_ma(y, 51)
        k0 = int(np.searchsorted(t, r["t1"]))
        ref = float(np.median(y[k0:k0 + 50]))
        kk = int(np.searchsorted(t, r["t1"] + min(30.0, r["plateau"] - 0.5)))
        cT = float(np.median(y[max(kk - 50, k0):kk + 1]) - ref)
        if not np.isfinite(cT) or abs(cT) < 1e-9:
            continue
        tt = np.array([t[min(int(np.searchsorted(t, r["t1"] + g)), len(t) - 1)]
                       for g in grid]) - r["t1"]
        cc = np.array([y[min(int(np.searchsorted(t, r["t1"] + g)), len(t) - 1)]
                       for g in grid]) - ref
        out.append((tt, cc / cT, "%s %s" % (lab.split("/")[-1][-6:],
                                            "tot" if r["sig"] == "tot" else "ch%d" % r["ch"])))
    return out


def make_figures(rows, tot, ch):
    os.makedirs(T.FIGURE, exist_ok=True)
    ref_win = 50
    # ---- 图 1：实测归一化形状族 vs 1−exp(−t/8)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    u = np.arange(0.0, 30.01, 0.05)
    ref = (1 - np.exp(-u / 8.0))
    ref = ref / (1 - np.exp(-30.0 / 8.0))
    ax = axes[0]
    curves = raw_curves(ch)
    for tt, cc, _ in curves:
        ax.plot(tt, cc, color="#1f77b4", alpha=0.28, lw=0.9)
    ax.plot(u, ref, "k-", lw=3.0, label="模型 x1 形状：1−exp(−t/8)")
    if curves:
        grid = curves[0][0]
        med = np.nanmedian(np.vstack([c[1] for c in curves
                                      if len(c[1]) == len(grid)]), axis=0)
        ax.plot(grid, med, "o-", color="#d62728", lw=2.4, ms=5,
                label="逐通道实测中位（n=%d）" % len(curves))
    tc = raw_curves(tot, max_curves=40)
    if tc:
        grid = tc[0][0]
        medt = np.nanmedian(np.vstack([c[1] for c in tc
                                       if len(c[1]) == len(grid)]), axis=0)
        ax.plot(grid, medt, "s--", color="#2ca02c", lw=2.2, ms=6,
                label="通道总和实测中位（n=%d）" % len(tc))
    ax.set_xlabel("沿末之后时间 t (s)")
    ax.set_ylabel("归一化蠕变完成度 c(t)/c(30)")
    ax.set_title("图1a 实测快相归一化形状族 vs 模型 1−exp(−t/8)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.15, 1.3)
    ax.set_xlim(0, 30)

    ax = axes[1]
    taus = np.array([r["A_single_tau"] for r in ch if np.isfinite(r["A_single_tau"])])
    taus_b = np.array([r["B_single_tau"] for r in ch if np.isfinite(r["B_single_tau"])])
    tp = np.array([r["tau_phase"] for r in ch if np.isfinite(r["tau_phase"])])
    bins = np.logspace(np.log10(0.5), np.log10(60), 34)
    ax.hist(taus, bins=bins, alpha=0.55, color="#1f77b4",
            label="单指数 τ（窗口A, n=%d）" % len(taus))
    ax.hist(taus_b, bins=bins, alpha=0.45, color="#2ca02c",
            label="单指数 τ（窗口B, n=%d）" % len(taus_b))
    ax.hist(tp, bins=bins, alpha=0.45, color="#ff7f0e",
            label="相图法 τ_phase（n=%d）" % len(tp))
    ax.axvline(8.0, color="k", ls="--", lw=2, label="模型固定 τc1 = 8 s")
    ax.set_xscale("log")
    ax.set_xlabel("拟合时间常数 τ (s)")
    ax.set_ylabel("事件-通道数")
    ax.set_title("图1b 实测 τ 分布 vs 模型固定 8 s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(T.FIGURE, "T1_形状叠画与τ分布.png"), dpi=120)
    plt.close(fig)

    # ---- 图 2：拉伸指数 β、双指数结构、AIC
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ax = axes[0]
    bet = np.array([r["A_stretch_beta"] for r in ch if np.isfinite(r["A_stretch_beta"])])
    ba = np.array([r["beta_avrami"] for r in ch if np.isfinite(r["beta_avrami"])])
    ax.hist(bet, bins=np.arange(0.0, 3.05, 0.15), alpha=0.6, color="#1f77b4",
            label="拉伸指数拟合 β（n=%d）" % len(bet))
    ax.hist(ba, bins=np.arange(0.0, 3.05, 0.15), alpha=0.45, color="#9467bd",
            label="Avrami 图斜率 β（n=%d）" % len(ba))
    ax.axvline(1.0, color="k", ls="--", lw=2, label="纯单指数 β = 1")
    ax.set_xlabel("拉伸指数 β")
    ax.set_ylabel("事件-通道数")
    ax.set_title("图2a 拉伸指数 β：β<1 慢尾、β>1 快起")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    tt1, tt2, share = [], [], []
    for r in ch:
        a, b = r["A_double_C1"], r["A_double_C2"]
        if not (np.isfinite(r["A_double_tau1"]) and np.isfinite(r["A_double_tau2"])
                and np.isfinite(a) and np.isfinite(b) and (a + b) > 0):
            continue
        tt1.append(r["A_double_tau1"])
        tt2.append(r["A_double_tau2"])
        share.append(a / (a + b))
    t1 = np.asarray(tt1)
    t2 = np.asarray(tt2)
    share = np.asarray(share)
    sc = ax.scatter(t1, np.clip(t2, 1, 5000), s=26, alpha=0.8, c=share,
                    cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label="快分量占比 C1/(C1+C2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(8, color="k", ls="--", lw=1.5)
    ax.axhline(8, color="k", ls="--", lw=1.5)
    ax.set_xlabel("双指数快分量 τ1 (s)")
    ax.set_ylabel("双指数慢分量 τ2 (s)")
    ax.set_title("图2b 双指数分解（颜色 = 快分量占比 C1/(C1+C2)）")
    ax.grid(alpha=0.3)

    ax = axes[2]
    names = ["fixed8", "single", "stretch", "double"]
    rms = [[r["A_%s_rms" % m] for r in ch if np.isfinite(r["A_%s_rms" % m])]
           for m in names]
    bp = ax.boxplot(rms, tick_labels=names, showfliers=False, patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor("#cfe2f3")
    ax.set_ylabel("窗口内残差 RMS (ADC)")
    ax.set_title("图2c 四种形态的拟合残差（逐通道，窗口A）")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(T.FIGURE, "T1_形态族与残差.png"), dpi=120)
    plt.close(fig)

    # ---- 图 3：形状参数 vs 台阶幅度
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    d = np.array([r["dch"] for r in ch])
    ax = axes[0]
    for key, col in (("A_single_tau", "#1f77b4"), ("tau_phase", "#ff7f0e")):
        y = np.array([r[key] for r in ch])
        m = np.isfinite(d) & np.isfinite(y) & (d > 0)
        ax.scatter(d[m], y[m], s=24, alpha=0.6, color=col, label=key)
    ax.axhline(8, color="k", ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("通道自身加载台阶 Δ (ADC)")
    ax.set_ylabel("观测时间常数 τ (s)")
    ax.set_title("图3a τ vs 台阶幅度")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    y = np.array([r["A_stretch_beta"] for r in ch])
    m = np.isfinite(d) & np.isfinite(y)
    ax.scatter(d[m], y[m], s=24, alpha=0.65, color="#9467bd")
    ax.axhline(1.0, color="k", ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("通道自身加载台阶 Δ (ADC)")
    ax.set_ylabel("拉伸指数 β")
    ax.set_title("图3b β vs 台阶幅度")
    ax.grid(alpha=0.3)

    ax = axes[2]
    fr = np.array([r["c_frac30"] if np.isfinite(r.get("c_frac30", np.nan)) else np.nan
                   for r in ch])
    # c_frac30 在旧版列里；新版用 c30/dch
    fr = np.array([r["c30_A"] / r["dch"] if r["dch"] else np.nan for r in ch])
    m = np.isfinite(d) & np.isfinite(fr) & (d > 0) & (fr > 0)
    ax.scatter(d[m], fr[m] * 100, s=24, alpha=0.65, color="#2ca02c")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("通道自身加载台阶 Δ (ADC)")
    ax.set_ylabel("30 s 内蠕变占台阶比例 (%)")
    ax.set_title("图3c 蠕变幅度占比 vs 台阶幅度")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(T.FIGURE, "T1_形状参数相关性.png"), dpi=120)
    plt.close(fig)
    print("figures -> %s" % T.FIGURE)


if __name__ == "__main__":
    main()
