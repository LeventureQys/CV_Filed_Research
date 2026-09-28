# -*- coding: utf-8 -*-
"""T5A-Q4 / Q5：滤波能不能救过充 —— 因果滤波候选 × 参数 的 Pareto（**扩展第一轮，不重做**）。

与第一轮 `13-v6-assessment/results/filter_pareto.csv` 的关系（**扩展而非重做**）：
  · 第一轮只有 **一种**滤波器（输出端因果 EMA）× 7 档 τ ∈ {0, 0.1, 0.3, 0.5, 1, 2, 3} s
    × 13 份录制，指标只有 `T_stable / over_pct / err_1s_pct` 三列。
  · 本脚本保留其全部档位（`f_iir_ema` 行，`tau` 直接可比），并**新增**：
      - 4 种滤波器族：因果滑动中值 / 一阶 IIR（EMA）/ 滑动均值 / 限幅+速率限制
      - 非因果对照（标 "需 r 帧延迟"）：居中中值、Savitzky-Golay
      - **两种落点**：显示层（v6 输出之后）/ 内部状态层（进补偿器之前）
      - 两个新指标：`G`（台阶捕获比，Q5 的"变钝"代价）与 `MD`（最大偏差）
  · 口径与本任务 Q1/Q2 完全同源（同一 `metrics_at`、同一 T4-A 事件集）。

因果性声明：`causal=True` 的滤波器只用 t 及以前的样本，可在线实现；
`causal=False` 的行是**离线对照**：上线需要 r 帧延迟（r = 半窗），已在 `extra_delay_frames` 列标明。

产出：`results/t5a_filter_pareto.csv`（主表）、`t5a_filter_cost.csv`（Q5 代价表）、
      `t5a_filter_vs_firstround.csv`（与第一轮逐档对表）、`_t5a_filter.log`。

用法：`python scripts/t5a_filter.py`（长任务，建议后台）
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402

LOG = []
T0 = time.time()
FS = 100.0


def rec(m):
    s = f"[{time.time() - T0:7.1f}s] {m}"
    print(s, flush=True)
    LOG.append(s)


# ───────────────────────── 因果滤波器实现 ─────────────────────────

def _causal_median(x, win):
    w = max(1, int(round(win * FS)))
    if w <= 1:
        return x.copy()
    return pd.Series(x).rolling(w, min_periods=1).median().to_numpy()


def _iir_ema(x, tau):
    if tau <= 0:
        return x.copy()
    a = 1.0 - np.exp(-1.0 / (tau * FS))
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def _causal_mean(x, win):
    w = max(1, int(round(win * FS)))
    if w <= 1:
        return x.copy()
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def _rate_limit(x, frac_per_s, amp):
    """限幅 + 速率限制：|y(t)−y(t−1)| ≤ frac_per_s·amp·dt。amp 由调用方按事件幅度给；
    这里给一个**与事件无关**的因果实现：amp 取该录制的总量动态范围的 1/2（在线可由 max_tot 给）。"""
    lim = frac_per_s * max(amp, 1e-9) / FS
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        d = x[i] - y[i - 1]
        y[i] = y[i - 1] + (lim if d > lim else (-lim if d < -lim else d))
    return y


def _clip_raw(x, raw, frac):
    """过充硬限幅（有界跟随）：显示不得超过「原始 + frac·|事件幅度|」。
    在线可实现（只需当前原始帧 + 一个幅度估计）。"""
    cap = raw + frac * np.maximum(np.abs(raw), 1e-9)
    return np.minimum(x, cap)


FILTERS = []


def _reg(name, family, fn, causal=True, delay_frames=0, note=""):
    FILTERS.append(dict(fid=name, family=family, fn=fn, causal=causal,
                        extra_delay_frames=delay_frames, note=note))


def _mk_med(w):
    def f(x, r, a, _w=w):
        return _causal_median(x, _w)
    return f


def _mk_mean(w):
    def f(x, r, a, _w=w):
        return _causal_mean(x, _w)
    return f


def _mk_ema(t):
    def f(x, r, a, _t=t):
        return _iir_ema(x, _t)
    return f


def _mk_rate(ff):
    def f(x, r, a, _f=ff):
        return _rate_limit(x, _f, a)
    return f


def _mk_clip(ff):
    def f(x, r, a, _f=ff):
        return _clip_raw(x, r, _f)
    return f


def _mk_medc(w):
    def f(x, r, a, _w=w):
        return pd.Series(x).rolling(max(1, int(round(_w * FS))), center=True,
                                   min_periods=1).median().to_numpy()
    return f


def _mk_sg(w):
    def f(x, r, a, _w=w):
        return savgol_filter(x, _w, 3, mode="interp")
    return f


# 与第一轮 filter_pareto.csv 完全对齐的 7 档（waveform 一族）
for tau in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
    _reg(f"iir_ema_tau{tau:g}", "iir_ema", _mk_ema(tau),
         note="第一轮 filter_pareto.csv 的同一族")
# 因果滑动中值
for win in [0.03, 0.05, 0.10, 0.20, 0.30, 0.50]:
    _reg(f"med_w{win:g}", "causal_median", _mk_med(win), note="因果，延迟 ≈ w/2")
# 因果滑动均值
for win in [0.05, 0.10, 0.20, 0.50]:
    _reg(f"mean_w{win:g}", "causal_mean", _mk_mean(win))
# 限幅 + 速率限制（对总量，相对动态范围）
for f in [0.05, 0.10, 0.20, 0.40]:
    _reg(f"ratelimit_{f:g}", "rate_limit", _mk_rate(f),
         note="速率上限 = frac·(记录总量动态范围)/s")
# 过充硬限幅（有界跟随，相对当前原始）
for f in [0.02, 0.05, 0.10]:
    _reg(f"clip_{f:g}", "clip_raw", _mk_clip(f), note="显示 ≤ 原始 + frac·|原始|")
# 非因果对照（需未来样本）
for win in [0.05, 0.10, 0.20, 0.50]:
    _reg(f"medC_w{win:g}", "centered_median", _mk_medc(win),
         causal=False, delay_frames=int(round(win * FS / 2)),
         note="居中中值，非因果；上线需 r 帧延迟")
for win in [11, 21, 31]:
    _reg(f"savgol_w{win}", "savgol", _mk_sg(win),
         causal=False, delay_frames=int(win // 2),
         note="Savitzky-Golay（3 阶），非因果；上线需 r 帧延迟")


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    load = ev[ev.kind.isin(["onset", "restep"])]
    rec(f"事件 {len(ev)}（加载类 {len(load)}）；滤波器族 {len(FILTERS)} 档")

    # ── 预跑基线 v6（κ 现状 1.30/1.12），缓存 Y 与 Z ──
    base = {}
    for k, d in recs.items():
        out = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12)
        base[k] = dict(Y=out["Y"], Z=d["Z"], tu=d["tu"], span=d["span"])
    rec(f"基线 v6（κ 1.30/1.12）跑完 {len(base)} 份")

    rows = []
    # ── 显示层滤波 ──
    for f in FILTERS:
        for k, d in recs.items():
            sub = load[load.key == k]
            if not len(sub):
                continue
            b = base[k]
            Zy = b["Y"].sum(axis=1)
            rng_amp = float(np.percentile(b["Z"], 99) - np.percentile(b["Z"], 1))
            yf = f["fn"](Zy, b["Z"], rng_amp)
            for _, e in sub.iterrows():
                m = C.metrics_at(b["Z"], yf, b["tu"], float(e["t_on"]), b["span"])
                m.update(key=k, t_on=float(e["t_on"]), kind=e["kind"], dom=d["dom"],
                         fid=f["fid"], family=f["family"], layer="display",
                         causal=f["causal"], extra_delay_frames=f["extra_delay_frames"])
                rows.append(m)
        rec(f"  [display] {f['fid']}")
    disp = pd.DataFrame(rows)

    # ── 内部状态层滤波（输入进补偿器之前），只做因果族与前 3 档非因果对照 ──
    inp_filters = [f for f in FILTERS if f["causal"]]
    rows2 = []
    for f in inp_filters:
        for k, d in recs.items():
            sub = load[load.key == k]
            if not len(sub):
                continue
            X = d["Xu"]
            rng_amp = float(np.percentile(d["Z"], 99) - np.percentile(d["Z"], 1))
            Xf = np.empty_like(X)
            for c in range(X.shape[1]):
                Xf[:, c] = f["fn"](X[:, c], X[:, c], rng_amp)
            out = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12, X=Xf)
            yf = out["Y"].sum(axis=1)
            Zf = Xf.sum(axis=1)
            for _, e in sub.iterrows():
                m = C.metrics_at(Zf, yf, d["tu"], float(e["t_on"]), d["span"])
                m.update(key=k, t_on=float(e["t_on"]), kind=e["kind"], dom=d["dom"],
                         fid=f["fid"], family=f["family"], layer="internal_input",
                         causal=True, extra_delay_frames=0)
                rows2.append(m)
        rec(f"  [internal] {f['fid']}")
    intl = pd.DataFrame(rows2)

    allf = pd.concat([disp, intl], ignore_index=True)
    allf["t0pf"] = allf["t0"] if "t0" in allf else allf["t_on"]
    allf.to_csv(os.path.join(C.TASK, "results", "t5a_filter_perevent.csv"),
                index=False, encoding="utf-8-sig", float_format="%.6g")

    # ── Pareto 汇总 ──
    sm = []
    for (layer, fid, fam, causal, delay), g in allf.groupby(
            ["layer", "fid", "family", "causal", "extra_delay_frames"]):
        on = g[g.kind == "onset"]
        sm.append(dict(
            layer=layer, fid=fid, family=fam, causal=causal, extra_delay_frames=delay,
            n=len(g), n_onset=len(on),
            OS_med=g.OS_pct.median(), OS_p90=g.OS_pct.quantile(.90), OS_max=g.OS_pct.max(),
            OS_med_onset=on.OS_pct.median(), OS_max_onset=on.OS_pct.max(),
            OS_abs_med=g.OS_pct.abs().median(),
            T_med=g.T_stable.median(), T_p90=g.T_stable.quantile(.90),
            T_med_onset=on.T_stable.median(),
            err1_med=g.err_1s_pct.median(), err1_abs_med=g.err_1s_pct.abs().median(),
            G_med=g.G20.median(), G_min=g.G20.min(),
            MD_med=g.MD.median(), MD_p90=g.MD.quantile(.90),
            US_med=g.US_pct.median(), US_max=g.US_pct.max(),
        ))
    S = pd.DataFrame(sm)
    # 与基线（无滤波 = iir_ema_tau0）的配对差
    b0 = allf[(allf.layer == "display") & (allf.fid == "iir_ema_tau0")].set_index(
        ["key", "t_on", "kind"])
    d0 = []
    for (layer, fid), g in allf.groupby(["layer", "fid"]):
        gi = g.set_index(["key", "t_on", "kind"])
        common = gi.index.intersection(b0.index)
        d0.append(dict(layer=layer, fid=fid, n_pair=len(common),
                       dOS_med=(gi.loc[common, "OS_pct"] - b0.loc[common, "OS_pct"]).median(),
                       dT_med=(gi.loc[common, "T_stable"] - b0.loc[common, "T_stable"]).median(),
                       dG_med=(gi.loc[common, "G20"] - b0.loc[common, "G20"]).median(),
                       dE_abs_med=(gi.loc[common, "err_1s_pct"].abs()
                                   - b0.loc[common, "err_1s_pct"].abs()).median()))
    S = S.merge(pd.DataFrame(d0), on=["layer", "fid"], how="left")

    # Pareto 前沿（最小化 OS_abs_med / T_med / (1−G_med)）
    def is_pareto(df):
        v = df[["OS_abs_med", "T_med", "G_pen"]].to_numpy(float)
        keep = np.ones(len(df), bool)
        for i in range(len(df)):
            if not np.all(np.isfinite(v[i])):
                keep[i] = False
                continue
            for j in range(len(df)):
                if i == j or not np.all(np.isfinite(v[j])):
                    continue
                if (v[j] <= v[i]).all() and (v[j] < v[i]).any():
                    keep[i] = False
                    break
        return keep
    S["G_pen"] = (1.0 - S["G_med"]).abs()
    parts = []
    for layer, g in S.groupby("layer"):
        g = g.copy()
        g["pareto"] = is_pareto(g)
        parts.append(g)
    S = pd.concat(parts, ignore_index=True)
    S.to_csv(os.path.join(C.TASK, "results", "t5a_filter_pareto.csv"),
             index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_filter_pareto.csv")

    # ── Q5 代价表 ──
    cost = []
    for _, r in S.iterrows():
        cost.append(dict(
            layer=r["layer"], fid=r["fid"], family=r["family"], causal=r["causal"],
            extra_delay_frames=int(r["extra_delay_frames"]),
            OS_med=r["OS_med"], OS_max=r["OS_max"], T_med=r["T_med"],
            err1_abs_med=r["err1_abs_med"], G_med=r["G_med"], G_min=r["G_min"],
            MD_med=r["MD_med"],
            dOS_vs_base=r["dOS_med"], dT_vs_base=r["dT_med"], dG_vs_base=r["dG_med"],
            err1_penalty=(r["err1_abs_med"] - float(
                S[(S.layer == "display") & (S.fid == "iir_ema_tau0")]["err1_abs_med"].iloc[0])),
            step_edge_smear_frames=int(r["extra_delay_frames"]) if not r["causal"] else 0,
            pareto=bool(r["pareto"]),
            note=("非因果：上线需 r 帧延迟" if not r["causal"] else "因果可上线"),
        ))
    CD = pd.DataFrame(cost).sort_values(["layer", "OS_med"])
    CD.to_csv(os.path.join(C.TASK, "results", "t5a_filter_cost.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_filter_cost.csv")

    # ── 与第一轮 filter_pareto.csv 对表 ──
    p = os.path.join(C.PROG, "13-v6-assessment", "results", "filter_pareto.csv")
    lines = ["\n===== 与第一轮 filter_pareto.csv 的对表 ====="]
    if os.path.exists(p):
        fr = pd.read_csv(p)
        rows = []
        for tau in sorted(fr.tau.unique()):
            fid = f"iir_ema_tau{tau:g}"
            mine = S[(S.layer == "display") & (S.fid == fid)]
            sub = fr[fr.tau == tau]
            if not len(mine):
                continue
            rows.append(dict(tau=tau, n_rec_firstround=len(sub),
                             T_med_firstround=sub.T_stable.median(),
                             T_med_mine=mine.T_med.iloc[0],
                             OS_med_firstround=sub.over_pct.median(),
                             OS_med_mine=mine.OS_med.iloc[0],
                             err1_med_firstround=sub.err_1s_pct.median(),
                             err1_med_mine=mine.err1_med.iloc[0]))
        fr2 = pd.DataFrame(rows)
        fr2.to_csv(os.path.join(C.TASK, "results", "t5a_filter_vs_firstround.csv"),
                   index=False, encoding="utf-8-sig", float_format="%.6g")
        lines.append(fr2.to_string(index=False))
        lines.append("注：第一轮只对**恒载 9 组**给 T_stable，实录行 T_stable 为空；"
                     "本表的 T_med 是加载类 41 事件（含实录），故两者的 T_stable 不可直接横比。")
    txt = "\n".join(lines)
    rec(txt)

    # ── 摘要打印 ──
    lines = ["\n===== 滤波 Pareto 摘要（加载类 n=41）====="]
    for layer in ["display", "internal_input"]:
        s = S[(S.layer == layer)].sort_values("OS_med")
        lines.append(f"--- {layer} ---")
        lines.append(f"{'fid':20s} {'因果':>4s} {'OS_med':>8s} {'OS_max':>8s} "
                     f"{'T_med':>7s} {'err1|med|':>10s} {'G_med':>6s} {'G_min':>6s} "
                     f"{'MD_med':>9s} {'Pareto':>7s}")
        for _, r in s.iterrows():
            lines.append(f"{r['fid']:20s} {str(r['causal']):>4s} {r['OS_med']:8.3f} "
                         f"{r['OS_max']:8.3f} {r['T_med']:7.3f} {r['err1_abs_med']:10.3f} "
                         f"{r['G_med']:6.3f} {r['G_min']:6.3f} {r['MD_med']:9.1f} "
                         f"{str(bool(r['pareto'])):>7s}")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_filter.log"),
                "python scripts/t5a_filter.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
