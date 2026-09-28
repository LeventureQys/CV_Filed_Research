# -*- coding: utf-8 -*-
"""t3b_common.py —— T3-B 公共层 = `T4_*/scripts/t4b_common.py` 的副本（逐行相同，
只把 `import t4b_ad_lib` 改为 `import t3b_ad_lib`；`HERE/TASK/PLAN/ROOT` 由 `__file__` 推导，
自动落到本任务目录，未改任何数值口径）。

原文件说明：T4-B 公共层：数据清单 / 因果事件提取 / 因果判据特征 / T1 口径指标。

设计原则（对应需求文档 §3）：
  * **只读**：本文件不写任何 temp/ 下既有文件；
  * **时间轴**：`timestamp` 列 → 100 Hz 均匀网格（`np.interp`）；
  * **禁止 τ=2 s 平滑**：本文件所有平滑 k ≤ 0.5 s；
  * **因果性**：所有判据特征只允许用 t ≤ t_on + Δ 的帧（见 `FEATS` 的 `lag` 列）；
    内部用「因果中值平滑」（只回溯过去帧）计算，避免居中窗的隐性未来信息。

事件真值口径（对齐 00_共享/指标字典与口径.md §2.1，与 r2_shapes.py 一致）：
  t_on  = 人工可复核的"真沿"：在候选帧 ±0.6 s 内取**单帧最大跳变**所在帧；
  pre   = 中值因果平滑序列在 [t_on−0.5, t_on) 的均值 ÷ 该录制总量峰值；
  J(Δ)  = 电平(t_on+Δ) − 电平(t_on−0.05)，电平取 t 前 0.05 s 因果均值 ⇒ 只用 t ≤ t_on+Δ。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))            # T4_*/scripts
TASK = os.path.dirname(HERE)                                  # T4_*
PLAN = os.path.dirname(TASK)                                  # plan/v1.0
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))  # 仓库根
TEMP = os.path.join(ROOT, "temp")
PROG_REF = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_ad_lib as L  # noqa: E402

B = os.path.join(TEMP, "变化负载")
RECS = [
    ("右拇指指尖/数据1", os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv"), "显示域"),
    ("右拇指指尖/数据2", os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"), "显示域"),
    ("右拇指指尖/数据3", os.path.join(TEMP, "右拇指指尖", "数据3", "device_001_seg000.csv"), "显示域"),
    ("左拇指指尖/数据1", os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv"), "显示域"),
    ("左拇指指尖/数据2", os.path.join(TEMP, "左拇指指尖", "数据2", "device_001_seg000.csv"), "显示域"),
    ("左拇指指尖/数据3", os.path.join(TEMP, "左拇指指尖", "数据3", "device_001_seg000.csv"), "显示域"),
    ("四指指尖/数据1", os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv"), "显示域"),
    ("四指指尖/数据2", os.path.join(TEMP, "四指指尖", "数据2", "device_001_seg000.csv"), "显示域"),
    ("四指指尖/数据3", os.path.join(TEMP, "四指指尖", "数据3", "device_001_seg000.csv"), "显示域"),
    ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                        "20260917_133923_single_device_ee20bc",
                                        "device_001_seg000.csv"), "ADC域"),
    ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载",
                                "device_001_seg000.csv"), "ADC域"),
    ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "device_001_seg000.csv"), "ADC域"),
    ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv"), "ADC域"),
]

# 判据窗长 Δ（s）：判据只允许用 t ≤ t_on + Δ 的帧
LAGS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
# 形状采样网格（对齐既有 shape_stats.csv / rom 口径，采样点用因果窗）
TAUS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]
SMOOTH_S = 0.50          # 指标评估用中值平滑窗（= 指标字典 Z̄, k=0.5 s）
DET_SMOOTH = 3           # 检测/判据信号：3 帧因果中值
LVL_WIN_S = 0.05         # 电平取样因果窗
DET_WIN_S = 0.10         # 候选帧检测的因果平滑
EDGE_WIN_S = 0.15        # 候选帧检测窗（短滞后电平差）


def zbar(tot, dt):
    """指标字典的 Z̄ = median(Z, k)，k = round(0.5 s / dt)；居中（只用于指标评估与真值标注）。"""
    return L.med_smooth(tot, max(1, int(round(SMOOTH_S / dt))), causal=False)


def zbar_c(tot, dt):
    """因果版中值平滑（只回溯过去帧），指标评估用口径 Z̄，k = 0.5 s。

    ⚠️ **不要用它做 t_on 附近的判据特征**：因果中值在快速上升沿上滞后约 k/2 = 0.25 s，
    会把 J(0.05) 抬成 J(0.25)（实测 0.1118 → 0.1329 占峰值比，见 results/_t4b_dbg0.log）。
    判据特征一律用 `det_sig`（3 帧因果中值）。
    """
    return L.med_smooth(tot, max(1, int(round(SMOOTH_S / dt))), causal=True)


def det_sig(tot, dt):
    """检测/判据信号：3 帧因果中值（≈30 ms），滤掉单帧掉点又不引入可观测滞后。"""
    return L.med_smooth(tot, DET_SMOOTH, causal=True)


def find_true_edge(zc, k, dt):
    """真沿回溯：在候选帧 k 的 [−0.6, +0.1] s 内取单帧最大正跳变所在帧 +1 为 t_on。

    只用 t ≤ t_on 的数据估计 t_on 本身不可能，实际实现需要看到 t_on 后的 1 帧
    （≈10 ms 延迟）；报告里作为流水线固有延迟声明。
    """
    a = max(1, k - int(0.60 / dt))
    b = min(len(zc) - 1, k + int(0.10 / dt))
    if b <= a:
        return k
    return a + int(np.argmax(np.diff(zc[a:b]))) + 1


def detect_candidates(tot, dt, peak):
    """因果候选帧：短滞后电平差 > 3% 峰值且持续 3 帧；同一斜坡只报一次（越过 0.5 s 再续）。

    返回的是"检测命中帧"（斜坡上第一个越过门限的帧），真沿由 `find_true_edge` 回溯。
    """
    zc = L.med_smooth(tot, max(1, int(0.10 / dt)), causal=True)
    w = max(1, int(EDGE_WIN_S / dt))
    g = max(1, int(0.20 / dt))
    d = np.zeros_like(zc)
    d[w + g:] = zc[w + g:] - zc[:-w - g]
    thr = 0.03 * peak
    skip = max(1, int(0.50 / dt))
    out, i, n = [], w + g, len(d)
    while i < n:
        if d[i] > thr:
            run = 0
            while i + run < n and d[i + run] > thr:
                run += 1
            if run >= 3:
                out.append(i)                 # 只报斜坡的第一个命中帧
            i += max(run, skip)               # 越过整段斜坡再续，防止一段斜坡报多次
        else:
            i += 1
    return out


def lvl_c(zc, i, dt, win_s=LVL_WIN_S):
    """时刻 i 的**因果**电平 = [i−win, i] 均值。"""
    w = max(1, int(win_s / dt))
    a = max(0, i - w)
    return float(zc[a:i + 1].mean()) if i >= a else float(zc[i])


def build_events(recs=None, verbose=True):
    """返回 (events DataFrame, cache dict)。events 每行一个加载/减载事件。

    列：rec/dom/ch/t_on/pre_frac/pre_jump/A5_self/... + 各 Δ 的因果判据特征 + 事后参考量。
    """
    recs = recs if recs is not None else RECS
    rows, cache = [], {}
    for name, path, dom in recs:
        if not os.path.isfile(path):
            if verbose:
                print("!! 缺文件:", path)
            continue
        d = L.prep(path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        tot = d["tot"]
        peak = float(tot.max())
        c = det_sig(tot, dt)                   # 检测/判据信号（3 帧因果中值）
        f = zbar(tot, dt)                      # 事后参考用（居中 0.5 s 中值）
        cd = L.med_smooth(tot, max(1, int(DET_WIN_S / dt)), causal=True)
        cand = detect_candidates(tot, dt, peak)
        ev_idx = []
        for k in cand:
            t_on = find_true_edge(tot, k, dt)
            # 候选帧可能对应同一事件（去重）
            if ev_idx and t_on - ev_idx[-1] < int(1.5 / dt):
                continue
            ev_idx.append(t_on)
        cache[name] = dict(tu=tu, Xu=Xu, dt=dt, tot=tot, peak=peak, c=c, f=f, dom=dom,
                           ev_idx=ev_idx)
        if verbose:
            print("%-20s 候选事件 %2d" % (name, len(ev_idx)))
    # ── 逐事件建行 ──
    for name, _ in [(r[0], r[1]) for r in recs]:
        if name not in cache:
            continue
        cc = cache[name]
        tu, dt, c, f, peak, tot = cc["tu"], cc["dt"], cc["c"], cc["f"], cc["peak"], cc["tot"]
        n = len(tu)
        pre_jump = lvl_c(c, max(0, cc["ev_idx"][0] - 1), dt) if cc["ev_idx"] else 0.0
        # 逐帧滚动峰值（因果）：max_{t'<=t} Z̄_c
        run_max = np.maximum.accumulate(c)
        for k in cc["ev_idx"]:
            pre = lvl_c(c, max(0, k - 1), dt) if k > 0 else lvl_c(c, k, dt)
            # 基线取 t_on 前 0.5 s 的因果均值（更稳）
            w0 = max(0, k - int(0.50 / dt))
            base = float(c[w0:k].mean()) if k > w0 else pre
            row = dict(rec=name, dom=cc["dom"], t_on=round(float(tu[k]), 3),
                       pre=base, peak_roll=float(run_max[max(0, k - 1)]),
                       pre_frac=base / max(run_max[max(0, k - 1)], 1e-9))
            # ── 因果判据特征：只用 t ≤ t_on + Δ；电平取 i 前 0.05 s 因果均值（测量点在 t_on+Δ−0.05）──
            for D in LAGS:
                i = min(n - 1, k + int(round(D / dt)))
                lv = lvl_c(c, i, dt) - base
                row["inc_%03d" % int(round(D * 100))] = lv
                row["Jpc_%03d" % int(round(D * 100))] = 100.0 * lv / max(peak, 1e-9)
            row["slope_frac_020"] = (row["inc_020"] - row["inc_000"]) / 0.20
            row["JF_020"] = row["inc_000"] / row["inc_020"] if row["inc_020"] > 1e-12 else np.nan
            row["shape_020"] = row["inc_020"] / row["inc_050"] if row["inc_050"] > 1e-12 else np.nan
            row["shape_010"] = row["inc_010"] / row["inc_020"] if row["inc_020"] > 1e-12 else np.nan
            row["noise_adc"] = float(np.std(np.diff(tot[max(0, k - int(1.0 / dt)):k]))) / np.sqrt(2)
            # ── T4-A 接口增量：**因果**输入速率估计（T4-A 的 T_ramp 是非因果反卷积）──
            #   T50c(Δ) = 事件增量首次达到 0.5·J(Δ) 所用时间；到量 T90c 同理。
            #   只用 t ≤ t_on+Δ 的帧 ⇒ 可以逐 Δ 累计评估"用输入速率需要多长观测窗"。
            for D in LAGS:
                Dp = int(round(D * 100))
                jd = min(n - 1, k + int(round(D / dt)))
                Jd = lvl_c(c, jd, dt) - base
                if not (Jd > 0):
                    row["T50c_%03d" % Dp] = np.nan
                    row["T90c_%03d" % Dp] = np.nan
                    row["R50c_%03d" % Dp] = np.nan
                    continue
                keys = [kk for kk in
                        (0, 1, 2, 3, 5, 10, 20, 30, 50, 100) if kk <= Dp]
                inc_t = {kk: (lvl_c(c, min(n - 1, k + int(round(kk / 100.0 / dt))), dt) - base)
                         for kk in keys}
                t50 = next((kk / 100.0 for kk in keys if inc_t[kk] >= 0.5 * Jd), np.nan)
                t90 = next((kk / 100.0 for kk in keys if inc_t[kk] >= 0.9 * Jd), np.nan)
                row["T50c_%03d" % Dp] = t50
                row["T90c_%03d" % Dp] = t90
                # 速率统计量：观测窗内已完成的比例（越高 ⇒ 输入越快）
                row["R50c_%03d" % Dp] = (inc_t[keys[-1]] / max(Jd, 1e-12))
            # 早期到达比例：level(t_on+Δ) 相对 level(t_on+0.01 s) 已经涨了多少
            #   （= T4-A 的 z_at_* 同族量，但基准取 t_on+0.01 s 而不是 pre，纯因果）
            lvl_001 = lvl_c(c, min(n - 1, k + 1), dt) - base
            for D in LAGS:
                if D < 0.02:
                    row["FR_%03d" % int(round(D * 100))] = np.nan
                    continue
                row["FR_%03d" % int(round(D * 100))] = (
                    row["inc_%03d" % int(round(D * 100))] / lvl_001
                    if lvl_001 > 1e-12 else np.nan)
            # ── 事后参考量（非因果，仅用于分组真值与效应量，不参与判据）──
            pre_nc = float(np.median(f[max(0, k - int(1.0 / dt)):max(1, k - int(0.15 / dt))]))
            post_nc = float(np.median(f[min(n - 1, k + int(4.0 / dt)):min(n, k + int(6.0 / dt))]))
            Jnc = post_nc - pre_nc
            row["pre_nc"] = pre_nc
            row["post_nc"] = post_nc
            row["J_nc"] = Jnc
            for T in TAUS:
                i = min(n - 1, k + int(round(T / dt)))
                j = max(1, int(0.03 / dt))
                seg = f[max(0, i - j):min(n, i + j + 1)]
                row["sh_%03d" % int(round(T * 100))] = (float(np.median(seg)) - pre_nc) / Jnc \
                    if abs(Jnc) > 1e-12 else np.nan
            row["A5_self"] = float(np.median(f[min(n - 1, k + int(4.8 / dt)):min(n, k + int(5.2 / dt))])) - base
            row["pre_frac_nc"] = pre_nc / max(peak, 1e-9)
            row["J_frac_nc"] = Jnc / max(peak, 1e-9)
            # ── T4-Q5 判据 P1（纯因果、Δ=0）──
            #    p1_ratio：pre 因果电平(0.5 s) ÷ 因果滚动峰值（全历史）——Q5 主口径
            #    p1s_ratio：同分子 ÷ 检测帧冻结的滚动峰值 V=max(zc[:k+1])——T4-Q8 的冻结实现口径
            row["p1_ratio"] = row["pre_frac"]
            row["p1s_ratio"] = base / max(float(run_max[max(0, k)]), 1e-9)
            row["p1_onset"] = bool(row["p1_ratio"] <= 0.30)
            row["p1s_onset"] = bool(row["p1s_ratio"] <= 0.30)
            rows.append(row)
    ev = pd.DataFrame(rows)
    # ── 真值标签（对齐指标字典 §2.1；用非因果参考量分组，判据不用它）──
    kind = []
    for _, r in ev.iterrows():
        if r.pre_frac_nc < 0.20 and r.J_nc > 0:
            kind.append("onset")
        elif r.pre_frac_nc >= 0.20 and r.J_nc > 0:
            kind.append("restep")
        elif r.pre_frac_nc >= 0.20 and r.post_nc / max(r.peak_roll, 1e-9) < 0.20:
            kind.append("decrement")
        elif r.J_nc < 0:
            kind.append("partial_unload")
        else:
            kind.append("other")
    ev["kind"] = kind
    # 边界带：pre_nc 偏离 20% 门限不到 5 个百分点 ⇒ 该事件的"真值标签本身"随门限摆动
    # （第一轮 events.csv 也把同一批事件在 0.15~0.20 之间给了不同标签，见 _t4b_dbg1.log）
    ev["label_margin"] = np.abs(ev.pre_frac_nc - 0.20)
    ev["label_confident"] = ev.label_margin >= 0.05
    # 臂标签（= 真值分支）：空载起 onset / 带载起 restep（边界带单列 ambiguous）
    ev["arm"] = np.where(ev.pre_frac_nc < 0.20, "zero_baseline", "loaded")
    ev.loc[~ev.label_confident & ev.kind.isin(["onset", "restep"]), "arm"] = "ambiguous"
    ev.loc[ev.kind == "decrement", "arm"] = "decrement"
    ev.loc[ev.kind == "partial_unload", "arm"] = "partial_unload"
    ev = ev.sort_values(["rec", "t_on"]).reset_index(drop=True)
    # ── 第二套归一（与既有 shape_stats.csv 口径可比）：f2(τ) = J(τ)/J(5s) ──
    for T in TAUS:
        col = "sh_%03d" % int(round(T * 100))
        ev[col + "b"] = ev[col] * ev.J_nc / np.maximum(np.abs(ev.A5_self), 1e-12)
    for name, cc in cache.items():
        cc["ev_t"] = [float(cc["tu"][i]) for i in cc["ev_idx"]]
    return ev, cache


# ─────────────── T1 口径指标（指标字典 §3）───────────────
def t1_metrics(y_disp, y_raw, k, dt, n, next_ev_i, J, win_s=30.0, drift_frac=0.05):
    """单个事件的 T1 口径指标。y_disp/y_raw = 显示/原始**总量**序列（已居中平滑 0.5 s）。

    返回 dict(T_stable, os_pct, us_pct, md_adc, d1, d2, win_s, ok)。窗口不足填 NaN。
    """
    i1 = min(n - 1, k + int(round(win_s / dt)))
    if next_ev_i is not None:
        i1 = min(i1, next_ev_i - 1)
    span = (i1 - k) * dt
    out = dict(win_s=round(span, 2), T_stable=np.nan, T_stable_drift=np.nan, os_pct=np.nan,
               us_pct=np.nan, md_adc=np.nan, d1=np.nan, d2=np.nan)
    if span < 3.0 or abs(J) < 1e-12:
        return out
    y = y_disp[k:i1 + 1]
    r = y_raw[k:i1 + 1]
    # 参考终值：优先 t_on+60 s 后的中位；不可得则用窗末 20% 中位（并记录）
    j60 = min(n - 1, k + int(60.0 / dt))
    if j60 - k > int(5.0 / dt) and j60 < n - 1:
        yf = float(np.median(y_disp[j60:min(n, j60 + int(10.0 / dt))]))
    else:
        m = max(1, int(0.2 * len(y)))
        yf = float(np.median(y[-m:]))
    # 超调/欠调（相对参考终值）
    dev = (y - yf) / J
    out["os_pct"] = round(100 * float(dev.max()), 3)
    out["us_pct"] = round(100 * float(dev.min()), 3)
    # 最大偏差（绝对 ADC）
    out["md_adc"] = round(float(np.abs(y - r).max()), 1)
    # 1 s / 2 s 相对终值偏差
    for tag, T in (("d1", 1.0), ("d2", 2.0)):
        i = k + int(round(T / dt))
        if i <= i1:
            out[tag] = round(100 * (float(np.median(y[max(0, i - 2):i + 3])) - yf) / J, 3)
    # T_stable：从 t_on 起到"显示读数首次停住"
    #   实现（两种口径都算，报告中同时给）：
    #   T_stable_dev：首个使此后**相对参考终值**|y−yf| ≤ 5%·J 的时刻（可测、与既有 T_stable 同义）
    #   T_stable_drift：首个使此后**窗内自身漂移**（p90−p10）≤ 5%·J 的时刻（指标字典字面口径，
    #                   在慢相永不收敛的实录上往往不可达 ⇒ 报告为 NaN，这本身是结论）
    tol = drift_frac * abs(J)
    first_dev, first_drift = None, None
    for idx in range(1, len(y)):
        if len(y) - idx < max(10, int(0.5 / dt)):
            break
        if first_dev is None and float(np.abs(y[idx:] - yf).max()) <= tol:
            first_dev = idx
        seg = y[idx:]
        dr = float(np.percentile(seg, 90) - np.percentile(seg, 10))
        if first_drift is None and dr <= tol:
            first_drift = idx
        if first_dev is not None and first_drift is not None:
            break
    if first_dev is not None:
        out["T_stable"] = round(first_dev * dt, 3)
    if first_drift is not None:
        out["T_stable_drift"] = round(first_drift * dt, 3)
    return out
