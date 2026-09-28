# -*- coding: utf-8 -*-
"""T6 公共层：路径、记录清单、事件几何、口径（含 legacy 口径逐字复刻）、扰动注入、指标。

口径声明（每个数字都必须带口径，见报告 §方法与口径）：
  · 网格：100 Hz 均匀网格（`np.interp`），时间轴一律取 `timestamp` 列，禁用 `elapsed`。
  · 域：恒载 9 组 = 力域（`calibration.display_mode=force`）；变载实录 4 份 = ADC 域。
    **跨域不可比绝对值**，一切跨域比较只用**相对量**（%·阶跃）。
  · 真沿 `t_on`：`first_onset`（= 第一轮 `pv_common.first_onset` 逐字复刻），
    以保证与 `13-v6-assessment/results/b_repeat_*.csv` 可直接对拍。
  · 两个真值口径（本轮反复强调：**换真值口径排名会反转**）：
      口径-R5   `pre + inc5`   （加载沿 +5 s 的原始电平；含快相蠕变）
      口径-Rstep `pre + a_step` （加载沿 +0.2 s 的机械台阶电平）
  · `T_stable`：T1-A 冻结口径**尚未交付**（`T1_*/results/t1a_settle_metrics.csv` 不存在），
    故本任务用「第一轮 legacy 口径」= `stable_time(tol=5%·step, hold=30 s)` 复刻，
    并**声明不可与 T1 绝对横比**；本报告一切横向比较以**同一次运行的配对差**为准。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
TEMP = os.path.join(ROOT, "temp")
PROG = os.path.join(TEMP, "v4.1flash", "progress")
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
sys.path.insert(0, HERE)

import t6_ad_lib as L                                              # noqa: E402
from t6_ad_lib import load_rec, med_smooth                         # noqa: E402

CJ = os.path.join(TEMP, "变化负载")
HOLD = {
    "右拇指指尖/数据1": os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv"),
    "右拇指指尖/数据2": os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"),
    "右拇指指尖/数据3": os.path.join(TEMP, "右拇指指尖", "数据3", "device_001_seg000.csv"),
    "左拇指指尖/数据1": os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv"),
    "左拇指指尖/数据2": os.path.join(TEMP, "左拇指指尖", "数据2", "device_001_seg000.csv"),
    "左拇指指尖/数据3": os.path.join(TEMP, "左拇指指尖", "数据3", "device_001_seg000.csv"),
    "四指指尖/数据1": os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv"),
    "四指指尖/数据2": os.path.join(TEMP, "四指指尖", "数据2", "device_001_seg000.csv"),
    "四指指尖/数据3": os.path.join(TEMP, "四指指尖", "数据3", "device_001_seg000.csv"),
}
VARY = {
    "切换负载-快相无责": os.path.join(CJ, "切换负载-快相无责的测试",
                                      "20260917_133923_single_device_ee20bc",
                                      "device_001_seg000.csv"),
    "零负载-切换负载-零负载-再切换负载": os.path.join(
        CJ, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv"),
    "中途切换-1d9493": os.path.join(CJ, "零负载-中途切换负载-零负载-切换负载",
                                    "device_001_seg000.csv"),
    "中途切换-最终目标-13ffca": os.path.join(CJ, "零负载-中途切换负载-零负载-切换负载",
                                             "最终测试目标", "device_001_seg000.csv"),
}
ALL = dict(HOLD)
ALL.update(VARY)
KIND = {k: "恒载" for k in HOLD}
KIND.update({k: "实采" for k in VARY})
DOMAIN = {}
DOMAIN.update({k: "force" for k in HOLD})
DOMAIN.update({k: "adc" for k in VARY})
GROUP = {"右拇指指尖": "右拇指", "左拇指指尖": "左拇指", "四指指尖": "四指"}
HOLD_TAGS = list(HOLD)
VARY_TAGS = list(VARY)

# ── 形状 τ 网格（C-2：T2 未交付前，本任务用 ROM 网格做骨架并声明） ──
TAU_GRID = np.array([0.20, 0.30, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00, 5.00])


# ─────────────────────────── 数据 ───────────────────────────
def grid(path, tmax=None, fs=100.0):
    """读录制 → 100 Hz 均匀网格。tmax 截断（秒，含端点前）。"""
    t, X = load_rec(path)
    if tmax is not None:
        m = t <= (t[0] + tmax)
        t, X = t[m], X[m]
    span = t[-1] - t[0]
    tu = np.arange(0.0, span, 1.0 / fs)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return dict(t=t - t[0], X=X, tu=tu, Xu=Xu, dt=1.0 / fs, span=span)


# ─────────────────────────── 口径：真沿 / T_stable ───────────────────────────
def first_onset(tu, tot, dtm):
    """第一轮 `pv_common.first_onset` 逐字复刻（真沿 + pre 电平）。"""
    peak = float(np.percentile(tot, 99.5))
    idx = np.where(tot > 0.5 * peak)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    pre = float(np.median(tot[max(0, i - int(1.5 / dtm)):max(1, i - int(0.3 / dtm))]))
    j = i
    while j > 0 and tot[j] > pre + 0.05 * (tot[i] - pre):
        j -= 1
    return j + 1, pre


def stable_time(tu, ytot, i0, step, dtm, hold_s=30.0, tol_frac=0.05):
    """legacy `T_stable`（第一轮口径，逐字复刻）：显示首次停住、此后 30 s 漂移 ≤5%·step。"""
    if step is None or not np.isfinite(step) or step <= 0:
        return np.nan
    n = len(ytot)
    H = int(hold_s / dtm)
    tol = tol_frac * step
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):
            break
        if np.max(np.abs(ytot[k:e] - ytot[k])) <= tol:
            return float(tu[k] - tu[i0])
    return np.nan


def onset_geometry(tu, tot, dtm):
    """一次加载事件的全部几何量（legacy 口径，可直接与 b_repeat_*.csv 对拍）。"""
    fo = first_onset(tu, tot, dtm)
    if fo is None:
        return None
    i_e, pre = fo
    ts1 = med_smooth(tot, max(1, int(round(0.1 / dtm))))
    n = len(tu)
    a_step = float(ts1[min(n - 1, i_e + int(0.2 / dtm))] - pre)
    inc5 = float(ts1[min(n - 1, i_e + int(5.0 / dtm))] - pre)
    tgt5 = float(np.median(tot[i_e + int(4.6 / dtm): i_e + int(5.4 / dtm)]))
    dom = dict(i_e=i_e, t_e=float(tu[i_e]), pre=pre, a_step=a_step, inc5=inc5,
               tgt5=tgt5, step=tgt5 - pre)
    return dom


# ─────────────────────────── 包结构与时序抖动 ───────────────────────────
def packet_layout(t_raw):
    """从原始 timestamp 列提取**真实包结构**（包内多帧共享同一到达时刻）。

    返回 (pkt_id_per_raw_row, n_pkt, pkt_period, frames_per_pkt)。
    判新包：与前一帧时间差 > 1 ms（同一包内时间差实测 ~1e-5 s）。
    """
    d = np.diff(t_raw)
    newp = np.concatenate([[True], d > 1e-3])
    pid = np.cumsum(newp) - 1
    nk = int(pid[-1]) + 1
    starts = t_raw[newp]
    P = float(np.median(np.diff(starts))) if nk > 2 else float(np.median(d))
    return pid, nk, P, float(len(t_raw)) / max(nk, 1)


def grid_packet_id(tu, t_raw, pid_raw):
    """把网格帧映射到包号：取「最后一个到达时刻 ≤ 该网格帧」的原始行所属包。"""
    j = np.searchsorted(t_raw, tu, side="right") - 1
    j = np.clip(j, 0, len(t_raw) - 1)
    return pid_raw[j]


def jitter_axis(tu, pkt, P, J, rng):
    """整包时序抖动：每包到达时刻加 δ_k~U(−J,J)，再 cummax 保序（到达顺序不可倒转）。

    **包内间隔保持不变**（帧值/帧序完全不动），因此这是"整包平移"而不是"单帧乱抖"。
    返回 (jittered_time_axis, reorder_frac)；reorder_frac = 被 cummax 抬起的包比例，
    是对"±J 是否超出物理可行域"的保真度提示。
    """
    nk = int(pkt.max()) + 1
    first = np.searchsorted(pkt, np.arange(nk))
    nominal = tu[first]
    raw = nominal + rng.uniform(-J, J, nk)
    T = np.maximum.accumulate(raw)
    reorder = float(np.mean(T - raw > 1e-12))
    return tu + (T[pkt] - nominal[pkt]), reorder


def add_bad_noise(Xu, A_rms, kind, rng, f_lo=0.3, f_hi=40.0):
    """复用 `t6_a_common.make_perturb`：**总量扰动的 RMS = A_rms (ADC)**。"""
    import t6_a_common as AC
    return Xu + AC.make_perturb(Xu, A_rms, kind, rng, f_lo=f_lo, f_hi=f_hi)


def dropout_frame(Xu, rng, n_drop=1):
    """单帧丢包：随机抽 1 帧，用前 1 帧值顶替（零阶保持，= 接收端最常见的丢帧隐藏）。"""
    Y = Xu.copy()
    n = Y.shape[0]
    idxs = rng.integers(60, n - 60, size=int(n_drop))
    for i in idxs:
        Y[i] = Y[i - 1]
    return Y, [int(i) for i in idxs]


# ─────────────────────────── 显示指标 ───────────────────────────
def display_metrics(tu, Y, dom, dtm):
    """在显示总量上算 L2 指标。Y = 逐通道输出矩阵。"""
    i_e, pre = dom["i_e"], dom["pre"]
    yt = Y.sum(axis=1)
    ds = med_smooth(yt, max(1, int(round(0.5 / dtm))))
    n = len(tu)
    t_e = float(tu[i_e])
    out = {}
    w = (tu >= t_e + 40.0) & (tu <= t_e + 60.0)
    out["plat_lvl"] = float(np.median(ds[w])) if w.any() else np.nan
    out["err_plat5_pct"] = (100.0 * ((out["plat_lvl"] - pre) / dom["inc5"] - 1.0)
                            if abs(dom["inc5"]) > 1e-12 else np.nan)
    out["err_platstep_pct"] = (100.0 * (out["plat_lvl"] - pre - dom["a_step"]) / dom["a_step"]
                               if abs(dom["a_step"]) > 1e-12 else np.nan)
    out["err5s"] = (100.0 * (yt[min(n - 1, i_e + int(5.0 / dtm))] / dom["tgt5"] - 1.0)
                    if abs(dom["tgt5"]) > 1e-12 else np.nan)
    win = (tu >= t_e) & (tu <= t_e + 30.0)
    ref = pre + dom["inc5"]
    if win.any() and abs(dom["inc5"]) > 1e-12:
        out["OS_pct"] = 100.0 * float(np.max(ds[win] - ref)) / abs(dom["inc5"])
        out["US_pct"] = 100.0 * float(np.min(ds[win] - ref)) / abs(dom["inc5"])
    else:
        out["OS_pct"] = out["US_pct"] = np.nan
    out["T_stable"] = stable_time(tu, yt, i_e, dom["step"], dtm)
    w2 = (tu >= t_e + 30.0) & (tu <= t_e + 60.0)
    out["flat5_pct"] = (100.0 * float(np.max(ds[w2]) - np.min(ds[w2])) / abs(dom["inc5"])
                        if w2.any() and abs(dom["inc5"]) > 1e-12 else np.nan)
    out["move3060_pct"] = (
        100.0 * (float(np.median(ds[(tu >= t_e + 50) & (tu <= t_e + 60)]))
                 - float(np.median(ds[(tu >= t_e + 30) & (tu <= t_e + 40)]))) / abs(dom["inc5"])
        if abs(dom["inc5"]) > 1e-12 else np.nan)
    return out


def raw_shape(tu, tot, dom, dtm, grid=TAU_GRID):
    """原始归一化快相形状 f(τ) = (Z̄(τ) − pre)/inc5（L1 输入层）。"""
    ts1 = med_smooth(tot, max(1, int(round(0.1 / dtm))))
    tau = tu - tu[dom["i_e"]]
    inc = ts1 - dom["pre"]
    if abs(dom["inc5"]) < 1e-12:
        return None
    return np.array([float(np.interp(x, tau, inc)) / dom["inc5"] for x in grid])


def rms_diff(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.mean(d * d)))


def pct_dist(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    return float(np.percentile(v, q))


def summarize(v, name):
    """报数规范：n≥20 报 mean±std，n<20 报中位 + p10~p90 + 极差。总是给 n。"""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(metric=name, n=0)
    d = dict(metric=name, n=int(len(v)), median=float(np.median(v)),
             p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
             rng=float(np.max(v) - np.min(v)))
    if len(v) >= 20:
        d.update(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)))
    if len(v) >= 2:
        d["std_all"] = float(np.std(v, ddof=1))
    return d
