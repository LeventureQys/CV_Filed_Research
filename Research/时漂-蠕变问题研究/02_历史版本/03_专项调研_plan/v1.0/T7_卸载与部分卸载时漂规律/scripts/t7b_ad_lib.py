# -*- coding: utf-8 -*-
"""T7-B 数据层（席位 T7-B · 部分卸载与加载/卸载对称性专项）。

与 `04-v5/scripts/ad_lib.py` 的区别（有意为之，见 00_共享/数据与脚本复用清单.md §2）：
  ① CSV 数据段起始行由 `##Data` 标记**自动定位**，不写死 skiprows=24；
  ② 时间轴一律取 `timestamp` 列（禁用 `elapsed`），并重采样到 100 Hz 均匀网格；
  ③ **方向统一口径**：所有时间指标定义在归一化进度 `f(τ)` 上（加载/卸载同式），
     彻底避免"卸载取负"带来的符号陷阱。

坐标锚点：HERE=T7_*/scripts → TASK=T7_* → PLAN=plan/v1.0 → ROOT=仓库根。
本模块只读数据，不写任何既有文件。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))                   # T7_*/scripts
TASK = os.path.dirname(HERE)                                        # T7_*
PLAN = os.path.dirname(TASK)                                        # plan/v1.0
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))  # 仓库根
TEMP = os.path.join(ROOT, "temp")
PROG = os.path.join(TEMP, "v4.1flash", "progress")
RESULTS = os.path.join(TASK, "results")
FIGURES = os.path.join(TASK, "figures")
T4A_RES = os.path.join(PLAN, "T4_两种快相形态与分支判据", "results")

FS = 100.0
DT = 1.0 / FS

# ── 录制清单（key -> 相对 temp/ 的路径, 域） ──────────────────────────────
RECS = {
    "RT1": ("右拇指指尖/数据1/device_001_seg000.csv", "显示域"),
    "RT2": ("右拇指指尖/数据2/device_001_seg000.csv", "显示域"),
    "RT3": ("右拇指指尖/数据3/device_001_seg000.csv", "显示域"),
    "LT1": ("左拇指指尖/数据1/device_001_seg000.csv", "显示域"),
    "LT2": ("左拇指指尖/数据2/device_001_seg000.csv", "显示域"),
    "LT3": ("左拇指指尖/数据3/device_001_seg000.csv", "显示域"),
    "F41": ("四指指尖/数据1/device_001_seg000.csv", "显示域"),
    "F42": ("四指指尖/数据2/device_001_seg000.csv", "显示域"),
    "F43": ("四指指尖/数据3/device_001_seg000.csv", "显示域"),
    "SW1": ("变化负载/切换负载-快相无责的测试/20260917_133923_single_device_ee20bc/"
            "device_001_seg000.csv", "ADC域"),
    "SW2": ("变化负载/零负载-切换负载-零负载-再切换负载/device_001_seg000.csv", "ADC域"),
    "SW3": ("变化负载/零负载-中途切换负载-零负载-切换负载/device_001_seg000.csv", "ADC域"),
    "SW4": ("变化负载/零负载-中途切换负载-零负载-切换负载/最终测试目标/"
            "device_001_seg000.csv", "ADC域"),
}

# T4-A 冻结事件表里的 `rec` 名 -> key（**不重新检测事件**，直接复用其 t_on / kind）
RECNAME2KEY = {
    "右拇指指尖/数据1": "RT1", "右拇指指尖/数据2": "RT2", "右拇指指尖/数据3": "RT3",
    "左拇指指尖/数据1": "LT1", "左拇指指尖/数据2": "LT2", "左拇指指尖/数据3": "LT3",
    "四指指尖/数据1": "F41", "四指指尖/数据2": "F42", "四指指尖/数据3": "F43",
    "切换负载-快相无责": "SW1", "再切换负载": "SW2",
    "中途切换-1d9493": "SW3", "中途切换-13ffca": "SW4",
}

KEY2RECNAME = {v: k for k, v in RECNAME2KEY.items()}

# 域声明（跨域不可比绝对值；本任务所有结论都分域报）
DOMAIN = {k: v[1] for k, v in RECS.items()}


def rec_path(key):
    return os.path.join(TEMP, *RECS[key][0].split("/"))


def load_rec(path):
    """自动定位 `##Data`，返回 (t, X, meta)。

    t = `timestamp` 列（自 0 起，**禁用 elapsed**）；X = 全部 `ch*` 列（float）。
    """
    hdr = None
    meta = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("##Data"):
                hdr = i + 1
                break
            meta.append(line.rstrip("\n"))
    if hdr is None:
        raise RuntimeError("no ##Data marker in " + path)
    df = pd.read_csv(path, skiprows=hdr)
    ch = [c for c in df.columns if str(c).startswith("ch")]
    t = df["timestamp"].to_numpy(float)
    return t - t[0], df[ch].to_numpy(float), {"meta": meta, "cols": ch, "n": len(df)}


def pkt_dt(t_raw):
    """包周期（s）——用于 ±1 包敏感性。

    实测包结构（本任务复测，见 results/t7b_recon.csv）：时间戳呈**突发包**结构，
    包内多帧共享近乎同一时刻，包间gap ≈ 包周期。
      · 变载实录：4 帧/包、包周期 ≈ 0.040 s
      · 指尖恒载：~1.65 帧/包、包周期 ≈ 0.0166 s
    估计法：取帧间差分的 p60 以上部分的中位数的 1/2 作为包内/包间分界，
    包周期 = 总时长 / 包数。
    """
    d = np.diff(np.asarray(t_raw, float))
    d = d[d > 1e-9]
    span = float(t_raw[-1] - t_raw[0])
    if len(d) < 10:
        return float(np.median(d)) if len(d) else 0.0
    hi = d[d > np.percentile(d, 60)]
    thr = 0.5 * float(np.median(hi))
    n_pkt = 1 + int((d > thr).sum())
    return span / max(n_pkt, 1)


def n_packets(t_raw):
    d = np.diff(np.asarray(t_raw, float))
    d = d[d > 1e-9]
    if len(d) < 10:
        return len(t_raw)
    hi = d[d > np.percentile(d, 60)]
    thr = 0.5 * float(np.median(hi))
    return 1 + int((d > thr).sum())


def dup_frac(t_raw):
    """包内重复率 = 1 - 包数/帧数（对齐"~4 帧/包"的实测结构）。"""
    return float(1.0 - n_packets(t_raw) / max(len(t_raw), 1))


def to_grid(t, X, fs=FS):
    """np.interp 到 100 Hz 均匀网格。"""
    span = t[-1] - t[0]
    tu = np.arange(0.0, span, 1.0 / fs)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu


def med_smooth(x, k):
    """k 帧中位平滑（k 为奇数）。本项目禁止 τ=2 s 平滑；此处只用于评估口径。"""
    k = int(k)
    if k <= 1:
        return np.asarray(x, float)
    if k % 2 == 0:
        k += 1
    return pd.Series(np.asarray(x, float)).rolling(k, center=True,
                                                   min_periods=1).median().to_numpy()


def flat_median(z, tu, t0, t1):
    m = (tu >= t0) & (tu < t1)
    return float(np.median(z[m])) if m.any() else np.nan


# ── 方向统一口径 ────────────────────────────────────────────────────────
# 定义（全任务统一，报告 §2 明写）：
#   pre  = median Z on [t0-2, t0)          （加载前 / 卸载前电平）
#   post = median Z on [t0+4, t0+6]        （对齐 00_共享/指标字典 §2.1 的既有口径）
#   D    = post - pre                      （**带符号**：加载为正、卸载为负）
#   f(τ) = (Z(t0+τ) - pre) / D             （**归一化进度，方向无关**：0→1）
#   t_α  = f 首次 >= α/100 的 τ（α ∈ {50,80,90,95}）
# ⇒ 加载与卸载用同一个式子，符号差异不会被误当作机制差异。
TAU_GRID = np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30,
                     0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0])


def event_metrics(zu, tu, t0, fs=FS, pre_win=(-2.0, 0.0), post_win=(4.0, 6.0),
                  tau_grid=TAU_GRID):
    """方向统一的单个事件指标。zu=100 Hz 网格上的总量 Z。"""
    i0 = int(round(t0 * fs))
    if i0 < 1 or i0 >= len(zu):
        return None
    pre = flat_median(zu, tu, t0 + pre_win[0], t0 + pre_win[1])
    post = flat_median(zu, tu, t0 + post_win[0], t0 + post_win[1])
    if not np.isfinite(pre) or not np.isfinite(post):
        return None
    D = post - pre
    if abs(D) < 1e-9:
        return None
    out = {"pre": pre, "post": post, "J": D, "absJ": abs(D)}
    # 归一化轮廓
    fgrid = []
    for tau in tau_grid:
        i = int(round((t0 + tau) * fs))
        fgrid.append((zu[i] - pre) / D if 0 <= i < len(zu) else np.nan)
    out["f_grid"] = np.array(fgrid)
    # 时间指标：在 t0..t0+8 s 内找首次达标
    seg = zu[i0: min(len(zu), i0 + int(8 * fs) + 1)]
    f = (seg - pre) / D
    for a in (10, 50, 80, 90, 95):
        thr = a / 100.0
        idx = np.argmax(f >= thr) if (f >= thr).any() else -1
        out["t%d" % a] = (idx / fs) if idx >= 0 else np.nan
    # 单帧/前 3 帧完成比例（与 T4-A 同名口径，便于对照）
    dz = np.diff(seg)
    out["step_frame_frac"] = float(np.max(np.abs(dz)) / abs(D)) if len(dz) else np.nan
    out["top3_frame_frac"] = float(np.sort(np.abs(dz))[::-1][:3].sum() / abs(D)) \
        if len(dz) >= 3 else np.nan
    # 过冲/下冲（相对 post，带符号：>0 = 越过 post 继续同向）
    fmax = np.nanmax(f)
    fmin = np.nanmin(f)
    out["overshoot_pct"] = float((fmax - 1.0) * 100.0)
    out["undershoot_pct"] = float((fmin - 0.0) * 100.0)
    out["z_min"] = float(np.min(seg))
    out["z_max"] = float(np.max(seg))
    return out


def read_frozen_events():
    """读 T4-A 冻结事件表（60 事件），只做列名规整，**不重新检测**。"""
    df = pd.read_csv(os.path.join(T4A_RES, "t4a_morphology.csv"))
    df["key"] = df["rec"].map(RECNAME2KEY)
    df = df[df["key"].notna()].copy()
    df["t_on"] = df["t_on"].astype(float)
    return df


def read_input_recover():
    """读 T4-A 的等效输入斜坡表（`T_ramp` / `gain_ramp` / `rmse_*`）。"""
    df = pd.read_csv(os.path.join(T4A_RES, "t4a_input_recover.csv"))
    df["key"] = df["rec"].map(RECNAME2KEY)
    df = df[df["key"].notna()].copy()
    df["t_on"] = df["t_on"].astype(float)
    return df


def ensure_dirs():
    for d in (RESULTS, FIGURES):
        os.makedirs(d, exist_ok=True)
