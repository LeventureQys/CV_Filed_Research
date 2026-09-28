# -*- coding: utf-8 -*-
"""T3-A 公共层：路径锚点、13 份录制清单、100 Hz 网格缓存、T4-A 冻结事件表读取、
归一化快相形状 f(τ) 的采样、形状参数化、分层变量定义、日志。

路径锚点（00-项目组织文档.md §4.1）：
    HERE = <PLAN>/T3_*/scripts ; TASK = <PLAN>/T3_* ; PLAN = plan/v1.0
    ROOT = PLAN/../../../.. = 仓库根 ; TEMP = ROOT/temp（数据根）

口径（00_共享/指标字典与口径.md）：
    * 时间轴 `timestamp` 列；np.interp → 100 Hz 均匀网格（dt=0.01 s）；**禁用 elapsed**；
    * **禁止 τ=2 s 平滑**；只用 k=round(0.5/dt)=50 的**中位滤波** Z̄（仅指标评估）；
    * 主信号 Z = Σ_c X[:,c]（总量口径）；
    * 归一化形状 f(τ) = (Z̄(t_on+τ) − J_ref_base) / J，**主基准电平 = T4-A 的 `pre` 窗中位**
      （C-2：T4-A 已冻结该口径；样本表见 results/t3a_recon_crosscheck.csv）；
    * 主 τ 网格 = **T4-A 冻结的 z_at_* 采样点** [0.05,0.10,0.20,0.30,0.50,1.00,2.00,5.00]，
      加密网格 dτ=0.01 s 作为补充（两套都在结果表里标出 `grid` 列）。
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
TEMP = os.path.join(ROOT, "temp")
PROG = os.path.join(TEMP, "v4.1flash", "progress")
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
CACHE = os.path.join(RES, "cache")
for _d in (RES, FIG, CACHE):
    os.makedirs(_d, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t3a_ad_lib as L  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

FS = 100.0
DT = 1.0 / FS
KSM = int(round(0.5 / DT))                      # Z̄ 中位滤波窗 = 0.5 s

# —— 主 τ 网格（= C-2 处置：T4-A 已冻结的 z_at_* 采样点，唯一权威骨架）——
TAU_MAIN = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00, 5.00]
ZC = {0.05: "z_at_005", 0.10: "z_at_01", 0.20: "z_at_02", 0.30: "z_at_03",
      0.50: "z_at_05", 1.00: "z_at_10", 2.00: "z_at_20", 5.00: "z_at_50"}
TAU_MAIN_LABEL = "T4A_z_at_grid"

# —— 加密网格（补充，dτ=0.01 s；用于变异带图与分位数曲线）——
TAU_FINE = np.round(np.arange(0.01, 5.0 + 1e-9, DT), 4)
TAU_FINE_LABEL = "fine_0p01s"

# —— 形状参数拟合窗（T4-A 口径 τ∈[0.1,4] s；本任务改用 [0.2,5] s，理由见报告 §2.3）——
FIT_TAU_T4A = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
FIT_TAU = np.array([0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

# —— 冻结输入：T4-A 事件表（只读引用）——
T4A_DIR = os.path.join(PLAN, "T4_两种快相形态与分支判据")
T4A_MORPH = os.path.join(T4A_DIR, "results", "t4a_morphology.csv")
T4A_INPUT = os.path.join(T4A_DIR, "results", "t4a_input_recover.csv")
T4A_CONCL = os.path.join(T4A_DIR, "results", "conclusions_T4A.json")

# —— 第一轮产物（只读引用，用于反驳性对照）——
R1_DIR = os.path.join(PROG, "13-v6-assessment")
R1_SHAPE_STATS = os.path.join(R1_DIR, "results", "shape_stats.csv")
R1_TIMEWARP = os.path.join(R1_DIR, "results", "shape_timewarp.csv")
R1_ROM_COMPARE = os.path.join(R1_DIR, "results", "rom_compare.csv")
R1_ROM_LOO = os.path.join(R1_DIR, "results", "rom_loo.csv")
R1_EVENTS = os.path.join(R1_DIR, "results", "events.csv")

B = os.path.join(TEMP, "变化负载")
RECS = [
    dict(key="RT1", rec="右拇指指尖/数据1", dom="显示域", fam="右拇指", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="RT2", rec="右拇指指尖/数据2", dom="显示域", fam="右拇指", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="RT3", rec="右拇指指尖/数据3", dom="显示域", fam="右拇指", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="LT1", rec="左拇指指尖/数据1", dom="显示域", fam="左拇指", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="LT2", rec="左拇指指尖/数据2", dom="显示域", fam="左拇指", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="LT3", rec="左拇指指尖/数据3", dom="显示域", fam="左拇指", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="F41", rec="四指指尖/数据1", dom="显示域", fam="四指", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="F42", rec="四指指尖/数据2", dom="显示域", fam="四指", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="F43", rec="四指指尖/数据3", dom="显示域", fam="四指", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="SW1", rec="切换负载-快相无责", dom="ADC域", fam="实录", ch=4,
         path=os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc",
                           "device_001_seg000.csv")),
    dict(key="SW2", rec="再切换负载", dom="ADC域", fam="实录", ch=15,
         path=os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
    dict(key="SW3", rec="中途切换-1d9493", dom="ADC域", fam="实录", ch=6,
         path=os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
    dict(key="SW4", rec="中途切换-13ffca", dom="ADC域", fam="实录", ch=11,
         path=os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "最终测试目标",
                           "device_001_seg000.csv")),
]
REC_BY_KEY = {r["key"]: r for r in RECS}
# 事件表里的 rec 名字 → key（T4-A 用中文录制名）
KEY_BY_REC = {r["rec"]: r["key"] for r in RECS}


# ───────────────────────── 日志 ─────────────────────────
class _Tee(object):
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass
        self.f.flush()


def start_log(tag=None):
    tag = tag or os.path.splitext(os.path.basename(sys.argv[0]))[0]
    p = os.path.join(RES, "_%s.log" % tag)
    sys.stdout = _Tee(p)
    print("# cmd      : python scripts/%s.py" % tag)
    print("# cwd      : %s" % os.getcwd())
    print("# argv     : %s" % " ".join(sys.argv))
    print("# when     : %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    print("# env      : python %s | numpy %s | pandas %s" %
          (sys.version.split()[0], np.__version__, pd.__version__))
    print("# 口径     : timestamp 列 / 100 Hz 网格 / Z=Σch / 无 τ=2 s 平滑 / 中位滤波 0.5 s /")
    print("#            f(τ) 基准 = T4-A 的 pre 窗中位 / 主 τ 网格 = T4-A 的 z_at_* 采样点")
    print("")
    return p


# ───────────────────────── 数据 ─────────────────────────
def load_grid(key):
    """读一份录制并重采样到 100 Hz；带 npz 缓存。返回 (tu, Xu, Z, Zs, pkt, n_raw)。"""
    r = REC_BY_KEY[key]
    f = os.path.join(CACHE, "t3a_%s.npz" % key)
    if os.path.isfile(f):
        z = np.load(f)
        return z["tu"], z["Xu"].astype(float), z["Z"].astype(float), z["Zs"].astype(float), \
            float(z["pkt"]), int(z["n_raw"])
    if not os.path.isfile(r["path"]):
        raise RuntimeError("缺数据文件: %s" % r["path"])
    t, X = L.load_rec(r["path"])
    tu, Xu = L.to_grid(t, X, FS)
    Z = Xu.sum(axis=1)
    Zs = L.med_smooth(Z, KSM)
    pkt, _ = L.packet_dt(t)
    np.savez_compressed(f, tu=tu, Xu=Xu.astype(np.float32), Z=Z.astype(np.float32),
                        Zs=Zs.astype(np.float32), pkt=np.array(pkt), n_raw=np.array(len(t)))
    return tu, Xu, Z, Zs, pkt, len(t)


def med_win(a, b, arr, n):
    a = max(0, int(a))
    b = min(int(n), int(b))
    if b <= a:
        return np.nan
    return float(np.median(arr[a:b]))


def zref(Zs, k, pre, post, tau):
    """f(τ) = (Z̄(t_on+τ) − base) / J。返回 (数值, 用的 base, 用的 J)。"""
    n = len(Zs)
    i = k + int(round(tau / DT))
    if i >= n:
        return np.nan, np.nan, np.nan
    J = post - pre
    if not np.isfinite(J) or abs(J) < 1e-12:
        return np.nan, np.nan, np.nan
    return float((Zs[i] - pre) / J), pre, J


def load_t4a_events():
    """只读 T4-A 冻结事件表（60 行 × 51 列），补 key/dom/fam 派生列。"""
    df = pd.read_csv(T4A_MORPH)
    df["key"] = df["rec"].map(lambda s: KEY_BY_REC.get(s, "?"))
    df["t4a_row"] = np.arange(len(df))
    for t in TAU_MAIN:
        c = ZC[t]
        if c not in df.columns:
            df[c] = np.nan
    df["dom"] = df["key"].map(lambda k: REC_BY_KEY[k]["dom"] if k in REC_BY_KEY else "?")
    df["fam"] = df["key"].map(lambda k: REC_BY_KEY[k]["fam"] if k in REC_BY_KEY else "?")
    # 装载类 = jump>0（onset + restep）；unload / partial_unload 不并入
    df["is_load"] = df["kind"].isin(["onset", "restep"])
    return df


def load_t4a_input():
    """只读 T4-A 反卷积结果（T_ramp 等）。"""
    df = pd.read_csv(T4A_INPUT)
    return df


# ───────────────────────── 形状参数化 ─────────────────────────
def fit_single_tau(tau, f):
    """单 τ 模型 f(τ) = 1 − exp(−τ/τ1)。返回 (tau1, rms)。"""
    from scipy.optimize import curve_fit
    tau = np.asarray(tau, float)
    f = np.asarray(f, float)
    m = np.isfinite(f) & (tau > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    try:
        def g(t, t1):
            return 1.0 - np.exp(-t / t1)
        p, _ = curve_fit(g, tau[m], f[m], p0=[0.5], bounds=([0.005], [50.0]), maxfev=20000)
        r = float(np.sqrt(np.mean((g(tau[m], *p) - f[m]) ** 2)))
        return float(p[0]), r
    except Exception:
        return np.nan, np.nan


def fit_stretched_exp(tau, f):
    """拉伸指数 f(τ) = 1 − exp(−(τ/τe)^β)。返回 (tau_e, beta, rms)。"""
    from scipy.optimize import curve_fit
    tau = np.asarray(tau, float)
    f = np.asarray(f, float)
    m = np.isfinite(f) & (tau > 0)
    if m.sum() < 4:
        return np.nan, np.nan, np.nan
    try:
        def g(t, te, b):
            return 1.0 - np.exp(-np.power(np.maximum(t, 1e-6) / te, b))
        p, _ = curve_fit(g, tau[m], f[m], p0=[0.5, 1.0],
                         bounds=([0.005, 0.1], [50.0, 20.0]), maxfev=20000)
        r = float(np.sqrt(np.mean((g(tau[m], *p) - f[m]) ** 2)))
        return float(p[0]), float(p[1]), r
    except Exception:
        return np.nan, np.nan, np.nan


def fit_power(tau, f):
    """幂律 f(τ) = clip((τ/τp)^p, 0, 1)。返回 (p, tau_p, rms)。"""
    from scipy.optimize import curve_fit
    tau = np.asarray(tau, float)
    f = np.asarray(f, float)
    m = np.isfinite(f) & (tau > 0)
    if m.sum() < 4:
        return np.nan, np.nan, np.nan
    try:
        def g(t, p, tp):
            return np.clip(np.power(np.maximum(t, 1e-9) / tp, p), 0.0, 1.0)
        p, _ = curve_fit(g, tau[m], f[m], p0=[0.5, 0.3],
                         bounds=([0.05, 0.005], [5.0, 20.0]), maxfev=20000)
        r = float(np.sqrt(np.mean((g(tau[m], *p) - f[m]) ** 2)))
        return float(p[0]), float(p[1]), r
    except Exception:
        return np.nan, np.nan, np.nan


def fit_two_tau(tau, f):
    """双 τ 模型 f(τ) = w·[1−exp(−τ/τa)] + (1−w)·[1−exp(−τ/τb)]，τa ≤ τb。
    返回 (tau_a, tau_b, w, rms, ok)。搜索用多起点 + 有界最小二乘（numpy 可复现）。"""
    from scipy.optimize import least_squares
    tau = np.asarray(tau, float)
    f = np.asarray(f, float)
    m = np.isfinite(f) & (tau > 0)
    if m.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan, False
    tt, ff = tau[m], f[m]

    def resid(p):
        la, lb, w = p
        ta, tb = 10.0 ** la, 10.0 ** lb
        mm = w * (1.0 - np.exp(-tt / ta)) + (1.0 - w) * (1.0 - np.exp(-tt / tb))
        return mm - ff

    best = None
    for la0 in (-1.0, -0.5, 0.0):
        for lb0 in (0.3, 0.7, 1.2):
            for w0 in (0.2, 0.5, 0.8):
                try:
                    r = least_squares(resid, [la0, lb0, w0],
                                      bounds=([-2.3, -2.3, 0.0], [1.0, 1.7, 1.0]))
                except Exception:
                    continue
                ss = float(np.sum(r.fun ** 2))
                if best is None or ss < best[0]:
                    best = (ss, r.x)
    if best is None:
        return np.nan, np.nan, np.nan, np.nan, False
    _, (la, lb, w) = best
    ta, tb = 10.0 ** la, 10.0 ** lb
    if ta > tb:
        ta, tb, w = tb, ta, 1.0 - w
    rms = float(np.sqrt(best[0] / len(tt)))
    return float(ta), float(tb), float(w), rms, True


# ───────────────────────── 统计 ─────────────────────────
def band(v):
    """中位 + p10 + p90 + n（n<20 一律报这个，指标字典 §6-2）。"""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    return (float(np.median(v)), float(np.percentile(v, 10)),
            float(np.percentile(v, 90)), int(v.size))


def cv_of(v):
    """变异系数 std(ddof=1)/mean。返回 (cv, mean, std, n)；mean 接近 0 时 cv 不可用（另标）。"""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return np.nan, (float(np.mean(v)) if n else np.nan), np.nan, n
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    return (sd / mu if abs(mu) > 1e-12 else np.nan), mu, sd, n


def savefig(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print("  -> %s" % p)
    return p
