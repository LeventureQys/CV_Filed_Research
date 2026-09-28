# -*- coding: utf-8 -*-
"""T2 公共层：13 份录制清单、双时间轴（100 Hz 网格 / 原始包时刻）、事件表读取、阶段判据常量、统计与日志。

路径锚点（00 号文档 §4.1）：
    HERE = <PLAN>/T2_*/scripts ; TASK = <PLAN>/T2_* ; PLAN = plan/v1.0
    ROOT = PLAN/../../../../ = 仓库根 ; TEMP = ROOT/temp（数据根）

口径（00_共享/指标字典与口径.md）：
  * 时间轴用 `timestamp` 列；禁用 elapsed；**禁止 τ=2 s 平滑**；
  * 形状用**原始读数**（`Z`），另给 3 帧中值（0.03 s）与 0.5 s 中值（Z̄）变体做对照；
  * `J = Z̄(post 窗中位) − Z̄(pre 窗中位)`，pre=[t0−2,t0)、post=[t0+4,t0+6]（既有口径）；
  * t50/t80/t90/t95：从 t_on 起算首次达到该比例·J 的时刻（分母用 |J|，卸载沿取反号）；
  * 事件集**复用 T4-A 的冻结事件表**（`T4_*/results/t4a_morphology.csv`，60 行），不重新检测。
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
PROG = os.path.join(ROOT, "temp", "v4.1flash", "progress")
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
CACHE = os.path.join(RES, "cache")
T4A_RES = os.path.join(PLAN, "T4_两种快相形态与分支判据", "results")
R13_RES = os.path.join(PROG, "13-v6-assessment", "results")
R07_RES = os.path.join(PROG, "07-v6", "results")
for _d in (RES, FIG, CACHE):
    os.makedirs(_d, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t2_ad_lib as L  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"     # 中文缺失 → 图件一律英文标签
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

FS = 100.0
DT = 1.0 / FS
KSM = int(round(0.5 / DT))                       # Z̄ 中位滤波窗 = 0.5 s（=50 帧）
K3 = 3                                           # 3 帧中值 = 0.03 s
TM = 20.0                                        # 比例穿越搜索上限（s）
TM_V6 = 10.0                                     # 07-v6 ce_v6_estimator 的搜索上限，对照用

# τ 网格（**全项目复用接口，不得改**）
TAU_FIXED = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75,
             1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0]
# 完成度列名（固定，避免 0.1/1.0 混淆）
TAU_COL = {0.02: "f_002", 0.05: "f_005", 0.10: "f_010", 0.20: "f_020",
           0.50: "f_050", 1.0: "f_100", 2.0: "f_200", 5.0: "f_500"}

# 阶段边界判据常量（T2 自定，报告 §2.3 给出理由）
FRAC_S1 = 0.25          # S1 阶跃段末 = 首次达到 0.25·|J|
T_S1_MAX = 0.05         # 「阶跃成立」判据：t25 ≤ 0.05 s（=5 帧 = 3 个指尖包）
FRAC_FAST = 0.90        # S2 快相末 = t90（指标字典 §2.2 主判）
FRAC_FAST_ALT = 0.95    # 备选口径
STEP_SENS_FRAC = [0.15, 0.25, 0.35]
STEP_SENS_T = [0.02, 0.05, 0.10]
SLOW_CAP = 60.0         # 慢相窗上限（s）
SLOW_UNIF = 30.0        # 均匀慢相窗（s，跨事件可比）
GUARD_NEXT = 0.5        # 与下一事件之间的保护间隔（s）

B = os.path.join(TEMP, "变化负载")
RECS = [
    dict(key="RT1", rec="右拇指指尖/数据1", fam="右拇指", dom="显示域", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="RT2", rec="右拇指指尖/数据2", fam="右拇指", dom="显示域", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="RT3", rec="右拇指指尖/数据3", fam="右拇指", dom="显示域", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="LT1", rec="左拇指指尖/数据1", fam="左拇指", dom="显示域", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="LT2", rec="左拇指指尖/数据2", fam="左拇指", dom="显示域", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="LT3", rec="左拇指指尖/数据3", fam="左拇指", dom="显示域", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="F41", rec="四指指尖/数据1", fam="四指", dom="显示域", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="F42", rec="四指指尖/数据2", fam="四指", dom="显示域", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="F43", rec="四指指尖/数据3", fam="四指", dom="显示域", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="SW1", rec="切换负载-快相无责", fam="实录", dom="ADC域", ch=4,
         path=os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc",
                           "device_001_seg000.csv")),
    dict(key="SW2", rec="再切换负载", fam="实录", dom="ADC域", ch=15,
         path=os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
    dict(key="SW3", rec="中途切换-1d9493", fam="实录", dom="ADC域", ch=6,
         path=os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
    dict(key="SW4", rec="中途切换-13ffca", fam="实录", dom="ADC域", ch=11,
         path=os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "最终测试目标",
                           "device_001_seg000.csv")),
]
REC_BY_KEY = {r["key"]: r for r in RECS}
# 07-v6 v6_rise_times.csv 的 ds 名 → 本任务 key
V6_DS = {"右拇指_1": "RT1", "右拇指_2": "RT2", "右拇指_3": "RT3",
         "左拇指_1": "LT1", "左拇指_2": "LT2", "左拇指_3": "LT3",
         "四指_1": "F41", "四指_2": "F42", "四指_3": "F43",
         "切换负载": "SW1", "再切换": "SW2", "中途1d9493": "SW3", "中途13ffca": "SW4"}


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
    print("# cmd   : python scripts/%s.py" % tag)
    print("# when  : %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    print("# env   : python %s | numpy %s | pandas %s" %
          (sys.version.split()[0], np.__version__, pd.__version__))
    print("# 口径  : timestamp 列 / 100 Hz 网格 / Z=Σch / 形状用原始读数 / 禁用 τ=2 s 平滑")
    print("")
    return p


# ───────────────────────── 数据 ─────────────────────────
def load_all(verbose=True):
    """读 13 份录制 → dict: key -> {tu, Z, Z3, Zbar, tp, Zp, Zp3, Zpbar, pkt, n_raw, reach}。

    网格轴：`tu` = 100 Hz 均匀网格（np.arange(0, span, 1/fs)，与 T4-A 完全同式 ⇒ k_on 可对齐）。
    包轴  ：`tp` = 包时刻，`Zp` = 包内均值总量（口径同 07-v6 ce_v6_estimator）。
    """
    out = {}
    for r in RECS:
        f = os.path.join(CACHE, "t2_%s.npz" % r["key"])
        if os.path.isfile(f):
            z = np.load(f)
            d = dict(tu=z["tu"], Z=z["Z"].astype(float), tp=z["tp"], Zp=z["Zp"].astype(float),
                     pkt=float(z["pkt"]), pkt50=float(z["pkt50"]), n_raw=int(z["n_raw"]),
                     n_pkt=int(z["n_pkt"]))
        else:
            if not os.path.isfile(r["path"]):
                raise RuntimeError("缺数据文件: %s" % r["path"])
            t, X = L.load_rec(r["path"])
            tu, Xu = L.to_grid(t, X, FS)
            tp, Xp = L.load_packets(t, X)
            pkt, pkt50 = L.packet_dt(t)
            Z, Zp = Xu.sum(axis=1), Xp.sum(axis=1)
            np.savez_compressed(f, tu=tu.astype(np.float64), Z=Z.astype(np.float32),
                               tp=tp.astype(np.float64), Zp=Zp.astype(np.float32),
                               pkt=np.array(pkt), pkt50=np.array(pkt50),
                               n_raw=np.array(len(t)), n_pkt=np.array(len(tp)))
            d = dict(tu=tu, Z=Z, tp=tp, Zp=Zp, pkt=pkt, pkt50=pkt50, n_raw=len(t), n_pkt=len(tp))
            if verbose:
                print("  [read] %-4s n_raw=%6d n_pkt=%6d span=%7.2fs pkt=%.4fs ch=%d" %
                      (r["key"], len(t), len(tp), tu[-1] - tu[0], pkt, Xu.shape[1]))
        d.update(key=r["key"], rec=r["rec"], fam=r["fam"], dom=r["dom"], ch=r["ch"],
                 Z3=L.med_smooth(d["Z"], K3), Zbar=L.med_smooth(d["Z"], KSM),
                 Zp3=L.med_smooth(d["Zp"], K3), Zpbar=L.med_smooth(d["Zp"], KSM))
        out[r["key"]] = d
    return out


# ───────────────────────── 事件表（复用 T4-A 冻结表）─────────────────────────
def load_events(with_tramp=True):
    fp = os.path.join(T4A_RES, "t4a_morphology.csv")
    ev = pd.read_csv(fp)
    ev["fam"] = ev["key"].map({r["key"]: r["fam"] for r in RECS})
    ev["_t2"] = ev["t_on"].round(2)
    if with_tramp:
        fp2 = os.path.join(T4A_RES, "t4a_input_recover.csv")
        if os.path.isfile(fp2):
            rc = pd.read_csv(fp2)[["key", "t_on", "T_ramp", "rmse_ramp_pct", "rmse_step_pct",
                                   "gain_ramp", "A_ramp"]]
            rc["_t2"] = rc["t_on"].round(2)
            ev = ev.merge(rc.drop(columns=["t_on"]), on=["key", "_t2"], how="left")
    return ev


def rec_end(all_d, key):
    return float(all_d[key]["tu"][-1])


# ───────────────────── 比例穿越 / 形状剖面（双轴通用）─────────────────────
def cross_time(t_ax, y, t0, pre, J, frac, tmax=TM, sustain=0.0):
    """首次 sgn·(y − pre) ≥ frac·|J| 的时刻（相对 t0，s）。

    `sustain > 0` 时要求**此后 sustain 秒内持续 ≥ 阈值**（"持续穿越"口径）——
    用于剔除加载撞击瞬态（实测 restep 在 0.02~0.05 s 有 25~45%·J 的过冲、0.1 s 内回落）。
    返回持续段的**起点**（不引入系统延迟）。
    """
    if not np.isfinite(J) or abs(J) < 1e-12:
        return np.nan
    tt = t0 + np.arange(0.0, tmax + 1e-9, DT)
    tt = tt[tt <= t_ax[-1]]            # 超出录制末的查询一律排除（不做常值外推）
    if tt.size == 0:
        return np.nan
    yy = np.interp(tt, t_ax, y)
    sgn = 1.0 if J > 0 else -1.0
    ok = sgn * (yy - pre) >= frac * abs(J)
    if not ok.any():
        return np.nan
    if sustain <= 0:
        return float(np.argmax(ok)) * DT
    k = int(round(sustain / DT))
    for i in np.where(ok)[0]:
        if i + k < len(ok) and ok[i:i + k + 1].all():
            return float(i) * DT
    return np.nan


def f_at(t_ax, y, t0, pre, J, tau):
    if not np.isfinite(J) or abs(J) < 1e-12 or (t0 + tau) > t_ax[-1]:
        return np.nan
    return float((np.interp(t0 + tau, t_ax, y) - pre) / J)


def axis_series(d, axis):
    """返回 (t_ax, y_raw, y_bar, dt) —— axis ∈ {grid, grid3, gridbar, pkt, pkt3, pktbar}。"""
    if axis.startswith("pkt"):
        t_ax, yraw, ybar = d["tp"], d["Zp"], d["Zpbar"]
        y3 = d["Zp3"]
    else:
        t_ax, yraw, ybar = d["tu"], d["Z"], d["Zbar"]
        y3 = d["Z3"]
    y = {"grid": yraw, "grid3": y3, "gridbar": ybar,
         "pkt": yraw, "pkt3": y3, "pktbar": ybar}[axis]
    return t_ax, y, ybar, DT


def j_of(ybar, t_ax, t0, dt=DT):
    """J 与 pre/post（指标字典 §2.1 口径，落在给定轴的中值滤波序列上；时间窗版，用于包轴）。"""
    pre = float(np.median(np.interp(np.arange(t0 - 2.0, t0, dt), t_ax, ybar)))
    post = float(np.median(np.interp(np.arange(t0 + 4.0, t0 + 6.0, dt), t_ax, ybar)))
    return pre, post, post - pre


def j_of_idx(ybar, k, dt=DT):
    """J 与 pre/post（**索引窗**版：pre=[k−2 s, k)、post=[k+4 s, k+6 s]，与 T4-A/既有口径逐点一致）。"""
    n = len(ybar)
    a, b = max(0, int(k - 2.0 / dt)), min(n, int(k))
    c, e = min(n, int(k + 4.0 / dt)), min(n, int(k + 6.0 / dt))
    pre = float(np.median(ybar[a:b])) if b > a else np.nan
    post = float(np.median(ybar[c:e])) if e > c else np.nan
    return pre, post, post - pre


def band(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    return (float(np.median(v)), float(np.percentile(v, 10)),
            float(np.percentile(v, 90)), int(v.size))


def fmt_band(v, nd=3, unit=""):
    m, p10, p90, n = band(v)
    if n == 0:
        return "n/a (n=0)"
    s = "%.*f [%.*f, %.*f] (n=%d)%s" % (nd, m, nd, p10, nd, p90, n, unit)
    if n <= 3:
        s += " [仅定性参考]"
    return s


def savefig(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print("  -> %s" % p)
    return p
