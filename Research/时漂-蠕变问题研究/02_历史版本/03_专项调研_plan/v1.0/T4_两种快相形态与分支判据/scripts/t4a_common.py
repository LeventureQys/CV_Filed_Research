# -*- coding: utf-8 -*-
"""T4-A 公共层：13 份录制清单、100 Hz 网格缓存、事件检测、形态指标、可分性统计、日志。

路径锚点（00 号文档 §4.1）：
    HERE = <PLAN>/T4_*/scripts ; TASK = <PLAN>/T4_* ; PLAN = plan/v1.0
    ROOT = PLAN/../../.. = 仓库根 ; TEMP = ROOT/temp（数据根）

口径（00_共享/指标字典与口径.md）：
    * 时间轴用 `timestamp` 列；np.interp 重采样 100 Hz（dt=0.01）；禁用 elapsed；
    * **禁止 τ=2 s 平滑**；本文只用 k=round(0.5/dt)=50 的**中位滤波** Z̄（仅指标评估用）；
    * 主信号 Z = Σ_c X[:,c]（总量口径），与 07-v6 §2.2、13-v6-assessment/phases.csv 同口径；
    * J = Z̄(post 窗中位) − Z̄(pre 窗中位)，pre 窗 [t0−2, t0)、post 窗 [t0+4, t0+6]；
    * t50/t80/t90/t95 从 t_on 起算；z_at_* 以 pre 为基准电平。
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
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
CACHE = os.path.join(RES, "cache")
for _d in (RES, FIG, CACHE):
    os.makedirs(_d, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import t4a_ad_lib as L  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

FS = 100.0
DT = 1.0 / FS
KSM = int(round(0.5 / DT))                       # Z̄ 中位滤波窗 = 0.5 s
TAU_FINE = np.round(np.arange(0.0, 5.0 + 1e-9, DT), 4)   # 机制/卷积用细网格 0..5 s
Z_TAU = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00, 5.00]
ZC = {0.05: "z_at_005", 0.10: "z_at_01", 0.20: "z_at_02", 0.30: "z_at_03",
      0.50: "z_at_05", 1.00: "z_at_10", 2.00: "z_at_20", 5.00: "z_at_50"}
FR = [0.10, 0.50, 0.80, 0.90, 0.95]   # t10 用于 rise_frames_10_90，其余对齐指标字典 §2.3

B = os.path.join(TEMP, "变化负载")
RECS = [
    dict(key="RT1", rec="右拇指指尖/数据1", dom="显示域", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="RT2", rec="右拇指指尖/数据2", dom="显示域", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="RT3", rec="右拇指指尖/数据3", dom="显示域", ch=17,
         path=os.path.join(TEMP, "右拇指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="LT1", rec="左拇指指尖/数据1", dom="显示域", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="LT2", rec="左拇指指尖/数据2", dom="显示域", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="LT3", rec="左拇指指尖/数据3", dom="显示域", ch=18,
         path=os.path.join(TEMP, "左拇指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="F41", rec="四指指尖/数据1", dom="显示域", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv")),
    dict(key="F42", rec="四指指尖/数据2", dom="显示域", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据2", "device_001_seg000.csv")),
    dict(key="F43", rec="四指指尖/数据3", dom="显示域", ch=11,
         path=os.path.join(TEMP, "四指指尖", "数据3", "device_001_seg000.csv")),
    dict(key="SW1", rec="切换负载-快相无责", dom="ADC域", ch=4,
         path=os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc",
                           "device_001_seg000.csv")),
    dict(key="SW2", rec="再切换负载", dom="ADC域", ch=15,
         path=os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
    dict(key="SW3", rec="中途切换-1d9493", dom="ADC域", ch=6,
         path=os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
    dict(key="SW4", rec="中途切换-13ffca", dom="ADC域", ch=11,
         path=os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "最终测试目标",
                           "device_001_seg000.csv")),
]


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
    print("# 口径     : timestamp 列 / 100 Hz 网格 / Z=Σch / 无 τ=2 s 平滑 / 中位滤波 0.5 s")
    print("")
    return p


# ───────────────────────── 数据 ─────────────────────────
def load_grid(r):
    """读一份录制并重采样到 100 Hz；带 npz 缓存（results/cache/）。返回 (tu, Xu, pkt, n_raw)。"""
    f = os.path.join(CACHE, "t4a_%s.npz" % r["key"])
    if os.path.isfile(f):
        z = np.load(f)
        return z["tu"], z["Xu"].astype(float), float(z["pkt"]), int(z["n_raw"])
    if not os.path.isfile(r["path"]):
        raise RuntimeError("缺数据文件: %s" % r["path"])
    t, X = L.load_rec(r["path"])
    tu, Xu = L.to_grid(t, X, FS)
    pkt, pkt50 = L.packet_dt(t)
    np.savez_compressed(f, tu=tu, Xu=Xu.astype(np.float32),
                        pkt=np.array(pkt), pkt50=np.array(pkt50), n_raw=np.array(len(t)))
    return tu, Xu.astype(float), pkt, len(t)


def med(a, b, arr, n):
    a = max(0, int(a))
    b = min(int(n), int(b))
    if b <= a:
        return np.nan
    return float(np.median(arr[a:b]))


# ───────────────────────── 事件检测 ─────────────────────────
def _coarse(y, thr_abs, min_gap=1.0, det_win=0.30):
    """0.2 s 滞后电平差 + 绝对门限 → 候选沿（含该候选处的 |d| 强度）。"""
    n = len(y)
    w = max(1, int(0.10 / DT))
    ys = L.med_smooth(y, max(1, w))
    d = np.zeros(n)
    d[w:-w] = ys[2 * w:] - ys[:-2 * w]
    cand = np.where(np.abs(d) > thr_abs)[0]
    out, last = [], -10 ** 9
    for i in cand:
        if i - last < int(min_gap / DT):
            continue
        j0 = max(0, i - int(det_win / DT))
        j1 = min(n, i + int(det_win / DT))
        j = j0 + int(np.argmax(np.abs(d[j0:j1])))
        out.append((int(j), float(abs(d[j]))))
        last = j
    return out, d, ys


def _merge(cands, min_gap=1.0):
    """按位置排序后合并 1.0 s 内重复候选：同组内保留强度最大者。"""
    cands = sorted(cands, key=lambda x: x[0])
    groups, cur = [], []
    for j, s in cands:
        if cur and j - cur[-1][0] < int(min_gap / DT):
            cur.append((j, s))
        else:
            if cur:
                groups.append(cur)
            cur = [(j, s)]
    if cur:
        groups.append(cur)
    return [max(g, key=lambda x: x[1]) for g in groups]


def refine_jump(Z, j, win=0.20):
    """把候选锚点 j 定位到 ±win 内**最大单帧跳变**所在帧（= 第一轮的 t_edge 口径）。"""
    n = len(Z)
    a = max(0, int(j - win / DT))
    b = min(n - 1, int(j + win / DT))
    if b - a < 2:
        return int(j)
    return int(a + int(np.argmax(np.abs(np.diff(Z[a:b])))))


def _drift_base(Zs, k, fit_s=1.5):
    """用 [k−2.0, k−0.5] 的线性外推当**局部基线**（去掉慢相漂移），返回基线数组（对全序列）。"""
    n = len(Zs)
    a = max(0, k - int(2.0 / DT))
    b = max(a + 2, k - int(0.5 / DT))
    seg = np.asarray(Zs[a:b], float)
    x = np.arange(len(seg)) * DT
    try:
        p = np.polyfit(x, seg, 1) if len(seg) >= 3 else np.array([0.0, float(np.median(seg))])
    except Exception:
        p = np.array([0.0, float(np.median(seg))])
    idx = np.arange(a, min(n, k + int(1.0 / DT)))
    return idx, (p[0] * (idx - a) * DT + p[1])


def refine_edge5(Zs, k, jj, frac=0.05, sustain=5):
    """漂移校正后的「真实加载起点」：Z−(线性漂移基线) 首次持续 ≥ frac·|J| 的帧。
    对阶跃（偏差 ≤1 帧）与斜坡（回溯到斜坡起点附近）都成立；返回 (idx, 相对 t_on 的秒数)。"""
    if not np.isfinite(jj) or abs(jj) < 1e-12:
        return np.nan, np.nan
    idx, base = _drift_base(Zs, k)
    if len(idx) == 0:
        return np.nan, np.nan
    r = np.asarray(Zs[idx], float) - base
    sgn = 1.0 if jj > 0 else -1.0
    hit = np.where(sgn * r >= frac * abs(jj))[0]
    for h in hit:
        if h + sustain <= len(idx) and np.all(sgn * r[h:h + sustain] >= 0.6 * frac * abs(jj)):
            return float(idx[h]), float((idx[h] - k) * DT)
    return np.nan, np.nan


def detect_events(tu, Z, Zs=None, Zch=None, amp_ch=None, min_gap=1.0, k_sigma=8.0,
                  floor_frac=0.02, floor_ch=0.08, sigma_L_k=10.0):
    """事件检出（T4-A 口径，报告 §2 详述）：

      A. 候选：① 总量 Z 上 `|Δlag(0.2 s)| > max(8σ_d, 2%·记录极差)`；
                ② 主通道上 `|Δlag| > 8%·该通道极差`（= 第一轮 r2_shapes.find_edges 的规则）。
         两路并集、1.0 s 内去重（保留强度大者）——单路都会漏（Z 漏小增量、主通道漏多通道同发事件）。
      B. 校验：±(0.2~1.0 s) 电平变化 `|Δ| ≥ max(10·σ_L, 2%·记录极差)`，滤掉漂移/量化伪沿。
      C. t_on：锚点 ±0.20 s 内**最大单帧跳变**帧（与第一轮完全同法，保证数字可比；
         「真实输入起点」另由 `refine_edge5` 给出，作为 t4a_morphology.csv 的 `d_t_on5` 列）。

    返回 (ks, info)；ks 为 t_on 帧号列表。
    """
    n = len(Z)
    amp = float(np.percentile(Z, 99.5) - np.percentile(Z, 0.5))
    w = max(1, int(0.10 / DT))
    _, dz, _ = _coarse(Z, 0.0)
    core = dz[w:-w]
    sd = 1.4826 * float(np.median(np.abs(core))) if core.size else 0.0
    thr_z = max(k_sigma * sd, floor_frac * amp)
    cz, _, _ = _coarse(Z, thr_z, min_gap=min_gap)
    cands = list(cz)
    thr_ch = np.nan
    if Zch is not None and amp_ch and amp_ch > 0:
        thr_ch = floor_ch * amp_ch
        cc, _, _ = _coarse(Zch, thr_ch, min_gap=min_gap)
        cands += list(cc)
    co = _merge(cands, min_gap=min_gap)
    # 电平校验用的噪声尺度（Zs 的 3 帧差分稳健尺度 ÷ sqrt(6) 还原为电平噪声）
    zs = Zs if Zs is not None else L.med_smooth(Z, KSM)
    d3 = np.diff(zs, 3)
    sL = 1.4826 * float(np.median(np.abs(d3))) / np.sqrt(6.0) if d3.size else 0.0
    thr_L = max(sigma_L_k * sL, floor_frac * amp)
    ks, drop, anchors = [], 0, []
    for j, s in co:
        p1, p2 = max(0, j - int(1.0 / DT)), min(n, j)
        q1, q2 = min(n, j + int(0.2 / DT)), min(n, j + int(1.0 / DT))
        if q2 <= q1 or p2 <= p1:
            drop += 1
            continue
        dl = float(np.median(zs[q1:q2]) - np.median(zs[p1:p2]))
        if not np.isfinite(dl) or abs(dl) < thr_L:
            drop += 1
            continue
        anchors.append(int(j))
        ks.append(refine_jump(Z, j))
    ks = sorted(set(ks))
    info = dict(amp=amp, sigma_d=sd, thr_z=thr_z, thr_ch=thr_ch, sL=sL, thr_L=thr_L,
                n_coarse=len(co), n_kept=len(ks), n_drop=drop, anchors=anchors)
    return ks, info


def round1_times(rec_name, cache={}):
    """只读引用第一轮 13-v6-assessment/results/events.csv 的事件时刻（主通道口径）。"""
    if "tab" not in cache:
        fp = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment",
                          "results", "events.csv")
        cache["tab"] = pd.read_csv(fp) if os.path.isfile(fp) else None
    tab = cache["tab"]
    if tab is None:
        return []
    return sorted(float(x) for x in tab.loc[tab.rec == rec_name, "t_edge"].tolist())


# ───────────────────────── 形态指标 ─────────────────────────
def _cross(z, jj, frac):
    """z: 事件原点起的电平增量序列（步长 DT）；返回首次达到 frac·J 的时刻（s）。"""
    if not np.isfinite(jj) or abs(jj) < 1e-12:
        return np.nan
    sgn = 1.0 if jj > 0 else -1.0
    tgt = sgn * frac * abs(jj)
    idx = np.where(sgn * (z - tgt) >= 0)[0]
    return float(idx[0]) * DT if len(idx) else np.nan


def fit_stretched_exp(tau, f):
    """f(τ) = 1 − exp(−(τ/τe)^β)；返回 (tau_e, beta, rms)。"""
    try:
        from scipy.optimize import curve_fit
        m = np.isfinite(f) & (tau > 0)

        def g(t, te, b):
            return 1.0 - np.exp(-np.power(np.maximum(t, 1e-6) / te, b))

        p, _ = curve_fit(g, tau[m], f[m], p0=[0.5, 1.0],
                         bounds=([0.005, 0.1], [50.0, 20.0]), maxfev=20000)
        r = float(np.sqrt(np.mean((g(tau[m], *p) - f[m]) ** 2)))
        return float(p[0]), float(p[1]), r
    except Exception:
        return np.nan, np.nan, np.nan


def fit_power(tau, f):
    """f(τ) = clip((τ/τp)^p, 0, 1)；返回 (p, tau_p, rms)。"""
    try:
        from scipy.optimize import curve_fit
        m = np.isfinite(f) & (tau > 0)

        def g(t, p, tp):
            return np.clip(np.power(np.maximum(t, 1e-9) / tp, p), 0.0, 1.0)

        p, _ = curve_fit(g, tau[m], f[m], p0=[0.5, 0.3],
                         bounds=([0.05, 0.005], [5.0, 20.0]), maxfev=20000)
        r = float(np.sqrt(np.mean((g(tau[m], *p) - f[m]) ** 2)))
        return float(p[0]), float(p[1]), r
    except Exception:
        return np.nan, np.nan, np.nan


def event_metrics(tu, Z, Zs, k, peak, amp, pkt, all_k, fit_tau=None):
    """单事件形态指标。k = 真实加载沿所在帧（该帧之后一帧出现最大单帧跳变）。"""
    n = len(Z)
    fit_tau = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]) if fit_tau is None else fit_tau
    pre = med(k - int(2.0 / DT), k, Zs, n)
    post = med(k + int(4.0 / DT), k + int(6.0 / DT), Zs, n)
    jj = post - pre
    r = dict(t_on=round(float(tu[k]), 3), k_on=int(k), pre=pre, post=post, jump=jj,
             ratio=(abs(jj) / max(abs(pre), 1e-9)), peak=peak,
             pre_over_peak=(pre / peak if peak else np.nan),
             pre_over_jump=(abs(pre) / max(abs(jj), 1e-9)),
             trunc=bool(k + int(6.0 / DT) >= n))
    if not np.isfinite(jj) or abs(jj) < 1e-12:
        for fr in FR:
            r["t%02d" % int(fr * 100)] = np.nan
        for tt in Z_TAU:
            r[ZC[tt]] = np.nan
        r.update(step_frame_frac=np.nan, top3_frame_frac=np.nan, rise_frames_10_90=np.nan,
                 n_frames_rise=np.nan, frames_to_50=np.nan, frames_to_90=np.nan, T_slope=np.nan,
                 exp_tau=np.nan, exp_beta=np.nan, exp_rms=np.nan, pow_p=np.nan, pow_tau=np.nan,
                 pow_rms=np.nan, t_on5=np.nan, d_t_on5=np.nan, n_other_near=0, clean=False,
                 pkt_dt=pkt)
        return r
    i5, dt5 = refine_edge5(Zs, k, jj)
    r["t_on5"] = round(float(tu[int(i5)]), 3) if np.isfinite(i5) else np.nan
    r["d_t_on5"] = dt5
    for fr in FR:
        r["t%02d" % int(fr * 100)] = _cross(Zs[k:k + int(8.0 / DT)] - pre, jj, fr)
    for tt in Z_TAU:
        i = k + int(round(tt / DT))
        r[ZC[tt]] = ((Zs[i] - pre) / jj) if (i < n and abs(jj) > 1e-12) else np.nan
    # 上升沿帧结构（用**原始 Z** 的逐帧微分，不做平滑）
    a = max(0, k - 1)
    b = min(n - 1, k + int(1.5 / DT))
    if b - a >= 3 and abs(jj) > 1e-12:
        d = np.abs(np.diff(Z[a:b]))
        r["step_frame_frac"] = float(d.max() / abs(jj))
        r["top3_frame_frac"] = float(np.sort(d)[-3:].sum() / abs(jj)) if d.size >= 3 else np.nan
        r["rise_frames_10_90"] = ((r["t90"] - r["t10"]) / DT
                                  if np.isfinite(r.get("t90", np.nan)) and np.isfinite(r.get("t10", np.nan))
                                  else np.nan)
        # 承载上升的帧数：|ΔZ| > 5%·|J| 的帧数（限 [t_on, t_on+5 s]）
        seg = np.abs(np.diff(Z[k:k + int(5.0 / DT)]))
        r["n_frames_rise"] = int((seg > 0.05 * abs(jj)).sum())
        # 稳健「上升沿帧数」：承载 50% / 90% 台阶所需的帧数（按单帧增量降序累加，窗长 5 s）
        dsort = np.sort(np.diff(Z[k:k + int(5.0 / DT)]))[::-1] * np.sign(jj)
        cs = np.cumsum(dsort)
        r["frames_to_50"] = int(np.argmax(cs >= 0.5 * abs(jj)) + 1) if cs.size and cs[-1] >= 0.5 * abs(jj) else np.nan
        r["frames_to_90"] = int(np.argmax(cs >= 0.9 * abs(jj)) + 1) if cs.size and cs[-1] >= 0.9 * abs(jj) else np.nan
        # 初始 0.15 s 稳健斜率 → 等效输入斜坡时长（辅助估计，与反卷积独立）
        seg2 = Z[k:k + int(0.15 / DT)]
        if len(seg2) >= 6:
            sl = float(np.polyfit(np.arange(len(seg2)) * DT, seg2, 1)[0])
            r["T_slope"] = (abs(jj) / sl) if (sl * np.sign(jj) > 0) else np.nan
        else:
            r["T_slope"] = np.nan
    else:
        r["step_frame_frac"] = r["top3_frame_frac"] = np.nan
        r["rise_frames_10_90"] = r["n_frames_rise"] = r["T_slope"] = np.nan
    # 形状参数拟合（归一化轮廓 f(τ)=z(τ)/J，τ∈[0.1,4] s）
    if abs(jj) > 1e-12:
        tau = np.asarray(fit_tau, float)
        f = np.array([(Zs[k + int(round(t / DT))] - pre) / jj for t in tau])
        te, be, re = fit_stretched_exp(tau, f)
        pp, tp, rp = fit_power(tau, f)
    else:
        te = be = re = pp = tp = rp = np.nan
    r.update(exp_tau=te, exp_beta=be, exp_rms=re, pow_p=pp, pow_tau=tp, pow_rms=rp)
    # 干净事件判据（指标字典 §2.1）：前 3 s / 后 8 s 内无其它事件，且 |J| ≥ 2% 记录峰值
    other = [x for x in all_k if x != k and (k - int(3.0 / DT)) <= x <= (k + int(8.0 / DT))]
    r["n_other_near"] = len(other)
    r["clean"] = bool(len(other) == 0 and abs(jj) >= 0.02 * abs(peak))
    r["pkt_dt"] = pkt
    return r


def classify(r, thr=0.20):
    """armed 判定口径（需求 T4 §3-1）：pre 窗中位电平 ≥ thr × 记录峰值 ⇒ 带载起。
    返回 (kind, armed) ；jump<0 归 unload / partial_unload（不并入 restep 统计）。"""
    if r["jump"] < 0:
        frac = abs(r["jump"]) / max(abs(r["pre"]), 1e-12)
        return ("unload" if frac >= 0.8 else "partial_unload"), True
    armed = bool(r["pre_over_peak"] >= thr)
    return ("restep" if armed else "onset"), armed


# ───────────────────────── 统计 ─────────────────────────
def auc(xp, xn):
    """AUC（onset 为正类），秩统计量，含并列修正。"""
    xp = np.asarray(xp, float)
    xn = np.asarray(xn, float)
    xp = xp[np.isfinite(xp)]
    xn = xn[np.isfinite(xn)]
    if len(xp) < 2 or len(xn) < 2:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([xp, xn]))
    n1, n2 = len(xp), len(xn)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def overlap_ratio(pa, pb):
    """p10~p90 重叠度：交集长度 ÷ 较窄带宽（0=完全分离，1=窄带被完全包含）。"""
    (a0, a1), (b0, b1) = pa, pb
    if not all(np.isfinite([a0, a1, b0, b1])):
        return np.nan
    inter = min(a1, b1) - max(a0, b0)
    w = min(a1 - a0, b1 - b0)
    return float(max(0.0, inter) / w) if w > 0 else np.nan


def band(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    return (float(np.median(v)), float(np.percentile(v, 10)),
            float(np.percentile(v, 90)), int(v.size))


def phase_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def savefig(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print("  -> %s" % p)
    return p
