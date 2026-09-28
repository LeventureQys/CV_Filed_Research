# -*- coding: utf-8 -*-
"""t3b_settle.py —— T3-B 用的「稳定时间核 + 汇总 + 日志」层。

**口径来源（C-1 处置）**：`amp_5s / t_stable / t_stable_ev / t_settle / next_event_cut /
z_final_of / os_dir / fill_rate_summary / unload_index` **逐行复制**自
`plan/v1.0/T1_稳定时间定义与鲁棒性口径/scripts/t1a_common.py`（T1-A 冻结口径的代码实现），
只把命名前缀改为 t3b、日志名改为 `_t3b_*.log`。

所以本任务报出的 `T_stable` **就是 T1-A 的冻结主口径 D1**（完整 30 s 窗、不因后续事件截断、
漂移 ≤5%·|J_ref|）；实录类另给 **D1-ev**（窗在下一真实事件处截断、要求可用窗 ≥5 s）。

### 与 T1-A 实现的两处差异（均已在本文件内自带等价性自检，见 `t3b_00_recon.py`）
  1. `t_stable / t_stable_ev / t_settle` 增加**向量化实现**（`*_fast`）并作为默认实现：
     T1-A 的逐 k 扫描是 O(n·H)（H=3000 帧），在 raw 臂（永不稳定 ⇒ 每次扫到底）上
     单事件单序列就要 ~6×10⁷ 次比较，40 事件 ×4 序列不可接受。
     向量化用滚动/suffix max-min 表达**同一判据**，等价性由 `t3b_00_recon.py` 的
     逐元素比对证明（`t3b_patch_ab_zero.csv` 的 kernel 行）。
  2. 新增 `us_dir / md_abs / capture_ratio`（指标字典 §3 的 US% / MD / G），T1-A 未实现。

`n<20` 一律报中位 + p10~p90（指标字典 §6）；`n<=3` 标「仅定性参考」。
⚠ T1-A 的 `results/t1a_settle_metrics.csv` 截稿时尚未产出，本任务用同一实现自行复算，
T1 交付后需逐事件对齐一次（见报告「未验证项」）。
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

FS = 100.0
DT = 1.0 / FS


# ───────────────────────── 幅度（T1-A 冻结口径） ─────────────────────────

def _idx(k0, dt, off):
    return int(round(k0 + off / dt))


def win_median(Y, k0, off0, off1, dt=DT):
    """以 t_on+off 为窗（[off0, off1)）的中位电平。"""
    a = max(0, _idx(k0, dt, off0))
    b = min(len(Y), _idx(k0, dt, off1))
    if b <= a:
        return np.nan
    return float(np.median(Y[a:b]))


def amp_5s(Y, k0, dt=DT, pre=(-2.0, 0.0), post=(4.0, 6.0)):
    """指标字典 §2.1 口径：J = post 窗中位 − pre 窗中位；同时返回 (J, pre, post)。"""
    p = win_median(Y, k0, pre[0], pre[1], dt)
    q = win_median(Y, k0, post[0], post[1], dt)
    return (q - p), p, q


# ───────────────────────── 稳定时间核 ─────────────────────────

def _roll_max_min(Y, H):
    """window=H 的**前向**滚动 max/min：第 k 项覆盖 [k, k+H−1]（右侧不足 H 处为 NaN）。

    ⚠ pandas 的 `rolling` 是**后向（trailing）**窗，直接用会变成"过去 H 帧"，
    判据随之被写成"此后 30 s 平坦"却算成"此前 30 s 平坦" —— 实测会把 v6 的 1.0 s
    误报成 30.5 s（见 scripts/t3b_dbg_stable.py 的诊断）。故先反转序列再滚、再反回来。
    """
    s = pd.Series(np.asarray(Y, float)[::-1])
    rmax = s.rolling(H, min_periods=H).max().to_numpy()[::-1]
    rmin = s.rolling(H, min_periods=H).min().to_numpy()[::-1]
    return rmax, rmin


def t_stable_ref(tu, Y, k0, ref_amp, hold=30.0, tol_frac=0.05, dt=DT):
    """【T1-A 原实现，逐行复制】参考版本：O(n·H)，只用于等价性自检。"""
    n = len(Y)
    H = int(round(hold / dt))
    tol = tol_frac * abs(ref_amp)
    if not np.isfinite(ref_amp) or tol <= 0:
        return np.nan, True
    for k in range(k0, n):
        e = min(n, k + H)
        if e - k < min(H, n - k0):
            break
        if float(np.max(np.abs(Y[k:e] - Y[k]))) <= tol:
            return float(tu[k] - tu[k0]), False
    return np.nan, True


def t_stable(tu, Y, k0, ref_amp, hold=30.0, tol_frac=0.05, dt=DT):
    """【冻结主口径 D1】显示首次"停下"：存在 τ 使 t≥t_on+τ 后 hold 内自身漂移 ≤tol×|J_ref|。

    要求完整 hold 窗（不足即删失），**不因后续事件截断**。返回 (T, censored)。
    与 `t_stable_ref` 判据等价（向量化）。
    """
    n = len(Y)
    H = int(round(hold / dt))
    tol = tol_frac * abs(ref_amp)
    if not np.isfinite(ref_amp) or tol <= 0:
        return np.nan, True
    if n - k0 < H:                       # 记录尾窗不足：只剩 k=k0 一个候选（同参考实现的 break）
        if n <= k0:
            return np.nan, True
        if float(np.max(np.abs(Y[k0:n] - Y[k0]))) <= tol:
            return float(tu[k0] - tu[k0]), False
        return np.nan, True
    rmax, rmin = _roll_max_min(Y, H)
    ok = (rmax - Y <= tol) & (Y - rmin <= tol)
    ok = np.nan_to_num(ok, nan=0.0).astype(bool)
    hi = n - H
    idx = np.nonzero(ok[k0:hi + 1])[0]
    if idx.size:
        return float(tu[k0 + int(idx[0])] - tu[k0]), False
    return np.nan, True


def t_stable_ev_ref(tu, Y, k0, ref_amp, cut_k=None, hold=30.0, tol_frac=0.05,
                    min_hold=5.0, dt=DT):
    """【T1-A 原实现，逐行复制】D1-ev 参考版本，只用于等价性自检。"""
    n = len(Y)
    end = n if cut_k is None else min(n, int(cut_k))
    H = int(round(hold / dt))
    Hm = int(round(min_hold / dt))
    tol = tol_frac * abs(ref_amp)
    if not np.isfinite(ref_amp) or tol <= 0 or end - k0 < Hm:
        return np.nan, True, np.nan
    for k in range(k0, end - Hm + 1):
        e = min(end, k + H)
        if float(np.max(np.abs(Y[k:e] - Y[k]))) <= tol:
            return float(tu[k] - tu[k0]), False, float(tu[e - 1] - tu[k])
    return np.nan, True, np.nan


def t_stable_ev(tu, Y, k0, ref_amp, cut_k=None, hold=30.0, tol_frac=0.05,
                min_hold=5.0, dt=DT):
    """【修订口径 D1-ev】同 D1，但窗在下一个真实事件处截断（要求可用窗 ≥min_hold）。

    返回 (T, censored, 实际用到的窗长 s)。与 `t_stable_ev_ref` 判据等价（向量化）。
    """
    n = len(Y)
    end = n if cut_k is None else min(n, int(cut_k))
    H = int(round(hold / dt))
    Hm = int(round(min_hold / dt))
    tol = tol_frac * abs(ref_amp)
    if not np.isfinite(ref_amp) or tol <= 0 or end - k0 < Hm:
        return np.nan, True, np.nan
    rr = Y[:end]
    rmax, rmin = _roll_max_min(rr, H)
    # 完整 H 窗条件（k+H ≤ end）
    ok_full = (rmax - rr <= tol) & (rr - rmin <= tol)
    # 截断窗条件（k+H > end）：窗 = [k, end)
    smax = pd.Series(rr[::-1]).cummax().to_numpy()[::-1]
    smin = pd.Series(rr[::-1]).cummin().to_numpy()[::-1]
    ok_suf = (smax - rr <= tol) & (rr - smin <= tol)
    ok = np.where(np.arange(end) + H <= end, ok_full, ok_suf)
    ok = np.nan_to_num(ok, nan=0.0).astype(bool)
    idx = np.nonzero(ok[k0:end - Hm + 1])[0]
    if idx.size:
        k = k0 + int(idx[0])
        e = min(end, k + H)
        return float(tu[k] - tu[k0]), False, float(tu[e - 1] - tu[k])
    return np.nan, True, np.nan


def t_settle_ref(tu, Y, k0, ref_amp, frac, z_final, cut_k=None, dt=DT):
    """【T1-A 原实现，逐行复制】参考版本，只用于等价性自检。"""
    n = len(Y)
    end = n if cut_k is None else min(n, int(cut_k))
    if not np.isfinite(ref_amp) or not np.isfinite(z_final) or end <= k0:
        return np.nan, True
    tol = frac * abs(ref_amp)
    ok = np.abs(Y - z_final) <= tol
    for k in range(k0, end):
        if bool(ok[k:end].all()):
            return float(tu[k] - tu[k0]), False
    return np.nan, True


def t_settle(tu, Y, k0, ref_amp, frac, z_final, cut_k=None, dt=DT):
    """【D2】首次进入并保持 |Y−Z_final| ≤ frac×|J_ref|（保持到 cut/记录末）。"""
    n = len(Y)
    end = n if cut_k is None else min(n, int(cut_k))
    if not np.isfinite(ref_amp) or not np.isfinite(z_final) or end <= k0:
        return np.nan, True
    tol = frac * abs(ref_amp)
    seg = np.abs(np.asarray(Y[k0:end]) - z_final) <= tol
    if seg.size == 0:
        return np.nan, True
    sa = np.logical_and.accumulate(seg[::-1])[::-1]
    idx = np.nonzero(sa)[0]
    if idx.size:
        return float(tu[k0 + int(idx[0])] - tu[k0]), False
    return np.nan, True


def z_final_of(Y, k0, cut_k, dt=DT):
    """Z_final：t_on+60 s 后中位（指标字典 §3 口径）；不足则退回 t_on+10 s 后中位。"""
    n = len(Y)
    end = n if cut_k is None else min(n, int(cut_k))
    a = k0 + int(60.0 / dt)
    from_s = 60.0
    if end - a < int(3.0 / dt):
        a = k0 + int(10.0 / dt)
        from_s = 10.0
    if end - a < int(1.0 / dt):
        return np.nan, np.nan
    return float(np.median(Y[a:end])), from_s


def os_dir(Y, k0, target, ref_amp, dt=DT, hold=30.0):
    """阶跃方向上的最大过冲（%×|J|）= 指标字典 §3 的「超调 OS%」（负值即下冲）。"""
    n = len(Y)
    a, b = k0, min(n, k0 + int(hold / dt))
    if not np.isfinite(ref_amp) or abs(ref_amp) < 1e-12 or b <= a or not np.isfinite(target):
        return np.nan
    s = np.sign(ref_amp)
    return float(np.max((Y[a:b] - target) * s)) / abs(ref_amp) * 100.0


def us_dir(Y, k0, target, ref_amp, dt=DT, hold=30.0):
    """【US%】同 os_dir 但取最小（下冲，负值）。"""
    n = len(Y)
    a, b = k0, min(n, k0 + int(hold / dt))
    if not np.isfinite(ref_amp) or abs(ref_amp) < 1e-12 or b <= a or not np.isfinite(target):
        return np.nan
    s = np.sign(ref_amp)
    return float(np.min((Y[a:b] - target) * s)) / abs(ref_amp) * 100.0


def md_abs(Ydisp, Yraw, k0, cut_k, hold=30.0, dt=DT):
    """【MD】指定窗内 max|显示总量 − 原始总量|（绝对 ADC），与指标字典 §3 一致。"""
    n = len(Ydisp)
    a = k0
    b = min(n, k0 + int(round(hold / dt)))
    if cut_k is not None:
        b = min(b, int(cut_k))
    if b - a < 2:
        return np.nan
    return float(np.abs(np.asarray(Ydisp[a:b]) - np.asarray(Yraw[a:b])).max())


def capture_ratio(Ydisp, Yraw, k0, lag=20.0, dt=DT):
    """【G】台阶捕获比 = 事件后 lag 秒的「显示增量 / 原始增量」。

    与 `progress/06-.../免责期1s-3s-5s对比.md` §3.2 同义；调用方须自行限定
    「原始增量足够大」的事件（本任务另出 `G_den_adc` 列供筛选）。
    """
    i = k0 + int(round(lag / dt))
    if i >= len(Ydisp):
        return np.nan
    d_raw = float(Yraw[i]) - float(Yraw[k0])
    if abs(d_raw) < 1e-12:
        return np.nan
    return (float(Ydisp[i]) - float(Ydisp[k0])) / d_raw


def unload_index(y, k, dt=DT):
    """加载沿之后的最大负跳变（卸载沿）；逐字复制自 r4_filter_tradeoff.py。"""
    w = max(1, int(0.10 / dt))
    j0, j1 = k + int(3.0 / dt), len(y) - 1
    d = y[j0 + w:j1] - y[j0:j1 - w]
    if len(d) == 0:
        return None
    return j0 + int(np.argmin(d))


def next_event_cut(times, t_on, peak, min_frac=0.02):
    """下一个"真实事件"的时刻：t 严格晚于本事件 且 |J| ≥ min_frac×记录峰值。"""
    cand = [t for t, j in times
            if t > t_on + 1e-9 and j is not None and np.isfinite(j) and abs(j) >= min_frac * peak]
    return float(min(cand)) if cand else None


# ───────────────────────── 汇总 ─────────────────────────

def fill_rate_summary(v):
    """n<20 报中位 + p10~p90；n<=3 标"仅定性参考"。"""
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    n = int(a.size)
    if n == 0:
        return dict(n=0, med=np.nan, p10=np.nan, p90=np.nan, worst=np.nan, note="无可测样本")
    d = dict(n=n, med=float(np.median(a)), p10=float(np.percentile(a, 10)),
             p90=float(np.percentile(a, 90)), worst=float(np.max(a)), note="")
    if n <= 3:
        d["note"] = "仅定性参考(n<=3)"
    return d


def q(v, p):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.percentile(a, p)) if a.size else np.nan


def med(v):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.median(a)) if a.size else np.nan


# ───────────────────────── 日志（stdout 存档，含运行命令） ─────────────────────────

class _Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        try:
            self.stdout.write(s)
        except UnicodeEncodeError:
            self.stdout.write(s.encode("utf-8", "replace").decode("utf-8", "replace"))
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def start_log(res_dir, tag):
    """把 stdout 同时写进 <res>/_t3b_<tag>.log，并在头部记录运行命令。"""
    os.makedirs(res_dir, exist_ok=True)
    p = os.path.join(res_dir, "_t3b_%s.log" % tag)
    sys.stdout = _Tee(p)
    print("==== T3-B %s ==== %s" % (tag, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("cmd : %s %s" % (sys.executable, " ".join(sys.argv)))
    print("cwd : %s" % os.getcwd())
    return p
