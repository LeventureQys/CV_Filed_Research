# -*- coding: utf-8 -*-
"""T1-A 公共层：13 份录制表 / 100 Hz 网格装载 / 三实现仪表化运行 / 稳定时间核。

来源与差异（有意为之）：
  * `RECS` 的路径与主通道取自 `progress/13-v6-assessment/scripts/r4_filter_tradeoff.py` 的
    `CASES`（13 份录制、主通道 17/18/11/4/15/6/11），**只读引用后重写为本任务的表**；
  * `make_traced / run_traced` 复制自 `progress/13-v6-assessment/scripts/a_common.py`
    （只记录状态变化、不改变算法行为）；`make_traced_v5` 复制自 `ad_lib.make_traced`
    （v5.1 的钩子是 `_begin/_restep`，与 v6 的 `_new_event/_handoff` 不同）；
  * 稳定时间核 `t_stable / t_stable_ev / t_settle / t_band` 是**本任务冻结口径**的实现，
    与 `13-v6-assessment/scripts/r4_filter_tradeoff.py::stable_time`、
    `11-paper-v6/scripts/pv_common.py::{stable_time,settle_time}` 逐条对照（见 t1a_00_smoke.py 自检）。

算法原型一律从本目录 `t1a_glm53_*.py` 导入（副本），**不 import 既有桶**。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)                                   # plan/v1.0/T1_*
PLAN = os.path.dirname(TASK)                                   # plan/v1.0
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))   # 仓库根
TEMP = os.path.join(ROOT, "temp")
PROG = os.path.join(ROOT, "temp", "v4.1flash", "progress")
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
CACHE = os.path.join(RES, "cache")
for _p in (HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from t1a_glm53_v51 import GLM53v51      # noqa: E402
from t1a_glm53_v6 import GLM53v6        # noqa: E402
from t1a_glm53_v61 import GLM53v61      # noqa: E402

FS = 100.0
DT = 1.0 / FS

ARMS = ["raw", "v5.1", "v6", "v6.1"]
CLS = {"v5.1": GLM53v51, "v6": GLM53v6, "v6.1": GLM53v61}

# rec 键与 `t4a_morphology.csv` 的 `rec` 列逐字一致（复用其 60 个事件的前提）
_R = os.path.join(TEMP, "右拇指指尖")
_L = os.path.join(TEMP, "左拇指指尖")
_F = os.path.join(TEMP, "四指指尖")
_B = os.path.join(TEMP, "变化负载")

RECS = []
for _loc, _base, _ch, _fam in (("右拇指指尖", _R, 17, "右拇指"), ("左拇指指尖", _L, 18, "左拇指"),
                               ("四指指尖", _F, 11, "四指")):
    for _i in (1, 2, 3):
        RECS.append(dict(rec="%s/数据%d" % (_loc, _i), family=_fam, dom="显示域", ch=_ch,
                         path=os.path.join(_base, "数据%d" % _i, "device_001_seg000.csv")))
RECS += [
    dict(rec="切换负载-快相无责", family="实录", dom="ADC域", ch=4,
         path=os.path.join(_B, "切换负载-快相无责的测试",
                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
    dict(rec="再切换负载", family="实录", dom="ADC域", ch=15,
         path=os.path.join(_B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
    dict(rec="中途切换-1d9493", family="实录", dom="ADC域", ch=6,
         path=os.path.join(_B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
    dict(rec="中途切换-13ffca", family="实录", dom="ADC域", ch=11,
         path=os.path.join(_B, "零负载-中途切换负载-零负载-切换负载", "最终测试目标",
                           "device_001_seg000.csv")),
]
REC_BY_KEY = {r["rec"]: r for r in RECS}

# `数据与脚本复用清单.md` §1 的帧数，用于 t1a_00_smoke.py 的"读到别的文件"自检
N_FRAMES = {"右拇指指尖/数据1": 16356, "右拇指指尖/数据2": 23537, "右拇指指尖/数据3": 19138,
            "左拇指指尖/数据1": 19740, "左拇指指尖/数据2": 19975, "左拇指指尖/数据3": 19402,
            "四指指尖/数据1": 19140, "四指指尖/数据2": 19114, "四指指尖/数据3": 20033,
            "切换负载-快相无责": 25693, "再切换负载": 7129,
            "中途切换-1d9493": 6421, "中途切换-13ffca": 12121}


def load_grid(rec, fs=FS):
    """读一份录制 → 100 Hz 网格。返回 (tu, Xu, meta)；meta 含包周期与域。"""
    import t1a_ad_lib as L
    r = REC_BY_KEY[rec] if isinstance(rec, str) else rec
    t, X = L.load_rec(r["path"])
    tu, Xu = L.to_grid(t, X, fs)
    p90, p50 = L.packet_dt(t)
    return tu, Xu, dict(rec=r["rec"], family=r["family"], dom=r["dom"], ch=int(r["ch"]),
                        n_raw=int(len(t)), span=float(t[-1] - t[0]), n_ch=int(X.shape[1]),
                        pkt_p90=p90, pkt_p50=p50, dup=int((np.diff(t) <= 0).sum()))


# ───────────────────────── 仪表化运行（复制自 a_common.py / ad_lib.py） ─────────────────────────

def make_traced(cls):
    """子类化 v6 系原型，只记录状态变化，不改变行为（逐行复制自 a_common.py::make_traced）。"""
    class T(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.tr_epoch = []     # (t0, kind, t_det, base, v0sum, y0sum)
            self.tr_revoke = []    # (ts, kind, tau)
            self.tr_handoff = []   # (ts, kind, A_hat, c_applied, A_sum, g)
            self.tr_unload = []    # (ts,)
            self.tr_gevent = []    # (ts, kind, d_like)
            self.tr_deny = []

        def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
            super()._new_event(t0, base, v0, y0, kind, hist, prev_state)
            self.tr_epoch.append((float(t0), str(kind), float(self.last_ts),
                                  float(base), float(np.sum(v0)), float(np.sum(y0))))

        def _handoff(self, ts, v, ev=None):
            ev = ev if ev is not None else self.ev
            rec = (float(ts), str(ev["kind"]), float(ev["A_hat"]), float(ev["c_applied"]))
            super()._handoff(ts, v, ev)
            self.tr_handoff.append(rec + (float(np.sum(self.A)), float(self.g)))

        def _to_idle(self):
            super()._to_idle()
            self.tr_unload.append(float(self.last_ts))

        def _run_event(self, ts, v, total, dt, eps, idle_now):
            ev = self.ev
            kind = str(ev["kind"]) if ev is not None else None
            n_rev = self.n_revoke
            e0 = None
            if ev is not None and kind == "decrease":
                e0 = float(ev["t0"])
            out = super()._run_event(ts, v, total, dt, eps, idle_now)
            if kind == "decrease" and self.n_revoke > n_rev:
                self.tr_gevent.append((float(ts), kind, e0))
            return out

        def process(self, ts, v):
            epoch_n = len(self.tr_epoch)
            rev_n = self.n_revoke
            out = super().process(ts, v)
            if self.n_revoke > rev_n and len(self.tr_epoch) == epoch_n:
                self.tr_revoke.append((float(ts), "revoke"))
            return out
    return T


def make_traced_v5(cls):
    """v5.1 的仪表化（钩子 `_begin/_restep`，复制自 ad_lib.make_traced）。"""
    class T(cls):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))
    return T


TRACED = {"v5.1": make_traced_v5(GLM53v51), "v6": make_traced(GLM53v6), "v6.1": make_traced(GLM53v61)}


def run_arm(rec, arm, tu=None, Xu=None):
    """跑一条臂。raw 直接返回输入；其余返回仪表化输出。

    返回 dict(Y=输出矩阵, ch=Y[:, main_ch], tot=Y.sum(1), yraw_ch=原始主通道, yraw_tot=原始总量,
              epoch/revoke/handoff/unload 列表, comp=实例)
    """
    if tu is None:
        tu, Xu, _ = load_grid(rec)
    ch = REC_BY_KEY[rec]["ch"] if isinstance(rec, str) else rec["ch"]
    ych = Xu[:, ch].copy()
    ytot = Xu.sum(axis=1)
    if arm == "raw":
        return dict(Y=Xu, ch=ych, tot=ytot, yraw_ch=ych, yraw_tot=ytot,
                    epoch=[], revoke=[], handoff=[], unload=[], comp=None)
    c = TRACED[arm](Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    ep = getattr(c, "epoch_t", None)
    if ep is None:
        ep = getattr(c, "tr_epoch", [])
    return dict(Y=Y, ch=Y[:, ch], tot=Y.sum(axis=1), yraw_ch=ych, yraw_tot=ytot,
                epoch=list(ep),
                revoke=list(getattr(c, "tr_revoke", [])),
                handoff=list(getattr(c, "tr_handoff", [])),
                unload=list(getattr(c, "tr_unload", [])), comp=c)


# ───────────────────────── 稳定时间核（本任务冻结口径） ─────────────────────────

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


def t_stable(tu, Y, k0, ref_amp, hold=30.0, tol_frac=0.05, dt=DT):
    """【冻结主口径 D1】显示首次"停下"：存在 τ 使 t≥t_on+τ 后 hold 内自身漂移 ≤tol×|J_ref|。

    口径与 `07-v6/MANIFEST.md`、`11-paper-v6/pv_common.stable_time`、
    `13-v6-assessment/r4_filter_tradeoff.stable_time` 一致：**要求完整 hold 窗**（不足即删失），
    **不因后续事件截断**。返回 (T, censored)。
    """
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


def t_stable_ev(tu, Y, k0, ref_amp, cut_k=None, hold=30.0, tol_frac=0.05, min_hold=5.0, dt=DT):
    """【修订口径 D1-ev】同 D1，但窗在**下一个真实事件**处截断（要求可用窗 ≥min_hold）。

    作用：让多次变载的实录类工况也能给出 T_stable（既有 30 s 口径在实录上恒删失）。
    返回 (T, censored, 实际用到的窗长 s)。
    """
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


def t_settle(tu, Y, k0, ref_amp, frac, z_final, cut_k=None, dt=DT):
    """【候选 D2】首次进入并保持 |Y−Z_final| ≤ frac×|J_ref|（保持到 cut/记录末）。"""
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


def t_band(tu, Y, k0, ref_amp, frac, target, hold=30.0, dt=DT):
    """【候选 D3】进入真值带 ±frac×J 并保持 hold（`11-paper-v6/pv_common.settle_time` 口径）。"""
    n = len(Y)
    H = int(round(hold / dt))
    tol = frac * abs(ref_amp)
    if not np.isfinite(ref_amp) or tol <= 0 or not np.isfinite(target):
        return np.nan, True
    ok = np.abs(Y - target) <= tol
    for k in range(k0, n):
        e = min(n, k + H)
        if e - k < min(H, n - k0):
            if bool(ok[k:e].all()):
                return float(tu[k] - tu[k0]), False
            break
        if bool(ok[k:e].all()):
            return float(tu[k] - tu[k0]), False
    return np.nan, True


def unload_index(y, k, dt=DT):
    """加载沿之后的最大负跳变（卸载沿）；逐字复制自 r4_filter_tradeoff.py。"""
    w = max(1, int(0.10 / dt))
    j0, j1 = k + int(3.0 / dt), len(y) - 1
    d = y[j0 + w:j1] - y[j0:j1 - w]
    if len(d) == 0:
        return None
    return j0 + int(np.argmin(d))


def first_onset_edge(tot, dt=DT):
    """11-paper-v6 的"回溯真沿"：首个超过 50% 峰值帧回退到 5% 增量处。逐字复制 pv_common。"""
    peak = float(np.percentile(tot, 99.5))
    idx = np.where(tot > 0.5 * peak)[0]
    if not len(idx):
        return None, np.nan
    i = int(idx[0])
    pre = float(np.median(tot[max(0, i - int(1.5 / dt)):max(1, i - int(0.3 / dt))]))
    j = i
    while j > 0 and tot[j] > pre + 0.05 * (tot[i] - pre):
        j -= 1
    return j + 1, pre


def next_event_cut(times, t_on, peak, min_frac=0.02):
    """下一个"真实事件"的时刻：t 严格晚于本事件 且 |J_tot| ≥ min_frac×记录峰值（脏/干净界）。"""
    cand = [t for t, j in times
            if t > t_on + 1e-9 and j is not None and np.isfinite(j) and abs(j) >= min_frac * peak]
    return float(min(cand)) if cand else None


def fill_rate_summary(v):
    """n<20 报中位 + p10~p90；n≤3 标"仅定性参考"。返回 dict，供汇总脚本统一填充。"""
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    n = int(a.size)
    if n == 0:
        return dict(n=0, med=np.nan, p10=np.nan, p90=np.nan, worst=np.nan, note="无可测样本")
    d = dict(n=n, med=float(np.median(a)), p10=float(np.percentile(a, 10)),
             p90=float(np.percentile(a, 90)), worst=float(np.max(a)), note="")
    if n <= 3:
        d["note"] = "仅定性参考(n<=3)"
    return d


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


def start_log(tag):
    """把 stdout 同时写进 results/_t1a_<tag>.log，并在头部记录运行命令。"""
    import datetime
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, "_t1a_%s.log" % tag)
    sys.stdout = _Tee(p)
    print("==== T1-A %s ==== %s" % (tag, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("cmd : %s %s" % (sys.executable, " ".join(sys.argv)))
    print("ROOT: %s" % ROOT)
    print("脚本目录: %s" % HERE)
    return p

