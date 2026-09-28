# -*- coding: utf-8 -*-
"""T5-A 公共层：路径锚点 / 录制清单 / κ 可调原型 / 过充事件集 / T1 口径指标。

本模块是 T5-A（过充/下冲归因 + 统一 κ 扫描）的唯一公共层。设计要点：

1. **路径锚点**：一切路径由 `__file__` 推导（本文件在 `<PLAN>/T5_*/scripts` 下，上溯 4 级 = 仓库根）。
2. **κ 可调原型**：`TracedV6K` / `TracedV61K` 由 `t5a_a_common.make_traced` 包一层，
   再套一个 `_KappaSet` 子类，把 `KAPPA_ONSET / KAPPA_RESTEP` 变成**实例可设置**的参数。
   `κ_onset=1.30, κ_restep=1.12` 时与 v6 默认行为**逐帧相同**（`t5a_patch_ab_zero.csv` 给 A/B 零差证明）。
3. **事件集**：直接读 T4-A 冻结表 `results/t4a_morphology.csv`（60 事件，含 `t_on`），
   **不重新发明事件检测**，保证与 T4-A / T4-B 的事件原点同口径；本模块只做子集筛选与幅度重算。
4. **指标**：`OS% / T_stable / MD / err_1s / G` 全部按《指标字典与口径》§3 定义实现，
   `OS%` 用任务书写死的 `Z_final = t_on+60 s 后中位`（不足则用后 1/3 段中位并标记）。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))                    # T5_*/scripts
TASK = os.path.dirname(HERE)                                          # T5_*
PLAN = os.path.dirname(TASK)                                          # plan/v1.0
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))    # 仓库根
TEMP = os.path.join(ROOT, "temp")
PROG = os.path.join(TEMP, "v4.1flash", "progress")
T4DIR = os.path.join(PLAN, "T4_两种快相形态与分支判据")
for p in (HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)

import t5a_a_common as AC                                            # noqa: E402
from t5a_a_common import load_uniform, make_traced, run_traced, run_plain  # noqa: E402
from t5a_glm53_v6 import GLM53v6                                     # noqa: E402
from t5a_glm53_v61 import GLM53v61                                   # noqa: E402

# ───────────────────────── 13 份录制清单 ─────────────────────────
# 恒载 9 组 = processed_display 显示域；变载实录 4 份 = ADC 域（跨域不可比绝对值）
RECS = [
    # key, 相对 TEMP 的路径, 域
    ("RT1", r"右拇指指尖\数据1\device_001_seg000.csv", "显示域"),
    ("RT2", r"右拇指指尖\数据2\device_001_seg000.csv", "显示域"),
    ("RT3", r"右拇指指尖\数据3\device_001_seg000.csv", "显示域"),
    ("LT1", r"左拇指指尖\数据1\device_001_seg000.csv", "显示域"),
    ("LT2", r"左拇指指尖\数据2\device_001_seg000.csv", "显示域"),
    ("LT3", r"左拇指指尖\数据3\device_001_seg000.csv", "显示域"),
    ("F41", r"四指指尖\数据1\device_001_seg000.csv", "显示域"),
    ("F42", r"四指指尖\数据2\device_001_seg000.csv", "显示域"),
    ("F43", r"四指指尖\数据3\device_001_seg000.csv", "显示域"),
    ("SW1", r"变化负载\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc\device_001_seg000.csv", "ADC域"),
    ("SW2", r"变化负载\零负载-切换负载-零负载-再切换负载\device_001_seg000.csv", "ADC域"),
    ("SW3", r"变化负载\零负载-中途切换负载-零负载-切换负载\device_001_seg000.csv", "ADC域"),
    ("SW4", r"变化负载\零负载-中途切换负载-零负载-切换负载\最终测试目标\device_001_seg000.csv", "ADC域"),
]
REC_PATH = {k: os.path.abspath(os.path.join(TEMP, p)) for k, p, _ in RECS}
REC_DOM = {k: d for k, _, d in RECS}
# T4-A 事件表的 rec 列名 → 本任务 key
T4A_KEY = {
    "右拇指指尖/数据1": "RT1", "右拇指指尖/数据2": "RT2", "右拇指指尖/数据3": "RT3",
    "左拇指指尖/数据1": "LT1", "左拇指指尖/数据2": "LT2", "左拇指指尖/数据3": "LT3",
    "四指指尖/数据1": "F41", "四指指尖/数据2": "F42", "四指指尖/数据3": "F43",
    "切换负载-快相无责": "SW1", "再切换负载": "SW2",
    "中途切换-1d9493": "SW3", "中途切换-13ffca": "SW4",
}

_T4A_CACHE = {}
# T_stable 的最小有效窗长（秒）：口径 = 「此后 |显示 − Z_final| ≤ 5%·J 且不再回越」，
# 段末不足 30 s 时窗被截断；有效窗 < 10 s 视为不可测（记 NaN）。
T_MIN_WIN_S = 10.0


def t4a_events():
    """读 T4-A 冻结事件表（只读引用）。返回 DataFrame，新增 `key` 列。"""
    p = os.path.join(T4DIR, "results", "t4a_morphology.csv")
    df = pd.read_csv(p)
    df["key"] = df["rec"].map(T4A_KEY)
    return df


def recordings():
    """逐份读入 100 Hz 网格，带进程内缓存 key -> dict。"""
    if _T4A_CACHE.get("_recs") is None:
        cache = {}
        for k, rel, dom in RECS:
            d = load_uniform(REC_PATH[k], 100.0)
            d["dom"] = dom
            d["Z"] = d["Xu"].sum(axis=1)
            cache[k] = d
        _T4A_CACHE["_recs"] = cache
    return _T4A_CACHE["_recs"]


# ───────────────────────── κ 可调原型 ─────────────────────────

def _make_kappa_cls(cls_name, base):
    """把 KAPPA_ONSET / KAPPA_RESTEP 变成实例属性可写。默认值 = 原型值。"""
    Traced = make_traced(base)

    class K(Traced):
        def __init__(self, n):
            super().__init__(n)
            self.kappa_onset = type(self).KAPPA_ONSET
            self.kappa_restep = type(self).KAPPA_RESTEP

        def run(self, tu, X):
            """就地跑一遍并返回仪表化字典（用实例已设好的参数）。"""
            n = len(tu)
            Y = np.empty_like(X)
            tot = np.empty(n)
            ev_flag = np.zeros(n, bool)
            st_code = np.zeros(n, np.int8)
            kappa_hist = np.full(n, np.nan)
            ahat_hist = np.full(n, np.nan)
            inc_hist = np.full(n, np.nan)
            g_hist = np.full(n, np.nan)
            for i in range(n):
                Y[i] = self.process(tu[i], X[i])
                tot[i] = X[i].sum()
                ev_flag[i] = self.ev is not None
                st_code[i] = {"idle": 0, "event": 1, "slow": 2}[self.state]
                if self.ev is not None:
                    ahat_hist[i] = self.ev["A_hat"]
                    inc_hist[i] = tot[i] - self.ev["base"]
                    kappa_hist[i] = (self.kappa_onset if self.ev["kind"] == "onset"
                                     else self.kappa_restep)
                g_hist[i] = self.g
            return dict(Y=Y, tot=tot, ev=ev_flag, st=st_code, comp=self,
                        kappa=kappa_hist, ahat=ahat_hist, inc=inc_hist, g=g_hist,
                        epoch=self.tr_epoch, revoke=self.tr_revoke,
                        handoff=self.tr_handoff, unload=self.tr_unload,
                        gevent=self.tr_gevent, A=self.A.copy(), gamma=self.gamma.copy())

        def process(self, ts, v):
            self.KAPPA_ONSET = self.kappa_onset
            self.KAPPA_RESTEP = self.kappa_restep
            return super().process(ts, v)

    K.__name__ = cls_name
    return K


KV6 = _make_kappa_cls("T5AKV6", GLM53v6)
KV61 = _make_kappa_cls("T5AKV61", GLM53v61)


def run_v6(d, kappa_onset=1.30, kappa_restep=1.12, rom_scale=None, rate_max=None,
           rate_down=None, ho_min=None, cls=None, X=None):
    """在 100 Hz 网格上跑一遍 κ 可调 v6（或 v6.1），返回仪表化字典。"""
    c = (cls or KV6)((X if X is not None else d["Xu"]).shape[1])
    c.kappa_onset = float(kappa_onset)
    c.kappa_restep = float(kappa_restep)
    if rom_scale is not None:
        c.ROM_SCALE = float(rom_scale)
    if rate_max is not None:
        c.RATE_MAX = float(rate_max)
    if rate_down is not None:
        c.RATE_DOWN = float(rate_down)
    if ho_min is not None:
        c.HO_MIN = float(ho_min)
    return c.run(d["tu"], X if X is not None else d["Xu"])


def run_v6_arm(d, arm):
    """按 arm 字典跑一遍（arm 含 cls=KV6/KV61 与各参数；`name/raw` 是元信息，不传给原型）。"""
    a = dict(arm)
    a.pop("name", None)
    a.pop("id", None)
    a.pop("raw", None)
    cls = a.pop("cls", KV6)
    return run_v6(d, cls=cls, **a)


# ───────────────────────── 事件与指标 ─────────────────────────

def clip_win(tu, t0, t1):
    i0 = int(np.searchsorted(tu, t0))
    i1 = int(np.searchsorted(tu, t1))
    return i0, min(i1, len(tu))


def z_final_of(Z, tu, t0, span, variant="v61"):
    """参考电平 `Z_final`。

    **为什么不用任务书的字面规则**（这是本任务的显式口径偏离，已在报告 §④ 登记）：
    字面规则「`Z_final` = `t_on+60 s` 之后的中位」在**恒载 9 组**上会落到**卸载之后**
    （实测：恒载 onset 在 8~15 s，卸载在 115~147 s ⇒ `t_on+60 s` 之后整段是空载），
    于是 `Z_final ≈ 0`，`OS% = (显示 − 0)/J ≈ +100%` —— 那是口径伪影，不是过充。
    （未修前实测：RT1@8.27 s OS%=+100.8、F41@10.94 s OS%=+101.1。）

    因此采用 **`08-v6.1` 已有口径**（`results/v61_overshoot.csv` 的 `P` 列，全项目已有数字都用它）：
        `variant="v61"`  ⇒ `Z_final` = 原始总量在 `[t_on+4.6, t_on+5.4] s` 的中位
    并同时给出任务书字面口径 `variant="task_literal"` 供对照（列 `OS_pct_task_literal`）。

    > 对 **unload / partial_unload**（台阶为负）`Z_final` 是卸载后的低电平，OS% 定义不适用；
    > 这两类改用 `metrics_at` 的 `dev_pre_pct`（显示相对事件前电平的有符号偏离 ÷ |J|）。
    """
    if variant == "task_literal":
        i0 = int(np.searchsorted(tu, t0 + 60.0))
        if (span - t0) >= 60.0 and i0 < len(Z) - 5:
            return float(np.median(Z[i0:])), "t0+60s后中位"
        j0 = int(len(Z) * 2.0 / 3.0)
        return float(np.median(Z[j0:])), "后1/3段中位(不足60s)"
    a, b = clip_win(tu, t0 + 4.6, t0 + 5.4)
    if b - a < 5:
        a, b = clip_win(tu, t0 + 4.0, min(span, t0 + 6.0))
    if b <= a:
        return float(Z[-1]), "段末帧(退化)"
    return float(np.median(Z[a:b])), "t_on+4.6~5.4s中位(=08-v6.1 P 口径)"


def j_and_pre(Zs, tu, t0, span):
    """台阶幅度 J 与 pre 电平：pre 窗 [t0-2, t0)，post 窗 [t0+4, t0+6]（指标字典 §2.1）。"""
    a0, a1 = clip_win(tu, t0 - 2.0, t0)
    b0, b1 = clip_win(tu, t0 + 4.0, t0 + 6.0)
    if b1 - b0 < 10:
        b0, b1 = clip_win(tu, t0 + 4.0, min(span, t0 + 6.0))
    pre = float(np.median(Zs[a0:a1])) if a1 > a0 else float(Zs[0])
    post = float(np.median(Zs[b0:b1])) if b1 > b0 else float(Zs[-1])
    return pre, post, post - pre


def _tstab_scan(seg, tu, i0s, t0, nwin, thr):
    """T_stable 的**穷举完整窗**实现（与 `t5a_metric_selfcheck.py` 的参考实现同语义）。

    只接受 `τ + win ≤ 段末` 的完整窗，返回最早的安静起点；找不到 ⇒ NaN。

    为什么必须穷举而不是只扫前 nwin 个起点：T_stable 允许把窗推迟到**载荷已经稳定**的
    任何位置（例如卸载后空载段），早期起点不安静并不意味着答案不存在
    —— 这一条正是被 `t5a_metric_selfcheck.py` 的对拍抓出来的（第一版有界搜索漏报 2/60）。
    """
    n = len(seg)
    lim = n - nwin
    for i in range(i0s, lim + 1):
        if seg[i:i + nwin].max() <= thr:
            return float(tu[i] - t0)
    return np.nan


def metrics_at(Zs, Ys, tu, t0, span, win_s=30.0, eps_frac=0.05, t_win_s=30.0):
    """单事件指标（口径见《指标字典与口径》§3 与 `z_final_of`）。

    · OS%  = max_t (Ys(t) − Zf) / J，t ∈ [t_on, t_on + win_s]；负值 = 下冲
    · US%  = −min_t (Ys(t) − Zf) / J（下冲幅度，正数）
    · T_stable = 最早 τ 使 `[τ, τ+t_win_s]` 内 |Ys − Zf| ≤ 5%·|J|（**只接受完整窗**；
                 放不下 ⇒ NaN，并由 `T_stable_10` 给 10 s 水平窗的版本）
    · MD   = 窗内 max|Ys − Zs|（ADC，与电平同报）
    · err_1s = (Ys(t0+1) − Ys(t0)) / J − 1
    · G(20 s) = (Ys(t0+20) − Ys(t0)) / (Zs(t0+20) − Zs(t0))
    """
    n = len(Zs)
    pre, post, J = j_and_pre(Zs, tu, t0, span)
    Zf, zf_mode = z_final_of(Zs, tu, t0, span, variant="v61")
    Zf_task, zf_task_mode = z_final_of(Zs, tu, t0, span, variant="task_literal")
    i0, i1 = clip_win(tu, t0, min(span, t0 + win_s))
    i0 = min(i0, n - 1)
    y = Ys[i0:i1]

    def os_us(upto):
        """在 [t0, t0+upto] 内算 OS% / US%（相对 Zf 与 J）。"""
        a, b = clip_win(tu, t0, min(span, t0 + upto))
        a = min(a, n - 1)
        yy = Ys[a:b]
        if not len(yy) or abs(J) <= 1e-12:
            return np.nan, np.nan
        d = (yy - Zf) / J
        return float(d.max()) * 100.0, float(-d.min()) * 100.0

    os5, us5 = os_us(5.0)          # 与 08-v6.1 台账同窗长（可直接对表）
    os30, us30 = os_us(win_s)      # 《指标字典与口径》§3 写死的 30 s 窗
    os_pct, us_pct = os30, us30
    # T_stable（完整窗，主口径 t_win_s=30 s；另给 10 s 水平窗版本）
    dt = float(tu[1] - tu[0])
    seg = np.abs(Ys - Zf)
    thr = eps_frac * abs(J)
    i0s = min(int(np.searchsorted(tu, t0)), n - 1)

    def tstab(win):
        nw = int(round(win / dt))
        if nw < 2 or (n - i0s) < nw:
            return np.nan
        return _tstab_scan(seg, tu, i0s, t0, nw, thr)

    tstab30 = tstab(30.0)
    tstab10 = tstab(10.0)
    tstab_main = tstab(t_win_s)
    t_trunc = bool(np.isnan(tstab30) and not np.isnan(tstab10))
    # MD 与台阶捕获比
    md = float(np.abs(Ys[i0:i1] - Zs[i0:i1]).max()) if i1 > i0 else np.nan
    a0 = min(int(np.searchsorted(tu, t0)), n - 1)
    b0 = min(int(np.searchsorted(tu, min(span - 1e-9, t0 + 20.0))), n - 1)
    raw_gain = float(Zs[b0] - Zs[a0])
    y_gain = float(Ys[b0] - Ys[a0])
    G = (y_gain / raw_gain) if abs(raw_gain) > 1e-9 else np.nan
    j1 = min(int(np.searchsorted(tu, min(span - 1e-9, t0 + 1.0))), n - 1)
    err1 = ((float(Ys[j1] - Ys[a0]) / J) - 1.0) * 100.0 if abs(J) > 1e-12 else np.nan
    # 卸载/部分卸载专用：显示相对事件前电平的**有符号**偏离 ÷ |J|
    dev_pre = (y - pre) / abs(J) * 100.0 if abs(J) > 1e-12 else np.zeros_like(y)
    # 显示−原始 的有符号差（用户直接看得见的量）
    dd = Ys[i0:i1] - Zs[i0:i1] if i1 > i0 else np.array([])
    return dict(pre=pre, post=post, J=J, Z_final=Zf, zf_mode=zf_mode,
                Z_final_task=Zf_task, zf_task_mode=zf_task_mode,
                OS_pct=os_pct, OS_pct_5s=os5, OS_pct_30s=os30,
                US_pct=us_pct, US_pct_5s=us5, US_pct_30s=us30,
                OS_pct_task_literal=(
                    float(((y - Zf_task) / J).max()) * 100.0
                    if (len(y) and abs(J) > 1e-12) else np.nan),
                T_stable=tstab_main, T_stable30=tstab30,
                T_stable10=tstab10, T_stable_trunc=t_trunc,
                MD=md, G20=G, err_1s_pct=err1,
                dev_mid_max_adc=float(dd.max()) if len(dd) else np.nan,
                dev_mid_min_adc=float(dd.min()) if len(dd) else np.nan,
                dev_pre_pct_max=float(dev_pre.max()) if len(dev_pre) else np.nan,
                dev_pre_pct_min=float(dev_pre.min()) if len(dev_pre) else np.nan,
                peak_dev_adc=float((y - Zf).max()) if len(y) else np.nan,
                trough_dev_adc=float((y - Zf).min()) if len(y) else np.nan)


def run_and_measure(d, arm, ev):
    """跑一遍算法，返回 (仪表化字典, 逐事件指标表)。arm 里 key='raw' 时用原始信号。"""
    if arm.get("raw"):
        Y = d["Xu"].copy()
        out = dict(Y=Y, comp=None, epoch=[], revoke=[], handoff=[], unload=[], gevent=[],
                   A=None, gamma=None, kappa=None, ahat=None, inc=None, g=None,
                   st=None, ev=None)
    else:
        out = run_v6_arm(d, arm)
    Zs = d["Z"]
    Ys = out["Y"].sum(axis=1)
    rows = []
    for _, e in ev.iterrows():
        m = metrics_at(Zs, Ys, d["tu"], float(e["t_on"]), d["span"])
        m.update(key=e["key"], t_on=float(e["t_on"]), kind=e["kind"], dom=d["dom"],
                 arm=arm.get("name", arm.get("id", "?")), clean=bool(e["clean"]))
        rows.append(m)
    return out, pd.DataFrame(rows)


def summarize(df, by=("arm", "kind"), col="OS_pct"):
    g = df.groupby(list(by))[col]
    return pd.DataFrame({"n": g.size(), "med": g.median(),
                         "p10": g.quantile(0.10), "p90": g.quantile(0.90)})


def ev_load_frozen():
    """T4-A 的 60 事件表（本任务的事件集接口）。"""
    df = t4a_events()
    return df[["key", "rec", "dom", "t_on", "kind", "clean", "pre", "post", "jump",
               "z_at_02", "z_at_10"]].copy()


def write_log(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
