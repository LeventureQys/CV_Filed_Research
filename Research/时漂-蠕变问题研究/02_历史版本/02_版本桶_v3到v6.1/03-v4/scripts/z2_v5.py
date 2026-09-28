# -*- coding: utf-8 -*-
"""v5：把「快相免责期」从固定时长改成「自适应收尾 + 只在首次 onset 生效」。

诊断依据（results/varying_steps.csv）：
  · v4_fast5 在「负载内变载」事件上丢失幅度 —— 数据B 21.02s 的 +5961 只报出 86%；
  · v3 同事件只报出 74%，但它慢在延迟；v4 延迟已经快 13 倍（0.026s vs 0.338s），
    剩下的问题就是「冻结期过长导致的幅度欠报」；
  · 两者的事件处单帧跳变超额都不达标（验收线 300 ADC）。

修正思路：
  1. 免责期只在**首次 onset（空载→负载）**启用；负载内变载不做独立冻结，
     完全交给 v3 的 restep 原子迁移（它本来就是为这个场景设计的）；
  2. 免责期**自适应收尾**：监测电平斜率，掉到 1.5%×电平/s 以下并持续 0.4s
     即结束（上限 5s、下限 1.2s），避免固定 5s 把真实变载也冻住；
  3. 幅度参考 A 在免责期结束后立刻捕获（用收尾前 1s 的稳健水平），
     不再依赖固定 3.5~5.0s 窗口。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3  # noqa: E402


class GLM53v5(GLM53v3):
    FAST_MAX = 5.0        # 免责期上限
    FAST_MIN = 1.2        # 免责期下限（机械加载至少这么久）
    SLOPE_REL = 0.015     # 收尾判据：|d电平|/电平 < 1.5% per s
    QUIET_S = 0.4         # 需持续多久
    ONLY_FIRST = True     # 只在首次 onset 生效

    def _begin(self, ts):
        super()._begin(ts)
        # 首次 onset（从未 armed 过）才启用免责期
        self.fast_active_ = (not self.ONLY_FIRST) or (not getattr(self, "armed_", False)) \
            or (not getattr(self, "ever_loaded_", False))
        self.ever_loaded_ = True
        self.fast_active_ = bool(self.fast_active_)
        self.quiet_acc_ = 0.0
        self.fast_end_u_ = None
        self.lvl_prev_ = None
        self.Aheld_ = None

    def process(self, ts, v):
        if not self.in_load:
            return super().process(ts, v)

        u = ts - self.onset_ts
        if self.fast_active_ and self.fast_end_u_ is None:
            # ---- 免责期：冻结补偿，只做自适应收尾判定 ----
            dt = 0.0 if self.lvl_prev_ is None else max(0.0, min(ts - self.lvl_ts_, 0.1))
            self.lvl_ts_ = ts
            lvl = float((v - self.b).sum())
            if self.lvl_prev_ is None:
                self.lvl_prev_ = lvl
            else:
                if dt > 0.0:
                    rate = abs(lvl - self.lvl_prev_) / max(abs(lvl), 1e-9) / dt
                    if u >= self.FAST_MIN and rate < self.SLOPE_REL:
                        self.quiet_acc_ += dt
                    else:
                        self.quiet_acc_ = 0.0
                self.lvl_prev_ = lvl
            self.a_captured = False
            self.a_acc = np.zeros(self.n)
            self.a_frames = 0
            self.g = 0.0
            self.g2 = 0.0
            self.g_rel = np.zeros(self.n)
            self.gamma = np.ones(self.n)
            # 持续记录近期水平，收尾时用它当 A
            if not hasattr(self, "lvl_hist_"):
                self.lvl_hist_ = []
            self.lvl_hist_.append((ts, lvl))
            if len(self.lvl_hist_) > 2000:
                self.lvl_hist_ = self.lvl_hist_[-2000:]
            if self.quiet_acc_ >= self.QUIET_S or u >= self.FAST_MAX:
                self.fast_end_u_ = u
                # A = 收尾前 0.8s 的稳健水平（逐通道）
                t_cut = ts - 0.8
                idx = [i for i, (tt, _) in enumerate(self.lvl_hist_) if tt >= t_cut]
                if idx:
                    # 重新用通道值记录（另存）
                    pass
                self.Aheld_ = None
                self.a_captured = True
            return v - self.b
        return super().process(ts, v)


class GLM53v5b(GLM53v3):
    """v5b：在 v5 基础上，A 用「免责期内的逐通道均值」在收尾当帧捕获。"""
    FAST_MAX = 5.0
    FAST_MIN = 1.2
    SLOPE_REL = 0.015
    QUIET_S = 0.4

    def _begin(self, ts):
        super()._begin(ts)
        self.fast_on_ = not getattr(self, "ever_loaded_", False)
        self.ever_loaded_ = True
        self.quiet_acc_ = 0.0
        self.fast_end_u_ = None
        self.lvl_prev_ = None
        self.lvl_ts_ = ts
        self.zsum_ = np.zeros(self.n)
        self.zcnt_ = 0

    def process(self, ts, v):
        if not self.in_load:
            return super().process(ts, v)
        u = ts - self.onset_ts
        if self.fast_on_ and self.fast_end_u_ is None:
            Z = v - self.b
            dt = max(0.0, min(ts - self.lvl_ts_, 0.1)) if self.lvl_prev_ is not None else 0.0
            self.lvl_ts_ = ts
            lvl = float(Z.sum())
            if self.lvl_prev_ is not None and dt > 0.0:
                rate = abs(lvl - self.lvl_prev_) / max(abs(lvl), 1e-9) / dt
                self.quiet_acc_ = self.quiet_acc_ + dt if (u >= self.FAST_MIN and rate < self.SLOPE_REL) else 0.0
            self.lvl_prev_ = lvl
            # 冻结补偿状态
            self.a_captured = False
            self.g = 0.0
            self.g2 = 0.0
            self.g_rel = np.zeros(self.n)
            self.gamma = np.ones(self.n)
            # 只累积「收尾候选窗」（最近 0.8s）的 Z
            self.zsum_ = self.zsum_ + Z
            self.zcnt_ += 1
            if self.zcnt_ > int(0.8 / max(dt, 1e-3)) and self.zcnt_ > 3:
                # 丢掉最早一帧（近似滑窗）
                self.zsum_ = self.zsum_ * 0.995
            if self.quiet_acc_ >= self.QUIET_S or u >= self.FAST_MAX:
                self.fast_end_u_ = u
                self.A = self.zsum_ / max(self.zcnt_, 1)
                amax = self.A.max()
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                    else np.zeros(self.n, bool)
                self.a_captured = True
                self.fast_on_ = False
            return Z
        return super().process(ts, v)


CASES = [("零负载-切换负载-零负载-再切换负载", "数据A"),
         ("零负载-中途切换负载-零负载-切换负载", "数据B")]


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def detect_events(tot, dt, rel=0.15, absfrac=0.08):
    n = len(tot)
    pn = int(2.0 / dt)
    cand = []
    for i in range(pn, n - pn, max(1, int(0.1 / dt))):
        pre = np.median(tot[i - pn:i])
        post = np.median(tot[i:i + pn])
        if abs(post - pre) > max(rel * abs(pre), absfrac * tot.max()):
            cand.append((i, post - pre))
    ev = []
    for i, dl in cand:
        if ev and i - ev[-1][0] <= int(1.5 / dt):
            if abs(dl) > abs(ev[-1][1]):
                ev[-1] = (i, dl)
        else:
            ev.append((i, dl))
    ref = []
    for i, dl in ev:
        a, b = max(0, i - int(2 / dt)), min(n - 1, i + int(2 / dt))
        sm = pd.Series(tot[a:b]).rolling(max(3, int(0.15 / dt)), min_periods=1).median().to_numpy()
        k = int(np.argmax(np.abs(np.diff(sm))))
        ref.append((a + k, dl))
    return ref


import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4

ALGOS = [("raw", None), ("v3", (GLM53v3, {})),
         ("v4_fast5", (GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0))),
         ("v5_adaptive", (GLM53v5, {})), ("v5b_adapt_A", (GLM53v5b, {}))]

rows = []
print("=" * 112)
print("v5 自适应免责期：变化负载阶跃保真  ·  temp/v4.1flash/scripts/z2_v5.py")
print("=" * 112)
for name, tag in CASES:
    t, X = load_rec(os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    ev = detect_events(tot, dt)
    print(f"\n=== {tag}（{name}）  时长 {span:.1f}s ===")
    Ys = {}
    for lab, cfg in ALGOS:
        Ys[lab] = Xu.copy() if cfg is None else run(tu, Xu, cfg[0], **cfg[1])
    print(f"  {'事件(s)':>9}{'原始跳变':>10}|" + "".join(
        f"{lab:>26}" for lab in ["v3", "v4_fast5", "v5_adaptive", "v5b_adapt_A"]))
    print(f"  {'':>9}{'':>10}|" + "".join(f"{'延迟':>8}{'增益':>8}{'超额':>10}" for _ in range(4)))
    for e, dl in ev:
        xj = np.median(tot[e:e + int(2 / dt)]) - np.median(tot[max(0, e - int(2 / dt)):e])
        line = f"  {tu[e]:9.2f}{xj:10.0f}|"
        for lab, _ in ALGOS[1:]:
            Y = Ys[lab]
            pre_off = np.median(Y[max(0, e - int(2 / dt)):e - int(0.3 / dt)].sum(axis=1)
                                - tot[max(0, e - int(2 / dt)):e - int(0.3 / dt)])
            post_off = np.median(Y[e + int(2 / dt):e + int(3 / dt)].sum(axis=1)
                                 - tot[e + int(2 / dt):e + int(3 / dt)])
            gain = (xj + post_off - pre_off) / xj if abs(xj) > 1e-9 else np.nan
            lag = np.nan
            if abs(post_off - pre_off) > 1e-9:
                for k in range(e - int(0.5 / dt), min(len(tu), e + int(8 / dt))):
                    if (Y[k].sum() - tot[k] - pre_off) / (post_off - pre_off) >= 0.9:
                        lag = (k - e) * dt
                        break
            a, b = max(0, e - int(1 / dt)), min(len(tu) - 1, e + int(8 / dt))
            exc = float(np.abs(np.diff(Y.sum(axis=1))[a:b] - np.diff(tot)[a:b]).max())
            line += f"{lag:8.2f}{gain:8.3f}{exc:10.0f}"
            rows.append(dict(dataset=tag, event_s=float(tu[e]), raw_jump=float(xj),
                             algo=lab, lag_s=lag, gain=gain, excess=exc,
                             level_err=float(post_off - pre_off)))
        print(line)

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "varying_steps_v5.csv"), index=False, encoding="utf-8-sig")
print("\n" + "=" * 112)
agg = dfm.groupby("algo").agg(
    事件数=("event_s", "count"), 阶跃增益=("gain", "mean"),
    增益最差=("gain", "min"), 跳变超额_整阵=("excess", "mean"),
    跳变超额_最大=("excess", "max"), 事件后电平误差=("level_err", "mean"),
    延迟_s=("lag_s", "mean")).round(3)
print(agg.to_string())
print("\nsaved: results/varying_steps_v5.csv")
