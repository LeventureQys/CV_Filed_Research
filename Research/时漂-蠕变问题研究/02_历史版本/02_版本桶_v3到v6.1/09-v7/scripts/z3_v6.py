# -*- coding: utf-8 -*-
"""v6：首次 onset 用固定 5s 免责期（与恒载验证过的 v4 一致），负载内变载不做冻结。

v5 的自适应收尾在变载事件上表现好（增益 0.984、最差 0.945），但把「低电平→高电平」
的首次 onset 也改成自适应后，在恒载上可能退回到 v3 的行为（A 参考窗又落到快相里）。
v6 取两者之长：首次 onset 固定冻结 5s（恒载已验证），变载事件不冻结（变载已验证）。
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
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4
_z2 = importlib.import_module("z2_v5")
GLM53v5 = _z2.GLM53v5


class GLM53v6(GLM53v3):
    """首次 onset（空载→负载）固定 5s 免责；负载内变载完全交给 restep。"""
    FAST_S = 5.0
    A_W0 = 3.5
    A_W1 = 5.0

    def _begin(self, ts):
        super()._begin(ts)
        first = not getattr(self, "ever_loaded_", False)
        self.ever_loaded_ = True
        self.fast_on_ = first
        self.fast_done_ = not first

    def process(self, ts, v):
        if not self.in_load:
            return super().process(ts, v)
        u = ts - self.onset_ts
        if self.fast_on_ and not self.fast_done_:
            if u < self.FAST_S:
                self.a_captured = False
                self.a_acc = np.zeros(self.n)
                self.a_frames = 0
                self.g = 0.0
                self.g2 = 0.0
                self.g_rel = np.zeros(self.n)
                self.gamma = np.ones(self.n)
                if self.A_W0 <= u <= self.A_W1:
                    self.a_acc = self.a_acc + (v - self.b)
                    self.a_frames += 1
                return v - self.b
            self.fast_done_ = True
            self.a_captured = True
            if self.a_frames > 0:
                self.A = self.a_acc / self.a_frames
                amax = self.A.max()
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                    else np.zeros(self.n, bool)
            else:
                self.A = v - self.b
                amax = self.A.max()
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                    else np.zeros(self.n, bool)
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


rows = []
print("=" * 100)
print("v6 分段策略（首次 onset 固定免责 / 变载不冻结）  ·  scripts/z3_v6.py")
print("=" * 100)
for name, tag in CASES:
    t, X = load_rec(os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    ev = _z2.detect_events(tot, dt)
    Ys = {"v3": run(tu, Xu, GLM53v3),
          "v5_adaptive": run(tu, Xu, GLM53v5),
          "v6_split": run(tu, Xu, GLM53v6)}
    print(f"\n=== {tag} ===")
    print(f"  {'事件':>8}{'原始跳变':>10}|" + "".join(f"{k:>26}" for k in Ys))
    print(f"  {'':>8}{'':>10}|" + "".join(f"{'延迟':>8}{'增益':>8}{'超额':>10}" for _ in Ys))
    for e, dl in ev:
        xj = np.median(tot[e:e + int(2 / dt)]) - np.median(tot[max(0, e - int(2 / dt)):e])
        line = f"  {tu[e]:8.2f}{xj:10.0f}|"
        for lab, Y in Ys.items():
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
            rows.append(dict(dataset=tag, event_s=float(tu[e]), raw_jump=float(xj), algo=lab,
                             lag_s=lag, gain=gain, excess=exc, level_err=float(post_off - pre_off)))
        print(line)

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "varying_steps_v6.csv"), index=False, encoding="utf-8-sig")
agg = dfm.groupby("algo").agg(事件数=("event_s", "count"), 阶跃增益=("gain", "mean"),
                              增益最差=("gain", "min"), 跳变超额_整阵=("excess", "mean"),
                              跳变超额_最大=("excess", "max"), 事件后电平误差=("level_err", "mean"),
                              延迟_s=("lag_s", "mean")).round(3)
print("\n" + "=" * 100)
print(agg.to_string())
print("\nsaved: results/varying_steps_v6.csv")
