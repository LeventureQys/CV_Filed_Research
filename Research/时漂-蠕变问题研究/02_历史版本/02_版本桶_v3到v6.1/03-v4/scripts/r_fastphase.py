# -*- coding: utf-8 -*-
"""验证「快相免责期」改造：把补偿推后到快相结束，只对抗慢相蠕变。

对照（全部在 temp 的恒载 9 组 + 变化负载 2 组上跑）：
  raw        原始
  v3         现有实现（快相与慢相一起处理；A 窗 1~3s，抑制窗 6s）
  v4_fast5   快相免责期 = onset 后 0~5s（冻结补偿 + 冻结状态估计），
             A 窗改为 3.5~5.0s，5.0s 起清零 g/γ 累加器，只对慢相增量建蠕变模型
  v4_fast3   同上但免责期 = 3s（比较免责期长短）
  v4_fast8   同上但免责期 = 8s

指标：与既有 GLM53 分析口径一致（时漂残余、噪声比、平坦度、零漂残余、阶跃保真、事件跳变超额），
另加「慢相段漂移」——只统计免责期结束后的区间，这才是本次改造要优化的目标量。
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(OUT, "figures")

DATASETS = [(loc, f"数据{i}") for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"] for i in (1, 2, 3)]
VARYING = [("变化负载", "零负载-切换负载-零负载-再切换负载"),
           ("变化负载", "零负载-中途切换负载-零负载-切换负载")]


class GLM53v4(GLM53v3):
    """在 v3 上加「快相免责期」：onset 后 FAST_S 秒内冻结补偿与状态估计。"""

    FAST_S = 5.0          # 免责期长度
    A_W0_V4 = 3.5         # 幅度采集窗起点（落在免责期末段）
    A_W1_V4 = 5.0         # 幅度采集窗终点 = 免责期终点

    def _begin(self, ts):
        super()._begin(ts)
        self.fast_done_ = False
        self.creep_started_ = False

    def process(self, ts, v):
        # 免责期内：完全冻结（不积分 g/γ、不更新 A、不累积），显示直通
        state = getattr(self, "fast_done_", True)
        u = (ts - self.onset_ts) if self.in_load else 0.0
        if self.in_load and not state and u < self.FAST_S:
            # 仍需要跑 v3 的状态机（事件检测/卸载判定），但把补偿冻结掉：
            # 用一个很短的 dt 让 EMA 正常演化，同时把 a_captured_ 置 False 并清空累加
            self.a_captured = False
            self.a_acc = np.zeros(self.n)
            self.a_frames = 0
            self.g = 0.0
            self.g2 = 0.0
            self.g_rel = np.zeros(self.n)
            self.gamma = np.ones(self.n)
            # 在免责期末段采集幅度
            if self.A_W0_V4 <= u <= self.A_W1_V4:
                self.a_acc = self.a_acc + (v - self.b)
                self.a_frames += 1
            return v - self.b          # 直通（仅扣零漂基线）
        if self.in_load and not state and u >= self.FAST_S:
            self.fast_done_ = True
            self.creep_started_ = True
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


def run(t, X, cls, **kw):
    c = cls(X.shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    Y = np.empty_like(X)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i])
    return Y


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def segs(total, frac=0.15):
    """返回全部负载区间（按长度降序）"""
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


ALGOS = {
    "raw": ("原始", None),
    "v3": ("现有 v3", dict(cls=GLM53v3, kw={})),
    "v4_fast3": ("快相免责 3s", dict(cls=GLM53v4, kw=dict(FAST_S=3.0, A_W0_V4=2.0, A_W1_V4=3.0))),
    "v4_fast5": ("快相免责 5s", dict(cls=GLM53v4, kw=dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0))),
    "v4_fast8": ("快相免责 8s", dict(cls=GLM53v4, kw=dict(FAST_S=8.0, A_W0_V4=6.0, A_W1_V4=8.0))),
}

if os.environ.get("FASTPHASE_WITH_V5") == "1":
    import importlib
    _z2 = importlib.import_module("z2_v5")
    ALGOS["v5_adaptive"] = ("v5 自适应免责", dict(cls=_z2.GLM53v5, kw={}))

rows = []
print("=" * 104)
print("「快相免责期」改造实测  ·  temp/v4.1flash/scripts/r_fastphase.py")
print("=" * 104)
for loc, name in DATASETS:
    t, X = load_rec(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    s0r, s1r = segs(X.sum(axis=1))[0]
    s0, s1 = int(np.searchsorted(tu, t[s0r])), int(np.searchsorted(tu, t[s1r]))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    loaded = amp > 0.10 * amp.max()
    tot = Xu.sum(axis=1)
    base = tot[:s0].mean()
    thr = base + 0.05 * (tot[s0:s1].max() - base)
    n_on = next((i for i in range(s0, s0 + 300) if tot[i] > thr), s0)
    nL = s1 - s0
    ys = {"raw": Xu.copy()}
    for key, (lab, cfg) in ALGOS.items():
        if cfg is None:
            continue
        ys[key] = run(tu, Xu, cfg["cls"], **cfg["kw"])
    # 指标：全负载段 + 慢相段（免责 5s 之后）
    for key, Y in ys.items():
        L = Y[s0:s1]
        by, bx = Y[:s0, m].mean(), Xu[:s0, m].mean()
        dr = L[-nL // 10:].mean(axis=0) - L[: nL // 10:].mean(axis=0)
        # 慢相段：onset+5s 到末端
        a5 = int(np.searchsorted(tu, tu[n_on] + 5.0))
        L5 = Y[max(a5, s0):s1]
        n5 = len(L5)
        dr5 = L5[-n5 // 10:].mean(axis=0) - L5[: n5 // 10:].mean(axis=0)
        i1, i2 = s0 + int(0.5 / dt), s0 + int(2.5 / dt)
        sx = Xu[i1:i2, m].mean() - bx
        step_ratio = ((Y[i1:i2, m].mean() - by) / sx) if abs(sx) > 1e-9 else np.nan
        tts = tu[s0:s1] - tu[s0]

        def dstd(sig, tq):
            k, b0 = np.polyfit(tq, sig, 1)
            return (sig - (k * tq + b0)).std()

        ny = dstd(L[10:, m], tts[10:])
        nx = dstd(Xu[s0 + 10:s1, m], tts[10:])
        j1, j2 = s1 + int(5.0 / dt), min(len(tu), s1 + int(30.0 / dt))
        dY, dX = np.diff(Y[:, m]), np.diff(Xu[:, m])
        rows.append(dict(location=loc, dataset=name, algo=key,
                         drift_main=100 * dr[m] / amp[m],
                         drift_loaded=100 * np.median(dr[loaded] / amp[loaded]),
                         drift_slow_main=100 * dr5[m] / amp[m],
                         noise_ratio=ny / nx if nx > 1e-12 else np.nan,
                         flat_main=100 * ny / amp[m],
                         zero_resid=(100 * (Y[j1:j2, m].mean() - by) / amp[m]) if j2 > j1 else np.nan,
                         step_ratio=step_ratio,
                         jump_excess=float(np.abs(dY - dX)[s0:s1].max())))
    print(f"[{loc}/{name}] " + "  ".join(
        f"{k}={100*(ys[k][s0:s1][-nL//10:, m].mean()-ys[k][s0:s1][:nL//10, m].mean())/amp[m]:+.1f}%"
        for k in ALGOS if k in ys))

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "fastphase_metrics.csv"), index=False, encoding="utf-8-sig")
LAB = {k: v[0] for k, v in ALGOS.items()}
agg = dfm.groupby("algo").agg(
    时漂残余_全段=("drift_main", lambda s: s.abs().mean()),
    时漂残余_慢相段=("drift_slow_main", lambda s: s.abs().mean()),
    时漂残余_受载中位=("drift_loaded", lambda s: s.abs().mean()),
    噪声比=("noise_ratio", "mean"),
    平坦度=("flat_main", "mean"),
    零漂残余=("zero_resid", lambda s: s.abs().mean()),
    阶跃保真=("step_ratio", "mean"),
    事件跳变超额=("jump_excess", "max"),
).reindex([k for k in ALGOS if k in set(dfm.algo)])
agg.index = [LAB[i] for i in agg.index]
agg.to_csv(os.path.join(RES, "fastphase_summary.csv"), encoding="utf-8-sig")
print("\n===== 9 组恒载汇总（越小越好，阶跃保真越接近 1 越好）=====")
print(agg.round(2).to_string())
print("\n重点看「时漂残余_慢相段」——这是本次改造的目标量。")
print("saved: results/fastphase_metrics.csv / fastphase_summary.csv")
