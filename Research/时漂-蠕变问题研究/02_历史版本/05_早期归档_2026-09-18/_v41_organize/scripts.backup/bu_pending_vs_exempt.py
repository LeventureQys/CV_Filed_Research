# -*- coding: utf-8 -*-
"""把「pending 冻结」与「免责期」在同一段数据上逐帧分开量出来。

背景：这两个环节都表现为"一段时间内不产生新扣除"，容易被当成一回事。本脚本在
**负载内变载（restep）**上把四段分开测量（这是两者真正分道扬镳的场景）：

    ① 检测窗    真实台阶 → pending 置位（判据要看够 0.3~0.8s）
    ② pending 冻结  pending 置位 → 阶跃确认（≥kStepPersistS=2.5s）
                    `hold_=True`：不积分 g/γ，扣除量冻结在 hold_comp_（旧段扣除）
    ③ 免责期     确认 → epoch 起点 + FAST_S（输出 Z−carry_，保留旧载已累积的蠕变）
    ④ 重建期     免责期结束 → 逐通道 γ 与 g 重新建立

对照：**首次加载（空载→负载）**时 `hold_` 在空载分支被显式置 False（`glm53_v3.py:122`），
且空载态根本没有扣除量 ⇒ 那 2.5 s 不是"冻结"，而是"还没进入负载态"。

产出：results/pending_vs_exempt.csv、控制台表（调用方 tee 到 results/_pending_vs_exempt.log）
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

B = os.path.join(TEMP, "变化负载")
RECS = [("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                         "最终测试目标", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"))]
FAST = 3.0


class Probe(GLM53v51):
    def __init__(self, n):
        super().__init__(n)
        self.log = []
        self.epoch_t = []

    def process(self, ts, v):
        y = super().process(ts, v)
        self.log.append((float(ts), int(self.in_load), int(self.pending), int(self.hold),
                         float(self.g), float(self.A.max()) if self.A.size else 0.0))
        return y

    def _begin(self, ts):
        super()._begin(ts)
        self.epoch_t.append(float(ts))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.epoch_t.append(float(ts))


rows = []
for tag, path in RECS:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
    ev = L.event_table(d, [e for e, _ in L.detect_events(d["tot"], dtm)], {"raw": Xu}, [], algos=[])
    ml = ev[(ev.jump.abs() >= 2000) & (ev.pre > 0.30 * ev.pre.max())]
    c = Probe(Xu.shape[1])
    c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = FAST, FAST / 3.0, FAST
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    st = np.array(c.log)
    ded = L.med_smooth((Xu - Y).sum(axis=1), 0.5 / dtm)
    print("=" * 124)
    print(f"[{tag}] 负载内变载 {len(ml)} 个（免责 {FAST}s 档）；括号内 = 该阶段扣除量变化 Δ")
    print("-" * 124)
    for _, e in ml.iterrows():
        i0 = int(np.searchsorted(tu, e.t))
        m = (st[:, 0] >= e.t - 1.0) & (st[:, 0] <= e.t + 12.0)
        idx = np.where(m)[0]
        # 四段边界
        pen = st[idx][st[idx, 2] > 0]
        t_pen = pen[0, 0] if len(pen) else np.nan
        i_pen = int(np.searchsorted(st[:, 0], t_pen)) if np.isfinite(t_pen) else i0
        ep = [x for x in c.epoch_t if x >= e.t - 0.25]
        t_ep = ep[0] if ep else np.nan
        i_ep = int(np.searchsorted(st[:, 0], t_ep)) if np.isfinite(t_ep) else i0
        i_end = int(np.searchsorted(st[:, 0], t_ep + FAST)) if np.isfinite(t_ep) else i0
        segs = [("① 检测窗", i0, i_pen), ("② pending 冻结", i_pen, i_ep),
                ("③ 免责期", i_ep, i_end), ("④ 重建期(12s 内)", i_end, min(i_end + int(12 / dtm), len(ded) - 1))]
        print(f"  变载 @{e.t:7.2f}s  台阶 {e.jump:+7.0f}（占电平 {e.ratio:.2f}）")
        for nm, a, b in segs:
            if b <= a:
                continue
            d0, d1 = float(ded[a]), float(ded[b - 1])
            g0, g1 = float(st[a, 4]), float(st[b - 1, 4])
            print(f"     {nm:>16} {tu[a] - e.t:6.2f}s → {tu[b - 1] - e.t:6.2f}s"
                  f"  扣除 {d0:8.0f} → {d1:8.0f} (Δ{d1 - d0:+8.0f})"
                  f"   g {g0:+.4f} → {g1:+.4f}   hold={int(st[a, 3])}")
            rows.append(dict(rec=tag, t_edge=float(e.t), jump=float(e.jump), stage=nm,
                             t0=float(tu[a] - e.t), t1=float(tu[b - 1] - e.t),
                             ded0=d0, ded1=d1, dded=d1 - d0, g0=g0, g1=g1, hold=int(st[a, 3])))
    print("-" * 124)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "pending_vs_exempt.csv"), index=False, encoding="utf-8-sig")
print("\n各阶段汇总（3 档 · 全部负载内变载事件）")
print(df.groupby("stage").agg(
    时长s=("t0", lambda s: np.nan), n=("dded", "count"),
    扣除变化均值=("dded", "mean"), 扣除变化最大=("dded", "max"),
    g变化均值=("g0", lambda s: np.nan)).to_string())
agg = df.groupby("stage").apply(
    lambda g: pd.Series({"事件数": len(g), "阶段时长中位(s)": (g.t1 - g.t0).median(),
                         "Δ扣除均值": g.dded.mean(), "Δ扣除绝对值均值": g.dded.abs().mean(),
                         "Δg均值": (g.g1 - g.g0).mean()}), include_groups=False)
print(agg.round(2).to_string())
