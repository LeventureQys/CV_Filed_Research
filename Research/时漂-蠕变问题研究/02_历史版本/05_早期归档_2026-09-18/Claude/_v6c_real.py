# -*- coding: utf-8 -*-
"""v6c vs v5.1 vs v6 实录对比（临时脚本，口径沿用 ci_v6_vs_1s_3s.py）：

  epoch 数 = len(epoch_t)
  负载内变载的 gap（扣除后显示与原始电平的偏差）：event_table 口径
  慢相残余时漂：负载段 [onset+5s, 末] 的线性去趋势标准差
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, os.pardir, "v4.1flash", "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402
from glm53_v6 import GLM53v6                           # noqa: E402
from glm53_v6c import GLM53v6c                         # noqa: E402

B = os.path.join(os.path.dirname(os.path.dirname(SCRIPTS)), "变化负载")
RECS = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]


def make(cls):
    class T(cls):
        def __init__(self, n):
            super().__init__(n)
            # v6c 的 _mark_epoch 由基类 _begin/_restep 触发，会写入 self.epoch_t；
            # 这里只统计本层包装器收到的 _begin/_restep 次数（两算法同口径）。
            if not hasattr(self, "epoch_t"):
                self.epoch_t = []
            self._n51 = 0

        def process(self, ts, v):
            return super().process(ts, v)

        def _begin(self, ts):
            super()._begin(ts)
            self._n51 += 1

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self._n51 += 1

    return T


def nep(c):
    """epoch 数：两算法同口径 = _begin/_restep 实际被调用的次数。"""
    return int(getattr(c, "_n51", 0))


def run(cls, tu, Xu, **kw):
    c = make(cls)(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


print("=" * 116)
print(f"{'录制':>18} {'算法':>16} {'epoch':>5} {'gap中位':>8} {'gap最大':>8} {'全程最大':>9} "
      f"{'占峰值%':>7} {'慢相残余%':>9} {'A_max':>8} {'g_end':>8}")
print("-" * 116)
for tag, path in RECS:
    if not os.path.exists(path):
        print(f"  [{tag}] 文件不存在：{path}")
        continue
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    peak = float(tot_s.max())
    ev = [e for e, _ in L.detect_events(tot, dtm)]
    for nm, cls, kw in [("v5.1", GLM53v51, {}), ("v6c", GLM53v6c, {}), ("v6", GLM53v6, {})]:
        try:
            Y, c = run(cls, tu, Xu, **kw)
        except Exception as ex:                       # noqa: BLE001
            print(f"  {tag:>16} {nm:>16} 运行失败：{type(ex).__name__}: {ex}")
            continue
        y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        gap = np.abs(y - tot_s)
        d["Ys"] = {"a": Y}
        try:
            et = L.event_table(d, ev, {"a": Y}, [], gain_s=6.0, algos=["a"])
            et["big"] = et["jump"].abs() >= 2000.0
            thr = 0.30 * et["pre"].max() if len(et) else 0.0
            et["mid"] = (et["pre"] > thr) & (et["post"] > thr) if len(et) else False
            ml = et[et.mid & et.big] if len(et) else et
            gmed = float(ml["gap_a"].median()) if len(ml) else float("nan")
            gmax = float(ml["gap_a"].max()) if len(ml) else float("nan")
            nev = int(len(ml))
        except Exception:                             # noqa: BLE001
            gmed = gmax = float("nan")
            nev = -1
        # 慢相残余：最大负载段 onset+5s 之后，主通道线性去趋势标准差占幅度比
        try:
            s0, s1 = L.find_periods(tot, dtm)[0]
            amp_v = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
            m = int(np.argmax(amp_v))
            i5 = s0 + int(5.0 / dtm)
            if s1 - i5 > int(10.0 / dtm):
                tq = tu[i5:s1] - tu[i5]
                sig = Y[i5:s1, m]
                k, b0 = np.polyfit(tq, sig, 1)
                res = float((sig - (k * tq + b0)).std())
                res_pct = 100 * res / amp_v[m]
            else:
                res_pct = float("nan")
        except Exception:                             # noqa: BLE001
            res_pct = float("nan")
        print(f"  {tag:>16} {nm:>16} {nep(c):5d} {gmed:8.0f} {gmax:8.0f} "
              f"{gap.max():9.0f} {100*gap.max()/peak:7.2f} {res_pct:9.2f} "
              f"{float(c.A.max()):8.0f} {float(c.g):+8.4f}")
print("=" * 116)
