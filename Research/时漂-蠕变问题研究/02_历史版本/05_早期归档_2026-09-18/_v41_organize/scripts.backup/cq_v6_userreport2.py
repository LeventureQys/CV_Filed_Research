# -*- coding: utf-8 -*-
"""第三轮用户报告：① 右拇指/数据2 卸载处的"奇怪下降"；② 切换负载 175~180 s 的过充。
逐帧转储 raw / v5-3s / v6 + v6 内部状态（A.sum / g / c_applied / 事件 τ / kind）。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                       # noqa: E402
from glm53_v51 import GLM53v51           # noqa: E402
from glm53_v6 import GLM53v6             # noqa: E402


def dump(path, label, wins):
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)

    class _T(GLM53v51):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    c5 = _T(Xu.shape[1]); c5.FAST_S, c5.EXEMPT_AWIN, c5.LEV_ARM_S = 3.0, 1.0, 3.0
    Y5 = np.empty_like(Xu)
    for i in range(len(tu)):
        Y5[i] = c5.process(tu[i], Xu[i])

    c = GLM53v6(Xu.shape[1]); c.debug = True
    n = len(tu)
    Y = np.empty(n)
    st = np.empty(n, dtype=object); Asum = np.zeros(n); gs = np.zeros(n)
    capp = np.zeros(n); tau = np.zeros(n); kind = np.empty(n, dtype=object)
    for i in range(n):
        out = c.process(tu[i], Xu[i])
        Y[i] = out.sum()
        st[i] = c.state; Asum[i] = c.A.sum(); gs[i] = c.g
        if c.ev is not None:
            tau[i] = tu[i] - c.ev["t0"]; kind[i] = c.ev["kind"]; capp[i] = c.ev["c_applied"]
        else:
            tau[i] = np.nan; kind[i] = ""
    print("=" * 128)
    print(f"{label}   事件 {len(c.epoch_t)}  撤销 {c.n_revoke}")
    for t, k in c.kind_log:
        print(f"      @{t:8.2f}s {k}")
    for r in c.dbg:
        if r[1] in ("STALL", "REVOKE", "REVOKE-", "UNLOAD", "HANDOFF"):
            rest = "  ".join(f"{x:12.4g}" for x in r[3:])
            print(f"      DBG {r[0]:8.2f} {r[1]:8s} {str(r[2]):14s} {rest}")
    for name, a, b, step in wins:
        i0 = int(np.searchsorted(tu, a)); i1 = int(np.searchsorted(tu, b))
        print(f"\n--- {name}：{a}~{b}s（每 {step}s）---")
        print(f"{'t(s)':>9}{'raw':>11}{'v5-3s':>11}{'v6':>11}{'v6-raw':>10}{'state':>8}"
              f"{'τ':>7}{'kind':>13}{'A.sum':>10}{'g':>10}{'c_app':>9}")
        for i in range(i0, min(i1, n), max(1, int(step / dtm))):
            print(f"{tu[i]:9.2f}{tot[i]:11.2f}{Y5[i].sum():11.2f}{Y[i]:11.2f}{Y[i]-tot[i]:10.2f}"
                  f"{str(st[i]):>8}{tau[i]:7.2f}{str(kind[i]):>13}{Asum[i]:10.2f}"
                  f"{gs[i]:10.4f}{capp[i]:9.2f}")


dump(os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"),
     "右拇指指尖/数据2 —— 卸载段", [("卸载前后", 157.0, 164.0, 0.1)])

dump(os.path.join(TEMP, "变化负载", "切换负载-快相无责的测试",
                  "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"),
     "切换负载-快相无责 —— 175~185s", [("过充段", 172.0, 188.0, 0.25)])
