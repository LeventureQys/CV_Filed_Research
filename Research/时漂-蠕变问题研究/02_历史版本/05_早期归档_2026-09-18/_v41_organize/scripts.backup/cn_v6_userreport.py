# -*- coding: utf-8 -*-
"""逐帧状态转储：定位用户报告的两个现象
   ① 右拇指/数据2 在 t≈150 s 的"大过冲 + 平缓下降"
   ② 切换负载 在首个加载段的"先稳住又缓慢抬升 / 先过充再回落"
逐帧记录 raw / 显示 / state / A.sum / g / c_applied / 当前事件 τ。
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
from glm53_v6 import GLM53v6             # noqa: E402


def dump(path, label, wins):
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    c = GLM53v6(Xu.shape[1])
    c.debug = True
    n = len(tu)
    Y = np.empty(n)
    st = np.empty(n, dtype=object)
    Asum = np.zeros(n); gs = np.zeros(n); capp = np.zeros(n)
    loaded = np.zeros(n, dtype=int); tau = np.zeros(n); kind = np.empty(n, dtype=object)
    for i in range(n):
        out = c.process(tu[i], Xu[i])
        Y[i] = out.sum()
        st[i] = c.state
        Asum[i] = c.A.sum(); gs[i] = c.g; loaded[i] = int(c.loaded.sum())
        if c.ev is not None:
            tau[i] = tu[i] - c.ev["t0"]; kind[i] = c.ev["kind"]
            capp[i] = c.ev["c_applied"]
        else:
            tau[i] = np.nan; kind[i] = ""
    print("=" * 124)
    print(f"{label}  事件 {len(c.epoch_t)}")
    for t, k in c.kind_log:
        print(f"      @{t:8.2f}s {k}")
    for r in c.dbg:
        if r[1] in ("STALL", "NEW", "HANDOFF", "REVOKE", "REVOKE-", "UNLOAD"):
            rest = "  ".join(f"{x:12.4g}" for x in r[3:])
            print(f"      DBG {r[0]:8.2f} {r[1]:9s} {str(r[2]):14s} {rest}")
    for name, a, b, step in wins:
        i0 = int(np.searchsorted(tu, a)); i1 = int(np.searchsorted(tu, b))
        print(f"\n--- {name}：{a}~{b}s（每 {step}s）---")
        print(f"{'t(s)':>9}{'raw':>11}{'v6':>11}{'v6-raw':>10}{'state':>8}"
              f"{'τ_ev':>8}{'kind':>14}{'A.sum':>10}{'g':>10}{'ld':>5}{'c_app':>10}")
        for i in range(i0, min(i1, n), max(1, int(step / dtm))):
            print(f"{tu[i]:9.2f}{tot[i]:11.2f}{Y[i]:11.2f}{Y[i]-tot[i]:10.2f}"
                  f"{str(st[i]):>8}{tau[i]:8.2f}{str(kind[i]):>14}{Asum[i]:10.2f}"
                  f"{gs[i]:10.4f}{loaded[i]:5d}{capp[i]:10.2f}")


dump(os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"),
     "右拇指指尖/数据2", [("过冲段", 140, 165, 0.5)])

dump(os.path.join(TEMP, "变化负载", "切换负载-快相无责的测试",
                  "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"),
     "切换负载-快相无责", [("首个加载段", 10.4, 22.0, 0.25), ("第二段", 26.0, 40.0, 0.5)])
