# -*- coding: utf-8 -*-
"""v2.0 阶段二 · F2 对「高位驻留时长」的作用（用户可见的那个量）。

对每个数据集，逐加载沿给出：
  驻留 = 事件内「显示超前量 > 事件后稳态超前量 + 2%·台阶」的最长连续时长
对比 base / C / C+F2F3 三臂。
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

spec = importlib.util.spec_from_file_location("dfr", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "design_full_regress.py"))
DFR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DFR)


def dwell(el, pre, out, tu):
    res = []
    for t in tu:
        m0 = (el >= t - 1.2) & (el < t - 0.35)
        if m0.sum() < 5:
            continue
        bp = float(np.median(pre[m0]))
        te = min(t + 30.0, el[-1] - 0.2)
        m1 = (el >= te - 1.5) & (el <= te)
        ps = float(np.median(pre[m1]))
        os_ = float(np.median(out[m1]))
        A = ps - bp
        if abs(A) < 500:
            continue
        idx = np.where((el >= t - 0.05) & (el <= te))[0]
        lead = out[idx] - pre[idx]
        thr = (os_ - ps) + 0.02 * abs(A)
        over = lead > thr
        tt = el[idx]
        best = run = 0.0
        for k in range(len(over)):
            if over[k]:
                run += (tt[k] - tt[k - 1]) if k else 0.0
                best = max(best, run)
            else:
                run = 0.0
        res.append((t, float(np.max(lead)) / abs(A), best))
    return res


def main():
    cases = [("★新录制", os.path.join(L.DS_ZERO, "device_001_pre_seg0.csv"))]
    base_dir = os.path.join(L.ROOT, "temp", "原始数据only")
    root = os.path.join(base_dir, "变化负载")
    for d in sorted(os.listdir(root)):
        for dp, _dn, fn in os.walk(os.path.join(root, d)):
            if "device_001_seg000.csv" in fn:
                cases.append((f"变化负载/{d}", os.path.join(dp, "device_001_seg000.csv")))

    for name, path in cases:
        el, V = DFR.read_simple(path)
        pre = V.sum(1)
        tu = DFR.edges(el, pre)
        outs = {}
        for tag, kw in (("base", dict()), ("C", dict(clamp=0.005)),
                        ("C+F2F3", dict(clamp=0.005, f2=True, f3=True))):
            outs[tag] = DFR.run(el, V, **kw)
        print(f"\n=== {name}  帧={len(el)}  加载沿={[round(t,2) for t in tu]} ===")
        print(f"{'沿(s)':>9}{'base驻留':>10}{'C驻留':>9}{'C+F2F3驻留':>12}"
              f"{'base超前':>10}{'C超前':>9}{'C+F2F3超前':>12}")
        d0 = dwell(el, pre, outs["base"], tu)
        d1 = dwell(el, pre, outs["C"], tu)
        d2 = dwell(el, pre, outs["C+F2F3"], tu)
        n = min(len(d0), len(d1), len(d2))
        for i in range(n):
            print(f"{d0[i][0]:9.2f}{d0[i][2]:10.2f}{d1[i][2]:9.2f}{d2[i][2]:12.2f}"
                  f"{100*d0[i][1]:9.2f}%{100*d1[i][1]:8.2f}%{100*d2[i][1]:11.2f}%")
        if n:
            print(f"{'中位':>9}{np.median([d0[i][2] for i in range(n)]):10.2f}"
                  f"{np.median([d1[i][2] for i in range(n)]):9.2f}"
                  f"{np.median([d2[i][2] for i in range(n)]):12.2f}"
                  f"{100*np.median([d0[i][1] for i in range(n)]):9.2f}%"
                  f"{100*np.median([d1[i][1] for i in range(n)]):8.2f}%"
                  f"{100*np.median([d2[i][1] for i in range(n)]):11.2f}%")


if __name__ == "__main__":
    main()
