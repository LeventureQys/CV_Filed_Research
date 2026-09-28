# -*- coding: utf-8 -*-
"""临时：回放中定位每次事件交接帧，并核算 A_new = y0 + share·Â 的两项来源。

用法: python v31_probe_handoff.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")


def main():
    ds = L.load_dataset(DS)
    el, V = ds["pre"]["el"], ds["pre"]["V"]
    vm = ds["main"]["V"]
    lines = ["%d" % V.shape[1]]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % x for x in row))
    p = subprocess.run([os.path.join(HERE, "build", "v30_runner.exe")],
                       input="\n".join(lines) + "\n", capture_output=True,
                       text=True, encoding="utf-8")
    rows = p.stdout.splitlines()
    frames = int(rows[1].split()[2])
    X = np.array([[float(x) for x in r.split()] for r in rows[2:2 + frames]])
    cols = rows[0].split()
    D = {c: X[:, i] for i, c in enumerate(cols)}
    tin = V.sum(1)
    yout = D["sum_out"]

    # 事件期 -> 慢相 的跳变 = 交接帧
    st = D["state"]
    ev = D["ev_valid"]
    ho = []
    for i in range(1, len(el)):
        if st[i - 1] == 1 and st[i] == 2:
            ho.append(i)
    print("=== 交接帧（state 1→2）共 %d 次 ===" % len(ho))
    print("%9s %9s %9s %9s %9s %9s %9s %9s %9s %9s" %
          ("交接t", "in", "out", "A_sum", "y0窗[.30,.05]均值", "空载显示约",
           "A_sum−y0", "A_sum−out", "ded@+3s", "ded@+20s"))
    for i in ho:
        t = el[i]
        # y0 窗 = [t0−0.30, t0−0.05]，t0 ≈ 交接前的建事件时刻；此处用 tau 反推
        # 直接取 [t-4.3-0.30, t-4.3-0.05] 作为近似（kHoMinS=3.5，t0≈t-3.5~4.3）
        t0 = t - 4.0
        m = (el > t0 - 0.30) & (el <= t0 - 0.05)
        y0w = float(np.mean(yout[m])) if m.any() else float("nan")
        m2 = (el >= t - 2.0) & (el < t - 1.0)
        idle_out = float(np.median(yout[m2])) if m2.any() else float("nan")
        d3 = float(np.median(tin[(el > t + 3) & (el < t + 3.5)]) -
                   np.median(yout[(el > t + 3) & (el < t + 3.5)]))
        d20 = float(np.median(tin[(el > t + 20) & (el < t + 20.5)]) -
                    np.median(yout[(el > t + 20) & (el < t + 20.5)]))
        print("%9.2f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f" %
              (t, tin[i], yout[i], D["A_sum"][i], y0w, idle_out,
               D["A_sum"][i] - y0w, D["A_sum"][i] - yout[i], d3, d20))
    print("")
    print("=== 关键：交接帧前 6 s 的 输入/显示 轨迹（看 display 是否被拉高）===")
    for i in ho:
        t = el[i]
        if t < 240:
            continue
        print("  --- 交接 @ %.2f ---" % t)
        print("      %8s %9s %9s %9s" % ("t", "in", "out", "A_sum"))
        for dt in (-6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1):
            m = (el >= t + dt) & (el < t + dt + 0.05)
            if m.any():
                k = int(np.where(m)[0][0])
                print("      %8.2f %9.0f %9.0f %9.0f" %
                      (el[k], tin[k], yout[k], D["A_sum"][k]))


if __name__ == "__main__":
    main()
