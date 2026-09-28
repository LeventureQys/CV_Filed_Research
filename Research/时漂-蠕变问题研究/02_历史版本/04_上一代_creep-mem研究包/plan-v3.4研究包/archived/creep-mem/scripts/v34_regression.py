# -*- coding: utf-8 -*-
"""v3.4 全量回归：所有可发现会话，mem OFF vs ON 的输出流差异 + C 限幅不变量。"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "build", "v34_runner.exe")


def run(ds_dir, extra):
    pre = L.load_stream(ds_dir, "device_001_pre_seg0.csv")
    n = pre["V"].shape[1]
    lines = [str(n)]
    for i in range(pre["n"]):
        lines.append("%.6f " % pre["ts"][i] +
                     " ".join("%.1f" % x for x in pre["V"][i]))
    p = subprocess.run([RUNNER] + list(extra), input="\n".join(lines),
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=HERE)
    rows = []
    for ln in p.stdout.splitlines():
        f = ln.split()
        if not f or f[0] in ("t", "OK", "END", "CH"):
            continue
        try:
            rows.append([float(x) for x in f])
        except ValueError:
            pass
    arr = np.array(rows)
    return pre, arr[:, 1], arr[:, 2]


def main():
    sessions = L.discover_sessions()
    print("发现 %d 个会话" % len(sessions))
    worst = []
    for label, d in sessions:
        try:
            _, s0, o0 = run(d, ["--mem", "0"])
            _, s1, o1 = run(d, ["--mem", "120"])
        except Exception as e:
            print("  %-58s 跳过(%s)" % (label, e))
            continue
        if len(o0) != len(o1) or len(o0) == 0:
            print("  %-58s 帧数异常" % label)
            continue
        diff = o1 - o0
        med = float(np.median(np.abs(diff)))
        mx = float(np.max(np.abs(diff)))
        changed = int(np.sum(np.abs(diff) > 1.0))
        viol = int(np.sum(o1 > s1 + 0.005 * np.abs(s1) + 1.0))
        print("  %-58s 帧%6d 改变%6d 中位|Δ|%8.1f 最大|Δ|%9.1f C越界%d"
              % (label, len(o0), changed, med, mx, viol))
        worst.append((mx, label))
    worst.sort(reverse=True)
    print("\n变化最大的 5 个会话:")
    for mx, label in worst[:5]:
        print("  %9.1f  %s" % (mx, label))


if __name__ == "__main__":
    main()
