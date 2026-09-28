# -*- coding: utf-8 -*-
"""临时：把回放内部量按 1 s 桶打印（A_sum / g / gamma / pct / ded），用于定位基线丢失。"""
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
    t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 238.0
    t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 312.0
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    print("%8s %9s %9s %9s %8s %8s %8s %8s %8s %8s %8s" %
          ("t", "in", "out", "ded", "state", "A_sum", "g", "gam_med", "pct_sum", "A_hat", "c_appl"))
    for t in np.arange(t0, t1, step):
        m = (el >= t) & (el < t + 0.6)
        if not m.any():
            continue
        f = lambda k: float(np.median(D[k][m]))  # noqa: E731
        print("%8.1f %9.0f %9.0f %9.0f %8.0f %8.0f %8.4f %8.3f %8.1f %8.0f %8.0f" %
              (t, float(np.median(tin[m])), f("sum_out"), f("comp_total"), f("state"),
               f("A_sum"), f("g"), f("gam_med"), f("pct_sum"), f("A_hat"), f("c_applied")))


if __name__ == "__main__":
    main()
