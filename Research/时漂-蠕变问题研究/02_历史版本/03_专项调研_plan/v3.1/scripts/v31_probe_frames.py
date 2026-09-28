# -*- coding: utf-8 -*-
"""临时：回放内部量逐帧（可指定窗口/步长）。用法: python v31_probe_frames.py t0 t1 step"""
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
    t0, t1, st = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
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
    idx = np.where((el >= t0) & (el <= t1))[0]
    print("%8s %8s %8s %7s %6s %5s %6s %8s %8s %8s %8s %8s %8s %8s" %
          ("t", "in", "out", "ded", "state", "ev", "kind", "A_sum", "g",
           "gam_med", "pct_sum", "A_hat", "inc_max", "c_appl"))
    for i in idx[::st]:
        g = lambda k: D[k][i]  # noqa: E731
        print("%8.3f %8.0f %8.0f %7.0f %6.0f %5.0f %6.0f %8.0f %8.4f %8.3f %8.1f %8.0f %8.0f %8.0f" %
              (el[i], tin[i], g("sum_out"), g("comp_total"), g("state"),
               g("ev_valid"), g("ev_kind"), g("A_sum"), g("g"), g("gam_med"),
               g("pct_sum"), g("A_hat"), g("inc_max"), g("c_applied")))


if __name__ == "__main__":
    main()
