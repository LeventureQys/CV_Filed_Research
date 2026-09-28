# -*- coding: utf-8 -*-
"""临时：在指定时刻截断输入流喂给 runner，dump 逐通道内部量（A/gamma/elig/pct）。

用法: python v31_probe_ch_at.py <t_cut>
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
    t_cut = float(sys.argv[1]) if len(sys.argv) > 1 else 260.0
    ds = L.load_dataset(DS)
    el, V = ds["pre"]["el"], ds["pre"]["V"]
    m = el <= t_cut
    el, V = el[m], V[m]
    lines = ["%d" % V.shape[1]]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % x for x in row))
    env = dict(os.environ)
    env["V30_DUMP_CH"] = "1"
    p = subprocess.run([os.path.join(HERE, "build", "v30_runner.exe")],
                       input="\n".join(lines) + "\n", capture_output=True,
                       text=True, encoding="utf-8", env=env)
    rows = p.stdout.splitlines()
    frames = int(rows[1].split()[2])
    last = rows[2 + frames - 1].split()
    cols = rows[0].split()
    print("t_cut=%.1f frames=%d" % (t_cut, frames))
    print("last frame: " + " ".join("%s=%s" % (c, v) for c, v in
                                    zip(cols[:3], last[:3])))
    print("state=%s A_sum=%s g=%s gam_med=%s comp_total=%s pct_sum=%s" %
          (last[cols.index("state")], last[cols.index("A_sum")],
           last[cols.index("g")], last[cols.index("gam_med")],
           last[cols.index("comp_total")], last[cols.index("pct_sum")]))
    for r in rows:
        if r.startswith("CH k") or r.startswith("CH "):
            print(r)


if __name__ == "__main__":
    main()
