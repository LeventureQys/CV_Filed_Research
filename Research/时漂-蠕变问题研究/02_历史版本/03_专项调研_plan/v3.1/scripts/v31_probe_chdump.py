# -*- coding: utf-8 -*-
"""临时：dump 逐通道末帧内部量（gamma / A / pct / 扣除）。"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
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
    env = dict(os.environ)
    env["V30_DUMP_CH"] = "1"
    p = subprocess.run([os.path.join(HERE, "build", "v30_runner.exe")],
                       input="\n".join(lines) + "\n", capture_output=True,
                       text=True, encoding="utf-8", env=env)
    rows = p.stdout.splitlines()
    hdr = [r for r in rows if r.startswith("CH k")]
    ch = [r for r in rows if r.startswith("CH ")]
    print(hdr[0] if hdr else "(no header)")
    for r in ch:
        print(r)


if __name__ == "__main__":
    main()
