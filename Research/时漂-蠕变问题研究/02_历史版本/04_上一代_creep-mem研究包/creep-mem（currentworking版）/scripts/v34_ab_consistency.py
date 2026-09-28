# -*- coding: utf-8 -*-
"""v3.4 A/B：蠕变记忆开/关，全录制各保持段的显示一致性。"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "build", "v34_runner.exe")
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")

# (label, t_start, t_end) 保持段；t 取自沿检测（v34_probe_15v245.py）
HOLDS = [
    ("满载#1  t16-36",   17.0,  35.5),
    ("满载#2  t42-56",   43.0,  55.5),
    ("满载#3  t63-216",  64.5, 215.0),
    ("满载#4  t233-239", 234.0, 238.5),
    ("满载#5  t246-251", 247.0, 250.5),
    ("满载#6  t259-260", 259.5, 260.4),
    ("满载#7  t262-274", 263.0, 274.0),
    ("满载#8  t279-281", 279.5, 281.2),
    ("满载#9  t290-294", 290.5, 294.2),
    ("满载#10 t304-308", 304.5, 307.5),
    ("半载#1  t37-41",   37.5,  40.5),
    ("半载#2  t57-62",   57.5,  62.0),
    ("半载#3  t253-254", 253.3, 254.4),
    ("半载#4  t283-288", 283.0, 288.0),
    ("半载#5  t295-296", 295.5, 296.2),
]


def run(extra):
    ds = L.load_dataset(DS)
    pre = ds["pre"]
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
    ts, el = pre["ts"], pre["el"]
    a = (el[-1] - el[0]) / (ts[-1] - ts[0])
    b = el[0] - a * ts[0]
    return a * arr[:, 0] + b, arr[:, 1], arr[:, 2]


def report(tag, el, sin, sout):
    print("\n[%s]" % tag)
    fulls, halfs = [], []
    for lab, t0, t1 in HOLDS:
        m = (el >= t0) & (el <= t1)
        if m.sum() < 20:
            continue
        d = float(np.median(sout[m]))
        iv = float(np.median(sin[m]))
        ded = iv - d
        fulls.append(d) if lab.startswith("满") else halfs.append(d)
        print("  %-16s 输入=%7.0f 显示=%7.0f 扣除=%+7.0f" % (lab, iv, d, ded))
    if fulls:
        print("  满载族显示: min=%7.0f max=%7.0f 极差=%6.0f std=%5.0f" %
              (min(fulls), max(fulls), max(fulls) - min(fulls),
               float(np.std(fulls))))
    if halfs:
        print("  半载族显示: min=%7.0f max=%7.0f 极差=%6.0f std=%5.0f" %
              (min(halfs), max(halfs), max(halfs) - min(halfs),
               float(np.std(halfs))))


def main():
    el, s0, o0 = run(["--mem", "0"])
    el1, s1, o1 = run(["--mem", "120"])
    report("mem OFF (plan-v3.1 行为)", el, s0, o0)
    report("mem ON  τ=120s (plan-v3.4)", el1, s1, o1)
    # C 限幅不变量：out <= in + 0.005*|in|（总量近似口径，逐帧）
    viol = np.sum(o1 > s1 + 0.005 * np.abs(s1) + 1.0)
    print("\nmem ON 帧级 C 限幅越界(总量口径, 容差1): %d / %d" % (viol, len(o1)))


if __name__ == "__main__":
    main()
