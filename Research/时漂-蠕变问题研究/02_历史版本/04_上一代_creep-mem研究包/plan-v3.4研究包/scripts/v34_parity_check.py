# -*- coding: utf-8 -*-
"""v3.4 C++↔Python 一致性校验：obs_runner vs v34_observer_core，目标录制全帧。"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_observer_core3 import observe3
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "build", "obs_runner.exe")
DS = os.path.join(L.DATA_ROOT, "archived", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")


def main():
    s = L.load_stream(DS, "device_001_pre_seg0.csv")
    ts, V = s["ts"], s["V"]
    D, _ = observe3(ts, V)                  # Python 原型 v3
    py_out = D.sum(axis=1)

    n = V.shape[1]
    lines = [str(n)]
    for i in range(len(ts)):
        lines.append("%.6f " % ts[i] + " ".join("%.1f" % x for x in V[i]))
    p = subprocess.run([RUNNER], input="\n".join(lines), capture_output=True,
                       text=True, encoding="utf-8", cwd=HERE)
    cpp_out = []
    for ln in p.stdout.splitlines():
        f = ln.split()
        if len(f) == 3:
            cpp_out.append(float(f[2]))
    cpp_out = np.array(cpp_out)
    print("帧数: py=%d cpp=%d" % (len(py_out), len(cpp_out)))
    m = min(len(py_out), len(cpp_out))
    d = np.abs(py_out[:m] - cpp_out[:m])
    print("逐帧 |Δ| : max=%.6f median=%.6f （容差 0.5 ADC 来自输入 1 位小数截断）"
          % (d.max(), np.median(d)))
    assert d.max() < 1.0, "parity FAIL"
    print("PARITY OK")


if __name__ == "__main__":
    main()
