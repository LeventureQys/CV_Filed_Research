# -*- coding: utf-8 -*-
"""C++（含运行期 SetFastPhase）↔ Python v5 原型 逐帧对拍：3s 与 5s 两档都要一致。

C++ 侧：cpp_v5_check 直接编译 src/domain/drift/drift_compensator.cpp，启动时调 SetFastPhase(fast)
Python 侧：glm53_v5（FAST_S/EXEMPT_AWIN/LEV_ARM_S 同步为该档）
"""
import os
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from glm53_v5 import GLM53v5                                          # noqa: E402

EXE = os.path.join(OUT, "cpp_v5_check", "build", "Debug", "v5_cpp_check.exe")
TAU, AMP = 199.0, 0.1564
FS, DT, DUR, T_LOAD, T_ADD = 100.0, 0.01, 400.0, 20.0, 300.0


def cpp_run(ratio, fast):
    out = subprocess.run([EXE, f"{ratio}", f"{fast}"], capture_output=True,
                         text=True, check=True)
    rows = [l.split(",") for l in out.stdout.strip().splitlines()]
    return (np.array([float(r[0]) for r in rows]),
            np.array([float(r[1]) for r in rows]),
            np.array([float(r[2]) for r in rows]))


def py_run(ratio, fast):
    c = GLM53v5(1)
    c.FAST_S = fast
    c.EXEMPT_AWIN = fast / 3.0          # 与 C++ SetFastPhase 的推导口径一致
    c.LEV_ARM_S = fast
    tt = np.arange(0.0, DUR, DT)
    Y = np.empty(len(tt))
    for i in range(len(tt)):
        t = tt[i]
        v = 0.0
        if t > T_LOAD:
            v += 10000.0 * (1 + AMP * (1 - np.exp(-(t - T_LOAD) / TAU)))
        if t > T_ADD:
            v += ratio * 10000.0 * (1 + AMP * (1 - np.exp(-(t - T_ADD) / TAU)))
        Y[i] = c.process(t, np.array([v]))[0]
    idx = np.arange(0, len(tt), 5)
    return tt[idx], Y[idx], c


print("=" * 104)
print("C++（运行期 SetFastPhase）↔ Python glm53_v5 逐帧对拍")
print("=" * 104)
ok_all = True
for fast in (3.0, 5.0):
    for ratio in (0.25, 0.50, 1.00):
        _, _, d_c = cpp_run(ratio, fast)
        _, d_p, comp = py_run(ratio, fast)
        n = min(len(d_c), len(d_p))
        dtmax = float(np.abs(d_c[:n] - d_p[:n]).max())
        ideal = 10000.0 * (1.0 + ratio)
        tail = d_c[int(0.9 * len(d_c)):].mean()
        ok = dtmax < 1e-4
        ok_all &= ok
        print(f"免责 {fast:.0f}s / 台阶 {ratio:.0%}×10N（真值 {ideal:.0f}）: "
              f"末端 C++ {tail:8.1f}（相对真值 {100*(tail-ideal)/ideal:+.2f}%）  "
              f"逐帧最大差 {dtmax:.2e}  {'✅' if ok else '❌'}")
print()
print("结论: " + ("两档免责期下 C++ 与 Python 原型均逐帧一致（含运行期切换路径）。"
                 if ok_all else "存在不一致，需排查。"))
