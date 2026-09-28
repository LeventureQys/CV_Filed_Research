# -*- coding: utf-8 -*-
"""123_force_shape：力值会话输入/显示波形结构（加载瞬间、保压期形状、卸载）。

回答：E 口径、加载快相的形状与幅度、保压期是「蠕变」还是「斜坡加压」、
算法何时开始扣除（前 3.5 s 空窗的结构来源）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

fl = import_module("122_force_lib")
OUT = TEMP / "palm7" / "out"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "force_1d925c.npz")
    t, tin, tout = z["t"], z["tot_pre"], z["tot_seg"]
    # 累加台阶电平：逐通道把「每级台阶后的新增弹性电平」相加（这里是单级，全卸载重算）
    print("=== 加载瞬间（t=1.17s 前后 1.2 s，逐帧 0.05 s 抽样）===")
    i0 = int(np.searchsorted(t, 1.17))
    for i in range(max(0, i0 - 20), i0 + 40, 2):
        print(f"  t={t[i]:6.3f}  输入={tin[i]:8.3f}  显示={tout[i]:8.3f}  "
              f"输入−显示={tin[i]-tout[i]:7.3f}")
    print("\n=== 段1 保压期（1.17~15.68s，每 0.5 s）===")
    for tt in np.arange(1.2, 15.7, 0.5):
        i = int(np.searchsorted(t, tt))
        if i >= len(t):
            break
        print(f"  t={t[i]:6.3f}  输入={tin[i]:8.3f}  显示={tout[i]:8.3f}  "
              f"扣除={tin[i]-tout[i]:7.3f}  (输入−E1={tin[i]-12.224:+7.3f})")
    print("\n=== 段2 保压期（18.86~32.95s，每 0.5 s）===")
    for tt in np.arange(18.9, 32.9, 0.5):
        i = int(np.searchsorted(t, tt))
        if i >= len(t):
            break
        print(f"  t={t[i]:6.3f}  输入={tin[i]:8.3f}  显示={tout[i]:8.3f}  "
              f"扣除={tin[i]-tout[i]:7.3f}  (输入−E3={tin[i]-13.221:+7.3f})")
    print("\n=== 卸载/再加载（15.68~19.5s）===")
    for tt in np.arange(15.6, 19.5, 0.2):
        i = int(np.searchsorted(t, tt))
        if i >= len(t):
            break
        print(f"  t={t[i]:6.3f}  输入={tin[i]:8.3f}  显示={tout[i]:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
