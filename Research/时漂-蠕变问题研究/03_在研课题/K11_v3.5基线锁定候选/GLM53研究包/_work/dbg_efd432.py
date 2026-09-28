# -*- coding: utf-8 -*-
"""efd432 长会话：−503 持续下冲是算法造成还是输入本身欠冲。"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
import sweep_lib  # noqa: E402
sweep_lib.EXE = ST / "build" / "v34_sweep_k11.exe"
from sweep_lib import Set, write_setfile  # noqa: E402
from sensor_common import read_session_csv  # noqa: E402
from retune_v4 import m1, read_bin  # noqa: E402

p = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\真实复杂工况\算法数据&原始数据"
         r"\算法数据&原始数据\working\零基线-同一荷载测试\回撤测试\20260920_175017_single_device_efd432"
         r"\device_001_pre_seg0.csv")
d = read_session_csv(p)
t = d["t"] - d["t"][0]
V = d["V"] - d["V"][0]
n, m = V.shape
WORK = HERE / "_dbg2"
WORK.mkdir(exist_ok=True)
f = WORK / "_s.bin"
with f.open("wb") as fh:
    fh.write(np.int32(n).tobytes()); fh.write(np.int32(m).tobytes())
    rows = np.empty((n, m + 1)); rows[:, 0] = t; rows[:, 1:] = V
    fh.write(rows.astype(np.float64).tobytes())

LOCK = {"r_fast": 0.0, "slope_gate_frac": 1e-6, "r_slow_max": 0.0, "hold_eps": 0.0,
        "hold_lock_freeze_s": 2.0, "edge_slope_thres": 150.0, "hold_lock_tau_s": 0.15}
sf = write_setfile([Set("k11", LOCK)], WORK / "_p.txt")
dd = WORK / "_d"
if dd.exists(): shutil.rmtree(dd)
dd.mkdir()
r = subprocess.run([str(sweep_lib.EXE), str(f), str(sf), "0.0", "0.0",
                    "--time", "raw", "--zero", "--dump", str(dd)],
                   capture_output=True, text=True, encoding="utf-8")
a = read_bin(dd / "k11.bin")
tin, out = a[:, 1], a[:, 2]
sm_in, sm_out = m1(tin, t), m1(out, t)
i = int(np.argmin(sm_out))
print(f"显示最深持续下冲 {sm_out[i]:.0f} @ t={t[i]:.0f}s；此时输入 m1={sm_in[i]:.0f}")
print("窗口（每 30s）：t, 输入m1, 显示m1")
for k in range(max(i - 9000, 0), min(i + 9000, n), 3000):
    print(f"  t={t[k]:7.0f}s  in={sm_in[k]:9.0f}  out={sm_out[k]:9.0f}")
print(f"末段：in={sm_in[-1]:.0f} out={sm_out[-1]:.0f}")
