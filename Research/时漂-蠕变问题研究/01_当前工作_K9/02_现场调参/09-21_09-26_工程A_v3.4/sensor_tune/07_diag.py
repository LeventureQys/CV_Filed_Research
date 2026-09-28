# -*- coding: utf-8 -*-
"""07 诊断：拿一份会话把 v3.4 观测器的内部状态逐帧打出来，解释「为什么扣不动 / 为什么过扣」。

同时做一次 C++↔Python 逐帧对拍（同一会话同一参数，比较显示总值 max|Δ|），
确认 Python 版与产品源码同口径。

用法：python 07_diag.py [会话键] [参数覆盖 k=v,...]
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from sensor_common import LIVE, OUT_DIR, load  # noqa: E402
from sweep_lib import EXE, PREP, write_setfile, Set  # noqa: E402

PROBE_T = [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 150, 179]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    key = sys.argv[1] if len(sys.argv) > 1 else "左拇指指腹/d1"
    over = {}
    if len(sys.argv) > 2:
        for tok in sys.argv[2].split(","):
            k, _, v = tok.partition("=")
            over[k] = float(v)
    p = replace(LIVE, **over)
    info = PREP[key]

    d = load(info["path"].split("data\\", 1)[-1].replace("\\", "/"))
    t = d["t"] - d["t"][0]
    V = d["V"] - d["V"][0]

    # ── Python 复算（带 trace）──
    c = CreepObserverK9(p)
    n = len(t)
    out = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
    tr = c.traces()

    # ── C++ 对拍 ──
    sf = write_setfile([Set("diag", over)], OUT_DIR / "_paramsets" / "_diag.txt")
    r = subprocess.run([str(EXE), info["path"], str(sf), f"{info['Esum_channels']:.6f}",
                        f"{info['t0']:.6f}", "--time", "uniform", "--zero",
                        "--dump", str(OUT_DIR / "_dump_diag")],
                       capture_output=True, text=True, encoding="utf-8")
    (OUT_DIR / "_dump_diag").mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(EXE), info["path"], str(sf), f"{info['Esum_channels']:.6f}",
                        f"{info['t0']:.6f}", "--time", "uniform", "--zero",
                        "--dump", str(OUT_DIR / "_dump_diag")],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print("C++ 失败:", r.stderr)
        return 1
    binf = OUT_DIR / "_dump_diag" / "diag.bin"
    raw = np.fromfile(binf, dtype=np.int32, count=2)
    arr = np.fromfile(binf, dtype=np.float64, offset=8).reshape(int(raw[0]), int(raw[1]))
    dmax = float(np.max(np.abs(arr[:, 2] - out)))
    print(f"会话 {key}  参数 {over or '现役默认'}  ΣE={info['Esum_channels']:.1f} "
          f"蠕变={info['creep_total']:.1f}  t0={info['t0']:.2f}s")
    print(f"C++↔Python 逐帧对拍：显示总值 max|Δ| = {dmax:.3e} ADC （输入 1 位小数，容差 0.5）\n")

    x1 = tr["x_fast"].sum(axis=1)
    x2 = tr["x_slow"].sum(axis=1)
    ap = tr["applied"].sum(axis=1)
    dwell = tr["dwell"]
    ine = tr["e_now"].sum(axis=1)
    tin = V.sum(axis=1)

    print(f"{'t':>6s} {'输入':>9s} {'显示':>9s} {'y':>9s} {'e':>9s} {'slope和':>9s} "
          f"{'x1':>8s} {'x2':>8s} {'applied':>8s} {'dwell中位':>9s} {'扣%':>6s}")
    for tt in PROBE_T:
        i = int(np.searchsorted(t, tt))
        if i >= n:
            continue
        sl = float(tr["slope"][i].sum())
        print(f"{t[i]:6.1f} {tin[i]:9.1f} {out[i]:9.1f} {tr['zero'][i].sum() * 0 + 0:9.1f} "
              f"{ine[i]:9.1f} {sl:9.2f} {x1[i]:8.1f} {x2[i]:8.1f} {ap[i]:8.1f} "
              f"{np.median(dwell[i]):9.2f} "
              f"{(tin[i] - out[i]) / max(tin[i], 1e-9) * 100:6.1f}")
    # 逐通道门限尺度
    i = int(np.searchsorted(t, 30.0))
    e_i = tr["e_now"][i]
    sl_i = np.abs(tr["slope"][i])
    print(f"\n30 s 处逐通道门限：e 中位 {np.median(e_i):.1f} → "
          f"沿门 5%·max(e,1) 中位 {np.median(0.05 * np.maximum(e_i, 1)):.3f} ADC/s，"
          f"积分上限 1%·max(e,1) 中位 {np.median(0.01 * np.maximum(e_i, 1)):.4f} ADC/s；"
          f"该帧 |slope| 中位 {np.median(sl_i):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
