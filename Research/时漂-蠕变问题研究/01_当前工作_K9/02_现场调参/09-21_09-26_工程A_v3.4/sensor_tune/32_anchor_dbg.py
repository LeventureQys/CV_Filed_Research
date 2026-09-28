# -*- coding: utf-8 -*-
"""32 锚定机制调试：打印锚定窗口采样、anchor、hi、extra 与显示轨迹。"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sensor_common import LIVE, load  # noqa: E402
from sweep_lib import PREP  # noqa: E402
import importlib  # noqa: E402

m31 = importlib.import_module("31_anchor") if False else None
sys.path.insert(0, str(HERE))
# 直接内联导入（模块名以数字开头，用 importlib）
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("anchor31", HERE / "31_anchor.py")
A = importlib.util.module_from_spec(spec)
sys.modules["anchor31"] = A
spec.loader.exec_module(A)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    key = sys.argv[1] if len(sys.argv) > 1 else "左拇指指腹/d1"
    sensor, tag = key.split("/", 1)
    info = PREP[key]
    d = load(info["path"].split("data\\", 1)[-1].replace("\\", "/"))
    t = d["t"] - d["t"][0]
    V = d["V"] - d["V"][0]
    base = A.ParamsA(**{f.name: getattr(LIVE, f.name) for f in fields(A.Params)})
    p = A.replace(base, r_fast=0.0, slope_gate_frac=0.0, r_slow_max=0.001, slope_cap_frac=0.0)
    o = A.AnchorObserver(p)
    outs, his, exs, anc = [], [], [], []
    for i in range(len(t)):
        y = o.process(float(t[i]), V[i])
        outs.append(y.sum())
        his.append(o.hi.sum() if hasattr(o, "hi") else 0.0)
        exs.append((o.hi.sum() - o.anchor.sum()) if hasattr(o, "hi") else 0.0)
        anc.append(o.anchor.sum() if hasattr(o, "anchor") else 0.0)
    outs = np.array(outs)
    tin = V.sum(axis=1)
    print(f"{key}：输入 0→{tin[-1]:.0f}；采样帧数={len(o._samples) if o._samples else 0}；"
          f"anchored={o._anchored}")
    print(f"  self.anchor 合计 = {anc[-1]:.1f}；metrics 口径（[1,2]s 输出中位数）= "
          f"{np.median(outs[(t >= 1.0) & (t <= 2.0)]):.1f}；"
          f"输入在 [1,2]s 的中位数 = {np.median(tin[(t >= 1.0) & (t <= 2.0)]):.1f}")
    print(f"  self.hi 合计 = {his[-1]:.1f}（输入末值 {tin[-1]:.0f}，输入峰 {tin.max():.0f}）")
    print(f"\n  {'t':>6s} {'输入':>8s} {'核心输出':>9s} {'显示':>8s} {'hi':>8s} {'extra':>8s} {'anchor':>8s}")
    for x in [0.5, 1.0, 1.5, 2.0, 2.5, 3, 5, 10, 30, 60, 120, 179]:
        i = int(np.searchsorted(t, x))
        if i >= len(t):
            continue
        print(f"  {t[i]:6.1f} {tin[i]:8.1f} {tin[i]:9.1f} {outs[i]:8.1f} {his[i]:8.1f} "
              f"{exs[i]:8.1f} {anc[i]:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
