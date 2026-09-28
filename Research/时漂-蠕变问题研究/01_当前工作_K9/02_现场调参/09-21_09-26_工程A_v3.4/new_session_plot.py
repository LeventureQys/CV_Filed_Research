# -*- coding: utf-8 -*-
"""会话 20260924_155543（ADC 模式）推荐参数对比图：输入 / 现役 / 推荐 / 更紧 / 拟合弹性电平 E。

用 toolbox\\数据解析工具 的 dptool 出图（单面板 overlay、signal=sum；总值写单元0、其余补 0）。

用法：python new_session_plot.py [--out out]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402
from new_session_analyze import fit_elastic  # noqa: E402

DPTOOL_ROOT = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
CSV = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\20260924_155543_single_device_1201c1"
           r"\device_001_seg000.csv")


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


def write_total_csv(path: Path, t, total, n_ch):
    zeros = ",".join(["0.000000"] * (n_ch - 1))
    ts = t + 171642.0
    lines = ["##Session", "数据阶段,processed_display", "值阶段,processed_display",
             f"行数,{len(t)}", f"列数,{n_ch}", "##Data",
             "timestamp,elapsed,frame_index," + ",".join(f"ch{j}" for j in range(n_ch))]
    for i in range(len(t)):
        lines.append(f"{ts[i]:.6f},{t[i]:.6f},{i},{total[i]:.6f},{zeros}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    out_dir = HERE / args.out
    d = read_any_session_csv(CSV)
    t, V = d["t"], d["V"]
    tin = V.sum(axis=1)
    n_ch = V.shape[1]
    E = float(fit_elastic(t, tin)[0][0])

    curves = [
        ("00_输入总ADC", tin),
        ("01_拟合弹性电平E", np.full_like(tin, E)),
        ("02_现役默认_tc1=12", run(LIVE, t, V)),
        ("03_推荐_tc1=2", run(replace(LIVE, tau_c_fast_s=2.0), t, V)),
        ("04_更紧_tc1=1.5", run(replace(LIVE, tau_c_fast_s=1.5), t, V)),
    ]
    tmp = out_dir / "_dptool_new"
    if tmp.exists():
        shutil.rmtree(tmp)
    for label, y in curves:
        write_total_csv(tmp / label / "device_001_seg000.csv", t, y, n_ch)
        print(f"{label:20s} 末值 {y[-1]:9.1f}  与 E 偏差 {y[-1] - E:+8.1f} ADC")

    sys.path.insert(0, str(DPTOOL_ROOT))
    from dptool import api, merge as M  # noqa: E402
    dirs = [str(tmp / c[0]) for c in curves]
    png = out_dir / "new_session_推荐参数对比.png"
    res = api.plot_to_file(dirs, str(png), mode="overlay", series_by="dir",
                           series_stream="out", signal="sum", dpi=130,
                           suptitle="20260924_155543（ADC 显示）：推荐参数 vs 现役 vs 拟合弹性电平 E"
                                    "（52 通道总 ADC）")
    print("出图：", res.get("path") or res.get("error"))
    tbl, _ = M.merge(dirs, start=int(np.searchsorted(t, 1.0)))
    png2 = out_dir / "new_session_推荐参数对比_放大.png"
    M.plot_table(tbl, str(png2), mode="overlay", series_by="dir", series_stream="out",
                 signal="sum", ylim=(28800, 32300), dpi=130,
                 suptitle="同图放大（y 轴 28800~32300 ADC）：看现役的先高后沉与推荐的贴合度")
    print("放大图：", png2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
