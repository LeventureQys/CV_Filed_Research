# -*- coding: utf-8 -*-
"""四指指腹：推荐参数 vs 现役 的总读数对比图（含拟合弹性电平参考线）。

用 toolbox\\数据解析工具 的 dptool 出图（单面板 overlay、signal=sum；总值写单元0、其余补 0）。
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402
from data_finger_params import SESSIONS, fit_total, fit_channels  # noqa: E402

DPTOOL_ROOT = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
PLAN_J = dict(r_fast=0.02, r_slow_max=0.02, tau_c_fast_s=2.0)
PLAN_A = dict(r_fast=0.02, r_slow_max=0.03, tau_c_fast_s=2.0)


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
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    out_dir = HERE / "out"
    sys.path.insert(0, str(DPTOOL_ROOT))
    from dptool import api  # noqa: E402

    for key in ("d1/6a679f(277s)", "d2/857759(192s)", "d1/5ce2c4(198s)", "d2/8e127e(98s)"):
        d = read_any_session_csv(SESSIONS[key])
        t, V = d["t"], d["V"]
        tin = V.sum(axis=1)
        t0, (E, c1, tau1, c2, tau2) = fit_total(t, tin)
        _, Es = fit_channels(t, V, tau1, tau2)
        Esum = float(Es.sum())
        curves = [
            ("00_输入总ADC", tin),
            ("01_拟合弹性电平E", np.full_like(tin, Esum)),
            ("02_现役默认", run(LIVE, t, V)),
            ("03_推荐J_rf.02_rsm.02", run(replace(LIVE, **PLAN_J), t, V)),
            ("04_多扣档A_rf.02_rsm.03", run(replace(LIVE, **PLAN_A), t, V)),
        ]
        tag = key.split("/")[0] + "_" + key.split("/")[1].split("(")[0]
        tmp = out_dir / f"_dptool_finger_{tag}"
        if tmp.exists():
            shutil.rmtree(tmp)
        n_ch = V.shape[1]
        for label, y in curves:
            write_total_csv(tmp / label / "device_001_seg000.csv", t, y, n_ch)
        png = out_dir / f"四指指腹_推荐参数_{tag}.png"
        res = api.plot_to_file([str(tmp / c[0]) for c in curves], str(png),
                               mode="overlay", series_by="dir", series_stream="out",
                               signal="sum", dpi=130,
                               suptitle=f"{key}：推荐参数 vs 现役（52 通道总 ADC，虚线=拟合弹性电平）")
        print(f"{key:18s} E={Esum:.0f} 真实蠕变={tin[-1] - Esum:.0f} 出图："
              f"{Path(res.get('path', '?')).name if res.get('path') else res.get('error')}", flush=True)


if __name__ == "__main__":
    main()
