# -*- coding: utf-8 -*-
"""H/I 组出图：K11 锁定 vs 现役（dptool）。H=每会话（输入/零基准/现役/K11锁定），I=每类 overlay。"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
import sweep_lib  # noqa: E402
sweep_lib.EXE = ST / "build" / "v34_sweep_k11.exe"
from sweep_lib import PREP, SESSION_BIN, Set, write_setfile  # noqa: E402
from sensor_common import DPTOOL_ROOT, SENSORS  # noqa: E402
from retune_v4 import read_bin  # noqa: E402

FIG = HERE.parent / "figures"
TMP = HERE / "_figH"
V5 = json.loads((HERE / "retune5_all.json").read_text(encoding="utf-8"))


def run_dump(key, named):
    p = PREP[key]
    d = TMP / "_d"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    sf = write_setfile([Set(n, o) for n, o in named], TMP / "_s.txt")
    src = SESSION_BIN.get(key, p["path"])
    r = subprocess.run([str(sweep_lib.EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                        f"{p['t0']:.6f}", "--time", "raw", "--zero", "--dump", str(d)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300])
    return {n: read_bin(d / f"{n}.bin") for n, _ in named}


def write_csv(path, t, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = t + 171642.0
    lines = ["##Session", "方案名, tactile_recording_session", "数据阶段,processed_display",
             "值阶段,processed_display", f"行数,{len(t)}", "列数,2", "数据点数,2",
             "显示模式,adc", "##Data", "timestamp,elapsed,frame_index,ch0"]
    lines += [f"{ts[i]:.6f},{t[i]:.6f},{i},{y[i]:.6f}" for i in range(len(t))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    if TMP.exists():
        shutil.rmtree(TMP)
    sys.path.insert(0, str(DPTOOL_ROOT))
    from dptool import api
    for sensor, v in V5.items():
        best = v["best"]["params"]
        i_dirs = []
        for tag in SENSORS[sensor]["sessions"]:
            key = f"{sensor}/{tag}"
            arr = run_dump(key, [("live", {}), ("lock", best)])
            t = arr["live"][:, 0]
            g = TMP / sensor / tag.replace("/", "_")
            write_csv(g / "00_未补偿输入" / "d.csv", t, arr["live"][:, 1])
            write_csv(g / "01_零基准" / "d.csv", t, np.zeros_like(t))
            write_csv(g / "02_现役默认" / "d.csv", t, arr["live"][:, 2])
            write_csv(g / f"03_K11锁定(tau={best['hold_lock_tau_s']}s)" / "d.csv",
                      t, arr["lock"][:, 2])
            names = sorted(x.name for x in g.iterdir())
            png = FIG / f"H_{sensor}_{tag.replace('/', '_')}_K11锁定.png"
            res = api.plot_to_file([str(g / n) for n in names], str(png), mode="overlay",
                                   series_by="dir", series_stream="out", signal="sum",
                                   smooth_s=0.0, stats=False, dpi=130, width_in=15, height_in=7,
                                   suptitle=f"{sensor} {tag}：K11 基线锁定 vs 现役默认"
                                            f"（τ={best['hold_lock_tau_s']}s，理想显示=0 线；"
                                            f"锁定后 1~3s 入带、带宽 ≲150 ADC）")
            print(f"  H {png.name}: {'OK' if res.get('path') else res.get('error')}")
            gb = TMP / f"I_{sensor}" / tag.replace("/", "_")
            write_csv(gb / f"{tag.replace('/', '_')}·K11锁定" / "d.csv", t, arr["lock"][:, 2])
            write_csv(gb / f"{tag.replace('/', '_')}·现役默认" / "d.csv", t, arr["live"][:, 2])
            i_dirs.append(gb)
        dirs = [str(d / n) for d in i_dirs for n in sorted(x.name for x in d.iterdir())]
        png = FIG / f"I_{sensor}_K11锁定overlay.png"
        res = api.plot_to_file(dirs, str(png), mode="overlay", series_by="dir",
                               series_stream="out", signal="sum", smooth_s=0.0, dpi=130,
                               width_in=15, height_in=7,
                               suptitle=f"{sensor}：K11 锁定（各会话）vs 现役默认 —— 理想显示=0")
        print(f"  I {png.name}: {'OK' if res.get('path') else res.get('error')}")


main()
