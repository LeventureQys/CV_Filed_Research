# -*- coding: utf-8 -*-
"""D 组出图：拇指/手掌四类「无大幅过减」复调结果对比（dptool）。
每会话一张：未补偿输入 / 拟合真实轨迹 ref / 现役默认 / 旧最优(04_search) / 新最优(过减受限)。
另每类一张 overlay：显示值相对 ref 的偏差（0 = 钉住真实曲线，负 = 过减）。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))

from sensor_common import DPTOOL_ROOT, OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import EXE, PREP, SESSION_BIN, Set, write_setfile  # noqa: E402
from retune_over import ref_of, read_bin, eval_batch  # noqa: E402

FIG = HERE.parent / "figures"
TMP = HERE / "_figD"
SEARCH = json.loads((OUT_DIR / "04_search.json").read_text(encoding="utf-8"))
RETUNE = json.loads((HERE / "retune_all.json").read_text(encoding="utf-8"))


def run_dump(key: str, named: list[tuple[str, dict]]) -> dict:
    p = PREP[key]
    outdir = TMP / "_dump" / key.replace("/", "_")
    outdir.mkdir(parents=True, exist_ok=True)
    sf = write_setfile([Set(n, o) for n, o in named], TMP / "_sets.txt")
    src = SESSION_BIN.get(key, p["path"])
    r = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                        f"{p['t0']:.6f}", "--time", "uniform", "--zero",
                        "--dump", str(outdir)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return {n: read_bin(outdir / f"{n}.bin") for n, _ in named}


def write_csv(path: Path, t: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = t + 171642.0
    lines = ["##Session", "方案名, tactile_recording_session", "数据阶段,processed_display",
             "值阶段,processed_display", f"行数,{len(t)}", "列数,2", "数据点数,2",
             "显示模式,adc", "##Data", "timestamp,elapsed,frame_index,ch0"]
    lines += [f"{ts[i]:.6f},{t[i]:.6f},{i},{y[i]:.6f}" for i in range(len(t))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if TMP.exists():
        shutil.rmtree(TMP)
    sys.path.insert(0, str(DPTOOL_ROOT))
    from dptool import api  # noqa: E402

    for sensor, v in RETUNE.items():
        best = v["best"]["params"]
        old = SEARCH[sensor]["runs"][0]["params"]
        e_dirs = []
        for tag in SENSORS[sensor]["sessions"]:
            key = f"{sensor}/{tag}"
            arr = run_dump(key, [("live", {}), ("old", old), ("new", best)])
            t = arr["live"][:, 0]
            tin = arr["live"][:, 1]
            g = TMP / sensor / tag.replace("/", "_")
            write_csv(g / "00_未补偿输入" / "device_001_seg000.csv", t, tin)
            write_csv(g / "01_零基准(理想显示)" / "device_001_seg000.csv", t, np.zeros_like(t))
            write_csv(g / "02_现役默认" / "device_001_seg000.csv", t, arr["live"][:, 2])
            write_csv(g / "03_旧最优(过减未约束)" / "device_001_seg000.csv", t, arr["old"][:, 2])
            write_csv(g / "04_新最优(过减≤150约束)" / "device_001_seg000.csv", t, arr["new"][:, 2])
            names = sorted(x.name for x in g.iterdir())
            png = FIG / f"D_{sensor}_{tag.replace('/', '_')}_过减复调对比.png"
            res = api.plot_to_file([str(g / n) for n in names], str(png), mode="overlay",
                                   series_by="dir", series_stream="out", signal="sum",
                                   smooth_s=0.0, stats=False, dpi=130, width_in=15, height_in=7,
                                   suptitle=f"{sensor} {tag}：显示 vs 零基准（调零口径，理想显示=0；"
                                            f"跌破 0 = 过减：现役最深 −{v['live']['over_adc']:.0f} ADC → "
                                            f"新参数 −{v['best']['over_adc']:.0f} ADC，容差 150）")
            print(f"  D {png.name}: {'OK' if res.get('path') else res.get('error')}")
            # 偏差 overlay（相对零基准：负值深度即过减深度）
            gb = TMP / f"E_{sensor}" / tag.replace("/", "_")
            write_csv(gb / f"{tag.replace('/', '_')}·新最优" / "device_001_seg000.csv",
                      t, arr["new"][:, 2])
            write_csv(gb / f"{tag.replace('/', '_')}·现役默认" / "device_001_seg000.csv",
                      t, arr["live"][:, 2])
            write_csv(gb / f"{tag.replace('/', '_')}·未补偿" / "device_001_seg000.csv",
                      t, tin)
            e_dirs.append(gb)
        dirs = [str(d / n) for d in e_dirs for n in sorted(x.name for x in d.iterdir())]
        png = FIG / f"E_{sensor}_显示与零基准.png"
        res = api.plot_to_file(dirs, str(png), mode="overlay", series_by="dir",
                               series_stream="out", signal="sum", smooth_s=0.0, dpi=130,
                               width_in=15, height_in=7,
                               suptitle=f"{sensor}：显示值（调零口径，理想 = 0 线；"
                                        f"跌破 0 的深度即过减，容差 −150 ADC）")
        print(f"  E {png.name}: {'OK' if res.get('path') else res.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
