# -*- coding: utf-8 -*-
"""v3 复核：新参数在 raw 时间轴下的逐会话指标（过减 / 末值上漂 / 扣除率）。"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import EXE, PREP, SESSION_BIN, Set, write_setfile  # noqa: E402
from retune_v3 import read_bin  # noqa: E402

V3 = json.loads((HERE / "retune3_all.json").read_text(encoding="utf-8"))
V2 = json.loads((HERE / "retune_all.json").read_text(encoding="utf-8"))
WORK = HERE / "_verify3"
WORK.mkdir(exist_ok=True)

for sensor, v in V3.items():
    named = [("live", {}), ("v2", V2[sensor]["best"]["params"]), ("v3", v["best"]["params"])]
    print(f"\n===== {sensor}")
    print(f"  {'会话':<8s} {'参数':<5s} {'过减ADC':>8s} {'末上漂ADC':>10s} {'扣除%':>7s}")
    for tag in SENSORS[sensor]["sessions"]:
        key = f"{sensor}/{tag}"
        p = PREP[key]
        sf = write_setfile([Set(n, o) for n, o in named], WORK / "_sets.txt")
        d = WORK / "_d"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        src = SESSION_BIN.get(key, p["path"])
        r = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                            f"{p['t0']:.6f}", "--time", "raw", "--zero", "--dump", str(d)],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"{key}: {r.stderr[:300]}")
        creep = max(p["creep_total"], 1.0)
        for n, _ in named:
            a = read_bin(d / f"{n}.bin")
            out = a[:, 2]
            over = max(0.0, -float(out.min()))
            up = float(out[-1] - p["Esum_channels"])
            ded = float((a[:, 1][-1] - out[-1]) / creep * 100.0)
            print(f"  {tag:<8s} {n:<5s} {over:8.0f} {up:10.1f} {ded:7.1f}")
