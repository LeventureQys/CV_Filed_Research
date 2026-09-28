# -*- coding: utf-8 -*-
"""复核：新参数（过减受限）在 raw 时间轴（上位机实际批量到达时刻）下的逐会话指标。"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import EXE, PREP, SESSION_BIN, Set, write_setfile  # noqa: E402
from retune_over import read_bin  # noqa: E402

RETUNE = json.loads((HERE / "retune_all.json").read_text(encoding="utf-8"))
SEARCH = json.loads((OUT_DIR / "04_search.json").read_text(encoding="utf-8"))
WORK = HERE / "_verify"
WORK.mkdir(exist_ok=True)

for sensor, v in RETUNE.items():
    named = [("live", {}), ("old", SEARCH[sensor]["runs"][0]["params"]),
             ("new", v["best"]["params"])]
    print(f"\n===== {sensor}")
    print(f"  {'会话':<10s} {'参数':<5s} {'末残ADC':>9s} {'末残%':>7s} {'过减ADC':>8s} {'扣除%':>7s}")
    for tag in SENSORS[sensor]["sessions"]:
        key = f"{sensor}/{tag}"
        p = PREP[key]
        sf = write_setfile([Set(n, o) for n, o in named], WORK / "_sets.txt")
        d = WORK / "_d"
        if d.exists():
            import shutil
            shutil.rmtree(d)
        d.mkdir(parents=True)
        src = SESSION_BIN.get(key, p["path"])
        r = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                            f"{p['t0']:.6f}", "--time", "raw", "--zero",
                            "--dump", str(WORK / "_d")],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"{key}: {r.stderr[:300]}")
        creep = max(p["creep_total"], 1.0)
        for n, _ in named:
            a = read_bin(WORK / "_d" / f"{n}.bin")
            out = a[:, 2]
            over = max(0.0, -float(out.min()))
            err = float(out[-1] - p["Esum_channels"])
            ded = float((a[:, 1][-1] - out[-1]) / creep * 100.0)
            print(f"  {tag:<10s} {n:<5s} {err:9.1f} {err / creep * 100:7.1f} {over:8.0f} {ded:7.1f}")
