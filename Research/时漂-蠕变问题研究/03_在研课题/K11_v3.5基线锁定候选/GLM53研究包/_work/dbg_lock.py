# -*- coding: utf-8 -*-
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
from sweep_lib import EXE, PREP, Set, write_setfile  # noqa: E402

LOCK = {"r_fast": 0.0, "slope_gate_frac": 1e-6, "r_slow_max": 0.0,
        "hold_eps": 0.0, "hold_lock_tau_s": 0.3, "hold_lock_freeze_s": 2.0,
        "edge_slope_thres": 60.0}
sf = write_setfile([Set("lock03", LOCK)], HERE / "_dbg_sets.txt")
print("setfile:", sf.read_text(encoding="utf-8"))
p = PREP["右手掌/d1/A"]
r = subprocess.run([str(EXE), p["path"], str(sf), f"{p['Esum_channels']:.6f}",
                    f"{p['t0']:.6f}", "--time", "raw", "--zero"],
                   capture_output=True, text=True, encoding="utf-8")
print("EXE =", EXE, "mtime =", __import__("datetime").datetime.fromtimestamp(EXE.stat().st_mtime))
print(r.stdout)
print(r.stderr)
import os
tmp = os.environ.get("TEMP", "") + "\\lk2.txt"
if Path(tmp).exists():
    r2 = subprocess.run([str(EXE), p["path"], tmp, f"{p['Esum_channels']:.6f}",
                         f"{p['t0']:.6f}", "--time", "raw", "--zero"],
                        capture_output=True, text=True, encoding="utf-8")
    print("== TEMP lk2.txt via python ==")
    print(r2.stdout)
    print(r2.stderr)
