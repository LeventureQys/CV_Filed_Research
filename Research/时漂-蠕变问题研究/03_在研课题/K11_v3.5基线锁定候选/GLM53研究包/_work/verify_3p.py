# -*- coding: utf-8 -*-
"""retune3p 右手掌变体的 raw 逐会话复核。"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
from sensor_common import SENSORS  # noqa: E402
from sweep_lib import EXE, PREP, SESSION_BIN, Set, write_setfile  # noqa: E402
from retune_v3 import read_bin  # noqa: E402

P = json.loads((HERE / "retune3p_all.json").read_text(encoding="utf-8"))["右手掌"]["best"]["params"]
V3 = json.loads((HERE / "retune3_all.json").read_text(encoding="utf-8"))["右手掌"]["best"]["params"]
named = [("v3", V3), ("v3p", P)]
d = HERE / "_v3p"
for tag in SENSORS["右手掌"]["sessions"]:
    key = f"右手掌/{tag}"
    p = PREP[key]
    sf = write_setfile([Set(n, o) for n, o in named], d / "_s.txt")
    if (d / "_d").exists():
        shutil.rmtree(d / "_d")
    (d / "_d").mkdir(parents=True)
    src = SESSION_BIN.get(key, p["path"])
    r = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                        f"{p['t0']:.6f}", "--time", "raw", "--zero", "--dump", str(d / "_d")],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300])
    creep = max(p["creep_total"], 1.0)
    for n, _ in named:
        a = read_bin(d / "_d" / f"{n}.bin")
        out = a[:, 2]
        print(f"{tag:>5s} {n:>4s} 过减={max(0.0, -float(out.min())):6.0f} "
              f"上漂={float(out[-1] - p['Esum_channels']):8.1f} "
              f"扣除={(a[:, 1][-1] - out[-1]) / creep * 100:6.1f}%")
