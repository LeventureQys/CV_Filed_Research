# -*- coding: utf-8 -*-
"""K11 探针：hold_lock 纯锁定（x 支路关闭）在拇指/手掌上的行为。"""
import subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
from sensor_common import OUT_DIR  # noqa: E402
import sweep_lib  # noqa: E402
sweep_lib.EXE = ST / "build" / "v34_sweep_k11.exe"
from sweep_lib import PREP, SESSION_BIN, Set, write_setfile  # noqa: E402
EXE = sweep_lib.EXE
from retune_v4 import metrics, read_bin  # noqa: E402

WORK = HERE / "_k11"
WORK.mkdir(exist_ok=True)


def run(key, named):
    p = PREP[key]
    sf = write_setfile([Set(n, o) for n, o in named], WORK / "_s.txt")
    d = WORK / "_d"
    if d.exists():
        import shutil
        shutil.rmtree(d)
    d.mkdir()
    src = SESSION_BIN.get(key, p["path"])
    r = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                        f"{p['t0']:.6f}", "--time", "raw", "--zero", "--dump", str(d)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:400])
    return {n: read_bin(d / f"{n}.bin") for n, _ in named}


# 纯锁定：x1/x2 关（r_fast=0、gate 关到不积分），只留 lock_lp
LOCK = {"r_fast": 0.0, "slope_gate_frac": 1e-6, "r_slow_max": 0.0,
        "hold_eps": 0.0, "hold_lock_tau_s": 0.3, "hold_lock_freeze_s": 2.0,
        "edge_slope_thres": 60.0}

print("== 等价性（live）==")
r = run("右拇指指腹/d1", [("live", {})])
print(f"  右拇指/d1 live err_end={r['live'][:, 2][-1] - PREP['右拇指指腹/d1']['Esum_channels']:.1f}（历史 1909.1）")

for key in ("右拇指指腹/d1", "右拇指指腹/d2", "左拇指指腹/d1", "左拇指指腹/d2",
            "右手掌/d1/A", "右手掌/d2/B", "左手掌/d1/A", "左手掌/d2/B"):
    p = PREP[key]
    named = [("live", {}),
             ("lock03", LOCK),
             ("lock05", {**LOCK, "hold_lock_tau_s": 0.5}),
             ("lock08", {**LOCK, "hold_lock_tau_s": 0.8}),
             ("lock03+track", {**LOCK, "hold_lock_tau_s": 0.3,
                               "track_base_frac": 1.0, "slope_gate_frac": 5.0,
                               "slope_cap_frac": 0.3, "r_slow_max": 30.0,
                               "slow_confirm_s": 0.0, "soft_unfreeze_s": 0.25,
                               "tau_slope_s": 1.0, "tau_r_slow_s": 600.0})]
    r = run(key, named)
    print(f"== {key} ==")
    for n, _ in named:
        m = metrics(r[n], p["Esum_channels"], max(p["creep_total"], 1.0))
        print(f"    {n:<14s} 过减={m['over_adc']:5.0f} 带宽={m['band_adc']:5.0f} "
              f"入带={m['settle_s']:5.1f}s 末漂={m['up_adc']:7.1f} 扣除={m['ded_pct']:6.1f}% "
              f"score={m['score']:6.1f}")
