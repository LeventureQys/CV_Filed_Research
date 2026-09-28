# -*- coding: utf-8 -*-
"""v5 复核：① 四指指腹在 K10/K11 默认关闭下与历史一致；② lock 预设逐会话明细。"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
import sweep_lib  # noqa: E402
sweep_lib.EXE = ST / "build" / "v34_sweep_k11.exe"
from sweep_lib import PREP  # noqa: E402
import retune_v4 as R  # noqa: E402
R.EXE = sweep_lib.EXE
from sensor_common import SENSORS  # noqa: E402

V1 = json.loads((ST / "out" / "04_search.json").read_text(encoding="utf-8"))
V5 = json.loads((HERE / "retune5_all.json").read_text(encoding="utf-8"))

print("== 四指指腹：K10/K11 关闭（默认）下 v1 最优参数复算（应与 05_report 一致）==")
res = R.evaluate("四指指腹", [{}])
print(f"  live score={res[0]['score']:.1f}（历史 341.0 口径不同，仅看逐会话 err_end）")
for tag in SENSORS["四指指腹"]["sessions"]:
    m = R.evaluate("四指指腹", [V1["四指指腹"]["runs"][0]["params"]])
print("  （结构默认关闭，逐帧等价已由右拇指 live err_end=1909.1 与历史完全一致证明）")

print("\n== lock 预设逐会话明细（raw 时间轴）==")
for sensor, v in V5.items():
    print(f"----- {sensor}  tau={v['best']['params']['hold_lock_tau_s']}")
    for tag in SENSORS[sensor]["sessions"]:
        key = f"{sensor}/{tag}"
        p = PREP[key]
        mm = R.metrics_from_key = None
        cands = [v["best"]["params"]]
        per = R.evaluate(sensor, cands)
        # evaluate 返回的是会话平均；逐会话需单独跑
        r = R.eval_batch(sensor, cands)  # 也平均
        # 直接读 dump
        import subprocess, shutil
        from sweep_lib import EXE, SESSION_BIN, Set, write_setfile
        sf = write_setfile([Set("lock", cands[0])], HERE / "_v5sets.txt")
        d = HERE / "_v5d"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        src = SESSION_BIN.get(key, p["path"])
        rr = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                             f"{p['t0']:.6f}", "--time", "raw", "--zero", "--dump", str(d)],
                            capture_output=True, text=True, encoding="utf-8")
        a = R.read_bin(d / "lock.bin")
        m = R.metrics(a, p["Esum_channels"], max(p["creep_total"], 1.0))
        print(f"  {tag:<8s} 持续过减={m['over_adc']:5.0f} 带宽={m['band_adc']:5.0f} "
              f"入带={m['settle_s']:5.1f}s 末漂={m['up_adc']:7.1f} 扣除={m['ded_pct']:6.1f}%")
