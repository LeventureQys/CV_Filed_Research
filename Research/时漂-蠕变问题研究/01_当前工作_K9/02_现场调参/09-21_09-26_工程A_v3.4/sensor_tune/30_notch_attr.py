# -*- coding: utf-8 -*-
"""30 回落归因：把「显示回落 notch」拆成「输入自身的回落（真实压力波动）」与「算法额外造成的回落」。
   前者是工况本身（手指/夹具压力在变），后者才是算法该负责的部分。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import PREP, Set, run_sensor  # noqa: E402

PRESETS = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))


def med_filt(x: np.ndarray, k: int = 20) -> np.ndarray:
    n = x.size
    out = np.empty(n)
    for i in range(n):
        a, b = max(0, i - k), min(n, i + k + 1)
        out[i] = np.median(x[a:b])
    return out


def notch_of(x: np.ndarray, t: np.ndarray, t_from: float) -> tuple[float, float]:
    run = -1e300
    best, bt = 0.0, 0.0
    for i in range(t.size):
        if t[i] < t_from:
            continue
        run = max(run, x[i])
        if run - x[i] > best:
            best, bt = run - x[i], t[i]
    return float(best), float(bt)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"{'传感器':<12s} {'会话':<10s} {'输入回落':>8s} {'显示回落':>8s} {'算法多出':>8s} "
          f"{'过减ADC':>8s} {'欠减峰':>8s}")
    allrows = {}
    for sensor in SENSORS:
        res = run_sensor(sensor, [Set("cand", PRESETS[sensor]["params"])],
                         tag_name=f"attr_{sensor}")["cand"]
        # 显示曲线取 C++ 的 sm_*（0.2 s 中值）不够逐帧，这里用 Python 版重算同口径
        sys.path.insert(0, str(HERE.parent))
        from creep_observer_k9 import CreepObserverK9
        from sensor_common import LIVE, load
        from dataclasses import replace
        for tag in SENSORS[sensor]["sessions"]:
            key = f"{sensor}/{tag}"
            info = PREP[key]
            d = load(info["path"].split("data\\", 1)[-1].replace("\\", "/"))
            t = d["t"] - d["t"][0]
            V = d["V"] - d["V"][0]
            c = CreepObserverK9(replace(LIVE, **PRESETS[sensor]["params"]))
            c._trace_frame = lambda *a, **k: None
            out = np.array([c.process(float(t[i]), V[i]).sum() for i in range(len(t))])
            tin = V.sum(axis=1)
            n_in, _ = notch_of(med_filt(tin), t, info["t0"] + 3.0)
            n_out, _ = notch_of(med_filt(out), t, info["t0"] + 3.0)
            m = res[tag]
            print(f"{sensor:<12s} {tag:<10s} {n_in:8.1f} {n_out:8.1f} {n_out - n_in:8.1f} "
                  f"{m['low_exc']:8.1f} {m['peak_exc']:8.1f}")
            allrows[key] = {"notch_in": n_in, "notch_out": n_out,
                            "notch_extra": n_out - n_in, "low_exc": m["low_exc"],
                            "peak_exc": m["peak_exc"]}
        print()
    (OUT_DIR / "19_notch_attr.json").write_text(json.dumps(allrows, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    avg_in = np.mean([v["notch_in"] for v in allrows.values()])
    avg_out = np.mean([v["notch_out"] for v in allrows.values()])
    avg_ex = np.mean([v["notch_extra"] for v in allrows.values()])
    print(f"平均：输入自身回落 {avg_in:.0f} ADC，显示回落 {avg_out:.0f} ADC，"
          f"算法额外造成 {avg_ex:.0f} ADC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
