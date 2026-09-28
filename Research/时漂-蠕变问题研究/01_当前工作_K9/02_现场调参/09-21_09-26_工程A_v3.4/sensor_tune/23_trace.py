# -*- coding: utf-8 -*-
"""23 轨迹细看：把拇指/手掌几份会话在推荐预设下的显示轨迹逐秒打出来，
   找「x1 阶段的下陷（过减）」与「基线漂移」到底长什么样。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensor_common import OUT_DIR  # noqa: E402
from sweep_lib import PREP, Set, run_sensor  # noqa: E402

PRESETS = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))
PROBE = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180]
CASES = ["右拇指指腹/d1", "左拇指指腹/d1", "右手掌/d1/A", "左手掌/d1/A"]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    for key in CASES:
        sensor, tag = key.split("/", 1)
        best = PRESETS[sensor]["params"]
        res = run_sensor(sensor, [Set("live", {}), Set("preset", best)],
                         tag_name=f"trace_{sensor}")
        d = np.load(OUT_DIR / "_prep" / (key.replace("/", "_") + ".npz"))
        t = d["t"]
        tin = d["tin"]
        print(f"\n===== {key}  （真实蠕变 {PREP[key]['creep_total']:.0f} ADC）")
        print(f"  {'t':>6s} {'输入':>8s} │ {'现役显示':>9s} {'现役扣':>8s} │ "
              f"{'推荐显示':>9s} {'推荐扣':>8s} │ {'推荐下陷':>9s} {'推荐回升':>9s}")
        # 下陷 = 相对"到该时刻为止显示最高点"的回落；回升 = 相对"到该时刻为止最低点"的上抬
        for nm in ("live", "preset"):
            pass
        # 现役与推荐的显示轨迹先从 C++ 侧拿不到逐帧，用 Pythhon 版重算（已对拍等价）
        sys.path.insert(0, str(HERE.parent))
        from creep_observer_k9 import CreepObserverK9
        from sensor_common import LIVE, load
        from dataclasses import replace
        dd = load(PREP[key]["path"].split("data\\", 1)[-1].replace("\\", "/"))
        tt = dd["t"] - dd["t"][0]
        VV = dd["V"] - dd["V"][0]
        outs = {}
        for nm, over in (("live", {}), ("preset", best)):
            c = CreepObserverK9(replace(LIVE, **over))
            c._trace_frame = lambda *a, **k: None
            outs[nm] = np.array([c.process(float(tt[i]), VV[i]).sum() for i in range(len(tt))])
        runmax = np.maximum.accumulate(outs["preset"])
        runmin = np.minimum.accumulate(outs["preset"])
        for x in PROBE:
            i = int(np.searchsorted(tt, x))
            if i >= len(tt):
                continue
            print(f"  {tt[i]:6.1f} {tin[i]:8.1f} │ {outs['live'][i]:9.1f} "
                  f"{tin[i] - outs['live'][i]:8.1f} │ {outs['preset'][i]:9.1f} "
                  f"{tin[i] - outs['preset'][i]:8.1f} │ "
                  f"{runmax[i] - outs['preset'][i]:9.1f} {outs['preset'][i] - runmin[i]:9.1f}")
        dd_max = float(np.max(runmax - outs["preset"]))
        i_dd = int(np.argmax(runmax - outs["preset"]))
        print(f"  → 推荐预设最大下陷（相对历史最高点）= {dd_max:.1f} ADC @ t={tt[i_dd]:.1f}s；"
              f"末值 = {outs['preset'][-1]:.1f} ADC；"
              f"全程最高 = {outs['preset'].max():.1f} ADC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
