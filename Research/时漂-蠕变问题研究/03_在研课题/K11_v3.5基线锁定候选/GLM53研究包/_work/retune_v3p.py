# -*- coding: utf-8 -*-
"""右手掌加压：UP_W=4（上漂超限与过减同罚力度）从 v3 最优再坐标下降一轮。"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retune_v3 as R  # noqa: E402

R.UP_W = 4.0
R.WORK = HERE / "_retune3p"
R.DUMP = R.WORK / "_dump"
v3 = json.loads((HERE / "retune3_all.json").read_text(encoding="utf-8"))

out = {}
for sensor in ("右手掌", "左拇指指腹"):
    print(f"\n########## {sensor} (UP_W=4)", flush=True)
    start = dict(v3[sensor]["best"]["params"])
    m0 = R.evaluate(sensor, [start])[0]
    print(f"  v3: score={m0['score']:.2f} 过减={m0['over_adc']:.0f} 上漂={m0['up_adc']:.0f}", flush=True)
    p, val, tr = R.coord_descent(sensor, start, passes=4)
    mb = R.evaluate(sensor, [p])[0]
    print(f"  ★ score={val:.2f} 过减={mb['over_adc']:.0f} 上漂={mb['up_adc']:.0f} "
          f"扣除={mb['ded_pct']:.0f}%", flush=True)
    print(f"  {json.dumps(p, ensure_ascii=False)}", flush=True)
    out[sensor] = {"best": {**mb, "params": p}}
(HERE / "retune3p_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
print("\n写出 retune3p_all.json")
