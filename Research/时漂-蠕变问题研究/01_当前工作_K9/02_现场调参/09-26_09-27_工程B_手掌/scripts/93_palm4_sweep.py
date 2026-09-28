# -*- coding: utf-8 -*-
"""93_palm4_sweep：ADC 模式找「不过减 + 快收敛 + 慢漂受控」的参数组。

判据（需求）：不允许过减（E−段内最小 ≤ 容差 ~50 ADC），允许向上慢漂一点点；
稳定时间 1~2 s（1 s 中值末次偏离落点 >300 ADC 的时刻，自台阶起算）；
下坠（峰−末）小。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
OUT = TEMP / "palm4" / "out"
T0 = 1.45
REC = {"r_fast": 0.3, "tau_c_fast_s": 40, "slow_confirm_s": 1, "soft_unfreeze_s": 2,
       "slope_cap_frac": 0.006, "r_slow_max": 0.05, "tau_r_fast_s": 6,
       "tau_r_slow_idle_s": 0.5}


def medfilt1s(y: np.ndarray, fps: float) -> np.ndarray:
    k = max(int(round(fps)) | 1, 3)
    pad = np.pad(y, k // 2, mode="edge")
    idx = np.arange(y.size)[:, None] + np.arange(k)[None, :]
    return np.median(pad[idx], axis=1)


def evaluate(r: dict, t: np.ndarray, tin: np.ndarray, fps: float) -> dict:
    y = medfilt1s(r["out_tot"], fps)
    me = (t >= T0 + 0.1) & (t <= T0 + 0.6)
    E = float(tin[me].min())
    m = t >= T0 + 0.1
    seg_y, seg_t = y[m], t[m]
    end = float(seg_y[-1])
    drop = float(seg_y.max() - end)
    over = float(max(0.0, E - seg_y.min()))
    dev = end - E
    # 稳定时间：末次 |中值−落点| > 300 ADC
    bad = np.abs(seg_y - end) > 300.0
    tsettle = float(seg_t[np.flatnonzero(bad)[-1]] - T0) if bad.any() else 0.0
    # 末段残余斜率（最后 4 s，正=向上慢漂）
    mm = seg_t >= seg_t[-1] - 4.0
    slope_end = float(np.polyfit(seg_t[mm], seg_y[mm], 1)[0]) if mm.sum() > 20 else 0.0
    return {"E": E, "over": over, "dev": dev, "drop": drop, "tsettle": tsettle,
            "slope_end": slope_end, "peak_t": float(seg_t[np.argmax(seg_y)])}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, pre, tin = z["t"], z["pre"], z["tot_pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])

    cands = [("rec", REC, "录制参数 rf.3")]
    for rf, tc1, cap, rsm in itertools.product(
            [0.06, 0.08, 0.10, 0.12], [4.0, 8.0, 15.0, 40.0],
            [0.004, 0.006], [0.05, 0.10]):
        cands.append((f"a_{rf}_{tc1:g}_{cap}_{rsm}",
                      {**REC, "r_fast": rf, "tau_c_fast_s": tc1,
                       "slope_cap_frac": cap, "r_slow_max": rsm},
                      f"rf={rf:g} τc1={tc1:g} cap={cap:g} rsm={rsm:g}"))
    rows = []
    for name, over_p, label in cands:
        r = obs.run(t, pre, obs.default_with(over_p))
        m = evaluate(r, t, tin, fps)
        rows.append({"name": name, "label": label, "params": over_p, "m": m})
        a = m
        print(f"{label:<34s} 过减={a['over']:7.1f} 落点={a['dev']:+8.1f} 下坠={a['drop']:7.1f} "
              f"稳定={a['tsettle']:5.2f}s 末斜率={a['slope_end']:+6.1f}ADC/s")
    (OUT / "93_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("\n== 合规章（过减≤50 且 稳定≤2.5s），按 |落点| 排序 ==")
    ok = [r for r in rows if r["m"]["over"] <= 50 and r["m"]["tsettle"] <= 2.5]
    for r in sorted(ok, key=lambda r: abs(r["m"]["dev"]))[:12]:
        a = r["m"]
        print(f"  {r['label']:<34s} 过减={a['over']:6.1f} 落点={a['dev']:+7.1f} "
              f"下坠={a['drop']:6.1f} 稳定={a['tsettle']:4.2f}s 末斜率={a['slope_end']:+5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
