# -*- coding: utf-8 -*-
"""151_b2d896_eval：b2d896（ADC、429 s、单台阶、不卸载）上的复算与「清尾漂」判据。

判据（ADC 口径，1 s 中值）：
  E        = 台阶后 0.1~0.6 s 输入最小值
  落点     = 段末显示 − E（读数固定偏移）
  过扣     = E − 段内最小显示（穿到目标电平以下的深度）——用户要「尽可能少」
  下坠     = 段内峰值 − 段末
  尾端斜率 = 末 60 s 线性拟合（「清尾漂」的主判据）
  10s→末  = 显示从 10 s 到末的变化量（沿用上一轮口径）
  冻结时刻 = 之后 |1 s 中值 − 段末| 再未超过 100 ADC 的最后时刻
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
fl = import_module("122_force_lib")
OUT = TEMP / "palm10" / "out"
E = 16502.0
T0 = 1.20
WIN = T0 + 0.1

CANDS = {
    "① 我上一轮力值推荐B（rf.05 τc1 2 cap.009 rsm.35）":
        {"r_fast": 0.05, "tau_c_fast_s": 2.0, "slope_cap_frac": 0.009, "r_slow_max": 0.35},
    "② 152928 ADC 推荐档（rf.06 τc1 2 conf2 soft1.5 cap.005 rsm.15）":
        {"r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0, "soft_unfreeze_s": 1.5,
         "slope_cap_frac": 0.005, "r_slow_max": 0.15},
    "③ 上一轮 161747 最小档（只改 cap.005 rsm.15）":
        {"slope_cap_frac": 0.005, "r_slow_max": 0.15},
    "④ 右手掌复检二次推荐（cap.012 rsm.25）":
        {"r_fast": 0.04, "tau_c_fast_s": 2.0, "slope_cap_frac": 0.012, "r_slow_max": 0.25},
    "⑤ 上一轮 161747 现参数（rf.04 τc1 1 conf1.5 soft1.5 cap.025 rsm.03）":
        {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
         "slope_cap_frac": 0.025, "r_slow_max": 0.03},
    "⑥ C++ 出厂默认（rf.01 τc1 40 conf5 soft8 cap.05 rsm.2）":
        {},
}


def evaluate(out_tot: np.ndarray, t: np.ndarray, fps: float) -> dict:
    y = s93.medfilt1s(out_tot, fps)
    m = t >= WIN
    yy, tt = y[m], t[m]
    end = float(yy[-1])
    lo = float(yy.min())
    hi = float(yy.max())
    tail = tt >= tt[-1] - 60.0
    slope_tail = float(np.polyfit(tt[tail], yy[tail], 1)[0])
    mid = (tt >= tt[-1] - 180.0) & (tt <= tt[-1] - 60.0)
    slope_mid = float(np.polyfit(tt[mid], yy[mid], 1)[0]) if mid.sum() > 20 else 0.0
    i10 = int(np.searchsorted(tt, 10.0))
    d10 = float(yy[-1] - yy[i10]) if i10 < yy.size else 0.0
    bad = np.abs(yy - end) > 100.0
    freeze = float(tt[np.flatnonzero(bad)[-1]]) if bad.any() else 0.0
    return {"end": end, "dev": end - E, "over": max(0.0, E - lo), "drop": hi - end,
            "slope_tail": slope_tail, "slope_mid": slope_mid, "d10": d10,
            "freeze": freeze, "peak": hi, "min": lo}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "b2d896.npz")
    t, V, tot = z["t"], z["V"], z["tot"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    print(f"n={len(t)} 时长={t[-1]:.1f}s fps={fps:.2f}  E={E:.0f} ADC")
    print(f"\n{'参数集':<52s} {'落点':>7s} {'过扣':>6s} {'下坠':>6s} "
          f"{'尾端斜率':>9s} {'中段斜率':>9s} {'10s→末':>7s} {'冻结@':>7s}")
    rows = []
    for name, over in CANDS.items():
        p = {**obs.Params().__dict__, **over} if False else obs.default_with(over)
        r = obs.run(t, V, p)
        e = evaluate(r["out_tot"], t, fps)
        print(f"{name:<52s} {e['dev']:+7.0f} {e['over']:6.0f} {e['drop']:6.0f} "
              f"{e['slope_tail']:+9.3f} {e['slope_mid']:+9.3f} {e['d10']:+7.0f} "
              f"{e['freeze']:7.1f}")
        rows.append({"name": name, "params": over, **e})
    (OUT / "151_cands.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
