# -*- coding: utf-8 -*-
"""106_why_climb：为什么改参后仍缓慢上爬——速率跟踪 vs 电平回归。

验证三件事（152928 真实输入）：
① 推荐档的显示分窗斜率：锁定过程中显示仍在缓慢上爬，多久后斜率→0；
② "存量欠补"不可恢复：x2 只积分 (slope − dx1_rate)，稳态只追**速率**，
   头几秒没补上的缺口永远留在显示上（显示最终平台 = E + 存量）;
③ 要把存量压小/收束更快的组合及其下坠代价。
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
s93 = import_module("93_palm4_sweep")
s100 = import_module("100_palm5_synth")
OUT = TEMP / "palm6" / "out"
E1 = 16481.0


def ev(p, label, show_win=True):
    z = np.load(OUT / "streams.npz")
    t, pre = z["t"], z["pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    r = obs.run(t, pre, obs.default_with(p))
    y = s93.medfilt1s(r["out_tot"], fps)
    m = t >= 1.6
    ts, ys = t[m], y[m]
    end = ys[-1]
    if show_win:
        wins = []
        for a, b in ((3, 8), (8, 15), (15, 25), (25, 35), (35, 41)):
            mm = (ts >= a) & (ts < b)
            if mm.sum() > 50:
                wins.append(f"{a}~{b}s:{np.polyfit(ts[mm], ys[mm], 1)[0]:+6.1f}")
        print(f"  分窗斜率(ADC/s)：{'  '.join(wins)}")
    t2 = np.arange(0, 30.0, 1.0 / 100.7)
    yf = s100.synth(t2, level_n=17.0)
    Vf = np.repeat(yf[:, None] * 1000.0 / 71.0, 71, axis=1)
    rf_ = obs.run(t2, Vf, obs.default_with(p))
    yflat = s93.medfilt1s(rf_["out_tot"] / 1000.0, 100.7)
    mf = t2 >= 2.2
    print(f"{label:<44s} 落点={end-E1:+7.1f}({(end-E1)*0.001:+.3f}N) 下坠={ys.max()-end:6.1f} "
          f"过减={max(0.0, E1-ys.min()):5.1f} 恒压下坠={(yflat[mf].max()-yflat[mf][-1])*1000:6.1f}")
    return {"dev": end - E1, "drop": ys.max() - end, "over": max(0.0, E1 - ys.min())}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    NEW = {"r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0,
           "soft_unfreeze_s": 1.5, "slope_cap_frac": 0.005, "r_slow_max": 0.15,
           "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5}
    print("① 上轮推荐档的分窗爬升速率（152928 输入）：")
    ev(NEW, "推荐档 rf.06 τc1=2 conf2 cap.005 rsm.15")
    print("\n② 收束更快的组合（提前开慢态 / 加大 rf，152928 输入 + 恒压合成）：")
    BASE = dict(NEW)
    rows = []
    for rf, conf, soft, cap in itertools.product(
            [0.06, 0.08], [0.5, 1.0, 2.0], [1.5], [0.004, 0.006, 0.008]):
        p = {**BASE, "r_fast": rf, "slow_confirm_s": conf, "soft_unfreeze_s": soft,
             "slope_cap_frac": cap}
        print(f"--- rf={rf:g} conf={conf:g} soft={soft:g} cap={cap:g} rsm=0.15")
        ev(p, f"rf={rf:g} conf={conf:g} soft={soft:g} cap={cap:g}", show_win=False)
        rows.append((p, f"rf={rf:g} conf={conf:g} soft={soft:g} cap={cap:g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
