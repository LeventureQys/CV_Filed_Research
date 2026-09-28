# -*- coding: utf-8 -*-
"""132_palm8_minset：161747 长保压——"最小参数清理尾漂"验证。

用户假设：快漂很快结束、尾端蠕变速率很慢 ⇒ 尾漂不需要大参数即可 cover，且过扣最小。
验证：① 实测尾端输入速率 vs cap/门限的裕量；② 只动 rsm / 只再压 cap 的最小改动档；
      ③ 稳定性指标：尾段(50~419s)显示 max-min、以及落点/下坠/过扣。
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
OUT = TEMP / "palm8" / "out"
E1 = 16502.0
T0 = 1.2

CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
       "slope_cap_frac": 0.025, "r_slow_max": 0.03, "tau_r_fast_s": 6.0,
       "tau_r_slow_idle_s": 0.5}


def load():
    z = np.load(OUT / "streams.npz")
    return z["t"], z["V"]


def med(y, fps):
    return s93.medfilt1s(y, fps)


def ev(t, V, p, fps):
    r = obs.run(t, V, obs.default_with(p))
    y = med(r["out_tot"], fps)
    m = t >= T0 + 0.6
    i10 = int(np.searchsorted(t, T0 + 10))
    i50 = int(np.searchsorted(t, T0 + 50))
    tail = y[i50:]
    end = float(y[-1])
    mm = t[m] >= t[m][-1] - 5.0
    return {"dev": end - E1, "climb": float(end - y[i10]),
            "tail_span": float(tail.max() - tail.min()),
            "drop": float(y[m].max() - end), "over": float(max(0.0, E1 - y[m].min())),
            "slope": float(np.polyfit(t[m][mm], y[m][mm], 1)[0]),
            "x2": float(r["x2"][-1]), "x1": float(r["x1"][-1])}


def ev_flat(p, secs=60.0):
    tt = np.arange(0, secs, 1.0 / 100.7)
    ytot = s100.synth(tt, level_n=17.0)
    V = np.repeat(ytot[:, None] * 1000.0 / 71.0, 71, axis=1)
    r = obs.run(tt, V, obs.default_with(p))
    y = med(r["out_tot"] / 1000.0, 100.7)
    m = tt >= 2.2
    return float(y[m].max() - y[m][-1])


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, V = load()
    fps = (len(t) - 1) / (t[-1] - t[0])
    tin = V.sum(axis=1)

    # ① 尾端输入速率实测
    print("== 输入蠕变速率（1 s 中值后差分，ADC/s 总值 / 折合每通道）")
    yin = med(tin, fps)
    for a, b in [(2, 5), (5, 10), (10, 30), (30, 60), (60, 120), (120, 240), (240, 419)]:
        ia, ib = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
        sl = np.polyfit(t[ia:ib], yin[ia:ib], 1)[0]
        print(f"  {a:3d}~{b:3d}s: {sl:+7.2f} ADC/s 总（每通道 {sl/71:+6.3f}，即 {sl/1000:+.4f} N/s）")

    # ② 候选：现参数基础上最小改动
    cands = [("cur_rsm03", CUR, "现参数（rsm.03 封顶）")]
    for rsm in [0.06, 0.10, 0.15, 0.30]:
        cands.append((f"rsm{rsm:g}", {**CUR, "r_slow_max": rsm}, f"只改 rsm={rsm:g}"))
    for rsm, cap in itertools.product([0.10, 0.15], [0.005, 0.01]):
        cands.append((f"rsm{rsm:g}_cap{cap:g}", {**CUR, "r_slow_max": rsm, "slope_cap_frac": cap},
                      f"rsm={rsm:g} + cap={cap:g}"))
    cands.append(("min_rf02", {**CUR, "r_fast": 0.02, "r_slow_max": 0.15, "slope_cap_frac": 0.005},
                  "rf.02+rsm.15+cap.005（压快态下坠）"))
    cands.append(("rec", {**CUR, "r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0,
                          "slope_cap_frac": 0.005, "r_slow_max": 0.15}, "推荐档（参照）"))
    print(f"\n{'参数集':<26s}{'落点(N)':>9s}{'10s→末(N)':>10s}{'尾段带宽(N)':>11s}{'下坠(N)':>9s}"
          f"{'过扣(N)':>9s}{'末斜率':>8s}{'x1末':>7s}{'x2末':>7s}{'恒压下坠':>9s}")
    rows = []
    for name, p, label in cands:
        a = ev(t, V, p, fps)
        f = ev_flat(p)
        rows.append({"name": name, "label": label, "params": p, "real": a, "flat": f})
        print(f"{name:<26s}{a['dev']/1000:+9.3f}{a['climb']/1000:+10.3f}"
              f"{a['tail_span']/1000:11.3f}{a['drop']/1000:9.3f}{a['over']/1000:9.3f}"
              f"{a['slope']:+8.2f}{a['x1']:7.0f}{a['x2']:7.0f}{f:9.3f}")
    (OUT / "132_minset.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
