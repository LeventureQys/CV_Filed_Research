# -*- coding: utf-8 -*-
"""97_palm5_tune：在力值（N）流上复算，按「下坠≤0.2N、上漂可接受」重调 8 参数。

会话 20260927_102924（display=force、算法关、15.5 s）：seg 即算法输入口径（调零后 N）。
假设用户写入 = 上轮推荐表（rf.10 τc1=15 conf1 soft2 cap.008 rsm.10 τr1=6 τrsi=0.5），
先复算其表现，再扫 cap/rf/τc1 找 N 尺度下的合规档。
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
OUT = TEMP / "palm5" / "out"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, V, tin = z["t"], z["seg"], z["tot_seg"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    # 台阶
    kk = 30
    d = np.abs(tin[kk:] - tin[:-kk])
    hot = np.flatnonzero(d > 3.0)
    i0 = int(np.argmax(d))
    T0 = float(t[i0])
    me = (t >= T0 + 0.1) & (t <= T0 + 0.6)
    E = float(tin[me].min())
    m = t >= T0 + 0.1
    print(f"台阶@{T0:.2f}s  E={E:.3f} N  输入峰={tin.max():.3f}@{t[np.argmax(tin)]:.2f}s  "
          f"末={tin[-1]:.3f} N")
    print(f"输入自身：峰−末={tin[m].max()-tin[m][-1]:.3f} N（不含算法的天然回调）")

    def ev(p, label):
        r = obs.run(t, V, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"], fps)
        seg_y, seg_t = y[m], t[m]
        end = float(seg_y[-1])
        out = {"drop": float(seg_y.max() - end), "dev": end - E,
               "over": float(max(0.0, E - seg_y.min())),
               "peak_t": float(seg_t[np.argmax(seg_y)])}
        bad = np.abs(seg_y - end) > 0.2
        out["tsettle"] = float(seg_t[np.flatnonzero(bad)[-1]] - T0) if bad.any() else 0.0
        mm = seg_t >= seg_t[-1] - 4.0
        out["slope_end"] = float(np.polyfit(seg_t[mm], seg_y[mm], 1)[0])
        print(f"{label:<40s} 下坠={out['drop']:6.3f} 落点={out['dev']:+7.3f} "
              f"过减={out['over']:6.3f} 稳定={out['tsettle']:5.2f}s 末斜率={out['slope_end']:+7.4f}N/s")
        return out

    WRITTEN = {"r_fast": 0.10, "tau_c_fast_s": 15, "slow_confirm_s": 1,
               "soft_unfreeze_s": 2, "slope_cap_frac": 0.008, "r_slow_max": 0.10,
               "tau_r_fast_s": 6, "tau_r_slow_idle_s": 0.5}
    rows = []
    print(f"\n{'参数集':<40s} 下坠(N)  落点(N)  过减(N)  稳定(s)  末斜率")
    rows.append(("written", WRITTEN, "已写入（假设=上轮推荐 rf.10 tc15 cap.008）",
                 ev(WRITTEN, "已写入（假设=上轮推荐 rf.10 tc15 cap.008）")))
    base = dict(WRITTEN)
    for rf, tc1, cap, rsm in itertools.product(
            [0.03, 0.06, 0.10], [8.0, 15.0], [0.003, 0.005, 0.006, 0.008],
            [0.05, 0.10]):
        p = {**base, "r_fast": rf, "tau_c_fast_s": tc1, "slope_cap_frac": cap,
             "r_slow_max": rsm}
        lab = f"rf={rf:g} τc1={tc1:g} cap={cap:g} rsm={rsm:g}"
        rows.append((f"{rf}_{tc1:g}_{cap}_{rsm}", p, lab, ev(p, lab)))
    ok = [r for r in rows if r[3]["drop"] <= 0.2 and r[3]["over"] <= 0.05]
    print("\n== 红线内（下坠≤0.2N、过减≤0.05N），按 |落点| 排序 ==")
    for r in sorted(ok, key=lambda r: abs(r[3]["dev"]))[:10]:
        a = r[3]
        print(f"  {r[2]:<40s} 下坠={a['drop']:6.3f} 落点={a['dev']:+7.3f} "
              f"稳定={a['tsettle']:5.2f}s 末斜率={a['slope_end']:+7.4f}")
    (OUT / "97_tune.json").write_text(json.dumps(
        [{"name": n, "label": l, "params": p, "m": mm} for n, p, l, mm in rows],
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
