# -*- coding: utf-8 -*-
"""132_adc_cross：把力值模式整定出的参数放到 ADC 显示会话上复算，量「跨模式代价」。

会话 20260927_152928_3efae9（display=adc、算法开、pre/seg 双流、41.8 s、71 通道、记录参数
rf.04/τc1 1/conf 1.5/soft 1.5/cap.025/rsm.03）。
ADC 口径：E = 台阶后 0.1~0.6 s 输入最小值（ADC）；过减 = E − 段内最小；落点 = 段末 − E；
红线：过减 ≤ 50 ADC、稳定 ≤ 2.5 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT6 = TEMP / "palm6" / "out"
OUT7 = TEMP / "palm7" / "out"


def segs_adc(t: np.ndarray, tin: np.ndarray) -> list[dict]:
    k = 30
    d = np.abs(tin[k:] - tin[:-k])
    hot = np.flatnonzero(d > 400)
    groups = [[int(hot[0])]]
    for i in hot[1:]:
        if i - groups[-1][-1] <= 60:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    st = [(float(t[g[int(np.argmax(d[g]))]]), g[int(np.argmax(d[g]))]) for g in groups]
    segs = []
    for j, (tt, i) in enumerate(st):
        m = (t >= tt + 0.1) & (t <= tt + 0.6)
        E = float(tin[m].min()) if m.any() else float(tin[i])
        nxt = st[j + 1][0] if j + 1 < len(st) else float(t[-1])
        if nxt - tt < 2.0 or E <= 100:
            continue
        segs.append({"t0": tt, "t1": nxt, "E": E, "E_fast": E,
                     "i0": int(np.searchsorted(t, tt + 0.1)),
                     "i1": int(np.searchsorted(t, nxt - 0.6))})
    return segs


def eval_adc(out_tot, t, segs, fps):
    y = s93.medfilt1s(out_tot, fps)
    rows = []
    for s in segs:
        a, b = s["i0"], max(s["i1"], s["i0"] + 1)
        yy, ttv = y[a:b], t[a:b]
        E = s["E"]
        end = float(yy[-1])
        peak = float(yy.max())
        lo = float(yy.min())
        bad = np.abs(yy - end) > 300.0
        tset = float(ttv[np.flatnonzero(bad)[-1]] - s["t0"]) if bad.any() else 0.0
        mm = ttv >= ttv[-1] - 4.0
        sl = float(np.polyfit(ttv[mm], yy[mm], 1)[0]) if mm.sum() > 20 else 0.0
        rows.append({"t0": s["t0"], "t1": s["t1"], "E": E, "dev": end - E,
                     "over": max(0.0, E - lo), "drop": peak - end,
                     "tsettle": tset, "slope_end": sl})
    return rows


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT6 / "streams.npz")
    t, pre = z["t"], z["pre"]
    tin = pre.sum(1)
    fps = (len(t) - 1) / (t[-1] - t[0])
    sg = segs_adc(t, tin)
    print(f"ADC 会话 3efae9：{len(t)} 帧 {t[-1]:.1f}s，{len(sg)} 段")
    for j, s in enumerate(sg):
        print(f"  段{j+1} {s['t0']:.2f}→{s['t1']:.2f}s E={s['E']:.0f} ADC "
              f"输入段末={float(tin[s['i1']]):.0f}")
    print(f"\n{'参数集':<40s} " + " | ".join(
        f"段{j+1} 过减/落点/下坠/稳定" for j in range(len(sg))))
    from_here = Path(__file__).resolve().parent
    sys.path.insert(0, str(from_here))
    import json
    cands = json.loads((OUT7 / "132_cands.json").read_text(encoding="utf-8")) \
        if (OUT7 / "132_cands.json").is_file() else {}
    for name, p in cands.items():
        r = obs.run(t, pre, obs.default_with(p))
        rows = eval_adc(r["out_tot"], t, sg, fps)
        cell = " | ".join(f"{x['over']:5.0f}/{x['dev']:+7.0f}/{x['drop']:5.0f}/{x['tsettle']:4.2f}s"
                          for x in rows)
        print(f"{name:<40s} {cell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
