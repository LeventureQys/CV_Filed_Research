# -*- coding: utf-8 -*-
"""t4b_dbg2_margin.py（调试件）—— 判据统计量 s 对"观察帧相对 t_on 偏移"的敏感性。

Q5 的判据 P1 用事件前的基线，理论上 Δ=0 就能定。但工程上需要知道：
**观察点比真沿早/晚几帧时，s 还稳不稳**（真沿估计误差 = 单帧跳变回溯的 ±1~2 帧，
外加变载实录的 ±1 包 ≈ 40 ms）。本脚本直接给出 s(偏移) 曲线。

产物：results/t4b_dbg_margin.csv
运行：python scripts/t4b_dbg2_margin.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_ad_lib as L          # noqa: E402
import t4b_common as C          # noqa: E402

OFFS = [-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10]     # 帧（10 ms/帧）


def main():
    ev, cache = C.build_events(verbose=False)
    rise = ev[ev.kind.isin(["onset", "restep"])]
    rows = []
    for name, cc in cache.items():
        loc = rise[rise.rec == name]
        if not len(loc):
            continue
        ts, dt, n = cc["tu"], cc["dt"], len(cc["tu"])
        zc = L.med_smooth(cc["tot"], C.DET_SMOOTH, causal=True)
        for _, r in loc.iterrows():
            k0 = int(np.searchsorted(ts, r.t_on))
            v = float(np.max(zc[:k0 + 1]))
            row = dict(rec=name, t_on=r.t_on, kind=r.kind, V=v, peak=cc["peak"])
            for o in OFFS:
                k = min(max(1, k0 + o), n - 1)
                i0 = max(1, k - int(0.50 / dt))
                base = float(zc[i0:k].mean()) if k > i0 else float(zc[k])
                row["s_off_%+03d" % o] = round(base / max(v, 1e-9), 4)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RES, "t4b_dbg_margin.csv"), index=False, encoding="utf-8-sig")
    cols = ["s_off_%+03d" % o for o in OFFS]
    print("== s 随观察帧偏移的变化（中位；偏移 = 观察点 − 真沿，负 = 比真沿早）==")
    print(pd.concat([df.groupby("kind")[cols].median(),
                     df.groupby("kind")[cols].max().add_prefix("max_")], axis=1).to_string())
    print("\n== 用门限 0.30 时的误判数（每档偏移，全体 35 事件）==")
    for c in cols:
        pred_on = df[c] <= 0.30
        err = int((((df.kind == "onset") & ~pred_on) | ((df.kind == "restep") & pred_on)).sum())
        print("  偏移 %s 帧（%+5.0f ms）：误判 %d / %d" % (c[6:], int(c[6:]) * 10, err, len(df)))
    print("\n-> results/t4b_dbg_margin.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
