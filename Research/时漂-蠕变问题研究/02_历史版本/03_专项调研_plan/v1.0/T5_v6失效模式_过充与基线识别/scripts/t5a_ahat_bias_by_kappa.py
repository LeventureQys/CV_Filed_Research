# -*- coding: utf-8 -*-
"""**实测** `Â/J − 1` 随 κ 的变化（onset，n=22）—— 定量说明"κ 压低反演高估"。

不依赖探针、不做时间对齐：对每个 κ 跑一遍 `t5a_common.KV6`，**每个事件单独重跑**，
在事件 τ=1 s 处直接读 `comp.ev["A_hat"]`（封顶后的目标值），除以该事件的原始台阶 `J`
（口径 = `t5a_common.j_and_pre`）。报中位 / p90 / max 与"相对基线的降幅"。

产出：`results/t5a_ahat_bias_by_kappa.csv`、`t5a_ahat_bias_perevent.csv`。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402

LOG = []


def rec(m):
    print(m, flush=True)
    LOG.append(m)


KS = [1.40, 1.35, 1.30, 1.25, 1.20, 1.15, 1.10, 1.05, 1.00, 0.95]


def ahat_at(ahat_hist, tu, t0, tau=1.0):
    """取事件 τ 处的 `Â`（`t5a_common.KV6.run` 返回的 `ahat` 逐帧数组）。"""
    i = int(np.searchsorted(tu, t0 + tau))
    if ahat_hist is None or i >= len(ahat_hist):
        return np.nan
    return float(ahat_hist[i])


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    onset = ev[ev.kind == "onset"]
    rows = []
    for kap in KS:
        for k, d in recs.items():
            sub = onset[onset.key == k]
            if not len(sub):
                continue
            out = C.run_v6(d, kappa_onset=kap, kappa_restep=1.12)
            tu = d["tu"]
            for _, e in sub.iterrows():
                t0 = float(e["t_on"])
                pre, post, J = C.j_and_pre(d["Z"], tu, t0, d["span"])
                if abs(J) < 1e-9:
                    continue
                a = ahat_at(out["ahat"], tu, t0, 1.0)
                if not np.isfinite(a):
                    continue
                rows.append(dict(kappa=kap, key=k, t_on=t0, J=J, A_hat_at_1s=a,
                                 bias_pct=(a / abs(J) - 1.0) * 100.0))
        rec(f"  κ={kap:.2f} 完成")
    PE = pd.DataFrame(rows)
    PE.to_csv(os.path.join(C.TASK, "results", "t5a_ahat_bias_perevent.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")
    summ = []
    base = None
    for kap, g in PE.groupby("kappa"):
        b = g.bias_pct
        if base is None:
            base = float(b.median())
        summ.append(dict(kappa=kap, n=len(g), bias_med=b.median(), bias_p90=b.quantile(.90),
                         bias_max=b.max(), bias_p10=b.quantile(.10), bias_min=b.min(),
                         n_over_5pct=int((b > 5).sum()),
                         d_vs_kappa140_med=float(b.median()) - base))
    S = pd.DataFrame(summ).sort_values("kappa", ascending=False)
    p = os.path.join(C.TASK, "results", "t5a_ahat_bias_by_kappa.csv")
    S.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"产出 {p}")
    lines = ["\n===== 实测 Â/J−1（onset，τ=1 s 快照）随 κ 的变化 =====",
             f"{'κ':>5s} {'n':>4s} {'中位':>8s} {'p90':>8s} {'max':>8s} {'min':>8s} "
             f"{'>5% 事件数':>10s} {'相对 κ=1.40 的中位变化':>16s}"]
    for _, r in S.iterrows():
        lines.append(f"{r.kappa:5.2f} {int(r.n):4d} {r.bias_med:7.3f}% {r.bias_p90:7.3f}% "
                     f"{r.bias_max:7.3f}% {r.bias_min:7.3f}% {int(r.n_over_5pct):10d} "
                     f"{r.d_vs_kappa140_med:15.3f} pt")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_ahat_bias.log"),
                "python scripts/t5a_ahat_bias_by_kappa.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
