# -*- coding: utf-8 -*-
"""a2 抖动注入扫描：v6 在不同幅度·带宽扰动下的行为。

核心判据（全部相对「同底噪、无扰动」的基线运行）：
  n_epoch      事件（epoch）数 —— 无扰动基线 = 真实变载数，多出来的是误触发
  dAmax        max_t(ΣA) 的最大变化 —— 基线锚点的最大错误（ADC）
  maxdev       显示(总量)相对基线的最大偏差（ADC）
  slow_dev     两者都处于慢相帧上的显示偏差中位数 —— **稳态/基线错误**（"基线混乱"指标）
  after_dev    扰动停止后（最后 10% 时间）的显示偏差
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import (REC, load_uniform, make_perturb, run_traced,   # noqa: E402
                      TRACED_V6, ROOT)

RES = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
os.makedirs(RES, exist_ok=True)

AMPS = [200, 500, 1000, 2000]
KINDS = [("white", dict(f_hi=40.0), "独立白噪"),
         ("band", dict(f_lo=0.3, f_hi=5.0), "带限0.3-5Hz")]
RECS = os.environ.get("A2_RECS", "切换负载-快相无责;零负载-切换负载-零负载-再切换负载").split(";")
SEEDS = [int(s) for s in os.environ.get("A2_SEEDS", "0,1").split(",")]


def perturb_on_mask(X, A, kind, rng, kw, ch_mask):
    E = make_perturb(X, A, kind, rng, **kw) * ch_mask[None, :]
    s = E.sum(axis=1).std()
    return E * (A / s) if s > 1e-12 else E


def main():
    rows = []
    for name in RECS:
        d = load_uniform(REC[name])
        tu, X = d["tu"], d["Xu"]
        tot0 = X.sum(axis=1)
        lvl = np.median(tot0[tot0 > 0.5 * tot0.max()])
        max_tot = tot0.max()
        ch_mask = np.ones(X.shape[1])
        b6 = run_traced(TRACED_V6, tu, X)
        Y0 = b6["Y"]
        totY0 = Y0.sum(axis=1)
        A0 = np.sum(b6["comp"].A)
        n_ep0 = len(b6["epoch"])
        print(f"== [{name}] 受载电平≈{lvl:.0f} max_tot={max_tot:.0f} "
              f"空载额外门限10%max={0.10*max_tot:.0f} 受载门限5%lvref≈{0.05*lvl:.0f} "
              f"基线 epoch={n_ep0} ΣA={A0:.0f}", flush=True)
        print("   基线 epoch: " + "; ".join(f"{e[0]:.2f}({e[1]})" for e in b6["epoch"]), flush=True)
        rows.append(dict(rec=name, kind="baseline", amp=0, seed=-1, n_epoch=n_ep0,
                         n_epoch_delta=0, maxdev=0.0, slow_dev=0.0, after_dev=0.0,
                         dAmax=0.0, revoke=len(b6["revoke"]), handoff=len(b6["handoff"]),
                         t_epochs=";".join(f"{e[0]:.2f}({e[1]})" for e in b6["epoch"])))
        for kind, kw, lbl in KINDS:
            for amp in AMPS:
                for seed in SEEDS:
                    t0 = time.time()
                    rng = np.random.default_rng(seed)
                    Xp = X + perturb_on_mask(X, amp, kind, rng, kw, ch_mask)
                    r6 = run_traced(TRACED_V6, tu, Xp)
                    Y = r6["Y"]
                    dev = Y.sum(axis=1) - totY0
                    n = len(tu)
                    tail = slice(int(0.9 * n), n)
                    slow = (r6["st"] == 2) & (b6["st"] == 2)
                    ep = r6["epoch"]
                    rows.append(dict(
                        rec=name, kind=lbl, amp=amp, seed=seed, n_epoch=len(ep),
                        n_epoch_delta=len(ep) - n_ep0,
                        maxdev=round(float(np.abs(dev).max()), 1),
                        slow_dev=round(float(np.median(dev[slow])), 1) if slow.sum() > 50 else "",
                        after_dev=round(float(np.max(np.abs(dev[tail]))), 1),
                        dAmax=round(float(np.sum(r6["comp"].A) - A0), 1),
                        revoke=len(r6["revoke"]), handoff=len(r6["handoff"]),
                        t_epochs=";".join(f"{e[0]:.2f}({e[1]})" for e in ep)))
                    print(f"  {lbl:11s} A={amp:5d} s={seed} epoch={len(ep):3d}(Δ{len(ep)-n_ep0:+d}) "
                          f"maxdev={np.abs(dev).max():7.0f} slow_dev={np.median(dev[slow]):8.0f} "
                          f"dA={np.sum(r6['comp'].A)-A0:8.0f} ho={len(r6['handoff'])} "
                          f"rev={len(r6['revoke'])} [{time.time()-t0:.1f}s]", flush=True)
                    if amp == 200 and seed == 0:
                        print("     epoch: " + "; ".join(f"{e[0]:.2f}({e[1]})" for e in ep), flush=True)
                        print("     handoff: " + "; ".join(
                            f"{h[0]:.2f}/{h[1]}/{h[2]:.0f}" for h in r6["handoff"]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a2_jitter_sweep.csv"), index=False, encoding="utf-8-sig")
    sub = df[df.kind != "baseline"]
    print(sub.groupby(["rec", "kind", "amp"])[["n_epoch_delta", "maxdev", "slow_dev", "dAmax"]]
          .mean().to_string())


if __name__ == "__main__":
    main()
