# -*- coding: utf-8 -*-
"""a0 侦察：录制基本统计 + 无扰动基线下 v6/v5.1 的 epoch 结构（作为误触发的基准）。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import REC, load_uniform, run_traced, TRACED_V6, GLM53v51, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "figures"),
            exist_ok=True)

rows = []
base_epochs = {}
for name, path in REC.items():
    d = load_uniform(path)
    Xu, tu = d["Xu"], d["tu"]
    tot = Xu.sum(axis=1)
    # 底噪：空载段（总量 < 10% 最大）的逐帧总量差分的稳健 σ
    thr = 0.10 * tot.max()
    idle_m = tot < thr
    d1 = np.diff(tot)
    sig_idle = 1.4826 * np.median(np.abs(d1[idle_m[1:]] - np.median(d1[idle_m[1:]]))) if idle_m[1:].sum() > 100 else np.nan
    sig_all = 1.4826 * np.median(np.abs(d1 - np.median(d1)))
    t = d["t"]
    rows.append(dict(rec=name, n=len(t), span_s=round(d["span"], 2),
                     fs_real=round(len(t) / d["span"], 2),
                     tot_max=round(float(tot.max()), 1),
                     tot_idle_med=round(float(np.median(tot[idle_m])) if idle_m.any() else np.nan, 1),
                     idle_ratio=round(float(idle_m.mean()), 3),
                     sig_frame_idle=round(float(sig_idle), 2),
                     sig_frame_all=round(float(sig_all), 2)))

    r6 = run_traced(TRACED_V6, tu, Xu)
    Y51, c51 = None, None
    import time
    t0 = time.time()
    Y51, c51 = GLM53v51(Xu.shape[1]), None
    c51 = GLM53v51(Xu.shape[1])
    Y51 = np.empty_like(Xu)
    for i in range(len(tu)):
        Y51[i] = c51.process(tu[i], Xu[i])
    dt51 = time.time() - t0
    ep51 = sorted(getattr(c51, "epoch_t", []))
    base_epochs[name] = dict(v6=r6["epoch"], v51=ep51,
                             v6_revoke=r6["revoke"], v6_handoff=r6["handoff"],
                             v6_unload=r6["unload"], v6_gevent=r6["gevent"])
    rows[-1].update(v6_epoch=len(r6["epoch"]), v51_epoch=len(ep51),
                    v6_revoke=len(r6["revoke"]), v6_down=len(r6["gevent"]),
                    v6_unload=len(r6["unload"]), v6_handoff=len(r6["handoff"]),
                    v51_s=round(dt51, 1))
    print(f"[{name}] n={len(t)} span={d['span']:.1f}s fs={len(t)/d['span']:.1f}Hz "
          f"totmax={tot.max():.0f} v6_epoch={len(r6['epoch'])} v51_epoch={len(ep51)} "
          f"revoke={len(r6['revoke'])} down={len(r6['gevent'])} unload={len(r6['unload'])}",
          flush=True)
    for e in r6["epoch"]:
        print("    v6 epoch t0=%.3f kind=%-14s t_det=%.3f base=%.1f v0=%.0f y0=%.0f" % e, flush=True)
    for e in ep51:
        print("    v51 epoch %.3f" % e, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "a0_baseline_stats.csv"), index=False, encoding="utf-8-sig")
np.save(os.path.join(OUT, "a0_baseline_epochs.npy"), np.array([base_epochs], dtype=object),
        allow_pickle=True)
print(df.to_string())
