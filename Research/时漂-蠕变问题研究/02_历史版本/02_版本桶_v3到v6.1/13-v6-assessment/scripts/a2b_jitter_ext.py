# -*- coding: utf-8 -*-
"""a2b 抖动加固扫描：加大幅度 / 换带宽 / 同相注入，找「误触发」与「漏触发」的分界。

- 同时复刻检测器（a1_detector.detector_trace）在扰动下的 σ_d 与门限，说明丢事件机理；
- 对每档记录 epoch 数（漏/误）、ΣA 偏差、稳态显示偏差、以及真实阶跃的检出时刻漂移。
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import REC, load_uniform, make_perturb, run_traced, TRACED_V6, ROOT  # noqa: E402
from a1_detector import detector_trace                                          # noqa: E402

RES = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
os.makedirs(RES, exist_ok=True)

# (标签, kind, kwargs)
CONF = [
    ("独立白噪", "white", dict(f_hi=40.0)),
    ("带限0.3-5Hz", "band", dict(f_lo=0.3, f_hi=5.0)),
    ("带限0.3-40Hz", "band", dict(f_lo=0.3, f_hi=40.0)),
    ("同相白噪", "common", dict(f_hi=40.0)),
    ("同相1-5Hz", "common_band", dict(f_lo=1.0, f_hi=5.0)),
]
AMPS = [200, 1000, 2000, 4000]
RECS = ["零负载-切换负载-零负载-再切换负载", "中途切换-最终测试目标"]


def perturb(X, A, kind, rng, kw):
    if kind == "common_band":
        n = X.shape[0]
        from a_common import _bandlimited
        s = _bandlimited(n, 100.0, kw["f_lo"], kw["f_hi"], rng)
        s = s / s.std()
        base = np.abs(X).mean(axis=0)
        w = base / base.sum()
        E = np.outer(s, w)
    else:
        E = make_perturb(X, A, kind, rng, **kw)
    t = E.sum(axis=1)
    s = t.std()
    return E * (A / s) if s > 1e-12 else E


def main():
    rows = []
    for name in RECS:
        d = load_uniform(REC[name])
        tu, X = d["tu"], d["Xu"]
        tot0 = X.sum(axis=1)
        lvl = np.median(tot0[tot0 > 0.5 * tot0.max()])
        max_tot = tot0.max()
        b6 = run_traced(TRACED_V6, tu, X)
        totY0 = b6["Y"].sum(axis=1)
        A0 = np.sum(b6["comp"].A)
        ep0 = {round(e[0], 1) for e in b6["epoch"]}
        n_ep0 = len(b6["epoch"])
        tr0 = detector_trace(tu, X)
        print(f"== [{name}] 电平≈{lvl:.0f} max={max_tot:.0f} 空载门={0.10*max_tot:.0f} "
              f"受载门=5%lvref≈{0.05*lvl:.0f} | 基线 epoch={n_ep0} ΣA={A0:.0f} "
              f"σ_d(中位)={np.median(tr0['sig']):.1f}", flush=True)
        rows.append(dict(rec=name, kind="baseline", amp=0, n_epoch=n_ep0, n_miss=0, n_extra=0,
                         sig_med=round(float(np.median(tr0["sig"])), 1),
                         thr_med=round(float(np.median(tr0["thr"])), 1),
                         maxdev=0.0, slow_dev=0.0, dA=0.0,
                         miss_t="", extra_t=""))
        for lbl, kind, kw in CONF:
            for amp in AMPS:
                t0 = time.time()
                rng = np.random.default_rng(7)
                Xp = X + perturb(X, amp, kind, rng, kw)
                r6 = run_traced(TRACED_V6, tu, Xp)
                ep = {round(e[0], 1) for e in r6["epoch"]}
                # 与基线"配对"（±0.8 s 内视为同一变载）
                miss = [e for e in ep0 if not any(abs(e - x) <= 0.8 for x in ep)]
                extra = [e for e in ep if not any(abs(e - x) <= 0.8 for x in ep0)]
                dev = r6["Y"].sum(axis=1) - totY0
                slow = (r6["st"] == 2) & (b6["st"] == 2)
                tr = detector_trace(tu, Xp)
                rows.append(dict(
                    rec=name, kind=lbl, amp=amp, n_epoch=len(ep),
                    n_miss=len(miss), n_extra=len(extra),
                    sig_med=round(float(np.median(tr["sig"])), 1),
                    thr_med=round(float(np.median(tr["thr"])), 1),
                    maxdev=round(float(np.abs(dev).max()), 1),
                    slow_dev=round(float(np.median(dev[slow])), 1) if slow.sum() > 50 else "",
                    dA=round(float(np.sum(r6["comp"].A) - A0), 1),
                    miss_t=";".join(f"{x:.1f}" for x in sorted(miss)),
                    extra_t=";".join(f"{x:.1f}" for x in sorted(extra))))
                print(f"  {lbl:12s} A={amp:5d} epoch={len(ep):3d}(漏{len(miss)}/误{len(extra)}) "
                      f"σ_d={np.median(tr['sig']):9.1f} 门限={np.median(tr['thr']):7.0f} "
                      f"maxdev={np.abs(dev).max():7.0f} slow={np.median(dev[slow]) if slow.sum()>50 else 0:8.0f} "
                      f"dA={np.sum(r6['comp'].A)-A0:8.0f} [{time.time()-t0:.1f}s]", flush=True)
                if len(extra):
                    print(f"     误触发 t0: {sorted(extra)}", flush=True)
                if len(miss):
                    print(f"     漏掉 t0: {sorted(miss)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a2b_jitter_ext.csv"), index=False, encoding="utf-8-sig")
    print(df.groupby(["kind", "amp"])[["n_miss", "n_extra", "sig_med", "slow_dev"]].mean().to_string())


if __name__ == "__main__":
    main()
