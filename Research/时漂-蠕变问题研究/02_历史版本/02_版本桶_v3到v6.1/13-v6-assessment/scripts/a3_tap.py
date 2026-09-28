# -*- coding: utf-8 -*-
"""a3 拍击类瞬态：是否被当成一次真实加载？显示冲多高？多久回到真值？

形态：上升沿 rise(30/50/80 ms) + 冲高 amp(500~6000 ADC) + 保持 hold(50/100/200 ms) + 回落。
对照：同幅度、同上升沿的**真实 restep**（不回落）——同一流水线比较可辨识特征。
所有指标相对「无拍击」的基线运行（Y0），以排除算法自身的既有偏差。
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import (REC, load_uniform, add_tap, add_step, run_traced,   # noqa: E402
                      TRACED_V6, ROOT)

RES = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
os.makedirs(RES, exist_ok=True)

PLAT = {
    "零负载-切换负载-零负载-再切换负载": [
        (12.5, 30.5, "负载平台2610"),
        (36.5, 50.0, "负载平台2742"),
        (57.5, 70.3, "负载平台1891"),
    ],
    "中途切换-最终测试目标": [
        (45.5, 60.7, "负载平台3354"),
    ],
}
TAPS = [(30, 50), (50, 100), (80, 200)]
AMPS = [500, 1000, 2000, 4000, 6000]


def metrics(tu, r6, Y0, i0, dur_s, n_ep0, n_ho0, ep0_t):
    disp = r6["Y"].sum(axis=1)
    disp0 = Y0.sum(axis=1)
    n = len(tu)
    j1 = min(n, i0 + int(dur_s / 0.01))
    win = slice(i0, j1)
    dev = disp[win] - disp0[win]
    lvl = np.median(disp0[max(0, i0 - 300):i0])
    tol = 0.05 * max(lvl, 1.0)
    over = np.abs(dev) > tol
    back = np.nan
    if over.any():
        last = np.where(over)[0][-1]
        rest = np.where(~over[last:])[0]
        if len(rest):
            back = round(float(tu[i0 + last + rest[0]] - tu[i0]), 2)
    ep = r6["epoch"]
    ep_new = [e for e in ep if any(abs(e[0] - x) <= 0.8 for x in ep0_t) is False]
    ep_lost = [x for x in ep0_t if not any(abs(e[0] - x) <= 0.8 for e in ep)]
    return dict(
        n_ep0=n_ep0, n_ep=len(ep), n_ep_new=len(ep_new), n_ep_lost=len(ep_lost),
        new_t=";".join(f"{e[0]:.2f}:{e[1]}" for e in ep_new),
        max_over=round(float(dev.max()), 1), min_over=round(float(dev.min()), 1),
        max_abs=round(float(np.abs(dev).max()), 1),
        back_s=(None if back != back else back),
        n_revoke=len(r6["revoke"]), d_ho=len(r6["handoff"]) - n_ho0,
        dA_end=round(float(np.sum(r6["comp"].A)), 1),
        state_end=r6["st"][-1])


def main():
    rows = []
    for name, plats in PLAT.items():
        d = load_uniform(REC[name])
        tu, X = d["tu"], d["Xu"]
        b6 = run_traced(TRACED_V6, tu, X)
        Y0 = b6["Y"]
        n_ep0 = len(b6["epoch"])
        n_ho0 = len(b6["handoff"])
        ep0_t = [e[0] for e in b6["epoch"]]
        A0 = float(np.sum(b6["comp"].A))
        print(f"== [{name}] 基线 epoch={n_ep0} ΣA={A0:.0f}", flush=True)
        for (a, b, tag) in plats:
            i0 = int(round(((a + b) / 2.0) / 0.01))
            for amp in AMPS:
                for (rise, hold) in TAPS:
                    Xp, _ = add_tap(X, i0, amp, rise_ms=rise, hold_ms=hold)
                    r6 = run_traced(TRACED_V6, tu, Xp)
                    m = metrics(tu, r6, Y0, i0, (2 * rise + hold) / 1000.0 + 5.0,
                                n_ep0, n_ho0, ep0_t)
                    m.update(rec=name, tag=tag, shape=f"拍击{rise}/{hold}", amp=amp,
                             t_inj=round(float(tu[i0]), 2), A0=A0,
                             dA=round(float(np.sum(r6["comp"].A)) - A0, 1))
                    rows.append(m)
                    print(f"  {tag} 拍击{rise:02d}/{hold:03d} amp={amp:5d}: "
                          f"epoch {m['n_ep']}(新{m['n_ep_new']}/丢{m['n_ep_lost']}) "
                          f"dev=[{m['min_over']:7.0f},{m['max_over']:7.0f}] back={m['back_s']} "
                          f"revoke={m['n_revoke']} dHO={m['d_ho']} dA={m['dA']:8.0f} "
                          f"{m['new_t']}", flush=True)
            for amp in AMPS:
                Xp = add_step(X, i0, amp, rise_ms=50)
                r6 = run_traced(TRACED_V6, tu, Xp)
                m = metrics(tu, r6, Y0, i0, 10.0, n_ep0, n_ho0, ep0_t)
                m.update(rec=name, tag=tag, shape="真实restep50", amp=amp,
                         t_inj=round(float(tu[i0]), 2), A0=A0,
                         dA=round(float(np.sum(r6["comp"].A)) - A0, 1))
                rows.append(m)
                print(f"  {tag} 真实restep      amp={amp:5d}: "
                      f"epoch {m['n_ep']}(新{m['n_ep_new']}/丢{m['n_ep_lost']}) "
                      f"dev=[{m['min_over']:7.0f},{m['max_over']:7.0f}] back={m['back_s']} "
                      f"revoke={m['n_revoke']} dHO={m['d_ho']} dA={m['dA']:8.0f} {m['new_t']}",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a3_tap_transient.csv"), index=False, encoding="utf-8-sig")
    cols = ["rec", "tag", "shape", "amp", "n_ep_new", "n_ep_lost", "max_abs",
            "back_s", "n_revoke", "d_ho", "dA", "new_t"]
    print(df[cols].to_string())
    # 汇总：拍击 vs 阶跃
    print("\n-- 汇总（按形态×幅度，跨平台平均）--")
    print(df.groupby(["shape", "amp"])[["n_ep_new", "n_revoke", "d_ho", "max_abs"]]
          .mean().to_string())


if __name__ == "__main__":
    main()
