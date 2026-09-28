# -*- coding: utf-8 -*-
"""T3 诊断：打印指定会话/沿附近的输入、基线显示、前馈显示、x1 轨迹（0.5 s 采样）。

用法：python t3_dbg_edge_curves.py <会话关键字> [沿时刻] [窗口秒]
"""
import sys

import numpy as np

import t3_lib as T

KW = sys.argv[1] if len(sys.argv) > 1 else "7b3977"
T0 = float(sys.argv[2]) if len(sys.argv) > 2 else None
WIN = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

ARMS = [("base", None),
        ("A_now_a1.0", dict(mode="A", pred="now", alpha=1.0)),
        ("A_lag_a1.0", dict(mode="A", pred="lag", alpha=1.0)),
        ("C_a0.8_tb2.0", dict(mode="C", pred="now", alpha=0.8,
                              tau_boost=2.0, boost_s=1.5))]


def main():
    for tag, label, d in T.all_sessions():
        if KW not in label:
            continue
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        din = V.sum(axis=1)
        rec = T.load_recorded(d)
        drec = rec["V"].sum(axis=1) if rec else None
        edges, ms = T.find_edges(el, din)
        print("==", label, " 沿数", len(edges))
        for e in edges:
            print("   沿 t=%7.2f 台阶 %7.0f 基线 %7.0f 保压 %5.1fs 段末 %7.2f"
                  % (e["t"], e["step"], e["base"], el[e["j"]] - el[e["i"]], el[e["j"]]))
        cand = edges
        if T0 is not None:
            cand = [e for e in edges if abs(e["t"] - T0) < 1.0]
        for e in cand[:3]:
            i = e["i"]
            t0 = el[i]
            m = (el >= t0 - 1.0) & (el <= t0 + WIN)
            ee = el[m]
            print("\n   --- 沿 t=%.2f  台阶=%.0f  （窗口 %.0f s）---" % (t0, e["step"], WIN))
            head = "      t   |    输入 |   录制显示"
            for name, spec in ARMS:
                head += " | %12s" % name
            print(head)
            res = {}
            x1s = {}
            x2s = {}
            for name, spec in ARMS:
                ff = None
                if spec:
                    ff = dict(T.DET)
                    ff.update(spec)
                r = T.observe(ts, V, ff=ff)
                res[name] = r["D"].sum(axis=1)
                x1s[name] = r["X1"].sum(axis=1)
                x2s[name] = r["X2"].sum(axis=1)
                if spec and r["trig"]:
                    near = [g for g in r["trig"] if abs(el[g[1]] - t0) < 1.5]
                    print("   [%s] 沿附近触发 %d 次: %s"
                          % (name, len(near),
                             " ".join("ch%d@%.2f sl=%.0f e=%.0f x1旧=%.0f"
                                      % (c, el[i], sl, ep, xb)
                                      for c, i, sl, ep, xb in near[:8])))
            tgrid = np.arange(ee[0], ee[-1] + 1e-6, 0.5)
            for tt in tgrid:
                k = int(np.argmin(np.abs(el - tt)))
                row = "   %7.2f | %7.0f | " % (tt, din[k])
                row += ("%9.0f" % drec[k]) if drec is not None else "        -"
                for name, spec in ARMS:
                    row += " | %12.0f" % res[name][k]
                print(row)
            print("   --- 同刻 x1/x2 分解（base / A_lag / C_a0.8）---")
            for tt in np.arange(ee[0], ee[-1] + 1e-6, 2.0):
                k = int(np.argmin(np.abs(el - tt)))
                print("   %7.2f | x1 %6.0f/%6.0f/%6.0f | x2 %6.0f/%6.0f/%6.0f"
                      % (tt, x1s["base"][k], x1s["A_lag_a1.0"][k], x1s["C_a0.8_tb2.0"][k],
                         x2s["base"][k], x2s["A_lag_a1.0"][k], x2s["C_a0.8_tb2.0"][k]))


if __name__ == "__main__":
    main()
