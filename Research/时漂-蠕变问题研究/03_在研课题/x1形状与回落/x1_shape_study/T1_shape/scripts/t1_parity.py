# -*- coding: utf-8 -*-
"""步骤 3b：回放显示 vs 录制显示的逐秒对照（诊断部分会话 parity 不达标的原因）。
输出 results/t1_parity.txt
"""
import os
import sys

import numpy as np

import t1_lib as T

CASES = ["9c3ca5", "3f32c5", "f40a1b", "持续恒定负载", "0cb8b6", "71208f"]


def main():
    out = []
    w = out.append
    for tag, root in (("working", T.WORKING), ("archived", T.ARCHIVED)):
        for label, d in T.discover_sessions(root, 5):
            if not any(k in label for k in CASES):
                continue
            pre, rec = T.load_session(d)
            if rec is None or rec["n"] != pre["n"]:
                w("%s : 无录制显示或帧数不符" % label)
                continue
            t, V = pre["t"], pre["V"]
            r = T.replay(pre["ts"], V, {})
            D = r["D"].sum(axis=1)
            Drec = rec["V"].sum(axis=1)
            diff = D - Drec
            # 逐帧差可能与 seg 流有 1 帧偏移
            for lag in (-2, -1, 0, 1, 2):
                a = D[max(0, lag):len(D) + min(0, lag)]
                b = Drec[max(0, -lag):len(Drec) - max(0, lag)]
                n = min(len(a), len(b))
                pass
            w("=" * 108)
            w("%s" % label)
            w("  回放-录制  均值 %+.1f  标准差 %.1f  max|Δ| %.1f (逐通道)"
              % (float(diff.mean()), float(diff.std()),
                 float(np.max(np.abs((V - r["X1"] - r["X2"]) - rec["V"])))))
            tt, md = T.band_med(t, diff, 10.0)
            w("  差值 10 s 桶: " + " ".join("%+.0f" % v for v in md[:24]))
            tt2, mrec = T.band_med(t, Drec, 10.0)
            tt3, mrep = T.band_med(t, D, 10.0)
            w("  录制显示 10 s 桶: " + " ".join("%.0f" % v for v in mrec[:24]))
            w("  回放显示 10 s 桶: " + " ".join("%.0f" % v for v in mrep[:24]))
            w("  输入     10 s 桶: "
              + " ".join("%.0f" % v for v in T.band_med(t, T.total(pre), 10.0)[1][:24]))
    txt = "\n".join(out)
    os.makedirs(T.RESULTS, exist_ok=True)
    with open(os.path.join(T.RESULTS, "t1_parity.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    sys.stdout.write("written results/t1_parity.txt\n")


if __name__ == "__main__":
    main()
