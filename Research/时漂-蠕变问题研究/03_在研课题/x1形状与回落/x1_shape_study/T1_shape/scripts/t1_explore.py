# -*- coding: utf-8 -*-
"""步骤 0：数据体检 —— 采样率、电平范围、台阶检测、快相原始爬升量级。

输出写到 results/t1_explore.txt（控制台编码不可靠）。
用法： python t1_explore.py
"""
import os
import sys

import numpy as np

import t1_lib as T


def main():
    out = []
    w = out.append
    for label, d in T.discover_sessions(T.WORKING, 5):
        pre, rec = T.load_session(d)
        t, el, V = pre["t"], pre["el"], pre["V"]
        tot = T.total(pre)
        steps, sm = T.detect_steps(t, tot)
        w("=" * 108)
        w(label)
        w("  帧 %d  时长 %.1f s  平均帧间隔 %.4f s (%.1f Hz)  重复 elapsed %d/%d  "
          "缺口(>0.3s) %d 个"
          % (len(el), el[-1], pre["dt_mean"], 1.0 / pre["dt_mean"],
             int(np.sum(np.diff(el) <= 0)), len(el), len(pre["gap_idx"])))
        w("  总量 %.0f ~ %.0f (范围 %.0f)" % (tot.min(), tot.max(),
                                              tot.max() - tot.min()))
        chs = T.top_channels(pre, 5)
        amp = V.max(axis=0) - np.percentile(V, 5, axis=0)
        w("  通道幅度 top: " + "  ".join("ch%d=%.0f" % (c, amp[c]) for c in chs))
        ups = [s for s in steps if s["sign"] > 0]
        dns = [s for s in steps if s["sign"] < 0]
        w("  台阶：加载 %d，卸载 %d" % (len(ups), len(dns)))
        for s in steps:
            w("    %s t=%7.2f 沿宽 %.2f s  Δ=%+8.0f  (%.0f→%.0f)  斜率 %.0f ADC/s"
              % ("UP" if s["sign"] > 0 else "DN", s["t1"], s["dur"], s["d"],
                 s["level_pre"], s["level_post"], s["slope"]))
        if ups:
            c = chs[0]
            y = V[:, c]
            w("  快相原始爬升（ch%d，相对落点，ADC）:" % c)
            hdr = "    %-24s" % "台阶"
            for tt in (1, 2, 4, 8, 16, 30):
                hdr += "%8s" % ("+%ds" % tt)
            w(hdr)
            for s in ups[:20]:
                tc = s["t1"]
                land, _ = T.landing_value(t, y, tc, s["dur"])
                row = "    t=%7.2f Δ=%+7.0f " % (tc, s["level_post"] - s["level_pre"])
                for tt in (1, 2, 4, 8, 16, 30):
                    k = min(int(np.searchsorted(t, tc + tt)), len(t) - 1)
                    row += "%8.0f" % (y[k] - land)
                w(row)

    txt = "\n".join(out)
    os.makedirs(T.RESULTS, exist_ok=True)
    with open(os.path.join(T.RESULTS, "t1_explore.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    sys.stdout.write("written results/t1_explore.txt (%d lines)\n" % len(out))


if __name__ == "__main__":
    main()
