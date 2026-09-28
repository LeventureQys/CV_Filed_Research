# -*- coding: utf-8 -*-
"""v3.6 探针 1：目标录制总览 + 逐加载段「台阶 / 补偿量」一致性主表。

回答两件事：
  A. 用户报的「245 s 之后丢基线」在该录制上是否成立、范围多大；
  B. 「前面有补偿的段」与「后面没有补偿的段」之间的差距到底多大（一致性口径）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(OUT, exist_ok=True)


def main():
    ds = K.load(K.DS_TARGET)
    el = ds["pre"]["el"]
    tin, tout, ded = ds["tot_in"], ds["tot_out"], ds["ded"]
    n = ds["pre"]["n"]
    dt = np.diff(el)
    L = []
    L.append("== 目标录制总览 ==")
    L.append("目录: %s" % K.DS_TARGET)
    L.append("帧数: %d  时长: %.1f s  通道: %d" % (n, el[-1] - el[0], ds["pre"]["V"].shape[1]))
    L.append("dt>0 占比: %.1f%%   中位帧间隔: %.4f s" % (100.0 * np.mean(dt > 0), np.median(dt)))
    L.append("输入总量   min/med/max = %.0f / %.0f / %.0f" % (tin.min(), np.median(tin), tin.max()))
    L.append("显示总量   min/med/max = %.0f / %.0f / %.0f" % (tout.min(), np.median(tout), tout.max()))
    L.append("补偿 ded   min/med/max = %.0f / %.0f / %.0f" % (ded.min(), np.median(ded), ded.max()))
    L.append("")

    segs = K.load_segments(tin, el)
    L.append("== 加载/空载段切分（输入总量滞回 30%%/70%%）==")
    for k, i0, i1 in segs:
        L.append("  %-6s [%7.2f, %7.2f] s  (%5d 帧)  med=%.0f"
                 % (k, el[i0], el[i1], i1 - i0 + 1, np.median(tin[i0:i1 + 1])))
    L.append("")

    rows = K.seg_table(ds, segs, w0=10.0, w1=20.0)
    L.append("== 逐加载段（窗口 = 段起点后 [10,20] s）==")
    L.append(K.fmt_table(rows))
    L.append("")

    tag = np.array([("前(<=200s)" if r["t_on"] <= 200.0 else "后(>200s)") for r in rows])
    if np.any(tag == "前(<=200s)"):
        a = [r["ded_med"] for r in rows if r["t_on"] <= 200.0]
        b = [r["ded_med"] for r in rows if r["t_on"] > 200.0]
        L.append("== 一致性摘要 ==")
        L.append("前段(t<=200s)  n=%d  ded 中位 %.0f  极差 %.0f" % (len(a), np.median(a), max(a) - min(a)))
        if b:
            L.append("后段(t> 200s) n=%d  ded 中位 %.0f  极差 %.0f" % (len(b), np.median(b), max(b) - min(b)))
            L.append("前后差距（中位之差）: %.0f ADC" % (np.median(a) - np.median(b)))
    L.append("")

    # 显示电平 vs 输入电平（偏差一致性）
    L.append("== 段内「显示−输入」偏差（窗口 [10,20] s）==")
    for r in rows:
        L.append("  t=%7.2f s  step=%+7.0f  ded=%+7.0f  ded/step=%+6.3f  显示/输入=%.4f"
                 % (r["t_on"], r["step"], r["ded_med"], r["ded_frac"],
                    (r["lvl"] - r["ded_med"]) / r["lvl"] if abs(r["lvl"]) > 1e-9 else float("nan")))

    txt = "\n".join(L)
    p = os.path.join(OUT, "v36_segments.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)
    print("\n-> %s" % p)


if __name__ == "__main__":
    main()
