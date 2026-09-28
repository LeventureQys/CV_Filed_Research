# -*- coding: utf-8 -*-
"""T2 步骤11：时间轴核实——elapsed 重复帧比例，以及对回放结论的影响。

T1 报告已指出录制 elapsed 是批量到达时间戳（70% 帧与前帧重复）。本脚本量化：
  1. 每个会话的重复帧占比、实际 dt>0 帧率、elapsed 量化步长；
  2. 用「帧序号 / 平均帧率」重建均匀时间轴后重跑基线，比较关键指标是否改变；
  3. 因为两套实现都用同一段 dt 积分，总积分时间一致 ⇒ 结论不应改变（本脚本验证）。
"""
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

OUT = os.path.join(T.OUT_ROOT, "results")


def main():
    lines = []
    lines.append("%-46s %8s %9s %9s %9s %10s"
                 % ("会话", "帧数", "重复帧占比", "dt>0占比", "平均帧率", "量化步长s"))
    pack = []
    for label, d in T.all_sessions():
        s = T.load_pre(d)
        el = s["el"]
        dz = np.diff(el)
        pos = dz[dz > 0]
        step = float(np.median(pos)) if len(pos) else float("nan")
        fps = (len(el) - 1) / el[-1] if el[-1] > 0 else float("nan")
        lines.append("%-46s %8d %9.3f %9.3f %9.2f %10.5f"
                     % (label[-46:], len(el), float(np.mean(dz == 0)),
                        float(np.mean(dz > 0)), fps, step))
        pack.append((label, s, el, fps))
    lines.append("")
    lines.append("均匀时间轴重建后的对照（基线，回落/保证段偏差）：")
    lines.append("%-46s %14s %14s %14s %14s"
                 % ("会话", "原生轴 回落", "均匀轴 回落", "原生轴 保证段", "均匀轴 保证段"))
    for label, s, el, fps in pack:
        tot = T.total(s)
        zz = np.zeros(len(el))
        evs = T.load_events(el, tot, win=60.0)
        if not evs:
            continue
        keep = []
        for e in evs:
            m = T.event_metrics(zz, tot, el, e)
            if m and T.event_gate(m, e):
                keep.append(e)
        if not keep:
            continue
        ts_uni = np.arange(len(el)) / fps
        row = []
        for axis in (el, ts_uni):
            r = O.observe(axis.copy(), s["V"], "base")
            ylp = T.lowpass(r["D"], axis, 1.0)
            xlp = T.lowpass(tot, axis, 5.0)
            ds, es = [], []
            for e in keep:
                m = T.event_metrics(r["D"], tot, axis, e, xlp=xlp, ylp=ylp,
                                    x1=r["X1"])
                if m:
                    ds.append(m["drop"])
                    es.append(m["seg_err"])
            row.append((float(np.mean(ds)) if ds else float("nan"),
                        float(np.mean(es)) if es else float("nan")))
        lines.append("%-46s %14.1f %14.1f %14.1f %14.1f"
                     % (label[-46:], row[0][0], row[1][0], row[0][1], row[1][1]))
    lines.append("")
    lines.append("说明：两套时间轴下状态积分总量一致（dt 全为 0 的帧不推进状态），")
    lines.append("      故绝对指标差异应在噪声量级；如出现系统性偏移，说明"
                 "elapsed 量化掩盖了真实的 100 Hz 均匀帧率。")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "t2_timeaxis.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
