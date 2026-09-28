# -*- coding: utf-8 -*-
"""T3 诊断：沿检测器触发覆盖情况。

对每个会话跑一次前馈臂（默认 A_now_a1.0），统计：
  - 触发总次数、触发时刻分布；
  - 每个评估沿是否有触发（±1.5 s）、沿后首次触发延迟、触发通道数；
  - 沿后 1.5 s 内逐通道 slope 峰值（前 5 大通道）与同刻 e_pre、门限，用于解释为何触发/未触发。
"""
import sys

import numpy as np

import t3_lib as T

ARMS = {
    "A_now_a1.0": dict(mode="A", pred="now", alpha=1.0),
    "A_lag_a1.0": dict(mode="A", pred="lag", alpha=1.0),
}
KW = [a for a in sys.argv[1:] if not a.startswith("--")]


def main():
    lines = []
    P = lines.append
    for arm, spec in ARMS.items():
        ff = dict(T.DET)
        ff.update(spec)
        P("================ 臂 %s（门限 thr=%.0f + %.2f·0.02·max(e,1)）"
          % (arm, ff["thr"], ff["rel_on"]))
        for tag, label, d in T.all_sessions():
            if KW and not any(k in label for k in KW):
                continue
            s = T.load_input(d)
            el, ts, V = s["el"], s["ts"], s["V"]
            din = V.sum(axis=1)
            edges, ms = T.find_edges(el, din)
            r = T.observe(ts, V, ff=ff)
            # 统一为 (通道, 时间, slope, e_pre, 置位前 x1)
            trig = [(c, float(el[i]), sl, ep, xb) for c, i, sl, ep, xb in r["trig"]]
            P("== %s  沿 %d 触发 %d" % (label, len(edges), len(trig)))
            for e in edges:
                t0 = e["t"]
                near = [g for g in trig if -0.2 <= g[1] - t0 <= 2.0]
                lat = (min(g[1] for g in near) - t0) if near else None
                P("   t=%7.2f 台阶%7.0f | 触发通道 %2d 首次延迟 %s | 触发通道 e_pre 中位 %s"
                  % (t0, e["step"], len(near),
                     ("%.2f s" % lat) if lat is not None else "  --  ",
                     ("%.0f" % np.median([g[3] for g in near])) if near else "  --  "))
            if trig:
                tt = np.array([g[1] for g in trig])
                on_edge = sum(1 for e in edges
                              if any(-0.2 <= g[1] - e["t"] <= 2.0 for g in trig))
                P("   触发时刻（前 20）：%s"
                  % " ".join("%.1f" % x for x in tt[:20]))
                P("   有触发的评估沿 %d/%d；触发时间跨度 %.0f~%.0f s（共 %d 次触发，每沿平均 %.1f 通道）"
                  % (on_edge, len(edges), tt.min(), tt.max(), len(tt),
                     len(tt) / max(len(edges), 1)))
            e_pre_all = [g[3] for g in trig]
            if e_pre_all:
                P("   触发时 e_pre 分位：p10 %.0f p50 %.0f p90 %.0f （x1 置位增量 = 0.12·e_pre：p50 %.0f）"
                  % (np.percentile(e_pre_all, 10), np.percentile(e_pre_all, 50),
                     np.percentile(e_pre_all, 90), 0.12 * np.percentile(e_pre_all, 50)))
    out = r"D:\workshop\Processing\multi-device-cascade-host-cpp\Document\Update\Dev-Version\v2.7 - 抗蠕变补偿算法\v4.1flash\plan\v4\x1_shape_study\T3_feedforward\results\t3_trigger_cover.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
