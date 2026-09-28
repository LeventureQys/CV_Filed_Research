# -*- coding: utf-8 -*-
"""T3 诊断：核对评估侧沿切分结果（每个会话的沿表 + 保压段长度 + 卸载标记）。"""
import os
import sys

import numpy as np

import t3_lib as T


def main():
    want = None
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            want = a
    lines = []
    P = lines.append
    for tag, label, d in T.all_sessions():
        if want and want not in label:
            continue
        s = T.load_input(d)
        el, din = s["el"], s["V"].sum(axis=1)
        rng = float(din.max() - din.min())
        ed, ms = T.find_edges(el, din)
        P("== %s   量程 %.0f  min_step %.0f  沿 %d 个" % (label, rng, ms, len(ed)))
        for e in ed:
            hold = el[e["j"]] - el[e["i"]]
            P("   t=%7.2f  台阶 %7.0f  基线 %7.0f  保压 %6.1fs  卸载%s  帧 %d"
              % (e["t"], e["step"], e["base"], hold,
                 "有" if e["unload"] else "无(到段末)", e["j"] - e["i"]))
        if "--raw" in sys.argv:
            nb = 100
            prev = np.array([np.median(din[max(0, i - nb):max(1, i - 2)])
                             if i > 5 else din[i] for i in range(len(din))])
            rise = din - prev
            P("   原始候选（rise 超门限且距上一次 ≥2s）：")
            last = -1e9
            for i in range(len(el)):
                if rise[i] > ms and el[i] - last > 2.0:
                    P("     t=%7.2f rise=%8.0f 电平=%8.0f" % (el[i], rise[i], din[i]))
                    last = el[i]
        # 独立对照：简单阈值上升沿计数（不做保压/卸载切分）
        rise = din - np.concatenate([[din[0]] * 100, din[:-100]])
        n_simple = 0
        last = -1e9
        for i in range(len(el)):
            if rise[i] > ms and el[i] - last > 2.0:
                n_simple += 1
                last = el[i]
        P("   （简单对照：阈值上升沿计数 %d）" % n_simple)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "t3_edge_check.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
