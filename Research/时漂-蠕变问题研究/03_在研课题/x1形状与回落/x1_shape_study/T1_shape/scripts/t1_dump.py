# -*- coding: utf-8 -*-
"""步骤 2b：把若干代表事件的实测快相曲线按 2 s 桶打印出来（文字版曲线），用于目视判断形态。
用法： python t1_dump.py
输出： results/t1_curve_dump.txt
"""
import os
import sys

import numpy as np

import t1_lib as T

CASES = [
    ("working", "9c3ca5", 3.08, None),
    ("working", "9c3ca5", 36.74, None),
    ("working", "9c3ca5", 90.49, None),
    ("working", "73032d", 63.36, None),
    ("working", "3f32c5", 17.03, None),
    ("archived", "0cb8b6", 121.36, None),
    ("archived", "7b3977", 16.53, None),
    ("archived", "持续恒定负载", 4.23, None),
]


def find(label_key, root, t_target):
    for label, d in T.discover_sessions(root, 5):
        if label_key in label:
            pre, _ = T.load_session(d)
            return label, pre
    return None, None


def main():
    out = []
    w = out.append
    for tag, key, t_target, _ in CASES:
        root = T.WORKING if tag == "working" else T.ARCHIVED
        label, pre = find(key, root, t_target)
        if pre is None:
            w("!! 未找到 %s / %s" % (tag, key))
            continue
        t, V = pre["t"], pre["V"]
        tot = T.total(pre)
        steps, _ = T.detect_steps(t, tot)
        ups = [s for s in steps if s["sign"] > 0]
        s = min(ups, key=lambda q: abs(q["t1"] - t_target))
        if abs(s["t1"] - t_target) > 1.5:
            w("!! %s 未匹配到 t=%.2f 的加载沿（最近 %.2f）" % (key, t_target, s["t1"]))
            continue
        nxt = [q["t0"] for q in ups if q["t0"] > s["t1"]]
        plateau = (nxt[0] if nxt else t[-1]) - s["t1"]
        rng = V.max(axis=0) - V.min(axis=0)
        b = int(np.searchsorted(t, s["t1"]))
        resp = []
        for c in range(T.NCH):
            if rng[c] < 120:
                continue
            lv_pre = float(np.median(V[max(0, b - 70):b - 10, c]))
            lv_post = float(np.median(V[b:b + 60, c]))
            if lv_post - lv_pre >= 25:
                resp.append((int(c), lv_post - lv_pre))
        resp.sort(key=lambda kv: -kv[1])
        resp = resp[:4]
        w("=" * 112)
        w("%s  沿 t=%.2f 沿宽 %.2f s 总量Δ=%+.0f 保压 %.1f s  响应通道 %s"
          % (label, s["t1"], s["dur"], s["d"], plateau,
             " ".join("ch%d(Δ%+.0f)" % kv for kv in resp)))
        w("   %-28s" % "时间(s) 相对沿末" + "".join("%14s" % ("ch%d 蠕变/归一" % c)
                                                    for c, _ in resp))
        for tt in (0.5, 1, 2, 3, 4, 6, 8, 10, 14, 20, 25, 30, 40, 60, 90, 120):
            if tt > plateau or tt > 120:
                break
            k = int(np.searchsorted(t, s["t1"] + tt))
            k = min(k, len(t) - 1)
            seg = slice(int(np.searchsorted(t, s["t1"] + tt)), k + 1)
            row = "   t=%+8.1f s           " % tt
            for c, _ in resp:
                land = float(np.median(V[b:b + 50, c]))
                creep = V[k, c] - land
                row += "%14s" % ("%+.0f" % creep)
            w(row)
        # 归一化（用 +30 s 或窗口末的值）
        w("   归一化 c(t)/c(T)  T=%.0f s:" % min(30.0, plateau))
        Tmax = min(30.0, plateau)
        for c, _ in resp:
            land = float(np.median(V[b:b + 50, c]))
            kT = int(np.searchsorted(t, s["t1"] + Tmax))
            cT = float(np.median(V[kT - 30:kT + 1, c]) - land)
            vals = []
            for tt in (1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30):
                if tt > Tmax:
                    break
                k = min(int(np.searchsorted(t, s["t1"] + tt)), len(t) - 1)
                vals.append("%.2f" % ((V[k, c] - land) / cT if cT else 0))
            w("     ch%-3d c30=%+7.1f  归一 %s" % (c, cT, " ".join(vals)))
    txt = "\n".join(out)
    os.makedirs(T.RESULTS, exist_ok=True)
    with open(os.path.join(T.RESULTS, "t1_curve_dump.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    sys.stdout.write("written results/t1_curve_dump.txt\n")


if __name__ == "__main__":
    main()
