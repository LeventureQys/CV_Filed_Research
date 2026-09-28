# -*- coding: utf-8 -*-
"""探针 G：对 A/B 里两个「方向不一致」的数据集用**专用口径**复核。

① 振荡工况 f40a1b：按 v2.0 的口径量「振荡段偏移中位 / 偏移 std / 安静段偏移中位」，
   平台残漂指标在该数据集上无意义（平台段跨越振荡包，量到的是振荡包络）。
② 变化负载「零负载-中途切换负载-零负载-切换负载」（第 2 份）：看每次切换后的跟踪误差。

输出：results/v30_special.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402
import v30_ab as A  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
OUT = os.path.join(RES, "v30_special.txt")
ARMS = [("base", "base", []), ("pct10", "proto", ["--pct", "10"]),
        ("pct20", "proto", ["--pct", "20"])]


def osc_metrics(el, tin, tout):
    """振荡段 = 输入总量滚动 std(0.5 s) > 200 ADC 的帧。"""
    w = 50
    c = np.convolve(tin, np.ones(w) / w, mode="same")
    c2 = np.convolve(tin ** 2, np.ones(w) / w, mode="same")
    sd = np.sqrt(np.maximum(c2 - c * c, 0))
    osc = sd > 200.0
    quiet = sd < 60.0
    off = tout - tin
    return dict(osc_frac=float(osc.mean()),
                osc_med=float(np.median(off[osc])) if osc.any() else float("nan"),
                osc_std=float(np.std(off[osc])) if osc.any() else float("nan"),
                osc_pkpk=float(off[osc].max() - off[osc].min()) if osc.any() else float("nan"),
                quiet_med=float(np.median(off[quiet])) if quiet.any() else float("nan"))


def main():
    lines = []
    p = lines.append
    odd = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况")
    f40 = None
    if os.path.isdir(odd):
        for d in sorted(os.listdir(odd)):
            for cand in ("device_001_pre_seg0.csv", "device_001_seg000.csv"):
                q = os.path.join(odd, d, cand)
                if os.path.exists(q):
                    f40 = q
                    break
    p("=== ① 振荡工况 ===")
    p("数据 %s" % f40)
    if f40:
        el, V = A.read_any(f40)
        tin = V.sum(1)
        p("%-8s %8s %10s %10s %10s %10s" %
          ("臂", "振荡占比", "振荡段偏移中位", "振荡段std", "振荡段极差", "安静段偏移中位"))
        for name, rk, args in ARMS:
            D = A.run_arm(A.RUNNERS[rk], args, el, V)
            m = osc_metrics(el, tin, D["sum_out"])
            p("%-8s %8.3f %10.0f %10.0f %10.0f %10.0f" %
              (name, m["osc_frac"], m["osc_med"], m["osc_std"], m["osc_pkpk"],
               m["quiet_med"]))
    p("")
    p("=== ② 变化负载「零负载-中途切换负载」（A/B 里方向不一致的那份）===")
    base = os.path.join(L.ROOT, "temp", "原始数据only", "变化负载")
    tgt = []
    for d in sorted(os.listdir(base)):
        for dp, _dn, fns in os.walk(os.path.join(base, d)):
            for fn in fns:
                if fn == "device_001_seg000.csv":
                    tgt.append((d, os.path.join(dp, fn)))
    for d, path in tgt:
        el, V = A.read_any(path)
        tin = V.sum(1)
        segs = L.hysteresis_segments(tin, min_frames=50)
        step = 1.0
        lv = [float(np.median(tin[i0:i1 + 1])) for k, i0, i1 in segs]
        if len(lv) > 1:
            step = max(lv) - min(lv)
        p("  %s  段数 %d  step %.0f" % (d, len(segs), step))
        p("  %-8s %s" % ("臂", "  ".join("段%d(%s)漂移" % (i, k) for i, (k, _, _) in enumerate(segs))))
        for name, rk, args in ARMS:
            D = A.run_arm(A.RUNNERS[rk], args, el, V)
            tout = D["sum_out"]
            vals = []
            for k, i0, i1 in segs:
                dur = el[i1] - el[i0]
                if dur < 3:
                    vals.append("   nan")
                    continue
                a = float(np.median(tout[i0:i0 + 200])) if i1 - i0 > 200 else float(np.median(tout[i0:i1 + 1]))
                b = float(np.median(tout[max(i0, i1 - 200):i1 + 1]))
                vals.append("%6.3f" % ((b - a) / step))
            p("  %-8s %s" % (name, "  ".join(vals)))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
