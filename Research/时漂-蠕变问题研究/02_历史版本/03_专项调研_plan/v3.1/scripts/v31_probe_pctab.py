# -*- coding: utf-8 -*-
"""探针 E（v3.1）：把「卸载沿响应」在 PCT 开/关两臂下对照（同一回放输入，只看相对差）。

输出：results/v31_pct_ab.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3.0", "scripts")))
import v30_ab as A  # noqa: E402

WORK = os.path.join(L.DATA_ROOT, "working")
DS = os.path.join(WORK, "零基线-反复增减同一负载",
                  "20260919_152749_single_device_0cb8b6")
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
TE = (24.42, 61.55, 97.63, 115.15, 272.81)


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def main():
    el, pre = A.read_any(os.path.join(DS, "device_001_pre_seg0.csv"))
    tin = pre.sum(1)
    arms = {
        "PCT关(--pct 0)": ["--pct", "0"],
        "PCT开(默认)": [],
    }
    if os.environ.get("V31_FIXED") == "1":
        arms["PCT开(默认,+修复)"] = []
    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("")
    outs = {}
    for name, args in arms.items():
        outs[name] = A.run_arm(A.RUNNERS["base"], args, el, pre)["sum_out"]
    p("=== 卸载沿的 Δ（沿后 [3,5] s − 沿前 [−2.5,−0.5] s）与 G ===")
    p("%8s %10s %s" % ("沿t", "Δ输入", "  ".join("%-24s" % k for k in arms)))
    for t in TE:
        di = wmed(el, tin, t + 3, t + 5) - wmed(el, tin, t - 2.5, t - 0.5)
        cells = []
        for name in arms:
            o = outs[name]
            do = wmed(el, o, t + 3, t + 5) - wmed(el, o, t - 2.5, t - 0.5)
            cells.append("Δ%+7.0f  G %5.3f" % (do, do / di if abs(di) > 1 else float("nan")))
        p("%8.2f %10.0f %s" % (t, di, "  ".join("%-24s" % c for c in cells)))
    p("")
    p("=== 卸载沿的**瞬态形状**（沿前 0.5 s → 沿后 0.1/0.3/0.5/1/2/5 s 的显示变化）===")
    for t in TE:
        p("--- 沿 t=%.2f ---" % t)
        base_i = wmed(el, tin, t - 0.5, t)
        p("   %-22s %s" % ("臂", "  ".join("%8s" % ("+" + str(k) + "s") for k in
                                          (0.1, 0.3, 0.5, 1, 2, 5, 10))))
        for name in arms:
            o = outs[name]
            b = wmed(el, o, t - 0.5, t)
            vals = [wmed(el, o, t + k, t + k + 0.1) - b for k in (0.1, 0.3, 0.5, 1, 2, 5, 10)]
            p("   %-22s %s" % (name, "  ".join("%+8.0f" % v for v in vals)))
        p("   %-22s %s" % ("（输入，供对照）",
                           "  ".join("%+8.0f" % (wmed(el, tin, t + k, t + k + 0.1) - base_i)
                                     for k in (0.1, 0.3, 0.5, 1, 2, 5, 10))))
    p("")
    p("=== 全录制：显示漂移（末段 − 首段，总量） ===")
    for name in arms:
        o = outs[name]
        p("  %-22s 首 5 s %.0f  末 5 s %.0f  极差 %.0f" %
          (name, wmed(el, o, 15, 20), wmed(el, o, 330, 335), o.max() - o.min()))
    with open(os.path.join(RES, "v31_pct_ab.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % os.path.join(RES, "v31_pct_ab.txt"))


if __name__ == "__main__":
    main()
