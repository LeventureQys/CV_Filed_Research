# -*- coding: utf-8 -*-
"""探针 M（临时诊断）：把候选变体放到「反复增减同一负载」录制上并排比。

对每个变体（沙箱源码编译出的 runner）输出：
  · 每次加载沿后 5/10/20/40/60/90/120 s 的补偿量（ded = 输入 − 显示）
  · 每段受载平台的稳态补偿量中位
判据：**同一负载的第二次加载应当与第一次给出同量级的补偿**（基线保持）。

输出：results/v31_variant_ab.txt
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
VDIR = os.path.join(RES, "v31_variants", "build")
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")
ONSETS = [3.34, 15.18, 41.75, 63.41, 232.23, 245.31, 258.34, 261.72, 278.63,
          289.33, 300.61, 303.42]


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def run(runner, el, V):
    lines = ["%d" % V.shape[1]]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % x for x in row))
    p = subprocess.run([runner], input="\n".join(lines) + "\n", capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("runner rc=%d %s" % (p.returncode, p.stderr[:300]))
    rows = p.stdout.splitlines()
    frames = int(rows[1].split()[2])
    X = np.array([[float(x) for x in r.split()] for r in rows[2:2 + frames]])
    cols = rows[0].split()
    end = rows[2 + frames].split()
    return {c: X[:, i] for i, c in enumerate(cols)}, end


def main():
    ds = L.load_dataset(DS)
    el, V = ds["pre"]["el"], ds["pre"]["V"]
    tin = V.sum(1)
    t_end = float(el[-1])
    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p(""
      )
    arms = [("现场录制(plan-v3.0首版)", None)] + [
        (n, os.path.join(VDIR, "v31_%s.exe" % n))
        for n in ("cur", "anchor_base", "anchor_base_nocap", "anchor_obs")]
    store = {}
    for name, exe in arms:
        if exe is None:
            tout = ds["main"]["V"].sum(1)
        else:
            D, end = run(exe, el, V)
            tout = D["sum_out"]
        store[name] = tout
        p("=== 臂 %s ===" % name)
        p("%8s %9s %9s %9s %9s %9s %9s %9s" %
          ("加载沿t", "+5s", "+10s", "+20s", "+40s", "+60s", "+90s", "+120s"))
        for t in ONSETS:
            vals = []
            for dt in (5, 10, 20, 40, 60, 90, 120):
                if t + dt + 2 > t_end:
                    vals.append(float("nan"))
                    continue
                vals.append(wmed(el, tin, t + dt, t + dt + 2) -
                            wmed(el, tout, t + dt, t + dt + 2))
            p("%8.2f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f" % (t, *vals))
        # 受载平台稳态补偿
        segs = L.hysteresis_segments(tin, min_frames=200)
        p("  受载平台稳态补偿中位（段长≥20 s）:")
        for kind, i0, i1 in segs:
            if kind != "loaded" or el[i1] - el[i0] < 20:
                continue
            j = i0
            while j <= i1 and el[j] < el[i0] + 10:
                j += 1
            d = wmed(el, tin, el[j], el[i1]) - wmed(el, tout, el[j], el[i1])
            p("    t=[%7.1f,%7.1f] dur %5.1f s  ded %6.0f  (%.1f%% of in)" %
              (el[i0], el[i1], el[i1] - el[i0], d,
               0.0 if wmed(el, tin, el[j], el[i1]) < 1 else
               100.0 * d / wmed(el, tin, el[j], el[i1])))
        p("")
    # 综合：各次加载后 [20,40] s 的补偿
    p("=== 判据汇总：每次加载后 [20,40] s 的补偿量（ded = 输入 − 显示）===")
    p("%-24s %s" % ("臂", " ".join("%9.1f" % t for t in ONSETS)))
    for name, _ in arms:
        tout = store[name]
        cells = []
        for t in ONSETS:
            a, b = t + 20, min(t + 40, t_end - 0.1)
            if b - a < 5:
                cells.append("%9s" % "-")
            else:
                cells.append("%9.0f" % (wmed(el, tin, a, b) - wmed(el, tout, a, b)))
        p("%-24s %s" % (name, " ".join(cells)))
    p("")
    p("（首列 3.34 = 首次 onset；245.31 起为「完全卸载后重载」——与首列同量级才叫基线保持）")
    out = os.path.join(RES, "v31_variant_ab.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
