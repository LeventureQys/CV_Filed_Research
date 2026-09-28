# -*- coding: utf-8 -*-
"""plan-v3.2：种子折减 gain 的 A/B（同一录制、同一回放口径）。

臂：gain ∈ {0（= 关闭 R1，等价 v3.1）, 0.5, 0.7, 0.85, 1.0}
指标（全部总量口径）：
  · 每次「完全卸载后重载」沿后 [20,40] s 的补偿量 ded = 输入 − 显示（越大 = 基线回来了）
  · 重载后 [0.3,1.5] s 的**负向超调**（显示比输入低多少；越小越好，避免一帧打穿）
  · 全程 |ded| 的最大值（别出现离谱大扣）
  · 加载沿台阶保真 G（[3,5] s Δ显示/Δ输入，应接近 1）
输出：results/v32_gain_ab.txt
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3.1", "scripts")))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.abspath(os.path.join(HERE, ".."))
RUNNER = os.path.join(V32, "build", "v32_runner.exe")
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")
GAINS = [0.0, 0.5, 0.7, 0.85, 1.0]
ONSETS = [3.34, 15.18, 41.75, 63.41, 232.23, 245.31, 258.34, 261.72, 278.63,
          289.33, 300.61, 303.42]


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def run(gain, el, V):
    args = [RUNNER] + (["--seed-gain", "%.3f" % gain] if gain is not None else [])
    lines = ["%d" % V.shape[1]]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % x for x in row))
    p = subprocess.run(args, input="\n".join(lines) + "\n", capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("rc=%d %s" % (p.returncode, p.stderr[:300]))
    rows = p.stdout.splitlines()
    frames = int(rows[1].split()[2])
    X = np.array([[float(x) for x in r.split()] for r in rows[2:2 + frames]])
    return {c: X[:, i] for i, c in enumerate(rows[0].split())}


def main():
    ds = L.load_dataset(DS)
    el, V = ds["pre"]["el"], ds["pre"]["V"]
    tin = V.sum(1)
    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("")
    p("=== 各 gain 臂：重载沿后 [20,40] s 的补偿量 ded"
      "（越大越好；v3.1 约 -60 即“归零”）===")
    p("%-8s %s" % ("gain", " ".join("%9.1f" % t for t in ONSETS)))
    store = {}
    for gain in GAINS:
        D = run(gain, el, V)
        tout = D["sum_out"]
        store[gain] = D
        cells = []
        for t in ONSETS:
            a, b = t + 20, min(t + 40, float(el[-1]) - 0.1)
            cells.append("%9.0f" % (wmed(el, tin, a, b) - wmed(el, tout, a, b))
                         if b - a >= 5 else "%9s" % "-")
        p("%-8.2f %s" % (gain, " ".join(cells)))
    p("")
    p("=== 代价与不变量 ===")
    p("%-8s %14s %14s %12s %12s %10s" %
      ("gain", "重载后最小ded", "平均ded", "全程|ded|max", "G中位", "限幅命中"))
    for gain in GAINS:
        D = store[gain]
        tout = D["sum_out"]
        over, mx, gs = [], 0.0, []
        for t in (245.31, 258.34, 278.63, 289.33, 300.61):
            m = (el >= t) & (el < t + 4.0)
            if m.any():
                d = float(np.min(tin[m] - tout[m]))
                over.append(d)
                mx = max(mx, float(np.max(np.abs(tin[m] - tout[m]))))
        for t0 in ONSETS:
            if t0 + 5 > el[-1]:
                continue
            di = wmed(el, tin, t0 + 3, t0 + 5) - wmed(el, tin, t0 - 2.5, t0 - 0.5)
            do = wmed(el, tout, t0 + 3, t0 + 5) - wmed(el, tout, t0 - 2.5, t0 - 0.5)
            if abs(di) > 1000:
                gs.append(do / di)
        p("%-8.2f %14.0f %14.0f %12.0f %12.3f %10.0f" %
          (gain, min(over), float(np.mean(over)), mx, float(np.median(gs)),
           float(D["clamp_hits"][-1])))
    out = os.path.join(V32, "results", "v32_gain_ab.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
