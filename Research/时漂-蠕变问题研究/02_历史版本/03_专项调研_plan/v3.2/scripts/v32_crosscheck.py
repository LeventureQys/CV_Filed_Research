# -*- coding: utf-8 -*-
"""plan-v3.2：跨数据集快速核查（v3.1 vs v3.2）——是否只在"卸载-重载"工况上生效。

对数据根下**全部**会话逐条比较两臂的总量输出：
  · 差异帧数与最大 |Δ|
  · 全程 ded 中位（补偿量是否被整体改变）
  · 台阶保真 G 中位
判据：没有卸载-重载的录制应当几乎逐位相同（冷启动 creep_ratio_=0）。
输出：results/v32_crosscheck.txt
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
RUN_V31 = os.path.join(V32, "..", "v3.1", "scripts", "build", "v30_runner.exe")
RUN_V32 = os.path.join(V32, "build", "v32_runner.exe")


def read_any(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    idx = [i for i, h in enumerate(hdr) if h.startswith("ch") and not h.startswith("ch_")]
    data = [r.split(",") for r in rows[di + 2:] if r.strip()]
    if not idx:
        idx = list(range(3, len(data[0])))
    el = np.array([float(f[1]) for f in data])
    V = np.array([[float(f[c]) for c in idx] for f in data])
    return el, V


def run(exe, el, V):
    lines = ["%d" % V.shape[1]]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % x for x in row))
    p = subprocess.run([exe], input="\n".join(lines) + "\n", capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    rows = p.stdout.splitlines()
    frames = int(rows[1].split()[2])
    X = np.array([[float(x) for x in r.split()] for r in rows[2:2 + frames]])
    return {c: X[:, i] for i, c in enumerate(rows[0].split())}


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def main():
    items = []
    for lbl, d in L.discover_sessions():
        p = os.path.join(d, "device_001_pre_seg0.csv")
        if os.path.isfile(p):
            items.append((lbl, p))
    lines = []
    p = lines.append
    p("plan-v3.2 跨数据集核查（%d 条）：v3.1 vs v3.2" % len(items))
    p("%-58s %7s %10s %10s %9s %9s %9s" %
      ("数据集", "帧数", "差异帧", "max|Δ|", "ded31中位", "ded32中位", "G31/G32"))
    for lbl, path in items:
        el, V = read_any(path)
        try:
            a = run(RUN_V31, el, V)
            b = run(RUN_V32, el, V)
        except Exception as exc:  # noqa: BLE001
            p("%-58s RUN FAIL %s" % (lbl[:58], exc))
            continue
        tin = V.sum(1)
        d = b["sum_out"] - a["sum_out"]
        nz = int(np.sum(np.abs(d) > 1e-6))
        gs = []
        for name, o in (("a", a["sum_out"]), ("b", b["sum_out"])):
            g = []
            up, _ = L.edge_indices(tin, float(np.median(tin)))
            for i in up:
                x0 = max(0, i - 25)
                x1 = min(len(el) - 1, i + 450)
                x2 = min(len(el) - 1, i + 550)
                if x2 <= x1:
                    continue
                di = float(np.median(tin[x1:x2]) - np.median(tin[x0:i]))
                do = float(np.median(o[x1:x2]) - np.median(o[x0:i]))
                if abs(di) > 1000:
                    g.append(do / di)
            gs.append(float(np.median(g)) if g else float("nan"))
        p("%-58s %7d %10d %10.0f %9.0f %9.0f %9s" %
          (lbl[:58], len(el), nz, float(np.max(np.abs(d))),
           float(np.median(tin - a["sum_out"])), float(np.median(tin - b["sum_out"])),
           "%.3f/%.3f" % (gs[0], gs[1])))
    out = os.path.join(V32, "results", "v32_crosscheck.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
