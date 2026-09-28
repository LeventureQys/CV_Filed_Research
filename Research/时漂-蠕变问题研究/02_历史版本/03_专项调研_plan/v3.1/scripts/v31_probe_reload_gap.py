# -*- coding: utf-8 -*-
"""探针 N（临时诊断）：交接时刻的「显示落点」与「最终平台」之差 —— 重载基线丢失的定量证据。

对每次加载事件：
  t_step   = 真实加载沿（输入 5%→95% 的跨越帧）
  t_ho     = 交接帧（回放 state 1→2）
  out@ho   = 交接当帧显示（此时扣除被复位到 ~0，显示≈输入）
  plateau  = 沿后 [45,55] s 的输入中位（该次加载的最终电平）
  headroom = plateau − out@ho       ← 显示还能"靠扣除压下去"的空间
  ded_late = 沿后 [45,55] s 的补偿中位
判据：headroom 决定 ded_late。若交接把显示钉到接近最终平台 ⇒ headroom≈0 ⇒ 补偿≈0（基线丢失）。

输出：results/v31_reload_gap.txt
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def main():
    ds = L.load_dataset(DS)
    el, V = ds["pre"]["el"], ds["pre"]["V"]
    tin = V.sum(1)
    lines = ["%d" % V.shape[1]]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % x for x in row))
    p = subprocess.run([os.path.join(HERE, "build", "v30_runner.exe")],
                       input="\n".join(lines) + "\n", capture_output=True,
                       text=True, encoding="utf-8")
    rows = p.stdout.splitlines()
    frames = int(rows[1].split()[2])
    X = np.array([[float(x) for x in r.split()] for r in rows[2:2 + frames]])
    cols = rows[0].split()
    D = {c: X[:, i] for i, c in enumerate(cols)}
    tout = D["sum_out"]
    st, ev, kind = D["state"], D["ev_valid"], D["ev_kind"]

    # 真实加载沿：平滑输入 5%→95%
    sm = np.convolve(tin, np.ones(15) / 15.0, mode="same")
    steps = []
    i = 1
    n = len(el)
    while i < n:
        if sm[i] - sm[i - 1] > 300:
            j = i
            while j + 1 < n and sm[j + 1] - sm[j] > 300:
                j += 1
            pre = sm[max(0, i - 30):i].mean()
            post = sm[j:min(n, j + 200)].max()
            if post - pre > 5000:
                steps.append((float(el[i]), pre, post))
            i = j + 1
        else:
            i += 1
    # 交接帧：state 1→2
    ho = [k for k in range(1, n) if st[k - 1] == 1 and st[k] == 2]

    lines.append("数据集 %s" % DS)
    lines.append("")
    lines.append("=== 加载沿与交接 ===")
    lines.append("%9s %14s %10s %10s %10s %10s %10s %10s %10s" %
                 ("加载沿t", "台阶(平滑)", "交接t", "out@交接", "in@交接",
                  "平台(45-55s)", "headroom", "ded(45-55s)", "ded/平台"))
    for (t, pre, post) in steps:
        cand = [k for k in ho if 0 <= el[k] - t <= 8.0]
        t_ho = float(el[cand[0]]) if cand else float("nan")
        o_ho = tout[cand[0]] if cand else float("nan")
        i_ho = tin[cand[0]] if cand else float("nan")
        pl = wmed(el, tin, t + 45, t + 55)
        ded = wmed(el, tin, t + 45, t + 55) - wmed(el, tout, t + 45, t + 55)
        lines.append("%9.2f %14.0f %10.2f %10.0f %10.0f %10.0f %10.0f %10.0f %10.3f" %
                     (t, post - pre, t_ho, o_ho, i_ho, pl, pl - o_ho, ded,
                      0.0 if not np.isfinite(pl) or pl < 1 else ded / pl))
    lines.append("")
    lines.append("=== 对照：首次加载（3.30 s）与完全卸载后重载（245.27 s）的整段 ===")
    lines.append("%-8s %10s %10s %10s %10s %10s %10s" %
                 ("dt(s)", "in", "out", "ded", "ded/in", "in−in@1s", "A_sum"))
    for tag, t0 in (("首次3.30", 3.30), ("重载245.27", 245.27)):
        lines.append("  --- %s ---" % tag)
        for dt in (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 20, 40, 60, 90):
            a, b = t0 + dt, t0 + dt + 0.4
            i_in = wmed(el, tin, a, b)
            i_out = wmed(el, tout, a, b)
            lines.append("  %-8s %10.0f %10.0f %10.0f %10.3f %10.0f %10.0f" %
                         ("%.1f" % dt, i_in, i_out, i_in - i_out,
                          0.0 if i_in < 1 else (i_in - i_out) / i_in,
                          i_in - wmed(el, tin, t0 + 0.9, t0 + 1.1),
                          wmed(el, D["A_sum"], a, b)))
    out = os.path.join(RES, "v31_reload_gap.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
