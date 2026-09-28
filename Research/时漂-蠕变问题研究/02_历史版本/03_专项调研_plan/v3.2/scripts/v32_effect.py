# -*- coding: utf-8 -*-
"""plan-v3.2 效果图：算法输入 / 现役 v3.1 / 候选 v3.2 的**总量波形**对比。

用法：
    python v32_effect.py                       # 默认目标录制（完全卸载后重载）
    python v32_effect.py "<会话目录>" [输出png]
输出：
    ../figures/F1_total_waveform.png   （只画总波形，无其它内容）
    ../results/v32_effect.txt          （同口径数值，供核对）
"""
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
DATA = os.path.join(ROOT, "temp", "算法数据&原始数据")
RUNNER_V31 = os.path.join(ROOT, "temp", "v4.1flash", "plan", "v3.1",
                          "scripts", "build", "v30_runner.exe")
RUNNER_V32 = os.path.join(V32, "build", "v32_runner.exe")
FIGDIR = os.path.join(V32, "figures")
RESDIR = os.path.join(V32, "results")
DEFAULT_DS = os.path.join(DATA, "working", "零基线-反复增减同一负载",
                          "20260919_160854_single_device_7b3977")


def read_stream(path, nch=21):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    data = [r.split(",") for r in rows[di + 2:] if r.strip()]
    el = np.array([float(f[1]) for f in data])
    V = np.array([[float(x) for x in f[3:3 + nch]] for f in data])
    return el, V


def run(runner, el, V):
    lines = ["%d" % V.shape[1]]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % x for x in row))
    p = subprocess.run([runner], input="\n".join(lines) + "\n", capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("%s rc=%d %s" % (runner, p.returncode, p.stderr[:400]))
    rows = p.stdout.splitlines()
    frames = int(rows[1].split()[2])
    X = np.array([[float(x) for x in r.split()] for r in rows[2:2 + frames]])
    cols = rows[0].split()
    D = {c: X[:, i] for i, c in enumerate(cols)}
    end = rows[2 + frames].split()
    meta = dict(zip("clamp shape valley_exit reanchor_idle g_floor g_valley_reset "
                    "in_event in_slow g".split(), [float(x) for x in end[1:]]))
    return D, meta


def setup_cjk_font():
    """让中文标签可显示；找不到中文字体时返回 False（调用方改用英文标签）。"""
    import matplotlib
    from matplotlib import font_manager
    want = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
            "WenQuanYi Micro Hei", "SimSun"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in want:
        if name in have:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DS
    out_png = (sys.argv[2] if len(sys.argv) > 2
               else os.path.join(FIGDIR, "F1_total_waveform.png"))
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    el, pre = read_stream(os.path.join(ds, "device_001_pre_seg0.csv"))
    _, main = read_stream(os.path.join(ds, "device_001_seg000.csv"))
    tin, tmain = pre.sum(1), main.sum(1)
    d31, m31 = run(RUNNER_V31, el, pre)
    d32, m32 = run(RUNNER_V32, el, pre)
    t31, t32 = d31["sum_out"], d32["sum_out"]

    lines = []
    p = lines.append
    p("数据集 %s" % ds)
    p("帧 %d  时长 %.1f s" % (len(el), el[-1] - el[0]))
    p("v3.1 runner meta %s" % m31)
    p("v3.2 runner meta %s" % m32)
    p("")
    p("=== 完全卸载后重载（245.3 s）之后的补偿量 ded = 输入 − 显示 ===")
    p("%10s %10s %10s %10s" % ("沿后", "输入", "v3.1 ded", "v3.2 ded"))
    for t0 in (245.31,):
        for dt in (5, 10, 20, 40, 60):
            a = wmed(el, tin, t0 + dt, t0 + dt + 1)
            p("%10s %10.0f %10.0f %10.0f" %
              ("+%d s" % dt, a, a - wmed(el, t31, t0 + dt, t0 + dt + 1),
               a - wmed(el, t32, t0 + dt, t0 + dt + 1)))
    p("")
    p("=== 全程：|显示 − 输入| / 输入 的中位（越小 = 显示越贴近输入；这里看的是\"还扣不扣\"）===")
    for name, o in (("现场录制", tmain), ("v3.1", t31), ("v3.2", t32)):
        d = tin - o
        p("  %-8s ded 中位 %8.0f   最大 %8.0f   [240,312] s 段内 ded 中位 %8.0f" %
          (name, float(np.median(d)), float(np.max(d)),
           float(np.median(d[el >= 240]))))
    p("")
    p("=== 台阶保真 G（加载沿后 [3,5] s 的 Δ显示/Δ输入）与 C 限幅不变量 ===")
    for name, o, meta in (("v3.1", t31, m31), ("v3.2", t32, m32)):
        gs = []
        for t0 in (3.3, 15.2, 41.8, 63.4, 232.2, 245.3, 258.3, 261.7, 278.6,
                   289.3, 300.6, 303.4):
            if t0 + 5 > el[-1]:
                continue
            di = wmed(el, tin, t0 + 3, t0 + 5) - wmed(el, tin, t0 - 2.5, t0 - 0.5)
            do = wmed(el, o, t0 + 3, t0 + 5) - wmed(el, o, t0 - 2.5, t0 - 0.5)
            if abs(di) > 1000:
                gs.append(do / di)
        p("  %-6s G 中位 %.3f  范围 %.3f~%.3f  限幅命中 %d" %
          (name, float(np.median(gs)) if gs else float("nan"),
           min(gs) if gs else float("nan"), max(gs) if gs else float("nan"),
           int(meta["clamp"])))
    with open(os.path.join(RESDIR, "v32_effect.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ── 画图：只有总波形 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cjk = setup_cjk_font()
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(15.0, 8.4), dpi=130, sharex=False,
        gridspec_kw=dict(height_ratios=[1.55, 1.0], hspace=0.28))

    def draw(a, x0=None, x1=None, lw_scale=1.0):
        a.plot(el, tin, color="#9aa0a6", lw=0.9 * lw_scale, zorder=2,
               label=L("算法输入 pre（无补偿时应显示的值）", "algorithm input (pre)"))
        a.plot(el, t31, color="#1a73e8", lw=2.6 * lw_scale, alpha=0.55, zorder=3,
               label=L("现役 v3.1（完全卸载后补偿归零）",
                       "current v3.1 (compensation lost after unload)"))
        a.plot(el, t32, color="#d93025", lw=1.2, zorder=4,
               label=L("候选 v3.2（重载蠕变接力）", "candidate v3.2 (creep hand-over)"))
        a.grid(alpha=0.25, lw=0.5)

    draw(ax)
    ax.set_xlim(float(el[0]), float(el[-1]))
    ax.set_ylim(2200, max(float(tin.max()), float(t31.max()), float(t32.max())) * 1.08)
    ax.set_ylabel(L("总量 Σch (ADC)", "total sum (ADC)"), fontsize=11)
    ax.set_xlabel(L("时间 elapsed (s)", "elapsed (s)"), fontsize=11)
    ax.set_title(L("plan-v3.2 效果：完全卸载后重载，显示补偿是否回来（21 通道总量）",
                   "plan-v3.2 effect: total-value waveform"), fontsize=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.92)

    z0, z1 = 236.0, 300.0
    draw(ax2, lw_scale=1.25)
    ax2.set_xlim(z0, z1)
    ax2.set_ylim(2200, max(float(tin[(el >= z0) & (el <= z1)].max()),
                           float(t31[(el >= z0) & (el <= z1)].max()),
                           float(t32[(el >= z0) & (el <= z1)].max())) * 1.08)
    ax2.set_ylabel(L("总量 Σch (ADC)", "total sum (ADC)"), fontsize=11)
    ax2.set_xlabel(L("时间 elapsed (s)", "elapsed (s)"), fontsize=11)
    ax2.set_title(L("局部放大 %.0f~%.0f s：反复「完全卸载 → 重载」，v3.1 此后基本不再补偿，"
                    "v3.2 把补偿接回来" % (z0, z1),
                    "zoom %.0f-%.0f s" % (z0, z1)), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    print("\n".join(lines))
    print("\n-> %s" % out_png)
    print("-> %s" % os.path.join(RESDIR, "v32_effect.txt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
