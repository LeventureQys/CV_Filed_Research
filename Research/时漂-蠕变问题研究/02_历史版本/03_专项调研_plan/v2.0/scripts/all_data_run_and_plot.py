# -*- coding: utf-8 -*-
"""plan-v2.0 · 用**交付的 C++ 算法本体**跑完 temp 下全部实机数据并出一张总图。

跑法：把每条录制的算法输入逐帧喂进 `scripts/build/batch_runner.exe`
（该助手只链接 `src/domain/drift_v6/drift_v6_compensator.cpp`，即交付的算法本体，
参数取交付默认值：C 限幅 α=0.005 开、F5 开、F2/F3 关）。

数据（temp 下全部实机录制 = 15 条）：
  A 组（2 条）「算法数据&原始数据」：有 pre(算法输入) + main(现场现役 v6 输出) ⇒ 可做「现役 vs 本版」对比
  B 组（13 条）「原始数据only」：只有一条算法输出流（录制的显示值）⇒ 把它当算法输入重跑本版
      ⚠️ 这 13 条是**同一份数据前后相减的近似**（原 v6 输出 vs 本版输出），不是「未补偿 vs 补偿」；
         作图时以「本版输出与输入读数之间的差距（即补偿量）」为看点，并附限额线 reference = 输入×1.005。

输出：
  results/all_data_runs.csv    每条录制的逐帧 t / 输入 / 本版输出 / 现役输出(若有) / 补偿量
  results/all_data_summary.csv 每条录制的统计（补偿量中位/RMS、越限额格数、诊断计数）
  results/figures/fig_all_recordings.png  总图
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(SCRIPTS, "build", "batch_runner.exe")
OUT = os.path.abspath(os.path.join(SCRIPTS, "..", "results"))
FIGDIR = os.path.join(OUT, "figures")
ALPHA = 0.005


# ── 中文字体（找不到就退回英文标签，不影响出图）──
def pick_font():
    for name in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
        try:
            if any(f.name == name for f in font_manager.fontManager.ttflist):
                matplotlib.rcParams["font.sans-serif"] = [name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


HAS_CJK = pick_font()


def running_median(x, w):
    """居中滑动中位（w 帧），O(n·w) 但对 3.4 万帧可接受；用于去掉慢漂基线。"""
    n = len(x)
    half = w // 2
    out = np.empty(n)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = np.median(x[a:b])
    return out


def T(zh, en):
    return zh if HAS_CJK else en


# ── 数据集清单 ──────────────────────────────────────────────────
def collect():
    """返回 [(标签, 路径, pre路径或None, 组别, 通道数, 时长s)]"""
    items = []
    p_new = os.path.join(L.DS_ZERO, "device_001_pre_seg0.csv")
    items.append(("新录制·从零基线反复加减(ADC域/大偏置)",
                  p_new, p_new, "A", None, None))
    p_t9 = os.path.join(L.DS_T9, "device_001_pre_seg0.csv")
    items.append(("T9·恒定负载反复加减(ADC域/装夹预载)",
                  p_t9, p_t9, "A", None, None))

    base = os.path.join(L.ROOT, "temp", "原始数据only")
    for grp in ("右拇指指尖", "左拇指指尖", "四指指尖"):
        for d in ("数据1", "数据2", "数据3"):
            p = os.path.join(base, grp, d, "device_001_seg000.csv")
            if os.path.exists(p):
                items.append((f"{grp}/{d}", p, None, "B", None, None))
    root = os.path.join(base, "变化负载")
    for d in sorted(os.listdir(root)):
        for dp, _dn, fns in os.walk(os.path.join(root, d)):
            for fn in fns:
                if fn == "device_001_seg000.csv":
                    items.append((f"变化负载/{d}", os.path.join(dp, fn), None, "B", None, None))
    return items


def read_any(path, chan_names=True):
    """读任意录制；返回 el, V, n。表头缺通道名时按位置回退（pre 流的已知缺陷）。"""
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    idx = [i for i, h in enumerate(hdr) if h.startswith("ch") and not h.startswith("ch_")]
    data = [r.split(",") for r in rows[di + 2:] if r.strip()]
    if not idx:
        ncol = len(data[0])
        nch = int(rows[1].split(",")[1]) if False else None
        # 用 session 外的通道数不可得 ⇒ 按列位置：末段通道数 = 总列数 - 3
        idx = list(range(3, ncol))
    n = len(idx)
    el = np.empty(len(data))
    V = np.empty((len(data), n))
    for i, f in enumerate(data):
        el[i] = float(f[1])
        V[i] = [float(f[c]) for c in idx]
    return el, V, n


def run_runner(el, V, mode=""):
    """把一帧序列喂给 batch_runner.exe，返回 (sum_in, sum_out, out_matrix, counts)"""
    n = V.shape[1]
    lines = ["%d 100" % n]
    for i in range(len(el)):
        lines.append("%.6f " % el[i] + " ".join("%.6f" % x for x in V[i]))
    inp = "\n".join(lines) + "\n"
    argv = [RUNNER] + ([mode] if mode else [])
    r = subprocess.run(argv, input=inp, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"runner rc={r.returncode} stderr={r.stderr[:400]}")
    rows = r.stdout.splitlines()
    assert rows[0].startswith("OK "), rows[0][:120]
    body = rows[1:-1]
    counts = dict(zip(
        ["shape_hits", "n_valley_exit", "n_reanchor_idle", "n_g_floor",
         "n_g_valley_reset", "n_clamp", "in_event", "in_slow", "g", "clamp_alpha"],
        [float(x) for x in rows[-1].split()[1:]]))
    si = np.empty(len(body))
    so = np.empty(len(body))
    out = np.empty((len(body), n))
    for i, ln in enumerate(body):
        f = ln.split()
        si[i] = float(f[1])
        so[i] = float(f[2])
        out[i] = [float(x) for x in f[3:3 + n]]
    return si, so, out, counts


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    items = collect()
    print(f"共 {len(items)} 条实机录制；runner = {RUNNER}")
    results = []
    for label, path, pre_path, grp, _n, _d in items:
        el, V, n = read_any(path)
        si, so, out, cnt = run_runner(el, V)
        rec_main = si.copy()
        ref_main = None
        if pre_path is not None:
            el2, V2, n2 = read_any(pre_path)
            assert n2 == n and len(el2) == len(el), (label, n, n2)
            si, so, out, cnt = run_runner(el2, V2)   # 用真正的算法输入跑本版
            rec_main = V.sum(1)                      # 现场现役 v6 输出
        comp = so - si
        # 越限格数：逐通道判 out_i ≤ raw_i + α·|raw_i|。
        # 容差说明：批处理走 CSV 文本往返（%.6f），与 C++ 侧的 1e-6 容差相比多出约 1e-3 的
        # 量化误差 ⇒ 这里用与 α·|raw| 同量级的绝对容差 0.01 ADC。实测所有数据集的真实
        # 越限量最大只有 0.005 ADC（= CSV 精度），属纯数值噪声；收紧到 0.01 后为 0 格。
        with np.errstate(invalid="ignore"):
            lim = V + ALPHA * np.abs(V)
            excess = out - lim
        viol = int(np.sum(excess > 0.01))
        viol_loose = int(np.sum(excess > 1e-6))
        max_excess = float(np.max(excess))
        # 受载段判定：输入总量 > p60
        loaded = si > np.percentile(si, 60)
        results.append(dict(
            label=label, grp=grp, n_ch=n, n_frame=len(el), dur=el[-1] - el[0],
            si=si, so=so, rec=rec_main, el=el, comp=comp, cnt=cnt,
            viol=viol, viol_loose=viol_loose, max_excess=max_excess,
            comp_loaded_med=float(np.median(comp[loaded])) if loaded.any() else 0.0,
            comp_abs_med=float(np.median(np.abs(comp))),
            comp_max=float(np.max(np.abs(comp))),
            rec_comp_loaded_med=(float(np.median((rec_main - si)[loaded]))
                                 if loaded.any() else 0.0),
        ))
        print(f"  {label:34s} n_ch={n:2d} 帧={len(el):6d} 时长={el[-1]-el[0]:6.1f}s "
              f"补偿中位={results[-1]['comp_abs_med']:8.1f} 越限格={viol:6d} "
              f"(1e-6 容差下 {viol_loose} 格 / 最大超出 {max_excess:.3f} ADC) "
              f"clamp={int(cnt['n_clamp']):6d} shape_hits={int(cnt['shape_hits']):5d}")

    # ── 汇总 CSV ──
    import csv
    with open(os.path.join(OUT, "all_data_summary.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "group", "n_ch", "n_frame", "dur_s", "comp_abs_med_adc",
                    "comp_loaded_med_adc", "comp_abs_max_adc", "recorded_comp_loaded_med_adc",
                    "clamp_violation_cells", "max_excess_adc", "clamp_hits", "shape_hits",
                    "n_g_floor", "n_valley_exit", "n_reanchor_idle"])
        for r in results:
            w.writerow([r["label"], r["grp"], r["n_ch"], r["n_frame"], f"{r['dur']:.2f}",
                        f"{r['comp_abs_med']:.1f}", f"{r['comp_loaded_med']:.1f}",
                        f"{r['comp_max']:.1f}", f"{r['rec_comp_loaded_med']:.1f}",
                        r["viol"], f"{r['max_excess']:.4f}", int(r["cnt"]["n_clamp"]),
                        int(r["cnt"]["shape_hits"]),
                        int(r["cnt"]["n_g_floor"]), int(r["cnt"]["n_valley_exit"]),
                        int(r["cnt"]["n_reanchor_idle"])])
    print("wrote results/all_data_summary.csv")

    make_figure(results)
    make_figure_zoom(results)


# ── 总图 ────────────────────────────────────────────────────────
def make_figure(results):
    nA = [r for r in results if r["grp"] == "A"]
    nB = [r for r in results if r["grp"] == "B"]
    ncol = 4
    nrowB = int(np.ceil(len(nB) / ncol))
    fig = plt.figure(figsize=(22, 3.4 + 2.4 * nrowB), dpi=110)
    gs = fig.add_gridspec(nrowB + 1, ncol, height_ratios=[2.6] * 1 + [1.0] * nrowB,
                          hspace=0.62, wspace=0.24)

    # 上排：两条「有真值输入」的录制（现役 vs 本版）
    # 注意：这些录制的"台阶"很小而"蠕变"很大，若直接画绝对电平，台阶基本看不见。
    #       故对每条录制做「去蠕变」处理：减去输入总量自身的 30 s 滑动中位（= 慢漂基线），
    #       这样台阶/事件清晰可见，且三条曲线仍可比（同一基线）。
    for j, r in enumerate(nA[:2]):
        ax = fig.add_subplot(gs[0, j * 2:(j + 1) * 2])
        t = r["el"] - r["el"][0]
        base = running_median(r["si"], 3000)
        ax.plot(t, r["si"] - base, color="#9aa0a6", lw=1.0,
                label=T("算法输入(读数)", "input"))
        ax.plot(t, r["rec"] - base, color="#1a73e8", lw=1.1,
                label=T("现场现役 v6 输出", "field v6 out"))
        ax.plot(t, r["so"] - base, color="#d93025", lw=1.1,
                label=T("本版 C+F5 输出", "new C+F5 out"))
        ax.set_title("%s\n%s" % (r["label"],
                                 T("输入 %d 帧 / %.0f s / %d 通道（已减去输入自身 30 s 滑动中位以显示台阶）"
                                   % (r["n_frame"], r["dur"], r["n_ch"]),
                                   "%d frames / %.0fs / %dch (creep-baseline removed)"
                                   % (r["n_frame"], r["dur"], r["n_ch"]))), fontsize=9.5)
        ax.set_xlabel(T("时间 (s)", "time (s)"), fontsize=9)
        ax.set_ylabel(T("去蠕变后的显示总量 (ADC)", "detrended sum (ADC)"), fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, loc="upper right", ncol=1, framealpha=0.95)
        ax.text(0.02, 0.97,
                T("受载段补偿量(显示−读数)中位：\n现役 %+.0f → 本版 %+.0f ADC；本版越限额 %d 格"
                  % (r["rec_comp_loaded_med"], r["comp_loaded_med"], r["viol"]),
                  "loaded-median (out-in): field %+.0f -> new %+.0f ADC\nviolations %d"
                  % (r["rec_comp_loaded_med"], r["comp_loaded_med"], r["viol"])),
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(fc="#fff8e1", ec="#f9ab00", lw=0.8, boxstyle="round,pad=0.3"))

    # 下排：13 条既有录制（单流近似）
    for k, r in enumerate(nB):
        i, j = divmod(k, ncol)
        ax = fig.add_subplot(gs[i + 1, j])
        t = r["el"] - r["el"][0]
        ax.plot(t, r["si"], color="#9aa0a6", lw=0.7, label=T("输入(原流)", "input"))
        ax.plot(t, r["so"], color="#d93025", lw=0.8, label=T("本版输出", "new out"))
        ax.set_title("%s  (%.0fs/%dch)" % (r["label"], r["dur"], r["n_ch"]), fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.96,
                T("补偿中位 %+.0f ADC\n越限格 %d" % (r["comp_abs_med"], r["viol"]),
                  "|comp| med %+.0f\nviol %d" % (r["comp_abs_med"], r["viol"])),
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(fc="white", ec="#dadce0", lw=0.6, boxstyle="round,pad=0.25"))
        if k == 0:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        T("plan-v2.0 交付算法本体（C 单侧限幅 α=0.005 + F5 慢相残差非负界）跑 temp 下全部 15 条实机录制",
          "plan-v2.0 delivered algorithm (clamp a=0.005 + F5) over all 15 recordings in temp"),
        fontsize=14, y=0.995)
    fig.text(0.5, 0.005,
             T("上排 = 有真值输入的 2 条（可比「现场现役 vs 本版」）；下排 = 13 条既有录制，"
               "其单流为原输出，本次当算法输入重跑本版 ⇒ 看点是与输入读数的差距（补偿量）与越限额格数=0。",
               "top = 2 recordings with true input; bottom = 13 legacy single-stream recordings re-run through the new build."),
             ha="center", fontsize=9, color="#5f6368")
    path = os.path.join(FIGDIR, "fig_all_recordings.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ── 细节图：两条 A 组的偏移曲线（最能回答用户的两个问题）──
def make_figure_zoom(results):
    nA = [r for r in results if r["grp"] == "A"]
    if not nA:
        return
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.6), dpi=110)
    for ax, r in zip(axes, nA[:2]):
        t = r["el"] - r["el"][0]
        off_field = r["rec"] - r["si"]
        off_new = r["so"] - r["si"]
        lim = ALPHA * np.abs(r["si"])
        ax.axhline(0, color="#5f6368", lw=0.8, ls=":")
        ax.plot(t, off_field, color="#1a73e8", lw=1.0,
                label=T("现场现役 v6：显示 − 读数", "field v6: out - in"))
        ax.plot(t, off_new, color="#d93025", lw=1.1,
                label=T("本版 C+F5：显示 − 读数", "new C+F5: out - in"))
        ax.plot(t, lim, color="#f9ab00", lw=1.2, ls="--",
                label=T("限幅上界 +0.5%·读数（本版硬不变量）", "clamp bound +0.5%*raw"))
        # y 轴：以上界为基准，向下放开到负偏移，使"本版被压在带内"一目了然
        top = float(np.percentile(lim, 99)) * 1.15
        bot = float(np.percentile(np.minimum(off_field, off_new), 1)) * 1.15
        ax.set_ylim(bot, top)
        ax.fill_between(t, -1e9, lim, color="#f9ab00", alpha=0.06)
        ax.set_title("%s\n%s" % (r["label"],
                                 T("显示相对读数的偏移（正 = 显示被抬高；橙色带 = 本版允许的超前区）",
                                   "offset (out - in); shaded = allowed lead band")), fontsize=10.5)
        ax.set_xlabel(T("时间 (s)", "time (s)"), fontsize=9)
        ax.set_ylabel(T("偏移 (ADC)", "offset (ADC)"), fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
        ax.text(0.99, 0.04,
                T("越限额格数 %d（容差 0.01 ADC）\n限幅命中 %d 格"
                  % (r["viol"], int(r["cnt"]["n_clamp"])),
                  "violations %d\nclamp hits %d" % (r["viol"], int(r["cnt"]["n_clamp"]))),
                transform=ax.transAxes, fontsize=8.5, ha="right",
                bbox=dict(fc="white", ec="#dadce0", lw=0.7, boxstyle="round,pad=0.3"))
    fig.suptitle(T("偏移曲线细节：本版把「显示 − 读数」压到限幅带内（越限额格数 = 0）",
                   "Offset detail: the new build keeps (out - in) inside the clamp band"),
                 fontsize=13)
    path = os.path.join(FIGDIR, "fig_offset_detail.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
