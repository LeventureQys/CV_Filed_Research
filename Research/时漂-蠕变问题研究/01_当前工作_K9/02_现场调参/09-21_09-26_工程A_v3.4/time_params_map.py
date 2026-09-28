# -*- coding: utf-8 -*-
"""K9 观测器全部时间参数地图 + 在实测数据上按时间轴标注。

回答两个问题：
  1) 算法里可调的时间参数有哪些、各自管哪段（图上按加载沿对齐的时间窗标注）；
  2) x2 为什么启动晚（主体 = slow_confirm_s 受载确认窗），调短后的效果对比。

用法：python time_params_map.py [会话目录名，默认 20260922_095849_single_device_6cca99]
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_v34_on_csv import read_session_csv  # noqa: E402
from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402

LN2 = 0.69314718056

# ── 全部时间参数（名称 / 当前值 / 管什么 / 对 x2 启动的影响方向）──
TIME_PARAMS = [
    # (属性名, 现役值, 作用, 与 x2 启动的关系)
    ("slow_confirm_s",     10.0, "x2 受载确认窗：load_dwell 连续≥该时长才允许 x2 积分",
                                 "★ 启动延迟的主体：x2 最早 ≈ 加载后该时长"),
    ("soft_unfreeze_s",    4.0,  "x2 软冻结窗：沿后 t_edge > 该值×ln2(≈2.77s) 才解冻",
                                 "次要延迟项（与确认窗取大者）"),
    ("tau_slope_s",        1.0,  "斜率低通 τ（slope 估计的平滑度）",
                                 "影响沿检测/缓坡计时的灵敏度"),
    ("edge_boost_s",       2.0,  "沿后 x1 前馈窗长（窗内 τc 用 boost 值）",
                                 "无关（x1 阶段）"),
    ("edge_refract_s",     2.0,  "沿不应期（防重复触发沿）",
                                 "无关"),
    ("tau_c_fast_boost_s", 2.0,  "沿后前馈窗内 x1 的收敛 τ（越短 x1 冲得越快）",
                                 "无关（x1 阶段过激程度）"),
    ("tau_c_fast_s",       12.0, "x1 受载收敛 τ（x1 爬向 r_fast·e 的时间常数）",
                                 "无关启动，但决定 x1 长尾下沉时长"),
    ("tau_r_fast_s",       2.0,  "x1 空载恢复 τ（卸载后 x1 泄放）",
                                 "无关"),
    ("tau_r_slow_idle_s",  8.0,  "空载期 x2 快泄放 τ（K7）",
                                 "卸载后 x2 清空速度（重载前残留）"),
    ("tau_r_slow_s",       150.0,"非空载、e≤0 时 x2 泄放 τ（下漂收敛速度）",
                                 "回撤后残留补偿的滞留时长"),
    ("tau_zero_s",         8.0,  "近零带内零点跟踪 τ",
                                 "无关"),
    ("y_floor_tau_s",      300.0,"去趋势基线上行跟踪 τ",
                                 "决定 idle 判据 → 间接影响 dwell 起点"),
    ("y_max_tau_s",        600.0,"量程包络衰减 τ",
                                 "无关"),
    ("hold_tau_s",         0.5,  "K6 预留池限额低通 τ",
                                 "无关"),
    ("ramp_full_s",        4.0,  "K9 缓坡满计时（缓坡持续该时长后 tc1 全额 boost）",
                                 "无关"),
    ("bypass_base_tau_s",  10.0, "K7 旁路总值基线 EMA τ",
                                 "无关"),
]


def run(p: Params, t, V, want=("out", "x1", "x2")):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    outs = np.empty(n); x1s = np.empty(n); x2s = np.empty(n)
    for i in range(n):
        outs[i] = c.process(float(t[i]), V[i]).sum()
        x1s[i] = c.x_fast.sum(); x2s[i] = c.x_slow.sum()
    return outs, x1s, x2s


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sess = sys.argv[1] if len(sys.argv) > 1 else "20260922_095849_single_device_6cca99"
    csv = HERE / "长期数据" / sess / "device_001_seg000.csv"
    d = read_session_csv(csv)
    t, V = d["t"], d["values"]
    tin = V.sum(axis=1)

    p0 = Params()
    print(f"会话 {sess}：{t[-1]-t[0]:.0f}s × {V.shape[1]} 通道\n")
    print(f"{'参数':22s} {'现值':>7s}  作用 / 与 x2 启动的关系")
    for name, val, desc, rel in TIME_PARAMS:
        print(f"{name:22s} {val:7.1f}  {desc}；{rel}")

    base = replace(p0, r_fast=0.06)
    out, x1s, x2s = run(base, t, V)

    # x2 启动时刻检测（对照用）
    peak = tin.max(); base5 = float(np.percentile(tin, 5)); step = peak - base5
    i_on = int(np.argmax(tin > 0.2 * step + base5))

    def x2_start(x2):
        above = x2 > 0.005 * step
        return (t[int(np.argmax(above))] if above.any() else None)

    t_x2 = x2_start(x2s)

    # slow_confirm 扫描
    scan = []
    for sc in (10.0, 6.0, 4.0, 2.0):
        o, a1, a2 = run(replace(base, slow_confirm_s=sc), t, V)
        scan.append((sc, a2, x2_start(a2)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.6, 1.2], hspace=0.3)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharex=a1)
    a3 = fig.add_subplot(gs[2], sharex=a1)

    t0 = t[i_on]      # 加载沿为注释基准
    # ── 顶图：输入/显示 + 各时间窗标注（相对加载沿）──
    a1.plot(t, tin, color="#7f8c8d", lw=1.0, label="输入（未补偿）")
    a1.plot(t, out, color="#2980b9", lw=1.3, label="K9 新参显示")
    a1.axvline(t0, color="#e67e22", lw=1.4, ls="--")
    a1.annotate(f"加载沿 t={t0:.1f}s", xy=(t0, tin[i_on]), xytext=(6, 24),
                textcoords="offset points", fontsize=10, color="#b9770e")

    def band(dt_from, dt_to, color, label, y_frac=0.97):
        a1.axvspan(t0 + dt_from, t0 + dt_to, color=color, alpha=0.13, lw=0)
        a1.text(t0 + (dt_from + dt_to) / 2, a1.get_ylim()[1] * 0.0 + _ytop(a1) * y_frac,
                label, ha="center", va="top", fontsize=8.5, color=color)

    def _ytop(ax):
        lo, hi = ax.get_ylim()
        return hi - (hi - lo) * 0.02

    # 时间窗（画之前先 reserve ylim：临时 plot 再删）
    a1.set_ylim(0, tin.max() * 1.05)
    band(0.0, base.edge_boost_s, "#e67e22",
         f"edge_boost_s={base.edge_boost_s:g}s\nx1 前馈窗(τc_boost={base.tau_c_fast_boost_s:g}s)")
    band(0.0, base.soft_unfreeze_s * LN2, "#16a085",
         f"soft_unfreeze_s×ln2\n≈{base.soft_unfreeze_s*LN2:.1f}s x2 软冻结")
    band(0.0, base.slow_confirm_s, "#c0392b",
         f"slow_confirm_s={base.slow_confirm_s:g}s\nx2 受载确认窗（启动延迟主体）", 0.86)
    for tau, lab, col in ((base.tau_c_fast_s, f"τc_fast={base.tau_c_fast_s:g}s（x1 收敛 τ，长尾）", "#8e44ad"),
                          (base.tau_r_slow_idle_s, f"τr_slow_idle={base.tau_r_slow_idle_s:g}s（空载 x2 泄放）", "#2980b9")):
        a1.axvline(t0 + tau, color=col, lw=1.0, ls=":")
        a1.text(t0 + tau, _ytop(a1) * 0.72, lab, rotation=90, fontsize=8,
                color=col, va="top", ha="right")
    if t_x2 is not None:
        a1.axvline(t_x2, color="#1e8449", lw=1.6)
        a1.annotate(f"x2 实际启动 t={t_x2:.1f}s\n（加载后 {t_x2-t0:.1f}s）",
                    xy=(t_x2, out[int(np.searchsorted(t, t_x2))]),
                    xytext=(10, -30), textcoords="offset points",
                    fontsize=9.5, color="#1e8449",
                    arrowprops=dict(arrowstyle="->", color="#1e8449"))
    a1.set_ylabel("总量 (ADC)")
    a1.set_title(f"{sess}：K9 时间参数地图（各窗口按加载沿对齐；r_fast=0.06）",
                 loc="left", fontsize=11)
    a1.legend(loc="lower right", fontsize=9)
    a1.grid(alpha=0.25)

    # ── 中图：x1/x2 分解 + slow_confirm 扫描的 x2 ──
    a2.plot(t, x1s, color="#e67e22", lw=1.3, label="x1 快态合计")
    a2.plot(t, x2s, color="#27ae60", lw=1.6, label="x2 慢态合计（slow_confirm=10s 现役）")
    colors = ("#16a085", "#138d75", "#0b4f42")
    for (sc, a2c, ts), col in zip(scan[1:], colors):
        a2.plot(t, a2c, lw=1.1, ls="--", color=col,
                label=f"x2（slow_confirm={sc:g}s → 启动 {'' if ts is None else f'{ts-t0:.1f}s'}）")
    a2.axvline(t0, color="#e67e22", lw=1.2, ls="--")
    a2.set_ylabel("扣除量 (ADC)")
    a2.set_title("x2 启动提前量：slow_confirm_s 10→6→4→2（其他参数不动；只影响启动时刻，不影响每通道是否有蠕变）",
                 loc="left", fontsize=10)
    a2.legend(loc="upper left", fontsize=8.5)
    a2.grid(alpha=0.25)

    # ── 底图：时间参数总表 ──
    a3.axis("off")
    cols = ["参数", "现值(s)", "管什么", "与 x2 启动的关系"]
    rows = [[n, f"{v:g}", d, r] for (n, v, d, r) in TIME_PARAMS]
    tbl = a3.table(cellText=rows, colLabels=cols, loc="center",
                   cellLoc="left", colWidths=(0.17, 0.06, 0.44, 0.33))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#dfe6e9"); cell.set_text_props(weight="bold")
        elif rows[r - 1][0] in ("slow_confirm_s", "soft_unfreeze_s"):
            cell.set_facecolor("#fdedec")     # 与 x2 启动直接相关：淡红
        elif r % 2 == 0:
            cell.set_facecolor("#f8f9fa")

    out_png = HERE / "out" / "K9_时间参数地图.png"
    fig.savefig(out_png, dpi=125)
    print(f"\nx2 启动时刻（r_fast=0.06）：")
    for sc, _, ts in scan:
        print(f"  slow_confirm_s={sc:4.1f}s → x2 启动 {('t='+f'{ts:.1f}s' if ts else '未启动')}"
              f"（加载后 {f'{ts-t0:.1f}s' if ts else '—'}）")
    print(f"图已保存：{out_png}")


if __name__ == "__main__":
    main()
