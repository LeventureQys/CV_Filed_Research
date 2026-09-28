# -*- coding: utf-8 -*-
"""142_force_plot：力值模式终版对照图（6 面板）。

① 输入（pre 流）总力值 + 两段弹性电平 E_pl
② 1d925c 显示：现役 vs 推荐B vs 备选A（1 s 中值）
③ 03225d 显示（另一真实施力会话，算法关 → 输入即未补偿显示）
④ 恒压合成 17 N（纯阶跃、零蠕变）：量「无蠕变可扣时被扣掉多少」＝伪影代价
⑤ ADC 会话 3efae9 显示（同一套寄存器换到 ADC 显示）
⑥ 兑换律：cap×rf 网格上的「落点（上漂）× 下坠（回调）」Pareto 曲线
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")))
from importlib import import_module  # noqa: E402

from dptool.figure_export import export_small_multiples  # noqa: E402
from dptool.snapshot import CurveSnapshot, FigureSnapshot, PanelSnapshot  # noqa: E402

obs = import_module("91_observer")
fl = import_module("122_force_lib")
c128 = import_module("128_force_cases")
OUT7 = TEMP / "palm7" / "out"

CUR = dict(fl.REC8)
B = {**fl.REC8, "r_fast": 0.05, "tau_c_fast_s": 2.0, "slope_cap_frac": 0.009}
A = {**fl.REC8, "r_fast": 0.04, "tau_c_fast_s": 2.0, "slope_cap_frac": 0.009}
GREY, RED, GREEN, BLUE, ORANGE = "#7f7f7f", "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cases = {c["name"]: c for c in c128.build()}
    panels = []

    # ① 输入
    c = cases["hold_1d925c"]
    t, tin = c["t"], c["V"].sum(1)
    segs = c["segs"]
    seg_curves = [CurveSnapshot(f"E_pl 段{k+1}={s['E']:.2f} N",
                                np.array([s["t0"], s["t1"]]), np.array([s["E"]] * 2),
                                BLUE, 1.2, "dash") for k, s in enumerate(segs)]
    panels.append(PanelSnapshot(
        "① 输入（算法输入流 pre）：总力值 N",
        "加载是 ~0.45 s 斜坡而非阶跃；两段保压各 14.5/14.1 s，输入自身还上爬 +3.46/+3.57 N",
        "经过时间 (s)", "总力值 (N)",
        [CurveSnapshot("输入", t, tin, GREY, 1.0)] + seg_curves))

    # ② 1d925c 显示
    def disp(p, t, V):
        fps = (len(t) - 1) / (t[-1] - t[0])
        r = obs.run(t, V, obs.default_with(p))
        return fl.medfilt1s(r["out_tot"], fps), r

    y_cur, r_cur = disp(CUR, t, c["V"])
    y_b, r_b = disp(B, t, c["V"])
    y_a, r_a = disp(A, t, c["V"])
    fps = (len(t) - 1) / (t[-1] - t[0])
    ra = fl.evaluate(r_b["out_tot"], t, segs, fps)
    rc = fl.evaluate(r_cur["out_tot"], t, segs, fps)
    note = ("；".join(f"段{k+1}: 现役 上漂{rc[k]['dev']:+.2f}/回调{rc[k]['drop']:.2f} → "
                      f"推荐B 上漂{ra[k]['dev']:+.2f}/回调{ra[k]['drop']:.2f} N"
                      for k in range(len(segs))))
    panels.append(PanelSnapshot(
        "② 1d925c 显示总值（1 s 中值）", note, "经过时间 (s)", "总力值 (N)",
        [CurveSnapshot("现役 rf.03 τc1 40 cap.011", t, y_cur, RED, 1.6),
         CurveSnapshot("推荐B rf.05 τc1 2 cap.009", t, y_b, GREEN, 1.8),
         CurveSnapshot("备选A rf.04 τc1 2 cap.009", t, y_a, ORANGE, 1.2)] + seg_curves))

    # ③ 03225d
    c3 = cases["in_03225d"]
    t3, V3 = c3["t"], c3["V"]
    fps3 = (len(t3) - 1) / (t3[-1] - t3[0])
    y3_cur = fl.medfilt1s(obs.run(t3, V3, obs.default_with(CUR))["out_tot"], fps3)
    y3_b = fl.medfilt1s(obs.run(t3, V3, obs.default_with(B))["out_tot"], fps3)
    s3 = c3["segs"][0]
    panels.append(PanelSnapshot(
        "③ 另一真实力值会话 03225d（算法关录制 → 未补偿显示）",
        f"输入 33 s 上爬 +{float(V3.sum(1)[s3['i1']]) - s3['E']:.2f} N；"
        "现役末段显示仍以 +0.017 N/s 上漂，推荐B 收到 +0.011 N/s",
        "经过时间 (s)", "总力值 (N)",
        [CurveSnapshot("输入", t3, V3.sum(1), GREY, 1.0),
         CurveSnapshot("现役", t3, y3_cur, RED, 1.6),
         CurveSnapshot("推荐B", t3, y3_b, GREEN, 1.8),
         CurveSnapshot(f"E_pl={s3['E']:.2f} N",
                       np.array([s3["t0"], s3["t1"]]), np.array([s3["E"]] * 2), BLUE, 1.2, "dash")]))

    # ④ 恒压合成（零蠕变）
    cf = cases["synth_flat17"]
    tf, Vf = cf["t"], cf["V"]
    yf_cur = fl.medfilt1s(obs.run(tf, Vf, obs.default_with(CUR))["out_tot"], 100.0)
    yf_b = fl.medfilt1s(obs.run(tf, Vf, obs.default_with(B))["out_tot"], 100.0)
    yf_c2 = fl.medfilt1s(obs.run(tf, Vf, obs.default_with({**CUR, "slope_cap_frac": 0.002}))["out_tot"], 100.0)
    panels.append(PanelSnapshot(
        "④ 代价台：合成恒压 17 N、零蠕变（量「无蠕变可扣时被扣掉多少」）",
        "扣除全部来自「台阶尾被当成慢漂移积分」的伪影：现役 −1.52 N(8.9%)、推荐B −2.00 N(11.8%)、"
        "cap.002 −0.57 N(3.3%)。真实手掌按压都有 23~50% 蠕变，此台为最坏边界",
        "经过时间 (s)", "总力值 (N)",
        [CurveSnapshot("输入 17 N", tf, Vf.sum(1), GREY, 1.0),
         CurveSnapshot("现役", tf, yf_cur, RED, 1.6),
         CurveSnapshot("推荐B", tf, yf_b, GREEN, 1.8),
         CurveSnapshot("cap=0.002（伪影最小）", tf, yf_c2, BLUE, 1.2, "dash")]))

    # ⑤ ADC 会话
    z = np.load(TEMP / "palm6" / "out" / "streams.npz")
    ta, pa = z["t"], z["pre"]
    fa = (len(ta) - 1) / (ta[-1] - ta[0])
    ya_cur = fl.medfilt1s(obs.run(ta, pa, obs.default_with(CUR))["out_tot"], fa) / 1000.0
    ya_b = fl.medfilt1s(obs.run(ta, pa, obs.default_with(B))["out_tot"], fa) / 1000.0
    panels.append(PanelSnapshot(
        "⑤ 同一套寄存器换到 ADC 显示（会话 3efae9，每通道 e 远大于 1 → cap/rsm 变比例语义）",
        "落点（段末−E）：现役 +973 ADC、推荐B +686 ADC（更贴）；下坠 259→279 ADC",
        "经过时间 (s)", "总值 (kN，1 N=1000 ADC)",
        [CurveSnapshot("输入", ta, pa.sum(1) / 1000.0, GREY, 1.0),
         CurveSnapshot("现役", ta, ya_cur, RED, 1.6),
         CurveSnapshot("推荐B", ta, ya_b, GREEN, 1.8)]))

    # ⑥ 兑换律
    matrix = []
    for rf in (0.03, 0.04, 0.05):
        for cap in (0.008, 0.009, 0.010, 0.011):
            p = {**fl.REC8, "r_fast": rf, "tau_c_fast_s": 2.0, "slope_cap_frac": cap}
            rr = []
            for k in ("hold_1d925c", "in_03225d"):
                cc = cases[k]
                t_, V_ = cc["t"], cc["V"]
                f_ = (len(t_) - 1) / (t_[-1] - t_[0])
                rr += fl.evaluate(obs.run(t_, V_, obs.default_with(p))["out_tot"],
                                  t_, cc["segs"], f_)
            matrix.append((max(abs(x["dev"]) for x in rr), max(x["drop"] for x in rr), rf, cap))
    matrix.sort()
    front = []
    best = 1e9
    for up, dn, rf, cap in matrix:
        if dn < best - 1e-9:
            best = dn
            front.append((up, dn, rf, cap))
    panels.append(PanelSnapshot(
        "⑥ 取舍曲线：cap×rf 网格（τc1=2）上「上漂落点」与「回调下坠」的 Pareto 前沿",
        "两者在 8 个旋钮内不可兼得：把回调压到 0.4 N 需容忍 +1.0 N 上漂；"
        "把落点压到 0.75 N 需容忍 0.7 N 回调。荐点 = rf.05/cap.009",
        "max|落点| 上漂 (N)", "max 下坠 回调 (N)",
        [CurveSnapshot("Pareto 前沿", np.array([p[0] for p in front]),
                       np.array([p[1] for p in front]), BLUE, 1.8),
         CurveSnapshot("全部网格点", np.array([m[0] for m in matrix]),
                       np.array([m[1] for m in matrix]), GREY, 0.0, "dot")]))

    snap = FigureSnapshot(
        suptitle="手掌 · 力值 N 显示 · v3.4 观测器 8 参数整定：现役 vs 推荐（离线复算，parity ≤0.01 N）",
        panels=panels, ncol=2, width_in=15.4, panel_height_in=3.1, dpi=110,
        legend_panel=1, max_points_per_curve=9000)
    out = TEMP / "figures" / "手掌_力值模式_8参数终版对照.png"
    r = export_small_multiples(snap, str(out), check=True)
    print("check:", r.get("check"), "bytes:", r.get("bytes"))
    print("path:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
