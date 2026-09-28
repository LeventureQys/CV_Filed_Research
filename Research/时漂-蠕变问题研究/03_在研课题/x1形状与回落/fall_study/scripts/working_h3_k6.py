# -*- coding: utf-8 -*-
"""working 全会话：H3 vs K6(H3+预留池) 并行对比测试。
会话级 ProcessPoolExecutor 并行；每会话输出全程 + 两个最大台阶放大面板；指标摘要落盘。"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(HERE, *[".."] * 9))
sys.path.insert(0, os.path.join(REPO, "toolbox", "数据解析工具"))
sys.path.insert(0, HERE)

from dptool.snapshot import CurveSnapshot, PanelSnapshot, FigureSnapshot  # noqa: E402
from dptool.figure_export import export_small_multiples  # noqa: E402

import fall_study_run as fs  # noqa: E402

DATA_ROOT = os.path.join(fs.DATA_ROOT, "working")
# K6 先画（红虚线）、H3 后画（橙实线），避免两线重合时 H3 被盖住
VARIANTS = [("K6_H3+holdA2", "#d62728", "K6 H3+预留池", "dash"),
            ("H3_F12_w4", "#ff7f0e", "H3", "solid")]


def run_session(item):
    cond, root = item
    from dptool.session_csv import load_table
    t = load_table(os.path.join(root, "device_001_pre_seg0.csv"), use_cache=False)
    ch = np.asarray(t.channels(), dtype=np.float64)
    ch = ch[:, np.isfinite(ch).all(axis=0)]
    ts = t.time_axis("frame_index") / fs.FPS
    tot_in = fs.medfilt(ch.sum(axis=1))
    events, _ = fs.find_events(ts, tot_in)
    out = {"cond": cond, "ts": ts, "tot_in": tot_in,
           "events": [float(ts[i]) for i in events], "tot": {}}
    for vname, _c, _l, _s in VARIANTS:
        D, S1, S2 = fs.observe(ts, ch, fs.VARIANTS[vname])
        tot = fs.medfilt(D.sum(axis=1))
        out["tot"][vname] = tot
        met = []
        for i in events:
            m = fs.event_metrics(ts, tot, i, S1, S2, tot_in)
            if m is None:
                continue
            met.append((float(ts[i]), float(m[1]), float(m[1] - m[2]["din"]),
                        float(m[2].get("t_set", np.nan))))
        out["met:" + vname] = met
        # 台阶排序用（供放大面板选点）
        steps = []
        for i in events:
            j0 = max(0, i - int(1.5 * fs.FPS))
            step = np.median(tot_in[i + int(2 * fs.FPS):i + int(4 * fs.FPS)]) - np.median(tot_in[j0:i])
            steps.append((float(step), float(ts[i])))
        out["steps"] = sorted(steps, reverse=True)
    return out


def main():
    sessions = []
    for root, _dirs, files in os.walk(DATA_ROOT):
        if "device_001_pre_seg0.csv" in files:
            sessions.append((os.path.relpath(root, DATA_ROOT), root))
    with ProcessPoolExecutor(max_workers=min(8, len(sessions))) as ex:
        results = list(ex.map(run_session, sorted(sessions)))

    lines = ["working 全会话：H3 vs K6(H3+预留池)", "excess=同窗口显示回落-输入自身回落", ""]
    panels = []
    for r in results:
        lines.append(f"[{r['cond']}] 事件数 {len(r['events'])}")
        for vname, _c, _l, _s in VARIANTS:
            met = r["met:" + vname]
            if met:
                ex_ = np.array([m[2] for m in met])
                tst = np.array([m[3] for m in met])
                lines.append(f"  {vname:16s} fall中位={np.median([m[1] for m in met]):7.1f}  "
                             f"excess中位={np.median(ex_):6.1f}  max={ex_.max():7.1f}  "
                             f"t_set中位={np.nanmedian(tst):5.2f}s")
        curves = [CurveSnapshot("输入", r["ts"], r["tot_in"], color="0.6", width=1.0)]
        for vname, color, lab, style in VARIANTS:
            curves.append(CurveSnapshot(lab, r["ts"], r["tot"][vname], color=color, width=1.4, style=style))
        panels.append(PanelSnapshot(title=f"{r['cond'][:40]} 全程", x_label="t (s)",
                                    y_label="总值 (ADC)", curves=curves,
                                    show_legend=(len(panels) == 0)))
        for step, t_ev in r["steps"][:2]:
            zoom = (r["ts"] >= t_ev - 3.0) & (r["ts"] <= t_ev + 25.0)
            panels.append(PanelSnapshot(
                title=f"放大 @{t_ev:.0f}s 台阶≈{step:.0f} ADC",
                x_label="t-t_edge (s)", y_label="总值 (ADC)",
                curves=[CurveSnapshot(c.name, r["ts"][zoom] - t_ev, c.y[zoom],
                                      color=c.color, width=c.width, style=c.style) for c in curves]))
        lines.append("")
    snap = FigureSnapshot(suptitle="working 全会话 · H3(橙) vs K6=H3+预留池(红) · 灰=输入（每会话 1 全程 + 2 放大）",
                          panels=panels, ncol=3, width_in=21.0, panel_height_in=3.6,
                          legend_panel=0, max_points_per_curve=6000)
    out_png = os.path.join(HERE, "..", "figure", "working对比_H3_K6.png")
    info = export_small_multiples(snap, out_png, check=True)
    lines.append(f"图: {out_png} ({info['bytes']/1024:.0f} KB)")
    with open(os.path.join(HERE, "..", "results", "working_H3_K6.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
