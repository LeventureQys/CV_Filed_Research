# -*- coding: utf-8 -*-
"""K 系预留池聚焦对比（多进程并行）：base / H3 / K5(base+hold) / K6(H3+hold)。
会话级并行（ProcessPoolExecutor），并输出指标摘要 + 对比图。"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.realpath(os.path.join(HERE, *[".."] * 9)),
                                "toolbox", "数据解析工具"))

VARIANTS = {}
import fall_study_run as fs  # noqa: E402
VARIANTS = {k: fs.VARIANTS[k] for k in ("base", "H3_F12_w4", "K5_base+holdA2", "K6_H3+holdA2")}


def run_session(item):
    cond, root = item
    from dptool.session_csv import load_table
    t = load_table(os.path.join(root, "device_001_pre_seg0.csv"), use_cache=False)
    ch = np.asarray(t.channels(), dtype=np.float64)
    ch = ch[:, np.isfinite(ch).all(axis=0)]
    ts = t.time_axis("frame_index") / fs.FPS
    tot_in = fs.medfilt(ch.sum(axis=1))
    events, _ = fs.find_events(ts, tot_in)
    rows = []
    for vname, vp in VARIANTS.items():
        D, S1, S2 = fs.observe(ts, ch, vp)
        tot = fs.medfilt(D.sum(axis=1))
        for i in events:
            m = fs.event_metrics(ts, tot, i, S1, S2, tot_in)
            if m is None:
                continue
            rows.append((cond, float(ts[i]), vname, float(m[0]), float(m[1]),
                         float(m[2].get("din", np.nan)), float(m[2].get("t_set", np.nan))))
    return rows


def main():
    sessions = fs.find_sessions()
    with ProcessPoolExecutor(max_workers=min(8, len(sessions))) as ex:
        all_rows = [r for rows in ex.map(run_session, sessions) for r in rows]
    lines = ["K 系预留池聚焦对比（base / H3 / K5=base+hold / K6=H3+hold，限额 τ=0.5s+2ADC/s）", ""]
    for vname in VARIANTS:
        sub = [r for r in all_rows if r[2] == vname]
        if not sub:
            continue
        falls = np.array([r[4] for r in sub])
        ex_ = np.array([r[4] - r[5] for r in sub])
        tst = np.array([r[6] for r in sub])
        lines.append(f"{vname:16s} n={len(sub):2d}  fall中位={np.median(falls):7.1f}  "
                     f"excess中位={np.median(ex_):6.1f}  p90={np.percentile(ex_,90):6.1f}  "
                     f"max={ex_.max():7.1f}  t_set中位={np.nanmedian(tst):5.2f}s")
    txt = "\n".join(lines)
    with open(os.path.join(HERE, "..", "results", "k_hold_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    try:
        print(txt)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((txt + "\n").encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
