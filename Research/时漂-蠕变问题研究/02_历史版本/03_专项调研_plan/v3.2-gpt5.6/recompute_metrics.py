# -*- coding: utf-8 -*-
"""重算「零基线-反复增减同一负载」现场录制的沿/台阶/形状指标（只读 CSV，只写本目录 CSV）。

数据：temp/算法数据&原始数据/working/零基线-反复增减同一负载/20260919_160854_single_device_7b3977
      （session.json 的 algorithm.params.param_set = "plan-v3.0 PCT"，即现场跑的不是 plan-v3.4）
口径：总量 = 21 通道之和；ded = 算法输入(pre) − 现场显示(seg000)
输出：event_evidence_raw.csv / shape_15s_vs_245s_raw.csv / edge_zoom_245s_raw.csv
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DS = os.path.join(REPO, "temp", "算法数据&原始数据", "working",
                  "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")
EDGE_THR = 1200.0
GAP_S = 0.40


def load(path):
    ts, el, v = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 24:
                continue
            try:
                t = float(row[0])
                e = float(row[1])
                ch = [float(x) for x in row[3:24]]
            except ValueError:
                continue
            ts.append(t)
            el.append(e)
            v.append(ch)
    return np.array(ts), np.array(el), np.array(v, dtype=float)


def main():
    _, el, v_in = load(os.path.join(DS, "device_001_pre_seg0.csv"))
    _, el_o, v_out = load(os.path.join(DS, "device_001_seg000.csv"))
    n = min(len(el), len(el_o))
    el, v_in, v_out = el[:n], v_in[:n], v_out[:n]
    s_in = v_in.sum(axis=1)
    s_out = v_out.sum(axis=1)
    ded = s_in - s_out

    d_in = np.diff(s_in)
    d_out = np.diff(s_out)
    idx = np.where(np.abs(d_in) > EDGE_THR)[0] + 1

    # 把相邻（<GAP_S）的候选帧归并为一次沿，取 |d_in| 最大的那一帧为「最大单帧台阶」
    groups, cur = [], [idx[0]]
    for k in idx[1:]:
        if el[k] - el[cur[-1]] <= GAP_S:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)

    rows = []
    for g in groups:
        kbest = max(g, key=lambda k: abs(d_in[k - 1]))
        t = el[kbest]
        pre = s_in[max(0, kbest - 5):kbest].mean()
        post = s_in[kbest:kbest + 5].mean()
        win = (el >= t + 2.5) & (el <= t + 8.0)
        lvl = (el >= t + 5.0) & (el <= t + 20.0)
        rows.append({
            "t_edge_s": round(float(t), 2),
            "n_hit_frames": len(g),
            "dir": "up" if d_in[kbest - 1] > 0 else "dn",
            "in_span_s": "%0.2f~%0.2f" % (el[g[0]], el[g[-1]]),
            "t_in_pre": round(float(pre), 0),
            "t_in_post": round(float(post), 0),
            "t_in_step_avg": round(float(post - pre), 0),
            "in_dmax": round(float(d_in[kbest - 1]), 0),
            "out_dmax": round(float(d_out[kbest - 1]), 0),
            "in_dmin_signed": round(float(d_in[g][np.argmin(d_in[g])]), 0),
            "out_dmin_signed": round(float(d_out[g][np.argmin(d_in[g])]), 0),
            "in_settled_2.5_8s": round(float(s_in[win].mean()), 0) if win.any() else "",
            "out_settled_2.5_8s": round(float(s_out[win].mean()), 0) if win.any() else "",
            "ded_settled_2.5_8s": round(float(ded[win].mean()), 0) if win.any() else "",
            "in_level_5_20s": round(float(s_in[lvl].mean()), 0) if lvl.any() else "",
            "out_level_5_20s": round(float(s_out[lvl].mean()), 0) if lvl.any() else "",
            "ded_level_5_20s": round(float(ded[lvl].mean()), 0) if lvl.any() else "",
            "out_over_in_pct": round(float(100.0 * s_out[win].mean() / s_in[win].mean()), 2) if win.any() else "",
        })

    with open(os.path.join(HERE, "event_evidence_raw.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 形状对齐：沿后各时刻增量 / 本段落定增量
    grid = [0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0]
    sh = []
    for label, t0 in (("15.18s", 15.18), ("63.41s", 63.41), ("245.27s", 245.27)):
        i0 = int(np.argmin(np.abs(el - (t0 - 0.5))))
        b_in = s_in[max(0, i0 - 5):i0].mean()
        b_out = s_out[max(0, i0 - 5):i0].mean()
        j20 = int(np.argmin(np.abs(el - (t0 + 20.0))))
        inc = s_in[j20 - 10:j20 + 10].mean() - b_in
        rec = {"edge": label, "base_in": round(float(b_in), 0), "base_out": round(float(b_out), 0),
               "inc_ref": round(float(inc), 0)}
        for gt in grid:
            k = int(np.argmin(np.abs(el - (t0 + gt))))
            rec["in_%gs" % gt] = round(float((s_in[k] - b_in) / inc), 3)
            rec["out_%gs" % gt] = round(float((s_out[k] - b_out) / inc), 3)
        sh.append(rec)

    with open(os.path.join(HERE, "shape_15s_vs_245s_raw.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sh[0].keys()))
        w.writeheader()
        w.writerows(sh)

    # 245 s 附近逐帧明细
    m = (el >= 244.6) & (el <= 246.4)
    with open(os.path.join(HERE, "edge_zoom_245s_raw.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "d_t", "sum_in", "sum_out", "ded", "d_in", "d_out"])
        prev = None
        for i in np.where(m)[0]:
            w.writerow([round(float(el[i]), 3), round(float(el[i] - el[i - 1]), 4),
                        round(float(s_in[i]), 0), round(float(s_out[i]), 0),
                        round(float(ded[i]), 0),
                        "" if prev is None else round(float(s_in[i] - s_in[i - 1]), 0),
                        "" if prev is None else round(float(s_out[i] - s_out[i - 1]), 0)])
            prev = i

    # 交付用汇总表：事件级证据（现场流重算 + 项目侧回放引用值）
    def win(t0, t1, s):
        m = (el >= t0) & (el <= t1)
        return round(float(s[m].mean()), 0) if m.any() else ""

    ev_rows = [
        # 项目侧回放/报告引用值（臂见 src 列），现场列为本目录重算
        dict(id="E_offload_240", t_edge_s=240.0, kind="Decrease(整片卸载)", arm="现场 plan-v3.0 PCT",
             in_settled=win(240.5, 242, s_in), out_settled=win(240.5, 242, s_out),
             ded=win(240.5, 242, s_in) - win(240.5, 242, s_out),
             ref_in="", ref_out="", ref_ded="", step_in="", step_out="", src="本目录 recompute_metrics.py"),
        dict(id="E1_first_load", t_edge_s=3.3, kind="Onset(首次加载)", arm="现场 plan-v3.0 PCT",
             in_settled=win(6, 10, s_in), out_settled=win(6, 10, s_out),
             ded=win(6, 10, s_in) - win(6, 10, s_out), ref_in="", ref_out="", ref_ded="",
             step_in="", step_out="", src="本目录 recompute_metrics.py"),
        dict(id="E2", t_edge_s=15.18, kind="Restep(+1 砝码)", arm="现场 plan-v3.0 PCT",
             in_settled=win(17.6, 23, s_in), out_settled=win(17.6, 23, s_out),
             ded=win(17.6, 23, s_in) - win(17.6, 23, s_out),
             ref_in=16844, ref_out=15219, ref_ded=1625, step_in=2606, step_out="",
             src="现场=本目录重算；引用值=plan/v3.6-v41flash/results/v36_ab_final.txt（臂 v36）"),
        dict(id="E3", t_edge_s=41.65, kind="Restep(卸载→重载)", arm="现场 plan-v3.0 PCT",
             in_settled=win(44.2, 50, s_in), out_settled=win(44.2, 50, s_out),
             ded=win(44.2, 50, s_in) - win(44.2, 50, s_out),
             ref_in=17521, ref_out=15641, ref_ded=1880, step_in="", step_out="",
             src="现场=本目录重算；引用值=plan/v3.6-v41flash/分析报告_245s丢基线.md §1（臂 v31）"),
        dict(id="E4", t_edge_s=63.41, kind="Restep(卸载→重载)", arm="现场 plan-v3.0 PCT",
             in_settled=win(66, 72, s_in), out_settled=win(66, 72, s_out),
             ded=win(66, 72, s_in) - win(66, 72, s_out),
             ref_in=17609, ref_out=15425, ref_ded=2184, step_in="", step_out="",
             src="现场=本目录重算；引用值=plan/v3.6-v41flash/分析报告_245s丢基线.md §1（臂 v31）"),
        dict(id="E5", t_edge_s=232.23, kind="Restep(长保压后加重)", arm="现场 plan-v3.0 PCT",
             in_settled=win(234.8, 239, s_in), out_settled=win(234.8, 239, s_out),
             ded=win(234.8, 239, s_in) - win(234.8, 239, s_out),
             ref_in=17556, ref_out=15299, ref_ded=2258, step_in="", step_out="",
             src="现场=本目录重算；引用值=plan/v3.6-v41flash/分析报告_245s丢基线.md §1（臂 v31）"),
        dict(id="E6_reload_245", t_edge_s=245.27, kind="Onset(整片卸载后重载)", arm="现场 plan-v3.0 PCT",
             in_settled=win(247.8, 253, s_in), out_settled=win(247.8, 253, s_out),
             ded=win(247.8, 253, s_in) - win(247.8, 253, s_out),
             ref_in=17255, ref_out=17237, ref_ded=18, step_in=12639, step_out=12668,
             src="现场=本目录重算（台阶=0.5 s 分辨率口径）；引用值=plan/v3.6-v41flash/分析报告_245s丢基线.md §1（臂 v31）"),
        dict(id="E7", t_edge_s=258.26, kind="Onset(整片卸载后重载)", arm="现场 plan-v3.0 PCT",
             in_settled=win(260.8, 266, s_in), out_settled=win(260.8, 266, s_out),
             ded=win(260.8, 266, s_in) - win(260.8, 266, s_out),
             ref_in=17659, ref_out=17601, ref_ded=58, step_in="", step_out="",
             src="现场=本目录重算；引用值=plan/v3.6-v41flash/分析报告_245s丢基线.md §1（臂 v31）"),
        dict(id="E8", t_edge_s=278.53, kind="Onset(整片卸载后重载)", arm="现场 plan-v3.0 PCT",
             in_settled=win(281, 287, s_in), out_settled=win(281, 287, s_out),
             ded=win(281, 287, s_in) - win(281, 287, s_out),
             ref_in=17693, ref_out=17778, ref_ded=-85, step_in="", step_out="",
             src="现场=本目录重算；引用值=plan/v3.6-v41flash/分析报告_245s丢基线.md §1（臂 v31）"),
        dict(id="E9", t_edge_s=300.53, kind="Onset(整片卸载后重载)", arm="现场 plan-v3.0 PCT",
             in_settled=win(303, 309, s_in), out_settled=win(303, 309, s_out),
             ded=win(303, 309, s_in) - win(303, 309, s_out),
             ref_in=17371, ref_out=17393, ref_ded=-22, step_in="", step_out="",
             src="现场=本目录重算；引用值=plan/v3.6-v41flash/分析报告_245s丢基线.md §1（臂 v31）"),
        # v3.4 臂（真实 C++ 本体离线回放）
        dict(id="V34_handoff_245", t_edge_s=245.31, kind="Onset 交接后", arm="plan-v3.4 (mem ON)",
             in_settled=17268, out_settled=15278, ded=2009, ref_in=15234, ref_out=15234,
             ref_ded=2009, step_in="", step_out="",
             src="plan/v3.4/results/v34_replay_mem_on.txt:30（A=15234 ded=2009）"),
        dict(id="V34_pre_edge_239", t_edge_s=239.05, kind="卸载沿前（同一回放）", arm="plan-v3.4 (mem ON)",
             in_settled=17610, out_settled=15264, ded=2295, ref_in=15188, ref_out=15188,
             ref_ded=2295, step_in="", step_out="",
             src="plan/v3.4/results/v34_replay_mem_on.txt:13（A=15188 ded=2295）"),
        dict(id="V34_snapshot_240", t_edge_s=240.08, kind="Decrease 检测当帧快照", arm="plan-v3.4 (mem ON)",
             in_settled=1841, out_settled=1171, ded=2030, ref_in=15188, ref_out=15188,
             ref_ded=2030, step_in="", step_out="",
             src="plan/v3.4/results/v34_replay_mem_on.txt:16（A=15188 ded=2030）→ 快照比沿前低 265 ADC"),
        dict(id="V34_familyA_15s", t_edge_s=232.23, kind="Restep 交接后", arm="plan-v3.4 (mem ON)",
             in_settled="", out_settled="", ded=2382, ref_in=15188, ref_out=15188,
             ref_ded=2382, step_in="", step_out="",
             src="plan/v3.4/results/v34_replay_mem_on.txt:9-10"),
        dict(id="V32_ded_collapse_248", t_edge_s=248.8, kind="交接当帧（v3.2 seed 缺陷）", arm="plan-v3.2 (--seed-gain 1.0)",
             in_settled="", out_settled="", ded=-81, ref_in=14680, ref_out=14680,
             ref_ded=-81, step_in="", step_out="",
             src="plan/v3.6-v41flash/分析报告_245s丢基线.md §4（248.60 s +2298 → 248.80 s −81）"),
        dict(id="V36_handoff_245", t_edge_s=245.31, kind="Onset 交接后", arm="plan-v3.6 (seed-gain 1.0, v36_d 400)",
             in_settled=17288, out_settled="", ded=2818, ref_in=14621, ref_out="",
             ref_ded=2818, step_in="", step_out="",
             src="plan/v3.6-v41flash/results/v36_handoff_check.txt:8（A_ho=14621, ded=2818）"),
    ]
    with open(os.path.join(HERE, "事件证据.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "t_edge_s", "kind", "arm", "in_settled", "out_settled", "ded",
            "step_in", "step_out", "ref_in", "ref_out", "ref_ded", "src"])
        w.writeheader()
        w.writerows(ev_rows)

    print("frames=%d  edges=%d  max_t=%.1fs" % (n, len(rows), el[-1]))
    print("written: event_evidence_raw.csv, shape_15s_vs_245s_raw.csv, edge_zoom_245s_raw.csv, 事件证据.csv")


if __name__ == "__main__":
    main()
