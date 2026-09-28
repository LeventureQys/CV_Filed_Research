# -*- coding: utf-8 -*-
"""T3 步骤4：汇总表（读 results/t3_edges.csv）。

输出 results/t3_summary.txt（人读）与 results/t3_summary.json（出图/报告用）。
分层：事件类（首次大台阶 / 受载态小台阶 / 其他）× 保压稳定性（in_rng ≤ 0.05 视为静载保压）。
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

ARMS = ["base", "A_now_a1.0", "A_lag_a1.0", "A_now_a0.8", "A_now_a0.6",
        "B_tb1.0_1.5s", "B_tb2.0_2.0s", "C_a0.8_tb2.0_1.5s",
        "C_a1.0_tb2.0_1.5s", "C_a0.8_lag_tb2.0_1.5s"]
CLASSES = ["首次大台阶", "受载态小台阶", "其他"]
FLOATS = ("t", "step", "hold", "peak", "settle", "settle_base", "lvl_dev",
          "fall", "fall5", "dip", "in_rng", "in_step", "dev_min", "dev_max",
          "t5", "t_peak")


def load():
    with open(os.path.join(OUT, "t3_edges.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in FLOATS:
            r[k] = float(r[k])
    return rows


def med(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return float(np.median(v)) if v else float("nan")


def stat_block(rows):
    out = {}
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        if not rr:
            continue
        t5 = [r["t5"] for r in rr if r["t5"] >= 0]
        out[arm] = dict(
            n=len(rr),
            fall=med([r["fall"] for r in rr]),
            fall_p90=float(np.percentile([r["fall"] for r in rr], 90)),
            fall5=med([r["fall5"] for r in rr]),
            dip=med([r["dip"] for r in rr]),
            dip_p90=float(np.percentile([r["dip"] for r in rr], 90)),
            dip_max=float(np.max([r["dip"] for r in rr])),
            t5=med(t5), n_t5=len(t5),
            t5_p90=(float(np.percentile(t5, 90)) if t5 else float("nan")),
            lvl=med([r["lvl_dev"] for r in rr]),
            lvl_absmax=float(np.max(np.abs([r["lvl_dev"] for r in rr]))),
            dev_min=med([r["dev_min"] for r in rr]),
            dev_max=med([r["dev_max"] for r in rr]))
    return out


def fmt_table(block, title, lines):
    lines.append("")
    lines.append("== %s" % title)
    lines.append("   臂                    N | 回落fall 中位(p90) | 5s内回落 | 自回弹下冲 中位(p90/最大) | T±5% 中位(p90,n) | 电平偏置 中位 | 与基线波形差 中位(min/max)")
    base = block.get("base")
    for arm in ARMS:
        d = block.get(arm)
        if not d:
            continue
        mark = ""
        if base and arm != "base":
            mark = "  Δ回落%+6.0f Δ下冲%+6.0f" % (d["fall"] - base["fall"],
                                                d["dip"] - base["dip"])
        lines.append("   %-20s %3d | %7.0f (%7.0f) | %7.0f | %7.0f (%6.0f/%6.0f) | %6.1f (%6.1f,%3d) | %+7.0f | %+7.0f/%7.0f%s"
                     % (arm, d["n"], d["fall"], d["fall_p90"], d["fall5"],
                        d["dip"], d["dip_p90"], d["dip_max"],
                        d["t5"], d["t5_p90"], d["n_t5"], d["lvl"],
                        d["dev_min"], d["dev_max"], mark))


def main():
    rows = load()
    lines = []
    P = lines.append
    P("T3 前馈汇总（数据：results/t3_edges.csv，%d 条事件×臂记录）" % len(rows))
    P("口径（显示总量 = 21 通道之和，ADC；形状指标在 0.3 s 滑动均值上计算）：")
    P("  fall   回落幅度 = 沿后显示峰值 − 段末稳定电平（段末 2 s 中位）  ← 用户主诉「显示回落过大」")
    P("  fall5  沿后 5 s 内回落 = 峰值 − 峰后 5 s 内最小显示")
    P("  dip    自回弹下冲 = max_t( y[t] − min_{s≤t} y[s] )：先下探再回升的幅度（单调下降=0；过扣自愈>0）")
    P("  t5     从沿起进入并保持 settle±5%·台阶 的时刻（稳定时间）")
    P("  lvl    该臂段末电平 − 基线段末电平（负=比基线扣得更多）")
    sess = sorted(set(r["session"] for r in rows))
    n_edge = len([r for r in rows if r["arm"] == "base"])
    P("会话 %d 个，评估沿合计 %d 个（base 臂）" % (len(sess), n_edge))
    json_out = {"n_sessions": len(sess), "n_edges": n_edge, "blocks": {}}

    blk = stat_block(rows)
    fmt_table(blk, "全部评估沿（含非静载保压）", lines)
    json_out["blocks"]["all"] = blk

    quiet = [r for r in rows if r["in_step"] <= 0.15]
    blk_q = stat_block(quiet)
    fmt_table(blk_q, "无二次加载保压子集（保压中沿后 2 s 起 1 s 内输入上升 ≤15%%·台阶，共 %d 事件）"
              % (len(quiet) // len(ARMS)), lines)
    json_out["blocks"]["clean"] = blk_q

    for cls in CLASSES:
        sub = [r for r in rows if r["cls"] == cls]
        if not sub:
            continue
        b = stat_block(sub)
        fmt_table(b, "事件类：%s（%d 事件）" % (cls, len(sub) // len(ARMS)), lines)
        json_out["blocks"]["cls_" + cls] = b
        subq = [r for r in sub if r["in_step"] <= 0.15]
        if subq:
            bq = stat_block(subq)
            fmt_table(bq, "事件类 ∩ 无二次加载：%s（%d 事件）"
                      % (cls, len(subq) // len(ARMS)), lines)
            json_out["blocks"]["cls_" + cls + "_clean"] = bq

    P("")
    P("== 逐事件差异分布（同一沿：该臂 − 基线）——「前馈是否加重回落」的直接判据")
    P("   臂                   类        N | Δ回落 中位[p10,p90] 加重比例 | Δ下冲 中位[p10,p90] | Δ电平 中位[p10,p90] |Δ电平|>200 比例")
    by2 = {}
    for r in rows:
        by2.setdefault((r["session"], round(r["t"], 2)), {})[r["arm"]] = r
    for arm in ARMS[1:]:
        for cls in ["全部"] + CLASSES:
            df, dd, dl = [], [], []
            for k, v in by2.items():
                if "base" not in v or arm not in v:
                    continue
                b, a = v["base"], v[arm]
                if cls != "全部" and b["cls"] != cls:
                    continue
                df.append(a["fall"] - b["fall"])
                dd.append(a["dip"] - b["dip"])
                dl.append(a["lvl_dev"])
            if not df:
                continue
            worse = 100.0 * np.mean([x > 20.0 for x in df])
            big = 100.0 * np.mean([abs(x) > 200.0 for x in dl])
            P("   %-20s %-8s %3d | %+7.0f [%+7.0f,%+7.0f] %5.0f%% | %+6.0f [%+6.0f,%+6.0f] | %+7.0f [%+7.0f,%+7.0f] | %5.0f%%"
              % (arm, cls, len(df), np.median(df),
                 np.percentile(df, 10), np.percentile(df, 90), worse,
                 np.median(dd), np.percentile(dd, 10), np.percentile(dd, 90),
                 np.median(dl), np.percentile(dl, 10), np.percentile(dl, 90), big))

    P("")
    P("== 逐事件对照：base → A_now_a1.0 / A_lag_a1.0 / C_a0.8_tb2.0_1.5s")
    P("   格式：回落/下冲/T5/电平偏置；类 简称（大=首次大台阶，小=受载态小台阶，他=其他）")
    P("   会话                                   类   t     台阶  保压  静载 | base 回落/下冲/T5 | A_now | A_lag | C_a0.8")
    by = {}
    for r in rows:
        by[(r["session"], round(r["t"], 2), r["arm"])] = r
    keys = sorted(set((r["session"], round(r["t"], 2)) for r in rows))
    for sess_k, t_k in keys:
        b = by.get((sess_k, t_k, "base"))
        if not b:
            continue
        cells = []
        for arm in ("A_now_a1.0", "A_lag_a1.0", "C_a0.8_tb2.0_1.5s"):
            a = by.get((sess_k, t_k, arm))
            cells.append(("%6.0f/%5.0f/%5.1f/%+5.0f"
                          % (a["fall"], a["dip"], a["t5"], a["lvl_dev"])) if a
                         else " " * 26)
        P("   %-38s %-3s %6.1f %6.0f %5.1fs %s | %6.0f/%5.0f/%5.1f | %s | %s | %s"
          % (sess_k[-38:], b["cls"][0], b["t"], b["step"], b["hold"],
             "净" if b["in_step"] <= 0.15 else "扰",
             b["fall"], b["dip"], b["t5"], cells[0], cells[1], cells[2]))

    with open(os.path.join(OUT, "t3_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "t3_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(json_out, fh, ensure_ascii=False, indent=1)
    print("\n".join(lines[:120]))


if __name__ == "__main__":
    main()
