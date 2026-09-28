# -*- coding: utf-8 -*-
"""T2 步骤3：基线 vs 各「x1 自适应」变体 的统一指标回放对比。

指标口径（全部来自真实数据回放，通道和视角）：
  回落均/峰   沿后 10s 内显示峰值 − 沿后 10s 末显示中位（ADC，正=冲高后回落）
  Δ显示10s    显示(沿+10s) − 显示(沿+0s)：加载后 10s 显示净变化（负=下沉，用户主诉方向）
  保压偏差    保压稳定段 [沿+15s, 沿+30s] 内 (显示−输入低通) 均值：0 = 蠕变被完全补偿（显示持平）
  保压std     保压稳定段显示 std
  保压漂移    保压段 (Δ显示 − Δ输入低通)：0 = 显示跟随蠕变
  空载偏差    空载段 (显示−输入) 均值
  一致性      反复增减同一负载的台阶电平互差（±10%·台阶 判据）合格数
  稳时        进入 settle+2%·台阶 带且不再回升的时刻
  对录制      |显示 − 记录文件里的算法输出| 中位（参考）

事件门槛：窗口 ≥25s 且输入 std/台阶 ≤0.15（否则量的是载荷变化不是算法）。
"""
import csv
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

WIN = 60.0
OUT = os.path.join(T.OUT_ROOT, "results")
os.makedirs(OUT, exist_ok=True)

KEYS = ["drop_mean", "drop_max", "drop_rel_mean", "x1_exc_mean",
        "decline_mean", "decline_max", "tail_mean", "seg_err_mean",
        "std_mean", "seg_drift_mean", "idle", "t_settle_mean", "fid",
        "cons_worst"]


def run_variant(variant, p_over, sess, evcache, subset=False):
    rows = []
    for label, c in sess.items():
        evs = evcache[label]
        if not evs:
            continue
        r = O.observe(c["el"].copy(), c["V"], variant, p=p_over)
        ylp = T.lowpass(r["D"], c["el"], tau=1.0)
        s = T.summarize(label, r["D"], c["tot"], c["el"], evs, drec=c["drec"],
                        subset=subset, xlp=c["xlp"], ylp=ylp, x1=r["X1"])
        s["variant"] = variant
        rows.append(s)
    return rows


def macro(rows):
    out = {}
    for k in KEYS:
        v = [r[k] for r in rows if not np.isnan(r[k])]
        out[k] = float(np.mean(v)) if v else float("nan")
    out["cons_ok"] = sum(r["cons_ok"] for r in rows)
    out["cons_tot"] = sum(r["cons_tot"] for r in rows)
    out["n_ev"] = sum(r["n_ev"] for r in rows)
    return out


def load_all():
    sess, evcache = {}, {}
    for label, d in T.all_sessions():
        s = T.load_pre(d)
        el = s["el"]
        tot = T.total(s)
        evs = T.load_events(el, tot, win=WIN)
        evcache[label] = evs
        rec = T.load_recorded(d)
        drec = T.total(rec) if rec is not None else None
        sess[label] = dict(el=el, V=s["V"], tot=tot, drec=drec,
                           xlp=T.lowpass(tot, el, tau=5.0))
    return sess, evcache


def main(argv):
    only = argv[1:] if len(argv) > 1 else None
    sess, evcache = load_all()
    order = [k for k in O.VARIANTS if (not only or k in only)]
    base_rows = run_variant("base", O.VARIANTS["base"][1], sess, evcache)
    bmap = {r["tag"]: r for r in base_rows}
    bmacro = macro(base_rows)
    bmacro_clean = macro(run_variant("base", O.VARIANTS["base"][1], sess, evcache,
                                     subset=True))

    lines, allrows = [], []
    for name in order:
        variant, p_over, desc = O.VARIANTS[name]
        rows = base_rows if name == "base" else run_variant(variant, p_over, sess, evcache)
        m = macro(rows)
        mclean = (bmacro_clean if name == "base"
                  else macro(run_variant(variant, p_over, sess, evcache, subset=True)))
        allrows.append((name, desc, m, rows, mclean))
        lines.append("")
        lines.append("#" * 152)
        lines.append("变体 %-11s %s" % (name, desc))
        lines.append("#" * 152)
        lines.append(T.HEADER)
        lines.append("-" * 152)
        for r in rows:
            lines.append(T.fmt_row(r))
        lines.append("-" * 152)
        lines.append("宏平均(全事件) %-11s 回落%7.1f 峰%7.1f x1超调%+7.1f 绝对下降%7.1f "
                     "尾段爬升%+7.1f 保证段偏差%+8.1f 保压std%7.1f 保压漂%7.1f "
                     "空载%+7.1f 稳时%6.1f 一致%2d/%-2d 事件%d"
                     % (name, m["drop_mean"], m["drop_max"], m["x1_exc_mean"],
                        m["decline_mean"], m["tail_mean"], m["seg_err_mean"],
                        m["std_mean"], m["seg_drift_mean"], m["idle"],
                        m["t_settle_mean"], m["cons_ok"], m["cons_tot"], m["n_ev"]))
        lines.append("宏平均(干净保压) %-11s 回落%7.1f 峰%7.1f x1超调%+7.1f 绝对下降%7.1f "
                     "尾段爬升%+7.1f 保证段偏差%+8.1f 保压std%7.1f 保压漂%7.1f 事件%d"
                     % (name, mclean["drop_mean"], mclean["drop_max"],
                        mclean["x1_exc_mean"], mclean["decline_mean"],
                        mclean["tail_mean"], mclean["seg_err_mean"],
                        mclean["std_mean"], mclean["seg_drift_mean"], mclean["n_ev"]))
        lines.append("  相对基线(全事件): " + " ".join(
            "%s%+.1f" % (k, m[k] - bmacro[k]) for k in
            ["drop_mean", "drop_max", "x1_exc_mean", "decline_mean",
             "tail_mean", "seg_err_mean", "std_mean", "seg_drift_mean",
             "idle", "t_settle_mean", "fid"]))
        lines.append("  相对基线(干净保压): " + " ".join(
            "%s%+.1f" % (k, mclean[k] - bmacro_clean[k]) for k in
            ["drop_mean", "drop_max", "x1_exc_mean", "decline_mean",
             "tail_mean", "seg_err_mean", "std_mean", "seg_drift_mean"]))
        for r in rows:
            b = bmap[r["tag"]]
            lines.append("    Δ %-46s 回落%+7.1f/%+7.1f x1超调%+7.1f 尾段%+7.1f "
                         "保证段%+8.1f 保压std%+7.1f 漂%+7.1f 空载%+7.1f"
                         % (r["tag"][-46:], r["drop_mean"] - b["drop_mean"],
                            r["drop_max"] - b["drop_max"],
                            r["x1_exc_mean"] - b["x1_exc_mean"],
                            r["tail_mean"] - b["tail_mean"],
                            r["seg_err_mean"] - b["seg_err_mean"],
                            r["std_mean"] - b["std_mean"],
                            r["seg_drift_mean"] - b["seg_drift_mean"],
                            r["idle"] - b["idle"]))

    TBL_COLS = [("变体", 11, "s"), ("说明", 40, "s"), ("回落", 7, "f"),
                ("回落峰", 7, "f"), ("回落/台阶", 10, "f"), ("x1超调", 8, "f"),
                ("绝对下降", 8, "f"), ("尾段爬升", 8, "f"), ("保证段偏差", 10, "f"),
                ("保压std", 7, "f"), ("保压漂移", 8, "f"), ("空载偏差", 8, "f"),
                ("稳时", 6, "f"), ("一致性", 7, "s")]

    def _trow(vals):
        return " ".join(T._cell(v, w, k) for v, (_, w, k) in zip(vals, TBL_COLS))

    def _trow_head(vals):
        return " ".join(("%-" + str(w) + "s") % str(v)[:w]
                        for v, (_, w, _k) in zip(vals, TBL_COLS))

    def table(title, pick, data):
        lines.append("")
        lines.append("=" * 152)
        lines.append(title)
        lines.append("=" * 152)
        lines.append(_trow_head([t for t, _, _ in TBL_COLS]))
        lines.append("-" * 152)
        for name, desc, m, rows, mclean in data:
            mm = pick(m, mclean)
            lines.append(_trow([name, desc[:40], mm["drop_mean"],
                                mm["drop_max"], mm["drop_rel_mean"],
                                mm["x1_exc_mean"], mm["decline_mean"],
                                mm["tail_mean"], mm["seg_err_mean"],
                                mm["std_mean"], mm["seg_drift_mean"],
                                mm["idle"], mm["t_settle_mean"],
                                "%d/%d" % (mm["cons_ok"], mm["cons_tot"])]))
        lines.append("-" * 152)
        lines.append(_trow(["[基线]", "v3.4 固定形状 r1=0.12 τc1=8",
                            bmacro["drop_mean"], bmacro["drop_max"],
                            bmacro["drop_rel_mean"], bmacro["x1_exc_mean"],
                            bmacro["decline_mean"], bmacro["tail_mean"],
                            bmacro["seg_err_mean"], bmacro["std_mean"],
                            bmacro["seg_drift_mean"], bmacro["idle"],
                            bmacro["t_settle_mean"],
                            "%d/%d" % (bmacro["cons_ok"], bmacro["cons_tot"])]))

    table("汇总 A：全事件宏平均（事件门槛：窗口≥25s 且输入std/台阶≤0.15）",
          lambda m, mc: m, allrows)
    table("汇总 B：仅「干净保压」事件（输入std/台阶≤0.05）",
          lambda m, mc: dict(mc, cons_ok=m["cons_ok"], cons_tot=m["cons_tot"],
                             t_settle_mean=m["t_settle_mean"], fid=m["fid"],
                             idle=m["idle"]),
          allrows)

    lines.append("")
    lines.append("相对基线增量（全事件宏平均；单位 ADC，负=改善）：")
    DELTAS = ["drop_mean", "drop_max", "x1_exc_mean", "decline_mean",
              "tail_mean", "seg_err_mean", "std_mean", "seg_drift_mean",
              "idle", "t_settle_mean"]
    DTITLE = ["Δ回落", "Δ回落峰", "Δx1超调", "Δ绝对下降", "Δ尾段爬升",
              "Δ保证段偏差", "Δ保压std", "Δ保压漂移", "Δ空载", "Δ稳时"]
    lines.append("%-11s %s" % ("变体", " ".join("%9s" % t for t in DTITLE)))
    for name, desc, m, rows, mclean in allrows:
        lines.append("%-11s %s" % (name, " ".join(
            "%+9.1f" % (m[k] - bmacro[k]) for k in DELTAS)))

    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "t2_compare.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(OUT, "t2_compare.csv"), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh)
        hdr = ["variant", "session", "n_ev"] + KEYS + ["cons_ok", "cons_tot"]
        w.writerow(hdr)
        for name, desc, m, rows, mclean in allrows:
            for r in rows:
                w.writerow([name, r["tag"], r["n_ev"]]
                           + ["%.4f" % r[k] if not np.isnan(r[k]) else ""
                              for k in KEYS]
                           + [r["cons_ok"], r["cons_tot"]])
    for line in lines:
        print(line)
    print("写出:", os.path.join(OUT, "t2_compare.txt"))


if __name__ == "__main__":
    main(sys.argv)
