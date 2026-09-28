# -*- coding: utf-8 -*-
"""T3 步骤3：全数据集回放 —— 基线 vs F-A/F-B/F-C 前馈变体。

事件 = 输入总量台阶 ≥ max(500, 5%·量程) 的加载沿（评估侧非因果切分）。

逐事件指标（口径与 v34_probe_tstable 一致的位置参数）：
  peak    沿后显示峰值
  settle  段末稳定电平（保压段末 30% 中位）
  fall    回落幅度 = peak − settle        ← 用户主诉「显示回落过大」
  dip     下冲 = max(0, settle − 沿后最小显示) ← 前馈过扣的直接量度
  t5      从沿起进入并保持 settle±5%·step 的时刻（稳定时间）
  lvl_dev settle_arm − settle_base        ← 该臂相对基线的稳态电平偏置（过扣/欠扣）
  in_rng  峰值后输入电平波动 / 台阶（>0.03 视为「非静载保压」，dip/fall 仅供参考）

按事件类分层：首次大台阶（空载起大台阶）/ 受载态小台阶（用户主诉场景）/ 其他。

用法：python t3_replay.py [--quick]
"""
import csv
import os
import sys

import numpy as np

import t3_lib as T

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

DET_BASE = dict(thr=10.0, rel_on=0.5, hyst=0.35, min_gap=0.6)

ARMS = [
    ("base", None),
    ("A_now_a1.0", dict(mode="A", pred="now", alpha=1.0)),
    ("A_lag_a1.0", dict(mode="A", pred="lag", alpha=1.0)),
    ("A_now_a0.8", dict(mode="A", pred="now", alpha=0.8)),
    ("A_now_a0.6", dict(mode="A", pred="now", alpha=0.6)),
    ("B_tb1.0_1.5s", dict(mode="B", tau_boost=1.0, boost_s=1.5)),
    ("B_tb2.0_2.0s", dict(mode="B", tau_boost=2.0, boost_s=2.0)),
    ("C_a0.8_tb2.0_1.5s", dict(mode="C", pred="now", alpha=0.8,
                               tau_boost=2.0, boost_s=1.5)),
    ("C_a1.0_tb2.0_1.5s", dict(mode="C", pred="now", alpha=1.0,
                               tau_boost=2.0, boost_s=1.5)),
    ("C_a0.8_lag_tb2.0_1.5s", dict(mode="C", pred="lag", alpha=0.8,
                                   tau_boost=2.0, boost_s=1.5)),
]


def arm_ff(spec):
    if spec is None:
        return None
    d = dict(DET_BASE)
    d.update(spec)
    return d


def classify(e, rng):
    if e["base"] < 0.20 * rng and e["step"] > 0.30 * rng:
        return "首次大台阶"
    if e["base"] > 0.20 * rng and e["step"] < 0.25 * rng:
        return "受载态小台阶"
    return "其他"


def smooth(y, n=30):
    """0.3 s 滑动均值（抑制噪声，用于形状类指标）。

    注意：必须用端点填充而不是 np.convolve(..., mode="same")——后者按零填充，
    会把序列首尾各 ~n/2 帧拉向 0（在 ADC~1e4 的信号上制造出数千 ADC 的假下冲）。
    """
    y = np.asarray(y, dtype=float)
    if len(y) <= n:
        return y.copy()
    pad = n // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    k = np.ones(n) / n
    out = np.convolve(ypad, k, mode="valid")
    return out[:len(y)]


def metrics_arm(el, din, disp, edges, base_disp=None):
    """逐事件量测。

    fall    回落幅度 = 峰值 − 段末稳定电平（用户主诉「显示回落过大」）
    fall5   沿后 5 s 内回落 = 峰值 − [峰后 5 s 内最小显示]
    dip     自回弹下冲 = max_t ( y[t] − min_{s≤t} y[s] )（先下探再回升的最大幅度；
            单调下降到稳定 ⇒ 0；过扣后自愈 ⇒ >0），在 0.3 s 平滑后计算（与参考电平无关）
    t5      进入并保持 settle±5%·台阶 的时刻
    lvl_dev 该臂 settle − 基线 settle
    dev_min/dev_max  该臂与基线显示的逐帧最大负/绝对偏差（直接波形差）
    in_step 保压中（沿后 2 s 起）输入 1 s 内最大上升 / 台阶（识别二次加载）
    """
    out = []
    for e in edges:
        i, j = e["i"], e["j"]
        seg = disp[i:j]
        if len(seg) < 20:
            out.append(None)
            continue
        ys = smooth(seg, 30)
        ee = el[i:j]
        peak = float(ys.max())
        ip = int(np.argmax(ys))
        w = min(len(ys), 200)          # 段末 2 s（不足则全程）
        settle = float(np.median(ys[-w:]))
        step = e["step"]
        band = 0.05 * abs(step) if step else 1.0
        # t5 = 最后一次离开 ±5%·step 带之后的时刻（等价于「进入并保持」，
        # 但对段末仍在缓慢漂移的显示也良定义；与 v34_probe_tstable 同义）
        out_idx = np.nonzero(np.abs(ys - settle) > band)[0]
        if len(out_idx) == 0:
            t5 = 0.0
        elif out_idx[-1] >= len(ys) - 1:
            t5 = None
        else:
            t5 = float(ee[out_idx[-1] + 1] - ee[0])
        tail = ys[ip:]
        runmin = np.minimum.accumulate(tail)
        dip = float(np.max(tail - runmin)) if len(tail) > 1 else 0.0
        i5 = min(len(ys) - 1, ip + 500)
        fall5 = float(peak - ys[ip:i5 + 1].min())
        w2 = din[i:j]
        irng = ((float(w2[ip:].max() - w2[ip:].min()) / abs(step)) if step else 0.0)
        # 保压中的二次加载：沿后 2 s 起，1 s 窗内最大上升
        q0 = min(len(w2) - 1, ip + 200)
        mx = 0.0
        for q in range(q0 + 1, len(w2)):
            p0 = max(q0, q - 100)
            mx = max(mx, float(w2[q] - w2[p0:q].min()))
        in_step = (mx / abs(step)) if step else 0.0
        dev_min = dev_max = 0.0
        if base_disp is not None:
            bs = smooth(base_disp[i:j], 30)
            if len(bs) == len(ys):
                d = ys - bs
                dev_min = float(d.min())
                dev_max = float(np.max(np.abs(d)))
        out.append(dict(t=float(ee[0]), step=step, peak=peak,
                        t_peak=float(ee[ip] - ee[0]), settle=settle,
                        fall=peak - settle, fall5=fall5, dip=dip,
                        t5=t5, in_rng=irng, in_step=in_step,
                        dev_min=dev_min, dev_max=dev_max,
                        hold=float(ee[-1] - ee[0])))
    return out


def hold_std(el, disp, edges):
    vals = []
    for e in edges:
        i1 = int(np.searchsorted(el, el[e["i"]] + 1.0))
        seg = disp[i1:e["j"]]
        if len(seg) > 40:
            vals.append(float(np.std(seg)))
    return vals


def idle_bias(el, disp, din, rng):
    m = din < (din.min() + 0.15 * rng)
    if m.sum() < 50:
        return None
    return float(np.mean(disp[m] - din[m]))


def consistency(el, disp, edges):
    """同类负载重复出现的电平一致性（±10%·台阶）。"""
    used = [False] * len(edges)
    ok = tot = 0
    devs = []
    order = sorted(range(len(edges)), key=lambda k: edges[k]["settle"])
    for a in range(len(order)):
        if used[order[a]]:
            continue
        grp = [order[a]]
        used[order[a]] = True
        lv = edges[order[a]]["settle"]
        for b in range(a + 1, len(order)):
            if not used[order[b]] and abs(edges[order[b]]["settle"] - lv) <= 0.08 * max(abs(lv), 1):
                grp.append(order[b])
                used[order[b]] = True
        if len(grp) < 2:
            continue
        grp.sort(key=lambda k: edges[k]["t"])
        ref = None
        for k in grp:
            i, j = edges[k]["i"], edges[k]["j"]
            w = max(15, int(0.3 * (j - i)))
            val = float(np.median(disp[max(i + 1, j - w):j]))
            if ref is None:
                ref = val
                continue
            dev = val - ref
            devs.append(dev)
            tot += 1
            ok += abs(dev) <= 0.10 * abs(edges[k]["step"])
    return ok, tot, devs


def run_session(label, d, arms, rows, hold_rows, trig_rows, sum_rows):
    s = T.load_input(d)
    el, ts, V = s["el"], s["ts"], s["V"]
    din = V.sum(axis=1)
    rng = float(din.max() - din.min())
    edges, ms = T.find_edges(el, din)
    if not edges:
        print("   %s 无评估沿，跳过" % label, flush=True)
        return
    res_by_arm = {}
    for name, spec in arms:
        r = T.observe(ts, V, ff=arm_ff(spec))
        res_by_arm[name] = (r["D"].sum(axis=1), r["trig"])
    disp_base = res_by_arm["base"][0]
    for name, spec in arms:
        disp, trig = res_by_arm[name]
        mm = metrics_arm(el, din, disp, edges, disp_base)
        res_by_arm[name] = (mm, disp, trig)
    base_mm = res_by_arm["base"][0]
    for name, spec in arms:
        mm, disp, trig = res_by_arm[name]
        for e, met, bmet in zip(edges, mm, base_mm):
            if met is None:
                continue
            rows.append(dict(session=label, arm=name, t=met["t"], step=met["step"],
                             cls=classify(e, rng), hold=met["hold"],
                             peak=met["peak"], settle=met["settle"],
                             settle_base=bmet["settle"],
                             lvl_dev=met["settle"] - bmet["settle"],
                             fall=met["fall"], fall5=met["fall5"], dip=met["dip"],
                             in_rng=met["in_rng"], in_step=met["in_step"],
                             dev_min=met["dev_min"], dev_max=met["dev_max"],
                             t5=(met["t5"] if met["t5"] is not None else -1),
                             t_peak=met["t_peak"]))
        hs = hold_std(el, disp, edges)
        hb = idle_bias(el, disp, din, rng)
        cok, ctot, cdev = consistency(el, disp, edges)
        tset = [e["t"] for e in edges]
        n_on_edge = sum(1 for c, i, sl, ep, xb in trig
                        if any(abs(float(el[i]) - t0) < 1.2 for t0 in tset))
        jumps = [abs(0.12 * ep) * 21 for c, i, sl, ep, xb in trig]
        epre = [ep for c, i, sl, ep, xb in trig]
        t5s = [m["t5"] for m in mm if m is not None and m["t5"] is not None]
        sum_rows.append(dict(
            session=label, arm=name, n_edge=len(edges),
            n_trig=len(trig), n_trig_on_edge=n_on_edge,
            trig_per_edge=(len(trig) / len(edges)),
            epre_med=float(np.median(epre)) if epre else 0.0,
            jump_med=float(np.median(jumps)) if jumps else 0.0,
            jump_tot_med=float(np.median(jumps)) * 1.0 if jumps else 0.0,
            hold_std=float(np.median(hs)) if hs else -1.0,
            idle_bias=(hb if hb is not None else float("nan")),
            cons_ok=cok, cons_tot=ctot,
            cons_med_dev=float(np.median(cdev)) if cdev else 0.0,
            t5_med=float(np.median(t5s)) if t5s else -1.0))
        print("   %-46s %-20s 沿%2d 触发%4d(沿上%4d) 保压std%5.0f 空载%+6.0f 一致性%d/%d"
              % (label[-44:], name, len(edges), len(trig), n_on_edge,
                 (np.median(hs) if hs else -1), (hb if hb is not None else float("nan")),
                 cok, ctot), flush=True)


def main():
    quick = "--quick" in sys.argv
    sessions = T.all_sessions()
    if quick:
        sessions = sessions[:3]
    rows, hold_rows, trig_rows, sum_rows = [], [], [], []
    print("== 逐会话逐臂回放（%d 会话 × %d 臂）" % (len(sessions), len(ARMS)), flush=True)
    for k, (tag, label, d) in enumerate(sessions, 1):
        print("[%d/%d] %s" % (k, len(sessions), label), flush=True)
        run_session(label, d, ARMS, rows, hold_rows, trig_rows, sum_rows)
        # 基准臂基线与触发记录（每会话只写一次）
        if not any(h["session"] == label for h in hold_rows):
            s = T.load_input(d)
            el = s["el"]
            din = s["V"].sum(axis=1)
            edges, _ = T.find_edges(el, din)
            for e in edges:
                hold_rows.append(dict(session=label, t=e["t"], step=e["step"],
                                      base=e["base"], settle_in=e["settle"],
                                      hold=float(el[e["j"]] - el[e["i"]])))
            ff = arm_ff(ARMS[1][1])
            r = T.observe(s["ts"], s["V"], ff=ff)
            for c, i, sl, ep, xb in r["trig"]:
                trig_rows.append(dict(session=label, t=float(el[i]), ch=c,
                                      slope=sl, e_pre=ep, x1_before=xb))
    os.makedirs(OUT, exist_ok=True)
    files = [("t3_edges.csv", rows,
              ("session", "arm", "t", "step", "cls", "hold", "peak", "settle",
               "settle_base", "lvl_dev", "fall", "fall5", "dip", "in_rng",
               "in_step", "dev_min", "dev_max", "t5", "t_peak")),
             ("t3_hold_segments.csv", hold_rows,
              ("session", "t", "step", "base", "settle_in", "hold")),
             ("t3_triggers.csv", trig_rows,
              ("session", "t", "ch", "slope", "e_pre", "x1_before")),
             ("t3_arm_session.csv", sum_rows, None)]
    for fn, rr, hdr in files:
        cols = hdr if hdr else (list(rr[0].keys()) if rr else [])
        with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols))
            w.writeheader()
            if rr:
                w.writerows(rr)
        print("   -> results/%s (%d 行)" % (fn, len(rr)), flush=True)


if __name__ == "__main__":
    main()
