# -*- coding: utf-8 -*-
"""T3 步骤5：误触发 / 抖动风险测试（修订版）。

四类风险：
  R1 静置/非事件误触发：不在加载事件窗 [t0−0.5, t0+2] 内的触发次数与置位幅度
  R2 慢速连续加载：识别缓升段（无单帧台阶的连续上升），统计其间触发次数与显示偏移
  R3 振荡/随机切换：触发数 vs 事件数、与基线显示偏差（事件窗内/外）、x1 提前量
  R4 卸载：卸载沿窗 [u, min(下一个加载沿, u+3 s)] 内的触发次数（探测器只认正 slope，
     此处统计的是「卸载过程中/卸载后即刻」是否被误置位）

真实加载/卸载沿用独立简单口径（总量 1 s 差 > max(500, 5%·量程)，间隔 >2 s）。
"""
import os
import sys

import numpy as np

import t3_lib as T

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

ARM = dict(T.DET)
ARM.update(dict(mode="A", pred="lag", alpha=1.0))   # 最敏感臂，用于风险测试


def onsets(el, tot, sign=+1, min_step_abs=500.0, min_step_frac=0.05, gap=2.0):
    rng = float(tot.max() - tot.min())
    ms = max(min_step_abs, min_step_frac * rng)
    lag = np.concatenate([[tot[0]] * 100, tot[:-100]])
    d = (tot - lag) * sign
    ev, last = [], -1e9
    for i in range(len(el)):
        if d[i] > ms and el[i] - last > gap:
            ev.append(float(el[i]))
            last = el[i]
    return np.array(ev), ms


def ramps(el, tot, tmax=30.0, min_rise=1500.0):
    """缓升段：连续上升但无单帧台阶（1 s 差 < 0.5·ms）。"""
    rng = float(tot.max() - tot.min())
    ms = max(500.0, 0.05 * rng)
    out = []
    i, n = 0, len(el)
    while i < n:
        j = min(int(np.searchsorted(el, el[i] + tmax)), n - 1)
        seg = tot[i:j + 1]
        if len(seg) > 10 and seg[-1] - seg[0] > min_rise:
            d1 = np.diff(seg, prepend=seg[0])
            if np.max(d1) < 0.5 * ms:
                out.append((float(el[i]), float(el[j]), float(seg[-1] - seg[0])))
                i = j
                continue
        i += 50
    return out


def main():
    keys = [a for a in sys.argv[1:] if not a.startswith("--")]
    lines = []
    P = lines.append
    P("T3 误触发/抖动风险测试（臂 = A_lag_a1.0：thr=%.0f + %.2f·0.02·max(e,1)，迟滞%.2f，最小间隔%.1fs）"
      % (ARM["thr"], ARM["rel_on"], ARM["hyst"], ARM["min_gap"]))
    P("")
    P("会话                                  事件 | 触发 |事件窗内|窗 外| 卸载窗内触发 | x1提前量max | 空载帧显示偏差max | 窗外显示偏差p95")
    agg = dict(n_ev=0, n_trig=0, n_on=0, n_off=0, n_un_win=0, n_un=0,
               x1_max=0.0, idle_dev=0.0, dev_p95=0.0, n_ramp=0, ramp_trig=0,
               ramp_sec=0.0, ramp_dev=[])
    for tag, label, d in T.all_sessions():
        if keys and not any(k in label for k in keys):
            continue
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        din = V.sum(axis=1)
        ev, ms = onsets(el, din, +1)
        un, _ = onsets(el, din, -1)
        rf = T.observe(ts, V, ff=ARM)
        rb = T.observe(ts, V, ff=None)
        df = rf["D"].sum(axis=1)
        db = rb["D"].sum(axis=1)
        x1f = rf["X1"].sum(axis=1)
        x1b = rb["X1"].sum(axis=1)
        trig_t = np.array([float(el[i]) for c, i, sl, ep, xb in rf["trig"]])
        trig_i = np.array([i for c, i, sl, ep, xb in rf["trig"]], dtype=int)
        # 触发时输入自身的近 1 s / 3 s 变化（判定「是否为真实负载变化」）
        lo1 = np.array([din[max(0, q - 100):q + 1].min() for q in trig_i]) if len(trig_i) else np.array([])
        rise1 = np.array([din[q] - v for q, v in zip(trig_i, lo1)]) if len(trig_i) else np.array([])
        rise3 = np.array([din[q] - din[max(0, q - 300)] for q in trig_i]) if len(trig_i) else np.array([])
        ms_ = ms
        n_spur = int(np.sum(rise3 < 0.01 * ms_)) if len(trig_i) else 0        # 近 3 s 输入几乎没变
        n_fall = int(np.sum(rise1 < 0.0)) if len(trig_i) else 0             # 近 1 s 输入在下降
        jump_spur = (float(np.median(0.12 * np.array([ep for c, i, sl, ep, xb in rf["trig"]])[rise3 < 0.01 * ms_] * 21))
                     if n_spur else 0.0)
        onmask = T.event_mask(el, ev, 0.5, 2.0)
        idx = np.clip(np.searchsorted(el, trig_t), 0, len(el) - 1) if len(trig_t) else np.array([], dtype=int)
        on = onmask[idx] if len(trig_t) else np.array([], dtype=bool)
        n_on = int(np.sum(on))
        # 卸载窗：u → min(下一个加载沿, u+3s)
        un_win = np.zeros(len(el), dtype=bool)
        for u in un:
            nxt = ev[ev > u]
            end = min(u + 3.0, float(nxt[0]) if len(nxt) else u + 3.0)
            un_win |= (el >= u) & (el <= end)
        n_un_win = int(np.sum(un_win[idx])) if len(trig_t) else 0
        dev = df - db
        far = ~T.event_mask(el, ev, 0.5, 20.0)
        idle = T.idle_mask(din)
        devfar = np.abs(dev[far]) if far.sum() > 10 else np.array([0.0])
        devidle = np.abs(dev[idle]) if idle.sum() > 10 else np.array([0.0])
        x1d = float(np.max(x1f - x1b))
        P("%-36s %4d | %4d | %6d | %5d | %12d | %10.0f | %15.0f | %14.0f"
          % (label[-36:], len(ev), len(trig_t), n_on, len(trig_t) - n_on,
             n_un_win, x1d, float(np.max(devidle)), float(np.percentile(devfar, 95))))
        P("     触发时刻输入近 3 s 变化 <1%%·min_step（真·无负载变化）= %d 次；近 1 s 输入在下降 = %d 次"
          % (n_spur, n_fall))
        rp = ramps(el, din)
        rtrig = 0
        rdev = []
        for t0, t1, rise in rp:
            rtrig += int(np.sum((trig_t >= t0) & (trig_t <= t1)))
            m = (el >= t0) & (el <= t1)
            if m.sum() > 20:
                rdev.append(float(np.mean(dev[m])))
        if rp:
            P("     缓升段 %d 个：%s  段内显示偏移均值 %s"
              % (len(rp),
                 " ".join("[%.0f–%.0fs 升%.0fADC]" % (a, b, c) for a, b, c in rp[:6]),
                 " ".join("%+.0f" % x for x in rdev[:6])))
        agg["n_ev"] += len(ev)
        agg["n_trig"] += len(trig_t)
        agg["n_on"] += n_on
        agg["n_off"] += len(trig_t) - n_on
        agg["n_un_win"] += n_un_win
        agg["n_un"] += len(un)
        agg["x1_max"] = max(agg["x1_max"], x1d)
        agg["idle_dev"] = max(agg["idle_dev"], float(np.max(devidle)))
        agg["dev_p95"] = max(agg["dev_p95"], float(np.percentile(devfar, 95)))
        agg["n_ramp"] += len(rp)
        agg["ramp_trig"] += rtrig
        agg["ramp_sec"] += sum(b - a for a, b, c in rp)
        agg["ramp_dev"].extend(rdev)
        agg["n_spur"] = agg.get("n_spur", 0) + n_spur
        agg["n_fall"] = agg.get("n_fall", 0) + n_fall
        agg["spur_jump"] = max(agg.get("spur_jump", 0.0), jump_spur)
    P("")
    P("汇总：真实加载事件 %d 个；触发 %d 次，事件窗内 %d、窗外 %d（%.0f%%）"
      % (agg["n_ev"], agg["n_trig"], agg["n_on"], agg["n_off"],
         100.0 * agg["n_off"] / max(agg["n_trig"], 1)))
    P("      卸载沿 %d 个；卸载窗（卸载后至多 3 s 或到下一个加载沿）内触发 %d 次" % (agg["n_un"], agg["n_un_win"]))
    P("      真·无负载变化误触发（近 3 s 输入变化 <1%%·min_step）%d 次 / 共 %d 次（%.1f%%）；"
      "其中位置位增量 %.0f ADC"
      % (agg.get("n_spur", 0), agg["n_trig"],
         100.0 * agg.get("n_spur", 0) / max(agg["n_trig"], 1), agg.get("spur_jump", 0.0)))
    P("      近 1 s 输入在下降时仍触发 %d 次（正 slope 要求下应为个位数）" % agg.get("n_fall", 0))
    P("      缓升段 %d 个（合计 %.0f s）内触发 %d 次；段内显示相对基线偏移均值 %s"
      % (agg["n_ramp"], agg["ramp_sec"], agg["ramp_trig"],
         ("%.0f ADC" % np.mean(agg["ramp_dev"])) if agg["ramp_dev"] else "—"))
    P("      全数据集 x1 提前量 max（Σx1_前馈 − Σx1_基线）= %.0f ADC" % agg["x1_max"])
    P("      空载帧 |显示−基线| max = %.0f ADC；事件窗（沿后 20 s）外 p95 上界 = %.0f ADC"
      % (agg["idle_dev"], agg["dev_p95"]))
    with open(os.path.join(OUT, "t3_risk.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
