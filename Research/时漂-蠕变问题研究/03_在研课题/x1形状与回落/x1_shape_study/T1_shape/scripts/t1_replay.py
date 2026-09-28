# -*- coding: utf-8 -*-
"""步骤 3：显示"回落/过扣"量测（以录制显示为准）与归因（反事实重放）。

量测定义（21 通道之和 = tot，以及逐通道 = chN）：
  t0      = 加载沿起点
  t_end   = 窗口末 = min(t0 + W, 保压末)，两个窗口 W = 10 s 与 60 s
  t_peak  = 沿后显示最大值时刻（峰值后 0.5 s 中位作电平）
  fall    = D(t_peak) − D(t_end)          显示回落深度
  v_creep = v(t_end) − v(t_peak)          同窗口内输入自身爬升
理想补偿（显示钉在弹性电平）要求 fall = 0；fall 本身即净"多扣/欠扣"量：
      fall = Δx1 + Δx2 − v_creep
fall 优先取 device_001_seg000.csv（实机显示 = 真值，不受回放初始状态影响）；
反事实重放（只改 r1 / 只改 tc1 / 关 x1 / 关 x2）用差值给出 x1 / x2 的贡献。

输出：results/t1_replay_events.csv、results/t1_replay.txt
用法： python t1_replay.py
"""
import csv
import os
import sys

import numpy as np

import t1_lib as T

MIN_STEP_FRAC = 0.03      # 台阶占总量范围比例下限
MIN_PLATEAU = 12.0        # 保压时长下限 s
WINS = (10.0, 60.0)
MIN_CH_DCH = 150.0        # 逐通道分析的最小通道台阶

CASES = [
    ("base", {}),
    ("no_x1", dict(r1=0.0)),
    ("no_x2", dict(r2max=0.0)),
    ("tc1_2", dict(tc1=2.0)),
    ("tc1_20", dict(tc1=20.0)),
    ("r1_005", dict(r1=0.05)),
    ("r1_020", dict(r1=0.20)),
    ("shape_fit", dict(r1=0.08, tc1=2.0)),
]

COLS = ["ses", "cas", "sig", "ch", "W", "t0", "t1", "ramp", "plateau", "step",
        "dch", "win", "parity", "peak", "t_peak", "end_disp", "end_v",
        "fall", "fall_frac", "v_creep", "x1_end", "x2_end", "d_x1", "d_x2",
        "src"]


def measure(t, v, D, X1, X2, s, plateau, W, src, step):
    t0, t1 = s["t0"], s["t1"]
    end = min(t1 + W, t1 + plateau - 0.5, t[-1])
    k0 = int(np.searchsorted(t, t0))
    kp = int(np.searchsorted(t, end))
    if kp - k0 < 150:
        return None
    # 0.3 s 中值滤波：录制流里存在单帧跳变（例：73032d t=4.619 单帧 12062），
    # 直接用 argmax 会被单帧尖峰劫持。
    Ds = T.med_smooth(D, 31)
    vs = T.med_smooth(v, 31)
    seg = slice(k0, kp)
    ip = k0 + int(np.argmax(Ds[seg]))
    ip2 = min(max(ip + 50, ip + 1), kp)
    peak = float(np.median(Ds[ip:ip2]))
    ipk = ip + int(np.argmax(Ds[ip:ip2]))
    start = min(max(kp - 300, ip2), max(kp - 100, k0))
    tail = slice(start, kp)
    end_disp = float(np.median(Ds[tail]))
    end_v = float(np.median(vs[tail]))
    v_pk = float(np.median(vs[ip:ip2]))
    fall = peak - end_disp
    x1p = float(np.median(X1[ip:ip2])) if X1 is not None else float("nan")
    x2p = float(np.median(X2[ip:ip2])) if X2 is not None else float("nan")
    out = dict(
        W=W, t0=round(t0, 3), t1=round(t1, 3), ramp=round(s["dur"], 3),
        plateau=round(plateau, 2), step=round(step, 1), src=src,
        win=round(float(t[kp - 1] - t0), 2),
        peak=round(peak, 1), t_peak=round(float(t[ipk] - t0), 2),
        end_disp=round(end_disp, 1), end_v=round(end_v, 1),
        fall=round(fall, 1),
        fall_frac=round(fall / abs(step), 4) if abs(step) > 1e-9 else "",
        v_creep=round(end_v - v_pk, 1))
    if X1 is not None:
        out.update(x1_end=round(float(np.median(X1[tail])), 1),
                   x2_end=round(float(np.median(X2[tail])), 1),
                   d_x1=round(float(np.median(X1[tail])) - x1p, 1),
                   d_x2=round(float(np.median(X2[tail])) - x2p, 1))
    return out


def main():
    rows = []
    log = []
    w = log.append
    for tag, root in (("working", T.WORKING), ("archived", T.ARCHIVED)):
        for label, d in T.discover_sessions(root, 5):
            pre, rec = T.load_session(d)
            t, ts, V = pre["t"], pre["ts"], pre["V"]
            v = T.total(pre)
            steps, _ = T.detect_steps(t, v)
            rng = float(v.max() - v.min())
            rngc = V.max(axis=0) - V.min(axis=0)
            evs = []
            for k, s in enumerate(steps):
                if s["sign"] <= 0 or abs(s["d"]) < MIN_STEP_FRAC * rng:
                    continue
                nxt = steps[k + 1]["t0"] if k + 1 < len(steps) else t[-1]
                plateau = nxt - s["t1"]
                if plateau >= MIN_PLATEAU:
                    evs.append((s, plateau))
            if not evs:
                continue
            base = T.replay(ts, V, {})
            par = float("nan")
            if rec is not None and rec["n"] == pre["n"]:
                par = float(np.max(np.abs((V - base["X1"] - base["X2"]) - rec["V"])))
            others = {cn: T.replay(ts, V, ps) for cn, ps in CASES if cn != "base"}
            w("=" * 112)
            w("%s   事件 %d 个   回放 vs 录制 逐通道 max|Δ| = %.2f ADC"
              % (label, len(evs), par))
            # 逐通道响应通道
            resp = [c for c in range(T.NCH) if rngc[c] >= 5 * MIN_CH_DCH]
            w("   %-6s %-9s %5s %8s %7s %8s %8s %8s %8s %8s %8s %8s"
              % ("sig", "cas", "W", "t0", "t_peak", "replay回落", "录回落",
                 "Δx1", "Δx2", "输入爬升", "占台阶", "净差"))
            for s, plateau in evs:
                for W in WINS:
                    if W > plateau - 0.5:
                        continue
                    for cname in ["base"] + [c for c, _ in CASES if c != "base"]:
                        r = base if cname == "base" else others[cname]
                        m = measure(t, v, r["D"].sum(axis=1), r["X1"].sum(axis=1),
                                    r["X2"].sum(axis=1), s, plateau, W, "replay",
                                    s["d"])
                        if m is None:
                            continue
                        m.update(ses=label, cas=cname, sig="tot", ch=-1,
                                 dch=round(s["d"], 1), parity=round(par, 2))
                        rows.append(m)
                        if cname == "base" and W == 10.0:
                            mrec = None
                            if rec is not None and rec["n"] == pre["n"]:
                                mrec = measure(t, v, rec["V"].sum(axis=1), None,
                                               None, s, plateau, W, "rec", s["d"])
                            if mrec is not None:
                                mrec.update(ses=label, cas="recorded", sig="tot",
                                            ch=-1, dch=round(s["d"], 1),
                                            parity=round(par, 2))
                                rows.append(mrec)
                            w("   %-6s %-9s %5.0f %8.2f %7.2f %8.0f %8s %8.0f %8.0f "
                              "%8.0f %8s %8.0f"
                              % ("tot", "base", W, m["t0"], m["t_peak"], m["fall"],
                                 "%.0f" % mrec["fall"] if mrec else "-",
                                 m["d_x1"], m["d_x2"], m["v_creep"],
                                 "%.2f" % m["fall_frac"], m["fall"] - m["v_creep"]))
                    # 逐通道
                    for c in resp:
                        dch = abs(float(np.median(V[int(np.searchsorted(t, s['t1'])):][:60, c])
                                        - np.median(V[max(0, int(np.searchsorted(t, s['t0'])) - 60):
                                                      int(np.searchsorted(t, s['t0'])) + 1, c])))
                        if dch < MIN_CH_DCH or s["d"] <= 0:
                            continue
                        mbase = measure(t, V[:, c], base["D"][:, c], base["X1"][:, c],
                                        base["X2"][:, c], s, plateau, W, "replay",
                                        dch)
                        if mbase is None:
                            continue
                        mbase.update(ses=label, cas="base", sig="ch", ch=c,
                                     dch=round(dch, 1), parity=round(par, 2))
                        rows.append(mbase)
                        for cname, r in others.items():
                            mm = measure(t, V[:, c], r["D"][:, c], r["X1"][:, c],
                                         r["X2"][:, c], s, plateau, W, "replay", dch)
                            if mm is None:
                                continue
                            mm.update(ses=label, cas=cname, sig="ch", ch=c,
                                      dch=round(dch, 1), parity=round(par, 2))
                            rows.append(mm)
    os.makedirs(T.RESULTS, exist_ok=True)
    with open(os.path.join(T.RESULTS, "t1_replay_events.csv"), "w",
              encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in COLS})
    with open(os.path.join(T.RESULTS, "t1_replay.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    sys.stdout.write("written results/t1_replay_events.csv (%d 行)\n" % len(rows))


if __name__ == "__main__":
    main()
