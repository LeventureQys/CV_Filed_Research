# -*- coding: utf-8 -*-
"""v3.6 A/B 主程序：多臂 × 目标录制的**「同一负载 → 同一显示」一致性表** + 代价指标。

臂：
  v31               现役（产品源码）
  v32:<gain>        plan v3.2 的 R1（creep seed）
  v36:<参数...>     本轮候选（v3.2 基线 + v3.6 补丁）

口径（全部总量 Σch，ADC）：
  「满载荷平台」E1~E9：由输入电平表（results/v36_level_steps.txt）手工核验的落定窗
  「减一砝码平台」P1~P6：同上
  每段给出 in_lvl / out_lvl / ded = in−out / 相对"台阶"的补偿比
  一致性判据：同一名义负载的 out_lvl 极差（越小越一致）；
  另给 G（沿后 [3,5] s Δ显示/Δ输入）与稳定时间 stab_s（进入 ±3%×台阶 并保持）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
import v36_replay as R  # noqa: E402
from v36_probe_internal import R_arm  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

# 满载荷平台（输入总电平 ≈17.0~17.8 kADC）：标签, 参考沿, 窗起, 窗止
EPISODES = [
    ("E1 首次加载",        3.34,  10.0,  14.0),
    ("E2 +1砝码",         15.18,  22.0,  36.0),
    ("E3 卸载→重载",       41.75,  48.0,  56.0),
    ("E4 卸载→重载",       63.41,  70.0, 100.0),
    ("E5 卸载→重载(长保压)", 232.23, 234.0, 239.0),
    ("E6 完全卸载后重载",   245.31, 248.5, 251.5),
    ("E7 完全卸载后重载",   258.25, 268.0, 275.0),
    ("E8 完全卸载后重载",   278.50, 292.0, 295.0),
    ("E9 完全卸载后重载",   300.50, 305.0, 307.5),
]
PARTIAL = [
    ("P1 减一砝码",   37.60,  38.5,  41.0),
    ("P2 减一砝码",   57.90,  59.5,  62.5),
    ("P3 减一砝码",  219.50, 221.0, 230.0),
    ("P4 减一砝码",  252.00, 253.0, 254.5),
    ("P5 减一砝码",  282.50, 285.0, 288.5),
    ("P6 减一砝码",  295.50, 296.0, 297.0),
]
# 台阶保真：加载沿 + 其后第一段平台
G_EDGES = [3.34, 15.18, 41.75, 63.41, 232.23, 245.31, 261.50, 289.50, 303.10]


def wmed(t, x, a, b):
    m = (t >= a) & (t < b)
    return float(np.median(x[m])) if m.any() else float("nan")


def episode_table(st):
    t = st["t"]
    tin, tout = st["sum_in"], st["sum_out"]
    rows = []
    for (lab, te, w0, w1) in EPISODES + PARTIAL:
        # 台阶：沿前 [−3,−0.5] s 的输入电平
        base = wmed(t, tin, te - 3.0, te - 0.5)
        li = wmed(t, tin, w0, w1)
        lo = wmed(t, tout, w0, w1)
        rows.append(dict(lab=lab, te=te, base=base, in_lvl=li, out_lvl=lo,
                         ded=li - lo, step=li - base,
                         frac=(li - lo) / (li - base) if abs(li - base) > 1 else float("nan"),
                         kind="full" if lab.startswith("E") else "part"))
    return rows


def step_metrics(st):
    t = st["t"]
    tin, tout = st["sum_in"], st["sum_out"]
    out = []
    for te in G_EDGES:
        bi = wmed(t, tin, te - 2.5, te - 0.5)
        bo = wmed(t, tout, te - 2.5, te - 0.5)
        di = wmed(t, tin, te + 3.0, te + 5.0) - bi
        do = wmed(t, tout, te + 3.0, te + 5.0) - bo
        out.append((te, di, do, do / di if abs(di) > 1 else float("nan")))
    return out


def stab_time(st, te, w0, w1, step):
    t = st["t"]
    sout = st["sum_out"]
    target = wmed(t, sout, w0, w1)
    lo, hi = target - 0.03 * abs(step), target + 0.03 * abs(step)
    m = (t >= te) & (t < w1)
    tt, oo = t[m], sout[m]
    for i in range(len(tt)):
        if np.all((oo[i:] >= lo) & (oo[i:] <= hi)):
            return float(tt[i] - te)
    return float("nan")


def main():
    arms = sys.argv[1:] or ["v31", "v32:0.7", "v36:--seed-gain:0.7"]
    ds = K.load(K.DS_TARGET)
    lines = []
    for a in arms:
        exe, args = R_arm(a)
        st = R.run_arm(ds, exe, args)
        rows = episode_table(st)
        t = st["t"]
        gm = dict(
            viol=float(np.nanmax(np.abs(st["max_clamp_viol"]))),
            hits=float(st["clamp_hits"][-1]),
            hold_pkpk=float(np.percentile(st["sum_out"][(st["t"] >= 70) & (st["t"] <= 232)], 99) -
                            np.percentile(st["sum_out"][(st["t"] >= 70) & (st["t"] <= 232)], 1)),
        )
        L = ["== 臂 %s (args=%s) ==" % (a, " ".join(args)),
             "%18s %9s %9s %9s %8s %8s %8s %7s" %
             ("段", "in_lvl", "out_lvl", "ded", "ded@3-5s", "stab_s", "G", "ded/Δin")]
        sm = step_metrics(st)
        gmap = {round(te, 2): g for (te, _, _, g) in sm}
        for r in rows:
            early = wmed(t, st["ded"], r["te"] + 3.0, r["te"] + 5.0)
            # 稳定时间：以「该段落定窗」为参考，从沿后起算
            sb = stab_time(st, r["te"], r["te"] + 3.0, r["te"] + 8.0, r["step"])
            L.append("%18s %9.0f %9.0f %9.0f %8.0f %8.2f %8.3f %7.3f" %
                     (r["lab"], r["in_lvl"], r["out_lvl"], r["ded"], early, sb,
                      gmap.get(round(r["te"], 2), float("nan")), r["frac"]))
        full = [r for r in rows if r["kind"] == "full"]
        grp = full[1:]      # E2~E9 = 「同一次装配 + 全部砝码」的同一负载（E1 是更小的首次加载，单列）
        rr = [r["out_lvl"] for r in grp]
        dd = [r["ded"] for r in grp]
        L.append("E1(小载荷) out_lvl %.0f" % full[0]["out_lvl"])
        L.append("同负载组 E2~E9 out_lvl: 极差 %.0f 中位 %.0f | 前四段(E2~E5) 中位 %.0f vs 后五段(E6~E9) 中位 %.0f → 差 %.0f"
                 % (max(rr) - min(rr), np.median(rr), np.median(rr[:4]), np.median(rr[4:]),
                    np.median(rr[:4]) - np.median(rr[4:])))
        L.append("同负载组 E2~E9 ded: 极差 %.0f 中位 %.0f | 前4 中位 %.0f vs 后4 中位 %.0f → 差 %.0f"
                 % (max(dd) - min(dd), np.median(dd), np.median(dd[:4]), np.median(dd[4:]),
                    np.median(dd[:4]) - np.median(dd[4:])))
        gs = [g for (_, _, _, g) in sm if np.isfinite(g)]
        sb = [stab_time(st, r["te"], r["te"] + 3.0, r["te"] + 8.0, r["step"])
              for r in full[1:]]
        sb = [v for v in sb if np.isfinite(v)]
        L.append("G 中位 %.3f 最差 %.3f | 稳定时间中位 %.2f s 最坏 %.2f s | C限幅越界 %.3g 命中 %g | 长保压 pk-pk %.0f"
                 % (np.median(gs), min(gs), np.median(sb) if sb else float("nan"),
                    max(sb) if sb else float("nan"), gm["viol"], gm["hits"], gm["hold_pkpk"]))
        lines.append("\n".join(L))
        print("\n".join(L))
        print()
    p = os.path.join(RES, "v36_ab.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(lines) + "\n")
    print("-> %s" % p)


if __name__ == "__main__":
    main()
