# -*- coding: utf-8 -*-
"""t4a_07_crosscheck：口径交叉核对（回答「与既有结论的差异」时把差异归因到口径还是样本）。

三路对照，同一事件：
  ① 本任务总量 Z 口径（t4a_morphology.csv）；
  ② 同一 t_on 下的**主通道**口径（第一轮 r2_shapes.py 的信号选择）；
  ③ 第一轮 `13-v6-assessment/results/events.csv` 的现成数字（主通道 + 其自有 t_edge）。

产出：results/t4a_crosscheck.csv
日志：results/_t4a_07_crosscheck.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t4a_common as C
import t4a_ad_lib as L

MAP = {"右拇指指尖/数据1": "RT1", "右拇指指尖/数据2": "RT2", "右拇指指尖/数据3": "RT3",
       "左拇指指尖/数据1": "LT1", "左拇指指尖/数据2": "LT2", "左拇指指尖/数据3": "LT3",
       "四指指尖/数据1": "F41", "四指指尖/数据2": "F42", "四指指尖/数据3": "F43",
       "切换负载-快相无责": "SW1", "再切换负载": "SW2",
       "中途切换-1d9493": "SW3", "中途切换-13ffca": "SW4"}


def main():
    C.start_log()
    mor = pd.read_csv(os.path.join(C.RES, "t4a_morphology.csv"))
    recs = {}
    for r in C.RECS:
        tu, Xu, pkt, nraw = C.load_grid(r)
        Z = Xu.sum(axis=1)
        recs[r["key"]] = dict(tu=tu, Xu=Xu, Z=Z, Zs=L.med_smooth(Z, C.KSM),
                              Zch=L.med_smooth(Xu[:, r["ch"]], C.KSM),
                              amp=float(np.percentile(Z, 99.5) - np.percentile(Z, 0.5)),
                              peak=float(L.med_smooth(Z, C.KSM).max()), ch=r["ch"], pkt=pkt)
    rows = []
    for _, q in mor.iterrows():
        R = recs[q.key]
        k = int(q.k_on)
        mch = C.event_metrics(R["tu"], R["Xu"][:, R["ch"]], R["Zch"], k, float(R["Zch"].max()),
                              float(np.percentile(R["Xu"][:, R["ch"]], 99.5)
                                    - np.percentile(R["Xu"][:, R["ch"]], 0.5)), R["pkt"], [k])
        rows.append(dict(key=q.key, rec=q.rec, kind=q.kind, clean=q["clean"], t_on=q.t_on,
                         z_t50=q.t50, z_t90=q.t90, z_at_02=q.z_at_02, z_at_10=q.z_at_10,
                         ch_t50=mch["t50"], ch_t90=mch["t90"], ch_at_02=mch["z_at_02"],
                         ch_at_10=mch["z_at_10"], ch_jump=mch["jump"], z_jump=q.jump))
    cc = pd.DataFrame(rows)
    cc.to_csv(os.path.join(C.RES, "t4a_crosscheck.csv"), index=False, encoding="utf-8-sig")

    print("== ① vs ②：同一 t_on，总量 Z 与主通道的差异（全部 %d 事件）==" % len(cc))
    for a, b in (("z_at_02", "ch_at_02"), ("z_at_10", "ch_at_10"), ("z_t50", "ch_t50"),
                 ("z_t90", "ch_t90")):
        d = (cc[a] - cc[b]).abs().dropna()
        print("  |Δ| %-9s 中位 %.4f  p90 %.4f  max %.4f" % (a, d.median(), d.quantile(.9), d.max()))
    sub = cc[cc.z_jump > 0]
    print("  仅加载类（n=%d）：|Δ z_at_02| 中位 %.4f；|Δ t90| 中位 %.3f s"
          % (len(sub), (sub.z_at_02 - sub.ch_at_02).abs().median(),
             (sub.z_t90 - sub.ch_t90).abs().median()))

    # ③ 与第一轮 events.csv 对表
    fp = os.path.join(C.ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results",
                      "events.csv")
    r1 = pd.read_csv(fp)
    r1["key"] = r1.rec.map(MAP)
    out = []
    for _, q in r1.iterrows():
        s = cc[(cc.key == q.key) & ((cc.t_on - q.t_edge).abs() <= 0.5)]
        if len(s) == 0:
            continue
        s = s.iloc[(s.t_on - q.t_edge).abs().argmin()]
        out.append(dict(rec=q.rec, kind_r1=q.kind, kind_t4a=s.kind, t_edge_r1=q.t_edge,
                        t_on_t4a=s.t_on, d_t=round(float(s.t_on - q.t_edge), 3),
                        r1_step_frac=q.step_frac, t4a_z_at_02=s.z_at_02, ch_at_02=s.ch_at_02,
                        r1_frac1=q.frac1, t4a_z_at_10=s.z_at_10,
                        r1_fast_t90=q.fast_t90, t4a_t90=s.z_t90, t4a_ch_t90=s.ch_t90))
    m = pd.DataFrame(out)
    m.to_csv(os.path.join(C.RES, "t4a_vs_firstround.csv"), index=False, encoding="utf-8-sig")
    print("\n== ③ 与第一轮 events.csv 匹配上 %d / %d 个事件 ==" % (len(m), len(r1)))
    if len(m):
        for a, b, nm in (("r1_step_frac", "t4a_z_at_02", "0.2 s 完成度(0~1)"),
                         ("r1_step_frac", "ch_at_02", "0.2 s 完成度: 第一轮 vs 本任务主通道"),
                         ("r1_frac1", "t4a_z_at_10", "1.0 s 完成度(0~1)")):
            dd = (m[a] / 100.0 - m[b]).dropna()
            print("  %-34s n=%d 中位差 %+.4f  |差|中位 %.4f  max %.4f"
                  % (nm, len(dd), dd.median(), dd.abs().median(), dd.abs().max()))
        print("\n  加载类子集（第一轮 kind=onset/restep）:")
        for kd in ("onset", "restep"):
            s = m[(m.kind_r1 == kd)]
            if not len(s):
                continue
            print("    %-7s n=%2d  第一轮 0.2s 完成度 中位 %.1f%%  本任务(Z) %.1f%%  本任务(主通道) %.1f%%"
                  % (kd, len(s), s.r1_step_frac.median(), 100 * s.t4a_z_at_02.median(),
                     100 * s.ch_at_02.median()))
            print("            第一轮 fast_t90 中位 %.2f s  本任务 t90(Z) %.2f s  主通道 %.2f s"
                  % (s.r1_fast_t90.median(), s.t4a_t90.median(), s.t4a_ch_t90.median()))
        print("\n  逐事件明细（前 30）:")
        print(m.head(30).round(3).to_string(index=False))
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
