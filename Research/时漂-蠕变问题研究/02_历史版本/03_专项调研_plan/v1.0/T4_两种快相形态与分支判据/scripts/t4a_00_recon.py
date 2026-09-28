# -*- coding: utf-8 -*-
"""t4a_00_recon：13 份录制的网格/包结构核对 + 总量 Z 事件检测自检 + 与第一轮 events.csv 对表。

产出：results/t4a_recon.csv（每份录制一行）+ results/t4a_recon_events.csv（检出沿清单）
日志：results/_t4a_00_recon.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t4a_common as C
import t4a_ad_lib as L


def main():
    C.start_log()
    rows, ev = [], []
    for r in C.RECS:
        tu, Xu, pkt, nraw = C.load_grid(r)
        Z = Xu.sum(axis=1)
        Zs = L.med_smooth(Z, C.KSM)
        amp = float(np.percentile(Z, 99.5) - np.percentile(Z, 0.5))
        peak = float(Zs.max())
        ks, di = C.detect_events(tu, Z, Zs, Xu[:, r["ch"]], float(np.percentile(Xu[:, r["ch"]], 99.5)
                                                                   - np.percentile(Xu[:, r["ch"]], 0.5)))
        rows.append(dict(key=r["key"], rec=r["rec"], dom=r["dom"], n_raw=nraw, n_ch=Xu.shape[1],
                         n_grid=len(tu), span_s=round(float(tu[-1]), 2), pkt_dt=round(pkt, 4),
                         sigma_d=round(di["sigma_d"], 5), thr_z=round(di["thr_z"], 4),
                         thr_ch=round(di["thr_ch"], 4), sL=round(di["sL"], 5),
                         thr_L=round(di["thr_L"], 4), n_coarse=di["n_coarse"], n_drop=di["n_drop"],
                         Z_min=round(float(Z.min()), 2), Z_med=round(float(np.median(Z)), 2),
                         Z_max=round(float(Z.max()), 2), amp=round(amp, 2), peak=round(peak, 2),
                         n_edge=len(ks)))
        for k in ks:
            m = C.event_metrics(tu, Z, Zs, k, peak, amp, pkt, ks)
            m.update(key=r["key"], rec=r["rec"], dom=r["dom"], main_ch=r["ch"])
            ev.append(m)
        print("%-18s n_raw=%6d nch=%2d span=%7.2fs pkt=%.4fs thr_z=%9.3f thr_ch=%9.3f "
              "thr_L=%9.3f co=%3d drop=%3d amp=%10.2f peak=%10.2f events=%d"
              % (r["key"], nraw, Xu.shape[1], tu[-1], pkt, di["thr_z"], di["thr_ch"], di["thr_L"],
                 di["n_coarse"], di["n_drop"], amp, peak, len(ks)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RES, "t4a_recon.csv"), index=False, encoding="utf-8-sig")
    evd = pd.DataFrame(ev)
    evd["kind"] = [C.classify(x)[0] for x in evd.to_dict("records")]
    evd["armed"] = [C.classify(x)[1] for x in evd.to_dict("records")]
    evd.to_csv(os.path.join(C.RES, "t4a_recon_events.csv"), index=False, encoding="utf-8-sig")
    print("\n== 总量 Z 检出事件 %d 个 ==" % len(evd))
    print(evd.groupby(["kind"]).size().to_string())
    print("\n== 逐事件（前 60）==")
    cols = ["rec", "dom", "kind", "t_on", "pre", "jump", "pre_over_peak",
            "t50", "t90", "z_at_02", "z_at_10", "step_frame_frac", "clean"]
    print(evd[cols].to_string(index=False))

    # ── 与第一轮 events.csv 对表（只读引用）──
    fp = os.path.join(C.ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment",
                      "results", "events.csv")
    if os.path.isfile(fp):
        f1 = pd.read_csv(fp)
        print("\n== 第一轮 events.csv n=%d（主通道口径）==" % len(f1))
        cr = {"右拇指指尖/数据1": "RT1", "右拇指指尖/数据2": "RT2", "右拇指指尖/数据3": "RT3",
              "左拇指指尖/数据1": "LT1", "左拇指指尖/数据2": "LT2", "左拇指指尖/数据3": "LT3",
              "四指指尖/数据1": "F41", "四指指尖/数据2": "F42", "四指指尖/数据3": "F43",
              "切换负载-快相无责": "SW1", "再切换负载": "SW2",
              "中途切换-1d9493": "SW3", "中途切换-13ffca": "SW4"}
        hit, miss = 0, []
        for _, q in f1.iterrows():
            key = cr.get(q["rec"])
            sub = evd[evd.key == key]
            d = (sub.t_on - float(q["t_edge"])).abs() if len(sub) else pd.Series(dtype=float)
            if len(d) and d.min() <= 0.5:
                hit += 1
            else:
                miss.append((q["rec"], q["kind"], q["t_edge"]))
        print("匹配（±0.5 s）%d / %d；未匹配 %d 个：" % (hit, len(f1), len(miss)))
        for m in miss:
            print("   ", m)
    else:
        print("!! 未找到第一轮 events.csv（只读引用失败）")
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
