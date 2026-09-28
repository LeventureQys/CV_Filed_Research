# -*- coding: utf-8 -*-
"""t4a_01_morphology：T4-Q1 两形态特征表（固定接口名 t4a_morphology.csv，T4-B 直接读）。

主信号：Z = Σ_c ch_c（总量口径）；事件原点 t_on = 检出沿 ±0.20 s 内**最大单帧跳变**所在帧
（与 13-v6-assessment/r2_shapes.py 同法，便于与第一轮对表）。
armed 判定（需求 T4 §3-1）：pre 窗中位电平 ≥ thr × 记录峰值（Z̄ 峰值）⇒ 带载起；thr = 10/20/30%
（armed_10/20/30），主判 arm = 20%；另给 kind_prej（第一轮 r2 的 pre < 0.2·|J| 口径）用于敏感性。

产出：results/t4a_morphology.csv
日志：results/_t4a_01_morphology.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t4a_common as C
import t4a_ad_lib as L

COLS = ["key", "rec", "dom", "main_ch", "t_on", "k_on", "kind", "armed", "armed_10", "armed_20",
        "armed_30", "kind_prej", "clean", "n_other_near", "pre", "post", "jump", "ratio", "peak",
        "pre_over_peak", "pre_over_jump", "t10", "t50", "t80", "t90", "t95",
        "rise_frames_10_90", "n_frames_rise", "frames_to_50", "frames_to_90",
        "step_frame_frac", "top3_frame_frac",
        "z_at_005", "z_at_01", "z_at_02", "z_at_03", "z_at_05", "z_at_10", "z_at_20", "z_at_50",
        "exp_tau", "exp_beta", "exp_rms", "pow_p", "pow_tau", "pow_rms", "T_slope",
        "t_on5", "d_t_on5", "src_ev", "pkt_dt"]


def build():
    rows = []
    for r in C.RECS:
        tu, Xu, pkt, nraw = C.load_grid(r)
        Z = Xu.sum(axis=1)
        Zs = L.med_smooth(Z, C.KSM)
        amp = float(np.percentile(Z, 99.5) - np.percentile(Z, 0.5))
        peak = float(Zs.max())
        ych = Xu[:, r["ch"]]
        ks, di = C.detect_events(tu, Z, Zs, ych,
                                 float(np.percentile(ych, 99.5) - np.percentile(ych, 0.5)))
        src = {k: "t4a" for k in ks}
        # 与第一轮事件表并集（只读引用）：距已有事件 ≥0.6 s 才补入；t_on 仍在 Z 上重定位
        for t1 in C.round1_times(r["rec"]):
            j = int(round(t1 / C.DT))
            if all(abs(tu[j] - tu[k]) >= 0.6 for k in ks):
                k2 = C.refine_jump(Z, j)
                if k2 not in src:
                    src[k2] = "r1_union"
                    ks = sorted(set(ks + [k2]))
        for k in ks:
            m = C.event_metrics(tu, Z, Zs, k, peak, amp, pkt, ks)
            m.update(key=r["key"], rec=r["rec"], dom=r["dom"], main_ch=r["ch"], src_ev=src[k])
            rows.append(m)
    df = pd.DataFrame(rows)
    kinds, arms, kp = [], [], []
    for x in df.to_dict("records"):
        kd, ar = C.classify(x, 0.20)
        kinds.append(kd)
        arms.append(ar)
        kp.append("onset" if x["pre_over_jump"] < 0.20 else ("restep" if x["jump"] > 0 else "unload"))
    df["kind"] = kinds
    df["armed"] = arms
    for t in (10, 20, 30):
        df["armed_%d" % t] = df["pre_over_peak"] >= t / 100.0
    df["kind_prej"] = kp
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS]


def main():
    C.start_log()
    df = build()
    out = os.path.join(C.RES, "t4a_morphology.csv")
    df.round(6).to_csv(out, index=False, encoding="utf-8-sig")
    print("== 检出事件 n=%d（13 份录制，总量 Z 口径）==" % len(df))
    print(df.groupby("kind").size().to_string())
    print("\n== armed 门限敏感性（统一到「加载/卸载」与「是否带载起」）==")
    for t in (10, 20, 30):
        sub = df[df.jump > 0]
        print("  thr=%2d%%  onset n=%2d  restep n=%2d  |  与 thr=20%% 标签不一致 %d 个"
              % (t, int((~sub["armed_%d" % t]).sum()), int(sub["armed_%d" % t].sum()),
                 int((sub["armed_%d" % t] != sub["armed_20"]).sum())))
    ld = df[df.jump > 0]
    print("\n== 加载类事件按 armed_20 分组的形态中位（真形态对比，未剔除脏事件）==")
    g = ld.groupby("kind")[["t50", "t80", "t90", "t95", "z_at_02", "z_at_10", "step_frame_frac",
                            "top3_frame_frac", "rise_frames_10_90", "exp_tau", "exp_beta",
                            "pow_p", "T_slope", "pre_over_peak"]].median()
    print(g.to_string())
    print("\n== 干净事件的同一对比（指标字典 §2.1）==")
    cl = ld[ld.clean]
    print("  clean: onset n=%d  restep n=%d"
          % (int((cl.kind == "onset").sum()), int((cl.kind == "restep").sum())))
    print(cl.groupby("kind")[["t50", "t80", "t90", "t95", "z_at_02", "z_at_10", "step_frame_frac",
                              "top3_frame_frac", "rise_frames_10_90", "exp_tau", "exp_beta",
                              "pow_p", "T_slope"]].median().to_string())
    print("\n== 标签口径不一致事件（armed_20 vs 第一轮 kind_prej）==")
    bad = ld[ld.kind != ld.kind_prej]
    print("  n=%d" % len(bad))
    if len(bad):
        print(bad[["rec", "t_on", "kind", "kind_prej", "pre", "jump", "pre_over_peak",
                   "pre_over_jump", "z_at_02"]].round(3).to_string(index=False))
    print("\n= 列名（T4-B 接口）=")
    print("  " + ", ".join(COLS))
    print("\n-> %s" % out)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
