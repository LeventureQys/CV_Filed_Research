# -*- coding: utf-8 -*-
"""T5-B / 00 侦察：4 份实录的电平结构、v6 干净 epoch 清单、实录瞬态候选。

产物：results/t5b_recon.csv, results/t5b_clean_epochs.csv, results/t5b_transients.csv
      results/_t5b_00_probe.log
纪律：时间轴只用 `timestamp`（经 t5b_ad_lib.load_rec 自动定位 `##Data`），100 Hz 网格；
      不做 τ=2 s 平滑（只用 0.3 s 中值做电平提取，用于判定平台，不喂算法）。
"""
import os
import sys
import io
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
RES = os.path.join(TASK, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, HERE)

from t5b_a_common import REC, load_uniform, run_traced, TRACED_V6   # noqa: E402
from t5b_ad_lib import med_smooth, find_transients                  # noqa: E402


def plateau_profile(tu, Z):
    """把录制按"总量电平"粗分段，返回段表（用于识别 ~1w ADC 平台等）。"""
    Zs = med_smooth(Z, 30)          # 0.3 s 中值：仅用于电平提取
    n = len(Zs)
    segs = []
    i = 0
    win = 100                       # 1 s 判决窗
    while i < n:
        j = min(n, i + win)
        lvl = float(np.median(Zs[i:j]))
        if segs and abs(lvl - segs[-1]["lvl"]) < 0.05 * max(abs(segs[-1]["lvl"]), 1.0):
            segs[-1]["i1"] = j
            segs[-1]["lvl"] = float(np.median(Zs[segs[-1]["i0"]:j]))
        else:
            segs.append(dict(i0=i, i1=j, lvl=lvl))
        i = j
    out = []
    for s in segs:
        s["dur_s"] = (s["i1"] - s["i0"]) * (tu[1] - tu[0])
        if s["dur_s"] >= 2.0:
            out.append(dict(t0=float(tu[s["i0"]]), t1=float(tu[min(s["i1"], n - 1)]),
                            dur_s=round(s["dur_s"], 2), level=round(s["lvl"], 1)))
    return out


def main():
    rows, ep_rows, tr_rows, seg_rows = [], [], [], []
    for name, path in REC.items():
        if not os.path.exists(path):
            print(f"[缺] {name}: {path}")
            continue
        d = load_uniform(path)
        tu, Xu, t = d["tu"], d["Xu"], d["t"]
        Z = Xu.sum(axis=1)
        Zs = med_smooth(Z, 30)
        segs = plateau_profile(tu, Z)
        for s in segs:
            seg_rows.append(dict(rec=name, **s))
        rows.append(dict(
            rec=name, n_frames=len(t), n_ch=Xu.shape[1], span_s=round(d["span"], 2),
            grid_n=len(tu),
            pkt_dt_s=round(float(np.median(np.diff(np.unique(np.round(t, 4))))), 4),
            Z_med=round(float(np.median(Zs)), 1),
            Z_p05=round(float(np.percentile(Zs, 5)), 1),
            Z_p50=round(float(np.percentile(Zs, 50)), 1),
            Z_p95=round(float(np.percentile(Zs, 95)), 1),
            Z_max=round(float(Zs.max()), 1),
            n_plateau=len(segs),
            n_near_1w=int(sum(1 for s in segs if 7000 <= s["level"] <= 14000)),
            frac_near_1w=round(float(np.mean((Zs >= 7000) & (Zs <= 14000))), 4),
            n_below_500=round(float(np.mean(Zs < 500)), 4),
        ))
        # 干净运行：v6 + 仪表化
        r = run_traced(TRACED_V6, tu, Xu)
        print(f"[{name}] 帧={len(t)} ch={Xu.shape[1]} 平台段={len(segs)} "
              f"epoch={len(r['epoch'])} revoke={len(r['revoke'])} "
              f"handoff={len(r['handoff'])} unload={len(r['unload'])}", flush=True)
        for e in r["epoch"]:
            ep_rows.append(dict(rec=name, t0=round(e[0], 3), kind=e[1], t_det=round(e[2], 3),
                                base=round(e[3], 1), v0_sum=round(e[4], 1),
                                lvl_pre=round(float(Zs[min(len(Zs) - 1, max(0, int(e[0] / 0.01) - 100))]), 1)))
        # 实录瞬态候选（真实拍击类）
        for i, peak, back, pre in find_transients(Z, 0.01, up_frac=0.03, rec_s=0.60,
                                                  min_peak=300.0):
            # 空间模式
            i0 = max(0, i - 40)
            dv = Xu[i] - Xu[i0]
            tr_rows.append(dict(rec=name, t=round(float(tu[i]), 3), peak=round(peak, 1),
                                back=round(back, 1), pre_level=round(pre, 1),
                                ratio_peak_pre=round(abs(peak) / max(abs(pre), 1.0), 4),
                                ipk=i))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "t5b_recon.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(ep_rows).to_csv(os.path.join(RES, "t5b_clean_epochs.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(tr_rows).to_csv(os.path.join(RES, "t5b_transients.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(seg_rows).to_csv(os.path.join(RES, "t5b_plateaus.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 录制概览 ===")
    print(pd.DataFrame(rows).to_string())
    print("\n=== 干净 epoch（前 40）===")
    print(pd.DataFrame(ep_rows).head(40).to_string())
    print(f"\n实录瞬态候选 n={len(tr_rows)}")
    print(pd.DataFrame(tr_rows).head(40).to_string())
    print("\n=== 平台段（level >= 7000 的）===")
    ps = pd.DataFrame(seg_rows)
    if len(ps):
        print(ps[(ps.level >= 7000)].to_string())


if __name__ == "__main__":
    main()
