# -*- coding: utf-8 -*-
"""t4b_r0_recon.py —— T4-B 第 0 步：数据可达性冒烟 + 载荷量级普查（T4-Q9 的原始素材）。

做三件事（全部只读）：
  ① 用 t4b_ad_lib 自动定位 ##Data 读全部 13 份录制，打印帧数/时长/包间隔/重复时间戳数，
     并与 00_共享/数据与脚本复用清单.md §1 的帧数对表（不一致立刻可见）；
  ② 对每份录制做**平台电平普查**：因果去噪后用"停留时长 ≥ 2 s 且变化 < 5%·记录峰值"的
     平台识别，列出所有平台电平 → 回答 T4-Q9「有没有不同载荷量级的加载」；
  ③ 列出加载沿的幅度清单（跨录制），看幅度是否跨量级。

产物：results/t4b_recon.csv（每录制）、results/t4b_plateaus.csv（每平台）、
      results/_t4b_r0.log（stdout 存档，含运行命令）。

运行：python scripts/t4b_r0_recon.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # T4_*/scripts
TASK = os.path.dirname(HERE)                                # T4_*
PLAN = os.path.dirname(TASK)                                # plan/v1.0
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))  # 仓库根
TEMP = os.path.join(ROOT, "temp")
RES = os.path.join(TASK, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, HERE)

import t4b_ad_lib as L  # noqa: E402

B = os.path.join(TEMP, "变化负载")
RECS = [
    ("右拇指指尖/数据1", os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv"), "显示域", 16356),
    ("右拇指指尖/数据2", os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"), "显示域", 23537),
    ("右拇指指尖/数据3", os.path.join(TEMP, "右拇指指尖", "数据3", "device_001_seg000.csv"), "显示域", 19138),
    ("左拇指指尖/数据1", os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv"), "显示域", 19740),
    ("左拇指指尖/数据2", os.path.join(TEMP, "左拇指指尖", "数据2", "device_001_seg000.csv"), "显示域", 19975),
    ("左拇指指尖/数据3", os.path.join(TEMP, "左拇指指尖", "数据3", "device_001_seg000.csv"), "显示域", 19402),
    ("四指指尖/数据1", os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv"), "显示域", 19140),
    ("四指指尖/数据2", os.path.join(TEMP, "四指指尖", "数据2", "device_001_seg000.csv"), "显示域", 19114),
    ("四指指尖/数据3", os.path.join(TEMP, "四指指尖", "数据3", "device_001_seg000.csv"), "显示域", 20033),
    ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                        "20260917_133923_single_device_ee20bc",
                                        "device_001_seg000.csv"), "ADC域", 25693),
    ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载",
                                "device_001_seg000.csv"), "ADC域", 7129),
    ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "device_001_seg000.csv"), "ADC域", 6421),
    ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv"), "ADC域", 12121),
]


def plateaus(tot, tu, dt, peak, min_dur=2.0, tol_frac=0.05, smooth_s=0.30):
    """因果平滑后找平台：(电平, 起, 止, 时长)；容忍度 = tol_frac × 记录峰值。"""
    z = L.med_smooth(tot, max(1, int(smooth_s / dt)), causal=True)
    tol = tol_frac * peak
    out = []
    i, n = 0, len(z)
    while i < n:
        j = i
        lvl = z[i]
        while j + 1 < n and abs(z[j + 1] - lvl) <= tol:
            j += 1
            lvl = np.median(z[i:j + 1])
        if (j - i) * dt >= min_dur:
            out.append(dict(level=float(np.median(z[i:j + 1])), t0=float(tu[i]),
                            t1=float(tu[j]), dur=float((j - i) * dt)))
            i = j + 1
        else:
            i = i + 1
    return out


def main():
    rows, plats, edges = [], [], []
    for name, path, dom, n_exp in RECS:
        if not os.path.isfile(path):
            print("!! 缺文件:", path)
            continue
        d = L.prep(path)
        tu, tot, dt, span = d["tu"], d["tot"], d["dtm"], d["span"]
        peak = float(tot.max())
        # 平台
        ps = plateaus(tot, tu, dt, peak)
        lvls = sorted({round(p["level"] / max(peak, 1e-9), 3) for p in ps})
        # 加载沿（因果检测：0.2 s 短滞后电平差）
        z = L.med_smooth(tot, max(1, int(0.15 / dt)), causal=True)
        w = max(1, int(0.20 / dt))
        g = max(1, int(0.35 / dt))
        dstat = np.zeros_like(z)
        dstat[w + g:] = z[w + g:] - z[:-w - g]
        thr = max(0.05 * peak, 5 * 1.4826 * np.median(np.abs(dstat - np.median(dstat))))
        cand = np.where(dstat > thr)[0]
        last = -10 ** 9
        for i in cand:
            if i - last < int(1.5 / dt):
                continue
            last = i
            a, b = max(1, i - int(0.6 / dt)), min(len(z) - 1, i + 1)
            k = a + int(np.argmax(np.diff(z[a:b])))
            pre = float(np.median(z[max(0, k - int(1.0 / dt)):max(1, k - int(0.15 / dt))]))
            A5 = float(np.median(z[min(len(z) - 1, k + int(4.8 / dt)):min(len(z), k + int(5.2 / dt))]) - pre)
            edges.append(dict(rec=name, t=round(float(tu[k]), 3), pre=round(pre, 2),
                              A5=round(A5, 2),
                              pre_frac=round(pre / max(peak, 1e-9), 4),
                              A5_frac=round(A5 / max(peak, 1e-9), 4)))
        for p in ps:
            plats.append(dict(rec=name, dom=dom, level=round(p["level"], 3),
                              level_frac_peak=round(p["level"] / max(peak, 1e-9), 4),
                              t0=round(p["t0"], 2), t1=round(p["t1"], 2), dur=round(p["dur"], 2)))
        rows.append(dict(rec=name, dom=dom, n_frames_expected=n_exp, n_frames=len(d["t"]),
                         n_grid=len(tu), span_s=round(span, 2),
                         pkt_dt_ms=round(1000 * d["pkt_dt"], 2),
                         dup_ts=int(d["dup"]), n_ch=d["X"].shape[1],
                         peak=round(peak, 2), min=round(float(tot.min()), 2),
                         n_plateaus=len(ps), n_levels=len(lvls),
                         levels="|".join("%.3f" % v for v in lvls[:12])))
        print("%-22s 帧 %6d/%6d  时长 %7.2f s  包间隔 %6.2f ms  平台 %2d 档位 %2d"
              % (name, len(d["t"]), n_exp, span, 1000 * d["pkt_dt"], len(ps), len(lvls)))
    rc = pd.DataFrame(rows)
    pl = pd.DataFrame(plats)
    pe = pd.DataFrame(edges)
    rc.to_csv(os.path.join(RES, "t4b_recon.csv"), index=False, encoding="utf-8-sig")
    pl.to_csv(os.path.join(RES, "t4b_plateaus.csv"), index=False, encoding="utf-8-sig")
    pe.to_csv(os.path.join(RES, "t4b_edges_all.csv"), index=False, encoding="utf-8-sig")

    print("\n== 帧数核对（期望值取自 00_共享/数据与脚本复用清单.md §1）==")
    bad = rc[rc.n_frames != rc.n_frames_expected]
    print("  不一致录制数 = %d" % len(bad))
    if len(bad):
        print(bad[["rec", "n_frames", "n_frames_expected"]].to_string(index=False))

    print("\n== 载荷平台档位（T4-Q9 素材）==")
    print(pl.groupby("rec").agg(n_plat=("level", "size"),
                                lvl_min=("level_frac_peak", "min"),
                                lvl_max=("level_frac_peak", "max")).to_string())

    print("\n== 加载沿幅度（相对各录制峰值）==")
    ons = pe[pe.pre_frac < 0.20]
    res = pe[pe.pre_frac >= 0.20]
    print("  onset  n=%2d  A5/peak 中位 %.3f  p10 %.3f  p90 %.3f  极差 %.3f"
          % (len(ons), ons.A5_frac.median(), ons.A5_frac.quantile(.1),
             ons.A5_frac.quantile(.9), ons.A5_frac.max() - ons.A5_frac.min()))
    print("  restep n=%2d  A5/peak 中位 %.3f  p10 %.3f  p90 %.3f  极差 %.3f"
          % (len(res), res.A5_frac.median(), res.A5_frac.quantile(.1),
             res.A5_frac.quantile(.9), res.A5_frac.max() - res.A5_frac.min()))
    print("\n-> results/t4b_recon.csv(%d) / t4b_plateaus.csv(%d) / t4b_edges_all.csv(%d)"
          % (len(rc), len(pl), len(pe)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
