# -*- coding: utf-8 -*-
"""T1-B / 00：数据与速度探测（只读，不改任何既有文件）。

目的：
  1) 核对 13 份录制的帧数/通道/时间戳结构（包结构、重复时间戳比例）；
  2) 量测 v6 原型在本机的单帧耗时，用于给 T1-B 的扫描矩阵定规模。

产物：results/t1b_probe.csv、results/_t1b_00_probe.log
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
TEMP = os.path.join(ROOT, "temp")
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t1b_ad_lib import load_rec, med_smooth          # noqa: E402
from t1b_glm53_v6 import GLM53v6                     # noqa: E402


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


REC = {
    "RT1": os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv"),
    "RT2": os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"),
    "LT1": os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv"),
    "F41": os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv"),
    "SW1": os.path.join(TEMP, "变化负载", "切换负载-快相无责的测试",
                        "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"),
    "SW2": os.path.join(TEMP, "变化负载", "零负载-切换负载-零负载-再切换负载",
                        "device_001_seg000.csv"),
    "SW3": os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                        "device_001_seg000.csv"),
    "SW4": os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                        "最终测试目标", "device_001_seg000.csv"),
}


def main():
    log = Tee(os.path.join(RES, "_t1b_00_probe.log"))
    log("=== T1-B probe ===")
    log("python", sys.version.split()[0], "numpy", np.__version__, "pandas", pd.__version__)
    rows = []
    for k, p in REC.items():
        t, X = load_rec(p)
        dt = np.diff(t)
        uniq = np.unique(np.round(t, 4))
        # 包结构：同一时间戳连续出现的次数
        rep = pd.Series(np.round(t, 4)).value_counts()
        row = dict(key=k, path_ok=os.path.exists(p), n=len(t), n_ch=X.shape[1],
                   span_s=round(float(t[-1] - t[0]), 3),
                   fs=round(len(t) / max(t[-1] - t[0], 1e-9), 3),
                   n_unique_ts=int(len(uniq)),
                   dup_frac=round(float(1 - len(uniq) / len(t)), 4),
                   dt_med_ms=round(float(np.median(dt)) * 1000, 3),
                   pkt_frames_med=int(rep.median()), pkt_frames_max=int(rep.max()),
                   tot_med=round(float(np.median(X.sum(axis=1))), 2),
                   tot_max=round(float(X.sum(axis=1).max()), 2),
                   ch_scale_max=round(float(np.abs(X).max()), 4))
        rows.append(row)
        log(f"  {k}: n={row['n']} ch={row['n_ch']} span={row['span_s']} fs={row['fs']} "
            f"uniq_ts={row['n_unique_ts']} dup={row['dup_frac']} dt_med={row['dt_med_ms']}ms "
            f"pkt_frames(med/max)={row['pkt_frames_med']}/{row['pkt_frames_max']} "
            f"tot(med/max)={row['tot_med']}/{row['tot_max']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t1b_probe.csv"), index=False, encoding="utf-8-sig")

    # 速度：v6 单帧耗时（用 3000 帧的窗口）
    log("--- timing v6 (plain, 3000 frames) ---")
    for k in ("RT1", "SW2", "SW4"):
        t, X = load_rec(REC[k])
        n = min(3000, len(t))
        tu = t[:n] - t[0]
        c = GLM53v6(X.shape[1])
        t0 = time.perf_counter()
        for i in range(n):
            c.process(float(tu[i]), X[i])
        el = time.perf_counter() - t0
        log(f"  {k}: {n} frames in {el:.2f}s -> {1000*el/n:.3f} ms/frame, "
            f"full_record_est={el/ n * len(t):.1f}s")
    log.f.close()


if __name__ == "__main__":
    main()
