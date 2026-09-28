# -*- coding: utf-8 -*-
"""T6-01 盘点与计时：13 份录制的帧/包结构、域、真沿（legacy 口径）、三实现单跑耗时。

为什么先跑它：
  ① 时序抖动必须**按包平移**（指尖 ~16.7 ms/包、实录 ~40 ms/包），所以先量出每份录制的
     真实包结构（唯一 timestamp 数 / 每包帧数 / 包周期）；
  ② 抖动实验要 30+ 次不同种子 × 多条件 × 3 实现，必须先量单跑墙钟，才能定"跑多长的段"；
  ③ 真沿 t_on 用既有第一轮口径 `pv_common.first_onset`（也是 b_repeat_*.csv 的口径），
     保证 L1/L2 数字能与第一轮对拍。

产出：results/t6_probe.csv、results/_t6_01_probe.log
"""
import io
import json
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
sys.path.insert(0, HERE)

import t6_ad_lib as L                                              # noqa: E402
import t6_common as C                                              # noqa: E402
from t6_ad_lib import load_rec, med_smooth                         # noqa: E402
from t6_glm53_v51 import GLM53v51                                  # noqa: E402
from t6_glm53_v6 import GLM53v6                                    # noqa: E402
from t6_glm53_v61 import GLM53v61                                  # noqa: E402

RES = os.path.join(TASK, "results")
os.makedirs(RES, exist_ok=True)

CJ = os.path.join(TEMP, "变化负载")
HOLD = {
    "右拇指指尖/数据1": os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv"),
    "右拇指指尖/数据2": os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"),
    "右拇指指尖/数据3": os.path.join(TEMP, "右拇指指尖", "数据3", "device_001_seg000.csv"),
    "左拇指指尖/数据1": os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv"),
    "左拇指指尖/数据2": os.path.join(TEMP, "左拇指指尖", "数据2", "device_001_seg000.csv"),
    "左拇指指尖/数据3": os.path.join(TEMP, "左拇指指尖", "数据3", "device_001_seg000.csv"),
    "四指指尖/数据1": os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv"),
    "四指指尖/数据2": os.path.join(TEMP, "四指指尖", "数据2", "device_001_seg000.csv"),
    "四指指尖/数据3": os.path.join(TEMP, "四指指尖", "数据3", "device_001_seg000.csv"),
}
VARY = {
    "切换负载-快相无责": os.path.join(CJ, "切换负载-快相无责的测试",
                                      "20260917_133923_single_device_ee20bc",
                                      "device_001_seg000.csv"),
    "零负载-切换负载-零负载-再切换负载": os.path.join(
        CJ, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv"),
    "中途切换-1d9493": os.path.join(CJ, "零负载-中途切换负载-零负载-切换负载",
                                    "device_001_seg000.csv"),
    "中途切换-最终目标-13ffca": os.path.join(CJ, "零负载-中途切换负载-零负载-切换负载",
                                             "最终测试目标", "device_001_seg000.csv"),
}
ALL = dict(HOLD)
ALL.update(VARY)
KIND = {k: "恒载" for k in HOLD}
KIND.update({k: "实采" for k in VARY})

LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def first_onset(tu, tot, dtm):
    """既有第一轮口径（pv_common.first_onset，逐字复刻）——真沿 t_on 与 pre 电平。"""
    peak = float(np.percentile(tot, 99.5))
    idx = np.where(tot > 0.5 * peak)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    pre = float(np.median(tot[max(0, i - int(1.5 / dtm)):max(1, i - int(0.3 / dtm))]))
    j = i
    while j > 0 and tot[j] > pre + 0.05 * (tot[i] - pre):
        j -= 1
    return j + 1, pre


def packet_stats(t_raw):
    """真实包结构：判新包 = 与前一帧时间差 > 1 ms（同一包内实测 ~1e-5 s）。"""
    pid, nk, P, fpp = C.packet_layout(t_raw)
    return dict(n_pkt=int(nk), pkt_dt_med=float(P), frames_per_pkt=fpp,
                n_uniq=int(len(np.unique(t_raw))),
                dup_frac=float(np.mean(np.diff(t_raw) <= 1e-3)))


def domain_of(path):
    sj = os.path.join(os.path.dirname(path), "session.json")
    try:
        with io.open(sj, "r", encoding="utf-8") as f:
            j = json.load(f)
        return str(j.get("value_stage", "?")), str(j.get("calibration", {}).get("display_mode", "?"))
    except Exception:                                                  # noqa: BLE001
        return "?", "?"


def to_grid(t, X, fs=100.0):
    span = t[-1] - t[0]
    tu = np.arange(0.0, span, 1.0 / fs)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu


def timed_run(Cls, tu, Xu):
    c = Cls(Xu.shape[1])
    t0 = time.time()
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return time.time() - t0, Y, c


def main():
    rows = []
    for tag, path in ALL.items():
        t, X = load_rec(path)
        ps = packet_stats(t)
        tu, Xu = to_grid(t, X)
        vs, dm = domain_of(path)
        tot = Xu.sum(axis=1)
        fo = first_onset(tu, tot, tu[1] - tu[0])
        rows.append(dict(tag=tag, kind=KIND[tag], n_frames=len(t), nch=X.shape[1],
                         span_s=round(float(t[-1] - t[0]), 2),
                         value_stage=vs, display_mode=dm,
                         t_edge_s=round(float(tu[fo[0]]), 3) if fo else np.nan,
                         pre=round(fo[1], 3) if fo else np.nan,
                         **{k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in ps.items()}))
        p(f"{tag:>30} [{KIND[tag]}] n={len(t):6d} ch={X.shape[1]:2d} span={t[-1]-t[0]:7.2f}s "
          f"pkt={ps['n_pkt']:5d} pkt_dt={ps['pkt_dt_med']*1000:5.2f}ms "
          f"frames/pkt={ps['frames_per_pkt']:.2f} dup={ps['dup_frac']:.3f} "
          f"dom={dm} t_edge={tu[fo[0]] if fo else float('nan'):.2f}s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t6_probe.csv"), index=False, encoding="utf-8-sig")

    # ── 计时：抖动实验的场景候选（取抖动实验要用的段） ──
    p("")
    p("=== 计时（抖动实验的场景候选，逐段）===")
    cands = [("中途切换-1d9493", VARY["中途切换-1d9493"], None),
             ("四指指尖/数据1", HOLD["四指指尖/数据1"], 70.0),
             ("切换负载-快相无责", VARY["切换负载-快相无责"], 70.0)]
    tr = []
    for tag, path, tmax in cands:
        t, X = load_rec(path)
        m = t <= (t[-1] + 1.0 if tmax is None else tmax)
        t, X = t[m], X[m]
        tu, Xu = to_grid(t, X)
        for nm, Cls in (("v5.1", GLM53v51), ("v6", GLM53v6), ("v6.1", GLM53v61)):
            dt_s, _, _ = timed_run(Cls, tu, Xu)
            tr.append(dict(scene=tag, n_frames=len(tu), impl=nm, secs=round(dt_s, 2)))
            p(f"  {tag:>22} n={len(tu):5d} {nm:>5} {dt_s:6.2f}s")
    pd.DataFrame(tr).to_csv(os.path.join(RES, "t6_probe_timing.csv"), index=False,
                            encoding="utf-8-sig")
    with io.open(os.path.join(RES, "_t6_01_probe.log"), "w", encoding="utf-8") as f:
        f.write("T6-01 probe log\nROOT=" + ROOT + "\n\n" + "\n".join(LOG) + "\n")
    p("\n-> results/t6_probe.csv / t6_probe_timing.csv / _t6_01_probe.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
