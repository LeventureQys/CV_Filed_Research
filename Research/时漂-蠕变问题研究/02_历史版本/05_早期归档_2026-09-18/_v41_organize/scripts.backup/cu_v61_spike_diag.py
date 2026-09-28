# -*- coding: utf-8 -*-
"""定位用户报告的现象：**切换负载-快相无责** 的阶跃仍带来"尖峰"，而单一负载（恒载 9 组）没有。

做法：跑 v6 原型并逐帧记录全部内部状态（事件类型 / tau / A_hat / inc / target / c_applied /
显示总量 / 原始总量），然后在 (显示 − 原始) 上找"尖峰"：
  尖峰 = 该量出现局部极大，且其后 3 s 内回落 ≥ 其峰值的 30%（= 先冲高再回落，不是单调平台）。

产物：
  results/v61_spike_diag.csv      逐帧转储（切换负载，含内部状态）
  results/v61_spike_list.csv      尖峰台账（时间 / 幅度 / 事件类型 / 判据）
  results/_v61_spike_diag.log     控制台全文
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402

B = os.path.join(TEMP, "变化负载")
VARY = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]
HOLD = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
        for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]


class TraceV6(GLM53v6):
    """逐帧记录内部状态。"""

    def __init__(self, n):
        super().__init__(n)
        self.log = []

    def process(self, ts, v):
        ev_before = self.ev
        st_before = self.state
        out = super().process(ts, v)
        ev = self.ev
        tau = (ts - ev["t0"]) if ev is not None else np.nan
        self.log.append(dict(
            t=float(ts), raw=float(np.sum(v)), out=float(np.sum(out)),
            state=self.state, st_before=st_before,
            kind=(ev["kind"] if ev is not None else
                  (self.kind_log[-1][1] if self.kind_log else "-")),
            tau=float(tau) if np.isfinite(tau) else np.nan,
            A_hat=float(ev["A_hat"]) if ev is not None else np.nan,
            inc=float(np.sum(v) - ev["base"]) if ev is not None else np.nan,
            base=float(ev["base"]) if ev is not None else np.nan,
            base_y=float(ev["base_y"]) if ev is not None else np.nan,
            c_app=float(ev["c_applied"]) if ev is not None else np.nan,
            stalled=bool(ev["stalled"]) if ev is not None else False,
            A_sum=float(self.A.sum()), g=float(self.g),
            new_ev=bool(ev is not None and ev is not ev_before),
        ))
        return out


def spikes(tt, err, dtm, min_amp, drop_frac=0.30, win=3.0, min_prom=0.5):
    """err 上的"尖峰"：局部极大 + 其后 win 秒内回落 ≥ drop_frac×峰值。"""
    n = len(err)
    W = max(2, int(win / dtm))
    res = []
    for k in range(1, n - 1):
        if err[k] < min_amp or err[k] < err[k - 1] or err[k] < err[k + 1]:
            continue
        e = min(n, k + W)
        prom = err[k] - err[k:e].min()
        if prom >= drop_frac * err[k] and prom >= min_prom:
            res.append((float(tt[k]), float(err[k]), float(prom), int(k)))
    # 合并同一个峰附近（±1 s）的重复记录
    out = []
    for r in sorted(res, key=lambda z: -z[2]):
        if all(abs(r[0] - o[0]) > 1.5 for o in out):
            out.append(r)
    return sorted(out, key=lambda z: z[0])


def run(tag, path):
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    c = TraceV6(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    df = pd.DataFrame(c.log)
    df["raw_s"] = L.med_smooth(df["raw"].to_numpy(), 0.5 / dtm)
    df["out_s"] = L.med_smooth(df["out"].to_numpy(), 0.5 / dtm)
    df["err"] = df["out_s"] - df["raw_s"]
    return d, df, c


print("=" * 118)
print("尖峰定位：显示(去毛刺) − 原始(去毛刺) 上的'先冲高再回落'")
print("=" * 118)

rows = []
for tag, path in VARY + HOLD:
    if not os.path.exists(path):
        print(f"[skip] 缺文件 {tag}")
        continue
    kind = "恒载" if tag in [t for t, _ in HOLD] else "实采"
    d, df, c = run(tag, path)
    dtm = d["dtm"]
    amp = float(df["raw_s"].max())
    min_amp = 0.02 * amp
    sp = spikes(df["t"].to_numpy(), df["err"].to_numpy(), dtm, min_amp)
    print(f"\n{tag:>18} [{kind}] 峰值 {amp:,.0f}  epoch {len(c.epoch_t)}  "
          f"err 范围 [{df.err.min():+,.0f}, {df.err.max():+,.0f}]  尖峰数 {len(sp)}")
    for t, v, prom, k in sp[:8]:
        r = df.iloc[k]
        print(f"    t={t:8.2f}s  err={v:+8.0f} ({100*v/amp:+5.1f}%峰值)  回落={prom:7.0f}  "
              f"kind={r.kind:<14s} tau={r.tau:5.2f} inc={r.inc:8.0f} A_hat={r.A_hat:8.0f} "
              f"c_app={r.c_app:+8.0f} stalled={r.stalled}")
    for t, v, prom, k in sp:
        r = df.iloc[k]
        rows.append(dict(dataset=tag, kind=kind, t=t, err=v, err_pct=100 * v / amp,
                         drop=prom, ev_kind=r.kind, tau=r.tau, inc=r.inc,
                         A_hat=r.A_hat, c_app=r.c_app, stalled=bool(r.stalled),
                         raw=r.raw_s, out=r.out_s))
    if tag == "切换负载-快相无责":
        df.to_csv(os.path.join(RES, "v61_spike_diag.csv"), index=False, encoding="utf-8-sig")
        df[["t", "raw_s", "out_s", "err"]].to_csv(
            os.path.join(RES, "v61_spike_diag_curves.csv"), index=False, encoding="utf-8-sig")
        # 事件台账
        print("    事件台账：")
        for (t0, kd) in c.kind_log:
            print(f"      t0={t0:8.2f}s  {kd}")

pd.DataFrame(rows).to_csv(os.path.join(RES, "v61_spike_list.csv"),
                          index=False, encoding="utf-8-sig")
print(f"\n产物：results/v61_spike_diag.csv  results/v61_spike_list.csv")
