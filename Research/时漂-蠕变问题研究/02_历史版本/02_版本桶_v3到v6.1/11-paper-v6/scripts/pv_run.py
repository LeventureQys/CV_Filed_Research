# -*- coding: utf-8 -*-
"""paper_v6 复算入口：13 份录制 × 5 臂（raw / 无责1s / 无责3s / v6 / v6+trim）。

产出（paper_v6/results/）：
  metrics_all.csv        逐数据集 × 逐臂的全部指标
  metrics_settle.csv     首次加载的 T_band / T_stable / err5s
  epochs_all.csv         每个 epoch 的起点与工况种类（事件台账）
  cache/<tag>.npz        逐帧曲线（降采样）+ 事件台账 + v6 单事件内部状态轨迹
  paper_tables.txt       正文引用的汇总表（恒载 9 组 / 实采 4 份）

用法：python pv_run.py [--only 数据集关键字]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_common as C                                       # noqa: E402
import ad_lib as L                                          # noqa: E402
from glm53_v51 import GLM53v51                              # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402

DS = 4                      # 缓存降采样步长（≈25 Hz 落盘）
TRACE_EV_MAX = 12           # 每份数据最多留 12 个事件的内部轨迹


class RawStub:
    """raw 臂的占位状态：无补偿、无 epoch、A/g 记为 0（指标里 NaN 更诚实）。"""

    def __init__(self, n):
        self.A = np.zeros(n)
        self.g = 0.0
        self.epoch_t = []
        self.kind_log = []


class V6Trace(GLM53v6):
    """记录每个事件的逐帧内部状态（供论文画"滑行/停滞/交接"轨迹）。"""

    def __init__(self, n):
        super().__init__(n)
        self.traces = {}

    def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
        super()._new_event(t0, base, v0, y0, kind, hist, prev_state)
        if len(self.traces) < TRACE_EV_MAX:
            self.traces[round(float(t0), 4)] = []

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        key = round(float(ev["t0"]), 4) if ev is not None else None
        out = super()._run_event(ts, v, total, dt, eps, idle_now)
        if key is not None and key in self.traces:
            self.traces[key].append(dict(
                t=float(ts), tau=float(ts - key), inc=float(total - ev["base"]),
                A_hat=float(ev["A_hat"]), c_app=float(ev["c_applied"]),
                stalled=bool(ev["stalled"]), kind=ev["kind"],
                target=float(ev["base_y"] + ev["A_hat"]), out=float(np.sum(out))))
        return out


class V5Trace(GLM53v51):
    """v5.1 的 epoch 起点记录（_begin / _restep 各记一次），与 ci_v6_vs_1s_3s.py 同口径。"""

    def __init__(self, n):
        super().__init__(n)
        self.epoch_t = []
        self.kind_log = []

    def _begin(self, ts):
        super()._begin(ts)
        self.epoch_t.append(float(ts))
        self.kind_log.append((float(ts), "v5_begin"))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.epoch_t.append(float(ts))
        self.kind_log.append((float(ts), "v5_restep"))


def build(arm, n):
    if arm == "e1s":
        c = V5Trace(n)
        c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = 1.0, 1.0 / 3.0, 1.0
        return c
    if arm == "e3s":
        c = V5Trace(n)
        c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = 3.0, 1.0, 3.0
        return c
    if arm == "v6":
        return V6Trace(n)
    if arm == "v6trim":
        c = V6Trace(n)
        c.TRIM_RATE = 0.002
        return c
    raise ValueError(arm)


def run_arm(arm, tu, Xu):
    if arm == "raw":
        return Xu.copy(), RawStub(Xu.shape[1])
    c = build(arm, Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    if arm.startswith("v6"):
        c.A = np.full(Xu.shape[1], c.A_peak)        # 供 hold_metrics 复用 A.max()
    return Y, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    rows, settle, ep_rows = [], [], []
    t_start = time.time()
    for tag, path in C.ALL:
        if args.only and args.only not in tag:
            continue
        if not os.path.exists(path):
            print(f"[skip] 缺文件 {path}")
            continue
        d = L.prep(path)
        tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
        tot_raw = Xu.sum(axis=1)
        tot_s = L.med_smooth(tot_raw, 0.5 / dtm)
        peak = float(tot_raw.max())
        kind = C.KIND[tag]
        fo = C.first_onset(tu, tot_raw, dtm)
        i0 = fo[0] if fo else None
        tgt = float(np.median(tot_raw[i0 + int(4.6 / dtm):i0 + int(5.4 / dtm)])) if fo else np.nan
        step = (tgt - fo[1]) if fo else np.nan
        print(f"{tag:>18} [{kind}] {d['span']:5.1f}s ch={Xu.shape[1]:2d}")

        blob = dict(tu=tu[::DS], Xu=Xu[::DS], dtm=np.array([dtm]), ds=np.array([DS]),
                    tot_s=tot_s[::DS], i0=np.array([-1 if i0 is None else i0]),
                    tgt=np.array([tgt]), step=np.array([step]), n=self_n(Xu))
        for arm in C.ARMS:
            t0 = time.time()
            Y, c = run_arm(arm, tu, Xu)
            mm = (C.hold_metrics(Y, c, tu, Xu, dtm, tot_s, peak) if kind == "恒载"
                  else C.vary_metrics(Y, c, tu, Xu, dtm, tot_s, peak, d))
            mm.update(dataset=tag, kind=kind, algo=arm, span=d["span"], nch=Xu.shape[1],
                      secs=round(time.time() - t0, 1))
            rows.append(mm)
            blob["Y_" + arm] = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)[::DS]
            blob["Yraw_" + arm] = Y[::DS]
            if c is not None:
                blob["epoch_" + arm] = np.array(getattr(c, "epoch_t", []), float)
                blob["kindlog_" + arm] = np.array(
                    [f"{t:.3f}|{k}" for t, k in getattr(c, "kind_log", [])], dtype=object)
                blob["A_" + arm] = np.asarray(c.A, float)
                blob["g_" + arm] = np.array([float(c.g)])
                if arm == "v6":
                    blob["traces"] = np.array([json.dumps(c.traces)], dtype=object)
            if arm == "raw":
                blob["epoch_raw"] = np.array([])
            for t_e, k in getattr(c, "kind_log", []) if c is not None else []:
                ep_rows.append(dict(dataset=tag, kind=kind, algo=arm, t0=float(t_e), event=k))
            if fo:
                tb = C.settle_time(tu, Y.sum(axis=1), i0, tgt, step, dtm)
                ts_ = C.stable_time(tu, Y.sum(axis=1), i0, step, dtm)
                flat = stable_time_flat(tu, Y.sum(axis=1), i0, dtm)
                err5 = float(Y.sum(axis=1)[i0 + int(5.0 / dtm)] / tgt * 100 - 100)
                settle.append(dict(dataset=tag, kind=kind, algo=arm, t_edge=float(tu[i0]),
                                   target=tgt, pre=fo[1], step=step, t_settle=tb,
                                   t_stable=ts_, t_flat30=flat, err5s=err5))
            line = (f"    {arm:>6}  " + (f"全段 {mm['drift_main']:+6.2f}%  慢相段 {mm['drift_slow']:+6.2f}%  "
                                         f"受载 {mm['drift_loaded']:+6.2f}%  epoch {mm['epoch']:3d}"
                                         if kind == "恒载" else
                                         f"全程偏差 {mm['max_gap']:7.0f}  变载窗 {mm['gap_med']:7.0f}  "
                                         f"捕获 {mm['cap_med'] if np.isfinite(mm['cap_med']) else float('nan'):5.2f}  "
                                         f"epoch {mm['epoch']:3d}"))
            print(line + f"   ({time.time() - t0:.1f}s)")
        np.savez_compressed(C.cache_path(tag), **blob)
        print(f"    -> cache {os.path.basename(C.cache_path(tag))}  累计 {time.time() - t_start:.0f}s")

    df = pd.DataFrame(rows)
    st = pd.DataFrame(settle)
    ep = pd.DataFrame(ep_rows)
    C.save_table(df, "metrics_all.csv")
    C.save_table(st, "metrics_settle.csv")
    C.save_table(ep, "epochs_all.csv")
    report(df, st)
    print(f"\n总耗时 {time.time() - t_start:.0f}s")


def self_n(Xu):
    return np.array([Xu.shape[1]])


def stable_time_flat(tu, Ytot, i0, dtm, hold_s=30.0):
    """T_flat30：显示首次进入"此后 30 s 内自身漂移 ≤ 0.5%×该时刻电平"的时刻。
       用于量"平台是否绝对平"（T_stable 的阶跃相对口径在 A 偏大时会一并放大容差）。"""
    n = len(Ytot)
    H = int(hold_s / dtm)
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):
            break
        if np.max(np.abs(Ytot[k:e] - Ytot[k])) <= 0.005 * max(abs(Ytot[k]), 1e-9):
            return float(tu[k] - tu[i0])
    return np.nan


def report(df, st):
    hold = df[df.kind == "恒载"]
    vary = df[df.kind == "实采"]
    lines = []
    A = lines.append
    A("=" * 96)
    A("表 A 恒载 9 组（|·| 均值，除注明）")
    A("=" * 96)
    g = hold.groupby("algo").agg(**{
        "时漂残余_全段%": ("drift_main", lambda s: s.abs().mean()),
        "时漂残余_慢相段%": ("drift_slow", lambda s: s.abs().mean()),
        "受载通道中位%": ("drift_loaded", lambda s: s.abs().mean()),
        "噪声比": ("noise_ratio", "mean"),
        "平坦度%": ("flat", "mean"),
        "阶跃保真": ("step_ratio", "mean"),
        "首扣时延s": ("ded_delay", "mean"),
        "A占幅度": ("a_max", "mean"),
        "g末端": ("g_end", "mean"),
        "epoch数和": ("epoch", "sum"),
    }).reindex(C.ARMS).round(3)
    A(g.to_string())
    A("")
    A("=" * 96)
    A("表 B 实采 4 份")
    A("=" * 96)
    h = vary.groupby("algo").agg(**{
        "全程偏差中位": ("max_gap", "median"),
        "占峰值中位%": ("pct", "median"),
        "变载窗偏差中位": ("gap_med", "median"),
        "捕获比中位": ("cap_med", "mean"),
        "捕获比最小": ("cap_min", "min"),
        "有效事件": ("n_event", "sum"),
        "epoch数和": ("epoch", "sum"),
    }).reindex(C.ARMS).round(3)
    A(h.to_string())
    A("")
    A("逐份全程最大偏差（ADC）")
    A(vary.pivot_table(index="dataset", columns="algo", values="max_gap")
      .reindex(columns=C.ARMS).round(0).to_string())
    A("")
    A("逐份台阶捕获比（中位）")
    A(vary.pivot_table(index="dataset", columns="algo", values="cap_med")
      .reindex(columns=C.ARMS).round(3).to_string())
    A("")
    A("=" * 96)
    A("表 C 首次加载响应（秒；NaN=观测窗内未满足）")
    A("=" * 96)
    A("T_stable（显示首次停下，此后 30 s 自身漂移 ≤5%×阶跃）")
    A(st.pivot_table(index="dataset", columns="algo", values="t_stable")
      .reindex(columns=C.ARMS).round(2).to_string())
    A("")
    A("组中位：")
    A(st.groupby(["kind", "algo"])["t_stable"].median().unstack()
      .reindex(columns=C.ARMS).round(2).to_string())
    A("")
    A("T_band（进入 ±5%×真值带并保持 30 s）")
    A(st.pivot_table(index="dataset", columns="algo", values="t_settle")
      .reindex(columns=C.ARMS).round(2).to_string())
    A("")
    A("T_flat30（进入「此后 30 s 自身漂移 ≤0.5%×电平」的平台）")
    A(st.pivot_table(index="dataset", columns="algo", values="t_flat30")
      .reindex(columns=C.ARMS).round(2).to_string())
    A("")
    A("加载沿 +5 s 相对误差（显示/真值−1，%）")
    A(st[st.kind == "恒载"].assign(a5=lambda x: x.err5s).groupby("algo")["a5"]
      .agg(["median", "min", "max"]).reindex(C.ARMS).round(2).to_string())
    for kind in ("恒载", "实采"):
        A(f"  {kind} |误差| 中位: " + "  ".join(
            f"{a} {st[(st.kind == kind) & (st.algo == a)].err5s.abs().median():.2f}%"
            for a in C.ARMS))
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(C.RES, "paper_tables.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
