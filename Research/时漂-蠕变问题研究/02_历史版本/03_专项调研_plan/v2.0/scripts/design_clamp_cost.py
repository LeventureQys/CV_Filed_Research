# -*- coding: utf-8 -*-
"""v2.0 阶段二 · 限幅的真实代价量化：早期超前量与"蠕变抑制"的丢失量。

对每个数据集给出三组量（base vs α=0.005 vs α=0.02）：
  lead_early   = 加载后 0.3~1.2 s 的补偿量（显示 − 原始）中位   ← 限幅削掉多少"该领先"
  sat_time     = 补偿量触顶(≥ α·原始·0.9)的帧占比                ← 限幅是否成了"隐形常量"
  creep_supp   = 长保压段内「原始读数增长」被压掉的比例           ← 蠕变抑制能力
                 = 1 − (显示末−显示首)/(原始末−原始首)
  hold_drift   = 长保压段显示自身的漂移 |Δ显示| / 段内显示中位
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class Arm(P.GLM53v6):
    def __init__(self, n, alpha=None):
        super().__init__(n)
        self.KAPPA_ONSET, self.KAPPA_RESTEP, self.HO_MIN = 1.05, 1.12, 3.5
        self.alpha = alpha
        self.n_sat = 0
        self.n_tot = 0

    def process(self, ts, v):
        raw = np.asarray(v, float).copy()
        out = np.asarray(super().process(ts, raw.copy()), float).copy()
        if self.alpha is not None:
            lim = raw + self.alpha * np.abs(raw)
            self.n_tot += 1
            if (out >= lim).any():
                self.n_sat += 1
            out = np.minimum(out, lim)
        return out


def run(el, V, alpha):
    ch = V.shape[1]
    c = Arm(ch, alpha)
    out = np.empty((len(el), ch))
    for i in range(len(el)):
        out[i] = c.process(float(el[i]), V[i])
    return c, out.sum(1)


def edges(el, tot):
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tot, 3), np.percentile(tot, 97)
    up, _ = L.edges_from_tot(np.convolve(tot, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in up:
        if not tu or el[i] - tu[-1] > 1.0:
            tu.append(float(el[i]))
    return tu


def long_hold_segment(pre_tot, el):
    """最长受载段（pre > 60 分位）"""
    thr = np.percentile(pre_tot, 60)
    m = pre_tot > thr
    best = (0, 0, 0)
    i = 0
    n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        else:
            i += 1
    return best[1], best[2]


def summarize(name, el, pre, arms):
    print(f"\n=== {name} ===")
    tu = edges(el, pre)
    a0, b0 = long_hold_segment(pre, el)
    print(f"  最长受载段 {el[a0]:.1f}~{el[b0-1]:.1f}s（{el[b0-1]-el[a0]:.1f}s）")
    print(f"  {'臂':10s}{'早期超前中位':>12}{'早期超前最小':>12}{'触顶帧%':>9}"
          f"{'蠕变抑制%':>10}{'保压漂移%':>10}")
    for tag, (c, out) in arms.items():
        # 早期超前
        leads = []
        for t in tu:
            m = (el >= t + 0.3) & (el <= t + 1.2)
            if m.any():
                leads.append(float(np.median(out[m] - pre[m])))
        lead_med = np.median(leads) if leads else float("nan")
        lead_min = min(leads) if leads else float("nan")
        sat = 100.0 * c.n_sat / c.n_tot if c.n_tot else float("nan")
        # 蠕变抑制：保压段首末
        dpre = pre[b0 - 1] - pre[a0]
        dout = out[b0 - 1] - out[a0]
        creep_supp = 100.0 * (1 - dout / dpre) if abs(dpre) > 1e-9 else float("nan")
        hold_drift = 100.0 * abs(dout) / max(np.median(out[a0:b0]), 1.0)
        print(f"  {tag:10s}{lead_med:12.0f}{lead_min:12.0f}{sat:8.1f}%"
              f"{creep_supp:9.1f}%{hold_drift:9.2f}%")


def read_simple(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    idx = [i for i, h in enumerate(hdr) if h.startswith("ch") and not h.startswith("ch_")]
    if not idx:
        ncol = len(rows[di + 2].split(","))
        idx = list(range(ncol - L.NCH, ncol))
    el, vals = [], []
    for r in rows[di + 2:]:
        if not r.strip():
            continue
        f = r.split(",")
        el.append(float(f[1]))
        vals.append([float(f[i]) for i in idx])
    return np.asarray(el), np.asarray(vals)


def main():
    d = L.load_dataset(L.DS_ZERO)
    el = d["pre"]["el"]
    V = d["pre"]["V"]
    pre = V.sum(1)
    arms = {tag: run(el, V, a) for tag, a in
            (("base", None), ("α=0.005", 0.005), ("α=0.02", 0.02))}
    summarize("★新录制（ADC 域，唯一同负载反复加减工况）", el, pre, arms)

    base = os.path.join(L.ROOT, "temp", "原始数据only")
    cases = []
    for grp in ("四指指尖", "右拇指指尖", "左拇指指尖"):
        for dd in ("数据1", "数据2", "数据3"):
            p = os.path.join(base, grp, dd, "device_001_seg000.csv")
            if os.path.exists(p):
                cases.append((f"{grp}/{dd}", p))
    for name, path in cases:
        el2, V2 = read_simple(path)
        pre2 = V2.sum(1)
        arms2 = {tag: run(el2, V2, a) for tag, a in
                 (("base", None), ("α=0.005", 0.005), ("α=0.02", 0.02))}
        summarize(f"{name}（力域）", el2, pre2, arms2)


if __name__ == "__main__":
    main()
