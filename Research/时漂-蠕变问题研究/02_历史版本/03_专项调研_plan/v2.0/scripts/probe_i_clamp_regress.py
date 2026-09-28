# -*- coding: utf-8 -*-
"""v2.0 PROBE-I：单侧限幅（显示 ≤ 原始 + α·|原始|）在 13 份既有录制上的不劣化回归。

口径（与 probe_h 一致，且必须显式）：
  受载时漂残余 = (显示 5 s 窗最大 − 显示 5 s 窗最小) / 显示段中位      ← 越小越好（越"平"）
  台阶捕获比 G = (事件后 4~5 s 显示中位 − 事件前显示中位) / 台阶幅度   ← 越接近 1 越好（欠报则 <1）
  超前量峰值   = max(显示 − 原始) / 台阶幅度                          ← 越小越"不过充"
  到带时间 T5  = 显示进入 |显示 − 显示末段稳态| ≤ 5%·台阶 的首个 τ
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
PROTO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTO)


def replay(el, V, alpha):
    n = len(el)
    c = PROTO.GLM53v6(V.shape[1])
    c.KAPPA_ONSET, c.KAPPA_RESTEP, c.HO_MIN = 1.05, 1.12, 3.5
    out = np.empty((n, V.shape[1]))
    for i in range(n):
        y = np.asarray(c.process(float(el[i]), V[i].copy()), float)
        if alpha is not None:
            y = np.minimum(y, V[i] + alpha * np.abs(V[i]))
        out[i] = y
    return out.sum(1)


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


def flatness(el, out_tot, pre_tot, win=5.0):
    """受载段的时漂残余：以 pre 的 60% 分位为受载门限。"""
    thr = np.percentile(pre_tot, 60)
    m = pre_tot > thr
    if m.sum() < 200:
        return float("nan")
    seg = out_tot[m]
    # 滑窗极差中位 / 段中位
    w = int(win * 100)
    if len(seg) <= w:
        return float("nan")
    step = max(w // 4, 1)
    rs = [np.max(seg[i:i + w]) - np.min(seg[i:i + w]) for i in range(0, len(seg) - w, step)]
    return float(np.median(rs) / np.median(seg))


def edges(el, tot):
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tot, 3), np.percentile(tot, 97)
    up, _ = L.edges_from_tot(np.convolve(tot, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in up:
        if not tu or el[i] - tu[-1] > 1.0:
            tu.append(float(el[i]))
    return tu


def ev_metrics(el, pre_tot, out_tot, t_ups):
    rows = []
    for t_up in t_ups:
        m0 = (el >= t_up - 1.2) & (el < t_up - 0.35)
        if m0.sum() < 5:
            continue
        bp = float(np.median(pre_tot[m0]))
        bo = float(np.median(out_tot[m0]))
        t_end = min(t_up + 30.0, el[-1] - 0.2)
        m1 = (el >= t_end - 1.5) & (el <= t_end)
        ps, os_ = float(np.median(pre_tot[m1])), float(np.median(out_tot[m1]))
        A = ps - bp
        if abs(A) < 1e-9:
            continue
        idx = np.where((el >= t_up - 0.05) & (el <= t_end))[0]
        lead = (out_tot[idx] - pre_tot[idx]) / abs(A)
        t5 = float("nan")
        for k in range(len(idx)):
            if abs(out_tot[idx[k]] - os_) <= 0.05 * abs(A):
                t5 = el[idx[k]] - t_up
                break
        G = (float(np.median(out_tot[(el >= t_up + 4.0) & (el <= t_up + 5.0)])) - bo) / A
        rows.append(dict(lead_peak=float(np.max(lead)), t5=t5, G=G))
    return rows


def load_legacy():
    base = os.path.join(L.ROOT, "temp", "原始数据only")
    cases = []
    for grp in ("四指指尖", "右拇指指尖", "左拇指指尖"):
        for d in ("数据1", "数据2", "数据3"):
            p = os.path.join(base, grp, d, "device_001_seg000.csv")
            if os.path.exists(p):
                cases.append((f"{grp}/{d}", p))
    root = os.path.join(base, "变化负载")
    for d in sorted(os.listdir(root)):
        for dp, _dn, fn in os.walk(os.path.join(root, d)):
            if "device_001_seg000.csv" in fn:
                cases.append((f"变化负载/{d}", os.path.join(dp, "device_001_seg000.csv")))
    return cases


def main():
    cases = [("★新录制(ADC域)", os.path.join(L.DS_ZERO, "device_001_pre_seg0.csv"))]
    cases += load_legacy()
    print(f"{'数据集':30s}{'臂':>8}{'时漂残余':>10}{'G中位':>8}{'超前峰中位':>11}"
          f"{'超前峰最大':>11}{'T5%中位':>9}")
    for name, path in cases:
        try:
            el, V = read_simple(path)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:30s}  读取失败 {exc}")
            continue
        p = V.sum(1)
        tu = edges(el, p)
        if not tu:
            print(f"{name:30s}  无加载沿")
            continue
        base_metric = None
        for tag, alpha in (("base", None), ("α=0.005", 0.005), ("α=0.02", 0.02)):
            o = replay(el, V, alpha)
            fl = flatness(el, o, p)
            rows = ev_metrics(el, p, o, tu)
            if not rows:
                print(f"{name:30s}{tag:>8}  无有效事件")
                continue
            lp = [r["lead_peak"] for r in rows]
            ts = [r["t5"] for r in rows if r["t5"] == r["t5"]]
            gs = [r["G"] for r in rows]
            line = (f"{name:30s}{tag:>8}{fl:10.4f}{np.median(gs):8.3f}"
                    f"{np.median(lp):11.3f}{max(lp):11.3f}"
                    f"{(np.median(ts) if ts else float('nan')):9.2f}")
            if tag == "base":
                base_metric = (fl, np.median(gs), np.median(lp))
            elif base_metric is not None:
                dfl = fl - base_metric[0]
                dg = np.median(gs) - base_metric[1]
                dl = np.median(lp) - base_metric[2]
                line += f"   Δ残余={dfl:+.4f} ΔG={dg:+.3f} Δ超前={dl:+.3f}"
            print(line)


if __name__ == "__main__":
    main()
