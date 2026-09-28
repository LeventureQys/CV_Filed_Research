# -*- coding: utf-8 -*-
"""v3.4 域外综合测试：B 组原始数据（无算法录制，seg000 即输入）+ 算法数据集。

每个会话回放 mem OFF / ON 两臂，量测：
  A) 抗干扰：输入安静段（滞回平台内）的显示 std / 输入 std（抑制比 <1 为降噪）；
     显示无输入支撑的大幅游走（|d out| > 3%·电平 且 |d in| < 1%·电平，滑窗 0.5s）。
  B) 一致性：把输入平台按电平聚类（5% 容差），同类平台的显示中位极差。
  C) C 限幅不变量。
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "build", "v34_runner.exe")


def run_stream(ts, V, extra):
    n = V.shape[1]
    lines = [str(n)]
    for i in range(len(ts)):
        lines.append("%.6f " % ts[i] + " ".join("%.1f" % x for x in V[i]))
    p = subprocess.run([RUNNER] + list(extra), input="\n".join(lines),
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=HERE)
    rows = []
    for ln in p.stdout.splitlines():
        f = ln.split()
        if not f or f[0] in ("t", "OK", "END", "CH"):
            continue
        try:
            rows.append([float(x) for x in f])
        except ValueError:
            pass
    arr = np.array(rows)
    return arr[:, 1], arr[:, 2]          # sum_in, sum_out


def plateaus(el, s, lo_frac=0.15, hi_frac=0.85, min_frames=150):
    rng = s.max() - s.min()
    if rng <= 0:
        return []
    lo, hi = s.min() + lo_frac * rng, s.min() + hi_frac * rng
    segs = []
    st = None
    for i in range(1, len(s)):
        if st is None and s[i] > hi:
            st = i
        elif st is not None and s[i] < lo:
            if i - st > min_frames:
                segs.append((st, i))
            st = None
    if st is not None and len(s) - st > min_frames:
        segs.append((st, len(s)))
    return segs


def analyze(tag, el, sin, sout):
    lv = max(sin.max(), 1.0)
    segs = plateaus(el, sin)
    out = {"tag": tag, "n_segs": len(segs), "quiet": None, "groups": [],
           "exc": 0, "exc_max": 0.0, "exc_max_adc": 0.0}
    # A) 安静段噪声
    ratios = []
    for a, b in segs:
        # 段内去掉首尾 0.5s
        w0 = a + 50
        w1 = b - 50
        if w1 - w0 < 100:
            continue
        si = np.std(sin[w0:w1])
        so = np.std(sout[w0:w1])
        if si > 1.0:
            ratios.append(so / si)
    if ratios:
        out["quiet"] = float(np.median(ratios))
    # A2) 显示无支撑游走（0.5s 滑窗）
    k = 50
    di = np.abs(np.convolve(sin, np.ones(k) / k, "valid"))
    do = np.abs(np.convolve(sout, np.ones(k) / k, "valid"))
    base = np.convolve(sin, np.ones(k) / k, "valid")
    m = di < 0.01 * base
    if m.any():
        exc = do[m] - 0.01 * base[m]
        exc = exc[exc > 0]
        out["exc"] = int(np.sum(exc > 0.02 * base[m][:len(exc)] if len(exc) else []))
        out["exc_max"] = float(exc.max() / lv) if len(exc) else 0.0
        out["exc_max_adc"] = float(exc.max()) if len(exc) else 0.0
    # B) 平台电平聚类一致性
    meds_in, meds_out = [], []
    for a, b in segs:
        meds_in.append(float(np.median(sin[a:b])))
        meds_out.append(float(np.median(sout[a:b])))
    if meds_in:
        groups = []                       # [(level, [display...])]
        for mi, mo in zip(meds_in, meds_out):
            for g in groups:
                if abs(mi - g[0]) < 0.05 * max(mi, g[0]):
                    g[1].append(mo)
                    break
            else:
                groups.append([mi, [mo]])
        for g in groups:
            if len(g[1]) >= 2:
                out["groups"].append((g[0], len(g[1]),
                                      max(g[1]) - min(g[1]),
                                      float(np.std(g[1]))))
    return out


def fmt(r):
    q = ("%.2f" % r["quiet"]) if r["quiet"] is not None else "  - "
    g = "; ".join("L=%.0f n=%d 极差=%.0f std=%.0f" % (lv, n, rg, sd)
                  for lv, n, rg, sd in r["groups"]) or "无重复电平组"
    return ("平台%2d 抑制比%s 游走帧%4d(最大%.0f ADC) | %s" %
            (r["n_segs"], q, r["exc"], r["exc_max_adc"], g))


def load_input(ds_dir):
    """算法数据集用 pre（算法前读数）；B 组无 pre，用 seg000（原始输入）。"""
    pre = os.path.join(ds_dir, "device_001_pre_seg0.csv")
    name = "device_001_pre_seg0.csv" if os.path.isfile(pre) \
        else "device_001_seg000.csv"
    s = L.load_stream(ds_dir, name)
    return name, s


def main():
    roots = [L.DATA_ROOT, os.path.join(L.ROOT, "temp", "原始数据only")]
    rows = []
    for root in roots:
        for label, d in L.discover_sessions(root, 5):
            name, s = load_input(d)
            el = s["el"]
            sin0, o0 = run_stream(s["ts"], s["V"], ["--mem", "0"])
            _, o1 = run_stream(s["ts"], s["V"], ["--mem", "120"])
            viol = int(np.sum(o1 > sin0 + 0.005 * np.abs(sin0) + 1.0))
            r0 = analyze("OFF", el, sin0, o0)
            r1 = analyze("ON", el, sin0, o1)
            rows.append((label.replace("\\", "/").split("/")[-1], name, r0, r1,
                         viol, len(o0)))
            print("== %s  [输入=%s] 帧%d C越界%d" %
                  (rows[-1][0], name, len(o0), viol))
            print("   OFF: " + fmt(r0))
            print("   ON : " + fmt(r1))


if __name__ == "__main__":
    main()
