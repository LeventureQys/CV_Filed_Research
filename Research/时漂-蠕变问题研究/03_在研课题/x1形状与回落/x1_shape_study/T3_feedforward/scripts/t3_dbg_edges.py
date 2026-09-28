# -*- coding: utf-8 -*-
"""临时诊断：复刻 find_edges 内部量，定位剧烈变化工况只出 1 个沿的原因。"""
import numpy as np
import t3_lib as T

for tag, label, d in T.all_sessions():
    if "d675cf" not in label and "6b2e70" not in label:
        continue
    s = T.load_input(d)
    el, tot = s["el"], s["V"].sum(axis=1)
    n = len(el)
    rng = float(tot.max() - tot.min())
    ms = max(500.0, 0.05 * rng)
    nb = 100
    prev = np.array([np.median(tot[max(0, i - nb):max(1, i - 2)]) if i > 5 else tot[i]
                     for i in range(n)])
    rise = tot - prev
    raw = []
    i = 5
    while i < n:
        if rise[i] > ms:
            bbase = prev[i]
            j = i
            while (j > 1 and i - j < 150 and tot[j - 1] > bbase + 0.2 * ms):
                j -= 1
            if raw and el[j] - el[raw[-1]] < 2.0:
                raw[-1] = j
            else:
                raw.append(j)
            i = int(np.searchsorted(el, el[j] + 2.0))
        else:
            i += 1
    print("==", label, "ms=%.0f  raw=%d" % (ms, len(raw)))
    for k, j in enumerate(raw):
        base = float(np.median(tot[max(0, j - 100):max(1, j - 2)]))
        limit = raw[k + 1] if k + 1 < len(raw) else n
        seg = tot[j:limit]
        step0 = float(np.percentile(seg, 97)) - base
        drop_thr = max(300.0, 0.30 * step0)
        runmax = np.maximum.accumulate(seg)
        k_un = None
        for q in range(min(30, len(seg) - 1), len(seg)):
            if runmax[q] - seg[q] > drop_thr:
                k_un = j + q
                break
        he = k_un if k_un is not None else limit
        he = min(he, n - 1)
        print("   t=%7.2f base=%7.0f step0=%7.0f drop_thr=%7.0f un=%s hold=%5.2fs %s"
              % (el[j], base, step0, drop_thr,
                 ("%.2f" % el[k_un]) if k_un else "None", el[he] - el[j],
                 "" if el[he] - el[j] >= 1.0 else "<-- 丢弃(保压不足)"))
