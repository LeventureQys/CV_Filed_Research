# -*- coding: utf-8 -*-
"""T9-F：打印 36~63 s 的细节（符号翻转发生在哪一段、是否与某次卸载/加载对齐）。"""
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 6)))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

rows = list(csv.DictReader(open(os.path.join(OUT, "t9b_series_full.csv"), encoding="utf-8")))
print("   t     pre    main     raw      off      (0.25 s 采样)")
prev = None
for r in rows:
    t = float(r["t_s"])
    if 30.0 <= t <= 62.7:
        off = float(r["off"])
        mark = ""
        if prev is not None:
            if prev < 0 <= off:
                mark = "   <== 偏移由负转正"
            elif prev > 0 >= off:
                mark = "   <== 偏移由正转负"
        print(f"{t:6.2f} {float(r['pre_tot']):7.0f} {float(r['main_tot']):7.0f} "
              f"{float(r['raw_tot']):7.0f} {off:8.0f}{mark}")
        prev = off
