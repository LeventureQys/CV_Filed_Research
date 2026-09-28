# -*- coding: utf-8 -*-
"""汇总 04_search.json / 04_search_ext.json 的最优参数（GLM53 过程留痕用）。"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\v3.4\sensor_tune\out")
for f in ("04_search.json", "04_search_ext.json"):
    p = BASE / f
    if not p.exists():
        print(f, "MISSING"); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    print("=" * 30, f)
    for s, v in d.items():
        b = v["runs"][0]
        print(f"{s}: live={v['live_score']:.2f} best={b['score']:.2f}")
        print("  params:", json.dumps(b["params"], ensure_ascii=False))
        for r in v["runs"][1:3]:
            print(f"    (次优 {r['score']:.2f}: {json.dumps(r['params'], ensure_ascii=False)})")
