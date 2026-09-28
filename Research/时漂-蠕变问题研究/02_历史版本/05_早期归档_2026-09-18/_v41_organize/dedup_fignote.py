# -*- coding: utf-8 -*-
"""去掉三个桶图件说明里与自动「位置」行重复的前缀句。"""
import io
import os

g = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_manifests.py")
s = io.open(g, encoding="utf-8").read()
dups = [
    "放本桶 `docs/figures/`（与 `dsp方案评审报告.md` 同目录，正文按 `figures/…` 引用）。",
    "放本桶 `docs/figures/`（与论文同目录，正文按 `figures/…` 引用）。",
]
n = 0
for d in dups:
    if d in s:
        n += s.count(d)
        s = s.replace(d, "")
io.open(g, "w", encoding="utf-8", newline="\n").write(s)
print("去掉重复前缀句:", n, "处")
