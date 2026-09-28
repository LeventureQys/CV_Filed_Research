# -*- coding: utf-8 -*-
"""复核 13-v6-assessment 与 plan/v1.0 之间的交叉引用是否可达。"""
import io
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))          # 13-v6-assessment/scripts
BUCKET = os.path.dirname(BASE)
PLAN = os.path.join(os.path.dirname(os.path.dirname(BUCKET)), "plan", "v1.0")

FILES = [
    os.path.join(BUCKET, "docs", "v6评估与需求答复.md"),
    os.path.join(BUCKET, "docs", "b_卸载时漂与可重复性分析.md"),
    os.path.join(PLAN, "答复-索引.md"),
    os.path.join(PLAN, "第一轮现状与口径更正.md"),
]

for p in FILES:
    if not os.path.isfile(p):
        print("!! 缺失:", p)
        continue
    s = io.open(p, encoding="utf-8").read()
    d = os.path.dirname(p)
    bad = []
    # 反引号里的相对路径 + markdown 链接
    cands = re.findall(r"`([^`\n]*(?:progress|plan|figures|results|scripts|docs)/[^`\n]*)`", s)
    cands += [m.group(1) for m in re.finditer(r"\]\(([^)\n]+)\)", s)]
    for t in cands:
        t = t.strip().rstrip("；，。、）)")
        if t.startswith(("http", "#", "<")):
            continue
        if not re.search(r"(progress|plan|figures|results|scripts|docs)/", t):
            continue
        full = os.path.normpath(os.path.join(d, t.replace("/", os.sep)))
        if not os.path.exists(full):
            bad.append(t)
    print("%-34s 引用 %3d 条，断链 %d %s"
          % (os.path.basename(p), len(cands), len(bad), ("→ " + ", ".join(bad[:4])) if bad else "✓"))
