# -*- coding: utf-8 -*-
"""给 legacy/ 下两份历史 README 加一行归档注记（正文按原文保留，不改内容）。"""
import io
import os

D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "v4.1flash", "progress", "legacy", "docs")

NOTE = ("> **[归档注记]** 本文件是 `temp/v4.1flash/` **重组织前**的原文，按原样保留；"
        "其中的 `Document/`、`figures/`、`results/`、`scripts/`、`paper*/` 等路径与图路径都是**当时的历史路径**，"
        "重组织后不再指向实际位置 —— 现有结构与入口见 `../../README.md`。\n")

for fn in ("README-根目录总入口（原始）.md", "README-Document目录索引（原始，按原文恢复）.md"):
    p = os.path.join(D, fn)
    s = io.open(p, encoding="utf-8").read()
    if "[归档注记]" in s:
        print("已有注记，跳过:", fn)
        continue
    lines = s.split("\n")
    # 插在首个标题行之后
    i = 0
    while i < len(lines) and not lines[i].startswith("#"):
        i += 1
    lines.insert(i + 1, "\n" + NOTE.rstrip("\n"))
    io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print("已加注记:", fn, "->", os.path.getsize(p), "B")
