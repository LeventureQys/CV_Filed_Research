# -*- coding: utf-8 -*-
"""校验 `progress/currentworking/` 四份文档里的相对引用与关键锚点（交接包自检）。

用法: python v30_check_docsync.py
"""
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * 5)))
BASE = os.path.join(ROOT, "temp", "v4.1flash", "progress", "currentworking")
FILES = ["00_交接说明.md", "01_v6算法说明_当前实现.md", "02_算法更新日志.md", "MANIFEST.md"]
PAT = re.compile(r"`([^`]+?\.(?:md|py|cpp|h|json|txt|csv|ps1|exe))`")
# 必须出现的锚点（交接包的关键断言）
MUST = [
    ("00_交接说明.md", "plan-v3.1 PCT-fix"),
    ("00_交接说明.md", "discover_sessions()"),
    ("00_交接说明.md", "## 7. 踩过的坑"),
    ("00_交接说明.md", "## 8. 没做完的事 & 建议的下一步"),
    ("01_v6算法说明_当前实现.md", "plan-v3.1 PCT-fix"),
    ("01_v6算法说明_当前实现.md", "### 6.5 ★plan-v3.0 新增：PCT 逐通道蠕变跟踪（默认生效）"),
    ("01_v6算法说明_当前实现.md", "### 6.6 ★plan-v3.1 修复：慢相扣除的**连续性**（卸载/交接）"),
    ("01_v6算法说明_当前实现.md", "| **`kPctTauS`** | **10.0 s** |"),
    ("01_v6算法说明_当前实现.md", "### 9.2 ★★恒载长保压的残漂"),
    ("02_算法更新日志.md", "### 2.3 ★ ③ plan-v3.0 · PCT 逐通道蠕变跟踪"),
    ("02_算法更新日志.md", "### 2.2 ② plan-v2.0 · C + F5（＋ F2/F3 默认关）"),
    ("02_算法更新日志.md", "### 2.4 ★ ④ plan-v3.1 · 慢相扣除的连续性修复"),
    ("MANIFEST.md", "`plan-v3.1 PCT-fix`"),
    ("MANIFEST.md", "00_交接说明.md"),
]


def main():
    bad, checked = [], 0
    for f in FILES:
        txt = open(os.path.join(BASE, f), encoding="utf-8").read()
        for m in sorted(set(PAT.findall(txt))):
            if not m.startswith("../"):
                continue
            checked += 1
            if not os.path.exists(os.path.normpath(os.path.join(BASE, m))):
                bad.append((f, m))
    print("相对引用：检查 %d 处，坏链 %d 处" % (checked, len(bad)))
    for b in bad:
        print("  坏链 %s -> %s" % b)

    print()
    miss = []
    for f, needle in MUST:
        txt = open(os.path.join(BASE, f), encoding="utf-8").read()
        if needle not in txt:
            miss.append((f, needle))
    print("关键锚点：应有 %d 处，缺失 %d 处" % (len(MUST), len(miss)))
    for m in miss:
        print("  缺失 %s -> %s" % m)

    print()
    print("文件数（应恰为 %d）：%d" % (len(FILES), len([x for x in os.listdir(BASE)])))
    for f in FILES:
        print("  %-34s %6d B" % (f, os.path.getsize(os.path.join(BASE, f))))
    return 0 if not bad and not miss else 1


if __name__ == "__main__":
    raise SystemExit(main())
