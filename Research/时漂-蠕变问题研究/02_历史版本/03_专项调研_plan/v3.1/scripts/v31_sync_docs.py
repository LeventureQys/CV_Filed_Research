# -*- coding: utf-8 -*-
"""把 currentworking/01 的常量表·参数集谱系·变更记录同步到 plan-v3.1。"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * 5)))
P = os.path.join(ROOT, "temp", "v4.1flash", "progress", "currentworking",
                 "01_v6算法说明_当前实现.md")

REPL = [
    ("| `kPctMono` | false | 关 | PCT 只加不减（备选） |",
     "| `kPctMono` | false | 关 | PCT 只加不减（备选；实测能换回卸载保真但保压偏离恶化 ⇒ 不采用） |\n"
     "| **`kFreezeSlowOnHit`** | **true** | **生效** | **[P3.1]** 检测器命中当帧起暂停慢相自适应（不更新 `g`、不积分 PCT） |"),
    ('                           "pct_hi_frac": 4.0, "pct_mono": 0,\n'
     '                           "param_set": "plan-v3.0 PCT" } }',
     '                           "pct_hi_frac": 4.0, "pct_mono": 0,\n'
     '                           "freeze_slow_on_hit": 1,\n'
     '                           "param_set": "plan-v3.1 PCT-fix" } }'),
    ("| **`plan-v3.0 PCT`** | ＋ 慢相逐通道蠕变跟踪 τ=10 s / 上界 4·A_k | ＋4 字段（当前） |",
     "| `plan-v3.0 PCT` | ＋ 慢相逐通道蠕变跟踪 τ=10 s / 上界 4·A_k | ＋4 字段 |\n"
     "| **`plan-v3.1 PCT-fix`** | ＋ 扣除连续性四处修复 ＋ 检测器命中冻结慢相 | ＋1 字段（`freeze_slow_on_hit`）（当前） |"),
    ("- 2026-09-19（**plan-v3.0，本次同步**）",
     "- 2026-09-19（**plan-v3.1**）：新增 §6.6「慢相扣除的连续性（卸载/交接）」与常量 `kFreezeSlowOnHit`；\n"
     "  参数集改为 `plan-v3.1 PCT-fix`。定因与实测见 `../../plan/v3.1/`。\n"
     "- 2026-09-19（**plan-v3.0，上次同步**）"),
]

s = open(P, encoding="utf-8").read()
for i, (a, b) in enumerate(REPL, 1):
    if a not in s:
        print("!! 第 %d 处未命中" % i)
    else:
        s = s.replace(a, b, 1)
        print("  命中 %d" % i)
open(P, "w", encoding="utf-8", newline="\n").write(s)
print("-> %s" % P)
