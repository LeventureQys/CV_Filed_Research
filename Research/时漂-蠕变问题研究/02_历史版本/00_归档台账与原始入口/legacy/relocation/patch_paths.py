# -*- coding: utf-8 -*-
"""搬迁后修正脚本里的路径锚点。

背景：脚本原来位于 temp/v4.1flash/scripts/，用
    HERE = dirname(__file__)          -> scripts/
    OUT  = dirname(HERE)              -> temp/v4.1flash
    TEMP = dirname(OUT)               -> temp/            （数据集在这里）
现位于 temp/v4.1flash/progress/<桶>/scripts/，于是 OUT 变成桶目录（results/figures 恰好也在桶里），
TEMP/ROOT 少算了两级，跨桶引用（paper_v6 的 pv_common、paper 的 pd）也要改指。
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(os.path.dirname(HERE), "v4.1flash", "progress")

# 全局替换（对所有桶的脚本生效）
GLOBAL = [
    # 数据集根：temp/
    ("TEMP = os.path.dirname(OUT)",
     "TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))"),
    # 仓库根
    ("ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))",
     "ROOT = " + "os.path.dirname(" * 5 + "HERE" + ")" * 5),
    # v6.1 曾用 OUT/paper_v6/scripts 引入 pv_common；该模块现在随桶复制到同目录
    ('sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))',
     'sys.path.insert(0, HERE)'),
    # 早期临时脚本按 cwd 猜路径
    ('sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))',
     'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))'),
    ('RES = os.path.join(os.getcwd(), "results")',
     'RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")'),
    # 数据集拼路径
    ('os.path.join(os.path.dirname(OUT), loc, f"数据{i}", "device_001_seg000.csv")',
     'os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(OUT))), loc, f"数据{i}", "device_001_seg000.csv")'),
]

# 单文件替换
SPECIAL = {
    os.path.join("08-v6.1", "scripts", "pv_common.py"): [
        ("FLASH = os.path.dirname(ROOT)",
         "FLASH = os.path.dirname(os.path.dirname(ROOT))"),
        ('sys.path.insert(0, os.path.join(FLASH, "scripts"))',
         'sys.path.insert(0, HERE)'),
    ],
    os.path.join("11-paper-v6", "scripts", "pv_common.py"): [
        ("FLASH = os.path.dirname(ROOT)",
         "FLASH = os.path.dirname(os.path.dirname(ROOT))"),
        ('sys.path.insert(0, os.path.join(FLASH, "scripts"))',
         'sys.path.insert(0, HERE)'),
    ],
    os.path.join("10-paper-v5-route", "scripts", "pd.py"): [
        ("FLASH = os.path.dirname(ROOT)",
         "FLASH = os.path.dirname(os.path.dirname(ROOT))"),
        ('sys.path.insert(0, os.path.join(FLASH, "scripts"))',
         'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))'),
    ],
    os.path.join("10-paper-v5-route", "scripts", "pf5_scenarios.py"): [
        ('sys.path.insert(0, os.path.join(FLASH, "scripts"))',
         'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))'),
    ],
    # 论文 v6 要读 v6 的既有分析产物（现落在 07-v6 桶）
    os.path.join("11-paper-v6", "scripts", "pf1_physical.py"): [
        ('OLD = os.path.join(C.FLASH, "results")',
         'OLD = os.path.join(C.FLASH, "progress", "07-v6", "results")'),
    ],
    os.path.join("11-paper-v6", "scripts", "pf3_invert_glide.py"): [
        ('os.path.join(C.FLASH, "results", "v6_detect_latency.csv")',
         'os.path.join(C.FLASH, "progress", "07-v6", "results", "v6_detect_latency.csv")'),
    ],
    # v_shape_diag 的锚点写法与别处不同，整块重写
    os.path.join("02-dsp-route", "scripts", "v_shape_diag.py"): [
        ("TEMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
         "TEMP = os.path.dirname(TEMP)\n"
         "OUT = os.path.dirname(os.path.dirname(TEMP))\n"
         "ROOT = os.path.dirname(TEMP)\n",
         "OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
         "TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))\n"
         "ROOT = os.path.dirname(TEMP)\n"
         "RES = os.path.join(OUT, \"results\")\n"),
        ('os.path.join(TEMP, "v4.1flash", "results", "_shape_diag.txt")',
         'os.path.join(RES, "_shape_diag.txt")'),
    ],
    # C++↔Python 对拍脚本：产物 exe 随桶搬到 cpp_check/
    os.path.join("04-v5", "scripts", "_diff.py"): [
        ('OUT = r"D:\\workshop\\Processing\\multi-device-cascade-host-cpp\\temp\\v4.1flash"',
         'OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))'),
        ('sys.path.insert(0, os.path.join(OUT, "scripts"))',
         'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))'),
        ('EXE = os.path.join(OUT, "cpp_v5_check", "build", "Debug", "v5_cpp_check.exe")',
         'EXE = os.path.join(OUT, "cpp_check", "build", "Debug", "v5_cpp_check.exe")'),
    ],
}


def patch_file(path, rules):
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    hits = []
    for old, new in rules:
        n = txt.count(old)
        if n:
            txt = txt.replace(old, new)
            hits.append((old.split("\n")[0][:64], n))
    if hits:
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            f.write(txt)
    return hits


def main():
    apply = "--apply" in sys.argv
    n_files = n_hits = 0
    tally = {}
    for bucket in sorted(os.listdir(PROG)):
        sdir = os.path.join(PROG, bucket, "scripts")
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".py"):
                continue
            rel = os.path.join(bucket, "scripts", fn)
            p = os.path.join(sdir, fn)
            rules = GLOBAL + SPECIAL.get(rel, [])
            if apply:
                hits = patch_file(p, rules)
            else:
                with io.open(p, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                hits = [(o.split("\n")[0][:64], txt.count(o)) for o, _n in rules if txt.count(o)]
            if hits:
                n_files += 1
                n_hits += sum(n for _o, n in hits)
                for o, n in hits:
                    tally[o] = tally.get(o, 0) + n
    print(("已应用" if apply else "[dry-run] 将修改") + " %d 个文件，%d 处替换" % (n_files, n_hits))
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print("  %3d  %s" % (v, k))
    left = [r for r in SPECIAL if not os.path.isfile(os.path.join(PROG, r))]
    if left:
        print("!! SPECIAL 指定的文件不存在：%s" % left)
    return 0


if __name__ == "__main__":
    sys.exit(main())
