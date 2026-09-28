# -*- coding: utf-8 -*-
"""整理 temp/：保留 v4.1flash 与原始数据，其余移入 temp/archived/。"""
import os
import shutil
import sys

TEMP = os.path.dirname(os.path.abspath(__file__))  # 本脚本就在 temp/ 下
ARCH = os.path.join(TEMP, "archived")

# ① 整目录搬走
DIRS = ["GLM53", "glm53_cpp_harness", "Claude", "creep_check", "v41_review",
        "audit_crops", "_v41_organize"]
# ② 零散文件
FILES = ["_archive_log.ps1", "_archive_log2.ps1", "_archive_log3.ps1", "_archive_log4.ps1"]
# ③ 右拇指指尖 里除原始数据（数据1~3）之外的部分
SPLIT_SRC = "右拇指指尖"
SPLIT_KEEP = {"数据1", "数据2", "数据3"}
SPLIT_DST = os.path.join(ARCH, "右拇指指尖-分析")

DRY = "--apply" not in sys.argv


def count(p):
    n = 0
    for _r, _d, fs in os.walk(p):
        n += len(fs)
    return n


def main():
    before = count(TEMP)
    print("temp/ 现有文件 %d 个" % before)
    os.makedirs(ARCH, exist_ok=True)

    jobs = []
    for d in DIRS:
        s = os.path.join(TEMP, d)
        if os.path.isdir(s):
            jobs.append((s, os.path.join(ARCH, d)))
    for f in FILES:
        s = os.path.join(TEMP, f)
        if os.path.isfile(s):
            jobs.append((s, os.path.join(ARCH, f)))

    src = os.path.join(TEMP, SPLIT_SRC)
    if os.path.isdir(src):
        for name in sorted(os.listdir(src)):
            if name in SPLIT_KEEP:
                continue
            jobs.append((os.path.join(src, name), os.path.join(SPLIT_DST, name)))

    total = 0
    for s, d in jobs:
        n = count(s) if os.path.isdir(s) else 1
        total += n
        print("  %s%-46s -> archived/%s" % ("[dry] " if DRY else "", os.path.relpath(s, TEMP),
                                            os.path.relpath(d, ARCH)))
    print("待搬 %d 项 / %d 个文件" % (len(jobs), total))
    if DRY:
        print("（dry-run；加 --apply 执行）")
        return 0

    for s, d in jobs:
        if os.path.exists(d):
            print("!! 目标已存在，跳过:", d)
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.move(s, d)

    after = count(TEMP)
    arch = count(ARCH)
    print("完成：temp/ %d 个（其中 archived %d 个）" % (after, arch))
    left = [os.path.join(TEMP, n) for n in sorted(os.listdir(TEMP))]
    print("temp/ 顶层现在剩：")
    for p in left:
        print("   %-22s %s" % (os.path.basename(p) + ("/" if os.path.isdir(p) else ""), count(p) if os.path.isdir(p) else "1 个文件"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
