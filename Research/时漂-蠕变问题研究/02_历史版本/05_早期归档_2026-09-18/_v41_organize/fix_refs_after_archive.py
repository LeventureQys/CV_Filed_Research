# -*- coding: utf-8 -*-
"""归档后同步「活引用」：v4.1flash 文档、src/ 头注释、project_summary 指路，以及缓存 SHA 刷新。"""
import hashlib
import io
import json
import os
import re
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # temp/archived/_v41_organize -> 仓库根
TEMP = os.path.join(REPO, "temp")
PROG = os.path.join(TEMP, "v4.1flash", "progress")
ARCH = os.path.join(TEMP, "archived")

EDITS = [
    # ① v4.1flash 的正式文档（不含 legacy 的历史快照）
    (os.path.join(PROG, "02-dsp-route", "docs", "dsp方案评审报告.md"),
     [("`temp/GLM53/dsp.md`", "`temp/archived/GLM53/dsp.md`"),
      ("`temp/GLM53/算法说明_混合蠕变补偿.md`", "`temp/archived/GLM53/算法说明_混合蠕变补偿.md`")]),
    (os.path.join(PROG, "03-v4", "docs", "快相与慢相分离分析.md"),
     [("`temp/GLM53/变化负载分析与v2方案.md`", "`temp/archived/GLM53/变化负载分析与v2方案.md`")]),
    # ② src/ 头注释里的「另见」指针
    (os.path.join(REPO, "src", "domain", "drift", "drift_compensator.h"),
     [("temp/GLM53/", "temp/archived/GLM53/")]),
    (os.path.join(REPO, "src", "domain", "creep", "change_point_creep_compensator.h"),
     [("temp/claude/", "temp/archived/Claude/")]),
    # ③ project_summary 指路
    (os.path.join(REPO, "project_summary", "modules", "display.force-adc-pressure", "analysis.md"),
     [("`temp/claude/Document/变点感知对数蠕变补偿算法详细设计.md`",
       "`temp/archived/Claude/Document/变点感知对数蠕变补偿算法详细设计.md`"),
      ("`temp/claude/s4_final.py`", "`temp/archived/Claude/s4_final.py`"),
      ("`temp/creep_check/check_estimator.py`", "`temp/archived/creep_check/check_estimator.py`")]),
]


def git_hash(path):
    """与 git hash-object 等价的 blob 哈希。"""
    data = open(path, "rb").read()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def main():
    apply = "--apply" in os.sys.argv

    # 把我这次的搬迁脚本一并归入 archived
    src_tool = os.path.join(TEMP, "_archive_temp.py")
    if os.path.isfile(src_tool):
        dst_tool = os.path.join(ARCH, "_v41_organize", "_archive_temp.py")
        if apply:
            shutil.move(src_tool, dst_tool)
            print("搬迁脚本归入 archived/_v41_organize/")
        else:
            print("[dry] 搬迁脚本 -> archived/_v41_organize/_archive_temp.py")

    changed = []
    for path, pairs in EDITS:
        if not os.path.isfile(path):
            print("!! 文件不存在:", path)
            continue
        s = io.open(path, encoding="utf-8").read()
        n = 0
        for a, b in pairs:
            if a in s:
                n += s.count(a)
                s = s.replace(a, b)
        if n:
            changed.append((path, n))
            if apply:
                io.open(path, "w", encoding="utf-8", newline="").write(s)
            print("%s %-58s %d 处" % ("[改]" if apply else "[dry]", os.path.relpath(path, REPO), n))
        else:
            print("     %-58s 无需修改" % os.path.relpath(path, REPO))

    # 刷新被改动 src 文件的 content_sha
    json_path = os.path.join(REPO, "project_summary", "modules",
                             "display.force-adc-pressure", "analysis.json")
    if apply and changed:
        d = json.load(io.open(json_path, encoding="utf-8"))
        touched = []
        for path, _n in changed:
            rel = os.path.relpath(path, REPO).replace(os.sep, "/")
            for entry in d.get("source_files", []):
                if (entry.get("path") or "").replace("\\", "/") == rel:
                    old = entry.get("content_sha")
                    new = git_hash(path)
                    entry["content_sha"] = new
                    entry["source_state"] = "modified"
                    touched.append((rel, old[:12] if old else "-", new[:12]))
        d["analyzed_at"] = "2026-09-18"
        io.open(json_path, "w", encoding="utf-8").write(
            json.dumps(d, ensure_ascii=False, indent=2) + "\n")
        print("已刷新 SHA:", touched)
    return 0


if __name__ == "__main__":
    os.sys.exit(main())
