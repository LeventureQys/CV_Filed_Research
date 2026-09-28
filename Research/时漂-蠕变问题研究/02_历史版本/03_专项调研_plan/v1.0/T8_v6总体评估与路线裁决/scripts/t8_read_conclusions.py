# -*- coding: utf-8 -*-
"""t8_read_conclusions.py -- T8 收口席：批量抽取 T1~T7 的结构化交接件。

只读；不重跑任何参数扫描。输出两件：
  1) results/_t8_conclusions_dump.txt   全量 JSON 原样转储（人工核对用）
  2) results/t8_handoff_index.csv       每个 seat 的 headline/verdict/correction/gap 计数索引

用法：python scripts/t8_read_conclusions.py
"""
import os
import sys
import json
import glob
import csv

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
RESULTS = os.path.join(TASK, "results")

PATTERN = os.path.join(PLAN, "T*_*", "results", "conclusions*.json")


def is_handoff(path):
    """交接件 = 目录名以 T<数字>_ 开头、且不是 T8 自己的 conclusions.json。"""
    task_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if not (len(task_dir) > 1 and task_dir[0] == "T" and task_dir[1].isdigit()):
        return False
    return task_dir.split("_")[0] != "T8"


def main():
    files = sorted(p for p in glob.glob(PATTERN) if is_handoff(p))
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    dump = []
    for p in files:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        task_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
        rel = os.path.relpath(p, PLAN).replace("\\", "/")
        dump.append("=" * 100)
        dump.append("FILE: " + rel)
        dump.append(json.dumps(d, ensure_ascii=False, indent=1))
        rows.append({
            "task_dir": task_dir,
            "seat": d.get("seat", ""),
            "date": d.get("date", ""),
            "n_headline": len(d.get("headline", []) or []),
            "n_verdicts": len(d.get("verdicts", []) or []),
            "n_corrections": len(d.get("corrections", []) or []),
            "n_gaps": len(d.get("gaps", []) or []),
            "n_blockers": len(d.get("blockers", []) or []),
            "keys": "|".join(sorted(d.keys())),
        })
    with open(os.path.join(RESULTS, "_t8_conclusions_dump.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(dump) + "\n")
    with open(os.path.join(RESULTS, "t8_handoff_index.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("files:", len(files))
    for r in rows:
        print(r["seat"], r["n_headline"], r["n_verdicts"], r["n_corrections"], r["n_gaps"])


if __name__ == "__main__":
    main()
