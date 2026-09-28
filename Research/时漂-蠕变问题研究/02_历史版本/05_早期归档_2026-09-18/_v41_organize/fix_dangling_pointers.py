# -*- coding: utf-8 -*-
"""修正两处「原本就悬空」的指针：档案里不存在 Document/…详细设计.md 与 s4_final.py。"""
import io
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # temp/archived/_v41_organize -> 仓库根

EDITS = [
    (os.path.join(REPO, "src", "domain", "creep", "change_point_creep_compensator.h"),
     [("temp/archived/Claude/Document/变点感知对数蠕变补偿算法详细设计.md §4",
       "temp/archived/Claude/02-v6c算法说明.md")]),
    (os.path.join(REPO, "project_summary", "modules", "display.force-adc-pressure", "analysis.md"),
     [("`temp/archived/Claude/Document/变点感知对数蠕变补偿算法详细设计.md`",
       "`temp/archived/Claude/02-v6c算法说明.md`"),
      ("`temp/archived/Claude/s4_final.py`", "`temp/archived/Claude/glm53_v6c.py`")]),
]


def main():
    apply = "--apply" in os.sys.argv
    for path, pairs in EDITS:
        s = io.open(path, encoding="utf-8").read()
        n = 0
        for a, b in pairs:
            if a in s:
                n += s.count(a)
                s = s.replace(a, b)
        print("%s %-58s %d 处" % ("[改]" if apply else "[dry]", os.path.relpath(path, REPO), n))
        if apply and n:
            io.open(path, "w", encoding="utf-8", newline="").write(s)

    # 复查：所有指向 archived 的指针都应存在
    import re
    print("\n复查指向 temp/archived 的路径是否存在：")
    for path in [os.path.join(REPO, "src", "domain", "creep", "change_point_creep_compensator.h"),
                 os.path.join(REPO, "src", "domain", "drift", "drift_compensator.h"),
                 os.path.join(REPO, "project_summary", "modules",
                              "display.force-adc-pressure", "analysis.md")]:
        txt = io.open(path, encoding="utf-8").read()
        for m in sorted(set(re.findall(r"temp/archived/[^\s`，。）)]+", txt))):
            p = os.path.join(REPO, m.replace("/", os.sep))
            print("   %-72s %s" % (m, "存在" if os.path.exists(p) else "!! 缺失"))


if __name__ == "__main__":
    os.sys.exit(main())
