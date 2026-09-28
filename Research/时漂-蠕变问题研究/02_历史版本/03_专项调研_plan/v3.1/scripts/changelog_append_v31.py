# -*- coding: utf-8 -*-
"""给 `Document/ChangingLog/完整更新日志.md` 追加一条 v3.1 收尾条目，并按规则归档。

规则（见 AGENTS.md「主文件 100KB 上限与归档规则」）：
  - 主文件上限 102400 字节；
  - 超限时从**最旧端**整条迁出「约 100KB」（累计 90~110KB 即停），原样移入 Archived/；
  - 归档文件命名 `完整更新日志_<起始日期>至<结束日期>.md`，同名追加 _2/_3；
  - 同步更新 `Archived/归档索引.md` 与主文件顶部说明。
用法: python changelog_append_v31.py
"""
import os
import re
from datetime import date

ROOT = r"D:\workshop\Processing\multi-device-cascade-host-cpp"
CL = os.path.join(ROOT, "Document", "ChangingLog")
MAIN = os.path.join(CL, "完整更新日志.md")
ARCH = os.path.join(CL, "Archived")
LIMIT = 102400

ENTRY = """
## 2026-09-19 交接包整理：把 v6 算法的接续信息集中到 `progress/currentworking/`（Agent 会话，只改文档）

- 需求（用户原文）：「请你把关键的信息补充到 `temp\\v4.1flash\\progress\\currentworking` 这个里面，之后我如果需要将工作转交给其他agent，请你把它需要知道的信息放在这个目录下」。
- 做了什么：把该桶由「3 份（MANIFEST + 01 说明 + 02 更新日志）」扩成**交接包**，新增入口文档 **`00_交接说明.md`**（约 20 KB），九节：① 三十秒版现状（六句话）；② 推荐阅读顺序（本桶 → plan v3.1 → v3.0 → v2.0 → v1.0 → archived/07-v6，各自"读它是为了什么"）；③ **文件地图**（本桶 4 份 / plan 四个交付目录各一句话结论 / 源码 6 个文件 / project_summary / ChangingLog / archived 13 桶）；④ **当前算法状态**——参数集四代谱系表、**生效开关表（含"关掉会怎样"的实测数字）**、5 条已知缺陷与取舍；⑤ **数据在哪怎么找**——明确写出"数据根被用户重组过"（`working/` 在用、`archived/` 归档、原 `测试台工况一览/` 已移出），并要求改用 `v30_lib.discover_sessions()` 而非写死路径，另附三流 CSV 的列位置与"`pre` 流表头缺通道名"的坑；⑥ 复现与验证命令（编译 / 两个离线复算助手 / 12 个脚本用途速查 / 缓存工具）；⑦ **项目验证纪律 7 条**（禁止构建 `test_*`、只增量构建主程序、`/m:4`、界面手测交用户、SubAgent 规则、改动必同步缓存与两份日志、**本桶文档必须跟改**）；⑧ **踩过的坑 10 条**（`dt=0` 帧占 70% ⇒ 输出施加必须与 dt 无关；"扣除向量"必须含新增通路；离线回放 ≠ 现场；数据根会被重组；增量构建时间戳陷阱；指标口径别用"末段−首段漂移"；振荡工况平台指标不可信；`pre` 流列位置；阳性对照要 touch 源文件；稳定时间必须与偏置并报）；⑨ 未关闭项与建议下一步（P0 真机 A/B 与"保压稳定↔卸载保真"的用户拍板 / P1 回归锁单测 / P2 回放对齐与 v6.1 ROM_SCALE / P3 振荡工况与已否决路线）；⑩ 改动与追溯纪律六条。
- 连带更新：`MANIFEST.md` 重写为**交接入口版**（新增"文档（4）"表、生效开关、7 条已知问题含"离线回放 ≠ 现场"、复现命令改为 v3.1 为主）；`progress/README.md` 的 §2.1 与 §3 阅读路径把 `00_交接说明.md` 列为第一读，§10.4 文件数由 3 改为 4（合计 860 → 861，并注明是重组当时的快照）；`project_summary/modules/display.force-adc-pressure/analysis.md` 的指路改为"交接入口 = `00_交接说明.md`"并补当前参数集；`plan/v3.0/scripts/v30_check_docsync.py` 扩成**交接包自检**（4 份文档：相对引用 17 处、关键锚点 14 处、文件数 4；实测 **0 坏链 / 0 缺失 / 文件数 4**）。
- 影响范围：**只改文档，未改任何源码、未重新构建、未跑测试**。本轮无源码改动 ⇒ `project_summary` 的 `content_sha` 无需重算（已复核 `display.force-adc-pressure` 仍 VALID）。
"""


def split_main(text):
    """返回 (头部, [条目...])；条目以行首 `## ` 为边界。"""
    idx = [m.start() for m in re.finditer(r"(?m)^## ", text)]
    if not idx:
        return text, []
    head = text[:idx[0]]
    entries = []
    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else len(text)
        entries.append(text[s:e])
    return head, entries


def entry_date(ent):
    m = re.search(r"^##\s+(\d{4}-\d{2}-\d{2})", ent)
    return m.group(1) if m else "0000-00-00"


def main():
    text = open(MAIN, encoding="utf-8").read()
    if ENTRY.strip().splitlines()[0] in text:
        print("主文件已含本条，跳过追加")
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += ENTRY
        open(MAIN, "w", encoding="utf-8", newline="\n").write(text)
        print("已追加条目")
    size = os.path.getsize(MAIN)
    print("主文件大小 = %d 字节（上限 %d）" % (size, LIMIT))
    if size <= LIMIT:
        print("未超限，无需归档")
        return 0

    head, entries = split_main(open(MAIN, encoding="utf-8").read())
    order = sorted(range(len(entries)), key=lambda i: (entry_date(entries[i]), i))
    moved, total = [], 0
    for i in order:
        b = len(entries[i].encode("utf-8"))
        if total + b > 110 * 1024:
            break
        moved.append(i)
        total += b
        if total >= 90 * 1024:
            break
    if not moved:
        print("!! 无可迁出条目")
        return 1
    moved.sort()
    dates = [entry_date(entries[i]) for i in moved if entry_date(entries[i]) != "0000-00-00"]
    d0, d1 = (min(dates), max(dates)) if dates else ("0000-00-00", "0000-00-00")
    os.makedirs(ARCH, exist_ok=True)
    base = "完整更新日志_%s至%s.md" % (d0, d1)
    path = os.path.join(ARCH, base)
    k = 2
    while os.path.exists(path):
        path = os.path.join(ARCH, base[:-3] + "_%d.md" % k)
        k += 1
    body = "".join(entries[i] for i in moved)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("> 归档文件：`Document/ChangingLog/完整更新日志.md` 的历史条目（**原样迁出，内容未改**）。\n")
        fh.write("> 归档日期区间：**%s ~ %s**（共 %d 条，约 %.1f KB）；归档时间：%s。\n"
                 "> 归档清单见同目录 `归档索引.md`。\n\n"
                 % (d0, d1, len(moved), total / 1024.0, date.today().isoformat()))
        fh.write(body)
    print("归档 %d 条（约 %.1f KB）-> %s" % (len(moved), total / 1024.0, os.path.basename(path)))

    rest = head + "".join(e for i, e in enumerate(entries) if i not in set(moved))
    open(MAIN, "w", encoding="utf-8", newline="\n").write(rest)
    print("主文件新大小 = %d 字节" % os.path.getsize(MAIN))

    # 归档索引
    idxp = os.path.join(ARCH, "归档索引.md")
    files = sorted(f for f in os.listdir(ARCH) if f.startswith("完整更新日志_") and f.endswith(".md"))
    lines = ["# Archived 归档索引", "",
             "> 由 `Document/ChangingLog/完整更新日志.md` 按「每档约 100KB」规则整批迁出的历史条目，",
             "> **原样保留、内容未改**。主文件只保留最近新增条目。", "",
             "| 归档文件 | 日期区间 | 文件大小 |", "|---|---|---|"]
    for f in files:
        m = re.match(r"完整更新日志_(.+)至(.+)\.md$", f)
        rng = "%s ~ %s" % (m.group(1), m.group(2)) if m else "—"
        lines.append("| `%s` | %s | %.1f KB |" % (f, rng, os.path.getsize(os.path.join(ARCH, f)) / 1024.0))
    open(idxp, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("归档索引已更新（%d 档）" % len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
