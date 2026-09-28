# -*- coding: utf-8 -*-
"""① 整理 temp/ 的更新日志条目；② 主文件超 100KB 前按「每档约 100KB」规则整批归档；③ 校验条目守恒。

规则（AGENTS.md）：主文件上限 102400 字节；超限时从最旧端按条目日期整批迁出约 100KB（90~110KB），
归档文件内条目按日期升序、内容逐字不改，并同步 Archived/归档索引.md 与主文件顶部指路。
"""
import io
import os
import re

REPO = r"D:\workshop\Processing\multi-device-cascade-host-cpp"
MAIN = os.path.join(REPO, "Document", "ChangingLog", "完整更新日志.md")
ARCH_DIR = os.path.join(REPO, "Document", "ChangingLog", "Archived")
INDEX = os.path.join(ARCH_DIR, "归档索引.md")

ENTRY = """
## 2026-09-18 整理 temp/ 目录：只保留 v4.1flash 与原始数据，其余 540 个文件移入 temp/archived/（Agent 会话，只动 temp/ 与三处指路）

- 需求来源：用户「整理一下 temp 目录下的内容：除了 v4.1flash 和原始数据，其他内容放在 temp/archived 目录下」。
- 搬了什么（17 项 / 540 个文件，整目录平移、内容未改）：`GLM53/`（混合蠕变补偿前身的分析与实测，含被评审的 `dsp.md`）、`glm53_cpp_harness/`（GLM53 的 C++↔Python 对拍工程）、`Claude/`（变点感知对数蠕变补偿 v6c 的设计与原型）、`creep_check/`（v6c 估计器复核）、`v41_review/` 与 `audit_crops/`（两批图件复核裁剪）、`_v41_organize/`（v4.1flash 归档台账与回退点）、4 个 `_archive_log*.ps1`（日志归档脚本），以及 **`temp/右拇指指尖/` 里除原始数据外的全部分析产物**（`漂移分析与算法对比报告.md`、`scripts/ results/ figures/ DSv4/ DSv4.1flash/`）→ 归为 `archived/右拇指指尖-分析/`。
- 保留了什么：`temp/v4.1flash/`（版本化归档，入口 `v4.1flash/progress/README.md`）与原始数据 `右拇指指尖/数据1~3`、`左拇指指尖/数据1~3`、`四指指尖/数据1~3`、`变化负载/`（共 29 个数据文件）。整理后 `temp/` = v4.1flash(813) + 数据(29) + archived(548)。
- 连带更新（活引用，避免指向已搬走的路径）：① `src/domain/drift/drift_compensator.h`、`src/domain/creep/change_point_creep_compensator.h` 的「另见」注释改为 `temp/archived/...`；② `project_summary/modules/display.force-adc-pressure/analysis.md` 三处指路同步，同目录 `analysis.json` 的 `content_sha` 重算（`check_stale.ps1 -Module display.force-adc-pressure` 为 **VALID**）；③ `v4.1flash/progress/02-dsp-route/docs/dsp方案评审报告.md`、`.../03-v4/docs/快相与慢相分离分析.md` 里的 `temp/GLM53/…` 路径；④ `v4.1flash/progress/README.md` 的 `_v41_organize` 回退点路径、并新增 §9.4 记录本次整理。**未改**历史记录：`Document/ChangingLog/` 旧条目与 `v4.1flash/progress/legacy/` 的历史 README 保留当时路径。
- 顺带修正两处**原本就悬空**的指针：注释/缓存里引用的 `Claude/Document/变点感知对数蠕变补偿算法详细设计.md` 与 `Claude/s4_final.py` 在档案里并不存在，已改指实际文档 `Claude/02-v6c算法说明.md` 与原型 `Claude/glm53_v6c.py`。
- 归档内脚本的路径注意（写入 `temp/archived/README.md`）：整体深了一层，约 **93 个历史脚本**的「数据集根」锚点会指到 `temp/archived/...`，本次只对 **14 个文件**做了自动补级（7 个 `GLM53/scripts/*` + 7 个归档工具，后者必须能跑），其余保持历史原样并给出批量修正脚本 `patch_archived_paths.py`；`Claude/*.py` 的 `SCRIPTS = HERE/../v4.1flash/scripts` 需按目标算法改指到 `progress/<桶>/scripts`。
- 验证：① `temp/` 整理后实跑通过——从 `temp/右拇指指尖/数据1` 读入 **16354 帧**并跑 `GLM53v5` 正常输出（证明数据集未动、`TEMP` 锚点仍为 `temp/`）；② v4.1flash 归档复校：锚点 **203/203** 正确、同目录 import（含动态）**0** 失败；③ archived 侧补级后锚点存在性审计 **0** 待修；④ 文件数守恒：搬迁 540 项、`temp/` 顶层只剩 v4.1flash + 4 个数据集目录 + archived。
- 影响范围：**只动 `temp/` 与上面列出的三处指路（两个 src 头注释、一个 project_summary 缓存、两篇 temp 内文档）**；未改任何功能代码逻辑、未改 CMake、未构建、未跑测试。
"""


def split_entries(text):
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("## ")]
    head = "\n".join(lines[:idx[0]]) if idx else text
    entries = []
    for n, i in enumerate(idx):
        j = idx[n + 1] if n + 1 < len(idx) else len(lines)
        seg = "\n".join(lines[i:j]).rstrip("\n")
        m = re.search(r"(\d{4}-\d{2}-\d{2})", lines[i])
        entries.append({"date": m.group(1) if m else None, "text": seg,
                        "bytes": len(seg.encode("utf-8")) + 1})
    return head, entries


def main():
    apply = "--apply" in os.sys.argv
    text = io.open(MAIN, encoding="utf-8").read()
    head, entries = split_entries(text)
    print("主文件 %d B；条目 %d 条 / %d B；头部 %d B"
          % (len(text.encode("utf-8")), len(entries),
             sum(e["bytes"] for e in entries), len(head.encode("utf-8"))))

    # 按日期从最旧端累计到 90KB 以上（无日期视为最旧）
    order = sorted(range(len(entries)), key=lambda i: (entries[i]["date"] or "", i))
    batch, acc = [], 0
    for i in order:
        if acc >= 90000:
            break
        batch.append(i)
        acc += entries[i]["bytes"]
    keep = [i for i in range(len(entries)) if i not in batch]
    dates = sorted(entries[i]["date"] for i in batch if entries[i]["date"])
    print("拟迁出 %d 条 / %d B（%s ~ %s）；主文件保留 %d 条 / %d B"
          % (len(batch), acc, dates[0], dates[-1], len(keep),
             sum(entries[i]["bytes"] for i in keep)))

    name = "完整更新日志_%s至%s.md" % (dates[0], dates[-1])
    dst = os.path.join(ARCH_DIR, name)
    if os.path.exists(dst):
        n = 2
        while os.path.exists(dst):
            name = "完整更新日志_%s至%s_%d.md" % (dates[0], dates[-1], n)
            dst = os.path.join(ARCH_DIR, name)
            n += 1
    print("归档文件：Archived/" + name)

    if not apply:
        print("（dry-run；加 --apply 执行）")
        return 0

    # 归档文件：按日期升序、内容逐字不改
    hdr = ("# 完整更新日志归档 · %s 至 %s\n\n"
           "> 2026-09-18 按「每档约 100KB」规则从最旧端整批迁出（%d 条，约 %.1fKB）；"
           "条目内容逐字未改，按日期升序排列。迁移原因：主文件在写入新条目后将超过 100KB 上限。\n\n"
           % (dates[0], dates[-1], len(batch), acc / 1024))
    body = "\n\n".join(entries[i]["text"] for i in
                       sorted(batch, key=lambda i: (entries[i]["date"] or "", i)))
    io.open(dst, "w", encoding="utf-8", newline="\n").write(hdr + body + "\n")

    # 主文件：头部改写指路 + 保留条目（原顺序）
    new_head = re.sub(
        r"> \*\*最近一次归档（\d{4}-\d{2}-\d{2}）\*\*：.*?\n",
        "> **最近一次归档（2026-09-18）**：主文件写入新条目后将超过 100KB（102400 字节）上限，"
        "按规则从最旧端整批迁出 **%s ~ %s 的 %d 条**（约 %.1fKB）至 "
        "`Archived/%s`；条目在「归档 + 主文件」中各自恰好出现一次、内容逐字未改。\n"
        % (dates[0], dates[-1], len(batch), acc / 1024, name),
        head + "\n", count=1, flags=re.S)
    if new_head == head + "\n":
        new_head = head + ("\n\n> **最近一次归档（2026-09-18）**：按「每档约 100KB」规则迁出 "
                           "%s ~ %s 的 %d 条（约 %.1fKB）至 `Archived/%s`。\n"
                           % (dates[0], dates[-1], len(batch), acc / 1024, name))
    keep_text = "\n\n".join(entries[i]["text"] for i in keep)
    io.open(MAIN, "w", encoding="utf-8", newline="\n").write(
        new_head.rstrip("\n") + "\n\n" + keep_text + "\n")

    # 追加本次新条目
    io.open(MAIN, "a", encoding="utf-8", newline="\n").write(ENTRY)

    # 归档索引
    ixn = ("\n- %s：%s 至 %s（%d 条历史条目，约 %.1fKB，2026-09-18 按「每档约 100KB」规则"
           "从最旧端整批迁出；主文件在写入「整理 temp/ 目录」条目后将超 100KB 上限触发归档；"
           "条目内容逐字未改，归档内按日期升序排列）\n" % (name, dates[0], dates[-1],
                                                    len(batch), acc / 1024))
    with io.open(INDEX, "a", encoding="utf-8", newline="\n") as f:
        f.write(ixn)

    # 校验守恒
    main_now = io.open(MAIN, encoding="utf-8").read()
    arch_now = io.open(dst, encoding="utf-8").read()
    _, kept = split_entries(main_now)
    _, moved = split_entries(arch_now)
    print("校验：主文件 %d B / %d 条；归档 %d B；条目 %d = %d(归档) + %d(主文件)"
          % (len(main_now.encode("utf-8")), len(kept), len(arch_now.encode("utf-8")),
             len(entries) + 1, len(moved), len(kept)))
    if len(moved) + len(kept) != len(entries) + 1:
        print("!! 条目数不守恒")
    if len(main_now.encode("utf-8")) > 102400:
        print("!! 主文件仍超上限")
    return 0


if __name__ == "__main__":
    os.sys.exit(main())
