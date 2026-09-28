# -*- coding: utf-8 -*-
"""执行 temp/v4.1flash 的版本化归档搬迁（依据 classify.py 生成的 plan.json）。

规则（对应用户要求）：
  · 只属于一个版本桶的文件 —— 剪切（move），原件消失；
  · 被多个版本桶共有的文件 —— 复制到每个桶，然后删除原件。
任何目标路径已存在且非空时立即中止，避免误覆盖。
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "v4.1flash")
PLAN = os.path.join(HERE, "plan.json")
DEST_ROOT = os.path.join(SRC, "progress")

FIG_ROOTS = (os.path.join("figures", ""),
             os.path.join("paper", "figures", ""),
             os.path.join("paper_v6", "figures", ""))
AUDIT_DIRS = ("_audit", "_crop", "_review_crops", "_zoom")


def dest_rel(bucket, kind, rel):
    """返回相对 DEST_ROOT 的目标路径。"""
    if kind == "scripts":
        return os.path.join(bucket, "scripts", os.path.basename(rel))
    if kind == "docs":
        return os.path.join(bucket, "docs", os.path.basename(rel))
    if kind == "results":
        sub = rel[len("results" + os.sep):] if rel.startswith("results" + os.sep) else os.path.basename(rel)
        return os.path.join(bucket, "results", sub)
    if kind == "figures":
        for root in FIG_ROOTS:
            if rel.startswith(root):
                return os.path.join(bucket, "figures", rel[len(root):])
        top = rel.split(os.sep)[0]
        if top in AUDIT_DIRS:
            return os.path.join(bucket, "figures", rel)
        return os.path.join(bucket, "figures", os.path.basename(rel))
    if kind in ("source", "cpp_check"):
        top = rel.split(os.sep)[0]
        sub = rel[len(top) + 1:] if os.sep in rel else os.path.basename(rel)
        return os.path.join(bucket, kind, sub)
    raise SystemExit("未知类别: %s" % kind)


def main():
    dry = "--apply" not in sys.argv
    plan = json.load(open(PLAN, encoding="utf-8"))
    detail = plan["_detail"]
    shared = {k: v for k, v in plan["shared"].items()}

    jobs = []          # (src, dst, is_shared)
    for bucket, kinds in detail.items():
        for kind, rels in kinds.items():
            for rel in rels:
                s = os.path.join(SRC, rel)
                d = os.path.join(DEST_ROOT, dest_rel(bucket, kind, rel))
                jobs.append((s, d, rel in shared, bucket))

    missing = [s for s, _d, _sh, _b in jobs if not os.path.exists(s) and not os.path.exists(_d)]
    if missing:
        print("!! 源与目标均缺失 %d 个，示例：%s" % (len(missing), missing[:5]))
        return 1
    clash = [d for s, d, sh, b in jobs
             if os.path.exists(d) and not sh and not os.path.samefile(s, d) and os.path.isfile(d)]
    if clash:
        print("!! 目标已存在 %d 个：%s" % (len(clash), clash[:5]))
        return 1

    print("作业数: %d（其中共享多投 %d）" % (len(jobs), sum(1 for j in jobs if j[2])))
    if dry:
        for s, d, sh, b in jobs[:15]:
            print("  %-6s %s -> %s" % ("shared" if sh else "move",
                                       os.path.relpath(s, SRC), os.path.relpath(d, DEST_ROOT)))
        print("  ... （dry-run，未落盘；加 --apply 执行）")
        return 0

    # 共享文件：先复制到所有目标，再统一删除原件（同一源只删一次）
    shared_srcs = {}
    for s, d, sh, _b in jobs:
        if sh:
            shared_srcs.setdefault(s, []).append(d)

    n_move = n_copy = n_skip = 0
    errors = []
    for i, (s, d, sh, _b) in enumerate(jobs):
        if not os.path.exists(s):
            if os.path.exists(d):
                n_skip += 1                      # 幂等：已搬迁完成（断点续跑）
                continue
            errors.append((s, d, "源与目标都不存在"))
            continue
        try:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if s in shared_srcs:
                if d != shared_srcs[s][-1]:      # 其余目标复制
                    shutil.copy2(s, d)
                    n_copy += 1
                    continue
                shutil.move(s, d)                 # 最后一个目标用 move 收掉原件
                n_move += 1
            else:
                shutil.move(s, d)
                n_move += 1
        except Exception as e:                    # noqa: BLE001
            errors.append((s, d, repr(e)))
        if (i + 1) % 200 == 0:
            print("  ... %d/%d" % (i + 1, len(jobs)))

    print("完成：move %d，copy %d，跳过(已完成) %d，错误 %d" % (n_move, n_copy, n_skip, len(errors)))
    for s, d, msg in errors[:20]:
        print("  ERR %s -> %s : %s" % (s, d, msg))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
