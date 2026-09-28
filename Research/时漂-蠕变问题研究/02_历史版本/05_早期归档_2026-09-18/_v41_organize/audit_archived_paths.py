# -*- coding: utf-8 -*-
"""审计 archived/ 下脚本的数据路径锚点：搬深一层后哪些脚本读不到数据集了。

对每个 .py：按顶层赋值求值出路径锚点（HERE/OUT/TEMP/ROOT/BASE/DATA/SCRIPTS/...），
看其中「指向数据集或既有产物」的锚点在(新位置)是否成立；同时模拟(原位置)作对照，
从而区分「本来就坏」与「被这次搬迁弄坏」。
"""
import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # temp/archived/_v41_organize -> 仓库根
ARCH = os.path.dirname(HERE)
TEMP = os.path.dirname(ARCH)
DATASETS = ("右拇指指尖", "左拇指指尖", "四指指尖", "变化负载")

ANCHORS = {"HERE", "OUT", "TEMP", "ROOT", "BASE", "DATA", "SCRIPTS", "RES", "FIG",
           "PROJ", "FLASH", "DS", "D", "SRC", "ART"}


def anchors(path, file_for_eval=None):
    """求值顶层赋值里的路径锚点；file_for_eval 可注入假想的 __file__（用于模拟旧位置）。"""
    try:
        tree = ast.parse(io.open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return {}
    ns = {"os": os, "sys": sys, "__file__": file_for_eval or path}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id not in ANCHORS:
            continue
        try:
            ns[t.id] = eval(compile(ast.Expression(node.value), "<a>", "eval"), ns)  # noqa: S307
        except Exception:                                                     # noqa: BLE001
            pass
    return {k: v for k, v in ns.items() if k in ANCHORS and isinstance(v, str)}


def interesting(v):
    """只看指向 temp 数据集/产物的锚点。"""
    if not isinstance(v, str):
        return False
    tail = v.replace("/", os.sep)
    return (any(d in tail for d in DATASETS)) or tail.rstrip(os.sep).endswith("temp") \
        or tail.rstrip(os.sep).endswith("v4.1flash")


def main():
    broken, okcnt, was_broken = [], 0, 0
    for root, _d, fs in os.walk(ARCH):
        for f in sorted(fs):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            a = anchors(p)
            rel = os.path.relpath(p, ARCH)
            hits = {k: v for k, v in a.items() if interesting(v)}
            if not hits:
                continue
            good = all(os.path.exists(v) for v in hits.values())
            if good:
                okcnt += 1
                continue
            # 模拟原位置（少一层 archived/）：只改注入的 __file__，源文件本身仍在原处读取
            old_p = p.replace(os.sep + "archived" + os.sep, os.sep, 1)
            if os.sep + "右拇指指尖-分析" + os.sep in p:
                old_p = p.replace(os.sep + "archived" + os.sep + "右拇指指尖-分析" + os.sep,
                                  os.sep + "右拇指指尖" + os.sep, 1)
            a_old = anchors(p, file_for_eval=old_p)
            old_hits = {k: v for k, v in a_old.items() if interesting(v)}
            old_ok = bool(old_hits) and all(os.path.exists(v) for v in old_hits.values())
            bad = {k: v for k, v in hits.items() if not os.path.exists(v)}
            if old_ok:
                broken.append((rel, bad))
            else:
                was_broken += 1
    print("锚点指向数据集/产物的脚本：正常 %d 个" % okcnt)
    print("被这次搬迁弄坏（原位置可用、新位置不可用）：%d 个" % len(broken))
    print("本来就坏（原位置也不可用，多为例外/已删产物）：%d 个" % was_broken)
    for rel, bad in broken[:40]:
        k, v = next(iter(bad.items()))
        print("   !! %-46s %s=%s" % (rel, k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
