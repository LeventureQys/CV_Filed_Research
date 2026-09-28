# -*- coding: utf-8 -*-
"""把 archived/ 下被「多一层目录」弄坏的路径锚点补回一级。

判定：某锚点求值结果不存在，但把它里面的第一段 `archived\\` 去掉后存在 ⇒ 属于深度错位。
修法：对该锚点赋值表达式的「前缀链」外面再加一层 os.path.dirname(...)：
  · `os.path.join(<链>, "x")` → 只给第一个实参加一层；
  · 裸链 → 整体加一层。
"""
import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ARCH = os.path.dirname(HERE)

ANCHORS = {"HERE", "OUT", "TEMP", "ROOT", "BASE", "DATA", "SCRIPTS", "RES", "FIG",
           "PROJ", "FLASH", "DS", "D", "SRC", "ART", "PROG", "OLD", "CACHE"}


def dearchive(p):
    return p.replace(os.sep + "archived" + os.sep, os.sep, 1)


def eval_anchors(path):
    src = io.open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, None, {}
    ns = {"os": os, "sys": sys, "__file__": path}
    nodes = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id not in ANCHORS:
            continue
        try:
            ns[t.id] = eval(compile(ast.Expression(node.value), "<a>", "eval"), ns)  # noqa: S307
            nodes[t.id] = node
        except Exception:                                                        # noqa: BLE001
            pass
    return src, tree, {k: (v, nodes[k]) for k, v in ns.items() if k in ANCHORS and isinstance(v, str)}


def broken_anchors(path):
    _src, _tree, a = eval_anchors(path)
    out = []
    for k, (v, node) in a.items():
        if os.path.exists(v):
            continue
        if os.path.exists(dearchive(v)):
            out.append((k, v, node))
    return out


def patch(path, apply=False):
    src, tree, _a = eval_anchors(path)
    bad = broken_anchors(path)
    if not bad:
        return 0
    new_src = src
    for k, v, node in bad:
        val = node.value
        seg = ast.get_source_segment(src, val)
        if seg is None:
            print("   !! 无法定位表达式:", path, k)
            continue
        if (isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute)
                and val.func.attr == "join" and len(val.args) >= 1):
            first = val.args[0]
            seg_first = ast.get_source_segment(src, first)
            new_seg = seg.replace(seg_first, "os.path.dirname(%s)" % seg_first, 1)
        else:
            new_seg = "os.path.dirname(%s)" % seg
        new_src = new_src.replace(seg, new_seg, 1)
    if apply and new_src != src:
        io.open(path, "w", encoding="utf-8", newline="").write(new_src)
    return len(bad)


def main():
    apply = "--apply" in sys.argv
    total = files = 0
    for root, _d, fs in os.walk(ARCH):
        for f in sorted(fs):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            bad = broken_anchors(p)
            if not bad:
                continue
            files += 1
            total += len(bad)
            print("%s %-52s %s" % ("[改]" if apply else "[dry]", os.path.relpath(p, ARCH),
                                   ", ".join("%s=%s" % (k, os.path.basename(v)) for k, v, _n in bad)))
            if apply:
                patch(p, apply=True)
    print("待修 %d 个文件 / %d 处锚点" % (files, total))
    if not apply:
        print("（dry-run；加 --apply 执行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
