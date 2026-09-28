# -*- coding: utf-8 -*-
"""枚举 archived/ 下「数据集根锚点」失准的脚本：脚本里用到 数据1~3 / 数据集名，
但没有任何一个路径锚点 + 该名字能在磁盘上找到。"""
import ast
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE)
ANCHORS = {"HERE", "OUT", "TEMP", "ROOT", "BASE", "DATA", "SCRIPTS", "RES", "FIG",
           "PROJ", "FLASH", "DS", "D", "SRC", "ART", "PROG", "OLD", "CACHE"}
LIT = re.compile(r'["\'](数据[123]|右拇指指尖|左拇指指尖|四指指尖|变化负载)["\']')


def eval_anchors(path):
    src = io.open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, {}
    ns = {"os": os, "sys": sys, "__file__": path}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id not in ANCHORS:
            continue
        try:
            ns[t.id] = eval(compile(ast.Expression(node.value), "<a>", "eval"), ns)  # noqa: S307
        except Exception:                                                        # noqa: BLE001
            pass
    return src, {k: v for k, v in ns.items() if k in ANCHORS and isinstance(v, str)}


def main():
    bad, ok = [], 0
    for root, _d, fs in os.walk(ARCH):
        for f in sorted(fs):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            src, a = eval_anchors(p)
            lits = sorted(set(LIT.findall(src)))
            if not lits or not a:
                continue
            reach = [l for l in lits
                     if any(os.path.exists(os.path.join(v, l)) for v in a.values())]
            if reach:
                ok += 1
            else:
                bad.append((os.path.relpath(p, ARCH), lits, a))
    print("用到数据集名且锚点可达的脚本：%d 个" % ok)
    print("数据集根锚点失准（需按注释重指到 temp/<数据集>）：%d 个" % len(bad))
    for rel, lits, a in bad:
        print("   %-52s 提到 %s" % (rel, "/".join(lits)))
        for k, v in list(a.items())[:2]:
            print("        %s = %s" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
