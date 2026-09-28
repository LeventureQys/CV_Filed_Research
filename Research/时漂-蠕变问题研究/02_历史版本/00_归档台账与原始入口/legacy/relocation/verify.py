# -*- coding: utf-8 -*-
"""搬迁 + 补丁后的自检：锚点解析、同目录 import、结果/图片引用可达性。"""
import ast
import io
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FLASH = os.path.join(os.path.dirname(HERE), "v4.1flash")
PROG = os.path.join(FLASH, "progress")
TEMP = os.path.dirname(FLASH)
REPO = os.path.dirname(TEMP)

STDLIB = set(sys.stdlib_module_names) | {"numpy", "pandas", "matplotlib", "scipy",
                                         "PIL", "cv2", "deepseek", "openai", "requests"}


def anchors(path):
    """按顺序求值顶层的路径锚点赋值（不执行 import / 函数调用语句）。"""
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"__syntax_error__": repr(e)}
    ns = {"os": os, "sys": sys, "__file__": path}
    keep = {"HERE", "OUT", "TEMP", "ROOT", "RES", "FIG", "FLASH", "CACHE", "PROG"}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id not in keep:
            continue
        try:
            ns[t.id] = eval(compile(ast.Expression(node.value), "<a>", "eval"), ns)  # noqa: S307
        except Exception:                                            # noqa: BLE001
            pass
    return {k: ns[k] for k in keep if k in ns and isinstance(ns[k], str)}


def main():
    buckets = [b for b in sorted(os.listdir(PROG)) if os.path.isdir(os.path.join(PROG, b))]
    scripts = {}            # bucket -> [names]
    all_modules = defaultdict(list)
    for b in buckets:
        sdir = os.path.join(PROG, b, "scripts")
        if os.path.isdir(sdir):
            scripts[b] = [f for f in sorted(os.listdir(sdir)) if f.endswith(".py")]
            for f in scripts[b]:
                all_modules[f[:-3]].append(b)

    problems = []
    n_ok = 0

    # --- 1) 锚点 ---
    for b, files in scripts.items():
        bdir = os.path.join(PROG, b)
        for f in files:
            p = os.path.join(bdir, "scripts", f)
            a = anchors(p)
            if "__syntax_error__" in a:
                problems.append("%s/%s 语法错误 %s" % (b, f, a["__syntax_error__"]))
                continue
            exp = {"OUT": bdir, "RES": os.path.join(bdir, "results"),
                   "FIG": os.path.join(bdir, "figures"), "TEMP": TEMP}
            bad = []
            for k, want in exp.items():
                if k in a and os.path.normcase(os.path.normpath(a[k])) != os.path.normcase(os.path.normpath(want)):
                    bad.append("%s=%s(应为 %s)" % (k, a[k], want))
            # ROOT 语义按脚本而定：仓库根 / 桶目录 / temp 都接受，只记录不报错
            if "ROOT" in a and os.path.normcase(a["ROOT"]) not in (
                    os.path.normcase(REPO), os.path.normcase(bdir), os.path.normcase(TEMP)):
                bad.append("ROOT=%s(非仓库根/桶/temp)" % a["ROOT"])
            if bad:
                problems.append("%s/%s 锚点不符: %s" % (b, f, "; ".join(bad)))
            else:
                n_ok += 1

    # --- 2) 同目录 import ---
    imp_bad = []
    for b, files in scripts.items():
        local = {f[:-3] for f in files}
        for f in files:
            p = os.path.join(PROG, b, "scripts", f)
            with io.open(p, "r", encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [al.name.split(".")[0] for al in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module.split(".")[0]]
                for m in mods:
                    if m in STDLIB or m in local:
                        continue
                    where = all_modules.get(m)
                    if where:                      # 归档里有这个模块，但不在本桶
                        imp_bad.append("%s/%s import %s（该模块在 %s）" % (b, f, m, where))

    # --- 3) 结果 / 图片引用可达性 ---
    res_missing = []
    known = {}
    for b in buckets:
        rd = os.path.join(PROG, b, "results")
        if os.path.isdir(rd):
            for root, _d, fs in os.walk(rd):
                for f in fs:
                    known.setdefault(f, []).append(b)

    for b, files in scripts.items():
        bdir = os.path.join(PROG, b)
        have_res = set()
        rd = os.path.join(bdir, "results")
        if os.path.isdir(rd):
            for root, _d, fs in os.walk(rd):
                have_res |= set(fs)
        for f in files:
            p = os.path.join(bdir, "scripts", f)
            with io.open(p, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            for i, line in enumerate(lines):
                for r in set(re.findall(r'"([A-Za-z0-9_\-\.]+\.(?:csv|npz|txt|log|json))"', line)):
                    if r in have_res:
                        continue
                    # 只关心“读”，写产物不算
                    if not re.search(r"read_csv|np\.load|load_npz|read_table|read_excel", line):
                        continue
                    others = [x for x in known.get(r, []) if x != b]
                    if others:
                        res_missing.append("%s/%s:%d 读 %s —— 本桶缺失，位于 %s"
                                           % (b, f, i + 1, r, others))

    print("== 锚点自检 ==")
    print("  通过 %d 个脚本；问题 %d" % (n_ok, len(problems)))
    for x in problems[:40]:
        print("   ! " + x)
    print("== 同目录 import 自检 ==")
    print("  问题 %d" % len(imp_bad))
    for x in sorted(set(imp_bad))[:40]:
        print("   ! " + x)
    print("== 结果引用可达性（跨桶读取提示，未必是错误）==")
    print("  提示 %d" % len(res_missing))
    for x in sorted(set(res_missing))[:40]:
        print("   - " + x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
