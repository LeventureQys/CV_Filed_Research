# -*- coding: utf-8 -*-
"""T6-00 设置：把既有原型/工具**复制**到本任务 scripts/ 并改写 import（前缀 t6_）。

只做复制与 import 改写，不改任何既有文件。改写内容逐条列在 REWRITE 里，可复查。
唯一一处"口径修正"是 `load_rec`：既有 `ad_lib.load_rec` 写死 `skiprows=24`，
本项目硬性要求**自动定位 `##Data` 行**，故在副本里替换该函数体（不改原型算法）。

产出：scripts/t6_{ad_lib,ad_v4,glm53_v3,glm53_v5,glm53_v51,glm53_v6,glm53_v61,a_common}.py
      results/_t6_00_setup_copy.log
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
assert os.path.isdir(os.path.join(ROOT, "temp", "v4.1flash", "progress")), ROOT
PROG = os.path.join(ROOT, "temp", "v4.1flash", "progress")
RES = os.path.join(TASK, "results")

NEED = ["t6_ad_lib.py", "t6_ad_v4.py", "t6_glm53_v3.py", "t6_glm53_v5.py",
        "t6_glm53_v51.py", "t6_glm53_v6.py", "t6_glm53_v61.py", "t6_a_common.py"]

SRC = {
    "t6_ad_lib.py":     (os.path.join(PROG, "07-v6", "scripts", "ad_lib.py"), [
        ("from glm53_v3 import GLM53v3", "from t6_glm53_v3 import GLM53v3"),
        ("from ad_v4 import GLM53v4, GLM53v4r", "from t6_ad_v4 import GLM53v4, GLM53v4r"),
    ]),
    "t6_ad_v4.py":      (os.path.join(PROG, "04-v5", "scripts", "ad_v4.py"), [
        ("from glm53_v3 import GLM53v3", "from t6_glm53_v3 import GLM53v3"),
    ]),
    "t6_glm53_v3.py":   (os.path.join(PROG, "04-v5", "scripts", "glm53_v3.py"), []),
    "t6_glm53_v5.py":   (os.path.join(PROG, "04-v5", "scripts", "glm53_v5.py"), [
        ("from glm53_v3 import GLM53v3", "from t6_glm53_v3 import GLM53v3"),
    ]),
    "t6_glm53_v51.py":  (os.path.join(PROG, "05-v5.1", "scripts", "glm53_v51.py"), [
        ("from glm53_v5 import GLM53v5", "from t6_glm53_v5 import GLM53v5"),
    ]),
    "t6_glm53_v6.py":   (os.path.join(PROG, "07-v6", "scripts", "glm53_v6.py"), []),
    "t6_glm53_v61.py":  (os.path.join(PROG, "08-v6.1", "scripts", "glm53_v61.py"), [
        ("from glm53_v6 import GLM53v6", "from t6_glm53_v6 import GLM53v6"),
        ("from glm53_v6 import ROM_TAU, ROM_G", "from t6_glm53_v6 import ROM_TAU, ROM_G"),
        ("from glm53_v6 import _cap", "from t6_glm53_v6 import _cap"),
    ]),
    "t6_a_common.py":   (os.path.join(PROG, "13-v6-assessment", "scripts", "a_common.py"), [
        ('HERE = os.path.dirname(os.path.abspath(__file__))\n'
         'ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))  # 仓库根\n'
         'P04 = os.path.join(ROOT, "temp", "v4.1flash", "progress", "04-v5", "scripts")\n'
         'P06 = os.path.join(ROOT, "temp", "v4.1flash", "progress", "07-v6", "scripts")\n'
         'P051 = os.path.join(ROOT, "temp", "v4.1flash", "progress", "05-v5.1", "scripts")\n'
         'for p in (P04, P06, P051):\n'
         '    if p not in sys.path:\n'
         '        sys.path.insert(0, p)\n',
         'HERE = os.path.dirname(os.path.abspath(__file__))\n'
         'TASK = os.path.dirname(HERE)\n'
         'PLAN = os.path.dirname(TASK)\n'
         'ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))   # 仓库根\n'
         'if HERE not in sys.path:\n'
         '    sys.path.insert(0, HERE)          # 只从本任务 scripts/ 取模块，不依赖既有桶\n'),
        ("from ad_lib import load_rec, med_smooth", "from t6_ad_lib import load_rec, med_smooth"),
        ("from glm53_v6 import GLM53v6", "from t6_glm53_v6 import GLM53v6"),
        ("from glm53_v51 import GLM53v51", "from t6_glm53_v51 import GLM53v51"),
    ]),
}

# ── load_rec：自动定位 ##Data（替换副本里的函数体） ──
OLD_LOAD = '''def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)'''
NEW_LOAD = '''def _data_hdr(path):
    """自动定位 ##Data 段：返回列名所在行号（0-based），**不得写死 skiprows**。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("##Data"):
                return i + 1
    raise RuntimeError("no ##Data marker in " + path)


def load_rec(p):
    """读录制：时间轴一律用 timestamp 列（禁用 elapsed）。"""
    df = pd.read_csv(p, skiprows=_data_hdr(p))
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)'''


def main():
    os.makedirs(RES, exist_ok=True)
    log = []
    ok = True
    for dst, (src, reps) in SRC.items():
        with open(src, "r", encoding="utf-8") as f:
            txt = f.read()
        n_rep = 0
        for a, b in reps:
            assert a in txt, f"pattern not found in {src}: {a[:60]!r}"
            txt = txt.replace(a, b)
            n_rep += 1
        if dst == "t6_ad_lib.py":
            assert OLD_LOAD in txt
            txt = txt.replace(OLD_LOAD, NEW_LOAD)
        head = (f'# -*- coding: utf-8 -*-\n'
                f'# [T6] 由 {os.path.relpath(src, ROOT)} 复制并改写 import（T6-00 设置脚本生成）。\n'
                f'# 改写: {n_rep} 处 import' + ('，+ load_rec 改自动定位 ##Data' if dst == "t6_ad_lib.py" else '') + '\n')
        outp = os.path.join(HERE, dst)
        with open(outp, "w", encoding="utf-8") as f:
            f.write(head + txt)
        # 自检：不得残留对既有桶模块名的 import
        bad = []
        for mod in ("ad_lib", "ad_v4", "glm53_v3", "glm53_v5", "glm53_v51", "glm53_v6", "glm53_v61"):
            for line in txt.splitlines():
                s = line.strip()
                if s.startswith(("import ", "from ")) and mod in s and not s.startswith(("from t6_", "import t6_")):
                    if f"from {mod} " in s or f"import {mod}\n" in s:
                        bad.append(s)
        st = "OK " if not bad else "BAD"
        if bad:
            ok = False
        log.append(f"{st} {dst:22s} <- {os.path.relpath(src, ROOT)}  rewrite={n_rep}  "
                   f"bytes={len(txt)}  residual_import={bad}")
    with io.open(os.path.join(RES, "_t6_00_setup_copy.log"), "w", encoding="utf-8") as f:
        f.write("T6-00 setup copy log\nROOT=" + ROOT + "\n" + "\n".join(log) + "\n")
    print("\n".join(log))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
