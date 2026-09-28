# -*- coding: utf-8 -*-
"""重写 a8_cpp_proto_consts.csv：SubAgent 最后一次重跑时 C++ 侧解析失败（cpp=NaN/same=False），
这里用「直接读 C++ 头文件常量 + 原型类属性」的方式重新核对并落盘，保证产物可复查。"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FLASH = os.path.dirname(os.path.dirname(OUT))
RES = os.path.join(OUT, "results")
sys.path.insert(0, os.path.join(FLASH, "progress", "07-v6", "scripts"))
from glm53_v6 import GLM53v6  # noqa: E402

CPP = os.path.join(FLASH, "..", "..", "src", "domain", "drift_v6", "drift_v6_compensator.h")
CPP = os.path.normpath(CPP)

PAIRS = [("kDetFastS", "DET_FAST"), ("kDetGapS", "DET_GAP"), ("kDetLagS", "DET_LAG"),
         ("kDetK", "DET_K"), ("kDetRel", "DET_REL"), ("kDetAbsFrac", "DET_ABS_FRAC"),
         ("kDetIdleFrac", "DET_IDLE_FRAC"), ("kDetPersist", "DET_PERSIST"),
         ("kIdleSettleS", "IDLE_SETTLE"), ("kRevokeS", "REVOKE"),
         ("kTailGateS", "TAIL_GATE_S"), ("kTailGateFrac", "TAIL_GATE_FRAC"),
         ("kBackdateS", "BACKDATE_S")]


def main():
    src = io.open(CPP, encoding="utf-8", errors="replace").read()
    rows = []
    for cpp_name, proto_name in PAIRS:
        m = re.search(r"%s\s*=\s*([0-9.]+)" % re.escape(cpp_name), src)
        cpp_v = float(m.group(1)) if m else None
        proto_v = float(getattr(GLM53v6, proto_name, float("nan")))
        rows.append((cpp_name, cpp_v, proto_v, cpp_v is not None and abs(cpp_v - proto_v) < 1e-9))
    lines = ["const,cpp,proto,same"]
    for name, c, p, ok in rows:
        lines.append("%s,%s,%s,%s" % (name, "" if c is None else c, p, ok))
    io.open(os.path.join(RES, "a8_cpp_proto_consts.csv"), "w", encoding="utf-8",
            newline="\n").write("\n".join(lines) + "\n")
    n_ok = sum(1 for *_x, ok in rows if ok)
    print("-> results/a8_cpp_proto_consts.csv  %d/%d 一致" % (n_ok, len(rows)))
    for name, c, p, ok in rows:
        if not ok:
            print("   !! 不一致:", name, c, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
