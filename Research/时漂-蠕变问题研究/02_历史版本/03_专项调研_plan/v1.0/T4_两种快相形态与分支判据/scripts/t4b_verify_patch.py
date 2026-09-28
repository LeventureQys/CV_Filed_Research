# -*- coding: utf-8 -*-
"""t4b_verify_patch.py（调试件）—— 证明 t4b_glm53_v6.py 的补丁**未改变算法行为**。

方法：同一份录制、同一 100 Hz 网格，分别用
  * 原版 `progress/07-v6/scripts/glm53_v6.py`
  * 补丁版 `scripts/t4b_glm53_v6.py`（默认 G_TAU/G_G）
跑完整链路，比较逐帧输出的最大绝对差。必须为 0（bit-level 相同）。

运行：python scripts/t4b_verify_patch.py
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    orig_path = os.path.join(C.ROOT, "temp", "v4.1flash", "progress", "07-v6",
                             "scripts", "glm53_v6.py")
    m1 = load_mod(orig_path, "glm53_v6_orig")
    m2 = load_mod(os.path.join(HERE, "t4b_glm53_v6.py"), "t4b_glm53_v6_patched")
    name, path, dom = C.RECS[10]      # 再切换负载：含 onset/restep/decrement
    d = __import__("t4b_ad_lib").prep(path)
    tu, Xu = d["tu"], d["Xu"]
    out = {}
    for tag, mod in (("orig", m1), ("patch", m2)):
        c = mod.GLM53v6(Xu.shape[1])
        Y = np.empty_like(Xu)
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
        out[tag] = Y
        print("%-6s epochs=%d  输出总量末值=%.6f" % (tag, len(c.epoch_t), float(Y.sum())))
    dm = float(np.abs(out["orig"] - out["patch"]).max())
    print("\n逐帧输出最大绝对差 = %.3e   → %s" % (dm, "PASS（补丁不改变行为）" if dm == 0 else "FAIL"))
    return 0 if dm == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
