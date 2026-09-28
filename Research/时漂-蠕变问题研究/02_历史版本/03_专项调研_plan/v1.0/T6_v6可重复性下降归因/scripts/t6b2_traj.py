# -*- coding: utf-8 -*-
"""T6-B2：微扰下的输出轨线族（供 `figures/T6_02_jitter_path.png` 左/中面板）。

只做一件事：实录 `中途切换-1d9493` 上，±1 包时序抖动（J = 100%·包周期，8 个种子）
对 v5.1 / v6 / v6.1 的**显示总量轨线**与基线轨线一起落盘（10 Hz 抽稀）。

产出：results/t6_jitter_traj_a.csv、results/_t6b2_traj.log
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import t6_common as C                                              # noqa: E402
import t6_a_common as AC                                            # noqa: E402
import t6_ad_lib as AL                                             # noqa: E402
from t6_glm53_v51 import GLM53v51                                  # noqa: E402
from t6_glm53_v6 import GLM53v6                                    # noqa: E402
from t6_glm53_v61 import GLM53v61                                  # noqa: E402

RES = C.RES
TAG = "中途切换-1d9493"
N_SEED = 8
TRACED = {"v6": AC.make_traced(GLM53v6), "v6.1": AC.make_traced(GLM53v61)}
LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def run(impl, tu, Xu):
    if impl == "v5.1":
        c = AL.make_traced(GLM53v51)(Xu.shape[1])
        Y = np.empty_like(Xu)
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
        return Y.sum(axis=1)
    r = AC.run_traced(TRACED[impl], tu, Xu)
    return r["Y"].sum(axis=1)


def main():
    d = C.grid(C.ALL[TAG])
    tu, Xu = d["tu"], d["Xu"]
    pid_raw, nk, P, fpp = C.packet_layout(d["t"])
    pkt = C.grid_packet_id(tu, d["t"], pid_raw)
    dec = np.arange(0, len(tu), 10)
    out = {"t": tu[dec], "raw": Xu.sum(axis=1)[dec]}
    for impl in ("v5.1", "v6", "v6.1"):
        out[f"base_{impl}"] = run(impl, tu, Xu)[dec]
        for s in range(N_SEED):
            rng = np.random.default_rng(20000 + s)
            tj, _ = C.jitter_axis(tu, pkt, P, 1.00 * P, rng)
            out[f"jit_{impl}_s{s}"] = run(impl, tj, Xu)[dec]
        p(f"  traj {impl} done")
    pd.DataFrame(out).to_csv(os.path.join(RES, "t6_jitter_traj_a.csv"), index=False,
                             encoding="utf-8-sig")
    with io.open(os.path.join(RES, "_t6b2_traj.log"), "w", encoding="utf-8") as fh:
        fh.write("T6-B2 traj log\n" + "\n".join(LOG) + "\n")
    p("-> results/t6_jitter_traj_a.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
