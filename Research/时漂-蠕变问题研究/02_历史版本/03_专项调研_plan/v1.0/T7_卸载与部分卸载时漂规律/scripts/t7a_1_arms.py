# -*- coding: utf-8 -*-
"""T7-A / step1：13 份录制 × 4 臂（raw / v5.1(e3s) / v6 / v5）逐帧运行并落缓存。

只有 §B（算法耦合）与零点"两条线"需要算法输出；**§A 纯数据规律一律只用 raw**。
缓存写 results/cache/<tag>.npz（本任务目录内，可重复生成；不写其它桶）。

产出：results/cache/*.npz、results/_t7a_1_arms.log
用法：python scripts/t7a_1_arms.py [--only 关键字]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402

LOG = C.Log("1_arms")


def main():
    C.log_reconfigure()
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    LOG("=" * 100)
    LOG("T7-A step1：逐帧运行 4 臂（100 Hz 网格）。臂定义见 t7a_common.build_arm：")
    for a in C.ARMS:
        LOG(f"  {a:<5} {C.ARM_LABEL[a]}")
    LOG("=" * 100)
    LOG(f"{'录制':<18}{'nch':>4}{'帧数':>7}" + "".join(f"{a+'/s':>9}" for a in C.ARMS)
        + f"{'ded_eff@idle中位':>18}")
    t_all = time.time()
    for tag, path in C.ALL:
        if only and only not in tag:
            continue
        if not os.path.exists(path):
            LOG(f"[缺文件] {tag}")
            continue
        d = C.prep(tag)
        arms = {}
        ts = []
        for a in C.ARMS:
            t0 = time.time()
            arms[a] = C.run_arm(a, d["tu"], d["Xu"])
            ts.append(time.time() - t0)
        C.save_rec_cache(d, arms)
        ds = C.zbar(d["tot"], d["dtm"])
        gaps, eps, floor, peak = C.parse_plateaus(ds, d["tu"], d["dtm"])
        idle = np.zeros(len(ds), bool)
        for g in gaps:
            idle[g["i0"]:g["i1"]] = True
        # 空载段的实际生效扣除（应≈0：现役两臂空载不归零）
        msg = []
        for a in ("e3s", "v6", "v5"):
            de = arms[a]["ded_eff"]
            msg.append(f"{a} {np.median(de[idle]):+.3f}" if idle.any() else f"{a} n/a")
        LOG(f"{tag:<18}{d['nch']:>4}{len(d['tu']):>7}" + "".join(f"{t:>9.1f}" for t in ts)
            + "   " + "  ".join(msg))
    LOG(f"\n总耗时 {time.time() - t_all:.0f} s")
    LOG.close("python scripts/t7a_1_arms.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
