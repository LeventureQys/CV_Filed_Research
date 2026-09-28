# -*- coding: utf-8 -*-
"""探针 F（v3.1）：让离线回放**对齐现场**——试不同的「录制前历史」前置长度。

录制开始时算法已在运行（本次录制前用户先开了 v6），录制文件里没有那段历史；
本探针用「把首帧空载电平前置 K 秒」来近似，看能否把回放拉回录制结果。

输出：results/v31_align.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3.0", "scripts")))
import v30_ab as A  # noqa: E402

WORK = os.path.join(L.DATA_ROOT, "working")
DS = os.path.join(WORK, "零基线-反复增减同一负载",
                  "20260919_152749_single_device_0cb8b6")
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def main():
    el, pre = A.read_any(os.path.join(DS, "device_001_pre_seg0.csv"))
    _, main = A.read_any(os.path.join(DS, "device_001_seg000.csv"))
    tin, tout = pre.sum(1), main.sum(1)
    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("现场：录制开始前算法已在运行（manifest 显示录制时就已开 v6），前置历史不可得")
    p("")
    p("=== 前置 K 秒空载（用首帧值）后回放，与录制的吻合度 ===")
    p("%6s %10s %10s %10s %10s %10s" %
      ("K(s)", "中位|Δ|", "RMS", "max|Δ|", "首5s偏移", "末段偏移"))
    for K in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0):
        if K > 0:
            n0 = int(K * 100)
            pad_el = np.arange(0, K, 1.0 / 100.0)[:n0] - K
            padV = np.repeat(pre[:1], n0, axis=0)
            e2 = np.concatenate([pad_el, el])
            V2 = np.concatenate([padV, pre], axis=0)
        else:
            e2, V2 = el, pre
        D = A.run_arm(A.RUNNERS["base"], [], e2, V2)
        o = D["sum_out"][-len(el):]
        d = o - tout
        p("%6.0f %10.1f %10.1f %10.1f %10.0f %10.0f" %
          (K, np.median(np.abs(d)), float(np.sqrt((d ** 2).mean())), np.abs(d).max(),
           np.median(d[(el >= 15) & (el < 20)]), np.median(d[(el >= 330) & (el < 335)])))
    p("")
    p("=== 参照：完全无前置时，逐 10 s 的 Δ（复算−录制）===")
    D = A.run_arm(A.RUNNERS["base"], [], el, pre)
    d = D["sum_out"] - tout
    for a in np.arange(0, 343, 20.0):
        m = (el >= a) & (el < a + 20)
        if m.any():
            p("  t=%6.1f  Δ 中位 %+8.0f   输入 %8.0f  录制 %8.0f  复算 %8.0f" %
              (a, np.median(d[m]), np.median(tin[m]), np.median(tout[m]),
               np.median(D["sum_out"][m])))
    with open(os.path.join(RES, "v31_align.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % os.path.join(RES, "v31_align.txt"))


if __name__ == "__main__":
    main()
