# -*- coding: utf-8 -*-
"""v3.6 探针 6：逐事件的**交接帧内部量**对账。

每个上行沿，取该 epoch 交接（state Event→Slow）的当帧，打印：
  base(原始沿前电平) / base_y(显示沿前电平) / y0_sum / Â / inc_max / 交接当帧输入
  / 该次加载的**实测落定增量**(由电平表外部给出：沿后 6~16 s 中位 − 沿前 3 s 中位)
用于回答：「Â 相对真实落定电平偏低多少？蠕变比 r 取多少才把 A 拉回无蠕变电平？」
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
import v36_replay as R  # noqa: E402
from v36_probe_internal import R_arm  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

# 上行沿 + 该次加载的「落定窗」（手工核验，见 results/v36_level_steps.txt）
EVENTS = [
    (3.34, 10.0, 14.0), (15.18, 22.0, 33.0), (41.75, 48.0, 56.0), (63.41, 70.0, 90.0),
    (232.23, 234.0, 239.0), (245.31, 248.0, 251.5), (258.25, 262.5, 275.0),
    (278.50, 282.0, 282.4), (289.50, 293.0, 295.0), (300.50, 305.0, 307.0),
]


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "v36:--seed-gain:1.0"
    ds = K.load(K.DS_TARGET)
    exe, args = R_arm(arm)
    st = R.run_arm(ds, exe, args)
    t = st["t"]
    tin, tout = ds["tot_in"], ds["tot_out"]
    el = ds["pre"]["el"]

    def wmed(x, a, b):
        m = (el >= a) & (el < b)
        return float(np.median(x[m])) if m.any() else float("nan")

    lines = ["== 臂 %s：逐事件交接帧对账 ==" % arm,
             "%8s %9s %9s %9s %9s %9s %9s %9s %8s %8s" %
             ("t_on", "base", "base_y", "y0_sum", "A_hat", "inc_max", "落定增量", "A_ho", "r_est", "ded_ho")]
    for (te, w0, w1) in EVENTS:
        m = (t >= te) & (t <= te + 8.0)
        idx = np.where(m)[0]
        # 交接帧 = 该窗内第一帧 state 由 1 变 0/2 的帧
        ho = None
        for i in idx:
            if i > 0 and st["state"][i - 1] == 1 and st["state"][i] != 1:
                ho = i
                break
        base_in = wmed(tin, te - 3.0, te - 0.5)
        set_in = wmed(tin, w0, w1)
        dset = set_in - base_in
        if ho is None:
            lines.append("%8.2f %9.0f %9s %9s %9s %9s %9.0f %9s %8s %8s"
                         % (te, base_in, "-", "-", "-", "-", dset, "-", "-", "-"))
            continue
        last_ev = ho - 1   # ev_ 在 Handoff 内被清空 ⇒ 取交接前一帧（事件末帧）
        ah, im, ab, aby, ay0 = (st["A_hat"][last_ev], st["inc_max"][last_ev],
                                st["ev_base"][last_ev], st["ev_base_y"][last_ev],
                                st["ev_y0_sum"][last_ev])
        aho = st["A_sum"][ho]
        rest = aho - aby
        r_est = (im - rest) / im if abs(im) > 1 else float("nan")
        lines.append("%8.2f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %8.3f %8.0f"
                     % (te, base_in, aby, ay0, ah, im, dset, aho, r_est,
                        st["sum_in"][ho] - st["sum_out"][ho]))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(RES, "v36_handoff_check.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print("-> %s" % os.path.join(RES, "v36_handoff_check.txt"))


if __name__ == "__main__":
    main()
