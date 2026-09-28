# -*- coding: utf-8 -*-
"""v2.0 阶段二 · 长保压段的"原始 vs 显示"分段量化（用于限幅的收益/代价裁决）。

对每个受载平台段（不跨事件）给出：
  pre_slope   原始读数在该段的漂移（占段中位 %）
  base_slope  base 显示的漂移
  a005_slope  α=0.005 显示的漂移
  supp_base   base 压掉的比例 = 1 − dOut/dPre
  supp_a005   α=0.005 压掉的比例
  off_zero    段内 |显示−原始| 是否被限幅压在 α·读数 以内
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class Arm(P.GLM53v6):
    def __init__(self, n, alpha=None):
        super().__init__(n)
        self.KAPPA_ONSET, self.KAPPA_RESTEP, self.HO_MIN = 1.05, 1.12, 3.5
        self.alpha = alpha

    def process(self, ts, v):
        raw = np.asarray(v, float).copy()
        out = np.asarray(super().process(ts, raw.copy()), float).copy()
        if self.alpha is not None:
            out = np.minimum(out, raw + self.alpha * np.abs(raw))
        return out


def run(el, V, alpha):
    c = Arm(V.shape[1], alpha)
    out = np.empty((len(el), V.shape[1]))
    for i in range(len(el)):
        out[i] = c.process(float(el[i]), V[i])
    return out.sum(1)


def main():
    d = L.load_dataset(L.DS_ZERO)
    el = d["pre"]["el"]
    V = d["pre"]["V"]
    pre = V.sum(1)
    ob = run(el, V, None)
    o5 = run(el, V, 0.005)
    o2 = run(el, V, 0.02)

    segs = L.plateau_segments(pre, el, min_dur=5.0)
    print(f"{'段':>3}{'kind':>8}{'t0':>8}{'t1':>8}{'dur':>7}"
          f"{'dPre':>9}{'dBase':>9}{'dA005':>9}{'supp_base':>10}{'supp_a005':>10}"
          f"{'off_base':>9}{'off_a005':>9}")
    for i, (kind, a, b) in enumerate(segs):
        if kind != "loaded" or b - a < 300:
            continue
        # 去掉首尾各 10% 避免事件边界
        pad = int(0.10 * (b - a))
        i0, i1 = a + pad, b - pad
        if i1 - i0 < 200:
            continue
        dp = pre[i1] - pre[i0]
        db = ob[i1] - ob[i0]
        d5 = o5[i1] - o5[i0]
        pb = np.median(pre[i0:i1])
        sb = 100 * (1 - db / dp) if abs(dp) > 1 else float("nan")
        s5 = 100 * (1 - d5 / dp) if abs(dp) > 1 else float("nan")
        offb = np.median(ob[i0:i1] - pre[i0:i1])
        off5 = np.median(o5[i0:i1] - pre[i0:i1])
        print(f"{i:>3}{kind:>8}{el[a]:8.1f}{el[b]:8.1f}{el[b]-el[a]:7.1f}"
              f"{dp:9.0f}{db:9.0f}{d5:9.0f}{sb:9.1f}%{s5:9.1f}%"
              f"{offb:9.0f}{off5:9.0f}")


if __name__ == "__main__":
    main()
