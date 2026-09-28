# -*- coding: utf-8 -*-
"""v2.0 阶段二 · 单侧限幅（C）的深度核查：它到底改了什么、有没有隐患。

关心三件事（全部逐帧可复核）：
  1. 限幅**实际触发频率与触发量级**（多久一次、每次削掉多少 ADC）——判断它是不是"隐形常量 0"
  2. 限幅对**空载/卸载段**的影响（负向偏移是否变差、ToIdle 是否被抑制）
  3. 限幅在**"显示该领先"的时段**（加载后快相窗）是否把有效补偿削掉
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

KAPPA_ONSET, KAPPA_RESTEP, HO_MIN = 1.05, 1.12, 3.5
ALPHA = 0.005


class Recorder(P.GLM53v6):
    def __init__(self, n, alpha=None):
        super().__init__(n)
        self.KAPPA_ONSET, self.KAPPA_RESTEP, self.HO_MIN = KAPPA_ONSET, KAPPA_RESTEP, HO_MIN
        self.alpha = alpha
        self.n_clip = 0
        self.clip_sum = 0.0
        self.clip_max = 0.0
        self.clip_ts = []

    def process(self, ts, v):
        raw = np.asarray(v, float).copy()
        out = np.asarray(super().process(ts, raw.copy()), float).copy()
        if self.alpha is not None:
            lim = raw + self.alpha * np.abs(raw)
            over = out - lim
            m = over > 0
            if m.any():
                self.n_clip += 1
                self.clip_sum += float(over[m].sum())
                self.clip_max = max(self.clip_max, float(over[m].max()))
                if len(self.clip_ts) < 40:
                    self.clip_ts.append((float(ts), float(over[m].sum())))
                out = np.minimum(out, lim)
        return out


def run(alpha):
    d = L.load_dataset(L.DS_ZERO)
    el = d["pre"]["el"]
    V = d["pre"]["V"]
    n, ch = V.shape
    c = Recorder(ch, alpha)
    out = np.empty((n, ch))
    for i in range(n):
        out[i] = c.process(float(el[i]), V[i])
    return el, V.sum(1), out.sum(1), c


def main():
    el, pre, o_base, c_base = run(None)
    el2, _, o_clip, c_clip = run(ALPHA)
    n = len(el)

    print(f"限幅 α={ALPHA} 触发统计：")
    print(f"  触发帧数 = {c_clip.n_clip} / {n}（{100*c_clip.n_clip/n:.1f}%）")
    print(f"  累计削掉 = {c_clip.clip_sum:.0f} ADC·帧；单帧最大削掉 = {c_clip.clip_max:.0f} ADC")
    print(f"  相对总量量级：单帧最大削掉 / 受载电平 17500 = {100*c_clip.clip_max/17500:.3f}%")
    print("  触发时刻（前 20 处，t 与削掉量）：")
    for t, s in c_clip.clip_ts[:20]:
        print(f"    t={t:8.2f}  clip={s:8.1f}")

    # 分区间比较
    print(f"\n{'区间(s)':>16}{'base off 中位':>14}{'限幅 off 中位':>14}"
          f"{'Δ':>9}{'base |off|max':>14}{'限幅 |off|max':>14}")
    spans = [(0, 5), (5, 40), (40, 50), (50, 110), (110, 120), (120, 137),
             (137, 150), (150, 210), (210, 220), (220, 287), (287, 320), (320, 337)]
    for a, b in spans:
        m = (el >= a) & (el < b)
        if not m.any():
            continue
        ob = o_base[m] - pre[m]
        oc = o_clip[m] - pre[m]
        print(f"{f'{a}~{b}':>16}{np.median(ob):14.0f}{np.median(oc):14.0f}"
              f"{np.median(oc)-np.median(ob):9.0f}"
              f"{np.max(np.abs(ob)):14.0f}{np.max(np.abs(oc)):14.0f}")

    # 加载后快相窗：显示领先量是否被削
    print("\n加载后 0~1.2 s 的补偿量（显示 − 原始；负 = 显示低于原始）：")
    print(f"{'沿(s)':>9}{'base 中位':>11}{'限幅 中位':>11}{'base 最小':>11}{'限幅 最小':>11}")
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(pre, 3), np.percentile(pre, 97)
    up, _ = L.edges_from_tot(np.convolve(pre, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in up:
        if not tu or el[i] - tu[-1] > 1.0:
            tu.append(float(el[i]))
    for t in tu:
        m = (el >= t) & (el <= t + 1.2)
        if not m.any():
            continue
        db = o_base[m] - pre[m]
        dc = o_clip[m] - pre[m]
        print(f"{t:9.2f}{np.median(db):11.0f}{np.median(dc):11.0f}"
              f"{db.min():11.0f}{dc.min():11.0f}")


if __name__ == "__main__":
    main()
