# -*- coding: utf-8 -*-
"""v5.1（空载不归零）验证：零点行为 + 负载跟踪指标是否退化。

A. 3 份含零负载段的实采录制：零点段被钳成 0 的时长、零点段显示均值、卸载沿后 0.5s 显示；
B. 4 份变化负载录制：台阶捕获比（事件后 6s 显示增量/原始增量）与变载窗最大偏差；
C. 恒载 9 组：负载段时漂残余（末 10% − 首 10%，占电平 %）。
对照：raw / v3（现役）/ v5（含自动归零）/ v51（本次修改）。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402
from glm53_v51 import GLM53v51                                        # noqa: E402

B = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
ROOT = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp"
RECS = [("零负载-切换负载-零负载-再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("零负载-中途切换-零负载-切换负载", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"))]
STATIC = [(f"{pos}/数据{k}", os.path.join(ROOT, pos, f"数据{k}", "device_001_seg000.csv"))
          for pos in ("右拇指指尖", "四指指尖", "左拇指指尖") for k in (1, 2, 3)]
ALGOS = [("v3", GLM53v3), ("v5", GLM53v5), ("v51", GLM53v51)]


def run(cls, tu, Xu):
    c = cls(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def zero_segments(tot_s, dtm):
    peak = float(tot_s.max())
    zero = tot_s < 0.05 * peak
    segs, i = [], 0
    while i < len(zero):
        if zero[i]:
            j = i
            while j + 1 < len(zero) and zero[j + 1]:
                j += 1
            if (j - i) * dtm >= 2.0:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs, peak


print("=" * 116)
print("A. 零点（零负载段）行为：显示被显示层钳成 0 的时长 / 零点段显示均值 / 卸载沿后 0.5s 显示总量")
print(f"{'录制 / 零负载段':44s} {'算法':5s} {'被钳0时长':>10s} {'显示均值':>10s} {'卸载沿后0.5s':>12s}")
print("-" * 116)
for tag, path in RECS:
    if not os.path.exists(path):
        print(f"[skip] {tag}")
        continue
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    segs, peak = zero_segments(tot_s, dtm)
    Y = {"raw": Xu.copy()}
    for name, cls in ALGOS:
        Y[name] = run(cls, tu, Xu)
    load = tot_s >= 0.05 * peak
    trans = np.where((load[:-1]) & (~load[1:]))[0] + 1
    for a, b in segs:
        for name in ("raw", "v3", "v5", "v51"):
            s = Y[name][a:b + 1].sum(axis=1)
            clamp = (s <= 1.0).sum() * dtm
            edge = []
            for e in trans:
                if a <= e <= b + 1:
                    edge.append(float(Y[name][e:e + int(0.5 / dtm)].sum(axis=1).mean()))
            lbl = f"{tag[:26]} [{tu[a]:6.1f}~{tu[b]:6.1f}s]"
            print(f"{lbl:44s} {name:5s} {clamp:9.2f}s {s.mean():10.0f} "
                  f"{(' '.join(f'{v:.0f}' for v in edge) if edge else '-'):>12s}")
    print("-" * 116)

print("\nB. 负载跟踪指标（4 份变化负载录制；台阶捕获比 = 事件后 6s 显示增量 ÷ 原始增量，1.0 为完整透传）")
print(f"{'录制':30s} {'算法':5s} {'捕获比中位/最小':>16s} {'变载窗最大偏差中位(ADC)':>22s} {'全程最大偏差(ADC)':>16s}")
print("-" * 116)
for tag, path in RECS:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    ev = [e for e, _ in L.detect_events(tot, dtm)]
    Ys = {name: run(cls, tu, Xu) for name, cls in ALGOS}
    for name in ("v3", "v5", "v51"):
        y = L.med_smooth(Ys[name].sum(axis=1), 0.5 / dtm)
        gains, gaps = [], []
        for e in ev:
            pre = float(np.median(tot_s[max(0, e - int(2 / dtm)):e]))
            post = float(np.median(tot_s[min(len(tu) - 1, e + int(4 / dtm)):min(len(tu), e + int(6 / dtm))]))
            dl = post - pre
            if abs(dl) < 0.08 * tot_s.max():
                continue
            i0, i1 = max(0, int(e - 1 / dtm)), min(len(tu) - 1, int(e + 20 / dtm))
            d_raw = float(tot_s[i1] - tot_s[i0])
            d_y = float(y[i1] - y[i0])
            if abs(d_raw) > 1000:
                gains.append(d_y / d_raw)
            aa, bb = max(0, int(e - 1 / dtm)), min(len(tu), int(e + 12 / dtm))
            gaps.append(float(np.abs(y[aa:bb] - tot_s[aa:bb]).max()))
        gmax = float(np.abs(y - tot_s).max())
        gm = f"{np.median(gains):.2f}/{np.min(gains):.2f}" if gains else "-"
        print(f"{tag:30s} {name:5s} {gm:>16s} {(f'{np.median(gaps):.0f}' if gaps else '-'):>22s} {gmax:16.0f}")
    print("-" * 116)

print("\nC. 恒载 9 组：负载段时漂残余（末 10% − 首 10%，占本段电平 %；越小越好）")
print(f"{'数据集':22s} {'raw':>9s} {'v3':>9s} {'v5':>9s} {'v51':>9s}")
print("-" * 116)
acc = {"raw": [], "v3": [], "v5": [], "v51": []}
for tag, path in STATIC:
    if not os.path.exists(path):
        print(f"[skip] {tag}")
        continue
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    per = L.find_periods(tot, dtm)
    if not per:
        continue
    a, b = max(per, key=lambda p: p[1] - p[0])
    b = min(b, len(tu) - 1)
    n = b - a
    seg = slice(a, b)
    lvl = float(np.median(tot_s[seg]))
    row = {}
    for name in ("raw", "v3", "v5", "v51"):
        Y = Xu if name == "raw" else run(dict(ALGOS)[name], tu, Xu)
        s = Y[seg].sum(axis=1)
        nq = max(1, n // 10)
        row[name] = 100.0 * (s[-nq:].mean() - s[:nq].mean()) / max(lvl, 1e-9)
        acc[name].append(row[name])
    print(f"{tag:22s} {row['raw']:9.2f} {row['v3']:9.2f} {row['v5']:9.2f} {row['v51']:9.2f}")
print("-" * 116)
print(f"{'9 组中位':22s} " + " ".join(
    f"{np.median(acc[k]):9.2f}" for k in ("raw", "v3", "v5", "v51")))
