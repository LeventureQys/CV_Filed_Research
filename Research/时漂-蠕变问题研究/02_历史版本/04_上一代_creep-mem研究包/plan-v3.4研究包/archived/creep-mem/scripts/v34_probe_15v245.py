# -*- coding: utf-8 -*-
"""v3.4 第一步：零基线-反复增减同一负载 会话的沿检测 + 15s vs 245s 形状对比。

口径：
- 输入流 = pre（算法前显示域读数）；显示流 = main（算法结果）。
- 沿检测对 pre 总量做：50 帧中值滤波 -> 差分阈值（台阶 5% 以上）-> 合并。
- 每个加载沿输出：事件后 [0,0.5,1,2,3,5] s 的 (pre总, main总, main-pre)。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v30_lib as L

DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")


def medfilt(x, k):
    from scipy.signal import medfilt
    return medfilt(x, k)


def edges(el, tot_med, frac=0.05):
    """返回 [(t_edge, dir, step)]，dir=+1 加载 / -1 卸载。"""
    rng = tot_med.max() - tot_med.min()
    thr = frac * rng
    out = []
    prev = tot_med[0]
    t_prev = el[0]
    for i in range(200, len(el)):
        if el[i] - t_prev < 0.5:
            continue
        a = tot_med[i - 100:i].mean()
        b = tot_med[i:i + 100].mean()
        if abs(b - a) > thr:
            out.append((el[i], 1 if b > a else -1, b - a))
            t_prev = el[i]
    return out


def main():
    ds = L.load_dataset(DS)
    pre, main = ds["pre"], ds["main"]
    tp = pre["V"].sum(axis=1)
    tm = main["V"].sum(axis=1)
    el = pre["el"]
    tp_m = np.copy(tp)
    # 简易中值（无 scipy 依赖）：5 帧滑窗中值
    for i in range(2, len(tp) - 2):
        tp_m[i] = np.median(tp[i - 2:i + 3])

    es = edges(el, tp_m)
    print("检测到沿（t, 方向, 台阶）:")
    for t, d, s in es:
        print("  t=%7.2f  %s  step=%8.1f" % (t, "UP  " if d > 0 else "DOWN", s))

    print("\n每个加载沿后各时刻 (pre总, main总, 显示-输入):")
    for t, d, s in es:
        if d < 0:
            continue
        line = ["t=%7.2f" % t]
        for dt in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
            j = int(np.argmin(np.abs(el - (t + dt))))
            line.append("dt=%.1f:(%7.0f,%7.0f,%+6.0f)" % (dt, tp[j], tm[j], tm[j] - tp[j]))
        print("  " + " | ".join(line))


if __name__ == "__main__":
    main()
