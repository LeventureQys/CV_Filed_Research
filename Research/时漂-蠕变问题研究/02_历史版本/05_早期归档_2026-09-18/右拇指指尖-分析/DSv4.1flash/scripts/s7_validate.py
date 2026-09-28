# -*- coding: utf-8 -*-
"""步骤7：算法库自检——在真实数据上验证每个算法的输出合理性。

真值定义（本次评测的核心约定）
------------------------------
恒定负载的物理真值是常数。把"阶跃建立后、漂移尚未显著"的窗口
    t_ref ∈ [1.0, 3.0] s（相对负载起点）
的均值定义为该通道的参考真值 F_ref。
所有算法的误差都相对 F_ref 计算，因此"把信号整体缩放"这种作弊
（例如把信号乘以 0.5）会被正确判为大误差。
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, DATASETS
import tac_algorithms as AL

np.set_printoptions(precision=4, suppress=True)

for name in DATASETS:
    D = load_dataset(name)
    fi = D["frame_index"]
    k, b = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), D["t"], fs=fs)
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    base = X[a:bb].mean(axis=0)
    Xn = X - base
    resp = Xn[c0:d0].max(axis=0)
    act = np.where(resp > 0.05)[0]
    main = int(act[np.argmax(resp[act])])
    w = lambda s: int(round(s * fs))
    Fref = Xn[c0 + w(1.0):c0 + w(3.0), main].mean()
    print("=" * 100)
    print(f"### {name}  主通道={D['ch_cols'][main]}  F_ref(1~3s)={Fref*1000:.3f} mN")

    cases = [
        ("Raw", AL.Raw()),
        ("ArrayShared-power", AL.ArraySharedShape("power", 0.5)),
        ("ArrayShared-log", AL.ArraySharedShape("log", 0.5)),
        ("ArrayShared-noparam", AL.ArraySharedShape(None, 0.5)),
        ("ArrayShared-late-power", AL.ArraySharedShape("power", 0.5, weights="late")),
        ("ModelFit-power", AL.ModelFit("power", 0.5)),
        ("ModelFit-log", AL.ModelFit("log", 0.5)),
        ("ModelFit-exp", AL.ModelFit("exp", 0.5)),
        ("SegBaseline-9", AL.SegBaseline(9, 2.0, 0.5)),
        ("SegBaseline-15", AL.SegBaseline(15, 2.0, 0.5)),
        ("CMR-mean-5s", AL.CommonModeRemoval("mean", 5.0, 0.5)),
        ("CMR-mean-20s", AL.CommonModeRemoval("mean", 20.0, 0.5)),
        ("DriftSub-k1", AL.DriftSubspace(1, 0.5)),
        ("DriftSub-k2", AL.DriftSubspace(2, 0.5)),
        ("CMRefFit-mean-o1", AL.CommonModeRefFit("mean", 1, 0.5)),
        ("ANC-0.5-64", AL.AdaptiveNoiseCanceller(0.5, 64, 0.5)),
        ("HPF-0.01", AL.ButterHighPass(0.01)),
        ("HPF-0.02", AL.ButterHighPass(0.02)),
        ("MA-5s", AL.MovingAverage(5.0)),
        ("Kalman-q1e-8", AL.KalmanDrift(1e-8, 1e-5, 1.0)),
    ]
    print(f"{'算法':<24s} {'末5s[mN]':>9s} {'误差%':>7s} {'相对漂移%':>9s} "
          f"{'0.5-1s/真值':>11s} {'1-2s/真值':>9s} {'噪声比%':>8s}")
    for label, c in cases:
        try:
            Y = c.transform(Xn, t, fs, c0)
        except Exception as e:
            print(f"{label:<24s}   FAILED: {e}")
            continue
        y = Y[:, main]
        late = y[d0 - w(5):d0].mean()
        err = (late - Fref) / Fref * 100
        drift = (y[d0 - w(5):d0].mean() - y[c0 + w(2):c0 + w(5)].mean()) / Fref * 100
        r1 = y[c0 + w(0.5):c0 + w(1.0)].mean() / Fref
        r2 = y[c0 + w(1.0):c0 + w(2.0)].mean() / Fref
        nz = np.diff(y[d0 - w(10):d0]).std() / np.sqrt(2)
        nz0 = np.diff(Xn[d0 - w(10):d0, main]).std() / np.sqrt(2)
        print(f"{label:<24s} {late*1000:>9.3f} {err:>7.2f} {drift:>9.2f} "
              f"{r1:>11.3f} {r2:>9.3f} {100*nz/nz0:>8.1f}")
    # 打印关键算法拟合出的形态参数
    for label in ("ArrayShared-power", "ModelFit-power"):
        c = dict(cases)[label]
        Y = c.transform(Xn, t, fs, c0)
        print(f"  [{label}] 共享形态参数={getattr(c, 'params', None)}  "
              f"R²={getattr(c, 'r2', None)}")
