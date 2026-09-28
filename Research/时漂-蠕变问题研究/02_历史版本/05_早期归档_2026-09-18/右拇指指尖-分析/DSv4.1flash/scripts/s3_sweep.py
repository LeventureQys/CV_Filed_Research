# -*- coding: utf-8 -*-
"""步骤3：算法实装与横向评测（多进程并行版）。

评测协议（关键）
----------------
1. 时间轴：frame_index 线性映射到均匀时间（fs≈100.51Hz），避免 timestamp 重复。
2. 归一化起点：所有方法输入的都是"前空载基线已归零"的数据（统一前置步骤，不算算法差异）。
3. 真值代理：恒定负载的物理真值是常数。取"阶跃后 1.0~2.0s 的均值"作为参考真值 F_ref。
4. 指标（均在受载通道上，主通道另列）：
   - resid          = 负载末 5s 均值 - 负载 1~2s 均值    [N]
   - resid_ratio    = resid / step_amp                  [%]
   - step_fid       = 0~0.15s 响应 / 原始同窗口响应      [%]
   - noise_ratio    = 负载末 10s 高频噪声σ / 原始        [%]
   - zero_post      = 后空载末 3s 均值（应≈0）          [N]
"""
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, dump_json, DATASETS, RES

NPROC = 6


def prepare(name):
    D = load_dataset(name)
    fi = D["frame_index"]
    k, b = np.polyfit(fi, D["t"], 1)
    tu = fi * k
    fs = 1.0 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), D["t"], fs=fs)
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    return dict(name=name, X=X, t=tu, fs=fs, pre=(int(a), int(bb)),
                load=(int(c0), int(d0)), post=(int(d0), len(tu)), ch=D["ch_cols"])


def build_cases():
    import tac_algorithms as AL
    cs = []
    cs.append(AL.Raw())
    for w in (0.5, 1.0, 2.0, 5.0, 10.0):
        cs.append(AL.MovingAverage(w))
    for w in (0.5, 1.0, 2.0, 5.0):
        cs.append(AL.MedianFilter(w))
    for tau in (1.0, 5.0, 20.0):
        cs.append(AL.EWMA(tau))
    for fc in (0.005, 0.01, 0.02, 0.05, 0.1):
        cs.append(AL.ButterHighPass(fc, 2))
    for w in (1.0, 3.0):
        cs.append(AL.SavitzkyGolay(w, 2))
    cs.append(AL.WaveletDenoise("db4", 4))
    cs.append(AL.ModelFitCompensator("exp1", 0.2))
    cs.append(AL.ModelFitCompensator("exp2", 0.2))
    cs.append(AL.ModelFitCompensator("power", 0.2))
    for lam in (0.999, 0.9995, 0.9999, 0.99998):
        cs.append(AL.RLSDetrend(lam, 1, 0.2))
    for q, r in ((1e-9, 1e-5), (1e-8, 1e-5), (1e-7, 1e-5), (1e-6, 1e-5), (1e-5, 1e-5)):
        cs.append(AL.KalmanDrift(q, r, 0.2))
    cs.append(AL.ModelFitShapeShared(0.2, "exp1"))
    cs.append(AL.ModelFitShapeShared(0.2, "exp2"))
    for use, sm in (("mean", 5.0), ("mean", 20.0), ("median", 5.0)):
        cs.append(AL.CommonModeRemoval(use, sm, 0.2, True))
    cs.append(AL.CommonModeRemoval("mean", 5.0, 0.2, False))
    for k in (1, 2, 3):
        cs.append(AL.PCASubspace(k, False, 0.2))
    cs.append(AL.PCASubspace(1, True, 0.2))
    for ref, order in (("mean", 1), ("mean", 2), ("median", 1), ("max", 1)):
        cs.append(AL.CommonModeRefFit(ref, order, 0.2))
    for mu, M in ((0.2, 32), (0.5, 64), (1.0, 128), (0.5, 256)):
        cs.append(AL.AdaptiveNoiseCanceller(mu, M, "mean", 0.2))
    return cs


def metrics(Y, X, D, j):
    t, fs = D["t"], D["fs"]
    a, bb = D["pre"]; c0, d0 = D["load"]; n = len(t)
    w = lambda s: int(round(s * fs))
    y, x = Y[:, j], X[:, j]
    ref_pre = X[a:bb].mean(axis=0)[j]
    step_amp = X[c0:c0 + w(0.15)].mean() - ref_pre
    y1 = y[c0 + w(1.0):c0 + w(2.0)].mean()
    y_late = y[max(d0 - w(5.0), c0):d0].mean()
    resid = y_late - y1
    step_fid = ((y[c0:c0 + w(0.15)].mean() - y[a:bb].mean()) / step_amp
                if abs(step_amp) > 1e-9 else np.nan)
    seg_y = y[max(d0 - w(10.0), c0):d0]
    seg_x = x[max(d0 - w(10.0), c0):d0]
    nz_y = np.diff(seg_y).std() / np.sqrt(2)
    nz_x = np.diff(seg_x).std() / np.sqrt(2)
    z_pre = y[max(bb - w(1.0), 0):bb].mean() - ref_pre
    z_post = y[max(n - w(3.0), d0):].mean() - ref_pre
    return dict(resid=float(resid),
                resid_ratio=float(resid / step_amp * 100) if abs(step_amp) > 1e-9 else np.nan,
                step_amp=float(step_amp), step_fid=float(step_fid * 100),
                noise=float(nz_y),
                noise_ratio=float(nz_y / nz_x * 100) if nz_x > 0 else np.nan,
                ripple=float(seg_y.std()), zero_pre=float(z_pre), zero_post=float(z_post))


def rough(M):
    return float(np.mean(np.abs(np.diff(M, axis=1))))


_G = {}


def _init(D):
    _G["D"] = D


def _run_one(idx):
    import tac_algorithms as AL
    D = _G["D"]
    cases = build_cases()
    c = cases[idx]
    X, t, fs = D["X"], D["t"], D["fs"]
    a, bb = D["pre"]; c0, d0 = D["load"]
    base = X[a:bb].mean(axis=0)
    Xn = X - base[None, :]
    resp = Xn[c0:d0].max(axis=0)
    act = np.where(resp > 0.05)[0]
    main = int(act[np.argmax(resp[act])])
    t0 = time.time()
    try:
        Y = c.transform(Xn, t, fs, c0)
        ok = True
        err = ""
    except Exception as e:
        Y = Xn; ok = False; err = str(e)
    el = (time.time() - t0) * 1000
    m_main = metrics(Y, Xn, D, main)
    mm = [metrics(Y, Xn, D, j) for j in act]
    resid_abs = np.array([x["resid"] for x in mm])
    stepf = np.array([x["step_fid"] for x in mm])
    nzr = np.array([x["noise_ratio"] for x in mm])
    rr = np.array([x["resid_ratio"] for x in mm])
    w5 = int(round(5 * fs))
    y_late = Y[max(d0 - w5, c0):d0]
    x_late = Xn[max(d0 - w5, c0):d0]
    return dict(algo=c.name, group=getattr(c, "group", "?"), ok=ok, err=err,
                ms=el, main_ch=D["ch"][main], main=int(main),
                act=[int(i) for i in act], m_main=m_main,
                m_mean_resid=float(resid_abs.mean()),
                m_max_resid=float(np.abs(resid_abs).max()),
                m_med_resid_ratio=float(np.median(rr)),
                m_max_abs_resid_ratio=float(np.abs(rr).max()),
                m_mean_step_fid=float(np.nanmean(stepf)),
                m_worst_step_fid=float(np.nanmin(stepf)),
                m_med_noise_ratio=float(np.nanmedian(nzr)),
                m_rough_ratio=float(rough(y_late) / rough(x_late)) if rough(x_late) > 0 else np.nan)


if __name__ == "__main__":
    lines = []

    def P(s=""):
        print(s, flush=True)
        lines.append(s)

    summary = {}
    ncase = len(build_cases())
    for name in DATASETS:
        D = prepare(name)
        P("=" * 110)
        P(f"### {name}  fs={D['fs']:.3f}Hz  前空载[0,{D['pre'][1]}]  "
          f"负载{D['load']}  后空载{D['post']}  候选算法 {ncase} 个")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=NPROC, initializer=_init,
                                 initargs=(D,)) as ex:
            rows = list(ex.map(_run_one, range(ncase), chunksize=1))
        P(f"    评测耗时 {time.time()-t0:.1f}s")
        main = rows[0]["main"]
        act = rows[0]["act"]
        P(f"    主通道={D['ch'][main]}  受载通道数={len(act)}")
        summary[name] = dict(main=main, act=act, rows=rows)
        order = sorted(rows, key=lambda x: abs(x["m_med_resid_ratio"]))
        P(f"\n    {'算法':<34s} {'主通道残余[N]':>12s} {'相对%':>8s} {'阵列|均值|[N]':>12s} "
          f"{'阶跃保真%':>9s} {'噪声%':>7s} {'零漂后[mN]':>10s} {'耗时ms':>7s}")
        for x in order:
            mm = x["m_main"]
            P(f"    {x['algo']:<34s} {mm['resid']:>12.4f} {mm['resid_ratio']:>8.2f} "
              f"{x['m_mean_resid']:>12.4f} {mm['step_fid']:>9.1f} "
              f"{mm['noise_ratio']:>7.1f} {mm['zero_post']*1000:>10.3f} {x['ms']:>7.0f}")
    dump_json(summary, "C_sweep.json")
    with open(os.path.join(RES, "C_sweep.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    P("\n[ok] 步骤3 完成")
