# -*- coding: utf-8 -*-
"""校验快照 glm53_v6c_base_agentA.py 是否等于已验证版本（临时脚本）。"""
import importlib.util
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, os.pardir, "v4.1flash", "scripts"))
sys.path.insert(0, SCRIPTS)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

spec = importlib.util.spec_from_file_location(
    "v6c_snap", os.path.join(HERE, "glm53_v6c_base_agentA.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
GLM53v6c = M.GLM53v6c
g_shape = M.g_shape

DT, N = 0.01, 5
rng = np.random.default_rng(7)


def synth(load_sched, creep=0.35, tau_c=25.0, T=150.0, noise=3.0, base=40.0, drop=None):
    t = np.arange(0.0, T, DT)
    X = np.zeros((len(t), N))
    for i in range(len(t)):
        X[i] = base
        for ts, lv in load_sched:
            if t[i] < ts:
                continue
            uu = t[i] - ts
            fast = float(g_shape(uu)) if uu < 5.0 else 1.0
            X[i] += lv * fast * (1.0 + creep * (1.0 - np.exp(-uu / tau_c))) / N
    X *= (1.0 + 0.05 * np.arange(N) / N)
    X += rng.normal(0.0, noise, X.shape)
    if drop is not None:
        X[int(drop / DT)] *= 0.3
    return t, X


def run(cls, t, X, **kw):
    c = cls(X.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(X)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i])
    return Y, c


print("A. 快照回归等价性（两开关全关）vs v5.1：")
for tag, sched, dr in [("恒载 3000", [(5.0, 3000.0)], None),
                       ("两级加载", [(5.0, 2000.0), (40.0, 3000.0)], None),
                       ("掉点", [(5.0, 3000.0)], 12.0)]:
    t, X = synth(sched, drop=dr)
    ya, _ = run(GLM53v51, t, X)
    yb, _ = run(GLM53v6c, t, X, SHAPE_CORRECT=False, ACCEL_G_INIT=False)
    print(f"   {tag:10s} 最大逐帧差 = {np.max(np.abs(ya - yb)):.3e}")

t, X = synth([(5.0, 3000.0)], drop=12.0)
for nm, cls in [("v5.1", GLM53v51), ("v6c-snap", GLM53v6c)]:
    Y, c = run(cls, t, X)
    D = (X - Y).sum(axis=1)
    j = int(12.0 / DT)
    print(f"E. {nm:9s} 掉点附近扣除单帧最大变化 = {np.max(np.abs(np.diff(D[j - 100:j + 100]))):8.1f} ADC")

B = os.path.join(os.path.dirname(os.path.dirname(SCRIPTS)), "变化负载")
RECS = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]
print()
print("实录复核（快照）：")
for tag, path in RECS:
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    ev = [e for e, _ in L.detect_events(tot, dtm)]
    ne = {}
    for nm, cls in [("v5.1", GLM53v51), ("snap", GLM53v6c)]:
        Y = np.empty_like(Xu)
        c = cls(Xu.shape[1])
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
        gap = np.abs(L.med_smooth(Y.sum(axis=1), 0.5 / dtm) - tot_s)
        d["Ys"] = {"a": Y}
        et = L.event_table(d, ev, {"a": Y}, [], gain_s=6.0, algos=["a"])
        et["big"] = et["jump"].abs() >= 2000.0
        thr = 0.30 * et["pre"].max() if len(et) else 0.0
        et["mid"] = (et["pre"] > thr) & (et["post"] > thr) if len(et) else False
        ml = et[et.mid & et.big] if len(et) else et
        ne[nm] = (len(getattr(c, "epoch_t", [])) if hasattr(c, "epoch_t") else -1,
                  float(ml["gap_a"].median()) if len(ml) else float("nan"),
                  gap.max(), 100 * gap.max() / tot_s.max(), float(c.A.max()), float(c.g))
    for nm in ("v5.1", "snap"):
        e, gm, gx, gp, am, ge = ne[nm]
        print(f"  {tag:>16} {nm:>5} epoch={e:3d} gap中位={gm:8.0f} 全程最大={gx:7.0f} "
              f"({gp:5.2f}%) A_max={am:7.0f} g_end={ge:+.4f}")
