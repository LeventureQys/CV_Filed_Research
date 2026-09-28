# -*- coding: utf-8 -*-
"""T9-G：验证「空载不可识别（never-idle）」是否就是现场基线漂移的成因。

做法：回放现场 pre 流，但把原型的历史最小电平 min_ts 在首帧之后预置为低值
      （= 模拟"算法在装夹/加载之前就已启用"，这正是现场录制的起始状态：
       录制第 0 帧算法就已在扣除，说明补偿器不是从本录制起步的）。

对照：min_ts 不预置（fresh）时，同一段数据几乎不产生任何补偿（偏移 ≈ 0）。
"""
import csv
import importlib.util
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 6)))
DATA = os.path.join(ROOT, "temp", "算法数据&原始数据", "恒定负载下反复加减同一个负载",
                    "20260919_092417_single_device_602c03")
PROTO = os.path.join(ROOT, "temp", "v4.1flash", "progress", "07-v6", "scripts", "glm53_v6.py")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
NCH = 21

spec = importlib.util.spec_from_file_location("glm53_v6", PROTO)
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)
BASE = proto.GLM53v6


class WithHyst(BASE):
    REVOKE_HOLD_N = 1

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        if ev is None or self.REVOKE_HOLD_N <= 1:
            return super()._run_event(ts, v, total, dt, eps, idle_now)
        tau = ts - ev["t0"]
        inc_s = self._win_mean(ts - 0.10, ts)
        inc_s = (inc_s - ev["base"]) if inc_s is not None else (total - ev["base"])
        block = False
        if tau < self.REVOKE and ev["inc_max"] > eps and inc_s < 0.5 * ev["inc_max"]:
            ev["_rvn"] = ev.get("_rvn", 0) + 1
            if ev["_rvn"] < self.REVOKE_HOLD_N:
                ev["_saved_incmax"] = ev["inc_max"]
                ev["inc_max"] = 0.0
                block = True
        else:
            ev["_rvn"] = 0
        out = super()._run_event(ts, v, total, dt, eps, idle_now)
        if block and self.ev is not None:
            self.ev["inc_max"] = self.ev.get("_saved_incmax", self.ev["inc_max"])
        return out


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    ch = [i for i, h in enumerate(hdr) if h.startswith("ch") or h.startswith("raw_ch")]
    if not ch:
        ncol = len(rows[di + 2].split(","))
        ch = list(range(ncol - NCH, ncol))
    el, vals = [], []
    for r in rows[di + 2:]:
        if not r:
            continue
        f = r.split(",")
        el.append(float(f[1]))
        vals.append([float(f[ci]) for ci in ch])
    return np.array(el), np.array(vals)


def replay(el, V, kappa, ho, hold_n, seed=None, tag="x"):
    c = WithHyst(NCH)
    c.KAPPA_ONSET = kappa
    c.HO_MIN = ho
    c.REVOKE_HOLD_N = hold_n
    out = np.zeros_like(V)
    for i in range(len(el)):
        out[i] = c.process(float(el[i]), V[i])
        if i == 0 and seed is not None:
            c.min_ts = float(seed)       # 首帧之后预置：模拟历史最小电平锁在低值
    return dict(tot=out.sum(axis=1), epochs=list(c.kind_log), revokes=c.n_revoke,
                g=c.g, A=float(c.A.sum()), state=c.state, tag=tag)


el, PRE = load("device_001_pre_seg0.csv")
_, MAIN = load("device_001_seg000.csv")
tp, tm = PRE.sum(axis=1), MAIN.sum(axis=1)

plats = []
with open(os.path.join(OUT, "t9b_cycles.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        plats.append((r["kind"], float(r["t_start"]), float(r["t_end"]), float(r["pre_lvl"])))

AMP = 2999.0
runs = {
    "fresh_min_ts": replay(el, PRE, 1.05, 3.5, 3),
    "seed1900_new": replay(el, PRE, 1.05, 3.5, 3, seed=1900),
    "seed1900_old": replay(el, PRE, 1.30, 5.0, 1, seed=1900),
    "seed8000_new": replay(el, PRE, 1.05, 3.5, 3, seed=8000),
    "seed8000_old": replay(el, PRE, 1.30, 5.0, 1, seed=8000),
}

print("平台偏移（replay_tot − pre_tot，ADC）；现场 = 录制 main − pre")
hdr = f"{'kind':7s}{'t':>8s}{'pre':>7s}{'现场':>9s}"
for k in runs:
    hdr += f"{k[:14]:>17s}"
print(hdr)
for kind, a, b, p in plats:
    m = (el >= a) & (el <= b)
    line = f"{kind:7s}{a:8.2f}{p:7.0f}{np.median(tm[m])-p:9.0f}"
    for k, r in runs.items():
        line += f"{np.median(r['tot'][m])-p:17.0f}"
    print(line)

print("\n事件序列（t0, kind）：")
for k, r in runs.items():
    print(f"  {k:14s} n={len(r['epochs']):2d} revoke={r['revokes']:2d} " +
          ", ".join(f"{t:.1f}{kk[:4]}" for t, kk in r["epochs"][:26]))

print("\n末态：")
for k, r in runs.items():
    print(f"  {k:14s} state={r['state']:6s} A_sum={r['A']:9.0f} g={r['g']:+.4f}")

with open(os.path.join(OUT, "t9g_seed.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t_s", "pre_tot", "main_tot"] + list(runs))
    for i in range(0, len(el), 10):
        w.writerow([f"{el[i]:.3f}", f"{tp[i]:.0f}", f"{tm[i]:.0f}"] +
                   [f"{runs[k]['tot'][i]:.1f}" for k in runs])
print("\nwrote results/t9g_seed.csv")
