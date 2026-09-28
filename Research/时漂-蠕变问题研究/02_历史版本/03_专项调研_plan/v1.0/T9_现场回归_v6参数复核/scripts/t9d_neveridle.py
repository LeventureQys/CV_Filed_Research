# -*- coding: utf-8 -*-
"""T9-D：验证「never-idle」假设 —— 把 min_ts_ 预置为低电平（模拟"算法在装夹前就已启用"），
再回放现场 pre 流，比较 旧参数 vs 新参数 的输出，并对照现场录制的 main 流。

若预置后能复现现场那种「空载时整段偏移且不回落」，而两套参数都复现 ⇒ 病因是状态机的
「空载不可识别」，与本次参数改动无关（参数只改变偏移的符号/量级）。
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


def replay(el, V, kappa, ho, hold_n, min_ts_seed=None, tag="x", warm_bare_s=0.0):
    c = WithHyst(NCH)
    c.KAPPA_ONSET = kappa
    c.HO_MIN = ho
    c.REVOKE_HOLD_N = hold_n
    out = np.zeros_like(V)
    gl, Al, st = [], [], []
    tshift = 0.0
    if warm_bare_s > 0.0:
        # 预热：裸传感器低电平 3 s（模拟"算法已启用、样品还没装上"）
        for i in range(int(warm_bare_s * 100)):
            c.process(tshift, V[0] * 0.15)
            tshift += 0.01
        # 装夹台阶：跳到真实起始电平并保持 2 s
        for i in range(200):
            c.process(tshift, V[0])
            tshift += 0.01
    if min_ts_seed is not None:
        c.min_ts = float(min_ts_seed)          # 模拟历史最小电平被锁在低值
    for i in range(len(el)):
        out[i] = c.process(float(tshift + el[i]), V[i])
        gl.append(c.g)
        Al.append(float(c.A.sum()))
        st.append(c.state)
    return dict(tot=out.sum(axis=1), g=np.array(gl), A=np.array(Al), state=st,
                epochs=list(c.kind_log), revokes=c.n_revoke, tag=tag)


el, PRE = load("device_001_pre_seg0.csv")
_, MAIN = load("device_001_seg000.csv")
tp, tm = PRE.sum(axis=1), MAIN.sum(axis=1)

plats = []
with open(os.path.join(OUT, "t9b_cycles.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        plats.append((r["kind"], float(r["t_start"]), float(r["t_end"]), float(r["pre_lvl"])))

AMP = 2999.0

runs = {
    "fresh_new": replay(el, PRE, 1.05, 3.5, 3),
    "fresh_old": replay(el, PRE, 1.30, 5.0, 1),
    "neveridle_new(min_ts=1900)": replay(el, PRE, 1.05, 3.5, 3, min_ts_seed=1900),
    "neveridle_old(min_ts=1900)": replay(el, PRE, 1.30, 5.0, 1, min_ts_seed=1900),
    "warmbare_new": replay(el, PRE, 1.05, 3.5, 3, warm_bare_s=3.0),
    "warmbare_old": replay(el, PRE, 1.30, 5.0, 1, warm_bare_s=3.0),
}

print("平台偏移 = replay_tot − pre_tot（ADC）；括号内 = 占负载幅度 2999 的 %")
print(f"{'kind':7s}{'t_start':>8s}{'pre':>7s}{'现场':>9s}  " +
      "".join(f"{k[:26]:>28s}" for k in runs))
for kind, a, b, p in plats:
    m = (el >= a) & (el <= b)
    line = f"{kind:7s}{a:8.2f}{p:7.0f}{(np.median(tm[m]) - p):9.0f}  "
    for k, r in runs.items():
        off = np.median(r["tot"][m]) - p
        line += f"{off:12.0f}({off/AMP*100:+5.1f}%)".rjust(28)
    print(line)

print("\n事件序列：")
for k, r in runs.items():
    print(f"  {k:28s} n={len(r['epochs']):2d} revoke={r['revokes']:2d} " +
          ", ".join(f"{t:.1f}{kk[:4]}" for t, kk in r["epochs"][:20]))

print("\n末段（52~62 s）状态与 g / A：")
for k, r in runs.items():
    m = el > 52
    gs = r["g"][m]
    As = r["A"][m]
    print(f"  {k:28s} g: [{gs.min():+.3f},{gs.max():+.3f}]  A_sum: [{As.min():.0f},{As.max():.0f}] "
          f"末态={r['state'][-1]}")

# 落盘（便于画图/复核）
with open(os.path.join(OUT, "t9d_neveridle.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t_s", "pre_tot", "main_tot"] + [f"{k}" for k in runs])
    for i in range(0, len(el), 10):
        w.writerow([f"{el[i]:.3f}", f"{tp[i]:.0f}", f"{tm[i]:.0f}"] +
                   [f"{runs[k]['tot'][i]:.1f}" for k in runs])
print("\nwrote results/t9d_neveridle.csv")
