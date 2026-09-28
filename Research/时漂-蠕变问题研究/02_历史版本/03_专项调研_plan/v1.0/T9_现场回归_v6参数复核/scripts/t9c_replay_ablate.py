# -*- coding: utf-8 -*-
"""T9-C：把现场录制的「算法前」流离线回放到 v6 原型上，并做参数消融。

目的：判断现场「基线飞掉」是否由本次参数改动（A1 κ_onset 1.05 / A4 HO_MIN 3.5 / A5a 撤销迟滞 3 帧）
      引入，还是 v6 本身在「同一负载反复加减」工况下的固有失效。

方法：
  1) 用原型 temp/v4.1flash/progress/07-v6/scripts/glm53_v6.py（C++ 逐行对齐）回放 pre 流；
  2) 与现场录制的 main 流逐帧比对（先验证原型能否复现 C++ 的输出）；
  3) 对 3 个参数做组合消融（新/旧/单改），比较每个空载平台上的基线偏移。
输出：results/t9c_replay_{tag}.csv、stdout 对比表
"""
import csv
import importlib.util
import os
import sys

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
    """A5a：撤销判据连续 REVOKE_HOLD_N 帧确认（=1 即原型原行为）。"""
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


def replay(el, V, kappa, ho_min, hold_n, tag):
    c = WithHyst(NCH)
    c.KAPPA_ONSET = kappa
    c.HO_MIN = ho_min
    c.REVOKE_HOLD_N = hold_n
    out = np.zeros_like(V)
    gtrace, Atrace, state = [], [], []
    for i in range(len(el)):
        o = c.process(float(el[i]), V[i])
        out[i] = o
        gtrace.append(c.g)
        Atrace.append(float(c.A.sum()))
        state.append(c.state)
    tot_out = out.sum(axis=1)
    with open(os.path.join(OUT, f"t9c_replay_{tag}.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "pre_tot", "replay_tot", "main_tot", "off_replay", "g", "A_sum", "state"])
        for i in range(0, len(el), 10):
            w.writerow([f"{el[i]:.3f}", f"{V[i].sum():.0f}", f"{tot_out[i]:.1f}",
                        "", f"{tot_out[i]-V[i].sum():.0f}", f"{gtrace[i]:.4f}",
                        f"{Atrace[i]:.0f}", state[i]])
    info = dict(tag=tag, epochs=len(c.epoch_t), revokes=c.n_revoke, c5=c.n_c5,
                kind_log=c.kind_log, tot_out=tot_out, g=gtrace, A=Atrace, state=state)
    return info


el, PRE = load("device_001_pre_seg0.csv")
_, MAIN = load("device_001_seg000.csv")
tp, tm = PRE.sum(axis=1), MAIN.sum(axis=1)

# 现场空载/受载平台（沿用 T9-B 的切段结果）
plats = []
with open(os.path.join(OUT, "t9b_cycles.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        plats.append((r["kind"], float(r["t_start"]), float(r["t_end"]),
                      float(r["pre_lvl"]), float(r["main_lvl"])))

AMP = 2999.0
variants = [
    ("new_A1A4A5a", 1.05, 3.5, 3),
    ("old_baseline", 1.30, 5.0, 1),
    ("A1only_k105", 1.05, 5.0, 1),
    ("A4only_ho35", 1.30, 3.5, 1),
    ("A5aonly_h3", 1.30, 5.0, 3),
]
res = {}
for tag, k, ho, hn in variants:
    res[tag] = replay(el, PRE, k, ho, hn, tag)
    r = res[tag]
    print(f"[{tag:14s}] κ={k} HO={ho} holdN={hn}  epochs={r['epochs']:3d} revokes={r['revokes']:3d} "
          f"c5={r['c5']:3d}")

print("\n各变体在每个平台上的偏移（replay − pre）/ 现场实际偏移（main − pre），单位 ADC")
hdr = f"{'kind':6s} {'t_start':>8s} {'t_end':>7s} {'pre_lvl':>8s} {'现场off':>9s} " + \
      " ".join(f"{t:>12s}" for t, *_ in variants)
print(hdr)
for kind, a, b, p_lvl, m_lvl in plats:
    m = (el >= a) & (el <= b)
    row = f"{kind:6s} {a:8.2f} {b:7.2f} {p_lvl:8.0f} {m_lvl-p_lvl:9.0f} "
    for tag, *_ in variants:
        o = res[tag]["tot_out"]
        row += f"{np.median(o[m]) - p_lvl:12.0f} "
    print(row)

print("\n（空载平台偏移占负载幅度 2999 ADC 的比例，负=显示偏低，正=显示偏高）")
for kind, a, b, p_lvl, m_lvl in plats:
    if kind != "idle":
        continue
    m = (el >= a) & (el <= b)
    row = f"  idle {a:7.2f}s  现场 {(m_lvl-p_lvl)/AMP*100:+7.1f}%  "
    for tag, *_ in variants:
        o = res[tag]["tot_out"]
        row += f"{tag}={(np.median(o[m])-p_lvl)/AMP*100:+7.1f}%  "
    print(row)

print("\n各变体事件序列（t0, kind）：")
for tag, *_ in variants:
    kl = res[tag]["kind_log"]
    print(f"  {tag:14s} n={len(kl):2d} " + ", ".join(f"{t:.1f}{k[:4]}" for t, k in kl[:24]) +
          (" ..." if len(kl) > 24 else ""))
