# -*- coding: utf-8 -*-
"""v2.0 阶段二 · 失效事件计数卡：逐帧量化三条候选各自有多少"可干预瞬间"。

在归档原型上埋计数器（不改原型文件），统计：
  A. 事件期内**真空载**（idle_now 为真）的连续时长 —— 现有卸载出口本该生效却因阈值/时序错过多久
  B. 现有卸载出口（`tau > UNLOAD_FAST and idle_now`）**实际触发次数**
  C. 慢相期内 idle_now 为真的帧数（ToIdle 机会）
  D. 长保压段内"显示相对原始的偏移"的分布（决定限幅是否真有活干）
  E. 事件期内 inc 回落到 inc_max 一半以下的时刻与 tau（判断"载荷中途减少"能否被现有机制看见）
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


class Counter(P.GLM53v6):
    def __init__(self, n):
        super().__init__(n)
        self.KAPPA_ONSET, self.KAPPA_RESTEP, self.HO_MIN = 1.05, 1.12, 3.5
        self.n_unload_exit = 0          # B
        self.idle_in_event_frames = 0   # A
        self.unload_would_fire = 0      # tau>UNLOAD_FAST 且 idle_now 且仍在事件内
        self.idle_in_slow_frames = 0    # C
        self.toidle_in_slow = 0
        self.drop_in_event = []         # E: (ts, tau, inc, inc_max)

    def process(self, ts, v):
        was_event = self.ev is not None
        was_slow = (self.state == "slow")
        ev_before = self.ev
        idle_before = None
        n_frame = super().process(ts, v)
        ev_after = self.ev
        if was_event and ev_after is not None and ev_before is ev_after:
            tau = ts - ev_after["t0"]
            inc = float(np.sum(v)) - ev_after["base"]
            inc_s = inc
            if inc_s < 0.5 * ev_after["inc_max"] and tau > self.REVOKE and ev_after["inc_max"] > 1.0:
                self.drop_in_event.append((float(ts), tau, inc_s, ev_after["inc_max"]))
        if was_event and ev_after is None:
            self.n_unload_exit += 1
        return n_frame


def main():
    d = L.load_dataset(L.DS_ZERO)
    el = d["pre"]["el"]
    V = d["pre"]["V"]
    n, ch = V.shape
    c = Counter(ch)
    out = np.empty((n, ch))
    idle_mask = np.zeros(n, dtype=bool)
    state_log = []
    for i in range(n):
        total = float(V[i].sum())
        # 复算本帧 idle_now（与算法同式，仅供统计）
        out[i] = c.process(float(el[i]), V[i])
        lvl = max(c.level_ref, 1e-6 * (1 + abs(c.max_tot)))
        idle = (c.ts_smooth < 0.10 * lvl) or (c.ts_smooth < 1.5 * c.min_ts + 1e-6)
        idle_mask[i] = idle
        state_log.append(c.state)

    state_arr = np.array(state_log)
    ev_frames = np.where(state_arr == "event")[0]
    slow_frames = np.where(state_arr == "slow")[0]
    print(f"帧数={n}")
    print(f"事件态帧数 = {len(ev_frames)}；其中 idle_now 为真 = "
          f"{int(idle_mask[ev_frames].sum())} 帧"
          f"（{100*idle_mask[ev_frames].mean() if len(ev_frames) else 0:.1f}%）")
    print(f"慢相态帧数 = {len(slow_frames)}；其中 idle_now 为真 = "
          f"{int(idle_mask[slow_frames].sum())} 帧"
          f"（{100*idle_mask[slow_frames].mean() if len(slow_frames) else 0:.1f}%）")
    print(f"空载态帧数 = {int((state_arr=='idle').sum())}；其中 idle_now 为真 = "
          f"{int(idle_mask[state_arr=='idle'].sum())} 帧")
    print(f"\nB 现有卸载出口实际触发（事件→无事件）次数 = {c.n_unload_exit}")

    # A：连续 idle 段（事件态内）
    runs = []
    cur = 0
    for i in ev_frames:
        if idle_mask[i]:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    runs = [r for r in runs if r >= 2]
    if runs:
        print(f"\nA 事件态内的主动 idle 连续段（≥2 帧）共 {len(runs)} 段，"
              f"时长中位 {np.median(runs)/100:.2f}s、最大 {max(runs)/100:.2f}s")
    else:
        print("\nA 事件态内没有出现 ≥2 帧的主动 idle 段")

    # E：事件内 inc 回落到一半以下的时刻
    print(f"\nE 事件期内 inc < 0.5·inc_max 的帧数 = {len(c.drop_in_event)}")
    if c.drop_in_event:
        print("   前 15 处（ts, tau, inc, inc_max）：")
        for t, tau, inc, mx in c.drop_in_event[:15]:
            print(f"     t={t:8.2f}  tau={tau:6.2f}  inc={inc:8.0f}  inc_max={mx:8.0f}"
                  f"  inc/inc_max={inc/mx:.2f}")

    # D：长保压段的偏移分布
    pre = V.sum(1)
    osum = out.sum(1)
    off = osum - pre
    segs = L.plateau_segments(pre, el, min_dur=5.0)
    print("\nD 各受载平台的偏移（显示−原始）分布：")
    print(f"{'段':>3}{'t0':>8}{'dur':>7}{'off 中位':>10}{'off p10':>9}{'off p90':>9}"
          f"{'off 最大':>9}{'|off|>0.5%读数 的帧%':>20}")
    for i, (kind, a, b) in enumerate(segs):
        if kind != "loaded" or b - a < 200:
            continue
        seg = off[a:b + 1]
        pr = pre[a:b + 1]
        frac = 100 * np.mean(np.abs(seg) > 0.005 * np.abs(pr))
        print(f"{i:>3}{el[a]:8.1f}{el[b]-el[a]:7.1f}{np.median(seg):10.0f}"
              f"{np.percentile(seg,10):9.0f}{np.percentile(seg,90):9.0f}"
              f"{np.max(np.abs(seg)):9.0f}{frac:19.1f}%")


if __name__ == "__main__":
    main()
