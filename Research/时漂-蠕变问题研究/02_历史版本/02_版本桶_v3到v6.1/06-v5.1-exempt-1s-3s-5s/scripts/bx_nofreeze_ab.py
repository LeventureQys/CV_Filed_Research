# -*- coding: utf-8 -*-
"""A/B：把 pending 期的「冻结」关掉，看它到底冻住了什么。

背景：`pending` 命中很快（实测 0.12 s），随后 2.5 s 确认窗内 `hold_=True`：
不积分 g/γ、输出保持 `hold_comp_`。本脚本用**源码打补丁**的方式造一个对照变体
（把 pending 分支里的 `self.hold = True` 改成 `False`），在同一段数据上对比：

    变体 A = 当前实现（pending 期冻结）
    变体 B = 不冻结（pending 期照常积分/扣除）

注意 B 除了不冻结，还会连带失去 D2 的前提（`hold_comp_` 不再被记录 ⇒ Restep 时
无法"扣掉旧载已累积的蠕变"），这正是"没有冻结"这一类实现的完整行为，故如实标注。

产出：results/nofreeze_ab.csv、控制台表（调用方 tee 到 results/_nofreeze_ab.log）
"""
import os
import re
import sys
import types
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

FAST, AWIN, ARM = 3.0, 1.0, 3.0

# ── 用源码补丁造「不冻结」变体（改一处：pending 分支的 hold） ──
src = open(os.path.join(HERE, "glm53_v51.py"), encoding="utf-8").read()
old = "        elif self.pending:\n            self.hold = True"
new = "        elif self.pending:\n            self.hold = False"
assert src.count(old) == 1, f"补丁锚点不唯一：{src.count(old)}"
mod = types.ModuleType("glm53_v51_nofreeze")
exec(compile(src.replace(old, new), "glm53_v51_nofreeze", "exec"), mod.__dict__)
NoFreeze = mod.GLM53v51
print("A/B 变体已构造：A = 当前实现（pending 冻结）；B = 不冻结（源码补丁 1 处）")


def run(cls, tu, Xu):
    c = cls(Xu.shape[1])
    c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = FAST, AWIN, ARM
    Y = np.empty_like(Xu)
    st = []
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
        st.append((int(c.in_load), int(c.pending), int(c.hold), float(c.g)))
    return Y, np.array(st)


B = os.path.join(TEMP, "变化负载")
RECS = [("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"))]

rows = []
for tag, path in RECS:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
    peak = float(d["tot"].max())
    ev = L.event_table(d, [e for e, _ in L.detect_events(d["tot"], dtm)], {"raw": Xu}, [], algos=[])
    ml = ev[(ev.jump > 2000) & (ev.pre > 0.30 * ev.pre.max())]
    YA, stA = run(GLM53v51, tu, Xu)
    YB, stB = run(NoFreeze, tu, Xu)
    ya = L.med_smooth(YA.sum(axis=1), 0.5 / dtm)
    yb = L.med_smooth(YB.sum(axis=1), 0.5 / dtm)
    print("=" * 118)
    print(f"[{tag}] 负载内加重事件 {len(ml)} 个（免责 {FAST} s）")
    print(f"{'事件':>8} {'台阶':>7} | {'A 检测':>7} {'A pending 期Δ扣除':>17} | "
          f"{'B pending 期Δ扣除':>17} | {'台阶后6s捕获 A/B':>16} | {'变载窗偏差 A/B':>16}")
    print("-" * 118)
    for _, e in ml.iterrows():
        i0 = int(np.searchsorted(tu, e.t))
        pen = np.where((stA[:, 1] > 0) & (stA[:, 0] == 1))[0]
        pen = pen[pen >= i0 - 2]
        if not len(pen):
            continue
        p0 = int(pen[0])
        p1 = p0
        while p1 + 1 < len(stA) and stA[p1 + 1, 1] > 0 and stA[p1 + 1, 0] == 1:
            p1 += 1                      # 该次 pending 的**连续**区间
        dedA = (Xu - YA).sum(axis=1)
        dedB = (Xu - YB).sum(axis=1)
        dA = float(dedA[p1] - dedA[p0])  # 同一时间窗内，两个变体的扣除量变化
        dB = float(dedB[p1] - dedB[p0])
        j6 = min(len(tu) - 1, i0 + int(6 / dtm))
        capA = (ya[j6] - ya[i0]) / (tot_s[j6] - tot_s[i0])
        capB = (yb[j6] - yb[i0]) / (tot_s[j6] - tot_s[i0])
        aa, bb = max(0, i0 - int(1 / dtm)), min(len(tu), i0 + int(12 / dtm))
        gapA = float(np.abs(ya[aa:bb] - tot_s[aa:bb]).max())
        gapB = float(np.abs(yb[aa:bb] - tot_s[aa:bb]).max())
        print(f"{e.t:8.2f} {e.jump:7.0f} | {tu[p0] - e.t:7.2f} {dA:17.0f} | {dB:17.0f} | "
              f"{capA:7.2f} /{capB:6.2f} | {gapA:7.0f} /{gapB:6.0f}")
        rows.append(dict(rec=tag, t_edge=float(e.t), jump=float(e.jump),
                         t_detect=float(tu[p0] - e.t), pend_dur=float(tu[p1] - tu[p0]),
                         ded_delta_frozen=dA, ded_delta_nofreeze=dB,
                         cap6_frozen=capA, cap6_nofreeze=capB,
                         gap_frozen=gapA, gap_nofreeze=gapB,
                         disp_at_edge_frozen=float(ya[i0]),
                         disp_after1s_frozen=float(ya[min(len(tu) - 1, i0 + int(1 / dtm))]),
                         disp_after1s_nofreeze=float(yb[min(len(tu) - 1, i0 + int(1 / dtm))]),
                         raw_after1s=float(tot_s[min(len(tu) - 1, i0 + int(1 / dtm))])))
    gA = float(np.abs(ya - tot_s).max())
    gB = float(np.abs(yb - tot_s).max())
    print(f"  全程最大偏差：A（冻结）{gA:.0f} ADC（占峰值 {100*gA/peak:.1f}%）  |  "
          f"B（不冻结）{gB:.0f} ADC（占峰值 {100*gB/peak:.1f}%）")
    print("-" * 118)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "nofreeze_ab.csv"), index=False, encoding="utf-8-sig")

print("\n汇总（负载内加重事件，均值）")
print(f"  pending 期时长           : {df.pend_dur.mean():.2f} s（≈ kStepPersistS = 2.5 s）")
print(f"  pending 期扣除量变化     : 冻结 {df.ded_delta_frozen.mean():+7.0f} ADC  "
      f"vs 不冻结 {df.ded_delta_nofreeze.mean():+7.0f} ADC")
print(f"  → 冻结挡掉的误扣量        : {df.ded_delta_nofreeze.mean() - df.ded_delta_frozen.mean():+,.0f} ADC"
      f"（占这些台阶均值 {df.jump.mean():,.0f} ADC 的 "
      f"{100*(df.ded_delta_nofreeze.mean()-df.ded_delta_frozen.mean())/df.jump.mean():.1f}%）")
print(f"  台阶后 6 s 捕获比        : 冻结 {df.cap6_frozen.mean():.2f}  vs 不冻结 {df.cap6_nofreeze.mean():.2f}")
print(f"  变载窗最大偏差           : 冻结 {df.gap_frozen.median():.0f} ADC  vs 不冻结 {df.gap_nofreeze.median():.0f} ADC")
