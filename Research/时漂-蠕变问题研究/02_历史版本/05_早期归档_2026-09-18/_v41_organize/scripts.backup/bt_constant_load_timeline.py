# -*- coding: utf-8 -*-
"""「零基线 → 恒压压上 → 算法开始回调」全过程时间线实测。

用户问法：从零基线到实际激发得到一个恒压、直到算法开始回调，一共需要多少时间？怎么理解？

时间线定义（全部相对**真实加载沿** t=0）：
    ① 检测延迟      pending 置位（判据要看够数据）
    ② 确认          kStepPersistS = 2.5s → epoch 起点
    ③ 免责期        FAST_S = 1/3/5s（显示完全跟随原始，即"还没动手"）
    ④ 首扣          扣除量 > 0（= ③ 的终点，理论上 epoch+FAST）
    ⑤ 可见回调      扣除量超过"看得见"阈值（实测口径）
    ⑥ 停止上漂      显示总量的峰值时刻（扣除速度追上蠕变速度）
    ⑦ 拉回完成      显示回落到「加载沿读数 + 5%×幅度」以内
    ⑧ 扣到位 95%    首扣 + 3τ（τ=3s）

数据：恒载 9 组（真实恒压：右/左拇指指尖、四指指尖各 3 次，display 域）
      + 4 份实录的「空载→负载」沿（ADC 域）。
产出：results/constant_load_timeline.csv、results/_constant_load_timeline.log（调用方 tee）
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

KS = ["1s", "3s", "5s"]
FASTOF = {"1s": 1.0, "3s": 3.0, "5s": 5.0}
CONFIRM, TAU_G = 2.5, 3.0
CFG = [(k, dict(FAST_S=FASTOF[k], EXEMPT_AWIN=FASTOF[k] / 3.0, LEV_ARM_S=FASTOF[k])) for k in KS]

STATIC = [(loc, f"数据{i}") for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
B = os.path.join(TEMP, "变化负载")
RECS = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]


def run(kw, tu, Xu):
    c = GLM53v51(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def timeline(y_tot, tot_s, dtm, i0, amp, b_end, span_s=120.0):
    """返回"回调"全过程的关键时刻（全部相对加载沿 i0，秒）。

    口径（都比"上漂/压住"本身稳健，不依赖峰值这类噪声敏感量）：
      t_vis   已生效扣除 > 0.5%×幅度        → 看得见的回调开始
      t63/t90 扣除量达到"末段稳态值"的 63%/90% → 蠕变被压住的过程
      t_flat  显示在随后 20s 窗内起伏 < 2%×幅度 → 显示走平（不再上漂）
    """
    a = i0
    b = min(len(y_tot) - 1, i0 + int(span_s / dtm), b_end)
    ded = np.asarray(tot_s[:len(y_tot)]) - np.asarray(y_tot)
    dd = ded[a:b]
    hit = np.where(dd > 0.005 * amp)[0]
    t_vis = float(hit[0] * dtm) if len(hit) else np.nan
    tail = dd[int(0.75 * len(dd)):]
    final = float(np.median(tail)) if len(tail) else np.nan
    t63 = t90 = np.nan
    if np.isfinite(final) and final > 0:
        for frac, name in ((0.63, "63"), (0.90, "90")):
            h = np.where(dd[int(20 / dtm):] >= frac * final)[0]
            t = float((h[0] + int(20 / dtm)) * dtm) if len(h) else np.nan
            if name == "63":
                t63 = t
            else:
                t90 = t
    t_flat = np.nan
    w = int(20 / dtm)
    y = np.asarray(y_tot)[a:b]
    for i in range(0, max(1, len(y) - w), max(1, int(1 / dtm))):
        seg = y[i:i + w]
        if (seg.max() - seg.min()) < 0.02 * amp:
            t_flat = float(i * dtm)
            break
    return t_vis, t63, t90, t_flat


rows = []
print("=" * 132)
print("A. 恒载 9 组（真实恒压：零基线 → 压上恒压 → 算法回调）：全过程时刻（相对真实加载沿，s）")
print("=" * 132)
print(f"{'数据':>14} {'档位':>5} {'首扣理论':>8} {'可见回调':>8} {'扣到63%':>8} {'扣到90%':>8} "
      f"{'显示走平':>8} {'末段残余':>9}")
print("-" * 132)
for loc, name in STATIC:
    p = os.path.join(TEMP, loc, name, "device_001_seg000.csv")
    if not os.path.exists(p):
        continue
    d = L.prep(p)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
    thr = 0.15 * d["tot"].max()
    ld = d["tot"] > thr
    dd = np.diff(ld.astype(int))
    s = list(np.where(dd == 1)[0] + 1)
    e = list(np.where(dd == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    a0, b0 = max(zip(s, e), key=lambda z: z[1] - z[0])
    b0 = min(b0, len(tu) - 1)
    base = float(np.median(tot_s[max(0, a0 - int(2 / dtm)):a0])) if a0 > 0 else float(tot_s[0])
    i0 = next((i for i in range(a0, min(a0 + int(5 / dtm), b0))
               if tot_s[i] > base + 0.05 * (tot_s[b0] - base)), a0)
    amp = float(tot_s[b0] - base)
    tail = tot_s[max(i0, b0 - int((b0 - a0) / 10)):b0]
    for k, kw in CFG:
        Y = run(kw, tu, Xu)
        y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        tv, t63, t90, tfl = timeline(y, tot_s, dtm, i0, amp, b0)
        res = 100 * (tail.mean() - float(y[i0])) / max(amp, 1e-9)
        rows.append(dict(scope="static", ds=f"{loc}/{name}", algo=k, t_vis=tv, t63=t63, t90=t90,
                         t_flat=tfl, resid_tail=res, t_theo_first=CONFIRM + FASTOF[k],
                         t_theo95=CONFIRM + FASTOF[k] + 3 * TAU_G))
        print(f"{loc + '/' + name:>14} {k:>5} {CONFIRM + FASTOF[k]:8.1f} {tv:8.1f} {t63:8.1f} "
              f"{t90:8.1f} {tfl:8.1f} {res:8.1f}%")
    tv, t63, t90, tfl = timeline(tot_s, tot_s, dtm, i0, amp, b0)
    rows.append(dict(scope="static", ds=f"{loc}/{name}", algo="raw", t_vis=np.nan, t63=np.nan,
                     t90=np.nan, t_flat=tfl, resid_tail=100 * (tail.mean() - tot_s[i0]) / max(amp, 1e-9),
                     t_theo_first=np.nan, t_theo95=np.nan))
    print(f"{loc + '/' + name:>14} {'raw':>5} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {tfl:8.1f} "
          f"{100 * (tail.mean() - tot_s[i0]) / max(amp, 1e-9):8.1f}%")

print("\n" + "=" * 132)
print("B. 4 份实录的「空载→负载」沿（ADC 域）")
print("=" * 132)
print(f"{'录制':>20} {'加载沿':>8} {'档位':>5} {'首扣理论':>8} {'可见回调':>8} {'扣到63%':>8} "
      f"{'扣到90%':>8} {'显示走平':>8}")
print("-" * 132)
for tag, path in RECS:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
    ev = L.event_table(d, [x for x, _ in L.detect_events(d["tot"], dtm)], {"raw": Xu}, [], algos=[])
    ed = ev[(ev.jump > 2000) & (ev.pre < 0.05 * float(d["tot"].max()))]
    segs = L.find_periods(d["tot"], dtm)
    b_end = min(len(tu) - 1, max(b for _, b in segs)) if segs else len(tu) - 1
    for _, e in ed.iterrows():
        i0 = int(np.searchsorted(tu, e.t))
        amp = float(abs(tot_s[min(len(tu) - 1, i0 + int(10 / dtm))] - tot_s[i0]))
        for k, kw in CFG:
            Y = run(kw, tu, Xu)
            y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
            tv, t63, t90, tfl = timeline(y, tot_s, dtm, i0, amp, min(b_end, i0 + int(40 / dtm)))
            rows.append(dict(scope="rec", ds=tag, algo=k, t_vis=tv, t63=t63, t90=t90, t_flat=tfl,
                             resid_tail=np.nan, t_edge=float(e.t),
                             t_theo_first=CONFIRM + FASTOF[k], t_theo95=CONFIRM + FASTOF[k] + 3 * TAU_G))
            print(f"{tag:>20} {e.t:8.2f} {k:>5} {CONFIRM + FASTOF[k]:8.1f} {tv:8.1f} {t63:8.1f} "
                  f"{t90:8.1f} {tfl:8.1f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "constant_load_timeline.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 132)
print("汇总（均值）：各时刻相对真实加载沿的秒数")
print("=" * 132)
for scope, lbl in (("static", "恒载 9 组（display 域，真实恒压）"), ("rec", "实录空载→负载沿（ADC 域）")):
    sub = df[df.scope == scope]
    if not len(sub):
        continue
    print(f"\n[{lbl}]")
    print(sub.groupby("algo").agg(
        首扣理论=("t_theo_first", "mean"), 可见回调=("t_vis", "mean"),
        扣到63=("t63", "mean"), 扣到90=("t90", "mean"), 显示走平=("t_flat", "mean"),
        扣到位95=("t_theo95", "mean"), n=("t_vis", "count"))
        .reindex([k for k in KS] + ["raw"]).round(2).to_string())
