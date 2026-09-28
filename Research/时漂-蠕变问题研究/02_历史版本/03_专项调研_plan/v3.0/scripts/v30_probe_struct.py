# -*- coding: utf-8 -*-
"""探针 E：稳定工况残漂的**结构化归因**（全部基于录制口径 pre/main，权威）。

回答四个问题：
  Q1 慢相扣除 ded_i(t) 在长保压内是不是「冻结」的？（冻结 ⇒ 增量蠕变全部进显示）
  Q2 蠕变的空间形状是否**时不变**（rank-1：creep_i(t) = c_i·h(t)）？
     —— 时不变 ⇒ 比例模型 γ_i·A_i·g 原则上能扣干净；时变 ⇒ 比例模型结构性扣不掉。
  Q3 实际扣除 ded_i(t) 跟的是「当前蠕变」还是「本 epoch 蠕变的时均值」？
     —— 这是 γ_i 无遗忘累积的最小二乘的必然结果。
  Q4 上限对照：若把 γ 换成**瞬时**（逐通道 EMA 跟踪当前残差），残余能降到多少？

输出：results/v30_structured.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

TAG = "20260919_141824_single_device_110871"
DS = os.path.join(L.OVERVIEW, TAG)
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
OUT = os.path.join(RES, "v30_structured.txt")


def ema(x, dt, tau):
    y = np.empty_like(x)
    a = 0.0
    y[0] = x[0]
    for i in range(1, len(x)):
        w = min(dt[i], 0.1) / tau
        a = y[i - 1] + w * (x[i] - y[i - 1])
        y[i] = a
    return y


def main():
    ds = L.load_dataset(DS)
    el = ds["pre"]["el"]
    pre = ds["pre"]["V"]
    main = ds["main"]["V"]
    dt = np.diff(el, prepend=el[0])
    dt = np.clip(dt, 0.0, 0.1)
    ref = pre[int(20 * 100):int(40 * 100)].mean(0)
    idle = pre[:430].mean(0)
    amp = ref - idle
    loaded = amp > 0.10 * amp.max()
    creep = pre - ref                 # 真蠕变（相对无蠕变电平）
    ded = pre - main                  # 算法实际扣除
    resid = main - ref                # 显示残漂

    lines = []
    p = lines.append
    p("数据集 %s（权威口径：pre=算法输入, main=算法结果）" % TAG)
    p("ref = pre 在 [20,40] s 的均值（无蠕变电平代理）；受载通道 %d 个" % int(loaded.sum()))
    p("")

    # ── Q1：扣除是否冻结 ──
    p("=== Q1：慢相扣除在长保压内是否「冻结」 ===")
    t0, t1 = int(60 * 100), len(el) - 1
    seg = (el >= 60) & (el <= 390)
    tc, td = creep.sum(1), ded.sum(1)
    p("  [60,390] s：creep 增长 %+.0f ADC（%.2f ADC/s）" %
      (tc[seg][-1] - tc[seg][0], (tc[seg][-1] - tc[seg][0]) / 330.0))
    p("              ded   增长 %+.0f ADC（%.2f ADC/s）" %
      (td[seg][-1] - td[seg][0], (td[seg][-1] - td[seg][0]) / 330.0))
    p("              ⇒ 增量蠕变被扣掉的比例 = %.1f%%（0%% = 完全没扣）" %
      (100 * (td[seg][-1] - td[seg][0]) / max(tc[seg][-1] - tc[seg][0], 1e-9)))
    # 分段：扣掉比例逐 40 s
    p("  逐 40 s 窗口的「增量扣除率」：")
    for a in range(60, 390, 40):
        m = (el >= a) & (el <= a + 40)
        dc = tc[m][-1] - tc[m][0]
        dd = td[m][-1] - td[m][0]
        p("    [%3d,%3d] s  Δcreep %+7.0f  Δded %+7.0f  扣除率 %6.1f%%" %
          (a, a + 40, dc, dd, 100 * dd / dc if abs(dc) > 1 else float("nan")))
    p("")

    # ── Q2：rank-1（时不变形状）检验 ──
    p("=== Q2：蠕变空间形状是否时不变（比例模型的前提） ===")
    ks = [k for k in np.nonzero(loaded)[0]]
    marks = [60, 120, 180, 240, 300, 360]
    prof = []
    for a in marks:
        m = (el >= a) & (el <= a + 20)
        prof.append(creep[m].mean(0))
    prof = np.array(prof)                       # [mark, ch]
    norm = prof / prof[-1][None, :]             # 每个通道归一到末值
    p("  归一化蠕变形状（各通道 / 该通道末值）：")
    p("  %6s %s" % ("t", " ".join("ch%-5d" % k for k in ks)))
    for i, a in enumerate(marks):
        p("  %6d %s" % (a, " ".join("%7.3f" % norm[i, k] for k in ks)))
    spread = norm.max(0) - norm.min(0)
    p("  各通道形状极差：中位 %.3f  最大 %.3f（0 = 完美 rank-1）" %
      (float(np.median(spread[ks])), float(spread[ks].max())))
    # rank-1 近似的最优残差（SVD 第一主成分能解释多少）
    M = creep[:].T                              # [ch, frame]
    M = M - M.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(M[:, ::50], full_matrices=False)
    p("  蠕变矩阵 SVD 能量占比：PC1 %.1f%%  PC2 %.1f%%  PC3 %.1f%%" %
      (100 * S[0] ** 2 / (S ** 2).sum(), 100 * S[1] ** 2 / (S ** 2).sum(),
       100 * S[2] ** 2 / (S ** 2).sum()))
    p("")

    # ── Q3：扣除跟的是当前蠕变还是时均蠕变 ──
    p("=== Q3：实际扣除 ded_i(t) 跟的是「当前」还是「epoch 时均」蠕变 ===")
    p("  %-5s %10s %10s %10s %10s %10s" %
      ("ch", "creep(360s)", "ded(360s)", "时均creep", "ded/当前", "ded/时均"))
    m360 = (el >= 350) & (el <= 380)
    mmean = (el >= 60) & (el <= 380)
    for k in ks:
        cu = creep[m360, k].mean()
        d = ded[m360, k].mean()
        cm = creep[mmean, k].mean()
        p("  ch%-3d %10.0f %10.0f %10.0f %9.2f %10.2f" %
          (k, cu, d, cm, d / cu if abs(cu) > 1 else float("nan"),
           d / cm if abs(cm) > 1 else float("nan")))
    p("  （ded/时均 ≈ 1 而 ded/当前 < 1 ⇒ 扣除锁在 epoch 时均值上，即 γ 无遗忘累积的必然结果）")
    p("")

    # ── Q4：上限对照 ──
    p("=== Q4：若把「共识标量 g × 固定 A」换成「逐通道瞬时跟踪」，残余能降到多少 ===")
    p("  （因果、可用实现：ded_i(t) = EMA_τ( pre_i(t) − ref_i )，ref/无蠕变电平在交接时刻已可得）")
    for tau in (5.0, 15.0, 30.0, 60.0):
        ded_ct = np.stack([ema(creep[:, k], dt, tau) for k in range(pre.shape[1])], 1)
        r = (creep - ded_ct).sum(1)
        mm = (el >= 40)
        p("    τ=%5.0f s  显示残漂(相对 ref) 末段 %+7.0f  |  漂移量(60→390s) %+7.0f  |  残差 RMS %7.0f" %
          (tau, r[-500:].mean(), r[seg][-1] - r[seg][0],
           float(np.sqrt((r[mm] ** 2).mean()))))
    p("  实况（现役算法）:        显示残漂(相对 ref) 末段 %+7.0f  |  漂移量(60→390s) %+7.0f  |  残差 RMS %7.0f" %
      (resid[-500:].sum(1).mean(), resid.sum(1)[seg][-1] - resid.sum(1)[seg][0],
       float(np.sqrt((resid[mm].sum(1) ** 2).mean()))))
    p("  不补偿（原样显示）:      显示残漂(相对 ref) 末段 %+7.0f  |  漂移量(60→390s) %+7.0f  |  残差 RMS %7.0f" %
      (creep[-500:].sum(1).mean(), creep.sum(1)[seg][-1] - creep.sum(1)[seg][0],
       float(np.sqrt((creep[mm].sum(1) ** 2).mean()))))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
