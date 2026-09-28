# -*- coding: utf-8 -*-
"""探针 F：残漂的**因果分解**（录制口径）。

把 [60,390] s 内显示漂移  ΔΣmain = ΔΣcreep − ΔΣded  按三个机制拆开：
  M1 未受载通道被排除（loaded_ 门 A_k > 0.10·maxA）—— 它们的 creep 结构性不被扣
  M2 受载通道的「共识标量 × 固定 A」比例模型扣不准（rank-1 限制 / γ 无遗忘）
  M3 其它

并给出「把 M1 关掉（门限降到 2%）」与「M2 用 forgetting γ」各自的上限收益。

输出：results/v30_causal.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

TAG = "20260919_141824_single_device_110871"
DS = os.path.join(L.OVERVIEW, TAG)
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
OUT = os.path.join(RES, "v30_causal.txt")

W0, W1 = 60.0, 390.0


def main():
    ds = L.load_dataset(DS)
    el = ds["pre"]["el"]
    pre, main = ds["pre"]["V"], ds["main"]["V"]
    ref = pre[int(20 * 100):int(40 * 100)].mean(0)
    idle = pre[:430].mean(0)
    amp = ref - idle
    m = (el >= W0) & (el <= W1)
    i0 = np.nonzero(m)[0][0]
    i1 = np.nonzero(m)[0][-1]

    lines = []
    p = lines.append
    p("窗口 [%.0f,%.0f] s（i=%d..%d）" % (W0, W1, i0, i1))
    p("")

    def inc(x):
        return x[i1] - x[i0]

    creep = pre - ref
    ded = pre - main
    resid = main - ref
    tot_c, tot_d, tot_r = inc(creep.sum(1)), inc(ded.sum(1)), inc(resid.sum(1))
    p("总量增量：Δcreep %+.0f  Δded %+.0f  Δ显示漂移 %+.0f（= Δcreep − Δded）"
      % (tot_c, tot_d, tot_r))
    p("")

    for frac in (0.10, 0.05, 0.02, 0.01):
        loaded = amp > frac * amp.max()
        dc_l = inc(creep[:, loaded].sum(1))
        dd_l = inc(ded[:, loaded].sum(1))
        dc_u = inc(creep[:, ~loaded].sum(1))
        dd_u = inc(ded[:, ~loaded].sum(1))
        p("门限 %.0f%%·maxA（受载 %2d 个通道）" % (100 * frac, int(loaded.sum())))
        p("   受载通道  : Δcreep %+7.0f  Δded %+7.0f  → 通过 %+7.0f" %
          (dc_l, dd_l, dc_l - dd_l))
        p("   未受载通道: Δcreep %+7.0f  Δded %+7.0f  → 通过 %+7.0f" %
          (dc_u, dd_u, dc_u - dd_u))
        p("   合计通过 %+7.0f" % (tot_c - dd_l - dd_u))
        p("")

    # ── 机制 M2：受载通道内的扣不准 ──
    loaded = amp > 0.10 * amp.max()
    p("=== M2 细看（现行 10%% 门）===")
    p("%-5s %10s %10s %12s %10s" % ("ch", "Δcreep", "Δded", "末值creep", "末值ded"))
    for k in np.nonzero(loaded)[0]:
        p("  ch%-3d %10.0f %10.0f %12.0f %10.0f" %
          (k, inc(creep[:, k]), inc(ded[:, k]), creep[i1, k], ded[i1, k]))
    p("")

    # ── 上限对照 ──
    p("=== 上限对照（全部因果、只用 t≤now 的数据）===")
    dt = np.clip(np.diff(el, prepend=el[0]), 0.0, 0.1)

    def ema(x, tau):
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = y[i - 1] + (dt[i] / tau) * (x[i] - y[i - 1])
        return y

    def rep(name, ded_mat):
        r = (creep - ded_mat).sum(1)
        dr = r[i1] - r[i0]
        rms = float(np.sqrt((r[m] ** 2).mean()))
        p("  %-34s Δ显示漂移 %+7.0f  残差RMS %6.0f  末值 %+7.0f" %
          (name, dr, rms, r[i1]))

    rep("不补偿", np.zeros_like(creep))
    rep("实况（现役算法）", ded)
    p("  --- 候选：逐通道瞬时跟踪 ded_k = EMA_τ(v_k − A_k) ---")
    for tau in (3.0, 5.0, 10.0, 20.0, 40.0):
        rep("  τ=%4.0f s，全部通道" % tau,
            np.stack([ema(creep[:, k], tau) for k in range(pre.shape[1])], 1))
    for tau in (5.0, 10.0, 20.0):
        dd = np.zeros_like(creep)
        for k in np.nonzero(loaded)[0]:
            dd[:, k] = ema(creep[:, k], tau)
        rep("  τ=%4.0f s，仅现行受载通道" % tau, dd)
    p("")
    p("=== A 锚点偏差（算法 A vs 真无蠕变电平 ref）===")
    a_hat_proxy = None
    p("  ref 总量 %.0f" % ref.sum())
    # 由录制口径反推算法 A 的等效值：ded_i 在蠕变≈0 时应为 0 ⇒ A ≈ v − ded at early hold
    m_early = (el >= 42) & (el <= 60)
    ded_e = ded[m_early].mean(0)
    pre_e = pre[m_early].mean(0)
    A_proxy = pre_e - ded_e
    p("  由 (pre − ded) 在 [42,60] s 反推：算法等效 A 总量 %.0f（比 ref %+.1f%%）" %
      (A_proxy.sum(), 100 * (A_proxy.sum() / ref.sum() - 1)))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
