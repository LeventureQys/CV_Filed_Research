# -*- coding: utf-8 -*-
"""t4b_crosscheck_t4a.py —— T4-B ↔ T4-A 独立复核（只读 T4-A 产物，不重算它的形态表）。

T4-A（同任务 A 席位）已交付 `results/t4a_morphology.csv`（61 行 × 51 列）与
`t4a_separability.csv`。本脚本做三件事，全部**只读**：

① **事件对表**：把 T4-A 的 40 个加载类事件与本任务的 35 个加载类事件按 (rec, t_on±0.6 s) 配对，
   报告匹配率与 Δt 分布（两个独立检测器的一致性）。
② **判据交叉核对**：在同一批配对事件上，比较
   - T4-A 的 `pre_over_peak`（它的 `armed` 定义量）与本任务 `pre_frac`（P1）→ 秩相关 + 门限一致率；
   - T4-A 的 `z_at_02`（主通道/总量口径）与本任务 `sh_020b`（归一化形状）→ 秩相关；
   如果两者一致，说明"最简判据"这个结论在**两套独立实现**下都成立。
③ **告警列**：把 T4-A 的 `clean`、`T_ramp`（非因果）、`T_slope` 与我们的 P1 判据结果并排放，
   列出门限附近的 8 个事件（T4-A 报告 §3.4(e) 的 C/D 类），给 T4-Q8 的迟滞设计做依据。

产物：results/t4b_t4a_crosscheck.csv、results/t4b_t4a_agreement.csv
运行：python scripts/t4b_crosscheck_t4a.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C  # noqa: E402

T4A = os.path.join(C.TASK, "results", "t4a_morphology.csv")
T4A_IR = os.path.join(C.TASK, "results", "t4a_input_recover.csv")


def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan, np.nan
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p)


def main():
    if not os.path.isfile(T4A):
        print("!! 找不到 T4-A 产物：", T4A)
        return 1
    a = pd.read_csv(T4A)
    a = a[a.kind.isin(["onset", "restep"])].copy()
    mine, _ = C.build_events(verbose=False)
    m = mine[mine.kind.isin(["onset", "restep"])].copy()

    # ── ① 事件对表 ──
    pairs = []
    for _, r in m.iterrows():
        c = a[(a.rec == r.rec) & (np.abs(a.t_on - r.t_on) <= 0.6)]
        if not len(c):
            continue
        j = c.iloc[(c.t_on - r.t_on).abs().argmin()]
        pairs.append(dict(rec=r.rec, t_on_t4b=round(r.t_on, 3), t_on_t4a=round(j.t_on, 3),
                          dt=round(j.t_on - r.t_on, 3),
                          kind_t4b=r.kind, kind_t4a=j.kind,
                          pre_frac_t4b=round(r.pre_frac, 4), pre_over_peak_t4a=round(j.pre_over_peak, 4),
                          sh020_t4b=round(r.sh_020b, 4) if r.sh_020b == r.sh_020b else np.nan,
                          z_at_02_t4a=round(j.z_at_02, 4),
                          t4a_clean=bool(j.clean), t4a_armed=bool(j.armed),
                          label_confident=bool(r.label_confident)))
    p = pd.DataFrame(pairs)
    p.to_csv(os.path.join(C.RES, "t4b_t4a_crosscheck.csv"), index=False, encoding="utf-8-sig")
    print("== ① 事件对表：本任务 %d 个加载类事件 → 匹配到 T4-A %d 个（±0.6 s）=="
          % (len(m), len(p)))
    print("   |Δt| 中位 %.3f s   p90 %.3f s   最大 %.3f s" %
          (p.dt.abs().median(), p.dt.abs().quantile(.9), p.dt.abs().max()))
    print("   类别一致率 %.3f" % float((p.kind_t4b == p.kind_t4a).mean()))
    dis = p[p.kind_t4b != p.kind_t4a]
    if len(dis):
        print("   类别不一致的事件（标签门限差异，不是检测器差异）：")
        print(dis[["rec", "t_on_t4b", "kind_t4b", "kind_t4a", "pre_frac_t4b",
                   "t4a_armed"]].to_string(index=False))

    # ── ② 判据交叉核对 ──
    rows = []
    rho, pv = spearman(p.pre_frac_t4b, p.pre_over_peak_t4a)
    rows.append(dict(check="P1: pre_frac(T4B) vs pre_over_peak(T4A)", n=len(p),
                     statistic=round(rho, 4), p=round(pv, 5),
                     note="两套独立实现的同一物理量（pre 电平 ÷ 记录峰值）"))
    lab_b = p.pre_frac_t4b <= 0.40
    lab_a = p.t4a_armed
    rows.append(dict(check="P1 θ=0.40 与 T4-A armed(20%) 的标签一致率", n=len(p),
                     statistic=round(float((lab_b == (~lab_a)).mean()), 4), p=np.nan,
                     note="一致率越高 ⇒ 判据对实现细节越不敏感"))
    rho, pv = spearman(p.sh020_t4b, p.z_at_02_t4a)
    rows.append(dict(check="shape: sh_020(T4B) vs z_at_02(T4A)", n=len(p),
                     statistic=round(rho, 4), p=round(pv, 5),
                     note="归一化形状的同一物理量（0.2 s 完成度）"))
    if os.path.isfile(T4A_IR):
        ir = pd.read_csv(T4A_IR)
        kcol = "key" if "key" in ir.columns else ir.columns[0]
        q = p.merge(a[["rec", "t_on", "key"]], left_on=["rec", "t_on_t4a"],
                    right_on=["rec", "t_on"], how="left")
        q = q.merge(ir[[kcol, "t_on", "T_ramp"]], on=["key", "t_on"], how="left",
                    suffixes=("", "_ir"))
        rho, pv = spearman(q.pre_over_peak_t4a, q.T_ramp)
        rows.append(dict(check="T4-A 内部：pre_over_peak vs T_ramp", n=int(q.T_ramp.notna().sum()),
                         statistic=round(rho, 4) if rho == rho else np.nan, p=round(pv, 5) if pv == pv else np.nan,
                         note="T4-A 报告 §3.4(c)：预载只是输入斜坡时长的代理（R²=0.467）"))
        rho, pv = spearman(q.T_ramp, q.sh020_t4b)
        rows.append(dict(check="T_ramp(T4A) vs sh_020(T4B)", n=int(q.T_ramp.notna().sum()),
                         statistic=round(rho, 4) if rho == rho else np.nan, p=round(pv, 5) if pv == pv else np.nan,
                         note="非因果驱动量 vs 因果形状量"))
        q.to_csv(os.path.join(C.RES, "t4b_t4a_merge.csv"), index=False, encoding="utf-8-sig")
    ag = pd.DataFrame(rows)
    ag.to_csv(os.path.join(C.RES, "t4b_t4a_agreement.csv"), index=False, encoding="utf-8-sig")
    print("\n== ② 判据交叉核对 ==")
    print(ag.to_string(index=False))

    # ── ③ 门限附近事件（迟滞设计依据）──
    near = p[(p.pre_frac_t4b - 0.35).abs() <= 0.15].sort_values("pre_frac_t4b")
    print("\n== ③ 建议门限 θ=0.35 附近（±0.15）的事件：迟滞设计必须覆盖这些 ==")
    print(near[["rec", "t_on_t4b", "kind_t4b", "kind_t4a", "pre_frac_t4b",
                "t4a_clean"]].to_string(index=False))
    print("   共 %d / %d 个事件落在门限 ±0.15 内" % (len(near), len(p)))
    print("\n-> results/t4b_t4a_crosscheck.csv / t4b_t4a_agreement.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
