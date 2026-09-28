# -*- coding: utf-8 -*-
"""T1-B / 04b：噪声引起的**显示偏置**溯源（诊断脚本，支撑 §3 的"稳定但偏了"结论）。

对同一事件、同一臂：对比无扰动与加噪（白噪 r=2% 电平）的
  ① 交接时刻 Â 与 ΣA、② 慢相显示与无扰动基线的偏差、③ epoch 序列。
产物：results/t1b_noise_bias_probe.csv、results/_t1b_04b_bias.log
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

import t1b_lib as L                        # noqa: E402
from t1_common import make_perturb, med_smooth   # noqa: E402


def main():
    logp = os.path.join(RES, "_t1b_04b_bias.log")
    f = open(logp, "a", encoding="utf-8")

    def w(s):
        print(s)
        f.write(s + "\n")

    w("=== T1-B / 04b 噪声引起的显示偏置溯源 ===")
    ev = L.event_table()
    rows = []
    for _, e in ev[ev.dom == "显示域"].iterrows():
        key, t_on, Lh = e["key"], float(e["t_on"]), float(e["L_hold"])
        d = L.get_grid(key)
        tu, Xu = L.window_of(d, dict(kind="hold", t_on=t_on, span=80.0))
        rc = L.run_arm("v6", tu, Xu)
        Zc = rc["Z"]
        for r in (0.005, 0.02, 0.05):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed * 31 + 7)
                Xp = Xu + make_perturb(Xu, r * Lh, "white", rng)
                rn = L.run_arm("v6", tu, Xp)
                Zn = rn["Z"]
                i0 = int(round(t_on / 0.01))
                a = i0 + int(10 / 0.01)
                b = min(len(Zn), i0 + int(70 / 0.01))
                db = float(np.median(Zn[a:b]) - np.median(Zc[a:b]))
                # 交接时刻的 Â 与 ΣA
                hc = [h for h in rc["handoff"] if h[0] >= t_on - 2]
                hn = [h for h in rn["handoff"] if h[0] >= t_on - 2]
                rows.append(dict(ev=e["ev"], key=key, r=r, seed=seed, L_hold=Lh,
                                 bias_display=db, bias_pct=100 * db / Lh,
                                 Ahat_clean=(hc[0][2] if hc else np.nan),
                                 Ahat_noisy=(hn[0][2] if hn else np.nan),
                                 Asum_clean=(hc[0][4] if hc else np.nan),
                                 Asum_noisy=(hn[0][4] if hn else np.nan),
                                 n_ho_clean=len(hc), n_ho_noisy=len(hn),
                                 n_ep_clean=len(rc["epoch"]), n_ep_noisy=len(rn["epoch"])))
    df = pd.DataFrame(rows)
    df["dAhat"] = df.Ahat_noisy - df.Ahat_clean
    df["dAsum"] = df.Asum_noisy - df.Asum_clean
    df["dAhat_pct"] = 100 * df.dAhat / df.Ahat_clean
    df.to_csv(os.path.join(RES, "t1b_noise_bias_probe.csv"), index=False,
              encoding="utf-8-sig")
    for r, g in df.groupby("r"):
        w(f"r={r:6.3f}: 显示偏置中位 {g.bias_display.median():+7.3f} "
          f"({g.bias_pct.median():+6.2f}%L)；Â 变化中位 {g.dAhat.median():+8.2f} "
          f"({g.dAhat_pct.median():+6.2f}%)；ΣA 变化中位 {g.dAsum.median():+9.2f}；"
          f"epoch {g.n_ep_clean.median():.1f}->{g.n_ep_noisy.median():.1f}")
    w("说明：显示偏置 = 加噪运行的 [t_on+10, t_on+70] 显示中位 − 无扰动同段中位；"
      "Â/ΣA 取事件后首次交接的记录 —— 用于判定偏置来自'形状反演'还是'状态机'。")
    f.close()


if __name__ == "__main__":
    main()
