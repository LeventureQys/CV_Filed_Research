# -*- coding: utf-8 -*-
"""T3 步骤9：回落成因分解（x1 vs x2 vs 输入自身）。

对每个加载事件、每个臂，取沿后峰值帧 ip 到段末的区间，统计：
  fall_end = D[ip] − D[段末]（显示实际回落）
  dX1 = Σx1[段末] − Σx1[ip]，dX2 同理，dV = Σv[段末] − Σv[ip]
恒等式：fall_end ≈ dX1 + dX2 − dV
  dX1_share = dX1 / (dX1 + dX2)：快态在「新增扣除」里占的份额
输出 results/t3_decomp.csv + results/t3_decomp.txt（按事件类分层的中位数）。
"""
import csv
import os

import numpy as np

import t3_lib as T
import t3_replay as R

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

ARMS = [("base", None),
        ("A_now_a1.0", dict(mode="A", pred="now", alpha=1.0)),
        ("A_lag_a1.0", dict(mode="A", pred="lag", alpha=1.0)),
        ("C_a0.8_tb2.0_1.5s", dict(mode="C", pred="now", alpha=0.8,
                                   tau_boost=2.0, boost_s=1.5))]


def main():
    rows = []
    for tag, label, d in T.all_sessions():
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        din = V.sum(axis=1)
        rng = float(din.max() - din.min())
        edges, _ = T.find_edges(el, din)
        for name, spec in ARMS:
            ff = None
            if spec:
                ff = dict(R.DET_BASE)
                ff.update(spec)
            r = T.observe(ts, V, ff=ff)
            D = r["D"].sum(axis=1)
            X1 = r["X1"].sum(axis=1)
            X2 = r["X2"].sum(axis=1)
            for e in edges:
                i, j = e["i"], e["j"]
                if j - i < 30:
                    continue
                ys = R.smooth(D[i:j], 30)
                ip = int(np.argmax(ys))
                k = i + ip
                kend = j - 1
                rows.append(dict(
                    session=label, arm=name, cls=R.classify(e, rng), t=float(el[i]),
                    step=e["step"], hold=float(el[kend] - el[i]),
                    fall_end=float(D[k] - D[kend]),
                    dX1=float(X1[kend] - X1[k]),
                    dX2=float(X2[kend] - X2[k]),
                    dV=float(din[kend] - din[k]),
                    x1_share=(float((X1[kend] - X1[k]) /
                                    max(abs(X1[kend] - X1[k]) + abs(X2[kend] - X2[k]),
                                        1e-9)))))
        print("   %-46s 沿%3d" % (label[-46:], len(edges)), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "t3_decomp.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["T3 回落成因分解（沿后峰值帧 → 保压段末；总量 ADC）",
             "恒等式（逐事件成立）：fall_end = dX1 + dX2 − dV  ⇒  均值可加，中位不可加",
             "",
             "臂                  类           N | fall_end 均值 |  dX1 均值 |  dX2 均值 |  dV 均值 | x1 占新增扣除 中位/均值",
             ""]
    for name, spec in ARMS:
        for cls in ("全部", "首次大台阶", "受载态小台阶", "其他"):
            rr = [r for r in rows if r["arm"] == name and (cls == "全部" or r["cls"] == cls)]
            if not rr:
                continue
            shares = [r["x1_share"] for r in rr if abs(r["dX1"] + r["dX2"]) > 50]
            lines.append("%-20s %-10s %3d | %12.0f | %9.0f | %9.0f | %7.0f | %10.2f / %6.2f"
                         % (name, cls, len(rr),
                            float(np.mean([r["fall_end"] for r in rr])),
                            float(np.mean([r["dX1"] for r in rr])),
                            float(np.mean([r["dX2"] for r in rr])),
                            float(np.mean([r["dV"] for r in rr])),
                            float(np.median(shares)) if shares else float("nan"),
                            float(np.mean(shares)) if shares else float("nan")))
        lines.append("")
    with open(os.path.join(OUT, "t3_decomp.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
