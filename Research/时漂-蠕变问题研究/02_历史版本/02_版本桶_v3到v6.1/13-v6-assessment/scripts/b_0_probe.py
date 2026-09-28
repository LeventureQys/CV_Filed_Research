# -*- coding: utf-8 -*-
"""b0：数据盘点 + 卸载沿库存（不跑算法，只读 CSV 与既有 npz 缓存）。

产出：results/b_probe_segments.csv（每份录制每个卸载沿的几何）
      results/_b0_probe.log（盘点日志）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_common as B                                          # noqa: E402

LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def main():
    B.log_reconfigure()
    rows = []
    p("=" * 112)
    p("b0 盘点：13 份录制（恒载 9 + 实录 4）；卸载沿用多尺度迟滞（4 档阈值取并集）定位")
    p("=" * 112)
    p(f"{'录制':<18}{'家族':<6}{'nch':>4}{'span/s':>8}{'dt/ms':>7}{'底':>10}{'峰':>10}"
      f"{'段数':>5}{'卸载沿':>7}{'全卸':>5}{'部分':>5}")
    for tag, path in B.ALL:
        d = B.load_csv(tag, path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        tot = Xu.sum(axis=1)
        segs, idle, peak, ts = B.load_segments(tot, dt)
        evs, idle2, peak2, ds = B.unload_events(d)
        nf = sum(1 for e in evs if e["ev_class"] == "full")
        for e in evs:
            rows.append(dict(rec=tag, kind=d["kind"], nch=Xu.shape[1], span=d["span"],
                             **{k: v for k, v in e.items() if k not in ("rec", "kind")}))
        p(f"{tag:<18}{d['kind']:<6}{Xu.shape[1]:>4}{d['span']:>8.1f}{dt * 1000:>7.2f}"
          f"{idle:>10.2f}{peak:>10.1f}{len(segs):>5}{len(evs):>7}{nf:>5}{len(evs) - nf:>5}")
        z = np.load(B.cache_path(tag), allow_pickle=True)
        dtu = float(z["tu"][1] - z["tu"][0])
        if abs(dtu - float(z["ds"][0]) * dt) > 1e-9:
            p(f"    !! 缓存栅格不匹配 {tag}: {dtu:.6f} vs {float(z['ds'][0]) * dt:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(B.RES, "b_probe_segments.csv"), index=False, encoding="utf-8-sig")
    p("")
    p("=" * 112)
    p(f"卸载沿库存：{len(df)} 处（full=回空载，partial=只卸一部分，eof 已剔除）")
    p("=" * 112)
    cols = ["rec", "t_dn", "ev_class", "hold_s", "pre", "plat", "post", "drop",
            "step_prox", "creep_acc", "creep_pct", "fall_t", "pre_src", "post_src"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        p(df[cols].round(2).to_string(index=False))
    p("")
    p("按家族 / 类型计数：")
    p(df.groupby(["kind", "ev_class"]).size().to_string())
    with open(os.path.join(B.RES, "_b0_probe.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print("\n-> results/b_probe_segments.csv, results/_b0_probe.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
