# -*- coding: utf-8 -*-
"""T7-B 步骤 0：录制盘点 + 冻结事件清单核对 + 包结构（±1 包基准）。

只读数据；输出：
  results/_t7b_00_recon.log        运行命令与全部 stdout
  results/t7b_recon.csv            13 份录制结构（帧数/唯一时间戳/包步长/重复率/域）
  results/t7b_frozen_events_t7b.csv  T4-A 冻结事件表（60 事件）按 key 挂上路径与域
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t7b_ad_lib import (RECS, DOMAIN, rec_path, load_rec, pkt_dt, dup_frac, to_grid,  # noqa: E402
                        n_packets, read_frozen_events, RESULTS, ensure_dirs)

ensure_dirs()
LOG = os.path.join(RESULTS, "_t7b_00_recon.log")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def main():
    print("CMD: python scripts/t7b_00_recon.py")
    print("=" * 78)
    rows = []
    for key, (rel, dom) in RECS.items():
        p = rec_path(key)
        if not os.path.exists(p):
            print("[MISS] %-4s %s" % (key, p))
            rows.append(dict(key=key, domain=dom, exists=False, path=p))
            continue
        t, X, meta = load_rec(p)
        dtp = pkt_dt(t)
        n_pkt = n_packets(t)
        rows.append(dict(key=key, domain=dom, exists=True, path=rel,
                         n_frames=len(t), n_pkt=n_pkt, pkt_dt=dtp,
                         frames_per_pkt=len(t) / max(n_pkt, 1),
                         dup_frac=dup_frac(t), n_ch=X.shape[1],
                         span_s=round(float(t[-1] - t[0]), 3),
                         z_mean=float(X.sum(1).mean()), z_max=float(X.sum(1).max())))
        print("%-4s %-7s frames=%-6d pkt=%-6d frames/pkt=%.2f pkt_dt=%.4f ch=%d span=%.1fs "
              "Zmax=%.0f" % (key, dom, len(t), n_pkt, len(t) / max(n_pkt, 1), dtp,
                             X.shape[1], t[-1] - t[0], X.sum(1).max()))
    rec = pd.DataFrame(rows)
    rec.to_csv(os.path.join(RESULTS, "t7b_recon.csv"), index=False, encoding="utf-8-sig")
    print("=" * 78)

    ev = read_frozen_events()
    print("T4-A 冻结事件表：n=%d，kind 分布：%s" % (len(ev), dict(ev["kind"].value_counts())))
    ev["domain"] = ev["key"].map(DOMAIN)
    ev["ratio_meas"] = (ev["jump"] / ev["pre"].replace(0, np.nan)).abs()
    # 实测减重比例（|jump| / pre，pre>0 时）
    ev["drop_frac_meas"] = np.where(ev["pre"] > 0, (ev["jump"].abs() / ev["pre"]), np.nan)
    for k in ("unload", "partial_unload"):
        sub = ev[ev["kind"] == k]
        print("\n--- kind=%s (n=%d) ---" % (k, len(sub)))
        cols = ["key", "t_on", "pre", "post", "jump", "drop_frac_meas", "t50", "t90",
                "step_frame_frac", "top3_frame_frac", "z_at_005", "z_at_02", "clean"]
        print(sub[cols].to_string(index=False))
    # 减重比例分档（用 20%/80% 判据重看 T4-A 的 partial_unload 标签）
    sub = ev[ev["kind"] == "partial_unload"].copy()
    sub["band"] = pd.cut(sub["drop_frac_meas"], [0, 0.05, 0.20, 0.80, 1.01],
                         labels=["<5%", "5-20%", "20-80%", ">=80%"])
    print("\npartial_unload 按实测减重比例分档：\n%s" % sub["band"].value_counts().to_string())
    ev.to_csv(os.path.join(RESULTS, "t7b_frozen_events_t7b.csv"), index=False,
              encoding="utf-8-sig")
    print("\nWROTE results/t7b_recon.csv, results/t7b_frozen_events_t7b.csv")


if __name__ == "__main__":
    tee = Tee(LOG)
    _o, _e = sys.stdout, sys.stderr
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = _o
        tee.flush()
        tee.f.close()
