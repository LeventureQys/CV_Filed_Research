import pandas as pd, numpy as np
d = pd.read_csv(r"results\v5_midload_events.csv")
d["j"] = d["jump"].abs()
for k in ("v3","v4","v5"):
    r = (d[f"gap_{k}"]/d["j"])
    print(f"{k}: 最坏欠报 均值={100*r.mean():.1f}%  最大={100*r.max():.1f}%  "
          f"事件数={len(r)}  |jump|中位={d['j'].median():.0f}")
print()
print(d[["rec","t","ratio","jump","gap_v3","gap_v4","gap_v5"]].round(0).to_string(index=False))
