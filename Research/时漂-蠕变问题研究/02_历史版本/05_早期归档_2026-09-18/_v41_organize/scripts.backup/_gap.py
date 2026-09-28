import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
RES = os.path.join(os.getcwd(), "results")
d = np.load(os.path.join(RES, "new_switch_load.npz"))
tu, Xu = d["tu"], d["Xu"]; periods, events = d["periods"], d["events"]
Ys = {k[2:]: d[k] for k in d.files if k.startswith("Y_")}
tot = Xu.sum(axis=1); dtm = tu[1]-tu[0]
sm = lambda x: pd.Series(x).rolling(max(3,int(0.5/dtm)),center=True,min_periods=1).median().to_numpy()
tot_s = sm(tot)
for lab in ("v3","v4_fast5","v4r_fast5"):
    y = sm(Ys[lab].sum(axis=1)); g = y - tot_s
    k = int(np.argmax(np.abs(g)))
    print(f"{lab}: 全程 max|显示-原始| = {abs(g[k]):7.1f} ADC @ t={tu[k]:6.2f}s  (显示={y[k]:7.0f} 原始={tot_s[k]:7.0f})")
    for w in (60,120,200,245):
        i=int(w/dtm); print(f"        t={w:3d}s 显示={y[i]:7.0f} 原始={tot_s[i]:7.0f} 差={g[i]:+7.0f} "
                            f"({100*g[i]/max(1,tot_s[i]):+6.1f}%)")
