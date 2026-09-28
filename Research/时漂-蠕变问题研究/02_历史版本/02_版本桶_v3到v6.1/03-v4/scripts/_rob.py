import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
d = np.load(os.path.join(RES, "new_switch_load.npz"))
tu, Xu = d["tu"], d["Xu"]; periods, events = d["periods"], d["events"]
Ys = {k[2:]: d[k] for k in d.files if k.startswith("Y_")}
tot = Xu.sum(axis=1); dtm = tu[1]-tu[0]
tot_s = pd.Series(tot).rolling(int(0.5/dtm), center=True, min_periods=1).median().to_numpy()
cuts = sorted({round(tu[a],3) for a,_ in periods} | {round(tu[b],3) for _,b in periods} |
              {round(float(tu[e]),3) for e in events})
order = ["raw","v3","v4_fast5","v4r_fast5"]
for guard, minw in ((8.0,6.0),(12.0,6.0),(20.0,6.0)):
    rows=[]
    for i in range(len(cuts)-1):
        w0 = int(np.searchsorted(tu, cuts[i]+guard)); w1 = int(np.searchsorted(tu, cuts[i+1]))
        if (w1-w0)*dtm < minw: continue
        lvl = float(np.median(tot_s[w0:w1])); nL = w1-w0
        for k in order:
            L = Ys[k][w0:w1].sum(axis=1)
            sd = L[-max(1,nL//5):].mean() - L[:max(1,nL//5)].mean()
            rows.append((cuts[i], cuts[i+1], (w1-w0)*dtm, k, 100*sd/lvl))
    df = pd.DataFrame(rows, columns=["s","e","dur","algo","drift"])
    g = df.groupby("algo")["drift"].apply(lambda s: s.abs().mean()).round(2)
    gm = df.groupby("algo")["drift"].apply(lambda s: s.abs().max()).round(2)
    print(f"guard={guard}s 窗数={df[['s','e']].drop_duplicates().shape[0]}  均值: " +
          "  ".join(f"{k}={g[k]}" for k in order) + " | 最大: " + "  ".join(f"{k}={gm[k]}" for k in order))

# v4-v3 各分歧段最大差 + 该段电平
err = np.abs(Ys["v4_fast5"]-Ys["v3"]).max(axis=1)
idx = np.where(err>1e-6)[0]
segs=[]; a=idx[0]
for k in range(1,len(idx)):
    if idx[k]!=idx[k-1]+1: segs.append((a,idx[k-1])); a=idx[k]
segs.append((a,idx[-1]))
for a,b in segs:
    print(f"  分歧段 [{tu[a]:7.2f},{tu[b]:7.2f}]  最大|Δ|={err[a:b+1].max():7.1f} ADC  "
          f"该段电平中位={np.median(tot_s[a:b+1]):7.0f}  占比={100*err[a:b+1].max()/max(1,np.median(tot_s[a:b+1])):.2f}%")

