import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ad_lib as L
z = np.load(os.path.join("results", "rec_13ffca.npz"))
tu, Xu = z["tu"], z["Xu"]
Ys = {k[2:]: z[k] for k in z.files if k.startswith("Y_")}
dtm = tu[1]-tu[0]
tot = Xu.sum(axis=1)
sm = lambda x: pd.Series(x).rolling(int(0.5/dtm), center=True, min_periods=1).median().to_numpy()
r, v3, v4 = sm(tot), sm(Ys["v3"].sum(axis=1)), sm(Ys["v4_fast5"].sum(axis=1))
print("13ffca 录制 · 27.7s(+4672 漏检) 与 45.3s(-3627 漏检) 两处台阶后的显示轨迹")
print(f"{'t(s)':>7}{'原始':>9}{'v3显示':>9}{'v3扣除':>9}{'v4显示':>9}{'显示-v3偏差':>12}")
for w in (25,27,27.6,29,31,34,38,42,45.3,47,50,54,58,62,66,69):
    i = min(len(tu)-1, int(w/dtm))
    print(f"{w:7.1f}{r[i]:9.0f}{v3[i]:9.0f}{r[i]-v3[i]:9.0f}{v4[i]:9.0f}{v3[i]-r[i]:12.0f}")
