import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from glm53_v3 import GLM53v3
from ad_v4 import GLM53v4
REC = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
       r"\零负载-中途切换负载-零负载-切换负载\device_001_seg000.csv")
df = pd.read_csv(REC, skiprows=24); ch = [c for c in df.columns if c.startswith("ch")]
t = df["timestamp"].to_numpy(float); t = t-t[0]; X = df[ch].to_numpy(float)
span=t[-1]-t[0]; dt=span/(len(t)-1); tu=np.arange(0.0,span,dt)
Xu=np.vstack([np.interp(tu,t,X[:,c]) for c in range(X.shape[1])]).T
sm=lambda x: pd.Series(x).rolling(int(0.5/dt),center=True,min_periods=1).median().to_numpy()
tot_s=sm(Xu.sum(axis=1))
out={}
for tag,cls in (("v3",GLM53v3),("v4",GLM53v4)):
    c=cls(Xu.shape[1]); c.FAST_S,c.A_W0_V4,c.A_W1_V4=5.0,3.5,5.0
    Y=np.empty_like(Xu)
    for i in range(len(tu)): Y[i]=c.process(tu[i],Xu[i])
    out[tag]=sm(Y.sum(axis=1))
    c2=cls(Xu.shape[1])
print("数据B 中途变载 @20.95s：显示相对原始总量的偏差（ADC / 占本次跳变 7067）")
for w in (21.5,22.5,23.5,24.5,26.0,28.0):
    i=int(w/dt)
    print(f"  t={w:5.1f}s  原始={tot_s[i]:7.0f}  v3偏差={out['v3'][i]-tot_s[i]:+8.0f}"
          f"({100*(out['v3'][i]-tot_s[i])/7067:+6.1f}%)   v4偏差={out['v4'][i]-tot_s[i]:+8.0f}"
          f"({100*(out['v4'][i]-tot_s[i])/7067:+6.1f}%)")
