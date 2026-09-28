import os, importlib.util, numpy as np, pandas as pd
spec=importlib.util.spec_from_file_location('fv','temp/GLM53/scripts/f_varying_load.py'); fv=importlib.util.module_from_spec(spec); spec.loader.exec_module(fv)
cases=[('右拇指指尖','数据1','C1'),('右拇指指尖','数据2','C2'),('右拇指指尖','数据3','C3'),
       ('左拇指指尖','数据1','C4'),('左拇指指尖','数据2','C5'),('左拇指指尖','数据3','C6'),
       ('四指指尖','数据1','C7'),('四指指尖','数据2','C8'),('四指指尖','数据3','C9')]
rows=[]
for loc,name,tag in cases:
    p=os.path.join('temp',loc,name,'device_001_seg000.csv')
    t,X,tc=fv.load_csv(p)
    total=X.sum(axis=1)
    segs=fv.find_segment(total); s0,s1=max(segs,key=lambda z:z[1]-z[0])
    main=int(np.argmax(X[s0:s1].mean(axis=0)-X[:s0].mean(axis=0)))
    amp=X[s0:s1,main].mean()-X[:s0,main].mean()
    Yv3,_=fv.run_case(fv.CompV3,X,t)
    cdf=pd.read_csv(f'temp/glm53_cpp_harness/cpp_{tag}.csv')
    Ycpp=cdf[[c for c in cdf.columns if c.startswith('ch')]].to_numpy()
    for label,Y in [('py_v3',Yv3),('cpp',Ycpp)]:
        L=Y[s0:s1,main]; nL=len(L)
        drift=100*(L[-nL//10:].mean()-L[:nL//10].mean())/amp
        rows.append((loc,name,label,drift))
for r in rows: print(r)
PY
