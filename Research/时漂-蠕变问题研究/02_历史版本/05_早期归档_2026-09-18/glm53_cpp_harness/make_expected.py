import os, numpy as np, pandas as pd, importlib.util
HERE = r'temp/GLM53/scripts/f_varying_load.py'
spec = importlib.util.spec_from_file_location('fv', HERE)
fv = importlib.util.module_from_spec(spec); spec.loader.exec_module(fv)
BASE='temp'
cases=[('变化负载','零负载-切换负载-零负载-再切换负载'),('变化负载','零负载-中途切换负载-零负载-切换负载')]
for prefix,(base,loc) in [('A',cases[0]),('B',cases[1])]:
    p=os.path.join(BASE,base,loc,'device_001_seg000.csv')
    t,X,tc=fv.load_csv(p)
    Y,_=fv.run_case(fv.CompV3,X,t)
    np.savez_compressed(f'temp/glm53_cpp_harness/expected_{prefix}.npz', t=t, Y=Y)
    print(prefix, t.shape, X.shape, Y.sum(axis=1)[:3], Y.sum(axis=1)[-3:])
