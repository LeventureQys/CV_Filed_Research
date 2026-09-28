import importlib.util, os, numpy as np, pandas as pd
spec=importlib.util.spec_from_file_location('cmpy','temp/glm53_cpp_harness/cpp_median_py.py')
cmpy=importlib.util.module_from_spec(spec); spec.loader.exec_module(cmpy)
fv=cmpy.fv
for tag,(base,loc) in [('A',('变化负载','零负载-切换负载-零负载-再切换负载')),('B',('变化负载','零负载-中途切换负载-零负载-切换负载'))]:
    t,X,tc=fv.load_csv(os.path.join('temp',base,loc,'device_001_seg000.csv'))
    c=cmpy.CppMedianV3(); Y=np.empty_like(X,float)
    for i in range(len(t)): Y[i]=c.process(t[i],X[i].astype(float))
    np.savez_compressed(f'temp/glm53_cpp_harness/cppmedian_{tag}.npz', Y=Y)
    cpp=pd.read_csv(f'temp/glm53_cpp_harness/cpp_{tag}.csv',header=None,names=['i','t','tot'])
    d=cpp['tot'].to_numpy()-Y.sum(axis=1)
    print(tag,'cpp vs py_upper max_abs',np.max(np.abs(d)),'rms',np.sqrt(np.mean(d*d)),'argmax',np.argmax(np.abs(d)),'t',cpp['t'].iloc[np.argmax(np.abs(d))])
PY
