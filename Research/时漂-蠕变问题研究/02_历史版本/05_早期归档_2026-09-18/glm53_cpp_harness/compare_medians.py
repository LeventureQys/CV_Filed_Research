import importlib.util, os, numpy as np, pandas as pd
spec=importlib.util.spec_from_file_location('cmpy','temp/glm53_cpp_harness/cpp_median_py.py'); cmpy=importlib.util.module_from_spec(spec); spec.loader.exec_module(cmpy)
for tag in ['A','B']:
    y0=np.load(f'temp/glm53_cpp_harness/expected_{tag}.npz')['Y'].sum(axis=1)
    y1=np.load(f'temp/glm53_cpp_harness/cppmedian_{tag}.npz')['Y'].sum(axis=1)
    d=y1-y0
    print(tag,'orig vs upper max_abs',np.max(np.abs(d)),'rms',np.sqrt(np.mean(d*d)),'argmax',np.argmax(np.abs(d)))
PY
