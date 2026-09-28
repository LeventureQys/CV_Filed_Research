import numpy as np, pandas as pd
for tag in ['A','B']:
    exp = np.load(f'temp/glm53_cpp_harness/expected_{tag}.npz')
    cpp = pd.read_csv(f'temp/glm53_cpp_harness/cpp_{tag}.csv', header=None, names=['i','t','tot'])
    y = exp['Y'].sum(axis=1)
    d = np.abs(cpp['tot'].to_numpy()-y)
    first = np.where(d>1e-6)[0]
    print('\n',tag,'first idx', first[:10], 't', cpp['t'].iloc[first[:10]].to_numpy() if len(first) else [])
    # print difference around first several and largest windows
    for idx in list(first[:5]) + [np.argmax(d)]:
        lo=max(0,idx-5); hi=min(len(d),idx+6)
        print('idx',idx,'t',cpp['t'].iloc[idx])
        for j in range(lo,hi):
            print(j, round(cpp['t'].iloc[j],4), round(cpp['tot'].iloc[j],3), round(y[j],3), round(cpp['tot'].iloc[j]-y[j],3))
PY
