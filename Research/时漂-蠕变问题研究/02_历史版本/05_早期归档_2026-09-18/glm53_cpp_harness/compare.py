import numpy as np, pandas as pd
for tag in ['A','B']:
    exp = np.load(f'temp/glm53_cpp_harness/expected_{tag}.npz')
    cpp = pd.read_csv(f'temp/glm53_cpp_harness/cpp_{tag}.csv', header=None, names=['i','t','tot'])
    y = exp['Y'].sum(axis=1)
    c = cpp['tot'].to_numpy()
    d = c - y
    print(tag, 'n', len(d), 'max_abs', np.max(np.abs(d)), 'rms', np.sqrt(np.mean(d*d)), 'argmax', np.argmax(np.abs(d)), 't_at_max', cpp['t'].iloc[np.argmax(np.abs(d))])
    print('first/last', d[0], d[-1], 'cpp first/last', c[0], c[-1], 'py first/last', y[0], y[-1])
PY
