# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'D:\workshop\文档\v2.7 - 抗蠕变补偿算法\v3.4\sensor_tune')
import numpy as np
from sensor_common import OUT_DIR
from sweep_lib import PREP

for key in ('右拇指指腹/d1', '右手掌/d1/A'):
    p = PREP[key]
    z = np.load(OUT_DIR / '_prep' / (key.replace('/', '_') + '.npz'))
    print(key, 'Esum=%.1f creep=%.1f tau=(%.1f,%.1f) t0=%.2f' %
          (p['Esum_channels'], p['creep_total'], p['tau1'], p['tau2'], p['t0']))
    print('  t:', z['t'][0], '->', z['t'][-1], 'n=', len(z['t']), 'tin_end=', z['tin'][-1])
    rel = np.maximum(z['t'] - p['t0'] - 0.6, 0.0)
    ramp = np.minimum(np.maximum(z['t'] - p['t0'], 0) / 0.6, 1)
    creep = ((z['cs'][:, 0:1] * (1 - np.exp(-rel / p['tau1']))
              + z['cs'][:, 1:2] * (1 - np.exp(-rel / p['tau2']))) * ramp).sum(axis=0)
    ref = z['Es'].sum() + creep
    print('  ref_end=%.1f  max|ref-tin|=%.1f' % (ref[-1], np.abs(ref - z['tin']).max()))
