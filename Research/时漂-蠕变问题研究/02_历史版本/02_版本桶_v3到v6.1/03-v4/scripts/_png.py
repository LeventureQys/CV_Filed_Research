import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
for f in ("figures/new_switch_load_result.png","figures/new_switch_load_metrics.png"):
    a = imread(f)
    # 粗略检查：非白像素占比（判断不是空白图）
    g = a[...,:3].mean(axis=2)
    print(f, a.shape, f"非白像素占比={100*(g<0.98).mean():.1f}%")
