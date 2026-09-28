import matplotlib; matplotlib.use("Agg")
from matplotlib.image import imread
import numpy as np
a = imread("figures/new_switch_load_result.png")
print("shape", a.shape, "dtype", a.dtype)
print("corner(5,5) =", a[5,5], " center =", a[a.shape[0]//2, a.shape[1]//2])
rgb = a[...,:3]
print("alpha min/max =", a[...,3].min(), a[...,3].max())
print("纯白像素占比 =", 100*np.all(rgb>0.98, axis=2).mean())
vals, cnt = np.unique((rgb.mean(axis=2)*20).astype(int), return_counts=True)
for v,c in sorted(zip(vals,cnt), key=lambda z:-z[1])[:6]:
    print(f"  亮度档 {v/20:.2f}: {100*c/rgb.shape[0]/rgb.shape[1]:.1f}%")
