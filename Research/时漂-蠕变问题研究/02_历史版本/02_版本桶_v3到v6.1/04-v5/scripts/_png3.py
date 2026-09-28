import matplotlib; matplotlib.use("Agg")
from matplotlib.image import imread
import numpy as np
a = imread("figures/v5_compare.png")
print("shape", a.shape, "dtype", a.dtype)
rgb = a[...,:3]
print("纯白占比", round(100*np.all(rgb>0.98, axis=2).mean(),1), "%")
print("corner", a[3,3], "center", a[a.shape[0]//2, a.shape[1]//2])
