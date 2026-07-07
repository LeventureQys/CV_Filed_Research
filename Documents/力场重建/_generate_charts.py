import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

OUTDIR = Path(r'D:\workshop\Github\CVResearch\Documents\力场重建')
OUTDIR.mkdir(parents=True, exist_ok=True)

session_path = Path(r'D:\Temp\20260706_161256_single_device_e25428\session.json')
session = json.loads(session_path.read_text(encoding='utf-8'))
profile = session['devices'][0]['layout_profile']
cells = profile['cells']
W_MM = float(profile['bounds']['width_mm'])
H_MM = float(profile['bounds']['height_mm'])

csv_path = Path(r'D:\Temp\20260706_161256_single_device_e25428\device_001.csv')
lines = csv_path.read_text(encoding='utf-8').splitlines()
header_idx = next(i for i, line in enumerate(lines) if line.startswith('timestamp,'))
reader = csv.reader(lines[header_idx:])
next(reader)
frames = np.array([[float(x) for x in row[3:]] for row in reader], dtype=np.float64)
values = frames.mean(axis=0)
# Also pick one representative mid-frame
single_values = frames[len(frames)//2]

points = np.array([[cell['centroid']['x'], cell['centroid']['y']] for cell in cells], dtype=np.float64)
grid_w, grid_h = 420, 980
x = np.linspace(0, W_MM, grid_w)
y = np.linspace(0, H_MM, grid_h)
xx, yy = np.meshgrid(x, y)
query = np.column_stack([xx.ravel(), yy.ravel()])
tree = cKDTree(points)
vmax = max(float(np.percentile(values, 99)), 1.0)

# ---- Methods ----
def nearest_field(vals):
    nearest = griddata(points, vals, query, method='nearest').reshape(grid_h, grid_w)
    return nearest

def point_gaussian(vals):
    field = np.zeros((grid_h, grid_w), dtype=np.float64)
    for cell in cells:
        v = vals[int(cell['id'])]
        if v <= 0:
            continue
        sigma_mm = float(cell.get('side_length', 30.0)) * 0.5
        dx = xx - float(cell['centroid']['x'])
        dy = yy - float(cell['centroid']['y'])
        dist2 = dx * dx + dy * dy
        mask = dist2 <= (3.0 * sigma_mm)**2
        field[mask] += v * np.exp(-dist2[mask] / (2.0 * sigma_mm * sigma_mm))
    return field

def idw_field(vals, k=6, p=2.0, sigma=0.8):
    dist, idx = tree.query(query, k=k)
    dist, idx = np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=int)
    eps = 1e-6
    w = 1.0 / np.maximum(dist, eps)**p
    exact = dist[:,0] < eps
    out = np.sum(w * vals[idx], axis=1) / np.sum(w, axis=1)
    out[exact] = vals[idx[exact,0]]
    field = np.clip(out.reshape(grid_h, grid_w), 0, None)
    return gaussian_filter(field, sigma=sigma) if sigma>0 else field

def weighted_plane(vals, k=8, p=2.0, sigma=0.8):
    dist, idx = tree.query(query, k=k)
    dist, idx = np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=int)
    out = np.zeros(len(query), dtype=np.float64)
    eps = 1e-6
    for qi, q in enumerate(query):
        ids = idx[qi]; pts = points[ids]; vs = vals[ids]
        if dist[qi,0] < eps: out[qi] = vs[0]; continue
        w = 1.0 / np.maximum(dist[qi], eps)**p
        A = np.column_stack([pts[:,0]-q[0], pts[:,1]-q[1], np.ones(len(ids))])
        sw = np.sqrt(w); Aw = A * sw[:,None]; bw = vs * sw
        try:
            coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            pred = coef[2]
        except: pred = np.average(vs, weights=w)
        out[qi] = np.clip(pred, np.min(vs), np.max(vs))
    field = np.clip(out.reshape(grid_h, grid_w), 0, None)
    return gaussian_filter(field, sigma=sigma) if sigma>0 else field

def wendland_field(vals, R=60.0, sigma=1.5):
    scale = min(grid_w/W_MM, grid_h/H_MM)
    ox = (grid_w - W_MM*scale)/2.0; oy = (grid_h - H_MM*scale)/2.0
    radius = R*scale; radius_sq = radius*radius
    bin_size = radius; bins_x = int(np.ceil(grid_w/bin_size)); bins_y = int(np.ceil(grid_h/bin_size))
    bins_x = max(1,bins_x); bins_y = max(1,bins_y)
    bin_cells = [[] for _ in range(bins_x*bins_y)]
    for ci, cell in enumerate(cells):
        v = vals[int(cell['id'])]
        if v<=0:
            continue
        cx = cell['centroid']['x']*scale+ox; cy = cell['centroid']['y']*scale+oy
        if cx<0 or cx>=grid_w or cy<0 or cy>=grid_h: continue
        bx = min(bins_x-1, int(cx/bin_size)); by = min(bins_y-1, int(cy/bin_size))
        bin_cells[by*bins_x+bx].append((ci, cx, cy, v))
    field = np.zeros((grid_h, grid_w), dtype=np.float64)
    for py in range(grid_h):
        by = min(bins_y-1, int(py/bin_size))
        for px in range(grid_w):
            bx = min(bins_x-1, int(px/bin_size))
            numer, denom = 0.0, 0.0
            for dby in [-1,0,1]:
                aby = by+dby
                if aby<0 or aby>=bins_y: continue
                for dbx in [-1,0,1]:
                    abx = bx+dbx
                    if abx<0 or abx>=bins_x: continue
                    for ci, cx, cy, v in bin_cells[aby*bins_x+abx]:
                        dx = cx-px; dy = cy-py; dsq = dx*dx+dy*dy
                        if dsq <= radius_sq:
                            r = np.sqrt(dsq)/radius
                            omr = 1.0-r; wgt = omr*omr*omr*omr*(4.0*r+1.0)
                            numer += v*wgt; denom += wgt
            if denom>0:
                field[py,px] = (numer/denom)*min(1.0,denom)
    return gaussian_filter(field, sigma=sigma) if sigma>0 else field

# ---- Generate ----
SAVE_PAIRS = [
    ('Nearest (raw)', nearest_field),
    ('Point Gaussian', point_gaussian),
    ('KNN-IDW k=6', lambda v: idw_field(v, k=6)),
    ('Weighted Plane k=8', lambda v: weighted_plane(v, k=8)),
    ('Wendland R=60mm', lambda v: wendland_field(v)),
]

for name, fn in SAVE_PAIRS:
    f = fn(values)
    fig, ax = plt.subplots(figsize=(5.8,13.5), dpi=140)
    im = ax.imshow(f, origin='lower', extent=[0,W_MM,0,H_MM], cmap='jet', vmin=0, vmax=vmax, aspect='equal')
    ax.scatter(points[:,0], points[:,1], c='white', s=2, alpha=0.30, linewidths=0)
    ax.set_title(name, fontsize=11); ax.set_xlabel('x mm'); ax.set_ylabel('y mm')
    fig.colorbar(im, ax=ax, shrink=0.75, label='ADC avg'); fig.tight_layout()
    slug = name.lower().replace(' ','_').replace('=','').replace('(','').replace(')','').replace('-','')
    fig.savefig(OUTDIR / f'{slug}.png'); plt.close(fig)

# Composite chart
fig, axes = plt.subplots(1,5, figsize=(20, 13), dpi=140)
for ax, (name, fn) in zip(axes, SAVE_PAIRS):
    f = fn(values)
    ax.imshow(f, origin='lower', extent=[0,W_MM,0,H_MM], cmap='jet', vmin=0, vmax=vmax, aspect='equal')
    ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUTDIR / '00_comparison_all.png'); plt.close(fig)

# Slice detail: zoom to active region y=850..1280
fig, axes = plt.subplots(1,5, figsize=(20, 9), dpi=140)
for ax, (name, fn) in zip(axes, SAVE_PAIRS):
    f = fn(values)
    ax.imshow(f, origin='lower', extent=[0,W_MM,0,H_MM], cmap='jet', vmin=0, vmax=vmax, aspect='equal')
    ax.set_ylim(850, 1280); ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUTDIR / '01_comparison_zoom.png'); plt.close(fig)

# Metrics CSV
metrics = []
for name, fn in SAVE_PAIRS:
    f = fn(values)
    fm = np.max(f)
    pc = 0
    for iy in range(1, f.shape[0]-1):
        for ix in range(1, f.shape[1]-1):
            v = f[iy,ix]
            if v >= 0.3*fm and v >= np.max(f[iy-1:iy+2, ix-1:ix+2]): pc+=1
    metrics.append((name, f'{fm:.1f}', pc))
with open(OUTDIR / 'metrics.csv', 'w') as fp:
    fp.write('method,max_value,local_peaks_30pct\n')
    for name, fm, pc in metrics: fp.write(f'{name},{fm},{pc}\n')

print('Charts written to', OUTDIR)
for p in sorted(OUTDIR.glob('*.png')): print(p)
