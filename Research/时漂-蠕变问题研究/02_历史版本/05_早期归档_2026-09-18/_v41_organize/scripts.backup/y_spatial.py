# -*- coding: utf-8 -*-
"""v4.1flash 步骤3：dsp.md §4「空间 PSF 逆卷积（3×3 拉普拉斯形高通核）」实测。

§4 的主张：对 10×10 阵列用微型 3×3 二维空间高通卷积核（拉普拉斯算子形）
做反卷积，即可压制受压区域边缘的溢出形变。

本脚本用实测载荷图检验它是否成立，量化三件事：
  1. 总力守恒：∑(反卷积后) / ∑(原始)  —— 高通核直流增益为 0，必然丢总量；
  2. 负值比例：反卷积后出现负读数的通道占比（物理上不可能）；
  3. 噪声放大：核的 ℓ2 增益对白噪声的放大倍数；
  4. 对「已知真值」的重建误差：构造一个 5×5 方形真值载荷，卷积高斯 PSF 后反卷积，
     看能否复原（PSF 未标定时，用固定拉普拉斯核等价于假设特定 PSF）。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
LINES = []


def say(s=""):
    LINES.append(str(s))


def load(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def find_segment(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]


say("=" * 96)
say("dsp.md §4 空间反卷积实测（temp/v4.1flash/scripts/y_spatial.py）")
say("=" * 96)
say("§4 原话：「通过微型 3×3 二维空间高通卷积核（拉普拉斯算子形）进行反卷积，")
say("          即可压制受压区域边缘的边缘溢出形变。」")
say("")
say("标准 4 邻域拉普拉斯核 L = [[0,1,0],[1,-4,1],[0,1,0]]。反卷积写作 (I − αL) 的逆，")
say("α 由 PSF 决定；§4 没有给 α 的标定方法，最简形式（α=1，即「高通核」）就是直接作用 L。")

K_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
say(f"\n核 L = \n{K_LAP}")
say(f"  核的直流增益（∑核）= {K_LAP.sum():.1f}  ⇒ 对任何常数场输出恒为 0")
say(f"  核的 ℓ2 增益（白噪放大）= {np.sqrt((K_LAP**2).sum()):.4f}")

# ---------------------------------------------------------------- 1. 实测载荷图
say("\n" + "-" * 96)
say("[1] 作用在实测载荷图上（8×5 四指指尖 / 9×7 右拇指指尖）")
say("-" * 96)
CASES = [("四指指尖", "数据1", 8, 5), ("四指指尖", "数据2", 8, 5),
         ("右拇指指尖", "数据1", 9, 7), ("右拇指指尖", "数据2", 9, 7)]
say(f"{'数据':<22}|{'∑原始':>10}|{'∑反卷积':>10}|{'总量比':>9}|{'负值通道':>9}|{'|负值|/幅值':>11}|{'峰值变化':>9}")
say("-" * 96)
sp_rows = []
for loc, name, rows, cols in CASES:
    ddir = os.path.join(TEMP, loc, name)
    t, X = load(os.path.join(ddir, "device_001_seg000.csv"))
    s0, s1 = find_segment(X.sum(axis=1))
    nL = s1 - s0
    frame = X[s0 + int(0.2 * nL):s0 + int(0.5 * nL)].mean(axis=0)      # 负载已建立
    base = X[:s0].mean(axis=0)
    f = frame - base
    # 映射回 rows×cols 网格（layout_mask 有空洞，缺失位置置 0）
    import json
    with open(os.path.join(ddir, "session.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    dev = meta["devices"][0] if "devices" in meta else meta
    mask = np.array([c == "1" for c in dev["layout_mask"]]).reshape(rows, cols)
    grid = np.zeros((rows, cols))
    grid[mask] = f
    # 逐位置卷积（边界用零填充）
    conv = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            acc = 0.0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    acc += grid[rr, cc]
            conv[r, c] = acc - 4 * grid[r, c]
    gv, cv = grid[mask], conv[mask]
    tot_ratio = cv.sum() / gv.sum() if abs(gv.sum()) > 1e-12 else np.nan
    neg = int((cv < 0).sum())
    negmag = np.abs(cv[cv < 0]).sum() / np.abs(gv).sum() if (cv < 0).any() else 0.0
    peak = cv.max() / gv.max() if gv.max() > 1e-12 else np.nan
    say(f"{loc + '/' + name:<22}|{gv.sum():10.2f}|{cv.sum():10.2f}|{tot_ratio:9.4f}|"
        f"{neg:4d}/{len(gv):<4d}|{negmag:11.4f}|{peak:9.3f}")
    sp_rows.append(dict(location=loc, dataset=name, sum_raw=gv.sum(), sum_deconv=cv.sum(),
                        total_ratio=tot_ratio, neg_channels=neg, n_channels=len(gv),
                        neg_mag_ratio=negmag, peak_ratio=peak))
pd.DataFrame(sp_rows).to_csv(os.path.join(RES, "spatial_deconv.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 2. 已知真值重建
say("\n" + "-" * 96)
say("[2] 已知真值重建：10×10 阵、5×5 方形载荷、高斯 PSF(σ=1.0 格)")
say("-" * 96)


def gauss_psf(sigma, rad=3):
    ax = np.arange(-rad, rad + 1)
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def conv2_same(img, k):
    r, c = img.shape
    kr, kc = k.shape
    pr, pc = kr // 2, kc // 2
    pad = np.pad(img, ((pr, pr), (pc, pc)))
    out = np.zeros_like(img, dtype=float)
    for i in range(r):
        for j in range(c):
            out[i, j] = (pad[i:i + kr, j:j + kc] * k).sum()
    return out


def deconv_laplacian(img, alpha):
    """(I − α·L) 的逆：用 FFT 在频域做（等价于周期边界的最小二乘反卷积）"""
    r, c = img.shape
    L = np.zeros((r, c))
    L[0, 0] = L[0, -1] = L[-1, 0] = L[-1, -1] = -4
    L[0, 1] = L[0, -2] = L[-1, 1] = L[-1, -2] = 1
    L[1, 0] = L[2, 0] = L[-2, 0] = L[-1, 0] = 0
    # 直接构造周期拉普拉斯算子
    Lp = np.zeros((r, c))
    Lp[0, 0], Lp[0, 1], Lp[0, -1], Lp[1, 0], Lp[-1, 0] = -4, 1, 1, 1, 1
    H = np.fft.fft2(np.eye(r, c) if r == c else np.eye(r, c))
    # 用频域符号算：L 的特征值 = -4 + 2cos(2πu/r) + 2cos(2πv/c)
    u = np.fft.fftfreq(r)[:, None]
    v = np.fft.fftfreq(c)[None, :]
    lam = -4 + 2 * np.cos(2 * np.pi * u) + 2 * np.cos(2 * np.pi * v)
    denom = 1 - alpha * lam
    denom[np.abs(denom) < 1e-9] = 1e-9
    return np.real(np.fft.ifft2(np.fft.fft2(img) / denom))


R, C = 10, 10
true = np.zeros((R, C))
true[3:8, 3:8] = 1.0
psf = gauss_psf(1.0)
meas = conv2_same(true, psf)
say(f"真值 5×5 方形（中心 25 格=1）：∑真值 = {true.sum():.1f}")
say(f"PSF 卷积后测量图：∑测量 = {meas.sum():.1f}（卷积保持总量）")
say("")
say(f"{'反卷积 α':>10}|{'∑重建':>9}|{'总量比':>8}|{'负值格数':>9}|{'重建 RMSE':>10}|{'峰值/真值':>10}")
say("-" * 72)
for alpha in [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]:
    rec = deconv_laplacian(meas, alpha)
    rmse = float(np.sqrt(np.mean((rec - true) ** 2)))
    say(f"{alpha:10.2f}|{rec.sum():9.2f}|{rec.sum()/true.sum():8.3f}|"
        f"{int((rec < -1e-9).sum()):9d}|{rmse:10.4f}|{rec.max()/true.max():10.3f}")
say("")
say("【说明】α = 0 表示不做反卷积（重建 = 测量图，RMSE 即 PSF 模糊造成的最小误差）。")
rec0 = meas
say(f"         α = 0（不处理）基线：∑={rec0.sum():.2f}  RMSE={np.sqrt(np.mean((rec0-true)**2)):.4f}  "
    f"峰值/真值={rec0.max()/true.max():.3f}")
say("")
say("=" * 96)
say("结论")
say("=" * 96)
say("1. 拉普拉斯高通核的直流增益为 0：直接作用会把阵列总力打成接近 0（见 [1] 的总量比），")
say("   这等于把「测到的力」整体丢掉——§4 把它当作可用的反卷积算子是错的。")
say("2. 正确的反卷积必须是 (I − αL) 的逆（分母），而不是 L 本身；§4 的措辞")
say("   「用高通核进行反卷积」把二者混为一谈，照着实现必然得出负读数与总量错乱。")
say("3. 即便写成 (I − αL)⁻¹，α 必须由 PSF 标定；§4 未给出 PSF 辨识步骤，")
say("   而 PSF 随指尖位置、载荷大小、接触面形状变化，单组固定 α 无法通用。")
say("4. 反卷积在频域是高通放大：噪声与量化误差被 α 放大，且阵列边缘（含 layout 空洞）")
say("   会产生明显振铃——而柔性阵列的量化噪声本来就比力信号精细。")

with open(os.path.join(RES, "spatial_check.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES) + "\n")
print("saved:", os.path.join(RES, "spatial_check.txt"))
