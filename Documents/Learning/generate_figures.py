from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, median_filter, rotate, zoom
from scipy.signal import convolve2d

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def test_image(size=180):
    """合成测试图：左半圆、右矩形，中间渐变条"""
    img = np.zeros((size, size), dtype=np.float32)
    rr, cc = np.meshgrid(np.arange(size), np.arange(size))
    mask_circle = (rr - size * 0.28) ** 2 + (cc - size * 0.5) ** 2 < (size * 0.22) ** 2
    mask_rect = (rr > size * 0.15) & (rr < size * 0.85) & (cc > size * 0.62) & (cc < size * 0.88)
    grad = np.clip(cc / size * 200 + 20, 20, 240)
    img[mask_circle] = 210
    img[mask_rect] = 60
    img[~mask_circle & ~mask_rect] = grad[~mask_circle & ~mask_rect]
    img += np.random.default_rng(42).normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def texton_pattern(size=180):
    """条纹+网格测试图案"""
    img = np.ones((size, size), dtype=np.float32) * 128
    rr, cc = np.meshgrid(np.arange(size), np.arange(size))
    img += 40 * np.sin(2 * np.pi * rr / 8)
    img += 40 * np.sin(2 * np.pi * cc / 6)
    img += np.random.default_rng(7).normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _readable_size(px):
    if px >= 1024:
        return f"{px/1024:.1f}KB"
    return f"{px}B"


# ═══════════════════════════════════════════════════════════════════
# Chapter 01 — 采样与量化
# ═══════════════════════════════════════════════════════════════════

def ch01_fig01():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.linspace(0, 1, 400)
    y = 0.5 + 0.35 * np.sin(2 * np.pi * 3 * x) + 0.08 * np.sin(2 * np.pi * 14 * x)
    axes[0].plot(x, y, lw=2.2, color="#2F5597")
    axes[0].set_title("(a) 连续亮度信号", fontsize=13)
    axes[0].set_xlabel("空间位置 x"); axes[0].set_ylabel("亮度 f(x)")
    xs_dense = np.linspace(0, 1, 40)
    ys_dense = np.interp(xs_dense, x, y)
    axes[1].plot(x, y, color="0.75", lw=1)
    markers, _, _ = axes[1].stem(xs_dense, ys_dense, linefmt="C0-", markerfmt="C0o", basefmt=" ")
    axes[1].set_title("(b) 密集采样 (40 点)", fontsize=13)
    axes[1].set_xlabel("空间位置 x")
    xs_sparse = np.linspace(0, 1, 11)
    ys_sparse = np.interp(xs_sparse, x, y)
    axes[2].plot(x, y, color="0.75", lw=1)
    axes[2].stem(xs_sparse, ys_sparse, linefmt="C3-", markerfmt="C3s", basefmt=" ")
    axes[2].set_title("(c) 稀疏采样 (11 点)", fontsize=13)
    axes[2].set_xlabel("空间位置 x")
    for ax in axes:
        ax.set_ylim(0.05, 1.05)
        ax.grid(alpha=0.2)
    save("ch01_fig01_continuous_to_sampled.png")


def ch01_fig02():
    size = 180
    rr, cc = np.meshgrid(np.arange(size), np.arange(size))
    freq = 12
    pattern = 128 + 127 * np.sin(2 * np.pi * rr / (size / freq))
    aliased = pattern[::6, ::6]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(pattern, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("(a) 细密条纹图案 (高频)", fontsize=13)
    axes[1].imshow(aliased, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                   extent=[0, size, 0, size])
    axes[1].set_title("(b) 欠采样结果 (出现假低频)", fontsize=13)
    from scipy.ndimage import gaussian_filter
    filtered = gaussian_filter(pattern.astype(np.float32), sigma=2.5)
    anti = filtered[::6, ::6]
    axes[2].imshow(anti, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                   extent=[0, size, 0, size])
    axes[2].set_title("(c) 先低通再降采样 (无混叠)", fontsize=13)
    for ax in axes:
        ax.axis("off")
    save("ch01_fig02_aliasing.png")


def ch01_fig03():
    size = 180
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    img = 0.25 + 0.25 * xx + 0.25 * yy + 0.25 * np.sin(2 * np.pi * xx * 3) * np.cos(2 * np.pi * yy * 4)
    img = np.clip((img * 255), 0, 255)
    bits = [2, 3, 4, 5, 6, 8]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for idx, b in enumerate(bits):
        ax = axes[idx // 3, idx % 3]
        levels = 2 ** b
        q = (np.floor(img / (256 / levels)) * (256 / levels)).astype(np.uint8)
        ax.imshow(q, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{b} bit / {levels} 级", fontsize=13)
        ax.axis("off")
    save("ch01_fig03_quantization_levels.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 02 — 点运算与代数运算
# ═══════════════════════════════════════════════════════════════════

def ch02_fig01():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    f = np.linspace(0, 255, 256)
    a, b = 1.4, -50
    g = np.clip(a * f + b, 0, 255)
    axes[0].plot(f, f, "k--", lw=1, alpha=0.4, label="恒等 g=f")
    axes[0].plot(f, g, "C0", lw=2.5, label=f"g = {a}f + ({b})")
    axes[0].fill_between(f[:80], 0, g[:80], alpha=0.15, color="C0")
    axes[0].fill_between(f[180:], g[180:], 255, alpha=0.15, color="C3")
    axes[0].annotate("暗部被裁为0", (20, 10), fontsize=9, color="C0")
    axes[0].annotate("亮部被裁为255", (200, 260), fontsize=9, color="C3")
    axes[0].set_title("(a) 对比度增强映射", fontsize=13)
    axes[0].set_xlabel("输入灰度 f")
    axes[0].set_ylabel("输出灰度 g")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    a2, b2 = 0.5, 60
    g2 = np.clip(a2 * f + b2, 0, 255)
    axes[1].plot(f, f, "k--", lw=1, alpha=0.4, label="恒等 g=f")
    axes[1].plot(f, g2, "C2", lw=2.5, label=f"g = {a2}f + {b2}")
    axes[1].set_title("(b) 对比度压缩映射", fontsize=13)
    axes[1].set_xlabel("输入灰度 f"); axes[1].set_ylabel("输出灰度 g")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    ti = test_image(120)
    enhanced = np.clip(1.4 * ti.astype(np.float32) - 50, 0, 255).astype(np.uint8)
    compressed = np.clip(0.5 * ti.astype(np.float32) + 60, 0, 255).astype(np.uint8)
    axes[2].imshow(np.hstack([ti, enhanced, compressed]), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("(c) 原图 | 增强 | 压缩", fontsize=13)
    axes[2].axis("off")
    axes[2].text(60, 130, "原图", ha="center", fontsize=9, color="w")
    axes[2].text(180, 130, "增强", ha="center", fontsize=9, color="w")
    axes[2].text(300, 130, "压缩", ha="center", fontsize=9, color="w")
    save("ch02_fig01_linear_transform.png")


def ch02_fig02():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    f = np.linspace(1, 255, 256)
    gammas = [0.3, 0.5, 0.7, 1.0, 1.5, 2.2, 3.0]
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(gammas)))
    for gm, c in zip(gammas, colors):
        axes[0].plot(f, 255 * (f / 255) ** gm, color=c, lw=1.8, label=f"γ={gm}")
    axes[0].plot(f, f, "k--", lw=1, alpha=0.5)
    axes[0].set_title("(a) Gamma 曲线族", fontsize=13)
    axes[0].set_xlabel("输入灰度"); axes[0].set_ylabel("输出灰度")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.2)
    ti = test_image(120)
    dark = np.clip(255 * (ti.astype(np.float32) / 255) ** 2.2, 0, 255).astype(np.uint8)
    bright = np.clip(255 * (ti.astype(np.float32) / 255) ** 0.45, 0, 255).astype(np.uint8)
    axes[1].imshow(np.hstack([ti, dark]), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("(b) 原图 vs γ=2.2 (变暗)", fontsize=13)
    axes[1].axis("off")
    axes[2].imshow(np.hstack([ti, bright]), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("(c) 原图 vs γ=0.45 (变亮)", fontsize=13)
    axes[2].axis("off")
    save("ch02_fig02_gamma.png")


def ch02_fig03():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ti = test_image(150).astype(np.float32)
    ti_dark = ti * 0.55 + np.random.default_rng(1).normal(0, 6, ti.shape)
    ti_dark = np.clip(ti_dark, 0, 255).astype(np.uint8)
    axes[0, 0].imshow(ti_dark, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) 低对比度图像", fontsize=13)
    axes[0, 0].axis("off")
    hist, bins = np.histogram(ti_dark.ravel(), 256, [0, 256])
    axes[0, 1].bar(bins[:-1], hist, width=1, color="C0", alpha=0.8)
    axes[0, 1].set_title("(b) 直方图 (集中在窄区间)", fontsize=13)
    axes[0, 1].set_xlim(0, 255); axes[0, 1].set_ylabel("像素数")
    cdf = hist.cumsum()
    cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    eq = cdf_norm[ti_dark].astype(np.uint8)
    axes[1, 0].imshow(eq, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("(c) 直方图均衡化后", fontsize=13)
    axes[1, 0].axis("off")
    hist_eq, _ = np.histogram(eq.ravel(), 256, [0, 256])
    axes[1, 1].bar(bins[:-1], hist_eq, width=1, color="C2", alpha=0.8)
    axes[1, 1].set_title("(d) 均衡化后直方图 (更分散)", fontsize=13)
    axes[1, 1].set_xlim(0, 255); axes[1, 1].set_ylabel("像素数")
    save("ch02_fig03_histogram_eq.png")


def ch02_fig04():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ti = test_image(150)
    t_vals = [90, 140]
    for idx, t in enumerate(t_vals):
        ax_img = axes[idx, 0]
        ax_img.imshow(ti, cmap="gray", vmin=0, vmax=255)
        ax_img.set_title(f"(a{'c' if idx else 'a'}) 原图 & 阈值 T={t}", fontsize=13)
        y = t
        ax_img.axhline(y=y, color="r", lw=2, linestyle="--")
        ax_img.text(5, t - 10, f"T={t}", color="r", fontsize=12, fontweight="bold")
        ax_img.axis("off")
        thresh = np.where(ti > t, 255, 0).astype(np.uint8)
        ax_th = axes[idx, 1]
        ax_th.imshow(thresh, cmap="gray", vmin=0, vmax=255)
        ax_th.set_title(f"(b{'d' if idx else 'b'}) 阈值分割 T={t}", fontsize=13)
        ax_th.axis("off")
    save("ch02_fig04_threshold.png")


def ch02_fig05():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    ti = test_image(120).astype(np.float32)
    noise = np.random.default_rng(2).normal(0, 15, ti.shape)
    bg = ti + noise
    obj = ti.copy()
    obj[30:50, 30:50] = 180
    obj = obj + np.random.default_rng(3).normal(0, 15, ti.shape)
    bg_clip = np.clip(bg, 0, 255).astype(np.uint8)
    obj_clip = np.clip(obj, 0, 255).astype(np.uint8)
    diff = np.clip(obj.astype(np.float32) - bg.astype(np.float32) + 128, 0, 255).astype(np.uint8)
    axes[0, 0].imshow(bg_clip, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) 背景帧", fontsize=12); axes[0, 0].axis("off")
    axes[0, 1].imshow(obj_clip, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("(b) 含目标帧", fontsize=12); axes[0, 1].axis("off")
    axes[0, 2].imshow(diff, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("(c) 差分图像", fontsize=12); axes[0, 2].axis("off")
    frames = []
    for i in range(8):
        f = ti + np.random.default_rng(i + 10).normal(0, 20, ti.shape)
        frames.append(np.clip(f, 0, 255))
    single = frames[0].astype(np.uint8)
    avg = np.clip(np.mean(frames, axis=0), 0, 255).astype(np.uint8)
    axes[1, 0].imshow(single, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("(d) 单帧 (噪声大)", fontsize=12); axes[1, 0].axis("off")
    axes[1, 1].imshow(avg, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("(e) 8 帧平均 (降噪)", fontsize=12); axes[1, 1].axis("off")
    axes[1, 2].imshow(np.abs(single.astype(float) - avg.astype(float)), cmap="hot")
    axes[1, 2].set_title("(f) 噪声残差", fontsize=12); axes[1, 2].axis("off")
    save("ch02_fig05_algebraic.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 03 — 几何变换
# ═══════════════════════════════════════════════════════════════════

def ch03_fig01():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    colors = ["#FFF2CC", "#E2F0D9", "#DDEBF7", "#FCE4D6"]
    texts = ["(a) 输出网格\n逐像素遍历", "(b) 逆变换\nM⁻¹", "(c) 输入浮点坐标\n(x, y)", "(d) 插值\n取灰度"]
    xs = [1.0, 4.0, 7.5, 4.0]
    ys = [5.5, 5.5, 5.5, 2.0]
    ws = [2.2, 1.8, 2.2, 3.0]
    for idx in range(4):
        rect = plt.Rectangle((xs[idx], ys[idx]), ws[idx], 1.4, fc=colors[idx],
                              ec="#2F5597", lw=1.8, joinstyle="round")
        ax.add_patch(rect)
        ax.text(xs[idx] + ws[idx] / 2, ys[idx] + 0.7, texts[idx],
                ha="center", va="center", fontsize=11)
    arrows = [(3.2, 6.2, 4.0, 6.2), (5.8, 6.2, 7.5, 6.2),
              (8.6, 5.5, 5.5, 3.4), (5.2, 3.4, 2.5, 5.5)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", lw=2.5, color="#2F5597"))
    ax.text(6.6, 2.3, "填回输出像素", ha="center", fontsize=10, color="#2F5597")
    ax.set_title("图3-1  反向映射：输出找输入，逐个像素赋值", fontsize=14, pad=12)
    ax.axis("off")
    save("ch03_fig01_backward_mapping.png")


def ch03_fig02():
    pattern = np.zeros((48, 48), dtype=np.float32)
    rr, cc = np.meshgrid(np.arange(48), np.arange(48))
    pattern[(rr % 8 < 2) | (cc % 8 < 2)] = 1.0
    pattern[(rr > 20) & (rr < 28) & (cc > 20) & (cc < 28)] = 0.8
    scale = 6
    nn = zoom(pattern, scale, order=0)
    bl = zoom(pattern, scale, order=1)
    bc = zoom(pattern, scale, order=3)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(pattern, cmap="gray", interpolation="nearest")
    axes[0].set_title("(a) 原图 48×48", fontsize=12)
    for ax, img, title in zip(axes[1:], [nn, bl, bc],
                               ["(b) 最近邻", "(c) 双线性", "(d) 双三次"]):
        ax.imshow(img[24:-24, 24:-24], cmap="gray")
        ax.set_title(title, fontsize=12)
    for ax in axes:
        ax.axis("off")
    save("ch03_fig02_interpolation.png")


def ch03_fig03():
    size = 120
    gx, gy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    grid = ((np.abs(gx * 10 % 1 - 0.5) < 0.04) | (np.abs(gy * 10 % 1 - 0.5) < 0.04)).astype(np.float32)
    grid[grid < 0.3] = 0.3
    rotated = rotate(grid, 25, reshape=True, order=1)
    scaled = zoom(grid, 1.6, order=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(grid, cmap="gray")
    axes[0].set_title("(a) 原网格", fontsize=13)
    axes[1].imshow(rotated, cmap="gray")
    axes[1].set_title("(b) 旋转 25°", fontsize=13)
    axes[2].imshow(scaled, cmap="gray")
    axes[2].set_title("(c) 放大 1.6×", fontsize=13)
    for ax in axes:
        ax.axis("off")
    save("ch03_fig03_affine_grid.png")


def ch03_fig04():
    size = 180
    rr, cc = np.meshgrid(np.arange(size), np.arange(size))
    pattern = (128 + 120 * np.sin(2 * np.pi * rr / 4.5)).astype(np.float32)
    naive = pattern[::6, ::6]
    filt = gaussian_filter(pattern, sigma=2.8)
    anti = filt[::6, ::6]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(pattern, cmap="gray")
    axes[0].set_title("(a) 高密度条纹原图", fontsize=13)
    axes[1].imshow(naive, cmap="gray", interpolation="nearest",
                   extent=[0, size, 0, size])
    axes[1].set_title("(b) 直接降采样 (出现混叠条纹)", fontsize=13)
    axes[2].imshow(anti, cmap="gray", interpolation="nearest",
                   extent=[0, size, 0, size])
    axes[2].set_title("(c) 先低通再降采样 (抑制混叠)", fontsize=13)
    for ax in axes:
        ax.axis("off")
    save("ch03_fig04_antialiasing.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 04 — 时频变换
# ═══════════════════════════════════════════════════════════════════

def ch04_fig01():
    size = 180
    rr, cc = np.meshgrid(np.arange(size), np.arange(size))
    img = (np.sin(2 * np.pi * (rr + cc) / 16) + 0.4 * np.sin(2 * np.pi * rr / 5)
           + 0.3 * np.sin(2 * np.pi * cc / 7))
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))
    phase = np.angle(np.fft.fftshift(np.fft.fft2(img)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("(a) 条纹 + 斜纹 空域图", fontsize=13)
    axes[1].imshow(spec, cmap="magma")
    axes[1].set_title("(b) 幅度谱 (对数)", fontsize=13)
    axes[2].imshow(phase, cmap="twilight")
    axes[2].set_title("(c) 相位谱", fontsize=13)
    for ax in axes:
        ax.axis("off")
    save("ch04_fig01_spectrum.png")


def ch04_fig02():
    size = 160
    rr, cc = np.meshgrid(np.arange(size), np.arange(size))
    img = (np.sin(2 * np.pi * (rr + cc) / 20) + 0.5 * np.sin(2 * np.pi * rr / 4)
           + 0.3 * np.sin(2 * np.pi * cc / 5.5))
    noise = np.random.default_rng(5).normal(0, 0.08, img.shape)
    img_noisy = img + noise
    f = np.fft.fft2(img_noisy)
    fshift = np.fft.fftshift(f)
    cy, cx = size // 2, size // 2
    lpf = np.zeros((size, size))
    hpf = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
            lpf[i, j] = 1 / (1 + (d / 18) ** 4)
            hpf[i, j] = 1 - 1 / (1 + (d / 18) ** 4)
    low = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * lpf)))
    high = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * hpf)))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes[0, 0].imshow(img_noisy, cmap="gray")
    axes[0, 0].set_title("(a) 含噪条纹", fontsize=12); axes[0, 0].axis("off")
    axes[0, 1].imshow(lpf, cmap="viridis")
    axes[0, 1].set_title("(b) 低通滤波器 H_LP", fontsize=12); axes[0, 1].axis("off")
    axes[0, 2].imshow(low, cmap="gray")
    axes[0, 2].set_title("(c) 低通结果 (平滑)", fontsize=12); axes[0, 2].axis("off")
    axes[1, 0].imshow(img_noisy, cmap="gray")
    axes[1, 0].set_title("(d) 含噪条纹", fontsize=12); axes[1, 0].axis("off")
    axes[1, 1].imshow(hpf, cmap="viridis")
    axes[1, 1].set_title("(e) 高通滤波器 H_HP", fontsize=12); axes[1, 1].axis("off")
    axes[1, 2].imshow(high, cmap="gray")
    axes[1, 2].set_title("(f) 高通结果 (边缘)", fontsize=12); axes[1, 2].axis("off")
    save("ch04_fig02_filter_demo.png")


def ch04_fig03():
    n = 8
    basis = np.zeros((n * n, n, n))
    for u in range(n):
        for v in range(n):
            for x in range(n):
                for y in range(n):
                    basis[u * n + v, x, y] = np.cos((2 * x + 1) * u * np.pi / (2 * n)) * \
                                             np.cos((2 * y + 1) * v * np.pi / (2 * n))
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for u in range(n):
        for v in range(n):
            ax = axes[u, v]
            ax.imshow(basis[u * n + v], cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("图4-3  DCT 8×8 标准基图像", fontsize=14, y=0.92)
    save("ch04_fig03_dct_basis.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 05 — 图像增强
# ═══════════════════════════════════════════════════════════════════

def ch05_fig01():
    ti = test_image(160).astype(np.float32)
    ti = ti * 0.5 + 40
    ti += np.random.default_rng(8).normal(0, 8, ti.shape)
    ti = np.clip(ti, 0, 255).astype(np.uint8)
    stretch = np.clip((ti.astype(float) - ti.min()) / (ti.max() - ti.min()) * 255, 0, 255).astype(np.uint8)
    cdf = np.histogram(ti.ravel(), 256, [0, 256])[0].cumsum()
    cdf_n = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    he = cdf_n[ti].astype(np.uint8)
    ti_smooth = gaussian_filter(ti.astype(float), sigma=1.0)
    detail = ti.astype(float) - ti_smooth
    unsharp = np.clip(ti.astype(float) + 2.0 * detail, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    labels = [
        ("(a) 低对比原图", ti),
        ("(b) 线性拉伸", stretch),
        ("(c) 直方图均衡化", he),
        ("(d) 原图(局部)", ti[30:90, 40:100]),
        ("(e) 高斯平滑", gaussian_filter(ti, sigma=2.5).astype(np.uint8)[30:90, 40:100]),
        ("(f) 反锐化掩膜", unsharp[30:90, 40:100]),
    ]
    for idx, (title, img) in enumerate(labels):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    save("ch05_fig01_enhancement_methods.png")


def ch05_fig02():
    ti = test_image(120).astype(np.float32)
    ti_noisy = ti + np.random.default_rng(9).normal(0, 22, ti.shape)
    ti_noisy = np.clip(ti_noisy, 0, 255).astype(np.uint8)
    mean3 = convolve(ti_noisy.astype(float), np.ones((3, 3)) / 9)
    gauss = gaussian_filter(ti_noisy.astype(float), sigma=1.2)
    median = median_filter(ti_noisy, size=3)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    labels = [("(a) 加噪图像", ti_noisy), ("(b) 均值 3×3", mean3),
              ("(c) 高斯 σ=1.2", gauss), ("(d) 中值 3×3", median)]
    for ax, (title, img) in zip(axes, labels):
        ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    save("ch05_fig02_denoising.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 06 — 图像恢复
# ═══════════════════════════════════════════════════════════════════

def ch06_fig01():
    size = 140
    ti = test_image(size).astype(np.float32)
    length = 15
    angle_deg = 30
    angle = np.deg2rad(angle_deg)
    kernel = np.zeros((length * 2 + 1, length * 2 + 1))
    cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
    for i in range(-length, length + 1):
        x = int(cx + i * np.cos(angle))
        y = int(cy + i * np.sin(angle))
        if 0 <= x < kernel.shape[1] and 0 <= y < kernel.shape[0]:
            kernel[y, x] = 1
    kernel = kernel / kernel.sum()
    blurred = convolve2d(ti, kernel, mode="same")
    noise = np.random.default_rng(10).normal(0, 2, blurred.shape)
    degraded = np.clip(blurred + noise, 0, 255)
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    H = fft2(kernel, s=degraded.shape)
    G = fft2(degraded)
    k_val = 0.008
    H_conj = np.conj(H)
    Wiener = H_conj / (np.abs(H) ** 2 + k_val)
    F_hat = G * Wiener
    restored = np.real(ifft2(F_hat))
    restored = np.clip(restored, 0, 255)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes[0, 0].imshow(ti, cmap="gray")
    axes[0, 0].set_title("(a) 原始图像", fontsize=12); axes[0, 0].axis("off")
    axes[0, 1].imshow(kernel, cmap="hot")
    axes[0, 1].set_title(f"(b) 运动模糊核 (len={length}, θ={angle_deg}°)", fontsize=12); axes[0, 1].axis("off")
    axes[0, 2].imshow(degraded, cmap="gray")
    axes[0, 2].set_title("(c) 退化图像 (模糊+噪声)", fontsize=12); axes[0, 2].axis("off")
    axes[1, 0].imshow(restored, cmap="gray")
    axes[1, 0].set_title("(d) Wiener 滤波恢复", fontsize=12); axes[1, 0].axis("off")
    k2 = 0.05
    Wiener2 = H_conj / (np.abs(H) ** 2 + k2)
    over = np.real(ifft2(G * Wiener2))
    over = np.clip(over, 0, 255)
    axes[1, 1].imshow(over, cmap="gray")
    axes[1, 1].set_title("(e) K 过大 (过度平滑)", fontsize=12); axes[1, 1].axis("off")
    k3 = 0.0001
    Wiener3 = H_conj / (np.abs(H) ** 2 + k3)
    under = np.real(ifft2(G * Wiener3))
    under = np.clip(under, 0, 255)
    axes[1, 2].imshow(under, cmap="gray")
    axes[1, 2].set_title("(f) K 过小 (噪声放大)", fontsize=12); axes[1, 2].axis("off")
    save("ch06_fig01_restoration.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 07 — 图像分割
# ═══════════════════════════════════════════════════════════════════

def ch07_fig01():
    size = 150
    ti = test_image(size).astype(np.float32)
    ti = gaussian_filter(ti, sigma=1.5)
    sobel_x = convolve2d(ti, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode="same")
    sobel_y = convolve2d(ti, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode="same")
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge = np.where(mag > 50, 255, 0).astype(np.uint8)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes[0, 0].imshow(ti, cmap="gray")
    axes[0, 0].set_title("(a) 原图", fontsize=12); axes[0, 0].axis("off")
    axes[0, 1].imshow(sobel_x, cmap="RdBu", vmin=-120, vmax=120)
    axes[0, 1].set_title("(b) Sobel Gx (水平梯度)", fontsize=12); axes[0, 1].axis("off")
    axes[0, 2].imshow(sobel_y, cmap="RdBu", vmin=-120, vmax=120)
    axes[0, 2].set_title("(c) Sobel Gy (垂直梯度)", fontsize=12); axes[0, 2].axis("off")
    axes[1, 0].imshow(mag, cmap="hot")
    axes[1, 0].set_title("(d) 梯度幅值 |∇f|", fontsize=12); axes[1, 0].axis("off")
    axes[1, 1].imshow(edge, cmap="gray")
    axes[1, 1].set_title("(e) 二值化边缘 (T=50)", fontsize=12); axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    save("ch07_fig01_edge_detection.png")


def ch07_fig02():
    size = 150
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    bg = 80 + 40 * xx
    obj = 200 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)
    img = bg + obj + np.random.default_rng(11).normal(0, 8, (size, size))
    img = np.clip(img, 0, 255).astype(np.uint8)
    global_t = 130
    global_th = np.where(img > global_t, 255, 0).astype(np.uint8)
    t_otsu = 128
    otsu_th = np.where(img > t_otsu, 255, 0).astype(np.uint8)
    block_size = 35
    adaptive = np.zeros_like(img)
    pad = block_size // 2
    padded = np.pad(img, pad, mode="reflect")
    for i in range(size):
        for j in range(size):
            block = padded[i:i + block_size, j:j + block_size]
            adaptive[i, j] = 255 if img[i, j] > block.mean() - 5 else 0
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) 光照不均原图", fontsize=12); axes[0, 0].axis("off")
    axes[0, 1].imshow(global_th, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"(b) 全局阈值 T={global_t} (失败)", fontsize=12); axes[0, 1].axis("off")
    axes[1, 0].imshow(otsu_th, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("(c) Otsu 阈值 (失败)", fontsize=12); axes[1, 0].axis("off")
    axes[1, 1].imshow(adaptive, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("(d) 自适应阈值 35×35 (成功)", fontsize=12); axes[1, 1].axis("off")
    save("ch07_fig02_threshold_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 08 — 特征提取
# ═══════════════════════════════════════════════════════════════════

def ch08_fig01():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    t = np.linspace(0, 2 * np.pi, 200)
    r = 1 + 0.15 * np.sin(5 * t) + 0.06 * np.sin(9 * t)
    px, py = r * np.cos(t), r * np.sin(t)
    axes[0].fill(px, py, fc="#DDEBF7", ec="#2F5597", lw=2)
    axes[0].scatter(0, 0, c="C3", s=80, zorder=5)
    axes[0].text(0, -1.5, "质心", ha="center", fontsize=10, color="C3")
    rect_w = px.max() - px.min()
    rect_h = py.max() - py.min()
    axes[0].add_patch(plt.Rectangle((px.min(), py.min()), rect_w, rect_h,
                                      fc="none", ec="C2", lw=1.5, ls="--"))
    axes[0].set_title("(a) 几何特征: 面积/周长/质心/外接框", fontsize=12)
    axes[0].axis("equal")
    coords = np.column_stack([px, py])
    hull = coords[[0, 40, 70, 95, 130, 170]]
    axes[0].plot(np.append(hull[:, 0], hull[0, 0]),
                 np.append(hull[:, 1], hull[0, 1]), "C1--", lw=1.8, label="凸包")
    axes[0].legend(fontsize=9)
    ss = np.linspace(0, 2 * np.pi, 24)
    circle = np.column_stack([np.cos(ss), np.sin(ss)])
    axes[1].plot(circle[:, 0], circle[:, 1], "k-", lw=1.8, label="圆")
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    axes[1].plot(square[:, 0], square[:, 1], "C0-", lw=1.8, label="正方形")
    star = np.column_stack([r * np.cos(t), r * np.sin(t)])
    axes[1].plot(star[:, 0], star[:, 1], "C2-", lw=1.8, label="星形")
    axes[1].set_title("(b) 不同形状的圆度差异", fontsize=12)
    axes[1].legend(fontsize=9); axes[1].axis("equal")
    axes[1].text(-1.2, 0.6, "圆度≈1", fontsize=9, color="k")
    axes[1].text(1.3, 0.6, "圆度≈0.79", fontsize=9, color="C0")
    axes[1].text(1.3, -0.8, "圆度≈0.45", fontsize=9, color="C2")
    texture = np.random.default_rng(1).normal(size=(100, 100))
    axes[2].imshow(texture, cmap="gray")
    axes[2].set_title("(c) 随机纹理模式", fontsize=12)
    axes[2].axis("off")
    save("ch08_fig01_geometric_features.png")


def ch08_fig02():
    size = 120
    fine = np.random.default_rng(2).normal(size=(size, size))
    fine = gaussian_filter(fine, sigma=0.8)
    coarse = np.random.default_rng(3).normal(size=(size, size))
    coarse = gaussian_filter(coarse, sigma=4.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(fine, cmap="gray")
    axes[0, 0].set_title("(a) 细纹理", fontsize=12); axes[0, 0].axis("off")
    axes[0, 1].imshow(coarse, cmap="gray")
    axes[0, 1].set_title("(b) 粗纹理", fontsize=12); axes[0, 1].axis("off")
    profile_fine = fine[size // 2, :]
    profile_coarse = coarse[size // 2, :]
    axes[1, 0].plot(profile_fine, lw=1.2, color="C0")
    axes[1, 0].set_title("(c) 细纹理灰度剖面 (高频)", fontsize=12)
    axes[1, 0].set_ylim(-3, 3)
    axes[1, 1].plot(profile_coarse, lw=1.2, color="C2")
    axes[1, 1].set_title("(d) 粗纹理灰度剖面 (低频)", fontsize=12)
    axes[1, 1].set_ylim(-3, 3)
    save("ch08_fig02_texture.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 09 — 形态学
# ═══════════════════════════════════════════════════════════════════

def ch09_fig01():
    size = 100
    img = np.zeros((size, size))
    img[25:75, 25:75] = 1
    img[28:72, 28:72] = 0
    img[35:65, 35:65] = 1
    img[10, 15] = 1; img[10, 85] = 1
    img[90, 50] = 1; img[50, 90] = 1
    se = np.ones((3, 3))
    def erode(a, kernel=se):
        out = np.zeros_like(a)
        pad = kernel.shape[0] // 2
        padded = np.pad(a, pad, mode="constant")
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out[i, j] = np.all(padded[i:i + kernel.shape[0], j:j + kernel.shape[1]] >= kernel)
        return out

    def dilate(a, kernel=se):
        out = np.zeros_like(a)
        pad = kernel.shape[0] // 2
        padded = np.pad(a, pad, mode="constant")
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out[i, j] = np.any(padded[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
        return out
    eroded = erode(img)
    dilated = dilate(img)
    opened = dilate(erode(img))
    closed = erode(dilate(img))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    labels = [("(a) 原始二值掩膜", img), ("(b) 腐蚀 (变小/去噪点)", eroded),
              ("(c) 膨胀 (变大/填断缝)", dilated),
              ("", np.zeros((10, 10))),
              ("(d) 开运算 (去噪点, 不缩小主体)", opened),
              ("(e) 闭运算 (补孔洞, 连接断裂)", closed)]
    for idx, (title, data) in enumerate(labels):
        ax = axes[idx // 3, idx % 3]
        if title == "":
            ax.axis("off")
            continue
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    save("ch09_fig01_morphology_ops.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 10 — 模式识别
# ═══════════════════════════════════════════════════════════════════

def ch10_fig01():
    rng = np.random.default_rng(42)
    n = 80
    cls_a = rng.normal(loc=[2.5, 2.5], scale=[0.6, 0.6], size=(n, 2))
    cls_b = rng.normal(loc=[5.5, 5.0], scale=[0.7, 0.5], size=(n, 2))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(cls_a[:, 0], cls_a[:, 1], c="C0", alpha=0.7, edgecolors="k", s=50, label="A 类")
    axes[0].scatter(cls_b[:, 0], cls_b[:, 1], c="C3", alpha=0.7, edgecolors="k", s=50, label="B 类")
    w = np.array([-1.2, 0.8])
    b0 = 2.0
    xx = np.linspace(0, 8, 100)
    yy = -(w[0] * xx + b0) / w[1]
    axes[0].plot(xx, yy, "k-", lw=2, label="线性决策边界")
    axes[0].fill_between(xx, yy, 8, alpha=0.08, color="C0")
    axes[0].fill_between(xx, 0, yy, alpha=0.08, color="C3")
    axes[0].set_title("(a) 二维特征空间与线性分类器", fontsize=13)
    axes[0].set_xlabel("特征 x₁"); axes[0].set_ylabel("特征 x₂")
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(0, 8); axes[0].set_ylim(0, 8)
    axes[1].axis("off")
    box_data = [
        (0.5, 5.5, 2.5, 1.5, "预处理后\n图像", "#E2F0D9"),
        (3.4, 5.5, 2.5, 1.5, "特征提取\n(面积/圆度/纹理)", "#FFF2CC"),
        (6.2, 5.5, 2.5, 1.5, "分类器\n(阈值/模板/Bayes)", "#DDEBF7"),
        (3.4, 2.5, 2.5, 1.5, "输出类别\n+ 置信度", "#FCE4D6"),
    ]
    for x, y, w, h, text, color in box_data:
        axes[1].add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#2F5597", lw=1.5))
        axes[1].text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    axes[1].set_xlim(0, 9.5); axes[1].set_ylim(0, 8)
    axes[1].set_title("(b) 模式识别流水线", fontsize=13)
    save("ch10_fig01_pattern_recognition.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 11 — 图像压缩
# ═══════════════════════════════════════════════════════════════════

def ch11_fig01():
    ti = test_image(140).astype(np.float32)
    block = ti[30:38, 70:78].copy()
    dct_block = np.zeros_like(block)
    n = 8
    for u in range(n):
        for v in range(n):
            cu = np.sqrt(2 / n) if u > 0 else np.sqrt(1 / n)
            cv = np.sqrt(2 / n) if v > 0 else np.sqrt(1 / n)
            s = 0
            for x in range(n):
                for y in range(n):
                    s += (block[x, y] - 128) * np.cos((2 * x + 1) * u * np.pi / (2 * n)) * \
                         np.cos((2 * y + 1) * v * np.pi / (2 * n))
            dct_block[u, v] = cu * cv * s
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32)
    quantized = np.round(dct_block / (Q * 1.5)) * (Q * 1.5)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(block, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) 8×8 原图像块", fontsize=12)
    for i in range(8):
        for j in range(8):
            axes[0, 0].text(j, i, f"{block[i, j]:.0f}", ha="center", va="center", fontsize=6, color="r")
    axes[0, 1].imshow(dct_block, cmap="RdBu_r", vmin=-200, vmax=200)
    axes[0, 1].set_title("(b) DCT 系数", fontsize=12)
    axes[0, 2].imshow(quantized, cmap="RdBu_r", vmin=-200, vmax=200)
    axes[0, 2].set_title("(c) 量化后系数 (多数高频归零)", fontsize=12)
    sizes = [1, 2, 3, 4, 8, 16]
    nonzeros = []
    for s in sizes:
        qs = np.round(dct_block / (Q * s)) * (Q * s)
        nz = np.count_nonzero(np.abs(qs) > 0.5)
        nonzeros.append(nz)
    axes[1, 0].bar(range(len(sizes)), nonzeros, tick_label=[f"Q={s}" for s in sizes],
                   color=["#2F5597", "#4472C4", "#5B9BD5", "#9DC3E6", "#BDD7EE", "#DEEBF7"])
    axes[1, 0].set_title("(d) 量化越强，非零系数越少", fontsize=12)
    axes[1, 0].set_ylabel("非零系数数量")
    q_vals = [0.5, 1, 3, 6, 12, 25]
    for idx, qq in enumerate(q_vals):
        qtemp = np.round(dct_block / (Q * qq)) * (Q * qq)
        recon = np.zeros_like(block)
        for x in range(n):
            for y in range(n):
                s = 0
                for u in range(n):
                    for v in range(n):
                        cu = np.sqrt(2 / n) if u > 0 else np.sqrt(1 / n)
                        cv = np.sqrt(2 / n) if v > 0 else np.sqrt(1 / n)
                        s += cu * cv * qtemp[u, v] * np.cos((2 * x + 1) * u * np.pi / (2 * n)) * \
                             np.cos((2 * y + 1) * v * np.pi / (2 * n))
                recon[x, y] = s + 128
        axes[1, 1 + (idx % 2)].imshow(np.clip(recon, 0, 255), cmap="gray", vmin=0, vmax=255)
        axes[1, 1 + (idx % 2)].set_title(f"Q×{qq:.1f}" if idx < 3 else f"Q×{qq:.0f}", fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    save("ch11_fig01_jpeg_dct.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 12 — 分形
# ═══════════════════════════════════════════════════════════════════

def ch12_fig01():
    def sierpinski(ax, depth, x, y, size):
        if depth == 0:
            ax.fill([x, x + size, x + size / 2], [y, y, y + size * np.sqrt(3) / 2],
                    fc="#2F5597", ec="k", lw=0.5)
        else:
            sierpinski(ax, depth - 1, x, y, size / 2)
            sierpinski(ax, depth - 1, x + size / 2, y, size / 2)
            sierpinski(ax, depth - 1, x + size / 4, y + size * np.sqrt(3) / 4, size / 2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for idx, depth in enumerate([1, 3, 5]):
        ax = axes[idx]
        sierpinski(ax, depth, 0, 0, 1)
        ax.set_title(f"(a{'b c'[idx]}) 迭代 depth={depth}", fontsize=13)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
    save("ch12_fig01_fractal_sierpinski.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 13 — 加密
# ═══════════════════════════════════════════════════════════════════

def ch13_fig01():
    ti = test_image(120).astype(np.float32)
    size = ti.shape[0]
    def arnold(img, n_iter):
        out = img.copy()
        for _ in range(n_iter):
            new = np.zeros_like(out)
            for i in range(size):
                for j in range(size):
                    ni = (i + j) % size
                    nj = (i + 2 * j) % size
                    new[ni, nj] = out[i, j]
            out = new
        return out
    a1 = arnold(ti, 1)
    a5 = arnold(ti, 5)
    a10 = arnold(ti, 10)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    labels = [("(a) 原图", ti), ("(b) Arnold 1 次迭代", a1),
              ("(c) Arnold 5 次迭代", a5), ("(d) Arnold 10 次迭代", a10)]
    for ax, (title, img) in zip(axes.ravel(), labels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    save("ch13_fig01_arnold_scramble.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 14 — 水印
# ═══════════════════════════════════════════════════════════════════

def ch14_fig01():
    ti = test_image(140).astype(np.float32)
    bs = 8
    dct_coeffs = np.zeros_like(ti)
    for i in range(0, ti.shape[0] - bs, bs):
        for j in range(0, ti.shape[1] - bs, bs):
            block = ti[i:i + bs, j:j + bs] - 128
            dct_coeffs[i:i + bs, j:j + bs] = block
    embed = ti.copy()
    strength = 4
    embed[4::bs, 4::bs] += strength
    embed = np.clip(embed, 0, 255).astype(np.uint8)
    diff = np.abs(embed.astype(float) - ti) * 40
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(ti.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("(a) 载体图像", fontsize=13); axes[0].axis("off")
    axes[1].imshow(embed, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("(b) 嵌入水印后 (不可见)", fontsize=13); axes[1].axis("off")
    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("(c) 水印差异 (放大显示)", fontsize=13); axes[2].axis("off")
    save("ch14_fig01_watermark_embed.png")


# ═══════════════════════════════════════════════════════════════════
# Chapter 15 — 综合实践
# ═══════════════════════════════════════════════════════════════════

def ch15_fig01():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    colors = ["#DDEBF7", "#E2F0D9", "#FFF2CC", "#FCE4D6", "#EADCF8", "#C6EFCE"]
    texts = ["采集：相机/光源\n采样与校正", "预处理：降噪\n增强与校正", "分析：分割\n特征与识别",
             "决策：分类\n阈值与报警", "记录：压缩\n存档与水印", "闭环：回测\n参数更新"]
    xs = [0.3, 2.3, 4.3, 6.3, 8.3, 10.3]
    for idx in range(6):
        rect = plt.Rectangle((xs[idx], 3.0), 1.5, 1.5, fc=colors[idx], ec="#2F5597", lw=1.8, joinstyle="round")
        ax.add_patch(rect)
        ax.text(xs[idx] + 0.75, 3.75, texts[idx], ha="center", va="center", fontsize=10)
    for idx in range(5):
        ax.annotate("", xy=(xs[idx + 1], 3.75), xytext=(xs[idx] + 1.5, 3.75),
                     arrowprops=dict(arrowstyle="->", lw=2.5, color="#2F5597"))
    ax.annotate("", xy=(xs[0] + 0.75, 2.5), xytext=(xs[5] + 0.75, 3.0),
                 arrowprops=dict(arrowstyle="->", lw=2, color="C3",
                                 connectionstyle="arc3,rad=-0.3"))
    ax.text(6, 1.7, "持续迭代与验证", ha="center", fontsize=12, color="C3", fontweight="bold")
    ax.set_title("图15-1  视觉工程系统闭环", fontsize=14, pad=12)
    ax.axis("off")
    save("ch15_fig01_pipeline.png")


def main():
    ch01_fig01(); ch01_fig02(); ch01_fig03()
    ch02_fig01(); ch02_fig02(); ch02_fig03(); ch02_fig04(); ch02_fig05()
    ch03_fig01(); ch03_fig02(); ch03_fig03(); ch03_fig04()
    ch04_fig01(); ch04_fig02(); ch04_fig03()
    ch05_fig01(); ch05_fig02()
    ch06_fig01()
    ch07_fig01(); ch07_fig02()
    ch08_fig01(); ch08_fig02()
    ch09_fig01()
    ch10_fig01()
    ch11_fig01()
    ch12_fig01()
    ch13_fig01()
    ch14_fig01()
    ch15_fig01()


if __name__ == "__main__":
    main()
