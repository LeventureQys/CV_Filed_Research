from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import xatlas
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from scipy.ndimage import gaussian_filter


matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "独立热力载体对比"
HELPER_PATH = ROOT / "generate_real_surface_cases.py"
IMAGE_SIZE = 720


def load_helper():
    specification = importlib.util.spec_from_file_location("real_cases", HELPER_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


HELPER = load_helper()


def field_colormap():
    positions = [0.00, 0.12, 0.35, 0.60, 0.80, 1.00]
    colors = np.array(
        [[140, 140, 148], [32, 76, 180], [0, 190, 220], [245, 220, 55], [245, 125, 35], [220, 35, 35]],
        dtype=float,
    ) / 255.0
    return LinearSegmentedColormap.from_list("surface_field", list(zip(positions, colors)), N=256)


CMAP = field_colormap()


def camera_projection(vertices, azimuth=-52.0, elevation=17.0):
    points = np.asarray(vertices)[:, [0, 2, 1]]
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    forward = np.array(
        [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
    )
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return np.column_stack([points @ right, points @ up, points @ forward]), forward


def projection_bounds(mesh):
    projected, _ = camera_projection(mesh.vertices)
    active = (mesh.vertices[:, 1] >= 8.0) & (np.abs(mesh.vertices[:, 0]) <= 22.0) & (np.abs(mesh.vertices[:, 2]) <= 22.0)
    local = projected[active]
    minimum = local[:, :2].min(axis=0)
    maximum = local[:, :2].max(axis=0)
    center = (minimum + maximum) * 0.5
    extent = np.max(maximum - minimum) * 0.58
    return np.array([center[0] - extent, center[0] + extent, center[1] - extent, center[1] + extent])


def rasterize(
    mesh,
    bounds,
    mode,
    vertex_values=None,
    face_values=None,
    uv=None,
    texture=None,
    procedural=None,
    image_size=IMAGE_SIZE,
):
    projected, forward = camera_projection(mesh.vertices)
    x_min, x_max, y_min, y_max = bounds
    screen = np.empty((len(projected), 2), dtype=float)
    screen[:, 0] = (projected[:, 0] - x_min) / (x_max - x_min) * (image_size - 1)
    screen[:, 1] = (1.0 - (projected[:, 1] - y_min) / (y_max - y_min)) * (image_size - 1)
    depth = projected[:, 2]
    image = np.full((image_size, image_size, 3), 0.965, dtype=float)
    depth_buffer = np.full((image_size, image_size), -np.inf)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    face_normals = np.asarray(mesh.face_normals)[:, [0, 2, 1]]
    light = np.array([-0.35, -0.25, 0.90])
    light /= np.linalg.norm(light)

    for face_index, face in enumerate(faces):
        triangle = screen[face]
        minimum = np.floor(triangle.min(axis=0)).astype(int)
        maximum = np.ceil(triangle.max(axis=0)).astype(int)
        minimum = np.maximum(minimum, 0)
        maximum = np.minimum(maximum, image_size - 1)
        if np.any(maximum < minimum):
            continue
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-9:
            continue
        x_coordinates = np.arange(minimum[0], maximum[0] + 1)
        y_coordinates = np.arange(minimum[1], maximum[1] + 1)
        xx, yy = np.meshgrid(x_coordinates, y_coordinates)
        first = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
        second = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
        third = 1.0 - first - second
        inside = (first >= -1e-7) & (second >= -1e-7) & (third >= -1e-7)
        if not np.any(inside):
            continue
        fragment_depth = first * depth[face[0]] + second * depth[face[1]] + third * depth[face[2]]
        depth_slice = depth_buffer[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1]
        visible = inside & (fragment_depth > depth_slice)
        if not np.any(visible):
            continue

        if mode == "flat":
            scalar = np.full_like(first, face_values[face_index])
        elif mode == "vertex":
            scalar = first * vertex_values[face[0]] + second * vertex_values[face[1]] + third * vertex_values[face[2]]
        elif mode == "texture":
            fragment_uv = (
                first[..., None] * uv[face[0]]
                + second[..., None] * uv[face[1]]
                + third[..., None] * uv[face[2]]
            )
            scalar = sample_texture(texture, fragment_uv)
        elif mode == "procedural":
            world = (
                first[..., None] * vertices[face[0]]
                + second[..., None] * vertices[face[1]]
                + third[..., None] * vertices[face[2]]
            )
            scalar = procedural(world)
        else:
            raise ValueError(mode)

        colors = CMAP(np.clip(scalar, 0.0, 1.0))[..., :3]
        illumination = 0.78 + 0.22 * abs(float(np.dot(face_normals[face_index], light)))
        colors *= illumination
        image_slice = image[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1]
        image_slice[visible] = colors[visible]
        depth_slice[visible] = fragment_depth[visible]
    return np.clip(image, 0.0, 1.0)


def sample_texture(texture, uv):
    height, width = texture.shape
    u = np.clip(uv[..., 0], 0.0, 1.0) * (width - 1)
    v = (1.0 - np.clip(uv[..., 1], 0.0, 1.0)) * (height - 1)
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = u - x0
    ty = v - y0
    return (
        texture[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + texture[y0, x1] * tx * (1.0 - ty)
        + texture[y1, x0] * (1.0 - tx) * ty
        + texture[y1, x1] * tx * ty
    )


def bake_texture(uv, faces, vertex_values, resolution=1024):
    texture = np.zeros((resolution, resolution), dtype=float)
    coverage = np.zeros((resolution, resolution), dtype=float)
    pixel_uv = np.column_stack([uv[:, 0] * (resolution - 1), (1.0 - uv[:, 1]) * (resolution - 1)])
    for face in faces:
        triangle = pixel_uv[face]
        minimum = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        maximum = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), resolution - 1)
        if np.any(maximum < minimum):
            continue
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-10:
            continue
        xx, yy = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1),
            np.arange(minimum[1], maximum[1] + 1),
        )
        first = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
        second = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
        third = 1.0 - first - second
        inside = (first >= -1e-7) & (second >= -1e-7) & (third >= -1e-7)
        values = first * vertex_values[face[0]] + second * vertex_values[face[1]] + third * vertex_values[face[2]]
        target = texture[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1]
        mask = coverage[minimum[1] : maximum[1] + 1, minimum[0] : maximum[0] + 1]
        target[inside] = values[inside]
        mask[inside] = 1.0
    numerator = gaussian_filter(texture * coverage, sigma=2.1)
    denominator = gaussian_filter(coverage, sigma=2.1)
    smoothed = np.divide(numerator, denominator, out=np.zeros_like(texture), where=denominator > 1e-5)
    return smoothed, coverage


def make_uv_mesh(mesh, vertex_values):
    mapping, indices, uv = xatlas.parametrize(
        np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32)
    )
    uv_mesh = mesh.__class__(vertices=np.asarray(mesh.vertices)[mapping], faces=indices, process=False)
    uv_values = np.asarray(vertex_values)[mapping]
    texture, mask = bake_texture(uv, indices, uv_values)
    return uv_mesh, np.asarray(uv), texture, mask


def normalized_wendland(distance, support):
    ratio = distance / support
    weight = np.zeros_like(ratio)
    inside = ratio < 1.0
    remainder = 1.0 - ratio[inside]
    weight[inside] = remainder**4 * (4.0 * ratio[inside] + 1.0)
    return weight


def make_decal_field(centers, values):
    projector_origin = centers.mean(axis=0)
    projector_normal = np.array([0.0, 1.0, 0.0])
    tangent_x = np.array([1.0, 0.0, 0.0])
    tangent_z = np.array([0.0, 0.0, 1.0])
    sensor_u = (centers - projector_origin) @ tangent_x
    sensor_v = (centers - projector_origin) @ tangent_z

    def evaluate(world):
        shape = world.shape[:-1]
        flattened = world.reshape(-1, 3)
        relative = flattened - projector_origin
        u = relative @ tangent_x
        v = relative @ tangent_z
        depth = np.abs(relative @ projector_normal)
        distance = np.sqrt((u[:, None] - sensor_u) ** 2 + (v[:, None] - sensor_v) ** 2)
        weights = normalized_wendland(distance, 20.0)
        weight_sum = weights.sum(axis=1)
        interpolated = np.divide(weights @ values, weight_sum, out=np.zeros_like(weight_sum), where=weight_sum > 1e-12)
        coverage = np.minimum(1.0, weight_sum) * np.clip(1.0 - depth / 18.0, 0.0, 1.0)
        return (interpolated * coverage).reshape(shape)

    return evaluate


def make_volume_field(centers, values):
    def evaluate(world):
        shape = world.shape[:-1]
        flattened = world.reshape(-1, 3)
        distance = np.linalg.norm(flattened[:, None, :] - centers[None, :, :], axis=2)
        weights = normalized_wendland(distance, 24.0)
        weight_sum = weights.sum(axis=1)
        interpolated = np.divide(weights @ values, weight_sum, out=np.zeros_like(weight_sum), where=weight_sum > 1e-12)
        return (interpolated * np.minimum(1.0, weight_sum)).reshape(shape)

    return evaluate


def save_preview(images, titles):
    figure, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=175, constrained_layout=True)
    for axis, image, title in zip(axes.flat, images, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    figure.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
        ax=axes,
        shrink=0.78,
        label="同一固定 0～1 热力色标",
    )
    figure.suptitle("同一真实模型、同一 Cell：六种热力场载体/投影方式", fontsize=17)
    figure.savefig(OUTPUT_DIR / "00_六种常见办法_总预览.png", bbox_inches="tight")
    plt.close(figure)
    filenames = [
        "01_逐面颜色.png",
        "02_原网格顶点标量.png",
        "03_独立Overlay_Mesh.png",
        "04_UV_Atlas热力纹理.png",
        "05_局部Decal_Projector.png",
        "06_世界空间3D场.png",
    ]
    for image, title, filename in zip(images, titles, filenames):
        figure, axis = plt.subplots(figsize=(8.2, 7.2), dpi=175, constrained_layout=True)
        axis.imshow(image)
        axis.set_title(title, fontsize=14)
        axis.axis("off")
        figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=axis, shrink=0.78)
        figure.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
        plt.close(figure)


def save_uv_atlas(texture, mask):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=180, constrained_layout=True)
    axes[0].imshow(mask, cmap="gray", origin="upper")
    axes[0].set_title("xatlas 自动生成的热力专用 UV 岛")
    axes[1].imshow(texture, cmap=CMAP, vmin=0, vmax=1, origin="upper")
    axes[1].set_title("1024×1024 单通道热力纹理")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=axes, shrink=0.78)
    figure.savefig(OUTPUT_DIR / "07_UV_Atlas与热力纹理.png", bbox_inches="tight")
    plt.close(figure)
    Image.fromarray(np.uint8(np.clip(CMAP(texture)[..., :3], 0, 1) * 255)).save(OUTPUT_DIR / "heat_texture_preview.png")


def write_report(mesh, fine_mesh):
    report = f"""# 任意三角模型热力图：独立载体方案对比

## 测试条件

- 模型来自真实 `simple.3dlp`，不是理想圆柱替代物。
- 使用真实 Cell 位置 C0、C7、C8，演示值固定为 `0.72、1.00、0.58`。
- 所有方案使用同一相机、同一目标区域和同一 0～1 色标。
- 原始焊接网格：{len(mesh.vertices)} 顶点、{len(mesh.faces)} 三角形。
- Overlay/UV 计算网格：{len(fine_mesh.vertices)} 顶点、{len(fine_mesh.faces)} 三角形。
- 预览由 CPU 像素级栅格器生成；顶点方案在每个像素进行重心插值，不再把顶点值平均成逐面颜色。

## 六种办法

| 方案 | 热力场存在哪里 | 预览目的 | 工程判断 |
|---|---|---|---|
| 原始面片逐面颜色 | 原始三角面 RGB/标量 | 展示碎片化基线 | 不推荐连续场 |
| 原网格顶点标量 | 原始顶点 float | 展示最轻量 GPU 方案 | 网格足够密时可用 |
| 独立 Overlay Mesh | 独立细密覆盖网格顶点 | 与原模型网格密度解耦 | 任意局部曲面推荐 |
| UV Atlas 热力纹理 | 独立 `R16F/R32F` 纹理 | 与模型几何分辨率解耦 | 静态模型/稳定 UV 推荐 |
| 局部 Decal/Projector | 二维投影纹理/解析场 | 快速交互式局部覆盖 | 视觉预览可用，注意穿透拉伸 |
| 世界空间 3D 隐式场 | `F(x,y,z)` 或 3D Texture | 完全不依赖 UV/网格 | 只适合真实体场；表面场会串到邻面 |

## 怎么选

1. **任意模型、动态选一块区域**：优先独立 Overlay Mesh。它能沿真实曲面遮挡，分辨率可独立控制，也能尊重拓扑边界。
2. **模型静态、允许预处理**：优先热力专用 UV Atlas + 单通道浮点纹理。要做 UV padding、岛边缘 dilation 和统一 texel density。
3. **只做快速交互预览**：Decal/Projector 成本最低，但必须用深度、法线和对象 ID 限制投影，不能让它穿到背面。
4. **真正的空间温度/流场**：使用 3D Texture、OpenVDB/NanoVDB 或隐式场；不要把表面压力误当体场。
5. **原模型本身已经非常细密且拓扑稳定**：顶点标量 + Fragment Shader 是性能最好的简化方案。

## 行业共识

最重要的不是“彻底摆脱三角形渲染”，而是：

`原模型只提供几何和遮挡；热力场用独立分辨率的标量载体；颜色最后由 Shader LUT 映射。`

Overlay、UV 纹理和 Decal 最终仍会由 GPU 光栅化到三角形像素，但热力图的存储、计算分辨率和原始 CAD 面片已经解耦，所以不会被原网格的三角形尺寸直接决定。

## 文件

- `00_六种常见办法_总预览.png`：主要决策图。
- `01`～`06`：每种方案独立大图。
- `07_UV_Atlas与热力纹理.png`：真实 xatlas UV 岛与 1024² 热力纹理。
- `heat_texture_preview.png`：可直接查看的热力纹理。
- `carrier_comparison_data.npz`：顶点场、UV 和纹理数据。
"""
    (OUTPUT_DIR / "README_独立热力载体对比.md").write_text(report, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    coarse_mesh, centers = HELPER.load_project()
    medium_mesh = HELPER.subdivide(coarse_mesh, 1)
    fine_mesh = HELPER.subdivide(coarse_mesh, 2)

    coarse_closest, _, _, coarse_field = HELPER.reconstruct(coarse_mesh, centers, "geodesic")
    _, _, _, fine_field = HELPER.reconstruct(fine_mesh, centers, "geodesic")
    face_field = np.round(coarse_field[coarse_mesh.faces].mean(axis=1) * 6.0) / 6.0
    bounds = projection_bounds(coarse_mesh)

    uv_mesh, uv, heat_texture, uv_mask = make_uv_mesh(fine_mesh, fine_field)
    decal = make_decal_field(centers, HELPER.ACTIVE_VALUES)
    volume = make_volume_field(centers, HELPER.ACTIVE_VALUES)

    images = [
        rasterize(coarse_mesh, bounds, "flat", face_values=face_field),
        rasterize(coarse_mesh, bounds, "vertex", vertex_values=coarse_field),
        rasterize(fine_mesh, bounds, "vertex", vertex_values=fine_field),
        rasterize(uv_mesh, bounds, "texture", uv=uv, texture=heat_texture),
        rasterize(medium_mesh, bounds, "procedural", procedural=decal),
        rasterize(medium_mesh, bounds, "procedural", procedural=volume),
    ]
    titles = [
        "逐面颜色：碎片化基线",
        "原网格顶点标量：轻量但受网格密度限制",
        "独立 Overlay Mesh：局部细密热力层",
        "UV Atlas 浮点纹理：独立纹理分辨率",
        "局部 Decal/Projector：快速投影预览",
        "世界空间 3D 场：可能穿到邻面/背面",
    ]
    save_preview(images, titles)
    save_uv_atlas(heat_texture, uv_mask)
    np.savez_compressed(
        OUTPUT_DIR / "carrier_comparison_data.npz",
        coarse_vertices=np.asarray(coarse_mesh.vertices),
        coarse_faces=np.asarray(coarse_mesh.faces),
        coarse_field=coarse_field,
        overlay_vertices=np.asarray(fine_mesh.vertices),
        overlay_faces=np.asarray(fine_mesh.faces),
        overlay_field=fine_field,
        uv_vertices=np.asarray(uv_mesh.vertices),
        uv_faces=np.asarray(uv_mesh.faces),
        uv=uv,
        heat_texture=heat_texture,
        centers=centers,
        values=HELPER.ACTIVE_VALUES,
    )
    write_report(coarse_mesh, fine_mesh)
    print(f"Generated independent heat-carrier previews in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
