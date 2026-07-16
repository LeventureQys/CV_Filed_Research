from __future__ import annotations

from pathlib import Path

import gmsh
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
STEP_PATH = ROOT / "半圆柱压头φ12_.stp"
OUTPUT_DIR = ROOT / "表面重建用例_v2"
RADIUS = 6.0
SUPPORT = 8.0
THETA_LIMIT = np.deg2rad(90.0)
Z_LIMIT = 8.0

SENSOR_Z = np.repeat(np.array([-6.0, 0.0, 6.0]), 4)
SENSOR_THETA = np.tile(np.deg2rad([-70.0, -25.0, 25.0, 70.0]), 3)
SENSOR_VALUES = np.array([0.18, 0.42, 0.55, 0.28, 0.32, 0.82, 1.00, 0.48, 0.46, 0.68, 0.76, 0.36])


def field_colormap() -> LinearSegmentedColormap:
    positions = [0.00, 0.12, 0.35, 0.60, 0.80, 1.00]
    colors = np.array(
        [[140, 140, 148], [32, 76, 180], [0, 190, 220], [245, 220, 55], [245, 125, 35], [220, 35, 35]],
        dtype=float,
    ) / 255.0
    return LinearSegmentedColormap.from_list("surface_field", list(zip(positions, colors)), N=256)


CMAP = field_colormap()


def wendland(distance: np.ndarray, support: float) -> np.ndarray:
    ratio = np.asarray(distance, dtype=float) / support
    weights = np.zeros_like(ratio)
    inside = ratio < 1.0
    remainder = 1.0 - ratio[inside]
    weights[inside] = remainder**4 * (4.0 * ratio[inside] + 1.0)
    return weights


def make_grid(theta_count: int = 241, z_count: int = 181):
    theta = np.linspace(-THETA_LIMIT, THETA_LIMIT, theta_count)
    z = np.linspace(-Z_LIMIT, Z_LIMIT, z_count)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = RADIUS * np.sin(theta_grid)
    y_grid = 40.0 - RADIUS * np.cos(theta_grid)
    return theta_grid, z_grid, x_grid, y_grid


def reconstruct_fields(theta_grid: np.ndarray, z_grid: np.ndarray, support: float = SUPPORT):
    arc_delta = RADIUS * (theta_grid[..., None] - SENSOR_THETA)
    z_delta = z_grid[..., None] - SENSOR_Z
    surface_distance = np.sqrt(arc_delta**2 + z_delta**2)

    x_grid = RADIUS * np.sin(theta_grid)
    y_grid = 40.0 - RADIUS * np.cos(theta_grid)
    sensor_x = RADIUS * np.sin(SENSOR_THETA)
    sensor_y = 40.0 - RADIUS * np.cos(SENSOR_THETA)
    euclidean_distance = np.sqrt(
        (x_grid[..., None] - sensor_x) ** 2
        + (y_grid[..., None] - sensor_y) ** 2
        + z_delta**2
    )

    nearest = SENSOR_VALUES[np.argmin(surface_distance, axis=2)]

    gaussian_sigma = support * 0.48
    gaussian_weights = np.exp(-0.5 * (surface_distance / gaussian_sigma) ** 2)
    gaussian_sum = np.sum(SENSOR_VALUES * gaussian_weights, axis=2)
    gaussian_sum = np.clip(gaussian_sum, 0.0, 1.0)

    euclidean_weights = wendland(euclidean_distance, support)
    euclidean_sum = euclidean_weights.sum(axis=2)
    euclidean_interpolated = np.divide(
        np.sum(SENSOR_VALUES * euclidean_weights, axis=2),
        euclidean_sum,
        out=np.zeros_like(euclidean_sum),
        where=euclidean_sum > 1e-12,
    )
    euclidean_effective = euclidean_interpolated * np.minimum(1.0, euclidean_sum)

    surface_weights = wendland(surface_distance, support)
    weight_sum = surface_weights.sum(axis=2)
    interpolated = np.divide(
        np.sum(SENSOR_VALUES * surface_weights, axis=2),
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 1e-12,
    )
    coverage = np.minimum(1.0, weight_sum)
    effective = interpolated * coverage
    return {
        "最近邻（块状基线）": nearest,
        "高斯直接叠加（伪峰）": gaussian_sum,
        "3D 欧氏距离 Wendland": euclidean_effective,
        "曲面 Wendland，仅 P": interpolated,
        "曲面 Wendland，P×C": effective,
        "coverage C": coverage,
    }


def draw_unwrapped(axis, field, title: str, show_points: bool = True):
    image = axis.imshow(
        field,
        origin="lower",
        extent=(-90, 90, -8, 8),
        aspect="auto",
        cmap=CMAP,
        norm=Normalize(0, 1),
        interpolation="bilinear",
    )
    if show_points:
        axis.scatter(np.rad2deg(SENSOR_THETA), SENSOR_Z, c=SENSOR_VALUES, cmap=CMAP, vmin=0, vmax=1, s=34, edgecolor="white", linewidth=0.8)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("圆周角 θ / °")
    axis.set_ylabel("轴向 z / mm")
    axis.set_xticks([-90, -45, 0, 45, 90])
    return image


def save_algorithm_comparison(theta_grid, z_grid, fields):
    panels = [
        "最近邻（块状基线）",
        "高斯直接叠加（伪峰）",
        "3D 欧氏距离 Wendland",
        "曲面 Wendland，仅 P",
        "coverage C",
        "曲面 Wendland，P×C",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=190, constrained_layout=True)
    for axis, title in zip(axes.flat, panels):
        draw_unwrapped(axis, fields[title], title)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=axes, shrink=0.83, label="固定 0～1 色标")
    figure.suptitle("同一 φ12 工作表面、同一 12 个 Cell：重建算法对比", fontsize=16)
    figure.savefig(OUTPUT_DIR / "01_算法对比_展开面.png", bbox_inches="tight")
    plt.close(figure)


def style_local_axis(axis, title: str, elev: float, azim: float):
    axis.set_xlim(-7.2, 7.2)
    axis.set_ylim(32.5, 41.0)
    axis.set_zlim(-9.0, 9.0)
    axis.set_box_aspect((14.4, 8.5, 18.0))
    axis.view_init(elev=elev, azim=azim)
    axis.set_xlabel("x / mm")
    axis.set_ylabel("y / mm")
    axis.set_zlabel("z / mm")
    axis.grid(False)
    axis.set_title(title, fontsize=11)


def draw_surface(axis, theta_grid, z_grid, field, stride=1):
    x_grid = RADIUS * np.sin(theta_grid)
    y_grid = 40.0 - RADIUS * np.cos(theta_grid)
    axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        facecolors=CMAP(np.clip(field, 0, 1)),
        rstride=stride,
        cstride=stride,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    sensor_x = RADIUS * np.sin(SENSOR_THETA)
    sensor_y = 40.0 - RADIUS * np.cos(SENSOR_THETA) - 0.08
    axis.scatter(sensor_x, sensor_y, SENSOR_Z, c=SENSOR_VALUES, cmap=CMAP, vmin=0, vmax=1, s=28, edgecolor="white", linewidth=0.8, depthshade=False)


def save_multi_view(theta_grid, z_grid, effective):
    views = [
        (7, -90, "正视工作面：热点和轴向变化"),
        (14, -52, "左斜视：观察曲率与贴面效果"),
        (14, -128, "右斜视：观察另一侧传播"),
    ]
    figure = plt.figure(figsize=(16, 6), dpi=190, constrained_layout=True)
    for index, (elev, azim, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        draw_surface(axis, theta_grid, z_grid, effective)
        style_local_axis(axis, title, elev, azim)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=figure.axes, shrink=0.70, label="effective E")
    figure.suptitle("推荐方案：曲面测地距离 + Wendland C2 + coverage", fontsize=15)
    figure.savefig(OUTPUT_DIR / "02_推荐方案_局部三视角.png", bbox_inches="tight")
    plt.close(figure)


def save_components(fields):
    panels = [
        (fields["曲面 Wendland，仅 P"], "插值 P：附近测值是多少"),
        (fields["coverage C"], "coverage C：表面被覆盖多少"),
        (fields["曲面 Wendland，P×C"], "有效值 E=P×C：最终送入色带"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=190, constrained_layout=True)
    for axis, (field, title) in zip(axes, panels):
        draw_unwrapped(axis, field, title)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=axes, shrink=0.82, label="固定 0～1 色标")
    figure.suptitle("为什么上一版不明显：只看 P 会形成大块同色，coverage 必须进入标量", fontsize=15)
    figure.savefig(OUTPUT_DIR / "03_P_C_E_可见性解释.png", bbox_inches="tight")
    plt.close(figure)


def save_radius_sweep(theta_grid, z_grid):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=190, constrained_layout=True)
    for axis, support in zip(axes, [5.0, 8.0, 12.0]):
        field = reconstruct_fields(theta_grid, z_grid, support)["曲面 Wendland，P×C"]
        draw_unwrapped(axis, field, f"R={support:g} mm")
        note = {5.0: "过小：覆盖孔洞/孤岛", 8.0: "推荐：连续且保留主峰", 12.0: "过大：热点过度混合"}[support]
        axis.text(0.5, -0.24, note, transform=axis.transAxes, ha="center", fontsize=10, color="#333333")
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=axes, shrink=0.82, label="固定 0～1 色标")
    figure.suptitle("支撑半径敏感性：同一方法仅改变 R", fontsize=15)
    figure.savefig(OUTPUT_DIR / "04_支撑半径_R_对比.png", bbox_inches="tight")
    plt.close(figure)


def save_render_carriers(effective):
    grids = [(13, 9, "粗网格逐面常色：三角块明显"), (49, 37, "细分顶点场：GPU 重心插值"), (241, 181, "参数展开纹理：高分辨率表面贴图")]
    figure = plt.figure(figsize=(16, 6), dpi=190, constrained_layout=True)
    for index, (theta_count, z_count, title) in enumerate(grids, start=1):
        theta_grid, z_grid, _, _ = make_grid(theta_count, z_count)
        field = reconstruct_fields(theta_grid, z_grid)["曲面 Wendland，P×C"]
        axis = figure.add_subplot(1, 3, index, projection="3d")
        draw_surface(axis, theta_grid, z_grid, field)
        style_local_axis(axis, title, 12, -58)
        if index == 1:
            x_grid = RADIUS * np.sin(theta_grid)
            y_grid = 40.0 - RADIUS * np.cos(theta_grid)
            axis.plot_wireframe(x_grid, y_grid, z_grid, color=(0.08, 0.08, 0.08, 0.48), linewidth=0.45)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=figure.axes, shrink=0.70, label="effective E")
    figure.suptitle("算法相同，只比较把标量场贴回曲面的显示载体", fontsize=15)
    figure.savefig(OUTPUT_DIR / "05_显示载体_粗面片_顶点_纹理.png", bbox_inches="tight")
    plt.close(figure)


def load_step_mesh(mesh_size=2.2):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(STEP_PATH))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.75)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coordinates).reshape(-1, 3)
        lookup = {int(tag): index for index, tag in enumerate(tags)}
        triangles = []
        types, _, nodes = gmsh.model.mesh.getElements(2)
        for element_type, block in zip(types, nodes):
            _, dimension, _, count, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if dimension == 2 and count >= 3:
                rows = np.asarray(block).reshape(-1, count)[:, :3]
                triangles.extend([[lookup[int(tag)] for tag in row] for row in rows])
        return vertices, np.asarray(triangles, dtype=int)
    finally:
        gmsh.finalize()


def save_model_locator(theta_grid, z_grid, effective):
    vertices, triangles = load_step_mesh()
    figure = plt.figure(figsize=(15, 7), dpi=190, constrained_layout=True)
    views = [(16, -72, "整机定位：红框区域才是本次目标表面"), (12, -38, "斜视定位：确认圆柱曲面而非底面")]
    for index, (elev, azim, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        gray = np.tile(np.array([0.76, 0.77, 0.80, 0.22]), (len(triangles), 1))
        collection = Poly3DCollection(vertices[triangles], facecolors=gray, edgecolors=(0.25, 0.27, 0.30, 0.16), linewidths=0.08)
        axis.add_collection3d(collection)
        draw_surface(axis, theta_grid[::3, ::3], z_grid[::3, ::3], effective[::3, ::3])
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        axis.set_xlim(minimum[0], maximum[0])
        axis.set_ylim(minimum[1], maximum[1])
        axis.set_zlim(minimum[2], maximum[2])
        axis.set_box_aspect(maximum - minimum)
        axis.view_init(elev=elev, azim=azim)
        axis.set_xlabel("x / mm")
        axis.set_ylabel("y / mm")
        axis.set_zlabel("z / mm")
        axis.grid(False)
        axis.set_title(title)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=figure.axes, shrink=0.74, label="目标面重建值")
    figure.suptitle("完整 STEP 中的目标工作面定位", fontsize=15)
    figure.savefig(OUTPUT_DIR / "00_整机与目标表面定位.png", bbox_inches="tight")
    plt.close(figure)


def write_report():
    report = """# φ12 工作表面重建用例 v2

## 为什么重做

上一版有两个直接问题：

1. 相机没有锁定实际工作圆柱面，整机视角主要看到了底座/底面。
2. 静态渲染把三个顶点标量先平均成一个三角面颜色，等价于重新做了一次逐面常色，没有真正表现 GPU 像素级重心插值；同时大面积低值背景压缩了连续场的可见差异。

本版根据 `3D filed rebuild document/3D曲面力场重建方法.md`，只选择 STEP 中半径 6 mm、轴线沿 z、轴线经过 `(x=0,y=40)` 的 φ12 圆柱侧面局部带：`z∈[-8,8] mm`、`θ∈[-90°,90°]`。使用文档定义的 12 个 Cell、固定 0～1 色标和 `R=8 mm`。

## 图片怎么读

- `00_整机与目标表面定位.png`：先确认重建区域在完整零件上的位置，避免把底面当作目标面。
- `01_算法对比_展开面.png`：最重要的算法公平对比；展开面能完整看到所有点和边界，不受遮挡影响。
- `02_推荐方案_局部三视角.png`：推荐方案的正视、左右斜视，确认结果确实贴在曲面上。
- `03_P_C_E_可见性解释.png`：解释“其他模式为什么看起来不明显”。只显示插值 P 时，覆盖区域容易成为大块同色；应让 coverage 进入标量得到 `E=P×C`，然后只查一次色带。
- `04_支撑半径_R_对比.png`：R 太小会形成孤岛，太大会抹平热点，8 mm 对当前点距较合适。
- `05_显示载体_粗面片_顶点_纹理.png`：重建算法完全相同，只比较逐面常色、细分顶点场、参数纹理三种表面呈现。

## 常见做法的结论

| 做法 | 结果特征 | 本项目建议 |
|---|---|---|
| 最近邻/逐面赋值 | Voronoi 色块、边界硬 | 仅用于调试和分类 |
| 高斯直接叠加 | 点密集处出现没有数据依据的伪峰 | 不采用 |
| 3D 欧氏距离归一化核 | 简单，但薄壁、背面和近邻零件可能串色 | 只可用于几何非常简单的预览 |
| 曲面 Wendland 只显示 P | 数值插值正确，但支撑边缘可能硬切，大面积同色 | 不作为最终显示标量 |
| 曲面 Wendland + coverage | 沿曲面传播、边缘自然回落、没有密度伪峰 | 推荐算法 |
| 粗网格顶点/逐面颜色 | 算法正确也会因显示采样不足而块状 | 不推荐作为最终载体 |
| 细分顶点标量 + Shader | 动态更新快，工程实现直接 | 通用实时默认方案 |
| 参数化/UV 浮点纹理 | 分辨率不受原网格限制，局部表面最细腻 | 规则圆柱或稳定 UV 的优选方案 |

## 对该 φ12 圆柱面的优先建议

这个目标面是规则可展圆柱面，优先级建议为：

1. 将 Cell 投影到目标圆柱面，并转换成二维参数坐标 `(s=Rθ,z)`。
2. 在展开域使用归一化 Wendland C2 重建 `P`、`C` 和 `E=P×C`。
3. 将 `E` 写入单通道 `R16F/R32F` 局部纹理，或写入足够细的局部覆盖网格顶点属性。
4. Fragment Shader 用同一张 256 项 LUT 着色，不再做第二次灰色/热色 RGB coverage 混合。
5. 只有跨出规则圆柱面、进入圆角或任意曲面时，才切换到焊接网格上的截断 Dijkstra/Heat Method。

这比直接依赖 CAD 原始三角面更稳定：场的计算分辨率由参数纹理或局部覆盖网格控制，不会被 STEP 三角化密度和 CAD 面边界绑死。

## 复现

```powershell
python .\\generate_surface_rebuild_cases_v2.py
python .\\generate_real_surface_cases.py
```
"""
    report = report.replace(
        "- `00_整机与目标表面定位.png`：先确认重建区域在完整零件上的位置，避免把底面当作目标面。\n",
        "- `06_实际模型_目标Cell定位.png`：使用 `.3dlp` 的真实 STL 和真实 Cell 位置，确认重建区域在压头上部。\n"
        "- `07_实际表面_推荐方案_局部三视角.png`：实际修剪曲面上的推荐结果，不使用理想圆柱覆盖实体。\n"
        "- `08_实际表面_欧氏与测地对比.png`：在真实模型上对比空间直线与曲面路径。\n"
        "- `09_实际表面_原始与细分网格.png`：说明即使算法相同，显示网格过粗仍会产生面片感。\n",
    )
    report = report.replace(
        "本版根据 `3D filed rebuild document/3D曲面力场重建方法.md`，只选择 STEP 中半径 6 mm、轴线沿 z、轴线经过 `(x=0,y=40)` 的 φ12 圆柱侧面局部带：`z∈[-8,8] mm`、`θ∈[-90°,90°]`。使用文档定义的 12 个 Cell、固定 0～1 色标和 `R=8 mm`。",
        "本版分成两类用例。`01`～`05` 根据文档定义的分析性 φ12 圆柱带、12 个 Cell 和 `R=8 mm`，用于无遮挡地比较算法与显示载体；该规则圆柱带是理论演示域，不等于 STEP 上完整存在的 CAD 修剪面。`06`～`09` 则读取 `.3dlp` 中的真实 STL 和真实 Cell 附着位置，只在实际三角曲面上做局部重建。",
    )
    report += """

## 实际模型用例

`06`～`09` 使用 `simple.3dlp` 中的真实焊接 STL。为突出“选取部分位置重建”的业务方式，仅启用压头上部的 C0、C7、C8 三个真实 Cell 位置，演示值分别为 `0.72、1.00、0.58`，支撑半径为 `24 mm`。原始网格为 814 顶点/1656 三角形；显示对比中再细分两级，使静态图不被粗三角面主导。

实际生产中不应按 `y>某阈值` 自动选择目标面，而应保存 Cell 的 `triangleId + barycentricCoordinates`，再由这些附着面、允许传播边界和连通分量定义目标区域。
"""
    (OUTPUT_DIR / "README_表面重建用例_v2.md").write_text(report, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    theta_grid, z_grid, _, _ = make_grid()
    fields = reconstruct_fields(theta_grid, z_grid)
    save_algorithm_comparison(theta_grid, z_grid, fields)
    save_multi_view(theta_grid, z_grid, fields["曲面 Wendland，P×C"])
    save_components(fields)
    save_radius_sweep(theta_grid, z_grid)
    save_render_carriers(fields["曲面 Wendland，P×C"])
    np.savez_compressed(
        OUTPUT_DIR / "surface_rebuild_fields_v2.npz",
        theta=theta_grid,
        z=z_grid,
        sensor_theta=SENSOR_THETA,
        sensor_z=SENSOR_Z,
        sensor_values=SENSOR_VALUES,
        nearest=fields["最近邻（块状基线）"],
        gaussian_sum=fields["高斯直接叠加（伪峰）"],
        euclidean_wendland=fields["3D 欧氏距离 Wendland"],
        surface_interpolated=fields["曲面 Wendland，仅 P"],
        coverage=fields["coverage C"],
        effective=fields["曲面 Wendland，P×C"],
    )
    write_report()
    print(f"Generated v2 surface reconstruction cases in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
