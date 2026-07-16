from __future__ import annotations

import heapq
from pathlib import Path

import gmsh
import matplotlib
import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import sparse


matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
STEP_PATH = ROOT / "半圆柱压头φ12_.stp"
OUTPUT_DIR = ROOT / "热力图方法对比"
CMAP = colormaps["viridis"]


def load_step_mesh(mesh_size: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(STEP_PATH))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.65)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coordinates, dtype=float).reshape(-1, 3)
        tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}

        element_types, _, element_nodes = gmsh.model.mesh.getElements(2)
        triangle_blocks = []
        for element_type, nodes in zip(element_types, element_nodes):
            _, dimension, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if dimension != 2 or nodes_per_element < 3:
                continue
            block = np.asarray(nodes, dtype=np.int64).reshape(-1, nodes_per_element)[:, :3]
            triangle_blocks.append(
                np.asarray([[tag_to_index[int(tag)] for tag in row] for row in block], dtype=np.int32)
            )
        triangles = np.vstack(triangle_blocks)
        return vertices, triangles
    finally:
        gmsh.finalize()


def build_graph(vertices: np.ndarray, triangles: np.ndarray) -> sparse.csr_matrix:
    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.concatenate([weights, weights])
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(vertices), len(vertices)))


def nearest_vertex(vertices: np.ndarray, target: np.ndarray) -> int:
    scale = np.ptp(vertices, axis=0)
    normalized = (vertices - target) / np.maximum(scale, 1e-9)
    return int(np.argmin(np.sum(normalized * normalized, axis=1)))


def dijkstra(graph: sparse.csr_matrix, source: int) -> np.ndarray:
    distances = np.full(graph.shape[0], np.inf)
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        start, end = graph.indptr[vertex], graph.indptr[vertex + 1]
        for neighbor, weight in zip(graph.indices[start:end], graph.data[start:end]):
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, int(neighbor)))
    return distances


def normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values[np.isfinite(values)], [1, 99])
    return np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)


def create_fields(vertices: np.ndarray, triangles: np.ndarray, graph: sparse.csr_matrix):
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extent = maximum - minimum
    targets = np.asarray(
        [
            minimum + extent * [0.50, 0.92, 0.50],
            minimum + extent * [0.27, 0.63, 0.68],
            minimum + extent * [0.72, 0.48, 0.34],
        ]
    )
    source_indices = [nearest_vertex(vertices, target) for target in targets]
    amplitudes = np.asarray([1.0, 0.72, 0.52])
    sigma = np.linalg.norm(extent) * 0.095

    euclidean = np.zeros(len(vertices))
    geodesic = np.zeros(len(vertices))
    for source, amplitude in zip(source_indices, amplitudes):
        euclidean_distance = np.linalg.norm(vertices - vertices[source], axis=1)
        euclidean += amplitude * np.exp(-0.5 * (euclidean_distance / sigma) ** 2)
        surface_distance = dijkstra(graph, source)
        geodesic += amplitude * np.exp(-0.5 * (surface_distance / (sigma * 1.15)) ** 2)

    adjacency = graph.copy()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    smoothed = normalize(geodesic)
    for _ in range(18):
        neighbor_average = adjacency @ smoothed / np.maximum(degree, 1.0)
        smoothed = 0.72 * smoothed + 0.28 * neighbor_average
        smoothed[source_indices] = np.maximum(smoothed[source_indices], amplitudes)

    centroids = vertices[triangles].mean(axis=1)
    sensor_points = vertices[source_indices]
    nearest_sensor = np.argmin(
        np.linalg.norm(centroids[:, None, :] - sensor_points[None, :, :], axis=2), axis=1
    )
    face_direct = amplitudes[nearest_sensor]
    face_direct *= 0.72 + 0.28 * np.cos(np.arange(len(triangles)) * 1.618)
    face_direct = np.round(normalize(face_direct) * 5.0) / 5.0

    euclidean = normalize(euclidean)
    geodesic = normalize(geodesic)
    smoothed = normalize(smoothed)
    vertex_linear = np.round(euclidean * 8.0) / 8.0
    engineering = np.round(smoothed * 10.0) / 10.0

    fields = {
        "逐面直接赋值": face_direct,
        "顶点标量线性插值": vertex_linear[triangles].mean(axis=1),
        "欧氏距离高斯扩散": euclidean[triangles].mean(axis=1),
        "曲面测地距离扩散": geodesic[triangles].mean(axis=1),
        "测地场 + 网格扩散平滑": smoothed[triangles].mean(axis=1),
        "工程分级色带样式": engineering[triangles].mean(axis=1),
    }
    return fields, source_indices, (euclidean, geodesic, smoothed)


def configure_axis(axis, vertices: np.ndarray, title: str) -> None:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = np.max(maximum - minimum) * 0.54
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-52)
    axis.set_axis_off()
    axis.set_title(title, fontsize=12, pad=3)


def draw_mesh(axis, vertices, triangles, face_values, source_indices) -> None:
    colors = CMAP(np.clip(face_values, 0.0, 1.0))
    collection = Poly3DCollection(
        vertices[triangles],
        facecolors=colors,
        edgecolors=(0.05, 0.05, 0.05, 0.09),
        linewidths=0.08,
        antialiased=True,
    )
    axis.add_collection3d(collection)
    points = vertices[source_indices]
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], c="white", edgecolors="black", s=28, depthshade=False)


def save_figures(vertices, triangles, fields, source_indices) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    figure = plt.figure(figsize=(17, 11), dpi=180, facecolor="white")
    for index, (title, values) in enumerate(fields.items(), start=1):
        axis = figure.add_subplot(2, 3, index, projection="3d")
        draw_mesh(axis, vertices, triangles, values, source_indices)
        configure_axis(axis, vertices, title)
    figure.subplots_adjust(left=0.01, right=0.94, top=0.95, bottom=0.04, wspace=0.01, hspace=0.08)
    color_axis = figure.add_axes([0.955, 0.16, 0.014, 0.68])
    colorbar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=CMAP), cax=color_axis)
    colorbar.set_label("归一化热力值", fontsize=11)
    figure.suptitle("半圆柱压头 STEP 模型：曲面热力图方法对比", fontsize=17, y=0.985)
    figure.savefig(OUTPUT_DIR / "00_热力图方法总览.png", bbox_inches="tight")
    plt.close(figure)

    for index, (title, values) in enumerate(fields.items(), start=1):
        figure = plt.figure(figsize=(9, 7), dpi=180, facecolor="white")
        axis = figure.add_subplot(111, projection="3d")
        draw_mesh(axis, vertices, triangles, values, source_indices)
        configure_axis(axis, vertices, title)
        scalar = matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=CMAP)
        colorbar = figure.colorbar(scalar, ax=axis, fraction=0.035, pad=0.02, shrink=0.78)
        colorbar.set_label("归一化热力值")
        figure.savefig(OUTPUT_DIR / f"{index:02d}_{title}.png", bbox_inches="tight")
        plt.close(figure)


def write_report(vertices, triangles, source_indices) -> None:
    points = vertices[source_indices]
    report = f"""# 半圆柱压头曲面热力图方法对比

## 输出内容

- `00_热力图方法总览.png`：六种显示/场重建方法的统一视角对比。
- `01`～`06` 图片：每种方法的独立大图。
- `heatmap_fields.npz`：三角网格、采样点和三种连续标量场，可供后续程序直接读取。
- `generate_heatmap_comparison.py`：从 STEP 重新生成全部结果的脚本。

## 本次模型与测试设置

- 输入模型：`半圆柱压头φ12_.stp`
- STEP 三角化结果：{len(vertices)} 个顶点，{len(triangles)} 个三角面。
- 白色圆点为 3 个模拟载荷/温度采样位置。
- 采样点坐标：{np.array2string(points, precision=2, separator=', ')}
- 全部图片共用 `viridis` 色带和 0～1 归一化范围，便于直接比较。
- 本示例值是为了比较算法而构造的合成数据，不代表压头的真实受力仿真结果。

## 六种方法的观察结论

| 方法 | 视觉表现 | 主要问题 | 推荐用途 |
|---|---|---|---|
| 逐面直接赋值 | 面片边界和离散色块明显 | 对网格密度敏感，容易碎片化 | 仅用于面级分类或调试 |
| 顶点标量线性插值 | 比逐面着色连续，实时性最好 | 粗网格仍会显露三角形，重复顶点会造成接缝 | 动态实时热力图的默认方案 |
| 欧氏距离高斯扩散 | 平滑、实现简单 | 可能穿过薄壁、凹槽或相邻但不连通的表面 | 几何简单、允许近似时 |
| 曲面测地距离扩散 | 热量沿网格表面传播，不易穿模 | 需要网格邻接和最短路计算 | 任意曲面上的稀疏采样重建 |
| 测地场 + 网格扩散平滑 | 连续自然，抑制局部碎片 | 平滑过强会抹掉真实突变 | 本模型最推荐的展示方案 |
| 工程分级色带样式 | 数值区间清楚，便于读等级 | 分级边界是显示效果，不是真实不连续 | 报告、阈值告警、质量分级 |

## 推荐落地流程

1. 将传感器点或仿真点投影到三角面，保存 `triangleId + barycentricCoordinates`，不要只吸附到最近顶点。
2. 单独建立去重后的“计算网格”，避免 CAD 面边界、硬法线或 UV 接缝造成重复顶点断裂。
3. 使用曲面测地距离高斯核或离散 Laplace-Beltrami 方程，把稀疏值重建为连续顶点标量场。
4. 只平滑标量值，不平滑 RGB；渲染阶段用 1D 色带纹理完成标量到颜色的映射。
5. 动态场优先上传顶点标量；需要高频细节且模型 UV 稳定时，再烘焙到 `R16F/R32F` 热力纹理。
6. 用固定色标范围比较不同时间帧；若使用百分位裁剪，应同时显示真实最小值、最大值和裁剪规则。

## 对本模型的建议

本模型包含多个 CAD 曲面边界。生产实现中建议把“CAD 三角化后的渲染顶点”和“用于场传播的焊接计算顶点”分开维护。实时显示可采用：

`采样点投影 → 焊接网格上的测地/RBF 重建 → 少量 Laplacian 平滑 → 顶点 float 标量 → Fragment Shader 查 viridis/cividis 色带`

如果受力值来自有限元节点，则优先保留有限元网格及其形函数插值，不应先把结果离散成逐面颜色。

## 复现

Python 环境安装依赖：

```powershell
python -m pip install gmsh numpy scipy matplotlib
python .\\generate_heatmap_comparison.py
```

脚本中的模拟采样点、影响半径、相机角度和色带均可调整。静态 PNG 使用每个三角面的平均标量着色；在实际 GPU Shader 中，顶点标量会在像素级重心插值，因此“顶点标量”方案的最终画面通常会比本静态图更细腻。
"""
    (OUTPUT_DIR / "README_热力图方法对比.md").write_text(report, encoding="utf-8")


def main() -> None:
    vertices, triangles = load_step_mesh()
    graph = build_graph(vertices, triangles)
    fields, source_indices, vertex_fields = create_fields(vertices, triangles, graph)
    save_figures(vertices, triangles, fields, source_indices)
    np.savez_compressed(
        OUTPUT_DIR / "heatmap_fields.npz",
        vertices=vertices,
        triangles=triangles,
        source_indices=np.asarray(source_indices),
        euclidean=vertex_fields[0],
        geodesic=vertex_fields[1],
        geodesic_smoothed=vertex_fields[2],
    )
    write_report(vertices, triangles, source_indices)
    print(f"Generated comparison files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
