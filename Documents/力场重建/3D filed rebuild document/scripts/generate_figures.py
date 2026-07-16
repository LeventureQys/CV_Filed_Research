import csv
import heapq
import io
import json
import os
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


BASE_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"
SUPPORT_MM = 30.0
ROI_MARGIN_MM = 8.0
FOLD_LAMBDA = 2.0
MIN_VALUE = 0.0
MAX_VALUE = 1.0
ACTIVE_CELL_INDICES = np.array([0, 1, 7, 8], dtype=int)


def find_project_path():
    candidates = [
        Path(os.environ.get("SIMPLE_3DLP_PATH", "")),
        Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\mvp_modules\simple.3dlp"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 simple.3dlp；可通过 SIMPLE_3DLP_PATH 指定文件位置")


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def field_colormap():
    positions = [0.00, 0.12, 0.35, 0.60, 0.80, 1.00]
    colors = np.array(
        [[140, 140, 148], [32, 76, 180], [0, 190, 220], [245, 220, 55], [245, 125, 35], [220, 35, 35]],
        dtype=float,
    ) / 255.0
    return LinearSegmentedColormap.from_list("full_surface_field", list(zip(positions, colors)), N=256)


def load_project(project_path=None):
    project_path = project_path or find_project_path()
    with zipfile.ZipFile(project_path) as archive:
        project = json.loads(archive.read("project.json").decode("utf-8"))
        model_name = project["model_file"]
        mesh = trimesh.load(io.BytesIO(archive.read(model_name)), file_type="stl", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("项目中的模型不是单一三角网格")
    cells = project["cells"]
    centers = np.array(
        [[cell["center_3d"][axis] for axis in ("x", "y", "z")] for cell in cells], dtype=float
    )
    normals = np.array(
        [[cell["normal"][axis] for axis in ("x", "y", "z")] for cell in cells], dtype=float
    )
    values = synthetic_values(centers)
    return project, mesh, centers, normals, values


def synthetic_values(centers):
    scale = np.array([28.0, 18.0, 28.0])
    hotspot_a = np.exp(-np.sum(((centers - np.array([8.0, 18.0, 8.0])) / scale) ** 2, axis=1))
    hotspot_b = np.exp(-np.sum(((centers - np.array([-22.0, 8.0, -18.0])) / np.array([20.0, 14.0, 20.0])) ** 2, axis=1))
    head = np.clip((centers[:, 1] - 20.0) / 26.0, 0.0, 1.0)
    values = 0.12 + 0.58 * hotspot_a + 0.23 * hotspot_b + 0.17 * head
    return np.clip(values, 0.0, 1.0)


def wendland_c2(distance, support=SUPPORT_MM):
    normalized = np.asarray(distance, dtype=float) / support
    weights = np.zeros_like(normalized)
    inside = normalized < 1.0
    remaining = 1.0 - normalized[inside]
    weights[inside] = remaining**4 * (4.0 * normalized[inside] + 1.0)
    return weights


def build_surface_graph(mesh, fold_lambda=FOLD_LAMBDA):
    vertices = np.asarray(mesh.vertices)
    face_normals = np.asarray(mesh.face_normals)
    edge_faces = {}
    for face_index, face in enumerate(mesh.faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            edge_faces.setdefault(edge, []).append(face_index)
    adjacency = [[] for _ in range(len(vertices))]
    for (first, second), faces in edge_faces.items():
        length = float(np.linalg.norm(vertices[first] - vertices[second]))
        penalty = 1.0
        if len(faces) >= 2:
            dot = abs(float(np.dot(face_normals[faces[0]], face_normals[faces[1]])))
            angle = np.arccos(np.clip(dot, 0.0, 1.0))
            penalty += fold_lambda * (angle / np.pi) ** 2
        cost = length * penalty
        adjacency[first].append((second, cost))
        adjacency[second].append((first, cost))
    return adjacency


def attach_cells(mesh, centers):
    closest, distances, face_ids = trimesh.proximity.closest_point_naive(mesh, centers)
    faces = mesh.faces[face_ids]
    vertices = np.asarray(mesh.vertices)
    seeds = []
    for point, face in zip(closest, faces):
        seeds.append(int(face[np.argmin(np.linalg.norm(vertices[face] - point, axis=1))]))
    return closest, distances, np.asarray(face_ids), np.asarray(seeds)


def dijkstra(adjacency, source, cutoff=None, return_predecessor=False):
    distances = np.full(len(adjacency), np.inf)
    predecessors = np.full(len(adjacency), -1, dtype=int)
    distances[source] = 0.0
    queue = [(0.0, int(source))]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        if cutoff is not None and distance > cutoff:
            continue
        for neighbor, cost in adjacency[vertex]:
            candidate = distance + cost
            if candidate < distances[neighbor] and (cutoff is None or candidate <= cutoff):
                distances[neighbor] = candidate
                predecessors[neighbor] = vertex
                heapq.heappush(queue, (candidate, neighbor))
    return (distances, predecessors) if return_predecessor else distances


def dijkstra_from_attachment(adjacency, vertices, face, point, cutoff=None):
    distances = np.full(len(adjacency), np.inf)
    queue = []
    for vertex in face:
        initial = float(np.linalg.norm(vertices[vertex] - point))
        if cutoff is None or initial <= cutoff:
            distances[vertex] = min(distances[vertex], initial)
            heapq.heappush(queue, (initial, int(vertex)))
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        if cutoff is not None and distance >= cutoff:
            continue
        for neighbor, cost in adjacency[vertex]:
            candidate = distance + cost
            if candidate < distances[neighbor] and (cutoff is None or candidate < cutoff):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def build_overlay(mesh, closest, face_ids, support=SUPPORT_MM, margin=ROI_MARGIN_MM, subdivisions=2):
    vertices = np.asarray(mesh.vertices)
    adjacency = build_surface_graph(mesh)
    radius = support + margin
    multi_source_distance = np.full(len(vertices), np.inf)
    for point, face_id in zip(closest, face_ids):
        distances = dijkstra_from_attachment(
            adjacency,
            vertices,
            mesh.faces[face_id],
            point,
            cutoff=radius,
        )
        multi_source_distance = np.minimum(multi_source_distance, distances)
    roi_faces = np.any(multi_source_distance[mesh.faces] <= radius, axis=1)
    roi_mesh = mesh.submesh([np.flatnonzero(roi_faces)], append=True, repair=False)
    overlay = roi_mesh.copy()
    for _ in range(subdivisions):
        overlay = overlay.subdivide()
    overlay.remove_unreferenced_vertices()
    return roi_mesh, overlay, multi_source_distance, roi_faces


def reconstruct(mesh, centers, values, support=SUPPORT_MM):
    adjacency = build_surface_graph(mesh)
    closest, attachment_error, face_ids, seeds = attach_cells(mesh, centers)
    distances = np.vstack([dijkstra(adjacency, seed, cutoff=support) for seed in seeds])
    weights = wendland_c2(distances, support=support)
    weight_sum = weights.sum(axis=0)
    interpolated = np.divide(
        values @ weights,
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 1e-12,
    )
    coverage = np.minimum(1.0, weight_sum)
    effective = interpolated * coverage
    return {
        "adjacency": adjacency,
        "closest": closest,
        "attachment_error": attachment_error,
        "face_ids": face_ids,
        "seeds": seeds,
        "distances": distances,
        "weights": weights,
        "weight_sum": weight_sum,
        "interpolated": interpolated,
        "coverage": coverage,
        "effective": effective,
    }


def reconstruct_overlay(overlay, sample_points, values, support=SUPPORT_MM, top_k=8):
    adjacency = build_surface_graph(overlay)
    vertices = np.asarray(overlay.vertices)
    closest, attachment_error, face_ids, _ = attach_cells(overlay, sample_points)
    distances = np.vstack(
        [
            dijkstra_from_attachment(adjacency, vertices, overlay.faces[face_id], point, cutoff=support)
            for point, face_id in zip(closest, face_ids)
        ]
    )
    weights = wendland_c2(distances, support=support)
    weight_sum_all = weights.sum(axis=0)
    coverage = np.minimum(1.0, weight_sum_all)
    normalized = np.zeros_like(weights)
    retained_counts = np.zeros(len(vertices), dtype=int)
    for vertex in range(len(vertices)):
        nonzero = np.flatnonzero(weights[:, vertex] > 0.0)
        if nonzero.size == 0:
            continue
        if nonzero.size > top_k:
            order = np.argsort(weights[nonzero, vertex])[-top_k:]
            nonzero = nonzero[order]
        retained = weights[nonzero, vertex]
        normalized[nonzero, vertex] = retained / retained.sum()
        retained_counts[vertex] = len(nonzero)
    interpolated = values @ normalized
    effective = interpolated * coverage
    return {
        "adjacency": adjacency,
        "closest": closest,
        "attachment_error": attachment_error,
        "face_ids": face_ids,
        "distances": distances,
        "weights": weights,
        "weight_sum": weight_sum_all,
        "normalized_weights": normalized,
        "retained_counts": retained_counts,
        "interpolated": interpolated,
        "coverage": coverage,
        "effective": effective,
    }


def style_3d(axis, mesh, elev=24, azim=-52):
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    axis.set_box_aspect((extents[0], extents[2], extents[1]))
    axis.set_xlabel("x / mm")
    axis.set_ylabel("z / mm")
    axis.set_zlabel("y / mm")
    axis.view_init(elev=elev, azim=azim)
    axis.grid(False)
    axis.set_facecolor("#f7f8fa")


def draw_mesh(axis, mesh, vertex_values=None, cmap=None, alpha=1.0, edge=False):
    vertices = np.asarray(mesh.vertices)
    display_vertices = vertices[:, [0, 2, 1]]
    triangles = display_vertices[mesh.faces]
    if vertex_values is None:
        face_colors = np.tile(np.array([0.72, 0.73, 0.76, alpha]), (len(mesh.faces), 1))
    else:
        face_colors = cmap(np.asarray(vertex_values)[mesh.faces].mean(axis=1))
        face_colors[:, 3] = alpha
    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors="#5f6570" if edge else "none",
        linewidths=0.12 if edge else 0.0,
    )
    axis.add_collection3d(collection)
    bounds = mesh.bounds
    axis.set_xlim(bounds[:, 0])
    axis.set_ylim(bounds[:, 2])
    axis.set_zlim(bounds[:, 1])
    return collection


def set_mesh_bounds(axis, mesh):
    bounds = mesh.bounds
    axis.set_xlim(bounds[:, 0])
    axis.set_ylim(bounds[:, 2])
    axis.set_zlim(bounds[:, 1])


def draw_cells(axis, closest, values, cmap, size=34, annotate=False):
    points = closest[:, [0, 2, 1]]
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=values, cmap=cmap, vmin=0, vmax=1,
        s=size, edgecolor="white", linewidth=0.8, depthshade=False,
    )
    if annotate:
        for index, point in enumerate(points):
            axis.text(*point, f"C{index}", fontsize=5.5, color="#202124")


def save_model_definition(mesh):
    figure = plt.figure(figsize=(13.0, 6.0), constrained_layout=True)
    for index, (elev, azim, title) in enumerate(
        [(24, -52, "完整模型：底座、柱体、圆头与过渡面"), (18, 132, "完整模型背侧与底面结构")], start=1
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        draw_mesh(axis, mesh, edge=True)
        style_3d(axis, mesh, elev=elev, azim=azim)
        axis.set_title(title)
    figure.suptitle(f"全模型三角曲面：{len(mesh.vertices)} 个焊接顶点，{len(mesh.faces)} 个三角形", fontsize=14)
    figure.savefig(FIGURE_DIR / "01_model_surface_definition.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_sensor_layout(mesh, closest, values, cmap, active_indices=ACTIVE_CELL_INDICES):
    figure = plt.figure(figsize=(11.2, 6.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    draw_mesh(axis, mesh, alpha=0.94)
    all_points = closest[:, [0, 2, 1]]
    axis.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], color="#b9bec8", s=18, depthshade=False)
    draw_cells(axis, closest[active_indices], values[active_indices], cmap, size=64, annotate=False)
    for cell_index in active_indices:
        point = closest[cell_index][[0, 2, 1]]
        axis.text(*point, f"C{cell_index}", fontsize=7, color="#202124")
    style_3d(axis, mesh)
    axis.set_title("31 个 Cell 的完整布局；彩色标记为本次 4 个活动 Cell", pad=12)
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=axis, shrink=0.72, pad=0.07)
    colorbar.set_label("演示用归一化标量")
    figure.savefig(FIGURE_DIR / "02_sensor_layout.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_overlay_construction(mesh, roi_mesh, overlay, closest, values, cmap):
    figure = plt.figure(figsize=(15.8, 5.4), constrained_layout=True)
    panels = [
        ("原始 Render Mesh", mesh, False),
        ("沿 Cell 曲面扩展得到 ROI", roi_mesh, True),
        ("ROI 细分后的独立 Overlay", overlay, True),
    ]
    for index, (title, panel_mesh, show_cells) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        if index > 1:
            draw_mesh(axis, mesh, alpha=0.13)
        draw_mesh(axis, panel_mesh, alpha=0.92, edge=index == 3)
        if show_cells:
            draw_cells(axis, closest, values, cmap, size=20)
        set_mesh_bounds(axis, mesh)
        style_3d(axis, mesh)
        axis.set_title(title)
    figure.suptitle(
        f"Overlay 构建：原模型 {len(mesh.faces)} 面 → ROI {len(roi_mesh.faces)} 面 → Overlay {len(overlay.faces)} 面",
        fontsize=14,
    )
    figure.savefig(FIGURE_DIR / "03_overlay_mesh_construction.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_kernel_curve():
    normalized = np.linspace(0.0, 1.2, 600)
    weights = wendland_c2(normalized, support=1.0)
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.plot(normalized, weights, color="#2457a6", linewidth=3)
    samples = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sample_weights = wendland_c2(samples, support=1.0)
    axis.scatter(samples, sample_weights, color="#d94841", zorder=4)
    for sample, weight in zip(samples, sample_weights):
        axis.annotate(f"{weight:.3f}", (sample, weight), xytext=(4, 7), textcoords="offset points")
    axis.axvline(1.0, color="#666666", linestyle="--", linewidth=1.4)
    axis.set(xlim=(0, 1.2), ylim=(-0.03, 1.05), xlabel="归一化曲面路径 r=d_surface/R", ylabel="权重 φ(r)")
    axis.set_title("Wendland C2 紧支撑核：支撑边界处平滑降为 0")
    axis.grid(alpha=0.24)
    figure.savefig(FIGURE_DIR / "05_wendland_kernel.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def select_distance_pair(mesh, adjacency, seeds):
    vertices = np.asarray(mesh.vertices)
    best = None
    for source in np.unique(seeds):
        distances = dijkstra(adjacency, int(source))
        euclidean = np.linalg.norm(vertices - vertices[source], axis=1)
        candidates = np.where((euclidean > 4.0) & np.isfinite(distances))[0]
        if candidates.size == 0:
            continue
        scores = distances[candidates] - euclidean[candidates]
        target = int(candidates[np.argmax(scores)])
        candidate = (float(scores.max()), int(source), target)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1], best[2]


def restore_path(predecessors, source, target):
    path = [target]
    while path[-1] != source and predecessors[path[-1]] >= 0:
        path.append(int(predecessors[path[-1]]))
    return np.asarray(path[::-1], dtype=int)


def save_distance_comparison(mesh, result):
    source, target = select_distance_pair(mesh, result["adjacency"], result["seeds"])
    distances, predecessors = dijkstra(result["adjacency"], source, return_predecessor=True)
    path = restore_path(predecessors, source, target)
    vertices = np.asarray(mesh.vertices)
    euclidean = float(np.linalg.norm(vertices[target] - vertices[source]))
    surface = float(distances[target])
    compare_support = max(SUPPORT_MM, surface * 1.08)
    weights = [wendland_c2(euclidean, compare_support), wendland_c2(surface, compare_support)]
    figure = plt.figure(figsize=(13.2, 5.7), constrained_layout=True)
    axis_3d = figure.add_subplot(121, projection="3d")
    draw_mesh(axis_3d, mesh, alpha=0.62)
    display_path = vertices[path][:, [0, 2, 1]]
    endpoints = vertices[[source, target]][:, [0, 2, 1]]
    axis_3d.plot(display_path[:, 0], display_path[:, 1], display_path[:, 2], color="#d94841", linewidth=3, label="网格曲面路径")
    axis_3d.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], color="#2457a6", linestyle="--", linewidth=2, label="空间直线")
    axis_3d.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], color="#202124", s=34)
    style_3d(axis_3d, mesh)
    axis_3d.set_title("完整模型上的同一对表面点")
    axis_3d.legend(loc="upper left", fontsize=8)
    axis = figure.add_subplot(122)
    x = np.arange(2)
    bars = axis.bar(x - 0.18, [euclidean, surface], width=0.36, color=["#4c78a8", "#e45756"], label="距离 / mm")
    axis.set_xticks(x, ["欧氏直线", "曲面路径"])
    axis.set_ylabel("距离 / mm")
    axis.grid(axis="y", alpha=0.22)
    twin = axis.twinx()
    twin.plot(x, weights, color="#2a9d8f", marker="o", linewidth=2.4, label="对应 Wendland 权重")
    twin.set_ylim(0, 1.05)
    twin.set_ylabel("权重")
    for bar in bars:
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}", ha="center", va="bottom")
    axis.set_title("空间近不等于沿完整表面近")
    figure.suptitle("完整模型上的欧氏距离与曲面路径比较", fontsize=14)
    figure.savefig(FIGURE_DIR / "04_distance_and_weight_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_overlay_components(mesh, overlay, closest, values, result, cmap):
    panels = [
        (result["interpolated"], "归一化插值 P"),
        (result["coverage"], "覆盖置信度 C"),
        (result["effective"], "显示标量 E=P×C"),
    ]
    figure = plt.figure(figsize=(16.0, 5.2), constrained_layout=True)
    for index, (field, title) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        draw_mesh(axis, mesh, alpha=0.12)
        draw_mesh(axis, overlay, field, cmap)
        draw_cells(axis, closest, values, cmap, size=18)
        style_3d(axis, mesh)
        axis.set_title(title)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=figure.axes, shrink=0.76, label="归一化值")
    figure.suptitle("独立 Overlay 上的归一化插值、coverage 与最终显示标量", fontsize=14)
    figure.savefig(FIGURE_DIR / "06_overlay_components.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_surface_reconstruction(mesh, overlay, closest, values, result, cmap):
    figure = plt.figure(figsize=(13.6, 6.0), constrained_layout=True)
    for index, (elev, azim, title) in enumerate([(24, -52, "主视角"), (20, 132, "背侧视角")], start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        draw_mesh(axis, mesh, alpha=1.0)
        draw_mesh(axis, overlay, result["effective"], cmap)
        draw_cells(axis, closest, values, cmap, size=25)
        style_3d(axis, mesh, elev=elev, azim=azim)
        axis.set_title(title)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=figure.axes, shrink=0.76, label="effective scalar")
    figure.suptitle(f"独立 Overlay 曲面力场重建（R={SUPPORT_MM:g} mm，Top-K=8）", fontsize=14)
    figure.savefig(FIGURE_DIR / "07_overlay_surface_reconstruction.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_workflow():
    build_labels = ["Render Mesh\n焊接拓扑 + BVH", "Cell 曲面附着", "曲面扩展 ROI", "生成独立 Overlay", "截断 Dijkstra", "Top-K Influence Cache"]
    frame_labels = ["读取 Sample value", "稀疏加权得到 E", "上传单通道 Scalar", "Shader 插值 + LUT"]
    figure, axis = plt.subplots(figsize=(14.6, 5.0), constrained_layout=True)
    axis.set_xlim(0, 14.6)
    axis.set_ylim(0, 5.0)
    axis.axis("off")
    axis.text(0.2, 4.55, "低频构建阶段：几何或配置变化时执行", fontsize=12.5, weight="bold", color="#2f4b7c")
    axis.text(0.2, 2.15, "高频实时阶段：采样值变化时循环执行", fontsize=12.5, weight="bold", color="#9c4f12")
    for index, label in enumerate(build_labels):
        x = index * 2.35 + 0.2
        box = FancyBboxPatch((x, 3.25), 1.95, 0.88, boxstyle="round,pad=0.08,rounding_size=0.08", linewidth=1.4, edgecolor="#3b4a6b", facecolor="#eef3fb")
        axis.add_patch(box)
        axis.text(x + 0.975, 3.69, label, ha="center", va="center", fontsize=9.2)
        if index < len(build_labels) - 1:
            axis.add_patch(FancyArrowPatch((x + 1.97, 3.69), (x + 2.28, 3.69), arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color="#677083"))
    for index, label in enumerate(frame_labels):
        x = index * 3.25 + 0.7
        box = FancyBboxPatch((x, 0.78), 2.45, 0.82, boxstyle="round,pad=0.08,rounding_size=0.08", linewidth=1.4, edgecolor="#a45a18", facecolor="#fff0df")
        axis.add_patch(box)
        axis.text(x + 1.225, 1.19, label, ha="center", va="center", fontsize=9.6)
        if index < len(frame_labels) - 1:
            axis.add_patch(FancyArrowPatch((x + 2.48, 1.19), (x + 3.18, 1.19), arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color="#a06a3c"))
    axis.add_patch(FancyArrowPatch((13.55, 0.77), (0.7, 0.77), connectionstyle="arc3,rad=-0.16", arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#a06a3c"))
    axis.add_patch(FancyArrowPatch((13.0, 3.2), (13.0, 1.66), arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color="#677083"))
    axis.text(13.15, 2.43, "缓存就绪", fontsize=9, color="#4f596b", va="center")
    axis.set_title("独立 Overlay 曲面力场重建：构建阶段与实时阶段分离", fontsize=14, pad=8)
    figure.savefig(FIGURE_DIR / "08_overlay_algorithm_workflow.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_sensor_data(centers, normals, values, closest, attachment_error):
    path = DATA_DIR / "sensor_data.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "surface_x", "surface_y", "surface_z", "attachment_error_mm", "normalized_value"])
        for index in range(len(centers)):
            writer.writerow([f"C{index}", *centers[index], *normals[index], *closest[index], attachment_error[index], values[index]])


def write_verification_data(project, mesh, centers, result, roi_mesh, overlay):
    covered = result["weight_sum"] > 1e-12
    metrics = {
        "model": project["model_file"],
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "connected_components": int(len(mesh.split(only_watertight=False))),
        "bounds_mm": mesh.bounds.tolist(),
        "cell_count": int(len(centers)),
        "support_radius_mm": SUPPORT_MM,
        "roi_margin_mm": ROI_MARGIN_MM,
        "roi_triangles": int(len(roi_mesh.faces)),
        "overlay_vertices": int(len(overlay.vertices)),
        "overlay_triangles": int(len(overlay.faces)),
        "max_attachment_error_mm": float(result["attachment_error"].max()),
        "covered_vertex_count": int(covered.sum()),
        "interpolated_min_on_covered": float(result["interpolated"][covered].min()),
        "interpolated_max_on_covered": float(result["interpolated"][covered].max()),
        "coverage_min": float(result["coverage"].min()),
        "coverage_max": float(result["coverage"].max()),
        "effective_min": float(result["effective"].min()),
        "effective_max": float(result["effective"].max()),
        "all_values_finite": bool(all(np.isfinite(result[key]).all() for key in ("interpolated", "coverage", "effective"))),
    }
    (DATA_DIR / "verification_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    project, mesh, centers, normals, values = load_project()
    base_result = reconstruct(mesh, centers, values)
    active_points = base_result["closest"][ACTIVE_CELL_INDICES]
    active_values = values[ACTIVE_CELL_INDICES]
    active_face_ids = base_result["face_ids"][ACTIVE_CELL_INDICES]
    roi_mesh, overlay, _, _ = build_overlay(
        mesh,
        active_points,
        active_face_ids,
    )
    result = reconstruct_overlay(overlay, active_points, active_values)
    cmap = field_colormap()
    save_model_definition(mesh)
    save_sensor_layout(mesh, base_result["closest"], values, cmap)
    save_overlay_construction(mesh, roi_mesh, overlay, active_points, active_values, cmap)
    save_distance_comparison(mesh, base_result)
    save_kernel_curve()
    save_overlay_components(mesh, overlay, result["closest"], active_values, result, cmap)
    save_surface_reconstruction(mesh, overlay, result["closest"], active_values, result, cmap)
    save_workflow()
    write_sensor_data(centers, normals, values, base_result["closest"], base_result["attachment_error"])
    write_verification_data(project, mesh, centers, result, roi_mesh, overlay)


if __name__ == "__main__":
    main()
