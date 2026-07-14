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
FOLD_LAMBDA = 2.0
MIN_VALUE = 0.0
MAX_VALUE = 1.0


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


def save_sensor_layout(mesh, closest, values, cmap):
    figure = plt.figure(figsize=(11.2, 6.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    draw_mesh(axis, mesh, alpha=0.94)
    draw_cells(axis, closest, values, cmap, size=52, annotate=True)
    style_3d(axis, mesh)
    axis.set_title("完整模型上的 31 个真实 Cell 位置（颜色为演示标量）", pad=12)
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=axis, shrink=0.72, pad=0.07)
    colorbar.set_label("演示用归一化标量")
    figure.savefig(FIGURE_DIR / "02_sensor_layout.png", dpi=220, bbox_inches="tight")
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
    figure.savefig(FIGURE_DIR / "03_wendland_kernel.png", dpi=220, bbox_inches="tight")
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


def save_full_model_components(mesh, closest, values, result, cmap):
    panels = [
        (result["interpolated"], "归一化插值 P"),
        (result["coverage"], "覆盖置信度 C"),
        (result["effective"], "显示标量 E=P×C"),
    ]
    figure = plt.figure(figsize=(16.0, 5.2), constrained_layout=True)
    for index, (field, title) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        draw_mesh(axis, mesh, field, cmap)
        draw_cells(axis, closest, values, cmap, size=18)
        style_3d(axis, mesh)
        axis.set_title(title)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=figure.axes, shrink=0.76, label="归一化值")
    figure.suptitle("完整模型表面的插值、coverage 与最终显示标量", fontsize=14)
    figure.savefig(FIGURE_DIR / "05_unwrapped_components.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_surface_reconstruction(mesh, closest, values, result, cmap):
    figure = plt.figure(figsize=(13.6, 6.0), constrained_layout=True)
    for index, (elev, azim, title) in enumerate([(24, -52, "主视角"), (20, 132, "背侧视角")], start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        draw_mesh(axis, mesh, result["effective"], cmap)
        draw_cells(axis, closest, values, cmap, size=25)
        style_3d(axis, mesh, elev=elev, azim=azim)
        axis.set_title(title)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=figure.axes, shrink=0.76, label="effective scalar")
    figure.suptitle(f"完整模型曲面路径 Wendland C2 重建（R={SUPPORT_MM:g} mm）", fontsize=14)
    figure.savefig(FIGURE_DIR / "06_surface_reconstruction.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_workflow():
    labels = ["完整模型三角化", "Cell 附着到表面", "全模型截断最短路", "Wendland C2 权重", "插值 P 与 coverage C", "全模型统一着色"]
    figure, axis = plt.subplots(figsize=(14.2, 2.8), constrained_layout=True)
    axis.set_xlim(0, len(labels) * 2.2)
    axis.set_ylim(0, 2.2)
    axis.axis("off")
    for index, label in enumerate(labels):
        x = index * 2.2 + 0.15
        box = FancyBboxPatch((x, 0.7), 1.75, 0.78, boxstyle="round,pad=0.08,rounding_size=0.08", linewidth=1.4, edgecolor="#3b4a6b", facecolor="#eef3fb" if index < 4 else "#fff0df")
        axis.add_patch(box)
        axis.text(x + 0.875, 1.09, label, ha="center", va="center", fontsize=9.5)
        if index < len(labels) - 1:
            axis.add_patch(FancyArrowPatch((x + 1.78, 1.09), (x + 2.13, 1.09), arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color="#677083"))
    axis.set_title("完整模型 3D 曲面场重建流程", fontsize=14, pad=10)
    figure.savefig(FIGURE_DIR / "07_algorithm_workflow.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_sensor_data(centers, normals, values, closest, attachment_error):
    path = DATA_DIR / "sensor_data.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "surface_x", "surface_y", "surface_z", "attachment_error_mm", "normalized_value"])
        for index in range(len(centers)):
            writer.writerow([f"C{index}", *centers[index], *normals[index], *closest[index], attachment_error[index], values[index]])


def write_verification_data(project, mesh, centers, result):
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
    result = reconstruct(mesh, centers, values)
    cmap = field_colormap()
    save_model_definition(mesh)
    save_sensor_layout(mesh, result["closest"], values, cmap)
    save_kernel_curve()
    save_distance_comparison(mesh, result)
    save_full_model_components(mesh, result["closest"], values, result, cmap)
    save_surface_reconstruction(mesh, result["closest"], values, result, cmap)
    save_workflow()
    write_sensor_data(centers, normals, values, result["closest"], result["attachment_error"])
    write_verification_data(project, mesh, centers, result)


if __name__ == "__main__":
    main()
