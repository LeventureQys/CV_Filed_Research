from __future__ import annotations

import heapq
import io
import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "表面重建用例_v2"
PROJECT_PATH = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\mvp_modules\simple.3dlp")
SUPPORT = 24.0
FOLD_PENALTY = 2.0
ACTIVE_CELL_IDS = np.array([0, 7, 8])
ACTIVE_VALUES = np.array([0.72, 1.00, 0.58])


def field_colormap():
    positions = [0.00, 0.12, 0.35, 0.60, 0.80, 1.00]
    colors = np.array(
        [[140, 140, 148], [32, 76, 180], [0, 190, 220], [245, 220, 55], [245, 125, 35], [220, 35, 35]],
        dtype=float,
    ) / 255.0
    return LinearSegmentedColormap.from_list("surface_field", list(zip(positions, colors)), N=256)


CMAP = field_colormap()


def load_project():
    with zipfile.ZipFile(PROJECT_PATH) as archive:
        project = json.loads(archive.read("project.json").decode("utf-8"))
        mesh = trimesh.load(io.BytesIO(archive.read(project["model_file"])), file_type="stl", process=True)
    centers = np.array(
        [[project["cells"][index]["center_3d"][axis] for axis in ("x", "y", "z")] for index in ACTIVE_CELL_IDS],
        dtype=float,
    )
    return mesh, centers


def subdivide(mesh: trimesh.Trimesh, levels: int):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    for _ in range(levels):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def build_graph(mesh: trimesh.Trimesh):
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.face_normals)
    edge_faces = {}
    for face_index, face in enumerate(mesh.faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(face_index)
    adjacency = [[] for _ in vertices]
    for (first, second), faces in edge_faces.items():
        cost = float(np.linalg.norm(vertices[first] - vertices[second]))
        if len(faces) >= 2:
            dot = abs(float(np.dot(normals[faces[0]], normals[faces[1]])))
            angle = np.arccos(np.clip(dot, 0.0, 1.0))
            cost *= 1.0 + FOLD_PENALTY * (angle / np.pi) ** 2
        adjacency[first].append((second, cost))
        adjacency[second].append((first, cost))
    return adjacency


def attach_cells(mesh: trimesh.Trimesh, centers: np.ndarray):
    closest, _, face_ids = trimesh.proximity.closest_point_naive(mesh, centers)
    seeds = []
    vertices = np.asarray(mesh.vertices)
    for point, face_id in zip(closest, face_ids):
        face = mesh.faces[face_id]
        seeds.append(int(face[np.argmin(np.linalg.norm(vertices[face] - point, axis=1))]))
    return closest, np.asarray(seeds)


def dijkstra(adjacency, source, cutoff):
    distances = np.full(len(adjacency), np.inf)
    distances[source] = 0.0
    queue = [(0.0, int(source))]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        for neighbor, cost in adjacency[vertex]:
            candidate = distance + cost
            if candidate < distances[neighbor] and candidate <= cutoff:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def wendland(distances):
    ratio = distances / SUPPORT
    weights = np.zeros_like(ratio)
    inside = ratio < 1.0
    remainder = 1.0 - ratio[inside]
    weights[inside] = remainder**4 * (4.0 * ratio[inside] + 1.0)
    return weights


def reconstruct(mesh, centers, distance_mode):
    closest, seeds = attach_cells(mesh, centers)
    vertices = np.asarray(mesh.vertices)
    if distance_mode == "euclidean":
        distances = np.vstack([np.linalg.norm(vertices - vertices[seed], axis=1) for seed in seeds])
    else:
        adjacency = build_graph(mesh)
        distances = np.vstack([dijkstra(adjacency, seed, SUPPORT) for seed in seeds])
    weights = wendland(distances)
    weight_sum = weights.sum(axis=0)
    interpolated = np.divide(
        ACTIVE_VALUES @ weights,
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 1e-12,
    )
    coverage = np.minimum(1.0, weight_sum)
    effective = interpolated * coverage
    return closest, interpolated, coverage, effective


def display_vertices(mesh):
    return np.asarray(mesh.vertices)[:, [0, 2, 1]]


def add_mesh(axis, mesh, effective=None, alpha=1.0, edges=False):
    triangles = display_vertices(mesh)[mesh.faces]
    if effective is None:
        colors = np.tile(np.array([0.73, 0.74, 0.77, alpha]), (len(mesh.faces), 1))
    else:
        face_values = effective[mesh.faces].mean(axis=1)
        colors = CMAP(np.clip(face_values, 0, 1))
        colors[:, 3] = np.where(face_values > 0.012, alpha, 0.13)
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors=(0.12, 0.13, 0.15, 0.16) if edges else "none",
        linewidths=0.08 if edges else 0.0,
    )
    axis.add_collection3d(collection)


def add_cells(axis, closest):
    points = closest[:, [0, 2, 1]]
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=ACTIVE_VALUES, cmap=CMAP, vmin=0, vmax=1, s=48, edgecolor="white", linewidth=1.0, depthshade=False)
    for label, point in zip(["C0", "C7", "C8"], points):
        axis.text(*point, label, fontsize=8, color="#202124")


def style_axis(axis, mesh, title, elev, azim, local=False):
    if local:
        axis.set_xlim(-15, 15)
        axis.set_ylim(-16, 16)
        axis.set_zlim(12, 47)
        axis.set_box_aspect((30, 32, 35))
    else:
        bounds = mesh.bounds
        axis.set_xlim(bounds[:, 0])
        axis.set_ylim(bounds[:, 2])
        axis.set_zlim(bounds[:, 1])
        axis.set_box_aspect((70, 70, 46))
    axis.view_init(elev=elev, azim=azim)
    axis.set_xlabel("x / mm")
    axis.set_ylabel("z / mm")
    axis.set_zlabel("y / mm")
    axis.grid(False)
    axis.set_title(title)


def save_locator(mesh, closest):
    figure = plt.figure(figsize=(15, 6.8), dpi=190, constrained_layout=True)
    for index, (elev, azim, title) in enumerate([(14, -90, "正视：Cell 位于压头上部"), (18, -42, "斜视：目标区域与底座分离")], start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        add_mesh(axis, mesh, alpha=0.56, edges=True)
        add_cells(axis, closest)
        style_axis(axis, mesh, title, elev, azim)
    figure.suptitle("实际 .3dlp 模型：本用例只重建 C0/C7/C8 所在的上部表面邻域", fontsize=15)
    figure.savefig(OUTPUT_DIR / "06_实际模型_目标Cell定位.png", bbox_inches="tight")
    plt.close(figure)


def save_real_views(mesh, closest, effective):
    figure = plt.figure(figsize=(16, 5.8), dpi=190, constrained_layout=True)
    views = [(10, -90, "正视局部"), (16, -48, "左斜视局部"), (16, -132, "右斜视局部")]
    for index, (elev, azim, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        add_mesh(axis, mesh, effective)
        add_cells(axis, closest)
        style_axis(axis, mesh, title, elev, azim, local=True)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=figure.axes, shrink=0.72, label="effective E")
    figure.suptitle(f"实际三角曲面：测地 Wendland C2 局部重建（R={SUPPORT:g} mm）", fontsize=15)
    figure.savefig(OUTPUT_DIR / "07_实际表面_推荐方案_局部三视角.png", bbox_inches="tight")
    plt.close(figure)


def save_distance_compare(mesh, closest, euclidean, geodesic):
    figure = plt.figure(figsize=(15, 6.3), dpi=190, constrained_layout=True)
    for index, (field, title) in enumerate([(euclidean, "3D 欧氏距离：可能走空间捷径"), (geodesic, "曲面路径：沿实际拓扑传播")], start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        add_mesh(axis, mesh, field)
        add_cells(axis, closest)
        style_axis(axis, mesh, title, 15, -52, local=True)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=figure.axes, shrink=0.74, label="固定 0～1 色标")
    figure.suptitle("实际模型上的距离定义对比", fontsize=15)
    figure.savefig(OUTPUT_DIR / "08_实际表面_欧氏与测地对比.png", bbox_inches="tight")
    plt.close(figure)


def save_mesh_density(coarse_mesh, fine_mesh, coarse_data, fine_data):
    figure = plt.figure(figsize=(15, 6.3), dpi=190, constrained_layout=True)
    panels = [
        (coarse_mesh, coarse_data[0], coarse_data[1], "原始网格：814 顶点，面片轮廓明显", True),
        (fine_mesh, fine_data[0], fine_data[1], "细分计算/显示网格：连续度明显提高", False),
    ]
    for index, (mesh, closest, field, title, edges) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        add_mesh(axis, mesh, field, edges=edges)
        add_cells(axis, closest)
        style_axis(axis, mesh, title, 15, -52, local=True)
    figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=figure.axes, shrink=0.74, label="相同算法和固定色标")
    figure.suptitle("实际表面上，显示网格分辨率本身会决定是否碎片化", fontsize=15)
    figure.savefig(OUTPUT_DIR / "09_实际表面_原始与细分网格.png", bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    coarse_mesh, centers = load_project()
    fine_mesh = subdivide(coarse_mesh, 2)
    coarse_closest, _, _, coarse_effective = reconstruct(coarse_mesh, centers, "geodesic")
    fine_closest, fine_p, fine_c, fine_effective = reconstruct(fine_mesh, centers, "geodesic")
    _, _, _, euclidean_effective = reconstruct(fine_mesh, centers, "euclidean")
    save_locator(coarse_mesh, coarse_closest)
    save_real_views(fine_mesh, fine_closest, fine_effective)
    save_distance_compare(fine_mesh, fine_closest, euclidean_effective, fine_effective)
    save_mesh_density(
        coarse_mesh,
        fine_mesh,
        (coarse_closest, coarse_effective),
        (fine_closest, fine_effective),
    )
    np.savez_compressed(
        OUTPUT_DIR / "real_surface_fields.npz",
        vertices=np.asarray(fine_mesh.vertices),
        triangles=np.asarray(fine_mesh.faces),
        centers=centers,
        values=ACTIVE_VALUES,
        interpolated=fine_p,
        coverage=fine_c,
        geodesic_effective=fine_effective,
        euclidean_effective=euclidean_effective,
    )
    print(f"Generated real-surface cases in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
