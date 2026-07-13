import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"
STEP_PATH = BASE_DIR.parent / "3D Models" / "半圆柱压头φ12_.stp"
RADIUS_MM = 6.0
LENGTH_MM = 16.0
SUPPORT_MM = 8.0
AXIS_CENTER_Y_MM = 40.0
MIN_VALUE = 0.0
MAX_VALUE = 1.0

SENSORS = np.array(
    [
        [-6.0, -70.0, 0.18],
        [0.0, -70.0, 0.32],
        [6.0, -70.0, 0.46],
        [-6.0, -25.0, 0.42],
        [0.0, -25.0, 0.82],
        [6.0, -25.0, 0.68],
        [-6.0, 25.0, 0.55],
        [0.0, 25.0, 1.00],
        [6.0, 25.0, 0.76],
        [-6.0, 70.0, 0.28],
        [0.0, 70.0, 0.48],
        [6.0, 70.0, 0.36],
    ],
    dtype=float,
)


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
    return LinearSegmentedColormap.from_list("beta4_surface_field", list(zip(positions, colors)), N=256)


def verify_step_geometry(step_path=STEP_PATH):
    text = step_path.read_text(encoding="utf-8")
    center_match = re.search(
        r"CARTESIAN_POINT\('',\(0\.E0,4\.E1,0\.E0\)\).*?"
        r"CYLINDRICAL_SURFACE\('',#[0-9]+,6\.E0\)",
        text,
        flags=re.DOTALL,
    )
    if center_match is None:
        raise ValueError("STEP 中未找到中心 (0,40,0)、半径 6 mm 的圆柱面证据")
    if "CARTESIAN_POINT('',(0.E0,4.E1,8.E0))" not in text:
        raise ValueError("STEP 中未找到 z=8 mm 的轴向端点")
    if "CARTESIAN_POINT('',(0.E0,4.E1,-8.E0))" not in text:
        raise ValueError("STEP 中未找到 z=-8 mm 的轴向端点")
    return {
        "step_file": step_path.name,
        "surface": "cylindrical",
        "radius_mm": RADIUS_MM,
        "diameter_mm": 2.0 * RADIUS_MM,
        "axis_center_y_mm": AXIS_CENTER_Y_MM,
        "axial_min_mm": -LENGTH_MM / 2.0,
        "axial_max_mm": LENGTH_MM / 2.0,
        "step_entity_evidence": ["#1483", "#1487", "#587", "#625"],
    }


def wendland_c2(distance, support=SUPPORT_MM):
    normalized = np.asarray(distance, dtype=float) / support
    remaining = np.clip(1.0 - normalized, 0.0, 1.0)
    weights = remaining**4 * (4.0 * normalized + 1.0)
    return np.where(normalized < 1.0, weights, 0.0)


def reconstruct(axial_z, angle_deg, sensors=SENSORS):
    axial_grid, angle_grid = np.meshgrid(axial_z, angle_deg)
    arc_grid = RADIUS_MM * np.deg2rad(angle_grid)
    sensor_arc = RADIUS_MM * np.deg2rad(sensors[:, 1])
    distance = np.sqrt(
        (axial_grid[..., None] - sensors[:, 0]) ** 2
        + (arc_grid[..., None] - sensor_arc) ** 2
    )
    weights = wendland_c2(distance)
    weight_sum = weights.sum(axis=-1)
    interpolated = np.divide(
        (weights * sensors[:, 2]).sum(axis=-1),
        weight_sum,
        out=np.full_like(weight_sum, MIN_VALUE),
        where=weight_sum > 1e-12,
    )
    coverage = np.minimum(1.0, weight_sum)
    effective = MIN_VALUE + (interpolated - MIN_VALUE) * coverage
    return axial_grid, angle_grid, interpolated, coverage, effective, weight_sum


def surface_xyz(axial_grid, angle_grid, radius=RADIUS_MM):
    angle = np.deg2rad(angle_grid)
    x = radius * np.sin(angle)
    y = AXIS_CENTER_Y_MM - radius * np.cos(angle)
    return x, y, axial_grid


def sensor_xyz(offset_mm=0.0, sensors=SENSORS):
    angle = np.deg2rad(sensors[:, 1])
    radius = RADIUS_MM + offset_mm
    x = radius * np.sin(angle)
    y = AXIS_CENTER_Y_MM - radius * np.cos(angle)
    return x, y, sensors[:, 0]


def style_3d(axis):
    axis.set_box_aspect((1.0, 1.0, 1.7))
    axis.set_xlabel("STEP x / mm")
    axis.set_ylabel("STEP y / mm")
    axis.set_zlabel("轴向 z / mm")
    axis.view_init(elev=23, azim=-48)
    axis.grid(False)
    axis.set_facecolor("#f7f8fa")


def save_model_definition():
    axial = np.linspace(-8.0, 8.0, 160)
    angle = np.linspace(-90.0, 90.0, 220)
    axial_grid, angle_grid = np.meshgrid(axial, angle)
    x, y, z = surface_xyz(axial_grid, angle_grid)
    figure = plt.figure(figsize=(13.0, 5.7), constrained_layout=True)
    axis_3d = figure.add_subplot(121, projection="3d")
    base = np.full(x.shape + (4,), [0.72, 0.73, 0.76, 1.0])
    axis_3d.plot_surface(x, y, z, facecolors=base, linewidth=0, shade=True)
    style_3d(axis_3d)
    axis_3d.set_title("STEP 中提取的 φ12 半圆柱接触面")

    axis_2d = figure.add_subplot(122)
    profile_angle = np.deg2rad(np.linspace(-90.0, 90.0, 500))
    profile_x = RADIUS_MM * np.sin(profile_angle)
    profile_y = AXIS_CENTER_Y_MM - RADIUS_MM * np.cos(profile_angle)
    axis_2d.plot(profile_x, profile_y, color="#3b4a6b", linewidth=3)
    axis_2d.plot([-RADIUS_MM, RADIUS_MM], [AXIS_CENTER_Y_MM] * 2, color="#9298a5")
    axis_2d.scatter([0], [AXIS_CENTER_Y_MM], color="#202124", s=30, zorder=4)
    axis_2d.annotate(
        "R = 6 mm",
        xy=(RADIUS_MM / np.sqrt(2), AXIS_CENTER_Y_MM - RADIUS_MM / np.sqrt(2)),
        xytext=(1.0, 42.0),
        arrowprops={"arrowstyle": "->", "color": "#d94841"},
        color="#d94841",
        fontsize=11,
    )
    axis_2d.annotate("轴向范围 z = -8~8 mm", xy=(0, 40), xytext=(-5.8, 42.8), fontsize=11)
    axis_2d.set_aspect("equal")
    axis_2d.set(xlim=(-7.5, 7.5), ylim=(32.5, 44.0), xlabel="STEP x / mm", ylabel="STEP y / mm")
    axis_2d.set_title("接触面截面与尺寸依据")
    axis_2d.grid(alpha=0.22)
    figure.suptitle("模型几何定义（依据 STEP 实体 #1483、#1487、#587、#625）", fontsize=14)
    figure.savefig(FIGURE_DIR / "01_model_surface_definition.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_sensor_layout(cmap):
    axial_grid, angle_grid = np.meshgrid(np.linspace(-8.0, 8.0, 180), np.linspace(-90.0, 90.0, 220))
    x, y, z = surface_xyz(axial_grid, angle_grid)
    figure = plt.figure(figsize=(10.8, 6.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    base = np.full(x.shape + (4,), [0.72, 0.73, 0.76, 1.0])
    axis.plot_surface(x, y, z, facecolors=base, linewidth=0, shade=True, alpha=0.96)
    sx, sy, sz = sensor_xyz(offset_mm=0.18)
    axis.scatter(sx, sy, sz, c=SENSORS[:, 2], cmap=cmap, norm=Normalize(0, 1), s=82, edgecolor="white", linewidth=1.4, depthshade=False)
    for index, (px, py, pz) in enumerate(zip(sx, sy, sz), start=1):
        axis.text(px, py - 0.38, pz, f"S{index}", fontsize=7, ha="center")
    style_3d(axis)
    axis.set_title("φ12 接触半圆柱面：12 个有意布置的曲面采样点", pad=12)
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=axis, shrink=0.72, pad=0.08)
    colorbar.set_label("演示用归一化压力")
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
    axis.set(xlim=(0, 1.2), ylim=(-0.03, 1.05), xlabel="归一化曲面路径 r = d_surface / R", ylabel="权重 φ(r)")
    axis.set_title("Wendland C2 紧支撑核：边界处平滑降为 0")
    axis.grid(alpha=0.24)
    figure.savefig(FIGURE_DIR / "03_wendland_kernel.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_distance_comparison():
    angle = np.linspace(0.0, 180.0, 700)
    radians = np.deg2rad(angle)
    surface_distance = RADIUS_MM * radians
    euclidean_distance = 2.0 * RADIUS_MM * np.sin(radians / 2.0)
    surface_weight = wendland_c2(surface_distance)
    euclidean_weight = wendland_c2(euclidean_distance)
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    axes[0].plot(angle, euclidean_distance, linewidth=2.5, label="三维欧氏距离（弦长）")
    axes[0].plot(angle, surface_distance, linewidth=2.5, label="曲面路径距离（弧长）")
    axes[0].axhline(SUPPORT_MM, color="#666666", linestyle="--", label="支撑半径 8 mm")
    axes[0].fill_between(angle, euclidean_distance, surface_distance, alpha=0.14, color="#d94841")
    axes[0].set(xlabel="沿圆柱面的角度差 / °", ylabel="距离 / mm", title="空间直线会低估表面传播距离")
    axes[0].grid(alpha=0.24)
    axes[0].legend()
    axes[1].plot(angle, euclidean_weight, linewidth=2.5, label="用欧氏距离得到的权重")
    axes[1].plot(angle, surface_weight, linewidth=2.5, label="用曲面路径得到的权重")
    axes[1].fill_between(angle, surface_weight, euclidean_weight, alpha=0.14, color="#d94841")
    axes[1].set(xlabel="沿圆柱面的角度差 / °", ylabel="Wendland 权重", title="距离低估会被放大为过度传播")
    axes[1].grid(alpha=0.24)
    axes[1].legend()
    figure.suptitle("φ12 圆柱面上的欧氏距离与曲面路径距离", fontsize=14)
    figure.savefig(FIGURE_DIR / "04_distance_and_weight_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_unwrapped_components(cmap):
    axial = np.linspace(-8.0, 8.0, 320)
    angle = np.linspace(-90.0, 90.0, 380)
    axial_grid, angle_grid, interpolated, coverage, effective, _ = reconstruct(axial, angle)
    arc_grid = RADIUS_MM * np.deg2rad(angle_grid)
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)
    panels = [(interpolated, "归一化插值 P"), (coverage, "覆盖置信度 C"), (effective, "显示标量 E=P×C")]
    image = None
    for axis, (values, title) in zip(axes, panels):
        image = axis.pcolormesh(axial_grid, arc_grid, values, cmap=cmap, vmin=0, vmax=1, shading="auto")
        axis.scatter(
            SENSORS[:, 0],
            RADIUS_MM * np.deg2rad(SENSORS[:, 1]),
            c=SENSORS[:, 2],
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=36,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_title(title)
        axis.set_xlabel("轴向 z / mm")
        axis.set_aspect("equal")
    axes[0].set_ylabel("圆弧坐标 s=Rθ / mm")
    figure.colorbar(image, ax=axes, shrink=0.82, label="归一化值")
    figure.suptitle("半圆柱曲面展开：插值、coverage 与最终显示标量", fontsize=14)
    figure.savefig(FIGURE_DIR / "05_unwrapped_components.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_surface_reconstruction(cmap):
    axial = np.linspace(-8.0, 8.0, 260)
    angle = np.linspace(-90.0, 90.0, 320)
    axial_grid, angle_grid, _, _, effective, _ = reconstruct(axial, angle)
    x, y, z = surface_xyz(axial_grid, angle_grid)
    figure = plt.figure(figsize=(11.3, 6.1), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(x, y, z, facecolors=cmap(effective), linewidth=0, antialiased=False, shade=False)
    sx, sy, sz = sensor_xyz(offset_mm=0.15)
    axis.scatter(sx, sy, sz, c="white", s=26, edgecolor="#202124", linewidth=0.7, depthshade=False)
    style_3d(axis)
    axis.set_title(f"曲面路径 Wendland C2 重建（支撑半径 R={SUPPORT_MM:g} mm）", pad=12)
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=axis, shrink=0.74, pad=0.08)
    colorbar.set_label("effective pressure")
    figure.savefig(FIGURE_DIR / "06_surface_reconstruction.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_workflow():
    labels = [
        "Cell 曲面采样",
        "附着到三角面",
        "截断表面最短路",
        "Wendland C2 权重",
        "归一化插值 + coverage",
        "统一色表显示",
    ]
    figure, axis = plt.subplots(figsize=(14.2, 2.8), constrained_layout=True)
    axis.set_xlim(0, len(labels) * 2.2)
    axis.set_ylim(0, 2.2)
    axis.axis("off")
    for index, label in enumerate(labels):
        x = index * 2.2 + 0.15
        box = FancyBboxPatch(
            (x, 0.7),
            1.75,
            0.78,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#3b4a6b",
            facecolor="#eef3fb" if index < 4 else "#fff0df",
        )
        axis.add_patch(box)
        axis.text(x + 0.875, 1.09, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (x + 1.78, 1.09),
                    (x + 2.13, 1.09),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.3,
                    color="#677083",
                )
            )
    axis.set_title("3D 曲面力场重建的数据流", fontsize=14, pad=10)
    figure.savefig(FIGURE_DIR / "07_algorithm_workflow.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_sensor_data():
    rows = ["id,x_mm,y_mm,z_mm,theta_deg,arc_mm,nx,ny,nz,normalized_pressure"]
    for index, (axial, angle, pressure) in enumerate(SENSORS, start=1):
        radians = np.deg2rad(angle)
        x = RADIUS_MM * np.sin(radians)
        y = AXIS_CENTER_Y_MM - RADIUS_MM * np.cos(radians)
        rows.append(
            f"S{index},{x:.3f},{y:.3f},{axial:.3f},{angle:.1f},"
            f"{RADIUS_MM * radians:.3f},{np.sin(radians):.6f},"
            f"{-np.cos(radians):.6f},0.000000,{pressure:.2f}"
        )
    (DATA_DIR / "sensor_data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_verification_data(model_evidence):
    _, _, interpolated, coverage, effective, weight_sum = reconstruct(
        np.linspace(-8.0, 8.0, 320), np.linspace(-90.0, 90.0, 380)
    )
    covered = weight_sum > 1e-12
    metrics = {
        "model_evidence": model_evidence,
        "sensor_count": int(SENSORS.shape[0]),
        "support_radius_mm": SUPPORT_MM,
        "interpolated_min_on_covered": float(interpolated[covered].min()),
        "interpolated_max_on_covered": float(interpolated[covered].max()),
        "coverage_min": float(coverage.min()),
        "coverage_max": float(coverage.max()),
        "effective_min": float(effective.min()),
        "effective_max": float(effective.max()),
        "all_values_finite": bool(
            np.isfinite(interpolated).all()
            and np.isfinite(coverage).all()
            and np.isfinite(effective).all()
        ),
    }
    (DATA_DIR / "verification_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_evidence = verify_step_geometry()
    cmap = field_colormap()
    save_model_definition()
    save_sensor_layout(cmap)
    save_kernel_curve()
    save_distance_comparison()
    save_unwrapped_components(cmap)
    save_surface_reconstruction(cmap)
    save_workflow()
    write_sensor_data()
    write_verification_data(model_evidence)


if __name__ == "__main__":
    main()
