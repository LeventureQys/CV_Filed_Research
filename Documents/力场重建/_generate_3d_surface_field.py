from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


OUTPUT_DIR = Path(__file__).parent / "figure" / "3d_surface_field"
RADIUS_MM = 6.0
LENGTH_MM = 16.0
SUPPORT_MM = 8.0
AXIS_CENTER_Y_MM = 40.0

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
    ]
)


def field_colormap():
    positions = [0.00, 0.12, 0.35, 0.60, 0.80, 1.00]
    colors = np.array(
        [
            [140, 140, 148],
            [32, 76, 180],
            [0, 190, 220],
            [245, 220, 55],
            [245, 125, 35],
            [220, 35, 35],
        ]
    ) / 255.0
    return LinearSegmentedColormap.from_list(
        "beta4_surface_field", list(zip(positions, colors))
    )


def wendland_c2(distance, support):
    normalized = distance / support
    remaining = np.clip(1.0 - normalized, 0.0, 1.0)
    weight = remaining**4 * (4.0 * normalized + 1.0)
    return np.where(normalized < 1.0, weight, 0.0)


def reconstruct(axial_z, angle_deg):
    axial_grid, angle_grid = np.meshgrid(axial_z, angle_deg)
    arc_grid = RADIUS_MM * np.deg2rad(angle_grid)
    sensor_arc = RADIUS_MM * np.deg2rad(SENSORS[:, 1])
    distance = np.sqrt(
        (axial_grid[..., None] - SENSORS[:, 0]) ** 2
        + (arc_grid[..., None] - sensor_arc) ** 2
    )
    weights = wendland_c2(distance, SUPPORT_MM)
    weight_sum = weights.sum(axis=-1)
    interpolated = np.divide(
        (weights * SENSORS[:, 2]).sum(axis=-1),
        weight_sum,
        out=np.zeros_like(weight_sum),
        where=weight_sum > 1e-10,
    )
    coverage = np.minimum(1.0, weight_sum)
    effective = interpolated * coverage
    return axial_grid, angle_grid, interpolated, coverage, effective


def surface_xyz(axial_grid, angle_grid):
    angle = np.deg2rad(angle_grid)
    x = RADIUS_MM * np.sin(angle)
    y = AXIS_CENTER_Y_MM - RADIUS_MM * np.cos(angle)
    z = axial_grid
    return x, y, z


def sensor_xyz(offset_mm=0.0):
    angle = np.deg2rad(SENSORS[:, 1])
    radius = RADIUS_MM + offset_mm
    x = radius * np.sin(angle)
    y = AXIS_CENTER_Y_MM - radius * np.cos(angle)
    z = SENSORS[:, 0]
    return x, y, z


def style_3d(axis):
    axis.set_box_aspect((1.0, 1.0, 2.0))
    axis.set_xlabel("STEP x / mm")
    axis.set_ylabel("STEP y / mm")
    axis.set_zlabel("轴向 z / mm")
    axis.view_init(elev=22, azim=-48)
    axis.grid(False)
    axis.set_facecolor("#f7f8fa")


def save_sensor_layout(cmap):
    axial = np.linspace(-LENGTH_MM / 2, LENGTH_MM / 2, 180)
    angle = np.linspace(-90.0, 90.0, 220)
    axial_grid, angle_grid = np.meshgrid(axial, angle)
    x, y, z = surface_xyz(axial_grid, angle_grid)

    figure = plt.figure(figsize=(10.5, 5.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    base = np.full(x.shape + (4,), [0.72, 0.73, 0.76, 1.0])
    axis.plot_surface(x, y, z, facecolors=base, linewidth=0, shade=True, alpha=0.96)

    sx, sy, sz = sensor_xyz(offset_mm=0.18)
    axis.scatter(
        sx,
        sy,
        sz,
        c=SENSORS[:, 2],
        cmap=cmap,
        norm=Normalize(0, 1),
        s=78,
        edgecolor="white",
        linewidth=1.4,
        depthshade=False,
    )
    for index, (px, py, pz) in enumerate(zip(sx, sy, sz), start=1):
        axis.text(px, py - 0.35, pz, f"S{index}", fontsize=7, ha="center")

    style_3d(axis)
    axis.set_title("STEP φ12 接触半圆柱面：12 点传感器布局", pad=12)
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    colorbar = figure.colorbar(scalar, ax=axis, shrink=0.72, pad=0.08)
    colorbar.set_label("归一化压力")
    figure.savefig(OUTPUT_DIR / "sensor_layout.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction(cmap):
    axial = np.linspace(-LENGTH_MM / 2, LENGTH_MM / 2, 260)
    angle = np.linspace(-90.0, 90.0, 320)
    axial_grid, angle_grid, _, _, effective = reconstruct(axial, angle)
    x, y, z = surface_xyz(axial_grid, angle_grid)

    figure = plt.figure(figsize=(11.2, 6.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        x,
        y,
        z,
        facecolors=cmap(effective),
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    sx, sy, sz = sensor_xyz(offset_mm=0.15)
    axis.scatter(
        sx,
        sy,
        sz,
        c="white",
        s=24,
        edgecolor="#202124",
        linewidth=0.7,
        depthshade=False,
    )
    style_3d(axis)
    axis.set_title(
        f"曲面路径 Wendland C2 重建（支撑半径 R={SUPPORT_MM:g} mm）", pad=12
    )
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    colorbar = figure.colorbar(scalar, ax=axis, shrink=0.74, pad=0.08)
    colorbar.set_label("effective pressure")
    figure.savefig(OUTPUT_DIR / "surface_reconstruction.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_unwrapped(cmap):
    axial = np.linspace(-LENGTH_MM / 2, LENGTH_MM / 2, 320)
    angle = np.linspace(-90.0, 90.0, 380)
    axial_grid, angle_grid, interpolated, coverage, effective = reconstruct(axial, angle)
    arc = RADIUS_MM * np.deg2rad(angle_grid)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    panels = [
        (interpolated, "归一化插值 P"),
        (coverage, "覆盖置信度 C"),
        (effective, "显示标量 E=P×C"),
    ]
    for axis, (values, title) in zip(axes, panels):
        image = axis.pcolormesh(
            axial_grid,
            arc,
            values,
            cmap=cmap,
            vmin=0,
            vmax=1,
            shading="auto",
        )
        axis.scatter(
            SENSORS[:, 0],
            RADIUS_MM * np.deg2rad(SENSORS[:, 1]),
            c=SENSORS[:, 2],
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=35,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_title(title)
        axis.set_xlabel("轴向 z / mm")
        axis.set_aspect("equal")
    axes[0].set_ylabel("圆弧坐标 s=Rθ / mm")
    figure.colorbar(image, ax=axes, shrink=0.82, label="归一化值")
    figure.suptitle("半圆柱曲面展开：插值、coverage 与最终显示标量", fontsize=14)
    figure.savefig(OUTPUT_DIR / "unwrapped_components.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_distance_comparison():
    angle = np.linspace(0.0, 180.0, 500)
    radians = np.deg2rad(angle)
    geodesic = RADIUS_MM * radians
    euclidean = 2.0 * RADIUS_MM * np.sin(radians / 2.0)

    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.plot(angle, euclidean, linewidth=2.5, label="三维欧氏距离（弦长）")
    axis.plot(angle, geodesic, linewidth=2.5, label="曲面路径距离（弧长）")
    axis.axhline(SUPPORT_MM, color="#666666", linestyle="--", label="支撑半径 8 mm")
    axis.fill_between(angle, euclidean, geodesic, alpha=0.14, color="#d94841")
    axis.set_xlabel("沿圆柱面的角度差 / °")
    axis.set_ylabel("距离 / mm")
    axis.set_title("STEP φ12 圆柱面：空间直线会低估真实表面距离")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "distance_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_sensor_csv():
    rows = [
        "id,x_mm,y_mm,z_mm,theta_deg,arc_mm,nx,ny,nz,normalized_pressure"
    ]
    for index, (axial, angle, pressure) in enumerate(SENSORS, start=1):
        radians = np.deg2rad(angle)
        arc = RADIUS_MM * np.deg2rad(angle)
        x = RADIUS_MM * np.sin(radians)
        y = AXIS_CENTER_Y_MM - RADIUS_MM * np.cos(radians)
        normal_x = np.sin(radians)
        normal_y = -np.cos(radians)
        rows.append(
            f"S{index},{x:.3f},{y:.3f},{axial:.3f},{angle:.1f},{arc:.3f},"
            f"{normal_x:.6f},{normal_y:.6f},0.000000,{pressure:.2f}"
        )
    (OUTPUT_DIR / "sensor_data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    cmap = field_colormap()
    save_sensor_layout(cmap)
    save_reconstruction(cmap)
    save_unwrapped(cmap)
    save_distance_comparison()
    save_sensor_csv()


if __name__ == "__main__":
    main()
