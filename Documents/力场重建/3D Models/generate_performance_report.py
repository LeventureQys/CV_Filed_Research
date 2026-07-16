from __future__ import annotations

import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt


matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "独立热力载体对比"


def load_results():
    return json.loads((OUTPUT_DIR / "performance_benchmark.json").read_text(encoding="utf-8"))


def save_chart(results):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.4), dpi=180, constrained_layout=True)

    atlas = results["uv_atlas_preprocess"]
    names = list(atlas)
    triangles = [atlas[name]["triangles"] for name in names]
    times = [atlas[name]["median_ms"] for name in names]
    axes[0].plot(triangles, times, marker="o", linewidth=2.4, color="#3267a8")
    axes[0].set_title("UV Atlas：一次性预处理")
    axes[0].set_xlabel("三角形数量")
    axes[0].set_ylabel("中位时间 / ms")
    axes[0].grid(alpha=0.25)
    for x, y in zip(triangles, times):
        axes[0].annotate(f"{y:.0f} ms", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)

    overlay = results["overlay_frame_update"]
    for cells, color in zip((3, 16, 64), ("#4c78a8", "#f2a541", "#d64f4f")):
        rows = [row for row in overlay if row["cells"] == cells]
        axes[1].plot(
            [row["items"] for row in rows],
            [row["p95_ms"] for row in rows],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{cells} Cell",
        )
    axes[1].axhline(16.67, color="#222222", linestyle="--", linewidth=1.4, label="60 Hz 总帧预算")
    axes[1].set_title("Overlay：缓存后的每帧 CPU 更新")
    axes[1].set_xlabel("Overlay 顶点数量")
    axes[1].set_ylabel("P95 时间 / ms")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    texture = results["texture_frame_update"]
    for cells, color in zip((3, 16, 64), ("#4c78a8", "#f2a541", "#d64f4f")):
        rows = [row for row in texture if row["cells"] == cells]
        axes[2].plot(
            [row["resolution"] for row in rows],
            [row["p95_ms"] for row in rows],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{cells} Cell",
        )
    axes[2].axhline(16.67, color="#222222", linestyle="--", linewidth=1.4, label="60 Hz 总帧预算")
    axes[2].set_title("CPU 全量热力纹理：每帧更新")
    axes[2].set_xlabel("纹理边长 / pixel")
    axes[2].set_ylabel("P95 时间 / ms")
    axes[2].set_xticks([256, 512, 1024])
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    figure.suptitle("i5-12600HX：热力图预处理与每帧更新基准", fontsize=16)
    figure.savefig(OUTPUT_DIR / "08_性能基准总览.png", bbox_inches="tight")
    plt.close(figure)


def write_report(results):
    atlas = results["uv_atlas_preprocess"]
    overlay = results["overlay_frame_update"]
    texture = results["texture_frame_update"]
    report = f"""# UV Atlas、Overlay 与 60 Hz 性能评估

## 测试机器与口径

- CPU：12th Gen Intel Core i5-12600HX，12 核 16 线程。
- Python 3.14 + NumPy 单进程参考实现。
- 60 Hz 一帧总预算为 `16.67 ms`，实际热力更新最好控制在 `2~4 ms`，给渲染、UI 和业务逻辑留余量。
- 当前结果包含 CPU 标量更新，不包含真实 OpenGL/Direct3D 驱动上传、Draw Call 和屏幕合成成本。
- 每帧测试使用预计算影响缓存，每个位置最多保留 8 个有效 Cell。这是生产建议，不是每帧重新计算距离。

## UV Atlas 是否和面片数量有关

有关。UV Atlas 至少需要处理每个三角形的邻接、Chart 切分、参数化和打包，成本通常随三角形数量、拓扑复杂度、接缝数量一起增长，不应假设严格线性。

本机 `xatlas` 实测：

| 网格 | 三角形 | UV 预处理中位时间 |
|---|---:|---:|
| 原始 | {atlas['原始网格']['triangles']:,} | {atlas['原始网格']['median_ms']:.1f} ms |
| 细分 1 级 | {atlas['细分1级']['triangles']:,} | {atlas['细分1级']['median_ms']:.1f} ms |
| 细分 2 级 | {atlas['细分2级']['triangles']:,} | {atlas['细分2级']['median_ms']:.1f} ms |
| 细分 3 级 | {atlas['细分3级']['triangles']:,} | {atlas['细分3级']['median_ms']:.1f} ms |

约 10.6 万三角形在本机约 `0.91 s`，因此高面数模型运行时临时生成全局 UV 会造成明显卡顿。但正确做法是：

1. 导入模型或后台任务中生成一次。
2. 把 UV、Chart 和纹理缓存保存到项目文件。
3. Cell 数值变化时绝不重新生成 UV。
4. 只显示局部热区时，只对目标 Patch 参数化，不处理完整模型。
5. 百万面 CAD 建议先生成简化代理/局部 Patch，或离线生成 UV，再把参数坐标映射回渲染模型。

## Overlay 每帧能否达到 60 Hz

可以，而且当前规模余量较大。

| Overlay 顶点 | Cell | CPU 中位 | CPU P95 | 结论 |
|---:|---:|---:|---:|---|
"""
    for row in overlay:
        conclusion = "充足" if row["p95_ms"] <= 4.0 else "可达，但应继续优化"
        report += f"| {row['items']:,} | {row['cells']} | {row['median_ms']:.3f} ms | {row['p95_ms']:.3f} ms | {conclusion} |\n"

    report += f"""

当前预览使用的 13,234 顶点 Overlay：

- 3 Cell：P95 `{next(row['p95_ms'] for row in overlay if row['items'] == 13234 and row['cells'] == 3):.3f} ms`。
- 16 Cell：P95 `{next(row['p95_ms'] for row in overlay if row['items'] == 13234 and row['cells'] == 16):.3f} ms`。
- 64 Cell：P95 `{next(row['p95_ms'] for row in overlay if row['items'] == 13234 and row['cells'] == 64):.3f} ms`。
- 每帧只需上传约 `52 KB` 的 R32F 标量，若使用 R16F 则约 `26 KB`。

即使 52,978 顶点、64 Cell，P95 也约 `{next(row['p95_ms'] for row in overlay if row['items'] == 52978 and row['cells'] == 64):.3f} ms`。因此在当前 CPU 上，**缓存后的 Overlay 标量更新达到 60 Hz 是现实的**。

## UV 热力纹理每帧能否达到 60 Hz

如果在 CPU 上每帧全量更新所有 texel：

| 分辨率 | Cell | CPU 中位 | CPU P95 | 判断 |
|---:|---:|---:|---:|---|
"""
    for row in texture:
        judgment = "CPU 可用" if row["p95_ms"] <= 8.0 else ("接近/占满帧预算" if row["p95_ms"] <= 16.67 else "CPU 60 Hz 不可用")
        report += f"| {row['resolution']}² | {row['cells']} | {row['median_ms']:.3f} ms | {row['p95_ms']:.3f} ms | {judgment} |\n"

    report += """

结论：

- `256²` CPU 全量更新可以做到 60 Hz，但 16/64 Cell 已消耗约 5 ms P95。
- `512²` 只有少量 Cell 比较稳；16 Cell 已超过完整帧预算，不建议 CPU 每帧全量更新。
- `1024²` CPU 全量更新约 35～51 ms P95，不能达到 60 Hz。
- UV 纹理方案要做 60 Hz，应使用 Compute Shader/CUDA/OpenCL，在 GPU 上根据预计算权重更新，或采用脏矩形、分块更新、降低更新频率。
- 热力场通常没有必要和相机以相同频率重建。可以 20～30 Hz 更新数据纹理，渲染仍保持 60 Hz，并在两个标量场之间插值。

## 不要每帧执行的操作

- UV Atlas 生成。
- Overlay 网格生成或细分。
- 顶点焊接和邻接图构建。
- Cell 投影和目标面识别。
- 每个 Cell 的 Dijkstra/Heat Method。

这些只在模型、Cell 位置、影响半径或传播边界变化时执行。本机 Python 截断 Dijkstra 在 13k 顶点上，3/16/64 Cell 分别约 `10.6/45.9/272.9 ms`；如果把它放进每帧，当然无法 60 Hz。

## 每帧只做什么

预处理阶段缓存：

```text
offsets[itemCount + 1]
sampleIndices[influenceCount]
weights[influenceCount]
coverage[itemCount]
```

每帧只做：

```text
effective[item] = Σ(sampleValue[index] × normalizedWeight) × coverage
```

然后上传一个 R16F/R32F 标量 Buffer 或 Texture。颜色 LUT、灯光和透明度全部在 Fragment Shader 中处理，不回传 RGB。

## 推荐选择

### 当前模型与常规 CPU

首选：**局部 Overlay Mesh + 预计算 CSR 权重缓存 + 每帧更新单通道顶点标量**。

- Overlay 建议控制在 `10k~50k` 顶点/活动区域。
- 每顶点保留最重要的 `4~8` 个 Cell 影响。
- CPU 更新后上传 R16F/R32F 标量。
- Overlay 网格、拓扑距离和权重只在布局变化时重建。
- 多个热区使用多个局部 Overlay，避免给完整大模型铺一层超密网格。

### 需要纹理级细节

使用 **UV Atlas + GPU Compute 更新**，或 CPU 只更新 `256²/局部脏块`。UV 生成成本不是主要运行时问题，因为它应当缓存；真正要关注的是每帧 texel 数量和上传带宽。

### 60 Hz 验收口径

最终需要在目标程序内分别记录：

1. `fieldUpdateCpuMs`：标量更新。
2. `gpuUploadMs`：Buffer/Texture 上传。
3. `heatDrawGpuMs`：Overlay/热力材质 GPU 时间。
4. `totalFrameP95Ms`：整帧 P95，必须小于 `16.67 ms`。
5. 运行至少 60 秒，统计 P50/P95/P99，而不是只看平均 FPS。

本基准证明 CPU 标量更新方案具有 60 Hz 可行性，但不能替代最终 OpenGL/Direct3D 集成测试。
"""
    (OUTPUT_DIR / "README_性能与60Hz评估.md").write_text(report, encoding="utf-8")


def main():
    results = load_results()
    save_chart(results)
    write_report(results)
    print(f"Generated performance report in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
