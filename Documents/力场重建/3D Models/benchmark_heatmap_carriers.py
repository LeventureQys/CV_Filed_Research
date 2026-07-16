from __future__ import annotations

import gc
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import xatlas


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "独立热力载体对比"
HELPER_PATH = ROOT / "generate_real_surface_cases.py"
RNG = np.random.default_rng(20260612)


def load_helper():
    specification = importlib.util.spec_from_file_location("real_cases", HELPER_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


HELPER = load_helper()


def measure(function, warmup=3, repeats=30):
    for _ in range(warmup):
        function()
    gc.disable()
    try:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            function()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    finally:
        gc.enable()
    values = np.asarray(samples)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
    }


def atlas_benchmark(mesh, repeats=3):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    def run():
        return xatlas.parametrize(vertices, faces)

    result = measure(run, warmup=0, repeats=repeats)
    mapping, indices, uv = run()
    result.update(
        {
            "vertices": int(len(vertices)),
            "triangles": int(len(faces)),
            "uv_vertices": int(len(mapping)),
            "uv_bytes": int(uv.nbytes + indices.nbytes + mapping.nbytes),
        }
    )
    return result


def influence_cache(item_count, cell_count, neighbor_count=8):
    active = min(cell_count, neighbor_count)
    indices = RNG.integers(0, cell_count, size=(item_count, active), dtype=np.int32)
    weights = RNG.random((item_count, active), dtype=np.float32)
    weights *= RNG.random((item_count, 1), dtype=np.float32)
    weight_sum = weights.sum(axis=1)
    coverage = np.minimum(1.0, weight_sum)
    normalized = np.divide(weights, weight_sum[:, None], out=np.zeros_like(weights), where=weight_sum[:, None] > 1e-8)
    return indices, normalized, coverage.astype(np.float32)


def frame_update_benchmark(item_count, cell_count, repeats):
    indices, weights, coverage = influence_cache(item_count, cell_count)
    values = RNG.random(cell_count, dtype=np.float32)
    output = np.empty(item_count, dtype=np.float32)

    def update():
        output[:] = np.sum(values[indices] * weights, axis=1) * coverage
        return output

    result = measure(update, warmup=8, repeats=repeats)
    result.update(
        {
            "items": int(item_count),
            "cells": int(cell_count),
            "influences_per_item": int(indices.shape[1]),
            "cache_bytes": int(indices.nbytes + weights.nbytes + coverage.nbytes),
            "output_bytes_R32F": int(output.nbytes),
            "estimated_output_bytes_R16F": int(item_count * 2),
        }
    )
    return result


def copy_benchmark(item_count, dtype, repeats=200):
    source = RNG.random(item_count).astype(dtype)
    destination = np.empty_like(source)
    result = measure(lambda: np.copyto(destination, source), warmup=10, repeats=repeats)
    result.update({"items": int(item_count), "dtype": str(np.dtype(dtype)), "bytes": int(source.nbytes)})
    return result


def geodesic_cache_benchmark(mesh, cell_count):
    vertices = np.asarray(mesh.vertices)
    chosen = RNG.choice(len(vertices), size=cell_count, replace=False)
    centers = vertices[chosen]
    adjacency = HELPER.build_graph(mesh)

    def run():
        return [HELPER.dijkstra(adjacency, int(source), HELPER.SUPPORT) for source in chosen]

    result = measure(run, warmup=0, repeats=2)
    result.update({"vertices": int(len(vertices)), "triangles": int(len(mesh.faces)), "cells": int(cell_count)})
    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    coarse_mesh, _ = HELPER.load_project()
    meshes = {
        "原始网格": coarse_mesh,
        "细分1级": HELPER.subdivide(coarse_mesh, 1),
        "细分2级": HELPER.subdivide(coarse_mesh, 2),
        "细分3级": HELPER.subdivide(coarse_mesh, 3),
    }

    results = {
        "environment": {
            "python": platform.python_version(),
            "processor": platform.processor(),
            "benchmark_note": "Python/NumPy CPU reference; excludes OpenGL/Direct3D driver upload and draw time",
        },
        "uv_atlas_preprocess": {name: atlas_benchmark(mesh) for name, mesh in meshes.items()},
        "geodesic_cache_preprocess": [],
        "overlay_frame_update": [],
        "texture_frame_update": [],
        "memory_copy_lower_bound": [],
    }

    for name in ("细分1级", "细分2级", "细分3级"):
        for cell_count in (3, 16, 64):
            results["geodesic_cache_preprocess"].append(
                {"mesh": name, **geodesic_cache_benchmark(meshes[name], cell_count)}
            )

    for vertex_count in (len(meshes["细分1级"].vertices), len(meshes["细分2级"].vertices), len(meshes["细分3级"].vertices)):
        for cell_count in (3, 16, 64):
            results["overlay_frame_update"].append(
                frame_update_benchmark(vertex_count, cell_count, repeats=160)
            )

    for resolution in (256, 512, 1024):
        texels = resolution * resolution
        for cell_count in (3, 16, 64):
            row = frame_update_benchmark(texels, cell_count, repeats=50 if resolution == 1024 else 100)
            row["resolution"] = resolution
            results["texture_frame_update"].append(row)

    for item_count, label in [
        (len(meshes["细分2级"].vertices), "Overlay 13k"),
        (512 * 512, "Texture 512²"),
        (1024 * 1024, "Texture 1024²"),
    ]:
        for dtype in (np.float16, np.float32):
            results["memory_copy_lower_bound"].append(
                {"label": label, **copy_benchmark(item_count, dtype)}
            )

    output_path = OUTPUT_DIR / "performance_benchmark.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark results: {output_path}")


if __name__ == "__main__":
    main()
