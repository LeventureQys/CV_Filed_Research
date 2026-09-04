# 归档目录说明

> 最后整理: 2026-07-11

## 目录结构

```
archived/
├── README.md                    ← 本文件
│
├── wiener_route/                ← WienerGate 路线（已废弃）
│   ├── WienerGate研究报告.md
│   ├── 算法可视化对比报告.md
│   └── output/                  ← 对比图片
│
├── pca_svd_route/               ← PCA+SVD 统一对比路线（已废弃）
│   └── output/                  ← final_*.png 对比图
│
├── pca_route/                   ← PCA 子空间独立路线（已废弃）
│   ├── PCA子空间路线说明.md
│   └── output/                  ← pca_*.png
│
├── svd_route/                   ← SVD hard 路线（已废弃）
│   ├── SVD降噪研究报告.md
│   └── output/                  ← svd_*.png
│
├── nmf_route/                   ← NMF 时序路线（已废弃）
│   └── output/                  ← nmf_*.png
│
├── membrane_route/              ← 膜片中值滤波路线（已废弃）
│   └── output/                  ← membrane_*.png
│
└── scripts/                     ← 所有历史脚本
    ├── algo_experiments.py
    ├── evaluate_algorithms.py
    ├── evaluate_full_comparison.py
    ├── generate_synthetic_data.py
    ├── wiener_gate.py
    ├── membrane_analysis.py
    ├── svd_denoiser.py
    ├── pca_subspace_denoiser.py
    ├── nmf_temporal_denoiser.py
    ├── render_comparison.py
    ├── render_final.py
    ├── render_nmf_comparison.py
    └── render_pca_comparison.py
```

## 各路线废弃原因

| 路线 | 核心问题 |
|------|----------|
| WienerGate | 逐通道独立处理，不符合整体 layout 要求 |
| StatGate/Spatial/Hybrid | 硬切/团化/失真 |
| SVD hard | 单帧硬截断，不看时序 |
| PCA 子空间 | whitening+shrinkage 过度削弱真信号 (SRR 0.06-0.20) |
| NMF 时序 | 方向对但当前实现未突破 (SRR 0.58-0.61) |
| Median 膜片 | 仅针对膜片边沿，不通用 |
