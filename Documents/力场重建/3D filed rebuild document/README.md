# 3D 曲面力场重建文档包

## 入口

- 主文章：`3D曲面力场重建方法.md`
- 问题清单：`问题清单.md`
- 设计说明：`设计文档.md`
- 开发过程：`开发文档.md`
- 验收方式：`验收文档.md`
- 完成情况：`完成报告.md`

## 目录

- `figures/`：文章使用的 8 张可复现原创图片
- `data/sensor_data.csv`：12 个采样点的坐标、法线和演示压力
- `data/verification_metrics.json`：生成后的数值核验结果
- `scripts/generate_figures.py`：从真实模型和 Cell 数据生成局部 Overlay、重建结果及 8 张配图
- `scripts/test_generate_figures.py`：数值单元测试

## 快速复现

```powershell
python -m unittest "Documents\力场重建\3D filed rebuild document\scripts\test_generate_figures.py" -v
python "Documents\力场重建\3D filed rebuild document\scripts\generate_figures.py"
```
