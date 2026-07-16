# 织物垫实物研究汇报 PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 beta4 研究报告生成一份 11 页、面向管理层、包含公司 Logo 与效果图的可编辑 PowerPoint 汇报文件。

**Architecture:** 使用 Python `python-pptx` 构建统一母版式页面，通过 Pillow 将 ICO 转为透明 PNG，并把已有算法图表按页面需要等比缩放。生成后通过 `python-pptx` 回读检查页数、文本和图片数量；若本机 PowerPoint 可用，再导出 PDF 或预览图进行视觉检查。

**Tech Stack:** Python 3.14、python-pptx、Pillow、Matplotlib、PowerPoint COM（可选）

---

### Task 1: 准备品牌与图像资源

**Files:**
- Create: `Document/Update/v1.0.10 - research/beta4/output/ppt_assets/logo.png`
- Create: `Document/Update/v1.0.10 - research/beta4/output/ppt_assets/metrics_focus.png`
- Create: `Document/Update/v1.0.10 - research/beta4/output/ppt_assets/effect_focus.png`

- [ ] **Step 1: 转换公司 Logo**

使用 Pillow 读取 `assets/logo.ico` 的最高分辨率帧并保存为透明 PNG。

- [ ] **Step 2: 生成 PPT 专用图像**

从 `fabric_metrics_summary.png` 与 `fabric_final_comparison.png` 读取图片，保持纵横比并生成适合 16:9 页面排版的副本，不拉伸原图。

- [ ] **Step 3: 验证资源尺寸**

Run: `python -c "from PIL import Image; ..."`

Expected: Logo、指标图、效果图均可打开，宽高大于 300 px。

### Task 2: 生成 11 页 PPT

**Files:**
- Create: `Document/Update/v1.0.10 - research/beta4/scripts/build_presentation.py`
- Create: `Document/Update/v1.0.10 - research/beta4/织物垫实物研究汇报.pptx`

- [ ] **Step 1: 定义主题和公共组件**

实现宽屏页面、统一标题、页脚、页码、Logo、结论卡片、流程箭头和图片等比缩放函数。

- [ ] **Step 2: 创建 11 页内容**

按 `PPT设计文档.md` 的页面结构写入标题、要点、决策矩阵、流程图和三张研究图表。

- [ ] **Step 3: 保存 PPT**

Run: `python "Document/Update/v1.0.10 - research/beta4/scripts/build_presentation.py"`

Expected: 生成 `织物垫实物研究汇报.pptx`，无异常。

### Task 3: 结构与视觉验证

**Files:**
- Inspect: `Document/Update/v1.0.10 - research/beta4/织物垫实物研究汇报.pptx`
- Create: `Document/Update/v1.0.10 - research/beta4/output/ppt_preview/`

- [ ] **Step 1: 回读结构**

用 `python-pptx` 打开生成文件，断言幻灯片数量为 11，标题包含问题背景、问题剖析、横向算法调研、效果展示和决策建议。

- [ ] **Step 2: 检查元素边界**

遍历所有 shape，检查 `left/top/width/height` 不越出页面边界；检查字号不低于 10 pt。

- [ ] **Step 3: 尝试渲染预览**

若本机安装 PowerPoint，通过 COM 导出 PDF 或 PNG；否则保留结构检查结果并报告未进行像素级预览。

- [ ] **Step 4: 修正并重新验证**

若发现越界、文字过密或图片比例错误，修改 `build_presentation.py`，重新生成并重复上述检查。

### Task 4: 拆分实测效果页

**Files:**
- Modify: `Document/Update/v1.0.10 - research/beta4/scripts/build_presentation.py`
- Modify: `Document/Update/v1.0.10 - research/beta4/织物垫实物研究汇报.pptx`

- [ ] **Step 1: 恢复问题背景页**

第 3 页只保留问题、用户感知和研究目标，不再嵌入两张曲线图。

- [ ] **Step 2: 新增实测优化效果页**

第 4 页上下排列 `background_raw_vs_ema.png` 和 `background_raw_vs_ema_zoom.png`，只指定宽度、由图片原始比例自动计算高度；添加简短图注。

- [ ] **Step 3: 更新后续页码**

所有后续页码顺延，PPT 总页数调整为 12。

- [ ] **Step 4: 验证**

Run: `python "Document/Update/v1.0.10 - research/beta4/scripts/build_presentation.py"`

Expected: PPT 共 12 页；第 3 页只有 Logo 图片，第 4 页有 Logo、整体图、局部图共 3 张图片；两张曲线图的 PPT 显示比例与源图片纵横比误差小于 1%。
