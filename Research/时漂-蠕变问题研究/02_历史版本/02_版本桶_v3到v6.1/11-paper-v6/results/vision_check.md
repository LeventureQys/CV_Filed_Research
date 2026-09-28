# DeepSeek vision 图件检查记录

模型：`deepseek-v4-flash-vision-exp`；检查时间：2026-09-18 16:58:32

| 图 | 判定 | 面板数 | 文字重叠 | 文字截断 | 空白面板 | 乱码 | 结论 |
|---|---|:--:|---|---|---|---|---|
| F1_two_phase.png | PASS | 4 | 0 | 0 | 0 | False | 四面板图均完整呈现，中文标签清晰无乱码，各面板元素与期望描述一致。 |
| F2_overview.png | PASS | 7 | 0 | 0 | 0 | False | 该图完整展示了v6算法总览，包含逐帧数据流、六类工况、慢相蠕变模块及三条输出链路，文字清晰无重叠，箭头连接明确。 |
| F3_invert_glide.png | PASS | 4 | 0 | 0 | 0 | False | 图表为2×2面板，包含原始读数与反演目标对比、输出随时间变化、逐帧修正量轨迹及检测时延统计，中文标签清晰无乱码，曲线区分明显。 |
| F4_slow_creep.png | PASS | 4 | 0 | 0 | 0 | False | 该图完整呈现了慢相蠕变估计与扣除的四个面板，包含残差分布、g 轨迹、γ 分布及扣除量限幅示意，中文标签清晰无乱码。 |
| F5_overview13.png | PASS | 13 | 0 | 0 | 0 | False | 13 格小倍数总览布局正确，右下 3 格留空，各面板曲线与中文标题均完整清晰，无乱码或截断。 |
| F6_fast_tradeoff.png | PASS | 4 | 0 | 0 | 0 | False | 图表包含四个面板，分别展示了分组柱状图、加载沿放大曲线、双轴柱状与散点图以及epoch数对比，所有中文标签和图例均清晰可辨，无重叠或截断问题。 |
| F7_slow_tradeoff.png | PASS | 4 | 0 | 0 | 0 | False | 图表包含四个面板，分别展示了平台静态偏置、慢修正死区、trim档位消融和取舍平面，中文标签清晰无乱码，无文字重叠或截断。 |

## 逐图明细

### F1_two_phase.png
```json
{
  "panels_ok": true,
  "panel_count": 4,
  "text_overlap": [],
  "text_clipped": [],
  "empty_panel": [],
  "missing_or_broken": [],
  "legend_covers_data": false,
  "garbled_text": false,
  "verdict": "PASS",
  "summary": "四面板图均完整呈现，中文标签清晰无乱码，各面板元素与期望描述一致。",
  "issues": [],
  "_file": "F1_two_phase.png",
  "_bytes": 256218,
  "_secs": 1.6
}
```

### F2_overview.png
```json
{
  "panels_ok": true,
  "panel_count": 7,
  "text_overlap": [],
  "text_clipped": [],
  "empty_panel": [],
  "missing_or_broken": [],
  "legend_covers_data": false,
  "garbled_text": false,
  "verdict": "PASS",
  "summary": "该图完整展示了v6算法总览，包含逐帧数据流、六类工况、慢相蠕变模块及三条输出链路，文字清晰无重叠，箭头连接明确。",
  "issues": [],
  "_file": "F2_overview.png",
  "_bytes": 319171,
  "_secs": 1.8
}
```

### F3_invert_glide.png
```json
{
  "panels_ok": true,
  "panel_count": 4,
  "text_overlap": [],
  "text_clipped": [],
  "empty_panel": [],
  "missing_or_broken": [],
  "legend_covers_data": false,
  "garbled_text": false,
  "verdict": "PASS",
  "summary": "图表为2×2面板，包含原始读数与反演目标对比、输出随时间变化、逐帧修正量轨迹及检测时延统计，中文标签清晰无乱码，曲线区分明显。",
  "issues": [],
  "_file": "F3_invert_glide.png",
  "_bytes": 203875,
  "_secs": 1.9
}
```

### F4_slow_creep.png
```json
{
  "panels_ok": true,
  "panel_count": 4,
  "text_overlap": [],
  "text_clipped": [],
  "empty_panel": [],
  "missing_or_broken": [],
  "legend_covers_data": false,
  "garbled_text": false,
  "verdict": "PASS",
  "summary": "该图完整呈现了慢相蠕变估计与扣除的四个面板，包含残差分布、g 轨迹、γ 分布及扣除量限幅示意，中文标签清晰无乱码。",
  "issues": [],
  "_file": "F4_slow_creep.png",
  "_bytes": 208477,
  "_secs": 1.7
}
```

### F5_overview13.png
```json
{
  "panels_ok": true,
  "panel_count": 13,
  "text_overlap": [],
  "text_clipped": [],
  "empty_panel": [],
  "missing_or_broken": [],
  "legend_covers_data": false,
  "garbled_text": false,
  "verdict": "PASS",
  "summary": "13 格小倍数总览布局正确，右下 3 格留空，各面板曲线与中文标题均完整清晰，无乱码或截断。",
  "issues": [],
  "_file": "F5_overview13.png",
  "_bytes": 356360,
  "_secs": 1.8
}
```

### F6_fast_tradeoff.png
```json
{
  "panels_ok": true,
  "panel_count": 4,
  "text_overlap": [],
  "text_clipped": [],
  "empty_panel": [],
  "missing_or_broken": [],
  "legend_covers_data": false,
  "garbled_text": false,
  "verdict": "PASS",
  "summary": "图表包含四个面板，分别展示了分组柱状图、加载沿放大曲线、双轴柱状与散点图以及epoch数对比，所有中文标签和图例均清晰可辨，无重叠或截断问题。",
  "issues": [],
  "_file": "F6_fast_tradeoff.png",
  "_bytes": 234591,
  "_secs": 2.2
}
```

### F7_slow_tradeoff.png
```json
{
  "panels_ok": true,
  "panel_count": 4,
  "text_overlap": [],
  "text_clipped": [],
  "empty_panel": [],
  "missing_or_broken": [],
  "legend_covers_data": false,
  "garbled_text": false,
  "verdict": "PASS",
  "summary": "图表包含四个面板，分别展示了平台静态偏置、慢修正死区、trim档位消融和取舍平面，中文标签清晰无乱码，无文字重叠或截断。",
  "issues": [],
  "_file": "F7_slow_tradeoff.png",
  "_bytes": 241774,
  "_secs": 2.0
}
```

---

---

## 附注（2026-09-18 归档后修订）

以上记录是**重编号前**的原始检查结果（文件名保留当时的名字），检查结论对图的内容依然有效。
此后按「图号 = 正文出现顺序」重排了三张图的编号，并重新出图（图内标题里的图号同步更新）：

| 检查时的文件名 | 重编号后 | 正文位置 |
|---|---|---|
| `F5_overview13.png` | `F6_overview13.png` | §8.1 复算结果 |
| `F6_fast_tradeoff.png` | `F7_fast_tradeoff.png` | §8.3 首次加载的响应 |
| `F7_slow_tradeoff.png` | `F5_slow_tradeoff.png` | §7.4 取舍平面 |

F1~F4 未改名。重出图只改了输出文件名与图内标题里的图号，绘图逻辑与数据未变；
`results/vision_check.json` 同为重编号前的原始记录，不再回写。
重出图后未重跑 vision 验收（需 `DEEPSEEK_API_KEY`），逐图 `figcheck`（面板数/越界/文字重叠）通过。
