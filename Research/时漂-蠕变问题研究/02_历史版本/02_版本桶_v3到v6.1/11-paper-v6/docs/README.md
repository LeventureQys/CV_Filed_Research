# paper_v6 —— 抗蠕变补偿算法（v6）论文工作区

> 任务：把 `temp/v4.1flash/` 里与 **v6 算法**（即"最终画图/最终交付"的那一版）相关的
> **纯抗蠕变算法部分**节选出来，写成一篇论文；并完整写出**为拟合实机数据所做的全部 trade-off**。
> 日期：2026-09-19 · 执行：DSH Agent

## 交付物

| 文件 | 内容 |
|---|---|
| **`抗蠕变补偿算法_v6.md`** | **论文正文**（约 4.5 万字，8 章 + 3 附录，7 张图） |
| `figures/F1~F7.png` | 全部配图（Python 重绘，每张都经 DeepSeek vision 模型检查）。**图号 = 正文出现顺序**（F1 §1 / F2 §3 / F3 §5 / F4 §6 / F5 §7.4 / F6 §8.1 / F7 §8.3），脚本 `pfN_*.py` 与图号一一对应，出图目录也是本 `figures/` |
| `results/metrics_all.csv` | 13 份录制 × 5 臂（raw / 无责1s / 无责3s / v6 / v6+trim）的全部指标 |
| `results/metrics_settle.csv` | 首次加载的 T_stable / T_band / T_flat30 / err5s |
| `results/trim_ablation.csv` | A 慢修正三档消融（绝对平 vs 绝对准的量化依据） |
| `results/epochs_all.csv` | 每个 epoch 的起点与工况种类（事件台账） |
| `results/paper_tables.txt` | 正文引用的汇总表（脚本自动生成） |
| `results/vision_check.md` / `.json` | **7 张图的 vision 检查记录**（模型：`deepseek-v4-flash-vision-exp`） |
| `results/cache/*.npz` | 逐帧曲线 + 事件内部状态轨迹（供画图复用，无需重算） |
| `scripts/` | 全部可复现脚本（见下） |

## 一句话结论

**v6 只换掉了"快相处理"这一段**（把"等快相走完"换成"用标定形状把快相算掉"），
慢相蠕变模块逐行沿用 v5；收益是首次加载平稳时刻 **3.82 s → 0.55 s**，
代价是阶跃保真 1.00 → 1.08、epoch 数 1.6~1.7 倍、实录全程最大偏差 1889 → 4583 ADC。
**单次加载/长保压类工况可替代现役实现，多次变载的实录类工况还不能。**

## 脚本

```powershell
$env:PYTHONIOENCODING='utf-8'      # 控制台中文/特殊符号需要

python scripts/pv_run.py              # ① 主体复算（13×5 臂，≈5 分钟）→ metrics_*.csv + cache/
python scripts/pv_trim_ablation.py    # ② trim 三档消融（≈4 分钟）→ trim_ablation.csv

python scripts/pf1_physical.py        # 图 F1 加载形状的实测事实（§1）
python scripts/pf2_overview.py        # 图 F2 v6 算法总览（§3）
python scripts/pf3_invert_glide.py    # 图 F3 形状约束反演与滑行器（§5）
python scripts/pf4_slow_creep.py      # 图 F4 慢相蠕变估计与扣除（§6）
python scripts/pf5_slow_tradeoff.py   # 图 F5 慢相取舍（§7.4）
python scripts/pf6_overview13.py      # 图 F6 13 份录制总览（§8.1）
python scripts/pf7_fast_tradeoff.py   # 图 F7 快相取舍（§8.3）

python scripts/pv_vision_check.py     # ③ 图件视觉验收（DeepSeek vision 逐张）
python scripts/pv_audit.py            # ④ 论文数字审计（期望 OK 74 / BAD 0）
```

依赖：Python 3.14 + numpy 2.4.6 / scipy 1.17.1 / pandas 3.0.3 / matplotlib 3.10.9（本机已有）。
密钥：`pv_vision_check.py` 从 `~/.dsh/.credentials.yaml` 读取 `DEEPSEEK_API_KEY`，不硬编码。

### 图件检查口径（两层）

1. **机器自检**（出图脚本内置 `figcheck`）：面板数、空白面板、**文字**越界、同面板文本重叠
   —— 只对文本判越界，因为线条/矩形的 bounding box 反映数据范围而非可见范围，会产生大量假阳性；
2. **视觉验收**（`pv_vision_check.py`）：把图与「期望描述」一起送 DeepSeek vision 模型，
   要求逐项回答面板数、文字重叠、文字截断、空白面板、缺曲线、图例遮挡、乱码。
   最新一次：**7 张全部 PASS**（记录见 `results/vision_check.md`）。

## 与其他目录的关系

| 目录/文档 | 关系 |
|---|---|
| `temp/v4.1flash/scripts/glm53_v6.py` | v6 原型（本文算法的唯一实现来源；慢相模块逐行沿用 `glm53_v51.py`） |
| `temp/v4.1flash/Document/07-v6算法说明.md` | v6 规格 + 四轮实测修正；**本文数字与该文 §12.6 末表逐项一致** |
| `temp/v4.1flash/paper/` | 无责 3 s/5 s 路线的论文（6 图）。本文与其**并列**，口径一致（同时间轴、同指标定义），互不覆盖 |
| `temp/v4.1flash/v6与免责1s3s对比.md` | v6 与免责 1 s/3 s 的对比报告；本文是其论文化重写 + 取舍台账化 |

## 未做 / 未验证

1. **未修改 `src/` 任何代码**（全部为离线分析与文档）；
2. **未做真机 / 界面手测**（按项目规范交用户）；
3. **未做参数扫描**：v6 的新参数只有检测窗做过扫描（§7.2），`TAU_REF`/`κ`/`TAIL_GATE`/`HO_MIN` 等
   仍是一次粗定值；
4. **形状库使用测试集自身标定**（13 份合并中位），属**乐观上界**；
5. 时间轴沿用 `ad_lib.prep` 的 timestamp 插值口径，对 $\tau<0.2$ s 有畸变——v6 比 v5 更吃
   时间轴精度，故本文 v6 数字可能**低估**其真实能力。
