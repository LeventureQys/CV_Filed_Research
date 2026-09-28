# 02_现场调参 —— K9 的调参台账（2026-09-21 ~ 09-27）

> 这一层是**当前算法的参数从哪来**的完整留痕。所有结论都是**离线复算**得到的，
> 复算器与产品源码 `creep_observer.cpp` 逐帧对拍过（多数会话 parity 均差 ≤0.1%、最大 ≤1.8 ADC）。
> **未改源码、未构建、未跑测试**（除工程 A 的离线扫参助手）。
>
> **要拿到"直接写进设备"的值 → 看 §1 的首件参数表。**
> **要复现某个数字 → 看 §3 / §4 的工程与脚本清单。**

---

## 1. 最终产物：首件参数表（2026-09-27）

`09-27_首件参数列表/9.27 首件算法参数列表.xlsx`

| 位置 | r_fast | tau_c_fast_s | slow_confirm_s | soft_unfreeze_s | slope_cap_frac | r_slow_max | tau_r_fast_s | tau_r_slow_idle_s |
|---|---|---|---|---|---|---|---|---|
| 四指指腹 / 拇指指腹 | 0.03 | 2 | 3 | 2 | 0.02 | 0.1 | 0.1 | 0.2 |
| 左手掌 | 0.04 | 1 | 1.5 | 1.5 | 0.05 | 0.15 | 6 | 0.5 |
| 右手掌 | 0.02 | 1 | 1.5 | 1.5 | 0.09 | 0.2 | 2 | 0.5 |
| 四指指尖 | N/A —— **待标定** | | | | | | | |

写入路径：补偿算法选择 → v3.4 观测器 → **参数...** →「**恢复默认参数**」→ 逐项填成上表 → 「**写入设备**」。
寄存器（200 使能 / 201 r_fast / 202 slow_confirm / 203 soft_unfreeze / 204 τrsi / 205 τr1 / 206 τc1 / 207 cap ×1000 / 208 rsm）
见 `../00_源码本体/参数与寄存器表.md` §4。

---

## 2. 三个子目录 / 三个交付物

| 目录 | 时间 | 是什么 | 主要产物 |
|---|---|---|---|
| **`09-21_09-26_工程A_v3.4/`** | 09-21 ~ 09-26 | **工程 A**：v3.4/K9 的**第一轮系统调参**（长期录制 + 多传感器分类扫参） | `out/` 的对比图与 `long_*` 判定；`sensor_tune/out/` 的分类预设（09/16/21 号 JSON）；`长期数据/` 4 份 09-22 长录 |
| **`09-26_09-27_工程B_手掌/`** | 09-26 ~ 09-27 | **工程 B**：**手掌专题**（ADC 模式 → 力值 N 模式）+ `palm4`~`palm10` 七轮工作区 | 9 份结论文档（见 §4.1）+ `figures/` 9 张对照图 |
| **`09-27_首件参数列表/`** | 09-27 | **交付**：首件（产线首件）参数表 xlsx | §1 的表 |
| `09-26_传感器分类调参交付包/` | 09-26 | 工程 A 的打包交付（zip，148 条目：图 + 预设 JSON + README） | `传感器分类调参_20260926.zip` |

---

## 3. 工程 A（`09-21_09-26_工程A_v3.4/`）

### 3.1 工具链

| 工具 | 作用 |
|---|---|
| `creep_observer_k9.py` | **K9 的 Python 参考实现**（当时的转写版） |
| `run_v34_on_csv.py` | 用 `obs_runner.exe`（链接产品源码的 C++ 离线助手）复放任意会话 CSV |
| `sensor_tune/v34_sweep.cpp` + `build/*.exe` | **离线扫参助手**：链接 `src/domain/drift_v6/creep_observer.cpp`，一次吃一份会话 + 一列参数集。`v34_sweep.exe`（K9）、`v34_sweep_v3.exe`（v3 列）、**`v34_sweep_k11.exe`（K10/K11 候选）** |
| `sensor_tune/sensor_common.py` | 数据装载 / 指标 / 公共层 |
| `drift_v6_py.py` | v6 的 Python 实现（对照用） |

### 3.2 脚本分组（根目录 39 个 `.py`）

| 组 | 脚本 | 主题 |
|---|---|---|
| 基线对照 | `compare_v34_v6.py`、`compare_params_on.py`、`tune_v34.py` | v3.4 vs v6 / 参数开关对照 |
| 长期保压 | `longterm_test.py`、`verdict_longterm.py`、`long_hold_drift.py`、`red_line_drift.py`、`x2_upward_drift.py` | 超长保压下的下漂/上漂定因（`长期数据/` 的 4 份会话） |
| 卸载与震荡 | `unload_osc_test.py`、`sc2_test.py`、`shift_x2_earlier.py` | 卸载衔接、x2 提前启动 |
| 另一路录制器 | `other_recorder_tune.py`、`other_recorder_param_scan.py`、`other_recorder_optimal.py`、`other_recorder_best_plot.py` | `other_recorder/` 会话的调参与最优解 |
| 新会话 | `new_session_analyze.py`、`new_session_plot.py` | 09-24 新录制分析 |
| 放大与标注 | `zoom_100539_seg1.py`、`zoom_100539_annotated.py`、`checkpoints.py` | 局部放大、加载后漂移检查点 |
| 四指指腹 | `data_finger_{params,sweep,pick,conservative,plot}.py` | 四指指腹档的取参 |
| 合成与结构 | `synth_converge.py`、`leak_prototype.py`、`time_params_map.py` | 合成激励收敛实验、时间参数地图 |
| K9 收口 | `final_test.py`、`final_params_check.py` | 「最终 5 参」/ 8 参的验收 |

### 3.3 `sensor_tune/`（分类扫参流水线）

编号脚本即执行顺序（`01_inventory` → `33_trackers`），产物在 `sensor_tune/out/`：

| 阶段 | 脚本 | 产物 |
|---|---|---|
| 盘点/画像 | `01_inventory` `02_profiles` `03_numbers` `04_rise_shape` `05_prep` | `01~05_*.json/txt` |
| 一维/网格扫参 | `06_sweep1d` `08_search` `26_refine` `29_search_4f` `24_search_trim` | `03/04/08/16/18_*.json` |
| 诊断 | `07_diag` `11_ceiling` `12_variant` `18_nozero` `19_ui_gap` `20_param_gap` `30_notch_attr` | 天花板、UI 覆盖缺口、参数缺口 |
| 定档 | `21_package` `25_pick` `27_focus` `28_finalize` `16_presets` `15_final` `17_check_figs` | `09_presets*.json`、交付 zip |
| 出图 | `09_plot` `10_report` `22_trim` `23_trace` `31_anchor` `32_anchor_dbg` `33_trackers` | `figures/`、`figures_archive/` |
| 校验 | `13_verify` `14_session_bin` `_npzcheck.txt` | 校验留痕 |

> **中间产物体量**：`out/_{raw,paramsets,sess,prep}/` 约 **3 万个文件 / 50 MB**（`_sess` 133 MB 为会话缓存）。
> **可整目录删除**，重跑 `01→05` 即可重建。

---

## 4. 工程 B（`09-26_09-27_工程B_手掌/`）

### 4.1 九份结论文档（按时间顺序）

| 文档 | 结论一句话 |
|---|---|
| `手掌_力值尺度_回调定因与调参.md` | 建立 **1 N = 1000 ADC** 映射；回调定因 |
| `手掌_力值尺度_回调修复终版.md` | 力值尺度回调的修复档 |
| `手掌_ADC模式_过减修复结论.md` | **过减定因**：把力值档原样搬进 ADC 模式（且 `r_fast` 填成 0.3 而推荐表是 0.03） |
| `v3.4_8参数_纯结构研究.md` | 抛弃推荐值，OFAT + rf×cap 90 点网格：给出**每个旋钮的杠杆跨度与死区/饱和** |
| `手掌_152928_上浮定因与推荐参数.md` | "慢态过调却仍上浮" = 观感过调、实质欠调（`r_slow_max=0.03` 把 x2 封顶） |
| `手掌_161747_长保压验证与终版参数.md` | 429 s 长保压验证与终版参数 |
| `手掌_161747_尾漂定因与最小处方.md` | 尾端漂移定因 + **最小参数处方** |
| `手掌_右手掌164332_复检与二次调参.md` | 另一份蠕变形态完全不同的按压（早段 770 ADC/s）暴露 `cap=0.005` 速率上限不够 |
| **`手掌_力值模式_8参数终版与代价.md`** | ★ **本轮总报告**：推荐档 B、全部取舍代价、结构性结论逐条实测证据、备选档 A/C、跨显示模式警告 |

### 4.2 工作区 `palm4` ~ `palm10`

每个 `palmN/out/` 是一轮实验的数值底座（`*.npz` / `*.json` / `*.bin`）。脚本按轮次编号：

| 脚本号段 | 轮次 |
|---|---|
| `90~95` | `palm4` ADC 模式过减 |
| `96~99` | `palm5` |
| `100~102` | `palm5` 合成/汇总 |
| `103~107` | `palm6` 力值尺度 |
| `108~109` | `palm6` 参数空间结构 |
| `110~114` | 长保压 / 慢相专项 / 折中 |
| `120~129` | `palm7` 力值模式（含 `91_observer.py` 移植器、`124_force_parity.py` 三方对拍） |
| `130~142` | `palm8` 细网格 / 结构结论 / 终版表 / 出图 |
| `150~156` | `palm10` b2d896 长保压尾漂 |
| `128~141`（另一支） | `palm9` |

### 4.3 关键方法与可信度

- **复算器**：`scripts/91_observer.py` 是 `creep_observer.cpp::Process()` 的忠实移植；
  另有链接产品源码的 C++ 离线复算器。**两者逐帧差 0.00000 N**。
- **三方对拍**：`124_force_parity.py`（复算器 vs Python 移植 vs 录制 seg 流）——加载段内 |差| ≤ 0.0094 N。
  **唯一大差在全卸载窗**（设备按 K7 空载直通整帧不扣，复算按状态泄放扣了）——该窗不影响任何保压段指标。
- **必须知道的两个数据坑**：`elapsed` 列约 70% 帧重复，须按 ~100.5 Hz 重建均匀时间轴；
  录制显示流有单帧尖峰，峰值类指标先做 0.3 s 中值滤波。

---

## 5. 复现命令

```powershell
# 工程 A：核心复算（需先构建 sensor_tune/build/v34_sweep*.exe）
cd 09-21_09-26_工程A_v3.4
python tune_v34.py
python final_test.py          ; python final_params_check.py
python longterm_test.py       ; python verdict_longterm.py

# 工程 A：分类扫参（约 10 分钟，产物在 sensor_tune/out/）
cd sensor_tune
python 01_inventory.py   ; python 05_prep.py
python 06_sweep1d.py     ; python 08_search.py
python 21_package.py     # 生成 09-26_传感器分类调参交付包\传感器分类调参_20260926.zip

# 工程 B：力值模式总流程（顺序见 手掌_力值模式_8参数终版与代价.md §九）
cd 09-26_09-27_工程B_手掌\scripts
python 121_force_prep.py ; python 124_force_parity.py
python 130_force_search2.py ; python 139_force_matrix.py
python 141_final_table.py   ; python 140_claims.py ; python 142_force_plot.py
```

> 依赖：Python 3 + numpy / scipy / pandas / matplotlib；C++ 离线助手需 MSVC 2022 + 仓库 `src` 与 `third_party/eigen-5.0.0`。
> **脚本里的路径锚点已按本次重构修正 6 处**（见根 `README.md` §8）；以 `__file__` 为基准的锚点随目录整块移动仍然有效。

---

## 6. 跨工程的注意事项

1. **工程 B 依赖工程 A 的扫参 exe**：`scripts/122_force_lib.py` 指向
   `../09-21_09-26_工程A_v3.4/sensor_tune/build/v34_sweep_k11.exe`（本次重构已同步修正）。
2. **`v34_sweep_k11.exe` 链接的是 K10/K11 版 `creep_observer.h`**，而产品源码里的 K10/K11
   **已在 2026-09-26 15:02 被回退**（详见根 `README.md` §7 第 1 条）。
   ⇒ 若要重跑工程 B 的力值流程并链接当前源码，请改用 `build_v34_sweep_v3.bat` 重新编译，
   **不要假设 exe 与当前 `src` 一致**。
3. **`data/` 与 `other_recorder/` 保持原位**（脚本用硬编码绝对路径读取）。
4. 这三份文档是同一批数据的三种口径（ADC / 力 / 合成台），**引用数字时必须连口径一起引用**——
   典型例子：`E` 的定义从"台阶后 0.1~0.6 s 最小值"改成"加载斜坡结束后 0.35~1.0 s 的平台"，落点会差 **2.7 N**。
