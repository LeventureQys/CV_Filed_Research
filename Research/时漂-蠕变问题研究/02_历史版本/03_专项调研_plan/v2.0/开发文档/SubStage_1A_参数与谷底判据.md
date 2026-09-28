# SubStage 1A · 运行期参数结构体 + 谷底判据（默认参数下行为必须与现役逐帧一致）

## SubStage: 1A
- **所属 Stage**：Stage 1
- **依赖前置**：无
- **并行状态**：**独立可执行**（无需等待任何前置，可立即开工）
- **所属阶段**：阶段三 - 开发

> **你是 SubAgent**。开工前必须先完整阅读仓库根 `AGENTS.md`（含「Agent 命令执行预算与 PowerShell 约束」「验证与构建总则」「SubAgent 编译测试分工」），
> 并遵守本文全部约束。不确定就停下并向 MainAgent 报告，不要自行扩大范围。

---

## 1. 背景与目标

v6 时漂补偿算法（`src/domain/drift_v6/drift_v6_compensator.{h,cpp}`）的全部常量都是 `static constexpr`，
现场无法 A/B，只能靠离线原型复算（本轮已因此受限）。本 SubStage 新增一个**运行期可注入的参数结构体**，
并落地后续三项修复都要用的**谷底判据**（`valley_now_`）。**本 SubStage 自身不改变任何补偿行为**：
默认参数下，输出必须与改动前逐帧一致。

---

## 2. 当前代码状态（相对路径 + 行号）

| 位置 | 内容 |
|---|---|
| `src/domain/drift_v6/drift_v6_compensator.h` L59~L159 | `class DriftV6Compensator` 的 public 段：常量区（L76~L159 全部 `static constexpr`） |
| `.h` L161~L288 | private 段：`enum class State` L162 / `struct BufMask` L174 / `struct EventCtx` L180 / 私有函数 L207~L236 / 私有成员 L238~L287 |
| `.h` L290~L310 | `class DriftV6CompensationCoordinator` |
| `.cpp` L54~L88 | `DriftV6Compensator::Reset()`（逐字段清零） |
| `.cpp` L90~L99 | `ResetFor(int n)`（调 `Reset()` 后重建 Eigen 缓冲） |
| `.cpp` L240~L404 | `Process()`；`ts_smooth_` 更新在 L261-264，`eps` 在 L269，检测器从 L284 开始 |
| `.cpp` L158~L162（构造函数） | 无显式构造函数（默认构造） |

---

## 3. 必须实现的接口契约（逐字实现，名字不得改）

### 3.1 新增常量（`.h` public 常量区，紧跟 L159 之后或就近）

```cpp
    // ── plan-v2.0 新增（C/F2/F3/F5 的默认值；全部可被 Params 覆盖） ──
    static constexpr double kClampAlpha      = 0.005;  // C: 显示 ≤ 原始 + α·|原始|
    static constexpr int    kValleyWinN      = 4000;   // 谷底窗长度(帧) ≈40 s @100 Hz
    static constexpr double kValleyFrac      = 0.15;   // 谷底相对位置门
    static constexpr double kValleyRangeFrac = 0.05;   // 窗内幅度门(相对 w_max)
    static constexpr double kEventValleyMinS = 0.60;   // F2: 事件内谷底出口最小 τ
    static constexpr double kReanchorValleyS = 0.20;   // F3: 重锚前谷底持续时长
    static constexpr double kGNegFloor       = -0.05;  // F5: g 的负向下界
    static constexpr double kGValleyResetS   = 0.30;   // F5: 谷底归零 g 的持续时长
    // F2/F3 的整组开关。默认 false: 14 份数据回归显示这两项收益未证实、
    // 且在「变化负载/切换负载-快相无责」上方向不一致(见设计文档 §3.4 / results/design_regress_summary.md)。
    static constexpr bool   kLegacyFixesOn   = false;
```

### 3.2 `Params` 结构体（`.h` public 段，`class DriftV6Compensator` 内、`Process` 声明之前）

```cpp
    // 运行期可注入参数(默认值 = 既有 static constexpr, 语义见各常量注释)
    struct Params {
        double clamp_alpha       = kClampAlpha;
        double valley_win_s      = 40.0;          // 仅用于落盘/文档; 运行期用 valley_win_n
        int    valley_win_n      = kValleyWinN;
        double valley_frac       = kValleyFrac;
        double valley_range_frac = kValleyRangeFrac;
        double event_valley_min_s= kEventValleyMinS;
        double reanchor_valley_s = kReanchorValleyS;
        double g_neg_floor       = kGNegFloor;
        double g_valley_reset_s  = kGValleyResetS;
        bool   legacy_fixes_enabled = kLegacyFixesOn;   // F2/F3 整组开关(默认 false)
        double kappa_onset       = kKappaOnset;
        double kappa_restep      = kKappaRestep;
        double ho_min_s          = kHoMinS;
        int    revoke_hold_n     = kRevokeHoldN;
        // 诊断计数(只写不读业务逻辑)
        long n_valley_exit    = 0;
        long n_reanchor_idle  = 0;
        long n_g_floor        = 0;
        long n_g_valley_reset = 0;
    };
```

### 3.3 public 访问器（`.h`，紧接 `shape_hits()` L72 之后）

```cpp
    const Params& params() const { return params_; }
    void SetParams(const Params& p);   // 只改参数与计数，不改补偿状态(n_/state_/A_/g_/环缓)
```

### 3.4 private 成员与函数（`.h` private 段）

```cpp
    Params params_;
    // 谷底判据: 长度 kValleyWinN 的环形缓冲(存 ts_smooth_)
    double valley_buf_[kValleyWinN];
    long   valley_n_   = 0;      // 累计写入帧数(用于判断"数据是否足够")
    bool   valley_now_ = false;
    double valley_run_ = 0.0;    // 谷底连续时长(秒)
    void   UpdateValley(double dt);
```

### 3.5 `UpdateValley` 语义（`.cpp` 新增实现）

```text
输入: dt = 本帧时间步(已 clamp 到 [0, 0.1], 与本文件既有口径一致)
行为:
  1. 把本帧的 ts_smooth_ 写入环形缓冲: idx = valley_n_ % kValleyWinN; valley_buf_[idx] = ts_smooth_; ++valley_n_;
  2. 若 valley_n_ < kValleyWinN(数据不足):
         valley_now_ = false; valley_run_ = 0.0; return;
  3. w_min = min(全部 kValleyWinN 个槽) ; w_max = max(全部 kValleyWinN 个槽)
  4. eps_v = 1e-6 * (1.0 + |w_max|)
  5. valley_now_ = (ts_smooth_ < w_min + params_.valley_frac * (w_max - w_min + eps_v))
                   && ((w_max - w_min) > params_.valley_range_frac * std::max(std::abs(w_max), eps_v))
  6. valley_run_ = (valley_now_ && dt > 0.0) ? (valley_run_ + dt) : (valley_now_ ? valley_run_ : 0.0)
     // dt <= 0 时若仍处于谷底则保持不累加，否则清零
```

### 3.6 调用点（`.cpp` `Process`）

在 L269（`const double eps = ...`）之后、L271（3 帧中值总量块）之前，插入**一行**：

```cpp
    UpdateValley(dt);
```

**位置不许改**：必须在 `ts_smooth_` 更新（L261-264）之后、任何使用 `valley_now_` 的状态判断之前。

### 3.7 `Reset()` 补充（`.cpp` L54~L88 的末尾）

> ⚠️ **本节已被 SubStage 4A 的实测推翻（2026-09-19，见设计文档 §4.1.1 C-5）**：
> 原稿要求 `Reset()` 里写 `params_ = Params();`，但那会让"首帧之前注入的参数"
> 在首帧的 `ResetFor → Reset` 中被静默抹掉（协调器叠加注入也救不回来）。
> **最终落定**：`Reset()` **不**执行 `params_ = Params();`，只清补偿状态与谷底状态。

```cpp
    valley_n_ = 0;
    valley_now_ = false;
    valley_run_ = 0.0;
```

（`valley_buf_` 不必清零——`valley_n_` 归零后不会读到陈旧槽。）

---

## 4. 输入输出

- **输入**：无外部数据；纯代码改动。
- **输出**：`.h` / `.cpp` 修改后的可编译源码。
- **行为要求**：默认参数下 `Process` 的输出**逐帧与改动前完全一致**（`valley_now_` 尚无消费者）。

---

## 5. 验收标准（可测试、可观察）

| # | 标准 | 判定方式 |
|---|---|---|
| A | `cmake --build build-app --config Debug --target TactileSense -- /m:4` **exit 0** 且无新增 warning | 命令输出 |
| B | **默认参数下行为零差**：用 `tests/fixtures/drift_v6/seg_b_handoff_pre.csv` 回放，改动前后的输出逐帧差 = 0 | 见下「自测脚本」 |
| C | `params()` 返回默认值（`clamp_alpha==0.005`、`valley_win_n==4000`、`g_neg_floor==-0.05`、**`legacy_fixes_enabled==false`**） | 自测脚本断言 |
| D | `SetParams` 后再 `Reset()`，`params()` 回到默认值 | 自测脚本断言 |
| E | `valley_now_` 在数据不足（< 4000 帧）时恒为 `false` | 自测脚本断言（前 3999 帧） |

### 自测脚本（**必须实际执行并贴出输出**）

在 `tests/data/test_drift_v6_compensator.cpp` **尚未建立前**（本 SubStage 属 Stage 1，测试骨架由 1B 建立），
用一次性临时 main 验证 A/B：在 `temp/v4.1flash/plan/v2.0/__scratch_1a/` 下新建一个**不参与主工程**的小 CMake 工程，
直接编译 `src/domain/drift_v6/drift_v6_compensator.cpp` + Eigen（include 路径 `thirdparty/eigen-5.0.0`），
读 `tests/fixtures/drift_v6/seg_b_handoff_pre.csv` 逐帧喂入，打印输出总量。
**零差对照**：先用 `git stash` 前的内容/或用 `git show HEAD:src/domain/drift_v6/drift_v6_compensator.cpp` 取出原版，
在同一个临时工程里编两个 main 对比（或先把原版输出存成 txt，再改代码后重跑对比）。
**完成后必须删除 `__scratch_1a/`**（它是临时验证工程，不是交付物；删除前在报告里写明理由）。

---

## 6. 禁止事项

1. **禁止**改动 §3.5~§3.7 之外的任何逻辑（检测器、状态机、慢相、限幅、κ、ROM、`idle_now`）。
2. **禁止**修改 `src/` 下除 `drift_v6_compensator.{h,cpp}` 之外的任何文件。
3. **禁止**修改 `temp/v4.1flash/progress/**`（归档只读）与 `plan/v1.0/**`。
4. **禁止**新建构建树；**禁止**不带 `--target` 的构建 / `ALL_BUILD` / 全量 ctest。
5. **禁止**构建或运行任何 `test_*` 目标（本 SubStage 不加测试）。
6. **禁止**改 `tests/` 下任何既有文件（测试骨架由 SubStage 1B 负责）。
7. 不确定就停并报告，不要猜。

---

## 7. 构建要求（必须照此执行）

- 只构建主程序目标：`cmake --build build-app --config Debug --target TactileSense -- /m:4`
  （主程序构建权限说明：本 SubStage 需要确认改动可编译；若 MainAgent 已在跑同一目标构建，则等待其完成，**不要并发**。）
- 先确认 `build-app` 已存在：`Get-ChildItem -Directory -Filter 'build*'`。**不要**重新 configure（未增删源文件）。
- 若 `.cpp`/`.h` 变动触发重编较多翻译单元，用 `run_in_background` + 60~120 s `job_output` 轮询。

---

## 8. 完成报告要求

报告必须包含：① 改动文件与行号；② §5 五项验收标准的**实测结果与命令输出**；③ 零差对照的具体做法与结果；
④ 临时工程是否已删除；⑤ 任何不确定项。**禁止**只写"编译通过/看起来正常"。
