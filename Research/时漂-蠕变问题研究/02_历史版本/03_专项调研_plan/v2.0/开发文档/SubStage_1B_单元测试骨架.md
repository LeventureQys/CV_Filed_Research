# SubStage 1B · 单元测试骨架（夹具加载 + 注册进 data 套件）

## SubStage: 1B
- **所属 Stage**：Stage 1
- **依赖前置**：SubStage 1A（**需等待**：1A 完成后，`.h` 中才会出现 `Params` 与 `params()`）
- **并行状态**：**需等待依赖**（在 1A 交付后开工；可与 SubStage 2A/2B 的**准备**并行，但不得先改 `.cpp`）
- **所属阶段**：阶段三 - 开发

> **你是 SubAgent**。开工前必须先完整阅读仓库根 `AGENTS.md`（含「Agent 命令执行预算」「验证与构建总则」「SubAgent 编译测试分工」），并遵守本文全部约束。

---

## 1. 背景与目标

`tests/` 下**没有任何**引用 `drift_v6` 的测试目标（`tests/suites.cmake` 无登记），v6 落地至今 0 单元测试。
本 SubStage 建立测试骨架：**加载实机夹具 → 逐帧喂入补偿器 → 断言既有行为**。
本 SubStage **只加测试，不改产品代码**（产品改动在 1A/2A/2B）。

---

## 2. 当前代码状态

| 位置 | 内容 |
|---|---|
| `tests/fixtures/drift_v6/` | 已生成：`seg_a_onset_{pre,main}.csv`(1005 帧)、`seg_b_handoff_{pre,main}.csv`(221 帧)、`seg_c_cycle_{pre,main}.csv`(945 帧)、`README.md`。**约 20 Hz 抽样**（`--stride 5`） |
| `tests/fixtures/drift_v6/README.md` | 夹具来源、口径与限制（**开工必读**） |
| `tests/CMakeLists.txt` L388 / L411~L423 | 既有「夹具目录 + `target_compile_definitions(... TEST_FIXTURE_DIR=...)`」的写法，**照此模式** |
| `tests/suites.cmake` L18~L33 | 套件 id 注册表；`data` 套件在其中 |
| `tests/data/test_data_handler.cpp` | 同目录既有测试的写法参考（gtest） |
| `src/domain/drift_v6/drift_v6_compensator.h` | 被测类；`Process(double, Eigen::VectorXd&)` 为唯一入口（1A 之后另有 `params()`/`SetParams`） |

---

## 3. 必须实现的交付物

### 3.1 新文件 `tests/data/test_drift_v6_compensator.cpp`

要求：
1. **夹具解析必须按列位置**：`*_pre.csv` 的 `##Data` 表头**缺 21 个通道名**（已知交付缺陷，见夹具 README），
   数据行有 24 列 ⇒ 通道 = 数据行的**第 3..23 列**（0-based：3+0 .. 3+20）。
   必须自行定位 `##Data` 行（不要写死行号），并断言每行列数 == 24。
2. **时间轴用 `elapsed`（第 2 列）**，单位为秒。
3. 喂入方式：`Eigen::VectorXd v(21); ...; comp.Process(t, v);`，每帧取 `out = v`（`Process` 就地改写）。
4. 本 SubStage 至少实现下列用例（**全部用 `EXPECT_*`，不得只用 `ASSERT` 提前退出**）：

| 用例名 | 目的 | 期望 |
|---|---|---|
| `FixtureLoadsAndShapeIsConsistent` | 夹具自检 | `seg_b_handoff` 221 帧、`seg_a_onset` 1005 帧、`seg_c_cycle` 945 帧；每帧 21 通道；时间戳单调不减 |
| `IdlePassThrough` | 空载直通 | 对 `seg_a_onset` 的前 100 帧（全空载），输出逐帧等于输入（容差 1e-9） |
| `LoadCausesCompensation` | 加载后出现补偿 | 对 `seg_a_onset` 全段：存在帧使 `out_sum < in_sum − 100`（补偿为负 = 显示被压低） |
| `CompensationNeverExceedsRaw` | 限幅不变量（本版新语义的**前置**断言） | 全段 `out_i ≤ in_i + 0.005·|in_i| + 1e-6`（SubStage 2A 完成后该断言必须成立；**2A 之前会失败，属预期** —— 用 `GTEST_SKIP()` 或 `#if` 或按 `params().clamp_alpha` 判定后再断言） |
| `ResetClearsState` | 复位 | `Reset()` 后重新喂 `seg_b_handoff` 前 50 帧，输出与首次喂前 50 帧逐帧相同 |
| `ChannelCountChangeResets` | 通道数变化 | 21 通道喂 50 帧 → 换 15 通道喂 50 帧不崩溃；再换回 21 通道，输出等于首次 |

> **注**：`CompensationNeverExceedsRaw` 的正确写法是
> `if (comp.params().clamp_alpha > 0.0) { EXPECT_LE(out_i, in_i + comp.params().clamp_alpha * std::abs(in_i) + 1e-6); }`
> —— 这样在 1A（限幅尚未接入）时该断言会被跳过而用例仍通过，2A 之后开始真正生效。

### 3.2 `tests/CMakeLists.txt` 追加（照既有模式，勿改其它目标）

```cmake
add_tactile_test(test_drift_v6_compensator data/test_drift_v6_compensator.cpp)
if(TARGET test_drift_v6_compensator)
    target_compile_definitions(test_drift_v6_compensator PRIVATE
        DRIFT_V6_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/drift_v6")
    target_link_libraries(test_drift_v6_compensator PRIVATE ...)   # 按既有目标的链接写法补全
endif()
```
（`add_tactile_test` 的既有用法与必需链接库请**照 `tests/CMakeLists.txt` 中同套件目标原样抄**，
不要自创 helper；若 `add_tactile_test` 已自动链接主库，则不要重复链接。）

### 3.3 `tests/suites.cmake` 登记

把 `test_drift_v6_compensator` 加入 **`data`** 套件的目标列表（`tests/suites.cmake` 中 `data` 段落）。
**登记遗漏不会丢目标，但 configure 会打印提醒**，必须消除该提醒。

---

## 4. 接口契约（供断言使用；与 SubStage 1A/2A/2B 共用的定义）

```text
夹具文件格式（UTF-8）：
  ... 头部以 "##Session" 开头，键值行 ...
  ##Data
  timestamp,elapsed,frame_index[,ch0..ch20 或 缺省]
  174016.336635,0.026000,0,53.0,7.0,...        ← 24 列
读取约定：定位 "##Data" 行；数据行 = 其后的非空行；通道 = 列 3..23；时间 = 列 1(elapsed)。

补偿器接口（阶段三完成后）：
  void Process(double timestamp_s, Eigen::VectorXd& values_io);   // 就地改写
  void Reset();
  const Params& params() const;                                   // 1A 新增
  void SetParams(const Params&);                                  // 1A 新增
```

---

## 5. 验收标准

| # | 标准 | 判定 |
|---|---|---|
| A | `cmake --build build-tests-all --config Debug --target test_drift_v6_compensator -- /m:4` **exit 0** | 命令输出 |
| B | `ctest --test-dir build-tests-all -C Debug -R "^test_drift_v6_compensator\." --output-on-failure` **全部通过** | ctest 输出 |
| C | §3.1 六个用例全部存在且至少一个含 `EXPECT_*` 断言 | 代码检查 + ctest 的用例列表 |
| D | `tests/suites.cmake` configure 无「未登记目标」提醒 | configure 输出 |

---

## 6. 禁止事项

1. **禁止**修改 `src/` 下任何文件（本 SubStage 只加测试）。
2. **禁止**新建构建树。**本版的测试构建树 = `build-tests-all`（已存在，`BUILD_TESTS=ON`、`TACTILE_TEST_SUITES=all`、`BUILD_BENCHMARKS=OFF`、`BUILD_PORTABLE_TOOLS=OFF`）**，
   它已覆盖 `data` 套件 ⇒ **直接复用它**，只构建本目标。
   **禁止**改用/新建 `build-tests-all` 等其它树；**禁止**改动 `build-tests-all` 的 `TACTILE_TEST_SUITES` 值。
3. **禁止**运行全套件 ctest（`build-tests-all` 里其它套件目标未构建），禁止构建 `ALL_BUILD`、禁止构建其它套件目标。
4. **禁止**修改 `tests/` 下除 `test_drift_v6_compensator.cpp` 之外的既有测试文件；
   `tests/CMakeLists.txt` 与 `tests/suites.cmake` 只允许**追加**。
5. 不确定就停并报告。

---

## 7. 构建要求

- **既有测试树**：`build-tests-all`（见 §6 第 2 条）。
- 新增源文件后 configure 一次（**这是唯一允许的 configure**）：
  ```powershell
  cmake -S . -B build-tests-all -DBUILD_TESTS=ON -DTACTILE_TEST_SUITES=all -DBUILD_BENCHMARKS=OFF -DBUILD_CALIBRATION_FLOW_PROTOTYPE=OFF -DBUILD_PORTABLE_TOOLS=OFF
  ```
- 只构建本目标：`cmake --build build-tests-all --config Debug --target test_drift_v6_compensator -- /m:4`
- 只跑本目标：`ctest --test-dir build-tests-all -C Debug -R "^test_drift_v6_compensator\." --output-on-failure`
- 长构建用后台作业 + 60~120 s 轮询。

---

## 8. 完成报告要求

① 新增/修改文件与关键行号；② A~D 四项的实测命令与输出；③ 六个用例的通过情况；
④ 夹具列解析的具体断言；⑤ 不确定项。**禁止**只写"测试通过"。
