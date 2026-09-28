# SubStage 3A · 回归锁测试 + 参数落盘

## SubStage: 3A
- **所属 Stage**：Stage 3
- **依赖前置**：SubStage 2A（限幅）+ SubStage 2B（F2/F3/F5）（**需等待两者都完成**）
- **并行状态**：可与 SubStage 3B（文档同步）**并行**
- **所属阶段**：阶段三 - 开发

> **你是 SubAgent**。开工前必须先完整阅读仓库根 `AGENTS.md`（含「Agent 命令执行预算」「验证与构建总则」「SubAgent 编译测试分工」），并遵守本文全部约束。

---

## 1. 背景与目标

SubStage 1B 建立了测试骨架（夹具加载 + 既有行为断言），2A/2B 落地了四项修复。
本 SubStage 把**四项修复各自的失效模式**变成**回归锁测试**，并把新增参数写入录制 manifest，
使真机录一次就能追溯参数集。**本 SubStage 不改补偿逻辑**，只加测试与参数落盘。

---

## 2. 当前代码状态

| 位置 | 内容 |
|---|---|
| `tests/data/test_drift_v6_compensator.cpp` | 1B 建立；含夹具加载 `LoadFixture(name)` 与六个用例 |
| `tests/CMakeLists.txt` | 1B 追加的 `add_tactile_test(test_drift_v6_compensator ...)` + `DRIFT_V6_FIXTURE_DIR` 宏 |
| `src/data/data_handler.cpp` L129~L142 | `CurrentDisplayAlgorithm()`：v6 分支填 `kappa_onset/kappa_restep/ho_min_s/revoke_hold_n/param_set` |
| `src/domain/drift_v6/drift_v6_compensator.h` | 1A 新增 `Params` / `params()` / `SetParams` |
| `.cpp` L749~L759、L1652~L1662 | v6 的两条挂载链路（串口 / 无线） |

---

## 3. 必须实现的交付物

### 3.1 `tests/data/test_drift_v6_compensator.cpp` 追加用例（**回归锁**）

| 用例名 | 失效模式（必须锁住的行为） | 断言 |
|---|---|---|
| `ClampLimitsLeadOverRaw` | C：显示超前量不得超 `α·\|原始\|` | 三段夹具全帧：`out_i ≤ raw_i + params().clamp_alpha·\|raw_i\| + 1e-6`；且**至少一帧**该不等式取等（证明限幅真的生效过，而不是恒真） |
| `ClampDisabledIsPassThrough` | C 的开关语义 | `Params p; p.clamp_alpha = 0.0; comp.SetParams(p);` 后重放，输出与"未接入限幅的 2A 前版本"逐帧一致（用 `≥1e-9` 容差比较两段 `std::vector<double>` 快照；快照可用断言中的期望常量表） |
| `EventUnloadValleyExitFires` | **F2**：事件期内回到谷底必须出事件 | 构造合成序列：空载 20 s → 阶跃到 +14000（保压 4 s）→ 回落到空载（保压 2 s）；**测试开头先 `SetParams` 把 `legacy_fixes_enabled = true`**；期望在该 6 s 内出现一次 `n_valley_exit ≥ 1`。**若 `n_valley_exit == 0`，用例失败**。另加一条断言：默认参数（`false`）下同一序列 `n_valley_exit == 0`（锁住默认关闭语义） |
| `ReanchorValleyZeroesResidual` | **F3**：回到谷底重锚必须归零 | 同上合成序列 + **`legacy_fixes_enabled = true`**；期望 `n_reanchor_idle ≥ 1`，且卸载后的空载段总输出与总输入差 `≤ 1%·台阶` |
| `GNegFloorBoundsSlowResidual` | **F5**：`g` 不得低于下界 | 用 `seg_c_cycle` 全段回放，逐帧断言 `comp.g()`（若 1A/2B 未暴露 getter，则用返回**新增只读 getter** `double slow_residual_g() const { return g_; }`）满足 `≥ params().g_neg_floor − 1e-9` |
| `ParamsRoundTripAndReset` | P：参数注入与复位 | `SetParams` 非默认值 → `params()` 反映新值 → `Reset()` → `params()` 回到默认值 |
| `NoLivelockOnLongHold` | 防"永不交接"（D1 §6 G9） | `seg_a_onset` 全段回放后 `comp.in_event() == false`（不得停在事件态） |

**辅助要求**：
- 若需要读内部 `g_`，在 `.h` 增加**只读诊断 getter**：
  `double slow_residual_g() const { return g_; }`（放在 `shape_hits()` 附近，不得改成非 const）。
  这是本 SubStage 允许的唯一产品代码改动。

### 3.2 参数落盘（`src/data/data_handler.cpp` L129~L142）

在既有 v6 分支的 `obj` 中**追加**四个字段（不要删改既有五个）：

```cpp
            obj[QStringLiteral("clamp_alpha")]          = drift_v6_comp_.params().clamp_alpha;
            obj[QStringLiteral("valley_win_n")]         = drift_v6_comp_.params().valley_win_n;
            obj[QStringLiteral("g_neg_floor")]          = drift_v6_comp_.params().g_neg_floor;
            obj[QStringLiteral("legacy_fixes_enabled")] = drift_v6_comp_.params().legacy_fixes_enabled;
            obj[QStringLiteral("param_set")]            = QStringLiteral("plan-v2.0 C+F5");
```

**接口前置**：`DriftV6Compensator::params()` 是 public 只读（SubStage 1A 提供）。
若 `drift_v6_comp_` 的**访问方式**与既有 `kappa_onset` 行（L136）不同（例如需要 `targets()` 或协调器接口），
**照 L136~L139 的既有写法原样处理**，不要自创新的访问路径。
`param_set` 字段原已存在（值为 `plan-v1.0 A1+A4+A5a`），本 SubStage 把它**改为** `plan-v2.0 C+F5`
（F2/F3 默认关闭 ⇒ 参数集名只标默认生效项）。

---

## 4. 接口契约

```text
新增只读诊断接口（.h, public）：
    const Params& params() const;            // 1A
    void SetParams(const Params&);           // 1A
    double slow_residual_g() const;          // 本 SubStage 新增（inline，返回 g_）
    long shape_hits() const;                 // 既有 L72

Params 字段（1A 定义，本 SubStage 只读）：
    clamp_alpha / valley_win_s / valley_win_n / valley_frac / valley_range_frac /
    event_valley_min_s / reanchor_valley_s / g_neg_floor / g_valley_reset_s /
    kappa_onset / kappa_restep / ho_min_s / revoke_hold_n /
    n_valley_exit / n_reanchor_idle / n_g_floor / n_g_valley_reset

录制 manifest 字段（本 SubStage 写入）：
    algorithm.params.{kappa_onset,kappa_restep,ho_min_s,revoke_hold_n,
                      clamp_alpha,valley_win_n,g_neg_floor,param_set}
```

---

## 5. 验收标准

| # | 标准 | 判定 |
|---|---|---|
| A | `cmake --build build-tests-all --config Debug --target test_drift_v6_compensator -- /m:4` **exit 0** | 命令输出 |
| B | `ctest --test-dir build-tests-all -C Debug -R "^test_drift_v6_compensator\." --output-on-failure` **全部通过**（1B 的 6 个 + 本 SubStage 的 7 个 = 13 个） | ctest 输出 |
| C | **阳性对照**：把 `Params::clamp_alpha` 改成 `0.0` 后 `ClampLimitsLeadOverRaw` **必须失败**（证明它不是恒真断言）；把 `valley_frac` 改成 `0.0001` 后 `EventUnloadValleyExitFires` **必须失败**。做完后**改回默认值**并在报告中留痕 | 手工改一次、运行、改回、贴输出 |
| D | `cmake --build build-app --config Debug --target TactileSense -- /m:4` **exit 0**（参数落盘改动可编译） | 命令输出 |
| E | 参数落盘字段齐全：用 grep 在 `data_handler.cpp` 中确认 8 个字段名全部出现 | 命令输出 |

---

## 6. 禁止事项

1. **禁止**修改补偿逻辑（`Process`/`RunEvent`/`SlowStep`/`Reanchor`/`Handoff`/`ToIdle` 的算法部分）；
   本 SubStage 在产品代码上**只允许**加 `slow_residual_g()` getter 与 §3.2 的参数落盘。
2. **禁止**改动 1B 已注册的测试目标名与 `DRIFT_V6_FIXTURE_DIR` 宏。
3. **禁止**新建构建树 / 无 `--target` 构建 / `ALL_BUILD` / 全套件 ctest。
4. **禁止**删除或放宽既有用例的断言来"让测试通过"。
5. 不确定就停并报告。

---

## 7. 构建要求

- 复用既有 `build-tests-all`；新增用例不需要 configure（已在 CMake 目标内）。
  若新增了源文件才需 configure 一次。
- 命令：`cmake --build build-tests-all --config Debug --target test_drift_v6_compensator -- /m:4`
  → `ctest --test-dir build-tests-all -C Debug -R "^test_drift_v6_compensator\." --output-on-failure`
- 主程序：`cmake --build build-app --config Debug --target TactileSense -- /m:4`
- 长构建用后台作业 + 60~120 s 轮询；**先 `job_list` 查是否已有同目标作业**。

---

## 8. 完成报告要求

① 新增/修改文件与行号；② A~E 的实测输出；③ 阳性对照的两次失败输出（§5-C）；
④ 13 个用例的逐个通过情况；⑤ 不确定项。**禁止**只写"测试通过"。
