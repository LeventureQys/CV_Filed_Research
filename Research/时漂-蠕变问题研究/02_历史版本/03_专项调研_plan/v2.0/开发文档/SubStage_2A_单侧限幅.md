# SubStage 2A · C 路修复：显示单侧限幅

## SubStage: 2A
- **所属 Stage**：Stage 2
- **依赖前置**：SubStage 1A（**需等待**：`Params`/`params()`/`valley_*` 已落地）
- **并行状态**：可与 SubStage 2B **并行**（两者改同一文件的不同函数，见 §2 的「冲突约定」）
- **所属阶段**：阶段三 - 开发

> **你是 SubAgent**。开工前必须先完整阅读仓库根 `AGENTS.md`（含「Agent 命令执行预算」「验证与构建总则」「SubAgent 编译测试分工」），并遵守本文全部约束。

---

## 1. 背景与目标

实机录制（33908 帧 / 337.5 s / 显示域 ADC / 21 通道）上实测：加载沿后显示相对传感器读数出现
**+4.60%（中位）/ +5.52%（最大）·台阶**的瞬态超前，并在长保压期内累积成 +808 → +1438 ADC 的偏移。
本 SubStage 落地用户裁决的 **C 路修复**：**显示单侧限幅** `显示 ≤ 原始 + α·|原始|`，**α = 0.005**（可通过
`Params::clamp_alpha` 运行期覆盖）。

**实测收益**（`temp/v4.1flash/plan/v2.0/results/design_ab_verify.txt`）：
超前量峰值中位 4.72% → **0.59%**·台阶、最大 13.27% → 0.75%；到带时间 T5% 中位 4.78 → **2.51 s**；
保压段偏移中位 438 → **76 ADC**、最大 1273 → 455 ADC。
**实测代价**（必须写进完成报告）：加载后 0.3~1.2 s 的早期超前量 506 → 65 ADC（削掉约 87%）。

---

## 2. 当前代码状态与冲突约定

| 位置 | 内容 |
|---|---|
| `.cpp` L376~L384 | 事件期出口：`RunEvent` 返回 true 时 `prev_out_ = out; values_io = out; Push(...); return;` |
| `.cpp` L386~L399 | 慢相态出口：`ToIdle` 或 `SlowStep` 后 `prev_out_ = out; values_io = out; Push(...); return;` |
| `.cpp` L401~L403 | 直通出口：`prev_out_ = v; values_io = v; Push(timestamp_s, v, y);` |
| `.cpp` L240 | `Process(double timestamp_s, Eigen::VectorXd& values_io)` 函数头；`v` 是 L245 的 `const Eigen::VectorXd v = values_io;`（**原始读数**，正是限幅的参考量） |
| `.h` L207~L236 | 私有函数声明区（新函数加在这里） |

**冲突约定（与 SubStage 2B 并行时有约束）**：
- 2A **只改** `Process` 的三个出口与新增 `ApplyOutputLimit`；
- 2B **只改** `RunEvent` 开头、`Reanchor` 开头、`SlowStep` 内；
- 两者都**不得**改动 `Process` L240~L374 的其余部分（2B 需要的 `UpdateValley` 调用点已在 1A 落好）。
- 若两人同时产出，冲突仅可能出现在 `Process` 的 L376~L403 区段（2A 独占）。

---

## 3. 接口契约（逐字实现）

### 3.1 新增私有函数（`.h` 私有函数声明区）

```cpp
    // 显示单侧限幅: out_i = min(out_i, raw_i + alpha·|raw_i|)。只改输出、不改任何内部状态。
    void ApplyOutputLimit(const Eigen::VectorXd& raw, Eigen::VectorXd* out) const;
```

### 3.2 实现（`.cpp`，放在 `Process` 之前或 `MedianInPlace` 之后）

```cpp
void DriftV6Compensator::ApplyOutputLimit(const Eigen::VectorXd& raw,
                                          Eigen::VectorXd* out) const {
    if (out == nullptr) return;
    if (params_.clamp_alpha <= 0.0) return;                 // 0 = 关闭(与既有安全语义一致)
    if (raw.size() != out->size()) return;                  // 尺寸不符直接跳过, 不越界
    for (int k = 0; k < raw.size(); ++k) {
        const double lim = raw(k) + params_.clamp_alpha * std::abs(raw(k));
        if ((*out)(k) > lim) (*out)(k) = lim;
    }
}
```

### 3.3 三个出口的接入（`.cpp` L376~L403 的重写）

```cpp
    if (ev_.valid) {
        Eigen::VectorXd out;
        if (RunEvent(timestamp_s, v, total, dt, eps, idle_now, &out)) {
            ApplyOutputLimit(v, &out);          // ← 新增（在写回 values_io 之前）
            prev_out_ = out;
            values_io = out;
            Push(timestamp_s, v, y);
            return;
        }
    }

    if (state_ == State::Slow) {
        Eigen::VectorXd out;
        if (idle_now && (timestamp_s - ev_end_) > kUnloadFastS) {
            ToIdle();
            out = v;
        } else {
            out = SlowStep(v, dt);
        }
        ApplyOutputLimit(v, &out);              // ← 新增
        prev_out_ = out;
        values_io = out;
        Push(timestamp_s, v, y);
        return;
    }

    prev_out_ = v;
    values_io = v;
    Push(timestamp_s, v, y);
```

**注意**：直通出口（第三段）**不需要**限幅调用——`out == raw` 时 `raw ≤ raw + α|raw|` 恒成立。
若为统一风格而调用也无害，但**不要**为它额外改动其它逻辑。

**关键约束**：
- 限幅**必须**在 `prev_out_ = out` **之前**施加 —— `prev_out_` 是下一帧 `y0` 的来源，
  限幅后的值才是"真正显示出去的值"，事件继承（`NewEvent` 的 `c0 = base_y − base`）必须基于它。
- 限幅**不得**修改 `A_/g_/gamma_/loaded_/ev_/hold_comp_` 等任何内部状态（与 `Capped` 同性质，见 `.cpp` L40 注释）。

---

## 4. 输入输出

- **输入**：`raw`（本帧原始显示域读数，21 或 31 维）、`out`（补偿器算出的显示值）。
- **输出**：被就地限幅的 `out`；无返回值、不抛异常、不写内部状态。
- **错误处理**：`out == nullptr`、尺寸不等、`clamp_alpha ≤ 0` 三种情况**静默跳过**（不做任何修改）。

---

## 5. 验收标准

| # | 标准 | 判定 |
|---|---|---|
| A | `cmake --build build-app --config Debug --target TactileSense -- /m:4` **exit 0** | 命令输出 |
| B | 不变量成立：对 `seg_a_onset` / `seg_b_handoff` / `seg_c_cycle` 三段夹具，全帧 `out_i ≤ raw_i + 0.005·|raw_i| + 1e-6` | 单元测试 `CompensationNeverExceedsRaw` 由跳过变为**真实通过** |
| C | `params().clamp_alpha = 0.0` 时输出与"未接入限幅"逐帧一致（关闭开关有效） | 单元测试或一次性脚本 |
| D | 保压段偏移改善：`seg_b_handoff` 上「受载段 `\|显示−原始\|` 中位」由 ≈438 ADC 降到 **≤120 ADC** | 一次性脚本（见下） |
| E | 既有 13 份录制的时漂残余与台阶捕获比 **Δ = 0.0000**（逐位不变） | `design_full_regress.py` 由 MainAgent 复算；SubAgent 只负责提供可复现命令 |

### 自测参考（SubAgent 必做 B/C/D）

用 `tests/fixtures/drift_v6/seg_b_handoff_pre.csv` 与 `_main.csv`：
```powershell
$env:PYTHONIOENCODING='utf-8'
python temp\v4.1flash\plan\v2.0\scripts\check_drift_v6_fixture.py    # 夹具自检（已有）
```
若需 C++ 侧断言，请**优先扩展 `tests/data/test_drift_v6_compensator.cpp` 的既有用例**（SubStage 1B 建立），
不要另建测试目标。

---

## 6. 禁止事项

1. **禁止**改动 `Process` 的 L240~L374（检测器/建事件段）与 `RunEvent`/`SlowStep`/`Reanchor`/`Handoff`/`ToIdle`。
2. **禁止**改动 `kKappaOnset`（保持 1.05）、`kHoMinS`（3.5）、停滞判据、形状 ROM、`idle_now`。
3. **禁止**修改 `src/` 下除 `drift_v6_compensator.{h,cpp}` 之外的任何文件。
4. **禁止**新建构建树 / 无 `--target` 构建 / `ALL_BUILD` / 全套件 ctest。
5. **禁止**同时构建主程序（与 MainAgent 的构建作业冲突）：先 `job_list` 检查，必要时等待。
6. 不确定就停并报告。

---

## 7. 构建要求

- 主程序目标：`cmake --build build-app --config Debug --target TactileSense -- /m:4`（**禁止**重新 configure）。
- 测试目标（复用既有树）：`cmake --build build-tests-all --config Debug --target test_drift_v6_compensator -- /m:4`
  然后 `ctest --test-dir build-tests-all -C Debug -R "^test_drift_v6_compensator\." --output-on-failure`。
- 长构建用 `run_in_background` + 60~120 s `job_output` 轮询。

---

## 8. 完成报告要求

① 改动文件与行号；② A~D 的实测命令与输出；③ **明确写出"代价"这一项**（早期超前量削掉多少，用 §1 的数字对照）；
④ 是否有行为超出预期；⑤ 不确定项。**禁止**只写"编译通过/测试通过"。
