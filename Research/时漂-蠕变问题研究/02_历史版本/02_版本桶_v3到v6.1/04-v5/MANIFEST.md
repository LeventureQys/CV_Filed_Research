# 04-v5 —— 四项定点修复（首个真正落地的版本）

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-17 11:46 ~ 15:22（`glm53_v5.py` 15:04）
- 状态：**已落地 `src/domain/drift/drift_compensator.{h,cpp}`**，与 Python 原型逐帧对拍 ≤ 5e−07
- 主文档：`docs/05-v5算法说明.md`（该文 §10 之后为 v5.1，故本桶与 `05-v5.1` 各存一份）

v5 = **v3 + 4 处定点修复**，修的是实测暴露的确定性缺陷，不是调参：

| # | 缺陷（v3 存在） | 症状 | v5 的修法 |
|---|---|---|---|
| P1 | 变载识别门限被 EMA 分离增益放大到 ≈27% 电平 | 小台阶被当蠕变扣掉，显示被「拉回」原读数 | 增加短滞后电平差判据，门限降到 5% |
| P2 | restep 把旧载已累积的蠕变锁进新基线 `A` | 台阶稳定后永久偏高 +5~8% | 基线扣除已生效的蠕变 |
| P3 | v4 的免责期只覆盖空载→负载 | 「变载也免责」从未生效 | onset 与 restep 都生效 |
| P4 | v4 原型免责期内跳过状态机、丢掉旧扣除 | 免责期内无法识别卸载；变载瞬间跳变 | 插在状态机之后，输出 `Z − carry` |

**收益**（4 份实录、11 个负载内变载事件、免责 3 s）：台阶捕获比 0.89 → **0.98**；
全程最大偏差占峰值 31.3% → **9.0%**；变载窗最大偏差中位 2464 → **1681 ADC**。
**代价**：恒载 9 组全段 1.88%（v3 1.58 / v4 2.60）、慢相段 1.61%（v3 1.49 / v4 0.95）。

**落地核对**：`cpp_check/` 是把 `glm53_v5.py` 与 C++ 实现逐帧对拍的独立工程
（`v5_cpp_check` 可执行体 + `_diff.py`）；`bg_v5_cpp_parity.py` 是同口径的 Python 侧对拍。

## 文档（1）

| 文件 | 大小 |
|---|---|
| `docs/05-v5算法说明.md` | 53 KB |

## 图件（2）

> 位置：`figures/`（桶根，文档按 `../figures/…` 引用）。

`v5_compare.png`（v5 vs v3 vs v4 的对比总图）、`v5_diff_worst.png`（最差用例差异定位）。

- `figures/v5_compare.png`（192 KB）
- `figures/v5_diff_worst.png`（631 KB）

## 脚本（31）

`glm53_v5.py` 是 v5 算法本体；`z2_v5 / z4_v5_trace / z5/z6_diag_v5` 是早期变体与追踪诊断；`ba_* ~ bi_*` 为 v5 的场景/录制/恒载 9 组/epoch/门限/图件/对拍脚本；`_*.py` 是当时的临时脚本（`_diff` 为 C++↔Python 对拍入口）。

```text
_diff.py  _gapcalc.py  _none.py  _png3.py  _runbh.py  ad_lib.py
ad_v4.py  ba_v5_scenarios.py  bb_v5_debug.py  bb_v5_debug2.py  bc_v5_recordings.py  bd_v5_static9.py
be_v5_epochs.py  be_v5_threshold.py  bf_v5_fig.py  bg_v5_cpp_parity.py  bh_v5_why_fig.py  bi_v5_5s_awin.py
glm53_v3.py  glm53_v5.py  r_fastphase.py  z2_v5.py  z4_v5_trace.py  z5_diag_v5.py
z6_diag_v5b.py
（编译缓存：ad_lib.cpython-314.pyc, ad_v4.cpython-314.pyc, glm53_v3.cpython-314.pyc, glm53_v5.cpython-314.pyc, r_fastphase.cpython-314.pyc, z2_v5.cpython-314.pyc）
```

## 数据（17）

```text
_v5_rec.log  _v5_rec3s.log  _v5_static.log  _v5_static3s.log
v5_maxgap.csv  v5_midload_events.csv  v5_static9_metrics.csv  v5_static9_summary.csv
v5_中途切换-13ffca.npz  v5_中途切换-13ffca_windows.csv  v5_中途切换-1d9493.npz  v5_中途切换-1d9493_windows.csv
v5_再切换负载.npz  v5_再切换负载_windows.csv  v5_切换负载-快相无责.npz  v5_切换负载-快相无责_windows.csv
varying_steps_v5.csv
```

## `cpp_check/`（65）

```text
cpp_check/CMakeLists.txt
cpp_check/build/ALL_BUILD.vcxproj
cpp_check/build/ALL_BUILD.vcxproj.filters
cpp_check/build/CMakeCache.txt
cpp_check/build/CMakeFiles/0eab482ea0f545e45ad1de472bf50e2b/generate.stamp.rule
cpp_check/build/CMakeFiles/4.3.2/CMakeCXXCompiler.cmake
cpp_check/build/CMakeFiles/4.3.2/CMakeDetermineCompilerABI_CXX.bin
cpp_check/build/CMakeFiles/4.3.2/CMakeRCCompiler.cmake
cpp_check/build/CMakeFiles/4.3.2/CMakeSystem.cmake
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/CMakeCXXCompilerId.cpp
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/CompilerIdCXX.exe
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/CompilerIdCXX.vcxproj
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CMakeCXXCompilerId.obj
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.exe.recipe
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/CL.command.1.tlog
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/CL.read.1.tlog
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/CL.write.1.tlog
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/Cl.items.tlog
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/CompilerIdCXX.lastbuildstate
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/link.command.1.tlog
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/link.read.1.tlog
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/link.secondary.1.tlog
cpp_check/build/CMakeFiles/4.3.2/CompilerIdCXX/Debug/CompilerIdCXX.tlog/link.write.1.tlog
cpp_check/build/CMakeFiles/4.3.2/VCTargetsPath.txt
cpp_check/build/CMakeFiles/4.3.2/VCTargetsPath.vcxproj
cpp_check/build/CMakeFiles/4.3.2/VCTargetsPath/x64/Debug/VCTargetsPath.recipe
cpp_check/build/CMakeFiles/4.3.2/VCTargetsPath/x64/Debug/VCTargetsPath.tlog/VCTargetsPath.lastbuildstate
cpp_check/build/CMakeFiles/CMakeConfigureLog.yaml
cpp_check/build/CMakeFiles/InstallScripts.json
cpp_check/build/CMakeFiles/TargetDirectories.txt
cpp_check/build/CMakeFiles/cmake.check_cache
cpp_check/build/CMakeFiles/generate.stamp
cpp_check/build/CMakeFiles/generate.stamp.depend
cpp_check/build/CMakeFiles/generate.stamp.list
cpp_check/build/Debug/v5_cpp_check.exe
cpp_check/build/Debug/v5_cpp_check.pdb
cpp_check/build/ZERO_CHECK.vcxproj
cpp_check/build/ZERO_CHECK.vcxproj.filters
cpp_check/build/cmake_install.cmake
cpp_check/build/v5_cpp_check.dir/Debug/drift_compensator.obj
… 其余 25 个文件（构建产物）
```

## 注意

`cpp_check/build/` 是随桶搬来的构建产物，其中的绝对路径已失效；如需对拍请重新 configure。

## 复现

```powershell
cd temp/v4.1flash/progress/04-v5
$env:PYTHONIOENCODING='utf-8'
python scripts/ba_v5_scenarios.py    # 场景复算
python scripts/bd_v5_static9.py      # 恒载 9 组基准
python scripts/bg_v5_cpp_parity.py   # C++↔Python 对拍（需先构建 cpp_check）
python scripts/bf_v5_fig.py          # 出图 v5_compare.png
```
