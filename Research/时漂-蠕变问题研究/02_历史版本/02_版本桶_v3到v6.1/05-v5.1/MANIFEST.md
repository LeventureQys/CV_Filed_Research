# 05-v5.1 —— 取消空载强制归零（当前现役算法）

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-17 16:46 ~ 17:14
- 状态：**已落地并作为当前现役实现**（`src/domain/drift/`，菜单「设备 → 时漂/零漂补偿（无责 1s/3s/5s）」）
- 主文档：`docs/06-抗蠕变漂移补偿算法说明.md`（自包含规格，只讲这一个算法，当前推荐版）

v5.1 只改一件事：**取消 v3 的「卸载门控自动归零」**——空载不再被强制拉回 0，空载显示即传感器当前读数。
起因是用户报告的「零点快速塌陷进 0」：负值被显示层 `ApplyThreshold` 钳成硬 0。
另加一条逐通道封顶「扣除 ≤ max(当前读数, 0)」，只改输出、不改任何内部状态。

诊断链条（`bj_*` → `bl/bm/bn_*`）：先探查零点与卸载行为，分解零点构成，用合成零负载钉住问题，
再追踪基线，最后 `bo_v51_verify.py` 用 3 份含零负载段的录制做前后对比。

**实测**：零点段被钳 0 的时长 v3 0~6.05 s / v5 0~6.64 s → **v5.1 0~0.27 s**；
卸载沿后 0.5 s 显示由 −569~−2503 回到 15~73；负载跟踪不劣反更优
（台阶捕获比中位 0.91/0.95/0.95 → 0.96/1.03/0.96）；恒载 9 组时漂残余 3.37% → 3.41%（基本不变）。

## 文档（2）

| 文件 | 大小 |
|---|---|
| `docs/05-v5算法说明.md` | 53 KB |
| `docs/06-抗蠕变漂移补偿算法说明.md` | 64 KB |

## 图件（0）

无独立图件：v5.1 的验证以表格与日志形式给出（`results/metrics_current*.csv`、`scenarios_v51.csv`）。


## 脚本（18）

`bj_probe_zero / bk_trace_unload / bl_decompose_zero / bm_synth_zero_pin / bn_trace_baseline` 为零点问题的诊断链；`bo_v51_verify.py` 为 v3/v5/v5.1 三方复核；`bv_scenarios_v51 / bw_metrics_current` 为场景与指标复核；`glm53_v51.py` 是当前现役算法本体。

```text
ad_lib.py  ad_v4.py  bj_probe_zero.py  bk_trace_unload.py  bl_decompose_zero.py  bm_synth_zero_pin.py
bn_trace_baseline.py  bo_v51_verify.py  bv_scenarios_v51.py  bw_metrics_current.py  glm53_v3.py  glm53_v5.py
glm53_v51.py
（编译缓存：ad_lib.cpython-314.pyc, ad_v4.cpython-314.pyc, glm53_v3.cpython-314.pyc, glm53_v5.cpython-314.pyc, glm53_v51.cpython-314.pyc）
```

## 数据（5）

```text
_metrics_current.log  _scenarios_v51.log  metrics_current.csv  metrics_current_slowwin.csv
scenarios_v51.csv
```

## 复现

```powershell
cd temp/v4.1flash/progress/05-v5.1
$env:PYTHONIOENCODING='utf-8'
python scripts/bo_v51_verify.py     # v3/v5/v5.1 前后对比（零点 + 负载跟踪 + 恒载 9 组）
python scripts/bv_scenarios_v51.py  # 场景复算
python scripts/bw_metrics_current.py # 当前版本指标
```
