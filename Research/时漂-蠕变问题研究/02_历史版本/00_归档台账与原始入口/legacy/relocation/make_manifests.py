# -*- coding: utf-8 -*-
"""按桶生成 MANIFEST.md：手工写的版本定位 + 自动生成的完整文件清单。"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(os.path.dirname(HERE), "v4.1flash", "progress")

B = {}

B["01-v3-baseline"] = dict(
    title="v3 —— 现役基线（GLM53 混合方案）",
    meta=[
        "时间：2026-09-03 落地，2026-09-17 起被 v5 取代",
        "状态：**曾被实施到 `src/domain/drift/drift_compensator.{h,cpp}`**（本桶只保留当时的源码快照）",
        "定位：所有后续版本的对照基线，指标表中的 `raw` / `v3` 两列即来自它",
    ],
    body="""
`v3` 是显示层时漂补偿的第一版实装：**卸载门控自动归零**（空载期用 τ=2 s 跟踪基线 `b_` 并从显示里减掉）
+ **负载比例蠕变场**（median 共识 `g` + 逐通道 `γ` + 限幅）+ 短滞后/长滞后双 EMA 的电平判据。

本桶不含文档：v3 的算法描述与缺陷清单写在 `04-v5` 的 `05-v5算法说明.md`（P1~P4 四项缺陷就是 v3 的）
与 `05-v5.1` 的 `06-抗蠕变漂移补偿算法说明.md`（当前算法，语义与之对照）中。
`source/` 下是当时的 C++ 快照备份（`.v3bak`），可用于逐行核对「v5 到底改了什么」。

**基线数字**（恒载 9 组，来自 `04-v5` / `06-v5.1-exempt-1s-3s-5s` 的复算）：
全段时漂 **1.58%**、慢相段 **1.49%**；变载台阶捕获比 0.89、全程最大偏差占峰值 31.3%。
""",
    fig_note="无独立图件（v3 曲线出现在各版本的对比图中作为基准线）。",
    script_note="`glm53_v3.py` 是 v3 的逐行 Python 移植（对应 C++ 原文件，参数一字未改），"
                "被 v4/v5/v6/v6.1 全部复算脚本 import —— 属**跨版本共享模块**，每个桶各存一份。",
    repro=["python scripts/glm53_v3.py   # 模块本体，无独立入口；由各桶脚本 import 使用"],
)

B["02-dsp-route"] = dict(
    title="dsp.md 逆滤波路线 —— 评审与实测否决",
    meta=[
        "时间：2026-09-17 10:01 ~ 10:31（本工作区最早的一轮）",
        "来源：评审 `temp/GLM53/dsp.md`《柔性触觉压感阵列抗蠕变漂移与动态力保真 DSP 补偿方案指南》",
        "结论：**理论方向正确、工程细节失守，不建议采用**（时漂残余 1.55% → 15.2~32.3%）",
    ],
    body="""
这条路线与 v4/v5 是**并行技术路线**，不是版本演进的一环：它主张用二阶逆 IIR（§6.1）直接反演蠕变模型，
再用拉普拉斯高通核做空间反卷积（§4）。本桶是把它**照着实现并放到 9 组实测上跑**的全部证据链。

评审结论分六处（详见 `docs/dsp方案评审报告.md`）：
§2.1 两套公式不自洽（直流增益差 N₀ = 1+a₁+a₂ 倍）；§6.2 硬编码系数与 §6.1 公式不是同一个滤波器；
§3.3 工况 A/B 的行为描述错误；§3.1「高频增益限制」未实现（全带噪声放大 N₀ 倍）；
§5 离线标定在 9 组实测上退化；§4 空间反卷积把总力打成负数（52~68% 通道出负值）。
§6.1 的数学推导本身经频域逐点验证是正确的（误差 < 1e-6，零极点精确对消）。

**留痕**：`results/superseded/` 保留了本轮两次被推翻的误判（用 `sum(b)/sum(a)` 判直流增益、
早期自写并行实现漏归一化系数），README §复核留痕 有说明。
""",
    fig_note="`f_review_1~4.png` 为评审报告配图（§6.1 阶跃响应/幅频、9 组形状检验、空间反卷积失败、主通道三方对比）；"
             "`f1_timeseries_grid / f2_load_zoom / f3_step_edge` 为 9 组恒载上 dsp vs v3 的对比。",
    script_note="`v_*` 为解析性探针与 IIR 正确性裁决；`x_model_id / y_spatial / z_shape_look` 为模型辨识、"
                "空间反卷积与形状诊断；`w_dsp_vs_v3 / w2_dsp_vs_v3 / e_dsp_fail` 为真实数据上的对比与失败点定位；"
                "`inventory.py` 产出 11 组数据盘点；`f_review_figs.py` 出图。",
    repro=[
        "python scripts/v_dsp_review_probe.py   # §1~§3 解析性复核（秒级）",
        "python scripts/v_final_check.py        # IIR 正确性裁决（秒级）",
        "python scripts/y_spatial.py            # §4 空间反卷积实测（秒级）",
        "python scripts/inventory.py            # 11 组数据盘点（约 1 分钟）",
        "python scripts/e_dsp_fail.py           # §5 标定失败定位（约 2 分钟）",
        "python scripts/w2_dsp_vs_v3.py         # 9 组恒载总对比（约 4 分钟）",
        "python scripts/f_review_figs.py        # 生成评审报告图（约 1 分钟）",
    ],
    notes="`results/superseded/` 是被推翻结论的留痕，不要当作结论引用。",
)

B["03-v4"] = dict(
    title="v4 —— 快相免责期改造（本工作区的主线起点）",
    meta=[
        "时间：2026-09-17 11:32 ~ 14:14",
        "状态：**原型，未落地 C++**（其落地骨架 `04-C++实现骨架.md` 部分内容仍适用）",
        "核心思想：加载瞬间的「快相」不要求算法处理，把补偿推后到快相结束，只对抗后续的慢相蠕变",
    ],
    body="""
本桶回答的问题是「**快相 0~4 s 的快速爬升能不能放它过去，只治之后的慢相**」。
先把两段结构量化（`快相与慢相分离分析.md`：归一化轮廓 σ≈0.016，高度可复现），
再把免责期从 3 s / 5 s / 8 s 扫一遍，最终定为 **3 s 与 5 s 两档**并完成 5 算法 × 11 组对比。

三份正式文档：
- `docs/02-v4算法说明.md` —— v4 的物理依据、算法定义、参数、代价与限制；
- `docs/03-验证与实测结果.md` —— 5 算法 × 11 组验证、图表索引、复现命令；
- `docs/04-C++实现骨架.md` —— v4 的 C++ 改动骨架（§3.0/3.1/3.2/3.4 要点仍适用，§3.3 关于 restep 起免责期的描述有误）。

本轮暴露的四个缺陷正是 v5 的修复对象；`五算法实测对比.md` 是这一轮的收官对比
（raw / v3 / v4-3s / v4-5s / dsp.md 逆滤波）。

**关键发现**：变载可辨识性 —— v4 免责期在 8/8 变载事件上欠报不大于 v3（均值 9.6% → 5.9%）；
「免责期只在首次 onset 生效」的变体（v7）在变化负载上与 v3 逐帧完全相同 ⇒ 无收益，见 `09-v7`。
""",
    fig_note="`F1_overview / F2_zoom / F3_metrics`（4 算法 × 11 组）、**`G1~G4`（5 算法 × 11 组最终对比，含变化负载长图）**、"
             "`f_fastphase_1`（快相归一化轮廓）、`f_varying_change`（中途变载逐帧）、"
             "`new_switch_load_*` / `rec_*`（实录验证）、`weight_scenario`（合成砝码 7 场景）。",
    script_note="`ad_v4 / r_fastphase` 是 v4 算法本体（后者同时是原型入口）；`q_*` 快慢相量化；"
                "`s_* / t_* / u_* / v2_track / w2_repeat` 免责期专项；`y2/y3/z8/z9` 变载可辨识性；"
                "`aa/ab/ac_*` 三轮最终对比与出图；`ad_* / ae_* / af_*` 实录与砝码场景；"
                "`cpp_check/` 是「文档里的 C++ 骨架片段抽出来能否编译」的自检工程。",
    repro=[
        "python scripts/ac_final5.py       # 5 算法 × 11 组 + G1~G4（约 5 分钟）",
        "python scripts/r_fastphase.py     # 免责 3/5/8 s 扫描（约 3 分钟）",
        "python scripts/z8_gap.py          # 变载可辨识性（约 2 分钟）",
        "python scripts/x2_fastfig.py      # 快相轮廓图（约 1 分钟）",
    ],
)

B["04-v5"] = dict(
    title="v5 —— 四项定点修复（首个真正落地的版本）",
    meta=[
        "时间：2026-09-17 11:46 ~ 15:22（`glm53_v5.py` 15:04）",
        "状态：**已落地 `src/domain/drift/drift_compensator.{h,cpp}`**，与 Python 原型逐帧对拍 ≤ 5e−07",
        "主文档：`docs/05-v5算法说明.md`（该文 §10 之后为 v5.1，故本桶与 `05-v5.1` 各存一份）",
    ],
    body="""
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
""",
    fig_note="`v5_compare.png`（v5 vs v3 vs v4 的对比总图）、`v5_diff_worst.png`（最差用例差异定位）。",
    script_note="`glm53_v5.py` 是 v5 算法本体；`z2_v5 / z4_v5_trace / z5/z6_diag_v5` 是早期变体与追踪诊断；"
                "`ba_* ~ bi_*` 为 v5 的场景/录制/恒载 9 组/epoch/门限/图件/对拍脚本；"
                "`_*.py` 是当时的临时脚本（`_diff` 为 C++↔Python 对拍入口）。",
    repro=[
        "python scripts/ba_v5_scenarios.py    # 场景复算",
        "python scripts/bd_v5_static9.py      # 恒载 9 组基准",
        "python scripts/bg_v5_cpp_parity.py   # C++↔Python 对拍（需先构建 cpp_check）",
        "python scripts/bf_v5_fig.py          # 出图 v5_compare.png",
    ],
    notes="`cpp_check/build/` 是随桶搬来的构建产物，其中的绝对路径已失效；如需对拍请重新 configure。",
)

B["05-v5.1"] = dict(
    title="v5.1 —— 取消空载强制归零（当前现役算法）",
    meta=[
        "时间：2026-09-17 16:46 ~ 17:14",
        "状态：**已落地并作为当前现役实现**（`src/domain/drift/`，菜单「设备 → 时漂/零漂补偿（无责 1s/3s/5s）」）",
        "主文档：`docs/06-抗蠕变漂移补偿算法说明.md`（自包含规格，只讲这一个算法，当前推荐版）",
    ],
    body="""
v5.1 只改一件事：**取消 v3 的「卸载门控自动归零」**——空载不再被强制拉回 0，空载显示即传感器当前读数。
起因是用户报告的「零点快速塌陷进 0」：负值被显示层 `ApplyThreshold` 钳成硬 0。
另加一条逐通道封顶「扣除 ≤ max(当前读数, 0)」，只改输出、不改任何内部状态。

诊断链条（`bj_*` → `bl/bm/bn_*`）：先探查零点与卸载行为，分解零点构成，用合成零负载钉住问题，
再追踪基线，最后 `bo_v51_verify.py` 用 3 份含零负载段的录制做前后对比。

**实测**：零点段被钳 0 的时长 v3 0~6.05 s / v5 0~6.64 s → **v5.1 0~0.27 s**；
卸载沿后 0.5 s 显示由 −569~−2503 回到 15~73；负载跟踪不劣反更优
（台阶捕获比中位 0.91/0.95/0.95 → 0.96/1.03/0.96）；恒载 9 组时漂残余 3.37% → 3.41%（基本不变）。
""",
    fig_note="无独立图件：v5.1 的验证以表格与日志形式给出（`results/metrics_current*.csv`、`scenarios_v51.csv`）。",
    script_note="`bj_probe_zero / bk_trace_unload / bl_decompose_zero / bm_synth_zero_pin / bn_trace_baseline` "
                "为零点问题的诊断链；`bo_v51_verify.py` 为 v3/v5/v5.1 三方复核；"
                "`bv_scenarios_v51 / bw_metrics_current` 为场景与指标复核；`glm53_v51.py` 是当前现役算法本体。",
    repro=[
        "python scripts/bo_v51_verify.py     # v3/v5/v5.1 前后对比（零点 + 负载跟踪 + 恒载 9 组）",
        "python scripts/bv_scenarios_v51.py  # 场景复算",
        "python scripts/bw_metrics_current.py # 当前版本指标",
    ],
)

B["06-v5.1-exempt-1s-3s-5s"] = dict(
    title="免责期 1 s / 3 s / 5 s 三档专项",
    meta=[
        "时间：2026-09-18 00:13 ~ 01:18",
        "提问：「把无责窗从 3 s 压到 **1 s** 会怎样？请与 3 s / 5 s 同图对比」",
        "结论：**1 s 不可取**；3 s 仍是长保压默认折中，5 s 在变载跟踪上全面最好",
    ],
    body="""
这是对现役 v5.1 **运行期可切参数**（菜单三档）的一次专项扫描：同一算法本体
（`glm53_v51.py`，与 C++ 逐帧一致）只改免责期长度与随之派生的 A 窗、武装延时（三处联动）。

| 口径 | 1 s | 3 s | 5 s |
|---|---|---|---|
| 恒载 9 组 全段时漂 | 1.79% | 1.59% | 1.53% |
| 变载窗最大偏差（中位） | 3258 ADC | 1889 ADC | **512 ADC** |
| 台阶捕获比（中位） | 0.76~0.90 | 0.96 | **0.99** |
| pending 占比 / epoch 数 | 21.2% / 31 | 16.3% / 29 | 16.3% / — |

1 s 的问题在于**把真实加载当成蠕变扣掉**（台阶捕获比最小 0.76，全程最大偏差中位是 5 s 档的 3.7 倍），
并引入额外的假变载重锚。`H2_exempt_1s_ablation.png` 把 1 s 档拆成「免责期长度 × 武装延时」两因子，
证明欠报来自它们共同作用，而不是单一参数。

> **后续**：菜单最终按用户明确要求加入了「无责 1s」档，并在 tooltip 中标注欠报代价
> （见项目 `src/ui/menubar.{h,cpp}`）。
""",
    fig_note="`H1`（3×3 主对比：时序/加载沿放大+首扣/首扣时延/恒载时漂/变载跟踪/扣除量/A 窗/逐事件/全程偏差）、"
             "`H2`（1 s 两因子拆解）、`H3`（13ffca 加载沿）、`H4`（13ffca 时间分段）、"
             "`H5`（1868 s pending 冻结放大）、`H6/H7`（13 份录制 1 s 档总览与指标）。",
    script_note="`bp_exempt_sweep.py` 是主扫描（37 KB，产出 H1/H2 与 `exempt_sweep_*` 全套）；"
                "`bq_probe_1s.py` 机制探针；`br/bz/by` 为 13ffca 与 1868 s 的定点放大；"
                "`bs/bt/bu/bx` 为加载沿时延、恒载时间线、pending vs 免责、无冻结 A/B；"
                "`ca_all_datasets_1s.py` 产出 13 份录制的 1 s 档总览。",
    repro=[
        "python scripts/bp_exempt_sweep.py   # 主扫描（约 10 分钟，产出 H1/H2 + exempt_sweep_*）",
        "python scripts/ca_all_datasets_1s.py # 13 份录制 1 s 档总览（H6/H7）",
        "python scripts/br_target_13ffca.py  # 13ffca 加载沿（H3）",
    ],
)

B["07-v6"] = dict(
    title="v6 —— 形状约束反演 + 滑行器（已落地 C++）",
    meta=[
        "时间：2026-09-18 14:19 ~ 16:38（`glm53_v6.py` 16:33）",
        "状态：**已落地 `src/domain/drift_v6/drift_v6_compensator.{h,cpp}`**（菜单「时漂/零漂补偿（v6 快速稳定）」），未做真机验证",
        "目标：把「加载沿 → 显示稳定」从 3.82 s 压到 ≤ 1.0 s",
    ],
    body="""
v6 **只换掉「快相处理」这一段**：把 v5 的「等快相走完（免责期）」换成「用标定形状把快相算掉」；
慢相模块（median 共识 `g` + 逐通道 `γ` + 限幅 + 输出封顶）**逐行沿用 v5**，保证可比。
前端为：短滞后电平差 + 在线 σ 门限检测（回溯真沿）→ 四类工况分类 → 形状约束反演 `Â = Σy·g/Σg²`
（14 点形状 ROM）→ 速率受限 smoothstep 滑行器 → τ_ho=5 s 交接给慢相模块。

**收益**：T_stable（显示首次停下、此后 30 s 自身漂移 ≤5%×阶跃）恒载 9 组中位 **3.82 s → 0.55 s**；
恒载多数稳态指标反超 v5（全段时漂 1.46% vs 1.50/1.85）。
**代价**：阶跃保真 1.00 → 1.08、epoch 数 1.6~1.7 倍、**实录类全程最大偏差 1889 → 4571~4583 ADC**。
⇒ 单次加载/长保压类工况可替代现役实现，**多次变载的实录类工况还不能**。

本桶另含一条对旧文档的实证修正（`07-v6算法说明.md` §内）：既有「快相 0~4 s 爬升」轮廓是分析脚本
τ=2 s 平滑造成的伪影。`v6与免责1s3s对比.md` §9 是用户报告两个具体现象后的第二轮修复与逐帧根因。
""",
    fig_note="`I1`（13 份录制小倍数总览）、`I2`（4 面板指标）、`J1`（检测窗扫描）、`K1`（第二轮修复验证）。",
    script_note="`cb~ch_*` 为形状/原始上升/尾部/估计器/泛化/自形状/单点诊断（v6 规格的实测依据）；"
                "`ci_v6_vs_1s_3s.py` 为 13×3 路对比（I1/I2）；`cj~cs_*` 为冒烟/追踪/保持/离群/用户报告复现/卸载/状态演化；"
                "`cp_v6_detwin_sweep.py` 检测窗扫描（J1）；`ct_v6_verify_figs.py` 修复验证（K1）。",
    repro=[
        "python scripts/cb_v6_stepshape.py     # 加载形状实测",
        "python scripts/ci_v6_vs_1s_3s.py      # v6 vs 免责 1s/3s × 13 份（约 5 分钟，I1/I2）",
        "python scripts/cp_v6_detwin_sweep.py  # 检测窗扫描（J1）",
        "python scripts/ct_v6_verify_figs.py   # 修复验证（K1）",
    ],
)

B["08-v6.1"] = dict(
    title="v6.1 —— v6 的阶跃尖峰定点修复（原型，未落地）",
    meta=[
        "时间：2026-09-18 16:47 ~ 21:27",
        "状态：**原型已跑通并实测，未落地 C++**",
        "只解决一个问题：切换负载类数据上阶跃仍带**尖峰**，而单一负载（恒载）看不到",
    ],
    body="""
**现象**：同一套 v6 代码，阶跃在两类数据上表现不同——恒载 9 组冲到真值之上 3.5% 后原始继续爬升把它「吃掉」，
肉眼不可见；`切换负载-快相无责` 是「快速到位 + 保压」，原始不再爬升 ⇒ 前置量无处可去 ⇒
显示 0.5 s 内冲到真值之上 +3,696 ADC，随后被停滞检测一步拉回 ⇒ **尖峰**。

**根因**（逐帧核对）：逆模型的形状库是 **13 份录制的中位形状**，比现场加载慢
（本例 0.2 s 已走完真值增量的 88%，ROM 只认 79%）⇒ `Â` 系统性高估 +2.7~+10.9%。

**修复**：形状库按「最快实测形状」**上包络重标**（只对 onset 生效，一行系数）+ 前置量下行限速。
**结果**：尖峰「峰后回落」由 **+11.9% → +0.4%**，恒载全套指标同步变好；`08-v6.1算法说明.md` 含 C++ 落地补丁与代价清单。
""",
    fig_note="`L1_v61_spike_fix.png`（尖峰修复前后逐帧）、`L2_v61_metrics.png`（指标对比）、`M1_v61_13ffca.png`（13ffca 定点）。",
    script_note="`cu/cv` 尖峰诊断与过冲量化；`cw_v61_ab.py` 13 份全指标 A/B（复用 `pv_common` 口径）；"
                "`cx/cz/dd` 冒烟与等价性；`cy_v61_shapecal.py` 形状库标定；`da_v61_window.py` 窗口扫描；"
                "`db_v61_overest.py` Â 高估量化；`dc_v61_figs.py` 出图；`de_v61_ledger.py` 逐事件台账；"
                "`df_v61_ablate.py` 消融；`dg_v61_panelcheck.py` 面板自检；`dh/di` 13ffca 定点与消融。",
    repro=[
        "python scripts/cu_v61_spike_diag.py  # 尖峰诊断（约 1 分钟）",
        "python scripts/cw_v61_ab.py          # 13 份全指标 A/B（约 5 分钟）",
        "python scripts/dc_v61_figs.py        # 出图 L1/L2",
        "python scripts/dh_v61_13ffca.py      # 13ffca 定点图 M1",
    ],
    notes="本桶脚本 `import pv_common`，该模块（`11-paper-v6` 的公共层）已随桶复制到本桶 `scripts/`，同目录可导入。",
)

B["09-v7"] = dict(
    title="v7 —— 已否决的变体（存档）",
    meta=[
        "时间：2026-09-17 11:47 ~ 11:53（与 v4 同期）",
        "状态：**未采用**，仅作为对照留档",
        "设计：免责期**只在首次 onset 生效**，负载内变载完全交给 v3 的 restep 原子迁移",
    ],
    body="""
v7 想解决的是「v4 免责期把真实变载也冻住导致欠报」，思路是把免责期限定在首次 onset。
自检确认它确实在补偿（`A_max = 1.3063`、末端补偿 +0.81），但
**在变化负载上与 v3 逐帧完全相同** ⇒ 对变载工况没有任何改善，
所以推荐方案仍是 v4/v5 的语义（首次 onset 与变载都走免责期）。

> ⚠ 命名坑：`z3_v6.py` 的名字写的是 v6，语义实为后来的 v7（首次 onset 固定 5 s 冻结、变载不冻结），
> 因此归入本桶；真正的 v6 是 `07-v6` 的 `glm53_v6.py`（形状约束反演）。
> 另：`05-v5算法说明.md` §1.3 指出 `02` §5.3 的「v7 ≡ v3」结论表述不成立，以 `05` 的复核为准。
""",
    fig_note="无独立图件（结论为「与 v3 逐帧相同」，见 `results/varying_steps_v7.csv`）。",
    script_note="`glm53_v7.py` 算法本体；`z7_v7_test.py` 自检；`ad_check_v7.py` 对照检查；`z3_v6.py` 同名异构的早期版本。",
    repro=["python scripts/z7_v7_test.py   # v7 自检（对照 v3 / v4）"],
)

B["10-paper-v5-route"] = dict(
    title="论文（一）：无责 3 s / 5 s 路线",
    meta=[
        "时间：2026-09-18 10:54 ~ 13:52",
        "交付：`docs/抗蠕变漂移补偿算法.md`（论文正文）+ 6 张配图 + 全部可复现脚本与数据",
        "口径：与 `11-paper-v6` 并列、同时间轴同指标定义，互不覆盖",
    ],
    body="""
论文正文写的是**已落地的 v5.1 路线**：把「零漂」与「时漂（蠕变）」显式区分为两种不同性质的量，
只处理后者，并用三条语义把它锁住（非负载段输出等于输入 / 快相免责期 / 输出封顶）。

**摘要数字**：9 组恒载负载段内原始漂移占幅度 5.32%~33.75%（均值 15.43%）；
算法把**慢相段时漂残余从 11.79% 压到 1.41%**，变载台阶透传比中位 0.96，
并用独立参考实现逐帧对拍确认落地实现与原型相等。
代价明确写出：从真实加载沿到扣除量到位约 17.7 s（2.5 s 抗瞬态误判 + 3 s 隔离快相 + 其余过程平滑）。

**图件审查留痕**：本轮 6 张图在定稿前做过像素级复核（裁剪放大、逐面板核对文字越界与重叠），
过程产物在 `12-fig-audit`。
""",
    fig_note="`F1_two_phase`（快慢相两段结构）、`F2_overview`（算法总览）、`F3_step_detect`（阶跃判据）、"
             "`F4_latency`（时延分解）、`F5_scenarios`（场景）、`F6_static9`（恒载 9 组）。",
    script_note="`pd.py` 是公共层（数据装载、口径、指标），`figstyle.py` 是绘图层；`pf1~pf6` 逐图产出。"
                "跨桶依赖 `ad_lib / glm53_v3 / ad_v4 / glm53_v5 / glm53_v51` 已随桶复制到本桶 `scripts/`。",
    repro=[
        "python scripts/pf1_physical.py   # F1 加载形状的实测事实",
        "python scripts/pf2_overview.py   # F2 算法总览",
        "python scripts/pf3_step_mech.py  # F3 阶跃判据",
        "python scripts/pf4_timeline.py   # F4 时延分解",
        "python scripts/pf5_scenarios.py  # F5 场景",
        "python scripts/pf6_static9.py    # F6 恒载 9 组",
    ],
)

B["11-paper-v6"] = dict(
    title="论文（二）：v6 抗蠕变补偿算法",
    meta=[
        "时间：2026-09-18 16:57 ~ 18:04",
        "交付：`docs/抗蠕变补偿算法_v6.md`（约 4.5 万字，8 章 + 3 附录 + 7 图）+ `docs/README.md`（工作区说明）",
        "特点：把**为拟合实机数据所做的全部 trade-off** 完整写成台账",
    ],
    body="""
论文只节选 v6 **纯抗蠕变算法**部分（即「最终画图/最终交付」的那一版），
并写出为拟合实机数据做的全部取舍。

**一句话结论**：v6 只换掉「快相处理」这一段（把「等快相走完」换成「用标定形状把快相算掉」），
慢相蠕变模块逐行沿用 v5；收益是首次加载平稳时刻 **3.82 s → 0.55 s**，
代价是阶跃保真 1.00 → 1.08、epoch 数 1.6~1.7 倍、实录全程最大偏差 1889 → 4583 ADC。
**单次加载/长保压类工况可替代现役实现，多次变载的实录类工况还不能。**

**图件验收是两层口径**：出图脚本内置 `figcheck`（面板数、空白面板、文字越界与重叠）；
另有 `pv_vision_check.py` 把图与「期望描述」一起送 vision 模型逐项回答
（面板数/重叠/截断/空白/缺曲线/图例遮挡/乱码），最新一次 **7 张全部 PASS**，记录在 `results/vision_check.md`。
`pv_audit.py` 对论文里引用的数字做审计（期望 OK 74 / BAD 0）。

**未做/未验证**（论文 §末已声明）：未改 `src/`；未做真机与界面手测；新参数只有检测窗做过扫描；
形状库用测试集自身标定（**乐观上界**）；时间轴对 τ<0.2 s 有畸变，v6 数字可能低估其真实能力。
""",
    fig_note="`F1_two_phase`、`F2_overview`、`F3_invert_glide`（形状约束反演与滑行器）、`F4_slow_creep`、"
             "`F5_overview13`（13 份录制）、`F6_fast_tradeoff`、`F7_slow_tradeoff`。",
    script_note="`pv_common.py` 是公共层（数据清单、指标口径、缓存），`pv_style.py` 绘图层；"
                "`pv_run.py` 主体复算（13×5 臂）；`pv_trim_ablation.py` trim 三档消融；"
                "`pf1~pf7` 逐图产出；`pv_vision_check.py` 图件视觉验收；`pv_audit.py` 数字审计。",
    repro=[
        "python scripts/pv_run.py             # ① 主体复算 13×5 臂（≈5 分钟）→ metrics_*.csv + cache/",
        "python scripts/pv_trim_ablation.py   # ② trim 三档消融（≈4 分钟）",
        "python scripts/pf1_physical.py       # 以下逐图",
        "python scripts/pv_vision_check.py    # ③ 图件视觉验收（需 DEEPSEEK_API_KEY）",
        "python scripts/pv_audit.py           # ④ 数字审计（期望 OK 74 / BAD 0）",
    ],
    notes="`pf1_physical.py` / `pf3_invert_glide.py` 要读 v6 的既有分析产物，路径已指向 `progress/07-v6/results/`；"
          "`pv_vision_check.py` 从 `~/.dsh/.credentials.yaml` 读 `DEEPSEEK_API_KEY`，不硬编码。",
)

B["12-fig-audit"] = dict(
    title="图件审查留痕（交付图的像素级复核）",
    meta=[
        "时间：2026-09-18 10:01 ~ 13:20（穿插在论文（一）出图过程中）",
        "性质：**过程留痕**，不是交付物；审查对象是 `10-paper-v5-route` 的 6 张交付图（中间修订版）",
        "用途：追溯「某张图为什么改」——裁剪放大、逐面板像素核对文字越界/重叠/串色的原始证据",
    ],
    body="""
四个子目录对应四种复核手法：

| 子目录 | 文件数 | 内容 |
|---|---|---|
| `figures/_crop/` | 83 | 按行/按面板/按图例带状裁剪，附 `f4_*.txt` / `f6_*.txt` 像素扫描文本（如红/灰曲线在 x=2/10/25/45/60/75% 处的 y 坐标） |
| `figures/_review_crops/` | 54 | 按 a/b/c 三张图分组的可见性问题裁剪（图例、标注、重叠、标题、轴标签） |
| `figures/_zoom/` | 28 | 逐面板放大（P1~P9）与四角/边缘/缝隙检查 |
| `figures/_audit/` | 14 | 定稿前的最终复核裁剪（F4 柱状图与 F6 各面板） |

> ⚠ 这些裁剪来自**中间修订版**的图（其后脚本又改过版面并重出图），
> 因此与 `10-paper-v5-route/figures/` 下最终图的尺寸并不一一对应——这是预期内的情况，不是归档错位。
>
> 其中的 `.txt` 是像素扫描/ASCII 映射结果，与 `.png` 同属审查证据，故一并保留在 `figures/` 下。
""",
    fig_note="见上表（共 179 个文件，全部为本桶内容，无脚本与数据）。",
    script_note="无脚本：这些裁剪由当时的临时 Python 片段生成，未落盘为可复现脚本。",
    repro=["# 无复现入口：过程留痕，仅作追溯用"],
)

ORDER = ["01-v3-baseline", "02-dsp-route", "03-v4", "04-v5", "05-v5.1",
         "06-v5.1-exempt-1s-3s-5s", "07-v6", "08-v6.1", "09-v7",
         "10-paper-v5-route", "11-paper-v6", "12-fig-audit"]


def tree_of(bdir, sub):
    p = os.path.join(bdir, sub)
    out = []
    if os.path.isdir(p):
        for root, dirs, files in os.walk(p):
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(root, f), bdir).replace(os.sep, "/")
                out.append((rel, os.path.getsize(os.path.join(root, f))))
    return sorted(out)


def main():
    for b in ORDER:
        bdir = os.path.join(PROG, b)
        info = B[b]
        docs = tree_of(bdir, "docs")
        figs = tree_of(bdir, "figures")
        scr = tree_of(bdir, "scripts")
        res = tree_of(bdir, "results")
        extra = {}
        for sub in ("source", "cpp_check"):
            t = tree_of(bdir, sub)
            if t:
                extra[sub] = t
        L = []
        L.append("# %s —— %s\n" % (b, info["title"]))
        L.append("> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。\n")
        for m in info["meta"]:
            L.append("- " + m)
        L.append("")
        L.append(info["body"].strip() + "\n")
        # 文档
        L.append("## 文档（%d）\n" % len(docs))
        if docs:
            L.append("| 文件 | 大小 |")
            L.append("|---|---|")
            for rel, sz in docs:
                L.append("| `%s` | %s |" % (rel, "%d KB" % max(1, sz // 1024)))
        else:
            L.append("（无）")
        L.append("")
        # 图件
        L.append("## 图件（%d）\n" % len(figs))
        L.append(info["fig_note"] + "\n")
        if figs:
            if len(figs) <= 40:
                for rel, sz in figs:
                    L.append("- `%s`（%d KB）" % (rel, max(1, sz // 1024)))
            else:
                from collections import Counter
                c = Counter(rel.split("/")[1] if "/" in rel else "(根)" for rel, _ in figs)
                L.append("| 子目录 | 张数 |")
                L.append("|---|---|")
                for k, v in sorted(c.items()):
                    L.append("| `figures/%s` | %d |" % (k, v))
        L.append("")
        # 脚本
        L.append("## 脚本（%d）\n" % len(scr))
        L.append(info["script_note"] + "\n")
        if scr:
            names = [rel[len("scripts/"):] for rel, _ in scr]
            py = [n for n in names if n.endswith(".py")]
            other = [n for n in names if not n.endswith(".py")]
            L.append("```text")
            for i in range(0, len(py), 6):
                L.append("  ".join(py[i:i + 6]))
            if other:
                L.append("（编译缓存：" + ", ".join(other) + "）")
            L.append("```")
        L.append("")
        # 数据
        L.append("## 数据（%d）\n" % len(res))
        if res:
            names = [rel[len("results/"):] for rel, _ in res]
            L.append("```text")
            for i in range(0, len(names), 4):
                L.append("  ".join(names[i:i + 4]))
            L.append("```")
        else:
            L.append("（无）")
        L.append("")
        for sub, t in extra.items():
            L.append("## `%s/`（%d）\n" % (sub, len(t)))
            L.append("```text")
            for rel, _sz in t[:40]:
                L.append(rel)
            if len(t) > 40:
                L.append("… 其余 %d 个文件（构建产物）" % (len(t) - 40))
            L.append("```")
            L.append("")
        if info.get("notes"):
            L.append("## 注意\n")
            L.append(info["notes"] + "\n")
        L.append("## 复现\n")
        L.append("```powershell")
        L.append("cd temp/v4.1flash/progress/%s" % b)
        L.append("$env:PYTHONIOENCODING='utf-8'")
        for r in info["repro"]:
            L.append(r)
        L.append("```")
        L.append("")
        with io.open(os.path.join(bdir, "MANIFEST.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(L))
        print("%-28s docs=%3d figs=%3d scripts=%3d results=%3d" %
              (b, len(docs), len(figs), len(scr), len(res)))


if __name__ == "__main__":
    main()
