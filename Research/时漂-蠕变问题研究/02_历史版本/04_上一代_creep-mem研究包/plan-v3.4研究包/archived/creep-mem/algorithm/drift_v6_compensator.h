// Copyright (c) 2025 Modulus Technology. All rights reserved.
//
// 文件: src/domain/drift_v6/drift_v6_compensator.h
// 描述: 显示层「触觉阵列加载瞬态快速稳定算法 v6」——严格因果的在线补偿器,
//       与 src/domain/drift 的免责期方案(v5)和 src/domain/creep 的变点感知
//       对数蠕变方案互为替代(菜单「设备」下四档算法开关互斥, 同一时刻只启用一种)。
//
//       规格来源: temp/v4.1flash/Document/07-v6算法说明.md
//       原型:     temp/v4.1flash/scripts/glm53_v6.py (逐行对齐, 含 §12.2/§12.3/§12.4
//                 的全部实测修正; 冲突时以原型代码为准)
//
//       目标: 把「真实加载沿 → 显示稳定」从 v5 的 7.8 s(可见首扣)/14.5 s(扣到位)
//             压缩到 ≤1.0 s, 手段**不是**把免责期从 3 s 改成 1 s, 而是
//             「不再等快相走完, 而是把快相算出来扣掉」。
//
//       五个环节:
//         ① 检测器 D  短滞后电平差 + σ 门限(近窗 0.20 s / 间隔 0.15 s / 参考窗 0.30 s),
//                      连续 3 帧确认 + 抬升结束后 re-arm。**无 2.5 s 硬确认、无 6 s 抑制窗**,
//                      取而代之的是 0.4 s 可撤销(试探式 + 连续 3 帧迟滞)与尾部 10% 门;
//         ② 分类器     onset / restep / restep_reload(C5) / decrease 四类, 事件命中时锁定。
//                      (规格 §5 的 C3 减重/卸载恢复库未标定, 本版减重走「不启用逆模型」
//                       的保守策略: 沉降窗内保持上一段扣除, 之后重锚 + 慢相抑制);
//         ③ 逆模型 F   形状约束反演 Â = Σy·g / Σg²(电平域最小二乘, 窗口 [0.20, τ]),
//                      形状 ROM = 13 份录制 onset 合并标定的 14 点表;
//         ④ 滑行器 G   速率受限 + smoothstep 滑行, 每帧滚动重估, 重锚时「改速率不改跳变」;
//         ⑤ 慢相模块 S τ ≥ τ_ho(≥3.5 s) 接管, 完全沿用 v5(median 共识 g + 逐通道 γ +
//                      限幅 + 输出封顶), A 按「钉住值」锚定以保证交接无暂态。
//
//       五处实现要点(都是原型实测踩出来的, 不是理论推导):
//         1. 事件时间原点必须**回溯到真实加载沿**(命中前 ≤0.6 s 内的单帧最大跳变),
//            直接把命中帧当 t0 会让形状里 τ=0 处已有 10%~80% 的增量 ⇒ Â 系统性偏低;
//         2. 逆模型用电平域最小二乘而非尾巴外推式: 后者在短窗被 1/mean(f) 放大(可达 ×25),
//            实测把 14.2 的台阶算成 15.7;
//         3. A 是逐通道「无蠕变总电平」而非增量: 写成增量会让 restep 的 g_raw=(Z−A)/A 爆炸;
//         4. y0 必须取「事件前的显示值」(输出环形缓冲均值), 不能用 (v0 − 当前输出) 反推;
//         5. 检测/回溯一律用 3 帧中值总量: 原始总量会出现 −8% 的单帧掉点(实测 右拇指/数据2
//            @147.31 s), 足以伪造一次「减重」事件。
//
//       安全语义(与 v5 一致): 空载直通不归零、扣除量封顶为当前读数(显示不为负)、
//       未受载通道直通; 任何时刻估计不可靠都自然退化为直通, 绝不塌显示。
//
//       已知边界:
//         - 采样率: 原型与实测口径均为 ~100.5 Hz。检测的 3 帧持续判据与滑行器的单帧限速
//           都按帧计算, 低采样率(≤10 Hz)下 0.20 s 近窗只有 1~2 帧、滑行会明显变慢 ——
//           本版按 100 Hz 链路落地, 低帧率链路的参数需重新标定;
//         - 内存: 每个目标保留 2 × (通道数 × 4096) 双精度环形缓冲(约 2 MB @31 通道),
//           多目标(多设备/多 session)时按目标数线性增长, 由协调器随签名变化释放;
//         - restep 类的形状库未按现场标定(现值为杂样本中位, 只作占位), 且 1 s 内输入本身
//           只走完 64%, 故 restep **不承诺 1 s 到位**(规格 §8.2), 只承诺跟随斜坡、不做过度外推;
//         - 形状先验的跨工况/跨加载方式误差会直接进显示(规格 §12.3(a)), 现场标定是唯一出路;
//         - 未做真机/界面手测(按项目规范交用户执行)。

#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

namespace drift_v6 {

class DriftV6Compensator {
public:
    // 就地补偿一帧显示值。timestamp_s 单调递增(秒); 通道数变化自动重置。
    void Process(double timestamp_s, Eigen::VectorXd& values_io);

    void Reset();

    // ── 诊断/自测用只读状态 ──
    bool in_event() const { return ev_.valid; }
    bool in_slow() const { return state_ == State::Slow; }
    // 形状 ROM 命中计数(>0 说明形状库真的被用上了; =0 即静默退化)
    long shape_hits() const { return shape_hits_; }
    // 谷底判据状态(设计文档 §4.1.1 C-4: 供单元测试断言; 只读, 不改任何状态)
    bool   in_valley() const { return valley_now_; }
    double valley_run_s() const { return valley_run_; }
    // 慢相共识残余比 g(设计文档 §4.1: SubStage 3A 亦用; 只读)
    double slow_residual_g() const { return g_; }
    // C 限幅命中格点数(plan-v2.0 SubStage 3A: 供回归锁测试区分"限幅生效"与"恒真断言"; 只读)
    long clamp_hits() const { return params_.n_clamp; }

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
        // C 限幅的诊断计数(plan-v2.0 SubStage 3A): ApplyOutputLimit 实际改写过的**格点**数
        // (通道×帧)。=0 只可能出现在"限幅被关闭"或"全帧走直通出口"两种情形;
        // 供单元测试把 `out ≤ raw + α·|raw|` 从恒真断言区分为"限幅真的生效过"。
        mutable long n_clamp   = 0;
        bool   legacy_fixes_enabled = kLegacyFixesOn;   // F2/F3 整组开关(默认 false)
        double kappa_onset       = kKappaOnset;
        double kappa_restep      = kKappaRestep;
        double ho_min_s          = kHoMinS;
        int    revoke_hold_n     = kRevokeHoldN;
        // ── plan-v3.0：慢相逐通道蠕变跟踪（PCT，见 kPctTauS 注释）──
        double pct_tau_s         = kPctTauS;      // 0 = 关闭（回到 plan-v2.0 行为，逐位一致）
        double pct_elig_frac     = kPctEligFrac;  // PCT 生效通道门（A_k > f·maxA）
        double pct_lo_frac       = kPctLoFrac;    // PCT 扣除下界（相对 A_k）
        double pct_hi_frac       = kPctHiFrac;    // PCT 扣除上界（相对 A_k）
        bool   pct_mono          = kPctMono;      // true = PCT 只允许加大扣除（蠕变单调）
        bool   freeze_slow_on_hit = kFreezeSlowOnHit;  // 检测器命中当帧起暂停慢相自适应
        // ── plan-v3.4：卸载-重载蠕变记忆 ──
        double mem_tau_s         = kMemTauS;      // 0 = 关闭（回到 plan-v3.1 行为）
        double mem_max_frac      = kMemMaxFrac;   // 记忆扣除总量 / 新 Â 的上限
        // 诊断计数(只写不读业务逻辑)
        long n_valley_exit    = 0;
        long n_reanchor_idle  = 0;
        long n_g_floor        = 0;
        long n_g_valley_reset = 0;
        long n_pct_hits       = 0;   // PCT 真正写入过扣除量的帧·通道数
    };

    const Params& params() const { return params_; }
    void SetParams(const Params& p);   // 只改参数与计数, 不改补偿状态(n_/state_/A_/g_/环缓)

    // ── 参数(与原型 glm53_v6.py 一一对应) ──
    // 预处理: 总量平滑 / 近期电平参考 EMA
    static constexpr double kTauTotalS = 0.3;
    static constexpr double kTauLevelS = 10.0;

    // 检测器 D: 判据跨度 = FAST + GAP + LAG = 0.65 s(cp_v6_detwin_sweep 的扫描采用值:
    // 恒载全段 1.11%、慢相段 1.27% 最好; 拉长到 0.80/0.95 s 反而变差 —— 参考窗会把
    // 蠕变斜坡混进 lv_ref, 抬高门限也拖慢响应。T_stable 在 0.45~0.95 s 各档都是 0.55 s,
    // 即**拉长检测窗不牺牲速度**: 滑行器本身要 0.6~0.8 s, 把检出延迟吸收了)
    static constexpr double kDetFastS  = 0.20;   // 近窗
    static constexpr double kDetGapS   = 0.15;   // 间隔
    static constexpr double kDetLagS   = 0.30;   // 参考窗
    static constexpr double kDetK      = 5.0;    // σ 门限倍数
    static constexpr double kDetRel    = 0.05;   // 相对参考电平(= v5 的 kLevRel)
    static constexpr double kDetAbsFrac = 0.01;  // 相对历史最大总量(= v5 的 kLevAbsFrac)
    static constexpr double kDetIdleFrac = 0.10; // 空载态下额外要求
    static constexpr int    kDetPersist = 3;     // 连续帧数(≈30 ms @100Hz)
    static constexpr double kIdleSettleS = 0.50; // 回到空载后的静默期(卸载沿后的回弹不再开新 epoch)
    static constexpr double kUnloadBlockS = 0.80; // 一次减重/卸载处理完后的事件静默期
    static constexpr double kBackdateS = 0.60;   // 真沿回溯窗
    static constexpr double kTailGateS = 3.0;    // epoch 尾巴内只认 ≥10% 电平的真实变载
    static constexpr double kTailGateFrac = 0.10;

    // 逆模型 F
    static constexpr double kTauRef = 0.20;      // 估计锚点(时间戳量化在 0.20 s 后收窄到 4~9%)
    static constexpr double kAWin   = 0.60;      // 窗长: [τ_ref, τ_ref+0.6]
    // Â 的单侧上限 = min(A, κ·inc), **只有反演值超过该上限时才生效**。
    // onset 逐事件的 r_max 落在 1.207~2.925(中位 1.2466) ⇒ κ≥1.28 时对 22 个 onset 一个都不
    // 触发(原值 1.30 实际等同「从未启用」, 这也是「κ 时灵时不灵」的来源)。
    // 1.05 是 26 臂 × 13 份统一扫描下唯一同时满足 1~2 s 判据的档: T_stable(5 s) 中位
    // 3.66→0.79 s 且 p90 23.44→1.354 s(进带)、留一最坏折 T_stable(10 s) 78.50→1.96 s;
    // 代价是 onset 转为轻微欠报(−3.61%)、G 0.942→0.926(落在安全侧: 欠报由慢相自然补回)。
    // 与「形状库上包络重标(rom_scale)」同类 —— 二者都压 Â, **只上其一, 同时上会过度欠报**。
    // ⚠️ r_max 是本批 13 份录制的统计量、不是器件物理常数: 换夹具/加载方式/传感器后须重标。
    static constexpr double kKappaOnset  = 1.05;
    static constexpr double kKappaRestep = 1.12; // restep 不允许无界外推(输入 36% 还没发生)

    // 滑行器 G
    static constexpr double kGlideMinS = 0.40;
    static constexpr double kGlideMaxS = 0.80;
    static constexpr double kRateMax   = 0.8;    // 单帧最大变化比例(/s)
    // 交接时刻 = max(kHoMinS, τ_g0 + τ_glide), 即「形状模型走完之后」才交给慢相模块。
    // 原注释「必须取 τ=5 s, 越早越差」在聚合口径下不成立: HO_MIN 5.0→3.5 s 时
    // T_stable(5 s) 中位 3.66→1.96 s、err1 34.1→31.3%、G 0.942→0.933, 未见 07-v6 记录的
    // 3~9 s 暂态(那是单事件观察, 与聚合口径不矛盾、只是尺度不同) ⇒ 本版取 3.5 s。
    // ⚠️ 交接提前会让慢相 g 从锚定值一路爬到真实蠕变水平, 需真机 A/B 复核;
    // 同一常量在重复性口径下方向相反(5→10 s 使 std(R5) 5.09→2.57 pp), 但「加载后 10 s 内
    // 再次变载」的工况未验证, 而变载实录里这类工况常见 ⇒ 暂不采用 10 s。
    static constexpr double kHoMinS = 3.5;
    static constexpr double kRevokeS = 0.40;       // 试探撤销窗
    // 撤销判据的迟滞帧数: 判定连续成立 kRevokeHoldN 帧才真撤销(=3 ⇒ ≈30 ms @100Hz)。
    // ±1 包时序抖动下 inc_s 会跨过抬升沿瞬时读到 0, 单帧即撤销会让该次加载整段不补偿
    // (显示停在原始读数上约 10 s) —— 这是 v6 路径抖动是 v5.1 的 30.6 倍的主因;
    // 加迟滞后 L3 路径抖动 7.71%→0.72%·电平, 而 L2 与 T_stable 逐位不变(=1 即原型行为)。
    static constexpr int    kRevokeHoldN = 3;
    static constexpr double kRevokeCooldownS = 0.30;
    static constexpr double kDecreaseSettleS = 0.30;
    static constexpr double kReanchorSmoothS = 0.25;

    // 停滞检测(输入停住时不再「预判尾巴」): 判据 = 实测尾巴增长比 vs 模型尾巴增长比。
    // 无量纲、与加载方式无关: 正常尾巴(恒载族 0.11~0.13、实录族 0.19)都 ≥ 模型值,
    // 输入停住则实测 ≈0(噪声 <0.01), 差一个数量级。斜率门与「模型-实测落差」门都被实测证伪。
    static constexpr double kStallStartS = 0.60;
    static constexpr double kStallTailFrac = 0.50;
    static constexpr double kStallHoldS  = 0.45;
    static constexpr double kStallMinFrac = 0.30;

    // 交接后 A 的慢修正(默认关)。开启(TRIM_RATE=0.002)可把形状先验造成的平台静态偏置
    // 收进死区, 代价是显示在保压期内会缓慢移动 —— 用户评价「还不如修复前稳定」, 故回退。
    static constexpr double kTrimRate = 0.0;
    static constexpr double kTrimDeadFrac = 0.025;

    // 慢相模块(= v5)
    static constexpr double kTauG = 3.0;
    static constexpr double kLoadedFrac = 0.10;
    static constexpr double kGammaMin = 0.3;
    static constexpr double kGammaMax = 2.0;
    static constexpr double kCreepLoFrac = -0.5;
    static constexpr double kCreepHiFrac = 1.5;
    static constexpr double kGEnable = 0.02;
    static constexpr double kIdleFrac = 0.10;
    static constexpr double kUnloadMinRatio = 1.5;
    static constexpr double kUnloadFastS = 0.30;  // C4: v5 的 u>3s 收到 0.3s

    static constexpr int kCap = 4096;   // 帧环形缓冲(>=40 s @100Hz)
    static constexpr int kDcap = 1024;  // 检测统计量环形缓冲

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

    // ── plan-v3.0 新增：慢相逐通道蠕变跟踪（PCT, Per-Channel Tracking）──
    // 定位（temp/v4.1flash/plan/v3.0/results/分析报告_稳定工况残漂.md）:
    //   原慢相是 **rank-1 比例模型** `ded_k = γ_k·A_k·g(t)` —— 一个共识标量 g 乘一条
    //   **固定的**逐通道空间剖面 γ_k·A_k（A_k 来自加载幅度、γ_k 是无遗忘的历次最小二乘）。
    //   实测在「恒定负载长保压」工况（20260919_141824, 单次加载后保压 393 s、输入自身
    //   蠕变 +2672 ADC/+23%）下，保压期残漂的 **96% 来自 A_k 小于 10%·maxA 因而不在
    //   loaded_ 掩码里的通道**（ch5 独占蠕变的 21.4%）；且受载通道内部扣除呈「冻结→跳变」
    //   阶梯（γ 无遗忘 ⇒ 跟踪的是 epoch 时均蠕变而非当前蠕变）。
    // PCT 是**级联在 rank-1 之外**的一条慢环：把每个通道的**剩余电平误差**积分掉，
    //   目标 = 该通道的 A_k（交接时刻钉住的无蠕变电平）。
    //   ded_k = clamp(γ_k A_k g, …) + pct_k,  pct_k += (dt/τ)·((v_k − A_k) − ded_rank1_k − pct_k)
    //   量纲是**绝对电平**（不是相对比例），因此不受「蠕变远大于该通道载荷幅度」的限制。
    // 实测（18 条录制，真实 C++ 本体，指标 = 保压期内显示相对落定电平的偏离 / 台阶）:
    //   目标录制 0.0325 → **0.0021**（−94%，15.5×）；15/17 条录制改善；
    //   2 条轻微反向（右拇指指尖/数据3 +8%、振荡工况 f40a1b −5% 属中性）；
    //   C 限幅不变量仍 0 越界。详见 plan/v3.0/results/。
    // τ 从 5 s 到 30 s 效果相近（目标录制 0.0058~0.0095），取 10 s。
    // pct_hi_frac 是**残余漏扣的主导约束**：实测末帧 ch1/ch2 的 crept 量超过 2×A_k 被截住
    //   （ch1 creep 160 / A=45、ch2 creep 133 / A=31），4.0 正好放开它们；4.0 → 8.0 无进一步收益
    //   且在其余 17 条录制上与 2.0 逐位相同 ⇒ 取 4.0。
    static constexpr double kPctTauS      = 10.0;   // 0 = 关闭 PCT
    static constexpr double kPctEligFrac  = 0.02;   // A_k > 2%·maxA 的通道才跟踪
    static constexpr double kPctLoFrac    = -0.5;   // pct 下界(相对 A_k)
    static constexpr double kPctHiFrac    = 4.0;    // pct 上界(相对 A_k)
    static constexpr bool   kPctMono      = false;  // true = 只加大不回调（备选，见 v3.0 报告）

    // ── plan-v3.1 新增：检测器命中当帧起暂停慢相自适应 ──
    // 定因（temp/v4.1flash/plan/v3.1/）：慢相 `g` 是 τ=3 s 的 EMA，而行/卸载发生时
    // `g_raw` 会瞬间跳变；检测器要 3 帧 + 0.2 s 近窗才确认（≈0.1~0.4 s），这段时间里
    // `g` 已经朝新电平走了一截 ⇒ 扣除被侵蚀；随后 Reanchor 用"当前（已侵蚀）的扣除"
    // 反推 A_new，把侵蚀**固化**。实测现场「反复增减同一负载」录制里每次卸载后显示比
    // 应有值高 ≈8~15%·台阶（卸载 G 中位 0.92，而加载是 0.98）。
    // 处置：`raw_hit` 为真的帧**只施加扣除、不更新 g、也不积分 PCT**（阶跃不是蠕变）。
    static constexpr bool   kFreezeSlowOnHit = true;

    // ── plan-v3.4 新增：卸载-重载蠕变记忆（creep memory）──
    // 定因（temp/v4.1flash/plan/v3.4/，working/零基线-反复增减同一负载 20260919_160854）:
    //   整片卸载（4 级台阶、~1.5 s 塌到基线）走 Decrease → Reanchor → ToIdle 后，慢相
    //   扣除被全部清空；随后重新加载作为「全新 Onset」只用形状反演定锚（A ≈ 快相终点
    //   电平），把整段受载历史里累积的蠕变补偿（实测 ~2000 ADC）全部遗忘 ⇒ 显示 ≈
    //   原始输入（仅剩 C 限幅上限），而未经整片卸载的加载段显示 = 输入 − 累积补偿，
    //   同一负载的两族显示相差 ~2000 ADC（一致性丧失，即用户报的「丢基线」）。
    // 处置：Decrease 事件建立当帧（慢相状态尚未被 Reanchor 侵蚀）快照总扣除向量与
    //   检测器参考窗电平；下一次从 Idle 出发的 Onset 在滑行目标与 Handoff 锚定里
    //   减去按新台阶比例缩放、随卸载后时间指数衰减的记忆扣除。
    // mem_step 缩放：重载台阶小于被记忆的卸载台阶时按比例折减（避免小负载过扣）；
    // mem_max_frac：记忆扣除总量 ≤ 该比例 × 新 Â（安全上限）。
    static constexpr double kMemTauS   = 120.0;  // 记忆衰减时间常数（0 = 关闭，回到 v3.1 行为）
    static constexpr double kMemMaxFrac = 0.50;  // 记忆扣除总量相对新 Â 的上限

private:
    enum class State { Idle, Event, Slow };
    // 工况分类(规格 §5 六类中本版启用的四类; C3 减重/卸载恢复的形状库未标定,
    // 故 Decrease 走「不启用逆模型」的保守策略):
    //   Onset        C1 空载→负载: 形状库 onset 档, κ=1.05, 目标 ≤1.0 s
    //   Restep       C2 负载内加重: κ=1.12 + 停滞检测, 稳定时刻受输入斜坡限制
    //   RestepReload C5 复合事件(epoch 内再变载 / 形状失配): A←Â_old+Δ̂, 不重启形状
    //   Decrease     C3 负载内减重: 直通 + 沉降窗后重锚 + 只压慢相
    enum class Kind { Onset, Restep, RestepReload, Decrease };

    // 环形缓冲窗口标记: 物理索引(升序)对应时间落在 (t0, t1] 的帧。
    // 时间戳不保证严格单调(同包时间戳几乎相同、重复时间戳帧只跳状态更新),
    // 因此窗口一律用「样本序号(time-based mask)」口径, **不用包时间戳做微分**。
    struct BufMask {
        int idx[kCap];
        int count = 0;
        bool any() const { return count > 0; }
    };

    struct EventCtx {
        bool valid = false;
        Kind kind = Kind::Onset;
        bool stalled = false;
        State prev_state = State::Idle;
        double t0 = 0.0;
        double t_det = 0.0;
        double base = 0.0;
        Eigen::VectorXd v0;
        Eigen::VectorXd y0;
        double base_y = 0.0;
        double c0 = 0.0;
        Eigen::VectorXd y_prev;
        double a_hat = 0.0;
        double inc_max = 0.0;
        double dec_max = 0.0;
        int revoke_run = 0;                          // 试探撤销判据的连续成立帧数(迟滞)
        bool inc_ref_set = false;
        double inc_ref = 0.0;
        double stall_t = 0.0;
        double c_applied = 0.0;
        bool tau_g0_set = false;
        double tau_g0 = 0.0;
        double tglide = kGlideMaxS;
        bool use_mem = false;                        // plan-v3.4：本事件是否套用蠕变记忆
        std::vector<std::pair<double, double>> hist;  // (τ, inc): τ∈[0,1.0] 全部样本
    };

    void ResetFor(int n);

    // 谷底判据: 每帧在 ts_smooth_ 更新之后、任何状态判断之前调用(见 .cpp Process)
    void UpdateValley(double dt);

    // 显示单侧限幅: out_i = min(out_i, raw_i + alpha·|raw_i|)。只改输出、不改任何内部状态。
    void ApplyOutputLimit(const Eigen::VectorXd& raw, Eigen::VectorXd* out) const;

    // ── 缓冲 ──
    void Push(double ts, const Eigen::VectorXd& v, const Eigen::VectorXd& y);
    void FillMask(double t0, double t1, BufMask* out) const;
    bool WinMean(double t0, double t1, double* out) const;          // 3 帧中值总量的窗均值
    bool MatMean(const Eigen::MatrixXd& buf, int n, double t0, double t1,
                 Eigen::VectorXd* out) const;                        // 逐通道窗均值
    void Backdate(double ts_now, double* t0, double* base,
                  std::vector<std::pair<double, double>>* hist) const;

    // ── 逆模型 ──
    bool InvEst(const std::vector<std::pair<double, double>>& hist,
                double tau, double kappa, double* a_out) const;

    // ── 事件 ──
    void NewEvent(double t0, double base, const Eigen::VectorXd& v0,
                  const Eigen::VectorXd& y0, Kind kind);
    // 返回 true 表示本帧已给出输出(out 有效)
    bool RunEvent(double ts, const Eigen::VectorXd& v, double total, double dt,
                  double eps, bool idle_now, Eigen::VectorXd* out);
    Eigen::VectorXd ShareVector(const EventCtx& ev, const Eigen::VectorXd& v) const;

    // ── 慢相 ──
    Eigen::VectorXd Rank1DeductionVector() const;   // 只含 rank-1（v5 口径，供 g 锚定用）
    Eigen::VectorXd DeductionVector() const;        // 慢相**实际施加**的总扣除（rank-1 + PCT）
    // plan-v3.4：本次事件应补齐的蠕变记忆扣除（逐通道；无效/关闭/无需补齐时返回零）。
    // 残差口径：目标记忆（按当前受载电平比例缩放、随卸载后静置衰减）减去显示血统
    // 中**已嵌入**的扣除（= 事件前的 base − base_y）——因此 Onset 从零补齐、
    // 家族 A 的重载事件（已带全额扣除）自然为零、楼梯式重载的后续事件只补差额。
    Eigen::VectorXd CreepMemory(double ts, double base, double a_hat,
                                double embedded_ded) const;
    void Handoff(double ts, const Eigen::VectorXd& v);
    void Reanchor(double ts, const Eigen::VectorXd& v);
    void ToIdle();
    Eigen::VectorXd SlowStep(const Eigen::VectorXd& v, double dt);
    void TrimA(double dt);

    int n_ = 0;
    bool first_frame_ = true;
    double last_ts_ = 0.0;
    double ts_smooth_ = 0.0;
    double level_ref_ = 0.0;
    double min_ts_ = 0.0;
    double max_ts_ = 0.0;
    double max_tot_ = 0.0;

    // 帧环形缓冲(时间戳/中值总量/逐通道原始/逐通道输出)
    double buf_t_[kCap];
    double buf_v_[kCap];        // 3 帧中值总量
    Eigen::MatrixXd buf_vx_;    // n × kCap 逐通道原始值
    Eigen::MatrixXd buf_yx_;    // n × kCap 逐通道输出值
    long buf_n_ = 0;
    double med_hist_[3];
    int med_n_ = 0;
    double prev_med_total_ = 0.0;   // 上一帧的 3 帧中值总量(写入缓冲的 buf_v_)

    // 检测统计量环形缓冲(σ 在线估计; 仅用非事件窗的 d 分布)
    double det_d_[kDcap];
    long det_n_ = 0;
    double sig_d_ = 0.0;
    int hit_run_ = 0;
    int quiet_run_ = 99;
    bool armed_ = true;

    // 运行期参数(plan-v2.0)
    Params params_;
    // 谷底判据: 长度 kValleyWinN 的环形缓冲(存 ts_smooth_)
    double valley_buf_[kValleyWinN];
    long   valley_n_   = 0;      // 累计写入帧数(用于判断"数据是否足够")
    bool   valley_now_ = false;
    double valley_run_ = 0.0;    // 谷底连续时长(秒)

    State state_ = State::Idle;
    EventCtx ev_;
    double ev_end_ = 0.0;
    double last_ev_ts_ = -1e9;
    double idle_since_ = -1e9;
    double ev_block_until_ = -1e9;

    // 慢相模块状态
    Eigen::VectorXd A_;
    std::vector<char> loaded_;
    double g_ = 0.0;
    bool   slow_freeze_ = false;   // plan-v3.1：本帧是否暂停慢相自适应(检测器命中)
    double g2_acc_ = 0.0;
    Eigen::VectorXd g_rel_acc_;
    Eigen::VectorXd gamma_;
    Eigen::VectorXd hold_comp_;   // size()==0 表示未捕获(仅减重沉降窗内使用)
    double trim_target_sum_ = -1.0;
    // plan-v3.0：逐通道蠕变跟踪（PCT）的扣除量与生效掩码（绝对电平口径，单位同输入）
    Eigen::VectorXd pct_ded_;
    std::vector<char> pct_elig_;

    // plan-v3.4：卸载-重载蠕变记忆。Decrease 建立当帧快照总扣除向量 mem_ded_ 与
    // 检测器参考窗电平 mem_lv_；下一次 Idle 出发的 Onset 套用（见 CreepMemory）。
    Eigen::VectorXd mem_ded_;
    double mem_ts_ = -1e9;      // 快照时刻
    double mem_lv_ = 0.0;       // 快照时的检测器参考窗电平（塌落前）
    bool   mem_valid_ = false;

    // 诊断计数（InvEst 为 const 查询式接口，形状命中计数随估计一并更新）
    mutable long shape_hits_ = 0;
    long n_revoke_ = 0;

    // 上一帧的显示值(y): y0 必须取「事件前的显示值」(见 .cpp 说明)
    Eigen::VectorXd prev_out_;
};

// 多目标协调器: 按 key 隔离补偿器状态; signature(显示值域签名)变化自动重置。
class DriftV6CompensationCoordinator {
public:
    void SetEnabled(bool on);
    bool enabled() const { return enabled_; }

    // signature 由调用方构造(显示模式/单位/数据域/标定代等), 变化即重置该目标。
    void Process(const std::string& key, const std::string& signature,
                 double timestamp_s, Eigen::VectorXd& values_io);

    void ResetAll();
    std::size_t target_count() const { return targets_.size(); }

    // 记忆式运行期参数: 记在协调器自身, 并在"新建条目 / signature 变化重置"后重新注入。
    // 背景见 temp/v4.1flash/plan/v2.0/设计文档.md §4.1.1 C-5:
    //   DriftV6Compensator 首帧会 ResetFor -> Reset -> params_ = Params(),
    //   因此"首帧之前"注入的参数会被静默清空。
    void SetParams(const DriftV6Compensator::Params& p);
    const DriftV6Compensator::Params& params() const { return params_; }

private:
    bool enabled_ = false;
    DriftV6Compensator::Params params_;
    struct Entry {
        DriftV6Compensator comp;
        std::string signature;
    };
    std::map<std::string, Entry> targets_;
};

}  // namespace drift_v6
