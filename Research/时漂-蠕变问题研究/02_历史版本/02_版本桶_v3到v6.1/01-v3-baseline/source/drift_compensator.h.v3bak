// Copyright (c) 2025 Modulus Technology. All rights reserved.
//
// 文件: src/domain/drift/drift_compensator.h
// 描述: 触觉阵列传感器显示层时漂/零漂补偿（GLM53 分析推荐的混合方案, v3
//       阶跃感知 + 切换连续性修正版）。
//       两级结构:
//         ① 卸载门控自动归零——空载期基线快速跟踪、负载期冻结, 解决零漂;
//         ② 负载比例蠕变场 + 逐通道增益——蠕变_i(t) ≈ γ_i·A_i·g(t),
//            g(t) 为受载通道归一化残差的 median 共识, γ_i 为过原点增量
//            最小二乘在线修正, 解决粘弹性蠕变(时漂)。
//       全部严格因果: 任意时刻输出只依赖当前与历史采样。
//       算法对比与三传感器实测见 temp/GLM53/（时漂 38.6%→4.8%, 弱蠕变
//       场景无过补偿; 延迟: 信号通路零群延迟, 补偿收敛 90%@13~17s）。
//       变化负载分析与 v3 方案见 temp/GLM53/变化负载分析与v2方案.md。
//
// v2/v3 相对 v1 的差异（阶跃感知混合算法）:
//   - v1 假设蠕变幅度 A 恒定, 负载中途变化时 A 过期: 真实阶跃被当作蠕变
//     扣除(受限幅钳制), 显示被"钉"在旧幅度附近并振荡; 部分卸载又被误判为
//     完全卸载而直接丢补偿。v2 用双 EMA 发散检测负载阶跃, 阶跃统一驱动
//     三种迁移(空载→负载 / 负载→负载 / 负载→空载), 并在每次迁移后重新捕获 A;
//   - 阶跃检测: fast(τ=0.7s) vs slow(τ=6s) 的绝对发散超过状态相关阈值
//     （空载态用 0.5×slow 高阈, 免疫近零空载域噪声; 负载态用
//     max(0.18×slow, 0.01×历史最大电平)——对数蠕变的发散理论上界约 6~11%）;
//     发散须持续 2.5s 才确认(手指调整/短暂磕碰 1~2s 内回落即取消);
//   - 抑制窗: onset 后 6s 内不做负载内阶跃判定, 避开蠕变快相;
//   - 卸载判据改为「电平跌破 10%×电平参考 或 1.5×历史最小」, u>3s 即可快速
//     判定(不等发散), 解决近零空载域 1.5×min_ts 失效;
//   - 基线跟踪门控改为「电平 < 20%×历史最大」;
//   - 修复 dt==0 快照缺陷: EMA 初始化只在首帧, 后续重复时间戳帧只跳过状态
//     更新, 绝不快照(否则快慢 EMA 在阶跃处瞬间同时收敛, 发散检测失效)。
//
// v3 相对 v2 的差异（切换连续性修正; 变化负载数据实测 v2 在切换处存在
// 单帧 ~7561 ADC 纯算法跳变, 用户反馈后修正, 详见变化负载分析与v2方案.md §4.3）:
//   - pending 期冻结蠕变补偿: 阶跃待确认期间不积分 g/γ, 补偿保持 pend
//     起始值 → 显示立即跟随真实阶跃, 不再被旧段 A/g 反向拖低;
//   - restep 原子迁移: A 当帧取 pending 期 Z(=v-b) 逐通道均值(无 1~3s
//     直通黑障), g 按 median(冻结补偿/(γ·A_new)) 锚定, γ 与 g2/g_rel
//     累加器保留不重置(否则新段 rel 从 0 起步、首帧 rel/g≈0.3 会把 γ
//     一帧砸到下限 0.3, 补偿坍缩 2365→638), restep 前后显示连续;
//   - 空载→负载 onset 加方向门(仅上升沿 fast>slow)与绝对下限
//     (1%×历史最大): 卸载后 fast/slow 回落 settling 不再误判为加载,
//     假 onset 及其连带误 restep 消失。
//
// 产品级安全语义:
//   - 冷启动/带载使能期间 b=0 → 直通, 绝不塌显示;
//   - 蠕变扣除仅作用于受载通道, 逐通道限幅 [-0.5·A, 1.5·A],
//     γ 限幅 [0.3, 2.0]; 无法建立可靠估计时自然退化为直通。

#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

namespace drift {

class DriftCompensator {
public:
    // 就地补偿一帧显示值。timestamp_s 单调递增(秒); 通道数变化自动重置。
    void Process(double timestamp_s, Eigen::VectorXd& values_io);

    void Reset();

private:
    void ResetFor(int n);
    // 进入负载段(首次加载): 重新捕获幅度 A 并清空蠕变状态(γ 回到 1)。
    void BeginLoad(double ts);
    // 负载内阶跃(加重/减轻)重捕获: v3 原子迁移——当帧用 pending 期 Z 均值
    // 给出 A, g 按补偿连续性锚定, γ 与 g2/g_rel 保留, 无 1~3s 直通窗。
    void Restep(double ts, const Eigen::VectorXd& z_now);
    // 迁移时对齐近期参考与快慢 EMA(发散归零, 参考电平取给定值)。
    void AlignRefs(double level_ref_value);

    // ── 参数(v2, 与 GLM53 原型一致, 恒载回归参数零重调) ──
    static constexpr double kTauTotalSmoothS = 0.3;   // 阵列总量平滑
    static constexpr double kTauFastS        = 0.7;   // 阶跃检测快 EMA
    static constexpr double kTauSlowS        = 6.0;   // 阶跃检测慢 EMA
    static constexpr double kTauLevelRefS    = 10.0;  // 近期电平参考 EMA
    static constexpr double kTauBaselineS    = 2.0;   // 空载基线跟踪
    static constexpr double kTauCreepSmoothS = 3.0;   // g(t) 平滑
    static constexpr double kOnsetRel        = 0.5;   // 空载→负载发散阈(相对 slow)
    static constexpr double kStepRel         = 0.18;  // 负载内阶跃发散阈(相对 slow)
    static constexpr double kStepAbsFrac     = 0.01;  // 负载内阶跃绝对下限(历史最大)
    static constexpr double kStepPersistS    = 2.5;   // 阶跃确认持续时长(瞬态拒绝)
    static constexpr double kStepSuppressS   = 6.0;   // onset 后负载内阶跃抑制窗
    static constexpr double kUnloadFastS     = 3.0;   // 快速卸载判定最小 u
    static constexpr double kIdleFrac        = 0.10;  // 卸载判据: 电平参考 10%
    static constexpr double kUnloadMinRatio  = 1.5;   // 卸载判据: 历史最小 1.5×
    static constexpr double kBaseGateFrac    = 0.20;  // 基线跟踪门控: 历史最大 20%
    static constexpr double kPendingResetFrac = 0.5;  // 发散回落至此比例即取消待确认
    static constexpr double kLoadedFrac      = 0.10;  // 受载通道入选阈
    static constexpr double kGammaMin        = 0.3;   // 逐通道增益限幅
    static constexpr double kGammaMax        = 2.0;
    static constexpr double kCreepLoFrac     = -0.5;  // 蠕变扣除限幅
    static constexpr double kCreepHiFrac     = 1.5;
    static constexpr double kGEnable         = 0.02;  // γ 更新门槛
    static constexpr double kAWindowStartS   = 1.0;   // 幅度 A 采集窗 [1,3]s
    static constexpr double kAWindowEndS     = 3.0;

    int n_ = 0;
    bool first_frame_ = true;
    double last_ts_ = 0.0;
    double t0_ = 0.0;         // 首帧时间戳(仅用于首帧判定)

    // 门控状态
    double ts_smooth_ = 0.0;
    double min_ts_ = 0.0;
    double max_ts_ = 0.0;
    double level_ref_ = 0.0;
    double fast_ = 0.0;       // 阶跃检测快 EMA
    double slow_ = 0.0;       // 阶跃检测慢 EMA
    bool pending_ = false;    // 阶跃待确认(发散已起, 尚未持续 kStepPersistS)
    double pending_ts_ = 0.0;
    bool armed_ = false;      // 已确认经历加载-卸载循环, 允许基线跟踪
    bool in_load_ = false;    // 当前处于负载段
    double onset_ts_ = 0.0;

    // v3 切换连续性: pending 期冻结补偿 + restep 原子迁移锚定
    bool hold_ = false;               // pending 期冻结蠕变补偿(不积分 g/γ)
    Eigen::VectorXd hold_comp_;       // 冻结补偿值(size()==0 表示未捕获)
    Eigen::VectorXd a_new_acc_;       // pending 期 Z(=v-b) 均值累加器
    int a_new_frames_ = 0;

    // 自动归零基线(初始 0 = 直通; 仅在 armed 且空载时更新)
    Eigen::VectorXd b_;

    // 负载段状态
    bool a_captured_ = false;
    Eigen::VectorXd a_acc_;
    int a_frames_ = 0;
    Eigen::VectorXd A_;
    std::vector<char> loaded_;

    // 蠕变场 + 逐通道增益
    double g_ = 0.0;
    double g2_acc_ = 0.0;
    Eigen::VectorXd g_rel_acc_;
    Eigen::VectorXd gamma_;
    std::vector<double> scratch_;  // median 工作区
};

// 多目标协调器: 按 key 隔离补偿器状态; signature(显示值域签名)变化自动重置。
class DriftCompensationCoordinator {
public:
    void SetEnabled(bool on);
    bool enabled() const { return enabled_; }

    // signature 由调用方构造(显示模式/单位/数据域/标定代等), 变化即重置该目标。
    void Process(const std::string& key, const std::string& signature,
                 double timestamp_s, Eigen::VectorXd& values_io);

    void ResetAll();
    std::size_t target_count() const { return targets_.size(); }

private:
    bool enabled_ = false;
    struct Entry {
        DriftCompensator comp;
        std::string signature;
    };
    std::map<std::string, Entry> targets_;
};

}  // namespace drift
