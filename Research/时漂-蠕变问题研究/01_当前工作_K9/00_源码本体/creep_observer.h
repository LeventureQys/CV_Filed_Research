// Copyright (c) 2025 Modulus Technology. All rights reserved.
//
// 文件: src/domain/drift_v6/creep_observer.h
// 描述: 显示层「在线双态蠕变观测器」（plan-v3.4，菜单 v3.4 档的当前实现）。
//
// 模型（原型 Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/v4.1flash/plan/v3.4/scripts/v34_observer_core3.py 的 C++ 落地）:
//   逐通道三个状态：零点 zero、快态 x1、慢态 x2，观测器跑在 y = v − zero 上：
//     y_max = 量程包络（慢衰减的 y 峰值）；idle = y < idle_frac·y_max
//     idle 时 zero 以 τ0 向读数靠拢（空载直通/零点再校准），受载时冻结；
//     e = max(y − x1 − x2, 0)                弹性估计（受载门）
//     x1：固定模型——受载朝 r1·e 以 τc1 收敛、空载以 τr1 恢复（吸收快相爬升）；
//     x2：慢漂移跟踪——受载时按输入低通导数（扣除 x1 份额）积分，上限 r2max·e
//         钳位；沿/快变(|slope|>gate)不积分；空载以 τr2 恢复。
//     显示 = v − x1 − x2                      （= zero + 纯弹性响应）
//   v3 变更（空载漂没修复）：v2 的 e=max(v−x1−x2,0) 把传感器零点偏置（实测 ~2120
//   ADC）当成弹性载荷 ⇒ 全程空载的录制里状态持续生长、显示下沉 −315 ADC；
//   v3 引入零点跟踪后空载显示≈原始读数（均值差 −20 ADC）。
//   v2 变更（恒载衰减修复）：慢态从固定幅度 r·e 改为实测慢漂移跟踪（蠕变幅度
//   因工况差 3 倍，固定幅度必在一边过扣一边欠扣）。
//   无事件分类、无记忆账本；状态跨卸载自然连续（恢复由 τr 决定）。
//   plan-v4 H3 变更（峰后回落修复，2026-09-19）：实测"阶跃后 ~2s 见峰随后回落"的
//   纯过补偿部分 75~85% 来自 x2 沿后过速积分（τ_slope=3s 低通导数高估）+ x1 偏快。
//   H3 = 常数修正（r1=0.10、τc1=12）+ 沿后前馈（沿后 2s 内 τc1→2）+ τ_slope=1 +
//   slope_gate=5% + 沿后 x2 软冻结（1-exp(-t/4s) 过半才开积分）。13 会话 73 事件
//   回放：纯过补偿回落中位 262→49 ADC，落位时间不变；新增状态仅每通道 1 个沿后
//   计时器。研究档案 Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/v4.1flash/plan/v4/。
//   K6 叠加（预留池，用户提议）：显示侧维护"实际施加补偿 applied"（预留池 = x1+x2 −
//   applied）。受载期 applied 增速 ≤ 输入快速导数（τ=0.5s）+2 ADC/s——快相期间会把
//   显示往下拉的部分先扣在预留池里，慢相阶段随输入爬升逐步释放；空载直通。效果
//   （13 会话 73 事件回放）：纯过补偿回落中位 49→27 ADC；反复增减工况 fall 中位
//   332→94、落位 9.2→6.6 s。新增状态：每通道 applied 与快速低通各 1 个。
//   K7 叠加（循环加载棘轮修复 + 空载直通，2026-09-20）：回撤测试（反复 4~6s 加卸载）
//   实测 applied 逐循环棘轮增长至负载的 ~29%（r_slow_max 上限附近）、空载输出塌向 0。
//   根因与修复：① 短保载超过 soft_unfreeze 后被 x2 当慢漂移吸收 → x2 积分需连续受载
//   确认 slow_confirm_s；② 空载残差使 e>0 挡住 x2 泄放 → 空载期按 tau_r_slow_idle_s
//   快泄放；③ 基线为包络的 ~7% 而 idle_frac=5%，原空载门限永不触发 → 改去趋势门限
//   （y0 − y_floor < idle_frac·(y_max − y_floor)，y_floor 为慢速最小值跟踪）；
//   ④ 全局总值旁路：空载基线仅空载期跟踪（能跟上下双向漂移），门限
//   max(1.15×基线, 基线+3σ)，总值低于门限整帧直通，超过 max(1.25×基线, 基线+4σ)
//   才恢复算法（迟滞防抖）。新增状态：每通道 y_floor 与 load_dwell 各 1 个 + 总值
//   基线/噪声标量 2 个。
//
// 实测：空载全程显示−输入均值 −20 ADC；恒载 346s 漂移 −108（输入 +1010）；
// 700s 长保压 +54（输入 +798）；反复增减一致性 10/10 段落在 ±10%·台阶内。
// 详见 Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/v4.1flash/plan/v3.4/算法说明_v3.4观测器.md。
//
// 现场调参（2026-09-23）：5 个参数在 `Params` 默认值里改为调参结论（见下方各成员注释的
// 「调参」标注），并可由「补偿算法选择... → v3.4 观测器 → 参数...」对话框
// （src/ui/dialogs/creep_observer_params_dialog.{h,cpp}）在运行期实时改写；
// 依据 Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/v3.4/（tune_v34 / final_test /
// sc2_test / x2_upward_drift 的「最终 5 参」：过补偿水位、x2 启动延迟与空载泄放一起收）。
// 2026-09-24 追加 2 个 UI 可调参数：tau_c_fast_s（x1 快态收敛 τ）与 slope_cap_frac
// （x2 慢态积分速率上限），二者已有设备寄存器 206/207、随「写入设备」下发；同日再追加
// r_slow_max（x2 慢态幅度上限）——它是让慢态**真正收敛**的那个旋钮：平台期斜率估计的
// 残余正偏会被积分器无限累积（实测 142 s 保压末段显示 −3.5 ADC/s 持续下漂，10 分钟 −6 %），
// 收紧上限后末段斜率收到 −0.02 ADC/s。r_slow_max 亦有设备寄存器 208（×100 定点）。
// 其余参数保持标定值不动，不经 UI 暴露。
// 2026-09-26 全量更新 8 个 UI 参数的默认值（r_fast 0.01 / r_slow_max 0.2 /
// slope_cap_frac 0.05 / slow_confirm_s 5 / soft_unfreeze_s 8 / tau_c_fast_s 40 /
// tau_r_fast_s 6 / tau_r_slow_idle_s 0.5，依据本轮现场调参结论）。

#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>

namespace drift_v6 {

class CreepObserverCompensator {
public:
    struct Params {
        double r_fast  = 0.01;        // 快态幅度比（plan-v4 H3：实测快相≈常数+3~5%·e）
                                      // 调参：0.10→0.06→0.01，快态终值 r_fast·e 决定长尾下漂水位
        double tau_c_fast_s = 40.0;   // 快态受载收敛时间常数（实测等效 τ 9.7~14 s；调参收紧）
        double tau_r_fast_s = 6.0;    // 快态空载恢复时间常数（调参：0.5→6s）
        double r_slow_max = 0.2;      // 慢态上限（相对弹性电平）
        double tau_r_slow_s = 150.0;  // 慢态空载恢复时间常数
        double tau_slope_s = 1.0;     // 慢漂移速率估计的低通时间常数（H3：3→1，减少阶跃后高估）
        double slope_gate_frac = 0.05;  // |dv/dt| 超过 5%·e/s 视为沿/快变，不积分（H3：2%→5%）
        double slope_cap_frac = 0.05;   // 慢态积分速率上限（调参：0.01→0.05）
        double tau_zero_s = 8.0;       // 零点跟踪时间常数（仅近零带内生效）
        double idle_frac = 0.05;       // 近零带：y < 5%·y_max 视为空载
        double y_max_tau_s = 600.0;    // 量程包络的衰减时间常数
        double edge_slope_thres = 60.0;   // H3：上升沿判定阈值（低通导数 ADC/s）
        double edge_refract_s = 2.0;      // H3：沿触发最小间隔
        double edge_boost_s = 2.0;        // H3：沿后前馈窗（窗内快态用 tau_c_fast_boost_s 收敛）
        double tau_c_fast_boost_s = 2.0;  // H3：沿后前馈期的快态收敛 τ
        double soft_unfreeze_s = 8.0;     // H3：沿后 x2 软冻结窗（权重 1-exp(-t/w) 过 0.5 才开积分）
                                          // 调参：4→2→8s
        double hold_eps = 2.0;            // K6：预留池限额余量（ADC/s，受载期施加补偿增速 ≤ 快速导数+eps）
        double hold_tau_s = 0.5;          // K6：预留池限额用的快速低通 τ（不能用 τ_slope，减速段会高估）
        double slow_confirm_s = 5.0;      // K7：x2 积分需连续受载确认时长（防短保载被误吸收为慢漂移）
                                          // 调参：10→2→5s
        double y_floor_tau_s = 300.0;     // K7：去趋势基线包络（y0 慢速最小值跟踪）的爬升 τ
        double tau_r_slow_idle_s = 0.5;   // K7：空载期 x2 快泄放 τ（≤0 回落 tau_r_slow_s）
                                          // 调参：8→2→0.5s，卸载后残留补偿快速清空
        double bypass_release_frac = 1.15;  // K7：总值低于基线×1.15（或基线+3σ）时空载直通
        double bypass_engage_frac = 1.25;   // K7：总值超过基线×1.25（或基线+4σ）才恢复算法（迟滞防抖）
        double bypass_noise_sigma = 3.0;    // K7：门限的噪声带宽（σ 倍）
        double bypass_base_tau_s = 10.0;    // K7：空载期总值基线/噪声的跟踪 τ
        double ramp_slope_min = 0.5;        // K9：缓坡前馈的斜率下限（ADC/s，低于沿阈值、高于蠕变噪声）
        double ramp_full_s = 4.0;           // K9：缓坡持续满该时长后 tc1 全额过渡到沿后前馈值
    };

    void Process(double timestamp_s, Eigen::VectorXd& values_io);
    void Reset();

    const Params& params() const { return params_; }
    void SetParams(const Params& p) { params_ = p; }

private:
    void ResetFor(int n);

    int n_ = 0;
    bool first_frame_ = true;
    double last_ts_ = 0.0;
    Eigen::VectorXd x_fast_;
    Eigen::VectorXd x_slow_;
    Eigen::VectorXd v_lp_;            // 慢漂移速率估计用的低通 v
    Eigen::VectorXd zero_;            // 逐通道零点（近零带内慢速跟踪，受载冻结）
    Eigen::VectorXd y_max_;           // 量程包络（慢衰减的 y 峰值，近零带判据用）
    Eigen::VectorXd t_edge_;          // H3：距上次上升沿触发的时长（驱动前馈/软冻结）
    Eigen::VectorXd applied_;         // K6：实际施加的补偿（预留池 = x1+x2 − applied_）
    Eigen::VectorXd v_fast_lp_;       // K6：预留池限额用的快速低通（τ=hold_tau_s）
    Eigen::VectorXd y_floor_;         // K7：y0 的慢速最小值跟踪（去趋势空载门限用）
    Eigen::VectorXd load_dwell_;      // K7：连续受载确认时长（x2 积分门控）
    Eigen::VectorXd ramp_dwell_;      // K9：缓坡（亚沿值正斜率）持续计时（tc1 渐进前馈用）
    double total_baseline_ = 0.0;     // K7：空载总值基线（仅空载期跟踪，受载冻结）
    double total_noise_ = 0.0;        // K7：空载总值噪声尺度（|total−baseline| 的慢速均值）
    Params params_;
};

// 多目标协调器：按 key 隔离状态；signature(显示值域签名)变化自动重置。
// 与 DriftV6CompensationCoordinator 同构（记忆式参数 + 开关切换全重置）。
class CreepObserverCoordinator {
public:
    void SetEnabled(bool on);
    bool enabled() const { return enabled_; }

    void Process(const std::string& key, const std::string& signature,
                 double timestamp_s, Eigen::VectorXd& values_io);

    void ResetAll();
    std::size_t target_count() const { return targets_.size(); }

    void SetParams(const CreepObserverCompensator::Params& p);
    const CreepObserverCompensator::Params& params() const { return params_; }

private:
    bool enabled_ = false;
    CreepObserverCompensator::Params params_;
    struct Entry {
        CreepObserverCompensator comp;
        std::string signature;
    };
    std::map<std::string, Entry> targets_;
};

}  // namespace drift_v6
