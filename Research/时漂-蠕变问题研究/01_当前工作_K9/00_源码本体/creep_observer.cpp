// Copyright (c) 2025 Modulus Technology. All rights reserved.
//
// 文件: src/domain/drift_v6/creep_observer.cpp
// 描述: src/domain/drift_v6/creep_observer.h 的实现
//       （plan-v3.4 在线双态蠕变观测器 v3：零点跟踪 + 快态模型 + 慢漂移跟踪）。

#include "domain/drift_v6/creep_observer.h"

#include <algorithm>
#include <cmath>

namespace drift_v6 {

void CreepObserverCompensator::Reset() {
    n_ = 0;
    first_frame_ = true;
    last_ts_ = 0.0;
    x_fast_.resize(0);
    x_slow_.resize(0);
    v_lp_.resize(0);
    zero_.resize(0);
    y_max_.resize(0);
    t_edge_.resize(0);
    applied_.resize(0);
    v_fast_lp_.resize(0);
    y_floor_.resize(0);
    load_dwell_.resize(0);
    ramp_dwell_.resize(0);
    total_baseline_ = 0.0;
    total_noise_ = 0.0;
}

void CreepObserverCompensator::ResetFor(int n) {
    Reset();
    n_ = n;
    x_fast_ = Eigen::VectorXd::Zero(n);
    x_slow_ = Eigen::VectorXd::Zero(n);
    applied_ = Eigen::VectorXd::Zero(n);
    load_dwell_ = Eigen::VectorXd::Zero(n);
    ramp_dwell_ = Eigen::VectorXd::Zero(n);
    t_edge_ = Eigen::VectorXd::Constant(n, 1.0e6);  // 初始视为"早已离沿"
}

void CreepObserverCompensator::Process(double timestamp_s,
                                       Eigen::VectorXd& values_io) {
    const int n = static_cast<int>(values_io.size());
    if (n <= 0) return;
    if (n_ != n) ResetFor(n);

    if (first_frame_) {
        first_frame_ = false;
        last_ts_ = timestamp_s;
        // 首帧初始化：零点/低通/包络都取首帧输入，状态从零开始
        zero_ = values_io;
        y_max_ = Eigen::VectorXd::Zero(n);
        v_lp_ = values_io;
        v_fast_lp_ = values_io;
        y_floor_ = values_io;
        load_dwell_ = Eigen::VectorXd::Zero(n);
        ramp_dwell_ = Eigen::VectorXd::Zero(n);
        t_edge_ = Eigen::VectorXd::Constant(n, 1.0e6);
        applied_ = Eigen::VectorXd::Zero(n);
        total_baseline_ = values_io.sum();
        total_noise_ = 0.0;
        values_io -= applied_;
        return;
    }

    double dt = timestamp_s - last_ts_;
    last_ts_ = timestamp_s;
    dt = (dt > 0.0) ? std::min(dt, 0.1) : 0.0;

    // ── K7 全局总值旁路：空载基线仅空载期跟踪（上下双向漂移都能跟上），门限取
    // max(1.15×基线, 基线+3σ)，低于门限整帧直通；超过 max(1.25×基线, 基线+4σ)
    // 才恢复算法（迟滞，防临界负载抖动）。噪声为 |total−baseline| 的慢速均值。
    // 直通帧仍跑逐通道状态恢复（零点跟踪/快慢态泄放/dwell 清零）——否则状态跨
    // 循环冻结累积（dwell 只进不清 → 确认门失效），空载塌陷换姿势复发。──
    bool bypass_frame = false;
    if (dt > 0.0) {
        const double total = values_io.sum();
        if (std::isfinite(total)) {
            const double alpha = std::min(dt / params_.bypass_base_tau_s, 1.0);
            const double release = std::max(params_.bypass_release_frac * total_baseline_,
                                            total_baseline_ + params_.bypass_noise_sigma * total_noise_);
            const double engage = std::max(params_.bypass_engage_frac * total_baseline_,
                                           total_baseline_ + (params_.bypass_noise_sigma + 1.0) * total_noise_);
            if (total < engage) {
                total_baseline_ += alpha * (total - total_baseline_);
                total_noise_ += alpha * (std::abs(total - total_baseline_) - total_noise_);
                if (total_noise_ < 0.0) total_noise_ = 0.0;
            }
            if (total < release) bypass_frame = true;
        }
    }

    if (dt > 0.0) {
        const Eigen::VectorXd& v = values_io;
        // ── 零点跟踪（K7 去趋势门限）：y_floor 跟踪 y0 慢速最小值（基线），判据改为
        // y0 − y_floor < idle_frac·(y_max − y_floor)。原判据 y0 < idle_frac·y_max 要求
        // 基线 < 5%·包络，实测基线≈7%·包络时空载永不触发（本会话根因③）。受载时冻结。──
        const double y_decay = std::exp(-dt / params_.y_max_tau_s);
        for (int k = 0; k < n; ++k) {
            const double y0 = v(k) - zero_(k);
            if (y0 < y_floor_(k)) {
                y_floor_(k) = y0;  // 下行瞬时跟随（卸载/基线下漂立刻跟）
            } else {
                y_floor_(k) += (dt / params_.y_floor_tau_s) * (y0 - y_floor_(k));
            }
            y_max_(k) = std::max(y_max_(k) * y_decay, std::max(y0, 0.0));
            const double span = std::max(y_max_(k) - y_floor_(k), 1.0);
            const bool idle =
                (y0 - y_floor_(k)) < params_.idle_frac * span;
            load_dwell_(k) = idle ? 0.0 : load_dwell_(k) + dt;
            if (idle) zero_(k) += (dt / params_.tau_zero_s) * (v(k) - zero_(k));
        }
        Eigen::VectorXd y(n);
        for (int k = 0; k < n; ++k) y(k) = v(k) - zero_(k);
        // ── 低通导数（更新前的 v_lp，解析式）：沿检测与慢态共用 ──
        Eigen::VectorXd slope(n);
        for (int k = 0; k < n; ++k)
            slope(k) = (v(k) - v_lp_(k)) / params_.tau_slope_s;
        // ── 快态 + 沿后前馈（H3）＋ 慢态（沿后软冻结）──
        // dx1_rate 与原型口径一致：用**更新前**的 x1 与 e_now 计算（斜率口径）。
        const Eigen::VectorXd e_now = (y - x_fast_ - x_slow_).cwiseMax(0.0);
        const double refract = std::max(params_.edge_refract_s, params_.edge_boost_s);
        for (int k = 0; k < n; ++k) {
            t_edge_(k) += dt;
            const bool fire = slope(k) > params_.edge_slope_thres &&
                              t_edge_(k) > refract;
            if (fire) t_edge_(k) = 0.0;
            // K9 缓坡前馈：亚沿值正斜率持续计时，满 ramp_full_s 后 tc1 线性过渡到
            // 沿后前馈值——缓坡加载（无过充、沿不触发）也能快速收敛。
            const bool ramp_on = slope(k) > params_.ramp_slope_min &&
                                 slope(k) <= params_.edge_slope_thres;
            ramp_dwell_(k) = ramp_on ? ramp_dwell_(k) + dt : 0.0;
            // K7：任一快变（升或降）都视为新加载段，受载确认窗重计——
            // 递减台阶工况下降挡不触发上升沿，dwell 会跨挡累积骗过确认门。
            if (std::abs(slope(k)) > params_.edge_slope_thres) load_dwell_(k) = 0.0;
            double tc1 =
                (params_.edge_boost_s > 0.0 && t_edge_(k) < params_.edge_boost_s)
                    ? params_.tau_c_fast_boost_s
                    : params_.tau_c_fast_s;
            if (params_.ramp_full_s > 0.0) {
                const double w = std::min(ramp_dwell_(k) / params_.ramp_full_s, 1.0);
                tc1 = tc1 * (1.0 - w) + params_.tau_c_fast_boost_s * w;
            }
            const double dx1_rate =
                (e_now(k) > 0.0)
                    ? (params_.r_fast * e_now(k) - x_fast_(k)) / tc1
                    : -x_fast_(k) / params_.tau_r_fast_s;
            x_fast_(k) = std::max(x_fast_(k) + dx1_rate * dt, 0.0);
            v_lp_(k) += (dt / params_.tau_slope_s) * (v(k) - v_lp_(k));
            // 慢态（慢漂移跟踪）：扣除快态已解释部分；沿后软冻结（H3）+ 受载确认（K7）。
            // 空载期按 tau_r_slow_idle_s 快泄放（K7：原仅在 e≤0 时泄放且 τ=150s，
            // 空载残差使 e 常为正 → 只进不出形成逐循环棘轮，本会话根因②）。
            const double e = std::max(y(k) - x_fast_(k) - x_slow_(k), 0.0);
            const double span_k = std::max(y_max_(k) - y_floor_(k), 1.0);
            const bool idle_k = (y(k) - y_floor_(k)) < params_.idle_frac * span_k;
            if (e > 0.0) {
                const double base = std::max(e, 1.0);
                const double cap = params_.slope_cap_frac * base;
                const bool soft_ok =
                    1.0 - std::exp(-t_edge_(k) / params_.soft_unfreeze_s) > 0.5;
                const bool dwell_ok = load_dwell_(k) >= params_.slow_confirm_s;
                if (dwell_ok && soft_ok &&
                    std::abs(slope(k)) < params_.slope_gate_frac * base) {
                    const double dx2 = std::clamp(slope(k) - dx1_rate, -cap, cap);
                    x_slow_(k) += dt * dx2;
                }
                const double hi = params_.r_slow_max * base;
                if (x_slow_(k) > hi) x_slow_(k) = hi;
                if (x_slow_(k) < 0.0) x_slow_(k) = 0.0;
            }
            if (e <= 0.0 || idle_k) {
                const double tau_r2 =
                    (idle_k && params_.tau_r_slow_idle_s > 0.0)
                        ? params_.tau_r_slow_idle_s
                        : params_.tau_r_slow_s;
                x_slow_(k) -= (dt / tau_r2) * x_slow_(k);
                if (x_slow_(k) < 0.0) x_slow_(k) = 0.0;
            }
            // ── K6 预留池：受载期施加补偿的增速 ≤ 输入快速导数+eps ──
            // 会把显示往下拉的部分先存进预留池（x1+x2−applied），慢相随输入爬升释放；
            // 空载直通（applied=x1+x2，保持空载显示≈原始读数语义）。
            if (params_.hold_eps > 0.0) {
                const double slope_a = (v(k) - v_fast_lp_(k)) / params_.hold_tau_s;
                v_fast_lp_(k) += (dt / params_.hold_tau_s) * (v(k) - v_fast_lp_(k));
                const double d_des = x_fast_(k) + x_slow_(k);
                const double a_up = applied_(k) + dt * std::max(slope_a + params_.hold_eps, 0.0);
                const double a_dn = applied_(k) + dt * (slope_a - params_.hold_eps);
                double a = (d_des > applied_(k)) ? std::min(d_des, a_up)
                                                 : std::max(d_des, a_dn);
                if (e > 0.0) {
                    a = std::clamp(a, 0.0, std::max(d_des, 0.0));
                } else {
                    a = d_des;
                }
                applied_(k) = a;
            }
        }
    }
    if (bypass_frame) {
        applied_ = x_fast_ + x_slow_;  // 空载直通：a=d_des，显示不扣
        return;
    }
    if (params_.hold_eps > 0.0) {
        values_io -= applied_;
    } else {
        values_io -= x_fast_;
        values_io -= x_slow_;
    }
}

void CreepObserverCoordinator::SetParams(const CreepObserverCompensator::Params& p) {
    params_ = p;
    for (auto& kv : targets_) kv.second.comp.SetParams(p);
}

void CreepObserverCoordinator::SetEnabled(bool on) {
    if (enabled_ == on) return;
    enabled_ = on;
    ResetAll();
}

void CreepObserverCoordinator::Process(const std::string& key,
                                       const std::string& signature,
                                       double timestamp_s,
                                       Eigen::VectorXd& values_io) {
    if (!enabled_) return;
    const bool is_new = (targets_.find(key) == targets_.end());
    Entry& entry = targets_[key];
    if (is_new || entry.signature != signature) {
        entry.comp.Reset();
        entry.comp.SetParams(params_);
        entry.signature = signature;
    }
    entry.comp.Process(timestamp_s, values_io);
}

void CreepObserverCoordinator::ResetAll() { targets_.clear(); }

}  // namespace drift_v6
