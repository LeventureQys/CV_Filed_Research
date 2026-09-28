// Copyright (c) 2025 Modulus Technology. All rights reserved.
//
// 文件: src/domain/drift/drift_compensator.cpp
// 描述: 见同目录 drift_compensator.h。

#include "domain/drift/drift_compensator.h"

#include <algorithm>
#include <cmath>

namespace drift {
namespace {

// 门控比较容差(随历史最大电平缩放, 兼容空载总量恰为 0 的工况)
inline double GateEps(double max_ts) {
    return 1e-6 * (1.0 + std::abs(max_ts));
}

}  // namespace

void DriftCompensator::Reset() {
    n_ = 0;
    first_frame_ = true;
    last_ts_ = 0.0;
    t0_ = 0.0;
    ts_smooth_ = 0.0;
    min_ts_ = 0.0;
    max_ts_ = 0.0;
    level_ref_ = 0.0;
    fast_ = 0.0;
    slow_ = 0.0;
    pending_ = false;
    pending_ts_ = 0.0;
    armed_ = false;
    in_load_ = false;
    onset_ts_ = 0.0;
    hold_ = false;
    a_captured_ = false;
    a_frames_ = 0;
    a_new_frames_ = 0;
    g_ = 0.0;
    g2_acc_ = 0.0;
    b_.resize(0);
    a_acc_.resize(0);
    A_.resize(0);
    g_rel_acc_.resize(0);
    gamma_.resize(0);
    hold_comp_.resize(0);
    a_new_acc_.resize(0);
    loaded_.clear();
    scratch_.clear();
}

void DriftCompensator::ResetFor(int n) {
    Reset();
    n_ = n;
    b_ = Eigen::VectorXd::Zero(n);          // 初始 0 = 直通安全语义
    a_acc_ = Eigen::VectorXd::Zero(n);
    A_ = Eigen::VectorXd::Zero(n);
    g_rel_acc_ = Eigen::VectorXd::Zero(n);
    gamma_ = Eigen::VectorXd::Ones(n);
    a_new_acc_ = Eigen::VectorXd::Zero(n);
    a_new_frames_ = 0;
    hold_ = false;
    hold_comp_.resize(0);
    loaded_.assign(n, 0);
    scratch_.reserve(n);
}

void DriftCompensator::BeginLoad(double ts) {
    in_load_ = true;
    onset_ts_ = ts;
    a_captured_ = false;
    a_acc_.setZero(n_);
    a_frames_ = 0;
    g_ = 0.0;
    g2_acc_ = 0.0;
    g_rel_acc_.setZero(n_);
    gamma_.setOnes(n_);
}

void DriftCompensator::AlignRefs(double level_ref_value) {
    level_ref_ = level_ref_value;
    fast_ = level_ref_value;
    slow_ = level_ref_value;
    pending_ = false;
    pending_ts_ = 0.0;
}

void DriftCompensator::Restep(double ts, const Eigen::VectorXd& z_now) {
    // v3 原子迁移: A 当帧取 pending 期 Z(=v-b) 的逐通道均值并立即生效
    // (无 1~3s 直通黑障); g 按补偿连续性锚定 median(冻结补偿/(γ·A_new)),
    // 保证 restep 前后显示连续(无单帧跳变); γ 与 g2_acc_/g_rel_acc_ 保留
    // 不重置——新段 rel 从 0 起步而 g 为携带值, 若重置则首帧 rel/g≈0.3
    // 会把 γ 一帧砸到下限 0.3, 补偿坍缩(实测 2365→638)。
    in_load_ = true;
    onset_ts_ = ts;
    a_acc_.setZero(n_);
    a_frames_ = 0;
    if (a_new_frames_ > 0) {
        A_ = a_new_acc_ / static_cast<double>(a_new_frames_);
    } else {
        A_ = z_now;
    }
    const double amax = A_.maxCoeff();
    if (amax > 1e-9) {
        for (int i = 0; i < n_; ++i)
            loaded_[i] = A_(i) > kLoadedFrac * amax ? 1 : 0;
    } else {
        std::fill(loaded_.begin(), loaded_.end(), 0);
    }
    a_captured_ = true;

    g_ = 0.0;
    scratch_.clear();
    if (hold_comp_.size() == static_cast<Eigen::Index>(n_)) {
        for (int i = 0; i < n_; ++i) {
            if (loaded_[i] && A_(i) > 1e-9) {
                const double denom = gamma_(i) * A_(i);
                scratch_.push_back(std::abs(denom) > 1e-9 ? hold_comp_(i) / denom : 0.0);
            }
        }
        if (!scratch_.empty()) {
            const auto mid = scratch_.begin() + static_cast<long>(scratch_.size() / 2);
            std::nth_element(scratch_.begin(), mid, scratch_.end());
            g_ = std::clamp(*mid, 0.0, 1.0);
        }
    }

    hold_ = false;
    hold_comp_.resize(0);
    a_new_acc_.setZero(n_);
    a_new_frames_ = 0;
    pending_ = false;
    pending_ts_ = 0.0;
}

void DriftCompensator::Process(double ts, Eigen::VectorXd& v) {
    const int n = static_cast<int>(v.size());
    if (n <= 0) return;
    if (n != n_) ResetFor(n);

    // ── 阵列总量与三级平滑(总量 / 快慢 EMA / 电平参考) ──
    const double total = v.sum();
    double dt = 0.0;
    if (first_frame_) {
        // EMA 初始化只在首帧: 后续重复时间戳帧(dt==0)只跳过状态更新,
        // 绝不快照到当前值(否则快慢 EMA 在阶跃处瞬间同时收敛, 发散恒为 0)
        first_frame_ = false;
        last_ts_ = ts;
        t0_ = ts;
        ts_smooth_ = total;
        fast_ = total;
        slow_ = total;
        min_ts_ = total;
        max_ts_ = total;
        level_ref_ = total;
    } else {
        dt = ts - last_ts_;
        last_ts_ = ts;
        if (!(dt > 0.0)) dt = 0.0;          // 重复时间戳/回退: 本帧不更新状态
        if (dt > 0.1) dt = 0.1;             // 大间隔(卡顿/暂停)截断
        if (dt > 0.0) {
            ts_smooth_ += (dt / kTauTotalSmoothS) * (total - ts_smooth_);
            fast_ += (dt / kTauFastS) * (total - fast_);
            slow_ += (dt / kTauSlowS) * (total - slow_);
            level_ref_ += (dt / kTauLevelRefS) * (ts_smooth_ - level_ref_);
        }
    }
    min_ts_ = std::min(min_ts_, ts_smooth_);
    max_ts_ = std::max(max_ts_, ts_smooth_);
    const double eps = GateEps(max_ts_);

    // ── 阶跃检测(双阈值 + 持续时间确认) ───────────
    //   负载态: max(0.18×slow, 0.01×历史最大电平), 与对数蠕变的发散
    //           理论上界(~6~11%)可分, 双向(加重/减轻)均检测;
    //   空载态: max(0.5×slow, 0.01×历史最大电平)高阈 + 方向门(仅上升沿
    //           fast>slow)——卸载后 fast 快于 slow 回落的 settling 发散
    //           (slow>fast)不再误判为加载, 免疫假 onset。
    const double div_onset_thr = kOnsetRel * std::max(slow_, eps);
    const double div_step_thr =
        std::max(kStepRel * std::max(slow_, eps), kStepAbsFrac * max_ts_);
    const double div = std::abs(fast_ - slow_);
    const double u = in_load_ ? ts - onset_ts_ : 0.0;
    const double thr =
        in_load_ ? div_step_thr : std::max(div_onset_thr, kStepAbsFrac * max_ts_);
    const bool step_now = in_load_ ? (div > thr) : (div > thr && fast_ > slow_);

    if (step_now) {
        if (!pending_) {
            pending_ = true;
            pending_ts_ = ts;
        }
    } else if (pending_ && div < kPendingResetFrac * thr) {
        pending_ = false;                   // 瞬态(手指调整/短暂磕碰): 取消
    }
    const bool step_confirmed = pending_ && (ts - pending_ts_ > kStepPersistS);

    // 卸载判据: 电平跌破 10%×参考 或 1.5×历史最小(任一成立)
    const bool idle = (ts_smooth_ < kIdleFrac * std::max(level_ref_, eps) ||
                       ts_smooth_ < kUnloadMinRatio * min_ts_ + eps);

    if (!in_load_) {
        if (std::abs(ts - t0_) <= 1e-9) {
            // 首帧: 带载使能/连接即受载 → 直接进负载段(空载时无害:
            // A 即空载偏置且 g≈0); 此后不再走比率路径(微小空载域会抖动)
            if (total > eps) {
                BeginLoad(ts);
                AlignRefs(ts_smooth_);
            }
        } else if (step_confirmed) {
            // 空载→负载
            BeginLoad(ts);
            AlignRefs(fast_);
        }
        hold_ = false;
    } else if (u > kUnloadFastS && idle) {
        // 快速卸载判定(不等发散, 避开抑制窗)
        in_load_ = false;
        armed_ = true;
        AlignRefs(ts_smooth_);
        hold_ = false;
        hold_comp_.resize(0);
        pending_ = false;
        pending_ts_ = 0.0;
    } else if (u > kStepSuppressS && step_confirmed) {
        if (idle) {
            // 负载→空载
            in_load_ = false;
            armed_ = true;
            AlignRefs(ts_smooth_);
            hold_ = false;
            hold_comp_.resize(0);
            pending_ = false;
            pending_ts_ = 0.0;
        } else {
            // 负载→负载(加重/减轻): v3 原子迁移重捕获(非 BeginLoad)
            Restep(ts, v - b_);
            AlignRefs(fast_);
        }
    } else if (pending_) {
        // v3: pending 期冻结蠕变补偿(不积分 g/γ, 补偿保持 pend 起始值),
        // 并积累新电平样本 Z=v-b 供 restep 原子迁移使用。
        hold_ = true;
        a_new_acc_ += v - b_;
        ++a_new_frames_;
    } else {
        hold_ = false;
        hold_comp_.resize(0);
        if (a_new_frames_ > 0) {
            a_new_acc_.setZero(n);
            a_new_frames_ = 0;
        }
    }

    // ── 空载基线跟踪(仅确认过加载-卸载循环, 且电平远低于历史最大) ──
    if (!in_load_ && armed_ && ts_smooth_ < kBaseGateFrac * max_ts_) {
        const double a = dt / kTauBaselineS;
        b_ += a * (v - b_);
    }

    Eigen::VectorXd Z = v - b_;

    if (!in_load_) {
        v = std::move(Z);
        return;
    }

    // ── 负载段: 幅度估计(加载后 1~3s 历史窗) ──────
    if (!a_captured_) {
        if (u >= kAWindowStartS && u <= kAWindowEndS) {
            a_acc_ += Z;
            ++a_frames_;
        }
        if (u > kAWindowEndS) {
            A_ = a_frames_ > 0 ? a_acc_ / static_cast<double>(a_frames_) : Z;
            const double amax = A_.maxCoeff();
            if (amax > 1e-9) {
                for (int i = 0; i < n; ++i)
                    loaded_[i] = A_(i) > kLoadedFrac * amax ? 1 : 0;
            } else {
                std::fill(loaded_.begin(), loaded_.end(), 0);
            }
            a_captured_ = true;
        } else {
            v = std::move(Z);
            return;
        }
    }

    // ── v3: pending 冻结——补偿保持 pend 起始值, 不积分 g/γ ──
    if (hold_) {
        if (hold_comp_.size() != static_cast<Eigen::Index>(n)) {
            // 惰性初始化: pend 可能在 A 捕获窗内开始, 当帧捕获完成后补算冻结值
            hold_comp_ = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < n; ++i) {
                if (loaded_[i] && A_(i) > 1e-9) {
                    hold_comp_(i) = std::clamp(gamma_(i) * A_(i) * g_,
                                               kCreepLoFrac * A_(i),
                                               kCreepHiFrac * A_(i));
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            double creep = 0.0;
            if (loaded_[i] && A_(i) > 1e-9)
                creep = hold_comp_(i);
            v(i) = Z(i) - creep;
        }
        return;
    }

    // ── 负载段: 蠕变场共识 g(t) + 逐通道增益 γ ─────
    scratch_.clear();
    for (int i = 0; i < n; ++i) {
        if (loaded_[i] && A_(i) > 1e-9)
            scratch_.push_back((Z(i) - A_(i)) / A_(i));
    }
    if (!scratch_.empty()) {
        const auto mid = scratch_.begin() + static_cast<long>(scratch_.size() / 2);
        std::nth_element(scratch_.begin(), mid, scratch_.end());
        const double g_raw = *mid;
        g_ += (dt / kTauCreepSmoothS) * (g_raw - g_);
    }

    if (g_ > kGEnable) {
        g2_acc_ += dt * g_ * g_;
        for (int i = 0; i < n; ++i) {
            if (loaded_[i] && A_(i) > 1e-9) {
                const double rel = (Z(i) - A_(i)) / A_(i);
                g_rel_acc_(i) += dt * g_ * rel;
            }
        }
        if (g2_acc_ > 1e-8) {
            for (int i = 0; i < n; ++i) {
                double gamma = 1.0;
                if (loaded_[i])
                    gamma = std::clamp(g_rel_acc_(i) / g2_acc_, kGammaMin, kGammaMax);
                gamma_(i) = gamma;
            }
        }
    }

    // ── 蠕变扣除(仅受载通道, 逐通道限幅) ──────────
    for (int i = 0; i < n; ++i) {
        double creep = 0.0;
        if (loaded_[i] && A_(i) > 1e-9) {
            creep = gamma_(i) * A_(i) * g_;
            creep = std::clamp(creep, kCreepLoFrac * A_(i), kCreepHiFrac * A_(i));
        }
        v(i) = Z(i) - creep;
    }
}

void DriftCompensationCoordinator::SetEnabled(bool on) {
    if (enabled_ == on) return;
    enabled_ = on;
    ResetAll();  // 开关切换后从全新状态开始
}

void DriftCompensationCoordinator::Process(const std::string& key,
                                            const std::string& signature,
                                            double timestamp_s,
                                            Eigen::VectorXd& values_io) {
    if (!enabled_) return;
    Entry& entry = targets_[key];
    if (entry.signature != signature) {
        entry.comp.Reset();
        entry.signature = signature;
    }
    entry.comp.Process(timestamp_s, values_io);
}

void DriftCompensationCoordinator::ResetAll() {
    targets_.clear();
}

}  // namespace drift
