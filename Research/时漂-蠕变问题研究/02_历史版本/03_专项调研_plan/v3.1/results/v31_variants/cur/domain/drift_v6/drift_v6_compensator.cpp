// Copyright (c) 2025 Modulus Technology. All rights reserved.
//
// 文件: src/domain/drift_v6/drift_v6_compensator.cpp
// 描述: src/domain/drift_v6/drift_v6_compensator.h 的实现。
//       逐行对齐原型 temp/v4.1flash/scripts/glm53_v6.py(class GLM53v6), 含
//       Document/07-v6算法说明.md §12.2(M1~M6)/§12.3(F1~F5)/§12.4 的全部实测修正。

#include "domain/drift_v6/drift_v6_compensator.h"

#include <algorithm>
#include <cmath>

namespace drift_v6 {
namespace {

// ── 形状 ROM: g(τ) = [Z(τ)−Z(0)] / [Z(5s)−Z(0)], 13 份录制 onset 合并标定 ──
// 与原型 ROM_TAU / ROM_G 逐点一致, **不可跨设备/跨加载方式搬用**(规格 §7.1)。
constexpr int kRomN = 14;
constexpr double kRomTau[kRomN] = {0.0,  0.05, 0.10, 0.20, 0.30, 0.50, 0.65,
                                   0.80, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00};
constexpr double kRomG[kRomN]   = {0.0,  0.680, 0.740, 0.790, 0.824, 0.854, 0.873,
                                   0.886, 0.904, 0.922, 0.941, 0.970, 0.989, 1.000};

// 形状 ROM 的线性插值(对应 numpy.interp(..., left=0, right=1))
double ShapeAt(double tau) {
    if (tau <= kRomTau[0]) return 0.0;
    if (tau >= kRomTau[kRomN - 1]) return 1.0;
    for (int i = 1; i < kRomN; ++i) {
        if (tau <= kRomTau[i]) {
            const double t0 = kRomTau[i - 1];
            const double t1 = kRomTau[i];
            const double w = (tau - t0) / (t1 - t0);
            return kRomG[i - 1] + w * (kRomG[i] - kRomG[i - 1]);
        }
    }
    return 1.0;
}

// 输出封顶: 扣除量不得超过当前读数(只封顶输出值, 不改内部状态)——同 v5.1。
double Capped(double z, double ded) { return ded > z ? std::max(z, 0.0) : ded; }

// 中位数(会就地排序传入的缓冲区)
double MedianInPlace(std::vector<double>& buf) {
    if (buf.empty()) return 0.0;
    const std::size_t mid = buf.size() / 2;
    std::nth_element(buf.begin(), buf.begin() + mid, buf.end());
    return buf[mid];
}

}  // namespace

void DriftV6Compensator::ApplyOutputLimit(const Eigen::VectorXd& raw,
                                          Eigen::VectorXd* out) const {
    if (out == nullptr) return;
    if (params_.clamp_alpha <= 0.0) return;                 // 0 = 关闭(与既有安全语义一致)
    if (raw.size() != out->size()) return;                  // 尺寸不符直接跳过, 不越界
    for (int k = 0; k < raw.size(); ++k) {
        const double lim = raw(k) + params_.clamp_alpha * std::abs(raw(k));
        if ((*out)(k) > lim) {
            (*out)(k) = lim;
            ++params_.n_clamp;                // 诊断计数(plan-v2.0 SubStage 3A)
        }
    }
}

// ───────────────────────────── 生命周期 ─────────────────────────────

void DriftV6Compensator::Reset() {
    n_ = 0;
    first_frame_ = true;
    slow_freeze_ = false;
    last_ts_ = 0.0;
    ts_smooth_ = 0.0;
    level_ref_ = 0.0;
    min_ts_ = 0.0;
    max_ts_ = 0.0;
    max_tot_ = 0.0;
    buf_n_ = 0;
    med_n_ = 0;
    prev_med_total_ = 0.0;
    det_n_ = 0;
    sig_d_ = 0.0;
    hit_run_ = 0;
    quiet_run_ = 99;
    armed_ = true;
    state_ = State::Idle;
    ev_ = EventCtx();
    ev_end_ = 0.0;
    last_ev_ts_ = -1e9;
    idle_since_ = -1e9;
    ev_block_until_ = -1e9;
    prev_out_.resize(0);
    A_.resize(0);
    loaded_.clear();
    g_ = 0.0;
    g2_acc_ = 0.0;
    g_rel_acc_.resize(0);
    gamma_.resize(0);
    hold_comp_.resize(0);
    trim_target_sum_ = -1.0;
    pct_ded_.resize(0);
    pct_elig_.clear();
    shape_hits_ = 0;
    n_revoke_ = 0;
    // 【plan-v2.0 修正】这里**不再**执行 `params_ = Params();`。
    // 原因: Process 的首帧必然会走 `if (n_ != n) ResetFor(n);` -> Reset(),
    //       于是任何在首帧之前注入的参数都会被静默抹回默认值
    //       (实测: 协调器 Reset()+SetParams() 注入后, 同一个 Process 调用内即被清除,
    //        导致 legacy_fixes_enabled / g_neg_floor 等参数永久停留默认值)。
    // 新语义: **Reset() 只重置补偿状态, 不动运行期参数**;
    //       参数只由 "本文件的 `Params` 默认成员初始化器 / 显式 `SetParams()`" 决定。
    valley_n_ = 0;
    valley_now_ = false;
    valley_run_ = 0.0;
}

void DriftV6Compensator::SetParams(const Params& p) { params_ = p; }

void DriftV6Compensator::ResetFor(int n) {
    Reset();
    n_ = n;
    buf_vx_.setZero(n, kCap);
    buf_yx_.setZero(n, kCap);
    A_ = Eigen::VectorXd::Zero(n);
    g_rel_acc_ = Eigen::VectorXd::Zero(n);
    gamma_ = Eigen::VectorXd::Ones(n);
    loaded_.assign(static_cast<std::size_t>(n), 0);
    pct_ded_ = Eigen::VectorXd::Zero(n);
    pct_elig_.assign(static_cast<std::size_t>(n), 0);
}

// ───────────────────────────── 缓冲 ─────────────────────────────

void DriftV6Compensator::Push(double ts, const Eigen::VectorXd& v,
                              const Eigen::VectorXd& y) {
    const int i = static_cast<int>(buf_n_ % kCap);
    buf_t_[i] = ts;
    buf_vx_.col(i) = v;
    buf_yx_.col(i) = y;
    // 与原型 _push 一致: 本帧写入的是**上一帧算出的 3 帧中值**(prev_med_total_),
    // 本帧中值在 Push 之后、本帧计算开始时才更新。因此窗口内任一帧的值都不会
    // 反向依赖「它自己之后才算出来的东西」。
    // 目的: 单帧掉点(实测 右拇指/数据2 @147.31 s 出现过 −8% 的单帧跌落)会把 0.10 s
    // 窗均值拉动 ~1 ADC 量级, 足以伪造一次「减重」事件。
    buf_v_[i] = prev_med_total_;
    ++buf_n_;
}

void DriftV6Compensator::FillMask(double t0, double t1, BufMask* out) const {
    out->count = 0;
    const long avail = std::min<long>(buf_n_, kCap);
    const long start = buf_n_ - avail;
    for (long k = start; k < buf_n_; ++k) {
        const int i = static_cast<int>(k % kCap);
        const double t = buf_t_[i];
        if (t > t0 && t <= t1) out->idx[out->count++] = i;
    }
}

bool DriftV6Compensator::WinMean(double t0, double t1, double* out) const {
    BufMask m;
    FillMask(t0, t1, &m);
    if (!m.any()) return false;
    double s = 0.0;
    for (int k = 0; k < m.count; ++k) s += buf_v_[m.idx[k]];
    *out = s / static_cast<double>(m.count);
    return true;
}

bool DriftV6Compensator::MatMean(const Eigen::MatrixXd& buf, int n, double t0, double t1,
                                 Eigen::VectorXd* out) const {
    BufMask m;
    FillMask(t0, t1, &m);
    if (!m.any() || n <= 0 || buf.rows() < n) return false;
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(n);
    for (int k = 0; k < m.count; ++k) acc += buf.col(m.idx[k]).head(n);
    *out = acc / static_cast<double>(m.count);
    return true;
}

void DriftV6Compensator::Backdate(double ts_now, double* t0, double* base,
                                  std::vector<std::pair<double, double>>* hist) const {
    // 事件时间原点必须回溯到真实加载沿: 检测统计量要 0.10 s 近窗才敢动, 直接把命中帧当 t0
    // 会让形状里 τ=0 处已有 10%~80% 的增量 ⇒ Â 被系统性低估(规格 §12.2 M1)。
    // 做法 = 命中前 ≤0.6 s 内取窗口最早的 1/4 作前置电平估计, 再找「越过
    // 前置 + 3%×总跳变」的第一帧, 取其前一帧为 t0, base 取 t0 起往前 5 帧中位。
    hist->clear();
    BufMask m;
    FillMask(ts_now - kBackdateS, ts_now, &m);
    if (!m.any()) {
        const int last = static_cast<int>((buf_n_ - 1) % kCap);
        *t0 = ts_now;
        *base = buf_v_[last];
        return;
    }
    // 按时间升序读窗口内的帧(环形缓冲的物理序不等于时间序)
    std::vector<std::pair<double, double>> tv;
    tv.reserve(static_cast<std::size_t>(m.count));
    for (int k = 0; k < m.count; ++k) tv.emplace_back(buf_t_[m.idx[k]], buf_v_[m.idx[k]]);
    std::sort(tv.begin(), tv.end(),
              [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                  return a.first < b.first;
              });
    if (tv.size() < 6) {
        *t0 = tv.front().first;
        *base = tv.front().second;
        return;
    }
    const std::size_t q = std::max<std::size_t>(3, tv.size() / 4);
    std::vector<double> head;
    head.reserve(q);
    for (std::size_t k = 0; k < q; ++k) head.push_back(tv[k].second);
    const double pre = MedianInPlace(head);
    const double jump = tv.back().second - pre;
    if (std::abs(jump) < 1e-12) {
        *t0 = tv.front().first;
        *base = pre;
        return;
    }
    const double tgt = pre + 0.03 * jump;
    std::size_t idx = 0;
    for (std::size_t k = 0; k < tv.size(); ++k) {
        if (jump > 0 ? (tv[k].second >= tgt) : (tv[k].second <= tgt)) {
            idx = (k > 0) ? (k - 1) : 0;
            break;
        }
    }
    const std::size_t b0 = (idx > 5) ? (idx - 5) : 0;
    std::vector<double> tail;
    tail.reserve(idx - b0 + 1);
    for (std::size_t k = b0; k <= idx; ++k) tail.push_back(tv[k].second);
    const double base_val = MedianInPlace(tail);
    const double t0_val = tv[idx].first;
    hist->reserve(tv.size() - idx);
    for (std::size_t k = idx; k < tv.size(); ++k)
        hist->emplace_back(tv[k].first - t0_val, tv[k].second - base_val);
    *t0 = t0_val;
    *base = base_val;
}

// ───────────────────────────── 逆模型 ─────────────────────────────

bool DriftV6Compensator::InvEst(const std::vector<std::pair<double, double>>& hist,
                                double tau, double kappa, double* a_out) const {
    // 电平域最小二乘: Â = Σy·g / Σg², 窗 [τ_ref, min(τ, τ_ref+AWIN)]。
    // **不是**尾巴外推式(Â = y(τ_ref)+Σ(y−y_ref)f/Σf²): 后者在窗口很短时被 1/mean(f)
    // 放大(短窗可达 ×25), 实测把 14.2 的台阶算成 15.7(+10%)(规格 §12.2 M2)。
    if (hist.size() < 4 || tau < kTauRef) return false;
    const double hi = std::min(tau, kTauRef + kAWin);
    double num = 0.0, den = 0.0;
    int cnt = 0;
    for (const auto& h : hist) {
        if (h.first < kTauRef || h.first > hi) continue;
        const double g = ShapeAt(h.first);
        num += h.second * g;
        den += g * g;
        ++cnt;
    }
    if (cnt < 5 || den < 1e-12) return false;
    const double inc = hist.back().second;
    if (inc <= 0.0) return false;
    double a = num / den;
    a = std::max(a, inc);
    *a_out = std::min(a, kappa * inc);
    ++shape_hits_;
    return true;
}

// ───────────────────────────── 谷底判据 ─────────────────────────────

void DriftV6Compensator::UpdateValley(double dt) {
    // 环缓只存 ts_smooth_(平滑总量), 判据不依赖「空载读数≈0」, 只问
    // 「当前电平是否处在近 kValleyWinN 帧的谷底附近, 且这段窗内确实发生过足够大的加载」。
    const int idx = static_cast<int>(valley_n_ % kValleyWinN);
    valley_buf_[idx] = ts_smooth_;
    ++valley_n_;
    if (valley_n_ < kValleyWinN) {
        // 数据不足: 保守 —— 任何消费方都不得触发出口
        valley_now_ = false;
        valley_run_ = 0.0;
        return;
    }
    double w_min = valley_buf_[0];
    double w_max = valley_buf_[0];
    for (int k = 1; k < kValleyWinN; ++k) {
        w_min = std::min(w_min, valley_buf_[k]);
        w_max = std::max(w_max, valley_buf_[k]);
    }
    const double eps_v = 1e-6 * (1.0 + std::abs(w_max));
    valley_now_ = (ts_smooth_ < w_min + params_.valley_frac * (w_max - w_min + eps_v)) &&
                  ((w_max - w_min) >
                   params_.valley_range_frac * std::max(std::abs(w_max), eps_v));
    // dt <= 0(首帧/重复时间戳)时谷底不累加; 但仍处于谷底则保持原值
    valley_run_ = (valley_now_ && dt > 0.0) ? (valley_run_ + dt)
                                            : (valley_now_ ? valley_run_ : 0.0);
}

// ───────────────────────────── 主流程 ─────────────────────────────

void DriftV6Compensator::Process(double timestamp_s, Eigen::VectorXd& values_io) {
    const int n = static_cast<int>(values_io.size());
    if (n <= 0) return;
    if (n_ != n) ResetFor(n);

    const Eigen::VectorXd v = values_io;
    const double total = v.sum();
    // y = 上一帧的显示值(首帧 = 本帧原始值)。y0 必须取「事件前的显示值」,
    // 不能用 (v0 − 当前输出) 反推 —— 后者把加载跳变误算成「扣除」, 实测目标偏高 0.74。
    const Eigen::VectorXd y = first_frame_ ? v : prev_out_;

    double dt = 0.0;
    if (first_frame_) {
        first_frame_ = false;
        last_ts_ = timestamp_s;
        ts_smooth_ = level_ref_ = total;
        min_ts_ = max_ts_ = total;
    } else {
        dt = timestamp_s - last_ts_;
        last_ts_ = timestamp_s;
        dt = (dt > 0.0) ? std::min(dt, 0.1) : 0.0;
        if (dt > 0.0) {
            ts_smooth_ += (dt / kTauTotalS) * (total - ts_smooth_);
            level_ref_ += (dt / kTauLevelS) * (ts_smooth_ - level_ref_);
        }
    }
    min_ts_ = std::min(min_ts_, ts_smooth_);
    max_ts_ = std::max(max_ts_, ts_smooth_);
    max_tot_ = std::max(max_tot_, total);
    const double eps = 1e-6 * (1.0 + std::abs(max_tot_));

    UpdateValley(dt);

    // 本帧的 3 帧中值总量(供下一帧 Push 使用; 检测/回溯统一走这片缓冲)
    if (med_n_ < 3) {
        med_hist_[med_n_++] = total;
    } else {
        med_hist_[0] = med_hist_[1];
        med_hist_[1] = med_hist_[2];
        med_hist_[2] = total;
    }
    {
        std::vector<double> m3(med_hist_, med_hist_ + med_n_);
        prev_med_total_ = MedianInPlace(m3);
    }

    // ── 检测器 D: 短滞后电平差 + σ 门限 ──────────────────────────
    double lv_now = 0.0, lv_ref = 0.0;
    const bool have_now = WinMean(timestamp_s - kDetFastS, timestamp_s, &lv_now);
    const bool have_ref = WinMean(timestamp_s - kDetFastS - kDetGapS - kDetLagS,
                                  timestamp_s - kDetFastS - kDetGapS, &lv_ref);
    const double d = (have_now && have_ref) ? (lv_now - lv_ref) : 0.0;
    const double ref_mag = std::abs(have_ref ? lv_ref : 0.0);

    det_d_[det_n_ % kDcap] = d;
    ++det_n_;
    if (det_n_ > 40) {
        // σ_d 由各录制噪声在线估计(1.4826·MAD); 实测门限只相当于电平的 0.003~0.01%。
        // 缓冲写满后只统计环形缓冲内的有效槽位(否则会把上一轮的陈旧样本混进来)。
        const int h = static_cast<int>(std::min<long>(det_n_, kDcap));
        std::vector<double> tmp(det_d_, det_d_ + h);
        const double med = MedianInPlace(tmp);
        for (std::size_t k = 0; k < tmp.size(); ++k) tmp[k] = std::abs(tmp[k] - med);
        sig_d_ = 1.4826 * MedianInPlace(tmp);
    }
    const double thr_d = std::max({kDetK * sig_d_, kDetRel * ref_mag, kDetAbsFrac * max_tot_});
    // epoch 尾巴内(事件中 / 交接后 kTailGateS 内): 只认 ≥10% 电平的真实变载,
    // 避免把本 epoch 自己的快相尾巴误判成「负载内变载」(v5 用 6 s 抑制窗达到同一目的)
    const double tau_ep = ev_.valid
        ? (timestamp_s - ev_.t0)
        : ((state_ == State::Slow) ? (timestamp_s - ev_end_) : 1e9);
    const double tail_gate =
        (state_ != State::Idle && tau_ep < kTailGateS) ? (kTailGateFrac * ref_mag) : 0.0;
    const bool raw_hit = std::abs(d) > std::max(thr_d, tail_gate);
    // plan-v3.1：检测器一有反应就暂停慢相自适应 —— 挡住"g 在确认期内朝新电平跑掉、
    // 随后被 Reanchor 固化"的那条通路（见 .h 的 kFreezeSlowOnHit 说明）。
    slow_freeze_ = params_.freeze_slow_on_hit && raw_hit;
    hit_run_ = raw_hit ? (hit_run_ + 1) : 0;
    quiet_run_ = raw_hit ? 0 : (quiet_run_ + 1);
    if (quiet_run_ >= kDetPersist) armed_ = true;   // 本次抬升结束后才允许再建事件
    bool hit = (hit_run_ >= kDetPersist) && armed_;

    const double lvl = std::max(level_ref_, eps);
    const bool idle_now = (ts_smooth_ < kIdleFrac * lvl) ||
                          (ts_smooth_ < kUnloadMinRatio * min_ts_ + eps);

    if (hit && d > 0 && state_ == State::Idle) {
        // 空载态: 既要「台阶足够显著」, 也要「已经从上一段卸载里静下来」
        // (否则卸载沿后的回弹会被当成新 onset —— 实测这是多余 epoch 的主要来源)
        if ((timestamp_s - idle_since_) < kIdleSettleS) {
            hit = false;
        } else if (d <= std::max(thr_d, kDetIdleFrac * max_tot_)) {
            hit = false;
        }
    }
    if (hit && timestamp_s < ev_block_until_) hit = false;   // 刚处理完减重/卸载: 静默

    if (hit && d > 0) {
        bool build = false;
        bool c5 = false;
        if (!ev_.valid) {
            if ((timestamp_s - last_ev_ts_) > kRevokeCooldownS) build = true;
        } else if (tau_ep > kRevokeS) {
            build = true;      // C5: epoch 内再来一次真实变载(原型曾写在 ev is None 分支里
            c5 = true;         //     ⇒ 分支不可达, 实录全程偏差 8509 ADC; M6 修正)
        }
        if (build) {
            double t0 = 0.0, base = 0.0;
            std::vector<std::pair<double, double>> hist;
            Backdate(timestamp_s, &t0, &base, &hist);
            Eigen::VectorXd v0, y0;
            const bool have_v0 = MatMean(buf_vx_, n, t0 - 0.30, t0 - 0.05, &v0);
            const bool have_y0 = MatMean(buf_yx_, n, t0 - 0.30, t0 - 0.05, &y0);
            if (have_v0 && have_y0 && hist.size() >= 4) {
                const bool idle_pre = (base < kIdleFrac * lvl) ||
                                      (base < kUnloadMinRatio * min_ts_ + eps);
                Kind kind = Kind::Restep;
                if (c5) {
                    kind = Kind::RestepReload;
                } else if (state_ == State::Idle && idle_pre) {
                    kind = Kind::Onset;      // C1: 空载→负载
                }
                NewEvent(t0, base, v0, y0, kind);
                armed_ = false;
            }
            last_ev_ts_ = timestamp_s;
        }
    } else if (hit && d < 0 && state_ != State::Idle && !ev_.valid) {
        if ((timestamp_s - last_ev_ts_) > kRevokeCooldownS) {
            double t0 = 0.0, base = 0.0;
            std::vector<std::pair<double, double>> hist;
            Backdate(timestamp_s, &t0, &base, &hist);
            Eigen::VectorXd v0, y0;
            const bool have_v0 = MatMean(buf_vx_, n, t0 - 0.30, t0 - 0.05, &v0);
            const bool have_y0 = MatMean(buf_yx_, n, t0 - 0.30, t0 - 0.05, &y0);
            if (have_v0 && have_y0 && hist.size() >= 4)
                NewEvent(t0, base, v0, y0, Kind::Decrease);
            last_ev_ts_ = timestamp_s;
        }
    }

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
            // C4 完全卸载: v5 的 u>3s 收到 0.3s(零负载段被钳 0 时长 0~0.71 s → ≤0.35 s)
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
}

// ───────────────────────────── 事件 ─────────────────────────────

void DriftV6Compensator::NewEvent(double t0, double base, const Eigen::VectorXd& v0,
                                  const Eigen::VectorXd& y0, Kind kind) {
    const State prev = state_;
    ev_ = EventCtx();
    ev_.valid = true;
    ev_.kind = kind;
    ev_.prev_state = prev;
    ev_.t0 = t0;
    ev_.t_det = last_ts_;
    ev_.base = base;
    ev_.v0 = v0;
    ev_.y0 = y0;
    ev_.base_y = y0.sum();
    // 继承当前修正: C5 新建事件时若把 c_applied 从 0 起算, 该帧显示 = 原始,
    // 整段修正被丢掉(实测「过充→回落→再抬升」里夹了一次跳变)。
    ev_.c0 = ev_.base_y - base;
    ev_.c_applied = ev_.c0;
    ev_.tglide = kGlideMaxS;
    ev_.y_prev = prev_out_;
    state_ = State::Event;
    hold_comp_.resize(0);
}

Eigen::VectorXd DriftV6Compensator::ShareVector(const EventCtx& ev,
                                                const Eigen::VectorXd& v) const {
    // A_i 由总量反演的 Â 按各通道增量占比分配(规格 §10.3 第 2 条: 逐通道形状未标定),
    // 因此修正量按同一 share 分摊到各通道。
    Eigen::VectorXd w = (v - ev.v0).cwiseMax(0.0);
    const double sw = w.sum();
    if (sw > 1e-12) return w / sw;
    return Eigen::VectorXd::Zero(v.size());
}

bool DriftV6Compensator::RunEvent(double ts, const Eigen::VectorXd& v, double total,
                                  double dt, double eps, bool idle_now,
                                  Eigen::VectorXd* out) {
    const double tau = ts - ev_.t0;
    const double inc = total - ev_.base;
    double inc_s = inc;
    double wm = 0.0;
    if (WinMean(ts - 0.10, ts, &wm)) inc_s = wm - ev_.base;

    // C6 试探撤销: 命中后 0.4 s 内电平回落到台阶的一半以下 ⇒ 手指调整/磕碰/振动,
    // 放弃本次事件、不建 epoch、不写状态(这是「取消 2.5 s 硬确认」的代价对冲)。
    // I7 迟滞: 该判据必须**连续 kRevokeHoldN 帧**成立才真撤销 —— 「整包到达」时序下
    // inc_s 会跨过抬升沿瞬时读到 0, 单帧即撤销会让这次加载整段不补偿。
    // 未达帧数时本帧不撤销, 且**不让这一帧的瞬时低值参与 inc_max**(否则下帧判据失效)。
    bool revoke_now = false;
    if (tau < kRevokeS && ev_.inc_max > eps && inc_s < 0.5 * ev_.inc_max) {
        ev_.revoke_run += 1;
        revoke_now = (ev_.revoke_run >= kRevokeHoldN);
    } else {
        ev_.revoke_run = 0;
    }
    ev_.inc_max = std::max(ev_.inc_max, inc_s);
    if (revoke_now) {
        ++n_revoke_;
        const State prev = ev_.prev_state;
        const Eigen::VectorXd y_prev = ev_.y_prev;
        ev_ = EventCtx();
        state_ = (prev == State::Idle || prev == State::Slow) ? prev : State::Idle;
        last_ev_ts_ = ts;
        if (prev == State::Idle && y_prev.size() == n_) {
            *out = y_prev;
        } else {
            *out = v;
        }
        return true;
    }

    // 减重的对称撤销: 单帧掉点/手指抖动造成的「假减重」必须在 0.4 s 内被撤掉, 否则会走
    // Reanchor 把 A 挪走(实测 右拇指/数据2 @146.6 s 就是这种假减重: 单帧 −8% 掉点)。
    ev_.dec_max = std::min(ev_.dec_max, inc_s);
    if (ev_.kind == Kind::Decrease && tau < kRevokeS && ev_.dec_max < -eps &&
        inc_s > 0.5 * ev_.dec_max) {
        ++n_revoke_;
        const State prev = ev_.prev_state;
        ev_ = EventCtx();
        state_ = (prev == State::Idle || prev == State::Slow) ? prev : State::Idle;
        last_ev_ts_ = ts;
        *out = v;
        return true;
    }

    // ── F2: 事件期内的卸载出口(不依赖 idle_now) ──
    // 本录制的实测结论: idle_now 在事件态内只真 1 帧(见 design_failure_census.txt),
    // 因此上面那条 C4 出口在本工况结构性不可达; valley_now_ 只依赖近 40 s 的电平形状,
    // 与"空载绝对值"无关, 故在事件期内也能正确认出"已回到谷底"。
    // 默认关闭(params_.legacy_fixes_enabled = false): 14 份数据的回归显示本项收益未证实,
    // 且在「变化负载/切换负载-快相无责」上方向不一致(见设计文档 §3.4)。
    if (params_.legacy_fixes_enabled && tau > params_.event_valley_min_s && valley_now_) {
        ++params_.n_valley_exit;
        ToIdle();
        *out = v;
        return true;
    }

    // C4 事件期内落入空载带
    if (tau > kUnloadFastS && idle_now) {
        ToIdle();
        *out = v;
        return true;
    }

    // 逆模型只需要 [0, 1.0] s 的样本(窗 [τ_ref, τ_ref+AWIN] = [0.2, 0.8]);
    // **不能按「最近 2 s」裁剪**, 否则 τ>2.8 s 后窗口被裁掉 ⇒ 估计失败 ⇒ 永不交接。
    if (tau <= 1.0) ev_.hist.emplace_back(tau, inc);

    // 半程后若前置电平其实很低, restep 被证伪为 onset(κ 与滑行策略随之放宽)
    if (ev_.kind == Kind::Restep && tau >= 0.30 && ev_.base < 0.5 * total) {
        ev_.kind = Kind::Onset;
    }

    if (ev_.kind == Kind::Decrease) {
        // 减重/卸载恢复的形状库(C3)完全未标定 ⇒ 本版不启用逆模型: 沉降窗内保持上一段
        // 扣除(显示连续), 沉降窗结束立即重锚并交给慢相模块继续抑制。
        // 沉降窗从**检测时刻**起算: 若检测本身晚于 t0+SETTLE(回溯到更早的沿),
        // 用 τ 会在建事件当帧立刻重锚, 等于没有沉降窗。
        if ((ts - ev_.t_det) < kDecreaseSettleS) {
            // hold_comp_ = **总扣除**（rank-1 + PCT）⇒ 沉降窗内显示严格随输入平移。
            // 逐通道施加条件用"本通道确实有扣除"，而不是 loaded_ —— 否则 PCT 独占的
            // 通道（eligible 但不在 loaded_ 里）在沉降窗里会完全没有扣除、显示上跳。
            if (hold_comp_.size() != n_) hold_comp_ = DeductionVector();
            Eigen::VectorXd o = v;
            for (int k = 0; k < n_; ++k) {
                if (std::abs(hold_comp_(k)) > 1e-12)
                    o(k) = v(k) - Capped(v(k), hold_comp_(k));
            }
            *out = o;
            return true;
        }
        ev_ = EventCtx();
        state_ = State::Slow;
        ev_end_ = ts;
        Reanchor(ts, v);
        ev_block_until_ = ts + kUnloadBlockS;
        *out = SlowStep(v, dt);
        return true;
    }

    // C1 onset / C2 restep 的 Â 上限(单侧)。onset 档 1.05 会**实际截断**(κ≥1.28 时对 22 个
    // onset 一个都不触发, 见头文件); C2 restep 只允许 1.12 —— 1 s 时输入只走完 64%,
    // 剩下 36% 尚未发生, 不允许无界外推(规格 §8.2)。
    const double kappa = (ev_.kind == Kind::Onset) ? kKappaOnset : kKappaRestep;
    double a_hat = 0.0;
    const bool ok = InvEst(ev_.hist, tau, kappa, &a_hat);
    if (ok && !ev_.stalled) ev_.a_hat = a_hat;   // 已判停滞: 不再让形状估计覆盖

    // ── 停滞检测(F5): 输入停住时不再「预判尾巴」 ──
    // 形状模型假定「载荷会继续爬到 5 s 电平」; 若输入中途停住(载荷分两级施加、中间保压),
    // 该假定被证伪, 继续钉在 Â 就是过充(实测 切换负载 @11.4~13.6 s 过充 +14.6%)。
    // 判据 = 实测尾巴增长比 < 0.5×模型尾巴增长比 持续 0.45 s: 无量纲、与加载方式无关;
    // 斜率门与「模型-实测落差」门都被实测证伪(前者在两族数据上必然误触发, 后者在先验
    // 偏 >6% 的正常记录上误判, 把 T5% 中位从 0.50 s 推到 5.42 s)。
    if (ok && !ev_.stalled && tau > kStallStartS) {
        const double amp = std::max(std::abs(ev_.a_hat), eps);
        if (!ev_.inc_ref_set && tau >= kTauRef && ev_.hist.size() >= 4) {
            const auto& h = ev_.hist;
            for (std::size_t k = 0; k + 1 < h.size(); ++k) {
                if (h[k].first <= kTauRef && h[k + 1].first >= kTauRef) {
                    const double span = h[k + 1].first - h[k].first;
                    const double w = (span > 1e-12) ? (kTauRef - h[k].first) / span : 0.0;
                    ev_.inc_ref = h[k].second + w * (h[k + 1].second - h[k].second);
                    ev_.inc_ref_set = true;
                    break;
                }
            }
        }
        bool stalled_now = false;
        if (ev_.inc_ref_set && ev_.inc_ref > kStallMinFrac * amp) {
            const double gr = ShapeAt(kTauRef);
            const double ratio_mod = (ShapeAt(tau) - gr) / gr;              // 模型应增长
            const double ratio_obs = (inc_s - ev_.inc_ref) / ev_.inc_ref;    // 实测增长
            if (ratio_mod > 0.03 && ratio_obs < kStallTailFrac * ratio_mod) stalled_now = true;
        }
        ev_.stall_t = stalled_now ? (ev_.stall_t + dt) : 0.0;
        if (ev_.stall_t >= kStallHoldS) {
            ev_.stalled = true;
            ev_.a_hat = std::max(inc_s, 0.0);
        }
    }
    if (ev_.stalled) ev_.a_hat = std::max(inc_s, 0.0);   // 停滞期内目标跟随实测电平

    const double target = ev_.base_y + ev_.a_hat;

    if (!ev_.tau_g0_set) {
        if (!ok || tau < kTauRef) {
            *out = v;      // Â 尚不可用(τ < 0.20 s): 直通
            return true;
        }
        ev_.tau_g0_set = true;
        ev_.tau_g0 = tau;
        const double amp = std::max(std::abs(ev_.a_hat), eps);
        ev_.tglide = std::clamp(std::abs(target - total) / (kRateMax * amp),
                                kGlideMinS, kGlideMaxS);
    }
    const double xx = std::clamp((tau - ev_.tau_g0) / ev_.tglide, 0.0, 1.0);
    const double w = xx * xx * (3.0 - 2.0 * xx);   // smoothstep: 单调、C¹ 连续
    // 修正量 = 继承值(1−W) + 新目标·W ⇒ W=0 时等于上一帧的修正, **不会瞬间跳回原始**
    double c_target = ev_.c0 * (1.0 - w) + (target - total) * w;
    // 速率受限: 重锚时「只改 y_target、不改跳变」, 由限速平滑过去
    const double amp = std::max(std::abs(ev_.a_hat), eps);
    const double lim = kRateMax * amp * std::max(dt, 1e-4);
    if (std::abs(c_target - ev_.c_applied) > lim)
        c_target = ev_.c_applied + ((c_target > ev_.c_applied) ? lim : -lim);
    ev_.c_applied = c_target;

    const double tau_ho = std::max(kHoMinS, ev_.tau_g0 + ev_.tglide);
    if (tau >= tau_ho && ok) {
        Handoff(ts, v);
        *out = SlowStep(v, dt);
        return true;
    }
    const Eigen::VectorXd share = ShareVector(ev_, v);
    *out = v + share * c_target;
    return true;
}

// ───────────────────────────── 分配 / 交接 ─────────────────────────────

// rank-1 部分（v5 口径，仅 loaded_ 通道）。**不能直接用它的总和代表"当前扣除"** ——
// 慢相实际施加的是 DeductionVector()（它还含 PCT，且作用域更宽）。
Eigen::VectorXd DriftV6Compensator::Rank1DeductionVector() const {
    Eigen::VectorXd d = Eigen::VectorXd::Zero(n_);
    for (int k = 0; k < n_; ++k) {
        if (loaded_[k] && A_(k) > 1e-9)
            d(k) = std::clamp(gamma_(k) * A_(k) * g_, kCreepLoFrac * A_(k),
                              kCreepHiFrac * A_(k));
    }
    return d;
}

// 慢相**当前实际施加**的逐通道扣除 = rank-1 + PCT。
// ★ 必须是"总扣除"：它被两处当成"上一段扣除"用来保持显示连续 ——
//   ① 减重/卸载沉降窗的 hold_comp_；② Reanchor 的 ded_old（用于 A_new = vs − ded_old）。
//   plan-v3.0 首版漏了 PCT，导致每次卸载时显示按 |pct| 先下跳、再上跳（实测现场
//   +67~+301 ADC 的台阶回跳），且重锚出的 A 被 pct 偏置。
Eigen::VectorXd DriftV6Compensator::DeductionVector() const {
    Eigen::VectorXd d = Rank1DeductionVector();
    if (pct_ded_.size() == n_) {
        for (int k = 0; k < n_; ++k) {
            if (k < static_cast<int>(pct_elig_.size()) && pct_elig_[k] && A_(k) > 1e-9)
                d(k) += pct_ded_(k);
        }
    }
    return d;
}

void DriftV6Compensator::Handoff(double, const Eigen::VectorXd& v) {
    const Eigen::VectorXd share = ShareVector(ev_, v);
    // 当前逐通道扣除 = −share·c_applied(显示 = v + share·c)
    const Eigen::VectorXd ded_old = -(share * ev_.c_applied);
    // ANCHOR_MODE = "pin": A = 钉住值(无蠕变电平), 与 g 锚定自洽 ⇒ 交接后无暂态。
    // (实测 "measured" 模式即 A←交接时刻实测电平: 显示钉在 Â 而 A 被挪到实测电平 ⇒
    //  g_raw≈0 但锚定 g>0 ⇒ g 以 τ=3 s 衰减、显示朝 raw 爬 3~9 s, 就是用户报告的
    //  「先稳住又缓慢抬升」。)
    Eigen::VectorXd A_new = (ev_.y0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);
    if (A_new.maxCoeff() <= 1e-9) A_new = ev_.y0.cwiseMax(0.0);
    A_ = A_new;
    const double amax = A_.maxCoeff();
    for (int k = 0; k < n_; ++k)
        loaded_[k] = (amax > 1e-9 && A_(k) > kLoadedFrac * amax) ? 1 : 0;
    // PCT 生效通道：比 loaded_ 宽（2% 而非 10%）。实测残漂的 96% 来自
    // 「加载时响应小、因此不在 loaded_ 里、却在保压期内持续蠕变」的通道。
    for (int k = 0; k < n_; ++k)
        pct_elig_[k] = (amax > 1e-9 && A_(k) > params_.pct_elig_frac * amax) ? 1 : 0;
    // ★ PCT 必须**清零**：交接后慢相施加的是 (ded1_new ＋ pct)，而 ded1_new 被下面
    //   锚定成"完整复现事件期的扣除 ded_old"（事件期没有 PCT 项）⇒ 若 pct 留用上一
    //   epoch 的残值，显示会在交接当帧跳 |pct|。清零后交接无跳变，PCT 再从 0 自然重积。
    if (pct_ded_.size() == n_) pct_ded_.setZero();

    // g 与当前扣除自洽(保持连续), 不能归零
    std::vector<double> vals;
    for (int k = 0; k < n_; ++k) {
        if (loaded_[k] && A_(k) > 1e-9 && std::abs(gamma_(k)) > 1e-12)
            vals.push_back(ded_old(k) / (gamma_(k) * A_(k)));
    }
    g_ = vals.empty() ? 0.0 : std::clamp(MedianInPlace(vals), kCreepLoFrac, kCreepHiFrac);

    // 交接时刻实测总电平(= v5 口径的幅度参考) → A 慢修正的目标(kTrimRate>0 时才用;
    // 默认关: 开启后显示在保压期内会持续缓慢移动, 用户评价「还不如修复前稳定」)
    trim_target_sum_ = ev_.base_y + (v.sum() - ev_.base);
    ev_ = EventCtx();
    state_ = State::Slow;
    ev_end_ = last_ts_;
}

void DriftV6Compensator::Reanchor(double ts, const Eigen::VectorXd& v) {
    // ── F3: 已回到谷底 ⇒ 不做"保持扣除连续", 直接归零 ──
    // Reanchor 原有语义是"把当前扣除量代代继承"(下方 ① 的理由), 前提是"实测电平里含真实受载";
    // 若已回到谷底, 该前提不成立, 继续继承会把错误扣除固化。
    // 默认关闭(与 F2 共用 params_.legacy_fixes_enabled)。
    if (params_.legacy_fixes_enabled
        && valley_now_ && valley_run_ >= params_.reanchor_valley_s) {
        ++params_.n_reanchor_idle;
        ToIdle();
        return;
    }
    // 减重后重锚。两处必须小心(都是实测踩出来的):
    //  ① **不能取瞬时帧**: 原始总量会出现 −8% 的单帧掉点(实测 右拇指/数据2 @147.31 s),
    //     用它算 A 会让 A 偏小 ≈4%, 随后 g 收敛到被高估的 g_raw, 显示在 5 s 内悄悄下滑
    //     ≈4% —— 就是用户看到的「奇怪下降」; 改为取近 kReanchorSmoothS 的逐通道窗均值;
    //  ② g 必须与当前扣除自洽(保持连续), 不能归零 —— 归零会让「下降段」变成 g 在重新
    //     填坑(实测 右拇指/数据2 @147.47 s 显示从 16.84 跳到 20.06(+19%), 再用 12 s 回落)。
    hold_comp_.resize(0);
    Eigen::VectorXd vs;
    if (!MatMean(buf_vx_, n_, ts - kReanchorSmoothS, ts, &vs)) vs = v;
    // ★ 两件事必须分开（plan-v3.0 首版把 ded_old 当成"总扣除"从而双重计入 PCT）：
    //  ① A_new 用**总扣除**（rank-1 + PCT）反推，才等于"当前显示的无蠕变电平"；
    //  ② g 只用 **rank-1 部分**锚定，因为交接后施加的是 (ded1 + pct)，用总扣除去解 g
    //     会把 pct 再加一遍，显示立刻下跳 |pct|。
    const Eigen::VectorXd ded1_old = Rank1DeductionVector();
    const Eigen::VectorXd ded_tot = DeductionVector();
    Eigen::VectorXd A_new = (vs - ded_tot).cwiseMax(0.0);
    if (A_new.maxCoeff() <= 1e-9) A_new = vs.cwiseMax(0.0);
    A_ = A_new;
    const double amax = A_.maxCoeff();
    for (int k = 0; k < n_; ++k)
        loaded_[k] = (amax > 1e-9 && A_(k) > kLoadedFrac * amax) ? 1 : 0;
    for (int k = 0; k < n_; ++k)
        pct_elig_[k] = (amax > 1e-9 && A_(k) > params_.pct_elig_frac * amax) ? 1 : 0;

    std::vector<double> vals;
    for (int k = 0; k < n_; ++k) {
        if (loaded_[k] && A_(k) > 1e-9 && std::abs(gamma_(k)) > 1e-12)
            vals.push_back(ded1_old(k) / (gamma_(k) * A_(k)));
    }
    g_ = vals.empty() ? 0.0 : std::clamp(MedianInPlace(vals), kCreepLoFrac, kCreepHiFrac);
}

void DriftV6Compensator::ToIdle() {
    ev_ = EventCtx();
    state_ = State::Idle;
    hold_comp_.resize(0);
    std::fill(loaded_.begin(), loaded_.end(), 0);
    std::fill(pct_elig_.begin(), pct_elig_.end(), 0);
    if (pct_ded_.size() > 0) pct_ded_.setZero();
    g_ = 0.0;
    idle_since_ = last_ts_;
    ev_block_until_ = last_ts_ + kUnloadBlockS;
}

void DriftV6Compensator::TrimA(double dt) {
    if (kTrimRate <= 0.0 || trim_target_sum_ < 0.0) return;
    const double cur = A_.sum();
    if (cur <= 1e-9) return;
    const double dev = trim_target_sum_ - cur;
    const double dead = kTrimDeadFrac * std::abs(trim_target_sum_);
    // 死区: |偏差| ≤ dead 视为「已经够准」⇒ 不修正, 显示保持绝对平
    const double dev_eff = (std::abs(dev) <= dead) ? 0.0 : (dev - (dev > 0 ? dead : -dead));
    const double lim = kTrimRate * std::abs(trim_target_sum_) * std::max(dt, 1e-6);
    const double dlt = std::clamp(dev_eff, -lim, lim);
    if (std::abs(dlt) > 1e-12) A_ = A_ * (1.0 + dlt / cur);
}

// ─────────── 慢相(沿用 v5; plan-v2.0 只在 g_ 更新后追加 F5 的有界化) ───────────
// 注意: 本节与 v5 的唯一差别是 F5 —— g_ 先被负向下界 kGNegFloor 夹住, 再在"已回到谷底"时归零。
// g_raw 仍取未限幅的原始读数 v(k), 因此 C 路的输出限幅不改变这里的任何内部量。

Eigen::VectorXd DriftV6Compensator::SlowStep(const Eigen::VectorXd& v, double dt) {
    TrimA(dt);
    std::vector<double> vals;
    for (int k = 0; k < n_; ++k) {
        if (loaded_[k] && A_(k) > 1e-9) vals.push_back((v(k) - A_(k)) / A_(k));
    }
    if (!vals.empty() && !slow_freeze_) {
        // g(t) = 受载通道归一化残差的 median 共识
        // plan-v3.1：`slow_freeze_`（检测器命中）时不更新 —— 阶跃不是蠕变。
        const double g_raw = MedianInPlace(vals);
        g_ += (dt / kTauG) * (g_raw - g_);
    }
    if (g_ < params_.g_neg_floor) {
        g_ = params_.g_neg_floor;
        ++params_.n_g_floor;
    }
    if (valley_now_ && valley_run_ >= params_.g_valley_reset_s && g_ < 0.0) {
        g_ = 0.0;
        ++params_.n_g_valley_reset;
    }
    if (g_ > kGEnable) {
        g2_acc_ += dt * g_ * g_;
        for (int k = 0; k < n_; ++k) {
            if (loaded_[k] && A_(k) > 1e-9)
                g_rel_acc_(k) += dt * g_ * ((v(k) - A_(k)) / A_(k));
        }
        if (g2_acc_ > 1e-8) {
            // 逐通道增益 γ_i: 过原点增量最小二乘, 限幅 [0.3, 2.0]
            for (int k = 0; k < n_; ++k) {
                gamma_(k) = loaded_[k]
                    ? std::clamp(g_rel_acc_(k) / g2_acc_, kGammaMin, kGammaMax)
                    : 1.0;
            }
        }
    }
    Eigen::VectorXd ded = Eigen::VectorXd::Zero(n_);
    for (int k = 0; k < n_; ++k) {
        if (loaded_[k] && A_(k) > 1e-9)
            ded(k) = std::clamp(gamma_(k) * A_(k) * g_,
                                kCreepLoFrac * A_(k), kCreepHiFrac * A_(k));
    }
    // ── plan-v3.0：逐通道蠕变跟踪（PCT）──
    // 目标 = 让「显示 − A_k」→ 0：pct_k += (dt/τ)·((v_k − A_k) − ded_k)。
    // 它是级联在 rank-1 之外的慢环（τ=10 s ≫ g 的 3 s），只做比例模型剩下的那部分：
    // pct_tau_s = 0 时整段不执行 ⇒ 与 plan-v2.0 逐位一致。
    if (params_.pct_tau_s > 0.0 && pct_ded_.size() == n_ && pct_elig_.size() == n_) {
        // 状态更新需要 dt>0（同包重复时间戳不积分）；**输出施加与 dt 无关**。
        // plan-v3.1：检测器命中时不积分（阶跃/卸载不是蠕变）。
        if (dt > 0.0 && !slow_freeze_) {
            for (int k = 0; k < n_; ++k) {
                if (!pct_elig_[k] || A_(k) <= 1e-9) continue;
                const double resid = (v(k) - A_(k)) - ded(k) - pct_ded_(k);
                double d = pct_ded_(k) + (dt / params_.pct_tau_s) * resid;
                // 备选：蠕变单调 ⇒ 只加大不回调（默认关，见 .h 的 kPctMono 说明）
                if (params_.pct_mono && d < pct_ded_(k)) d = pct_ded_(k);
                const double lo = params_.pct_lo_frac * A_(k);
                const double hi = params_.pct_hi_frac * A_(k);
                if (d < lo) d = lo;
                if (d > hi) d = hi;
                pct_ded_(k) = d;
                if (std::abs(d) > 1e-12) ++params_.n_pct_hits;
            }
        }
        for (int k = 0; k < n_; ++k) {
            if (pct_elig_[k] && A_(k) > 1e-9) ded(k) += pct_ded_(k);
        }
    }
    Eigen::VectorXd out = v;
    for (int k = 0; k < n_; ++k) {
        if (A_(k) > 1e-9 && std::abs(ded(k)) > 1e-12)
            out(k) = v(k) - Capped(v(k), ded(k));
    }
    return out;
}

// ───────────────────────────── 协调器 ─────────────────────────────

void DriftV6CompensationCoordinator::SetParams(const DriftV6Compensator::Params& p) {
    params_ = p;
    for (auto& kv : targets_) kv.second.comp.SetParams(p);   // 已存在的条目立即生效
}

void DriftV6CompensationCoordinator::SetEnabled(bool on) {
    if (enabled_ == on) return;
    enabled_ = on;
    ResetAll();   // 开关切换后从全新状态开始
}

void DriftV6CompensationCoordinator::Process(const std::string& key,
                                             const std::string& signature,
                                             double timestamp_s,
                                             Eigen::VectorXd& values_io) {
    if (!enabled_) return;
    const bool is_new = (targets_.find(key) == targets_.end());
    Entry& entry = targets_[key];
    if (is_new || entry.signature != signature) {
        entry.comp.Reset();
        entry.comp.SetParams(params_);   // 重置后重新注入记住的参数
        entry.signature = signature;
    }
    entry.comp.Process(timestamp_s, values_io);
}

void DriftV6CompensationCoordinator::ResetAll() { targets_.clear(); }

}  // namespace drift_v6
