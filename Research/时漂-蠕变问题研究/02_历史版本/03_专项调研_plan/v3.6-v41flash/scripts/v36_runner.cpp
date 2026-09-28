// plan-v3.6 离线诊断助手（不参与主工程、不改任何产品代码）
//
// 基线同 plan-v3.2 的 v32_runner：标准头先 include 完，再 `#define private public` 后 include
// 变体头，逐帧导出补偿器内部状态，并支持运行期参数注入做 A/B。
//
// 用法：v36_runner.exe [--pct f] [--pct-mono 0|1] [--pct-hi f] [--pct-elig f] [--clamp a]
//                      [--legacy 0|1] [--freeze 0|1] [--seed-gain f] [--creep-tau f]
//                      [--creep-hold f] [--creep-min f] [--creep-max f]
//                      [--v36 <名> <值> ...]   ← 本轮新增旋钮（见 kParamTable）
//
// 协议：stdin 第 1 行 <n_channels>，之后每行 <t> <v0..vn-1>；
//       stdout 第 1 行表头，第 2 行 `OK <n> <frames>`，之后逐帧一行，末行 END ...
//       V30_DUMP_CH=1 时追加逐通道末帧快照（CH 行）。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define private public
#include "domain/drift_v6/drift_v6_compensator.h"
#undef private

namespace {

double med_of(std::vector<double> x) {
    if (x.empty()) return 0.0;
    std::nth_element(x.begin(), x.begin() + x.size() / 2, x.end());
    return x[x.size() / 2];
}

const char* kHeader =
    "t sum_in sum_out state ev_valid ev_kind g A_sum n_loaded "
    "gam_med gam_min gam_max r_med r_wmean aggA_loaded num_loaded "
    "ded_unclamped ded_clamped ded_capped ideal_ded comp_total "
    "level_ref ts_smooth min_ts max_ts max_tot idle_now valley_now valley_run "
    "tau tau_g0 tglide A_hat inc_max c_applied stalled clamp_hits shape_hits trim_sum "
    "max_clamp_viol pct_sum n_pct creep_ratio seed_used w_seed ded1_sum ded_n "
    "ev_base ev_base_y ev_y0_sum";

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (!(std::cin >> n) || n <= 0) return 2;
    {
        std::string rest;
        std::getline(std::cin, rest);
    }

    drift_v6::DriftV6Compensator comp;
    {
        drift_v6::DriftV6Compensator::Params pp = comp.params();
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string k = argv[i];
            const double val = std::atof(argv[i + 1]);
            if (k == "--pct") pp.pct_tau_s = val;
            else if (k == "--pct-mono") pp.pct_mono = (val > 0.5);
            else if (k == "--pct-hi") pp.pct_hi_frac = val;
            else if (k == "--pct-lo") pp.pct_lo_frac = val;
            else if (k == "--pct-elig") pp.pct_elig_frac = val;
            else if (k == "--clamp") pp.clamp_alpha = val;
            else if (k == "--legacy") pp.legacy_fixes_enabled = (val > 0.5);
            else if (k == "--freeze") pp.freeze_slow_on_hit = (val > 0.5);
            else if (k == "--seed-gain") pp.creep_seed_gain = val;
            else if (k == "--creep-tau") pp.creep_tau_s = val;
            else if (k == "--creep-hold") pp.creep_hold_tau_s = val;
            else if (k == "--creep-min") pp.creep_ratio_min = val;
            else if (k == "--creep-max") pp.creep_ratio_max = val;
            // ── 本轮新增旋钮（未打补丁时静默忽略，保证 runner 可跨变体复用）──
            else if (k == "--v36-a") pp.v36_a = val;
            else if (k == "--v36-b") pp.v36_b = val;
            else if (k == "--v36-c") pp.v36_c = val;
            else if (k == "--v36-d") pp.v36_d = val;
        }
        comp.SetParams(pp);
    }

    std::ostringstream out;
    out.precision(9);
    Eigen::VectorXd v(n);
    Eigen::VectorXd last_v_ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd last_out_ = Eigen::VectorXd::Zero(n);
    long frames = 0;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        double t = 0.0, x = 0.0;
        if (!(is >> t)) continue;
        int got = 0;
        while (got < n && (is >> x)) v(got++) = x;
        if (got != n) continue;

        const Eigen::VectorXd vin = v;
        const double sum_in = vin.sum();
        comp.Process(t, v);
        const double sum_out = v.sum();
        last_v_ = vin;
        last_out_ = v;
        ++frames;

        const double g = comp.g_;
        const double A_sum = (comp.A_.size() == n) ? comp.A_.sum() : 0.0;

        std::vector<double> gam, r;
        double aggA = 0.0, num = 0.0, ded_un = 0.0, ded_cl = 0.0, ded_cp = 0.0;
        int nload = 0;
        for (int k = 0; k < n; ++k) {
            if (k >= static_cast<int>(comp.loaded_.size())) continue;
            if (!comp.loaded_[k] || comp.A_(k) <= 1e-9) continue;
            ++nload;
            const double rk = (vin(k) - comp.A_(k)) / comp.A_(k);
            const double dk = comp.gamma_(k) * comp.A_(k) * g;
            const double dc = std::clamp(dk, comp.kCreepLoFrac * comp.A_(k),
                                         comp.kCreepHiFrac * comp.A_(k));
            gam.push_back(comp.gamma_(k));
            r.push_back(rk);
            aggA += comp.A_(k);
            num += comp.A_(k) * rk;
            ded_un += dk;
            ded_cl += dc;
            ded_cp += (dc > vin(k)) ? std::max(vin(k), 0.0) : dc;
        }
        const double gam_med = gam.empty() ? 1.0 : med_of(gam);
        double gam_min = 1.0, gam_max = 1.0;
        if (!gam.empty()) {
            gam_min = *std::min_element(gam.begin(), gam.end());
            gam_max = *std::max_element(gam.begin(), gam.end());
        }
        const double r_med = r.empty() ? 0.0 : med_of(r);
        const double r_wmean = (aggA > 1e-12) ? (num / aggA) : 0.0;

        const double eps = 1e-6 * (1.0 + std::abs(comp.max_tot_));
        const double lvl = std::max(comp.level_ref_, eps);
        const int idle_now =
            ((comp.ts_smooth_ < comp.kIdleFrac * lvl) ||
             (comp.ts_smooth_ < comp.kUnloadMinRatio * comp.min_ts_ + eps)) ? 1 : 0;

        double max_viol = 0.0;
        {
            const double al = comp.params().clamp_alpha;
            if (al > 0.0)
                for (int k = 0; k < n; ++k)
                    max_viol = std::max(max_viol, v(k) - (vin(k) + al * std::abs(vin(k))));
        }

        double ded1_sum = 0.0, ded_n = 0.0;
        for (int k = 0; k < n; ++k) {
            if (k >= static_cast<int>(comp.loaded_.size())) continue;
            if (comp.A_(k) <= 1e-9) continue;
            const double dk = comp.gamma_(k) * comp.A_(k) * g;
            const double dc = std::clamp(dk, comp.kCreepLoFrac * comp.A_(k),
                                         comp.kCreepHiFrac * comp.A_(k));
            if (comp.loaded_[k]) ded1_sum += dc;
            ded_n += (comp.pct_elig_[k] ? dc : 0.0);
        }

        out << t << ' ' << sum_in << ' ' << sum_out << ' '
            << static_cast<int>(comp.state_) << ' ' << (comp.ev_.valid ? 1 : 0) << ' '
            << static_cast<int>(comp.ev_.kind) << ' '
            << g << ' ' << A_sum << ' ' << nload << ' '
            << gam_med << ' ' << gam_min << ' ' << gam_max << ' '
            << r_med << ' ' << r_wmean << ' ' << aggA << ' ' << num << ' '
            << ded_un << ' ' << ded_cl << ' ' << ded_cp << ' ' << num << ' '
            << (sum_in - sum_out) << ' '
            << comp.level_ref_ << ' ' << comp.ts_smooth_ << ' ' << comp.min_ts_ << ' '
            << comp.max_ts_ << ' ' << comp.max_tot_ << ' ' << idle_now << ' '
            << (comp.valley_now_ ? 1 : 0) << ' ' << comp.valley_run_ << ' '
            << (comp.ev_.valid ? (t - comp.ev_.t0) : -1.0) << ' '
            << (comp.ev_.valid && comp.ev_.tau_g0_set ? comp.ev_.tau_g0 : -1.0) << ' '
            << (comp.ev_.valid ? comp.ev_.tglide : -1.0) << ' '
            << (comp.ev_.valid ? comp.ev_.a_hat : 0.0) << ' '
            << (comp.ev_.valid ? comp.ev_.inc_max : 0.0) << ' '
            << (comp.ev_.valid ? comp.ev_.c_applied : 0.0) << ' '
            << (comp.ev_.valid && comp.ev_.stalled ? 1 : 0) << ' '
            << comp.clamp_hits() << ' ' << comp.shape_hits() << ' '
            << comp.trim_target_sum_ << ' ' << max_viol << ' '
            << (comp.pct_ded_.size() == n ? comp.pct_ded_.sum() : 0.0) << ' '
            << comp.params().n_pct_hits << ' '
            << comp.creep_ratio_ << ' ' << comp.ev_.seed_used << ' ' << comp.ev_.w_seed << ' '
            << ded1_sum << ' ' << ded_n << ' '
            << (comp.ev_.valid ? comp.ev_.base : 0.0) << ' '
            << (comp.ev_.valid ? comp.ev_.base_y : 0.0) << ' '
            << (comp.ev_.valid ? comp.ev_.y0.sum() : 0.0) << '\n';
    }
    const auto& p = comp.params();
    std::cout << kHeader << '\n';
    std::cout << "OK " << n << ' ' << frames << '\n';
    std::cout << out.str();
    std::cout << "END " << comp.clamp_hits() << ' ' << comp.shape_hits() << ' '
              << p.n_valley_exit << ' ' << p.n_reanchor_idle << ' ' << p.n_g_floor << ' '
              << p.n_g_valley_reset << ' ' << (comp.in_event() ? 1 : 0) << ' '
              << (comp.in_slow() ? 1 : 0) << ' ' << comp.g_ << ' '
              << p.n_creep_seed << '\n';
    const char* dump = std::getenv("V30_DUMP_CH");
    if (dump != nullptr && std::string(dump) == "1") {
        std::cout << "CH k loaded elig A gamma v_last out_last pct\n";
        for (int k = 0; k < n; ++k) {
            std::cout << "CH " << k << ' '
                      << ((k < static_cast<int>(comp.loaded_.size()) && comp.loaded_[k]) ? 1 : 0) << ' '
                      << ((k < static_cast<int>(comp.pct_elig_.size()) && comp.pct_elig_[k]) ? 1 : 0) << ' '
                      << comp.A_(k) << ' ' << comp.gamma_(k) << ' ' << last_v_(k) << ' '
                      << last_out_(k) << ' ' << comp.pct_ded_(k) << '\n';
        }
    }
    return 0;
}
