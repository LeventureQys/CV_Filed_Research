// plan-v3.0 离线诊断助手（**不参与主工程、不改任何产品代码**）
//
// 与 plan/v2.0/scripts/batch_runner.cpp 的差别：本助手把补偿器的**内部状态**逐帧导出，
// 用于回答「稳定工况下算法输出为什么还在缓慢漂」，并支持运行期参数注入做 A/B。
//
// 取内部状态的手法：标准头先 include 完，再 `#define private public` 后 include 产品头
// （仅离线诊断用；产品代码一行未动）。
//
// 用法：v30_runner.exe [--pct 0|10] [--pct-mono 0|1] [--pct-hi f] [--pct-elig f]
//                      [--clamp a] [--legacy 0|1]
//   --pct 0  = 回到 plan-v2.0 行为（PCT 关闭）
//
// 协议（stdin/stdout 文本流）：
//   stdin 第 1 行:  <n_channels>
//   之后每行:       <t> <v0> ... <v_{n-1}>
//   stdout 第 1 行: 表头
//   第 2 行:        OK <n> <frames>
//   之后每帧一行（字段见 kHeader）
//   末行:           END <clamp_hits> <shape_hits> <n_valley_exit> <n_reanchor_idle>
//                        <n_g_floor> <n_g_valley_reset> <in_event> <in_slow> <g>
//   V30_DUMP_CH=1 时追加逐通道末帧快照（CH 行）。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
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
    "max_clamp_viol pct_sum n_pct";

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

        const Eigen::VectorXd vin = v;          // 补偿前的逐通道输入
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
            << comp.params().n_pct_hits << '\n';
    }
    const auto& p = comp.params();
    std::cout << kHeader << '\n';
    std::cout << "OK " << n << ' ' << frames << '\n';
    std::cout << out.str();
    std::cout << "END " << comp.clamp_hits() << ' ' << comp.shape_hits() << ' '
              << p.n_valley_exit << ' ' << p.n_reanchor_idle << ' ' << p.n_g_floor << ' '
              << p.n_g_valley_reset << ' ' << (comp.in_event() ? 1 : 0) << ' '
              << (comp.in_slow() ? 1 : 0) << ' ' << comp.g_ << '\n';
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
