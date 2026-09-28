// v3.4 观测器离线扫参助手（链接产品源码 src/domain/drift_v6/creep_observer.cpp，仅分析用）。
//
// 用法：
//   v34_sweep.exe <session.csv> <params.txt> <Esum> <t0> [--time uniform|raw] [--dump DIR]
//
// session.csv : 上位机会话 CSV（##Session 头 + ##Data 块；前三列 timestamp/elapsed/frame_index）
// params.txt  : 每行一个参数集，格式  <名称>|key=value,key=value,...   （只写要覆盖的键）
// Esum        : 该会话的参考弹性电平（ΣE，由 Python 侧蠕变拟合给出），用于算残差指标
// t0          : 加载时刻（s）；保压窗口从其 +5 s 起算，平稳窗口从 +30 s 起算
// --time      : uniform（默认，按 frame_index 重建均匀时间轴，fps = (n-1)/elapsed 跨度）
//               raw（直接用 CSV 的 elapsed 列，即上位机实际喂给算法的批量到达时刻）
// --zero      : 「调零口径」——逐通道减去首帧值（等价于上位机「装好负载 → 调零 → 开算法」，
//               使算法内部 zero_ = 0，全局总值旁路以 0 为基线）
// --dump DIR  : 额外把 (t, 输入总值, 显示总值) 写成 <DIR>/<名称>.bin（int32 n, int32 3, float64 数据）
//
// stdout：CSV 行；out 与 in 的单位 = CSV 通道值之和（本数据为 ADC）

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "domain/drift_v6/creep_observer.h"

using drift_v6::CreepObserverCompensator;
using Params = CreepObserverCompensator::Params;

namespace {

struct Session {
    std::vector<double> t_raw, t_uni, v;   // v 行主序 n x m
    int n = 0, m = 0;
};

bool ReadSession(const std::string& path, Session& s) {
    // 二进制订阅格式（Python 侧预转换）：int32 n, int32 m, 然后 n 行 (elapsed + m 通道) float64
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".bin") == 0) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        std::int32_t n = 0, m = 0;
        if (std::fread(&n, 4, 1, f) != 1 || std::fread(&m, 4, 1, f) != 1) { std::fclose(f); return false; }
        if (n <= 1 || m <= 0) { std::fclose(f); return false; }
        std::vector<double> buf(static_cast<std::size_t>(n) * (m + 1));
        const std::size_t got = std::fread(buf.data(), 8, buf.size(), f);
        std::fclose(f);
        if (got != buf.size()) return false;
        s.n = n;
        s.m = m;
        s.t_raw.resize(n);
        s.v.resize(static_cast<std::size_t>(n) * m);
        for (int i = 0; i < n; ++i) {
            s.t_raw[i] = buf[static_cast<std::size_t>(i) * (m + 1)];
            for (int k = 0; k < m; ++k)
                s.v[static_cast<std::size_t>(i) * m + k] = buf[static_cast<std::size_t>(i) * (m + 1) + 1 + k];
        }
        const double span = s.t_raw.back() - s.t_raw.front();
        const double fps = (span > 0.0) ? (n - 1) / span : 100.0;
        s.t_uni.resize(n);
        for (int i = 0; i < n; ++i) s.t_uni[i] = i / fps;
        return true;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string line;
    bool in_data = false, seen_names = false;
    std::vector<double> row;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (!in_data) {
            if (line.rfind("##Data", 0) == 0) in_data = true;
            continue;
        }
        if (!seen_names) { seen_names = true; continue; }
        row.clear();
        const char* p = line.c_str();
        while (*p) {
            char* end = nullptr;
            const double x = std::strtod(p, &end);
            if (end == p) { row.push_back(std::nan("")); ++p; }
            else { row.push_back(x); p = end; }
            while (*p == ',' || *p == ' ' || *p == '\t') ++p;
        }
        if (row.size() < 4) continue;
        s.t_raw.push_back(row[1]);
        s.v.insert(s.v.end(), row.begin() + 3, row.end());
        ++s.n;
    }
    if (s.n <= 1) return false;
    s.m = static_cast<int>(s.v.size() / s.n);
    const double span = s.t_raw.back() - s.t_raw.front();
    const double fps = (span > 0.0) ? (s.n - 1) / span : 100.0;
    s.t_uni.resize(s.n);
    for (int i = 0; i < s.n; ++i) s.t_uni[i] = i / fps;
    return true;
}

bool Apply(Params& p, const std::string& k, double v) {
    if (k == "r_fast") p.r_fast = v;
    else if (k == "tau_c_fast_s") p.tau_c_fast_s = v;
    else if (k == "tau_r_fast_s") p.tau_r_fast_s = v;
    else if (k == "r_slow_max") p.r_slow_max = v;
    else if (k == "tau_r_slow_s") p.tau_r_slow_s = v;
    else if (k == "tau_slope_s") p.tau_slope_s = v;
    else if (k == "slope_gate_frac") p.slope_gate_frac = v;
    else if (k == "slope_cap_frac") p.slope_cap_frac = v;
    else if (k == "tau_zero_s") p.tau_zero_s = v;
    else if (k == "idle_frac") p.idle_frac = v;
    else if (k == "y_max_tau_s") p.y_max_tau_s = v;
    else if (k == "edge_slope_thres") p.edge_slope_thres = v;
    else if (k == "edge_refract_s") p.edge_refract_s = v;
    else if (k == "edge_boost_s") p.edge_boost_s = v;
    else if (k == "tau_c_fast_boost_s") p.tau_c_fast_boost_s = v;
    else if (k == "soft_unfreeze_s") p.soft_unfreeze_s = v;
    else if (k == "hold_eps") p.hold_eps = v;
    else if (k == "hold_tau_s") p.hold_tau_s = v;
    else if (k == "slow_confirm_s") p.slow_confirm_s = v;
    else if (k == "y_floor_tau_s") p.y_floor_tau_s = v;
    else if (k == "tau_r_slow_idle_s") p.tau_r_slow_idle_s = v;
    else if (k == "bypass_release_frac") p.bypass_release_frac = v;
    else if (k == "bypass_engage_frac") p.bypass_engage_frac = v;
    else if (k == "bypass_noise_sigma") p.bypass_noise_sigma = v;
    else if (k == "bypass_base_tau_s") p.bypass_base_tau_s = v;
    else if (k == "ramp_slope_min") p.ramp_slope_min = v;
    else if (k == "ramp_full_s") p.ramp_full_s = v;
    else if (k == "track_base_frac") p.track_base_frac = v;
    else if (k == "hold_lock_tau_s") p.hold_lock_tau_s = v;
    else if (k == "hold_lock_freeze_s") p.hold_lock_freeze_s = v;
    else return false;
    return true;
}

struct Set { std::string name; Params p; };

bool ReadSets(const std::string& path, std::vector<Set>& out) {
    std::ifstream in(path);
    if (!in) return false;
    const Params base{};
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::size_t bar = line.find('|');
        Set st;
        st.p = base;
        st.name = (bar == std::string::npos) ? line : line.substr(0, bar);
        while (!st.name.empty() && (st.name.back() == ' ' || st.name.back() == '\t')) st.name.pop_back();
        const std::string body = (bar == std::string::npos) ? std::string() : line.substr(bar + 1);
        std::size_t pos = 0;
        while (pos < body.size()) {
            std::size_t comma = body.find(',', pos);
            if (comma == std::string::npos) comma = body.size();
            const std::string tok = body.substr(pos, comma - pos);
            pos = comma + 1;
            if (tok.empty()) continue;
            const std::size_t eq = tok.find('=');
            if (eq == std::string::npos) continue;
            if (!Apply(st.p, tok.substr(0, eq), std::atof(tok.c_str() + eq + 1)))
                std::fprintf(stderr, "[warn] unknown key: %s\n", tok.c_str());
        }
        out.push_back(std::move(st));
    }
    return !out.empty();
}

double WindowMin(const std::vector<double>& out, const std::vector<double>& T,
                 double from, double to, double ref) {
    double m = 1e300;
    for (std::size_t i = 0; i < out.size(); ++i)
        if (T[i] >= from && T[i] <= to) m = std::min(m, out[i] - ref);
    return (m > 1e299) ? 0.0 : m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: v34_sweep <session.csv> <params.txt> <Esum> <t0> "
                             "[--time uniform|raw] [--dump DIR]\n");
        return 2;
    }
    const std::string csv = argv[1], pfile = argv[2];
    const double esum = std::atof(argv[3]);
    const double t0 = std::atof(argv[4]);
    bool use_raw = false, do_zero = false;
    std::string dump_dir;
    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            use_raw = (std::strcmp(argv[i + 1], "raw") == 0);
            ++i;
        } else if (std::strcmp(argv[i], "--zero") == 0) {
            do_zero = true;
        } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        }
    }

    Session s;
    if (!ReadSession(csv, s)) { std::fprintf(stderr, "read session failed\n"); return 3; }
    if (do_zero) {
        for (int k = 0; k < s.m; ++k) {
            const double z = s.v[k];
            for (int i = 0; i < s.n; ++i) s.v[static_cast<std::size_t>(i) * s.m + k] -= z;
        }
    }
    const std::vector<double>& T = use_raw ? s.t_raw : s.t_uni;
    std::vector<Set> sets;
    if (!ReadSets(pfile, sets)) { std::fprintf(stderr, "read params failed\n"); return 3; }

    std::printf("name,n,dur,in_end,out_end,ded_end,err_end,err_min,err_max,err_min_all,"
                "hold_mean_err,hold_std,tail30,disp_min,disp_min_t,disp_max,disp_max_t,"
                "notch,notch_t,sm_max,sm_min,sm_end\n");
    std::fprintf(stderr, "cols: %s\n", "v3");

    Eigen::VectorXd v(s.m);
    std::vector<double> tin(s.n, 0.0), tout(s.n, 0.0);

    for (const Set& st : sets) {
        CreepObserverCompensator comp;
        comp.SetParams(st.p);
        for (int i = 0; i < s.n; ++i) {
            const double* row = &s.v[static_cast<std::size_t>(i) * s.m];
            double sum = 0.0;
            for (int k = 0; k < s.m; ++k) { v(k) = row[k]; sum += row[k]; }
            comp.Process(T[i], v);
            double osum = 0.0;
            for (int k = 0; k < s.m; ++k) osum += v(k);
            tin[i] = sum;
            tout[i] = osum;
        }
        const double in_end = tin[s.n - 1], out_end = tout[s.n - 1];
        const double err_end = out_end - esum;
        const double err_min = WindowMin(tout, T, t0 + 5.0, 1e18, esum);
        const double err_min_all = WindowMin(tout, T, t0, 1e18, esum);
        double err_max = -1e300;
        for (int i = 0; i < s.n; ++i)
            if (T[i] >= t0 + 5.0) err_max = std::max(err_max, tout[i] - esum);
        if (err_max < -1e299) err_max = err_end;
        int i30 = s.n - 1;
        while (i30 > 0 && T[i30] >= T[s.n - 1] - 30.0) --i30;
        ++i30;
        double tail = 0.0;
        const int cnt = s.n - i30;
        if (cnt > 5) {
            double mx = 0, my = 0;
            for (int i = i30; i < s.n; ++i) { mx += T[i]; my += tout[i]; }
            mx /= cnt; my /= cnt;
            double sxy = 0, sxx = 0;
            for (int i = i30; i < s.n; ++i) {
                sxy += (T[i] - mx) * (tout[i] - my);
                sxx += (T[i] - mx) * (T[i] - mx);
            }
            tail = (sxx > 0) ? sxy / sxx : 0.0;
        }
        double hm = 0, hs = 0;
        int hn = 0;
        for (int i = 0; i < s.n; ++i) if (T[i] >= t0 + 30.0) { hm += tout[i]; ++hn; }
        if (hn > 0) {
            hm /= hn;
            for (int i = 0; i < s.n; ++i)
                if (T[i] >= t0 + 30.0) hs += (tout[i] - hm) * (tout[i] - hm);
            hs = std::sqrt(hs / hn);
        }
        // 显示绝对量程（跳过加载瞬间 0.5 s）：用于「过减深度 / 欠减高度」的绝对 ADC 判据
        const double w0 = t0 + 0.5;
        double dmin = 1e300, dmax = -1e300, dmin_t = 0.0, dmax_t = 0.0;
        for (int i = 0; i < s.n; ++i) {
            if (T[i] < w0) continue;
            if (tout[i] < dmin) { dmin = tout[i]; dmin_t = T[i]; }
            if (tout[i] > dmax) { dmax = tout[i]; dmax_t = T[i]; }
        }
        if (dmin > 1e299) { dmin = out_end; dmax = out_end; }
        // ── 「过减」= 显示被后续过度补偿拉回去的深度 ──
        // 先把显示做 0.2 s 中值滤波去毛刺（原始数据有单帧尖峰），
        // 再算 相对历史最高点的最大回落 notch（从 t0+3 s 起算，避开加载瞬间的上升）。
        std::vector<double> sm(s.n, 0.0);
        {
            const int k = 20;                    // ±20 帧 ≈ ±0.2 s
            std::vector<double> buf;
            buf.reserve(2 * k + 1);
            for (int i = 0; i < s.n; ++i) {
                const int a = std::max(0, i - k), b = std::min(s.n - 1, i + k);
                buf.assign(tout.begin() + a, tout.begin() + b + 1);
                const std::size_t mid = buf.size() / 2;
                std::nth_element(buf.begin(), buf.begin() + mid, buf.end());
                sm[i] = buf[mid];
            }
        }
        const double n0 = t0 + 3.0;
        double notch = 0.0, notch_t = 0.0, sm_max = -1e300, sm_min = 1e300;
        double runmax = -1e300;
        for (int i = 0; i < s.n; ++i) {
            if (T[i] < n0) continue;
            runmax = std::max(runmax, sm[i]);
            if (runmax - sm[i] > notch) { notch = runmax - sm[i]; notch_t = T[i]; }
            sm_max = std::max(sm_max, sm[i]);
            sm_min = std::min(sm_min, sm[i]);
        }
        if (sm_max < -1e299) { sm_max = sm_min = out_end; }
        const double sm_end = s.n > 0 ? sm[s.n - 1] : out_end;
        std::printf("%s,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,"
                    "%.4f,%.3f,%.4f,%.3f,%.4f,%.3f,%.4f,%.4f,%.4f\n",
                    st.name.c_str(), s.n, T[s.n - 1] - T[0], in_end, out_end,
                    in_end - out_end, err_end, err_min, err_max, err_min_all, hm - esum, hs, tail,
                    dmin, dmin_t, dmax, dmax_t,
                    notch, notch_t, sm_max, sm_min, sm_end);

        if (!dump_dir.empty()) {
            const std::string path = dump_dir + "/" + st.name + ".bin";
            if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                const std::int32_t n = s.n, cols = 3;
                std::fwrite(&n, 4, 1, f);
                std::fwrite(&cols, 4, 1, f);
                for (int i = 0; i < s.n; ++i) {
                    const double vals[3] = {T[i], tin[i], tout[i]};
                    std::fwrite(vals, 8, 3, f);
                }
                std::fclose(f);
            }
        }
    }
    return 0;
}
