// plan-v2.0 离线验证助手 v2（**实验版**：不参与主工程，不改任何产品代码）
//
// 臂（mode）：
//   raw      逐帧原样喂交付算法                                   —— 基线
//   packavg  同一 elapsed 的 1~4 帧**去涨落**后喂（候选②）         —— 始终生效
//   hybrid   仅在**检出抖动**时才对包内去涨落，其余逐帧原样（候选②′）★
// 包内统计量（agg）：mean | median
//
// 「包内去涨落」（逐通道）：
//   g_c = 该包内通道 c 的 agg（mean 或 median）
//   m_c = 该包内通道 c 的逐帧均值
//   v[k,c] += (g_c − m_c)          ← 只平移电平，**保留包内各帧的相对形状**
//   说明：等价于"把包内均值搬到 agg 上"，比"直接替换成均值"少一层失真；
//         agg=mean 时即"完全抹平包内波动"，agg=median 时保留一半。
//
// 抖动判据（hybrid）：**原始帧的包内极差 >= kIntraThresh**——
//   它是可直接观测的量，也正是「该不该去涨落」的定义。
//   （反例留痕：曾用「去涨落后」的总量做门限，3988 组只触发 4 组 —— 先抹平再判断，等于永不自证。）
//
// 用法：runner_hy <mode> <agg> <n_ch> <margin>
// 输出：OK 行 + 逐帧 "<t> <sum_in> <sum_out>" + COUNTS ... agg_groups groups jitter_frames

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "domain/drift_v6/drift_v6_compensator.h"

namespace {

constexpr double kIntraLow  = 150.0;    // 包内极差**下限**：> 它才认为"包内存在抖动"
constexpr double kIntraHigh = 2500.0;   // 包内极差**上限**：> 它说明包内跨越了**真实加载/卸载沿**
                                        //   （实测 t=111.061 一个包内含 17307 与 3495，去涨落会造出 10400
                                        //    这个**从未存在过**的电平，被算法当成一次真实加载 ⇒ 偏移 +6894 ADC）
                                        //   ⇒ 这种包必须原样保留，绝不能去涨落。

double median_of(std::vector<double> x) {
    if (x.empty()) return 0.0;
    std::nth_element(x.begin(), x.begin() + x.size() / 2, x.end());
    return x[x.size() / 2];
}


struct JitterDetUnused {};   // 保留占位（门限实现已改为"包内极差"，见文件头说明）

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "raw";
    const std::string aggmode = (argc > 2) ? argv[2] : "median";
    int n = 0;
    double margin = 100.0;
    if (!(std::cin >> n >> margin) || n <= 0) return 2;
    {
        std::string rest;
        std::getline(std::cin, rest);
    }

    std::vector<double> t;
    std::vector<std::vector<double>> ch;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        double tv = 0.0, x = 0.0;
        if (!(is >> tv)) continue;
        std::vector<double> row(n);
        int got = 0;
        while (got < n && (is >> x)) row[got++] = x;
        if (got != n) continue;
        t.push_back(tv);
        ch.push_back(std::move(row));
    }
    const std::size_t N = t.size();
    if (N == 0) {
        std::cout << "OK " << n << " 0\n";
        std::cout << "COUNTS 0 0 0 0 0 0 0 0 0 0 0 0\n";
        return 0;
    }

    // ── 预计算：每个包的 [i, j) 以及逐通道 (g_c − m_c) ──
    std::vector<std::size_t> gid(N);
    std::vector<std::vector<double>> delta;   // 每组的逐通道平移量
    std::vector<std::size_t> gstart, gend;
    {
        std::size_t i = 0;
        while (i < N) {
            std::size_t j = i + 1;
            while (j < N && t[j] == t[i]) ++j;
            const std::size_t g = gstart.size();
            gstart.push_back(i);
            gend.push_back(j);
            for (std::size_t k = i; k < j; ++k) gid[k] = g;
            std::vector<double> d(n, 0.0);
            std::vector<double> col(j - i);
            for (int c = 0; c < n; ++c) {
                double m = 0.0;
                for (std::size_t k = i; k < j; ++k) {
                    m += ch[k][c];
                    col[k - i] = ch[k][c];
                }
                m /= static_cast<double>(j - i);
                const double g_agg = (aggmode == "mean") ? m : median_of(col);
                d[c] = g_agg - m;
            }
            delta.push_back(std::move(d));
            i = j;
        }
    }

    // ── 逐组判定"是否抖动" ──
    // ★ 关键设计：门限必须用**原始帧**的包内极差（intra-packet range），
    //   而不是"包内去涨落后"的信号 —— 后者已经把要抑制的东西抹掉了，
    //   用它做门限等于"先证明没问题、再决定要不要修"，门限永远不触发
    //   （实测：用去涨落后的总量做门限，3988 组里只触发 4 组）。
    //   包内极差是**可直接观测**的量，也正是"该不该去涨落"的定义。
    std::vector<char> use(gstart.size(), 0);
    for (std::size_t g = 0; g < gstart.size(); ++g) {
        const std::size_t i = gstart[g], j = gend[g];
        double pk = 0.0;
        for (std::size_t k = i; k < j; ++k) {
            double s = 0.0;
            for (int c = 0; c < n; ++c) s += ch[k][c];
            pk = std::max(pk, s);
        }
        double pmin = 1e300;
        for (std::size_t k = i; k < j; ++k) {
            double s = 0.0;
            for (int c = 0; c < n; ++c) s += ch[k][c];
            pmin = std::min(pmin, s);
        }
        const double intra = pk - pmin;
        if (mode == "packavg") use[g] = 1;
        else if (mode == "hybrid") use[g] = (intra >= kIntraLow && intra <= kIntraHigh) ? 1 : 0;
    }

    // ── 喂算法 ──
    drift_v6::DriftV6Compensator comp;
    std::ostringstream out;
    Eigen::VectorXd v(n);
    long jitter_frames = 0, agg_groups = 0;
    static long cli_agg_groups = 0;
    for (std::size_t g = 0; g < gstart.size(); ++g) {
        const std::size_t i = gstart[g], j = gend[g];
        if (use[g]) ++agg_groups;
        for (std::size_t k = i; k < j; ++k) {
            for (int c = 0; c < n; ++c) v(c) = ch[k][c];
            const double sum_in = v.sum();
            if (use[g]) {
                for (int c = 0; c < n; ++c) v(c) += delta[g][c];
                ++jitter_frames;
            }
            comp.Process(t[k], v);
            out << t[k] << ' ' << sum_in << ' ' << v.sum() << '\n';
        }
    }

    std::cout << "OK " << n << ' ' << N << '\n';
    std::cout << out.str();
    const auto& p = comp.params();
    std::cout << "COUNTS " << comp.shape_hits() << ' ' << p.n_valley_exit << ' '
              << p.n_reanchor_idle << ' ' << p.n_g_floor << ' ' << p.n_g_valley_reset
              << ' ' << comp.clamp_hits() << ' ' << (comp.in_event() ? 1 : 0) << ' '
              << (comp.in_slow() ? 1 : 0) << ' ' << comp.slow_residual_g() << ' '
              << p.clamp_alpha << ' ' << agg_groups << ' ' << gstart.size() << ' '
              << jitter_frames << '\n';
    return 0;
}
