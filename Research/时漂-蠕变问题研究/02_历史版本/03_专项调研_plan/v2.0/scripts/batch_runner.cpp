// plan-v2.0 离线跑批助手（不参与主工程；只编 src/domain/drift_v6/drift_v6_compensator.cpp）
//
// 用途：把一条实机录制的**算法输入**逐帧喂进**真实 C++ 补偿器**，导出逐帧显示值与内部诊断，
//       供 Python 侧出图与统计。这样跑的就是交付的算法本体，而不是 Python 原型。
//
// 协议（都走 stdin/stdout 文本流，便于 Python 用 subprocess 驱动）：
//   stdin 第 1 行:  <n_channels> <margin>          单空格分隔；margin = 求和容差（通常 100）
//   stdin 之后每行: <t> <v0> <v1> ... <v_{n-1}>     t 用 elapsed（秒）
//   stdout 第 1 行: OK <n_channels> <frame_count>   逐帧输出行随后
//   stdout 之后每行: <t> <sum_in> <sum_out> <out0> <out1> ... <out_{n-1}>
//   末行:           COUNTS <shape_hits> <n_valley_exit> <n_reanchor_idle> <n_g_floor>
//                           <n_g_valley_reset> <n_clamp> <in_event> <in_slow> <g> <clamp_alpha>
//
// 说明：
//   * 每帧只调一次 Process()，且写入 values_io 的是**输入**通道值（与主程序 data_handler 的口径一致：
//     补偿器就地改写数据）。
//   * 若某帧的 |Σout − Σv_out| > margin，输出行末追加 " WARN_SUM"（自动跳过一帧重跑，见下）。
//   * 通道数变化时会自动 ResetFor（补偿器内建行为），本助手不干预。

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "domain/drift_v6/drift_v6_compensator.h"

int main(int argc, char** argv) {
    int n = 0;
    double margin = 100.0;
    if (!(std::cin >> n >> margin) || n <= 0) {
        std::cerr << "bad header\n";
        return 2;
    }
    std::string rest;
    std::getline(std::cin, rest);  // 吃掉表头行剩余

    drift_v6::DriftV6Compensator comp;
    if (argc > 1) {                       // 可选：注入参数（用于 A/B）
        drift_v6::DriftV6Compensator::Params p;
        const std::string key = argv[1];
        if (key == "noclamp") p.clamp_alpha = 0.0;
        else if (key == "legacy") p.legacy_fixes_enabled = true;
        comp.SetParams(p);
    }

    Eigen::VectorXd v(n);
    std::vector<double> buf(n);
    long frames = 0;
    std::ostringstream out;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        double t = 0.0, x = 0.0;
        if (!(is >> t)) continue;
        int got = 0;
        while (got < n && (is >> x)) buf[got++] = x;
        if (got != n) continue;
        for (int k = 0; k < n; ++k) v(k) = buf[k];
        const double sum_in = v.sum();
        comp.Process(t, v);
        const double sum_out = v.sum();
        out << t << ' ' << sum_in << ' ' << sum_out;
        for (int k = 0; k < n; ++k) out << ' ' << v(k);
        if (std::abs(sum_out - sum_in) > margin) out << " WARN_SUM";
        out << '\n';
        ++frames;
    }

    std::cout << "OK " << n << ' ' << frames << '\n';
    std::cout << out.str();
    const auto& p = comp.params();
    std::cout << "COUNTS " << comp.shape_hits() << ' ' << p.n_valley_exit << ' '
              << p.n_reanchor_idle << ' ' << p.n_g_floor << ' ' << p.n_g_valley_reset
              << ' ' << comp.clamp_hits() << ' ' << (comp.in_event() ? 1 : 0) << ' '
              << (comp.in_slow() ? 1 : 0) << ' ' << comp.slow_residual_g() << ' '
              << p.clamp_alpha << '\n';
    return 0;
}
