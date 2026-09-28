// plan-v3.4 观测器离线复算助手（链接产品源码 creep_observer.cpp，仅诊断用）。
// 协议：stdin 首行 <n>；之后每行 <t> <v0..v_{n-1}>；stdout 逐行输出总量与显示总量。
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "domain/drift_v6/creep_observer.h"

int main() {
    int n = 0;
    if (!(std::cin >> n) || n <= 0) return 2;
    {
        std::string rest;
        std::getline(std::cin, rest);
    }
    drift_v6::CreepObserverCompensator comp;
    Eigen::VectorXd v(n);
    std::string line;
    std::ostringstream out;
    out.precision(9);
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        double t = 0.0, x = 0.0;
        if (!(is >> t)) continue;
        int got = 0;
        while (got < n && (is >> x)) v(got++) = x;
        if (got != n) continue;
        const double sin = v.sum();
        comp.Process(t, v);
        out << t << ' ' << sin << ' ' << v.sum() << '\n';
    }
    std::cout << out.str();
    return 0;
}
