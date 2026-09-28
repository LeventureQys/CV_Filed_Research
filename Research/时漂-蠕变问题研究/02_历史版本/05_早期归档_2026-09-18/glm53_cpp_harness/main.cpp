#include "domain/drift/drift_compensator.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: drift_harness input.csv out.csv\n"; return 2; }
    std::ifstream in(argv[1]);
    if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 3; }
    std::string line;
    for (int i = 0; i < 24 && std::getline(in, line); ++i) {}
    std::getline(in, line);
    std::stringstream ss(line);
    std::vector<std::string> cols;
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    std::vector<int> idx;
    for (int i = 0; i < (int)cols.size(); ++i)
        if (cols[i].rfind("ch", 0) == 0) idx.push_back(i);
    int ti = -1;
    for (int i = 0; i < (int)cols.size(); ++i) if (cols[i] == "timestamp") { ti = i; break; }
    if (ti < 0 || idx.empty()) { std::cerr << "header missing\n"; return 4; }
    std::ofstream out(argv[2]);
    out << "i,t,total";
    for (int j = 0; j < (int)idx.size(); ++j) out << ",ch" << j;
    out << "\n";
    drift::DriftCompensator comp;
    int frames = 0;
    while (std::getline(in, line)) {
        std::stringstream ls(line);
        std::vector<std::string> f;
        std::string tok;
        while (std::getline(ls, tok, ',')) f.push_back(tok);
        if ((int)f.size() <= ti) continue;
        Eigen::VectorXd v(idx.size());
        for (int j = 0; j < (int)idx.size(); ++j)
            v(j) = std::stod(f[idx[j]]);
        const double ts = std::stod(f[ti]);
        comp.Process(ts, v);
        out << frames << "," << ts << "," << v.sum();
        for (int j = 0; j < (int)idx.size(); ++j) out << "," << v(j);
        out << "\n";
        ++frames;
    }
    std::cerr << frames << " frames\n";
    return 0;
}
