// v5 C++ 落地补丁的端到端行为自检 harness（不参与主工程构建）
// 直接 include 主工程里被修改后的 drift_compensator.{h,cpp}，喂同一组合成砝码信号，
// 输出逐帧显示值，交给 Python 侧与 glm53_v5.py 原型逐帧对拍。
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "domain/drift/drift_compensator.h"

// 与 Python 侧 glm53_v5 场景完全一致的合成信号：
//   0~20s 空载 → 20s 加载 10N(=10000) → 300s 再叠加 ratio×10000
//   蠕变律：15.64%·(1-e^(-t/199))（另一项在该时间尺度上为 0）
static const double kTau = 199.0;
static const double kAmp = 0.1564;

static double Creep(double u) {
    if (u <= 0.0) return 0.0;
    return kAmp * (1.0 - std::exp(-u / kTau));
}

int main(int argc, char** argv) {
    const double ratio = (argc > 1) ? std::atof(argv[1]) : 0.5;   // 第二档占 10N 的比例
    const double fast_s = (argc > 2) ? std::atof(argv[2]) : 3.0;  // 快相免责期(秒)
    const double fs = 100.0;
    const double dt = 1.0 / fs;
    const double dur = 400.0;
    const double t_load = 20.0, t_add = 300.0;

    drift::DriftCompensator comp;
    comp.Reset();
    comp.SetFastPhase(fast_s);          // 运行期切换（菜单「无责 3s / 无责 5s」）

    const int n = 1;
    for (int i = 0; i < static_cast<int>(dur * fs); ++i) {
        const double t = static_cast<double>(i) / fs;
        double val = 0.0;
        if (t > t_load) val += 10000.0 * (1.0 + Creep(t - t_load));
        if (t > t_add) val += ratio * 10000.0 * (1.0 + Creep(t - t_add));
        Eigen::VectorXd v(n);
        v(0) = val;
        comp.Process(t, v);
        if (i % 5 == 0) {                       // 每 0.05s 打一行，便于对拍
            std::printf("%.2f,%.6f,%.6f\n", t, val, v(0));
        }
    }
    return 0;
}
