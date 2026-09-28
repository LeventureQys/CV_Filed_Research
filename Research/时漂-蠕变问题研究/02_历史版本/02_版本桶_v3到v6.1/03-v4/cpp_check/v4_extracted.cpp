// 由 extract_check.py 从 04-C++实现骨架.md 自动生成 —— 请勿手工编辑
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <cstdio>

class V4Snippet {
public:
    // ── v4: 快相免责期（加载瞬间的确定性快速爬升不做补偿）──
    // 依据: 9 组实测数据显示加载后 0~1s 为机械加载、1~4s 为接触建立(4s 完成 92%)、
    //       4s 后进入慢相蠕变; 快相归一化形状 9 组逐点标准差 <= 0.026 (高度可复现)。
    static constexpr double kFastPhaseS   = 5.0;  // 免责期长度(首次 onset 后, 秒)
    static constexpr double kExemptAWinS  = 1.5;  // 幅度采集窗长度, 紧贴免责期末端

    static constexpr double kLoadedFrac  = 0.10;   // 来自 v3 既有参数(harness 补齐)
    void ResetFor(int n) {
        n_ = n;
        b_ = Eigen::VectorXd::Zero(n);
        A_ = Eigen::VectorXd::Zero(n);
        a_acc_ = Eigen::VectorXd::Zero(n);
        g_rel_acc_ = Eigen::VectorXd::Zero(n);
        gamma_ = Eigen::VectorXd::Ones(n);
        exempt_acc_ = Eigen::VectorXd::Zero(n);
        loaded_.assign(n, 0);
        fast_done_ = false;
        exempt_frames_ = 0;
        a_frames_ = 0;
        a_captured_ = false;
        g_ = 0.0;
        g2_acc_ = 0.0;
        in_load_ = true;
        onset_ts_ = 0.0;
    }

    void Process(double ts, Eigen::VectorXd& v) {
        // ---- 以下为文档 3.4 节片段的原样内容（v3 状态机在此前已跑完）----
    // ── v4: 快相免责期(首次 onset 与负载内变载都生效) ──────────────────
    // 免责期内不扣蠕变, 输出直通(只扣零漂基线 b_); 免责期结束当帧捕获幅度 A_
    // 并清零 g/γ, 让慢相蠕变的 epoch 从快相末开始计。
    if (in_load_ && !fast_done_) {
        const double ux = ts - onset_ts_;
        if (ux < kFastPhaseS) {
            // 冻结蠕变状态: 不积分 g/γ、不累积蠕变、不使用 v3 的 A 窗
            a_captured_ = false;
            A_.setZero();
            a_acc_.setZero();
            a_frames_ = 0;
            g_ = 0.0;
            g2_acc_ = 0.0;
            g_rel_acc_.setZero();
            gamma_.setOnes();

            // 幅度采集: 仅紧贴免责期末端的窗口 [kFastPhaseS-kExemptAWinS, kFastPhaseS]
            if (ux >= kFastPhaseS - kExemptAWinS) {
                exempt_acc_ += v - b_;
                ++exempt_frames_;
            }

            v = v - b_;                 // 直通(仅扣零漂基线), 不做蠕变扣除
            return;
        }

        // ── 免责期结束当帧: 收尾 ──
        // 注意: 这里不能写成 A_ = (frames>0) ? (acc/frames) : (v-b_);
        // Eigen 的 ?: 两个分支表达式类型不同 (商表达式 vs 差表达式), 无法隐式转换, 编译不过。
        // 必须先赋值再按需覆盖。
        A_ = v - b_;                    // 兜底: 采集窗为空时用当帧 Z
        if (exempt_frames_ > 0)
            A_ = exempt_acc_ / static_cast<double>(exempt_frames_);
        const double amax = A_.maxCoeff();
        if (amax > 1e-9) {
            for (int i = 0; i < n_; ++i)
                loaded_[i] = A_(i) > kLoadedFrac * amax ? 1 : 0;
        } else {
            std::fill(loaded_.begin(), loaded_.end(), 0);
        }
        a_captured_ = true;
        g_ = 0.0;
        g2_acc_ = 0.0;
        g_rel_acc_.setZero();
        gamma_.setOnes();
        exempt_acc_.setZero();
        exempt_frames_ = 0;
        fast_done_ = true;
        // 不 return: 当帧继续走 v3 的蠕变补偿逻辑
    }

        // ---- 片段结束 ----
        Eigen::VectorXd Z = v - b_;
        v = Z;
    }

    bool fast_done() const { return fast_done_; }
    bool a_captured() const { return a_captured_; }
    double amax() const { return A_.maxCoeff(); }
    int n_loaded() const {
        return static_cast<int>(std::count(loaded_.begin(), loaded_.end(), static_cast<char>(1)));
    }
    int exempt_frames() const { return exempt_frames_; }

private:
    int n_ = 0;
    Eigen::VectorXd b_, A_, a_acc_, g_rel_acc_, gamma_;
    bool fast_done_ = false;
    Eigen::VectorXd exempt_acc_;
    int exempt_frames_ = 0;
    std::vector<char> loaded_;
    bool a_captured_ = false, in_load_ = false;
    int a_frames_ = 0;
    double g_ = 0.0, g2_acc_ = 0.0, onset_ts_ = 0.0;
};

int main() {
    const int n = 6;
    V4Snippet s;
    s.ResetFor(n);
    const double dt = 0.01;

    // 免责期 0~5s：合成上升载荷（模拟快相），第 5 个通道弱受载
    for (int i = 0; i < 500; ++i) {
        const double t = i * dt;
        Eigen::VectorXd v = Eigen::VectorXd::Constant(n, 1.0 + 0.4 * (t / 5.0));
        v(5) = 0.05;
        s.Process(t, v);
    }
    const int frames_before_wrap = s.exempt_frames();
    // 收尾帧
    {
        Eigen::VectorXd v = Eigen::VectorXd::Constant(n, 1.4);
        v(5) = 0.05;
        s.Process(5.0 + dt, v);
    }

    std::printf("免责期末段采集帧数 = %d\n", frames_before_wrap);
    std::printf("fast_done=%d  a_captured=%d  A_max=%.6f  n_loaded=%d/%d\n",
                s.fast_done() ? 1 : 0, s.a_captured() ? 1 : 0,
                s.amax(), s.n_loaded(), n);
    const bool ok1 = s.amax() > 1e-9;
    const bool ok2 = s.n_loaded() > 0;
    const bool ok3 = s.fast_done();
    const bool ok4 = frames_before_wrap > 0;
    std::printf("自检: A非零=%d  loaded>0=%d  fast_done=%d  采集窗非空=%d\n",
                ok1 ? 1 : 0, ok2 ? 1 : 0, ok3 ? 1 : 0, ok4 ? 1 : 0);
    return (ok1 && ok2 && ok3 && ok4) ? 0 : 1;
}
