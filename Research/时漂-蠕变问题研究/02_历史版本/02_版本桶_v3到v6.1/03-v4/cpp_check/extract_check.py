# -*- coding: utf-8 -*-
"""从 04-C++实现骨架.md 抽取代码块，生成可编译的自检 harness。

目的：验证交付文档里的 C++ 片段**原样**能通过编译，并跑通「免责期 → 收尾 → 自检」流程。
只做片段级验证，不触碰主工程、不构建主程序。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(HERE), "Document", "04-C++实现骨架.md")
OUT = os.path.join(HERE, "v4_extracted.cpp")

src = open(DOC, encoding="utf-8").read()
blocks = re.findall(r"```cpp\n(.*?)```", src, re.S)
print(f"从文档抽取到 {len(blocks)} 个 cpp 代码块")

# 找出 3.4 节的 Process 分支（含 kFastPhaseS 与 exempt_acc_）
branch = None
for b in blocks:
    if "kFastPhaseS" in b and "exempt_acc_" in b and "if (in_load_" in b:
        branch = b
        break
if branch is None:
    print("ERROR: 未找到 3.4 节的分支片段")
    sys.exit(1)

# 找出 2.1 / 2.2 节的常量与成员：按「块内是否含目标标识符」整体取，再逐行过滤出声明
consts, members = [], []
for b in blocks:
    if "if (in_load_" in b or "ResetFor" in b or "BeginLoad" in b or "Process(" in b:
        continue                                   # 跳过 3.x 节的实现片段
    if "static constexpr" in b and "kFastPhaseS" in b:
        consts.append(b)
    if "Eigen::VectorXd" in b and ("exempt_acc_" in b or "kExempt" in b):
        for line in b.splitlines():
            s = line.strip()
            if not s or s.startswith("//") or s.startswith(">"):
                continue
            if "= Eigen::VectorXd::Zero(n)" in s:
                continue                           # 不是声明，是 ResetFor 里的赋值
            if "  //" in line:
                line = line.split("  //")[0].rstrip()
            members.append(line)
print(f"  常量块 {len(consts)} 个, 成员声明 {len(members)} 行")
print("  成员: " + " | ".join(m.strip() for m in members))
assert consts and members, "常量或成员块未找到"

consts_src = "\n".join(consts)
members_src = "\n".join(members)

harness = f'''// 由 extract_check.py 从 04-C++实现骨架.md 自动生成 —— 请勿手工编辑
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <cstdio>

class V4Snippet {{
public:
{consts_src}
    static constexpr double kLoadedFrac  = 0.10;   // 来自 v3 既有参数(harness 补齐)
    void ResetFor(int n) {{
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
    }}

    void Process(double ts, Eigen::VectorXd& v) {{
        // ---- 以下为文档 3.4 节片段的原样内容（v3 状态机在此前已跑完）----
{branch}
        // ---- 片段结束 ----
        Eigen::VectorXd Z = v - b_;
        v = Z;
    }}

    bool fast_done() const {{ return fast_done_; }}
    bool a_captured() const {{ return a_captured_; }}
    double amax() const {{ return A_.maxCoeff(); }}
    int n_loaded() const {{
        return static_cast<int>(std::count(loaded_.begin(), loaded_.end(), static_cast<char>(1)));
    }}
    int exempt_frames() const {{ return exempt_frames_; }}

private:
    int n_ = 0;
    Eigen::VectorXd b_, A_, a_acc_, g_rel_acc_, gamma_;
{members_src}
    std::vector<char> loaded_;
    bool a_captured_ = false, in_load_ = false;
    int a_frames_ = 0;
    double g_ = 0.0, g2_acc_ = 0.0, onset_ts_ = 0.0;
}};

int main() {{
    const int n = 6;
    V4Snippet s;
    s.ResetFor(n);
    const double dt = 0.01;

    // 免责期 0~5s：合成上升载荷（模拟快相），第 5 个通道弱受载
    for (int i = 0; i < 500; ++i) {{
        const double t = i * dt;
        Eigen::VectorXd v = Eigen::VectorXd::Constant(n, 1.0 + 0.4 * (t / 5.0));
        v(5) = 0.05;
        s.Process(t, v);
    }}
    const int frames_before_wrap = s.exempt_frames();
    // 收尾帧
    {{
        Eigen::VectorXd v = Eigen::VectorXd::Constant(n, 1.4);
        v(5) = 0.05;
        s.Process(5.0 + dt, v);
    }}

    std::printf("免责期末段采集帧数 = %d\\n", frames_before_wrap);
    std::printf("fast_done=%d  a_captured=%d  A_max=%.6f  n_loaded=%d/%d\\n",
                s.fast_done() ? 1 : 0, s.a_captured() ? 1 : 0,
                s.amax(), s.n_loaded(), n);
    const bool ok1 = s.amax() > 1e-9;
    const bool ok2 = s.n_loaded() > 0;
    const bool ok3 = s.fast_done();
    const bool ok4 = frames_before_wrap > 0;
    std::printf("自检: A非零=%d  loaded>0=%d  fast_done=%d  采集窗非空=%d\\n",
                ok1 ? 1 : 0, ok2 ? 1 : 0, ok3 ? 1 : 0, ok4 ? 1 : 0);
    return (ok1 && ok2 && ok3 && ok4) ? 0 : 1;
}}
'''

open(OUT, "w", encoding="utf-8").write(harness)
print(f"已生成 {OUT}")
