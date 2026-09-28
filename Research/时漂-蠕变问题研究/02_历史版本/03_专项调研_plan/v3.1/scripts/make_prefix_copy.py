# -*- coding: utf-8 -*-
"""生成「修复前」副本：把 src 的 v6 补偿器拷到 scripts/prefix/，并**反向**套用本轮 4 处修复。

用途：离线 A/B 的对照臂（修复前 vs 修复后），不参与产品构建。
用法: python make_prefix_copy.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
SRC = os.path.join(ROOT, "src", "domain", "drift_v6")
DST = os.path.join(HERE, "prefix", "domain", "drift_v6")

# (修复后的文本, 修复前的文本)
REVERSIONS = [
    # ① DeductionVector 拆出 Rank1DeductionVector（修复后新增的函数 → 还原为单一函数）
    ("""// rank-1 部分（v5 口径，仅 loaded_ 通道）。**不能直接用它的总和代表"当前扣除"** ——
// 慢相实际施加的是 DeductionVector()（它还含 PCT，且作用域更宽）。
Eigen::VectorXd DriftV6Compensator::Rank1DeductionVector() const {
    Eigen::VectorXd d = Eigen::VectorXd::Zero(n_);
    for (int k = 0; k < n_; ++k) {
        if (loaded_[k] && A_(k) > 1e-9)
            d(k) = std::clamp(gamma_(k) * A_(k) * g_, kCreepLoFrac * A_(k),
                              kCreepHiFrac * A_(k));
    }
    return d;
}
""", """Eigen::VectorXd DriftV6Compensator::Rank1DeductionVector() const {
    Eigen::VectorXd d = Eigen::VectorXd::Zero(n_);
    for (int k = 0; k < n_; ++k) {
        if (loaded_[k] && A_(k) > 1e-9)
            d(k) = std::clamp(gamma_(k) * A_(k) * g_, kCreepLoFrac * A_(k),
                              kCreepHiFrac * A_(k));
    }
    return d;
}
"""),
    # ② DeductionVector 加回 PCT（修复后含 PCT → 还原为只含 rank-1）
    ("""Eigen::VectorXd DriftV6Compensator::DeductionVector() const {
    Eigen::VectorXd d = Rank1DeductionVector();
    if (pct_ded_.size() == n_) {
        for (int k = 0; k < n_; ++k) {
            if (k < static_cast<int>(pct_elig_.size()) && pct_elig_[k] && A_(k) > 1e-9)
                d(k) += pct_ded_(k);
        }
    }
    return d;
}""", """Eigen::VectorXd DriftV6Compensator::DeductionVector() const {
    return Rank1DeductionVector();
}"""),
    # ③ Handoff 清零 pct（修复后新增 → 删除）
    ("""    if (pct_ded_.size() == n_) pct_ded_.setZero();
""", """"""),
    # ④ Reanchor：A_new 用总扣除、g 用 rank-1（修复后 → 还原为都用 ded_old）
    ("""    const Eigen::VectorXd ded1_old = Rank1DeductionVector();
    const Eigen::VectorXd ded_tot = DeductionVector();
    Eigen::VectorXd A_new = (vs - ded_tot).cwiseMax(0.0);""",
     """    const Eigen::VectorXd ded1_old = DeductionVector();
    Eigen::VectorXd A_new = (vs - ded1_old).cwiseMax(0.0);"""),
    # ⑤ 沉降窗逐通道施加条件（修复后 → 还原为只看 loaded_）
    ("""                if (std::abs(hold_comp_(k)) > 1e-12)
                    o(k) = v(k) - Capped(v(k), hold_comp_(k));""",
     """                if (loaded_[k] && A_(k) > 1e-9)
                    o(k) = v(k) - Capped(v(k), hold_comp_(k));"""),
    # ⑥ 头文件：保留两套声明（修复前后都能编过），不改
]


def main():
    os.makedirs(DST, exist_ok=True)
    for fn in ("drift_v6_compensator.h", "drift_v6_compensator.cpp"):
        shutil.copy2(os.path.join(SRC, fn), os.path.join(DST, fn))
    p = os.path.join(DST, "drift_v6_compensator.cpp")
    txt = open(p, encoding="utf-8").read()
    for i, (new, old) in enumerate(REVERSIONS, 1):
        if new not in txt:
            print("!! 第 %d 处未命中（源码已变？）" % i)
            return 1
        txt = txt.replace(new, old, 1)
        print("  反向套用 %d OK" % i)
    open(p, "w", encoding="utf-8", newline="\n").write(txt)
    print("-> %s（修复前副本）" % DST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
