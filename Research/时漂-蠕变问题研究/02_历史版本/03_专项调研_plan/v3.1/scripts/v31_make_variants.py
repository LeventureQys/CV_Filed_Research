# -*- coding: utf-8 -*-
"""生成「基线保持」候选变体源码（沙箱，不动产品源码）。

每个变体 = 产品源码 src/domain/drift_v6/* 复制到 results/v31_variants/<name>/ 后加补丁。
变体只在**慢相锚定 / 空闲语义**上不同，用来分离「基线丢失」的成因：

  cur      现役（对照）
  keep_g   空闲时冻结 g_ / γ（蠕变增益不随卸载清零）
  keep_slow 完全卸载时不回 Idle（保留 epoch 与慢相状态，重载走 C5/Reanchor）
  anchor_idle 交接锚定改为 A_new = min(y0 + share·Â, base_y + share·Â)
             （即"钉住=空载电平＋台阶"，不把已发生的蠕变算进无蠕变电平）

用法：python v31_make_variants.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
SRC = os.path.join(ROOT, "src", "domain", "drift_v6")
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "v31_variants"))


def patch(name, fn, text):
    if name == "cur":
        return text
    if name == "keep_g":
        if fn.endswith(".h"):
            return text
        # ToIdle 里 g_ = 0.0 → 保留（蠕变增益是负载属性，不是空闲属性）
        old = """    if (pct_ded_.size() > 0) pct_ded_.setZero();
    g_ = 0.0;
    idle_since_ = last_ts_;"""
        new = """    if (pct_ded_.size() > 0) pct_ded_.setZero();
    idle_since_ = last_ts_;"""
        assert old in text, "keep_g anchor not found"
        return text.replace(old, new)
    if name == "keep_slow":
        if fn.endswith(".h"):
            return text
        # 完全卸载不再回 Idle：保持锚定与 g，等重载走 C5/Reanchor
        old = """    if (state_ == State::Slow) {
        Eigen::VectorXd out;
        if (idle_now && (timestamp_s - ev_end_) > kUnloadFastS) {"""
        new = """    if (state_ == State::Slow) {
        Eigen::VectorXd out;
        if (false && idle_now && (timestamp_s - ev_end_) > kUnloadFastS) {"""
        assert old in text, "keep_slow anchor 1 not found"
        text = text.replace(old, new)
        old2 = """    // C4 事件期内落入空载带
    if (tau > kUnloadFastS && idle_now) {"""
        new2 = """    // C4 事件期内落入空载带
    if (false && tau > kUnloadFastS && idle_now) {"""
        assert old2 in text, "keep_slow anchor 2 not found"
        return text.replace(old2, new2)
    if name == "anchor_idle":
        if fn.endswith(".h"):
            return text
        # 交接锚定：无蠕变电平不得超过「空载电平 + 台阶」
        old = """    Eigen::VectorXd A_new = (ev_.y0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);"""
        new = """    Eigen::VectorXd A_new =
        (ev_.y0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);
    {
        Eigen::VectorXd lim = (ev_.v0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);
        A_new = A_new.cwiseMin(lim);
    }"""
        assert old in text, "anchor_idle anchor not found"
        return text.replace(old, new)
    if name == "anchor_base":
        if fn.endswith(".h"):
            return text
        # 决定性反事实：交接锚定改为「事件前的显示电平 + 台阶」，
        # 完全不用逆模型 Â 的绝对电平（只保留它作为上限）。
        # 若补偿在重载后回来 ⇒ 「锚定被 Â 的绝对电平带偏」就是基线丢失的根因。
        old = """    Eigen::VectorXd A_new = (ev_.y0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);"""
        new = """    Eigen::VectorXd A_new =
        (ev_.v0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);"""
        assert old in text, "anchor_base anchor not found"
        return text.replace(old, new)
    if name == "anchor_base_nocap":
        if fn.endswith(".h"):
            return text
        # 同上，但去掉 Â 只取「事件内实测增量 inc_max」⇒ 锚定 = 空载电平 + 实测台阶
        old = """    Eigen::VectorXd A_new = (ev_.y0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);"""
        new = """    Eigen::VectorXd A_new =
        (ev_.v0 + share * std::max(ev_.inc_max, 0.0)).cwiseMax(0.0);"""
        assert old in text, "anchor_base_nocap anchor not found"
        return text.replace(old, new)
    if name == "anchor_obs":
        if fn.endswith(".h"):
            return text
        # 对照：锚定改为「交接时刻实测逐通道电平」（历史纪要记载的 measured 模式）
        old = """    Eigen::VectorXd A_new = (ev_.y0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);"""
        new = """    Eigen::VectorXd A_new = v.cwiseMax(0.0);"""
        assert old in text, "anchor_obs anchor not found"
        return text.replace(old, new)
    raise SystemExit("unknown variant %s" % name)


VARIANTS = ("cur", "anchor_base", "anchor_base_nocap", "anchor_obs")


def main():
    for name in VARIANTS:
        d = os.path.join(OUT, name, "domain", "drift_v6")
        if os.path.isdir(os.path.join(OUT, name)):
            shutil.rmtree(os.path.join(OUT, name))
        os.makedirs(d)
        for fn in ("drift_v6_compensator.h", "drift_v6_compensator.cpp"):
            with open(os.path.join(SRC, fn), encoding="utf-8") as fh:
                text = fh.read()
            with open(os.path.join(d, fn), "w", encoding="utf-8") as fh:
                fh.write(patch(name, fn, text))
        print("variant %-12s -> %s" % (name, d))
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
