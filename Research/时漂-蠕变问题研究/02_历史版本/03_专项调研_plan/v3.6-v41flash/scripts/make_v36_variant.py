# -*- coding: utf-8 -*-
"""生成 plan-v3.6 候选算法源码（**不改产品源码**，只在 plan/v3.6-v41flash/ 下生成变体）。

基线 = `plan/v3.2/src_v32/`（= 产品 `src/` ＋ R1「重载蠕变接力」，由
`plan/v3.2/scripts/make_v32_variant.py` 生成、已通过 v3.2 的 A/B 验证）。
本脚本在其上叠加 v3.6 的补丁（见 V36_H / V36_C），**产品 `src/` 一行不动**。

为什么以 src_v32 为基线而不是重新跑一遍 v3.2 的补丁表：
`make_v32_variant.py` 当前的补丁表里有若干锚点已与 `src/` 漂移（例如把 `.h` 的常量锚点
写进了 `.cpp` 补丁组），直接重跑会 SystemExit。src_v32 是 v3.2 已验收的产物，
直接以它做基线可保证「v3.6 = v3.2 + 本轮改动」，A/B 的臂间可比性最强。

用法：
  python make_v36_variant.py
  cmd /c build_v36_runner.bat
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
V32_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "v3.2", "src_v32", "domain",
                                       "drift_v6"))
DST = os.path.abspath(os.path.join(HERE, "..", "src_v36", "domain", "drift_v6"))

# ─────────────────────────── v3.6 补丁（头文件）───────────────────────────
V36_H = [
    # 1) 运行期旋钮
    ("""        double creep_seed_gain    = kCreepSeedGain;   // 种子折减(1.0 = 全额; <1 更保守)
        // 诊断计数(只写不读业务逻辑)""",
     """        double creep_seed_gain    = kCreepSeedGain;   // 种子折减(1.0 = 全额; <1 更保守)
        // ── plan-v3.6：本轮实验旋钮（默认值 = 采用值；语义见 .cpp 顶部 v3.6 说明）──
        double v36_a = kV36A;
        double v36_b = kV36B;
        double v36_c = kV36C;
        double v36_d = kV36D;
        // 诊断计数(只写不读业务逻辑)"""),
    # 2) 常量
    ("""    static constexpr double kCreepRelMin     = 0.001;
""",
     """    static constexpr double kCreepRelMin     = 0.001;
    // plan-v3.6：本轮实验旋钮（默认值 = 基线，行为与 plan-v3.2 逐位一致）
    static constexpr double kV36A = 0.0;
    static constexpr double kV36B = 0.0;
    static constexpr double kV36C = 0.0;
    static constexpr double kV36D = 0.0;
"""),
]

# ─────────────────────────── v3.6 补丁（实现文件）───────────────────────────
V36_C = [
    # ── R2.1：seed 只在一处施加（事件目标里），交接自然连续 ──
    #  v3.2 的 seed 同时出现在 ① RunEvent 的输出项、② Handoff 的 ded_old、③ Handoff 的
    #  pct_ded_ 三处，且 ② 的符号与 ① 相反 ⇒ 每次带 seed 的交接都会让扣除量塌成负值、
    #  显示被 C 限幅顶到 raw×1.005 后花 ~3 s 慢慢滑回来（实测 245.31 s 重载：交接当帧
    #  ded 由 +2297 跳到 −81）。修法：seed 折进 target，其余各处全部删掉。
    ("""    const double target = ev_.base_y + ev_.a_hat;
""",
     """    // ── plan-v3.6 R2.1：seed 折进目标电平（唯一施加点）──
    // 事件期显示目标 = 沿前显示 + Â − seed；交接时 A = y0 + share·(Â − seed) 与之一致，
    // ded_old 直接取 c_applied 推出的实际扣除 ⇒ 交接处严格连续。
    const double seed_now = std::max(ev_.seed_used, 0.0);
    const double target = ev_.base_y + ev_.a_hat - seed_now;
"""),
    ("""    const Eigen::VectorXd share = ShareVector(ev_, v);
    *out = v + share * c_target;
    // R1：注入的蠕变种子按 w_seed 渐入（权重与实测上升同步 ⇒ 与 c 的过渡天然连续）
    if (ev_.seed_used > 0.0) {
        const double sw = share.sum();
        if (sw > 1e-12) *out += (share / sw) * (-ev_.seed_used);
    }
    return true;
}""",
     """    const Eigen::VectorXd share = ShareVector(ev_, v);
    *out = v + share * c_target;      // seed 已在 target 内（R2.1），此处不再另加
    return true;
}"""),
    ("""    Eigen::VectorXd ded_old = -(share * ev_.c_applied);
    // ── plan-v3.2 R1：seed 是「事件期的固定扣除」，必须一起交给慢相 ──
    // 否则交接当帧显示会跳 |seed|（plan-v3.0 漏 PCT 时踩过同一个坑，见 .cpp 的 ①② 说明）。
    // 用事件期**末帧**的 seed_used，慢相再按同一 share 施加 ⇒ 严格连续。
    if (ev_.seed_used > 0.0) {
        const double sw = share.sum();
        if (sw > 1e-12) ded_old += (share / sw) * (-ev_.seed_used);
    }
""",
     """    // plan-v3.6 R2.1：seed 已在 c_applied 内（事件目标含 seed）⇒ ded_old 就是实际扣除，
    // 不能再单独加一次（v3.2 在这里加了 −seed，符号与事件期相反，是"交接塌陷"的根因）。
    Eigen::VectorXd ded_old = -(share * ev_.c_applied);
"""),
    ("""    // R1：seed 以逐通道扣除的形式继续存在于慢相（与事件期末帧一致，交接无跳变）
    if (seed_amt > 0.0 && pct_ded_.size() == n_) {
        const double sw = share.sum();
        if (sw > 1e-12) pct_ded_ -= (share / sw) * seed_amt;
    }
""",
     ""),
    ("""    // ── plan-v3.2 R1：事件期注入的蠕变种子继续按同一分配向量施加 ──
    // 它不参与 g/PCT 的自适应（那两项只负责"再长出来的那一部分"），只把载荷的蠕变起点
    // 带回显示；作用域与事件期一致（share 向量），保证交接处严格连续。
    const double seed_amt = std::max(ev_.seed_used, 0.0);
    if (seed_amt > 1e-12) {
        const Eigen::VectorXd share = ShareVector(ev_, v);
        const double sw = share.sum();
        if (sw > 1e-12) {
            const Eigen::VectorXd seed_v = (share / sw) * (-seed_amt);
            for (int k = 0; k < n_; ++k) ded(k) += seed_v(k);
        }
    }
""",
     """    // plan-v3.6 R2.1：慢相不再单独施加 seed —— 它已经通过 A_ 与 ded1 的锚定进入扣除
    // （Handoff 的 A_new = y0 + share·(Â − seed)）。v3.2 在这里再施一次会让慢相双计。
"""),
    # ── R2.2：seed 作用域放宽到"所有加载事件"（由 v36_b 开关控制）──
    #  完全卸载后重载 = Onset-from-Idle（v3.2 已覆盖）；短保压后的部分卸载→重载 =
    #  Restep（v3.2 未覆盖，实测 289.5 / 303.1 s 两处补偿仍为 0）。
    ("""    if (kind == Kind::Onset && prev == State::Idle && creep_ratio_ >= params_.creep_ratio_min)
        ev_.creep_seed = -1.0;      // <0 = 允许注入(占位)，数值在 RunEvent 中填""",
     """    {
        const bool load_up = (kind == Kind::Onset || kind == Kind::Restep ||
                              kind == Kind::RestepReload);
        const bool onset_from_idle = (kind == Kind::Onset && prev == State::Idle);
        // v36_b > 0.5 ⇒ 对**所有**加载事件注入（含部分卸载后的重载 Restep）；
        // 默认(0) 保持 v3.2 语义：只对「从 Idle 起跳的 Onset」注入。
        const bool allow = load_up && creep_ratio_ >= params_.creep_ratio_min &&
                           (params_.v36_b > 0.5 || onset_from_idle);
        if (allow) ev_.creep_seed = -1.0;   // <0 = 允许注入(占位)，数值在 RunEvent 中填
    }"""),
    # ── R2.3：空载期蠕变比的衰减（v36_c>0 才开；默认 0 = 与 v3.2 同为"不衰减"）──
    ("""    if (state_ == State::Idle && dt > 0.0) {
        const double exp_factor = std::exp(-dt / params_.creep_hold_tau_s);
        creep_ratio_ = prev_creep_ + (creep_ratio_ - prev_creep_) * exp_factor;
    }""",
     """    if (state_ == State::Idle && dt > 0.0 && params_.v36_c > 0.0) {
        // v3.2 原式 `prev + (cur−prev)·e^(−dt/τ)` 因为 prev_creep_ 恒等于 cur ⇒ **实际不衰减**。
        // 物理上空载期蠕变会部分恢复。v36_c > 0 时按 τ=v36_c 向 0 衰减（本轮默认 0 = 不衰减，
        // 因为短时卸载（几秒）恢复量有限，而"记忆被抹掉"会让后续重载失去一致锚定）。
        creep_ratio_ *= std::exp(-dt / params_.v36_c);
        prev_creep_ = creep_ratio_;
    }"""),
    # ── R2.4：r 的更新改为"可单侧"（v36_d>0 ⇒ 下降用更慢的 τ；=0 保持对称）──
    ("""        if (std::abs(rel) >= kCreepRelMin && dt > 0.0 &&
            !(state_ == State::Event && ev_.creep_seed < 0.0)) {
            creep_ratio_ += (dt / params_.creep_tau_s) * (rel - creep_ratio_);""",
     """        if (std::abs(rel) >= kCreepRelMin && dt > 0.0 &&
            !(state_ == State::Event && ev_.creep_seed < 0.0)) {
            // v36_d > 0 ⇒ 下降沿用更慢的时间常数（避免"退化 episode 把会话记忆拖小 ⇒
            // 下一次 seed 更小 ⇒ 更退化"的自激下降；上升仍用 creep_tau_s）。
            double tau_r = params_.creep_tau_s;
            if (params_.v36_d > 0.0 && rel < creep_ratio_) tau_r = params_.v36_d;
            creep_ratio_ += (dt / tau_r) * (rel - creep_ratio_);"""),
]


def apply(text, patches, tag):
    for i, (old, new) in enumerate(patches, 1):
        if old not in text:
            raise SystemExit("[%s] 第 %d 处未命中（基线已变？）:\n%s" % (tag, i, old[:200]))
        if text.count(old) != 1:
            raise SystemExit("[%s] 第 %d 处命中 %d 次（应唯一）" % (tag, i, text.count(old)))
        text = text.replace(old, new, 1)
    return text


def build_v36_sources():
    with open(os.path.join(V32_SRC, "drift_v6_compensator.h"), encoding="utf-8") as fh:
        h = fh.read()
    with open(os.path.join(V32_SRC, "drift_v6_compensator.cpp"), encoding="utf-8") as fh:
        c = fh.read()
    h = apply(h, V36_H, "v36-h")
    c = apply(c, V36_C, "v36-cpp")
    c = c.replace("// ===== plan-v3.2 R1 变体（由 make_v32_variant.py 从 src/ 生成，勿手改）=====",
                  "// ===== plan-v3.6 候选 = v3.2(R1) + 本轮补丁（由 make_v36_variant.py 生成，勿手改）=====",
                  1)
    return h, c


def main():
    os.makedirs(DST, exist_ok=True)
    h, c = build_v36_sources()
    with open(os.path.join(DST, "drift_v6_compensator.h"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(h)
    with open(os.path.join(DST, "drift_v6_compensator.cpp"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(c)
    print("基线: %s" % V32_SRC)
    print("-> %s" % DST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
