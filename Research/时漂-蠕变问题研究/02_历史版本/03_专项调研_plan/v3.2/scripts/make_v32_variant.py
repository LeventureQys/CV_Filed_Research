# -*- coding: utf-8 -*-
"""生成 plan-v3.2 候选算法源码（**不改产品源码**，只在 plan/v3.2/ 下生成一份变体）。

本轮目标（v3.2 / R1「重载蠕变接力」）：修掉「完全卸载 → 重载后补偿归零」。
定因见 `../v3.1/results/v31_reload_gap.txt` 与 `问题清单.md`：
  `Handoff()` 的 `A_new = y0 + share·Â` 把「无蠕变电平」钉在**交接时刻的显示电平**上；
  首次加载时显示还停在快相平台（远低于最终平台）⇒ 慢相有 4~5 kADC 余量长出补偿；
  完全卸载后重载时，交接时刻的显示**已经是平台的慢相电平** ⇒ `g_raw≈0` ⇒ 补偿恒 0。

改法（两条，互相配套）：
  ① 会话级蠕变比记忆 `creep_ratio_`：慢相把「实际扣除 / 锚定电平」的比值按 τ=30 s 平滑，
     完全卸载时把该比值按 τ=45 s 衰减（物理上短时卸载只部分恢复），完全卸载后重载时
     作为**起点**注入新 epoch —— 蠕变（粘弹性）不会因为读数为零就消失。
  ② 锚定不再被"当前显示电平"钉住：交接时 `A_new = y0 + share·(Â − creep_seed)`，
     即从「无蠕变电平」里扣掉「本次加载已经长出来的蠕变量」，让慢相仍有余量。
     ③ 事件期把 seed 扣除按 1.5 s smoothstep 渐入（避免单帧跳变）。

用法：
  python make_v32_variant.py            # 生成 ../src_v32/domain/drift_v6/*
  python build_v32_runner.bat           # 编译 ../build/v32_runner.exe
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
SRC = os.path.join(ROOT, "src", "domain", "drift_v6")
DST = os.path.abspath(os.path.join(HERE, "..", "src_v32", "domain", "drift_v6"))

# ── 头文件补丁：(原文, 新文) ──────────────────────────────────────────────
H_PATCHES = [
    # 1) 运行期参数
    ("""        bool   freeze_slow_on_hit = kFreezeSlowOnHit;  // 检测器命中当帧起暂停慢相自适应
        // 诊断计数(只写不读业务逻辑)""",
     """        bool   freeze_slow_on_hit = kFreezeSlowOnHit;  // 检测器命中当帧起暂停慢相自适应
        // ── plan-v3.2 R1：重载蠕变接力（见 .cpp 顶部 R1 说明）──
        double creep_ratio_start  = kCreepRatioStart; // 会话蠕变比初值(冷启动 = 0)
        double creep_ratio_min    = kCreepRatioMin;   // 低于此值不参与种子(噪底/未收敛)
        double creep_ratio_max    = kCreepRatioMax;   // 上界钳位(防离谱换算)
        double creep_tau_s        = kCreepTauS;       // 慢相内蠕变比平滑时间常数
        double creep_hold_tau_s   = kCreepHoldTauS;   // 空载期蠕变比衰减时间常数(部分恢复)
        double creep_seed_gain    = kCreepSeedGain;   // 种子折减(1.0 = 全额; <1 更保守)
        // 诊断计数(只写不读业务逻辑)"""),
    # 2) 诊断计数
    ("""        long n_pct_hits       = 0;   // PCT 真正写入过扣除量的帧·通道数
    };""",
     """        long n_pct_hits       = 0;   // PCT 真正写入过扣除量的帧·通道数
        long n_creep_seed     = 0;   // R1: 重载注入过蠕变种子的次数
    };"""),
    # 3) 常量
    ("""    static constexpr double kUnloadFastS = 0.30;  // C4: v5 的 u>3s 收到 0.3s
""",
     """    static constexpr double kUnloadFastS = 0.30;  // C4: v5 的 u>3s 收到 0.3s

    // ── plan-v3.2 R1：重载蠕变接力 ──
    // 动机：完全卸载会把「已经长出来的蠕变补偿」清零，重载后 `Handoff` 又把锚定钉在当前
    // 显示电平上 ⇒ 补偿恒 0（实测 240 s 后每段 ded = −85~+180 ADC，而此前为 +2232）。
    // 做法：把「扣除/锚定」的比值当会话级状态带着走（物理上蠕变是粘弹性、不会因读数为零消失）。
    // τ 取 30 s / 45 s：前者平滑慢相内的量测（远慢于 g 的 3 s），后者近似「空载读数在
    // 读数域之上时的部分恢复」（本录制卸载间隙 4.5 s ⇒ 只衰减 ~10%）。
    static constexpr double kCreepRatioStart = 0.0;    // 冷启动不注入(保证首次加载逐位不变)
    static constexpr double kCreepRatioMin   = 0.02;   // 低于 2% 视为未收敛/噪底
    static constexpr double kCreepRatioMax   = 0.35;   // 上界钳位(实测本录制 0.13~0.14)
    static constexpr double kCreepTauS       = 30.0;
    static constexpr double kCreepHoldTauS   = 45.0;
    // 种子折减：注入多少"历史蠕变比"。1.0 = 全额（把历史蠕变当成本次加载已长出的量）。
    // 取值由 A/B 定（见 results/v32_gain_ab.txt）：全额在「连续多次卸载-重载」上补偿最足，
    // 但重载后的头 1~2 s 会因"事件期滑行修正与种子叠加"而多扣（显示偏低）。
    static constexpr double kCreepSeedGain  = 0.70;
    // 会话蠕变比只有在「有实际扣除量」时才更新，避免空载/未收敛期的 0 污染记忆
    static constexpr double kCreepRelMin     = 0.001;
"""),
    # 4) EventCtx 字段
    ("""        double a_hat = 0.0;
        double inc_max = 0.0;""",
     """        double a_hat = 0.0;
        double creep_seed = 0.0;   // plan-v3.2 R1: 本次事件注入的蠕变扣除(电平口径, ≤0)
        double t_seed = 0.0;       // plan-v3.2 R1: seed 注入的起始时刻
        double w_seed = 0.0;       // plan-v3.2 R1: seed 渐入权重(0→1, 时间与实测上升同步)
        double seed_used = 0.0;    // plan-v3.2 R1: 实际采用的电平(交接时冻结, 保证连续)
        double inc_max = 0.0;"""),
    # 5) 成员状态
    ("""    // plan-v3.0：逐通道蠕变跟踪（PCT）的扣除量与生效掩码（绝对电平口径，单位同输入）
    Eigen::VectorXd pct_ded_;
    std::vector<char> pct_elig_;""",
     """    // plan-v3.0：逐通道蠕变跟踪（PCT）的扣除量与生效掩码（绝对电平口径，单位同输入）
    Eigen::VectorXd pct_ded_;
    std::vector<char> pct_elig_;

    // plan-v3.2 R1：会话级蠕变比记忆(跨卸载/重载保留)
    double creep_ratio_ = kCreepRatioStart;   // 平滑后的「实际扣除 / 锚定电平」
    double prev_creep_ = 0.0;                 // 上一次卸载前一刻的瞬时比值(供衰减用)"""),
]

# ── 实现文件补丁：(原文, 新文) ────────────────────────────────────────────
C_PATCHES = [
    # 1) Reset
    ("""    shape_hits_ = 0;
    n_revoke_ = 0;""",
     """    shape_hits_ = 0;
    n_revoke_ = 0;
    // plan-v3.2 R1：Reset() 只重置**补偿状态**；会话蠕变比是"负载/器件属性"，
    // 与 A_/g_ 同类状态一起清空(重新开关算法 = 新会话)。
    creep_ratio_ = params_.creep_ratio_start;
    prev_creep_ = 0.0;"""),
    # 2) 两个新函数（插在 InvEst 之前）
    ("""// ───────────────────────────── 逆模型 ─────────────────────────────

bool DriftV6Compensator::InvEst(""",
     """// ─────────────────── plan-v3.2 R1：会话级蠕变比记忆 ───────────────────
//
// 语义：`creep_ratio_` = 「慢相当前实际施加的总扣除」/「锚定电平 A 的总和」。
//   它回答"这个负载在当前器件上已经长出了多少比例的蠕变"，是**负载/器件属性**，
//   不是 epoch 属性 —— 因此完全卸载（读数归零）不该把它抹掉。
//   · 慢相每帧按 τ=kCreepTauS 向瞬时比值平滑（只在"确有扣除量"时更新，避免空载 0 污染）；
//   · 空载期按 τ=kCreepHoldTauS 向 0 衰减（物理上短时卸载只部分恢复、长时间卸载会恢复）；
//   · 卸载前一刻的瞬时值记在 prev_creep_，衰减从它出发（避免卸载瞬间比值被 0 拉平）。
void DriftV6Compensator::CreepSeedMaintain(double ts, double dt) {
    (void)ts;
    const bool have_geom =
        (A_.size() == n_) && (pct_ded_.size() == n_) && (gamma_.size() == n_);
    const double asum = have_geom ? A_.sum() : 0.0;
    if (have_geom && asum > 1e-9) {
        const double ded_sum = DeductionVector().sum();
        const double rel = ded_sum / asum;
        if (std::abs(rel) >= kCreepRelMin && dt > 0.0 &&
            !(state_ == State::Event && ev_.creep_seed < 0.0)) {
            creep_ratio_ += (dt / params_.creep_tau_s) * (rel - creep_ratio_);
            creep_ratio_ = std::clamp(creep_ratio_, -params_.creep_ratio_max,
                                      params_.creep_ratio_max);
            prev_creep_ = creep_ratio_;
        }
    }
    if (state_ == State::Idle && dt > 0.0) {
        const double exp_factor = std::exp(-dt / params_.creep_hold_tau_s);
        creep_ratio_ = prev_creep_ + (creep_ratio_ - prev_creep_) * exp_factor;
    }
}

// ───────────────────────────── 逆模型 ─────────────────────────────

bool DriftV6Compensator::InvEst("""),
    # 3) NewEvent：登记 R1 元数据（seed 的**数值**要等实测增量出来才算得准，见 4)）
    ("""    ev_.tglide = kGlideMaxS;
    ev_.y_prev = prev_out_;
    state_ = State::Event;
    hold_comp_.resize(0);
}""",
     """    ev_.tglide = kGlideMaxS;
    ev_.y_prev = prev_out_;
    // ── plan-v3.2 R1：登记"允许注入蠕变种子"──
    // ① 只对「从 Idle 起跳的 Onset」注入：Idle 意味着真实卸载过（Slow 态完全卸载即 ToIdle）；
    // ② 数值本身必须等实测增量出来才算得准（建事件当帧 Â=0、inc_max=0），
    //    因此这里只置意图，实际扣减在 RunEvent 里按 inc_max 惰性给出（见 R1 说明）。
    ev_.creep_seed = 0.0;
    ev_.seed_used = 0.0;
    ev_.w_seed = 0.0;
    ev_.t_seed = 0.0;
    if (kind == Kind::Onset && prev == State::Idle && creep_ratio_ >= params_.creep_ratio_min)
        ev_.creep_seed = -1.0;      // <0 = 允许注入(占位)，数值在 RunEvent 中填
#ifdef V32_DBG
    if (std::getenv("V32_DBG_FRAME") != nullptr)
        std::fprintf(stderr, "NEWEV t=%.2f kind=%d prev=%d ratio=%.4f seed0=%.1f\\n",
                     last_ts_, static_cast<int>(kind), static_cast<int>(prev),
                     creep_ratio_, ev_.creep_seed);
#endif
    state_ = State::Event;
    hold_comp_.resize(0);
}"""),
    # 3b) RunEvent：R1 惰性计算 seed + 与实测上升同步的渐入
    ("""    const double target = ev_.base_y + ev_.a_hat;
""",
     """    // ── plan-v3.2 R1：完全卸载后重载 ⇒ 把"这个负载已经长出来的蠕变"注入本次事件 ──
    // 数值用 ev_.inc_max（实测电平增量），不用 Â：Â 在交接前的 1 s 内已收敛到 1.05~1.12×inc，
    // 而 inc_max 单调、无形状先验，正好是"蠕变比"的合适分母。
    // 渐入权重完全由**实测上升**驱动（w = min(2·inc_s / inc_max, 1)）：
    //   · 输入还接近 0 时不扣（否则 seed 会把显示一帧打穿、被 Capped 截成 0 反而失真）；
    //   · inc_s → inc_max（慢相起点）时 w→1，seed 全额生效；
    //   · 与滑行修正 c 共用同一上升过程 ⇒ 两者叠加天然连续，不需要额外时间窗。
    if (ev_.creep_seed < 0.0) {
        const double amt = params_.creep_seed_gain *
                           std::min(creep_ratio_, params_.creep_ratio_max) *
                           std::max(ev_.inc_max, 0.0);
        if (amt > 0.0) {
            ev_.creep_seed = -amt;
            const double inc_ref2 = std::max(ev_.inc_max, eps);
            ev_.w_seed = std::clamp(2.0 * inc_s / inc_ref2, 0.0, 1.0);
        }
        ev_.seed_used = -ev_.creep_seed * ev_.w_seed;      // 正数：本帧实际采用的电平
    }

    const double target = ev_.base_y + ev_.a_hat;
"""),
    # 4) 事件期输出：seed 渐入
    ("""    const Eigen::VectorXd share = ShareVector(ev_, v);
    *out = v + share * c_target;
    return true;
}""",
     """    const Eigen::VectorXd share = ShareVector(ev_, v);
    *out = v + share * c_target;
    // R1：注入的蠕变种子按 w_seed 渐入（权重与实测上升同步 ⇒ 与 c 的过渡天然连续）
    if (ev_.seed_used > 0.0) {
        const double sw = share.sum();
        if (sw > 1e-12) *out += (share / sw) * (-ev_.seed_used);
    }
    return true;
}"""),
    # 5) Handoff：慢相扣除也带上 seed（冻结事件期末值 ⇒ 交接无跳变）
    ("""    const Eigen::VectorXd share = ShareVector(ev_, v);
    // 当前逐通道扣除 = −share·c_applied(显示 = v + share·c)
    const Eigen::VectorXd ded_old = -(share * ev_.c_applied);""",
     """    const Eigen::VectorXd share = ShareVector(ev_, v);
    // 当前逐通道扣除 = −share·c_applied(显示 = v + share·c)
    Eigen::VectorXd ded_old = -(share * ev_.c_applied);
    // ── plan-v3.2 R1：seed 是「事件期的固定扣除」，必须一起交给慢相 ──
    // 否则交接当帧显示会跳 |seed|（plan-v3.0 漏 PCT 时踩过同一个坑，见 .cpp 的 ①② 说明）。
    // 用事件期**末帧**的 seed_used，慢相再按同一 share 施加 ⇒ 严格连续。
    if (ev_.seed_used > 0.0) {
        const double sw = share.sum();
        if (sw > 1e-12) ded_old += (share / sw) * (-ev_.seed_used);
    }"""),
    # 6) Handoff：A_new 扣掉种子
    ("""    Eigen::VectorXd A_new = (ev_.y0 + share * std::max(ev_.a_hat, 0.0)).cwiseMax(0.0);
    if (A_new.maxCoeff() <= 1e-9) A_new = ev_.y0.cwiseMax(0.0);
    A_ = A_new;
    const double amax = A_.maxCoeff();
    for (int k = 0; k < n_; ++k)
        loaded_[k] = (amax > 1e-9 && A_(k) > kLoadedFrac * amax) ? 1 : 0;
    // PCT 生效通道：比 loaded_ 宽（2% 而非 10%）。实测残漂的 96% 来自
    // 「加载时响应小、因此不在 loaded_ 里、却在保压期内持续蠕变」的通道。
    for (int k = 0; k < n_; ++k)
        pct_elig_[k] = (amax > 1e-9 && A_(k) > params_.pct_elig_frac * amax) ? 1 : 0;""",
     """    // ── plan-v3.2 R1：锚定不再等于"当前显示电平" ──
    // 原式 A_new = y0 + share·Â 隐含「交接时刻显示 = 无蠕变电平」。完全卸载后重载时该前提
    // 不成立：显示里已经含有本次加载长出的蠕变 ⇒ 锚定被抬高 ⇒ g_raw≈0 ⇒ 补偿恒 0。
    // 故从 Â 里扣掉 seed（本次加载已长出的蠕变量），把蠕变余量还给慢相。
    const double seed_amt = std::max(ev_.seed_used, 0.0);
    Eigen::VectorXd A_new =
        (ev_.y0 + share * std::max(ev_.a_hat - seed_amt, 0.0)).cwiseMax(0.0);
    if (A_new.maxCoeff() <= 1e-9) A_new = ev_.y0.cwiseMax(0.0);
    A_ = A_new;
    const double amax = A_.maxCoeff();
    for (int k = 0; k < n_; ++k)
        loaded_[k] = (amax > 1e-9 && A_(k) > kLoadedFrac * amax) ? 1 : 0;
    // PCT 生效通道：比 loaded_ 宽（2% 而非 10%）。实测残漂的 96% 来自
    // 「加载时响应小、因此不在 loaded_ 里、却在保压期内持续蠕变」的通道。
    for (int k = 0; k < n_; ++k)
        pct_elig_[k] = (amax > 1e-9 && A_(k) > params_.pct_elig_frac * amax) ? 1 : 0;
    // R1：seed 以逐通道扣除的形式继续存在于慢相（与事件期末帧一致，交接无跳变）
    if (seed_amt > 0.0 && pct_ded_.size() == n_) {
        const double sw = share.sum();
        if (sw > 1e-12) pct_ded_ -= (share / sw) * seed_amt;
    }"""),
    # 7) Process：每帧维护蠕变记忆（放在状态机之后，见 8) 的说明）
    ("""    if (ev_.valid) {
        Eigen::VectorXd out;
        if (RunEvent(timestamp_s, v, total, dt, eps, idle_now, &out)) {""",
     """    // plan-v3.2 R1：每帧维护会话级蠕变比。**必须在本帧状态机跑完之后**（否则拿到的是
    // 上一帧的 state_ / ev_，卸载或建事件的那一帧会记错"当前是否在慢相/事件"）。
    CreepSeedMaintain(timestamp_s, dt);

    if (ev_.valid) {
        Eigen::VectorXd out;
        if (RunEvent(timestamp_s, v, total, dt, eps, idle_now, &out)) {"""),
    # 8) Handoff 里 pct 清零必须在 seed 之前 ⇒ 调整顺序
    ("""    if (pct_ded_.size() == n_) pct_ded_.setZero();""",
     """    if (pct_ded_.size() == n_) pct_ded_.setZero();
    // R1 的 seed 紧接着重新写入 pct_ded_（顺序不能颠倒）"""),
    # 9) SlowStep 出口把 seed 一起施加
    ("""    Eigen::VectorXd out = v;
    for (int k = 0; k < n_; ++k) {
        if (A_(k) > 1e-9 && std::abs(ded(k)) > 1e-12)
            out(k) = v(k) - Capped(v(k), ded(k));
    }
    return out;
}""",
     """    // ── plan-v3.2 R1：事件期注入的蠕变种子继续按同一分配向量施加 ──
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
    Eigen::VectorXd out = v;
    for (int k = 0; k < n_; ++k) {
        if (std::abs(ded(k)) > 1e-12) out(k) = v(k) - Capped(v(k), ded(k));
    }
    return out;
}"""),
]

# 新函数的头文件声明
H_DECL_ANCHOR = """    // ── 逆模型 ──
    bool InvEst("""
H_DECL_NEW = """    // ── plan-v3.2 R1：会话级蠕变比记忆（重载蠕变接力）──
    void CreepSeedMaintain(double ts, double dt);

    // ── 逆模型 ──
    bool InvEst("""


def apply(text, patches, tag):
    for i, (old, new) in enumerate(patches, 1):
        if old not in text:
            raise SystemExit("[%s] 第 %d 处未命中（源码已变？）:\n%s" % (tag, i, old[:200]))
        if text.count(old) != 1:
            raise SystemExit("[%s] 第 %d 处命中 %d 次（应唯一）" % (tag, i, text.count(old)))
        text = text.replace(old, new, 1)
    return text


def main():
    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(SRC, "drift_v6_compensator.h"), encoding="utf-8") as fh:
        h = fh.read()
    with open(os.path.join(SRC, "drift_v6_compensator.cpp"), encoding="utf-8") as fh:
        c = fh.read()
    h = apply(h, H_PATCHES, "h")
    h = apply(h, [(H_DECL_ANCHOR, H_DECL_NEW)], "h-decl")
    c = apply(c, C_PATCHES, "cpp")
    c = c.replace("namespace drift_v6 {",
                  "// ===== plan-v3.2 R1 变体（由 make_v32_variant.py 从 src/ 生成，勿手改）=====\n"
                  "namespace drift_v6 {", 1)
    c = c.replace('#include <algorithm>\n#include <cmath>',
                  '#include <algorithm>\n#include <cmath>\n'
                  '#ifdef V32_DBG\n#include <cstdio>\n#include <cstdlib>\n#endif', 1)
    with open(os.path.join(DST, "drift_v6_compensator.h"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(h)
    with open(os.path.join(DST, "drift_v6_compensator.cpp"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(c)
    print("-> %s" % DST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
