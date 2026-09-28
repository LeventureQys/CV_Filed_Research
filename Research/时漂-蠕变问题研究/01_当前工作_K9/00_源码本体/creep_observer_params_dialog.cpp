// Copyright (c) 2025 Modulus Technology. All rights reserved.
//
// 文件: src/ui/dialogs/creep_observer_params_dialog.cpp
// 描述: 见同名头文件。参数表（键名/取值范围/步长/说明）集中在下方的 kSpecs，
//       默认值一律从 CreepObserverCompensator::Params 的默认构造读取，
//       保证「算法默认值」只有源码一处来源（改域层默认值本对话框自动跟随）。

#include "creep_observer_params_dialog.h"

#include "domain/drift_v6/creep_observer.h"

#include <QDialogButtonBox>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFont>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>

#include <array>
#include <cstring>

namespace {

using Params = drift_v6::CreepObserverCompensator::Params;

struct ParamSpec {
    const char* key;                        // user_settings / 录制 manifest 的键名
    const char* title;                      // 显示名
    double Params::* field;                 // 对应的 Params 成员（默认值来源 + 写回目标）
    double min_value;
    double max_value;
    double step;
    int decimals;
    const char* suffix;                     // 单位后缀（可为空）
    const char* special_zero;               // 取最小值时的替换文案（可为空）
    const char* help;
};

const char* const kRFastHelp =
    "快态（x1）的幅度比：加载后快态终值 = r_fast × 弹性残差 e，闭环下实际吃掉载荷的 "
    "r_fast/(1+r_fast)。\n\n"
    "调大：恒载段更快扣掉快相爬升，但卸载/快速回撤后残留的补偿更多，显示回落（下漂）更久；\n"
    "调小：过补偿水位降低（快速回撤后显示更稳），代价是加载后数秒内的上漂抑制变慢。\n\n"
    "现场调参结论 0.10 → 0.06 → 0.01（长尾下漂水位进一步下降）。默认 0.01；\n"
    "0 = 关闭快态补偿（x1 恒为 0，只有慢态 x2 在工作）。";

const char* const kSlowConfirmHelp =
    "慢态（x2）积分前的「连续受载确认」时长：逐通道的连续受载计时达到该值后，x2 才允许积分；"
    "期间任何一次快变（升或降）都会把它清零重计。\n\n"
    "调大：慢态启动更晚、长保压前段欠扣更多，但 4~6 s 的短保载不会被误吸收成慢漂移；\n"
    "调小：慢态启动提前、长保压更早进入补偿，代价是短保载工况的逐循环棘轮（补偿越扣越多）风险上升。\n\n"
    "x2 的最早启动时刻 ≈ 该值与「沿后软冻结窗」取大者。现场调参结论 10 → 5 s。\n"
    "默认 5 s；0 = 不设确认门槛（计时 ≥ 0 恒成立）。";

const char* const kSoftUnfreezeHelp =
    "慢态（x2）的沿后软冻结窗：加载沿之后要等到 t_edge > 该值 × ln2（≈ 0.69 × 该值）"
    "才允许 x2 积分。\n\n"
    "作用是沿后的斜率被台阶瞬态抬高时不许进 x2（防沿后过速积分造成的「峰后回落」）。\n"
    "调大：沿后更久不积慢态，回落更小但慢态启动更晚；\n"
    "调小：与「受载确认窗」一起决定 x2 能多早开始扣蠕变。\n\n"
    "现场调参结论 4 → 8 s（解冻点 ≈ 8×0.69 ≈ 5.5 s）。默认 8 s；必须 > 0。";

const char* const kTauCFastHelp =
    "快态（x1）受载收敛时间常数：受载（e > 0）时 x1 朝 r_fast × e 收敛的一阶 τ，也就是"
    "「加载后头几秒上漂被压多快」的直接旋钮。\n\n"
    "调大：快相补偿更保守、沿后回落更小，但加载后数秒内的上漂抑制更慢；\n"
    "调小：收敛更快，加载段与后续段的残余上漂更小，代价是卸载/快速回撤后 x1 残留撤得更急、"
    "显示瞬态更陡。\n\n"
    "注意：上升沿后 edge_boost_s（默认 2 s）内以及缓坡前馈满窗后，实际用的是 "
    "tau_c_fast_boost_s（2 s），本项决定前馈窗之外（长保压/后续段）的收敛速度。\n"
    "默认 40 s。本项随「写入设备」下发到寄存器 206（×100 定点）。";

const char* const kSlopeCapHelp =
    "慢态（x2）积分速率上限：每帧 x2 的增量被限制在 ±该值 × max(e, 1) 以内（单位 1/s）。\n\n"
    "x2 是积分器、没有自己的时间常数，从 0 涨到上限 r_slow_max × e 的等效时间 ≈ "
    "r_slow_max / 该值：默认 0.2 / 0.05 = 4 s —— 后续段（第二段及以后）的爬升由它决定。\n"
    "调大：后续段的补偿更快更足（实测 0.01 → 0.015/0.02 可显著降低第二段的残余上漂），"
    "代价是台阶/加载沿处容易过速积分，出现「峰后回落」与长保压过补偿下漂；\n"
    "调小：沿处更稳，后续段欠扣更多。\n\n"
    "0 = 关闭慢态积分（只剩快态 x1，补偿幅度很小）。\n"
    "本项按显示单位判定：显示 ADC（e ≫ 1）时按弹性电平的比例生效；显示力值（N，e ≪ 1）时 "
    "max(e,1) 恒为 1，上限退化成绝对值 该值 × 1（按 N 计），同一数值在两种显示模式下的"
    "实际强度不同。\n"
    "默认 0.05（5 %/s）。本项随「写入设备」下发到寄存器 207（×1000 定点，量化步 0.001，"
    "与界面步长一致）。";

const char* const kTauRSlowIdleHelp =
    "空载期慢态（x2）的泄放时间常数：判为空载（去趋势空载门内）时，x2 每帧按 x2 -= dt/τ × x2 "
    "快速衰减。\n\n"
    "它决定「卸载后残留的蠕变补偿清空多快」，也就是「同一负载反复增减时，上一轮的补偿会不会带到"
    "下一轮」——该值过大时每循环都会留下一点，逐循环累加（棘轮）会把空载显示越压越低。\n"
    "调大：卸载后残留补偿滞留更久（更接近「保持蠕变记忆」）；\n"
    "调小：卸载后显示更快回到原始读数，重载后重新起算。\n\n"
    "现场调参结论 8 → 2 → 0.5 s。默认 0.5 s；填 0 = 关闭该分支，回落到 150 s 的慢态恢复时间常数"
    "（仅在非空载且 e ≤ 0 时使用）。";

const char* const kTauRFastHelp =
    "快态（x1）的空载恢复时间常数：判为无载荷（e ≤ 0）时，x1 每帧按 x1 -= dt/τ × x1 泄放。\n\n"
    "它决定卸载后「快相补偿」撤掉多快，与上面的空载慢态泄放一起决定下一次加载时还剩多少残留补偿：\n"
    "调大：卸载后快态补偿滞留更久（短时间内重载读数更连续，但卸载后显示回落更慢）；\n"
    "调小：卸载后显示更快回到原始读数。\n\n"
    "现场调参结论 0.5 → 6 s。默认 6 s；必须 > 0（填 0 会在 e ≤ 0 分支产生 0/0）。";

const char* const kRSlowMaxHelp =
    "慢态（x2）幅度上限：x2 被钳在 该值 × 弹性残差 e 以内（单位：1，即相对载荷的比例）。\n\n"
    "它是**唯一**能让慢态真正收敛的旋钮：x2 是积分器，靠斜率估计驱动，平台期斜率估计的残余正偏"
    "会一直被积进去（实测 142 s 保压：载荷已平，x2 仍以输入的约 2 倍速率增长，显示 −3.5 ADC/s 持续下漂，"
    "10 分钟可达 −6 %）。把上限收紧后，x2 涨到上限即停，实测末段显示斜率由 −3.55 收到 −0.02 ADC/s。\n"
    "调大：能扣掉更多蠕变（适合蠕变占比大的传感器），但平台期持续下漂的风险回升；\n"
    "调小：平台期更稳（宁欠扣不过扣），代价是蠕变扣不完、显示偏高。\n\n"
    "取值建议：≈ 该传感器「慢漂占比」+ 一点余量。例如慢漂约 6 % 载荷 ⇒ 0.06~0.08。\n"
    "填 0 = 关闭慢态（x2 被钳死在 0，只靠快态与其它机制工作）。\n"
    "默认 0.2。本项随「写入设备」下发到寄存器 208（×100 定点）。";

const std::array<ParamSpec, 8> kSpecs = {{
    {"r_fast", "快态幅度比 r_fast", &Params::r_fast, 0.0, 0.500, 0.01, 3, "",
     "0（关闭快态）", kRFastHelp},
    {"tau_c_fast_s", "快态收敛 τ tau_c_fast_s", &Params::tau_c_fast_s, 0.5, 120.0, 0.5, 1,
     " s", nullptr, kTauCFastHelp},
    {"slow_confirm_s", "慢态受载确认窗 slow_confirm_s", &Params::slow_confirm_s, 0.0, 60.0, 0.5,
     2, " s", nullptr, kSlowConfirmHelp},
    {"soft_unfreeze_s", "慢态沿后软冻结窗 soft_unfreeze_s", &Params::soft_unfreeze_s, 0.1, 60.0,
     0.5, 2, " s", nullptr, kSoftUnfreezeHelp},
    {"slope_cap_frac", "慢态积分速率上限 slope_cap_frac", &Params::slope_cap_frac, 0.0, 0.050,
     0.001, 3, " 1/s", "0（关闭慢态）", kSlopeCapHelp},
    {"r_slow_max", "慢态幅度上限 r_slow_max", &Params::r_slow_max, 0.0, 0.600, 0.01, 3, "",
     "0（关闭慢态）", kRSlowMaxHelp},
    {"tau_r_fast_s", "空载快态恢复 τ tau_r_fast_s", &Params::tau_r_fast_s, 0.05, 300.0, 0.05, 2,
     " s", nullptr, kTauRFastHelp},
    {"tau_r_slow_idle_s", "空载慢态泄放 τ tau_r_slow_idle_s", &Params::tau_r_slow_idle_s, 0.0,
     300.0, 0.5, 2, " s", "0（回落 150 s）", kTauRSlowIdleHelp},
}};

const std::array<ParamSpec, 8>& Specs() { return kSpecs; }

// 设备侧有寄存器（200~208）的参数：全部 8 项均随「写入设备」下发。
constexpr const char* const kDeviceParamKeys[] = {
    "r_fast", "slow_confirm_s", "soft_unfreeze_s", "tau_r_slow_idle_s", "tau_r_fast_s",
    "tau_c_fast_s", "slope_cap_frac", "r_slow_max"};

bool IsDeviceParam(const char* key) {
    for (const char* k : kDeviceParamKeys)
        if (std::strcmp(k, key) == 0) return true;
    return false;
}

const Params& DefaultParams() {
    static const Params kDefaults{};
    return kDefaults;
}

}  // namespace

int CreepObserverParamsDialog::ParamCount() { return static_cast<int>(kSpecs.size()); }

QJsonObject CreepObserverParamsDialog::DefaultValues() {
    const Params& d = DefaultParams();
    QJsonObject obj;
    for (const auto& spec : Specs()) obj[QString::fromLatin1(spec.key)] = d.*(spec.field);
    return obj;
}

CreepObserverParamsDialog::CreepObserverParamsDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(QString::fromUtf8("观测器参数（v3.4，实时调整）"));
    setMinimumWidth(760);

    auto* root = new QVBoxLayout(this);

    auto* hint = new QLabel(
        QString::fromUtf8(
            "这 8 个参数直接作用于正在运行的观测器：改动立即生效，并且不重置观测器状态，"
            "因此可以一边改一边在显示上观察效果。\n"
            "参数随 user_settings.json 记住，重启软件后继续生效；「恢复默认参数」一键回到算法默认值。\n"
            "点每行右侧的「?」查看该参数的作用与调大/调小的代价；"
            "全部 8 个参数都随「写入设备」下发到寄存器 201~208（+ 使能位 200）："
            "slope_cap_frac 按 ×1000 定点，其余 7 项按 ×100 定点。"),
        this);
    hint->setWordWrap(true);
    hint->setStyleSheet(QStringLiteral("QLabel { color: #7ec8e3; }"));
    root->addWidget(hint);

    enable_check_ = new QCheckBox(
        QString::fromUtf8("启用观测器（v3.4 档，与其他补偿算法互斥，同一时刻只生效一种）"),
        this);
    connect(enable_check_, &QCheckBox::toggled, this, [this](bool checked) {
        if (!updating_) emit EnabledToggled(checked);
    });
    root->addWidget(enable_check_);

    auto* grid = new QGridLayout();
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setHorizontalSpacing(8);
    grid->setVerticalSpacing(6);
    grid->setColumnStretch(4, 1);
    root->addLayout(grid);

    for (int i = 0; i < ParamCount(); ++i) AddRow(grid, i, i);

    state_label_ = new QLabel(this);
    state_label_->setWordWrap(true);
    root->addWidget(state_label_);

    write_state_label_ = new QLabel(this);
    write_state_label_->setWordWrap(true);
    write_state_label_->setStyleSheet(QStringLiteral("QLabel { color: #8a8a8a; }"));
    root->addWidget(write_state_label_);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    write_button_ = buttons->addButton(QString::fromUtf8("写入设备"),
                                       QDialogButtonBox::ActionRole);
    write_button_->setToolTip(QString::fromUtf8(
        "把算法使能与 8 个参数下发到已连接的设备（标准 Modbus 寄存器 200~208："
        "200=算法使能，201=r_fast，202=slow_confirm_s，203=soft_unfreeze_s，"
        "204=tau_r_slow_idle_s，205=tau_r_fast_s，206=tau_c_fast_s，207=slope_cap_frac，"
        "208=r_slow_max）："
        "先停止数据自动上传，逐寄存器写入、每帧等到设备回执确认后才发下一帧，"
        "完成后（无论成败）自动恢复数据自动上传。\n"
        "某个寄存器 1 秒内没有回执（设备无应答或返回 Modbus 异常码）只会跳过它、继续写后面的，"
        "结束时逐条列出未确认的项，不影响其余参数。\n"
        "定点量化：slope_cap_frac 按 ×1000（量化步 0.001），其余 7 项按 ×100。"));
    connect(write_button_, &QPushButton::clicked, this,
            &CreepObserverParamsDialog::OnWriteToDevice);
    auto* restore = buttons->addButton(QString::fromUtf8("恢复默认参数"),
                                       QDialogButtonBox::ResetRole);
    restore->setToolTip(QString::fromUtf8(
        "回到算法默认值（r_fast=0.01、tau_c_fast_s=40、slow_confirm_s=5、soft_unfreeze_s=8、"
        "slope_cap_frac=0.05、r_slow_max=0.2、tau_r_fast_s=6、tau_r_slow_idle_s=0.5），"
        "立即生效并覆盖已记住的设置。"));
    connect(restore, &QPushButton::clicked, this, &CreepObserverParamsDialog::OnRestoreDefaults);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::close);
    root->addWidget(buttons);

    SetValues(DefaultValues());
}

QDoubleSpinBox* CreepObserverParamsDialog::AddRow(QGridLayout* grid, int row, int spec_index) {
    const ParamSpec& spec = Specs().at(static_cast<std::size_t>(spec_index));

    auto* title = new QLabel(QString::fromUtf8(spec.title), this);
    title->setMinimumWidth(250);
    grid->addWidget(title, row, 0);

    auto* spin = new QDoubleSpinBox(this);
    spin->setRange(spec.min_value, spec.max_value);
    spin->setSingleStep(spec.step);
    spin->setDecimals(spec.decimals);
    if (spec.suffix[0] != '\0') spin->setSuffix(QString::fromUtf8(spec.suffix));
    if (spec.special_zero) spin->setSpecialValueText(QString::fromUtf8(spec.special_zero));
    spin->setMinimumWidth(140);
    // 键盘输入按「编辑完成/回车」提交，方向键与滚轮仍逐档实时生效。
    spin->setKeyboardTracking(false);
    spin->setToolTip(QString::fromUtf8(spec.help));
    connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &CreepObserverParamsDialog::OnSpinChanged);
    grid->addWidget(spin, row, 1);
    spins_.append(spin);

    auto* help = new QToolButton(this);
    help->setText(QStringLiteral("?"));
    // 带边框的实体按钮（不 autoRaise），在任何配色下都能看出是个可点的「?」。
    help->setAutoRaise(false);
    help->setFixedSize(26, 26);
    {
        QFont font = help->font();
        font.setBold(true);
        help->setFont(font);
    }
    help->setToolTip(QString::fromUtf8(spec.help));
    help->setCursor(Qt::WhatsThisCursor);
    const QString help_title = QString::fromUtf8(spec.title);
    const QString help_text = QString::fromUtf8(spec.help);
    connect(help, &QToolButton::clicked, this, [this, help_title, help_text]() {
        QMessageBox::information(this, help_title, help_text);
    });
    grid->addWidget(help, row, 2);

    auto* defaults = new QLabel(
        IsDeviceParam(spec.key)
            ? QString::fromUtf8("默认 %1").arg(DefaultParams().*(spec.field), 0, 'g', 6)
            : QString::fromUtf8("默认 %1 ｜ 仅本机").arg(DefaultParams().*(spec.field), 0, 'g', 6),
        this);
    defaults->setStyleSheet(QStringLiteral("QLabel { color: #8a8a8a; }"));
    grid->addWidget(defaults, row, 3);

    return spin;
}

void CreepObserverParamsDialog::SetValues(const QJsonObject& values) {
    updating_ = true;
    for (int i = 0; i < spins_.size() && i < ParamCount(); ++i) {
        const ParamSpec& spec = Specs().at(static_cast<std::size_t>(i));
        const QString key = QString::fromLatin1(spec.key);
        if (!values.contains(key)) continue;
        const double v = values.value(key).toDouble(DefaultParams().*(spec.field));
        spins_.at(i)->setValue(v);
    }
    updating_ = false;
    EmitChanged();
}

QJsonObject CreepObserverParamsDialog::Values() const {
    QJsonObject obj;
    for (int i = 0; i < spins_.size() && i < ParamCount(); ++i)
        obj[QString::fromLatin1(Specs().at(static_cast<std::size_t>(i)).key)] =
            spins_.at(i)->value();
    return obj;
}

void CreepObserverParamsDialog::OnSpinChanged() {
    if (updating_) return;
    EmitChanged();
}

void CreepObserverParamsDialog::OnRestoreDefaults() {
    SetValues(DefaultValues());
}

void CreepObserverParamsDialog::SetAlgorithmEnabled(bool enabled) {
    updating_ = true;
    enable_check_->setChecked(enabled);
    updating_ = false;
}

bool CreepObserverParamsDialog::AlgorithmEnabled() const {
    return enable_check_->isChecked();
}

void CreepObserverParamsDialog::SetWriteState(bool busy, const QString& text) {
    write_busy_ = busy;
    if (write_button_) write_button_->setEnabled(!busy);
    if (write_state_label_) {
        write_state_label_->setText(text.isEmpty()
                                        ? QString::fromUtf8("参数尚未下发到设备（当前仅在本机生效）。")
                                        : text);
        write_state_label_->setStyleSheet(
            text.isEmpty() ? QStringLiteral("QLabel { color: #8a8a8a; }")
                           : QStringLiteral("QLabel { color: #e3a008; }"));
    }
}

void CreepObserverParamsDialog::OnWriteToDevice() {
    if (write_busy_) return;
    SetWriteState(true, QString::fromUtf8(
                          "正在下发：停止自动上传 → 逐帧写入寄存器 200~208（使能 + 8 个参数，"
                          "每帧等回执）；未回执的寄存器自动跳过、不影响其余参数，完成后逐条列出..."));
    emit WriteToDeviceRequested();
}

void CreepObserverParamsDialog::EmitChanged() {
    if (spins_.size() < ParamCount()) return;
    const QJsonObject current = Values();
    if (state_label_) {
        const QJsonObject defaults = DefaultValues();
        QStringList drifted;
        for (const auto& spec : Specs()) {
            const QString key = QString::fromLatin1(spec.key);
            if (current.value(key).toDouble() != defaults.value(key).toDouble())
                drifted << QString::fromLatin1(spec.key);
        }
        state_label_->setText(drifted.isEmpty()
                                  ? QString::fromUtf8("当前 8 个参数均为算法默认值。")
                                  : QString::fromUtf8("已偏离默认值：%1（可用「恢复默认参数」还原）")
                                        .arg(drifted.join(QStringLiteral("、"))));
        state_label_->setStyleSheet(drifted.isEmpty()
                                        ? QStringLiteral("QLabel { color: #8a8a8a; }")
                                        : QStringLiteral("QLabel { color: #e3a008; }"));
    }
    // 顺序与 kSpecs / Values() 一致：r_fast、tau_c_fast_s、slow_confirm_s、soft_unfreeze_s、
    // slope_cap_frac、r_slow_max、tau_r_fast_s、tau_r_slow_idle_s。
    emit ParamsChanged(current.value(QStringLiteral("r_fast")).toDouble(),
                       current.value(QStringLiteral("tau_c_fast_s")).toDouble(),
                       current.value(QStringLiteral("slow_confirm_s")).toDouble(),
                       current.value(QStringLiteral("soft_unfreeze_s")).toDouble(),
                       current.value(QStringLiteral("slope_cap_frac")).toDouble(),
                       current.value(QStringLiteral("r_slow_max")).toDouble(),
                       current.value(QStringLiteral("tau_r_fast_s")).toDouble(),
                       current.value(QStringLiteral("tau_r_slow_idle_s")).toDouble());
}
