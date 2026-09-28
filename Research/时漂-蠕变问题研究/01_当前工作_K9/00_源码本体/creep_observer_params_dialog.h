// Copyright (c) 2025 Modulus Technology. All rights reserved.
//
// 文件: src/ui/dialogs/creep_observer_params_dialog.h
// 描述: 「时漂/零漂补偿（v3.4 观测器）」的现场参数对话框
//       （非模态，跟随补偿算法选择对话框打开）。暴露 src/domain/drift_v6/creep_observer.h
//       的 CreepObserverCompensator::Params 中 8 个现场可调参数，每项配「?」说明。
//       改动实时生效（不重置正在运行的观测器状态），「恢复默认参数」一键回到算法默认值。
//       本对话框不含算法逻辑，只做取值、校验与投递。
//       8 项全部可下发：使能位 + 8 参数对应标准 Modbus 寄存器 200~208
//       （201 r_fast / 202 slow_confirm_s / 203 soft_unfreeze_s /
//        204 tau_r_slow_idle_s / 205 tau_r_fast_s / 206 tau_c_fast_s /
//        207 slope_cap_frac / 208 r_slow_max；slope_cap_frac ×1000 定点，
//        其余 ×100）。

#pragma once
#include <QDialog>
#include <QJsonObject>
#include <QVector>

class QDoubleSpinBox;
class QLabel;
class QCheckBox;
class QPushButton;

class CreepObserverParamsDialog : public QDialog {
    Q_OBJECT
public:
    explicit CreepObserverParamsDialog(QWidget* parent = nullptr);

    // 现场可调参数个数（与 Params 的 8 个可调成员一一对应）。
    static int ParamCount();

    // 算法默认值（取自 CreepObserverCompensator::Params 的默认构造，单一来源）。
    static QJsonObject DefaultValues();

    // 按 JSON 覆盖控件值：只处理出现的键，缺键保留当前值（首次打开即算法默认值）。
    void SetValues(const QJsonObject& values);
    QJsonObject Values() const;

    // 观测器算法的启用状态（与其他补偿算法互斥；
    // 由 controller 同步进来，也随勾选框发出 EnabledToggled 让 controller 走唯一开关点）。
    void SetAlgorithmEnabled(bool enabled);
    bool AlgorithmEnabled() const;

    // 参数下发链路的状态回显：busy=true 时禁用「写入设备」按钮；
    // text 为空表示回到空闲态。结果文案（成功/失败原因）也走这里。
    void SetWriteState(bool busy, const QString& text);

signals:
    // 任一参数被改动（实时）；顺序与 Values() / kSpecs 的键顺序一致
    // （末位 = tau_r_slow_idle_s，2026-09 调参批次新增的现场可调项）。
    void ParamsChanged(double r_fast, double tau_c_fast_s, double slow_confirm_s,
                       double soft_unfreeze_s, double slope_cap_frac, double r_slow_max,
                       double tau_r_fast_s, double tau_r_slow_idle_s);

    // 勾选框切换观测器算法的启用/停用（controller 走 ApplyCompensationAlgorithm）。
    void EnabledToggled(bool enabled);

    // 「写入设备」按钮：请求把使能位 + 8 个参数（寄存器 200~208）下发到设备
    // （完整回执链路在 controller / DeviceConnector 侧）。
    void WriteToDeviceRequested();

private slots:
    void OnSpinChanged();
    void OnRestoreDefaults();
    void OnWriteToDevice();

private:
    QDoubleSpinBox* AddRow(class QGridLayout* grid, int row, int spec_index);
    void EmitChanged();

    QVector<QDoubleSpinBox*> spins_;
    QCheckBox* enable_check_ = nullptr;
    QPushButton* write_button_ = nullptr;
    QLabel* state_label_ = nullptr;
    QLabel* write_state_label_ = nullptr;
    bool updating_ = false;
    bool write_busy_ = false;
};
