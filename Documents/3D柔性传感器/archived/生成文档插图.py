import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)
RESEARCH_DIR = ROOT / "research"

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle,
    facecolor,
    edgecolor,
    title_size=15,
    subtitle_size=10.5,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#172033",
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color="#445066",
        linespacing=1.35,
        zorder=3,
    )
    return box


def add_arrow(ax, start, end, color="#60708A", width=2.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=width,
        color=color,
        shrinkA=4,
        shrinkB=4,
        zorder=1,
    )
    ax.add_patch(arrow)


def add_arrow_label(ax, x, y, title, subtitle):
    ax.text(
        x,
        y + 0.012,
        title,
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#3D4A61",
        zorder=3,
    )
    ax.text(
        x,
        y - 0.012,
        subtitle,
        ha="center",
        va="top",
        fontsize=8.8,
        color="#657188",
        zorder=3,
    )


def load_session_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    header_index = next(index for index, row in enumerate(rows) if row and row[0] == "timestamp")
    header = rows[header_index]
    elapsed_index = header.index("elapsed")
    channel_indices = [index for index, name in enumerate(header) if name.startswith("ch")]
    elapsed = []
    values = []
    for row in rows[header_index + 1 :]:
        if len(row) < len(header):
            continue
        elapsed.append(float(row[elapsed_index]))
        values.append([float(row[index]) for index in channel_indices])
    return np.asarray(elapsed), np.asarray(values)


def load_legacy_csv(path, sample_rate_hz=10.0):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    header_index = next(index for index, row in enumerate(rows) if row and row[0] == "时间戳")
    header = rows[header_index]
    total_index = header.index("总值")
    totals = []
    for row in rows[header_index + 1 :]:
        if len(row) <= total_index:
            continue
        totals.append(float(row[total_index]))
    elapsed = np.arange(len(totals), dtype=float) / sample_rate_hz
    return elapsed, np.asarray(totals)


def moving_average(values, window):
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def draw_layered_model():
    fig, ax = plt.subplots(figsize=(14, 18), dpi=160)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "柔性压感传感器分层模型",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#172033",
    )
    ax.text(
        0.5,
        0.935,
        "从真实载荷到产品输出：每层只处理一种可解释误差",
        ha="center",
        va="center",
        fontsize=12,
        color="#657188",
    )

    states = [
        ("真实载荷", "总力 F(t) 与空间压力分布 p(x, y, t)", "#E8F1FF", "#3978C5"),
        ("理想 Cell 受力", "各 Cell 的等效局部载荷 x_i(t)", "#F0F8E8", "#6A9B35"),
        ("电学量", "电阻、电导或电容响应 q_i(t)", "#FFF0E7", "#C96C32"),
        ("原始 ADC", "原始观测 y_i(t)，保留时间戳与质量标志", "#F4ECFA", "#8756B3"),
        ("校正响应", "完成可解释校正后的响应 r_i(t)", "#EDEBFA", "#6658B6"),
        ("测量结果", "压力图、接触特征与总力估计值 F_est(t)", "#E9F2FA", "#3E7AA6"),
        ("产品输出", "稳定测量、质量状态与独立显示分支", "#E6F5F5", "#23888B"),
    ]
    transforms = [
        ("接触与结构传播 H_mech", "面积 · 压头 · 装夹边界"),
        ("材料静态非线性 f_i + 黏弹状态 z_i", "迟滞 · 蠕变 · 恢复"),
        ("阵列网络 A + 采集电路 g", "串扰 · 激励 · 复用 · 放大 · ADC"),
        ("基线 / 共模 / 坏点 / 串扰校正", "仅处理有依据、可验证的误差"),
        ("Cell 标定 + 空间积分", "ADC → 压力 / 力 / 接触特征"),
        ("因果稳定化 + 显示分流", "测量流保真，显示流可独立平滑"),
    ]

    x = 0.15
    width = 0.70
    height = 0.066
    top = 0.845
    step = 0.125

    for index, (title, subtitle, facecolor, edgecolor) in enumerate(states):
        y = top - index * step
        add_box(
            ax,
            x,
            y,
            width,
            height,
            title,
            subtitle,
            facecolor,
            edgecolor,
            title_size=15.5,
            subtitle_size=10.5,
        )
        if index < len(states) - 1:
            next_y = top - (index + 1) * step
            arrow_top = y - 0.006
            arrow_bottom = next_y + height + 0.006
            add_arrow(ax, (0.5, arrow_top), (0.5, arrow_bottom), width=2.2)
            add_arrow_label(
                ax,
                0.68,
                (arrow_top + arrow_bottom) / 2,
                transforms[index][0],
                transforms[index][1],
            )

    ax.text(
        0.5,
        0.035,
        "原则：保留原始数据 · 分层辨识 · 校正与显示分流 · 所有处理可回放",
        ha="center",
        va="center",
        fontsize=11,
        color="#526078",
    )

    fig.savefig(
        ASSET_DIR / "分层模型.png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def draw_product_architecture():
    fig, ax = plt.subplots(figsize=(18, 11), dpi=160)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "柔性压感系统推荐产品架构",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#172033",
    )
    ax.text(
        0.5,
        0.895,
        "采集、测量与显示分层；原始数据始终可获取",
        ha="center",
        va="center",
        fontsize=12,
        color="#657188",
    )

    add_box(
        ax,
        0.04,
        0.38,
        0.24,
        0.36,
        "FPC / 采集器",
        "原始有符号采样\n参考通道与时间戳\n板级 offset / gain 校准\n健康状态与可选轻量抗干扰",
        "#E8F1FF",
        "#3978C5",
    )

    add_box(
        ax,
        0.365,
        0.36,
        0.29,
        0.40,
        "上位机测量管线",
        "设备模式选择\n基线 / 共模 / 坏点校正\n有模型时进行空间串扰校正\nCell 与总力标定、迟滞状态\n因果时间稳定化",
        "#EAF7F2",
        "#27896A",
    )

    add_box(
        ax,
        0.745,
        0.57,
        0.22,
        0.25,
        "记录与业务输出",
        "原始 ADC\n校正响应与力值\n质量标志、参数版本\n可复现数据记录",
        "#FFF6DF",
        "#C58A18",
    )

    add_box(
        ax,
        0.745,
        0.20,
        0.22,
        0.27,
        "上位机显示管线",
        "从校正测量流分支\n显示死区与迟滞\n连通域与热图视觉处理\n可配置平滑，不反写测量数据",
        "#F4ECFA",
        "#8756B3",
    )

    add_arrow(ax, (0.28, 0.56), (0.365, 0.56), color="#3978C5", width=2.5)
    add_arrow(ax, (0.655, 0.62), (0.745, 0.695), color="#27896A", width=2.5)
    add_arrow(ax, (0.655, 0.48), (0.745, 0.335), color="#8756B3", width=2.5)

    ax.text(
        0.33,
        0.59,
        "原始帧 + 元数据",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#526078",
    )
    ax.text(
        0.70,
        0.70,
        "测量结果",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#526078",
    )
    ax.text(
        0.69,
        0.375,
        "显示分支",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#526078",
    )

    principle = FancyBboxPatch(
        (0.11, 0.075),
        0.78,
        0.075,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#A8B2C4",
        facecolor="#FFFFFF",
    )
    ax.add_patch(principle)
    ax.text(
        0.5,
        0.112,
        "架构原则：不可逆处理可绕过 · 测量与显示不互相污染 · 参数和模型均可追溯",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#445066",
    )

    fig.savefig(
        ASSET_DIR / "产品架构.png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def draw_host_algorithm_pipeline():
    fig, ax = plt.subplots(figsize=(18, 12), dpi=160)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "上位机核心算法管线",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#172033",
    )
    ax.text(
        0.5,
        0.895,
        "从可信原始帧到测量结果，再分支到显示处理",
        ha="center",
        va="center",
        fontsize=12,
        color="#657188",
    )

    stages = [
        ("01", "原始帧校验", "帧序号 · 时间戳\n完整性与采样率", "#E8F1FF", "#3978C5"),
        ("02", "设备 / 模式识别", "传感器类型 · signed\npositive_bias · rectified", "#E8F1FF", "#3978C5"),
        ("03", "基线与共模校正", "空载基线 · 慢漂移\n帧级共模", "#EAF7F2", "#27896A"),
        ("04", "坏点处理", "断路 · 短路 · 卡死\n饱和与质量标志", "#EAF7F2", "#27896A"),
        ("05", "空间响应校正", "仅对已有模型的设备\n补偿串扰与扩散", "#FFF6DF", "#C58A18"),
        ("06", "ADC → 力 / 压力", "Cell 非线性标定\n迟滞与状态补偿", "#FFF6DF", "#C58A18"),
        ("07", "特征与总力计算", "压力图 · 接触区域\n质心 · 面积 · 总力", "#F4ECFA", "#8756B3"),
        ("08", "因果时间稳定化", "EMA / IIR 可配置\n保持实时性与峰值", "#F4ECFA", "#8756B3"),
        ("09", "显示专用处理", "死区 · 迟滞 · 连通域\n热图插值与视觉平滑", "#E6F5F5", "#23888B"),
    ]

    positions = [
        (0.06, 0.64),
        (0.37, 0.64),
        (0.68, 0.64),
        (0.68, 0.35),
        (0.37, 0.35),
        (0.06, 0.35),
        (0.06, 0.06),
        (0.37, 0.06),
        (0.68, 0.06),
    ]
    box_width = 0.25
    box_height = 0.19

    for (number, title, subtitle, facecolor, edgecolor), (x, y) in zip(stages, positions):
        add_box(
            ax,
            x,
            y,
            box_width,
            box_height,
            title,
            subtitle,
            facecolor,
            edgecolor,
            title_size=14,
            subtitle_size=9.6,
        )
        ax.text(
            x + 0.018,
            y + box_height - 0.025,
            number,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=edgecolor,
            zorder=4,
        )

    horizontal_arrows = [
        ((0.31, 0.735), (0.37, 0.735)),
        ((0.62, 0.735), (0.68, 0.735)),
        ((0.68, 0.445), (0.62, 0.445)),
        ((0.37, 0.445), (0.31, 0.445)),
        ((0.31, 0.155), (0.37, 0.155)),
        ((0.62, 0.155), (0.68, 0.155)),
    ]
    vertical_arrows = [
        ((0.805, 0.64), (0.805, 0.54)),
        ((0.185, 0.35), (0.185, 0.25)),
    ]
    for start, end in horizontal_arrows + vertical_arrows:
        add_arrow(ax, start, end, width=2.3)

    fig.savefig(
        ASSET_DIR / "上位机核心算法.png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def draw_baseline_state_machine():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=160)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.93,
        "基线与慢漂移状态机",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#172033",
    )
    ax.text(
        0.5,
        0.875,
        "只在确定空载时学习基线，持续受压期间必须冻结",
        ha="center",
        va="center",
        fontsize=12,
        color="#657188",
    )

    states = [
        (
            0.07,
            0.43,
            "空载跟踪",
            "确认无接触\n慢速更新 baseline\n鲁棒 EMA / 分位数估计",
            "#EAF7F2",
            "#27896A",
        ),
        (
            0.39,
            0.43,
            "接触冻结",
            "检测到有效接触\n冻结受力 Cell 的 baseline\n继续保留原始响应",
            "#FFF6DF",
            "#C58A18",
        ),
        (
            0.71,
            0.43,
            "释放确认",
            "检测释放并等待稳定\n检查残余响应与坏点\n满足延迟后恢复更新",
            "#F4ECFA",
            "#8756B3",
        ),
    ]

    for x, y, title, subtitle, facecolor, edgecolor in states:
        add_box(
            ax,
            x,
            y,
            0.22,
            0.25,
            title,
            subtitle,
            facecolor,
            edgecolor,
            title_size=15,
            subtitle_size=10,
        )

    add_arrow(ax, (0.29, 0.555), (0.39, 0.555), color="#C58A18", width=2.5)
    add_arrow(ax, (0.61, 0.555), (0.71, 0.555), color="#8756B3", width=2.5)
    add_arrow(ax, (0.82, 0.43), (0.18, 0.43), color="#27896A", width=2.5)

    ax.text(0.34, 0.59, "检测到接触", ha="center", fontsize=10.5, color="#526078")
    ax.text(0.66, 0.59, "检测到释放", ha="center", fontsize=10.5, color="#526078")
    ax.text(0.50, 0.37, "稳定时间满足 → 延迟恢复基线更新", ha="center", fontsize=10.5, color="#526078")

    warning = FancyBboxPatch(
        (0.14, 0.12),
        0.72,
        0.11,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#C44D61",
        facecolor="#FCEAEC",
    )
    ax.add_patch(warning)
    ax.text(
        0.5,
        0.175,
        "禁止：持续受压时无条件更新基线，否则真实压力会被逐步学进 baseline",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#9B3E50",
    )

    fig.savefig(
        ASSET_DIR / "基线与慢漂移状态机.png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def draw_drift_problem():
    data_path = (
        RESEARCH_DIR
        / "DataSet"
        / "织物垫"
        / "空载1min-1kg 20min-空载1min"
        / "20260713_162110_single_device_c56f37"
        / "device_001.csv"
    )
    elapsed, channels = load_session_csv(data_path)
    total_adc = channels.sum(axis=1)
    smooth = moving_average(total_adc, 625)

    loaded = total_adc > np.percentile(total_adc, 60)
    loaded_indices = np.flatnonzero(loaded)
    start_index = loaded_indices[0]
    end_index = loaded_indices[-1]
    hold_time = elapsed[start_index : end_index + 1] - elapsed[start_index]
    hold_total = total_adc[start_index : end_index + 1]
    hold_smooth = smooth[start_index : end_index + 1]

    one_minute = 60.0
    first_mask = (hold_time >= 10) & (hold_time < 10 + one_minute)
    last_mask = (hold_time > hold_time[-1] - one_minute - 10) & (hold_time <= hold_time[-1] - 10)
    first_median = float(np.median(hold_total[first_mask]))
    last_median = float(np.median(hold_total[last_mask]))
    drift_percent = (last_median - first_median) / first_median * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=160, gridspec_kw={"width_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle("困境一：固定载荷下仍存在长时变化", fontsize=23, fontweight="bold", color="#172033", y=0.97)

    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    ax.plot(hold_time / 60, hold_total, color="#AFC2D8", linewidth=0.45, alpha=0.55, label="逐帧总 ADC")
    ax.plot(hold_time / 60, hold_smooth, color="#C44D61", linewidth=2.2, label="10 s 平滑趋势")
    ax.axvspan(10 / 60, (10 + one_minute) / 60, color="#3978C5", alpha=0.12)
    ax.axvspan((hold_time[-1] - one_minute - 10) / 60, (hold_time[-1] - 10) / 60, color="#C58A18", alpha=0.15)
    ax.set_title("1 kg 长时保持：逐帧波动之上仍有慢趋势", fontsize=14, fontweight="bold")
    ax.set_xlabel("保持时间 / min")
    ax.set_ylabel("96 Cell 总 ADC")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    ax.set_facecolor("#FFFFFF")
    bars = ax.bar(
        ["保持初段\n1 min", "保持末段\n1 min"],
        [first_median, last_median],
        color=["#3978C5", "#C58A18"],
        width=0.58,
    )
    ax.set_title("首尾稳态中位数对比", fontsize=14, fontweight="bold")
    ax.set_ylabel("总 ADC 中位数")
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, [first_median, last_median]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=11)
    ax.text(
        0.5,
        0.05,
        f"首尾变化：{drift_percent:+.2f}%\n需进一步区分材料蠕变、界面变化与温漂",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#9B3E50",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FCEAEC", "edgecolor": "#C44D61"},
    )

    fig.text(
        0.5,
        0.025,
        "结论：低通可以压低逐帧波动，但不能自动消除保持段均值随时间变化。数据为采集器 processed_display 输出。",
        ha="center",
        fontsize=11,
        color="#526078",
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.92))
    fig.savefig(ASSET_DIR / "困境-数据时漂.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_noise_problem():
    data_path = (
        RESEARCH_DIR
        / "DataSet"
        / "织物垫"
        / "1kg砝码压力"
        / "录制数据_20260710143635_part0.csv"
    )
    elapsed, total_adc = load_legacy_csv(data_path)
    sample_interval = np.median(np.diff(elapsed))
    if sample_interval > 1:
        elapsed = elapsed / 1000
    window_mask = (elapsed >= 12) & (elapsed <= 20)
    window_time = elapsed[window_mask]
    window_total = total_adc[window_mask]
    alpha = 1 - np.exp(-np.median(np.diff(window_time)) / 0.8)
    ema = np.empty_like(window_total)
    ema[0] = window_total[0]
    for index in range(1, len(window_total)):
        ema[index] = ema[index - 1] + alpha * (window_total[index] - ema[index - 1])
    raw_roughness = float(np.std(np.diff(window_total)))
    ema_roughness = float(np.std(np.diff(ema)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=160, gridspec_kw={"width_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle("困境二：固定载荷稳态段仍有明显帧间震荡", fontsize=23, fontweight="bold", color="#172033", y=0.97)

    ax = axes[0]
    ax.set_facecolor("#FFFFFF")
    ax.plot(window_time, window_total, color="#8AA9C7", linewidth=1.2, label="原始总 ADC")
    ax.plot(window_time, ema, color="#C44D61", linewidth=2.4, label="EMA 0.8 s")
    ax.set_title("稳态局部窗口：噪声与真实均值是不同问题", fontsize=14, fontweight="bold")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("96 Cell 总 ADC")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.set_facecolor("#FFFFFF")
    bars = ax.bar(["原始", "EMA 0.8 s"], [raw_roughness, ema_roughness], color=["#3978C5", "#27896A"], width=0.58)
    ax.set_title("帧间 roughness", fontsize=14, fontweight="bold")
    ax.set_ylabel("相邻帧差值标准差")
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, [raw_roughness, ema_roughness]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=11)
    reduction = (1 - ema_roughness / raw_roughness) * 100
    ax.text(
        0.5,
        0.05,
        f"帧间震荡降低约 {reduction:.0f}%\n但这不证明力值准确度同步提高",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#246F59",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#EAF7F2", "edgecolor": "#27896A"},
    )

    fig.text(0.5, 0.025, "结论：时间滤波适合改善显示稳定性；跨次均值差异、迟滞和漂移必须另行建模。", ha="center", fontsize=11, color="#526078")
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.92))
    fig.savefig(ASSET_DIR / "困境-数据噪声.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_crosstalk_problem():
    size = 64
    axis = np.arange(size)
    grid_x, grid_y = np.meshgrid(axis, axis)
    truth = np.exp(-((grid_x - 31.5) ** 2 + (grid_y - 31.5) ** 2) / (2 * 3.4**2))
    truth = truth / truth.max()
    observed = 0.84 * truth.copy()
    for shift_x, shift_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        observed += 0.04 * np.roll(np.roll(truth, shift_y, axis=0), shift_x, axis=1)
    residual = np.clip(observed - truth, 0, None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=160)
    fig.patch.set_facecolor("#F7F9FC")
    fig.suptitle("困境三：空间耦合会把真实接触扩散成外围伪影", fontsize=23, fontweight="bold", color="#172033", y=0.97)
    images = [truth, observed, residual]
    titles = ["理想局部压力", "含 16% 邻域泄漏的观测", "非接触区候选伪影"]
    color_maps = ["viridis", "viridis", "magma"]
    for ax, image, title, color_map in zip(axes, images, titles, color_maps):
        rendered = ax.imshow(image, cmap=color_map, origin="lower", vmin=0)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("列")
        ax.set_ylabel("行")
        fig.colorbar(rendered, ax=ax, fraction=0.046, pad=0.04)
    fig.text(
        0.5,
        0.025,
        "说明：本图是串扰机理示意，不是对当前膜片的定量结论；真实设备必须通过单点扫描和双点叠加辨识空间模型。",
        ha="center",
        fontsize=11,
        color="#9B3E50",
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.91))
    fig.savefig(ASSET_DIR / "困境-串扰伪影.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    draw_layered_model()
    draw_product_architecture()
    draw_host_algorithm_pipeline()
    draw_baseline_state_machine()
    draw_drift_problem()
    draw_noise_problem()
    draw_crosstalk_problem()
    print(f"Generated diagrams in: {ASSET_DIR}")
