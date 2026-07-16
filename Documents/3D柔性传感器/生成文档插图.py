from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)

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


if __name__ == "__main__":
    draw_layered_model()
    draw_product_architecture()
    draw_host_algorithm_pipeline()
    draw_baseline_state_machine()
    print(f"Generated diagrams in: {ASSET_DIR}")
