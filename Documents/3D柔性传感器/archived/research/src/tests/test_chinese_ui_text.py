import os


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_PATH = os.path.join(SRC_DIR, "main.py")


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_single_processor_ui_uses_chinese_explanatory_text():
    with open(MAIN_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    for phrase in [
        "单文件处理与统计",
        "选择一个录制 CSV",
        "k_sigma：噪声门限强度",
        "decay：额外扣除量",
        "处理并生成统计报告",
        "本功能用于检查某一个 CSV 在当前降噪参数下的效果",
    ]:
        assert_true(phrase in text, f"missing Chinese UI text: {phrase}")


def test_all_top_level_tabs_use_chinese_labels_and_help_text():
    with open(MAIN_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    for phrase in [
        "降噪动画回放",
        "算法横向 Benchmark",
        "单文件处理与统计",
        "这个页面是做什么的？",
        "选择录制 CSV",
        "启动动画窗口",
        "运行全量 Benchmark",
        "保存 Benchmark 结果",
        "本工具用于 v1.0.10 beta1 研究版本",
    ]:
        assert_true(phrase in text, f"missing full-GUI Chinese text: {phrase}")


if __name__ == "__main__":
    test_single_processor_ui_uses_chinese_explanatory_text()
    test_all_top_level_tabs_use_chinese_labels_and_help_text()
    print("Chinese UI text tests passed")
