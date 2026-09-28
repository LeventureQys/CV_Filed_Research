import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
names = {f.name for f in fm.fontManager.ttflist}
print("Microsoft YaHei 可用:", "Microsoft YaHei" in names, "| SimHei 可用:", "SimHei" in names)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fig, ax = plt.subplots(figsize=(3,2))
    ax.set_title("中文测试 切换负载 快相免责期 ①②③")
    fig.savefig("figures/_fonttest.png", dpi=80)
    print("glyph 警告数:", len([x for x in w if "missing from font" in str(x.message)]))
