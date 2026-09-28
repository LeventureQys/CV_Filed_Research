# -*- coding: utf-8 -*-
"""修正两处：① vision_check.md 附注被 PowerShell 反引号转义弄坏；② 生成器里的图件说明。"""
import io
import os

PROG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "v4.1flash", "progress")

# ---------- ① 重写 vision_check.md 的附注 ----------
p = os.path.join(PROG, "11-paper-v6", "results", "vision_check.md")
s = io.open(p, encoding="utf-8").read()
marker = "## 附注（2026-09-18 归档后修订）"
s = s[:s.index(marker)].rstrip("\n") if marker in s else s.rstrip("\n")
note = """

---

## 附注（2026-09-18 归档后修订）

以上记录是**重编号前**的原始检查结果（文件名保留当时的名字），检查结论对图的内容依然有效。
此后按「图号 = 正文出现顺序」重排了三张图的编号，并重新出图（图内标题里的图号同步更新）：

| 检查时的文件名 | 重编号后 | 正文位置 |
|---|---|---|
| `F5_overview13.png` | `F6_overview13.png` | §8.1 复算结果 |
| `F6_fast_tradeoff.png` | `F7_fast_tradeoff.png` | §8.3 首次加载的响应 |
| `F7_slow_tradeoff.png` | `F5_slow_tradeoff.png` | §7.4 取舍平面 |

F1~F4 未改名。重出图只改了输出文件名与图内标题里的图号，绘图逻辑与数据未变；
`results/vision_check.json` 同为重编号前的原始记录，不再回写。
重出图后未重跑 vision 验收（需 `DEEPSEEK_API_KEY`），逐图 `figcheck`（面板数/越界/文字重叠）通过。
"""
io.open(p, "w", encoding="utf-8", newline="\n").write(s + note)
print("vision_check.md 附注已重写:", os.path.getsize(p), "B")

# ---------- ② 生成器：docs 清单排除 docs/figures，并补图件说明 ----------
g = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_manifests.py")
s = io.open(g, encoding="utf-8").read()
pairs = [
    ('docs = tree_of(bdir, "docs")',
     'docs = [t for t in tree_of(bdir, "docs") if not t[0].startswith("docs/figures/")]'),
    ('fig_note="`f_review_1~4.png` 为评审报告配图',
     'fig_note="放本桶 `docs/figures/`（与 `dsp方案评审报告.md` 同目录，正文按 `figures/…` 引用）。'
     '`f_review_1~4.png` 为评审报告配图'),
    ('fig_note="`F1_two_phase`（快慢相两段结构）',
     'fig_note="放本桶 `docs/figures/`（与论文同目录，正文按 `figures/…` 引用）。`F1_two_phase`（快慢相两段结构）'),
    ('fig_note="`F1_two_phase`、`F2_overview`、`F3_invert_glide`（形状约束反演与滑行器）、`F4_slow_creep`、',
     'fig_note="放本桶 `docs/figures/`（与论文同目录，正文按 `figures/…` 引用）。按**正文出现顺序**编号，'
     '脚本 `pfN_*.py` 与图号一一对应：`F1_two_phase`（§1 加载形状实测）、`F2_overview`（§3 算法总览）、`F3_invert_glide`（§5 形状反演与滑行）、'),
    ('"`F5_overview13`（13 份录制）、`F6_fast_tradeoff`、`F7_slow_tradeoff`。"',
     '"`F4_slow_creep`（§6 慢相估计与扣除）、`F5_slow_tradeoff`（§7.4 慢相取舍）、`F6_overview13`（§8.1 13 份总览）、`F7_fast_tradeoff`（§8.3 快相取舍）。"'),
]
for a, b in pairs:
    assert a in s, "未匹配: " + a[:70]
    s = s.replace(a, b, 1)
io.open(g, "w", encoding="utf-8", newline="\n").write(s)
print("make_manifests.py 已更新:", len(pairs), "处")
