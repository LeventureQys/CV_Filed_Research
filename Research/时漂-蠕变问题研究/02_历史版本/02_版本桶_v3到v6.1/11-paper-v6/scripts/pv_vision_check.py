# -*- coding: utf-8 -*-
"""图件视觉验收：调用 DeepSeek vision 模型逐张检查 paper_v6/figures/*.png。

每个图带着自己的「期望描述」送检，模型回答四件事：
  ① 面板数与各面板内容是否与期望一致；② 是否有文字重叠/遮挡/截断；
  ③ 是否有空白面板或明显缺曲线；④ 是否出现图例遮挡数据、标签出界等排版问题。

用法：
  python pv_vision_check.py                # 检查全部（含期望描述已登记的图）
  python pv_vision_check.py F1 F3          # 只检查名字含 F1/F3 的图
  python pv_vision_check.py --list         # 只列出已登记图与期望
结果写 results/vision_check.md 与 results/vision_check.json。

密钥：从 ~/.dsh/.credentials.yaml 读取 DEEPSEEK_API_KEY（不硬编码在本文件里）。
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pv_common as C                                            # noqa: E402

API = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash-vision-exp"
CRED = os.path.join(os.path.expanduser("~"), ".dsh", ".credentials.yaml")

# ── 每张图的期望（人工写死，供模型核对；不依赖模型自由发挥）──────────
EXPECT = {
    "F1_two_phase.png": (
        "2×2 四面板图，中文标签。"
        "(a) 左上：对数横轴 0.02~5 s，9 条灰色细线 + 一条绿色粗中位线 + 绿色浅色 p10~p90 带，"
        "纵轴 0~1；另有棕色点线竖线标注 0.20 s；左上角有白底文本框写「中位形状：…」数值；"
        "(b) 右上：横轴 0~5 s、纵轴 0~1，三条线（绿色带圆点=原始、蓝色方块虚线=τ=2 s 平滑、"
        "灰点线=理论阶跃响应），0.02~0.2 s 处有一条浅橙色竖带；"
        "(c) 左下：t50/t80/t90/t95 四个分组的成对柱状图（绿=onset、橙=restep），柱顶有数值；"
        "(d) 右下：9 条水平条形图，条形右侧有百分数与秒数标注，纵轴为 9 个中文数据集名。"
        "全图无中文方框/乱码字符。"),
    "F2_overview.png": (
        "v6 算法总览示意图：上半部为逐帧数据流方框链（预处理 → 检测器 → 分类器 → 逆模型 → 滑行器 "
        "→ 慢相蠕变模块），下半部为三条输出链路（空载直通 / 滑行期 / 慢相扣除）与时间轴落点。"
        "中文文字为主，方框之间的箭头连接清晰，无文字压线。"),
    "F3_invert_glide.png": (
        "1×3 或 2×2 面板：单次真实加载事件的 (a) 原始读数与形状反演目标 Â 的对比、"
        "(b) 显示输出随时间（标注滑行段与交接时刻）、(c) 逐帧 Δ 或修正量轨迹。"
        "中文标签，曲线区分明显。"),
    "F4_slow_creep.png": (
        "慢相蠕变估计与扣除的面板图：包含逐通道归一化残差 (Z−A)/A 的分布与其中位共识 g 的时间轨迹、"
        "逐通道增益 γ 的分布/限幅条带、以及扣除量随时间的建立与输出封顶示意。中文标签。"),
    "F5_slow_tradeoff.png": (
        "慢相取舍图：含 A 慢修正（trim）开启/关闭的对照曲线或柱状（慢相段时漂、平台静态偏置、"
        "全程最大偏差）以及死区示意。中文标签。"),
    "F6_overview13.png": (
        "13 格小倍数总览（4 列 × 4 行，右下 3 格留空）：每格画一条灰色原始曲线与若干彩色补偿后曲线，"
        "每格标题为中文数据集名与指标数字。中文无乱码。"),
    "F7_fast_tradeoff.png": (
        "快相响应的取舍图：含 (a) T_stable 或首扣时延的分组柱状/箱线对比、(b) epoch 数与阶跃保真、"
        "(c) 加载沿附近放大曲线（显示快速滑到平台 vs 原始继续上漂）。中文标签与图例。"),
    "F8_ablation.png": (
        "消融/参数扫描图：柱状或折线展示若干参数档位（检测窗跨度、TRIM_RATE、死区）对关键指标的影响，"
        "有最优点标注。中文标签。"),
}

PROMPT = """你是图表质量检查员。下面给你一张科研论文配图，以及该图的"期望描述"。
请逐项核对并只回答 JSON（不要额外文字）：
{"panels_ok": true/false, "panel_count": 整数,
 "text_overlap": [], "text_clipped": [], "empty_panel": [],
 "missing_or_broken": [], "legend_covers_data": true/false,
 "garbled_text": true/false, "verdict": "PASS 或 FAIL",
 "summary": "两句话以内的中文结论", "issues": ["具体问题1", "具体问题2"]}
要求：
- text_overlap：列出你实际看到互相重叠/压住的文字内容；
- text_clipped：列出被画布边缘或其它元素截断的文字；
- garbled_text：中文出现方框、乱码、缺字时为 true；
- empty_panel：列出看起来应该画内容却是空白的面板名；
- missing_or_broken：期望里有、图上却没有或明显画错的元素；
- 如果一切正常，issues 为空数组。
期望描述：%s"""


def api_key():
    with open(CRED, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*DEEPSEEK_API_KEY:\s*(\S+)", line)
            if m:
                return m.group(1)
    raise RuntimeError("未找到 DEEPSEEK_API_KEY")


def ask(png_path, expect, key, retry=2):
    b64 = base64.b64encode(open(png_path, "rb").read()).decode()
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT % expect},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        ]}],
        "max_tokens": 32000,
        "temperature": 0,
        # 该模型的推理链会吃掉全部 output 配额（finish_reason=length 而 content 为空），
        # 本任务是"看图核对"，不需要推理：显式关闭 thinking、压低 reasoning effort。
        "reasoning_effort": "low",
        "thinking": {"type": "disabled"},
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + key})
    last = None
    for k in range(retry + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read().decode("utf-8"))
            txt = d["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if not m:
                return {"verdict": "PARSE_FAIL", "raw": txt[:600]}
            return json.loads(m.group(0))
        except Exception as e:                                   # noqa: BLE001
            last = e
            try:
                last = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            time.sleep(3 + 3 * k)
    return {"verdict": "ERROR", "raw": str(last)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        for k, v in EXPECT.items():
            print(f"{k}\n    {v}\n")
        return
    key = api_key()
    figs = sorted(f for f in os.listdir(C.FIG) if f.lower().endswith(".png"))
    if args:
        figs = [f for f in figs if any(a.lower() in f.lower() for a in args)]
    out, rows = {}, []
    for f in figs:
        exp = EXPECT.get(f)
        if exp is None:
            print(f"[skip] {f}：未登记期望描述")
            continue
        p = os.path.join(C.FIG, f)
        t0 = time.time()
        res = ask(p, exp, key)
        res["_file"] = f
        res["_bytes"] = os.path.getsize(p)
        res["_secs"] = round(time.time() - t0, 1)
        out[f] = res
        v = res.get("verdict", "?")
        print(f"[{v}] {f}  ({res['_bytes']/1024:.0f} KB, {res['_secs']}s)")
        print(f"      {res.get('summary', res.get('raw', ''))[:220]}")
        for it in (res.get("issues") or [])[:6]:
            print(f"      - {it}")
        if res.get("text_overlap"):
            print(f"      文字重叠: {res['text_overlap']}")
        if res.get("text_clipped"):
            print(f"      文字截断: {res['text_clipped']}")
        if res.get("empty_panel"):
            print(f"      空白面板: {res['empty_panel']}")
        if res.get("missing_or_broken"):
            print(f"      缺失/画错: {res['missing_or_broken']}")
        rows.append(res)
    with open(os.path.join(C.RES, "vision_check.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(C.RES, "vision_check.md"), "w", encoding="utf-8") as fh:
        fh.write("# DeepSeek vision 图件检查记录\n\n")
        fh.write(f"模型：`{MODEL}`；检查时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fh.write("| 图 | 判定 | 面板数 | 文字重叠 | 文字截断 | 空白面板 | 乱码 | 结论 |\n")
        fh.write("|---|---|:--:|---|---|---|---|---|\n")
        for f, r in out.items():
            fh.write(f"| {f} | {r.get('verdict')} | {r.get('panel_count','?')} "
                     f"| {len(r.get('text_overlap') or [])} | {len(r.get('text_clipped') or [])} "
                     f"| {len(r.get('empty_panel') or [])} | {r.get('garbled_text')} "
                     f"| {str(r.get('summary',''))[:110]} |\n")
        fh.write("\n## 逐图明细\n")
        for f, r in out.items():
            fh.write(f"\n### {f}\n```json\n{json.dumps(r, ensure_ascii=False, indent=2)}\n```\n")
    n_fail = sum(1 for r in rows if r.get("verdict") != "PASS")
    print(f"\n汇总：{len(rows)} 张，PASS {len(rows)-n_fail}，非 PASS {n_fail}"
          f"  → results/vision_check.md / .json")


if __name__ == "__main__":
    main()
