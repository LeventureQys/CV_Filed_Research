# -*- coding: utf-8 -*-
"""v2.0 阶段三/四 · 离线验收复算（最终配置口径）。

臂（与交付的 C++ 默认值一一对应）：
  base        现役（plan-v1.0 A1+A4+A5a）
  default     本版**默认交付配置**：C（限幅 α=0.005）+ F5（g 负向下界 −0.05），F2/F3 关
  legacy_on   默认 + F2/F3（`legacy_fixes_enabled = true`）—— 用于量化"打开 F2/F3"的后果

指标（口径与 `results/design_regress_summary.md` 一致）：
  resid  受载段时漂残余 = 滑窗(5 s)极差中位 / 段中位
  lead   加载沿后 0.3~1.2 s 超前量中位（占台阶）
  T5%    进入自身稳态 ±5%·台阶 的时间中位
  G      沿后 4~5 s 台阶捕获比中位
  offabs 受载平台 |显示 − 原始| 中位
"""
import importlib.util
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

spec = importlib.util.spec_from_file_location("dfr", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "design_full_regress.py"))
DFR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DFR)


class Final(DFR.Arm):
    """C + F5 默认开、F2/F3 由开关控制（与交付的 C++ 默认值对齐）。"""

    def __init__(self, n, clamp=0.005, legacy=False, g_protect=True):
        super().__init__(n, clamp=clamp, f2=legacy, f3=legacy, f5=g_protect)


def run(el, V, legacy=False, g_protect=True):
    c = Final(V.shape[1], clamp=0.005, legacy=legacy, g_protect=g_protect)
    out = np.empty((len(el), V.shape[1]))
    for i in range(len(el)):
        out[i] = c.process(float(el[i]), V[i])
    return c, out.sum(1)


def main():
    sel = os.environ.get("V20_CASES", "new").strip()
    want = set(x for x in sel.split(",") if x)
    cases = []
    if "new" in want or "all" in want:
        cases.append(("★新录制(ADC/大偏置)", os.path.join(L.DS_ZERO, "device_001_pre_seg0.csv")))
    base_dir = os.path.join(L.ROOT, "temp", "原始数据only")
    if "all" in want or "legacy" in want:
        for grp in ("四指指尖", "右拇指指尖", "左拇指指尖"):
            for d in ("数据1", "数据2", "数据3"):
                p = os.path.join(base_dir, grp, d, "device_001_seg000.csv")
                if os.path.exists(p):
                    cases.append((f"{grp}/{d}", p))
    if "all" in want or "vary" in want:
        root = os.path.join(base_dir, "变化负载")
        for d in sorted(os.listdir(root)):
            for dp, _dn, fn in os.walk(os.path.join(root, d)):
                if "device_001_seg000.csv" in fn:
                    cases.append((f"变化负载/{d}", os.path.join(dp, "device_001_seg000.csv")))

    print(f"{'数据集':30s}{'臂':>10}{'resid':>9}{'lead':>9}{'T5%':>8}{'G':>8}{'offabs':>9}",
          flush=True)
    agg = {}
    for name, path in cases:
        el, V = DFR.read_simple(path)
        if len(el) < 500:
            continue
        pre = V.sum(1)
        tu = DFR.edges(el, pre)
        bm = None
        o_base = DFR.run(el, V)
        bm = DFR.evaluate(el, pre, o_base, tu)
        print(f"{name:30s}{'base':>10}{bm['resid']:9.4f}{100*bm['lead']:8.2f}%"
              f"{bm['t5']:8.2f}{bm['G']:8.3f}{bm['offabs']:9.0f}", flush=True)
        for tag, lg in (("default", False), ("legacy_on", True)):
            c, o = run(el, V, legacy=lg)
            m = DFR.evaluate(el, pre, o, tu)
            print(f"{name:30s}{tag:>10}{m['resid']:9.4f}{100*m['lead']:8.2f}%"
                  f"{m['t5']:8.2f}{m['G']:8.3f}{m['offabs']:9.0f}"
                  f"   Δresid={m['resid']-bm['resid']:+.4f}"
                  f" Δlead={100*(m['lead']-bm['lead']):+.2f}pp"
                  f" ΔG={m['G']-bm['G']:+.3f}", flush=True)
            agg.setdefault(tag, []).append((m, bm))
        print(flush=True)

    print("== 汇总（相对 base 的中位变化；口径见文件头）==", flush=True)
    for tag in agg:
        d_r = np.nanmedian([m["resid"] - b["resid"] for m, b in agg[tag]])
        d_l = np.nanmedian([m["lead"] - b["lead"] for m, b in agg[tag]])
        d_g = np.nanmedian([m["G"] - b["G"] for m, b in agg[tag]])
        d_o = np.nanmedian([m["offabs"] - b["offabs"] for m, b in agg[tag]])
        worst_r = np.nanmax([m["resid"] - b["resid"] for m, b in agg[tag]])
        worst_g = np.nanmin([m["G"] - b["G"] for m, b in agg[tag]])
        print(f"  {tag:10s} Δresid中位={d_r:+.5f} 最坏={worst_r:+.5f}  "
              f"Δlead中位={100*d_l:+.2f}pp  ΔG中位={d_g:+.4f} 最坏={worst_g:+.4f}  "
              f"Δoffabs中位={d_o:+.0f}", flush=True)


if __name__ == "__main__":
    main()
