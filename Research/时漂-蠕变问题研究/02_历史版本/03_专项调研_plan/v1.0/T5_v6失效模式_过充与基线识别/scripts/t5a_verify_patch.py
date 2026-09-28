# -*- coding: utf-8 -*-
"""T5-A 补丁零差证明（`results/t5a_patch_ab_zero.csv`）。

三组 A/B，**全部要求最大绝对差 = 0**：

  V1 数据层补丁：原 `ad_lib.load_rec`（写死 skiprows=24）vs `t5a_ad_lib.load_rec`
     （自动定位 `##Data`）—— 必须给出同一 (t, X)。
  V2 原型补丁：`glm53_v6.GLM53v6`（原文件，未复制）vs `t5a_glm53_v6.GLM53v6`（复制改名后）
     在同一份录制上跑完整链路 —— 逐帧输出必须逐字节相同。
  V3 κ 补丁：`t5a_common.KV6` 在 `kappa_onset=1.30 / kappa_restep=1.12`（= 原型默认值）
     与 `t5a_glm53_v6.GLM53v6` 逐帧对比 —— 必须零差。这是后续一切 κ 扫描可信的前提。
  V4 v6.1 补丁：`t5a_glm53_v61.GLM53v61` vs `t5a_common.KV61`（κ = 1.30/1.12）—— 必须零差。

用法：`python scripts/t5a_verify_patch.py`
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                    # noqa: E402
from t5a_a_common import PROG, run_plain                  # noqa: E402

P06 = os.path.join(PROG, "07-v6", "scripts")
P081 = os.path.join(PROG, "08-v6.1", "scripts")
rows = []
log = []


def rec(msg):
    print(msg, flush=True)
    log.append(msg)


def add(vid, what, subject, nbytes, maxabs, extra=""):
    rows.append(dict(vid=vid, what=what, subject=subject, n_elem=nbytes,
                     max_abs_diff=maxabs, pass_=(maxabs == 0.0), note=extra))


def diff(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.abs(a - b).max())


def main():
    rec("=== T5-A 补丁零差证明 ===")
    # ── V1 数据层 ──
    sys.path.insert(0, P06)
    import ad_lib as A                                    # noqa: E402
    t_orig, X_orig = A.load_rec(C.REC_PATH["SW3"])
    t_new, X_new = C.load_uniform.__globals__["load_rec"](C.REC_PATH["SW3"])
    add("V1", "数据层 load_rec（timestamp 轴）", "SW3", len(t_orig),
        max(diff(t_orig, t_new), diff(X_orig, X_new)))
    rec(f"V1 load_rec: n={len(t_orig)}  max|Δt|={diff(t_orig, t_new):.3e}  "
        f"max|ΔX|={diff(X_orig, X_new):.3e}")

    # ── V2 原型补丁（v6）──
    import importlib
    d = C.recordings()["SW3"]
    g6 = importlib.import_module("glm53_v6")
    Y_orig, c_orig = run_plain(g6.GLM53v6, d["tu"], d["Xu"])
    from t5a_glm53_v6 import GLM53v6 as V6new
    Y_new, c_new = run_plain(V6new, d["tu"], d["Xu"])
    add("V2", "原型 glm53_v6 vs t5a_glm53_v6（逐帧输出）", "SW3", Y_orig.size,
        diff(Y_orig, Y_new), f"epoch {len(c_orig.epoch_t)} vs {len(c_new.epoch_t)}")
    rec(f"V2 v6 原型: max|ΔY|={diff(Y_orig, Y_new):.3e}  "
        f"epoch {len(c_orig.epoch_t)} vs {len(c_new.epoch_t)}")

    # ── V3 κ 补丁（v6）──
    out_k = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12)
    add("V3", "κ 补丁 KV6(1.30/1.12) vs 原型 v6（逐帧输出）", "SW3", Y_new.size,
        diff(Y_new, out_k["Y"]),
        f"epoch {len(c_new.epoch_t)} vs {len(out_k['epoch'])}")
    rec(f"V3 κ 补丁 v6: max|ΔY|={diff(Y_new, out_k['Y']):.3e}  "
        f"epoch {len(c_new.epoch_t)} vs {len(out_k['epoch'])}")

    # ── V4 κ 补丁（v6.1）──
    sys.path.insert(0, P081)
    g61 = importlib.import_module("glm53_v61")
    Y61_orig, c61 = run_plain(g61.GLM53v61, d["tu"], d["Xu"])
    out_k61 = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12, cls=C.KV61)
    add("V4", "κ 补丁 KV61(1.30/1.12) vs 原型 v6.1（逐帧输出）", "SW3", Y61_orig.size,
        diff(Y61_orig, out_k61["Y"]))
    rec(f"V4 κ 补丁 v6.1: max|ΔY|={diff(Y61_orig, out_k61['Y']):.3e}")

    # ── V5 v6.1 等价性自检（关 F1/F2 后必须 = v6）──
    out61_off = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12, cls=C.KV61,
                         rom_scale=1.0, rate_down=0.0)
    add("V5", "v6.1 关 F1/F2 后 vs v6（逐帧输出）", "SW3", Y_new.size,
        diff(Y_new, out61_off["Y"]))
    rec(f"V5 v6.1 关修复: max|ΔY|={diff(Y_new, out61_off['Y']):.3e}")

    # ── V6/V7/V8 κ 上限探针补丁的零差证明（MainAgent 2026-09-19 追加要求）──
    from t5a_kappa_probe import KV6P, KV61P
    o6p = KV6P(d["Xu"].shape[1])
    o6p.kappa_onset, o6p.kappa_restep = 1.30, 1.12
    o6p.enable_probe = True
    r6p = o6p.run(d["tu"], d["Xu"])
    add("V6", "κ 探针补丁 KV6P（探针开）vs 原型 v6（逐帧输出）", "SW3", Y_new.size,
        diff(Y_new, r6p["Y"]),
        f"探针有效帧 {int(np.isfinite(r6p['A_raw']).sum())} 帧")
    rec(f"V6 κ 探针 v6（开）: max|ΔY|={diff(Y_new, r6p['Y']):.3e}  "
        f"探针有效帧={int(np.isfinite(r6p['A_raw']).sum())}")
    o61p = KV61P(d["Xu"].shape[1])
    o61p.kappa_onset, o61p.kappa_restep = 1.30, 1.12
    o61p.enable_probe = True
    r61p = o61p.run(d["tu"], d["Xu"])
    add("V7", "κ 探针补丁 KV61P（探针开）vs 原型 v6.1（逐帧输出）", "SW3", Y61_orig.size,
        diff(Y61_orig, r61p["Y"]))
    rec(f"V7 κ 探针 v6.1（开）: max|ΔY|={diff(Y61_orig, r61p['Y']):.3e}")
    o6n = KV6P(d["Xu"].shape[1])
    o6n.kappa_onset, o6n.kappa_restep = 1.30, 1.12
    o6n.enable_probe = False
    r6n = o6n.run(d["tu"], d["Xu"])
    add("V8", "κ 探针补丁（探针关）vs 原型 v6（逐帧输出）", "SW3", Y_new.size,
        diff(Y_new, r6n["Y"]))
    rec(f"V8 κ 探针 v6（关）: max|ΔY|={diff(Y_new, r6n['Y']):.3e}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(C.TASK, "results", "t5a_patch_ab_zero.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    rec("")
    rec(df.to_string(index=False))
    rec(f"\nALL PASS = {bool(df['pass_'].all())}")
    rec(f"产出：{out_csv}")
    C.write_log(os.path.join(C.TASK, "results", "_t5a_verify_patch.log"),
                "python scripts/t5a_verify_patch.py\n\n" + "\n".join(log) + "\n")


def _imp(path, name):
    import importlib.util
    sys.path.insert(0, path)
    return importlib.import_module(name)


if __name__ == "__main__":
    main()
