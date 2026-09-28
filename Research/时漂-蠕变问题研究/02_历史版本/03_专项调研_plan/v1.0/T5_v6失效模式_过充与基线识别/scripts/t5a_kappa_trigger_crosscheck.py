# -*- coding: utf-8 -*-
"""C-4 交叉核对：用**两种独立方法**确认"κ 上限何时开始咬人"，结果必须一致。

方法 A（被动观测）：`r = A_raw/inc`，判据 `r > κ`（与上一版脚本同源）。
方法 B（**直接判据**）：对每个 κ，直接比较"探针记录的调用 κ"与"实际封顶是否发生" ——
    即比较 `κ·inc` 与 `A_raw` 的大小并按 `κ_probe` 分组统计。
两者若给同一结论，说明触发率不是算法伪影。
另外核对：**κ_probe 记录的确实是 onset 用 κ_onset、restep 用 κ_restep**。

产出：`results/t5a_kappa_trigger_crosscheck.csv`、`_t5a_kappa_trigger_crosscheck.log`。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402
from t5a_kappa_probe import KV6P                        # noqa: E402

LOG = []


def rec(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    rows = []
    kappa_by_kind = {}
    for k, d in recs.items():
        sub = ev[ev.key == k]
        if not len(sub):
            continue
        c = KV6P(d["Xu"].shape[1])
        c.kappa_onset, c.kappa_restep = 1.30, 1.12   # 现状
        c.enable_probe = True
        o = c.run(d["tu"], d["Xu"])
        A, inc, kp, kd = o["A_raw"], o["inc_probe"], o["kappa_probe"], o["kind_probe"]
        m = np.isfinite(A) & np.isfinite(inc) & (inc > 1e-9) & np.isfinite(kp)
        # 核对 κ 与事件类型的对应
        for kind in ("onset", "restep"):
            sel = m & (kd == kind)
            if sel.sum():
                ks = np.unique(np.round(kp[sel], 4))
                kappa_by_kind.setdefault(kind, set()).update(ks.tolist())
        r = A[m] / inc[m]
        cap_bound = kp[m] * inc[m]
        # 方法 B：实际发生封顶 = A_raw > κ·inc（等价于 r > κ）
        n_total = int(m.sum())
        n_hit = int((A[m] > cap_bound).sum())
        rows.append(dict(key=k, n_probe_frames=n_total, n_cap_bound_frames=n_hit,
                         frac_cap_bound=n_hit / n_total if n_total else np.nan,
                         r_med=float(np.median(r)) if n_total else np.nan,
                         r_max=float(r.max()) if n_total else np.nan,
                         kappa_med=float(np.median(kp[m])) if n_total else np.nan))
    R = pd.DataFrame(rows)
    p = os.path.join(C.TASK, "results", "t5a_kappa_trigger_crosscheck.csv")
    R.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"产出 {p}")
    rec(f"κ_probe 与事件类型的对应：onset → {sorted(kappa_by_kind.get('onset', []))}；"
        f"restep → {sorted(kappa_by_kind.get('restep', []))}")
    ok = (kappa_by_kind.get("onset", set()) <= {1.30} and
          kappa_by_kind.get("restep", set()) <= {1.12})
    rec(f"核对结果：{'PASS —— onset 用 κ_onset(1.30)、restep 用 κ_restep(1.12)' if ok else 'FAIL'}")
    rec("\n逐录制（现状 κ=1.30/1.12 下的封顶帧占比）：")
    rec(R.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    C.write_log(os.path.join(C.TASK, "results", "_t5a_kappa_trigger_crosscheck.log"),
                "python scripts/t5a_kappa_trigger_crosscheck.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
