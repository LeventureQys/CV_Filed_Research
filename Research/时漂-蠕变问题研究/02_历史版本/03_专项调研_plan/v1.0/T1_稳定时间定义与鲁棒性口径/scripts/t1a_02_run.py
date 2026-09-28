# -*- coding: utf-8 -*-
"""t1a_02_run：在 13 份录制上跑三条实现（v5.1 / v6 / v6.1）+ raw，落缓存。

口径：100 Hz 网格（`np.interp`，dt=0.01 s）、时间轴 `timestamp`；算法原型为本目录 `t1a_glm53_*.py`
副本（**不 import 既有桶、不修改原型**），经 `t1a_common.run_arm` 仪表化运行（只记录状态变化）。

缓存（results/cache/t1a_<rec>.npz）：tu、yraw_ch、yraw_tot、<arm>_ch、<arm>_tot（每份 <100 KB）。
状态机计数（results/t1a_arm_epochs.csv）：epoch / revoke / handoff / unload 次数与时刻。

用法：python scripts/t1a_02_run.py            # 全 13 份（约 5~12 min，建议后台作业）
      python scripts/t1a_02_run.py 左拇指指尖/数据2   # 只跑指定录制（冒烟/复算单份）
产出：results/cache/*.npz、results/t1a_arm_epochs.csv、results/_t1a_02_run.log
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import t1a_common as C


def cache_path(rec):
    return os.path.join(C.CACHE, "t1a_%s.npz" % rec.replace("/", "_"))


def alt_channel(rec, Xu, events):
    """该录制**最大正台阶（onset/restep）事件**的"台阶最大通道"（主通道定义的敏感性口径）。

    注意：必须排除卸载类事件 —— 卸载的 |J| 通常是全录制最大，但其"最大通道"是卸载后归零的通道，
    不能当加载主通道。
    """
    sub = events[(events["ds"] == rec) & (events["J_ok"]) & (events["J"] > 0)
                 & (events["kind"].isin(["onset", "restep"]))]
    if not len(sub):
        return None, None
    e = sub.loc[sub["J"].abs().idxmax()]
    k0 = int(e["k_on"])
    pre = np.median(Xu[max(0, k0 - 200):k0], axis=0)
    post = np.median(Xu[k0 + 400:k0 + 600], axis=0)
    return int(np.argmax(post - pre)), float(e["t_on"])


def main():
    C.start_log("02_run")
    os.makedirs(C.CACHE, exist_ok=True)
    events = pd.read_csv(os.path.join(C.RES, "t1_events.csv"), encoding="utf-8-sig")
    recs = sys.argv[1:] or [r["rec"] for r in C.RECS]
    if recs and recs[0] in ("--idx", "--index"):     # 避开控制台中文参数编码问题
        recs = [C.RECS[int(i)]["rec"] for i in recs[1:]]
    print("待跑录制 %d 份：%s" % (len(recs), ", ".join(recs)))
    rows = []
    for rec in recs:
        t0 = time.time()
        tu, Xu, m = C.load_grid(rec)
        ch = m["ch"]
        chalt, t_alt = alt_channel(rec, Xu, events)
        payload = dict(tu=tu, yraw_ch=Xu[:, ch], yraw_tot=Xu.sum(axis=1), n_ch=m["n_ch"],
                       dom=m["dom"], family=m["family"], main_ch=ch, pkt_p90=m["pkt_p90"],
                       chalt=(-1 if chalt is None else chalt),
                       yraw_chalt=(Xu[:, ch] if chalt is None else Xu[:, chalt]))
        print("\n[%s] n=%d  域=%s  指定主通道=%d  最大台阶通道=%d(@t=%.2f s)%s"
              % (rec, len(tu), m["dom"], ch, -1 if chalt is None else chalt,
                  -1 if t_alt is None else t_alt, "" if chalt == ch else "  **不一致**"))
        for arm in C.ARMS:
            ta = time.time()
            out = C.run_arm(rec, arm, tu, Xu)
            payload["%s_ch" % arm] = out["ch"]
            payload["%s_tot" % arm] = out["tot"]
            payload["%s_chalt" % arm] = (out["ch"] if chalt is None or chalt == ch
                                         else out["Y"][:, chalt])
            ne, nr = len(out["epoch"]), len(out["revoke"])
            ep_t = ";".join("%.2f" % (x[0] if isinstance(x, (tuple, list)) else x)
                            for x in out["epoch"]) if out["epoch"] else ""
            rows.append(dict(rec=rec, arm=arm, n_epoch=ne, n_revoke=nr,
                             n_handoff=len(out["handoff"]), n_unload=len(out["unload"]),
                             epoch_t=ep_t, sec=round(time.time() - ta, 1)))
            print("   %-5s %5.1f s  epoch=%2d revoke=%2d handoff=%2d unload=%2d  ch[末]=%.3f tot[末]=%.1f"
                  % (arm, time.time() - ta, ne, nr, len(out["handoff"]), len(out["unload"]),
                     float(out["ch"][-1]), float(out["tot"][-1])))
            del out
        np.savez_compressed(cache_path(rec), **payload)
        print("   -> %s（本份 %.1f s）" % (os.path.basename(cache_path(rec)), time.time() - t0))

    df = pd.DataFrame(rows)
    p = os.path.join(C.RES, "t1a_arm_epochs.csv")
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n== 状态机计数汇总（13 份 × 4 臂） ==")
    print(df.groupby("arm")[["n_epoch", "n_revoke", "n_handoff", "n_unload", "sec"]].agg(
        ["median", "max"]).to_string())
    print("-> %s" % p)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
