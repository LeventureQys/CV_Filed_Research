# -*- coding: utf-8 -*-
"""扫参公共层：把参数集喂给 C++ v34_sweep.exe（链接产品源码），把结果整理成指标表。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sensor_common import SENSORS  # noqa: E402

EXE = next((p for p in (HERE / "build" / "v34_sweep_v3.exe",
                        HERE / "build" / "v34_sweep.exe") if p.exists()),
           HERE / "build" / "v34_sweep_v3.exe")
OUT_DIR = HERE / "out"
PARAM_SETS_DIR = OUT_DIR / "_paramsets"
PREP = json.loads((OUT_DIR / "02_prep.json").read_text(encoding="utf-8"))
CACHE = OUT_DIR / "_raw"
# 会话二进制缓存（Python 侧一次性把 CSV 转成 [elapsed, ch...] 的 float64 二进制，
# C++ 侧直接读，省掉每次几秒的 CSV 解析）
SESS_BIN_DIR = OUT_DIR / "_sess"
SESSION_BIN: dict[str, Path] = {}
for _k in PREP:
    _f = SESS_BIN_DIR / (_k.replace("/", "_") + ".bin")
    if _f.exists():
        SESSION_BIN[_k] = _f


@dataclass
class Set:
    name: str
    over: dict = field(default_factory=dict)

    def line(self) -> str:
        body = ",".join(f"{k}={v!r}" for k, v in self.over.items())
        return f"{self.name}|{body}"


def write_setfile(sets: list[Set], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s.line() for s in sets) + "\n", encoding="utf-8")
    return path


def run_session(sensor: str, tag: str, sets: list[Set], *, time_mode: str = "uniform",
                zero: bool = True, tag_name: str = "") -> dict[str, dict]:
    """跑一份会话的所有参数集，返回 {参数集名: 指标}。结果缓存到 out/_raw。

    注意：缓存文件名**必须**同时含会话键与 tag（否则同传感器多会话会互相覆盖）。
    """
    key = f"{sensor}/{tag}"
    p = PREP[key]
    safe = key.replace("/", "_").replace("\\", "_")
    if not sets:
        return {}
    # 缓存名 = 会话键 + 时间口径 + 调零口径 + **参数集内容哈希**：
    # 并行跑多个搜索进程时也不会互相覆盖（曾因共用计数器命名而串档）。
    digest = hashlib.sha1("\n".join(s.line() for s in sets).encode("utf-8")).hexdigest()[:12]
    name = f"{safe}__{tag_name or 't'}__{digest}__{time_mode}{'_z' if zero else ''}"
    sf = write_setfile(sets, PARAM_SETS_DIR / f"{name}.txt")
    cache_f = CACHE / f"{name}.csv"

    def compute() -> str:
        src = SESSION_BIN.get(key, p["path"])
        cmd = [str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}", f"{p['t0']:.6f}",
               "--time", time_mode]
        if zero:
            cmd.append("--zero")
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"v34_sweep failed on {key}: {r.stderr}")
        return r.stdout

    def parse(txt: str) -> list[dict]:
        rows = list(csv.DictReader(io.StringIO(txt)))
        if len(rows) != len(sets):
            raise ValueError(f"行数不符：{len(rows)} != {len(sets)}")
        for row in rows:
            for k, v in row.items():
                if k != "name":
                    float(v)
        return rows

    txt = cache_f.read_text(encoding="utf-8") if cache_f.exists() else None
    try:
        rows = parse(txt) if txt else None
        if rows is None:
            raise ValueError("no cache")
    except Exception:
        txt = compute()
        rows = parse(txt)
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(txt, encoding="utf-8")
    out = {}
    esum = p["Esum_channels"]
    creep = max(p["creep_total"], 1.0)
    for row in rows:
        nm = row["name"]
        f = {k: float(v) for k, v in row.items() if k not in ("name",)}
        # 统一按「本该被扣掉的累计蠕变 creep」归一：ΣE 对「录制时已受载」的会话≈0，不能做分母。
        # 绝对 ADC 口径（用户验收关心的是"过减深度"的绝对值，不只看比例）：
        #   disp_min/disp_max 取自 t ≥ t0+0.5 s 的显示值；参考电平 ref：
        #   有加载台阶的会话 = ΣE（应钉在弹性电平）；无台阶的会话 = 0（调零后的起始基线）
        ref = esum if p.get("has_step") else 0.0
        dmin, dmax = f["disp_min"], f["disp_max"]
        sm_max, sm_min, sm_end = f["sm_max"], f["sm_min"], f["sm_end"]
        out[nm] = {
            "key": key, "Esum": esum, "creep": creep, "ref": ref,
            "resid_pct": f["err_end"] / creep * 100.0,        # 末值残漂（+欠扣 / −过扣）
            "over_pct": -f["err_min"] / creep * 100.0,        # 保压期最深过扣（正=已扣过头）
            "under_pct": f["err_max"] / creep * 100.0,        # 保压期最大欠扣
            "tail_pct": f["tail30"] / creep * 100.0 * 30.0,   # 末 30 s 残余漂移
            "std_pct": f["hold_std"] / creep * 100.0,         # 保压期抖动
            "ded_pct": f["ded_end"] / creep * 100.0,          # 蠕变扣除率
            "err_end_adc": f["err_end"], "err_min_adc": f["err_min"],
            "ded_end_adc": f["ded_end"], "in_end_adc": f["in_end"],
            # ── 绝对 ADC 口径（用户验收：过减用绝对值，50~150 可接受）──
            "notch": f["notch"], "notch_t": f["notch_t"],     # ★ 过减 = 相对历史最高点的最大回落
            "peak_exc": max(0.0, sm_max - ref),               # 欠减峰值（显示浮在参考线上方）
            "low_exc": max(0.0, ref - sm_min),                # 绝对过减（显示压到参考线下方）
            "end_exc": sm_end - ref,                          # 末值偏差（基线漂移的落点）
            "wander_adc": dmax - dmin, "disp_min_adc": dmin, "disp_max_adc": dmax,
            "sm_max": sm_max, "sm_min": sm_min, "sm_end": sm_end,
        }
    return out


def run_sensor(sensor: str, sets: list[Set], **kw) -> dict[str, dict[str, dict]]:
    res = {}
    for tag in SENSORS[sensor]["sessions"]:
        for nm, m in run_session(sensor, tag, sets, **kw).items():
            res.setdefault(nm, {})[tag] = m
    return res


def score(m: dict, over_tol: float = 10.0) -> float:
    """（旧口径，比例制）越小越好：末值残漂 + 末 30 s 残余漂移 + 过扣惩罚。"""
    return (abs(m["resid_pct"]) + abs(m["tail_pct"])
            + 2.0 * max(0.0, m["over_pct"] - over_tol))


# 用户验收口径（绝对 ADC）：
#   low_exc  = 显示压到参考电平**下方**多少 ADC —— 这才是字面意义的「过减」，可接受 50~150；
#   notch    = 显示相对历史最高点的回落（另一面，回落过大说明补偿追过头）；
#   peak_exc = 显示浮在参考电平上方多少 ADC —— 欠减/基线漂移。
LOW_TOL = 100.0
NOTCH_TOL = 150.0


def score_adc(m: dict, low_tol: float = LOW_TOL, notch_tol: float = NOTCH_TOL) -> float:
    """绝对口径目标函数（越小越好）：
        peak_exc + 0.5·|末值偏差| + 4·max(0, 过减 − 容差) + 1·max(0, 回落 − 容差)
    先把「显示沉到参考线下方超过容差」压住，再让显示尽量贴着参考线。"""
    return (m["peak_exc"] + 0.5 * abs(m["end_exc"])
            + 4.0 * max(0.0, m["low_exc"] - low_tol)
            + 1.0 * max(0.0, m["notch"] - notch_tol))


def agg(res: dict[str, dict[str, dict]], sensor: str) -> list[dict]:
    rows = []
    for nm, per in res.items():
        ms = list(per.values())
        row = {"set": nm, "n": len(ms), "score": sum(score(m) for m in ms) / len(ms)}
        for k in ("resid_pct", "over_pct", "under_pct", "tail_pct", "std_pct", "ded_pct",
                  "err_end_adc", "err_min_adc", "ded_end_adc"):
            row[k] = sum(m[k] for m in ms) / len(ms)
        row["resid_worst"] = max(abs(m["resid_pct"]) for m in ms)
        row["over_worst"] = max(m["over_pct"] for m in ms)
        rows.append(row)
    rows.sort(key=lambda r: r["score"])
    return rows


def print_table(rows: list[dict], sensor: str, top: int | None = None) -> None:
    print(f"\n===== {sensor}：按 score 排序（越小越好；百分比一律以「本该扣掉的蠕变」为分母）=====")
    print(f"{'参数集':<38s} {'score':>7s} {'残漂%':>8s} {'过扣%':>8s} {'欠扣%':>8s} "
          f"{'末30s%':>8s} {'抖动%':>7s} {'扣除%':>7s} {'末残ADC':>9s}")
    for r in (rows if top is None else rows[:top]):
        print(f"{r['set']:<38s} {r['score']:7.2f} {r['resid_pct']:8.2f} {r['over_pct']:8.2f} "
              f"{r['under_pct']:8.2f} {r['tail_pct']:8.2f} {r['std_pct']:7.2f} "
              f"{r['ded_pct']:7.1f} {r['err_end_adc']:9.1f}")
