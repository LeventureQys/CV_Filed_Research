# 09-v7 —— 已否决的变体（存档）

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-17 11:47 ~ 11:53（与 v4 同期）
- 状态：**未采用**，仅作为对照留档
- 设计：免责期**只在首次 onset 生效**，负载内变载完全交给 v3 的 restep 原子迁移

v7 想解决的是「v4 免责期把真实变载也冻住导致欠报」，思路是把免责期限定在首次 onset。
自检确认它确实在补偿（`A_max = 1.3063`、末端补偿 +0.81），但
**在变化负载上与 v3 逐帧完全相同** ⇒ 对变载工况没有任何改善，
所以推荐方案仍是 v4/v5 的语义（首次 onset 与变载都走免责期）。

> ⚠ 命名坑：`z3_v6.py` 的名字写的是 v6，语义实为后来的 v7（首次 onset 固定 5 s 冻结、变载不冻结），
> 因此归入本桶；真正的 v6 是 `07-v6` 的 `glm53_v6.py`（形状约束反演）。
> 另：`05-v5算法说明.md` §1.3 指出 `02` §5.3 的「v7 ≡ v3」结论表述不成立，以 `05` 的复核为准。

## 文档（0）

（无）

## 图件（0）

无独立图件（结论为「与 v3 逐帧相同」，见 `results/varying_steps_v7.csv`）。


## 脚本（15）

`glm53_v7.py` 算法本体；`z7_v7_test.py` 自检；`ad_check_v7.py` 对照检查；`z3_v6.py` 同名异构的早期版本。

```text
ad_check_v7.py  glm53_v3.py  glm53_v7.py  r_fastphase.py  z2_v5.py  z3_v6.py
z7_v7_test.py
（编译缓存：__pycache__/glm53_v3.cpython-314.pyc, __pycache__/glm53_v7.cpython-314.pyc, __pycache__/r_fastphase.cpython-314.pyc, __pycache__/z2_v5.cpython-314.pyc, glm53_v3.cpython-314.pyc, glm53_v7.cpython-314.pyc, r_fastphase.cpython-314.pyc, z2_v5.cpython-314.pyc）
```

## 数据（1）

```text
varying_steps_v7.csv
```

## 复现

```powershell
cd temp/v4.1flash/progress/09-v7
$env:PYTHONIOENCODING='utf-8'
python scripts/z7_v7_test.py   # v7 自检（对照 v3 / v4）
```
