# Issue #7 rework — omegaB97M-V UKS solution stability and the auto+verify route

**状态**: review_requested（暂用，未验收；本 rework 轮针对 Codex 2026-10-06
审查卡 `control/review-20261006.md` + `issue-7-next-round.md`）·
**本轮证据**: `/data/results/gpu4pyscf-agent/tasks/issue-7/2026-10-06-uks-stability/`
（sha256 锚定清单见该目录 `evidence_index.json`）·
**源码**: `e9c24f7` + 本 issue 提交 · **科学门**: scientific_gate_passed =
**false**（methoxy/def2-svp UKS wb97m-v 格的 1e-6 Ha 精度门在诚实语义下
仍未通过——原因现已归因为解的多重性，见 §2；fail 记录从未改标签）

## 1. 复现（本轮新证据，非旧结果复用）

`reproduce_diagnose.json`：methoxy/def2-svp UKS wb97m-v（设置与 #7 生产行
逐项一致：DF/def2-universal-jkfit、conv_tol 1e-10、minao 初猜、charge 0
spin 1）。

- fp64 参考道 E = -114.90169501301963 Ha（11 圈）；
- auto 道 3/3 独立重复收敛到 E = -114.9016962956088 Ha（19 圈），
  dE = **-1.282589e-6 Ha**，超 1e-6 门，复现成功；
- Mulliken 自旋布居两解一致到 1e-5 以内；S²（共线 UKS）= 0.75 相同；
  占据数 (9,8) 相同；两解轨道梯度残差均 ~1e-6（各自都是真不动点）。

## 2. 诊断：不同的定态解，不是收敛不足也不是积分精度误差

按审查卡做了四组对照，全部记录在本轮证据目录：

| 实验 | 文件 | 结论 |
|---|---|---|
| FP64/auto 交叉重启 | `reproduce_diagnose.json` | fp64 SCF 从 auto 密度重启 3 圈收敛在 auto 能量（-1.2831e-6 offset 保持）；auto 策略从 fp64 密度重启留在 fp64 盆地（1.9e-12）。**两个盆地互为独立的 fp64 不动点**——排除"收敛不足" |
| 密度/轨道/占据/S² 记录 | `orbital_analysis.json` | 总密度差 max 0.35；alpha 占据空间几乎相同（SVD 奇异值全 ~1）；**beta 占据空间差一个轨道**（SVD 谱 0.9999..×7, **0.2576**×1）——差异是一个 beta 占据轨道交换 |
| 几何位移有限差分（支线斜率） | `branch_slope_fd.json`, `branch_fd_tracked.json` | 沿 max\|Δg\| 方向（atom1-y）位移 ±0.005/0.01/0.02 bohr，两支线各自中心差分斜率 ∓0.037 Eh/Bohr，**斜率差 0.0744 = 2×Δg 分量（0.0369）**，且 3 个位移长度一致到 0.05%——证明两解是**能量相近但核斜率相反的两个定态分支**，梯度差完全由解选择解释 |
| NLC/实现一致性 | #7 既有 R10 + 本轮 `stability_selection.json` | NLC 分量跨车道一致（1.6e-8，#7 已证）；本轮再证两解的 VV10 项随密度差而差，不是 VV10 精度问题 |

**稳定性排序**（`stability_selection.json`，fp64 交叉重启测两支线能量）：
methoxy/svp wb97m-v：auto 支线**更低**（-1.283e-6）；methoxy/tzvp
wb97m-v：fp64 支线更低（-2.60e-8）；methoxy svp/tzvp b3lyp：fp64 支线
更低（-4.0e-8/-4.4e-8）。**哪个车道找到更低解是体系相关的**——这正是
审查卡"A small energy difference alone does not prove the solution
explanation"的定量回答：不是 auto 总是更低，而是 fp32 头部把 UKS 迭代
锁进不同盆地，两盆地能量差可正可负，幅度落在 1e-8..1e-6 Ha 区间。

与 #12 梯度发现的连接（`branch_slope_fd.json` 的 b3lyp 块）：
b3lyp/methoxy 两解占据空间几乎相同（beta SVD 最小 0.9996）但梯度差
1.16e-3 Eh/Bohr——同一多重解机制在小能量差（4e-8）下仍产生大梯度差，
完全解释 #12 的 b3lyp/wb97m-v methoxy UKS 梯度 fail（auto-fp64-grad
同值，fp32 内核项仅 1.6e-6——内核无罪）。

## 3. 处置：opt-in `precision_mode='auto+verify'`（默认不变）

**最小证据支持的修正**：新增 `gpu4pyscf/scf/stability_verify.py`
（`verify_solution_stability`）+ `scf/hf.py` 的 `auto+verify` 策略值 +
`grad/rhf.py _resolve_precision` 对该值的解析（梯度仍走 fp32 内核）。

语义：auto 策略收敛后，**同初猜再跑一条严格 fp64 参考轨迹**；若两轨迹
收敛到不同密度（max|dP| > 1e-3）且参考解更低，则采纳参考解的
轨道/能量/scf_summary 分量。选择记录（两解能量、圈数、轨道梯度残差、
密度差、selected/rejected）完整写入
`mf.scf_summary['stability_verification']`。主 SCF 不收敛时记录 skip
不崩溃。**默认 None/fp64/fp32/auto 路径逐行未动**（18 个既有 mixedprec
回归测试原样通过）。

实测（`auto_verify_functional.json` + `auto_verify_matrix.json`，
5 cell × 3 车道 × 3 独立样本，CUDA sync）：

| cell | dE(auto) vs fp64 | dE(auto+verify) vs fp64 | 选择 | 开销(vs auto) |
|---|---|---|---|---|
| methanol/svp 闭壳 | 2.8e-14 | 0.0 | 同解 | 2.03x |
| benzene/svp 闭壳 | 3.4e-13 | 4.6e-13 | 同解 | 2.07x |
| methoxy/svp UKS wb97m-v | **-1.283e-6 fail** | **-1.283e-6 fail（诚实保留）** | main=auto 支线（更低，采纳） | 1.65x |
| methoxy/tzvp UKS wb97m-v | 2.6e-8 | **0.0（选参考支线）** | reference | 2.14x |
| methoxy/svp UKS b3lyp | 4.0e-8 | **0.0（选参考支线）** | reference | 2.34x |

梯度（同 fp64 内核、只差 SCF 解选择，`auto_verify_matrix.json`）：
methoxy/tzvp wb97m-v max|Δg| = 2.7e-9（过 3e-5 门）；methoxy/svp b3lyp
= 1.2e-14（过门）；**methoxy/svp wb97m-v = 3.69e-2（仍不过）**——
验证道保留了 auto 支线（它更低），相对 fp64 参考道的梯度差是两支线
真实斜率差（FD 已证 0.0744），**这不是缺陷而是解多重性的物理后果**：
哪个解"正确"需要用户层面的稳定性分析（全谱不稳定检测超出本 issue
范围），auto+verify 的承诺是"两条轨迹中更低者+完整决策记录"。

## 4. 诚实声明与剩余问题

1. `auto+verify` **不宣称修复** methoxy/svp UKS 的 1e-6 门：两支线能量
   差本身 1.28e-6 超门，验证道选择其中更低者（auto 支线），相对 fp64
   参考道的差保持超门并如实 fail。它的价值是（a）消除 tzvp/b3lyp 等
   "auto 支线更高"格的解选择误差（实测 2.6e-8/4.0e-8 → 0）；（b）给出
   可审计的解选择记录。若用户需要与 fp64 车道逐位一致的 UKS 能量，
   现阶段的安全路线是 **fp64 SCF（可选配 fp32 梯度内核）**，
   #12 已实测其内核误差 ≤2.4e-6 Eh/Bohr。
2. 开销如实报告：1.65–2.34x（一条额外 fp64 SCF；大体系上应改用
   "从主解密度重启 + 微扰"的廉价验证，本轮未实现，非本轮范围）。
3. 全谱 UKS 稳定性分析（违背参考解的鞍点/不稳定方向搜索）未实现；
   本模块只比较两条确定性轨迹，**不是** `mf.stability()` 的完整替代。
4. 已知工具 quirk（#7 attempt-3 记录过）：GPU 锁心跳在持有 pid 退出后
   拒绝刷新（start CLI 的短命 pid）；本轮 deadline 03:43Z 前完成全部
   GPU 工作不受影响。

## 5. 复现命令

```bash
# 复现 + 诊断（~4 min GPU）
PYTHONPATH=. /home/qczcode/gpu4090-src/bin/python \
  /data/tmp/gpu4pyscf-agent/execution/issue7_rework/reproduce_diagnose.py
# 支线斜率 FD（~3 min GPU）
... branch_slope_fd.py   # 粗扫描（单轨道旋转，负结果保留）
... branch_slope_fd2.py  # 支线跟踪中心差分（决定性）
# 稳定性排序 + auto+verify 矩阵（~10 min GPU）
... stability_selection.py
... autoverify_matrix.py
# 单元回归（无网络）
PYTHONPATH=. /home/qczcode/gpu4090-src/bin/python -m unittest \
  gpu4pyscf.scf.tests.test_stability_verify
PYTHONPATH=. /home/qczcode/gpu4090-src/bin/python -m unittest \
  gpu4pyscf.df.tests.test_df_mixedprec
```
