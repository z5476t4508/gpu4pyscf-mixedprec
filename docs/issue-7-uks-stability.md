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
   **（LC-29/31 关闭此投机项，2026-10-07）**：实测物理否决了该
   路线——交叉重启双向保盆地（本节 §2：fp64 从 auto 密度重启留在
   auto 盆地，auto 从 fp64 密度重启留在 fp64 盆地），从主解密度
   出发的任何廉价验证**原理上无法发现**参照支线；发现 fp64 道盆地
   必须走同初猜的完整参照轨迹。实测新开销（含精化道）：
   auto+verify/auto 中位 1.97–2.64x（6 格矩阵，见 §7.3.1）。
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

## 6. LC-1 补遗 (2026-10-06 长链工作单元): 内稳定性 Hessian 分析 —— fp64 道支线是鞍点

第 4 节第 3 条遗留的"全谱稳定性分析"已完成（长链工作单元 LC-1，纯 CPU
pyscf 2.14.0 独立实现，基准匹配设置，证据
`/data/results/gpu4pyscf-agent/tasks/issue-7/2026-10-06-lc1-stability-hessian/`，
11 个文件 sha256 锚定，`CONCLUSIONS.md` 为完整结论）：

- **fp64 道支线是内不稳定鞍点**：最软轨道 Hessian 本征值
  **-1.670e-5 Ha**（含 VV10 二阶响应；stock `with_nlc=False` 版
  -2.050e-4 同样不稳定）。符号在能量曲面上独立验证：沿单位化最软模
  t=±0.001 单点能量 dE=-8.1e-12，隐含曲率 -1.61..-1.67e-5，与
  Davidson 值 3 位有效数字一致。
- **auto 道支线是稳定极小**：最软本征值 **+1.273e-5 Ha**
  （v^T H v 残差 2.5e-16；小振幅扫描呈奇宇称信号 = 数值平坦方向上的
  近零线性项，比非线性底低两个量级，非符号错误）。
- **归属修正（不改任何基准行）**：methoxy/svp wb97m-v 格的
  -1.282589e-6 Ha "auto 相对 fp64 道误差"不是混合精度缺陷——fp64
  参考道收敛到的是鞍点，auto 道找到的是更稳更低的极小。1e-6 门以
  "fp64 道之解"为参照在该格上比较的对象本身不是极小；是否把解多重
  格的参照改为"稳定极小解"是用户策略决定，本分析只钉死物理事实。
  `auto+verify` 在该格选择的正是稳定极小支线。
- **工具性结论**：stock pyscf 稳定性分析（`with_nlc=False`）对 NLC
  泛函是错误的曲面（两支线都被判不稳定）；含 VV10 响应
  （`pyscf.hessian.rks.get_vnlc_resp`，支持非限制密度）才是 wb97m-v
  的物理完备 Hessian。未来任何 NLC 稳定性工具必须带 NLC 响应项。
- 模式跟踪（振幅 1/3）：从未找到低于 auto 支线的第三盆地；最软模与
  支线交换方向仅 ~26% cos 对齐，expmat 旋转不直接实现交换，多稳态
  流形在支线方向近平坦——与 #12 的 FD 支线混合观察一致。

复现：`stability_hessian2.py`（两支线 × stock/NLC 四次 Hessian 求解 +
模式跟踪）、`quadratic_form_check.py`、`qf_mismatch_diagnosis.py`
（小振幅 + vHv 一致性），脚本镜像存于证据目录。

### 6.1 LC-65 补遗 (2026-10-07 独立评审后续): 本征残差、中心曲率与
定态性诊断的修正与补强

独立评审 (review-20261007-followup.md) 正确指出 LC-1 的两处方法学
缺陷：(a) 报告的 `abs(v.T@H@v-lambda)` 是 Rayleigh 一致性检查而非
本征向量残差 `norm(Hv-lambda*v)`；(b) auto 支线小振幅能量扰动由
奇/线性项主导。LC-65 用严格对称的中心差分与真实本征残差重做诊断
（证据 `tasks/issue-7/2026-10-07-lc65-stationarity-eigenresidual/`，
sha256 锚定；GPU 道密度在 fcde3cd 重测，LC-1 记录复现 <=8.5e-14）：

- **真实本征残差（Rayleigh 值的修正）**：fp64 支线
  `norm(Hv-lambda*v) = 6.18e-7`（|lambda|=1.67e-5 的 3.7%），auto
  支线 3.44e-7（2.7%）——不是 LC-1 报的 2.5e-16（那是
  `v.T@H@v - lambda` 自洽残差，机器精度只说明 Davidson 内洽，不说明
  本征方程满足）。残差下限的机制已定位：pyscf Davidson 收敛判据为
  `|de| < tol 且 dx_norm < sqrt(tol)`（`linalg_helper.py:502`），
  tol=1e-12 只要求更新范数 <1e-6，实测残差 6.17e-7 恰在该界内；
  双线性对称探针（12 对随机向量）证明算子本身**严格对称**
  （max|<x,h(y)>-<y,h(x)>| = 7.8e-16，相对 1.9e-18）且确定性
  （双重求值差 3.8e-16）——即本征**值**已收敛（|de|<1e-12，跨
  tol 1e-9/1e-11/1e-12 一致到 2e-14），本征**向量**残差受
  Davidson 宽松停机判据限制而非算子噪声。
- **紧残差 Davidson 给出符号判定的数学封闭**：由于算子严格对称
  （Hermitian），对单位近似本征对有
  `|lambda - lambda_true| <= ||Hv - lambda v||`。用显式
  `tol_residual=1e-9` 重解（`davidson1`，绕过 sqrt(tol) 宽判据）：
  fp64 支线 lambda=-1.669252e-5，残差 **3.57e-8**（误差界为
  |lambda| 的 1/467）；auto 支线 lambda=+1.272527e-5，残差
  **2.22e-8**（1/574）——两支线的符号分类在误差界内**不可能
  翻转**（次软本征值 0.316，模式分离 4 个量级）。第二软模的
  确认使 §6 鞍点/极小判定从"Davidson 收敛值"升级为"带紧误差界
  的数学事实"。**586×586 稠密精确对角化已完成（两支线）**：fp64
  支线 lambda_exact=-1.669252e-5，本征残差 2.58e-14（机器精度），
  Davidson-稠密偏差 +1.30e-13，模式重叠 0.99999999999994；auto
  支线 lambda_exact=+1.272527e-5，残差 2.58e-14，偏差 +2.56e-14，
  重叠 0.99999999999999；两支线算子对称缺陷 <=4.7e-14（相对），
  次软本征值 0.316——符号分类由稠密精确解最终确认，Davidson 的
  宽停机判据只影响向量残差（6.2e-7），对本征值的影响 <=2.6e-13。
- **奇/线性项的定量归因（评审点 b 的答案）**：严格反对称轨道旋转
  （`newton_ah.expmat`，R(-t)=R(t)^T）下的中心曲率
  k_c(t)=[E(+t)+E(-t)-2E(0)]/t^2 完全消除奇宇称项：fp64 支线奇项
  ~1e-13（纯噪声），k_c(t)=lambda+c4*t^2 加权拟合给出
  lambda_fit=-1.667e-5，与 Davidson -1.669e-5 一致（差 2.1e-8，
  符号相同）；auto 支线 lambda_fit=+1.276e-5 vs Davidson +1.273e-5
  （差 3.6e-8，符号相同）。LC-1 观察到的"奇宇称信号"定量等于
  残余梯度泄漏 ~2·||g||·t（auto 支线 ||g||~6.2e-6，t=0.1 时
  1.2e-6，与实测奇项 1.0e-6 同量级）——是**定点未完全收敛的梯度
  泄漏，不是势能面的真实线性项**。大振幅（t=0.1..0.4）中心曲率
  符号与 lambda 一致性：auto 支线 3/3 一致；fp64 支线在 t>=0.1 的
  有限振幅下中心曲率为正（+3.6e-5..+1.1e-3）但 t->0 拟合极限为
  负——鞍点下降方向极软（|lambda|~1.7e-5），t=0.1 时四次项
  c4*t^2~4.6e-3*t^2 已淹没二次项；小振幅扫描（t<=0.05）才恢复
  负曲率。这是软模鞍点的预期行为，不推翻判定。
- **定态性收紧尝试（两条路都到不了更紧的定点）**：auto 支线
  平面 DIIS 无法收紧到 conv_tol 1e-12（100 圈不收敛，能量已在道值
  1.9e-8 内——软模平坦盆地使 DIIS 失速）；Newton 收紧则**离开
  auto 盆地**到第三个点（+1.71e-6 Ha，dm 距 auto 0.075/距 fp64
  0.295，自身收敛失败且其最软 lambda=-8.93e-5）。即 auto 支线
  "稳定极小"的定态性只到其道底 ||g||~6.2e-6，无法用标准收紧工具
  进一步证明；其极小性判定依赖该定态性水平下的 Davidson lambda
  与中心曲率符号一致（上一条）。如实记录，不升级为"已证明全局
  极小"。
- **网格稳健性（9/9 完整）**：fp64 支线最软 lambda 在 grids×nlcgrids
  level (2,3,4)² 全部 9 格为负：-3.91e-5(g2/n2)…-1.67e-5(g3/n3)
  …-8.57e-5(g4/n4)，nlcgrids 变化影响 <=0.09%（NLC 响应贡献小），
  grids level 变化改变幅度 2-5 倍但**符号从不翻转**——鞍点判定
  网格稳健。每格本征残差均在 6.17-6.18e-7（算子残差下限，见上）。
- **§6 判定的最终状态**：fp64 支线鞍点判定被中心曲率 t->0 拟合、
  9/9 网格稳健性、真实本征残差（3.7% 相对）与稠密精确对角化
  （lambda_exact=-1.669252e-5，残差 2.6e-14）四方确认，置信度
  上调；auto 支线极小判定在道底定态性水平成立（稠密精确解
  lambda_exact=+1.272527e-5 同样机器精度），其收紧限制如实
  记录。§6 "归属修正"结论（fp64 参照道收敛到鞍点、1e-6 门比较
  对象在该格不是极小）不变。

复现：`stationarity_eigenresidual.py`（stage B）、
`central_curvature_scan.py`（stage C）、`stageB_completion.py`/
`stageB_dense_continuation.py`（评审后续补全）、
`operator_symmetry_probe.py`（对称性/确定性）、
`tight_residual_davidson.py`（紧误差界），命令与原始输出
均在证据目录。

## 7. LC-25 补遗 (2026-10-07 长链工作单元): 全基准矩阵 UKS 稳定性分类

LC-1 只分类了 wb97m-v/methoxy/def2-svp 一格。LC-25 用同一套已验证方法
（GPU 道收敛密度转储 → CPU fp64 重启 → 泛函适配响应的轨道 Hessian；
证据 `/data/results/gpu4pyscf-agent/tasks/issue-7/2026-10-07-lc25-stability-matrix/`，
sha256 锚定，含 3 个过程 bug 的诚实记录）把其余 5 个 UKS 格全部分类。
10 次道密度转储全部复现其记录能量（<=6e-9 Ha）。

| 格 | 支线 | 分裂 (Ha) | fp64 道判定 | auto 道判定 | 低者 |
|---|---|---|---|---|---|
| methoxy/svp/wb97m-v | 两条 | -1.28e-06 | **鞍点**（含 NLC 不稳定） | **稳定极小** | auto（LC-1） |
| methoxy/tzvp/wb97m-v | 两条 | +2.60e-08 | 稳定（NLC 面 +8.1e-5） | 稳定（+7.4e-5） | fp64 |
| methoxy/svp/b3lyp | 两条 | +4.03e-08 | 稳定 | 稳定 | fp64 |
| methoxy/tzvp/b3lyp | 两条 | +4.40e-08 | 稳定 | 稳定 | fp64 |
| methoxy/svp/m062x | 同一支线 | -3.9e-12（CPU 重启对；GPU 道对 -2.8e-12） | **不稳定** | **不稳定** | （同） |
| methoxy/mtzvpp/r2scan3c | 同一支线 | -3.7e-13（CPU 重启对；GPU 道对 +9.0e-12） | 稳定 | 稳定 | （同） |

（LC-51 勘误：原表 m062x 行未声明 CPU/GPU 基准、r2scan3c 行写
+1.7e-13——主证据（lc25 stability_classification.json /
r2scan3c_mtz.json / runlog）中不存在该值；CPU 重启对分裂为
-3.695e-13（符号与幅度均已按主证据改正），GPU 道对为
+8.995e-12。两行的"同一支线、稳定/不稳定"判定与 dm 相对差
（5.2e-6 / 1.5e-5）不受影响。）

新事实（不改任何基准行判据）：

1. **wb97m-v/tzvp：两条支线在含 NLC 曲面上都是稳定极小**；stock
   无-NLC Hessian 把两条都误判不稳定（-1.0e-4）——LC-1 的 "NLC 泛函
   必须带 VV10 响应" 规则在第二个基组上复现。fp64 道支线更低，是该格
   正确的比较参照；auto 道 +2.6e-8 Ha 偏移是门内精度路径效应，非支线
   问题。
2. **b3lyp 两个基组：两条互异且都稳定的极小**，fp64 道低
   4.0e-8/4.4e-8 Ha。auto 道落在同一泛函面的另一局部极小上——
   `auto+verify` 的"取低收敛解"规则在此类格上恰好选对。
3. **r2SCAN-3c methoxy UKS：单一稳定解**，两条道恒等（CPU 重启对
   分裂 -3.7e-13 Ha / dm 相对差 5.2e-6；GPU 道对 +9.0e-12，同为门
   内噪声级）。干净格。轨道 Hessian 曲面是纯 r2scan（D4/gCP
   为几何可加修正，不在 SCF 算符内——runner 源码验证）。
4. **m062x methoxy UKS：基准解本身是鞍点**（两道同支线，最软本征值
   -3.559e-3 Ha，比 wb97m-v 鞍点大 ~200 倍，beta 主导模）。沿最软模
   跟踪收敛到**稳定极小，低 2.628e-5 Ha**（-114.898236033025 对
   -114.898209749161，S^2 两解同为 0.7528，C1 无对称性限制可破）。
   混合精度门不受影响（两道同支线，dE -3.1e-12 PASS）——受影响的是
   参照解本身的物理解释：该格的 UKS 能量参考是"错误电子解"意义上
   的鞍点（定态、梯度自洽，但非该 UKS 面的基态解），高出级联极小
   26 倍于 1e-6 门。
   该发现**回溯解释** issue-12 的 m062x FD 异常（所有位移 SCF 都落在
   中心之下，max|d| 9.1e-3）：鞍点中心在任何位移方向都下坡逃逸。
   issue-12 "FD 不适用于 UKS methoxy 验证"的结论得到更强机制支持，
   benzene FD 验证不受影响。
5. 支线判别：能量分裂（<=1e-9 Ha）+ 自旋密度相对 Frobenius 差
   （<=1e-4）为操作判据（b3lyp 支线对 4e-8/9.7e-3 对 m062x/r2scan3c
   的 4e-12/1.5e-5——两个数量级的干净分离）；AO 交叠度量下的占据子
   空间 SVD 作为佐证（互异支线也 >=0.9996，旋转角 ~1e-2 rad）。

至此 LC-25 覆盖的每个 UKS 格都有稳定性分类（LC-28 补上遗漏的
m062x/tzvp 格，见 §7.2）。用户待决事项（1e-6 门在解多重格上的参照
改写）现在有完整的物理输入：只有 wb97m-v/svp 一格的 fp64 道是鞍点；
wb97m-v/tzvp 与 b3lyp 各格两支线皆为极小；m062x 两基组格参照解皆
为鞍点但与精度无关；r2scan3c 干净。

复现：`dump_lane_dm.py`（10 次转储）、`stability_matrix.py`
（增量持久化、可续跑）、`m062x_saddle_follow.py`、`m062x_cascade.py`
（级联到稳定极小）、`make_classification.py`（汇总表），全部镜像在
证据目录；attempt-1 度量 bug 归档于 `attempt1-metric-bug/`。

### 7.1 LC-26 补遗 (2026-10-07 长链工作单元): m062x 稳定极小的 GPU 道一致性

LC-25 找到的 m062x 级联稳定极小（低 2.628e-5 Ha）在 GPU 道上的行为
（证据 `tasks/issue-7/2026-10-07-lc26-m062x-min-gpu/`，3 样本/道）：

- **GPU fp64 从极小密度重启：留在极小**（+1.1e-12 Ha，1 圈）——低解
  的 CPU/GPU 实现一致性确认。
- **GPU auto 道（fp32 头）从极小密度重启：同样回到极小**（3 圈，
  +1.1e-11 Ha）——低解是 auto 策略的精度稳健吸引子；基准停在鞍点
  纯粹是 minao 入口轨迹问题，与精度无关。
- **auto+verify 从 minao 进入：诚实选择鞍点**（3 样本皆 selected=main
  ——两条轨迹同从 minao 盆地出发都落鞍点，`distinct_solutions=false`）。
  双轨迹策略能发现 fp32 头引起的盆地分裂（wb97m-v/svp、b3lyp 各格），
  但**不能发现级联型不稳定**——入口盆地本身即鞍点的情形。发现该极小
  需要稳定性 Hessian 模式跟踪（LC-25 路线，43 AO 约 40 s CPU），属
  独立工具，刻意不在生产 SCF 路径内（本阶段范围无 Hessian）。

结论：m062x 格的"参照解应是哪个"已为用户决策完整刻画——基准鞍点
定态自洽、门内比较有效；更低的稳定极小存在、CPU/GPU 一致、精度
稳健，但只能经稳定性跟踪到达。基准行判据不变。

### 7.2 LC-28 补遗 (2026-10-07 长链工作单元): m062x/def2-tzvp —— 鞍点与级联极小的基组稳健性

LC-25 的 UKS 分类漏了一格能量基准矩阵格：m062x/methoxy/def2-tzvp
（LC-25 的 CELLS 表止步于 svp；该格 GPU issue-6 文档 auto 道低
4.09e-8 Ha，恰为 b3lyp 互异支线分裂的量级，需分类）。LC-28 补齐
（证据 `tasks/issue-7/2026-10-07-lc28-m062x-tzvp-stability/`，
sha256 锚定；方法与 LC-25/LC-26 逐字相同）：

- **鞍点是基组稳健的**：tzvp 最软本征值 -2.921e-3 Ha（svp -3.559e-3），
  同一 beta 主导模（beta_share 0.951），两 GPU 道同支线（重启分裂
  -4.4e-12）。
- **级联极小同样稳健**：跟踪最软模收敛到稳定极小（+3.195e-3 Ha），
  低 **2.687e-5 Ha**（svp 2.628e-5——两个基组几乎同深度），S^2
  0.7537 不变，18 圈收敛。
- **GPU 道在极小上一致**（LC-26 语义的 tzvp 重演，3 样本确定性）：
  fp64 重启留在极小（+1.6e-12 Ha，1 圈）；auto 道回到极小
  （+1.6e-11 Ha，4 圈）。低解在第二个基组上同样是 auto 策略的
  精度稳健吸引子。
- **auto+verify 策略行为（LC-28 实测，3 样本确定性）**：与 svp 格
  （LC-26：`distinct_solutions=false` 的干净 no-op）不同，tzvp 格上
  策略的密度差判据被触发——`dm_max_abs_diff=1.16e-3`（阈 1e-3，
  仅高出 16%）——因为 fp32 定点位移本身（沿不稳定模 1.96e-3 相对
  密度差）大于阈值；`dE_ref_minus_main=+4.09e-8`，"取低收敛解"规则
  **选择了非定态的 auto 点**（selected=main）。记录诚实地保留了
  被采纳点的轨道梯度范数（8.9e-6，对照参照解 3.4e-6——fp64 面在
  位移点上的残差抬升），下游读者可见。即：两轨迹策略在此类
  "鞍点+fp32 定点位移"格上把位移误读为支线分裂并采纳非定态点；
  该行为不违反任何门（auto+verify 的比较仍是收敛解对收敛解，
  -4.09e-8 在 1e-6 门内 PASS），但"verified 解为 fp64 面定态点"
  的性质在该格上不成立。
- **新现象（诚实记录，G1 门 auto 道不通过）**：GPU auto 道本身停在
  鞍点下方 4.09e-8 Ha 处（沿不稳定模的位移方向——鞍点在该方向是
  局部极大），从其密度的 CPU fp64 重启**爬回鞍点**（8 圈，+4.09e-8）。
  即 auto 道收敛点在严格 fp64 面上**非定态**：fp32 头把它停在鞍点
  的下坡沿上。fp64 道重启复现门通过（-6.4e-12）。该格的
  auto-vs-fp64 门比较（GPU 文档 -4.09e-8，PASS 门内）不受影响——
  两道都在门内、都报告为鞍点支线的解；但"auto 道收敛点严格
  定态"这一性质在该格上不成立，与 svp 格（LC-25：两道都精确
  定态于鞍点）不同。
- 结论：m062x 家族（两基组）参照解均为鞍点、级联极小均存在且
  精度稳健、深度几乎基组无关（2.6-2.7e-5 Ha）。用户门改写决策
  的物理输入现在覆盖 m062x 的两个基组格。

复现：`dump_lane_dm.py`（两道密度转储）、`m062x_tzvp_stability.py`
（分类+级联，增量持久化）、`dump_cascade_dm.py`、`min_lane_test.py`
（GPU 道极小一致性，3 样本/道）、`autoverify_test.py`（auto+verify
策略行为，3 样本），全部镜像在证据目录。

## 7.3 LC-29 补遗 (2026-10-07 长链工作单元): auto+verify 位移/分裂判别 —— 有界 fp64 精化交叉重启

LC-28 的 auto+verify 测量暴露了策略代码的一个已测缺陷：密度差判据
（`dm_max_abs_diff > 1e-3`）无法区分"fp32 定点位移"与"真支线分裂"
——在 m062x/methoxy/def2-tzvp 鞍点格上，fp32 位移本身（1.16e-3）就
超阈，"取低收敛解"规则随即采纳了**非定态**的 auto 点（轨道梯度
8.9e-6，对照 fp64 面 3.4e-6 基线）。本单元把 issue #7 诊断阶段使用的
交叉重启检验**升级为策略本身的一等判据**：

**新判据**（仅当密度差测试触发时执行，默认路径逐行未动）：从 auto
解密度再跑一条**有界严格 fp64 精化 SCF**（默认 12 圈上限
`stability_verify_refine_max_cycle`）：

- 精化爬回参照密度（`dm_max_abs_diff_refined_vs_ref <= 1e-3`）→
  两轨迹本就是**同一解**，auto 点是 fp32 位移 → 采纳定态参照点，
  记录 `fp32_displacement_detected=True`、位移能
  `auto_point_displacement_e` 与被弃位移点全记录；
- 精化留在异盆地 → **真分裂**：比较两个**定态**点（精化 auto vs
  参照），取更低者（`adopted_stationarity_verified=True`）；
- 有界精化不收敛 → 诚实回退到原始两轨迹比较，
  `adopted_stationarity_verified=False`。

**四决定性格实测**（3 车道 × 3 独立样本，CUDA 同步，预测先行登记于
长链检查点；证据
`tasks/issue-7/2026-10-07-lc29-autoverify-refinement/`）：

| 格 | 精化结果 | 采纳 | dE(auto+verify vs fp64 道) | 前 |
|---|---|---|---|---|
| m062x/methoxy/tzvp | same_solution（8 圈爬回鞍点） | 参照鞍点 | **≤8.5e-14** | -4.09e-8（非定态点） |
| wb97mv/methoxy/svp | distinct（3 圈留 auto 盆地） | main=精化 auto 支线 | **-1.283e-6（诚实保留）** | -1.283e-6 |
| b3lyp/methoxy/svp | distinct（1 圈定态） | 参照支线 | **0.0** | +4.0e-8 |
| b3lyp/methanol/svp | 未触发（dm 2.0e-7） | main 不变 | 1.4e-13 | 同 |

- **G1 采纳定态性（tzvp）**：被采纳鞍点的轨道梯度回到基线
  （精化后 1.98e-07，对照参照 3.4e-6）；位移点 8.9e-6 的记录进入
  `displaced_auto_point` 留档。精化 dm 差 1.9e-6 与参照解吻合。
- **G2 无回归（wb97mv/svp）**：真分裂仍被识别（占据空间 SVD：
  alpha 0.9998 / **beta 0.2576** —— 正是 §2 记录的 beta 交换签名，
  判据与 SVD 佐证交叉一致）；精化 3 圈定于 auto 支线能量
  （-114.901696296），车道差 -1.283e-6 Ha 原样保留——**两定态支线
  真实相差该值，修复不掩盖、不改判**。
- **G3 b3lyp**：参照支线更低，采纳后 dE=0.0（原文档 4.0e-8 的
  解选择误差按预期消除）。
- **G4 闭壳层**：密度测试不触发，无精化键、能量不动（对 auto 道
  1.4e-13）。
- **G5/G6**：全部车道×样本收敛；不可变基准判定无一改动。

**开销**：触发格多付一条有界 fp64 SCF（实测精化 1-8 圈，从已收敛
邻域密度出发）；未触发格零新增开销。

**语义变化（诚实声明）**：auto+verify 的承诺从"两条轨迹中更低者"
收紧为"**两条轨迹可达的更低定态解** + 完整决策记录（含位移检测、
占据空间 SVD、精化能量/圈数）"。已测的 wb97mv/svp -1.283e-6 Ha
超门 fail 属于两定态支线的真实能差，**不是**缺陷修复对象——该格
的精确匹配路线仍是 fp64 SCF（可选 fp32 梯度内核）。模块记录为
向后兼容超集（新增键，旧键语义不变）。

复现：`refinement_cells.py`（增量持久化、逐格打印、期限保护），
`refinement_cells.json`（12 样本全记录），`confirm_sample.txt`
（精简后代码的逐位复现），`evidence_sha256.txt` 锚定源文件哈希。

**梯度后果（LC-30，2026-10-07）**：该修复不只是定态性簿记——
m062x/methoxy/tzvp 格的 auto 车道梯度误差实测 **1.95e-4 Eh/Bohr**
（超 3e-5 门 6.5×，完全来自解位移项；内核项仅 8.8e-7），修复后
auto+verify 交付 **1.09e-6**（采纳定态鞍点；180× 改善）。详见
issue-12 文档 §7 与
`tasks/issue-12/2026-10-07-lc30-autoverify-grad-consequence/`。

### 7.3.1 补全 6 格 methoxy UKS 矩阵 + 开销（LC-31，2026-10-07）

LC-29 代码下剩余两格（m062x/svp 无需重测：其密度测试不触发
LC-26，且不触发路径逐字节未改）。预测先行登记；证据
`tasks/issue-7/2026-10-07-lc31-autoverify-remaining-cells/`：

| 格 | 精化 | 采纳 | dE(auto+verify vs fp64 道) | dE_ref−main | 精化圈数 |
|---|---|---|---|---|---|
| wb97mv/tzvp | distinct | reference | ≤5.7e-14（旧码 0.0 复现） | -2.600e-8 | 1 |
| b3lyp/tzvp（首次实测） | distinct | reference | ≤1.1e-13 | -4.400e-8 | 1 |

- 两格的 dE_ref−main 与 issue-7 原 `stability_selection.json`
  （-2.60e-8 / -4.4e-8）**逐位吻合**——新判据路径复现旧码结果。
- 占据空间 SVD：beta 0.99967 / 0.99957（旋转型支线，非交换；
  与 §2 b3lyp 0.9996 先例一致——小能差支线对仍可产生大梯度差，
  见 #12）。
- 物理注记：这两格的精化 1 圈即收敛且留在原盆地（auto 点本身
  已是本盆地定态点），与 m062x/tzvp 的 8 圈爬回（auto 点非定态）
  形成两类情形的实测对照。
- **开销（中位墙钟，全部 6 格自动+verify/auto 比）**：svp 闭壳
  2.16x；wb97mv/svp 1.97x；b3lyp/svp 2.48x；m062x/tzvp 2.64x；
  wb97mv/tzvp 2.35x；b3lyp/tzvp 2.49x——触发格付参照道+精化道
  两条额外 fp64 SCF（精化 1-8 圈）；不触发格仅参照道。
  （旧码开销 1.65-2.34x；新增精化道使其升至 ~2.0-2.6x，如实
  报告。）

### 7.3.2 旋转型支线族的梯度后果（LC-33，2026-10-07）

§7.3.1 的两个旋转型格（wb97mv/tzvp、b3lyp/tzvp）此前只有能量
记录；LC-33 在 `#12` 文档 §8 补齐了梯度后果：auto 车道能量误差
仅 2.6e-8/4.4e-8 Ha（远小于 1e-6 门）但梯度误差 1.02e-3/
1.22e-3 Eh/Bohr（超 3e-5 梯度门 34-41 倍，全部为解项）；解项 /
dm_max_abs_diff 的实测放大率 0.183/0.188（tzvp 族聚集
0.17-0.19，svp 旋转型 ~0.10，m062x/tzvp 位移型 0.168）。
auto+verify 采纳 reference 支线后两格恢复 1.02e-6/8.06e-7。即：
本节 §7.3.1 表格中"精化 1 圈留在原盆地"的支线（auto 点本身
定态）同样携带大梯度差——**支线的能量接近不等于梯度接近**，
与 m062x/tzvp 位移格（非定态）殊途同归。dm×0.2 启发式可用于
未来格的预登记（详见 `docs/issue-12-gradients.md` §8，证据
`tasks/issue-12/2026-10-07-lc33-rotation-family-grad/`）。

## 8. LC-76/77 补遗 (2026-10-12 独立评审续): 严格 fp64-参照匹配路线 `auto+strict`

评审 2026-10-12 指出：auto+verify 的"分裂时选较低定态解"不是
"匹配意图 FP64 轨迹"的契约——数值接近的能量也不证明梯度一致，
且把 FP64 鞍点换成更低极小值不构成精度通过。LC-76（提交
9c23651）实现该评审要求的**最小显式严格参照/安全回退路线**：

- `precision_mode='auto+strict'`（`stability_verify.
  verify_reference_match`）：auto 策略 + **一条**同初始猜测的
  严格 fp64 参照 SCF（与 auto+verify 共用 `_build_fp64_sibling`
  全套旋钮/网格定义/色散钩子携带）。dm 差异超过容差时
  **无条件采纳参照解**（即使能量更高——这正是防鞍点替换契约），
  并诚实记录被拒绝的 auto 点（能量/圈数/轨道梯度）。参照未收敛
  或非有限时**fail-closed**：保留收敛的 main 解，
  `strict_match_verified=False` + 精确原因
  （reference_not_converged / reference_energy_nonfinite /
  dm_comparison_nonfinite），绝不比较、绝不断言同解、绝不采纳。
- 梯度侧 `_resolve_precision`：'auto+strict' 与 'auto+verify'
  同义——路线选择的是解，不是核精度（显式覆盖仍优先）。
- 基准车道 `auto-strict` + 验证器规则 **G7**：任一 repeat 的
  认证不完整（或缺失）则该行 verdict 必为 fail——未认证的解
  绝不是"已认证的 fp64 道匹配"。

LC-77（证据 `tasks/issue-7/2026-10-12-lc77-strict-route-proof/`，
59 文件 sha256 锚定，预登记 P1-P5+PD1-PD3）在生产 runner 上以
全部 repeat 证明该路线（推荐配置 16384 五格 + 缺省块尺寸两格
对格）：

| 格 | 配置 | auto 道 | auto-strict | 契约 |
|---|---|---|---|---|
| b3lyp/methoxy/svp UKS | 16384 | FAIL 1.343e-3 | **PASS 1.585e-6** 参照采纳 | 恢复族 |
| wb97mv/methoxy/svp UKS | 16384 | FAIL 3.005e-2 | **PASS 1.943e-6** 参照采纳 | 第三支线判别 |
| b3lyp/benzene/svp | 16384 | pass | **PASS 1.774e-6** 不触发 | 闭壳对照 |
| r2scan3c/methoxy/mtzvpp UKS | 16384 | pass | **PASS 1.516e-6** 不触发 | 3c 哨兵（D4+gCP 携带） |
| m062x/methoxy/tzvp UKS | 16384 | pass 1.589e-6 | **PASS 1.589e-6** 不触发 | 16384 下位移不触发（新发现） |
| m062x/methoxy/tzvp UKS | 缺省 | FAIL 1.952e-4 | **PASS 1.091e-6** 鞍点采纳 | **位移格: dm=1.1606e-3 触发（逐位复现 LC-29），位移点被拒 +4.086e-8** |
| wb97mv/methoxy/svp UKS | 缺省 | FAIL 3.691e-2 | **PASS 1.942e-6** 参照采纳 | **反鞍点替换: auto 支线低 1.2826e-6 却采纳更高的 fp64 道参照（每个 repeat）** |

- 采纳分数 12/21（恰为失效族诸格；不触发格保留 main 解）。
- **开销（中位端到端，诚实的全代价）**：auto-strict / 匹配 fp64
  = 1.79-2.94x；/ auto = 1.64-1.96x——一条完整严格 fp64 参照
  SCF，无精化道。按评审要求如实记录为安全开销，非加速；缺省
  块尺寸 wb97mv 格 2.94x 为最贵（参照 SCF 11.3s 中位）。
- 5 份合并文档（含 G7）全部 schema-VALID；auto 道诚实 fail 行
  原样保留。
- **新发现（块尺寸依赖）**：m062x/tzvp 位移仅在缺省 XC 网格
  块尺寸触发（16384 下 auto 轨迹自身收敛到鞍点，dm=9.3e-7），
  而 wb97mv/svp 的能量排序随块尺寸翻转（16384 下 fp64 支线反而
  低 4.25e-7）。fp32 轨迹景观对配置双向敏感——安全网必须在
  任何配置下可用，这正是本路线的存在理由。
- 该路线是**新增独立车道**；issue-7/-12 的历史失效行不变
  （immutable），不因新路线存在而改判。
