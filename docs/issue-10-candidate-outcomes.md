# Issue #10 — 热点受控混合精度：候选处置结果

**状态**: review_requested（暂用，未验收）· **证据**:
`/data/results/gpu4pyscf-agent/tasks/issue-10/`（sha256 锚定清单见
`evidence_sha256.txt`）· **源码**: `f85737c` + 本 issue 提交
（C1 内核 + 分析工具 + 测试；C2 已回退，拒绝实现保留为
`c2_rejected_implementation.patch`）

按 #9 候选清单的执行顺序（C1 → C2 →（C3/C4 等 #11）→ C6 已否决），
本 issue 落实"先量精度再动代码"：每个候选先做小体系精度探针，再实现
（或否决），最后全矩阵复测。安全开关默认关，严格 FP64 回退保留。

## 1. C1 — VV10 UWE 内核多 warp 变体（实现为 opt-in，默认关闭）

**候选来源**: #9 实测 ωB97M-V 的 77–84% kernel 时间在 NLC/VV10，
其中 93% 在单个 kernel `vv10_fock_eval_UWE_kernel`（benzene/def2-svp
5.777 s / 9 次调用）。参考内核每 block 只用 1 个 warp 走 j-循环。

**实现** (`gpu4pyscf/lib/gdft/vv10.cu`, commit a25a599):
- 新增 `vv10_fock_eval_UWE_multiwarp_kernel`：block→i 映射不变（ngrids
  个 block），j-循环分摊到全部 128 线程，per-i 求和用 block 归约；
  **fp64 算术表达式逐项不变**，与参考内核的唯一差别是 fp64 求和顺序
  （树归约 vs 串行链）。
- 选择器 `GPU4PYSCF_VV10_UWE_KERNEL`（进程级缓存一次）：未设/任意值
  = 参考 warp1 内核（**默认不变**）；`warp4` = 新内核；未知值打印警告
  并忽略。Python 侧不读该变量（单测钉死），opt-in 不改变任何其它路径。
- 严格 FP64 回退：默认路径就是历史 warp1 内核，逐位不变。

### (i) 精度探针（`c1_precision_probe.json`）

4 个已准入 cell（methanol/S0、methoxy-UKS/S1、benzene/S1、benzene/S2），
内核级直发复算（捕获 `_vv10nlc` 输入后直接调 libgdft 绑定）+ SCF 级：

| 量 | 实测（最大值） |
|---|---|
| 内核级 dU/dW/dE_grid（warp4−warp1） | 8.1e-18 / 8.1e-17 / 4.3e-16 |
| SCF 级 e_tot 差 | ≤ 5.7e-14 Ha |
| 逐位可复现（每变体两次运行） | 全部 bitwise 一致 |
| 探针预处理 vs `_vv10nlc` 真路径 | bitwise 一致 |

归约顺序噪声在 fp64 下 ~1e-16 量级，距 1e-6 门 10 个数量级；一阶落在
E_nlc、无下游衰减（§2.3 规律），风险判定为低，准予实现。

### (ii) 全矩阵复测（`c1_retest_*` + `c1_retest_analysis.json`）

5 个已准入 ωB97M-V cell × {warp1, warp4} × {fp64, auto} 车道，
rtx4090_runner 逐格独立计时（3 样本 + 弃置预热，源 commit 同为
a25a599，设置与 #7 生产运行逐项一致）。**验收量**：每变体内部
auto vs fp64 的 |dE| ≤ 1e-6 Ha；**诊断量**：warp4−warp1 能量差
（归约顺序噪声）与端到端中位时间比。逐格数值见 JSON 与 §3 汇总表。

**结论**: warp4 在已准入尺寸（48–222 AO，ng 57k–144k）上端到端收益
有限且不一致（内核微基准 methoxy 1.26x、其它 1.02–1.07x；端到端
-5.6%..+15%，见 §3 汇总表）——收益**不随**网格增大：warp4 的每-i
j-循环并行度优势在小网格兑现，大网格下其 128× j-网格 staging 流量
让内核转为带宽受限（§3 机理）。**不上默认**；作为 opt-in 保留，
使用边界见 §3。

## 2. C2 — DIIS 在精度切换时重置（否决，已回退）

**候选来源**: #7 deferred：methoxy/def2-svp UKS ωB97M-V auto 车道
3/3 收敛到比 GPU-fp64 低 **-1.28e-6 Ha** 的 DIIS 外推固定点
（门内差 1.28e-6 > 1e-6，issue #7 唯一精度门失败格）。

**探针**（`c2_diis_reset_probe.json`，3 系统 × 3 独立重复）:

| cell | auto 默认 max\|dE\| | auto+重置 max\|dE\| | 判定 |
|---|---|---|---|
| methoxy/def2-svp UKS（失败格） | 1.2826e-6 | **1.2830e-6** | 重置未消除偏移 |
| methoxy/def2-tzvp UKS（通过格） | 2.60e-8 | 2.60e-8 | 不扰动已通过格 |
| benzene/def2-svp（闭壳层对照） | 4.5e-13 | 1.2e-12 | 不扰动闭壳层 |

fp64 守卫格（开关打开的纯 fp64 运行）与默认 fp64 一致（仅 SCF 尾部
噪声量级差异）。**结论**: 偏移不是 fp32 头留在 DIIS 子空间的误差向量
所致——重置子空间后固定点几乎不动（-1.2826e-6 → -1.2830e-6），且
重置格多花 1 个循环（20 vs 19）。该假设被证伪，候选否决，实现回退
（diff 留档 `c2_rejected_implementation.patch`）；#7 deferred 项保持
unresolved，真实机制仍在排查（见 §4 后续建议）。

## 3. C1 全矩阵复测汇总表

（由 `c1_retest_analysis.py` 从 schema 文档 derive，`c1_retest_analysis.json`
sha256 锚定为权威源；dE_auto 为该变体内 auto vs fp64；speedup = 端到端
中位时间比 warp1/warp4。）

| cell | AO | dE_auto(warp1) | dE_auto(warp4) | \|dE(w4−w1)\| fp64 | speedup fp64 | speedup auto |
|---|---|---|---|---|---|---|
| s0_methanol_svp | 48 | 5.7e-14 | 5.7e-14 | 5.7e-14 | **1.152** | 1.033 |
| s1_benzene_svp | 114 | 5.1e-13 | 4.6e-13 | 1.1e-13 | 1.001 | 1.021 |
| s1_methoxy_svp_uks | 43 | **-1.283e-6** | **-1.283e-6** | 8.5e-14 | 1.068 | 1.101 |
| s2_benzene_tzvp | 222 | 2.2e-12 | 6.5e-12 | 2.3e-13 | **0.944** | 0.988 |
| s2_methoxy_tzvp_uks | 80 | 2.6e-8 | 2.6e-8 | 8.5e-14 | 1.031 | 1.088 |

读法与处置：

- **精度**：变体间差（归约顺序噪声）≤ 4.6e-12 Ha，距 1e-6 门 6 个
  数量级。4/5 cell 的变体内 auto 门通过（≤ 6.5e-12）；methoxy/def2-svp
  UKS 两个变体同为 -1.283e-6——这是 #7 遗留 deferred 失败的**原样复现**
  （变体间差 ~1e-12，内核更换既不修复也不恶化它），该格 verdict 仍为
  fail，从未改标签。
- **速度**：端到端收益小且不一致——methanol fp64 +15%、methoxy 格
  +3~10%、benzene/svp ±2% 以内、**benzene/tzvp fp64 -5.6%（负加速）**。
  机理：warp4 每个 block（单个 i）都要把整个 j-网格搬进 shared memory，
  全局读放大 128×（warp1 每 block 服务 128 个 i）；小网格时串行依赖链
  缩短占优，大网格时带宽饱和反超。#9 候选表预估的 "kernel 2x → 端到端
  ~1.7x" 上限在实测 kernel 收益只有 1.02–1.26x 的情况下不可达。
- **处置**：warp4 **不上默认**（默认路径仍是逐位不变的 warp1 参考
  内核 = 严格 FP64 回退），保持 opt-in 实验开关；使用边界如实记录：
  仅小网格/小体系（S0–S1 methanol/methoxy 类，≤ ~60k 网格点）有
  1.05–1.15x 收益，benzene 级 143k 网格点已到收支平衡甚至为负。
  未达"全矩阵净加速为正"验收线的候选按计划回退并记录——本候选的
  回退形式即"默认关闭 + 边界文档"。

## 4. 未解决事项与后续

1. **#7 methoxy UKS -1.28e-6 Ha 固定点**：C2 否决后仍是唯一未过精度
   门的格。已排除：NLC 分量精度（R10 一致到 1.6e-8）、DIIS 子空间残留
   （本 issue）、收敛慢（#7 已证 per-cycle Δ 到 1e-10 的固定点特征）。
   下一步候选（#12 前必须解决或明确边界）：auto 头部 fp32 迭代本身把
   轨道锁进 UKS 不同布居解邻域（fp64 尾在同一盆地里收敛）——探针是
   fp32 头逐轮 MO 占据/布居对比 fp64 头，或 auto 头部直接从 fp64 minao
   初猜起跑的对照。在解决前，该格保持 fail（从未改标签）。
2. **C3/C4（网格 GEMM fp32）**：按 #9 清单等 #11 更大 AO 实测
   （S1/S2 每 cycle fp32 仅 1.1–1.3x，launch-bound），本 issue 不动。
3. **C5/C7（AO 求值、eig/DIIS）**: 份额 <16%/<7%，S1/S2 上限 <1.1x，
   不做（负结果预期明确，避免无意义工程）。
4. **C6（CDERI fp32 累加）**: §2.3 已否决（Cholesky 相消），不重做。
5. **C1 后续设计方向（记录，不实施）**：warp4 的全局读放大（每 block
   整个 j-网格 × ngrids 个 block）可用 i-分块方案消除——每 block 服务
   k 个 i（寄存器）× 128 线程分摊 j，staging 摊薄 k 倍；k=8..32 可望
   在 benzene 级网格上转正。属 #11 之后的内核工程，须重走
   探针→实现→全矩阵流程。

## 5. 开关文档

| 开关 | 位置 | 默认 | 作用 |
|---|---|---|---|
| `GPU4PYSCF_VV10_UWE_KERNEL` | `gpu4pyscf/lib/gdft/vv10.cu`（C 侧读取，进程级一次） | 未设 = `warp1` 参考 | `warp4` 启用多 warp VV10 UWE 内核（fp64 算术不变，仅求和顺序差异）；未知值警告并忽略 |

复现（内核微基准 + 探针 + 全矩阵）：
```bash
/home/qczcode/gpu4090-src/bin/python \
  /data/results/gpu4pyscf-agent/tasks/issue-10/c1_precision_probe.py   # 探针
bash /data/tmp/gpu4pyscf-agent/execution/c1_retest_matrix.sh           # 全矩阵
/home/qczcode/gpu4090-src/bin/python \
  /data/work/gpu4pyscf-mixedprec/mixedprec/benchmark/rtx4090/c1_retest_analysis.py \
  --out /data/results/gpu4pyscf-agent/tasks/issue-10/c1_retest_analysis.json
```
