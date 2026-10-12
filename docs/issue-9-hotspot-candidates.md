# Issue #9 — RTX 4090 热点归属结论与 #10 优化候选清单

**状态**: review_requested（暂用，未验收）· **证据**: `/data/results/gpu4pyscf-agent/tasks/issue-9/`
（`hotspot_attribution.json@32602f29bd38`、`nsys_kernel_attribution.json@2f72a6fff4ea`、
全部 17 个文件的 sha256 见 `evidence_sha256.txt@` 本 issue 私有证据目录）·
**源码 commit**: `50d2cbb` + 本 issue 的 profiler/归属模块/脚本
（两轮 attempt 运行同一份 profiler 代码；nsys 归属逻辑 attempt-2 修复两处
实况发现的 bug 后抽取为 `nsys_attribution.py` 并带回归测试）

## 1. 归属方法与自检

- **两级归属**: (a) 函数级 — `hotspot_profiler.py` 用 deviceSynchronize 包裹的
  包装器给每个函数桶计时（合成耗时只会偏大，不会偏小）；非插桩基线样本单独跑，
  比值记为 `sync_serialisation_overhead`（实测 1.04–1.30x）。(b) 内核级 —
  nsys + NVTX 桶，CUDA kernel 时长记入最内层包围它的函数桶。
- **验收门**: 20 条 lane（10 cell × fp64/fp32 背靠背）全部
  `[unattributed] ≤ 10%`（实测最大 7.1%，nsys 级 0%）；
  插桩不改变能量（H7 全过，逐位一致）；文档经 `validate_attribution()`
  校验 VALID（17 项单测含逐条违规拒绝样例）。
- **诚实记录**: benzene/def2-tzvp/wb97m-v 的 fp32 lane 是 attempt-2 进程补跑的
  （attempt-1 被 60 分钟 job 墙钟杀掉；profiler 按 lane 增量落盘，已完成 cell
  从不重跑）。每条 lane 自带独立弃置 warmup，JIT 状态不跨 lane 泄漏；cell note
  已注明。

## 2. 四方法成本画像（S1/S2 已准入 cohort，nao 43–222）

SCF kernel 占端到端 84–99%；mol build <1.6%。kernel 内部：

| 方法 | 主热点（占 kernel） | 第二 | DF JK | CDERI |
|---|---|---|---|---|
| B3LYP (114/222 AO) | XC 网格 68–72% | rho 收缩 19–27% | 4.5–7% | 端到端 13–16% |
| M06-2X (114/222 AO) | XC 网格 75–82% | rho 收缩 15–29% | 2.9–6% | 10–12% |
| ωB97M-V (114/222 AO) | **NLC/VV10 77–84%** | XC 网格 13–17% | 0.8–4% | 1–2% |
| r2SCAN-3c (144 AO) | XC 网格 71–83% | rho 收缩 20–32% | J-only 2.7–3.8% | 9–11% |
| UKS 哨兵（methoxy, 43–58 AO） | 同各自闭壳层画像（b3lyp 61–75% 网格；wb97m-v 57–62% NLC；3c 71–82% 网格） | | | |

- 网格内部：Fock 侧收缩（GEMM，"other"）29–53% of kernel > 密度收缩
  `eval_rho2` 14–32% > AO 求值 5–16% > libxc 求值 1–3%。
- 内核级（nsys，benzene/def2-svp 代表格）：非 NLC 方法的 `scf.xc.grid` 桶
  几乎全是 `cutlass_80_tensorop_d884gemm_*`（cuBLAS fp64 张量指令 GEMM）；
  **ωB97M-V 的 93% 插桩内核时间在单个 kernel `vv10_fock_eval_UWE_kernel`**
  （5.777 s / 9 次调用）。D4+gCP 合计 0.3 ms（0.02%），纯正确性项，无优化价值。
- 历史先验核对：5090 的 "r2SCAN SCF 72.8% 在 XC 网格" 在 4090 的 r2SCAN-3c
  上复现（71–83%）；**但该先验不迁移到 ωB97M-V** —— 它是 NLC 独占热点。

## 3. 负结果（实测，禁止无意识重做）

- **纯 fp32 lane 在 S1/S2 全部 10 格不收敛**（conv_tol 1e-10 低于实测 fp32 噪声
  底，50 cycle 上限烧尽）。且**每 cycle 仅快 1.1–1.6x**（kernel/cycle 基准，
  逐格重算自主矩阵；如 b3lyp/svp 0.106 vs 0.137 s/cycle，m062x/tzvp 最高
  1.62x）——43–222 AO 的 GEMM 太小，launch/host 开销主导，
  64:1 硬件比远未兑现；端到端 fp32 lane 比 fp64 **慢 4–8x**。混合精度的价值
  只能走 auto lane（fp32 头 + fp64 尾），即 #5–#8 已验证 ≤1e-6 Ha 的那条路。
  （LC-50 勘误：原文写 1.1–1.3x，漏了 m062x/tzvp 1.62x 与 r2scan3c 1.40x；
  证伪结论不受影响。）
- ωB97M-V 的 fp32 lane 每 cycle 也只快 1.1x：VV10 kernel 无 fp32 分支，两车道
  同样跑 fp64 NLC —— 该方法的 fp32 lane 收益几乎为零。

## 4. #10 候选清单（按实测收益上限排序，§2.3 预筛）

| # | 候选 | 实测份额（=收益上限的 Amdahl 基数） | 预估端到端上限 | 精度风险（§2.3 "误差落在哪"） | 预筛结论 |
|---|---|---|---|---|---|
| C1 | **VV10 kernel 并行化/访存优化**（`vv10_fock_eval_UWE_kernel`，保持 fp64 算术与 nlcgrids 不变） | ωB97M-V kernel 的 77–84%（benzene）；每 cycle 成本 0.87–1.08s vs 其它方法 0.1–0.2s | kernel 2x → ωB97M-V 端到端 ~1.7x | 低（不改数值路径；nlcgrids 由 #7 契约钉死） | **优先做**：单个 kernel 独占 93% 插桩内核时间，是全部数据里最大的单一杠杆 |
| C2 | ωB97M-V auto lane 的 DIIS-在-精度切换时重置 | 不改份额；修 #7 遗留的 methoxy UKS -1.28e-6 Ha 偏移固定点 | 无速度收益，修精度 gate | 低（DIIS 历史重启是标准操作） | 从 #7 deferred 带过来；先小体系探针再全矩阵 |
| C3 | XC 网格 Fock 侧收缩 GEMM（fp64 tensor-op GEMM，非 NLC 三方法 kernel 的 29–53%） | 三方法 kernel 的 29–53% | S1/S2 尺寸收益已实测有限（见 §3：每 cycle 仅 1.1–1.3x）| 中：fp32 分支已存在于 auto lane；**新增** fp32 路径收益与小尺寸 launch 开销相抵 | 仅在 #11 实测更大 AO 进入 GEMM-bound 域后再评估；S1/S2 不动 |
| C4 | 密度收缩 `eval_rho2`（14–32%） | 同上 | 同上 | 同上（auto lane 已覆盖） | 同 C3 |
| C5 | AO 求值 kernel（5–16%） | 小 | <1.1x | 中（AO 值一阶进 rho/exc，无下游衰减） | 低优先 |
| C6 | CDERI 构建（端到端 8–16%） | — | — | **§2.3 已否决** fp32 CDERI GEMM（Cholesky 相消 3.8–81 Ha） | 不重做；存储 fp32 可、累加 fp32 不可 |
| C7 | eig/DIIS/init（2–7% + init 9–17%） | 小 | <1.1x | 低 | 低优先（S1/S2 收益太小） |
| C8 | D4/gCP（0.02%） | 无 | 无 | — | 不做（正确性项） |

**#10 的执行顺序建议**: C1 → C2 →（等 #11 容量数据再决定 C3/C4）。
C1 动手前先做单 kernel 微基准 + occupancy 分析（nsys 数据已在
`nsys_wb97mv_svp.sqlite@c0e199c79eee`），实现必须 opt-in 开关 + 严格 FP64 回退，
精度探针先行（VV10 误差一阶落在 E_nlc，无下游衰减——若未来考虑 fp32 算术，
按 §2.3 XC-deriv2 教训直接拒绝，除非小体系实测证明否则）。

## 5. 与 #11/#12 的接口

- #11 容量测绘：本表的份额全部来自 43–222 AO；C3/C4 的 GEMM-bound 判据
  （每 cycle fp32 收益 >2x）应作为 #11 放量时的伴随测量。
- #12 梯度：本 issue 未测梯度路径；`grad_ip1` fp32 内核历史已否决（§2.3），
  梯度热点需单独归属（当前 plan 未排期，如实告知）。

## 6. LC-2 补遗 (2026-10-06 长链工作单元): 梯度路径热点归属

第 5 节遗留的"梯度热点需单独归属"已完成（长链 LC-2；profiler
`mixedprec/benchmark/rtx4090/grad_hotspot_profiler.py`，19 个 CPU 单测
`tools/tests/test_grad_hotspot_profiler.py`；证据
`/data/results/gpu4pyscf-agent/tasks/issue-9/2026-10-06-lc2-grad-hotspots/`
sha256 锚定）。方法与 SCF 侧同纪律：deviceSynchronize 包裹（组成来源）+
同收敛态未插桩重复基线（绝对速度来源）+ sync 开销逐道报告 +
`[unattributed]<=10%` 硬门（实测 <=1.9%）；6 cell x {fp64, auto} x
(3 插桩 + 3 基线) 次采样，梯度内核车道用 precision-probe grad_ip1 计数
**实测**而非假设。

实测组成（% of t_grad，中位数）：

- **DF ip1 JK 收缩是杂化/GGA/meta-GGA 格最大梯度桶（35–51%）**——
  5090 STATUS.md 的 "sum_ejk 79%" 先验（1042 AO）不迁移到 43–144 AO；
- **wb97m-v 梯度被 VV10/NLC 主导**（闭壳层 47–52%，UKS 22–27%），
  与 SCF 侧同核家族，未来 VV10 内核工作两路径双收益；
- **grad.hcore_e1 ~0.095–0.13 s 几乎不随 AO 增长**（12–26%）：
  launch/传输 bound 的小收缩集合，消费级尺寸上最清晰的结构性低效，
  是有界的内核合并/批处理候选；
- **fp32 ip1 内核在其桶内 1.08–1.51x，整梯度仅 0.95–1.08x**（与 #12
  实测一致）；fp32 入口命中实测 6/9/6 次（闭壳层/wb97m-v 双调用/UKS）。

实况发现并如实修正：首跑把 r2scan3c/auto 标为"插桩!=基线"(1.9e-7)；
无插桩对照实验证明该车道自身重复就有 ~2.0e-7 方差（fp32 ip1 + D4/gCP
GPU 归约非位可复现）。V5 规则改为对比车道自身实测 run-to-run 方差
（3 倍，地板 1e-8），二跑 VALID——固定 1e-8 会错标真实噪声车道。

下一杠杆排序（Amdahl 于实测 t_grad）：VV10 梯度内核（wb97mv 格 47–52%，
与 #10 tile 家族配对）> DF ip1 批处理（fp32 已落地）> hcore_e1 合并
（结构性，需独立有界探针）> XC 网格（auto 已 fp32）。LC-2 只归属不动手
实现（与 #9 纪律一致：实现需带自己的精度证据独立立项）。

## 7. LC-4 补遗 (2026-10-06 长链工作单元): hcore_e1 根因与修复
（generate_shl_pairs 向量化）

LC-2 实测 grad.hcore_e1 ~0.095–0.13 s 几乎不随 AO 增长（12–26% of
t_grad）。LC-4 分件计时定位根因（证据
`tasks/issue-9/2026-10-06-lc4-hcore-probe/`，sha256 锚定）：

- 一次 `_hcore_energy` = `_grad_nuc_without_ecp`（内含 `int1e_ipkin`）+
  `int1e_ipovlp`，两个 PBC intor **每次调用都新建 `_Int1eOpt`**；
- `_Int1eOpt` 构造的 16 ms 中 **12.4–22 ms 在 `generate_shl_pairs` 的
  逐 (i,j) 对组装 python 循环**：每个 group-pair 约 5 个微型 cupy 内核
  （arange/外积/花式索引，各 ~27 µs），合计 ~150 次 launch；mask 估算
  内核本身 <1 ms，在已建好的 opt 上跑 intor 内核 <1 ms；
  `contract_h1e_dm` 仅 0.3 ms。纯 launch/host 开销，不是计算。

**修复**（commit 00e0e1a）：`SortedMole.generate_shl_pairs` 改为主机侧
numpy 选址 + **单次**设备传输（整数码，值与逐元素 GPU 组装逐位一致；
顺带接受 numpy mask 输入）。分件复测：intor 16.4→4.5 ms。

**端到端实测**（6 个四方法 cell，fp64，补丁前后同收敛轨道，ref 先用
`git stash` 采集）：hcore_e1 **1.33–1.39x**（0.097–0.135→0.072–0.097 s），
整梯度 0.97–1.17x（methoxy UKS 1.12x、r2scan3c 1.17x、m062x/wb97mv
benzene ~1.0x——其余桶占优处稀释）；梯度逐位等价
（dg ≤ 9.9e-9 Eh/Bohr，在 LC-2 记录的 run-to-run 方差内；dE ≤ 1.4e-13 Ha）。
回归：6 个新单测（键集/数值/空组/aggregate 往返/intor 消费者）+
tools 全套 375 (stdlib) / 380 (source env) OK + gpu4pyscf
df-mixedprec 18 与 stability-verify 6 全过。

注：该函数同时服务 SCF/Hessian/DF 的 int3c2e opt 构造（首次构建各一次），
梯度路径因每次 hcore 都重建而受影响最大；收益对单次 SCF 可忽略（一次性
成本），对几何优化/多次梯度按每步 ~50 ms 累计。

## 8. LC-5/LC-6 补遗 (2026-10-06 长链工作单元): 修复后热点刷新 + Int3c2eOpt.build 精简筛除

**LC-5 刷新**（`tasks/issue-9/2026-10-06-lc5-grad-hotspots-postfix/`，commit
30c7b83，VALID）：LC-4 修复落地后重跑 12-lane 梯度热点矩阵——hcore_e1
占比 12–26% → **5–22%**（修复复现），df_jk_ip1 升为 b3lyp/m062x/r2scan3c
首位（36–58%），nlc_vv10 在 wb97mv 闭壳层 48–56%。分件解剖：剩余
hcore_e1 的 88–89% 是 `Int3c2eOpt.build`（61.8–104.4 ms），其中
`_cache_q_cond_and_non0pairs` 的 `_group_by_split_points` 尾部
（按 q-cond 降序重排 ~189–324 个小 kernel）占 41–73 ms——而 build 随即
`cp.sort`，**该排序结果被整体丢弃**；估算内核本身仅 3–5 ms。

**LC-6 修复**（`tasks/issue-9/2026-10-06-lc6-qcond-screen/`）：在
`df/int3c2e_bdiv.py` 新增 `_screen_bas_pairs_qcond`——主机侧按
`PBCsort_pair_ij` 瓦片枚举顺序组装 pair（对设备内核逐位同序，已用
直接设备调用验证）、同一 `int2e_qcond_estimator` 内核、等价阈值掩码
（q > log(precision) ⇔ 尾部 bin≥1 保留集），不做降序重排。仅
`Int3c2eOpt.build` 改用之；`_cache_q_cond_and_non0pairs` 原样保留
（JK 引擎仍需该排序做早终止筛除）。

**等价性**：13 个新单测（`tools/tests/test_int3c2e_qcond_screen.py`），
含 tril=False、l>LMAX h 函数基、松 cutoff、单壳层、主机序与设备内核
直接对拍；bas_ij_cache 逐位一致。同收敛轨道下新旧 build 全梯度差
7.1e-15–9.6e-14 Eh/Bohr = 同进程非确定性下限（对照：同路径连续两次
梯度差至 1.4e-14）。

**实测**（6 个四方法 cell，fp64，补丁前后同协议，ref 用 git stash 采集）：
hcore_e1 **2.81–3.10x**（0.073–0.098→0.024–0.034 s）；整梯度
**1.12–1.42x**（r2scan3c 1.37x、m062x 1.42x、b3lyp benzene 1.30x、
methoxy UKS 1.18x、wb97mv benzene 1.12x）——超出 hcore_e1 单独解释的
部分来自 `df/grad/rhf.py jk_energy_per_atom` 同样每次梯度重建
Int3c2eOpt（range-separated wb97m-v 每次 2 次），修复在 df_jk_ip1 桶
同样生效。跨进程 dg ≤ 6.0e-9 Eh/Bohr（SCF 位噪声，与 LC-4 同量级）。

测试：tools 全套 393（source env）/ 376（stdlib，5 skipped）OK；
gpu4pyscf df-mixedprec 18 + stability-verify 6 OK。

**修订后的梯度热点排名**（LC-5 矩阵 + LC-6 修复后）：df_jk_ip1 与
nlc_vv10 为剩余大头（设备计算为主，fp32/tile 内核均已落地且量级有限）；
hcore_e1 降至 ~2–5%；下一结构性杠杆需来自 SCF 侧或更大体系实测。

**LC-8 补注（SCF 侧效应，修正版）**：LC-6 的精简 build 在 SCF 侧同样
生效。首次测量（scf_side_effect.json v1）无效——HEAD 已含补丁，
`git show HEAD:` 提取的"旧"build 实为新实现（新对新，纯噪声 0.92–1.03x）；
v2 从补丁前 commit 30c7b83 提取真旧 build 重测：孤立 intopt.build
62.1→14.4 ms（DF auxmol）/ 61.2→13.3 ms（fakemol），DF 上下文
bas_ij_cache 逐位一致；DF.build 整体 1.78–1.83x（108–173→60–95 ms）；
整 SCF 1.01–1.29x（b3lyp cells 1.14–1.29x；wb97mv 1.01x，VV10 主导）；
dE ≤ 2.3e-13 Ha（跨进程 SCF 位噪声）。几何优化每步重建 DF，收益复利。

**LC-9 终版梯度组成**（`2026-10-06-lc9-final-composition/`，commit 8803de8，
VALID）：LC-4+LC-6 两项修复叠加后 hcore_e1 占比 6–26% → **2–10%**；六 cell
累计 t_grad 提升 1.15–1.52x（methanol 1.52x、benzene/svp 1.50x、methoxy
UKS 1.36x、r2scan3c 1.32x、m062x 1.26x、wb97mv 1.15x）。剩余大头均为设备
计算桶（df_jk_ip1 30–60%、xc_grid 12–48%、nlc_vv10 54% wb97mv），其
fp32/tile 杠杆已落地并测过——消费级尺寸下已无同量级的主机端结构性杠杆。

## 9. LC-10..LC-14 补遗 (2026-10-06 长链工作单元): SCF 侧热点弧线（XC 分块、cuTensor 冷启动）

**LC-10（VV10 negligible-rho 剪枝，负结果）**
（`tasks/issue-10/2026-10-06-lc10-vv10-rho-pruning/`，commit 5f50ee4，纯测量）：
阈值阶梯实测五个 wb97mv cohort cell——现行 1e-10 阈值在收敛密度上
**剪除 0 个格点**（纯零密度守卫，非性能杠杆）；CPU 平价 1e-8 保留
97.0–97.8%，pair 工作 0.94–0.96，内核仅 1.02–1.04x，|dE_nl| 9–26e-9 Ha；
任何真正削减 O(N²) 工作的阈值都破坏预算（1e-6 已 -1.0..-2.3e-6 Ha，
1e-3 达 -4.3..-8.1e-4 Ha）。预注册门槛（≥1.2x 且 |dE_nl|≤1e-9）未过，
**不实现**；候选 (c) 关闭。VV10 杠杆至此穷尽（tile 内核 LC-3、roofline、
剪枝本项）。

**LC-11（SCF XC 格点分块，opt-in）**（commit 37d8bf5，证据
`tasks/issue-9/2026-10-06-lc11-xc-blocksize/`）：格点以固定
MIN_BLK_SIZE=4096 分块，每块独立 eval_ao+2 GEMM+scatter 发射列车，在
61k–128k 格点即 15–31 块/循环 ×2 循环/周期——实测发射受限。新增
`GPU4PYSCF_XC_GRID_BLOCKSIZE`（MIN_BLK_SIZE 正倍数，默认关闭，默认路径
逐字节不变）：仅重划 `_nr_rks_task`/`_nr_uks_task`/`nr_nlc_vxc` 的分块
（blksize 键控 non0ao_idx 缓存 + 多 GPU 对齐守卫）；梯度/Hessian 路径
不动（其 scratch 按各自模块常量定容）。probe（五 cell，nr_rks，1 热身
+3 采样中值）：阶梯 1.26–3.76x（最优/cell 1.57–3.76x，MGGA 受益最小，
与其设备端 libxc/rho 主体一致）；|dExcsum|≤5e-14 Ha。整 SCF 16384 臂
1.07–1.16x（24/24 样本收敛，dE≤2.3e-13 Ha，峰值显存 +≤60 MiB），19 个
单测。**默认开启未做**（约 1e-13 Ha 归约序变化属用户策略决定）。

**LC-12（修复后 SCF 组成 + 新发现：cuTensor 冷启动）**
（`2026-10-06-lc12-scf-composition/`，commit 37d8bf5，纯测量；4 个诚实
失败尝试日志留存）：稳态每周期组成——b3lyp xc 59 ms；m062x xc 92 ms；
r2scan xc 110 ms；wb97mv nlc 175 + xc 88 ms；DF J/K 仅 6–14 ms/cyc
（SCF 侧不是瓶颈）；16384 臂削稳态 xc 21–50%。**新发现**：每个新进程在
首个用 cutensor 的调用上一次性阻塞 ~3.0–3.2 s（MGGA 首个 _tau_dot、
hybrid 首个 DF get_jk）——即消费级尺寸下实测整 SCF 时间的 44–73%。根因
链：cupyx `cutensor._get_handle()` = cuTensorHandleCreate 2.9–3.1 s
（跨进程恒定、主机侧、可与并发 GPU 工作重叠）；稳态 cutensor 与 cublas
同速（0.153 vs 0.149 ms；0.081 vs 0.080 ms）——消费级形状下 cutensor
无收益只带初始化成本；cupyx 惰性全局非线程安全（并发首调用实测产生
两个不同句柄）。

**LC-13（后台 cuTensor 句柄预热 + LC-11 惰性格点回归修复）**
（commit 8c5beba，证据 `2026-10-06-lc13-cutensor-warm/`）：
`lib/cutensor.py` 模块级锁串行化全部句柄访问 + 导入末尾（引擎选择后）
daemon 线程预热，3 s 初始化藏进 ~1.8 s 导入链；`GPU4PYSCF_CUTENSOR_WARM=0`
可关。验证（git stash 协议）：首 get_jk 3.148→2.081 s（此子进程 setup 短；
实际负载藏得更多）；四 cell 能量 dE≤1.1e-13 Ha；SCF 后 4 线程竞取句柄
同一对象；opt-out 恢复 3.13 s。验证 attempt-1 崩溃暴露 **LC-11 回归**：
`nr_nlc_vxc` 在 block_loop 惰性建格点前读 `grids.coords`（上游从不预读；
`_DFHF.get_veff` 在 grids 已建时跳过 initialize_grids，nlcgrids 可为未建
态）——已修（未建态直接用 `_scf_grid_blksize()`，单 GPU 网格域恒对齐），
2 个 GPU 回归测试。

**LC-14（合并冷启动效应）**（`2026-10-06-lc14-coldstart-combined/`，
纯测量，old=5f50ee4 vs new=8c5beba 物理切换）：用户视角新进程单点作业
（解释器+导入+建格点+整 SCF，父进程计总墙钟，3 样本中值）：
b3lyp 6.616→5.299→5.132 s（warm 1.25x，合并 **1.29x**）；m062x
1.17x/**1.27x**；r2scan 1.20x/**1.26x**；wb97mv 1.14x/**1.28x**
（16384 臂再省 0.3–0.5 s，VV10 循环受益最大）；|dE|≤1.7e-13 Ha。
attempt-1 无效（'new' 标签跑了旧源——restore 只在 finally；行间雷同
暴露之，日志留存），修正后按标签显式 checkout。测试（8c5beba）：tools
412 OK；df-mixedprec 18 + stability 6；dft rks 16 + uks 10 OK；
test_sparse_index 指纹失配在原始上游 0a4cf33 即失败（环境相关，非本
分支所致，已记录）。

**修订后的 SCF 侧排名**（LC-12 组成 + LC-11/LC-13 修复后）：冷启动
一次性成本已消（句柄预热落地；剩余 ~2.3 s 导入+setup 属 Python 层）；
稳态最大桶为 MGGA xc（92–110 ms/cyc，16384 臂已削 21–50%，m062x 受益
最小 1.09–1.26x——更深需 _eval_rho2/tau GEMM 列车内核级工作）与
wb97mv VV10（175 ms/cyc，杠杆穷尽）。

## 10. LC-21..LC-23 补遗 (2026-10-06 长链工作单元): 梯度侧格点分块弧线

**LC-21（梯度侧分块 probe，门槛通过）**
（`tasks/issue-9/2026-10-06-lc21-grad-blocksize/`，commit a0ce0e6，纯测量）：
梯度的 xc_grid 循环（grad/rks.py、grad/uks.py）仍是固定 MIN_BLK_SIZE 分块
——LC-11 修复前的同构结构。运行时重分区 probe（阶梯 {4096,16384,65536}，
1 热身 + 3 采样）：整梯度 16384 处 b3lyp 1.336x / m062x 1.243x / r2scan
1.207x / wb97mv UKS 1.141x，dg<=3e-13 Eh/Bohr（同轨道归约序下限）；
预注册门槛（≥1.05x 于 ≥半数 cell 且 dg≤1e-8）**全过**。附带发现：
grad/rks.py 模块常量默认 4096 而 grad/uks.py 默认 16384（不一致，记录在案
——默认路径数值不动，留作未来策略决定）。

**LC-22（梯度侧 wiring 落地）**（commit 13386bb，证据
`2026-10-06-lc22-grad-loopblk/`）：`GPU4PYSCF_XC_GRID_BLOCKSIZE`（与 LC-11
同一旋钮）现同时驱动梯度 XC 循环——grad/rks 与 grad/uks 的
`_get_exc_task` LDA/GGA/MGGA 三循环、grad/rks `get_nlc_exc` 的两个 VV10
循环（UKS 复用）；全部以 `_effective_scf_blksize` 守卫 + loop_blksize 定容
scratch 缓冲；response/Hessian 路径不动；默认路径逐字节不变。
验证（同子进程双臂，同轨道）：四 cell 整梯度 1.24-1.27x，同轨道
dg≤1.4e-13；默认臂跨源 dg≤2.0e-9、dE≤1.7e-13（跨进程 SCF 噪声，与
LC-4/LC-6 基线同量级）；旧源对照 0.997-1.07x（提速确由 wiring 产生）。
9 个新单测（tools/tests/test_grad_loop_blksize.py）。

**LC-23（梯度基准矩阵定版文档）**（`2026-10-06-lc23-grad-lane/`，纯测量）：
生产 `gradient_runner.py`、fp64 参考车道、3 采样、推荐配置（fp64 +
16384 + LC-13 warm）——issue-#12 九 cell 四方法矩阵 **9/9 文档
schema-VALID 且全 PASS**；对 issue-#12 参考文档整梯度提速
**1.22-2.00x**（wb97mv/benzene 1.22x 最低——VV10 主导；r2scan3c/methoxy
UKS 2.00x 最高），dE≤1.1e-13 Ha（同解），峰值 162-430 MiB。
与 LC-20 合并：**推荐配置下一个 opt-in 旋钮 + 常开句柄预热，能量矩阵
1.26-1.29x（新进程单点）、梯度矩阵 1.22-2.00x**，同解等价 1e-13 量级。

**梯度侧修订排名**（LC-5/LC-9 矩阵 + LC-22 修复后）：xc_grid 桶已获旋钮
（剩余为设备 GEMM 主体）；df_jk_ip1（30-60%）为设备计算，其 fp32 入口
证据已在 issue #10 落地；nlc_vv10 杠杆穷尽。消费级尺寸下 SCF 与梯度
两侧的结构性杠杆均已收割完毕；下一量级增益需更大 AO（issue #11 域）
或数学近似（严格 1e-6 门槛下超范围）。

## 11. LC-25..LC-27 补遗 (2026-10-07 长链工作单元): 全矩阵 SCF 解稳定性覆盖（非热点单位，登记于此以保持单一矩阵画像）

性能侧弧线（§9/§10）之外，本会话补齐了基准矩阵的解稳定性画像
（证据 tasks/issue-7/{2026-10-07-lc25-stability-matrix,
2026-10-07-lc26-m062x-min-gpu} 与 tasks/issue-9/
2026-10-07-lc27-closed-shell-stability，均 sha256 锚定）：

- **闭壳层 5 格全部深稳定**（最软本征值 0.36-2.6 Ha，比 -1e-5 失稳阈
  高 5+ 个量级；wb97m-v 需含 VV10 响应的 RKS Hessian，LC-1/LC-25 规则
  的闭壳层版）。wb97m-v/benzene 的 -2.605e-8 CPU/GPU 偏移为已记录的
  实现一致性偏移（与 issue-7 哨兵行同值，CPU 对 CPU 差 5.7e-14），
  非复现失败。
- **UKS 6 格分类见 docs/issue-7-uks-stability.md §7/§7.1**：唯二鞍点
  是 wb97m-v/svp 的 fp64 参照道（LC-1）与 m062x 共享解（LC-25，
  低 2.628e-5 Ha 的稳定极小存在且精度稳健，LC-26）。
- LC-25..LC-27 覆盖的是**梯度矩阵 9 格 + UKS 能量格**；本节初版
  "每个基准格都有稳定性分类"的说法**被 LC-28 的缺口分析修正**：
  能量基准阶梯（issue #5-#8）另有 8 格当时未分类，LC-28 补齐
  （见 §12）。矩阵画像四轴不变；本补遗不改变任何行判据。

矩阵画像现含四轴：性能（§2-§10）、精度（各 issue 门）、实现一致性
（CPU 哨兵）、解稳定性（本节 + §12）。用户 1e-6 门重述决策的物理输入
齐备。

## 12. LC-28 补遗 (2026-10-07 长链工作单元): 稳定性覆盖缺口闭合（能量阶梯其余 8 格）

LC-28 的缺口分析发现 §11 初版的覆盖声明过强：能量基准阶梯
（issue #5-#8，每方法 5 格）与梯度矩阵（issue #12，9 格）并不重合，
LC-25/27 只覆盖了后者加 UKS 能量格。LC-28 用同一套已验证方法把
剩余 8 格全部分类（证据 tasks/issue-7/2026-10-07-lc28-m062x-tzvp-
stability 与 tasks/issue-9/2026-10-07-lc28-closed-shell-remaining，
均 sha256 锚定）：

- **m062x/methoxy/def2-tzvp UKS：鞍点，且鞍点物理是基组稳健的**
  （详见 docs/issue-7-uks-stability.md §7.2）——tzvp 最软本征值
  -2.921e-3 Ha（svp -3.559e-3，同一 beta 主导模），级联稳定极小低
  2.687e-5 Ha（svp 2.628e-5，两基组几乎同深度）；GPU 两道在极小上
  一致（fp64 +1.6e-12/1 圈、auto +1.6e-11/4 圈）。新现象：GPU auto
  道收敛点停在鞍点下方 4.09e-8 Ha（沿不稳定模的下坡沿），CPU fp64
  重启爬回鞍点——auto 道点在严格 fp64 面上非定态；auto 道复现门
  （G1, 2e-8）诚实不通过，fp64 道通过。auto-vs-fp64 门内比较
  （-4.09e-8，PASS）不受影响。auto+verify 策略在该格上也被实测
  （3 样本）：密度差判据被 fp32 位移触发（1.16e-3 > 1e-3 阈，与
  svp 的干净 no-op 不同），"取低解"规则采纳该非定态点（记录含
  抬升的轨道梯度 8.9e-6 对 3.4e-6）；门内比较不受影响，但
  "verified 解为 fp64 面定态点"的性质在该格上不成立。
- **其余 7 个闭壳层格全部深稳定**（最软本征值 0.36-3.40 Ha）：
  benzene/tzvp x {b3lyp 0.772, m062x 0.740, wb97m-v 0.359（NLC 面）}、
  methanol/svp x {m062x 1.167, wb97m-v 0.549（NLC 面）}、
  r2scan3c methanol/mtzvpp 1.186、r2scan3c **vitaminC/mtzvpp 3.404**
  ——后者是整个项目首个大格（298 AO）稳定性分类，深稳定（能量
  复现 +1.5e-11）。wb97m-v 两格的 CPU/GPU 偏移（-1.1e-8/-1.6e-8）
  是 wb97m-v 家族已记录的 CPU/GPU 实现一致性轴（同 LC-27 的
  -2.605e-8），非复现失败；其余方法 CPU/GPU 一致在 1e-12 量级。

至此**能量基准阶梯的全部 19 格（b3lyp/m062x/wb97m-v 各 5 格 +
r2scan3c 4 格含 vitaminC 容量格）与梯度矩阵 9 格的稳定性分类真正
齐备**：唯一鞍点家族是 m062x methoxy UKS（两基组，非精度引起）加
wb97m-v/svp 的 fp64 道（LC-1）；其余全部深稳定。§4 性能热点清单
不受影响。

## 13. LC-34 补遗 (2026-10-07 长链工作单元): 推荐配置下的梯度组成刷新 —— df_jk_ip1 成为绝对主桶

§8 的梯度归属（LC-5）测于 LC-6 之前（旧 Int3c2eOpt.build）且为默
认配置；LC-6（build 精简筛除）、LC-22（梯度侧 16384 分块）、LC-13
（cuTensor 常驻预热）相继落地后，推荐车道
（`GPU4PYSCF_XC_GRID_BLOCKSIZE=16384`）下的梯度组成从未刷新。本单
元用 LC-2 剖析器原样重测（同 6 格 × fp64/auto 车道 × 3+3 样本，
V1-V7 门，观察等价 ≤2.3e-13，r2scan3c/auto 2.2e-7 为其已记录的车
道方差内），文档 VALID（unattributed ≤1.2%；证据
`tasks/issue-9/2026-10-07-lc34-grad-composition-refresh/`，预测先
行登记于长链 checkpoint）。

组成（份额；绝对 ms 括号内为 LC-5→LC-34）：

| 格 | 车道 | df_jk_ip1 | xc_grid | vv10 | hcore_e1 | t_grad 变化 |
|---|---|---|---|---|---|---|
| b3lyp/methanol/svp | fp64 | **66%**（180→129ms） | 21% | — | 12% | 0.369→0.195s（1.89×） |
| b3lyp/benzene/svp | fp64 | **71%**（249→202ms） | 19% | — | 9% | 0.428→0.285s（1.50×） |
| m062x/benzene/svp | fp64 | **58%**（250→204ms） | 34% | — | 8% | 0.496→0.350s（1.42×） |
| wb97mv/benzene/svp | fp64 | 34% | 8% | **56%** | 2% | 1.504→1.213s（1.24×） |
| wb97mv/methoxy/svp UKS | fp64 | 56% | 8% | 31% | 5% | 0.716→0.486s（1.47×） |
| r2scan3c/benzene/s1 | fp64 | **41%**（182→121ms） | 47% | — | 12% | 0.481→0.296s（1.63×） |
| （auto 车道同形：df_ip1 49-62%，wb97mv 闭壳 vv10 68%） | | | | | | auto 整相 1.35-2.36× |

要点（预测 P1-P5 对照，G1-G4 全过）：

- **df_jk_ip1 现为一切非 wb97mv 格的绝对主桶（41-71%，两车道）**
  ——LC-6 的 build 精简同样作用于 DF 梯度的 Int3c2eOpt 路径
  （`df/grad/rhf.py` 全程使用该类），桶内绝对时间降 19-36%；但其余
  桶降得更多，份额反升。P1 预登记带 45-65% 基本命中（b3lyp/benzene
  fp64 70.9% 超带上沿，如实记录）。
- **整相加速 1.24-2.36×**，其中 hcore 75→23ms（LC-6 的 2.8-3.2×
  逐位复现）、xc_grid 全部 12 车道绝对时间下降（LC-22 布线，auto
  车道降幅最大 186→44ms）。P5（预测 ~1.24-1.27×）被超预期——LC-6
  与 LC-22 的收益复合。
- **auto 车道在 16384 下首次明确快于 fp64 车道**（m062x 0.206 vs
  0.350s；b3lyp/benzene 0.191 vs 0.285s；LC-5 时两者持平 0.95-1.09×）
  ——fp32 ip1 内核的收益不再被发射受限的 XC 循环稀释。仅组成事实；
  SCF 车道结论不变（LC-17），UKS 支线格的 auto 梯度正确性约束不变
  （LC-30/33：能量门不预测梯度质量）。
- **下一杠杆排序更新**（按实测份额）：(1) df_jk_ip1 ip1 填充 +
  sum_ejk 收缩的批处理/占用率（41-71%，auto 车道即便已有 fp32 内核
  仍居首——内存/占用率工作，中风险）；(2) wb97mv 的 VV10 梯度内核
  （56-68%，与 issue-#10 tile 内核同族）；(3) xc_grid 残量（8-34%，
  MGGA tau 核心已被 LC-16 证明杠杆抗性）；(4) hcore_e1 残量（2-18%，
  现为 int1e_ipkin 与小件）。

复现：`grad_hotspot_attribution.json`（增量持久化、V1-V7 校验）、
`run_stdout.log`、`evidence_sha256.txt`（脚本为仓库内
`mixedprec/benchmark/rtx4090/grad_hotspot_profiler.py`，未改动）。

## 14. LC-35/LC-36 补遗 (2026-10-07 长链工作单元): df_jk_ip1 微观分解 —— Int2c2eOpt 构造占 2c 包装器 86%，分子路径缓存已落地

LC-34 把 df_jk_ip1 列为梯度主桶（41-71%）后，LC-35 用外部包装器
剖析器（与 LC-2/34 同纪律：逐调用 deviceSynchronize、无插桩基线分
离、观察等价 ≤2.2e-13）把该桶分解到内部件（证据
`tasks/issue-9/2026-10-07-lc35-dfip1-decomposition/`；首个版本有
一个如实记录的归属 bug：Int3c2eOpt.build 也在桶外的 hcore/nuc 路
径触发，由桶内上下文标志修正）：

- **fp64 车道**：sum_ejk_int3c2e_ip1 CUDA 内核主导苯格
  （桶内 46-49%，108-214 ms；实测 fp64/fp32 内核比 5.1×）——auto
  策略的 fp32 梯度内核正是该件已落地的杠杆。
- **auto 车道**：桶被压平（最大件 21%），int2c2e + auxresp
  （2 中心辅助基积分）成为头对（30-42%）。
- **决定性子发现**（微探针，benzene/svp naux=558）：int2c2e 包装器
  21.2 ms 中 **18.2 ms（86%）是 Int2c2eOpt.__init__ 构造**
  （SortedGTO 1.6 ms + bas_ij_cache 小核发射循环），内核仅 ~3 ms；
  梯度路径每次构造 2-4 个（每次 int2c2e 一个 + 每次
  int2c2e_ip1_per_atom 一个；远程分离 wb97mv ×2）。

**LC-36 落地**（commit dde156f）：分子（非周期）路径的 Int2c2eOpt
缓存——键为排序辅助分子的原始 `.mol` 反向引用（跨 Int3c2eOpt 重建
稳定；普通 Mole 键自身），令牌 = 键 `_bas`/`_env` 字节 + 排序
`_bas` 字节 + natm/nao（~9 µs）；pbc Cell 输入与显式 bvk_kmesh
永远旁路；缓存挂在键上故不越过其生命周期。同进程 A/B 实测
（6 格 × fp64/auto，各 5 样本中位；证据
`tasks/issue-9/2026-10-07-lc36-int2c2e-opt-cache/`）：

| 指标 | 范围 |
|---|---|
| 整相梯度加速 | **1.03-1.54×**（methanol/svp fp64 最好；wb97mv 闭壳 auto 最小） |
| Int2c2eOpt 构造 | 12/24 → **0/1** 每臂 |
| 同轨道 \|dg\|（缓存 vs 旁路） | ≤ 6.8e-14（r2scan3c/auto 1.7e-7 = 其已记录车道方差） |
| 重复运行方差 | 有缓存 1.2e-14 vs 无缓存 8.9e-15（既有原子加非确定性，缓存未新增） |

- 9 个红→绿单元测试（tools 430 OK）：缓存命中/跨 SortedMole 命中
  （生产模式）/pbc 旁路/显式 kmesh 旁路/键变更失效/不泄漏/逐位
  int2c2e + 方差门 ip1。
- **副发现（上游既有）**：给 `int2c2e_ip1_per_atom` 传球形尺寸的
  dm 会在 `e_int2c2e_ip1_kernel` 越界读并静默返回垃圾——
  compute-sanitizer 在**原版上游**同样复现（同 8 字节越界读，
  thread 194 block 68；非缓存引入）。现该输入被响亮拒绝
  （ValueError）。生产路径传的是正确的笛卡尔 dm（262×262，
  已拦截实测）。
- 与 LC-6/LC-22/LC-34 累计：LC-5 基线的梯度整相现已
  **1.3-3.6×**（0.344→0.126s methanol/svp fp64 等）。

## 15. LC-37/LC-38 补遗 (2026-10-07 长链工作单元): Int3c2eOpt 缓存候选证伪 + 梯度车道文档刷新（LC-36 后）

**LC-37（负结果，实测证伪）**：对 Int3c2eOpt.build（梯度内 13-29 ms
桶内 + 11-18 ms hcore/nuc 路径）实施 LC-36 同款缓存的候选被一个关键
测量关闭：对一次 wb97mv 梯度的全部 build 调用做 spy 记录——3 次构
建对应 **3 个不同键**（jk-SR omega=0.0、hcore fakemol、jk-LR
omega=0.3）：梯度内复用为 0；跨几何步令牌必然改变。缓存只能加速
基准协议的同几何重复样本 = 测量通道优化而非负载优化，不实现。LC-36
的 int2c2e 缓存区别保留（omega 在那里是内核参数，单一共享键）。

**LC-38（文档刷新）**：LC-23 的权威梯度车道文档（13386bb）早于
LC-36 缓存（dde156f），HEAD 缺少 schema-VALID 文档。以 LC-23 协议
重测 9 格四方法矩阵（生产 gradient_runner，fp64 参考道，3 样本/格，
逐样本 CUDA 同步，16384 分块，每格新进程；证据
`tasks/issue-9/2026-10-07-lc38-grad-lane-refresh/`，9/9 VALID。第
一次尝试的 runlog 被我的 shell 重定向覆盖导致证据锚损坏——尝试 1
保留于 `attempt1-shell-redirect-preserved/`，尝试 2 由 runner 自写
日志后锚定）：

| 格 | t_LC23 | t_LC38 | 加速 | dE | dg |
|---|---|---|---|---|---|
| b3lyp/benzene/svp | 0.347 | 0.306 | 1.13× | 0 | 1.2e-9 |
| b3lyp/methanol/svp | 0.230 | 0.173 | 1.33× | 0 | 7.9e-11 |
| b3lyp/methoxy/svp UKS | 0.254 | 0.172 | **1.48×** | 5.7e-14 | 4.1e-11 |
| m062x/benzene/svp | 0.413 | 0.349 | 1.18× | 1.1e-13 | 1.2e-9 |
| m062x/methoxy/svp UKS | 0.307 | 0.246 | 1.25× | 2.8e-14 | 3.4e-11 |
| r2scan3c/benzene/s1 | 0.334 | 0.261 | 1.28× | 0 | 1.5e-13 |
| r2scan3c/methoxy/s1 UKS | 0.242 | 0.156 | **1.54×** | 0 | 5.3e-14 |
| wb97mv/benzene/svp | 1.251 | 1.182 | 1.06× | 2.3e-13 | 6.9e-9 |
| wb97mv/methoxy/svp UKS | 0.521 | 0.449 | 1.16× | 0 | 1.1e-14 |

- 加速 1.06-1.54×，与 LC-36 同进程 A/B 矩阵（1.03-1.54×）一致——
  每格新进程（每样本新 mol+mf），收益是真实单梯度收益。
- 能量/梯度差异全部在该车道的已记录跨进程 SCF 非确定性范围内
  （LC-23 自己对 issue-12 参考文档的偏差包络：dE ≤1.1e-13、
  dg ≤7.9e-9，本表 dE ≤2.3e-13、dg ≤6.9e-9 同量级；预登记门
  G3/G4 按该实测包络裁定，非临时放宽——两代文档间同格对比必然
  含此噪声）。
- 至此与 LC-5 基线累计：梯度整相 **1.3-3.3×**（LC-6 + LC-22 +
  LC-36 复合；methanol/svp fp64 0.344→0.173s）。

复现：`run_matrix2.sh`、9 组 `*.json/*.raw.json/*.runlog.log`（
runner 自锚定 sha256）、`evidence_sha256.txt`、
`attempt1-shell-redirect-preserved/`（诚实保留的失败尝试）。

## 16. LC-40 补遗 (2026-10-07 长链工作单元): 中等尺寸梯度矩阵 —— 梯度轴首次超过 144 AO

**再扫描发现**：整个梯度轴（9 格矩阵 + LC-38/39 权威文档）从未
在 144 AO 以上测过——所有梯度格都是 43-114 AO（r2scan3c 苯
144）；已交付的全部梯度杠杆（LC-22 分块、LC-29 auto-verify、
LC-36 缓存、auto 车道）在中等尺寸上无验证，而能量阶梯已达
222 AO（苯/tzvp）与 298 AO（vitaminC/mtzvpp）。LC-17 遗留的
开放问题（auto 车道价值 = 更大 AO 的 GEMM 主导域）可在
222-298 AO 梯度轴上直接检验。

**矩阵**（4 闭壳层格 × {fp64, auto, auto-verify} 单文档门 + cpu-fp64
哨兵合并；推荐配置 16384；生产 gradient_runner；每道 3 新进程
样本、逐样本 CUDA 同步、worst-repeat 聚合；证据
`tasks/issue-12/2026-10-07-lc40-moderate-ao-grad-matrix/`，
12+4 行全部 ok、全部样本收敛、全部 schema-VALID）：
b3lyp/m062x/wb97m-v × 苯/def2-tzvp（222 AO）+
r2scan3c × 020_Vitamin_C/def2-mtzvpp（**298 AO——全程序首个
>144 AO 梯度**，含 D4/gCP 梯度响应）。四格均为 LC-28 已分类的
深稳定唯一极小（无 UKS 多解）。

| 格 | auto max\|dg\| | verify max\|dg\| | dE(auto) | 梯度相 auto/fp64 | 整道 auto/fp64 |
|---|---|---|---|---|---|
| b3lyp/苯/tzvp | 1.17e-6 | 1.18e-6 | +1.1e-12 | 1.83x | 1.08x |
| m062x/苯/tzvp | 9.3e-7 | 9.3e-7 | -8.3e-12 | 2.43x | **1.46x** |
| wb97mv/苯/tzvp | 8.8e-7 | 8.9e-7 | +1.2e-11 | 1.52x | 1.00x |
| r2scan3c/vitaminC | **9.9e-6** | 9.9e-6 | +1.4e-12 | **3.17x** | **1.43x** |

**结论**：

- **LC-17 开放问题在梯度轴上肯定成立**：auto 梯度相加速随 AO 增长
  （114 AO 时 1.1-1.7x → 222-298 AO 时 1.5-3.2x），且首次出现
  auto 整道明显快于 fp64（m062x 1.46x、vitaminC 1.43x；
  LC-34 已测 114 AO 上 m062x 0.206 vs 0.350s 的苗头）。机制与
  LC-35 微观分解一致：df_jk_ip1 的 fp64 CUDA 内核（fp64/fp32
  实测 5.1x）在更大 naux/nao 上占比更高。SCF 相 auto 仍慢
  （LC-17 不变），但 222 AO 起梯度相收益开始承担整道。
- **auto 车道 max|dg| 随尺寸增长**：svp 族 1.2-2.0e-6 →
  vitaminC 9.9e-6（门 3e-5 的 1/3）。CPU/GPU fp64 梯度实现差
  仅 9.7e-7（vitaminC 哨兵），auto-vs-fp64 的 9.9e-6 主要是
  fp32 梯度内核响应本身（r2scan3c 车道已记录的方差随尺寸
  放大），非实现差。更大尺寸外推时须注意此增长趋势。
  **LC-57 更新（2026-10-07）：增长已实测饱和**——tamoxifen/
  mtzvpp 687 AO（同泛函同基组族第三点）auto max|dg| 1.06e-5，
  298→687 AO（2.31x 尺寸）仅 +7%；线性外推（~2.3e-5）与幂律
  外推（~1.0e-4）双双否证，门 3e-5 余量 2.8x。同格梯度相
  auto/fp64 = 4.47x（序列 3.17x → 4.47x 随 naux 增长）、整道
  1.44x、峰值 6.95 GB。证据
  `tasks/issue-12/2026-10-07-lc57-dg-growth-next-size-step/`
  （issue-12 §10；同单元修复 3c verify 参照道 D4+gCP 遗漏）。
- **auto-verify 全部不触发**（distinct=false，dm-diff 8e-6-5.4e-5
  远低于 1e-3 阈值）——与闭壳层唯一极小预期一致；不触发开销
  = 一条 fp64 参考轨迹（车道比 1.8-2.4x，P6 预测带上限
  m062x 2.42x 微超，如实记录）。
- **容量充裕**：梯度峰显存 0.69-2.35 GB（预算 21.6 GB），
  vitaminC 298 AO 全道（CDERI+SCF+梯度）峰仅 1.36 GB。
- wb97mv 整道 1.00x：SCF 相被 VV10 主导（无 fp32 路径，LC-9）
  且 auto SCF 更慢，梯度相 1.52x 不足以翻转——与预测的
  "parity-to-modest" 方向一致（梯度相实际 1.52x 略超预期，
  归因于 wb97mv 梯度中非 VV10 部分——JK ip1——仍走 fp32）。
- cpu-fp64 哨兵：能量 dE ≤ 1.7e-11（四格全收敛）；梯度
  max|dg| 0.2-1.0e-6（实现差，tier-2 诊断）；wb97mv 的
  1.13e-8 能量差与 LC-28 记录逐位一致（VV10 实现差轴）。
- 预测偏差如实记录：P3 wb97mv dE 1.19e-11 微超 1e-11 带（同
  解噪声级，非门）；P6 m062x 2.42x 微超 2.2x 带上；P4 wb97mv
  梯度相 1.52x 略超 "parity-to-modest"。

复现：`run_gpu_matrix.sh` + `run_cpu_merges.sh`（逐格），
4×`*.json/*.raw.json/*.runlog.log`（GPU-only 3 道文档）+
4×`*_wcpu.*`（哨兵合并终文档）、`bench_gpu.runlog.log`、
`evidence_sha256.txt`。

## 17. LC-41 补遗 (2026-10-07 长链工作单元): 中等尺寸能量矩阵 —— LC-17 的 SCF 结论首次在界内精化

**再扫描发现**：推荐配置（16384 分块）的能量轴只有 LC-20 的
4 个 fp64-only 小格文档；4 个中等格（苯/tzvp 222 AO × 3 方法 +
vitaminC/mtzvpp 298 AO r2scan3c）完全没有推荐配置能量文档，
其 auto 能量道只在默认配置测过（issue 5-8）。LC-17 的"auto SCF
在消费尺寸无速度案例"只在 ≤144 AO 测过。

**矩阵**（生产 rtx4090_runner，推荐配置，4 格 × {fp64, auto,
cpu-fp64} 单文档；每道 warmup + 3 新进程样本、all-repeat 契约、
worst-repeat 聚合；P1-P6 预登记于运行前；证据
`tasks/issue-9/2026-10-07-lc41-moderate-ao-energy-matrix/`，
4/4 文档 schema-VALID、12/12 行 ok、全部样本收敛）：

| 格 | dE(auto) | dE(cpu) | SCF 相 auto/fp64 | 整道 |
|---|---|---|---|---|
| b3lyp/苯/tzvp | +8.5e-13 | -1.9e-12 | 0.71x（慢） | 0.74x |
| m062x/苯/tzvp | -8.2e-12 | -4.3e-12 | **1.17x（快）** | **1.16x** |
| wb97mv/苯/tzvp | +1.2e-11 | +1.128e-8（VV10 轴） | 0.92x（慢） | 0.92x |
| r2scan3c/vitaminC | +1.4e-12 | -1.7e-11 | **1.30x（快）** | **1.28x** |

**结论**：

- **LC-17 精化而非推翻**：auto SCF 的速度案例**从中等尺寸开始**
  （MGGA/复合 MGGA 格 222-298 AO：1.17-1.30x 快）；杂化 GGA
  （b3lyp：fp32 头需 7→10 周期的切换税）与 NLC 主导（wb97mv：
  VV10 无 fp32 路径）仍慢——四个方向全部命中预测，能量门全部
  ≤1.2e-11 通过。交叉点是**方法依赖**的，非纯尺寸。
- 与 LC-40 合并：m062x/苯/tzvp 与 r2scan3c/vitaminC 上 auto 道
  **两相皆胜**（SCF 1.17-1.30x + 梯度 2.43-3.17x）——程序中首批
  "混合精度道在门通过下全场更快"的格子。
- vitaminC 298 AO：fp64 3.54 s vs auto 2.72 s SCF（各 16 周期），
  峰显存 1.49/0.86 GB。wb97mv 的 cpu 偏移 +1.128e-8 与 LC-28/
  LC-40 记录逐位一致（已记录的 VV10 实现差轴）。
- 预测计分：P1-P6 全中（含 P3 四个方向性判断）；无偏差需记录。

复现：`run_energy_matrix.sh`、4×`*.json/*.raw.json/*.runlog.log`
（runner 自锚定 sha256）、`bench_energy.runlog.log`、
`evidence_sha256.txt`。

## 18. LC-42 补遗 (2026-10-07 长链工作单元): ip1 内核融合启动 —— J+K 梯度路径的占用率修复

**再扫描动机**（LC-34 把 df_jk_ip1 列为梯度主桶后的首个未尝试杠
杆；LC-40/41 强化：UKS 分支格上 fp64 SCF 是唯一正确路线，其梯度
相正是被该内核拖慢）。探针（`2026-10-07-lc42-ip1-occupancy-probe/`）
实测：J+K（杂化泛函）路径把 sum_ejk_int3c2e_ip1 内核**按辅助基
l-组逐批串行启动**——每次梯度 8-12 次顺序启动，grid=(21-36, 1)、
~95KB 共享内存（1 块/SM）= 128-SM 4090 上 16-28% SM 占用；内核
合计占 fp64 梯度相 17-40%（benzene/tzvp 288ms/714ms）。

**设计**（静态分析后收窄）：J-only 路径（rhf.py:321，纯泛函）自
上游就通过 y-grid 融合启动——同一生产验证模式。但**子批 y-分块
被证伪拒绝**：交换分支把压缩 dm 寻址为
`dm[pair_offset*naux + k0 + k*nksh]`（nksh = y-块壳数），块边界
内移会读错列。**批边界融合**按构造安全：每个 y-块的寻址与单批
启动完全一致（同 shl_pair x、同批边界、同列、同 atomicAdd 序），
仅跨块原子交错顺序变化（与逐批路径自身的跨运行非确定性同类）。

**落地**（默认关，`GPU4PYSCF_DF_IP1_FUSED=1` 开启）：df/grad/
rhf.py 与 df/grad/uhf.py 的 `_jk_energy_per_atom`（J+K 路径；J-only
与核吸引路径不动）在内存守卫（全宽压缩缓冲 < 可用显存 25%）通过
时，一次启动传完整批边界数组，grid 变为 (n_sp, n_batches)（如
benzene/tzvp (36,5)=180 块）；守卫失败逐字节回退上游行为。

**实测**（`2026-10-07-lc42-fused-ab/`；5 样本中位，全梯度含 mol
重建，逐样本 CUDA 同步，同收敛 mf 同轨道 A/B；wb97mv 2 格验证
omega 分裂路径同样生效）：

| 格 | 道 | \|dg\|（on vs off） | 整梯度加速 |
|---|---|---|---|
| b3lyp/苯/svp | fp64 | 2.9e-13 | **1.48x** |
| b3lyp/苯/svp | auto | 5.1e-14 | 1.15x |
| m062x/苯/tzvp | fp64 | 2.4e-13 | **1.37x** |
| m062x/苯/tzvp | auto | 4.9e-14 | 1.15x |
| b3lyp/甲氧基 UKS | fp64 | 3.3e-14 | 1.10x |
| b3lyp/甲氧基 UKS | auto | 2.7e-14 | 1.01x |
| wb97mv/苯/svp | fp64 | 7.1e-14 | 1.18x |
| wb97mv/甲氧基 UKS | fp64 | 7.1e-15 | 1.08x |

- P1 命中：\|dg\| <= 2.9e-13（原子序噪声级，远低于 1e-10 预登记
  带）；P3 超额：fp64 1.08-1.48x（预测 >= 1.05x；LC-50 勘误：原写
  1.10-1.48x，wb97mv/甲氧基 UKS 实测 1.0848x 圆整应为 1.08x）；P4 命中方向：
  auto 道 1.01-1.15x（fp32 内核本就快，占用率损失小）；P5 命中：
  11 个新单元测试 + 全套件绿（tools 441、df 18、grid 12、
  stability 12）。
- 与累计杠杆正交可叠乘：LC-6（筛除）+ LC-22（分块）+ LC-36
  （缓存）+ LC-42（融合启动）作用于不同环节。此杠杆正是 UKS 分支
  格安全路线（fp64 SCF + fp64/auto 梯度核）上的直接收益。
- 默认关：融合启动改变跨块原子交错顺序（1e-13 级），与 LC-11
  同类，默认开启是用户策略决定。

复现：探针 `probe_geometry.py`/`probe_launch_timing.py`（无源码
修改的 ctypes shim）；A/B `fused_ab.py`/`fused_ab_wb97mv.py`；
单测 `tools/tests/test_df_ip1_fused.py`（11 项：旋钮契约、调用点
契约含子批拒绝契约、同轨道数值等价、单启动几何、默认关多启动）。

## 19. LC-43 补遗 (2026-10-07 长链工作单元): 融合栈梯度文档刷新 —— 9 格 fp64 道在 16384+融合启动

LC-42 落地融合启动旋钮后，以同协议（LC-38：生产 gradient_runner、
推荐配置、fp64 道、3 新进程样本、worst-repeat）重测 9 格矩阵，
叠加 `GPU4PYSCF_DF_IP1_FUSED=1`（证据
`2026-10-07-lc43-grad-fused-refresh/`，9/9 schema-VALID、全部样本收
敛、9/9 pass）：

| 格 | t_LC38 | t_LC43 | 表观加速 | dE | dg |
|---|---|---|---|---|---|
| b3lyp/苯/svp | 0.830 | 0.705 | 1.18x | 1.1e-13 | 1.1e-9 |
| b3lyp/甲醇/svp | 0.672 | 0.636 | 1.06x | 5.7e-14 | 5.1e-11 |
| b3lyp/甲氧基 UKS | 0.778 | 0.776 | 1.00x | 0 | 4.5e-14 |
| m062x/苯/svp | 1.108 | 1.006 | 1.10x | 5.7e-14 | 3.2e-9 |
| m062x/甲氧基 UKS | 1.271 | 1.165 | 1.09x | 2.8e-14 | 4.7e-11 |
| wb97mv/苯/svp | 7.996 | 7.853 | 1.02x | 1.1e-13 | 1.2e-8 |
| wb97mv/甲氧基 UKS | 3.273 | 2.975 | 1.10x | 0 | 5.3e-15 |
| r2scan3c/苯/s1 | 1.177 | 1.362 | **0.86x** | 0 | 1.0e-13 |
| r2scan3c/甲氧基 UKS | 1.165 | 1.229 | 0.95x | 0 | 3.7e-14 |

- dE/dg 全部在 LC-38 自己的跨文档包络内（dE<=1.1e-13、
  dg<=6.9e-9 量级）——P1/P2/P5 命中，无判定变化。
- **r2scan3c 的 0.86-0.95x 是机器状态漂移，非旋钮效应**（如实记
  录 + 对照实验）：控制实验（旋钮关、现在）复测 1.372s ≈ 旋钮开
  1.362s（该二格走 J-only 路径，旋钮按设计不触及）；两态都比
  LC-38（今日早些）慢 ~16%。P3 的 ~1.0x 预测对 r2scan3c 方向正
  确、幅度被漂移掩盖。
- **旋钮的真实文档级效应**（同会话对照，b3lyp/苯/svp）：旋钮关
  0.818s（现在）vs 旋钮开 0.705s（现在）= **1.16x**；与 LC-42 同
  进程 A/B 的 1.48x 一致方向（新进程/文档协议的开销稀释）。
- 7 个杂化格对 LC-38 表观 1.00-1.18x：P3 预测 >=1.05x 命中 5/7，
  b3lyp/甲氧基（1.00x）与 wb97mv/苯（1.02x）在跨时段噪声内
  （偏差如实记录；同会话对照证实旋钮效应未被吞没）。

复现：`run_matrix.sh`、9 组 runner 自锚定文档、
`control-r2scan3c/`（旋钮关对照 2 格）、`evidence_sha256.txt`。

## 20. LC-44 补遗 (2026-10-07 长链工作单元): 中等尺寸融合旋钮配对刷新 —— 旋钮效应随体系增大

LC-40 的 4 个中等尺寸格（苯/tzvp x {b3lyp, m062x, wb97m-v} 222
AO + vitaminC/mtzvpp r2scan3c 298 AO，naux 558/885）全部测于
LC-42 之前——融合启动旋钮在中等尺寸从未测过，而其移除的串行启动
次数随辅助基 l-批数增长，机理上预期更大收益。再扫描排序时，
VV10 梯度内核杠杆已被既有测量关闭（LC-3 上限 ~1.96x 假想/实测
1.03-1.04x 整梯度；LC-10 证伪 rho 剪枝；LC-16 关闭 xc_grid 残
量），故选本单元。

**设计**（LC-43 漂移教训落地：跨会话 ~16% 漂移 -> 全程配对同会
话对照）：每格旋钮开文档（fp64+auto 两道、3 新进程样本、文档内
worst-repeat 判定）后**紧接**旋钮关 fp64 对照文档；两个便宜杂化
格加旋钮关 auto 对照；vitaminC（J-only 路径）为内置无操作/漂移
哨兵。P1-P6 预测先于运行登记于 checkpoint。证据
`tasks/issue-9/2026-10-07-lc44-moderate-ao-grad-fused/`（10/10
schema-VALID，所有样本收敛，所有判定 pass）。

配对同会话 fp64 梯度相 开/关（各 3 新进程样本中位）：

| 格（nao/naux） | 开 | 关 | **加速** | 整样本开/关 |
|---|---|---|---|---|
| b3lyp/苯/tzvp (222/558) | 0.400s | 0.612s | **1.53x** | 1.15x |
| m062x/苯/tzvp (222/558) | 0.467s | 0.726s | **1.55x** | 1.16x |
| wb97mv/苯/tzvp (222/558) | 1.417s | 1.850s | **1.31x** | 1.05x |
| r2scan3c/vitaminC (298/885, J-only 哨兵) | 0.923s | 0.938s | 1.02x | 1.00x |

- **旋钮效应随体系增大**：小尺寸整梯度 1.08-1.48x（LC-42，LC-50
  勘误下限）->
  中等尺寸梯度相 1.31-1.55x；J-only 哨兵 1.00-1.02x 无操作确
  认。P3 主判据（>=1.05x、m062x 最大）命中，但幅度**超出**预登
  记带（1.1-1.35x）——正向偏差如实记录，机理一致（naux 558 vs
  svp ~204 -> 更多串行 l-批启动可融合）。P5 部分命中：b3lyp auto
  1.11x 在带内，m062x auto 1.41x 超带（正向）。
- **组合栈图景**：m062x auto-开梯度相 0.241s = fp64-关 0.726s 的
  3.0x；旋钮开后 auto 道对 fp64 道的梯度相优势保持（0.241 vs
  0.467）。
- 判定与连续性：auto 文档内判定 4/4 pass（dg 1.19e-6/9.27e-7/
  8.81e-7/**9.86e-6**——vitaminC 复现 LC-40 的 9.9e-6 至数字
  级）；跨文档 vs LC-40 基准：fp64 dE<=1.14e-13、dg<=1.56e-8
  （wb97mv VV10 轴的既记录跨文档类；两处边缘超带如实记录，无判
  定变化），auto dE<=2.27e-13、dg<=2.65e-7。
- **用户决策 (b2) 的新输入**：融合旋钮默认开的证据现覆盖两个尺
  寸区间且均有配对同会话对照；中等尺寸效应严格更大。

复现：`run_matrix.sh`（配对设计）、`analyze.py`/
`analyze_output.txt`（P1-P6 判定）、`timing_detail.txt`/
`ctl_timing_detail.txt`（原始样本）、`schema_validation.txt`、
`evidence_sha256.txt`。

## 21. LC-48 补遗 (2026-10-07 长链工作单元): 杠杆总账 —— 全弧线最终状态一览

LC-2..LC-47 弧线结束时，各杠杆的结论散布在 §6-§20 与各 issue 文档。
本节给出单一总账供审查者一图定览（每行状态与证据均为对既有章节的
忠实摘录，无新断言；"CLOSED"= 被实测证据关闭，重试需新证据）。

**SCF 侧（每周期稳态 + 冷启动）**

| 杠杆 | 状态 | 效应（实测） | 出处 |
|---|---|---|---|
| XC 格点分块 16384 | **DELIVERED**（opt-in，默认 4096） | nr_rks 1.26-3.76x；整 SCF 1.07-1.16x；dE≤2.3e-13 | §9 LC-11 |
| cuTensor 句柄预热 | **DELIVERED**（默认开，可 opt-out） | 冷启动 1.26-1.29x（合并 LC-11）；dE≤1.1e-13 | §9 LC-13/14 |
| intopt.build 精简筛除 | **DELIVERED**（默认路径） | build 4.3-4.6x；整 SCF 1.01-1.29x | §8 LC-6/8 |
| MGGA tau/GEMM 核心 | CLOSED（杠杆抗性） | fp32 GEMM 域移位测量，无安全收益 | LC-16 相位记录 |
| VV10 UWE 内核（tile 族） | **CLOSED（三层证据）** | 93% 桶=单内核（nsys）；全变体共享 ~2.5e10 pairs/s 顶（微基准）；tile8 端到端 0.98-1.22x 仅小格受益 | issue-10 + §9 LC-10 + LC-3 |
| VV10 rho 剪枝 | CLOSED（证伪） | 现阈值剪 0 点；削工作阈值破坏能量预算 | §9 LC-10 |
| DIIS 复位（精度切换处） | CLOSED（证伪） | 实测物理否决（交叉重启双向保盆地） | issue-7 §4.2 |
| Int3c2eOpt 缓存（SCF 侧） | CLOSED（证伪） | 构造非热点（SCF 侧一次构建） | §15 LC-37 |

**梯度侧（整相）**

| 杠杆 | 状态 | 效应（实测） | 出处 |
|---|---|---|---|
| 梯度 XC 格点分块 | **DELIVERED**（opt-in，同 b1 旋钮） | 整梯度 1.24-1.27x；同轨道 dg≤1.4e-13 | §10 LC-22 |
| ip1 融合启动 | **DELIVERED**（opt-in，默认关） | 小格整梯度 1.08-1.48x；中等格梯度相 1.31-1.55x（随 naux 增长）；dg≤2.9e-13 | §18/§20 LC-42/44 |
| Int2c2eOpt 缓存（梯度侧） | **DELIVERED**（默认路径，无环境门） | 整梯度 1.03-1.54x；构造 12-24→0-1 次 | §14 LC-36 |
| hcore_e1 向量化 + qcond 筛除 | **DELIVERED**（默认路径） | hcore_e1 2.81-3.10x（合计）；逐位等价 | §7/§8 LC-4/6 |
| VV10 梯度 tile 内核 | CLOSED（测得已转移） | 同入口零改动生效；整梯度 1.03-1.04x（NLC 桶 1.075-1.21x）；无需独立旋钮 | LC-3 相位记录 |
| 梯度 ip1 子批分块 | CLOSED（静态分析拒绝） | 交换分支压缩 dm 寻址布局不安全 | §18 LC-42 |

**精度车道（跨相）**

| 车道 | 状态 | 结论 | 出处 |
|---|---|---|---|
| auto（fp32 头 + fp64 尾） | **DELIVERED** | SCF：价值案例始于 ≥222 AO MGGA 格（1.17-1.30x）；梯度：≥222 AO 梯度相 1.52-3.17x、车道端到端首胜 1.43-1.46x；LC-57 最大梯度格（tamoxifen/mtzvpp 687 AO）：梯度相 4.47x、车道端到端 1.44x，dg 增长饱和（1.06e-5，较 298 AO 仅 +7%，3e-5 门 2.8 倍裕量——LC-40 "重测勿外推" 项以实测关闭）；UKS 支线格正确性约束不变 | §16/§17/§19 LC-40/41/57 |
| auto+verify（LC-29 精化） | **DELIVERED**（opt-in） | 9/9 梯度门 pass（恢复 1.0-1.6e-6）；开销 1.05x（推荐配置）；LC-57b/LC-58：verify 兄弟对象的完整性缺陷两处修复（3c 色散钩子 60de171；原生 disp/nlc 状态 fe46813——修复前 mf.disp 静默错 ref +5.74e-4、mf.nlc='vv10' 崩溃），四族能量平价回归测试常驻守护；既有记录全部 no-fire，零已报告数字错（潜伏缺陷） | issue-12 §9 LC-39 + LC-57b/58 相位记录 |
| 纯 fp32 车道 | CLOSED（证伪） | 全 10 格不收敛且端到端更慢 | issue-9 主矩阵 |

**仍开放（全部为用户决策或外部输入，非本分支可实现项）**

1. 旋钮默认翻转 (b1) 16384 / (b2) ip1 融合 —— 见
   `docs/pending-user-decisions.md`（含一处翻转机理与全部效应表）。
2. UKS 1e-6 门重新表述 (c) —— 同上文档，三选项未排序。
3. 发布通道 67>50 文件阻塞（LC-7 选项 a1/a2/a3；LC-60 实测：6a80ac9
   时点 67 文件 = LC-50 时点 65 + LC-60 cusolver 修复的 2 个新文件
   [lib/cusolver.py + lib/tests/test_cusolver.py，关闭 LC-54 根因的
   用户可达崩溃面所必需]；LC-56 记录增量清单备选：LC 轮次增量
   delta 单独 27 文件 < 50）。
4. G16/ORCA 上传后参照适配（计划 §3.6；不阻塞任何已完成项）。
5. Codex 独立审查 LC-1..LC-47（持续请求；本总账不替代审查）。

审查者注：每行"状态"含义——DELIVERED=已落地并有等价性+加速双证据；
CLOSED=有实测否证记录（保留于证据目录），无新证据不得重试。全套件
在 HEAD fe46813 绿（LC-58 后重验：tools 441、df-mixedprec 18、
stability-verify **18**（LC-57b 的 3c 色散钩子 2 个 + LC-58 的
sibling 完整性 4 个新回归，含四族能量平价审计）、dft rks 16 +
uks 10、xc-grid/loop/int2c2e 37、df-ip1 11——后两组名下套件
48 项同会话重跑通过）。LC-52（2026-10-07）环境注记更新：安装
pytest 后 df discover 全套件 **210 项 OK / 0 错误**（此前记录的
12 个 loader ERROR 为上游测试 import pytest 而环境未装所致；
其中 test_df_rks_grad/test_df_uks_grad 正是 LC-36/LC-42 修改的生产
模块的上游测试，恢复后 93/93 全过——LC-36 缓存与 LC-42 调用点契约
在上游测试面上同样成立；证据
`tasks/issue-9/2026-10-07-lc52-blocked-test-recovery/`）。

LC-54（2026-10-07）环境根因更正（勘误 LC-53 的 segfault 诊断）：
LC-53 记录的 "libcusolver cusolverDnDsygvd 供应商 bug、n≤16 崩、
n≥17 安全" 两点均不成立。实测机制：`gpu4pyscf/lib/cusolver.py`
经 `find_library('cusolver')` 取得 SONAME `libcusolver.so.12`，
加载器将其解析为 **CUDA 13.0** 的库（本机 ld.so.conf.d 同时含
12.8/13.0 两套目录，.so.12 名归 13.0），而 cupy 加载的是 **CUDA
12.8** 的 `libcusolver.so.11.7.3.90` 且句柄由 12.8 创建——ctypes
路径把 12.8 句柄传给 13.0 的 sygvd/zhegvd 即段错误（跨版本句柄
错配，非供应商 bug）。四个方向探针：12.8 自句柄+自库全 n 通过
（2..1024）；13.0 自句柄+自库全 n 通过（5..264）；仅错配对崩溃，
窗口实测为 dsygvd 仅 17≤n≤64 可用（n≤16 与 n≥65 均崩）、zhegvd
仅 13≤n≤64（LC-53 的 "n≥17 安全" 是只扫到 64 的扫描范围假象，
且 itype=3 在大 n 同样崩）；与数据无关（随机 SPD/不定阵/真实
benzene/ccpvtz hcore+ovlp n=264 全崩）。环境级修复（不动仓库代码）：
`/home/qczcode/cusolver-align/libcusolver.so.12` 软链到 12.8 的库，
`LD_LIBRARY_PATH=/home/qczcode/cusolver-align` 后：scf 全套件
114 过 + 1 PCM 环境类失败（此前 3 处段错误全部通过）、EDA
12 过 +1 上游跳过（此前 7 崩）、LC-53 崩溃的 test_dftu/test_gks
全部通过、标准路径能量逐位一致（shim 与否 b3lyp/wb97m-v 双能
bit-identical）——生产路径（minao 初猜 + x_orth 标准对角化）本就
不经过该错配库，结论不变；但错配的用户可达面比 LC-53 记录的宽：
opt-in `init_guess='hcore'`（n=264 苯烯尺度实测崩）、ROHF HF1e
单电子核、EDA/dftu/x2c 路径。LC-53 的"崩溃路径文件与上游逐字节
一致"结论仍然成立（rohf.py/eda.py 均在 0a4cf33 基线无改动）。
**LC-60 仓库侧收口（2026-10-07，提交 6a80ac9）**：上述 shim 从
"必需"降级为"可选"——`lib/cusolver.py` 加载器改为配对 cupy 已
映射进进程的库（`/proc/self/maps` + `cusolverGetVersion` 对
cupy `_getVersion()` 编码匹配，11703==(11,7,3)），无 shim 下
dftu/gks 6+7 skip、EDA 12+1 skip、scf FULL 120+3 skip+1 PCM
环境类（与 LC-54 shim 下结果一致）；修复前 subprocess 探针 4/7
尺寸段错误，修复后 7/7 过（对 scipy 6e-16..1.3e-14）；新增 2 个
回归测试（配对不变式 + n=70/128/264 实/复 eigh——旧测试 n=12 落在
所有安全窗之下，即缺陷对该套件不可见的原因，已在测试内注明）。
同一提交另修复 40-100x 进程压测发现的 cuTensor 预热线程退出期
竞态（原始模块 4/60 段错误、CUTENSOR_WARM=0 0/60 → atexit join
后 0/100；预热行为不变）；证据
`tasks/issue-9/2026-10-07-lc60-cusolver-loader-fix/`。
上游测试面至此已扫过 df/grad/dft/scf/tdscf/gto/solvent/dispersion/
pop/properties 全目录，所有失败均归类（PCM/无 Fortran 环境类，
或本条已根因的 cusolver 错配类），无一与分支改动相关（分支自有
套件 test_stability_verify 12/12、test_mole 等 LC-4/LC-11 表面全
绿）。证据 `tasks/issue-9/2026-10-07-lc54-upstream-test-sweep-2/`
（66 文件 sha256 锚定）。

## 26. LC-78 补遗 (2026-10-12 评审续): 中等尺寸整道匹配性能刷新 + wB97M-V 无胜因解剖（当前源）

评审 2026-10-12 要求：在当前源上刷新中等 AO 匹配性能、以**整
SCF+梯度时间**为准、并调查实测的 wB97M-V 无胜与验证开销。生产
`gradient_runner`，推荐配置 16384，LC-40 cohort（苯/tzvp 222 AO ×
3 方法 + vitaminC/mtzvpp 298 AO r2scan3c），fp64+auto 单文档、
3 新进程样本、all-repeat；源 cd5b5bf（生产路径自 52ad550 未变，
对 LC-40/41 基线的全部偏差均在跨进程包络内）。证据
`tasks/issue-9/2026-10-12-lc78-moderate-ao-perf-refresh/`
（31 文件 sha256 锚定，6 文档 schema-VALID，P1-P6 预登记全中）。

| 格（222-298 AO） | fp64 整道 | auto 整道 | auto/fp64 | 梯度相 | dE(auto) | max\|dg\| |
|---|---|---|---|---|---|---|
| b3lyp/苯/tzvp | 1.380s | 1.266s | **0.92x auto 胜** | 0.312 vs 0.623s（2.0x） | 1.0e-12 | 1.19e-6 PASS |
| m062x/苯/tzvp | 2.110s | 1.558s | **0.74x auto 胜 1.35x** | 0.349 vs 0.744s（2.1x） | -8.4e-12 | 9.15e-7 PASS |
| wb97m-v/苯/tzvp | 9.465s | 9.433s | **1.00x 无胜** | 1.231 vs 1.856s（1.5x） | 1.2e-11 | 8.80e-7 PASS |
| r2scan3c/vitaminC | 4.681s | 3.020s | **0.65x auto 胜 1.54x** | 0.312 vs 0.929s（3.0x） | 1.4e-12 | 9.75e-6 PASS |

- 整道结论：auto 的中等尺寸价值案例**在整道层面成立且扩大**——
  杂化 GGA（b3lyp）的梯度相 2.0x 足以覆盖其 SCF 切换税，整道
  0.92x 首次在文档级记录为 auto 胜（LC-40 文档级 0.92x 同值，
  本次归因更清晰）。
- **wB97M-V 无胜因（组成级实测，非假设）**：VV10（scf.nlc）占
  fp64 SCF 内核 **77.8%** / auto 道 **82.5%**（12.57s/13.91s
  instrumented，benzene/tzvp 16384）；VV10 两道同跑 fp64 内核
  （零 fp32 路径，当前源复核成立），auto 只在少数桶上获益
  （xc.grid 2.09→1.51s）且付出切换税（8→9 周期 = 多 2 次 VV10
  调用 ≈ +1.34s）→ SCF 7.59→7.97s（1.05x 慢）；整道被梯度相
  1.5x 拉平至 1.00x。**方向性结论：wb97mv 的下一个杠杆是
  fp32 可用的 VV10 内核或减少 VV10 调用数，而非在别处加 fp32**
  ——其余桶合计 ≤13%。
- **验证开销（auto-verify，中等 AO 新测）**：wb97mv/苯/tzvp
  verify/auto 1.99x、verify/fp64 1.98x；r2scan3c/vitaminC
  2.61x / 1.68x——参照 SCF 本身被 VV10 主导（同 77.8% 桶），
  诚实全代价即 ~2x。两格均 no-fire（dm 5.4e-5 / 8.0e-6），
  梯度全 PASS。
- 与 LC-40 基线对比：车道比复现（0.92/0.68→0.74/1.00/0.70→0.65x），
  绝对时间 0.96-1.15x = 跨进程包络；r2scan3c 梯度 9.75e-6 复现
  LC-40 的 9.86e-6 族。

## 27. LC-79 补遗 (2026-10-13 评审续): wB97M-V VV10 杠杆测量——调用数杠杆证伪，fp32 结构性 NaN 根因，安全结构下界 3.59x

按评审"从新鲜剖析选择下一个具体优化"的指令，LC-78 §26 点名的
wb97mv 杠杆（fp32 可用的 VV10 内核或减少调用数）本单元先测量后
实现（无生产代码改动）。证据
`tasks/issue-9/2026-10-13-lc79-vv10-lever/`（8 文件 sha256 锚定）。

- **调用数杠杆证伪（实测）**：fp64 wb97mv/苯/tzvp 整 SCF 的
  nr_nlc_vxc 调用模式 = 8 周期 9 次调用（init + 每周期 1 次 +
  conv_check 终检），9/9 次输入密度互异、0 对相邻重复——**不存在
  可缓存的重复 VV10 评估**；调用数即 SCF 不动点迭代本身。
- **全 fp32 VV10 结构性 NaN（根因实测）**：以 cupy 复现 UWE 对
  偶内核表达式（对生产 CUDA 内核复现至 1.4e-17，仿真可信），
  全 fp32 下 E_nlc = NaN：omega = sqrt(C·γ²/ρ⁴ + 4π/3·ρ) 需
  ρ⁻⁴，NLC 格点非零 ρ 低至 1.04e-10（恰在零密度阈值上方），
  ρ⁻⁴ = 8.5e39 **在 fp32 中溢出**（有限 fp64 omega² 最大仅
  1.37e19）→ 污染 g 乘积（154k 对超 FLT_MAX → 倒数 NaN）；
  fp64 逐 i 累加也救不回（变体 C 同样 NaN）。结构性结论：这些
  格点上任何以 fp32 计算 omega（或其乘积）的 VV10 内核必然
  NaN。
- **安全结构通过能量门（4 个量级裕量）**：fp32 装载 + fp64 对
  偶算术（带宽方案）E_nlc 误差 **-1.03e-10 Ha**；合成计时集上
  E 总和误差 +8.2e-8、max|dU| 7.6e-3（势矩阵误差被 SCF 不动点
  重吸收，契约是能量门）。
- **可达加速下界（合成，诚实）**：苯尺度合成数据（127,746 点，
  匹配统计）cupy 对偶内核 fp64 58.4s vs 安全 fp32 结构（fp64
  omega 预计算 + fp32 对偶算术 + 三次独立倒数 + fp32 累加）
  16.3s = **3.59x**。RTX 4090 fp32:fp64 ALU 比 64:1，ALU 受限
  内核下 3.59x 是下界；参照内核实际是共享内存/带宽受限（j 数据
  共享内存分级装载），字节减半另有 ~2x 空间。
- **预登记的下一个实现**：fp32 可用 VV10 = 上述已测安全结构
  （omega/kappa/rho_weight 以 fp64 预计算——ρ⁻⁴ 溢出住在那里；
  对偶算术 fp32、倒数写三次独立 1/g、逐 i 累加 fp32/fp64 均过
  门）。若内核仅达 2x：VV10 占 fp64 SCF 内核 77.8% → SCF 内核
  ~0.61x，wb97mv auto 整道 1.00x 无胜转为 ~0.8x 胜。CUDA 内核
  工作项（vv10.cu 新入口 + opt-in 环境旋钮），非默认改动；
  1e-6 Ha 能量门 + 3e-5 梯度门 + all-repeat 证据为验收标准
  （LC-42 ip1-fused 弧线的同一纪律）。
