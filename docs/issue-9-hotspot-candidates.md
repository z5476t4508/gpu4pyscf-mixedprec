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

## 28. LC-80 补遗 (2026-10-13 评审续): fp32 安全 VV10 UWE 内核落地 —— wb97mv 整 SCF 3.3x，无胜转胜（opt-in）

按 LC-79 §27 预登记的实测安全结构实现
`vv10_fock_eval_UWE_fp32safe_kernel`（gpu4pyscf/lib/gdft/vv10.cu，
选择器 `GPU4PYSCF_VV10_UWE_KERNEL=fp32`，opt-in；默认 warp1 与其它
选择器不变；Python 树不读该 env——issue-10 契约）。证据
`tasks/issue-9/2026-10-13-lc80-vv10-fp32-kernel/`（21 文件 sha256
锚定，P1-P5 预登记全中）。

- **结构（LC-79 实测安全形式）**：omega/kappa/rho_weight 以 fp64
  进入（fp64 omega_derivative 内核不变——ρ⁻⁴ 溢出留在 fp32 之外），
  装载时单次转 fp32 共享内存；对偶算术 fp32、**三次独立倒数**；
  **fp64 块部分和累加**（每 128-j 块 fp32 部分和 → fp64 累加器）。
- **诚实负发现（保留）**：第一版纯串行 fp32 累加在苯/tzvp 实测
  dE **+1.478e-6 Ha 超门**（编译器重结合使串行序不可控）；块部分
  和 + fp64 合并实测 **-1.83e-10**（diag_accum.py：tree-fp32
  +8.5e-10 / serial +1.2e-10 / fp64-acc -2.6e-11——落地形式达到
  fp64 累加水平）。
- **内核级**：苯/tzvp 127,746 点 dE_nlc -1.83e-10、max|dU| 1.0e-10、
  658.7→49.8ms = **13.2x**；甲醇/svp 61,078 点 +4.5e-11、
  158.7→19.6ms = **8.5x**。
- **整 SCF（生产 runner，16384，fp64+auto × 开/关，各 3 新进程
  样本全部收敛，4 文档 VALID）**：苯/tzvp fp64 SCF 7.549→2.308s
  (**3.27x**)，auto 8.197→2.204s (**3.72x**)；甲醇/svp fp64
  2.507→1.030s (**2.43x**)，auto 2.646→1.119s (**2.36x**)。开启时
  重复能量与关闭版在 1e-10 级一致；auto 对本道 fp64 dE<=1.2e-11。
- **LC-78 §26 的 wb97mv 整道无胜（auto/fp64=1.00x）开启旋钮后
  转为胜**：苯/tzvp auto 道 2.306s vs fp64 2.412s = **0.96x**
  （甲醇 1.08x——小体系下 auto 剩余切换税超过其 fp32 收益，如实
  记录）。
- **测试**：tools/tests/test_vv10_fp32_kernel.py 6 项（静态源契约
  + GPU 内核等价 dE_nlc<=1e-8 门 + 整 SCF <=1e-9；对修复前的
  1.5e-6 形式为红）。套件：tools 398 OK（392+6）、df+dft 33 通过；
  gdft 目标构建干净（全量构建的 libxc 克隆失败是无网络外部依赖，
  历史已记录，与本改动无关）。
- **范围诚实**：旋钮 opt-in，无默认改动（若默认开启属 LC-45 (b)
  族用户决策）；VV10 **梯度**内核是独立代码路径，本旋钮不覆盖
  （LC-3 tile 工作在梯度侧；fp32 安全结构需在那里单独实测）。
  **【LC-81 勘误】**上句"梯度内核不覆盖"经实测**不成立**：生产
  梯度（grid_response=False 类默认）走 `get_nlc_exc` →
  `numint._vv10nlc` → **同一个 UWE dispatcher**，旋钮直接生效
  （梯度相 1.4-1.9x 见 §29）；仅 offdiagonal 内核是梯度专属且
  只在 opt-in 的 grid_response=True 路径出现。详见 §29。

## 29. LC-81 补遗 (2026-10-13 评审续): 梯度侧 VV10 测量 —— 旋钮覆盖生产梯度路线（LC-80 记录勘误），梯度相 1.4-1.9x，生产路线无 offdiagonal 内核

测量单位（无生产代码改动；源 a1599c8；证据
`tasks/issue-9/2026-10-13-lc81-vv10-grad-measurement/`，sha256 锚定，
P1-P5 全中 / P6 因路线事实修正而 MISS——预登记基于错误路线，如实
记录）。

**路线事实（源码 + 首次探针失败证明）**：生产梯度
`ks_grad.grid_response = False`（RKS/UKS 类默认，pyscf 继承），故
`_get_denlc` 走 `get_nlc_exc`（grad/rks.py:136）；其 VV10 位势经
`numint._vv10nlc` 调用**与 SCF 相同的**
`VXC_vv10nlc_fock_eval_UWE` dispatcher——LC-80 旋钮对生产梯度
直接生效。offdiagonal 内核仅存在于 opt-in 的
`grid_response=True` 路径（`get_nlc_exc_full_response`），生产
不执行。据此 **fp32 offdiagonal 原型不立项**（预期收益为零）。

**Phase A：整 SCF+梯度道（生产 gradient_runner，16384，fp64+auto
× 开/关，各 3 新进程样本，4 文档 VALID，全部收敛）**：

| 格 | 道 | t_grad 关→开 | 梯度相 | 整样本 关→开 | 整道 |
|---|---|---|---|---|---|
| benzene/tzvp | fp64 | 1.880→1.270s | **1.48x** | 9.59→3.59s | **2.67x** |
| benzene/tzvp | auto | 1.249→0.677s | **1.85x** | 9.56→3.18s | **3.01x** |
| methanol/svp | fp64 | 0.473→0.335s | **1.41x** | 3.09→1.47s | **2.10x** |
| methanol/svp | auto | 0.453→0.312s | **1.45x** | 3.04→1.43s | **2.13x** |

门（4 文档全过）：道内 auto 对本道 fp64 dE<=1.2e-11、每次重复
max|dg|<=1.8e-6（对 3e-5 门 16-34x 裕量）——**两种旋钮态都过**；
跨旋钮 fp64 道 dE 1.1e-12/4.7e-11、max|dg| 9.8e-8（苯）/1.0e-9
（甲醇）。诚实细节：开启态苯 fp64 道 SCF 8→9 圈（auto 9→10），
1e-12 级轨迹扰动翻转收敛判据一圈；整道倍数已含此成本。

**Phase B：梯度相组成（benzene/tzvp，探针
`probe_nlc_grad_decomp.py`，sync 包夹 + 区域键子桶 + finally 还原
+ 观测无损验证 dg 2.1e-14 vs 本底方差 1.4e-14）**：

| 桶 | 关 | 开 |
|---|---|---|
| UWE 内核 | **0.658s（t_grad 的 35.0%）** | **0.050s（3.9%）** |
| e2_df_jk | 0.803s（42.8%） | 0.796s（62.6%） |
| xc_grid_total | 0.265s（14.1%） | 0.279s（21.9%） |
| nlc 总计 | 0.789s（42.1%） | 0.195s（15.4%） |
| t_grad 基线中位 | 1.877s | 1.271s |

UWE 桶 13.2x 收缩与 LC-80 内核级测量逐层一致；开启后梯度相由
df_jk 主导（62.6%）——**下一个梯度侧杠杆是 ip1 族（LC-35/42 已
测绘），不再是 VV10**。

**测试**：`test_vv10_fp32_kernel.py` +5（模块 11 OK）：
`FP32VV10GradientRoute`（梯度有限 / 能量 <=1e-9 / 跨旋钮
max|dg|<=1e-7 门=3e-5 的 300x 裕量，实测 1.0e-9）+
`VV10GradientRouteStatic`（生产路线经 `_vv10nlc`→共享 dispatcher、
offdiagonal 仅在 full_response 体、RKS/UKS grid_response 类默认
保持 False）。整套 tools 467 OK（HEAD 干净态 462 + 5；
LC-80 记录的"398"是较窄发现范围下的计数，当前全量计数为 462）。

**范围诚实**：测量+测试单位，无生产代码改动；旋钮仍 opt-in（
默认开启属 LC-45 (b) 族用户决策）；LC-80 原证据不可变，本节为
勘误 + 新增测量。


## 30. LC-82 补遗 (2026-10-13 评审续): 组合 opt-in 旋钮测量 —— VV10 fp32 × ip1-fused 在 wb97mv 梯度道上叠加生效（fp64 整道 3.07x）

测量单位（无代码改动；源 3306beb；证据
`tasks/issue-9/2026-10-13-lc82-combined-knobs/`，sha256 锚定，P1-P5
全中）。两个 opt-in 旋钮作用于**不相交的实测桶**（LC-81 Phase B：
UWE 35.0% + df_jk 42.8% of t_grad），此前从未同测。四态相邻配对
（A 双关 / B 仅 VV10 / C 仅 ip1 / D 双开），生产 gradient_runner，
16384，fp64+auto 单文档，各 3 新进程样本，8/8 文档 VALID、全部
收敛、全部道内门 PASS（dE<=1.2e-11，max|dg|<=1.8e-6）。

**benzene/tzvp（中位数）**：

| 态 | fp64 t_grad | fp64 整样本 | auto t_grad |
|---|---|---|---|
| A 双关 | 1.884s | 9.624s | 1.240s |
| B 仅 VV10 | 1.253s | 3.645s | 0.604s |
| C 仅 ip1 | 1.434s | 9.111s | 1.216s |
| **D 双开** | **0.820s（2.30x）** | **3.135s（3.07x）** | **0.556s** |

- **桶叠加近似可加**（fp64）：A→B 省 0.631s（UWE）+ A→C 省
  0.450s（ip1）= 1.081s ≈ A→D 实测省 1.064s；D t_grad 0.820s 对
  预登记 ~0.80s。
- **C 态逐位复现 LC-44**：1.434s（LC-44 ON 1.417s，1.31x 族；
  同格 6 天不同源提交——稳定）。
- **诚实 auto 道发现**：ip1 旋钮对 auto 道梯度**无可测效应**
  （C 1.216s ≈ A 1.240s；D 0.556s ≈ B 0.604s）——其作用仅限
  fp64 道每批路径；auto 道的 fp32 ip1 组成不同（LC-35：int2c2e+
  auxresp 主导、内核平坦），融合启动不触及。D-auto 整样本
  3.024s 略高于 B 2.830s，纯 SCF 相跨进程方差（t_scf 2.369 vs
  2.133；t_grad 反而更低），如实记录。
- methanol/svp：组合 t_grad fp64 1.48x / auto 1.40x；ip1 单独
  效应小（fp64 1.07x、auto 无）——41 AO 小体系符合预期
  （LC-44 vitaminC 298 AO J-only 哨兵 1.02x 同族）。
- 跨态 fp64 诊断：D-vs-A dE 1.1e-12、max|dg| 9.6e-8（VV10 旋钮
  水平主导；ip1 的 LC-42 同轨道 2.9e-13 在此粒度不可见，C-vs-A
  的 4.3e-9 是 LC-44 已记录的跨进程新 SCF 噪声类）。圈数细节复现
  LC-81：B/D 的 fp64 道 9 圈 vs A/C 8 圈（1e-12 轨迹扰动翻转收敛
  判据；整道倍数已含）。
- ip1 旋钮生效验证：C/D 日志无 guard 拒绝告警；fp64 道计时逐位
  复现 LC-44 实测效应；`grad_ip1_hits` 计的是 fp32 模式进入次数
  （auto 道 3/3），不是融合启动次数——避免误读。
- **范围诚实**：测量单位，双旋钮仍 opt-in、无默认改动（LC-45
  (b) 族用户决策现在明确覆盖"组合开启"这一选项）；D 态不是新
  生产道，是未来默认决策的购买力证据。

## 31. LC-83 补遗 (2026-10-13 评审续): 融合态 df_jk 桶微观分解 —— 融合启动结构级生效（18→2 调用，内核 4.78x），下一个命名杠杆是 ip1 内核本体

测量单位（无代码改动；源 3ca128e；证据
`tasks/issue-9/2026-10-13-lc83-dfjk-decomp-postknob/`，sha256 锚定）。
LC-35 探针复用（同步包夹逐调用计时、四 DF 梯度类桶边界、3 基线 +
1 仪器化、观测无损验证 equiv 2.0e-14/2.2e-14），wb97mv/benzene/tzvp
fp64 道，A（双关）/ D（双开）相邻新进程对。

**实测（D = VV10 fp32 + ip1 fused）**：

| 片 | A（关/关） | D（双开） |
|---|---|---|
| 桶总计（2 次调用） | 0.8402s | **0.3868s（2.17x）** |
| ip1_kernel | 0.5596s ×18 | **0.1170s ×2（4.78x）** |
| fill3c | 0.0723s ×18 | 0.0722s ×18 |
| opt_build | 0.0483s ×2 | 0.0488s ×2 |
| int2c2e | 0.0422s ×2 | 0.0414s ×2 |
| t_grad 基线中位 | 1.8536s | 0.8185s（2.26x） |

- **融合启动结构级生效**：ip1_kernel 调用数 18→2（每批 → 每次桶
  调用一次融合启动；wb97m-v 是距离分离杂化故桶触发两次），内核片
  4.78x 收缩，大于桶的 2.17x——正是去串行化启动的机制签名。
- fill3c/opt_build/int2c2e 在 A/D 间**逐位不变**（旋钮不触及）：
  LC-82 的不相交桶结论现在下沉到片级。
- **D 态梯度相的剩余最大可寻址片 = 融合 ip1_kernel 本体**
  （0.117s，t_grad 的 14.3%）。代码库存在
  `sum_ejk_int3c2e_ip1_f32` 但仅 precision_mode='fp32' 使用；在
  fp64 道启用它是**精度策略问题**（fp32 伪密度矩阵收缩经 metric
  求解放大，df/grad/rhf.py 源内注释与 LC-35 的 int2c2e+auxresp
  误差路径已记录），属用户决策族——本单元只命名候选及其实测上限
  （本格 <=0.117s），不做任何改动。
- 预登记裁定：P1 数字 MISS（预测 D 桶 ~0.80s，实测 0.387s——预测
  错误地把 ip1 旋钮当作桶中性，如实记录；A 桶 0.840s 与 LC-81 的
  0.803s 在同步开销包络内一致）；P2 HIT 更强（ip1 份额 66.6%→
  30.3%，预测降至 30-40%）；P3 部分 HIT；P4/P5 HIT。
- **范围诚实**：测量单位，旋钮 opt-in 不变；ip1-fp32-on-fp64-lane
  仅命名不实现（精度策略变化，用户决策族）。

## 32. LC-84 补遗 (2026-10-13 评审续): 安全路线整成本随旋钮栈刷新 —— auto-verify 绝对开销 3.03x 下降（18.77→5.90s），控制组逐位不动

测量单位（无代码改动；源 013b6be；证据
`tasks/issue-7/2026-10-13-lc84-verify-cost-knobstack/`，sha256 锚定，
P1-P5 全中；关联 issue-7 安全路线成本 + issue-9 杠杆出处）。
LC-78 的旋钮关态验证成本（verify/auto 1.99x wb97mv、2.61x
r2scan3c）由 VV10 主导的参考 SCF 构成；LC-80/81 的 fp32 VV10 内核
把该 SCF 缩 3.27x。本单元首次同测（生产 gradient_runner，16384，
fp64+auto+auto-verify 单文档，3 新进程样本，旋钮全程导出——含
验证参考；文档内设置一致）。

**实测（2/2 文档 VALID，全部收敛，全部门 PASS）**：

| 格 | 道 | 旋钮关（LC-78） | 旋钮开（LC-84） | 变化 |
|---|---|---|---|---|
| wb97mv/benzene/tzvp | fp64 整样本 | 9.51s | 3.640s | 2.61x |
| | auto 整样本 | 9.55s | 2.860s | 3.34x |
| | **auto-verify 整样本** | **18.77s** | **5.904s** | **3.18x** |
| | verify/auto 比 | 1.99x | 2.06x | 有界（分母同步缩小） |
| r2scan3c/vitaminC（控制组） | auto-verify 整样本 | 7.87s | 7.875s | **1.00x** |

- **安全路线的绝对开销（verify 道 − auto 道）9.2s → 3.0s = 3.03x
  下降**；比率基本不动（预登记即预期"比率有界、绝对成本大幅降"）。
- **证伪控制组成立**：r2scan3c 无 VV10，验证成本逐位不动
  （7.87s），dm 7.96e-6 ≈ LC-78 的 8.0e-6——wb97mv 的改进归因于
  VV10 内核而非会话漂移。
- 决策记录逐重复复现无触发族（policy=auto+verify，
  distinct_solutions=false，selected=main）；wb97mv dm 1.19e-5
  （旋钮关 5.4e-5——参考轨迹 1e-12 级扰动使 dm 在族内移动，判定
  不变；r2scan3c 控制 dm 基本不动，归因隔离）。梯度门全过
  （max|dg| 9.1e-7 / 7.9e-7 / 9.8e-6，LC-78 族 9.06e-7 / 9.82e-6
  复现）。
- **范围诚实**：测量单位；旋钮仍 opt-in——这是安全路线在推荐
  性能栈下的成本证据，不是默认改动，也不宣称验证免费（比率仍
  ~2.1x）；auto+strict（LC-76）的纯 fp64 参考成本仍是旋钮关列。

## 33. LC-85 补遗 (2026-10-13 评审续): 推荐旋钮栈下的中等尺寸整道矩阵补全 —— ip1 使 fp64 道逼近/抹平 auto 整道优势（方向与预登记相反，如实记录）

测量单位（无代码改动；源 e957b0b；证据
`tasks/issue-9/2026-10-13-lc85-knobstack-matrix/`，sha256 锚定）。
三个未在旋钮栈下测过的方法（b3lyp/m062x/r2scan3c；这三个方法
无 NLC，VV10 旋钮惰性——栈效应即 ip1 旋钮的 fp64 道梯度效应），
相邻配对关态控制文档（LC-78 文档已隔 2 个提交；LC-43 跨会话漂移
教训）。生产 gradient_runner，16384，fp64+auto 单文档，各 3 新
进程样本，6/6 文档 VALID、全部收敛、全部道内门 PASS。

| 格 | fp64 梯度关→栈 | 栈效应 | auto/fp64 整道分数 关→栈 |
|---|---|---|---|
| b3lyp/benzene/tzvp | 0.605→0.387s | **1.56x** | 0.919→**1.022**（auto 优势抹平） |
| m062x/benzene/tzvp | 0.712→0.458s | **1.56x** | 0.737→0.870（收窄） |
| r2scan3c/vitaminC（控制） | 0.941→0.915s | 1.03x 无操作 | 0.723→0.727（不变） |

- **头条发现（与预登记方向相反，P5 MISS 如实记录）**：auto/fp64
  整道分数**收窄**而非扩大——ip1 旋钮只加速 fp64 道梯度
  （LC-82 实测），分母侧竞争者变快使 auto 相对优势缩小；b3lyp
  该格 fp64 与 auto 达到平价（1.022）。机制：预登记时误推
  "fp64 变快会使 auto 相对胜扩大"，算术上恰好相反。
- 实际解读（证据而非默认改动）：推荐栈下，小-中尺寸混合精度
  的整道收益主要在 SCF 相（m062x auto SCF 1.045 vs fp64 1.113s；
  r2scan3c 2.66 vs 3.31s）；梯度相的 fp32 优势在杂化上被融合
  fp64 ip1 路径大体追平。wb97mv 仍是 VV10 旋钮主导的格
  （LC-82：整道 3.07x）。
- 门与诊断：道内 auto 对本道 fp64 dE<=8.1e-12、max|dg|<=1.17e-6
  （b3lyp/m062x）与 9.95e-6（r2scan3c，LC-40/78 族）全过；
  跨态 fp64 b3lyp/m062x max|dg| 2.5-3.3e-9（LC-44 跨进程新 SCF
  噪声类；同轨道水平 2.9e-13），**r2scan3c 控制组 3.0e-13 逐位
  级**——栈在 J-only 路径是无操作，机制隔离成立。
- 配对关态控制复现 LC-78 分数于跨会话包络内（b3lyp 0.919 vs
  0.92；m062x 0.737 vs 0.74；r2scan3c 0.723 vs 0.65——最大差，
  在 LC-43 ~16% 漂移类内；vitaminC auto 梯度相是易变分量）。
- **范围诚实**：测量单位；栈仍 opt-in；P5 方向 MISS 保留。

## 34. LC-86 补遗 (2026-10-13 评审续): ip1-fp32-on-fp64-lane 原型落地 —— LC-83 命名杠杆实测（内核 8.8x，fp64 梯度相 1.16-1.29x，opt-in 精度政策旋钮）

实现+测量单位（源 85f0a4e；证据
`tasks/issue-9/2026-10-13-lc86-ip1-f32-fp64lane/`，sha256 锚定）。
LC-83 命名的"最大剩余可寻址梯度碎片"（融合态 ip1_kernel 0.117s
= t_grad 的 14.3%）按预登记实现为 opt-in 旋钮
`GPU4PYSCF_DF_IP1_F32=1`：三个 DF ip1 内核选择点（rhf J+K、
rhf J-only、uhf J+K）由 `fp32 = get_precision()=='fp32' or
_ip1_f32()` 门控——内核及其伪 DM/dm+auxvec **输入**走 fp32，
j3c@dm 收缩、j2c 度规求解与 ejk/ejk_aux 原子累加目标**全部保持
既有 fp32 纪律的 fp64 侧**。默认关；核吸引路径（grad/rhf.py
fakemol）按设计不触及；auto 道梯度本就解析为 fp32，结构无操作。
+16 回归测试（解析器契约、三调用点源码契约、内核选择 shim 于
fp64 模式下验证 f32 真被调用、同轨道数值界），tools 483 OK
（467+16）、df-mixedprec 18、上游 df-grad 82 全绿。

三证据层级（全部新鲜进程、全部收敛）：

| 层级 | wb97mv/benzene/tzvp（D 态） | r2scan3c/vitaminC（J-only 控制） |
|---|---|---|
| 同轨道 A/B max\|dg\| | **7.99e-7**（A 态 7.37e-7；态内噪声 2e-14） | **8.94e-6**（A/D 态一致；态内 4e-13） |
| 同轨道 t_grad | 0.818→0.704s = **1.16x** | 0.887→0.754s = **1.18x** |
| 内核碎片（LC-83 探针配对） | ip1_kernel 0.1170→**0.0133s = 8.8x**；桶 0.389→0.297s；fill3c/opt_build/metric 逐位稳定 | — |
| 生产道 t_grad（3 样本中位） | fp64 0.834→0.727s = **1.15x**；整道 3.263→3.133s | fp64 0.950→0.734s = **1.29x**；整道 4.624→4.321s |
| 跨进程 fp64 旋钮效应 | 7.4-7.8e-7（复现同轨道值；dE<=2.3e-13） | 8.96-8.98e-6（dE=0 逐位级 SCF） |
| auto 道无操作验证 | 1.0-1.6e-7 = LC-82 A/C 包络（0.7-1.2e-7） | 2.9-4.6e-7 = LC-85 off/stack 包络（2.8-3.9e-7） |

- **头条机制发现（P2 量级 MISS 如实记录）**：fp64 ip1 内核在
  RTX 4090 上是 **fp64 计算率束缚**（1/64 rate），非占用/共享内存
  束缚——fp32 内核 **8.8x**，远超预登记的 1.3-2.3x（错误锚定：
  预登记假设占用瓶颈）。t_grad 1.16x（预测 5-8%，实测 14%）与
  整道 4%（预测 1-3%）同源偏差。P1/P4/P5/P6 HIT，P3 部分。
- **机制分解副产品**：r2scan3c auto-vs-fp64 差从 9.86e-6 降到
  1.99e-6（当 fp64 参考也带旋钮）——该格 auto 道误差 ~80% 就是
  ip1 内核交换本身；旋钮使两道共享同一内核精度。
- **精度政策诚实（本旋钮非纯收益类，与 (b1)/(b2) 不同）**：
  旋钮改变 fp64 道**自身**——开启时该道不再是严格 fp64 参考
  （杂化 ~8e-7、J-only 8.9e-6 的内核交换代价，3e-5 门 3.4x
  裕量；LC-57 尺寸增长警示：687 AO 未测，auto 道同类误差在
  tamoxifen 已到 1.06e-5）。门裁定必须用关态 fp64 参考；本单元
  D_f32 文档的 auto 门用同文档 fp64 参考仅为道内一致性检查，
  严格参考比较（auto vs 关态 fp64 = 9.86e-6）同样 PASS、无隐藏。
- 新待决策 **(b3) GPU4PYSCF_DF_IP1_F32 默认**（决策简报已加行）；
  旋钮保持 opt-in，默认逐字节上游。

### §34 续（LC-199，2026-10-18）：(b3) 具名 687 AO 未测缺口闭合 —— 带顶同轨道内核交换代价 1.09-1.11e-5（门裕量 2.71x），交换效应随尺寸增大（t_grad 1.18x@298 → 1.39-1.40x@687）

测量单位（无代码改动；源 HEAD 221143c，df/grad/rhf.py 自 LC-86 未变；
证据 `tasks/issue-9/2026-10-18-lc199-ip1f32-687ao-sameorbital/`，14 文件
sha256 锚定；预注册 cd846343 于运行前 08:19:35Z）。LC-86 协议逐字
（一次收敛 SCF conv_tol 1e-11、同轨道下每态 3 次梯度、_IP1_F32 模块
缓存切换、探针自证 env 旋钮未设）在**准入带顶 J-only 格**
tamoxifen/r2scan3c/def2-mtzvpp 687 AO（LC-198 同格；naux 2626）。
态 A/D（LC-86 约定）分离式串行 08:20:26-08:22:22Z = 1m56s，2/2
rc=0。**裁定 5/5 HIT**：P1 max|dg| off/on = **1.108e-5（D）/ 1.091e-5
（A）**，落在预注册带 [7e-6,2e-5] 内、恰在 LC-57 的 auto 道同类误差
1.06e-5 中心估计上；**3e-5 门裕量 3.4x@298 → 2.71x@687**（全已测带
>2.5x）。P2 态内原子噪声底 8.2-8.9e-13。P3 两态 e_tot 对冻结锚
+4.7e-10（conv_tol 1e-11-vs-1e-10 同族尾部）。P4 fp64 道 t_grad
off→on 5.05→3.63s = **1.391x（D）/ 1.403x（A）**——旋钮自身的梯度相
效应随体系增大（1.18x@298 naux 690 → 1.40x@687 naux 2626，与 (b3)
条目"效应随体系增大"一致）。P5 协议。**读数**：(1) (b3) 代价曲线在
两端闭合 {杂化@298 7.4-8.0e-7 / J-only@298 8.9-9.0e-6 / **J-only@687
1.09-1.11e-5**}；(2) 态平权成立（A/D 差 2%，J-only 无操作保持）；(3)
与 LC-198 分解自洽：本探针 off 态 t_grad 5.05s（fused 开、ip1 fp64）
→ 3.63s（ip1-f32）→ LC-198 完整栈 fp64 t_grad 1.25s（再加
GRAD_GRID_F32）——J-only t_grad 分解可复合。决策支持-only：(b3)
默认开/关仍是用户待决，不变；测量-only；默认与门槛不变；旋钮保持
opt-in。决策简报 (b3) 条目的"687 AO 未测"行已同步更新。发件箱保持
在 221143c 发射（本段折入下次发射）。

## 35. LC-87 补遗 (2026-10-13 评审续): 后 LC-86 梯度全桶分解 —— tau_grad_dot 是最大剩余可寻址碎片；NLC 梯度环 fp64 不对称（auto 道结构性缺口，源码级证实）

测量单位（无代码改动；源 ea24c69；证据
`tasks/issue-9/2026-10-13-lc87-xcgrid-grad-decomp/`，sha256 锚定）。
LC-81 探针扩展三个包裹件（tau_grad_dot/h1e_dm/eval_rho）后，在推荐栈
（VV10=fp32+IP1_FUSED=1）下对 wb97mv/benzene/tzvp 做全桶分解，含
LC-86 旋钮开关两种态 + auto 道控制；全部新鲜进程、3+3 样本、obs-only
1.5-4.4e-14；基线逐位复现 LC-86/LC-83（D 0.813s / D_f32 0.690s）。

| 桶 | D（fp64 道） | D_f32（fp64 道） | D_f32 auto 道 |
|---|---|---|---|
| t_grad | 0.813s | 0.690s | 0.542s |
| e2_df_jk | 0.349s (42.9%) | 0.241s (35.0%) | 0.244s (44.9%) |
| xc_grid_total | 0.271s (33.3%) | 0.263s (38.1%) | 0.109s (20.1%) |
| — tau_grad_dot | 0.134s | **0.133s (19.3%)** | 0.018s |
| — gga_grad_sum（xc 环） | 0.090s | 0.090s | 0.019s |
| — nlc_gga_grad_sum | 0.089s | 0.089s | **0.089s（16.5%）** |

- **头条 1**：`_tau_grad_dot_`（MGGA tau 项收缩）是后 LC-86 梯度单笔
  最大可寻址碎片（19.3% of t_grad）；auto 道 7.4x 快（AO 到达即
  fp32）——fp64 GEMM 束缚类（LC-86 同机制）。fp64 道上用它换速度是
  (b3) 同族精度政策决定，此处仅记录实测裕量 ~0.115s（tau+gga 两件）。
- **头条 2（结构性，源码级证实）**：auto 道主 XC 梯度环
  （`_get_exc_task` 三分支均有 `fp32_grid` AO cast）与 NLC 梯度环
  （`get_nlc_exc` deriv=2 循环的 `_gga_grad_sum_` 收缩，**无任何
  precision 处理**）不对称——dtype 实测：xc 环 AO=fp32（auto），
  NLC 环 AO=fp64（两道皆是）。代价实测：`nlc_gga_grad_sum` 两道同为
  0.089s（auto 道 t_grad 的 16.5%），若对齐主 XC 纪律 auto 道该桶
  约 -0.07s（t_grad 0.542→~0.47s，~13%）。非历史刻意排除
  （fp32_grid 块早于此分支存在；get_nlc_exc 仅 LC-22 动过分块）。
  **对 auto 道补齐该环属于既有 fp32 政策面的同类修改，非 (b3) 新
  精度决定**——下一实现单位（LC-88）预登记对象。
- P3 结构 MISS 如实记录：预测最大 xc 碎片为 deriv2_loop_ao（实测仅
  0.4-0.8%）——LC-81 探针未包裹 tau_grad_dot，v1 归入未解释内部，
  v2 扩展后暴露；AO 求值本身便宜，收缩才是成本。
- 范围诚实：测量单位；三旋钮仍 opt-in；tau 杠杆（fp64 道）记录为
  (b3) 族裕量、不实施；NLC 环对齐（auto 道）为下一单位的独立预登记
  对象。

## 36. LC-88 补遗 (2026-10-13 评审续): NLC 梯度环 fp32 对齐落地 —— auto 道 t_grad 1.09x（benzene），同轨道交换代价 1.1e-9（三阶低于道族误差）

实现+测量单位（源 a5c9325 + 本改动；证据
`tasks/issue-9/2026-10-13-lc88-nlc-loop-fp32/`，sha256 锚定）。
LC-87 发现的结构性不对称按预登记修复：`get_nlc_exc` 的 deriv=2
网格收缩环获得与主 XC 梯度环（`_get_exc_task` 的 `fp32_grid`，上游
继承）完全相同的纪律——fp32 模式下（auto 道已解析的梯度模式）AO
逐块 cast 一次，`_gga_grad_sum_` 的 GEMM 按操作数 dtype 运行，对
fp64 dm_mask 的 einsum 与 exc1 累加器保持 fp64。fp64 道不触发
（cast 仅在 fp32 模式）；UKS 共用该函数（uks.py 导入）。**同类
修改而非新政策**：无新旋钮、无默认翻转——precision_mode=None
（全 fp64）行为逐位不变；本修复使 NLC 路径服从车道已选择的精度
模式。rho 环与 full_response（opt-in 路线）不触及（源码契约测试
钉住）。

实测（推荐栈 VV10=fp32+IP1_FUSED=1+IP1_F32，全部新鲜进程、全部
收敛）：

| 证据层 | benzene/tzvp | methanol/svp |
|---|---|---|
| 同轨道 NLC 交换代价 max\|dg\| | **1.121e-9** | **1.157e-9** |
| 同轨道 t_grad | 0.545→0.485s（-0.060s） | 0.306→0.308s（无增益，诚实记录） |
| 生产道 auto t_grad（3 样本） | 0.579→**0.530s = 1.09x** | 0.329→0.308s（跨文档，LC-82 参考隔 6 天——漂移告示；机制级真值=同轨道：无增益） |
| 生产道 fp64 t_grad | 0.727→0.713s（噪声带，cast 不触发） | 0.315→0.297s（噪声带） |
| auto-vs-本道 fp64 门 | dE 1.08e-11 / max\|dg\| 5.533e-7（修复前 5.115e-7，同族）PASS | dE 1.08e-12 / 4.350e-7 PASS |

- +9 回归测试（源码契约×5：cast 单点门控于 deriv=2 环内、累加器/
  dm einsum 不变、full_response 不触及、UKS 共用；dtype shim×2：
  fp32 道 NLC AO=fp32 / fp64 道=fp64；同轨道数值界 5e-6（实测
  1.1e-9）；fp64 道重复底噪测试）。tools 492 OK（483+9）、
  df-mixedprec 18、RKS vv10/nlc 全绿。
- **P1 部分 HIT 如实记录**：benzene 生产道增益 0.049s（同轨道
  0.060s）vs 桶估计 0.07s——fp32 NLC 收缩不及主 XC 环测得的等价
  物快（NLC 环没有预置 fp32 缓冲，astype 逐块分配）。
- **范围诚实**：UKS 主 XC 梯度环（grad/uks.py::_get_exc_task）同样
  无 fp32_grid——LC-87 dtype 证据已记录，属下一同类候选（先测后
  改），本单元未动；tau_grad_dot（fp64 道）仍是 (b3) 族裕量。

## 37. LC-89 补遗 (2026-10-13 评审续): UKS 主 XC 梯度环 fp32 对齐 —— 先测后改的诚实负结果（不实施）

测量单位（源 26f80ca 无改动；证据
`tasks/issue-9/2026-10-13-lc89-uks-xc-fp32-gap/`，sha256 锚定）。
LC-88 收尾命名的下一同类候选按预登记"先测后改"纪律测量完毕：
**在范围内的 UKS 格上该对齐是净损失，候选关闭**。

**结构确认**（dtype 探针，两格、梯度 fp32=auto 道已解析模式）：
UKS `get_exc` 主 XC 区域 `_gga_grad_sum_`/`_tau_grad_dot_` 收到
fp64 AO（wb97mv 格 30+30 次调用、b3lyp 格 30 次），而 NLC 区域
（经 `_get_denlc` 别名）收到 fp32 AO（LC-88 后）——不对称性与
预登记 P1 完全一致：UKS 主 XC 环是 auto 道上最后一块 fp64 网格
收缩面。

**同轨道 A/B**（LC-88 协议：一次收敛 auto-SCF，每侧 3 梯度样本；
候选以忠实 monkeypatch 模拟——AO 逐块 cast、收缩辅助函数
`_gbuf(None)` 使 GEMM 按操作数 fp32、rho/dm_mask/exc1 累加器保持
fp64）：

| 格（配置） | t_grad A→B | 比值 | max\|dg\| | 判定 |
|---|---|---|---|---|
| wb97mv/甲氧基 UKS/svp | 0.512→0.533s | **0.960x** | 1.65e-7 | 更慢 |
| b3lyp/甲氧基 UKS/svp | 0.186→0.205s | **0.909x** | 2.74e-7 | 更慢 |
| m062x/甲氧基 UKS/tzvp（默认块） | 0.275→0.320s | **0.860x** | 4.10e-8 | 更慢 |
| m062x/甲氧基 UKS/tzvp（16384） | 0.205→0.214s | **0.956x** | 1.25e-7 | 更慢 |

数值上安全（dg 族 4e-8..2.7e-7，全部 ≤1e-6 裕量），但四种配置
一致为 4-14% 的 t_grad **损失**——预登记判定规则（增益 ≥5% 且
dg ≤1e-6 才实施）解析为**负证据：不实施**。UKS auto 道主 XC 梯度
收缩保持 fp64（比 RKS 等价物更保守，正确方向的不对称）。

**机制**（微基准，m062x/甲氧基/tzvp 真实形状，nao=118）：
默认块大小 4096 下 fp32 `_gga_grad_sum_` 单次调用反而 **0.66x**
（fp32 路径失去预置 fp64 缓冲，且 `_make_dR_dao_w` 从单一融合
内核退化为 ~21 次小启动——launch/alloc 受限而非计算率受限）；
16384 下逐次调用确有 fp32 优势（gga 1.95x、tau 3.05x）但真实
掩码环的 nao_sub 形状小于微基准满 NAO 形状、加上逐块 cast 流量，
整梯度净效应仍为负。RKS benzene/tzvp 的增益不迁移至范围内 UKS
格（更小分子、每块 AO 覆盖更稀、每块双自旋遍历）。

**预测判定**：P1 命中；P2 方向 MISS 如实记录（预测 +5..40%，实测
-4..-14%——迁移假设被证伪）；P3/P4/P5 命中（数值安全、双分支
GGA 与 MGGA+NLC 无 dtype/形状错误、A 侧中位数无补丁序伪影）。

**范围诚实**：无源码改动、无默认改动、旋钮全 opt-in。候选在
范围内尺寸关闭；假想的更大 NAO UKS 格可能翻号（16384 微基准
满 NAO 形状有逐调用优势），但小/中 Cohort 中不存在该格——重开
需新范围内格，不允许外推。tau_grad_dot（fp64 道）仍是 (b3) 族
裕量、不实施。

## 38. LC-90 补遗 (2026-10-13 评审续): LC-88 后 auto 道梯度组成刷新（推荐栈）—— 道内 fp32 转化已耗尽，无 ≥0.05s 新杠杆

测量单位（源 26f80ca；证据
`tasks/issue-9/2026-10-13-lc90-postlc88-auto-composition/`，sha256
锚定）。LC-87 探针按预登记复制扩展（UKS 自旋类分派 + uks.get_exc
区域包裹 + uhf/uks 类字典补丁；适配中两个闭包缺陷由运行期断言捕获
后修复，修复版为证据版），LC-83 df_jk 探针仅改 lane=auto。3 基线 +
3 插桩样本，观测等价 4.1e-14 / 3.6e-15。

**wb97mv/benzene/tzvp（RKS，LC-87 锚定格）：**

| 桶 | t (s) | 占比 |
|---|---|---|
| t_grad 基线 | **0.458**（0.446-0.488） | — |
| e2_df_jk（2 次调用） | 0.249 | **54.4%** |
| nlc_total | 0.106 | 23.2% |
| — uwe_kernel（fp32，LC-80） | 0.050 | 10.8% |
| — nlc_gga_grad_sum（fp32，LC-88） | **0.012** | 2.6% |
| xc_grid_total | 0.077 | 16.9% |
| hcore_e1 | 0.042 | 9.2% |

对 LC-87 的 LC-88 前 D_f32_auto 记录：t_grad 0.542→**0.458s**
（LC-88 生产道效应在组成级可见），nlc_gga_grad_sum 0.0894→
**0.012s**；e2_df_jk 0.244→0.249s（LC-86 auto 道 ip1-f32 结构
无操作复现）。

**wb97mv/甲氧基/svp UKS（首个 UKS auto 道组成）**：t_grad 0.314s；
e2_df_jk 59.7%；xc_grid 26.1%（tau_grad_dot 0.021s fp64 =
LC-89 关闭面；gga_grad_sum 0.012s fp64）；nlc 12.3%。

**auto 道 df_jk 内部**（benzene/tzvp，桶 0.279s）：fill3c 0.072s
（25.8%）> opt_build 0.049s（17.5%，梯度内缓存已被 LC-37 实测
证伪——3 次构建 3 个不同键）> int2c2e 0.041s > other 0.064 > … >
**ip1_kernel 仅 0.013s（4.8%，fp32 内核结构项）**。

**结论（P5 诚实分支）**：auto 道在推荐栈下已从精度受限转为
积分构建受限——ip1/UWE/RKS-XC/NLC 收缩全部 fp32；剩余 ≥0.05s
的片段均为不可约积分求值（fill3c）、已 fp32 的内核（uwe）、
已证伪的候选（opt_build 缓存）或同步/胶水残差。**auto 道梯度
优化面在现范围内实测耗尽**；fp64 道的 tau_grad_dot（0.133s，
LC-87）仍为 (b3) 族政策裕量。预测判定：P1 命中、P2 拆分（df_jk
主导命中 / nlc 占比 ≤10% 未中——uwe+胶水托住）、P3/P4 命中、
P5 命中（预登记的"无杠杆"诚实分支）。

## 39. LC-91 补遗 (2026-10-13 评审续): 推荐栈下 SCF 相组成（当前源）—— VV10 SCF 地板 88.7ms/cyc、UKS auto 道 11→19 周期税（新实测）

测量单位（源 7e9ed79；证据
`tasks/issue-9/2026-10-13-lc91-scf-composition-recommended/`，sha256
锚定）。LC-12 的 SCF 分解（2026-10-06）早于 LC-80/86/88 与全部旋钮
组合——本单元以 LC-12 探针扩展（臂重定义为四旋钮推荐栈 vs 全关；
增加 fp64/auto 道维；新增 wb97mv/benzene/tzvp 格）重测：5 格 × 2 臂
× 2 道 × 3 新鲜进程 = 60 个 SCF，全部收敛（最差同组能量展宽
3.4e-12）。纯测量；旋钮全 opt-in。

**推荐栈稳态每周期（ms/cyc，中位）**：

| 格 | xc fp64/auto | nlc fp64/auto | (j+k+jk) fp64 | t_scf fp64/auto (s) |
|---|---|---|---|---|
| b3lyp/benzene/svp | 27.5 / 30.8 | — | 5.1 | 3.16 / 3.22 |
| m062x/benzene/svp | 57.6 / 56.1 | — | 6.8 | 3.52 / 3.60 |
| wb97mv/甲氧基/svp UKS | 56.4 / 50.2 | 33.0 / 31.6 | 15.8 | 4.17 / 4.00 |
| r2scan/benzene/mtzvpp | 84.7 / 70.7 | — | 31.3 | 4.01 / 3.68 |
| wb97mv/benzene/tzvp | 96.0 / 62.5 | 88.7 / 86.2 | 24.2 | 5.11 / 4.97 |

（对照默认臂 nlc：tzvp 734.6、UKS 178.2 ms/cyc——LC-80 内核的 SCF
侧效应 8.3x / 5.4x。）

**发现**：
1. 推荐栈下 SCF 侧顶部稳态碎片 = xc 网格列车（tzvp 96.0ms/cyc，
   LC-12 已命名的 _eval_rho2/tau GEMM 内核级深工程方向）+ LC-80 后
   VV10 地板（88.7ms/cyc，"更少调用"已被 LC-79 证伪、内核已 fp32）；
   DF jk 仅 5-31 ms/cyc——SCF 侧为网格受限（LC-12 结论在当前源 +
   旋钮栈下再量化）。
2. **新实测：UKS 双重态在默认分块下 auto 道 SCF 周期税 11→19 周期**
   （每道全部 3 样本一致；+2.4s 整 SCF）；推荐栈恢复 11 周期。将
   LC-78 的 switch-tax 族扩展到周期维度：fp32 头拖慢该双重态的 DIIS
   收敛，VV10 fp32 内核 + 16384（每周期 5.4x）恢复之。
3. auto 道稳态 xc ≈ fp64（杂化格；b3lyp auto 反而略慢，整 SCF 平价
   3.22 vs 3.16s——与 LC-85 的 b3lyp 平价一致）：fp64 尾主导稳态窗口；
   auto 优势集中于 fp32 头覆盖更多周期的格（tzvp 1.54x、r2scan
   1.20x）。
4. **SCF 侧亦无新的 ≥0.05s 免费杠杆**；b3lyp jk 稳态臂间差
   （7.7→5.1ms/cyc）为噪声带（两 ip1 旋钮经源码验证仅梯度侧生效，
   df/grad/rhf.py）。

**预测判定**：P1 拆分（方向双格命中 8.3x/5.4x；≤60ms/cyc 阈值 UKS
命中 33.0 / tzvp 未中 88.7）；P2 命中（非 wb97mv 格仅 xc 桶随臂变
化，b3lyp 稳态 xc ~39% 削减 = LC-12 的 21-50% 族）；P3 MISS（部分
方向：b3lyp auto 稳态反而慢；最大优势在最大格而非预测的 MGGA——
fp64 尾主导）；P4/P5 命中。探针适配（臂/道/新格）如登记记录。

## 40. LC-92 补遗 (2026-10-13 评审续): UKS auto 道 19 周期税机制诊断 —— fp32 头的盆地导引（LC-29 支分裂格），非精度地板

诊断单位（源 082e6c3；证据
`tasks/issue-9/2026-10-13-lc92-uks-cycle-tax/`，sha256 锚定）。LC-91
发现 2 的四轨迹回调追踪（每周期 dE/|g|/|ddm|/活跃精度；车道
{fp64,auto} × 臂 {default,recommended}）：

| 运行 | 周期 | fp32 周期 | 终态 E (Ha) | t_scf |
|---|---|---|---|---|
| fp64/default | 11 | 0 | -114.9016950130 | 6.08s |
| auto/default | **19** | 6 | **-114.9016962956** | 8.85s |
| fp64/recommended | 11 | 0 | -114.9016950130 | 4.14s |
| auto/recommended | 11 | 6 | -114.9016945880 | 4.24s |

**机制（四轨迹一致）**：
1. fp32 头 6 个周期与 fp64 逐步一致（dE/|g|/|ddm| 三四位数字
   吻合），第 7 周期两 auto 臂同时在单触发 |dE|<1e-4 规则下切换
   ——H1（头部震荡）被证伪。
2. 税非"尾部再收敛"（H2 的预测形式被证伪）：auto/default 的第
   7-10 周期几乎逐位镜像 fp64 的 7-10。分歧在第 10-11 周期：auto
   到达**另一解** E=-114.9016962956，比 fp64 道**低 1.283e-6 Ha**
   ——正是本格的 LC-29 支分裂（occ-SVD beta~0.497, dm~0.298 历史族
   ）。多出的 8 周期 = 该支上近阈值 DIIS 行走：dE 停在 ~3e-10、
   |g| 九个周期贴 6.178e-6 地板，直至 |ddm| 衰减满足组合收敛。
3. 推荐臂导引到第三个邻近点（-114.9016945880，+4.25e-6 vs fp64）
   并 11 周期无停滞收敛——分块/VV10 旋钮组合改变 fp32 头轨迹，
   落在 DIIS 地貌正常的支上。**税是支-地貌依赖（fp32 盆地导引），
   非精度地板固有**；LC-77 已实测本格位移的分块依赖（默认触发、
   16384 不触发），本诊断与其同族。

**诚实框定**：19 周期运行收敛（到低能支——auto+verify 政策在本格
历史上为该支采纳 reference；支分裂本身仍待用户决定 (c)，此处不
改写）。周期税是该轨迹的成本。无范围内工程杠杆可跟：停滞是支的
DIIS 地貌在 conv_tol 1e-10 下的属性；切换规则四次运行逐同；更紧
conv_tol 或更早切换均属政策类。**推荐栈已在本格消除该税（11 周期）
——待决 (b1)+(b2) 默认化决定的又一实测收益证据，非新旋钮**。

预测判定：P1 结构 MISS 如实（多周期确在 fp64 尾但非慢尾再收敛
——是 LC-29 低支上的停滞）；P2 命中（第二子结果：切换点不变、
切换后轨迹/支改变）；P3 被轨迹证伪。诊断-only；旋钮全 opt-in；
默认不变。

## 41. LC-93 补遗 (2026-10-13 评审续): 全推荐栈下的验证整成本 —— 安全路线成本为参考-SCF-受限，ip1 旋钮对验证零增益；LC-86 内核共享副产品在验证语境复现

测量单位（无代码改动；源 e314e49；证据
`tasks/issue-7/2026-10-13-lc93-verify-cost-fullstack/`，sha256 锚定；
关联 issue-7 安全路线成本 + issue-9 杠杆出处）。LC-84 只测了
VV10-旋钮栈；两个 ip1 旋钮（LC-42/LC-86）与 LC-88 的 NLC 环改动
晚于它，在验证语境下未测过。协议与 LC-84 逐字相同（生产
gradient_runner，16384，fp64+auto+auto-verify 单文档，3 新进程样
本，四个旋钮全程导出；格 wb97mv/benzene/tzvp +
r2scan3c/vitaminC 无-VV10 控制组）。2/2 文档 VALID（各 3 行 3
过），全部道、全部重复收敛。

| 格 | 道 | LC-84（VV10 栈） | LC-93（全栈） | 变化 |
|---|---|---|---|---|
| wb97mv/benzene/tzvp | fp64 整样本 | 3.640s | 3.009s | **1.21x**（t_grad 1.256→0.702 = 1.79x） |
| | auto 整样本 | 2.860s | 3.002s | 0.95x（t_scf 漂移 +0.22s 净过 t_grad 增益 1.14x） |
| | **auto-verify 整样本** | **5.904s** | **5.852s** | **1.01x** |
| | verify/auto 比 | 2.06x | 1.950x | 绝对开销 3.0→2.85s |
| r2scan3c/vitaminC（控制） | auto-verify 整样本 | 7.875s | 7.800s | **1.01x** |
| | verify/auto 比 | 2.71x | 2.502x | 绝对开销 4.97→4.68s |

- **发现 1（P1 MISS 如实）**：验证整成本不再下降——auto-verify
  道由 fp64 **参考 SCF** 主导（wb97mv t_scf 5.21/5.85s；r2scan3c
  7.27/7.80s），而 ip1 旋钮为梯度侧（源码验证 df/grad/rhf.py），
  参考比较逻辑与参考 SCF 皆无增益。验证成本面现已测全三种栈态
  （LC-78 关 / LC-84 VV10 / LC-93 全栈）：**成本为参考-SCF-受限，
  VV10 内核是其唯一实测杠杆**。
- 发现 2：fp64 参考道整文档 1.21x（两 ip1 旋钮复合 1.79x t_grad，
  与 LC-42 的 1.08-1.48x + LC-86 的 1.16x 一致）——加速的是基准
  道而非安全路线。
- **发现 3（P4 命中）**：r2scan3c 文档内 auto-vs-fp64 门从
  8-10e-6 族降至 **1.9e-6**——LC-86 的机制副产品（该格 auto 误差
  ~80% 来自 ip1 内核差异）在验证语境复现：IP1_F32=1 使两道共享
  fp32 内核。**(b3) 语义诚实**：本文档的"fp64"道运行 fp32 ip1
  内核，其门是 auto-vs-旋钮开-fp64，非严格 fp64 参考门——严格
  参考证据仍是旋钮关记录（LC-78/LC-84），旋钮维持 opt-in 待决
  (b3)。
- 发现 4：wb97mv auto 道 t_grad 0.600→0.525s 在本格复现 LC-88 的
  生产效应（其文档 0.579→0.530s）；auto 道整文档 0.95x 为 t_scf
  跨会话漂移净过 t_grad 增益（ip1 旋钮对 auto 道结构无操作，
  LC-82/LC-86 已证）——无结构性回归。
- 决策记录：两格全部重复 no-fire、selected=main、
  distinct_solutions=false（LC-84 判定物理不动摇，P5 命中）；
  门：wb97mv dE -1.09e-11 / max|dg| 5.4e-7、5.0e-7；r2scan3c
  dE 1.4e-12 / 1.9e-6。
- 预测判定：P1 MISS（预测 5.3-5.6s；验证成本结构现已实测——
  梯度仅 ~0.53-0.65s/5.9s，参考 SCF 不受梯度侧旋钮影响）；P2/P3
  命中（fp64 t_grad 增益 1.79x/1.11x；verify/auto 1.950x 在预测
  1.9-2.2x 带内）；P4/P5 命中。
- **范围诚实**：测量单位；旋钮全 opt-in；默认不变。(b3) 语义注
  （发现 3）是 LC-86 精度政策诚实性在验证语境的重述，非新决定；
  文档内门为道内一致定义。

## 42. LC-94 补遗 (2026-10-13 评审续): 全推荐栈下中等尺寸整道矩阵（LC-88 后源）—— b3lyp 整道分数越过 1.0（auto 反慢 11%），wb97mv 持有整道胜

测量单位（无代码改动；源 594dfe4，运行时=26f80ca；证据
`tasks/issue-9/2026-10-13-lc94-wholelane-postlc88/`，sha256 锚定）。
LC-85 只测了双旋钮栈（无 IP1_F32）；wb97mv/benzene/tzvp 只在
VV10-旋钮态测过（LC-82）。本单元补全四方法 × {全关, 全栈} 配对
相邻控制（生产 gradient_runner，16384，fp64+auto 单文档，3 新进
程样本）；8/8 文档 VALID、全部道/重复收敛。

| 格 | fp64 样本 | fp64 t_grad | auto 样本 | auto t_grad | auto/fp64 分数 | 道内门 (auto-vs-fp64) |
|---|---|---|---|---|---|---|
| wb97mv/benzene/tzvp | 9.459→**3.245s**(2.91x) | 1.856→0.731s | 9.424→**3.151s**(2.99x) | 1.188→0.534s | 0.996→**0.971** | dE≤1.2e-11 / dg 8.9e-7→5.2e-7 |
| b3lyp/benzene/tzvp | 1.309→1.146s | 0.571→0.344s(1.66x) | 1.312→1.270s | 0.336→0.300s | 1.003→**1.109** | dg 1.1e-6→5.7e-7 |
| m062x/benzene/tzvp | 1.936→1.718s | 0.696→0.404s(1.72x) | 1.431→1.480s | 0.270→0.291s | 0.739→**0.862** | dg 9.3e-7→5.3e-7 |
| r2scan3c/vitaminC（控制） | 4.645→4.312s | 0.940→0.707s(**1.33x**) | 3.258→3.108s | 0.368→0.359s | 0.701→0.721 | dg 9.9e-6→**1.7e-6** |

跨态 fp64（栈开 vs 关）：wb97mv dE 1.2e-12/dg 8.9e-7；b3lyp
1.1e-13/9.3e-7；m062x 5.7e-14/7.9e-7；r2scan3c dE 0.0/dg 8.7e-6。

**发现**：
1. **wb97mv/benzene/tzvp 在全栈下保持整道 auto 胜（0.971；两道各
   ~2.9-3.0x 栈效应）**——首次在完整栈下测得；LC-82 的 0.96
   （VV10-旋钮态）复现。NLC 重的 fp32 优势在该格经受住 fp64 道
   ip1 增益。
2. **b3lyp 整道分数越过 1.0（1.003→1.109）**：该格 auto 道比旋钮开
   fp64 道慢 11%。机制（LC-91 组成）：b3lyp 的 auto 优势从来只在
   梯度相（杂化稳态 xc≈fp64、auto 整 SCF 略慢）；fp64 梯度压到
   0.344s 后剩余 ~0.04s fp32 梯度边不再覆盖 auto 的 SCF 税
   （+0.12s）。**全栈下混合精度在中小 AO 的整道价值集中于
   NLC/MGGA 格与大 AO SCF（LC-41/LC-78 族），不在普通 GGA 杂化**。
   （m062x 同向收窄至 0.862 但 auto 仍胜——其 MGGA SCF 优势更大。）
3. **r2scan3c J-only 控制组隔离 IP1_F32 语义**（P3 命中）：道内门
   9.9e-6→1.7e-6（内核共享，LC-86/LC-93 族）而 fp64 t_grad 1.33x
   （LC-86 J-only 1.29x 同轨道在文档级复现）；跨态 fp64 dg 8.7e-6
   = J-only 路径 (b3) 族内核交换代价（LC-86 同轨道测 8.9e-6）——
   **旋钮开 fp64 道非严格参考，配对关态文档才是**（LC-86/LC-93
   精度政策诚实性延续）。
4. 杂化 t_grad 栈效应 1.66-1.72x 超过 LC-85 双旋钮的 1.56x，恰为
   IP1_F32 增量（LC-86 的 1.16x 叠加；文档级含新 SCF 方差）——旋钮
   在 fp64 道按各级实测复合。

预测判定：P1 命中；P2 拆分（fp64 t_grad 增益/分数方向命中；b3lyp
1.109 的量级被"略高于 1.0"低估——SCF 税主导）；P3 命中；P4 命中；
P5 拆分（杂化跨态噪声类如预测；r2scan3c dE 逐位 0 但 dg 8.7e-6 =
预登记的内核交换类）。

**范围诚实**：测量单位；旋钮全 opt-in；默认不变。b3lyp 分数>1 是
待决 (b1)/(b2)/(b3) 语境的证据而非默认改动——旋钮关时 auto 仍与
关态 fp64 平价（1.312 vs 1.309s）；翻转是比较对象为**旋钮开**
fp64 道的属性。至本单元，评审续轮的测量面在当前源补全：梯度组成
（LC-90）、SCF 组成（LC-91）、周期税诊断（LC-92）、验证成本
（LC-93）、整道矩阵（LC-94）、LC-89 负结果关闭。

## 43. LC-95 补遗 (2026-10-13 评审续): SCF 侧 xc 列车**内部**分解 —— 有界第一步问题的实测答案是否定的（AO 复用关闭），MGGA vmat GEMM 列车是唯一剩余命名碎片

LC-91 把稳态 xc 列车测为 SCF 侧最大件（tzvp 96.0ms/cyc fp64）但整
桶未分解；检查点把内核级 `_eval_rho2`/tau GEMM 方向列为选项 (i)，
前置问题是"是否存在有界第一步"。本单元给出实测答案（测量单位；
运行时源不变 26f80ca）。

**结构事实（源码级）**：`_nr_rks_task`/`_nr_uks_task` 每次 SCF 调
用运行 `block_loop` **两次**（rho 通道 + vmat 通道），两次各自从头
重算 AO（non0ao_idx 稀疏缓存跨通道复用；仅 eval_ao 冗余）。

**探针**：共享流 CUDA 事件计时的内部桶（无逐块主机同步，默认块
launch-bound 臂不失真）：p1_loop/ao_p1/rho_mo（mo 路径收缩）/
libxc/p2_loop/ao_p2，vmat_rest=p2_loop−ao_p2 导出；宿主胶水=墙
−三主桶（诚实单独报告）。LC-91 收样协议逐字（新鲜进程×3、双臂、
双道）；4 格×2 臂×2 道×3 = **48/48 收敛**（最差样本内能量散布
2.8e-13；LC-91 的 UKS default/auto 19 周期税逐字复现 19/19/19）。

**验证**：V1 命中——recommended/fp64 稳态每周期墙 vs LC-91 xc
ms/cyc 四格比值 1.02–1.16（预登记 ±20% 带内）；V2 部分诚实——
覆盖中位 0.74–0.95，推荐臂部分低于 0.85 下限，缺口为宿主胶水
（nelec/excsum `.get()` 同步、缓冲准备），16384 下设备时间缩小使
其**占比**增大；事件桶仍是有效设备侧时间，胶水 7.8–14.1ms/cyc
单列为命名碎片。

推荐栈 fp64 稳态每周期组成（中位，ms）：

| 格 | 墙 | p1(rho 通道)[ao/rho] | libxc | p2(vmat 通道)[ao/rest] | ao 占比 |
|---|---|---|---|---|---|
| b3lyp/benzene/svp | 32.0 | 14.9 [2.9/9.6] | 2.0 | 7.4 [1.1/6.3] | 12.6% |
| m062x/benzene/svp | 60.8 | 21.7 [2.7/15.8] | 5.5 | 22.6 [1.1/21.4] | 6.5% |
| wb97mv/methoxy/svp UKS | 57.2 | 17.8 [1.0/14.2] | 5.6 | 19.7 [1.0/18.7] | 3.5% |
| wb97mv/benzene/tzvp | 103.0 | 23.2 [3.2/16.4] | 1.6 | 69.0 [1.8/67.2] | 4.8% |

**发现**：

1. **AO 复用候选关闭（P1 决定性 MISS，决策规则不实施）**：ao 占比
   3.5–12.6%，冗余 ao_p2 每周期 1.0–1.8ms，整 SCF 投影最大
   0.022s（tzvp auto）——低于 0.05s 杠杆门槛 2.3 倍。**内部是收缩
   约束而非 AO 约束**。
2. **按 xctype 的通道排序与 GEMM 计数一致（P2 命中）**：GGA b3lyp
   rho>vmat（14.9>7.4）；三 MGGA 格 vmat>rho（tzvp 3:1——MGGA
   vmat 通道含 `_tau_dot` 3 GEMM + 3 缩放 + GGA 收缩）。
3. **libxc 处处 ≤9.8%（P3 命中）**：单次融合全网格调用，
   1.2–5.6ms/cyc。
4. **LC-11 块尺寸杠杆原是 rho 通道杠杆（P4 拆分）**：
   default→recommended 内部缩 1.47x，分解为 rho 通道摊销 3.05x
   （70.8→23.2）而 **vmat_rest 对块尺寸近不变（70.0→67.2，
   1.04x）**——每块 vmat GEMM 列车为固定总 FLOPs 的 launch/计算
   约束，大块不摊销它。ao 占比随块尺寸**下降**（7.5→4.8%，方向
   MISS 诚实）。
5. **fp32 头 rho 收缩不快于 fp64 尾（P5 方向被驳倒）**：
   (p1_loop−ao_p1) 头/尾比 ≥1.0 于 12 个样本对中的 11 个
   （1.00–1.96）——逐块 AO cast + 失去预置缓冲抵消/超过 fp32 GEMM
   增益（LC-89 机制在 SCF 侧实测复现）；头整调用仍更便宜（tzvp
   头 54.9 vs 尾 89.5ms），**auto 头的增益定位在 vmat 通道**
   （fp32 模式下 vtmp/scale_buf 被丢弃而非重设尺寸，无缓冲问
   题）。
6. **vmat_rest 为剩余命名碎片**：tzvp 67.2ms/cyc = 0.61s/SCF(fp64)、
   0.46s(auto)；其余 MGGA 格 18.7–21.4ms/cyc。与梯度侧 LC-87/LC-90
   命名的 tau GEMM 列车同族——内核级深工程，小可评审类之外，
   **保持命名未实施**。
7. **诚实未诊断观察**：rho_mo 桶在运行内双峰（tzvp 9–19ms 模式；
   一个 fp64 样本全程 5.9–6.6ms）而 p2_loop 逐调用稳定且整调用墙
   与 LC-91 互证 7% 内——所有结论基于稳定桶或外部验证墙；
   rho 收缩拆分带 2–3x 模式不确定度（疑调度/性能态方差；未追
   诊，需专用内核踪迹单元）。

预测判定：P1 MISS（决策规则：不实施）；P2 命中；P3 命中；P4 拆分
（内部 1.47x 命中/ao 占比方向 MISS + 新分解 rho 3.05x vs vmat
1.04x）；P5 方向被驳倒（ao 相等半命中）。

**范围诚实**：测量单位；运行时源不变（26f80ca）；默认不变；旋钮
全 opt-in。2.3s python 导入冷启动在道文档计时块之外（LC-14 记录
维持）。nlc 桶整记（内部 LC-80 已闭）。UKS methoxy 每周期桶跨道
可比、整运行和不可比（19 周期税）。

**下一步**：检查点选项 (i) 以实测 NO 关闭——在测范围内 xc 列车内
不存在有界第一步；候选转向选项 (ii) 未分解小面（hcore_e1 0.042s、
grad_nuc）或评审反馈项。证据
tasks/issue-9/2026-10-13-lc95-scf-xc-train-decomp 8 文件
sha256 锚定；effects-index 19 数据集。

## 44. LC-96 补遗 (2026-10-13 评审续): 收缩引擎整道成本/收益实测 —— cutensor 默认维持（无稳态换益），新鲜进程句柄等待为预热类；plan-cache 杠杆证伪

LC-95 首调用桶暴露了一个未记录的 ~1.6–1.8s 一次性成本（每次新鲜
进程运行，落在第一个 cutensor `contract()` 消费者：MGGA 格首个 xc
调用 / GGA 杂化首个 DF jk）。本单元先做证伪链再上生产矩阵（测量
单位；运行时源不变 26f80ca；`CONTRACT_ENGINE` 为既有 opt-in 上游
旋钮）。

**证伪链（先于矩阵，全部保留在证据目录）**：
`cuTensorHandleCreate` = 2.85s 冷启动/进程；LC-13 预热线程仅隐藏
~1.1–1.3s（在 1.4–1.5s 导入链后面）；残余 ~1.5–1.8s 阻塞首个 SCF
周期。句柄就绪后首个**平凡** contract 仅 0.8ms（内核加载非成
本）；首个真实形状 41.5ms；稳态 0.674ms；真实 `_tau_dot` GEMM 形
状 cuBLAS 平价（0.605ms）。**plan-cache 杠杆证伪**：逐调用
descriptor+operator+plan 设置 = 0.017ms API + 总计 ~0.1ms/调用
（0.674 全程 vs 0.579 预规划）≈ vmat 桶的 5%——cutensor.py 中
未使用的 `_tensor_descriptors` 缓存是无害死代码，非杠杆。

**生产矩阵**：8 份 rtx4090-gradient-v1 文档（4 格 × 2 引擎，生产
gradient_runner，fp64+auto 道，每道 3 新鲜进程样本，双引擎同挂
推荐栈），8/8 schema-VALID；methoxy UKS 格带着已知的 LC-29/LC-39
诚实失败族行（两引擎均 1 pass + 1 fail——正确地未被翻转）。

**计时样本中位（生产道证据）**：

| 格 | 道 | t_scf ct→cupy | t_grad ct→cupy |
|---|---|---|---|
| b3lyp/svp | fp64 / auto | 0.493→0.481 / 0.542→0.663 | 0.210→0.214 / 0.225→0.225 |
| m062x/svp | fp64 / auto | 0.715→0.726 / 0.947→1.113 | 0.234→0.283 / 0.265→0.250 |
| wb97mv/methoxy UKS | fp64 / auto | 1.156→1.580 / 1.323→1.740 | 0.301→0.359 / 0.309→0.387 |
| wb97mv/tzvp | fp64 / auto | 2.525→2.313 / 2.517→2.456 | 0.727→0.735 / 0.550→0.571 |

**发现**：

1. **引擎问题以混合/负证据关闭（P1/P2 MISS）**：无一致稳态收益
   ——b3lyp/tzvp 平价；cupy 在 svp MGGA/UKS 格**慢 12–40%**；仅
   tzvp SCF 快 ~9%。
2. **~1.5–2.0s 新鲜进程句柄等待是预热类**：生产协议把它吸进未计
   时的 warmup 运行（cutensor fp64 道预热 +1.66–2.00s），计时中
   位不受影响——对单次单进程用户运行真实（预备探针：tzvp fp64
   scf 10.08→8.60s、b3lyp 2.58→1.03s），对基准道文档按设计不可
   见。
3. **无门翻转、无假通过**：诚实失败族在两引擎下均为 fail；干净
   格数值保持归约序类（fp64 跨引擎 dE ≤5.7e-14、max|dg|
   ≤8.0e-8；干净格 auto dE ≤4.9e-12）；methoxy auto 跨引擎
   dE 4.3e-7 = 支分裂族签名（LC-77 轨迹敏感类，非收缩噪声）。
4. **auto 道轨迹对引擎敏感**（b3lyp/m062x +1 周期；methoxy 落点
   移动）——引擎切换会扰动已记录的 auto 道证据基，默认维持的又
   一理由。
5. **决策规则：不提出引擎默认变更**；CONTRACT_ENGINE 保持
   opt-in 未设；未来若缓解句柄等待，属于 cutensor 路径自身（更
   早/惰性 warm join），不是引擎切换。

预测判定：P1 MISS（决定性——预热吸收）；P2 MISS（3/8 行出带，
cupy 慢）；P3 拆分（fp64 命中 ≤5.7e-14 / 干净格 auto 命中 /
methoxy auto MISS=族签名）；P4 MISS 经由既有族（非引擎回归；
6/8 文档全过；无引擎致翻转）；P5 数量级 MISS（3.5–8.0e-8 vs 预测
≤1e-8，类对量错，诚实记录）。

**范围诚实**：测量单位；运行时源不变；默认不变；推荐栈下的比较
（默认栈效应由预备探针确立，同构）。methoxy 失败行为不可变族，
未重标注。t_import 在道文档计时块之外（LC-14 维持）。

至本单元，检查点的新杠杆扫描双方向（xc 列车内部、引擎维度）均以
实测关闭；剩余下一单元候选：评审反馈项，或（若被指示）cutensor
路径句柄等待缓解的源改动类评估。证据
tasks/issue-9/2026-10-13-lc96-contraction-engine 17 个 sha256 锚
定文件 + 8 文档；effects-index 20 数据集。

## 45. LC-97 补遗 (2026-10-13 评审续): cuTensor 句柄早 warm —— 实测后按预登记决策规则回退的诚实负结果（杠杆关至地板）

**动机**：LC-96 §44 将新鲜进程 cuTensor 句柄等待（冷 2.85s，残余
~1.5–2.0s 阻塞单发用户运行的第一个 SCF 收缩）定性为预热类，并把
"cutensor 路径句柄等待缓解（更早 warm）"列为检查点候选（b）。本单
元先测后改：预登记落地门槛（最小格单发增益 ≥0.20s + 数值/道文档/
teardown 全过），未达标即回退。

**实现（97 行，已保留于证据目录、按规则回退）**：新私有模块
`gpu4pyscf/_cutensor_warm.py`（引擎/环境门 + 共享句柄锁 + 幂等守护
线程启动），`gpu4pyscf/__init__.py` 顶部（`_patch_pyscf` 之前）启
动，`lib/cutensor.py` 复用同一锁与幂等启动（LC-60 atexit-join
teardown 竞态修复与注册顺序原样保留）。

**机制发现（importtime，本单元首次测得）**：`lib/cutensor.py` 实际
在 `lib.diis` 子树内被导入 —— `lib/__init__.py` 的**第一个**导入即
拉入它 —— 故基线 warm 线程已在包导入 ~0.28s 处启动，而非线性阅读
`lib/__init__.py` 所估计的 ~0.45s；可回收偏移实测 ~0.16s。绝对地板
由 cupy 本身设定（句柄无法先于 cupy 导入创建）。

**实测（清洁基线 vs 打补丁，同会话相邻，每样本独立新进程，各 3
样本）**：

| 量 | 基线 | 补丁后 | Δ |
|---|---|---|---|
| wait 模式导入后残余句柄等待 (s) | 2.57–2.62 | 2.46–2.48 | −0.11…−0.16 |
| b3lyp/svp 单发 t_scf (s) | 2.823 | 2.653 | **−0.170** |
| wb97mv/tzvp 单发 t_scf (s) | 10.153 | 9.978 | −0.176 |
| methoxy UKS 单发 t_scf (s) | 5.794 | 5.600 | −0.194 |
| b3lyp 首个 get_jk (s) | 1.712 | 1.552 | −0.160 |
| 包导入 (s) | 0.644–0.687 | 0.539–0.567 | −0.09…−0.13 |
| 道文档 fp64/auto 中位 (s) | 2.989/2.999 | 3.115/2.882 | ±3.9% |

数值与安全：三格 fp64 跨状态能量最大差 1.7–2.3e-13 Ha（跨进程地
板）；成对生产道文档（gradient_runner、推荐四旋钮栈、3 新进程样
本）双双 VALID、门全过、中位 ±10% 内，in-doc 门 −1.080e-11/
−1.069e-11 与 dg 5.41e-7/5.73e-7 = LC-93/94 族复现；teardown 压
力 0/60（补丁）/ 0/30（回退后），import-and-exit 墙钟 4.25s vs
4.50s 无回归；`GPU4PYSCF_CUTENSOR_WARM=0` 与 `CONTRACT_ENGINE=cupy`
下均不起线程（子进程探针验证）。

**决策**：预登记规则判定**回退** —— 最小格增益 0.16–0.17s <
0.20s 门槛，而改动并非零成本（线程所有权移出引擎选择模块），对
生产基准协议零影响（预热两侧都吸收等待）。工作树恢复至 c932760
字节干净；实现以 `implementation.diff` + `_cutensor_warm.py.reverted`
保全于证据目录，供用户按决策重新落地（若单发用例重要）。

**杠杆关闭至地板**：残余 ~2.5s 等待即 `cuTensorHandleCreate` 本身
（cupy 版本锁定的主机侧成本），早于 cupy 导入不可达；进一步收益属
NVIDIA/cupy 上游而非本仓库。生产协议已由预热吸收全额 2.85s
（LC-96），单发暴露现界定为 ~2.5s 且位置已表征（杂化首个 get_jk、
MGGA 首个 xc 收缩）。

预测判定：P1 HIT（LC-96 残余族复现）；P2 量级 MISS（0.16–0.19s vs
0.20s 门槛；方向与机制如预测，偏移本身小于估计 —— importtime 嵌
套为此前未知）；P3/P4/P5 HIT；P6 已测子项全 HIT（单测子项因回退
未运行，如实记录）。

**范围诚实**：测量+条件实现单位；运行时源最终不变（c932760，运行
时 26f80ca）；默认不变；所有旋钮 opt-in。证据
tasks/issue-9/2026-10-13-lc97-cutensor-early-warm 93 个 sha256 锚
定文件；effects-index 21 数据集（LC-96 版已归档）。

## 46. LC-98 补遗 (2026-10-13 评审续): vmat_rest 内部分解 + stacked-tau A/B —— LC-95 命名碎片首次逐件测量，无隐藏开销，候选关闭

**动机**：LC-95 §43 将 `vmat_rest`（tzvp fp64 67.2ms/cyc，唯一剩余
命名碎片）归因于 "MGGA `_tau_dot` + GGA 收缩 GEMM 列车"——仅凭
GEMM 计数推断，无内部测量。存在未验证的算术张力：4 GEMM/块 ×
~13 块 × LC-96 稳态 0.674ms/调用 ≈ 35ms ≠ 67.2ms。要么真实掩码形
状下每调用慢数倍，要么非 GEMM 碎片占大头。本单元测之。

**探针（v2；v1 的仪器 bug 作为证据的一部分记录）**：对
`numint._tau_dot/_scale_ao/contract/_as/add_sparse` 忠实 monkeypatch
包装，**每调用独立 CUDA Event 对**（v1 跨嵌套调用复用同一对事件，
父层计时被子层重复 record 破坏——smoke run 中 total<interior 不可能
值当场捕获；v2 另在收敛 dm 上的**稳态** `nr_rks` 调用前重置桶，使
中位数描述单次稳态调用）。2 格 × 2 道 × 3 新进程样本 + A/B 模式
（同输入 stacked-tau，每道 2 样本），推荐四旋钮栈，24/24 收敛，
能量复现 LC-91/94 族位数；对照（未仪表化）调用确认仪器成本近零
（tzvp 98.4/102.2ms）。

**实测组成（稳态调用，fp64）**：

| 碎片 | tzvp (nao_sub 216) | m062x/svp (nao_sub 112) |
|---|---|---|
| tau 3-GEMM 列车 | 57.2 ms (9×6.48) | 26.3 ms |
| GGA GEMM | 17.2 ms (9×1.94) | 6.8 ms |
| _scale_ao | 2.2 ms | 1.3 ms |
| add_sparse / _as | 0.4 / 0.1 ms | 0.5 / 0.1 ms |
| **每 GEMM 速率** | **707 GFLOPS = fp64 峰值 55%**（GGA 61%） | 447 GFLOPS = 35%（GGA 44%） |

**发现**：
1. **算术张力解决：无隐藏开销**。tau 列车占仪表化 p2 碎片的 74%
   （≈LC-95 vmat_rest 的 85%），GGA ~22%，其余全部亚毫秒。LC-96
   的 0.674ms 是小得多的微基准形状；真实 216-AO 掩码形状每 GEMM
   2.15ms @ **RTX 4090 fp64 峰值（1.29 TFLOPS）的 55%** —— 列车
   为计算速率受限，即此类高瘦 fp64 形状的实际天花板类。
2. **stacked-tau 重分块实测 2.03× 更慢**（13.12 vs 6.45 ms/调用，
   双道一致；校验和精确为 0）：单个 (3n,ng)×(ng,3n) GEMM 需算出全
   部 9 个 (x1,x2) 块而只取 3 个对角块——3× FLOP 惩罚主导，大
   GEMM 效率无救援。视野内唯一 GEMM 重分块候选决定性关闭。
3. **auto 道稳态调用跑 fp64 列车**（两道同 6.45ms/调用）：更便宜
   的 fp32 vmat 是 SCF 头部专属效应（LC-95 发现 5）；精度切换后
   每次 vmat 调用付 fp64 速率。
4. **整调用交叉验证诚实拆分**：tzvp 复现 LC-95 在带内（0.96–
   0.99×）；m062x 对照 0.75×（更快——排除仪器开销成因；暖缓存/
   性能态类；mo_coeff 路径假设直接测得 96.4 vs 93.1ms 被拒）。

预测判定：P1 带上 MISS 方向对（tau 份额 74–85% 高于预测 40–70%
带——比预测更 GEMM 主导）；P2 量级 MISS（6.48ms/调用 = LC-96 微值
的 9.6× vs 预测 2–4×；类对带错，机制为形状差 + 实测 55% 峰值速
率）；P3 HIT（更慢分支 + 校验和精确）；P4 拆分；P5 HIT。

**决策**：预登记规则判定**不实施**（tau 份额过 ≥50% 门但 A/B 增益
条件决定性失败）。至此 vmat 列车表面测至组成地板：GEMM 计算速率受
限于峰值 55–61%、无 launch/plan/overhead 余量（LC-96 plan-cache 已
证伪；本单元 scale/add_sparse/_as 亚毫秒）、重分块关闭、引擎维度
关闭（LC-96）、fp32 政策已用于 auto 头（LC-95）。剩余即 LC-95 所
命名——深度内核工程（融合收缩内核）或硬件 fp64 速率，均出小可评审
类。SCF 侧杠杆台账就此完备：每个 ≥0.05s 碎片均已测量、归因，并分
别为已优化、政策门控或命名深度工程。

**范围诚实**：测量单位；运行时源不变（26f80ca）；默认不变；所有旋
钮 opt-in。证据 tasks/issue-9/2026-10-13-lc98-vmat-rest-decomp 37
个 sha256 锚定文件；effects-index 22 数据集。

## 47. LC-99 补遗 (2026-10-13 评审续): 安全路线代价面补全 —— auto-strict 全栈中等 AO 成本与首个全栈分歧格（交叉链接）

本节交叉链接 issue-7 §9（LC-99 单元的完整记录在
`tasks/issue-7/2026-10-13-lc99-strict-route-cost-fullstack/`，
18 文件 sha256 锚定）：LC-93（§41）只测了 auto-verify 在完整推荐
栈的整道代价；本单元以 LC-93 协议逐字复刻 + 同一文档内 auto-strict
道补全该面，并首次在完整推荐栈测量任一安全路线在正典 UKS 分歧格
（wb97mv/methoxy/svp）上的行为。与本文件性能台账相关的要点：

- **不触发格（wb97mv/benzene/tzvp + r2scan3c/vitaminC/mtzvpp）**：
  strict/verify 端到端比 1.048 / 1.002——两条安全路线同为
  **参照 SCF 受限**（LC-93 结论扩展到 strict 道），LC-93 中位数
  0.98-1.06x 第三会话复现，能量 dE ≤ 1.1e-13。
- **分歧格**：auto 道 11 圈收敛到 LC-92 第三点（+4.249e-7），梯度
  诚实 FAIL 3.0e-2；两条安全路线均采纳 fp64 道解，梯度恢复至
  2.3e-7；strict 比 verify 便宜 12%（3.543 vs 3.974s，无精化
  SCF）。verify 的精化 1 圈收敛回 auto 第三点（dm 差 6.7e-7）
  = 真定态二盆分裂（occ-SVD beta 0.497，LC-29 签名）。
- **对杠杆台账的意义**：安全路线的代价不是可优化碎片而是参照
  SCF 本身（fp64 全精度轨迹，语义上不可 warm-start——认证要求
  同初始猜测轨迹），其唯一的已测杠杆仍是 LC-80 VV10 内核
  （LC-84 3.18x）。本单元不改变任何旋钮/默认；3/3 文档 VALID，
  36/36 采样收敛，auto 诚实 fail 行原样保留；effects-index 23
  数据集。

## 48. LC-100 补遗 (2026-10-12 评审续交叉链接): 缺省栈路线×栈分歧表补全 —— 首份同文档 verify/strict 分歧行（issue-7 §10）

交叉链接（完整记录 `tasks/issue-7/2026-10-13-lc100-defaultstack-route-divergence/`，
14 文件 sha256 锚定）：缺省栈 4 道同文档在正典分裂格
（wb97mv/methoxy/svp UKS）首次于当前源实测**路线分歧**——
auto-verify 的较低定态规则保留低支线（诚实 FAIL：dE -1.283e-6 /
dg 3.69e-2），auto-strict 无条件采纳 fp64 道参照（PASS：dE 0.0
逐位 / dg 1.94e-6）；位移格（m062x/methoxy/tzvp UKS）两路线一致
（均采纳参照，dg 1.09/1.10e-6）；strict 判定跨源复现 LC-77
（dE 0.0 / 8.5e-14）。性能侧要点：strict 在任何测量格上都不慢
（两格均比 verify 省 12% = 无精化 SCF；LC-99 平价至省 12%），
且两安全路线在缺省栈分裂格的代价都被 auto 头的 19 圈 DIIS 停滞
（LC-92 族）放大（t_scf 11.2-12.8s）——推荐栈移除该停滞的实测
（LC-91/92）因此也是安全路线的整道成本杠杆。测量单位；运行时源
不变（26f80ca）；默认不变；effects-index 24 数据集。

## 49. LC-101 (2026-10-13 评审续): fp64 道梯度格点收缩 fp32 旋钮（GPU4PYSCF_GRAD_GRID_F32）—— LC-87 命名的最后一个开放杠杆落地（opt-in），实测 1.39-2.20x

LC-87 finding 1 命名了 fp64 梯度道上最后一个 ≥0.05s 的开放杠杆
（`tasks/issue-9/2026-10-13-lc87-xcgrid-grad-decomp/`）：fp64 道的
XC 格点收缩全 fp64 运行——`_tau_grad_dot_` 0.133s + `_gga_grad_sum_`
0.090s（wb97mv/benzene/tzvp@16384 全栈，t_grad 0.690s）加
get_nlc_exc 的 deriv=2 收缩 0.089s（LC-88 只在 fp32 模式施放 cast，
fp64 道照付）——而 auto 道以完全相同的形状/dtype fp32 运行这些
GEMM（0.018/0.019/0.012s）。LC-89 关闭的只是 UKS-auto 变体
（monkeypatch 4 配置全更慢）；RKS-fp64-lane 变体此前从未测过。
本单元以 LC-86 同-orbital A/B 协议测度先行、按预登记决策规则落地
opt-in 旋钮。

**候选与实现**（`gpu4pyscf/grad/rks.py`，默认关）：
`GPU4PYSCF_GRAD_GRID_F32=1` 在 fp64 解析梯度模式下把逐块 AO cast 为
float32 **仅用于收缩调用**——rho 保留 fp64 AO 与预分配缓冲（密度逐位
不变），`_gga_grad_sum_`/`_tau_grad_dot_`/`_d1_dot_` 以操作数 dtype
运行（即 auto 道既有代码路径），dm_mask einsum 与 exc1 累加器保持
float64；get_nlc_exc 的 LC-88 cast 条件扩为 `fp32_grid or 旋钮` 且
**仅限 RKS 路径（nset==1）**——UKS `_get_exc_task` 面不带旋钮
（LC-89 负结果维持）。fp32/auto 道结构性无操作。解析器与
GPU4PYSCF_DF_IP1_F32 同款（仅接受 '1'，垃圾值告警回退）。

**测度**（`tasks/issue-9/2026-10-13-lc101-grad-grid-f32/`，同-orbital，
推荐栈 env，3 样本/侧，全新进程）：

| 格 | t_grad off→on | 加速 | max\|dg\| 交换代价 |
|---|---|---|---|
| wb97mv/benzene/tzvp@16384 | 0.692→0.443s | **1.563x** | 4.53e-7 |
| b3lyp/benzene/tzvp@16384 | 0.261→0.325s | **0.805x（更慢）** | 4.08e-7 |
| m062x/benzene/tzvp@16384 | 0.427→0.299s | **1.426x** | 1.26e-7 |
| r2scan3c/vitaminC/mtzvpp@16384 | 0.738→0.299s | **2.469x** | 2.33e-6 |
| wb97mv@缺省分块 | 0.763→0.659s | **1.158x** | 4.41e-7 |

P1 HIT（1.563x ≥ 1.15x），P3 HIT 强（298 AO 格 2.47x——收益随
nao 增长，LC-98 的 fp64-GEMM 速率类），P4 HIT（全部 ≤2.3e-6 ≤
1e-5），**P2 诚实 MISS**（b3lyp 0.805x：纯 GGA 格只有 ~0.09s 的
gga 收缩可赢，逐块 AO cast 的访存开销反而净亏——LC-85/91 的
b3lyp 平价发现从另一方向重现，旋钮须按工作负载 opt-in），P5 方向
MISS（缺省分块仍有 1.158x）。实现复现探针（1.555x vs 1.563x，
max|dg| 4.533e-7 逐位一致，实现-ON 与探针-ON 梯度差 5.8e-8 =
fp32 类跨进程地板）。

**生产验证**（gradient_runner 全新进程文档，推荐栈，fp64+auto 两道
× 3 样本，4/4 VALID）：wb97mv/benzene/tzvp fp64 道文档级
0.705→0.508s = 1.39x；r2scan3c/vitaminC fp64 道 0.774→0.352s =
**2.20x**（两个独立 ON 态文档跨进程一致 0.352/0.322）；auto 道在
样本散布内无操作（旋钮结构性只作用于 fp64 道，dtype shim 测试证明）；
e_tot 两态逐位同（仅梯度旋钮）；所有门 PASS。副产品（LC-86 家族在
格点收缩面的重现）：r2scan3c 文档内 auto-vs-fp64 dg 1.724e-6 →
1.178e-6（两道共享 fp32 收缩——auto 误差的一部分本就是收缩交换类）。
(b3) 类精度政策语义：旋钮开启的 fp64 道不再是逐位严格 fp64 参考
（严格门裁定用关态 fp64 记录）。

**一次诚实记录**：vitaminC 首个 OFF 对照文档因验证脚本在 wb97mv ON
运行后未 unset 环境变量而实际运行在 ON 态（t_grad 0.322s ≈ ON 态
而非 LC-94 的 0.707s fp64 记录即其证据）——错标文档以
`*_mislabeled_off_was_on.*` 保留为证据，真 OFF 对照以干净 env 重跑
（raw 75f066a8d69e）。

测试：`tools/tests/test_grad_grid_f32.py` 12 项（解析器契约/源契约
含 UKS 面无旋钮与 nset==1 保护/dtype shim/同-orbital 界 5e-6/
关态 fp64 道地板）；tools 全套 504 OK（492+12）；上游
test_rks_grad 15 通过、test_uks_grad 12 通过（改动面）。默认不变；
全部旋钮 opt-in；测量+实现单位。effects-index 25 数据集。

## 50. LC-103 (2026-10-13 评审续): 四方法整道矩阵在完整 4 旋钮栈补全 —— wb97mv 整道分界穿越至平价（(b4) 决策面完成）

LC-101 的生产文档只在 wb97mv 与 r2scan3c/vitaminC 两格验证了第 4
旋钮（GRAD_GRID_F32），m062x 与 b3lyp 两格缺文档级 4 旋钮数字——
LC-94 的整道矩阵是 3 旋钮态。本单元以 LC-94 run_matrix 纪律逐字补齐
（`tasks/issue-9/2026-10-13-lc103-4knob-wholelane-matrix/`）：生产
gradient_runner，fp64+auto 两道×3 全新进程样本，S2，16384 两态同设；
配对相邻 3 旋钮对照（GRAD_GRID_F32 off = LC-94 栈在现源 d2135b6）。
env 防泄漏纪律按 LC-101 教训执行（每态显式 `env` 前缀 + 日志回显，
本次零泄漏）。4/4 文档 VALID，24/24 样本收敛，两态全门 PASS。

**新文档测度**（中位数；3knob→4knob 配对）：

| 格 | 道 | t_grad | e2e | 文档内门（4knob） |
|---|---|---|---|---|
| m062x/benzene/tzvp | fp64 | 0.484→**0.301s (1.611x)** | 1.824→1.622s | 锚行 |
| m062x/benzene/tzvp | auto | 0.290→0.246（噪声内） | 1.556→1.366s | pass dE −8.36e-12 / dg 5.67e-7 |
| b3lyp/benzene/tzvp | fp64 | 0.298→0.307（**0.969x 平价**） | 1.036→1.081s | 锚行 |
| b3lyp/benzene/tzvp | auto | 0.272→0.296（噪声内） | 1.123→1.186s | pass dE 9.66e-13 / dg 2.54e-7 |

**完成的四方法整道矩阵**（e2e sample_s，LC-94 口径；LC-101 off/on
文档供 wb97mv/r2scan3c 行，全部同源等价 d2135b6）：

| 格 | LC-94 3knob frac | **4knob frac** | fp64 e2e 3→4 | 判读 |
|---|---|---|---|---|
| wb97mv/benzene/tzvp (222 AO) | 0.971 | **1.035** | 3.090→2.921s | auto 整道优势被抹平——平价 |
| b3lyp/benzene/tzvp (222 AO) | 1.109 | **1.097** | 1.036→1.081s | auto 慢 10%，维持 |
| m062x/benzene/tzvp (222 AO) | 0.862 | **0.842** | 1.824→1.622s | auto 赢 16%，维持 |
| r2scan3c/vitaminC/mtzvpp (298 AO) | 0.721 | **0.788** | 4.562→4.175s | auto 赢 21%，SCF 主导 |

**发现**：(1) **头条——wb97mv 整道分界穿越**：完整 4 旋钮栈上，
auto 道在规范 NLC 格（222 AO）的整道优势（3 旋钮态 0.971，LC-94）
被抹平为 1.035；fp64 道真实变快（e2e 3.090→2.921，t_grad 0.705→
0.508 = LC-101 的 1.39x），auto 道结构性不变（switch 记录逐字同、
e_tot 1e-13 地板内、跨态 auto-auto max|dg| 4.7e-8–1.4e-7 = 同文档
跨进程地板 4.9e-8–1.2e-7——梯度级证明旋钮不触及 auto 道）。正确
判读是 ±10% auto-t_scf 带内的**平价**而非反转。对 (b4) 的决策
含义：四旋钮全开时 fp64 道在 222 AO NLC 格上是 auto 道的整道对手
——knob-on 严格-fp64 道成为该尺寸下混合精度的实用替代，逐位严格
证据仍需关态记录（(b3)/(b4) 语义）。(2) **auto 道在完整栈仍赢处**：
298 AO（0.788，SCF 相主导且 auto SCF 快 1.35x）与 m062x（0.842）；
混合精度的整道价值集中于大-AO SCF + tau 重 MGGA；纯 GGA b3lyp 维持
auto-负值（1.097，LC-94 穿越带第 4 旋钮仍立）。(3) **(b4) 的 fp64
道 e2e 增量**：1.06x/1.12x/1.09x（wb97mv/m062x/r2scan3c-298AO）/
~1.0x 诚实无操作（b3lyp）——梯度相收益（1.39-2.20x）被 e2e 的 SCF
占比稀释，与预期一致。

**对照漂移诚实记录**：m062x fp64 t_grad 跨会话噪声 ±15%（LC-94
0.404 / 本次重跑 0.436 / 首对照 0.484；LC-94 自己在该格的 auto 散布
0.225-0.363 = ±23%）；b3lyp 对照反向漂 1.153x（0.344→0.298）。
wb97mv/r2scan3c 对照复现 LC-94 于 4%/9.5% 内。配对相邻比（协议的
估计量）不受影响；跨会话 ±10% 对照条款对这些格过紧——记录之，
仅一次有界重跑，不再重复。P1/P2 SPLIT（配对比 HIT/端点命中，对照
条款 MISS）、P3 HIT（三重证明）、P4 SPLIT（"e_tot 逐位不变"条款
**误设**——全新进程样本在态内本就差 1e-13 地板，旋钮不超此地板：
fp64 锚行跨态 e_tot 差 1.1e-13/2.9e-14）、P5 SPLIT（wb97mv
"frac<1 保持" MISS——穿越即本单元头条发现）。

测量-only；运行时源 d2135b6 不变；默认不变；全部旋钮 opt-in；
effects-index 26 数据集。

## 51. LC-105 (2026-10-13 评审续): SCF 侧 vmat 收缩 fp32 旋钮（GPU4PYSCF_SCF_VMAT_F32）—— LC-98 命名的 vmat 列车的最后一个未试杠杆，先测量后落地（opt-in）

**动机**：LC-98 把 vmat 列车测至组成地板（tau 3-GEMM 列车 57.2ms/
稳态调用、计算速率受限于 fp64 峰值 55%），关闭了重分块/引擎/
plan-cache/启动维度，并记录 "更便宜的 fp32 vmat 是 SCF 头部专属
效应"（LC-95 发现 5）——即 numint `_nr_rks_task`/`_nr_uks_task`
的 fp32_grid cast 只在全局精度为 fp32 时生效。但 (b3)/(b4) 家族
模式（LC-86/LC-101：把 auto 道既有的 fp32 收缩机制以 opt-in 旋钮
扩展到 fp64 道）从未施加于 **SCF 侧 vmat 列车**——grep 全部文档/
阶段记录零命中。这是该命名碎片上唯一未尝试的廉价杠杆。

**候选与纪律**：`GPU4PYSCF_SCF_VMAT_F32=1` 在**纯 fp64 SCF 道**
（precision_mode None/'fp64'）上把 vmat 收缩循环的逐块 AO cast 到
fp32——rho/excsum/nelec 保持在循环前的 fp64 AO 上（密度与 excsum
逐位不变，收缩辅助函数按操作数 dtype 运行，vmat 累加器 fp64），
即 auto 头部既有代码路径原样扩展。auto 家族的 fp64 尾部与
stability_verify 的严格参照不触及（scf.hf 道门 + `_strict_fp64_
reference` 标记——无门控反事实验证见下）。梯度/UKS 梯度面不触及。

**先测量（2026-10-13-lc105-vmat-f32-scf-knob，预登记决策规则）**：
- 同-dm 换入代价（4 格）：dexcsum=dnelec=0 **逐位**（rho 路不受
  触碰的证明）；max|dvmat| 1.9e-6；稳态调用 2.40x wb97mv/tzvp、
  1.47x m062x、1.82x b3lyp、1.20x UKS methoxy。
- 整 SCF A/B（全新进程×3）：t_scf 1.122x wb97mv（9→8 周期）、
  1.137x m062x、b3lyp 1.003x 平价、UKS 0.979x 噪声内；收敛 e_tot
  位移 ≤6.8e-11 Ha（距 1e-6 门 4 个量级，远高于 1e-13 跨进程地板）。
- **无门控反事实**（P4）：把 cast 无条件应用到 auto 尾部使 tzvp
  auto e_tot 移动 3.5e-11 —— auto 尾部必须逐位 fp64 的直接证据，
  因此落地采用 scf.hf 道门（这是本旋钮与 (b3)/(b4) 的实现差异：
  那两个旋钮的梯度面天然区分 auto，这里 SCF 面必须显式区分）。
- 决策规则（≥1.05x 整 SCF + 逐位 excsum/nelec + e_tot ≤1e-7）
  **HIT**：落地。

**落地**（numint.py `_scf_vmat_f32()` 解析器 + `_scf_vmat_f32_lane`
道旗；scf/hf.py `_kernel` 道门；stability_verify `_build_fp64_
sibling` 标记 `_strict_fp64_reference`）：+10 回归测试
（tools/tests/test_scf_vmat_f32.py：解析器/源合同/dtype shim/
同-dm 界/整 SCF e_tot 界/auto 尾 fp64 保持）；tools 514 OK
（504+10）；上游 dft rks 16 + uks 10 + rhf/numint 40 全绿
（test_sparse_index 缺失为已记录的既有环境类失败，pristine 树
复验同样失败）。旋钮关态逐字节等价（floor 测试：pre-change
-231.971642476836763 vs 关态 -231.97164247683688 = 1.2e-13 跨进程
地板）。RHF 无 NLC 网格面，无循环导入（kernel 时惰性导入验证）。
auto+verify 的参照 SCF 在旋钮开启时仍逐位 fp64（methoxy UKS 实测：
29 次 nr_uks 调用道旗全 False，ref_e_tot = -114.9016950129849 =
fp64 道记录逐位）。

**生产文档级（修正后矩阵，8/8 VALID，3 全新进程样本/道，配对相邻
关态对照，完整推荐栈 5 旋钮）**：

| 格 | 道 | t_scf off→on | e_tot 位移 | 周期 |
|---|---|---|---|---|
| wb97mv/benzene/tzvp | fp64 | 2.377→**1.780s (1.336x)** | +6.5e-11 | 9→8 |
| m062x/benzene/tzvp | fp64 | 1.285→**0.794s (1.618x)** | +3.2e-11 | 8→8 |
| b3lyp/benzene/tzvp | fp64 | 0.611→0.608（平价） | +6.7e-11 | 7→8 |
| wb97mv/methoxy/svp UKS | fp64 | 1.456→1.507（噪声内） | −3.6e-11 | 11→11 |

auto 道在全部 4 格跨态**逐位不变**（e_tot 差 ≤2.8e-14，max|dg|
≤1.0e-7 = 跨进程地板，周期相同，开关记录逐字同）——旋钮在文档
级被证明不触及 auto 道；t_grad 全格平价（旋钮仅 SCF 面）。文档内
auto-vs-fp64 门全 pass（清洁格）；methoxy UKS auto 诚实 fail 原样
保留（LC-29/92 家族，两态同 fail）。梯度相互不改（fp64 道锚行
e_tot 位移即交换代价；严格裁定用关态记录——(b3)/(b4) 语义，
knob-on fp64 SCF 道非逐位严格参照）。

**协议缺陷诚实记录**（evidence 内 DEFECT_NOTE.md + 两个保存
目录）：矩阵脚本 v1 的清理 sed 误删了 ON 分支整行
`GPU4PYSCF_XC_GRID_BLOCKSIZE=16384 $STACK5 \`，使 v1-v3 的全部
vmatf32ON 文档跑在缺省分块 4096 上（fp64 9 周期轨迹
-232.226270667534、methoxy auto 落 LC-29 下支 19 周期 = LC-92
缺省栈签名）。三步去混杂诊断（auto-only 双态逐位同、fp64-only
ON/OFF/ON/OFF 四文档、配对/双-fp64 bisect）隔离出该缺陷；矩阵 v2
（env 全字面量 + 全 env 回显）后全部文档 VALID。v1 另含一次我方
并发 GPU 检查（单 GPU 任务纪律违规，已隔离保存）；当时的"并发污染
即主因"机制判断被缺省分块缺陷证据推翻——如实记录。缺陷文档作为
(1) 缺陷记录与 (2) 未预登记的缺省分块 knob-on 测量保存，不裁定。

**预测判定**：P1 拆分（逐位 excsum/nelec HIT；dvmat 1.7-1.9e-6
略超预测带 [1e-9,1e-6] 上缘）；P2 拆分（2.40x/1.82x HIT，
m062x 1.47x 与 UKS 1.20x 略低于带）；P3 拆分（e_tot 位移带 HIT；
wb97mv 整 SCF 1.122x 低于预测 1.15 带下缘，b3lyp 平价 MISS——
文档级 1.336x/1.618x 高于带为周期效应叠加）；P4 HIT（无门控
反事实验证了道门必要性）；P5 HIT（methoxy fp64 双态收敛）。
决策规则三条件全 HIT → 落地。

**决策面扩展**：(b5)-类精度政策旋钮——knob-on 时 fp64 SCF 道的
收敛能量位移 ~1e-11 类（**5 个量级低于 (b3)/(b4) 的 ~1e-6 类
dg 交换代价**：SCF 是不动点求解器，vmat 扰动被变分收敛二次项
吸收），但速度收益也仅 SCF 相（1.3-1.6x MGGA 格）。决策简报
(b5) 行已记录。默认不变；全部旋钮 opt-in。

测量+实现+验证单元；证据 tasks/issue-9/2026-10-13-lc105-vmat-f32-
scf-knob 74 个 sha256 锚定文件（含 2 个缺陷保存目录与去混杂
诊断文档）；effects-index 27 数据集。

## 52. LC-106 (2026-10-13 评审续): 5 旋钮栈整道分界 —— knob-on 严格-fp64 道在三格全部端到端反超 auto 道（(b5) 决策面完成）

**动机**：LC-103 测得 4 旋钮完整栈上 wb97mv/benzene/tzvp 整道分界
穿越至 1.035（平价），auto 整道赢保持于 298 AO（0.788）与 m062x
（0.842）。LC-105 落地第 5 旋钮（SCF vmat fp32，fp64 道 t_scf
1.336x/1.618x）后，整道分界在 5 旋钮栈的自然补全——本单元以
LC-94/103 纪律逐字测量（配对相邻 4 旋钮对照 = LC-103 栈在现源
1ac5b2d；SCF_VMAT_F32 为唯一变量；脚本为修正后的 v2 字面量 env
纪律）。测量-only；6/6 文档 VALID，3 全新进程样本/道，全收敛。

**头条——三格全部反超**（frac = auto e2e / fp64 e2e；<1 auto 更
便宜，>1 fp64 更便宜）：

| 格 | 4knob 对照 | **5knob** | fp64 e2e 增益 | fp64 e_tot 位移 |
|---|---|---|---|---|
| wb97mv/benzene/tzvp (222 AO) | 1.032（复现 LC-103 的 1.035 于 0.3%） | **1.254** | 1.219x | +6.5e-11 |
| m062x/benzene/tzvp (222 AO) | 0.995（对照漂移，见下） | **1.273** | 1.329x | +3.2e-11 |
| r2scan3c/vitaminC/mtzvpp (298 AO) | 0.809（复现 LC-103 的 0.788 于 2.7%） | **1.253** | **1.604x** | +3.0e-10 |

knob-on 严格-fp64 道在全部三格端到端便宜 20-27%——包括 298 AO
SCF 主导格（auto 在此的整道优势 0.721→0.788 曾是最强持有）：旋钮
的 SCF 增益随 AO 增长（t_scf 3.511→1.981s = 1.77x，vmat 列车随
AO 放大），**完整 5 旋钮栈上混合精度的整道价值在所测全部格上被
抹平**。对 (b5) 的决策含义：fp64 道不只是"实用替代"——在所测
尺寸带上它现在是**更便宜的整道选择**；逐位严格证据仍需关态记录
（(b3)/(b4)/(b5) 语义不变）。

**auto 道跨态不变（P4，结构证明扩展到三格）**：e_tot 位移
0-1.9e-12、max|dg| 4.8e-8-5.5e-7 —— 每格均等于**同文档内**跨
进程地板（wb97mv 1.3-3.2e-7、m062x 3-4.9e-8、298 AO 格
4.2-6.1e-7；后者地板大于 222 AO 格——预登记带按 222 AO 格标定
的 1e-7 写，如实修正）；周期与开关记录逐字同。

**诚实记录**：(1) m062x 4 旋钮对照分界 0.995 vs LC-103 的 0.842
——auto 道 e2e 跨会话慢 13%（该格已记录的 ±15-23% 噪声类内；
fp64 对照漂 4%）；配对相邻比（协议估计量）不受影响；按决策规则
一次有界重跑不触发（在带内）。(2) r2scan3c fp64 道周期 16→17
（一周期诚实记录）。(3) r2scan3c e_tot 位移 +3.0e-10 大于 222
AO 格的 ≤6.8e-11——交换代价随尺寸/周期数增长，仍距 1e-6 门 3.5
个量级。

**预测判定**：P1 数字带 MISS 方向错（预登记把"fp64 道获益"误写
为分界下降——fp64 变便宜时 auto/fp64 比值上升；工作单元问题
"fp64 道是否决定性更便宜"答：**是**，1.254）；P2/P3 MISS（预测
auto 保持 0.70-0.78，实测两格均反超至 1.27——旋钮效应强于预
测，298 AO 的 auto SCF 优势被 1.77x 的 fp64 t_scf 增益覆盖）；
P4 HIT（含 298 AO 地板修正）；P5 HIT（含对照漂移注）。

测量-only；运行时源 1ac5b2d 不变；默认不变；全部旋钮 opt-in；
无实现跟随（决策规则：文档表 + effects-index 记录即关闭）。
证据 tasks/issue-9/2026-10-13-lc106-wholelane-5knob 10 个 sha256
锚定文件；effects-index 28 数据集。

## 53. LC-108 (2026-10-13): 687 AO 顶格补全 —— (b5) 的尺寸-安全边界（687 AO fp64 道不收敛）与 auto 整道赢的顶部保持；安全道成本面到顶

**动机**：LC-106 的头条（5 旋钮栈上 knob-on fp64 道端到端全部反超
auto）明确限定于"所测尺寸带"（≤298 AO）。687 AO（r2scan3c /
057_Tamoxifen / def2-mtzvpp，容量准入顶格，LC-57 峰值 6.95 GB）
的最后一次测量是 LC-57（仅 16384 栈、旧源）：auto 整道分界 0.70
= 最大 auto 整道赢。两条尺寸趋势相反且在顶部均未测：auto 优势随
AO 增长（LC-40/41）与旋钮的 fp64 增益随 AO 增长（LC-101/105，
(b5) t_scf 1.34x@222→1.77x@298）。安全道成本面在 298 AO 以上
从未测过（auto-strict 一次都没有；auto-verify 仅在 16384 栈）。
预登记 P1-P7 于运行前写入（preregistration.json）。

**设计**：LC-106 v2 纪律逐字（两分支字面量 env + 每态全 env 回显；
生产 gradient_runner；3 全新进程样本/道；最差重复聚合；文档内
fp64 锚）。两相邻文档：文档 A = 4 旋钮栈（SCF_VMAT_F32 未设），
道 fp64+auto（新的 687-AO 4 旋钮分界行 + (b5) 配对相邻对照）；
文档 B = 完整 5 旋钮栈，道 fp64+auto+auto-verify+auto-strict。
加一个有界诊断（单进程，max_cycle=60，逐周期能量迹线）。

**头条 1——(b5) 在 687 AO 打破 fp64 道 SCF 收敛**：文档 B 的
fp64 锚 **3/3 全新进程样本在生产 50 周期上限不收敛**（conv=False，
cyc=50，e_tot -1137.6325817952 = 收敛族之上 +7.88e-9）；文档
schema-VALID、0 行 pass——失败被设计内的 fail-closed 保留（fp64
= fail；其余道 = no_reference，锚不收敛即无效化该格门行，绝不重
标）。机制（诊断，逐周期迹线 + 生产设置复核）：SCF 于 ~17 周期
抵达定点邻域后在 ~1e-10 振幅的**刀锋极限环**上漂移（逐周期 dE
5e-12..1.4e-10 = conv_tol 1e-10 边缘）；**收敛结局是上下文敏感
的**——生产跑批上下文 3/3 于 50 周期上限失败（截断能量
-1137.6325817952136，位移盆 +7.88e-9），两次独立诊断（其一为
生产逐字设置 max_cycle=50 / direct_scf_tol=1e-14）分别于 45/39
周期收敛（25.0s/21.9s，同一位移定点 +8.82/+8.88e-9）：微观算术
差异（进程/分配上下文——生产样本之前有同进程 warmup SCF）决定
判据是否在上限内闭合。**即使收敛也更慢**（21.9-25.0s vs 关态
14 干净周期 15.6s）。≤298 AO 同旋钮 e_tot 位移 ≤3.0e-10、周期
稳定（LC-105/106）：扰动/conv_tol 比在 298-687 AO 之间越过 DIIS
闭合阈值（r2scan3c 类负载）。**(b5) 因此是尺寸条件性的：≤298 AO
快而安全，687 AO 不可靠（收敛由上下文决定，生产上下文 3/3 失败）
且每测得上下文净更慢——(b1)-(b5) 旋钮组里第一个测得的尺寸-安全
边界**。配对估计量（文档 A vs B，
时间相邻）：(b5) t_scf 比 0.864（更慢）、t_grad 比 1.057（平价，
SCF-only 旋钮）、fp64 e_tot 位移 +7.88e-9。

**头条 2——auto 整道赢在顶部保持（4 旋钮栈）**：文档 A frac
(auto/fp64 e2e) = **0.836**：auto 16.19s vs knob-on-fp64 19.36s。
LC-106 的抹平结论是尺寸限定的：222-298 AO（1.254/1.273/1.253）
与 687 AO（0.836）方向相反；**混合精度在准入带顶部的整道价值是
测得事实，非外推**。栈背景 vs LC-57（仅 16384、跨源——仅背景，
绝不作估计量）：fp64 t_grad 5.19→1.28s（(b3)+(b4) 4.05x）；auto
e2e 16.2→16.2s 不变、e_tot 逐位同（-1137.632581803097）、周期不
变（fp64 14 / auto 17）；文档内 auto 门 dE -1.8e-12、max|dg|
2.40e-6 PASS（LC-57 同格 16384 栈测 1.06e-5；(b3)/(b4) 核共享
dg 下降（LC-94 族）在 687 AO 复现）。文档 VALID（2 行 2 pass）。

**auto 道跨态不变——证明扩至 687 AO**：e_tot 位移 **0.0 逐位**、
周期同（[17]==[17]）、max|dg| 7.24e-7 = **687-AO 跨进程地板**
（该尺寸首次测得；与 298-AO ~5e-7 地板同族）。auto 策略按设计不
触旋钮（hf.py 门，LC-105），逐位不触碰记录现覆盖 222-687 AO。

**头条 3——安全道成本面到顶（>298 AO 首测）**：auto-verify e2e
36.30s、auto-strict 36.29s（strict/verify 平价 0.9998 = LC-99 无
分歧平价族延伸至顶格）；verify/auto 开销 2.24x；两道均参考-SCF
主导（t_scf 32.8s / 36.3s）。两道全部重复无触发（selected=main、
distinct=False）：受保护的 fp64 参考姊妹道 3/3 收敛于 14 周期至
-1137.632581802616..7——fp64 道记录之上 +4.8e-10，恰为 LC-58
修复后 fixcheck 姊妹位移族（LC-57 的姊妹记录同位移类，已交叉核
验）；dm 判据未触发（dm_max_abs_diff 1.78e-6 ≪ tol 1e-3）。
**LC-105 的严格参考保护门在顶格测得"在岗"**：文档 B 内未受保护
的 fp64 道开旋钮 3/3 失败的同一格上，每一 verify/strict 参考姊
妹道干净收敛——安全道的证书在该旋钮打破裸道的格上仍然有效。
背景：verify 36.30s vs LC-57 的 36.0s（跨源）——验证成本在整个
旋钮栈历史中持平（参考-SCF 主导；LC-93/99 结论延伸至 687 AO）。

**预测判定**：P1 MISS 且机制找到——(b5) 在 687 AO 破坏收敛，
5 旋钮分界不是有效性能行；预登记的诚实备选（"frac<1 = 混合精度
在 687 AO 保持整道价值"）经文档 A 的 0.836 落地；抹平结论保持
其尺寸范围。P2 MISS（预测 1.7-2.3x 继续增长，实测 0.864x——尺
寸趋势在 ~298 AO 后反转进入不收敛）。P3 分裂（t_grad 平价 HIT；
e_tot 位移 +7.88e-9 = 界 3e-10 的 26 倍 MISS——≤298 AO 族不外
推）。P4 HIT（0.0 逐位 / [17] 同 / 7.24e-7 地板）。P5 HIT
（36.30s 于 32-40 带；0.9998；无触发 6/6；文档 A 门 2.40e-6
PASS——文档 B 门行因锚失败为 no_reference，如实保留，其下的
auto 道是同一条逐位不触碰道）。P6 HIT（峰值 ≤7664 MiB）。
P7 于"全收敛"子句 MISS（文档 B fp64 3/3 不收敛 = 本发现；无
OOM、无非有限、两文档 schema-VALID 完整）。

**后果（无源码变更跟随；全部记入待决事项）**：(1) (b5) 决策行
获得测得的尺寸-安全边界——任何未来的 (b5) 默认开提案必须尺寸
限定；严格裁决已用关态记录（不变，此处再次确认）。(2) (b1)-(b5)
整道决策面两端均已测量：222-298 AO 抹平（LC-106）vs 687-AO
4 旋钮栈 auto 保持（0.836）且 (b5) 不可用。(3) 安全道成本面至
顶格完整：strict/verify 平价、参考-SCF 主导、无触发、参考保护
门在裸道被旋钮打破的格上有效。

**诚实记录**：文档 B 的 0-pass 态是设计内 fail-closed 行为（不
收敛的 fp64 锚无效化该格门行），无任何重标；生产上下文 3/3 失败
而独立诊断（含一次生产逐字设置的复核）收敛于 45/39 周期——结局
是**上下文敏感的刀锋边缘**（诊断在跑批上下文之外单独运行；首诊
断有两处非生产设置偏差 max_cycle=60 / direct_scf_tol=1e-12，复核
诊断逐字生产设置仍收敛，故上下文而非这两个参数是决定变量），且
即使收敛也净更慢并落 +8.8e-9 位移定点；跨源 LC-57 比较全部标注
为背景非估计量；裁决器第一个嵌套梯度展平 bug 被 TypeError 当场
抓住并于任何数字读出前修复（缺陷输出从未使用）；诊断脚本两个早
期属性错（with_df / args 命名空间）同样当场抓住修复，全部保留
于日志。

测量-only；运行时源 1ac5b2d 不变；默认不变；全部旋钮 opt-in。
证据 tasks/issue-9/2026-10-13-lc108-687ao-5knob-wholelane 16 个
sha256 锚定文件；effects-index 29 数据集（112 源全部校验通过）。

## 54. LC-110 (2026-10-13): (b5) 尺寸-安全边界定位于 [420, 687) 窗口 —— 420 AO 安全且增益最大（2.29x），整道抹平延伸至带中部（0.716→1.393）

**动机**：LC-108 发现 (b5) 在 687 AO 生产上下文不收敛（刀锋极限
环）、≤298 AO 安全（位移 ≤3.0e-10）——边界位于 298-687 窗口内但
位置未测；带中部的整道抹平问题同样未测（298 AO 5 旋钮分界 1.253
fp64 赢；687 AO 旋钮不可用、4 旋钮分界 0.836 auto 赢）。

**设计更正（诚实记录，运行前）**：原设计 r2scan3c/057_Tamoxifen/
def2-svp（537 AO）被 schema 规则 R4 于参数校验阶段拦截——r2SCAN-3c
的规约基组 def2-mtzvpp 是方法定义的一部分、不可变更，298-687 AO
之间不存在 r2scan3c 阶梯点；两次尝试 rc=2、零 GPU 秒（日志保留：
run_matrix_v1_blocked_r4.log + 两个空 stdout）。更正为
m062x/020_Vitamin_C/def2-tzvp（420 AO）：tau 列车重的 MGGA-杂化
（旋钮受益类）。局限如实记录：>298 AO 阶梯跨方法，边界结论按工
作负载类给出，非纯尺寸外推。更正设计于成功运行前重新预登记。

**测量结果**（LC-106/108 v2 纪律，两相邻文档 off→on，道 fp64+auto，
3 全新进程样本/道，全收敛，两文档 schema-VALID 全 pass）：

| 态 | 道 | e2e | t_scf | t_grad | 周期 | e_tot | 文档内门 | 峰值 |
|---|---|---|---|---|---|---|---|---|
| 4 旋钮 | fp64 | 6.52s | 5.53s | 0.68s | 14 | -684.7276390820432 | 锚 | 4734 MiB |
| 4 旋钮 | auto | 4.67s | 3.72s | 0.63s | 14 | -684.7276390820405 | dE 2.7e-12, dg 7.39e-7 PASS | 4762 MiB |
| 5 旋钮 | fp64 | 3.36s | 2.41s | 0.63s | 14 | -684.7276390820136 | 锚 | 4902 MiB |
| 5 旋钮 | auto | 4.69s | 3.69s | 0.66s | 14 | -684.7276390820400 | dE 2.7e-12, dg 1.40e-6 PASS | 4762 MiB |

**头条 1——(b5) 安全括号收窄至 [420, 687)**：420 AO 上旋钮开
的 fp64 道生产 50 周期上限内 3/3 收敛、周期数不变（两态均 14）、
e_tot 位移 +2.96e-11（≤298-AO 族类，非 687-AO 位移盆），配对
t_scf 增益 **2.292x** = 迄今最大（阶梯：1.48x @222 m062x →
2.29x @420 m062x → 687 破坏）。与 LC-108 的 687-AO 生产上下文
3/3 失败合并：**(b5) 的尺寸-安全边界位于 tau 类负载的 420-687
窗口**——至少到 420 AO 快而安全，687 AO 不可用。

**头条 2——整道抹平延伸至带中部**：4 旋钮分界 (auto/fp64 e2e)
= **0.716** = 迄今最强 m062x auto 整道赢（随 AO 增长趋势的
420-AO 点；222-AO 4 旋钮 m062x 分界 0.842-0.995 噪声带）；旋钮
开翻转为 **1.393**（fp64 3.36s vs auto 4.69s = 端到端便宜 28%）。
LC-106 的抹平结论因此贯通带中部：**旋钮安全处（≤420 AO 每一
测得尺寸）knob-on fp64 道端到端击败 auto 道；唯一测得的保持点
是 687 AO——恰为 (b5) 不可用之处**。混合精度在准入带上的整道
价值现仅存于顶部，且是旋钮自身失效所致，非 auto 速度所致。

**auto 道不变（P4）**：跨态 e_tot 位移 4.5e-13、周期同
（[14]==[14]）、max|dg| 9.30e-8（420-AO 跨进程地板类）——auto
结构性不触碰证明扩至第四个尺寸点（222/298/420/687）。

**预测判定**：P1 安全 HIT（3/3 收敛，括号 [420, 687)）；P2 增益
带上方 MISS（实测 2.292x，增长趋势强于预测带顶 2.1x）；P3 位移
HIT（+2.96e-11）；P4 HIT（4.5e-13/[14]/9.3e-8）；P5 分裂——
4 旋钮带下方 MISS（0.716 vs 预测 0.80-0.95，auto 赢强于预测、
与 AO 趋势一致），5 旋钮 HIT（1.393 于 1.0-1.4 带内 = 抹平延伸
至 420 AO）；P6 峰值 HIT（≤4902 MiB）；P7 完整性 HIT（两文档
VALID，无 OOM、无非有限）。

**后果（无源码变更；(b5) 决策行记录）**：(1) (b5) 安全区现括号
于 [420, 687)（tau 类负载）：420 AO 安全+2.3x，687 AO 破坏——
任何默认开提案必须尺寸限定于边界之下。(2) 整道决策面在整条准入
带上形状连贯：222/298/420 抹平（开旋钮分界 1.25-1.39）vs 687
唯一保持点（0.836，旋钮不可用）——混合精度整道价值的尺寸范围
现在是完备测得事实。(3) effects-index 数据集
lc110-b5-boundary-420ao。

**诚实记录**：R4 拦截完整保留（零 GPU 秒；验证器的规约基组规则
在任务执行前完成本职）；5 旋钮文档内 auto 门（1.40e-6）按
knob-ON fp64 锚（e 位移 +3e-11）计算——(b5) 语义不变，knob-on
fp64 非逐位严格参照；两个裁决器一次性小 bug（None 格式崩溃与空
max 表达式）在打印任何数字前崩溃、均已修复，缺陷运行无数字被
使用。

测量-only；运行时源 1ac5b2d 不变；默认不变；全部旋钮 opt-in。
证据 tasks/issue-9/2026-10-13-lc110-b5-boundary-537ao 16 个
sha256 锚定文件；effects-index 30 数据集（114 源全部校验通过）。

## 55. LC-112 (2026-10-13): (b5) 安全括号收窄至 [537, 687) —— 537 AO 安全但整道翻转仅为边缘平价（1.029），增益随尺寸非单调（1.72x）

**动机（运行前预登记）**：LC-108/110 把 (b5) 尺寸-安全边界括号于
[420, 687)（tau 类负载）——64% 尺寸跨度的灰区对 ~500 AO 用户无指
引。一个 537-AO 点将其减半。m062x 保持与 LC-110 相同的工作负载类
（tau 重 MGGA-杂化；r2scan3c 被 R4 基组锁定；wb97mv 改变 NLC 构
成）；tamoxifen 保持与 687-AO 边界格同分子。

**测量结果**（LC-106/108/110 v2 纪律，两相邻文档 off→on，道
fp64+auto，3 全新进程样本/道，全收敛，两文档 schema-VALID 全 pass）：

| 态 | 道 | e2e | t_scf | t_grad | 周期 | e_tot | 文档内门 | 峰值 |
|---|---|---|---|---|---|---|---|---|
| 4 旋钮 | fp64 | 19.15s | 15.21s | 2.38s | 13 | -1136.6021160034602 | 锚 | 9894 MiB |
| 4 旋钮 | auto | 13.13s | 9.31s | 2.27s | 13 | -1136.6021160034566 | dg 1.28e-6 PASS | 9864 MiB |
| 5 旋钮 | fp64 | 12.71s | 8.85s | 2.30s | 13 | -1136.6021160034443 | 锚 | 9960 MiB |
| 5 旋钮 | auto | 13.08s | 9.22s | 2.24s | 13 | -1136.6021160034580 | dg 1.85e-6 PASS | 9864 MiB |

**头条 1——(b5) 安全括号收窄至 [537, 687)**：537 AO 上旋钮开的
fp64 道生产上限内 3/3 收敛、周期数不变（两态均 13——**无刀锋征
兆**，对照 LC-108 的 687-AO 3/3 上限失败）、e_tot 位移 +1.59e-11
（小位移族类，远离 687-AO 的 +7.9e-9 位移盆）。**~500 AO 灰区用户
现获得具体指引：537 AO 安全**。

**头条 2——537 AO 整道翻转仅为边缘平价，非决定性抹平**：4 旋钮
分界 **0.686**（m062x 阶梯最强 auto 整道赢：0.716@420 vitaminC/tzvp
→ 0.686@537 tamoxifen/svp → 0.836@687 tamoxifen/mtzvpp——混合阶梯
上非单调，见诚实记录）；旋钮开分界 **1.029**：越过 1.0 但仅至平
价——异于 420 的决定性 1.393。翻转幅度取决于工作负载构成而非随
AO 单调：537 的配对 (b5) t_scf 增益 **1.719x** 低于 420 的 2.292x
（阶梯 1.48@222 → 2.29@420 → 1.72@537 混合分子与基组质量；格内
配对估计量才是有效数字）。537 处更小的旋钮增益遇上更强的 auto 优
势，两者抵消。对整道形状陈述的修正（LC-110 的概括）：**旋钮安全
处 knob-on fp64 至少平价（222-420 分界 1.25-1.39、537 平价
1.029），唯一保持点仍是 687（旋钮不可用）——但抹平仅 ≤420 决定
性，537 为边缘；该工作负载类的交叉点位于 ~537**。

**auto 道不变（P4，第五尺寸点）**：跨态 e_tot 位移 -1.36e-12、
周期同（[13]==[13]）、max|dg| 6.32e-7（537-AO 跨进程地板类，介于
420 的 9.3e-8 与 687 的 7.2e-7 之间——地板随尺寸增长的规律保持）。

**预测判定**：P1 安全 HIT（3/3、周期不变；括号 [537, 687)）；P2
增益带下方 MISS（1.719x vs 2.0-2.6x 预测——增益依赖工作负载构
成，混合阶梯上非 AO 单调）；P3 位移 HIT（+1.59e-11，小端，无边
界临近信号）；P4 HIT（-1.36e-12/[13]/6.32e-7）；P5 分裂——4 旋钮
带下方 MISS（0.686 低于 0.72-0.84 带，auto 赢再次强于预测）、
5 旋钮 MISS（1.029 vs ≥1.2——平价非决定性抹平）；P6 峰值 MISS
（9960 MiB vs ≤6.5 GB 估计——svp 杂化 K 构建；仍距 24 GB 设备
2.4x、预算内、无 OOM）；P7 完整性 HIT。

**后果（无源码变更；(b5) 决策行记录）**：(1) (b5) 安全括号
[537, 687)——灰区再减半；(b5) 默认开的论证随尺寸弱化即使安全：
增益收缩（2.29x→1.72x）而 auto 优势增长（0.716→0.686）。
(2) 整道形状精化：决定性抹平 222-420（1.25-1.39）、537 边缘平价
（1.029）、687 唯一保持（旋钮不可用）。(3) effects-index 数据集
lc112-b5-safety-537ao。

**诚实记录**：m062x 分界阶梯（0.716@420/0.686@537/0.836@687）与
增益阶梯（2.29@420/1.72@537）混合分子与基组质量（vitaminC/tzvp、
tamoxifen/svp、tamoxifen/mtzvpp）——格内配对估计量为承载，跨格
趋势陈述为背景（LC-110 的跨方法警示现扩展为跨基组）；P6 内存
MISS 是估计量错误（svp 杂化 K 构建）而非安全事件（峰值 9960 MiB、
预算内、准入门通过）；5 旋钮文档内 auto 门（1.85e-6）按 knob-ON
fp64 锚（+1.6e-11 位移）计算——(b5) 语义不变。

测量-only；运行时源 1ac5b2d 不变；默认不变；全部旋钮 opt-in。
证据 tasks/issue-9/2026-10-13-lc112-b5-boundary-537ao 13 个
sha256 锚定文件；effects-index 31 数据集（116 源全部校验通过）。

## 56. LC-115 (2026-10-13): GPU4PYSCF_SCF_VMAT_F32=headtail 生产验证 PASS —— 687-AO 尺寸-安全边界被头/尾切分移除（1.51x）

实现已在 commit cd6cdef（LC-114 探针 5/5 HIT 后落地）：`headtail`
值让 LC-105 的 fp32 vmat 收缩仅在 SCF 头部运行，首次周期末
|dE| < 切换阈值（auto 策略同款单触发）即翻回，收敛尾部走精确
fp64 收缩图；`=1` 语义不变；hf 道闸（纯 fp64 道；auto 家族 +
严格参照保护）对两值同样生效；默认不变；13 单测
（test_scf_vmat_f32_headtail.py，聚焦 13/13 + 全 tools 527 于
LC-116 预注册中新鲜捕获）。

生产验证（本节补全：操作员交接保留的未裁定单元；6 文档
12:03-12:10 已测，本会话裁定）：3 格 × off/headtail 配对相邻
（r2scan3c/tamoxifen/mtzvpp 687 AO 破坏格、m062x/vitaminC/tzvp
420 AO、wb97mv/benzene/tzvp 222 AO），完整 5 旋钮栈上下文（逐态
env 回显），生产 gradient_runner，3 新鲜进程样本，P1-P5 全 HIT
（decision PASS）：

- **P1**：687-AO headtail fp64 道 3/3 收敛、14 周期（与 knob-off
  完全相同）、真 fp64 定点（e_shift +1.14e-11 vs 相邻 off fp64，
  段 [12,22] 内）——LC-108 的 `=1` 边界（3/3 不收敛 + 7.9e-9
  位移 + ~1e-10 极限环）在同格被移除。
- **P2**：687-AO t_scf 1.509x（15.61→10.34 s；探针 1.510x
  复现）；off 文档兼作 LC-108 doc A 的首次新会话复现（14 周期
  /15.6 s）。
- **P3**：420 AO 1.430x（e_shift -4.5e-13）、222 AO 1.338x
  （-3.6e-11，周期 9→8 记录不平均）；两 headtail 文档内 auto
  门 PASS。
- **P4**：auto 道跨态不变——e_shift 0.0/0.0/-1.7e-13、周期逐位
  相同、dg 差 ≤7.6e-8（道闸对新值在生产中成立）。
- **P5**：6/6 文档 schema-VALID、off fp64 锚各 3/3、12/12 行
  PASS、峰值 ≤11 GiB。

**(b5) 决策行双值形式**：<537 AO `=1`（安全处更大增益，LC-110/
112 阶梯 2.29x/1.72x）；顶部 `headtail`（687 AO 唯一可用值，
1.51x）。诚实语义：headtail fp64 道非逐位严格参照（轨迹不同），
严格门裁定继续用关态记录（与 `=1` 同 (b5) 类）。裁定晚于队列
冻结完成（LC-116..119 全部在冻结默认栈、全旋钮关下运行），与
盲测完整性无交互。证据
`tasks/issue-9/2026-10-13-lc115-headtail-knob/`（31 文件 sha256
锚定；判定 analysis.json）；effects-index 数据集
lc115-headtail-production。

## 57. LC-116..LC-119 (2026-10-13): 化学多样性固定队列 —— 18 开发分子 4 方法核心矩阵 + TZVP 规模延伸 + 6 预留盲测；跨化学机制规则建立

用户 2026-10-13 新授权：增加不同分子寻找共性规律，而非旧 5 分子
上的重复。冻结队列
`/data/datasets/gpu4pyscf-chemistry-cohort-20261013-v1/manifest.json`
（24 分子、seed 20261013、18 开发 + 6 预留、4 开壳层、几何仅 CPU
筛选——非 GPU/科学接受证明）。全部在冻结基线（源 cd6cdef、默认
栈全旋钮关、生产 runner、门槛不变）上测量。

- **LC-116 校准**（乙醇/吡啶/乙氧自由基 spin=1/双酚 A，B3LYP/
  def2-SVP，3 道 × 3 样本）：12/12 PASS，无新失败族；乙氧 UKS
  干净；双酚 A 318 AO 默认栈 auto 整道赢 0.762（fp32 梯度驱动）。
- **LC-117 开发矩阵**（72 格、清单顺序、相邻 lane、3 样本）：
  **70/72 auto PASS**；两个失败**全部是甲氧基**（b3lyp 4.03e-8/
  1.16e-3 = LC-33 旋转族；wb97mv -1.283e-6/3.69e-2 = LC-29/39
  分支劈裂逐位复现）——**分子特异性而非自由基族**（乙氧全 4 法
  1e-12 类通过）：队列第一条跨化学机制规则。UKS 循环税同为甲氧
  基特异（+11/+8/+8 vs 乙氧 0-1、O2 +1）。默认栈分界：~150 AO
  以下 fp64/平价（中位 1.12）、≥208 AO auto 赢（中位 0.874）；
  **wB97M-V 默认栈从不决定性赢**（最好 0.973 平价）；族中位数跟
  尺寸不跟化学；最大峰值 8992 MiB。记账勘误（LC-119 记录）：
  LC-117 实际新鲜重测全部 72 格（4 个 o0NN 拷贝为来源副本，裁定
  从未读过）。
- **LC-118 TZVP 规模延伸**（6 分子 M06-2X/TZVP 同分子配对）：
  6/6 PASS；**整道交叉由基组驱动**（SVP→TZVP 每分子 frac 移
  -0.12..-0.18；m062x 下 auto 自 117 AO 起 TZVP 赢）；同分子
  dg 地板**不**随 AO 增长（P5 诚实否定：比值 0.15-0.88，乙氧
  1.64e-5→2.46e-6）。
- **LC-119 预留盲测**（24 格：吡咯/NO2 双自由基/乙胺/硝基甲烷/
  乙酰胺/肌苷 ×4 法；冻结记录先于首格写入、结果一次性判读）：
  **24/24 auto PASS**（max |dE| 1.26e-11、max dg 9.22e-6——
  能量门槛内 5 个数量级、梯度 3.2x 裕度）、auto-strict 24/24
  且零独立解采纳、216/216 收敛、0 OOM/跳过/超时（max 峰值
  3642 MiB @ 肌苷 466 AO）。**全部开发集规则在 6 个化学独立
  新分子上复现**：<150 AO fp64/平价（20 格 frac 1.003-1.296）、
  ≥326 AO auto 赢（肌苷 0.773-0.973，中位 0.821）、wB97M-V
  平价（中位 1.050）、NO2 UKS 税 +1（甲氧基特异规则扩展到第
  二自由基类）、dg 地板在 326-466 AO 格最大（7.10-9.22e-6）。
  P1-P6 全 HIT。

**队列级工程规则**（78 独立开发格 + 24 预留格，24 分子）：
(1) 默认栈混合精度在消费级尺寸跨化学多样性**门槛安全**——
102 独立格中 100 通过，仅有的失败是 2 个甲氧基格（已诊断、
auto-strict 完全恢复）；自由基族**不是**风险轴。(2) **按 AO
数而非化学选道**：<~150 AO fp64；~200-330 AO 起 auto；基组升
级在分子内把交叉点拉向 auto。(3) wB97M-V 默认栈整道价值上限
  平价（其 auto 赢需推荐旋钮栈，(b1)+(b2)）。(4) auto-strict
是安全生产路线（24/24 + 72/72、0 误采纳）。(5) 24 分子是工程
覆盖试点（单冻结 seed），非统计功效样本——不据此声称普适。
证据：`tasks/issue-9/2026-10-13-lc116-newmolecule-calibration`
（20 文件）、`2026-10-13-lc117-devmatrix-4method`（301 文件）、
`2026-10-13-lc118-tzvp-sizeextension`（27 文件）、
`2026-10-13-lc119-heldout-freeze-eval`（105 文件，全部 sha256
锚定）；effects-index 36 数据集（168 源全部校验通过）。

## 58. LC-120 (2026-10-13): 队列最慢格类的 VV10 墙 —— 组成剖析 88.6% NLC + 现有 LC-80 fp32 旋钮在 385k 网格点上移除它（5.9x SCF / 4.7x 端到端）

LC-119 点名的后续工作单元（队列最慢格 inosine/wB97M-V
1603 s 墙钟、"已记录未优化"）。测量单元，无源码改动；运行时
ff328d4（树干净）；默认栈、单变量 A/B（`GPU4PYSCF_VV10_UWE_KERNEL`
unset vs fp32，其余五个旋钮两支全部 unset，LC-105 v2 字面 env 纪律）。

**Phase A 组成剖析**（观测性探针，新鲜进程，fp64 道）：
inosine/wb97mv/def2-SVP（326 AO）t_scf 93.8 s / 17 循环中
**nlc（nr_nlc_vxc）83.1 s = 88.6%**，其中 `_vv10nlc` 内核 79.8 s
（18 次调用、4.44 s/次）；xc 仅 7.2 s；梯度阶段 VV10 占 49.9%
（4.41/8.84 s，生产路线 grid_response=False，LC-81 路线修正
在该尺度成立）。NLC 网格 **385,024 点**（苯/tzvp 127,746 的
3.0 倍，对工作量 ~9.1 倍）。队列 fp64 调查的 wb97mv SCF 成本
随原子数 ~二次增长（0.96 s/cyc @12 原子 → 5.36 @31 → 16.0
@57 他莫昔芬）正是 VV10 对-作用签名。

**Phase B 配对生产 A/B**（4 个 VALID 文档、24/24 重复收敛、
峰值 2.9-3.5 GiB、循环数逐位不变 17/17 与 13/13）：

| 格 | 道 | t_scf 增益 | 端到端增益 | 跨旋钮 fp64 dE | 跨旋钮 max\|dg\| |
|---|---|---|---|---|---|
| inosine 326 AO | fp64 | **5.86x**（91.4→15.6 s） | **4.70x**（101.4→21.6 s） | +3.55e-10 Ha | 2.9e-8 |
| inosine 326 AO | auto | 5.70x | 5.17x | +3.62e-10 | 1.5e-6 |
| bisphenol_a 318 AO | fp64 | **6.01x**（72.2→12.0 s） | **4.66x**（82.1→17.6 s） | +2.53e-10 | 1.7e-8 |
| bisphenol_a 318 AO | auto | 5.90x | 5.29x | +2.33e-10 | 4.9e-7 |

文档内验收门（auto 对自身 fp64）在**两种旋钮状态、两个分子**
全部通过（|dE| ≤ 1.9e-11、dg ≤ 9.8e-6）；内核级 `_vv10nlc`
每次调用 4.44 → 0.232 s = **19.1x**（LC-80 苯点为 13.2x——更大
网格的占用率使 fp32 内核获益更多，P2 高于预测带诚实记录）。
旋钮安全括号由 61k-128k 点扩展到 **61k-385k NLC 点**（交叉旋钮
fp64 dE 3.5e-10 = 门槛内 3.5 个数量级；分立倒数 + fp64 块累加
安全结构在 3 倍尺度成立）。

**默认栈 frac 语义修正**：旋钮开后 inosine 0.970→0.882、
双酚 A 0.970→0.853——**wB97M-V"默认栈至多平价"的读数是
车道共担的 VV10 墙造成的组成假象**，非 wb97mv 本质：墙以
fp64 成本压在两条道上，掩盖了 auto 道的剩余优势；移除后
318-326 AO auto 整道赢出现。LC-119 规则 3 按原样成立（默认栈
上该墙存在），细化为其成因是组成性的。

**Phase A2 旋钮后组成**（决策规则输入）：t_scf 18.3 s（探针
同步开销 ~2.7 s vs 生产文档 15.6 s）中 nlc 7.5 s（41%，其中
UWE fp32 4.2 s、rho/胶水 3.4 s）、xc 7.3 s（40%）、jk+k 3.2 s
（18%）。**无新的 ≥0.05 s/循环杠杆**：xc 列车 = LC-95/98 已
逐件测量命名的深工程；jk = LC-90 已闭合的积分构建瓶颈；nlc
包装 = NLC 网格上必需的 rho 求值；UWE = 本旋钮。按预注册决策
规则，单元作为测量闭合——**LC-80 fp32 旋钮就是队列最慢格类的
现成工程答案**（318-326 AO 默认栈下 wb97mv 格从 ~101/82 s 降至
~19/18 s 每样本，精度 3.5e-10 Ha）。晋升默认仍是 (b) 族用户
决策；默认与门槛不变。

诚实边界：inosine 为预留分子，仅作裁定后性能测量对象（LC-119
点名后续），其冻结盲测记录逐位未动；tamoxifen/wb97mv（537 AO，
VV10 墙 ~2.4x 于肌苷）未在本单元测量（单元时限，诚实的下一个
尺寸点候选）；剖析为单样本+同步开销，计时主张由 Phase B 生产
文档（每道 3 新鲜进程样本）承载。证据：
`tasks/issue-9/2026-10-13-lc120-vv10-large-molecule`（27 文件
sha256 锚定）；effects-index 37 数据集（176 源全部校验通过）。

### §58 续：LC-121 —— 括号顶端补全（他莫昔芬 537 AO / 679,488 NLC 点）

LC-120 诚实边界点名的候选：对 cohort 最大 VV10 墙（tamoxifen/
wb97mv/svp，537 AO，冻结记录 fp64 t_scf 208.0 s / 13 循环 =
16.0 s/循环）做同一纪律单变量 A/B（默认栈、仅
`GPU4PYSCF_VV10_UWE_KERNEL` unset vs fp32、生产 runner、每道 3
新鲜进程样本、显式 charge/spin、绝对 geom、字面 env、相邻
off→on、3900 s/doc 超时）。NLC 网格 **679,488 点** = 肌苷的
1.766 倍（对工作量 ~3.12 倍；CPU 侧网格计数锚定于
`nlc_grid_count.txt`）。

| 道 | t_scf | 端到端 | 跨旋钮 fp64 dE | max\|dg\| | frac (auto/fp64 e2e) |
|---|---|---|---|---|---|
| fp64 | **6.42x**（207.86→32.36 s） | **4.86x**（236.7→48.7 s） | +6.52e-10 Ha | 4.67e-8 | 1.014 → **0.818** |
| auto | **6.99x**（219.2→31.4 s） | **6.03x**（240.1→39.9 s） | +7.04e-10 | 8.10e-7 | |

循环数逐位不变（13/13、14/14）；12/12 重复收敛；峰值
8.4-8.9 GiB 预算内；文档内验收门两种旋钮状态全过（dE ≤
3.9e-11、dg ≤ 1.13e-5）；旋钮关文档复现冻结 LC-117 o40 记录
（t_scf 100.1%、e_tot +9.1e-13、循环 13）。P1-P6 全 HIT。

**括号性质**：跨旋钮 fp64 dE 随对数 ~线性增长（+2.5e-10@
~330k → +3.5e-10@385k → +6.5e-10@679k 点）——与分立倒数 +
fp64 块累加结构一致；按此趋势在允许带内（< 他莫昔芬 7 倍点
数）仍 ~5e-9 级、门槛裕度不受威胁，但 679k 之外是趋势陈述
非测量。**fp32-VV10 旋钮的安全括号就此覆盖允许带自底至顶：
61,078（甲醇）→ 127,746（苯/tzvp）→ 385,024（肌苷）→
679,488（他莫昔芬）NLC 点**；wB97M-V 大分子案例的 (b) 族决策
面自底至顶补全（4.9-6.4x fp64 t_scf、dE ≤ 7e-10）。旋钮保持
opt-in；默认与门槛不变；单元作为测量闭合。诚实边界：顶端单
分子（dev split；537-AO r2scan3c 格非 NLC，cohort 内无第二
顶端 wb97mv 点）；旋钮开 auto 道保持其 14 循环（与冻结记录
同类，+1 auto-vs-fp64 循环为该格固有属性）。证据：
`tasks/issue-9/2026-10-13-lc121-vv10-tamoxifen-topband`（14 文件
sha256 锚定）；effects-index 38 数据集（181 源全部校验通过）。

### §58 续 2：LC-122 —— 完整 5 旋钮推荐栈在 NLC 带顶端：车道平价级；LC-106 的翻转不外推到 537 AO（2026-10-13）

栈 × 尺寸 × 方法面上未测的角：LC-106 的完整栈翻转（frac
1.25-1.27，fp64 端到端便宜 20-27%）只在 222-298 AO 测过；
LC-121 测了本格（tamoxifen/wb97mv/svp 537 AO）但只用 VV10 单
旋钮栈（frac 0.818，auto 决定性领先）；LC-108 的 687 AO 保留
读数是 4 旋钮非 NLC 格。本单元以 LC-121/LC-115 同纪律测完整栈
在带顶端的表现：2 份相邻文档（doc1 = VV10-only 对照 = LC-121
on 态同会话复现；doc2 = 完整 5 旋钮栈 = LC-115 规范 env：
16384 blocksize + VV10 fp32 + ip1 fused + ip1 f32 + grad grid
f32 + vmat headtail），fp64+auto 两道、各 3 新鲜进程样本、
12/12 重复收敛、无 fallback 事件、峰值 8.4-9.7 GiB 预算内。
运行时 565ad30（对 0f47fa7 仅文档差异已验证）；测量性单元，
源码与默认值不变。

| 道 | t_scf | t_grad | 端到端 | 循环 | frac (auto/fp64 e2e) |
|---|---|---|---|---|---|
| fp64 | 31.93→28.83 s（1.11x） | 14.76→**5.15 s（2.87x）** | 48.2→**35.5 s（1.36x）** | 13→13 | 0.824 → **0.929** |
| auto | 31.28→26.56 s（1.18x） | 6.92→4.90 s（1.41x） | 39.8→**33.0 s（1.20x）** | 14→14 | |

三项载荷性结论：

1. **LC-106 的完整栈翻转不外推到 NLC 带顶端。** 完整栈下 frac
   = 0.929（3 个样本全部 < 1.0，auto 领先 7-14%）：栈把 auto
   道的梯度优势抹平（t_grad 4.90 vs 5.15 = 平价，同 222-298
   AO 机制），但抹不平其在 537 AO 的 SCF 优势（t_scf 26.56 vs
   28.83，-8%）——fp64 道的 5 旋钮 SCF 效应（1.11x）小于其梯
   度效应（2.87x），车道排序保留。完整栈下的 frac 阶梯现为：
   1.25-1.27（222-298 AO，翻转）→ 0.93（537 AO NLC，平价级）；
   翻转是带中部属性，非普适。
2. **完整栈对两道都有利**：fp64 e2e 1.36x、auto 1.20x。带顶
   端用户的工程答案是推荐栈本身，与车道选择无关；混合精度在
   满旋钮 fp64 道之上再增益 ≤ 7%。
3. **auto 道在 537 AO 不是栈不变的**（P3 MISS，真实效应）：
   e2e 比 0.83（t_scf -15%、t_grad -29%）——唯一触及 auto 道的
   栈变量是 16384 blocksize（ip1_fused 在 auto 道为记录在案的
   no-op LC-82；ip1_f32/grad_grid_f32/vmat_f32 有 fp64 道门
   LC-86/101/105）。这是首个清晰的 auto 道 blocksize 增益：
   LC-17「消费尺寸无 auto 道案例」在带顶端有了测量反例——
   (b1) 相关：537 AO 处 16384 blocksize 对两道都有利。

数值：文档内验收门两份文档全过（dE ≤ 3.8e-11、dg ≤
1.12e-5）；跨栈 fp64 dE +6.4e-12 / max|dg| 1.08e-5（核交换
族，LC-94 的 8.7e-6 同类）；doc2 文档内 dg 1.89e-6 比对照的
1.12e-5 小 6 倍 = (b3)/(b4) 核共享效应（旋钮开 fp64 道非逐位
严格参考；严格判定保留配对 VV10-only 文档与冻结 LC-117 默认
栈记录）。doc1 复现 LC-121 on 态（fp64 t_scf -1.3%、t_grad
0.03%、e2e -0.9%、e_tot 至 12 位、循环 13/13；auto e2e
-0.2%、循环 14/14）。auto 轨迹细节：完整栈下 auto 重复循环
[14,13,14]（对照 14/14/14）、e_shift -2.3e-12——blocksize 移
动 fp32 头轨迹（LC-77 族），一次重复早一个循环收敛，全部收
敛、最差重复聚合 14。

预注册判定 2/6 HIT：P1 复现 HIT、P6 内存 HIT；P2 高侧 MISS
（t_grad 2.87x 超 [1.3,2.3] 带——旋钮效应随尺寸增长族在梯度
组成上的带顶延伸）、P3 MISS（auto 非不变性，真实效应）、P4
MISS（诊断带宽 8% 过紧，验收门本身全过）、P5 边缘 MISS
（0.9293 vs 平价带下沿 0.93，差 0.08% 远小于 8.1% 的样本内
散布——平价级、auto 微优）。按预注册规则读为平价分支：
(b) 族在带顶端的指引为「完整推荐栈下车道选择为平价级
（auto ≤ 7% e2e）；旋钮对两道都是工程答案；222-298 AO 的
fp64 道翻转不外推到 537 AO」。诚实边界：顶端单分子（dev
split，cohort 结构性限制同 LC-121）；该尺寸无 UKS/自由基格，
auto 道 blocksize 发现为闭壳层专属；fp32_entry_hits 计数器只
计 auto 道条目（旋钮在 fp64 道的投入由 2.87x t_grad 增益 +
~1.0 GiB 峰值增量佐证，LC-112 类）。证据：
`tasks/issue-9/2026-10-13-lc122-fullstack-537ao-nlc-bandtop`
（14 文件 sha256 锚定）；effects-index 39 数据集（186 源全部
校验通过）。

### §58 续 3：LC-122 补遗 —— 687 AO 完整栈 frac 自冻结 LC-115 文档提取（零 GPU）；完整栈车道排序随组成非单调

LC-122 落定 537 AO NLC 点后，完整栈 frac 阶梯只缺 687 AO 值——
而它已存在于冻结的 LC-115 生产文档中（该单元已判定 PASS、
sha256 锚定；本补遗零 GPU、纯文档级读取）：
`r2scan3c_tamoxifen_mtzpp_{off,headtail}.json`——4 旋钮态 frac
0.835（第三次复现 LC-108 的 0.836），完整 5 旋钮态（headtail）
**frac = 1.141：fp64 道 14% 便宜**。翻转完全来自 fp64 道的
vmat-headtail SCF 增益（15.61→10.34 s，LC-115 P2 的 1.509x）反超
auto SCF（10.34 < 12.51 s）；auto 道栈不变（16.09→16.10 s、循环
17/17，LC-115 P4）。

完整栈整道 frac 阶梯（允许带内）：

| 点 | 格（方法/基组） | frac | 读数 |
|---|---|---|---|
| 222-298 AO | 混合（LC-106） | 1.25-1.27 | fp64 翻转 |
| 537 AO | wb97mv/svp NLC 重（LC-122） | 0.93 | 平价级，auto ≤ 7% |
| 687 AO | r2scan3c/mtzvpp tau 重 J-only（LC-115 文档，本补遗） | 1.141 | fp64 便宜 14% |

**面随组成非单调**，载荷机制是旋钮效应尺寸随方法变化而非尺寸
本身：vmat-headtail 的 SCF 效应在 tau 重 MGGA 格最大（687 AO
1.51x，LC-105 MGGA 族）并翻转 SCF 排序；NLC 格（537 wb97mv）的
post-VV10 SCF 余项非 vmat 主导（fp64 SCF 栈效应 1.11x），auto
保留 SCF 优势；VV10 fp32 旋钮车道共担、从不改变排序
（LC-120/121）。诚实警示：阶梯的 537 与 687 点在方法、基组、
分子三方面都不同（LC-110/112 的阶梯混合警示延伸到完整栈 frac
阶梯）——格内配对估计量仍是载荷性证据，阶梯为语境。(b) 族指引
精化：完整推荐栈下，带顶端车道排序随工作负载组成而定——NLC
重格保留小幅 auto 优势（平价级），tau 重 MGGA 格翻向 fp64 道
（vmat 旋钮是翻转驱动）。旋钮在两侧都是工程答案；默认不变；
全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc122-fullstack-537ao-nlc-bandtop`
（15 文件 sha256 锚定，含 `addendum-687-frac-extraction.md`）。

## 59. LC-123 (2026-10-13): 完整 5 旋钮栈在 NLC 带中部 —— 318/326 AO 平价级（0.86→0.98 / 0.89→0.94）；auto 道优势的载体是 blocksize 创造的 SCF 增益，而非 ≥537 专属缓冲

LC-122 落定 537 AO NLC 点后，完整栈 frac 阶梯在 222-298 与
537 之间的 NLC 带中部（318/326 AO，队列最慢格类的另一端）仍
未测量——LC-120 只测了该尺寸的 VV10-only 栈（frac 0.853/0.882
= auto 显著赢）。本单元以 LC-122 纪律逐字补全该角：
bisphenol_a/wb97mv/def2-svp（318 AO，385,002 NLC 点）与
inosine/wb97mv/def2-svp（326 AO；预留分子冻结后仅作性能对象，
LC-120 先例），每分子 doc1 VV10-only 对照 vs doc2 完整栈相邻
配对，fp64+auto 两道、各 3 新鲜进程样本。4 份 VALID 文档、
24/24 重复收敛、无 fallback、峰值 2.9-4.3 GiB。运行时 8df5292
（对 ff328d4 [LC-120] 与 565ad30 [LC-122] 仅文档差异已验证）；
测量性单元，源码与默认值不变。

| 分子 | 栈 | fp64 t_scf | fp64 t_grad | fp64 e2e | auto t_scf | auto t_grad | auto e2e | frac |
|---|---|---|---|---|---|---|---|---|
| bisphenol 318 AO | VV10-only | 12.232 | 5.276 | 17.853 | 12.455 | 2.507 | 15.326 | 0.858 |
| bisphenol 318 AO | 完整栈 | 9.450 | 1.557 | 11.336 | 9.111 | 1.561 | 11.145 | **0.983** |
| inosine 326 AO | VV10-only | 15.503 | 5.572 | 21.446 | 16.033 | 2.664 | 19.066 | 0.889 |
| inosine 326 AO | 完整栈 | 12.116 | 1.745 | 14.236 | 11.242 | 1.733 | 13.343 | **0.937** |

（循环不变 13/13 与 17/17；完整栈配对效应：fp64 t_grad
**3.39x/3.19x**、t_scf 1.29x/1.28x、e2e 1.58x/1.51x——均超过
537 AO 的 2.87x/1.11x/1.36x。）

四项载荷性结论：

1. **带中部的完整栈 frac 为平价级**：0.858→0.983（bisphenol）、
   0.889→0.937（inosine）。预注册机制判别成立（P5 HIT）：
   318-326 AO 处 auto 道在 VV10-only 栈下**没有** SCF 优势
   （t_scf 落后 fp64 +2-3%，与 537 AO 的 -8% 缓冲相反），无缓冲
   之下完整栈把 auto 赢面侵蚀到 ≤6.3%。
2. **auto 道在完整栈下的 NLC 优势不是 ≥537 专属属性——
   16384 blocksize 在带中部直接创造它**：inosine auto t_scf 从
   +3.4% 落后翻到 7.3% 领先（16.03→11.24 vs fp64 15.50→12.12）；
   bisphenol 从 +2.3% 落后到 3.7% 领先。存活的 auto 优势由
   blocksize 驱动的 SCF 增益 + t_grad 平价（1.56/1.56、
   1.73/1.74）承载，而非内在大 AO SCF 缓冲——对 LC-122 读数
   的机制精化（非仅范围修正）。
3. **auto 道 16384 增益向更小 NLC 尺寸增强**：e2e 比 0.700
   （326）/<0.727（318）vs 0.83（537，LC-122）vs 687 r2scan3c
   栈不变（LC-115）。LC-17「消费尺寸无 auto 道案例」现于三个
   NLC 尺寸有测量反例；该增益为 NLC 类专属（J-only 687 格无）。
   (b1) 相关：NLC 格 318-537 AO 全带 16384 blocksize 对两道
   都有利。
4. **完整栈 frac 阶梯按组成分带、NLC 类内平坦**：
   1.25-1.27（222-298 AO 非 NLC）| 0.937-0.983（318-326 NLC，
   本单元）→ 0.929（537 NLC，LC-122）| 1.141（687 tau 重
   MGGA J-only，LC-115 文档提取）。NLC 格在完整推荐栈下的车道
   排序于**所有已测尺寸 318-537 AO 均为平价级（auto ≤ 7%）**；
   完整栈下的车道选择由工作负载组成（NLC vs tau 重 vs 普通混合）
   决定，而非 AO 单独决定——(b) 族指引在带中部收口。

数值：文档内验收门 4 份全过（auto dE -2.6e-11…+2.8e-11、
max|dg| 1.1e-6…1.0e-5）；跨栈 fp64 dE 6.8e-13/4.5e-13、
max|dg| 7.9e-6/9.8e-6（核交换族，LC-122 的 1.08e-5 同类）；
完整栈文档内 dg（1.1e-6/1.9e-6）比对照小约 5 倍 = (b3)/(b4)
核共享效应再次出现。doc1 逐位复现 LC-120 on 态：bisphenol
fp64 e_tot 12 位一致（-730.833368448368）、auto dE
-1.9554136088117957e-11 **精确一致**；inosine fp64 e_tot 12 位
一致；时间 ±2% 内、循环精确。

预注册判定 6/10 HIT：P1×2 HIT（复现）、P4×2 HIT（数值门）、
P5 HIT（头条）、P6 HIT（内存）；P2×2 信息性高侧 MISS（t_grad
3.2-3.4x 超预注册 [1.8,3.0] 带——带中部旋钮效应超过 537 的
2.87x 外推，增益随尺寸非单调族 [LC-112] 再添两点）、P3×2
真实效应低侧 MISS（auto e2e 比 0.70-0.73 低于 [0.80,1.05]
不变性带——blocksize 增益在带中部比 537 更强，发现 3）。按
预注册决策规则读为命中分支：LC-122 的「NLC 格保留小幅 auto
优势」精化为「NLC 类全带（318-537 AO）平价级、auto ≤ 7%，
载体为 blocksize 创造的 SCF 增益」。诚实边界：该角 2 分子/
1 方法/1 基组；blocksize 归因来自单变量栈配对（auto 道只触及
blocksize，同 LC-122 推断类）；闭壳层专属（无 UKS/自由基格）；
fp32_entry_hits 计数器只计 auto 道。证据：
`tasks/issue-9/2026-10-13-lc123-fullstack-bandmiddle-nlc`
（23 文件 sha256 锚定）；effects-index 40 数据集（193 源全部
校验通过）。

## 60. LC-124 (2026-10-13): 完整 5 旋钮栈在小尺寸带 48-92 AO —— 默认开启安全角：全带无回归（每道每格 1.22-3.21x 增益）；LC-101 单旋钮回归不组成进整栈

完整栈决策面此前只测过 222-687 AO（LC-106/108/110/112/115/122/
123）；小尺寸带——auto 道在默认栈下比 fp64 慢的区间（LC-117 交叉
150-208 AO 以下）——从未在完整栈下测过。这是 (b) 族默认开启决策
的安全角：栈是否**回归**小格？LC-101 曾单旋钮实测 GRAD_GRID_F32
的 0.805x b3lyp 回归（中等尺寸）。本单元以 LC-122/LC-123 纪律补全：
4 个 dev 格（不含预留分子）各配对 doc1 默认栈 vs doc2 完整栈，
fp64+auto 两道、各 3 新鲜进程样本——methanol/b3lyp/svp 48 AO
（回归哨兵）、methylamine/wb97mv/svp 53 AO（小 NLC）、
methanol/m062x/svp 48 AO（小 tau MGGA）、ethanol/r2scan3c/
mtzvpp 92 AO（小 3c J-only）。8 份 VALID 文档、48/48 重复收敛、
无 fallback、峰值 88-432 MiB。运行时 5160a23（cd6cdef→5160a23
仅文档差异已验证——LC-117 冻结记录同运行时）。

| 格 | 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|---|
| b3lyp/甲醇 48 | fp64 | 0.503→0.391（1.29x） | 0.208→0.176 | 0.780→0.637（**1.22x**） | 1.295→1.151 |
| b3lyp/甲醇 48 | auto | 0.636→0.473（1.35x） | 0.280→0.183 | 1.011→0.733（**1.38x**） | |
| wb97mv/甲胺 53 | fp64 | 3.760→0.997（**3.77x**） | 0.628→0.297 | 4.408→1.374（**3.21x**） | 1.133→1.181 |
| wb97mv/甲胺 53 | auto | 4.258→1.193（3.57x） | 0.669→0.328 | 4.996→1.623（3.08x） | |
| m062x/甲醇 48 | fp64 | 0.739→0.559（1.32x） | 0.298→0.213 | 1.113→0.876（**1.27x**） | 1.111→1.061 |
| m062x/甲醇 48 | auto | 0.888→0.615（1.44x） | 0.235→0.235 | 1.237→0.930（1.33x） | |
| r2scan3c/乙醇 92 | fp64 | 1.108→0.682（1.62x） | 0.266→0.197 | 1.446→1.005（**1.44x**） | 1.085→1.083 |
| r2scan3c/乙醇 92 | auto | 1.169→0.794（1.47x） | 0.307→0.206 | 1.568→1.089（1.44x） | |

（每道每格循环不变；跨栈 e_tot 全部 ≤ 2.1e-12。）

五项载荷性结论：

1. **P5 命中——完整栈在小尺寸带全带无回归**：每道每格 e2e 增益
   1.22x-3.21x，最大配对 doc2/doc1 比值 0.82（阈值 1.05）。(b) 族
   默认开启面补上小尺寸安全角：**整个允许尺寸范围 48-687 AO 上，
   完整栈对每条实测道都是增益或中性**（687 的 '=1' 不收敛例外已由
   LC-108/115 以 headtail 形式尺寸限定）。
2. **LC-101 单旋钮 0.805x 回归不组成进整栈**（48 AO 哨兵格反增
   1.22x/1.38x）：单旋钮回归被栈内其它效应在有益方向上支配。
   诚实边界：这证明的是 48 AO 整栈无回归，非该回归处处消失
   （未在其原始中等尺寸重测单旋钮态）。
3. **VV10 墙在 53 AO 就存在，旋钮在 3.77x 移除它**（fp64 t_scf
   3.76→1.00 s；LC-80 的 2.43x 整 SCF 锚被超过）；车道共担，两道
   ~3x——NLC 发现从 318-537 AO 一路延伸到队列最小 NLC 格。
4. **auto 道 blocksize 增益延伸到小尺寸带**：三个非 NLC 格上
   （blocksize 是唯一触及 auto 道的栈变量）auto e2e 比 0.69-0.75
   （b3lyp 0.73 / m062x 0.75 / r2scan3c 0.69）。LC-17「消费尺寸
   无 auto 道案例」现有 48/53/318/326/537 AO 五个反例，仅在
   687 J-only 格不出现——(b1) 相关：除 687 J-only 外 16384 在
   **所有已测尺寸**对两道都有利。
5. **完整栈 frac 阶梯至此在所有尺寸带有测点**：1.06-1.18
   （48-92，本单元）→ 1.25-1.27（222-298 非 NLC，LC-106）|
   0.937-0.983（318-326 NLC，LC-123）→ 0.929（537 NLC，
   LC-122）| 1.141（687 tau 重 J-only）。小带四格 auto 全部慢于
   fp64（frac 1.06-1.18）——fp64 优势排序延伸到默认栈交叉点
   以下；完整栈下唯一 auto 占优区域仍是 NLC 类 318-537 AO。

数值：文档内验收门 8 份全过（auto dE ≤ 7.7e-13、dg ≤ 2.3e-6）；
跨栈 fp64 dE ≤ 2.1e-12、max|dg| ≤ 3.9e-7（核交换类小格地板）。

预注册判定 11/18 HIT：P1×4 HIT（勘误：判定脚本最初把 LC-117
锚值按控制台打印精度 10 位小数转录，产生 3 个虚假 P1 MISS，已就地
修正为冻结 JSON 文档的全精度值并记录——无任何测量或科学结论受影
响）、P4×4 HIT、P5/P6 HIT；P2 三处高侧 MISS（wb97mv 3.77x 超
[1.8,3.2]、r2scan3c 1.62x 超 [1.05,1.40]、b3lyp 哨兵 1.22x 超带上
沿）、P3×4 低侧 MISS（auto e2e 比 0.32-0.75 低于 [0.55,1.15] 带）。
**每个 MISS 都是栈比预测帮得更多**——安全筛选的最优结果形态。
诚实边界：每组成类 1 分子；blocksize 归因来自单变量栈配对；亚秒
绝对时间（启动开销权重）；LC-117 的 methoxy UKS 失败族未重探
（本单元问的是栈响应，非精度类）。证据：
`tasks/issue-9/2026-10-13-lc124-fullstack-smallband`
（38 文件 sha256 锚定）；effects-index 41 数据集（204 源全部
校验通过）。

## 61. LC-125 (2026-10-13): 完整 5 旋钮栈在 UKS/开壳层格 —— 壳类角收口：开壳层全带无回归（1.60-3.69x）；UKS 上的 VV10 墙 4.09x 移除；frac 壳轴 fp64 占优

完整栈决策面至此只剩壳类未测：此前所有完整栈文档均为闭壳层 RKS
（LC-106/108/110/112/115/122/123/124，各自记录「仅闭壳层」诚实边
界），UKS 格只测过默认栈（LC-117）或 4 旋钮栈（LC-39）。完整栈 ×
UKS 首次同时行使：headtail 于 `_nr_uks_task`（LC-115 是 RKS）、
VV10-fp32 于 UKS NLC 格（LC-120/121/124 均为 RKS）、ip1_f32 的
UHF J+K 选择点、16384 blocksize 于 UKS auto 轨迹。三个干净 dev
UKS 格（methoxy 故意排除——支线分裂格，决策 (c) 门控）：ethoxy/
wb97mv/svp 67 AO s=1（UKS NLC，VV10 墙 8.1 s）、ethoxy/m062x/svp
67 AO s=1（UKS tau MGGA）、oxygen_triplet/b3lyp/svp 28 AO s=2
（三重态杂化）。6 份 VALID 文档、36/36 重复收敛、无 fallback、
峰值 70-394 MiB。运行时 b5f6b3d（cd6cdef→b5f6b3d 仅文档差异已验
证——LC-117 冻结记录同运行时）。

| 格 | 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|---|
| wb97mv/乙氧 67 UKS | fp64 | 8.261→2.022（**4.09x**） | 0.915→0.397 | 9.289→2.518（**3.69x**） | 1.015→1.224 |
| wb97mv/乙氧 67 UKS | auto | 8.480→2.558（3.31x） | 0.875→0.450 | 9.425→3.081（3.06x） | |
| m062x/乙氧 67 UKS | fp64 | 2.049→1.192（1.72x） | 0.392→0.205 | 2.508→1.470（**1.71x**） | 1.168→1.247 |
| m062x/乙氧 67 UKS | auto | 2.467→1.494（1.65x） | 0.404→0.265 | 2.930→1.833（1.60x） | |
| b3lyp/O₂三重态 28 | fp64 | 0.477→0.306（1.56x） | 0.204→0.098 | 0.748→0.451（**1.66x**） | 1.179→1.193 |
| b3lyp/O₂三重态 28 | auto | 0.619→0.358（1.73x） | 0.215→0.136 | 0.882→0.538（1.64x） | |

（循环不变，唯 O₂三重态 fp64 8→9（全部重复）——headtail 于三重态
的单循环轨迹效应，被 1.66x 增益吸收，全部收敛。）

五项载荷性结论：

1. **P5 命中——完整栈在 UKS/开壳层格全带无回归**：每道每格 e2e
   增益 1.60x-3.69x，最大配对 doc2/doc1 比 0.63。LC-124 的无回归
   结论扩展到**两个壳类**：整个允许尺寸范围 48-687 AO × RKS+UKS
   上，完整栈对每条实测道都是增益或中性。headtail × UKS 切换触
   发、VV10-fp32 × UKS NLC、ip1_f32 × UHF J+K 均无病理。
2. **UKS 上的 VV10 墙被 4.09x 移除**（fp64 t_scf 8.26→2.02 s；
   auto 3.31x）——大于 RKS 小 NLC 锚（53 AO 3.77x）；fp32-VV10
   安全括号现含 UKS NLC 格（跨栈 fp64 dE 4.6e-11，核交换类，为
   已测最大完整栈 fp64 位移，仍远低于 1e-6 门 4 个量级）。
3. **紧裕度 UKS 格在完整栈下保持梯度门裕度**：m062x/乙氧 auto
   dg 1.688e-5 vs 默认 1.638e-5（门 3e-5）——裕度保持而非消耗
   （无核共享缩减：该格 auto 误差为解项主导，LC-89 类）。
4. **完整栈 frac 壳轴以 fp64 占优收口**：UKS/NLC 1.224、UKS/
   MGGA 1.247、三重态杂化 1.193——连同 LC-124 小 NLC RKS 1.181，
   读数精化：完整栈下唯一 auto 占优区域是 **NLC 类 318-537 AO**
   （0.93-0.98）；小 NLC 格（53-67）与非 NLC 同为 fp64 占优——
   auto 优势是中-大尺寸 NLC 属性，非 NLC 类属性。
5. **auto 道 blocksize 增益延伸到 UKS 轨迹且无循环税**：auto
   e2e 比 0.33/0.63/0.61；UKS auto 循环全部不变（14/14/9→
   14/14/9）——16384 下这些格无 LC-91 类默认分块循环税。headtail
   × UKS 交互至少与 RKS 同强（m062x 1.72x UKS vs 1.32x RKS@48
   ——UKS 付两次 vmat 任务遍）。

预注册判定 11/14 HIT：P1×3（复现，fp64 e_tot 至 5e-14）、
P3×3、P4×3（文档内门全过；跨栈 fp64 dE ≤ 4.6e-11、dg ≤
2.2e-6）、P5、P6 HIT；P2×3 高侧 MISS（全为增益超带：4.09x vs
4.0 带缘、1.72x vs 1.45、1.66x vs 1.30——与 LC-124 相同的「帮得
比预测多」形态）。诚实边界：每类 1 分子；队列中 67 AO 以上无
UKS 自由基格（结构性）；methoxy 类 UKS 门风险未测（决策 (c) 门
控）；O₂单循环轨迹位移为良性记录。证据：
`tasks/issue-9/2026-10-13-lc125-fullstack-uks`
（30 文件 sha256 锚定）；effects-index 42 数据集（213 源全部
校验通过）。

至此完整栈决策面在允许空间的**每个角**均有测点：尺寸 48-92 /
222-298 / 318-326 NLC / 537 NLC / 687；壳类 RKS + UKS；组成
杂化/MGGA/NLC/3c。全测域无回归（687 的 '=1' 不收敛例外已由
LC-108/115 以 headtail 形式尺寸限定）。(b) 族决策面完整，等待
用户政策决定；全部旋钮保持 opt-in，默认与门槛不变。

## 62. LC-126 (2026-10-13): 完整 5 旋钮栈在 SVP NLC 交叉缺口 —— auto 占优区域下缘定位于 (109, 208] AO；缺口带 + 两个新化学家族无回归（4.81-6.86x）

「完整栈下唯一 auto 占优区域 = NLC 类 318-537 AO」这一旗舰工程规则
（LC-122..125）此前依赖一段 250-AO 宽的未测内插：SVP NLC 轴上 67
与 318 AO 之间无任何完整栈测点，规则所暗示的交叉点从未被定位。
本单元在缺口内测三个 development-分割 wb97m-v/def2-svp 格，每个
配对相邻 doc1（默认栈）/doc2（完整栈，六旋钮全开、RKS 无结构性
no-op）：butane 106 AO（饱和链族首个完整栈点）、pyridine 109 AO
（芳香杂环族首个）、vitamin_c 208 AO（多官能团；默认栈 auto 整道
赢规则恰始于该尺寸，LC-117/119）。6 份 VALID 文档、36/36 重复收
敛、无 fallback、峰值 208-1430 MiB。运行时 3b77d1b（cd6cdef→
3b77d1b 仅文档差异已验证——LC-117 冻结记录同运行时）。

| 格 | 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|---|
| wb97mv/butane 106 | fp64 | 9.178→1.569（**5.85x**） | 1.429→0.377（3.79x） | 10.659→2.016（**5.29x**） | 1.026→1.073 |
| wb97mv/butane 106 | auto | 9.459→1.699 | 1.414→0.404 | 10.937→2.163（5.06x） | |
| wb97mv/pyridine 109 | fp64 | 9.562→1.613（**5.93x**） | 1.360→0.358（3.80x） | 11.028→2.042（**5.40x**） | 1.009→1.134 |
| wb97mv/pyridine 109 | auto | 9.899→1.831 | 1.152→0.407 | 11.130→2.316（4.81x） | |
| wb97mv/vitamin_c 208 | fp64 | 33.601→4.775（**7.04x**） | 3.929→0.743（5.29x） | 37.701→5.660（**6.66x**） | 1.001→0.972 |
| wb97mv/vitamin_c 208 | auto | 34.601→4.582 | 3.080→0.798 | 37.753→5.505（6.86x） | |

（每道每格循环全部不变：8/8、11/11、14/14，全部重复。）

五项载荷性结论：

1. **P5 命中——交叉点定位于 (109, 208] AO**：完整栈 frac
   butane 1.073 / pyridine 1.134（fp64 占优）vs vitamin_c 0.972
   （auto 占优）。auto 占优区域从「NLC 318-537」精化为「NLC
   ~(109,208]-537 AO（SVP 轴）」：~208 AO 起的中尺寸 NLC 用户在
   完整栈下 auto 道是更快的安全选择；~110 AO 以下 knob-on fp64
   道更快。小内缘非单调（106→1.073 < 109→1.134，分子依赖），
   以上单调（208→0.972 ≥ 318→0.983 ≈ 326→0.937 ≈ 537→0.929）。
2. **VV10 墙随尺寸平滑增长（fp64 道）**：t_scf 增益
   3.77x@53（LC-124）→ 5.85x@106 → 5.93x@109 → 7.04x@208 →
   ~7.4x@318（LC-120/123）——小 NLC 与带中部锚点间的缺口被三级
   平滑阶梯闭合；auto 道增益更大（e2e 5.06-6.86x，auto e2e 比
   0.146-0.208）。
3. **P6 命中——无回归延伸到缺口带 + 两个新化学家族**：每道每格
   e2e 增益 4.81-6.86x（最大配对 doc2/doc1 比 0.21）；饱和链与
   芳香杂环族加入完整栈已行使家族。峰值随 16384 blocksize 上升
   （298→664、296→556、1182→1430 MiB），均远低于预算。
4. **最大缺口格的梯度裕度改善 8 倍**：vitamin_c auto dg
   1.005e-5（默认）→ 1.26e-6（完整栈）——LC-124 核共享缩减类在
   RKS NLC 格上的复现；循环不变（14/14）。
5. **跨栈 fp64 dE 随尺寸增长（NLC 轴）**：4.5e-11@106、5.8e-11
   @109、2.3e-10@208——与 LC-121 的 dE~点对数线性 VV10-fp32 机制
   一致（537 VV10-only 6.5e-10）；2.3e-10 为已测最大闭壳层完整
   栈 fp64 位移，仍低于 1e-6 门 4 个量级（核交换类，auto 道文档
   内门不受影响）。

精化后的完整栈 SVP NLC frac 阶梯：53→1.181（LC-124）、67 UKS→
1.224（LC-125）、**106→1.073、109→1.134、208→0.972（本单元）**、
318→0.983、326→0.937（LC-123）、537→0.929（LC-122）；非 SVP 语境：
222 tzvp→1.254（LC-106）、687 tau 重 J-only→1.141（LC-122 提取）。

预注册判定 10/14 HIT：P1×3（复现，fp64 e_tot 至 2.5e-13、循环
精确）、P4×3（文档内门全过，dE ≤ 2.4e-11、dg ≤ 1.005e-5；跨栈
fp64 dE ≤ 2.3e-10、dg ≤ 9.8e-6）、P5、P6 HIT；P2 vitamin_c
高侧 MISS（7.04x vs 带顶 6.5）与 P3×3 低侧 MISS（auto e2e 比
0.146-0.208 vs 带 [0.25,0.60]）——全部 MISS 均为「帮得比预测
多」形态（LC-124/125 复现）；|auto e_shift| ≤ 2.6e-10、循环
全不变。诚实边界：每个缺口点 1 分子（交叉区间两侧各由单格界
定，区间内部结构未测）；vitamin_c 的 auto 占优读数（0.972）低
于平价 2.8%——平价级而非决定性（~208 下缘读数偏软）；全部
development-分割（heldout 未触及）；完整栈文档 e2e 绝对时间
< 6.9 s、每道 3 计时样本；dE-尺寸观察基于 3+1 点非拟合律。
测量-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc126-fullstack-svp-nlc-crossover`
（31 文件 sha256 锚定）；effects-index 43 数据集（222 源全部
校验通过）。

## 63. LC-127 (2026-10-13): 苯 114 AO 完整栈 —— 交叉点为分子依赖而非单一 AO 阈值；§62 规则修正（苯在两种基组下均 fp64 占优）

LC-126 将完整栈 SVP NLC 交叉点定位于 (109, 208] AO。苯/wb97m-v/
def2-svp（114 AO）是区间内最后一个 development NLC-SVP 格；本配对格
（doc1 默认栈 vs doc2 完整栈）收紧区间并携带同分子基组问题（苯/
tzvp 222 AO 在完整栈下 fp64 占优 1.254，LC-106）。2 份 VALID 文档、
12/12 重复收敛、无 fallback、峰值 272-606 MiB。运行时 1bad30f
（cd6cdef→1bad30f 仅文档差异已验证——LC-117 冻结记录 o16 同运行时）。

| 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|
| fp64 | 7.755→1.302（**5.96x**） | 1.339→0.363（3.69x） | 9.191→1.733（**5.30x**） | 1.124→**1.258** |
| auto | 9.010→1.715 | 1.250→0.397 | 10.327→2.180（4.74x） | |

（循环：fp64 8→7——headtail 轨迹**快一个循环**（全部重复，全部收
敛；与 LC-125 O₂三重态的 +1 反号、同为良性）；auto 9→9 不变。）

四项载荷性结论：

1. **P5 信息性 MISS——苯@114 是已测最 fp64 占优的 SVP NLC 点
   （frac 1.258），落在预注册邻包络 [0.93, 1.20] 之外**（butane
   1.073、pyridine 1.134、vitamin_c 0.972）：小内缘非单调
   （1.073@106 < 1.134@109 < 1.258@114 → 0.972@208），平滑阶梯假
   设在带缘被打破——预注册决策规则的「分子依赖」分支如期触发：
   如实记录，不重贴标签。
2. **规则修正（本单元交付物）：完整栈车道交叉在 114-318 AO 内部
   是分子依赖的，不是单一 AO 阈值**。vitamin_c（208，多官能团）
   auto 占优（0.972）而苯（114 SVP 1.258；222 TZVP 1.254）在**两
   种基组**下均 fp64 占优至少到 222。**§62 的「NLC ~(109,208]-537」
   读数过度概括了 vitamin_c 单点**；分子无关的表述为：**~114 以下
   全面 fp64 占优；114-318 分子依赖（苯类芳香族 ≥222 仍 fp64 占
   优、vitamin_c 类多官能团 ~208 起 auto 占优）；≥318 AO 的每个
   已测 NLC 格均 auto 占优**（0.929-0.983）。
3. **同分子基组数据：苯的完整栈 frac 对基组不敏感**（SVP 1.258 vs
   TZVP 1.254）——苯无基组翻转；LC-118 默认栈的 SVP→TZVP frac 位移
   （-0.12..-0.18）在完整栈上不构成本分子的翻转。
4. **跨栈 fp64 dE 在小端是分子特异的**：苯 2.9e-12 vs butane
   4.5e-11 / pyridine 5.8e-11（原子数相近）——LC-121/126 的
   dE~点对数趋势获得反例；所有值仍为核交换类（≤2.3e-10，低于门
   4 个量级以上），但非尺寸的单调函数。

判定 5/6 HIT：P1（复现，fp64 e_tot 至 1.2e-13）、P2（t_scf 5.96x
在带内）、P3（auto e2e 比 0.211 在 [0.15,0.45]）、P4（文档内门全
过，dE ≤ 1.3e-11、dg ≤ 1.7e-6；跨栈 dE 2.9e-12、dg 1.75e-6）、P6
（无回归，最大 doc2/doc1 比 0.21；峰值 ≤ 606 MiB）HIT；P5 信息性
MISS（上述）。诚实边界：单格；分子依赖表述基于 4 个 SVP 内点 +
苯/tzvp@222（LC-106 记录、上一会话、未复测）+ ≥318 三格；「芳香
族 fp64 占优」仅概括一个分子（pyridine 109 亦 fp64 占优但在区间
下方）；(114,318) 内无芳香族测点；每道 3 计时样本。测量-only；
默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc127-benzene114-complete-stack`
（10 文件 sha256 锚定）；effects-index 44 数据集（227 源全部校验
通过）。

## 64. LC-128 (2026-10-13): 芳香族 TZVP 内部点 —— 芳香性读数推广（3/3 芳香族两种基组下均 fp64 占优）；TZVP 的 VV10 墙小于 SVP 外推

LC-127 的规则修正建立在一个芳香分子（苯）对一个多官能团分子
（vitamin_c）之上。本单元在另外两个芳香杂环上检验芳香性读数：
furan/def2-tzvp（179 AO）与 pyridine/def2-tzvp（216 AO）——均为
development-分割、此前从未以 wb97m-v 在 TZVP 下运行——外加
furan/def2-svp（90 AO）把 SVP 阶梯向下延伸。6 份 VALID 文档、
36/36 重复收敛、无 fallback、峰值 170-902 MiB。运行时 06ec046
（cd6cdef→06ec046 仅文档差异已验证——LC-117 冻结记录 o65 为
furan/svp doc1 锚）。

| 格 | 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|---|
| furan/svp 90 | fp64 | 6.666→1.469（**4.54x**） | 1.005→0.349（2.88x） | 7.757→1.892（**4.10x**） | 0.995→**1.055** |
| furan/tzvp 179 | fp64 | 7.706→1.971（**3.91x**） | 1.490→0.454（3.28x） | 9.296→2.511（**3.70x**） | 1.040→**1.109** |
| pyridine/tzvp 216 | fp64 | 11.224→2.513（**4.47x**） | 2.072→0.498（4.16x） | 13.410→3.121（**4.30x**） | 0.945→**1.111** |

（每道每格循环全部不变：11/11/12，全部重复；auto 道栈 e2e 增益
3.47-4.10x。）

五项载荷性结论：

1. **P5 命中——芳香性读数推广**：furan/tzvp@179（1.109）与
   pyridine/tzvp@216（1.111）均 fp64 占优；连同苯（SVP 1.258 /
   TZVP 1.254）与 pyridine/SVP@109（1.134），**三个芳香分子在
   每个所测基组下（90-222 AO）均 fp64 占优**，vitamin_c（0.972
   @208）仍是唯一已测 auto 占优内部点——分子依赖段的芳香类读数
   现有 3 个分子支撑。通用车道选择规则：**~114 以下全分子
   fp64 占优，芳香族延伸至 ≥222；多官能团类（vitamin_c 类）
   ~208 起 auto 占优；≥318 AO 每个已测 NLC 格均 auto 占优**。
2. **三个同分子完整栈基组对全部在符号上基组不敏感**：苯
   1.258/1.254、furan 1.055/1.109、pyridine 1.134/1.111——完整栈
   车道排序在任何已测分子上不随基组翻转（LC-118 默认栈的
   SVP→TZVP 位移不改变芳香族在完整栈的排序）。
3. **TZVP 的 VV10 墙小于 SVP 外推——完整栈系列首个「帮得比预测
   少」MISS**：furan 3.91x / pyridine 4.47x fp64 t_scf 增益 vs
   基于 SVP 格（5.85x@106→7.04x@208）与 LC-120/121 锚的
   [5.0, 8.0] 带。机制：TZVP 每原子 AO 约多 1.6 倍而 NLC 网格点
   相近，VV10 在 t_scf 中占比更低，fp32-VV10 旋钮移除的份额更
   小；e2e 增益 3.70x/4.30x。
4. **P6 命中——无回归保持**：每道每格 e2e 增益 3.47-4.10x（最
   大 doc2/doc1 比 0.29）；峰值随 16384 上升（256→410、606→830、
   890→902 MiB），均远低于预算；furan/svp@90 把 ~114 以下
   fp64 占优模式向下延伸（frac 1.055；默认栈两道平价 0.995）。
5. **数值干净**：文档内门全过 6 份（dE ≤ 4.4e-11、dg ≤
   3.7e-6）；跨栈 fp64 dE ≤ 6.7e-12、max|dg| ≤ 3.1e-6——系列
   中最小跨栈位移（TZVP 格 NLC 点更少，与 LC-121 点对数机制
   一致）；循环全部不变。

精化后的完整栈 NLC frac 图（LC-126/127/128 后）：SVP 53→1.181、
67 UKS→1.224、90 furan→1.055、106 butane→1.073、109 pyridine→
1.134、114 benzene→1.258、**208 vitamin_c→0.972（唯一 auto 占优
内部点）**、318→0.983、326→0.937、537→0.929；TZVP 179 furan→
1.109、216 pyridine→1.111、222 benzene→1.254。

判定 12/14 HIT：P1×3（furan/svp 复现冻结 o65，e_tot 至 4.0e-12；
两个 TZVP doc1 基线 3/3 收敛）、P3×3（auto e2e 比 0.258/0.288/
0.274 在带内）、P4×3、P5、P6 HIT；P2×2 低侧 MISS = TZVP 墙发现
（如实记录，非失败）。诚实边界：新增 2 分子；芳香读数 3 分子
（其中 2 个杂环）、多官能团读数 1 分子；墙机制由增益推断、本
单元未独立剖析；(114,318) SVP 内部仍只测苯/vitamin_c；每道 3
计时样本；两个 TZVP doc1 基线为首次测量（无冻结锚）。测量
-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc128-aromatic-tzvp-interior`
（16 文件 sha256 锚定）；effects-index 45 数据集（236 源全部
校验通过）。

## 65. LC-129 (2026-10-13): 羰基/酸/酮类 TZVP 内部点 —— vitamin_c 为分子特异离群点；内部段读数锐化为「除 vitamin_c 外全部 fp64 占优」（14/14 全 HIT）

LC-126..128 后，中尺寸 NLC 段读数为：芳香族全部 fp64 占优（3 分
子、两种基组）vs vitamin_c（多官能团）唯一 auto 占优内部点——一个
单分子类。本单元在区间内再测三个非芳香家族：acetaldehyde/
def2-tzvp（117 AO，羰基）、acetic_acid/def2-tzvp（148 AO，酸——
最小 COOH 分子，vitamin_c 官能签名的微缩版）、acetone/def2-tzvp
（160 AO，酮）。均为 development-分割；三个 SVP 对应格为 LC-117
冻结格；TZVP 组合为首次测量。6 份 VALID 文档、36/36 重复收敛、
无 fallback、峰值 196-1024 MiB。运行时 d2d2447（cd6cdef→
d2d2447 仅文档差异已验证）。

| 格 | 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|---|
| acetaldehyde/tzvp 117 | fp64 | 4.712→1.284（**3.67x**） | 0.855→0.366（2.34x） | 5.656→1.741（**3.25x**） | 1.036→**1.074** |
| acetic_acid/tzvp 148 | fp64 | 6.281→1.735（**3.62x**） | 1.098→0.370（2.97x） | 7.439→2.206（**3.37x**） | 1.024→**1.053** |
| acetone/tzvp 160 | fp64 | 7.539→1.780（**4.24x**） | 1.255→0.427（2.94x） | 8.893→2.309（**3.85x**） | 1.039→**1.069** |

（每道每格循环全部不变：11/12/11；auto 道栈 e2e 增益
3.13-3.85x。）

五项载荷性结论：

1. **P5 命中——vitamin_c 为分子特异离群点**：三个羰基/酸/酮内部
   点全部 fp64 占优（1.074/1.053/1.069）。最锐利的检验通过：
   acetic_acid（最小 COOH 分子）fp64 占优——vitamin_c 的 auto
   占优读数（0.972@208）**不是官能团类属性**。内部段读数锐化：
   **90-222 AO 双基组下每个已测分子均 fp64 占优，唯 vitamin_c
   例外；auto 占优 = vitamin_c（~208）与 ≥318 的每个 NLC 格**。
2. **六个同分子基组对全部在符号上基组不敏感**：苯 1.258/1.254、
   呋喃 1.055/1.109、吡啶 1.134/1.111、乙醛 1.166/1.074、乙酸
   1.088/1.053、丙酮 0.988/1.069——丙酮 SVP 默认栈 frac 为平价
   0.988 而完整栈 TZVP 读 1.069，无任何符号翻转：完整栈车道排序
   在六个已测分子上均不随基组翻转。
3. **14/14 全 HIT——完整栈系列首个全命中**：每条预测带（增益、
   auto 比、循环、数值、安全、内存）全中。LC-128 的 TZVP 墙读数
   精确迁移：fp64 t_scf 增益 3.62-4.24x（带 [3.5,5.5] 由 LC-128
   锚构建），小于 SVP 格的 5.85-7.04x。
4. **P6 命中——无回归保持**：每道每格 e2e 增益 3.13-3.85x（最
   大 doc2/doc1 比 0.32）；峰值随 16384 上升（292→630、462→742、
   488→1024 MiB），均远低于预算。
5. **数值干净**：文档内门全过 6 份（dE ≤ 1.3e-12、dg ≤
   5.6e-6——acetic_acid 的 SVP 裕度族完整栈下 dg 反缩 10 倍
   5.64e-6→5.53e-7，核共享类）；跨栈 fp64 dE ≤ 4.2e-11、
   max|dg| ≤ 6.4e-6（核交换类）；循环全部不变。

完整栈 NLC frac 图（LC-126..129 后，10 分子）：SVP 53→1.181、67
UKS→1.224、90 呋喃→1.055、106 丁烷→1.073、109 吡啶→1.134、114
苯→1.258、**208 vitamin_c→0.972（318 以下唯一 auto 占优点）**、
318→0.983、326→0.937、537→0.929；TZVP 117 乙醛→1.074、148 乙酸
→1.053、160 丙酮→1.069、179 呋喃→1.109、216 吡啶→1.111、222 苯
→1.254。

判定 14/14 HIT（P1×3/P2×3/P3×3/P4×3/P5/P6 全命中）。诚实边
界：三个新分子各一家族；「vitamin_c 离群」建立在 3 芳香 + 3 羰
基类 vs 1 多官能团之上（区间内无第二个多官能团分子——bisphenol
318 读平价 0.983、最近但不决定性）；三个 TZVP doc1 基线为首次测
量；vitamin_c 离群机制未诊断（候选：其 208-AO 尺寸恰处 auto 道
SCF 缓冲最大的区间中部——记为开放问题而非断言）。测量-only；
默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc129-carbonyl-acid-tzvp-interior`
（17 文件 sha256 锚定）；effects-index 46 数据集（245 源全部校验
通过）。

## 66. LC-130 (2026-10-13): 新分子中等尺寸的非 NLC 方法完整栈 —— 三方法全落平价级（1.012-1.050，低于两侧带内插值）；LC-117 的 m062x 小格「auto 赢」被判定为亚秒计时噪声

完整栈组成轴延伸：LC-124 的小带（48-92，旧小分子）与 LC-106 的旧
分子中等带（222-298）之外，在一个新分子（pyridine，dev-分割）上
测全部三个非 NLC 方法——b3lyp/svp 109、m062x/svp 109、r2scan3c/
mtzvpp 144（组成受控单分子）。6 份 VALID 文档、36/36 重复收敛、
无 fallback、峰值 212-620 MiB。运行时 06cd86a（cd6cdef→06cd86a
仅文档差异已验证——LC-117 冻结记录 o71/o38/o32 为 doc1 锚）。

| 格 | 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|---|
| b3lyp/pyridine 109 | fp64 | 0.948→0.778（1.22x） | 0.405→0.223（1.82x） | 1.432→1.085（**1.32x**） | 1.075→**1.050** |
| m062x/pyridine 109 | fp64 | 1.399→1.008（1.39x） | 0.444→0.240（1.85x） | 1.925→1.325（**1.45x**） | 1.065→**1.012** |
| r2scan3c/pyridine 144 | fp64 | 1.294→0.865（1.50x） | 0.392→0.189（2.08x） | 1.812→1.164（**1.56x**） | 1.154→**1.016** |

（每道每格循环全部不变：11/11/10；auto 道栈 e2e 增益 1.35-1.77x；
跨栈 fp64 dE ≤ 1.7e-13——非 NLC 格无 VV10-fp32 扰动，实为零。）

五项载荷性结论：

1. **三方法全落平价级（1.012/1.050/1.016）**——已测最低非 NLC
   完整栈 frac，**低于**小带（1.06-1.18@48-92）与旧分子中等带
   （1.25-1.27@222-298）之间的内插值：非 NLC 完整栈 frac 对尺寸
   非单调，在该分子上于 109-144 降至平价。实践读数：非 NLC 方
   法中等尺寸下车道选择几乎不重要（≤5%），而旋钮栈对两道都有
   益（1.32-1.77x）。
2. **LC-117 的 m062x/pyridine 默认栈 auto 赢（frac 0.903）未在本
   会话复现（doc1 1.065）**——两道各自都在 ±15% 计时带内，但 ~2 s
   运行上 ~10% 的 frac 会在会话间翻号：**LC-117 的 m062x 小格
   auto 赢（≤109 AO 的 ~0.90 frac）为噪声级**，不得在无复测会话
   确认下被引作默认栈 auto 赢；LC-119 的「auto 赢 ≥208 AO」规则
   不受影响（其 ≥208 格运行 10-40 s，frac 稳定）。冻结数字本身
   不变、正确——被修订的是读数。
3. **P6 命中——无回归保持**：每道每格 e2e 增益 1.32-1.77x；峰值
   随 16384 上升（262→458、276→524、212→620 MiB），均远低于预
   算；36/36 收敛；循环全不变。
4. **栈增益按组成排序如预注册**：b3lyp 1.32x（最弱旋钮方法）<
   m062x 1.45x（tau headtail）< r2scan3c 1.56x（tau 重 J-only，
   t_grad 2.08x）——P2×3 HIT。
5. **auto 道 blocksize 响应随方法权重增强**：auto e2e 比
   0.739/0.654/0.566——r2scan3c 低于预注册带 [0.60,0.85]（P3
   低侧 MISS，帮得比预测多）：16384 在 3c MGGA 列车上的收益大于
   普通杂化。

判定 11/14 HIT：P1 m062x/r2scan3c 精确复现（e_tot ≤ 3.3e-13）、
b3lyp t_scf 计时带 MISS（+18.7%，亚秒噪声；e2e 在带内、循环精
确、e_tot 至 3e-14）；P2×3、P4×3（文档内门全过，dE ≤ 1.5e-12、
dg 完整栈下缩 4-12 倍核共享类；r2scan3c D4+gCP 两态一致）HIT；
P3 r2scan3c 低侧 MISS；P5 信息性 MISS（平价级读数 + 噪声级发
现）；P6 HIT。诚实边界：单分子三方法（平价凹陷为单分子读数）；
亚秒至 2 s 道时间携带 ±5-10% 会话 frac 噪声（m062x 翻转所证
——平价结论对噪声稳健，更细排序不稳健）；r2scan3c 144 mtzvpp
与 109 SVP 不直接可比。测量-only；默认与门槛不变；全部旋钮
opt-in。证据：
`tasks/issue-9/2026-10-13-lc130-nonnlc-newmolecule-moderate`
（17 文件 sha256 锚定）；effects-index 47 数据集（254 源全部
校验通过）。

## 67. LC-131 (2026-10-13): 非 NLC 平价凹陷推广到第二个分子 —— 丁烷×三方法完整栈全落平价级（1.003-1.070，14/14 HIT，非 NLC 轴首个全中）

LC-130 诚实边界「单分子×三方法（平价凹陷为单分子读数）」的直接
后续：在平价凹陷带的第二个非 NLC 分子——丁烷（saturated，
dev-分割）——测全部三个非 NLC 方法：b3lyp/svp 106、m062x/svp 106、
r2scan3c/mtzvpp 126。与 LC-126（丁烷 wb97mv 1.073、pyridine
wb97mv 1.134）和 LC-130（pyridine×三方法 1.012/1.050/1.016）合
并，106-144 AO 带成为 2 分子×4 方法网格。运行时 d6d5a2f（树干
净；cd6cdef→d6d5a2f 仅文档差异链——LC-117 冻结记录 o59/o41/o14
为 doc1 锚）。6 份 VALID 文档、36/36 重复收敛、无 fallback、峰值
188-702 MiB；驱动墙 1.6 分钟。

| 格 | 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|---|
| b3lyp/丁烷 106 | fp64 | 0.806→0.535 (1.51x) | 0.369→0.212 (1.74x) | 1.240→0.838 | **1.48x** | 1.151→**1.003** |
| b3lyp/丁烷 106 | auto | 0.988→0.575 | 0.376→0.219 | 1.428→0.840 | 1.70x | |
| m062x/丁烷 106 | fp64 | 1.199→0.678 (1.77x) | 0.434→0.247 (1.76x) | 1.707→0.997 | **1.71x** | 0.955→**1.066** |
| m062x/丁烷 106 | auto | 1.186→0.770 | 0.379→0.245 | 1.630→1.063 | 1.53x | |
| r2scan3c/丁烷 126 | fp64 | 1.236→0.776 (1.59x) | 0.385→0.217 (1.78x) | 1.680→1.072 | **1.57x** | 1.112→**1.070** |
| r2scan3c/丁烷 126 | auto | 1.375→0.832 | 0.414→0.235 | 1.869→1.148 | 1.63x | |

（循环：b3lyp 9/9、m062x 8/8 两态不变；r2scan3c auto 9→10——
blocksize 响应族，在预注册 ±1 带内，fp64 9/9；auto e2e 仍
1.63x 增益。跨栈 fp64 dE ≤ 1.14e-13——m062x 与 r2scan3c 按位精
确（0.0）。）

**头部发现**：

1. **三格丁烷完整栈 frac 全落平价级（1.003/1.066/1.070）**——
   LC-130 的非 NLC 平价凹陷在 106-144 AO 推广到第二个分子：这是
   **带性质，不是 pyridine 性质**。106-144 带现为 2 分子×4 方法
   网格：丁烷 1.003/1.066/1.070/1.073（b3lyp/m062x/r2scan3c/
   wb97mv）、pyridine 1.012/1.050/1.016/1.134——八格全部在
   [1.00,1.14]，迄今测得的最紧完整栈平价簇。实用读数：非 NLC
   方法在中等尺寸（106-144 AO）下车道选择是抛硬币，而旋钮栈使
   两道皆受益（本单元 fp64 e2e 1.48-1.71x）。
2. **14/14 HIT——非 NLC 完整栈轴的首个全中**（LC-129 的全中是
   NLC TZVP 内点）。P1 复现精确（循环 9/8/9 精确、e_tot ≤
   3.9e-13、全部计时带内）；P2 组成排序增益；P3 auto blocksize
   响应 0.589/0.652/0.614 全在 [0.50,0.85]；P4 门；P5 平价带头
   条；P6 安全。
3. **m062x/丁烷默认 frac 会话内再翻（冻结 1.106→doc1 0.955），
   LC-130 噪声级读数的第三次确认**：两道各自都在 ±15% 计时带内
   （fp64 e2e 1.707 vs 冻结 1.697；auto 1.630 vs 冻结 1.877 带
   缘），但 ~15% 的亚 2 秒 frac 会话间翻号。完整栈 frac（1.066，
   道 ~1 s）更稳，但被引读数是平价带，不是排序——预注册噪声警
   告适用。
4. **栈增益组成排序：丁烷上 b3lyp 1.48x < r2scan3c 1.57x <
   m062x 1.71x**——tau 类内部排序相对 pyridine（m062x 1.45 <
   r2scan3c 1.56）**翻转**：tau 类内（1.45-1.71x）细排序为分子
   依赖；b3lyp-最弱排序两分子稳定（2/2）。
5. **核共享 dg 缩减复现至 6 格深**：auto-vs-fp64 max|dg| 完整栈
   下缩 5.6x/4.3x/2.7x（b3lyp 1.72e-6→3.07e-7、m062x
   1.91e-6→4.43e-7、r2scan3c 1.48e-6→5.43e-7）；r2scan3c
   D4+gCP 两态按位一致。auto e_shift ≤ 3.8e-12（blocksize 为唯
   一 engaged auto 旋钮）。

诚实边界：凹陷带内 2 分子（均为 dev-分割、亚 2.2 s 默认道，
±5-10% 会话 frac 噪声——被引结论是八格平价簇 [1.00,1.14]，不
是更细排序）；凹陷在 144-222 AO 间如何回升至 1.25 仍未在非 NLC
方法上测得；r2scan3c auto +1 循环为真实 blocksize 响应（在带
内记录，不改变 e2e 结论）。测量-only；默认与门槛不变；全部旋
钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc131-nonnlc-butane-complete-stack`
（31 文件 sha256 锚定）；effects-index 48 数据集（源全部校验
通过）。

## 68. LC-132 (2026-10-13): 208 AO 回升带顶点 —— m062x/vitamin_c 默认 auto 赢在完整栈被抹除（0.841→1.127）；凹陷界于 ~144-208；vitamin_c 的 auto 占优读数为 VV10 特异

LC-131 诚实边界「144-222 AO 间凹陷如何回升未测」的直接后续：在回升
区间顶点测 vitamin_c/svp 208 AO × b3lyp + m062x（均 dev-分割，
LC-117 冻结锚 o69/o43）。运行时 21263d3（树干净；对 d6d5a2f 仅文
档差异 = LC-131 两份文档）。4 份 VALID 文档、24/24 重复收敛、无
fallback、峰值 1066-1388 MiB；驱动墙 1.8 分钟。

| 格 | 道 | t_scf | t_grad | e2e（栈增益） | frac 默认→完整 |
|---|---|---|---|---|---|
| b3lyp/vitc 208 | fp64 | 1.825→1.219 (1.50x) | 1.029→0.326 (**3.15x**) | 2.993→1.681 | **1.78x** | 0.922→**1.114** |
| b3lyp/vitc 208 | auto | 1.965→1.362 | 0.631→0.396 | 2.759→1.874 | 1.47x | |
| m062x/vitc 208 | fp64 | 2.820→1.698 (1.66x) | 1.178→0.310 (**3.80x**) | 4.072→2.144 | **1.90x** | 0.841→**1.127** |
| m062x/vitc 208 | auto | 2.625→1.890 | 0.638→0.374 | 3.422→2.416 | 1.42x | |

（循环：fp64 13/13 两格两态；b3lyp auto 13→14（+1，blocksize 族，
带内）；m062x auto 13/13 不变。跨栈 fp64 dE ≤ 4.6e-13；跨栈 fp64
max|dg| 8.9e-6 / 1.06e-5——fp64 道梯度旋钮的核交换类，在预注册
1.1e-5 诊断边沿，按实测报告；严格参考仍为旋钮关闭的 doc1 记录。）

**头部发现**：

1. **m062x/vitamin_c 默认 auto 赢（冻结 0.859，本会话 0.841 = 复
   现，~4 s 道稳定级）在完整栈被抹除：frac 1.127**——LC-106 的抹
   除（m062x/tzvp 0.995→1.273 @222-298）下延至 208 AO；结合
   LC-130/131 的 106-144 平价簇（八格 [1.00,1.14]）与本单元
   1.114/1.127，**非 NLC 完整栈凹陷界于 ~144-208 并在 208 回升至
   222-298 带（1.25-1.27）方向**——非 NLC 轴的完整栈曲面现已在
   每个尺寸带有测（48-92 / 106-144 / 208 / 222-298 / 687）。
2. **vitamin_c 的 auto 占优读数为 VV10 特异，非分子整体性质**：
   同一分子同一完整栈下 wb97mv/svp 0.972（auto 占优，VV10 墙）
   vs b3lyp 1.114 / m062x 1.127（fp64 占优，无 VV10）vs
   r2scan3c/mtzvpp@298 1.253——无 NLC 时 vitamin_c 加入回升带。
   LC-129 的「分子特异离群点」锐化为「离群机制是其工作负载的
   VV10 份额」。
3. **梯度旋钮栈增益随尺寸强增长**：fp64 t_grad 3.15x/3.80x @208
   vs 1.74-1.82x @106-144（ip1_f32 + grad_grid_f32 + ip1_fused 效
   应随 df_jk/grid 占梯度份额放大）；b3lyp e2e 1.78x 超出预注册
   [1.15,1.60]（P2 高侧 MISS，帮得比预测多——复现族；该格上梯度
   相位主导增益，b3lyp-vs-tau 区分大体消失）。
4. **核共享 dg 缩减迄今最大**：文档内 auto-vs-fp64 max|dg| 完整
   栈下缩 5.7 倍（b3lyp 8.84e-6→1.55e-6）与 11.7 倍（m062x
   1.06e-5→9.05e-7——LC-117 非 NLC 矩阵中最大默认 dg 降至 1e-6
   以下）；auto e_shift ≤ 9.3e-12。

判定 9/10 HIT：P1×2 精确（循环 13/13、e_tot ≤ 4.6e-13、计时带
全过）；P2 m062x HIT（1.90x 恰在带上沿）、b3lyp 高侧 MISS；P3×2
HIT（0.679/0.706）；P4×2 HIT；P5 HIT（抹除 + 回升双读数）；P6
HIT（每道 1.42-1.90x、峰值 ≪ 预算、24/24 收敛）。诚实边界：208
为单分子×双方法（144-208 内点未测，凹陷上缘为括号内非定位）；
~3-4.5 s 道携带 ±5% 会话 frac 噪声（+0.29 的抹除与 +0.19 的
b3lyp 回升远在其外；1.114 vs 1.127 不被引）；VV10 特异读数为单
分子上一 NLC 格对两非 NLC 格的对比。测量-only；默认与门槛不
变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc132-vitc-nonnlc-riseback`
（23 文件 sha256 锚定）；effects-index 49 数据集（源全部校验通
过）。

## 69. LC-133 (2026-10-13): 完整 5 旋钮栈在 m062x/TZVP 全轴 —— 六格全落平价簇 [0.98,1.12]；pyridine@216 默认 auto 赢（0.753）被抹除；首个非 NLC 同分子基组对；auto blocksize 响应在 TZVP 系统性减弱

LC-132 诚实边界「144-208 内点未测（TZVP 轴）」的直接后续：在
m062x/def2-tzvp 全轴测完整栈——acetaldehyde 117、ethoxy 123
（UKS spin 1）、butadiene 160、furan 179、butane 184、pyridine 216
（全部 dev-分割，LC-118 冻结锚）。运行时 5a98f43（树干净；
cd6cdef→5a98f43 仅文档差异链）。12 份 VALID 文档、72/72 重复收敛、
无 fallback、峰值 198-894 MiB；驱动墙 5.5 分钟。

| 格 | fp64 e2e 默→完（增益） | t_grad 增益 | auto 比 | frac 默→完 | 循环 |
|---|---|---|---|---|---|
| acetaldehyde 117 | 1.380→1.186 (**1.16x**) | 1.52x | 0.804 | 1.072→**1.002** | 11→11 |
| ethoxy UKS 123 | 2.513→1.675 (**1.50x**) | 1.37x | 0.606 | 1.077→**0.979** | 12→12 |
| butadiene 160 | 1.571→1.153 (**1.36x**) | 1.85x | 0.802 | 0.957→**1.045** | 9→9 |
| furan 179 | 2.281→1.402 (**1.63x**) | 2.66x | 0.745 | 0.921→**1.116** | 11→11 |
| butane 184 | 2.222→1.185 (**1.88x**) | 2.41x | 0.659 | 0.841→**1.039** | 9→9 |
| pyridine 216 | 2.469→1.664 (**1.48x**) | 2.35x | 0.866 | 0.806→**1.036** | 11→11 |

（循环每道每格两态全不变——迄今最干净的栈不变性；auto e_shift ≤
1.1e-12；跨栈 fp64 dE ≤ 1.7e-13、max|dg| ≤ 2.6e-6；文档内 dg 完整
栈下每格缩 4-13 倍，核共享类非 NLC 计 12 格深。）

**头部发现**：

1. **六格全落平价簇 [0.979,1.116]**——任何方法×基组轴上测得的最
   紧完整栈簇（紧于 106-144 SVP 网格的 [1.00,1.14]）：非 NLC 凹
   陷在 TZVP 轴**延伸穿过 216**；冻结 LC-106 benzene/m062x/tzvp
   @222（1.273）将回升括号在 216-222 间——分子依赖或陡峭回升，
   记为张力未解（跨分子、跨会话）。
2. **pyridine/m062x/tzvp 默认 auto 赢（冻结 0.753、本会话 0.806 =
   复现，222 以下最强）在完整栈被抹除（frac 1.036）**——抹除读
   数现跨双基组 3/3（LC-106 benzene@222、LC-132 vitc@208 SVP、
   LC-133 pyridine@216 TZVP）；**687 以下无稳定级非 NLC 默认
   auto 赢在完整栈存活**。
3. **首个非 NLC 同分子完整栈基组对**：butane SVP 1.066 / TZVP
   1.039 与 pyridine SVP 1.012 / TZVP 1.036 **点稳定**（差 ≤
   0.027）——LC-129 基组符号不变性读数推广到非 NLC 闭壳层；
   **ethoxy UKS 对不是点稳定**（SVP 1.247 / TZVP 0.979，跨平价
   0.27 移动）：非 NLC 对为**平价带不变**，UKS 壳层轴对 frac 的
   移动大于基组轴。
4. **auto blocksize 响应在 TZVP 系统性弱于 SVP**：auto e2e 比
   0.804/0.802/0.866 vs SVP m062x 锚 0.652-0.706——TZVP 每原子
   ~1.6 倍 AO 稀释 16384 blocksize 在网格受限 auto 道上的收益
   （LC-128 VV10 墙的 AO 稀释机制在同一方向）。
5. **fp64 栈增益分子依赖而非尺寸排序**（1.16-1.88x 散布；184-AO
   butane 最大而 216-AO pyridine 最小；t_scf 1.07-1.80 与 t_grad
   1.37-2.66 独立散布）——尺寸排序带（从 SVP 锚外推）过度预测。
6. **acetaldehyde doc1 计时漂移 -20%/-26%**（fp64 e2e 1.380 vs 冻
   结 1.727；e_tot 精确至 1.7e-13、循环精确）——噪声级族扩展：
   亚 2 秒绝对道时间会话间漂移至 ~26%，超出 ±15% 复现带；P1
   MISS 诚实记录，完整 frac 平价读数不受影响、默认 frac
   （1.072 vs 冻结 0.929）为又一次噪声级翻号。

判定 19/26 HIT：P1 5 HIT + 1 MISS（漂移）；P2 3 HIT + 3 低侧
MISS（分子依赖增益）；P3 3 HIT + 3 高侧 MISS（TZVP blocksize 稀
释发现）；P4×6 HIT；P5 HIT（抹除 + 三基组对同侧 + 内点带）；P6
HIT（每道 1.16-1.88x fp64 / 0.61-0.87 auto 增益、峰值 ≪ 预算、
72/72 收敛）。诚实边界：单方法单基组（平价簇读数为 tau-MGGA
类）；216-vs-222 张力跨分子跨会话（括号回升，不定位边缘）；
ethoxy UKS 对跨会话（0.27 量级在噪声外但 UKS-轴归因被单对混
淆）；亚 2 秒格默认 frac 噪声级。测量-only；默认与门槛不变；
全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc133-m062x-tzvp-axis`
（55 文件 sha256 锚定）；effects-index 50 数据集（源全部校验通
过）。

## 70. LC-134 (2026-10-13): b3lyp/TZVP 轴完整栈 —— 平价簇读数跨方法类大部分迁移（六格之五 [1.02,1.18]）；butadiene 1.240 例外（共轭分子在凹陷区间内达到旧分子中等带读数）；b3lyp 在 5/6 格上高于 m062x；blocksize 稀释轴通用

LC-133 诚实边界「单方法单基组；b3lyp/TZVP 轴为候选下一单元」的直接
后续：在最弱旋钮方法上测同一 TZVP 轴——acetaldehyde 117、ethoxy
123（UKS spin 1）、butadiene 160、furan 179、butane 184、pyridine
216（全部 dev-分割；doc1 为首次测量基线，LC-129 先例——本单元冻
结该轴默认锚）。与 LC-133 的 m062x 格构成同分子跨方法对。运行时
b15c2ab（树干净）。12 份 VALID 文档、72/72 重复收敛、无 fallback、
峰值 194-880 MiB；驱动墙 5.7 分钟。

| 格 | fp64 e2e 默→完（增益） | t_grad 增益 | auto 比 | frac 默→完 | m062x 对应 | 循环 |
|---|---|---|---|---|---|---|
| acetaldehyde 117 | 1.292→0.886 (**1.46x**) | 1.65x | 0.662 | 1.056→**1.019** | 1.002 | 11→11 |
| ethoxy UKS 123 | 1.822→1.230 (**1.48x**) | 1.72x | 0.828 | 0.961→**1.178** | 0.979 | 12→12 |
| butadiene 160 | 1.235→0.797 (**1.55x**) | 1.71x | 0.657 | 1.217→**1.240** | 1.045 | 10→10 |
| furan 179 | 1.651→0.990 (**1.67x**) | 2.29x | 0.723 | 0.916→**1.104** | 1.116 | 10→10 |
| butane 184 | 1.519→1.032 (**1.47x**) | 1.90x | 0.718 | 1.043→**1.102** | 1.039 | 9→9 |
| pyridine 216 | 1.985→1.275 (**1.56x**) | 2.79x | 0.807 | 0.893→**1.122** | 1.036 | 11→11 |

（循环每道每格两态不变；auto e_shift ≤ 1.9e-12；跨栈 fp64 dE ≤
2.3e-13（两格按位）、max|dg| ≤ 2.5e-6；文档内 dg 每格缩 3.4-11 倍，
核共享类非 NLC 计 18 格深。）

**头部发现**：

1. **平价簇读数跨方法类大部分迁移：六格之五落 [1.019,1.178]**——
   完整栈平价区为双方法类的基组轴性质。例外：**butadiene 落
   1.240**，超预注册带 [0.90,1.20] 且进入 LC-106 旧分子中等带读数
   （1.25-1.27）——共轭分子为首个在凹陷区间内达到中等带读数的
   cohort 分子；其完整栈移动仅 +0.02（六格最小），例外主要为默认
   级（默认 frac 1.217 已是该轴最高）。
2. **b3lyp 在 5/6 格上高于 m062x**（中位差 +0.07；acetaldehyde 低
   0.017）——最弱旋钮方法在同分子同基组上读数更 fp64 占优，与
   headtail 类预测方向相反：b3lyp 的 fp64 栈增益同样大（1.46-1.67x，
   散布比 m062x 的 1.16-1.88x 更紧）而 auto blocksize 响应不强于
   ——梯度路径的 ip1/grad-grid 旋钮方法无关。
3. **TZVP blocksize 稀释读数在 b3lyp 上复现**（P3×6 HIT；auto 比
   0.657-0.828，UKS/大格处弱端）——稀释为**轴通用**（TZVP 每原子
   AO），非 tau 特异。
4. **UKS ethoxy 对再次跨平价移动**（默认 0.961 → 完整 1.178，
   0.22 移动，镜像其 SVP 对 LC-125）——壳层轴效应跨方法/基组
   4/4。
5. **pyridine/b3lyp/tzvp 默认 0.893 → 完整 1.122**：第四个 TZVP
   默认 auto 赢在完整栈被抹除的分子；**两方法上均无 TZVP 默认
   auto 赢在 222 以下于完整栈存活**。
6. **首次测量 doc1 基线冻结**：该轴默认计时/循环/能量记录在案
   （b3lyp/tzvp SCF 循环 9-12；e2e 1.235-1.985 s）。

判定 25/26 HIT：P1×6（首次测量锚）、P2×6（增益 1.46-1.67x 全带
内）、P3×6（稀释迁移）、P4×6（门全过）HIT；P5 信息性 MISS
（butadiene 1.240 出带 + ethoxy 跨方法差 0.199 超 0.15 子检）；
P6 HIT（每道 1.46-1.67x fp64 / 0.66-0.83 auto 增益、峰值 ≪ 预
算、72/72 收敛）。诚实边界：butadiene 例外为单分子（共轭类数
据点，非类规则）；doc1 基线首次测量（无跨会话复现）；跨方法差
比较相邻会话（~1 小时间隔，±5-10% 噪声——5/6-above 系统性为被
引结论，逐格差不引）；ethoxy UKS 移动跨会话/方法（无 UKS 轴闭
壳层对照）。测量-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc134-b3lyp-tzvp-axis`
（55 文件 sha256 锚定）；effects-index 51 数据集（源全部校验通
过）。

## 71. LC-135 (2026-10-13): wb97mv/TZVP 轴完成 —— LC-134 的 butadiene 例外不迁移到 NLC 方法（1.082，方法 1/3）；内部簇全 fp64 占优 [1.03,1.11]（任何方法×基组轴最紧）；butane 带该轴最大 VV10 墙

LC-134 遗留问题（共轭例外是否迁移）的直接后续：最后三个未测
wb97mv/def2-tzvp 格——butadiene 160、butane 184、ethoxy 123
（UKS spin 1）——wb97mv/TZVP 轴就此完成（9 格）。doc1 基线为首
次测量 wb97mv/tzvp 组合（LC-129 先例——本单元冻结该轴默认锚）。
运行时 c8b2907（树干净）。6 份 VALID 文档、36/36 重复收敛、无
fallback、峰值 290-880 MiB；驱动墙 5.7 分钟。

| 格 | fp64 e2e 默→完（增益） | t_scf 增益 | auto 比 | frac 默→完 | 循环 |
|---|---|---|---|---|---|
| butadiene 160 | 8.538→2.177 (**3.92x**) | 4.26x | 0.271 | 1.019→**1.082** | 10→10 |
| butane 184 | 12.285→2.532 (**4.85x**) | 5.35x | 0.221 | 1.002→**1.075** | 9→9 |
| ethoxy UKS 123 | 9.509→2.913 (**3.26x**) | 3.62x | 0.307 | 1.033→**1.034** | 14→14 |

（循环每道每格两态相同（10/9/14）；auto e_shift ≤ 1.3e-11；跨栈
fp64 dE 2.3-6.0e-11（VV10-fp32 核交换类）、max|dg| ≤ 2.3e-6；
文档内 dg 每格缩 3.0-4.7 倍——核共享类：1.79e-6→5.89e-7、
1.10e-6→3.48e-7、2.49e-6→5.30e-7。）

**头部发现**：

1. **LC-134 的 butadiene 例外不迁移到 NLC 方法**：wb97mv/tzvp
   butadiene 落 **1.082**，正处于该轴内部簇中心。共轭分子读数在
   同分子同基组完整栈上现为 **方法 1/3**（b3lyp 1.240、m062x
   1.045、wb97mv 1.082）——按预注册判定规则降级为 **b3lyp 默认级
   伪象**（其默认 frac 1.217 已是该轴最高、完整栈移动为六格最
   小）。无共轭类规则存活。
2. **wb97mv/TZVP 轴完成——9 格全部内部格 fp64 占优**：
   acetaldehyde 1.074、acetic_acid 1.053、acetone 1.069、
   butadiene 1.082、butane 1.075、furan 1.109、pyridine 1.111、
   ethoxy 1.034（benzene@222 1.254 冻结 LC-106）。内部簇
   [1.03,1.11] 为**任何方法×基组轴上最紧的完整栈簇**（m062x/TZVP
   [0.979,1.116]、b3lyp/TZVP [1.019,1.240]）；NLC 方法在 TZVP 上
   222 以下无 auto 占优格——SVP 轴的 auto 占优内部读数
   （vitamin_c 0.972 @208）无 TZVP 对应；auto 占优 NLC 区为
   **SVP+大 NLC（≥318 AO）性质**。
3. **ethoxy UKS 壳层轴格网闭至 6 格**（SVP 1.224/1.247/1.178、
   TZVP m062x 0.979 / wb97mv 1.034 / b3lyp 1.178）：壳层轴跨平价
   移动 frac 但在 TZVP 上**无一致方向**——LC-134 的 4/4 fp64 方向
   读数不扩展为 5/5；壳层轴是 frac **噪声放大器**（同分子同基组
   方法间散布 0.21）而非 fp64 定向器。
4. **butane 带该轴最大 VV10 墙**：默认 fp64 e2e 12.285 s（任何已
   测 TZVP 格最大默认道）；fp32-VV10 旋钮以 5.35x t_scf / 4.85x
   e2e 移除之——P2 helps-more-than-predicted MISS（良性方向）。
   连同 LC-121（6.42x @537）与 LC-128（3.91-4.47x），TZVP 墙梯
   123-216 AO 读 3.6-5.4x——TZVP 稀释是 VV10 份额效应，非硬上
   限。
5. **首次测量 doc1 基线冻结**：该轴默认计时/循环/能量记录在案
   （循环 9/10/14；e2e 8.5-12.3 s）。

判定 13/14 HIT：P1×3（首测锚；ethoxy UKS 干净——methoxy 仍为唯
一分裂格）、P2 2 HIT + 1 MISS（butane 4.85x/5.35x 超上带——该轴
最大墙）、P3×3（auto 比 0.221-0.307 带内、循环逐样本相同、
|e_shift| ≤ 1.3e-11）、P4×3（门全过；跨栈 fp64 dE ≤ 6.0e-11、
max|dg| ≤ 2.3e-6）、P5 HIT（butadiene 1.082 ∈ [1.00,1.20]、
butane 1.075 ∈ [0.95,1.20]、ethoxy 1.034 ∈ [0.90,1.30]）、P6
HIT（每道 3.26-4.85x fp64 / 0.22-0.31 auto 增益、峰值 ≪ 预算、
36/36 收敛）。诚实边界：共轭降级基于单分子单基组 3 个方法格
（"方法 1/3" 为准确表述，无更广共轭类断言——cohort 无第二个共
轭 dev 分子）；「TZVP 222 以下无 auto 占优」闭壳层限定（m062x
ethoxy UKS 0.979 跨到平价下）；doc1 基线首测（无跨会话复现）；
默认 frac（1.002-1.033）带 <2.5 s 噪声类——完整栈移动
（+0.06/+0.07/+0.00）为被引结论；ethoxy 跨方法比较跨 4 个会话
（±5-10%）。测量-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc135-wb97mv-tzvp-axis-completion`
（31 文件 sha256 锚定）；effects-index 52 数据集（源全部校验通
过）。

## 72. LC-136 (2026-10-13): r2scan3c/mtzvpp 内部轴完成（最后一个方法轴内部缺口闭合）—— 8 个内部格全 fp64 占优 [1.016,1.318]、3c 方法无平价凹陷；benzene-vs-pyridine 同尺寸 144 AO 最大 frac 散布 0.28；LC-135 的 butadiene 降级须修订（方法格 2/4）

LC-135 遗留（「最后一未测方法轴内部」）的直接后续：5 个 dev 格
——methanol 63、acetonitrile 77、ethoxy 87（UKS spin 1）、
butadiene 106、benzene 144——全部 r2scan3c/def2-mtzvpp（3c 规定
复合基组）。doc1 默认控制复现冻结 LC-117 锚（P1）。运行时
19ae0cd（树干净，LC-135 文档提交）。10 份 VALID 文档、60/60 重复
收敛、无 fallback、峰值 176-620 MiB；驱动墙 8.5 分钟。

| 格 | fp64 e2e 默→完（增益） | t_grad 增益 | auto 比 | frac 默→完 | 循环 fp64/auto |
|---|---|---|---|---|---|
| methanol 63 | 1.089→0.750 (**1.45x**) | 1.57x | 0.781 | 1.164→**1.318** | 9/9 → 9/9 |
| acetonitrile 77 | 1.199→0.860 (**1.39x**) | 1.26x | 0.680 | 1.143→**1.084** | 10/10 → 10/10 |
| ethoxy UKS 87 | 2.282→1.564 (**1.46x**) | 1.29x | 0.723 | 1.050→**1.107** | 12/12 → 12/12 |
| butadiene 106 | 1.487→0.842 (**1.77x**) | 1.56x | 0.588 | 1.170→**1.215** | 9/9 → 9/10 |
| benzene 144 | 1.526→1.101 (**1.39x**) | 1.83x | 0.855 | 1.095→**1.298** | 8/10 → 8/10 |

（auto 循环移动仅 butadiene +1（9→10，blocksize 族，预注册 ±1 带
内）；其余每道每格两态相同；auto e_shift ≤ 1.4e-12；跨栈 fp64 dE
≤ 7.3e-12（ethoxy 按位 0.0）、max|dg| ≤ 2.7e-6；文档内 dg 5 格中
4 格缩（acetonitrile 5.4x、ethoxy/butadiene 2.3x、benzene 1.7x）
——**methanol 反增 1.5x**（1.24e-6→1.83e-6：首个无核共享 dg 缩减
的 3c 格；门仍以 3.6x 裕度通过，诚实记录，无机制断言）；D4+gCP
两态相同携带。）

**头部发现**：

1. **3c 内部轴完成——最后一方法轴内部缺口闭合**。连同
   LC-124/130/131（ethanol 92 = 1.151、butane 126 = 1.070、
   pyridine 144 = 1.016），3c 内部现读 8 格 63-144 AO **全 fp64
   占优 [1.016,1.318]**：杂化方法上测得的 106-144 平价凹陷
   （pyridine/butane 格网 [1.00,1.14]）在 3c 方法上**不出现**
   （其仅有的平价格为 butane/pyridine），且其小带在更小尺寸读
   **更 fp64 占优**（methanol 63 = 1.318，92 AO 以下最高完整栈
   frac；LC-124 小带 [1.06,1.18] 不推广到 3c）。
2. **benzene-vs-pyridine 在同 144 AO、同方法+基组上是已测最大同
   尺寸 frac 散布：1.298 vs 1.016（0.28）**——固定尺寸下的分子
   依赖可与 UKS 壳层轴（0.21）匹敌。benzene 在其第三基组加入
   芳香 fp64 占优族（svp 1.258 / tzvp 1.254 / mtzvpp 1.298）而
   芳香**杂环**在同方法上读平价类——芳香性读数在 3c 上分裂为
   碳氢（fp64 占优）vs 杂环（平价）：LC-128 芳香类读数获得组成
   轴。
3. **LC-135 的 butadiene 降级须修订：共轭读数为方法格 2/4**
   （b3lyp/tzvp 1.240、3c/mtzvpp 1.215，对比 m062x/tzvp 1.045、
   wb97mv/tzvp 1.082）。butadiene 的完整栈 frac 在其默认 frac 处
   中档的两个格上都超 1.2；「b3lyp 默认级伪象」框架不成立——按
   预注册判定规则记录为真实的第二例外（诚实注记：4 格跨 2 基
   组——3c 规定 mtzvpp；共轭类仍单分子）。
4. **ethoxy UKS 格网闭至 7 格**（SVP 1.224/1.247/1.178、TZVP
   0.979/1.034/1.178、mtzvpp 1.107 居中）——壳层轴保持 frac 噪
   声放大（范围 0.979-1.247），无基组不变方向。
5. **t_grad 栈增益 2 格 helps-less-than-predicted**
   （acetonitrile 1.26x、ethoxy 1.29x 低于 [1.5,2.2] 带；其余
   1.56-1.83x）——J-only 梯度旋钮响应在 3c 轴上分子依赖（自
   LC-128 TZVP 墙发现以来非 NLC 轴首批亚带 t_grad 读数）。

判定 17/22 HIT：P1×4 HIT + benzene MISS（fp64 复现 -4.7%；auto
e2e -15.0%，超 ±15% 带 0.1 点——噪声类，LC-130 族：sub-2s 默认
道；循环与门精确复现）；P2×3 HIT + 2 MISS（acetonitrile/ethoxy
t_grad 低于带 = 发现 5；e2e 增益全 [1.39,1.77]）；P3×4 HIT +
benzene MISS（auto 比 0.855 超带上限 0.85 仅 0.005，五格最弱
blocksize 响应，边缘）；P4×5 HIT（门全过；跨栈 fp64 dE ≤
7.3e-12、max|dg| ≤ 2.7e-6；D4+gCP 相同）；P5 MISS（methanol
1.318 > 1.30 且 butadiene 1.215 > 1.20——两个例外即本单元发现
1 与 3；benzene 1.298 带内加入芳香族；ethoxy 1.107 < 1.30 ✓）；
P6 HIT（每道 1.39-1.77x fp64 / 0.59-0.86 auto 增益、峰值 ≪ 预
算、60/60 收敛）。诚实边界：共轭 2-of-4 跨 2 基组；benzene-
pyridine 散布为单分子对单尺寸；methanol dg 增长单格 J-only 观
察；benzene doc1 auto e2e -15.0% 会话计时漂移（其默认 frac
1.095 vs 冻结 1.228 = -11%——会话内完整栈移动 +0.20 为被引结
论）。测量-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc136-r2scan3c-interior-axis`
（47 文件 sha256 锚定）；effects-index 53 数据集（源全部校验通
过）。

## 73. LC-137 (2026-10-13): SVP 非 NLC 内部带（16 格）—— 凹陷下沿测绘完成，全格 [0.899,1.241]；共轭类候选弱化为基组关联（butadiene/SVP 1.153/1.132）；m062x/ethanol 0.899 为首个完整栈 auto 占优非 NLC 内部格

LC-136 遗留（SVP 平价凹陷下沿无内部点）的直接后续：8 分子 × 2
方法（b3lyp + m062x，全闭壳）——acetaldehyde 62、acetonitrile
57、methylamine 53、furan 90、butadiene 86、acetone 86、
acetic_acid 76、ethanol 72——测绘凹陷下沿并为共轭问题加第三基
组（SVP）。doc1 默认控制复现冻结 LC-117（14/16 干净，2 噪声类
漂移格）。运行时 6d136a0（树干净）。16 份 VALID 文档、96/96 重
复收敛、无 fallback、峰值 108-488 MiB；驱动墙 19 分钟。

| 格 | 默认 frac | 完整 frac | fp64 e2e 增益 | auto 比 | 循环 |
|---|---|---|---|---|---|
| b3lyp/methylamine 53 | 1.050 | **1.048** | 1.45x | 0.690 | 9 |
| b3lyp/acetaldehyde 62 | 0.993 | **1.102** | 1.38x | 0.805 | 10 |
| m062x/acetaldehyde 62 | 1.000 | **1.049** | 1.32x | 0.798 | 10 |
| b3lyp/ethanol 72 | 1.112 | **1.056** | 1.38x | 0.689 | 10 |
| m062x/ethanol 72 | 0.929 | **0.899** | 1.41x | 0.686 | 10 |
| b3lyp/acetic_acid 76 | 1.029 | **1.209** | 1.63x | 0.723 | 11 |
| m062x/acetic_acid 76 | 1.069 | **1.096** | 1.40x | 0.735 | 11 |
| b3lyp/butadiene 86 | 1.015 | **1.153** | 1.55x | 0.731 | 9 |
| m062x/butadiene 86 | 1.006 | **1.132** | 1.67x | 0.675 | 9 |
| b3lyp/acetone 86 | 1.157 | **1.241** | 1.39x | 0.771 | 10 |
| m062x/acetone 86 | 1.166 | **0.989** | 1.28x | 0.661 | 10 |
| b3lyp/acetonitrile 57 | 1.109 | **1.125** | 1.48x | 0.684 | 10 |
| m062x/acetonitrile 57 | 1.102 | **0.976** | 1.28x | 0.693 | 10 |
| b3lyp/furan 90 | 1.161 | **1.088** | 1.48x | 0.632 | 10 |
| m062x/furan 90 | 1.060 | **1.108** | 1.65x | 0.632 | 11 |

（循环每格两态相同，仅 m062x/acetic_acid auto +1（blocksize 族，
带内）；auto e_shift ≤ 1e-8 类；跨栈 fp64 dE ≤ 2.3e-13、
max|dg| ≤ 6.4e-6；文档内 dg 16 格中 15 格缩 **10.7-17.4x**——任
何轴上最大的核共享缩减（SVP 形状最高效共享 fp32 核）；唯一增长
格 b3lyp/methylamine（4.47e-7→5.68e-7，0.79x）带该轴最小绝对
dg（>50x 门裕度）——连同 LC-136 的 methanol，缩减依赖 dg 水平
（微小 dg 格的 fp32 舍入面更小）。文档内门 16 份全过。）

**头部发现**：

1. **SVP 非 NLC 凹陷下沿测绘完成：16 内部格全落
   [0.899,1.241]**——frac 从 48 AO 的 ~1.15（LC-124）经 53-90 内
   部（多为 1.05-1.24）降入 106-144 平价带（LC-130/131）。连同
   LC-124/125/130/131/132，**SVP 非 NLC 阶梯 28-208 完成**；带内
   无 TZVP butadiene 1.240 级的 fp64 占优例外。
2. **共轭类候选弱化为基组关联：butadiene/SVP 落 1.153/1.132**
   （双双 ≤ 1.20；冻结默认 1.035/1.127）。>1.2 读数仅存在于
   TZVP/mtzvpp（b3lyp/tzvp 1.240、3c/mtzvpp 1.215）= 方法格
   2/6 且**无 SVP 格超 1.2**。butadiene 的 6 格完整栈散布
   [1.045,1.240] 为已测最大同分子完整栈散布。
3. **m062x/ethanol 0.899 = 首个完整栈 auto 占优非 NLC 内部格**
   （会话内默认 0.929；auto 1.407 vs fp64 1.516 s）；另有 2 个
   m062x 格处平价或以下（acetonitrile 0.976、acetone 0.989）而
   b3lyp 在同分子上读 1.05-1.24——**方法轴（m062x 更强的
   headtail 响应）在 SVP 内部移动 frac 多于分子轴**，将
   LC-134 的 b3lyp-高于-m062x 读数扩展到 SVP 内部（m062x 中位
   1.04 vs b3lyp 中位 1.11，方向一致 14/16）。
4. **唯一 dg 增长格 b3lyp/methylamine**（0.79x）与 LC-136 的
   methanol 构成第二例无核共享缩减的小格——缩减依赖 dg 水平，
   非缺陷（两格门裕度 >50x）。
5. **首次测量内部基线冻结**（doc1 默认 frac 会话内
   0.993-1.166；冻结 LC-117 锚 0.995-1.286）。

判定 63/66 HIT：P1 14/16 HIT（b3lyp/acetaldehyde auto -17% 与
m062x/ethanol auto -21% 噪声类漂移 MISS——e_tot/循环精确，
LC-133/130 族）；P2×16 HIT（增益 1.28-1.67x 全带内；86-AO
butadiene/acetone 对显示固定尺寸分子依赖 1.55/1.67 vs
1.39/1.28）；P3×16 HIT（auto 比 0.628-0.805 带内、一循环移动
带内）；P4×16 HIT（门全过；跨栈 fp64 dE ≤ 2.3e-13、max|dg| ≤
6.4e-6）；P5 MISS（m062x/ethanol 0.899 < 0.95——带内首个 auto
占优非 NLC 内部格即发现 3，信息性 MISS；15/16 带内）；P6 HIT
（每道 1.28-1.67x fp64 / 0.63-0.81 auto 增益、峰值 ≪ 预算、
96/96 收敛）。诚实边界：m062x/ethanol 单格 ~1.4 s 道（sub-2.5s
噪声类，默认 frac 会话内已漂移 vs 冻结——作稳定 auto 占优格引
用前需重复会话确认，记为首次测量）；共轭弱化 = 单分子 2 SVP
格（「基组关联」为准确表述）；方法分裂读数比较单会话 8 个同
分子对（14/16 方向一致性为被引结论）；dg 缩减幅度为形状驱动
（绝对 dg 值最小、门裕度全 >10x，无精度回退含义）。测量
-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc137-svp-nnlc-interior`
（135 文件 sha256 锚定）；effects-index 54 数据集（源全部校验
通过）。

## 74. LC-138 (2026-10-13): wb97mv/SVP 内部完成 —— 六格全 [0.999,1.189] fp64 占优或平价；SVP NLC 阶梯 43-208 完整；auto 占优 NLC 区域边界确认于 (208,318]；acetone 默认 auto 赢被抹除

LC-137 之后 SVP NLC 内部（48-86 AO 无测点）的完成单元：6 个
dev 格——methanol 48、acetonitrile 57、acetaldehyde 62、ethanol
72、acetic_acid 76、acetone 86——wb97mv/SVP 阶梯 43-208 就此完整
（methoxy 43 UKS 按设计排除，决策 (c)）。doc1 默认控制 6/6 干净
复现冻结 LC-117（墙主导的 3.8-8.8 s 道会话稳定）。运行时
50e386b（树干净）。12 份 VALID 文档、72/72 重复收敛、无
fallback、峰值 58-502 MiB；驱动墙 9.5 分钟。

| 格 | 默认 frac | 完整 frac | fp64 e2e 增益 | t_scf 增益 | auto 比 | 循环 |
|---|---|---|---|---|---|---|
| methanol 48 | 1.080 | **1.134** | 3.09x | 3.55x | 0.340 | 10 |
| acetonitrile 57 | 1.027 | **1.129** | 2.81x | 3.22x | 0.391 | 10 |
| acetaldehyde 62 | 1.052 | **0.999** | 3.30x | 4.02x | 0.288 | 10 |
| ethanol 72 | 1.106 | **1.189** | 4.03x | 4.53x | 0.267 | 10 |
| acetic_acid 76 | 1.054 | **1.034** | 3.75x | 4.21x | 0.262 | 12 |
| acetone 86 | 0.915 | **1.167** | 4.81x | 5.38x | 0.265 | 11 |

（fp64 循环移动仅 acetonitrile doc2 +1（blocksize 族）；auto 循
环移动仅 acetone doc2 +1；其余每道每格两态相同；auto e_shift ≤
1e-8 类；跨栈 fp64 dE ≤ 7.8e-11（核交换类）、max|dg| ≤ 7.5e-6；
文档内 dg 每格缩 3.7-7.3x；门 12 份全过。）

**头部发现**：

1. **P5 头条 HIT：六个 wb97mv/SVP 内部 frac 全落
   [0.999,1.189]**——连同 LC-124/125/126/127/128，wb97mv/SVP 阶梯
   43-208 **完整**，且 90 以下无 auto 占优内部格：**auto 占优
   NLC 区域边界确认于 (208,318]**（vitamin_c 0.972 @208 为唯一
   内部例外；NLC 格 ≥318 全 auto 占优）。NLC 方法内部在两基组
   的每一已测尺寸 48-222 读 fp64 占优或平价。
2. **acetone 的默认 auto 赢在完整栈被抹除**（会话内 0.915 /
   冻结 0.988 → 1.167，0.25 移动 = 六格最大）：wb97mv/SVP 轴唯一
   亚平价默认格在完整栈加入 fp64 占优簇——抹除读数扩展到小 NLC
   内部（连同 LC-133/134：**闭壳层下两类方法均无默认 auto 赢
   在 222 以下于完整栈存活**）。
3. **VV10 墙梯在 SVP 上完成**：t_scf 增益 3.2-5.4x @48-86
   （内插 methylamine 3.77x @53 与 LC-126 的 5.85-7.04x
   @106-208）；acetone 墙最大（5.38x t_scf / 4.81x e2e；冻结默
   认 e2e 8.78 s 为该轴最慢格）。acetonitrile P2 MISS（e2e
   2.81x 低于 [3.0,7.0]；t_grad 1.90x 该轴最弱）
   helps-less-than-predicted。
4. **每道增益 2.8-4.8x fp64 / 0.26-0.39 auto**（P6）——完整栈
   非回退扩展到最后一个未测 SVP 带。

判定 25/26 HIT：P1×6（6/6 干净复现）、P2×5 + 1 MISS
（acetonitrile 良性低侧）、P3×6（auto 比 0.262-0.391 带内、两循
环移动带内）、P4×6（门全过；跨栈 fp64 dE ≤ 7.8e-11、max|dg| ≤
7.5e-6）、P5 HIT（头条）、P6 HIT。诚实边界：(208,318] 边界为
SVP 限定（TZVP 无任何尺寸的 auto 占优格，LC-135）；acetone 抹
除比较会话内默认（冻结 0.988 差噪声类；+0.25 移动远在其外）；
acetonitrile t_grad 弱为单格观察。测量-only；默认与门槛不变；
全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc138-wb97mv-svp-interior`
（55 文件 sha256 锚定）；effects-index 55 数据集（源全部校验通
过）。

## 75. LC-139 (2026-10-13): 3c dev 轴完成 —— bisphenol 413 默认 auto 赢（0.769/0.806）被抹除（1.038）：完整栈下 3c 方法 687 以下无 auto 占优 dev 格；furan 1.425 = 任何格已测最高完整栈 frac；首个亚平价 3c 格；bisphenol t_grad 3.75x = 最大 J-only 梯度旋钮响应

LC-138 之后 3c dev 轴剩余 6 格的完成单元：methylamine 68、
acetaldehyde 82、acetic_acid 106、acetone 111、furan 120、
bisphenol_a 413（methoxy 58 UKS 按设计排除，决策 (c)）——3c dev
轴就此完整。头条格 bisphenol_a 413：冻结默认 auto 赢 0.769（会
话内 0.806 = 复现，doc1 峰 2054 MiB）检验 3c 上带抹除。运行时
2decd8e（树干净）。12 份 VALID 文档、72/72 重复收敛、无
fallback、峰值 120-2614 MiB；驱动墙 12 分钟。

| 格 | 默认 frac | 完整 frac | fp64 e2e 增益 | t_scf | t_grad | auto 比 | 循环 |
|---|---|---|---|---|---|---|---|
| methylamine 68 | 1.352 | **1.087** | 1.25x | 1.38x | 1.17x | 0.644 | 9 |
| acetaldehyde 82 | 1.044 | **0.968** | 1.39x | 1.41x | 1.40x | 0.666 | 10 |
| acetic_acid 106 | 0.975 | **0.983** | 1.44x | 1.56x | 1.32x | 0.701 | 11 |
| acetone 111 | 1.094 | **1.128** | 1.48x | 1.55x | 1.44x | 0.698 | 10 |
| furan 120 | 1.163 | **1.425** | 1.66x | 1.75x | 1.51x | 0.740 | 10 |
| bisphenol_a 413 | 0.806 | **1.038** | 1.81x | 1.65x | **3.75x** | 0.713 | 13/14 |

（循环每道每格两态相同（bisphenol auto 14 vs fp64 13 在两份文
档中均在——栈不变）；auto e_shift ≤ 1e-8 类；跨栈 fp64 dE ≤
1.4e-12（3c J-only 类，已测最小）、max|dg| ≤ 6.6e-6；门 12 份全
过；D4+gCP 两态相同。bisphenol doc2 峰 2614 MiB = 已测最大完整
栈峰，3.5 GiB 准入预算内。）

**头部发现**：

1. **BISPHENOL 413 默认 auto 赢在完整栈被抹除**（0.769 冻结 /
   0.806 会话内 → 1.038）——3c 上带抹除完成（298:
   0.809→1.253、687: 0.836→1.141）：**完整栈下 3c 方法 687 以下
   无 auto 占优 dev 格**；连同 LC-133/134/138，闭壳层「默认
   auto 赢无存活」读数现于**全部四方法**成立。其 fp64 栈增益
   1.81x e2e 带已测最大 J-only t_grad 响应 **3.75x**（413 AO 的
   fp32 ip1 核——J-only 梯度旋钮增益随尺寸强增长，LC-132 的
   3.15-3.80x @208 杂化格先兆）。
   〔LC-151 勘误附注 2026-10-14：本条 298 的 1.253 是 LC-106
   **全 fp32-vmat 变体**读数（该变体后被 headtail 取代）；该格
   在规范 headtail 栈的三会话记录 [0.943, 1.098, 0.964] = 多数
   2/3 **亚平价**（auto 占优-在噪声带内，边际 0.036-0.057）——
   正确表述为"除 vitamin_c/r2scan3c/def2-mtzvpp 298 AO（噪声带
   内 auto 占优）外，3c 在 687 以下每格 fp64 占优或平价"。见
   §81。〕
2. **FURAN 1.425 = 任何格已测最高完整栈 frac**（此前最高：
   benzene/wb97mv/svp 1.258、methanol/3c 1.318）：furan 在每个
   方法上读 fp64 占优（1.055-1.425），其 3c 读数超 benzene 的
   1.298 达 0.11——**3c 方法上芳香碳氢-vs-杂环分裂反转**
   （furan 高于 benzene），与 wb97mv 相反（benzene 1.258 高于
   furan 1.055）。
3. **首个亚平价 3c 格：acetaldehyde 0.968 与 acetic_acid
   0.983**（双双处于恰好 1.0 的 ±5-10% 会话噪声内——记为亚平
   价类而**不**贴 auto 占优标签，与低 10% 于平价的
   m062x/ethanol 0.899 不同）；3c 簇放宽至 [0.968,1.425]，轴上
   分子依赖（furan vs acetaldehyde @82-120）匹敌壳层轴。
4. **小/中 3c 格复现 LC-136 模式**：增益 1.25-1.66x e2e、auto
   比 0.64-0.74、t_grad 1.17-1.51x（分子依赖；methylamine
   1.17x helps-less-than-predicted）；每道增益（P6）。
5. **3c dev 轴完成**（连同 LC-124/130/131/136）：每个 dev 分子
   × 3c 格（methoxy UKS 除外，决策 (c)）双栈态测毕，63-687 AO。

判定 21/26 HIT：P1 4/6 HIT（methylamine fp64 -21% sub-1.3s 噪声
类、acetic_acid auto -16% 边缘，e_tot/循环精确）；P2 4/6 HIT
（methylamine 分母效应 e2e 1.25x 低于带；bisphenol t_grad 3.75x
超带 helps-more-than-predicted；furan 1.66x/acetone 1.48x/
acetic_acid 1.44x/acetaldehyde 1.39x 带内）；P3×6 HIT（auto 比
0.644-0.740 带内；循环全 ±0）；P4×6 HIT（门全过；跨栈 fp64 dE
≤ 1.4e-12 已测最小、max|dg| ≤ 6.6e-6；D4+gCP 相同）；P5 MISS
（bisphenol 抹除头条 HIT，但 furan 1.425 > 1.35、两亚平价格
< 1.00——三个例外即本单元发现；3/5 小/中格带内）；P6 HIT（每道
1.25-1.81x fp64 / 0.64-0.74 auto 增益、峰值 2614 MiB 预算内、
72/72 收敛）。诚实边界：bisphenol 抹除为 413 单格（3c 的
318-537 段无其他 dev 格）；furan 反转为 2 分子 1 方法；亚平价格
处于会话噪声内（无 auto 占优过度声明）；methylamine P1/P2 共享
分母效应（doc1 fp64 快 21%）。测量-only；默认与门槛不变；全部
旋钮 opt-in。证据：
`tasks/issue-9/2026-10-13-lc139-r2scan3c-remaining-cells`
（55 文件 sha256 锚定）；effects-index 56 数据集（源全部校验通
过）。

## 76. LC-140 (2026-10-14): 完整栈 heldout 评估（24 格）—— inosine 非 NLC auto 赢在完整栈存活（b3lyp 0.889 / m062x 0.927）：抹除规则修正为 ~222 以下，非 NLC 完整栈面尺寸非单调；NLC ≥318 规则确认（wb97mv/inosine 0.948）；wb97mv/inosine 增益 7.14x e2e 为栈最大单格效应

cohort 程序的确认步骤：全部 24 个 heldout 方法格（pyrrole /
nitromethane / ethylamine / nitrogen_dioxide UKS / acetamide /
inosine × 4 方法）在完整 5 旋钮栈 vs 配对默认控制。**冻结完整性**：
候选设置与源码在运行前冻结于 9d843c0（树干净），无运行后调参。
第二次 heldout 使用（LC-119 默认栈为第一次）——标注为完整栈确认
而非新的盲测。运行时 9d843c0。48 份 VALID 文档、288/288 重复收
敛、无 fallback、峰值 50-4348 MiB；驱动墙 33 分钟（A+B+NO2 重
试）。

**协议注记（诚实）**：A 部循环给四个 nitrogen_dioxide 格传了
spin 0（spin-1 修正指向了循环形式中不存在的独立行）——四格同时
报错（"Electron number 23 and spin 0 are not consistent"，NO2 为
23 电子二重态）。16 个 attempt-1 错误文件保留于
attempt1-spin0-errors/；重试（尝试 2）以 spin 1 跑通四格。其他
格不受影响。

| 格 | 默认 frac | 完整 frac | fp64 e2e 增益 | t_scf | t_grad | auto 比 |
|---|---|---|---|---|---|---|
| b3lyp/pyrrole 95 | 1.026 | **1.096** | 1.45x | 1.51x | 1.63x | 0.734 |
| m062x/pyrrole 95 | 1.070 | **0.915** | 1.44x | 1.41x | 1.72x | 0.595 |
| wb97mv/pyrrole 95 | 1.040 | **1.203** | 4.87x | 5.44x | 3.42x | 0.237 |
| r2scan3c/pyrrole 125 | 1.092 | **1.082** | 1.63x | 1.65x | 1.88x | 0.608 |
| b3lyp/nitromethane 71 | 1.093 | **1.105** | 1.35x | 1.25x | 1.96x | 0.747 |
| m062x/nitromethane 71 | 1.203 | **1.200** | 1.52x | 1.58x | 1.69x | 0.658 |
| wb97mv/nitromethane 71 | 1.197 | **1.123** | 3.17x | 3.46x | 2.43x | 0.296 |
| r2scan3c/nitromethane 106 | 1.220 | **1.191** | 1.35x | 1.41x | 1.31x | 0.725 |
| b3lyp/ethylamine 77 | 1.013 | **1.084** | 1.62x | 1.64x | 1.78x | 0.659 |
| m062x/ethylamine 77 | 1.087 | **1.006** | 1.42x | 1.49x | 1.33x | 0.652 |
| wb97mv/ethylamine 77 | 1.013 | **1.076** | 4.22x | 4.79x | 2.79x | 0.252 |
| r2scan3c/ethylamine 97 | 1.140 | **0.948** | 1.43x | 1.58x | 1.22x | 0.580 |
| b3lyp/NO2 UKS 42 | 1.064 | **0.964** | 1.40x | 1.37x | 1.62x | 0.648 |
| m062x/NO2 UKS 42 | 1.094 | **1.001** | 1.41x | 1.45x | 1.24x | 0.648 |
| wb97mv/NO2 UKS 42 | 1.160 | **1.163** | 2.08x | 2.19x | 1.80x | 0.482 |
| r2scan3c/NO2 UKS 72 | 1.105 | **1.054** | 1.18x | 1.11x | 1.57x | 0.806 |
| b3lyp/acetamide 81 | 1.019 | **1.131** | 1.37x | 1.36x | 1.75x | 0.812 |
| m062x/acetamide 81 | 1.030 | **1.132** | 1.54x | 1.54x | 1.80x | 0.714 |
| wb97mv/acetamide 81 | 0.976 | **1.080** | 3.86x | 4.35x | 2.63x | 0.286 |
| r2scan3c/acetamide 111 | 1.059 | **1.119** | 1.39x | 1.44x | 1.23x | 0.762 |
| m062x/inosine 326 | 0.788 | **0.927** | 1.68x | 1.44x | 3.65x | 0.699 |
| b3lyp/inosine 326 | 0.732 | **0.889** | 1.57x | 1.25x | 3.53x | 0.771 |
| wb97mv/inosine 326 | 0.974 | **0.948** | **7.14x** | 7.55x | **5.61x** | 0.136 |
| r2scan3c/inosine 466 | 0.901 | **1.100** | 1.67x | 1.54x | **3.84x** | 0.730 |

（循环除 6 道 ±1 单样本类移动（双向，blocksize 族，带内）外全
同；auto e_shift ≤ 1e-8 类；跨栈 fp64 dE ≤ 3.6e-10（已测最大，
inosine VV10 格，对计数类）、max|dg| ≤ 9.95e-6（诊断带内）；
**文档内门 48 份全过——每个 heldout auto 重复均在 1e-6 Ha /
3e-5 dg 内匹配其同文档 fp64 参考**；峰值至 4348 MiB（m062x/
inosine doc2）在 4.5 GiB 准入预算内。）

**头部发现**：

1. **inosine 非 NLC auto 赢在完整栈存活：b3lyp 0.889、m062x
   0.927**（默认 0.732/0.788；仅 3c 格抹除，1.100）——dev 侧
   「687 以下无默认 auto 赢存活」规则是**外推**：每个 dev 非
   NLC 抹除都在 ≤216 AO 测得，222 以上无 dev 非 NLC 完整栈格
   （tamoxifen/bisphenol 非 NLC 未测）。**规则修正为「~222 以
   下无非 NLC 默认 auto 赢存活」**：非 NLC 完整栈面尺寸非单调
   （48-222 fp64 占优，~326 于 inoside 类富氢梯度重分子 auto
   占优，胜幅衰减 0.73→0.89 / 0.79→0.93 但未抹除）；222-537
   非 NLC 区间仅在 heldout inosine 326 测过——tamoxifen/
   bisphenol 非 NLC 完整栈格为**具名缺口**。
2. **NLC ≥318 规则确认：wb97mv/inosine 0.948 保持 auto 占优**
   （默认 0.974）——NLC auto 占优区（SVP ≥318）在规则赖以建立
   的精确格类上于 heldout 成立；NLC/非 NLC 完整栈区分经受住
   heldout 检验（非 NLC 抹除规则则没有）。
3. **wb97mv/inosine 增益 7.14x e2e（101.0→14.2 s）= 完整栈最
   大单格效应**（VV10 墙 7.55x t_scf + 梯度路径 5.61x t_grad；
   auto 道 0.136x；P2 超带 helps-more-than-predicted——带
   [4.0,6.5] 取自 LC-120 仅 VV10 旋钮的 4.70x）。
4. **小/中格 19/20 落入 dev 各方法簇**（b3lyp
   [1.084,1.131]、m062x [0.915,1.232]、wb97mv [1.076,1.203]、
   3c [0.948,1.191]）；唯一例外 r2scan3c/ethylamine 0.948 将
   3c 亚平价类扩展到 heldout（连同 LC-139 的 0.968/0.983）；
   m062x/pyrrole 0.915 为最深亚平价非 NLC 小格（m062x 亚平价
   类扩展到 heldout）。
5. **t_grad 尺寸梯延伸：326-466 AO 格上 3.5-3.8x**（每方法
   类；与 bisphenol 3.75x（LC-139）、vitamin_c 3.15-3.80x
   （LC-132）复合）——fp32 梯度旋钮增益 ≥326 AO 稳定于
   3.5-3.8x。
6. **nitrogen_dioxide UKS 格（spin 1 重试后）全收敛过门**：
   NO2 完整栈 frac 0.964/1.001/1.163/1.054——第三个 UKS 分子
   （b3lyp 亚平价如其 dev UKS 对；壳层轴再次噪声放大）。

判定 91/98 HIT：P1 22/24（b3lyp/ethylamine fp64 +17% sub-2s 噪
声类、r2scan3c/NO2 auto -20% 重试格 fp64 精确）；P2 21/24
（wb97mv/inosine 7.14x 超带 helps-more + wb97mv/NO2 2.08x 低
带——42 AO 为已测最小 VV10 墙 + r2scan3c/NO2 1.18x 低带）；
P3 23/24（wb97mv/inosine 0.136 低于带，helps-more）；P4×24 HIT
（门全过；跨栈 fp64 dE ≤ 3.6e-10、max|dg| ≤ 9.95e-6）；P5
MISS——头条：(a) 被驳（inosine 存活）、(b) 确认、(c) 19/20，按
预注册判定规则诚实记录，即本单元核心交付；P6 HIT（每道
1.18-7.14x fp64 / 0.14-0.81 auto 增益、峰值预算内、288/288 收
敛）。诚实边界：inosine 存活为单分子单尺寸 2 方法（0.889/0.927
低于平价 11-7%——在 ±5-10% 会话噪声外但不远；重复会话将巩固，
记为首次测量）；第二次 heldout 使用标注（完整栈规则首次接触
heldout，但 dev 规则晚于 LC-119）；NO2 attempt-1 错误为运行器
bug（已保留）。测量-only；默认与门槛不变；全部旋钮 opt-in。证
据：`tasks/issue-9/2026-10-14-lc140-heldout-complete-stack`
（229 文件 sha256 锚定）；effects-index 57 数据集（源全部校验
通过）。

## 77. LC-141 (2026-10-14): 222-537 非 NLC 缺口闭合 —— 四个完整栈 frac 全 < 1.0（bisphenol 0.922/0.913、tamoxifen 0.761/0.827）：非 NLC auto 占优带扩展至 318-537，LC-140 修正规则在 dev 分子上确认；非 NLC 规则完整（48-222 fp64 占优 / 318-537 auto 占优）

LC-140 具名缺口的直接后续：最后一个未测非 NLC 完整栈区间——
bisphenol_a 318 与 tamoxifen 537，各 b3lyp + m062x（def2-svp）。
LC-140 inosine 存活预测在 dev 分子上的直接检验。doc1 默认控制
4/4 干净复现冻结 LC-117（4-24 s 道会话稳定）。运行时 2a243d7
（树干净）。8 份 VALID 文档、48/48 重复收敛、无 fallback、峰值
2988-9962 MiB；驱动墙 10 分钟。

| 格 | 默认 frac | 完整 frac | fp64 e2e 增益 | t_scf | t_grad | auto 比 | 循环 |
|---|---|---|---|---|---|---|---|
| b3lyp/bisphenol 318 | 0.744 | **0.922** | 1.77x | 1.32x | 3.69x | 0.701 | 13 |
| m062x/bisphenol 318 | 0.749 | **0.913** | 1.86x | 1.48x | **4.22x** | 0.654 | 12 |
| b3lyp/tamoxifen 537 | 0.673 | **0.761** | 1.38x | 1.05x | 2.99x | 0.821 | 13 |
| m062x/tamoxifen 537 | 0.680 | **0.827** | 1.50x | 1.22x | 3.25x | 0.814 | 13 |

（循环每道每格两态相同；auto e_shift ≤ 6e-12；跨栈 fp64 dE ≤
1.4e-12（两格按位 0.0）；跨栈 fp64 max|dg| 于 tamoxifen 两格
1.20-1.21e-5——超预注册 1.1e-5 **诊断带** 0.1e-5，LC-132 核交换类
边缘（knob-on fp64 道非严格参考，knob-off 文档才是——(b3) 语义；
**科学门全过**：文档内 dE ≤ 6e-12、文档内 dg 1.2e-5→1.1-1.4e-6
缩 9-11 倍、8 份文档 auto 道全过）；峰值至 9962 MiB 于 11 GiB 准
入预算内。）

**头部发现**：

1. **P5 头条 HIT——四个完整栈 frac 全 < 1.0：非 NLC auto 占优带
   扩展至 318-537**。连同 heldout inosine（0.889/0.927 @326），
   auto 占优非 NLC 读数为 3 分子 × 2 方法 × 318-537 AO，且非
   NLC 规则**完整**：**48-222 fp64 占优（抹除带）、318-537
   auto 占优**——衰减模式处处成立（默认 0.66-0.79 → 完整
   0.76-0.92：栈收窄差距 30-60% 但从不抹除）。LC-140 修正规则
   在 dev 分子上确认（inosine 非分子特异）。
2. **t_grad 梯达峰：m062x/bisphenol 4.22x**（杂化上已测最大单道
   梯度旋钮响应）；326-466 梯延续（2.99-4.22x 分子依赖）；
   tamoxifen t_scf 增益仅 1.05-1.22x（J 重的 537-AO SCF 除
   headtail 外 fp32 面很少）——该带 e2e 增益 1.38-1.86x 为梯度
   路径驱动。
3. **auto blocksize 比随尺寸减弱**（0.65-0.70 @318 → 0.81-0.82
   @537）——LC-133 TZVP AO 稀释读数扩展到 SVP 大带。
4. **首次测量锚冻结**：已测最大非 NLC 完整栈格（tamoxifen
   537 AO，峰值 9962 MiB）与 bisphenol 非 NLC 完整栈格。

判定 15/18 HIT：P1×4（4/4 干净复现）、P2×3 + 1 MISS
（m062x/bisphenol t_grad 4.22x 超带 helps-more）；P3×4（auto 比
0.654-0.821 带内、循环全 ±0）、P4×2 + 2 MISS（tamoxifen 跨栈
dg 超**诊断带** 0.1e-5 = 核交换类边缘，科学门全过）、P5 HIT
（头条）、P6 HIT（每道 1.38-1.86x fp64 / 0.65-0.82 auto 增益、
峰值预算内、48/48 收敛）。诚实边界：带读数 = 3 分子 3 尺寸，
cohort 318-537 无内部点；222-318 与 537-687 带缘于非 NLC SVP 未
测（537-687 仅 tamoxifen/3c @687 存在）；dg 诊断超标按原样记录
（跨态核交换，非门失败）。测量-only；默认与门槛不变；全部旋钮
opt-in。证据：
`tasks/issue-9/2026-10-14-lc141-nnlc-222-537-gap`
（41 文件 sha256 锚定）；effects-index 58 数据集（源全部校验通
过）。

## 78. LC-142 (2026-10-14): 重复会话稳定性确认 —— 大带存活率 6/6 复现（非 NLC auto 占优 318-537 带经两个独立会话确证）；m062x/ethanol 0.899 与 3c 亚平价三重奏两个噪声类标签按预注册规则诚实撤回（1.005 / 1.239,1.019,1.139）；会话噪声类被刻画（auto 道 sub-2.5s 道跨会话漂移至 +30%，≥4s 道稳定 ±9%）

checkpoint 具名的稳定性跟进单元：每个被标记"需重复会话确认"
的首测读数。仅 10 份完整栈文档（doc2 类——frac 为文档内
auto/fp64 e2e 比，无需默认控制），fp64+auto 两道各 3 个新进程
样本，LC-122..141 纪律逐字复用（相同 env/旗标/runner/超时）。
运行时 988d6c1（树干净）；驱动墙 8 分钟；10 份 VALID 文档、
60/60 重复收敛、auto 道文档内门 10/10 全过、峰值 ≤ 9958 MiB。

| 格 | 锚 frac | 复现 frac | d | fp64 e2e 漂移 | auto e2e 漂移 | 循环 |
|---|---|---|---|---|---|---|
| m062x/ethanol 72 | 0.899 | **1.005** | +0.106 | 0.99x | +11% | 10/10 |
| r2scan3c/acetaldehyde 82 | 0.968 | **1.239** | +0.271 | 1.02x | +30% | 10/10 |
| r2scan3c/acetic_acid 106 | 0.983 | **1.019** | +0.036 | 0.99x | +3% | 11/11 |
| r2scan3c/ethylamine 130 | 0.948 | **1.139** | +0.191 | 0.97x | +17% | 10/10 |
| b3lyp/bisphenol 318 | 0.922 | **0.965** | +0.043 | 1.04x | +9% | 13/13 |
| m062x/bisphenol 318 | 0.913 | **0.945** | +0.032 | 1.06x | +9% | 12/12 |
| b3lyp/inosine 326 | 0.889 | **0.903** | +0.014 | 1.02x | +4% | 18/18 |
| m062x/inosine 326 | 0.927 | **0.969** | +0.042 | 0.95x | -0% | 17/17 |
| b3lyp/tamoxifen 537 | 0.761 | **0.769** | +0.008 | 1.00x | +1% | 13/13 |
| m062x/tamoxifen 537 | 0.827 | **0.824** | -0.003 | 0.99x | -1% | 13/13 |

（全部 dE ≤ 5.0e-12、max|dg| ≤ 1.51e-6——数值在跨进程底板复
现；每个重复收敛；循环逐字相同。）

**头部发现**：

1. **P3 头条 HIT——大带存活率 6/6 复现：非 NLC auto 占优
   318-537 带为会话稳定**。每格再次 < 1.0、每格距冻结锚
   ≤ ±0.08，≥4s 道在 ±9% 水平复现（tamoxifen 基本数字稳定：
   0.769/0.824 vs 0.761/0.827）。LC-140 诚实边界（"inosine 存活
   一个分子一个尺寸两个方法……重复会话将确证，记录首测"）现
   已解决：**确证——inosine 0.903/0.969 第二会话仍 auto 占
   优**。LC-141 修正非 NLC 规则（48-222 fp64 占优、318-537
   auto 占优）现在在全部六个带格上有**两个独立会话**支撑。
2. **P1 诚实 MISS（预注册降级分支触发）：m062x/ethanol
   0.899「首个完整栈 auto 占优非 NLC 内部格」标签撤回**。复现
   读数 1.005——恰好平价。机制为 sub-2.5s 道上的 auto 道计时
   漂移（+11%，文档内样本 [1.067, 0.893, 1.121] = 20% 散布），
   fp64 道稳定（0.99x）。按预注册决策规则降级为噪声类——两个
   方向都不做静默重标。
3. **P2 诚实 MISS——3c 亚平价类不复现：三重奏读数 1.239/
   1.019/1.139（全部 ≥ 1.0），三个格的亚平价标签全部撤回**。
   机制同为 sub-2.5s 道 auto 道漂移（+30%/+17%/+3%，样本散布
   至 20%），fp64 道稳定（0.97-1.02x）、循环相同。
4. **会话噪声类现已刻画：auto 道 sub-2.5s 道跨会话漂移至
   +30%；≥4s 道稳定 ±9% 以内**。被撤回的两个标签恰是
   checkpoint 标记为 sub-2.5s 噪声类候选的读数；没有一个
   ≥3s 道。引用实用规则：sub-2.5s 道上的 frac 读数带 ±0.15-
   0.25 会话不确定性，任何标签前需重复会话确认；≥4s 道读数
   单会话可引（整个大带现已有双会话证据）。
5. **规则结构经两次撤回反而强化**：ethanol 复位于 1.005 意味
   318 以下现在**完全没有** auto 占优非 NLC 格（内部规则
   "48-222 fp64 占优"失去其唯一被标记的例外）；3c 三重奏位于
   1.019-1.239 意味 3c 方法在 687 以下每个 dev 格 fp64 占优或
   平价（LC-139 的"687 以下无 auto 占优 dev 格"失去其亚平价
   软点）。干净的规则存活；只有噪声类软标签消失。
   〔LC-151 勘误附注 2026-10-14：vitamin_c/r2scan3c 在规范
   headtail 栈三会话 [0.943, 1.098, 0.964] 多数亚平价——该格
   成为 3c 在 687 以下的唯一 auto 占优（噪声带内）dev 格，本条
   的"每个 dev 格 fp64 占优或平价"随之修正；见 §81。〕

判定 2/5 HIT（P3/P4 HIT；P1/P2/P5 MISS——miss 即本单元的交付
物，两个标签按预注册规则撤回、噪声类被刻画）。诚实边界：两个
会话、一台机器、相邻日期（9d843c0/2a243d7/988d6c1 均为同源
系上的 measurement-only 提交）——稳定性陈述限同源同硬件；被
撤回标签在冻结 LC-137/139/140 证据与文档中作为首测读数保留，
本单元补充复现记录与撤回——文档两者并载；噪声类刻画基于
4 小格 + 6 大格，2.5-4s 道间隙在完整栈无 cohort 格落入（未
测）；heldout 第三次使用：inosine 与 ethylamine 现为稳定性确
认格、**非盲测**（盲测用途止于 LC-119/LC-140，记录在案）。
测量-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-14-lc142-repeat-session-stability`
（48 文件 sha256 锚定）；effects-index 59 数据集（源全部校验
通过）。

## 79. LC-144 (2026-10-14): 2.5-4s 道间隙带实测 —— 会话噪声类为中间类、frac 噪声阶梯按道墙单调（sub-2.5s 中位 0.149 > 2.5-4s 0.100 > ≥4s 0.023）；无规则标签翻转；ethoxy/tzvp 最紧簇成员资格按预注册规则诚实撤回

LC-142 具名诚实边界——"2.5-4s 道间隙未测（无 cohort 格落入）"
——实为对其自身 10 格的覆盖陈述：重扫冻结完整栈文档
（lc122..lc142）发现 **5 个单会话格**的 fp64 或 auto 道墙落在
[2.5, 4.0) s，其中三个承载规则性读数（芳香 fp64 占优类、UKS
噪声放大壳轴、wb97mv/TZVP 最紧簇轴）。本单元按 LC-142 纪律
逐字给每格第二会话记录，另加 bisphenol/b3lyp（fp64 3.5 s，
带内）作双会话已知 CONTROL（0.922 LC-141 → 0.965 LC-142）。
预注册 01:44:41Z（先于 01:45:11Z 运行）；运行时 2894153（树干
净）；驱动墙 2 分 50 秒；6 份 VALID 文档、36/36 重复收敛、全
道 admit、零回退、峰值 242-3688 MiB。

| 格 | 锚 frac | 锚源 | 复现 frac | d | fp64 漂移 | auto 漂移 |
|---|---:|---|---:|---:|---:|---:|
| wb97m-v/ethoxy/svp UKS 67 | 1.224 | LC-125 | **1.087** | -0.137 | 1.08x | 0.96x |
| wb97m-v/ethoxy/tzvp UKS 123 | 1.034 | LC-135 | **1.204** | **+0.170** | 0.90x | 1.05x |
| wb97m-v/butane/tzvp 184 | 1.075 | LC-135 | **1.098** | +0.023 | 1.00x | 1.02x |
| wb97m-v/furan/tzvp 179 | 1.109 | LC-128 | **1.209** | +0.100 | 0.90x | 0.98x |
| wb97m-v/pyridine/tzvp 216 | 1.111 | LC-128 | **1.103** | -0.008 | 1.02x | 1.01x |
| b3lyp/bisphenol 318（CONTROL） | 0.965 | LC-142 | **0.872** | -0.093 | 1.02x | 0.92x |

（锚 e_tot 差 ≤ 2.3e-13 Ha（数个逐位 0.0——跨进程底板）；
文档内 auto-vs-fp64 max|dg| ≤ 1.33e-6；fp64 循环 6/6 相同；
auto 循环仅 furan 11→12、pyridine 12→13 各 +1——auto 道近阈值
末循环决策翻转，能量在 1e-11 水平不受影响。）

**头部发现**：

1. **2.5-4s 带为中间噪声类，frac 噪声阶梯按道墙在整个已测范
   围单调**：sub-2.5s 中位 |d| 0.149 / 最大 0.271（LC-142，4
   格）> **2.5-4s 中位 0.100 / 最大 0.170（本单元，5 格）** >
   ≥4s 中位 0.023 / 最大 0.043（LC-142，6 格）。LC-142 的两端
   刻画扩展到实测中间。
2. **2.5-4s 的机制是比率复合，而非道漂移**：每条单独道墙在
   ±10% 内复现（fp64 [0.90,1.08]、auto [0.92,1.05]）——道级
   表现即 ≥4s 类行为——但 frac 是两条独立漂移道的比值，极值复
   合放大（ethoxy/tzvp：fp64 0.903x × auto 1.052x → frac
   +0.170）。sub-2.5s 机制（LC-142）是 auto 道单独漂移至
   +30%；2.5-4s 时**两道各自稳定而比值仍游移**。
3. **P2 HIT——无规则承载标签翻转**：芳香格复现 fp64 占优
   （furan 1.209、pyridine 1.103，均 ≥ 1.0；芳香类保持）；
   butane 复现簇内（1.098）；UKS ethoxy 对保持噪声放大包络内
   （1.087 / 1.204；方向再次非基组不变——LC-135 壳轴读数被
   确认，现具第二会话）。
4. **一次诚实撤回（预注册降级分支）：ethoxy/tzvp 的 LC-135 最
   紧簇成员资格（1.034 ∈ [1.03,1.11]）撤回**——复现读
   1.204，在最紧完整栈簇外。冻结锚保留为首测记录；文档两者
   并载。撤回仅弱化该簇的 UKS 成员：LC-135 轴读数"每个内部格
   fp64 占优 [1.03,1.11]"复现保持符号（1.204 仍 fp64 占优）
   但数值包络排除 UKS 格；闭壳簇格（butane 复现 1.098）不受
   影响。
5. **P5 MISS——CONTROL 移动 -0.093（0.965 → 0.872），超出其
   ±0.08 带外：预注册会话环境警示触发**。诚实双重读数（两者
   并录）：（a）警示适用——间隙格计时读数带会话旗、仅数值结
   论保留；（b）CONTROL 自身即 2.5-4s 带格（fp64 3.55 s /
   auto 3.10 s），其移动更简约解释为带属性而非总会话伪影
   （温度 22-33 C、时钟稳定、fp64 道 1.00-1.08x 全程）。该格
   auto 占优**符号**三会话稳定（0.922 / 0.965 / 0.872 全 <
   1.0）：非 NLC 318-537 带的 b3lyp 边保持，带边距不确定度放
   宽（0.87-0.97）。

**引用规则（LC-142 §78 规则现为三类）**：sub-2.5s 道 frac 不
确定度 ±0.15-0.25、任何标签前需重复会话（不变）；**2.5-4s 道
（新）：frac 不确定度 ~±0.10-0.17（中位 0.10、最大 0.17）——
仅符号读数在距决策边界 > ~0.2 时单会话可引；标签承载读数（簇
成员资格、紧数值包络）需重复会话确认**；≥4s 道 ±0.09（实测
最大 0.043）、单会话可引（不变）。

判定 3/5 HIT（P2/P3/P4 HIT；P1/P5 MISS 即交付物）。诚实边界：
每间隙格一次复现（每格两会话）、一台机器、同源系；5 间隙格 +
1 control、带上端（3.5-4.0 s）仅由 control 采样、4 间隙格位于
2.5-3.2 s；ethoxy/tzvp fp64 文档内散布 15.7%（首样本 3.015
离群高）——中位数缓解但不消除采样噪声；0.17 最大 |d| 由单一
格界定（UKS 壳轴格——已被标记噪声放大的类），闭壳间隙格最大
|d| 0.100。测量-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-14-lc144-lane-gap-stability`（32 文件
sha256 锚定）；effects-index 61 数据集。

## 80. LC-145 (2026-10-14): dev 完整栈覆盖闭合 —— 一次新审计抓到 checkpoint 的"每 dev 格已测"过度声明；剩余 7 个 dev 格在完整 5 旋钮栈全部 fp64 占优或平价（frac 1.046-1.152 = 全系列最紧簇）；oxyT/r2scan3c 默认亚平价 0.982 不存活（1.046）

**起点是一次防御性审计，不是新假设**：新会话重扫全部 96 个冻结
manifest 格（按几何 sha256 键控，对全部证据文档），发现 checkpoint
的"完整栈面在每 dev + heldout 格已测"声明不成立——85/96 已覆盖、
4 个 methoxy 格按设计排除（LC-125，决策 (c) 分支分裂格，已记录），
**7 个 dev 格仅有默认栈文档**（LC-116/117）：benzene/b3lyp +
benzene/m062x SVP 114、butadiene/wb97m-v SVP 86（NLC）、
ethoxy/b3lyp SVP 67 UKS、oxygen_triplet 的 m062x / r2scan3c /
wb97m-v 三格（28-48 AO UKS）。逐轴 phase 记录各自准确；聚合声明
不准确。本单元按 LC-126..139 纪律逐字补齐：production
gradient_runner.py、冻结 cohort 几何、显式 charge/spin、fp64+auto
两道、每道 3 新进程样本、doc1 默认栈 -> doc2 完整 5 旋钮栈相邻对。
预注册 18:15:56Z（sha256 8c1105d3...，先于 18:16:41Z 运行）；运行
时 9b2bbba（树干净）；驱动墙 4 分 11 秒；14 份 VALID 文档、84/84
重复收敛、零回退、峰值 ≤ 554 MiB。

| 格 | 冻结 d1 frac | d1 frac | d2 frac | fp64 增益 | auto 比 | 循环 |
|---|---|---|---|---|---|---|
| benzene/b3lyp/svp 114 | 1.159 | 1.175 | **1.080** | 1.414x | 0.650x | 7/8→7/8 |
| benzene/m062x/svp 114 | 1.113 | 1.160 | **1.141** | 1.589x | 0.619x | 8/9→8/9 |
| butadiene/wb97mv/svp 86 | 1.067 | 1.062 | **1.152** | 4.848x | 0.224x | 9/9→9/9 |
| ethoxy/b3lyp/svp 67 UKS | 1.068 | 1.130 | **1.126** | 1.492x | 0.667x | 11/11→11/11 |
| oxyT/m062x/svp 28 UKS | 1.212 | 1.150 | **1.093** | 1.346x | 0.706x | 8/9→8/9 |
| oxyT/r2scan3c/mtzvpp 48 UKS | 0.982 | 1.247 | **1.046** | 1.305x | 0.642x | 8/9→9/9 |
| oxyT/wb97mv/svp 28 UKS | 1.319 | 1.074 | **1.108** | 1.654x | 0.624x | 8/9→8/9 |

**头部发现（P5 HIT）**：

1. **全部 7 个完整栈 frac 落在 [1.046,1.152]——fp64 占优或平
   际，全系列最紧簇（散布 0.106；对照 furan 1.425 单点高值）**。
   无任何新 auto 占优格；dev 完整栈面在本单元后**真正闭合**
   （92/96 已测 + 4 methoxy 决策-(c) 设计排除；LC-148 审计重导
   出确认 92 = 68 dev + 24 heldout——96 格分母为 dev+heldout
   总格数，dev 单侧为 68/72；且 92 中 1 格 vitamin_c/r2scan3c
   的完整栈文档是 LC-106 的 `SCF_VMAT_F32=1` 全 fp32-vmat 变
   体（后被 687-AO LC-108 安全发现取代、LC-115 起 headtail 才
   是规范栈），headtail 规范栈计数为 91/96——vitamin_c/r2scan3c
   的 headtail 格为已具名未测间隙）。
2. **oxyT/r2scan3c 默认亚平价 0.982 不存活（1.046）**——LC-142
   撤回方向获得第四个 3c 格支持；3c 在 687 以下每个非 methoxy
   dev 格 fp64 占优或平价。
3. **UKS 壳轴 4 个新格全读 fp64 占优**（ethoxy/b3lyp 1.126；
   oxyT 三方法 1.046-1.108）——与 LC-125/133/135 的壳轴噪声放大
   读数一致方向：壳轴从不给出 basis 不变的 auto 占优格。
4. **benzene 芳香 fp64 占优族扩展到其非 NLC SVP 格**
   （1.080/1.141；与 wb97mv SVP 1.258 LC-127、TZVP 1.254/1.078
   LC-128 并列——benzene 在 4 个方法-基组组合上全 fp64 占优）；
   butadiene/wb97mv 1.152 落在 wb97mv/SVP 内部带簇内
   （LC-138 [0.999,1.189]）。

**P1 诚实 MISS（5/7）**：fp64 道复现冻结锚 0.944-1.105（7/7 在
±15% 内）；两条 auto 道漂出 ±15% 带外——ethoxy 1.158、oxyT/
r2scan3c 1.270（第三格 oxyT/wb97mv 0.900 在 ±15% 带内、以
0.0005 之差出 ±10% 道带）——全部为 sub-2.5s 噪声类道（LC-142/
LC-144 已刻画类），而其 e_tot 复现在底板（ethoxy auto
-5.7e-14 / fp64 -1.1e-13；oxyT/r2scan3c 与 oxyT/wb97mv 逐位
0.0）：纯会话计时噪声、物理同一；预注册引用规则适用于 doc1 锚。
（LC-148 勘误：原文"三条 auto 道漂出带外"把 ±10% 道带与
±15% 预注册带混写，且逐位三元组 (0.0/0.0/-1.1e-13) 的格映射
有误——本段按 analysis.json 逐格改写；adjudication.json 原文
无错。MISS 判定不变。）
P2 benzene/b3lyp 1.414x 超出其 [0.85,1.20] 哨兵带（MISS-high，
b3lyp 梯度旋钮 helps-more 族延伸至芳香 114-AO 格——b3lyp "无增
益"先验现仅限 SVP 内部小格）。P3 HIT 7/7；P4 HIT（文档内门
14/14 全过：auto 最大 |dE| 2.9e-11 / |dg| 4.0e-6；跨栈 fp64
dE ≤ 1.3e-10、max|dg| ≤ 4.8e-6、两个逐位）；P6 HIT（无道回
退；峰值 ≤ 554 MiB）。

**勘误（本单元自身）**：checkpoint 与 docs 的聚合声明"每 dev 格
已有完整栈文档"在本单元前不成立；本节即为更正记录——正确表述
为"除 4 个 methoxy 决策-(c) 设计排除外，每 dev 格均有完整栈文
档"。heldout 侧声明（LC-140，24/24）经同一审计验证为真。

判定 5/6 HIT + P1 诚实 MISS（噪声类，数值逐位）。诚实边界：单
会话、6/7 格 sub-2.5s 道类（引用规则不确定带适用——但簇散布
0.106 远在其内，且全部 7 格 ≥ 1.0）、一台机器、同源系。
测量-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-14-lc145-dev-coverage-closure`（63 文件
sha256 锚定，清单 1dd4dd4c...）；effects-index 62 数据集。

### §80 补（LC-146, 2026-10-14）: LC-145 七格的重复会话确认 —— 6/7 两会话确证；oxyT/r2scan3c 符号翻转（1.046 → 0.841）按预注册降级分支诚实撤回

LC-145 自己的引用规则要求 sub-2.5s 道上的标签承载读数需重复
会话确认——其 7 格中 6 格为 sub-2.5s 类，且 oxyT/r2scan3c 的
1.046 落在 1.0 决策边界的噪声带内。本单元按 LC-142 纪律逐字
执行（仅 doc2 类完整栈文档——frac 为文档内 auto/fp64 e2e 比，
无需默认控制）。预注册 18:29:17Z（先于 18:29:41Z 运行）；运行
时 9b2bbba runner 源（树含 docs-only 3121211，TREE_DIRTY=no）；
7 份 VALID 文档、42/42 重复收敛、锚 e_tot 4/7 逐位（其余
≤ 4.6e-13）、每道每格循环逐字相同。

| 格 | LC-145 锚 | 复现 | d |
|---|---|---|---|
| benzene/b3lyp/svp | 1.080 | **1.013** | -0.066 |
| benzene/m062x/svp | 1.141 | **1.134** | -0.007 |
| butadiene/wb97mv/svp | 1.152 | **1.064** | -0.088 |
| ethoxy/b3lyp/svp UKS | 1.126 | **1.095** | -0.031 |
| oxyT/m062x/svp UKS | 1.093 | **1.103** | +0.010 |
| oxyT/r2scan3c/mtzvpp UKS | 1.046 | **0.841** | **-0.205** |
| oxyT/wb97mv/svp UKS | 1.108 | **1.145** | +0.037 |

**头部**：六格两会话全 ≥ 1.0 —— LC-145 成员资格**确证**。
**oxyT/r2scan3c 符号翻转（1.046 → 0.841），按预注册降级分支
从 fp64 占优或平价集合撤回**：sub-2.5s 噪声带内跨越平价的格，
符号跨会话不稳（均值 0.944 偏 auto 占优但任一单会话均不可引
标签；机制为比率复合——auto 0.852x × fp64 1.060x，LC-144 签
名，且其 auto 样本 [0.85, 0.72, 0.72] 首样本离群高）。
**修正后的 dev 闭合表述：91/96 格带可引完整栈标签（85 +
6 两会话确证）、1 格（oxyT/r2scan3c 48 AO UKS）已测但不确
定（sub-2.5s 噪声类、符号不稳）、4 个 methoxy 格按设计排除
（决策 (c)）**。第二：3c 方法的 sub-2.5s 噪声族增至**五格**
（LC-142 三重奏 + 本格）——3c 小 UKS 格系统性在噪声带内跨越
平价，方法级签名而非分子特异。第三：比率复合机制首次被抓到
**翻转符号**——引用规则要求 sub-2.5s 标签承载读数重复会话的
最有力实证。P5 如写 N/A 诚实记录（预注册缺陷：指的是 doc2-only
设计不测的 doc1 默认道）。判定 3/4 可测 HIT + P2 诚实 MISS
（降级分支即交付物）。测量-only；默认与门槛不变。证据：
`tasks/issue-9/2026-10-14-lc146-lc145-repeat-session`（34 文件
sha256 锚定）；effects-index 63 数据集。

### §80 续（LC-147, 2026-10-14）: 不确定格的第三会话决胜 —— oxyT/r2scan3c 读 1.076（多数 2/3 fp64 占优或平价），标签带三会话记录恢复

LC-146 留下的唯一不确定 dev 格（oxyT/r2scan3c，两会话符号不
一致 1.046 vs 0.841）按引用规则的决胜程序测第三会话。预注册
18:35:58Z（先于 18:36:15Z 运行）；LC-142 纪律逐字（仅 doc2 类）；
驱动 12 秒；1 份 VALID 文档、6/6 重复收敛、锚 e_tot 与第二会话
逐位 0.0、循环 9/9 与两会话均相同、文档内门 PASS。

**第三会话 frac 1.076 ≥ 1.0 —— 多数 2/3 为 fp64 占优或平价，
标签带三会话记录 [1.046, 0.841, 1.076] 与符号不稳定警示恢复**
（0.841 保持离群；两个一致会话 1.046/1.076 紧凑于 ~1.06；auto
样本再次呈首样本离群模式 [0.934, 0.847, 0.791] 递减）。判定
4/4 HIT（恢复分支）。

**最终 dev 闭合表述（取代 LC-146 的；LC-148 勘误附注）：92/96
格带可引完整栈标签（85 + 6 两会话确证 + 1 多数 2/3 恢复）、0 不
确定、4 个 methoxy 格按设计排除（决策 (c)）**——96 格分母为
dev+heldout 总数（72+24）：92 = 68 dev + 24 heldout；且其中
vitamin_c/r2scan3c 一格的可引文档是 LC-106 全 fp32-vmat 变体
（headtail 规范栈为 91/96——见 §80 勘误附注）。3c sub-2.5s UKS
噪声族
（五格）保持其刻画——本格为其首个**已解决**成员：多数 2/3
程序作为引用规则决胜器有效。证据：
`tasks/issue-9/2026-10-14-lc147-oxyt3c-third-session`（9 文件
sha256 锚定）；effects-index 64 数据集。

## 81. LC-149..151 (2026-10-14): vitamin_c/r2scan3c headtail 规范栈间隙闭合 —— LC-148 审计点名的唯一 headtail 间隙格；三会话 [0.943, 1.098, 0.964] 多数 2/3 亚平价 = 687 以下首个带 auto 占优标签的 3c dev 格（噪声带内）；frac 移动 1.253→0.94 机制完全归因（fp64 headtail 税 0.810x × auto 会话噪声）

**起点是 LC-148 的具名间隙，不是新假设**：LC-148 重导出发现
vitamin_c/r2scan3c/def2-mtzvpp（298 AO）的完整栈文档只有 LC-106
的**全 fp32-vmat 变体**（`SCF_VMAT_F32=1`——LC-108 测得 687-AO
不收敛后由 headtail 取代的变体），headtail 规范栈计数 91/96、该格
为具名未测间隙。LC-149 按LC-142 纪律逐字补测（预注册 19:02:37Z
sha256 eb8e8b6a 先于 19:03:07Z 运行；runtime 8694895 树干净；驱动
30 秒；1 份 VALID 文档、6/6 收敛、零回退、峰 ~1.5 GiB）。

| 会话 | fp64 e2e | auto e2e | frac | 循环 | 锚数值 |
|---|---|---|---|---|---|
| 1（LC-149） | 3.175 | 2.993 | **0.943** | 16/16 | — |
| 2（LC-150） | 2.915 | 3.202 | **1.098** | 16/16 | e_tot 双道逐位 0.0 |
| 3（LC-151） | 3.239 | 3.121 | **0.964** | 16/16 | e_tot 双道逐位 0.0 |

**头部发现**：

1. **LC-149 读 0.943 亚平价（P2 MISS = 预注册 informative 分
   支）**——规范 headtail 栈上 687 以下第一个亚平价 3c 格；但
   2.5-4s 道类（3.0-3.2 s）、距平价边际 0.057 << ~0.2，引用规则
   禁止单会话符号引用 → LC-150 复测（预注册 6b32d21f）读
   **1.098：对 [0.943, 1.098] 跨越平价 = 不确定** → LC-151 第三
   会话决胜（预注册 fbfa7521，LC-147 多数 2/3 程序）读 **0.964
   → 多数 2/3 亚平价：标签 auto 占优-在噪声带内**（符号不稳警
   示随附；1.098 为离群会话；两会话紧凑于 ~0.95）。多数 2/3 决
   胜器第二次使用、与首次（oxyT/r2scan3c 恢复 fp64 占优）方向
   相反——机制确认方向无关。
2. **frac 移动 1.253 → 0.94 机制完全归因（LC-149 的干净交付
   物）**：vmat 旋钮按 LC-105 设计仅作用于 fp64 道——auto 道
   e_tot 与 LC-106 **逐位相同**（0.0）、其 e2e 差为纯会话噪声
   （0.929x），fp64 道付 headtail-vs-fullvmat 尾税（0.810x =
   保留 SCF 增益 1.24x，LC-114"增益的 60-90%"读数在该格）；
   分解 1.253 × 0.810 × 0.929 = 0.943（残差 < 0.001）。
3. **LC-139 聚合声明修正（§75/§78 附注已加）**：原"完整栈下
   3c 方法 687 以下无 auto 占优 dev 格"部分依赖本格的全 vmat
   1.253；正确表述为"**除 vitamin_c/r2scan3c 298 AO（噪声带内
   auto 占优，边际 0.036-0.057）外**，3c 在 687 以下每格 fp64
   占优或平价"。vitamin_c 在默认栈的 auto 占优内部点（wb97mv
   0.972 族，LC-126/132）不是孤立方法类伪影——该分子在 3c 规
   范栈同样携带。
4. **循环与数值**：fp64 循环 17（fullvmat）→ 16（headtail）=
   headtail 不加循环的 LC-115 方向第三格；三会话循环 16/16 全
   部逐字相同；in-doc 门三会话全过（dE ~1.1e-12 / dg
   1.2-1.4e-6）；跨变体 fp64 dE -2.97e-10（全 vmat 位移反向，
   P3 HIT）、auto 逐位 0.0。
5. **P3 诚实记录（LC-150/151）**：单道漂移在这条 3.0-3.2 s 道
   上达 ±11%（LC-150 auto +7.0% 出 ±10% 带；LC-151 fp64
   +11.1% 出带）——2.5-4s 刻画的边缘情形，方向双向；比率复合
   归因在三个会话对上全部精确（残差 < 0.001）。

**headtail 规范栈面现以 92/92 文档覆盖 + 标签闭合**（92/96 格
带完整栈标签 = 68 dev + 24 heldout；4 methoxy 设计排除不变；
其中本格标签为噪声带 auto 占优）。判定：LC-149 3/5 HIT（P2
MISS 即交付物；P4 前提错位记录）；LC-150 3/4 HIT（P3 边际
MISS）；LC-151 3/4 HIT（P3 SPLIT）。诚实边界：三会话但每道
~3 s（2.5-4s 带 ±0.10-0.17 是标签永久警示；距平价边际
0.036-0.057 远在其内）；一台机器；同源系；预注册时间戳缺陷
（LC-151 的 registered_at 字符串早于 sha256 锁数秒——以
sha256 文件对为约束性预运行记录，已勘误注明）。测量-only；
默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-14-lc149-vitc3c-headtail-gap`（11 文件
锚定）、`.../2026-10-14-lc150-vitc3c-repeat-session`（10）、
`.../2026-10-14-lc151-vitc3c-third-session`（10）；effects-index
68 数据集。

## 82. LC-153/154 (2026-10-17): Codex review-20261017 两项 P1 缺陷修复 —— (a) 验证器未收到实际初始密度（dm0 携带，7d5e74a）；(b) fp32-VV10 对刚性平移不稳（fp64 原点减法，50a702c）；两修复均以审查者自己的复现器红→绿验证

**本节起点是 2026-10-17 独立 Codex 审查（基点 ee72b11，needs changes
但授权继续开发）点名的两个未修 P1**，非新假设。分支
`codex/fix-20261017-review-p1s` 自当前 HEAD（ee72b11）建出，保留全部
继承代码与证据；默认、门槛、验收状态无任何变化。

### LC-153 —— scf() 向两条验证路线转发实际初始密度

- **缺陷**：`gpu4pyscf/scf/hf.py::scf` 把显式 dm0（或由已有
  mo_coeff/mo_occ 重建的 dm0）传给 `_kernel`，但调用
  `verify_reference_match(mf)` / `verify_solution_stability(mf)` 时
  未传——验证器从 `mf.init_guess` 重新构造密度，违背"相同起点的
  FP64 轨迹"契约（不同起点可落入不同 UKS 分支）。审查者的 AST 级
  CPU dispatch 测试在 2 策略 × 2 初始密度来源的 4 个子场景全红。
- **修复（commit 7d5e74a）**：最小改动——把 dm0 转发给两个验证器
  调用（其 `dm0=None` 形参原本未被使用）；`dm0=None` 回退路径与
  之前行为逐位一致。
- **红→绿**：dispatch 测试 4/4 红（`red_proof.txt`）→ 4/4 绿
  （`dispatch_test_green.txt`）；新增真实 GPU 回归
  `ActualInitialDensityCarryTests`（`gpu4pyscf/scf/tests/
  test_stability_verify.py`）：(1) 显式 dm0 以同一对象到达两条验证
  路线、且调用方数组经完整 scf() 逐位不可变（huckel dm0 vs minao
  默认，water RKS b3lyp）；(2) 第二次 kernel() 调用转发由
  mo_coeff/mo_occ 重建的密度（缺陷的重用波函数半边）；(3) 在实测
  wb97m-v/methoxy 双盆格上，从显式 dm0（auto 分支收敛密度）出发的
  验证参考与同密度独立 fp64 道吻合至 1e-9 Ha，而 fp64-from-minao
  落在另一分支（实测分裂 ~1.28e-6 Ha）——修复前验证器把错误分支
  读作"参考"（非空性断言 >1e-7 分离）。全套件 45 通过 + 7 子测试。
- **不可变机制**：`_kernel_body` 每循环重绑 `dm`、只读初始密度，
  从不写入；`cupy.asarray(dm0, order='C')` 别名或复制但不改动。
- **范围诚实**：仅 opt-in 策略（auto+verify / auto+strict）；默认道
  不进验证分支、默认行为无变化。历史 auto+verify/auto+strict 记录
  中主计算同样从 init_guess 出发（无显式 dm0/无先前波函数）者内部
  自洽；存在显式 dm0 或先前波函数的记录受影响方向为"现在正确的
  参考归属"，本节未重标任何一条。

### LC-154 —— fp32-VV10 UWE 内核刚性平移修复

- **缺陷**：`vv10_fock_eval_UWE_fp32safe_kernel`（gpu4pyscf/lib/
  gdft/vv10.cu）在坐标差之前把**绝对坐标**转 float32：同一份收敛
  密度仅平移 NLC 网格（rho/weights/omega/kappa 不变）时分量级
  ΔE_NLC +5.273374983710832e-6 Ha（1e6 Bohr），fp64 warp1 ~7e-15
  ——审查卡实测数字，本轮用审查者自己的探针逐位复现。
- **修复（commit 50a702c）**：在 fp32 cast **之前**以 fp64 减去第
  一个网格点作为原点（i-load 与 j-shared 两处 staging）；原点随
  网格平移，任意公共平移在 fp64 减法中相消；staged 值保持分子
  尺度。仅重建目标 gdft；.so 前后 sha256 均留存（88c09a35 →
  41190f26）。
- **红→绿**（`.so` 交换 A/B、同会话、审查者探针 `probe_red_green.txt`）：
  修复前退出码 1（1e6 Bohr 处 5.273e-6）；修复后退出码 0（全部平移
  ≤5.35e-13 Ha）。
- **整 SCF 刚性变换验证（4 格，预注册 07:12:32Z）**：
  - P1 HIT——fp32-knob 道平移不变性恢复：|dE(T)-dE(0)| ≤ 4.4e-10 Ha
    （1e6 Bohr；methanol 2.3e-10 / benzene 2.8e-10 / inosine 1.8e-9 /
    oxyT 4.4e-10），与 fp64 道差 <1e-12；梯度不变性 ≤6.1e-8
    Eh/Bohr。
  - P2 HIT——精度门 40/40 道对全过（dE ≤4.1e-10 vs 门 1e-6；
    max|dg| ≤2.3e-8 vs 门 3e-5；同一变换设置两道）。
  - P3/P4 MISS（预注册预期错误、物理已识别）——**旋转**不变性受
    网格求积限制而非精度：dE(rot) 9e-8（oxyT）/4.1e-7（methanol）/
    2.0e-6（benzene/tzvp）/1.5e-5（inosine/svp），两道**相同**（能量
    响应差 <1e-12、梯度 4 位一致）——旋转使 Lebedev/Becke 求积重
    定向（LC-33 网格取向族），level-3 网格性质，非内核或修复缺陷；
    预注册不变性门在 benzene/inosine 旋转行**两道皆败**——按失败
    记录，未重述。精度相关的陈述是 P2（同设置道间一致），通过。
  - P5 部分——内核成本 +1.3-3.5%（最小网格 61k 点 +3.0% 触及
    预注册 3% 调查线；机制为 staged 坐标新增 fp64 减法 + 原点广
    播，相对成本随 staging 带宽饱和衰减）；不阻塞：修复前内核在
    刚性平移下数值错误，LC-80 的 ~12.7x 内核加速保留。
  - P6 HIT——UKS oxygen_triplet 全门通过（dE ~1.1e-10、
    dg ≤2.5e-10；首次尝试驱动 spin=1 不自洽——诚实执行错误，两
    次运行均保留）。
- **套件**：LC-80 等价套件 11/11 绿 + 3 新测试（静态原点钉扎 +
  GPU 分量平移不变性，门 1e-8，修复前失败 5.3e-6）；tools 530 通过。
- **范围诚实**：旋钮保持 OPT-IN（`GPU4PYSCF_VV10_UWE_KERNEL=fp32`）；
  fp64 内核未动（warp1 探针前后一致）；旋转求积噪声（9e-8..1.5e-5）
  属网格定义性质，未改任何门。

两修复后 10/17 的新测量（LC-155/156、LC-158）全部运行在 50a702c
上——fp32-VV10 旋钮路径携带修复后的内核。判定均按各预注册；证据
`tasks/issue-7/2026-10-17-lc153-dm0-verify-carry`（dispatch 绿、
红证、sha256 锚定）、`tasks/issue-9/2026-10-17-lc154-vv10-
translation-fix`（16 文件含 .so 前后二进制与探针记录）。

## 83. LC-155/156 (2026-10-17): 随机相邻配对跨块计时记录 —— LC-155 试点定协议（最大道内散布 5.51%）；LC-156 完整 3 块 × 6 格 × 3 路线 × 3 样本 = 54 文档全过，每格 frac 以区间复现冻结标签（无多数票）；oxyT 默认道按预注册符号翻转条款报平价跨骑；LC-157 G2 后备清单冻结

**起点是审查卡 P2 的计时纪律要求**（"计时噪声不能靠多数票消除；
下一轮使用相邻配对、固定种子随机顺序、真实 CUDA 同步、预热隔离
…报告原始时间、散布及配对比值区间，不只报多数标签"），不是新假设。
运行时 50a702c（§82 两修复后）；预注册 08:07:22Z 先于首块。

### LC-155 试点（15:45-16:05 本地，预注册 07:44:52Z）

vitc3c（2.5-4s 噪声族）+ inosine（~100s NLC 格）双格定标：道内散布
0.4-5.5%；相邻配对协议下跨文档 fp64 稳定性亚 1%；vitc 跨日锚
-2.6%/-0.2%、e_tot 逐位复现 → 决策规则触发 **3 块 × 3 样本全 6 格**
的完整记录。试点文档即 block-1 的 vitc/inosine 样本（块内顺序用了
试点 seed-20261017 抽签而非 20261018——已记录偏差，配对性不受
影响）。

### LC-156 完整记录（54 文档 / 324 道样本，全部收敛、in-doc 门全过）

| 格 (AO) | 循环 | frac_default (b1,b2,b3 → 区间) | frac_stack (b1,b2,b3 → 区间) |
|---|---|---|---|
| vitamin_c/r2scan3c (298) | 16 | 0.862, 0.913, 0.900 → **[0.862, 0.913]** | 0.987, 1.017, 1.013 → **[0.987, 1.017]** |
| oxygen_triplet/r2scan3c (48) | 8/9 | 0.969, 1.077, 0.963 → **[0.963, 1.077] 平价跨骑** | 1.103, 1.059, 1.072 → **[1.059, 1.103]** |
| ethoxy/b3lyp UKS (67) | 11 | 1.002, 1.161, 1.141 → **[1.002, 1.161]** | 1.118, 1.052, 1.222 → **[1.052, 1.222]** |
| bisphenol_a/b3lyp (318) | 13 | 0.751, 0.745, 0.754 → **[0.745, 0.754]** | 0.910, 0.899, 0.942 → **[0.899, 0.942]** |
| inosine/wb97mv (326) | 17 | 0.975, 0.973, 0.972 → **[0.972, 0.975]** | 0.963, 0.948, 0.960 → **[0.948, 0.963]** |
| tamoxifen/wb97mv (537) | 13/14 | 1.015, 1.015, 1.016 → **[1.015, 1.016]** | 0.932, 0.940, 0.937 → **[0.932, 0.940]** |

**头部读数**：

1. **全部 6 格以区间复现冻结标签，无规则性标签变化**——tamoxifen
   默认平价/栈 auto 占优（锚 1.014/0.929，跨块散布 0.001！）；
   bisphenol 默认 auto 赢族/栈平价类；inosine 两路线 auto 占优；
   vitc 默认 auto 占优/栈平价（LC-151 噪声带类）；ethoxy UKS
   fp64 占优或平价。同路线优化增益块稳定：NLC 6.64-7.27（VV10
   墙）、非 NLC 1.20-1.69；strict/stack-auto 1.85-2.57。
2. **oxyT/r2scan3c 默认道是唯一平价跨骑格**（b2 读 1.077、b1/b3
   读 0.969/0.963）——预注册符号翻转条款触发：报区间**不投票**
   （其 3c/UKS sub-2.5s 噪声族特征与默认栈三会话记录
   [1.046, 0.841, 1.076] 一致）。
3. **协议本身是交付物**：≥10s 道跨块稳定性最高 5.4%（tamoxifen
   0.1-0.8%、inosine 0.4-1.7%），~3-5s 道 4.1%，sub-2.5s 道
   8.5-27.4% = 已刻画会话噪声类——相邻配对协议在 ≥10s 道上把跨块
   稳定性压到个位数百分比（单文档协议历史散布 9-15%）。
4. **跨日锚复现**（block 1 vs 冻结记录）：tamoxifen 默认 1.015 vs
   1.014 / 栈 0.932 vs 0.929、fp64 t_scf 208.6 vs 207.86 s；
   bisphenol 默认 [0.745,0.754] vs 0.762；vitc 栈 [0.987,1.017] vs
   0.964；inosine 栈 [0.948,0.963] vs 0.937/0.948——全部在道类
   引用带内。

**判定（vs 08:07:22Z 预注册）**：P1 HIT 6/6（每格默认 frac 区间在
冻结标签带内）；P2 HIT 6/6（栈 frac 同）；P3 HIT（fp64-SCF 路线
增益全 ≥1.0、NLC ≥4x、vitc 1.44、ethoxy UKS 顶 1.668 略超 1.6
带顶——informative helps-more）；P4 HIT（54/54 收敛、门全过、
strict 全过；最差 in-doc dE 3.73e-11）；P5 HIT（≥10s 道 ≤6%、
~3s 道 ≤15%，符号翻转条款如上触发）；P6 HIT（报告契约：原始
样本+中位+分块值+双增益区间，无多数票）。

**诚实边界**：一台机器；三块同日相隔 ~3-7 h（跨块非跨日——跨日
锚是上方冻结记录）；b3_tamoxifen_default 在前一会话 4 h 安全顶
处被杀（auto 道已完 E=-1136.632915551171、fp64 样本 3/3 在飞）——
部分输出原样保留于 `killed-20261017T1056Z/`、该文档从头重跑且落
在 b1/b2 族内（1.0158 vs 1.0154/1.0148）；54 文档 × 3 样本、不做
统计功效声明。

### LC-157 —— 中等 CHON 扩充筛查与冻结的公开后备清单

审查卡授权最多 12 个新中等 CHON 分子（公开可溯源几何）。离线清查：
ASE G2 源（队列自身溯源）上限 14 原子 = 小带；仓库其他几何越界
（Azadirachtin 95 原子无 N；r14 含 Ru）；G16 目录专有排除 →
**中等 CHON 带在离线服务器不可达，已记录操作员请求**（供给公开
中等 CHON 几何，或确认 G2+TZVP 后备为预期范围；未做任何网络动作）。
**冻结 12 分子 G2 后备**（`g2_fallback_screen.json`；CPU pyscf 筛查
闭壳层 0/0、AO 计入表；复核保留子集 C5H8/C3H9N/cyclobutane/H2CCHCN
不用于调参）。材料化时（`inventory/gpu4pyscf-g2-fallback-20261017-v1`，
manifest+几何 sha256，natm/symbols/AO 与冻结筛查逐项交叉验证通过）
发现 LC-157 相位记录的两处化学描述笔误并已勘误：**C5H8 = 螺戊烷**
（spiropentane，D2d；非"炔二烯"）、**C3H9N = 三甲胺**（非"氮丙
啶"；氮丙啶是 CH2NHCH2）——几何/AO/电荷自旋不受影响。后备矩阵
（12 分子 × 4 方法 × default+stack = 96 文档）预注册
2026-10-17T11:03:42Z 后以 6 个有界子单元在 LC-156 完成后自动开跑
（LC-158；结果见后续节）。

测量-only；默认与门槛不变；全部旋钮 opt-in。证据：
`tasks/issue-9/2026-10-17-lc155-paired-timing-pilot`（相位记录+
分析+6 文档，锚定）、`tasks/issue-9/2026-10-17-lc156-paired-
timing-full`（255 文件锚定，含 54 文档、3 驱动日志、killed-20261017T1056Z/、
aggregate_lc156.py）、`tasks/issue-9/2026-10-17-lc157-chon-expansion-screen`
（筛查+材料化脚本+勘误）；effects-index 73 数据集。

**§83 补（LC-156 block 4，2026-10-18）：跨日区间扩展块完成——4 块
记录闭合（72 文档 / 432 lane-sample）**。块 4（seed 20261021 两级
随机顺序；preregistration_block4.json sha256 0223688d，14:04:58Z 于
块 3 前注册）由**staged 门限等待器在 2026-10-18T00:00:15Z 自动发射**
（跨日要求满足：块 1-3 于 10-17、块 4 于 10-18 UTC；驱动 01:07:58Z
结束 = 67.7 分钟，落在预注册 [62,70] 带内；18/18 文档 rc=0，
108 lane-sample；tamoxifen default 文档 1913.6 s 为墙钟主导，在其
2400 s 限时内；峰值主机 RSS 1.30-1.34 GB；温度 ≤38 C）。**裁定
（对预注册 P1-P5）：P2 HIT**（每道全收敛、in-doc 门全过，strict 道
过）；**P3 HIT**——30/30 lane-中位稳定条目全在带内（≥10s 道最大
|dev| 1.7% inosine / 0.5% tamoxifen = 大道跨日极稳；~3s 道最大
12.2%；oxyT stack 两条 sub-1s 道噪声类按设计 within=None）；
**P1 SPLIT——12 格-路线中 11 在 blocks-1..3 ±0.05 带内，1 条越带
按预注册记为跨日漂移发现（不重述）：oxyT default frac 1.1981 越带
[0.913, 1.127]**——该格本就携带块 1-3 的预注册平价跨骑标签
（[0.963, 1.077]），块 4 读数保持跨骑且偏上，其两条底层道在各自
P3 带内（-9.3%/+12.2% = 比值复合，sub-2.5s UKS 会话噪声类）；
**P4 HIT**（原始样本+逐块中位+4 块区间，per_cell.four_block_intervals，
全程无多数票）；**P5 HIT**（67.7 分钟）。**4 块 frac 区间读数：
tamoxifen default 1.0157（区间 [1.015,1.016] 原样延展）/stack
0.9335（auto 占优）；bisphenol default 0.785（带内贴边）/stack
0.915；inosine 两路线 auto 占优（0.9708/0.9373）；vitc3c
0.9287/1.036；ethoxy UKS 1.1003/1.0197；oxyT stack 1.0708 带内，
oxyT default 上述唯一越带格**。同路线优化收益（块 4）：inosine
7.01x/7.26x、tamoxifen 6.63x/7.21x（VV10 墙类），非 NLC
1.37-1.63x；strict_over_stack_auto 2.0-2.4x（与块 1-3 一致）。
**执行注记**：分级脚本 aggregate_lc156_block4.py 的 g() 助手存在
键路径缺陷（嵌套 dict 上按扁平 'doc/lane' 查找，staged 会话遗留、
真实数据到达才暴露），首次运行当场捕获并修复，零数字从缺陷路径
报出——已记录于相位记录。至此审查卡 2026-10-17 的跨日配对计时
要求完成（4 块、区间报告、无多数票）。测量-only；默认与门槛不变；
全 knob opt-in。证据：块 4 增量文件（b4_*.json 18 + raw + 日志 +
block4_seed_order.json + preregistration_block4.json + 发射器 +
aggregate_lc156_block4.py + block4_analysis.json + monitor_block4.sh）
并入 `tasks/issue-9/2026-10-17-lc156-paired-timing-full`
（evidence_sha256.txt 重生成为 353 文件全超集，全验证通过）；
effects-index 99 数据集（lc156-block4-crossday）。

**§83 续（LC-195，2026-10-18）：oxyT 跨日漂移发现的第三日裁定 +
运行中时钟/温度采样交付（审查卡 item-4 记录要求闭合）**。预注册
028e5dcd（运行前入 checkpoint）；运行 05:17:38Z-05:21:12Z，3.5 分钟，
seed 20261022 两级随机顺序，**仅三个 sub-2.5s-lane 小格**（oxyT 漂移格
+ vitc3c/ethoxy 家族对照）× default+stack+stack_strict = 9 文档
（协议逐字复用块 4：同 runner、--samples 3、同 STACK_ENV/几何/基组/
方法/charge/spin），全程 **1-Hz nvidia-smi 运行中采样**（212 条：
时间戳/温度/SM 时钟/功率/利用率/显存）。9/9 rc=0，18/18 道收敛、
in-doc 门全过（最差 dE 4.547e-12、最差 max|dg| 9.164e-6）。
**裁定（对预注册 P1-P6）：P1 SPLIT/贴边-IN——第三日 oxyT default frac
1.0979：高于 b1-3 区间顶 1.0765 仅 2.0%、低于块 4 值 1.1981 达
8.4%、低于预注册持续性阈值 1.127 → 日级持续性被否证**；块 4 的越带
是当日 auto 道偏慢（1.163 s → 1.039 s，-10.6%；fp64 稳定 -2.5%），
该格保持平价跨骑标签，5 值包络 [0.9632, 1.1981]，无可引用方向、
无多数票。**P2 MISS（有结构）**：ethoxy 两路线带内（1.0317/
1.1891）；vitc3c stack 与 oxyT stack 贴边 +0.4%（1.0397 vs 1.036、
1.1074 vs 1.1026）；**vitc3c default 0.7938 = 低于 4 值并集
[0.8624, 0.9287] 达 7.5%**——两道同向移动（fp64 +5.6%、auto
-9.7%）于本块最长小格道（4.1-5.2 s）。**纯计时现象：全部 12 条 3c
道能量对块 4 逐位复现（偏差 0.000e+00）、ethoxy UKS ≤1.137e-13、
18/18 道循环数与块 4 完全一致（±0）**。**P4 HIT**——无降频事件：
计算样本全程 ≥2565 MHz（210 MHz 行为块前空闲 + 首文档爬升，现已
显式记录在案）、23-36 C、峰值 143.63 W；每文档运行中窗口
（12-48 条/文档）全部入 block5_analysis.json。**P5 HIT（±0，优于
±1 预算）、P6 HIT（3.5 分钟 ≤15）**。**P3 HIT**。附带 Part A
（零 GPU）：块 4 驱动日志的温度/时钟采样从未被聚合（analysis 的
temp_c_range/clocks_mhz_range 为 null，数据一直在日志里）——勘误
记录，18 文档解析入 `block4_thermal_context.json`（24-39 C、
块内 2565-2745 MHz、无逐文档不对称；粒度为文档前后，非运行中）。
**读数修订**：跨日单道中位漂移实测带——sub-2.5s 道 ±10-15%、
~4-5 s 道 ±5%（单条 -9.7%），能量逐位、循环零漂移 ⇒ frac 级跨日
弥散至 17%（vitc3c default 5 值序列 [0.7938, 0.9287]）——**~4-5 s
道的 frac/方向主张同样需跨会话复现后才可引用**（对 §78 会话噪声
规则的一级上延）。本会话前置审计：ee72b11 之后未复核源面（恰好
4 个源文件：molframe 旋钮+对称陀螺破结、vv10.cu 平移修复、hf.py
dm0 携带及其测试）逐行复核**未发现缺陷**（线性/球陀回落、右手系、
逐符号表旋转、NLC 网格经同一 build、scanner reset、UHF 经同一
scf() 分发；tools 530 测试全绿于 ada6877）。测量-only；默认与门槛
不变；全 knob opt-in。证据：
`tasks/issue-9/2026-10-18-lc195-oxyt-drift-third-day`
（53 文件 sha256 锚定）；effects-index 108 数据集
（lc195-oxyt-drift-third-day）。

**§83 续2（LC-196，2026-10-18；预注册 80f76d8e 于运行前 05:37:05Z）**：
**中带运行中时钟/温度/功率采样（审查卡 item-4 在 LC-195 小格带之外的
补全）**——LC-156 六格带中两个持续负载 NLC 中格：inosine 326 AO ×
{default, stack, stack_strict} + tamoxifen 537 AO × {default, stack}
（bisphenol/b3lyp 设计排除：非 NLC、默认道短于 tamoxifen = 热学域被
支配；tamoxifen/strict 同理省略，inosine/strict 携带 strict 路线窗口）；
LC-156 协议逐字（同 runner、每道预热 + 3 样本、同 STACK_ENV/几何/设置），
seed 20261023 两级随机序，全程 1-Hz 采样（3290 行，最大间隙 1.07s）。
分离式串行 05:37:29Z-06:32:21Z = 54.9 分钟，5/5 rc=0，10/10 道收敛、
in-doc 门全过，循环精确复现冻结族（inosine 17/17；tamoxifen 13 fp64/
14 auto），最差 |dE| 2.41e-11、最差 max|dg| 1.13e-5（均在 LC-156 家族
内），inosine stack 记录 e_tot bitwise。**P1 按预注册判定 MISS，结构
决定性**：预注册判据（每文档窗 <60 C、计算相时钟 ≥2565 MHz、≤250 W）
在 3/5 文档破——**破的全是 stack 文档**：default 路线的持续 fp64 道
（100s/240s VV10 墙负载）**完全没有节流签名**（计算相时钟最低
2730 MHz、≤40 C、≤138.5 W，见 lane_phase_trace.txt）= LC-156 中带跨
日稳定性无热学混杂；stack 路线更热（60-62 C、峰值 317.3 W）而计算相
时钟 2520-2535 MHz = **RTX 4090 额定加速时钟底**（基础 2235；无一
样本低于额定加速；62 C 远离 ~83 C 节流区；317 W < 450 W 板限）——
中带 fp32 内核族的 V/f 负载域（小带 stack 文档峰值功率仅 143.6 W），
**非热学节流**。LC-195 的 2565 MHz 小带计算底**不外推**到中带 stack
道；正确判据为"不低于额定加速 2520 MHz"。**每个 stack 文档两道共享
同一热学域**（30 秒桶迹）= 相邻配对 frac 内部一致；default 道
2730-2745 MHz 与 stack 道 2520-2730 MHz 的 ~7-8% 跨路线时钟域差是
会话稳定的（本会话 9 条锚定道中位全部 ±1.0% 复现 LC-156）——历史
default-vs-stack 增益不因此失效，但记为已命名、已测量的保守告警
（方向：低估 stack 增益）。**P2 HIT**：9/9 锚定道中位 ±1.0% 内复现
LC-156（inosine 默认 100.99/98.77 vs 101.00/98.49；stack 14.25/13.66
vs 14.32/13.79；strict 29.33 vs 29.29；tamoxifen 默认 236.16/240.01
vs 237.54/241.20；stack 35.66/33.41 vs 35.79/33.35）；frac 仅记上下文
（inosine 0.976/0.958、tamoxifen 1.016/0.937，均在冻结区间内，无新
标签）。**P4/P5 HIT**（5/5 rc=0、采样无 >2s 间隙、墙 54.9 分钟、峰值
显存 10210 MiB、host RSS 1.31-1.33 GiB）。**读数**：(1) 审查卡
item-4 的运行中仪表化现已覆盖全部已测尺寸带（LC-195 小带 48-298 AO +
本中带 326/537 AO）；(2) 未来计时轮的 stack 道无节流判据应使用额定
加速 2520 MHz 底；(3) 跨路线（default vs stack 道时间）比较携带
~7-8% 时钟域差，已命名。本会话另以两审查复现器在 HEAD c290d03 复验
LC-153/154 修复绿（test_independent_initial_guess 4/4 子场景 OK；
probe_vv10_translation exit 0，fp32 平移增量 ≤5.35e-13 Ha @1e6 Bohr，
与 LC-154 修复后记录同数字类）——前会话审计为源码审查 + tools 套件、
未复跑复现器，本单元补上该缺口。测量-only；默认与门槛不变；全 knob
opt-in。证据：`tasks/issue-9/2026-10-18-lc196-mediumband-thermal`
（35 文件 sha256 锚定）+ `tasks/issue-9/2026-10-18-head-green-confirmation`
（4 文件 sha256 锚定）；effects-index 110 数据集。发件箱保持在
ada6877 发射（本段折入下次发射）。

**§83 续3（LC-198，2026-10-18；预注册 8e2691de 于运行前
08:02:30Z）**：**687 AO 带顶跨日配对计时块 + 运行中热学仪表化——
审查卡 item-3/4 覆盖闭合到完整准入带 48-687 AO 的最后缺失尺寸类**。
单元 = 准入带顶 tamoxifen/r2scan3c/def2-mtzvpp 687 AO（57 原子/200
电子闭壳层；几何 sha256 f272e89c 与冻结 20261013 清单及 LC-115 的
057_Tamoxifen.xyz 逐位一致）：唯一既无跨日配对块（LC-156 六格止于
537）也无运行中仪表化（LC-195 小带 48-298 / LC-196 中带 326-537）的
已测尺寸类；其标签冻结于 2026-10-13 单会话——默认道 frac 0.908
（LC-117 o52）、完整栈 1.141（LC-115 headtail 文档；LC-122 的读数是
**提取**，从未整块重跑），均 (b5) 相关（fp32 安全括号顶、headtail
旋钮边界 [537,687)）。LC-156/196 协议逐字（生产 gradient_runner、
每道预热 + 3 新鲜进程样本、--pilot-stage S2、default/stack/stack_strict
三路线文档；STACK_ENV 为完整栈逐字，其中 VV10_UWE 对 3c 为无操作
——D4 无 VV10，VMAT_F32=headtail 即使 687 fp64 SCF 收敛的旋钮），
全程 1-Hz 采样（499 行，最大间隙 1.06 s），seed 20261024 抽序
[stack, default, stack_strict]。分离式串行 08:03:32-08:11:51Z =
8m19s（文档 125.8/167.1/205.7 s），3/3 rc=0、三文档全 VALID、6/6 道
收敛、in-doc 门全过（最差 |dE| 1.273e-11、最差 max|dg| 1.056e-5），
峰值显存 6,868-8,170 MiB（strict 道最高）、host RSS 1,207-1,316 MiB，
源 HEAD 221143c（r2scan3c 路径不受两项审查 P1 修复影响）。**裁定
6/6 全 HIT**：P1 默认道 frac 0.8979（带 [0.80,1.01]；冻结 0.9080，
-1.1%；fp64 道 +0.4%/auto 道 -0.7% 对 5 天前中位）；P2 完整栈 frac
**1.1469**（带 [1.04,1.24]；冻结 1.1405，+0.56%）= 带顶 (b5) 标签
"完整栈下 fp64 便宜 ~15%" 的**首次新鲜同文档测量**且跨日确认；
P3 正确性（fp64 e_tot 锚：默认道偏差 0.0 **逐位**复现 LC-117、
栈道 9.09e-12 对 LC-115；四锚定道循环精确 14/19/14/17）；P4 协议；
P5 热学（LC-196 额定加速判据在带顶成立：计算相时钟 2565-2730 MHz
全 ≥2520、24-43 C、峰值 210.4 W）；P6 成本（8m19s ≤30）。**读数**：
(1) 带顶四锚定道全部 ±0.72% 内复现 5 天前中位——该格类最紧跨日
复现水平；(2) 同路线 fp64 优化收益 default→完整栈 **1.529x** e2e
（t_scf 1.36x × t_grad 3.93x，与 LC-115/LC-117 分解 1.11×1.37 一致）；
(3) **auto 道在本格非栈不变**：stack/default auto 比 0.835（循环
19→17 = blocksize 循环效应）= LC-122 的 537 发现（0.83）在 687、
3c 方法上复现——**预注册 design_choices 文本勘误**（wb97mv 格的
auto 不变性不迁移；无预测受影响，P1/P2 带锚定在冻结 frac 上且全
HIT）；(4) strict 路线参考-SCF 主导（36.26 s ≈ LC-117 默认-strict
37.81 s；栈旋钮不加速参考道 = LC-93/96 族）；(5) 热学：本格道为
14-21 s 突发负载（非 100-240 s 持续 NLC 载荷）——默认文档计算底
2565 MHz = LC-195 小带底重现，stack 文档 2700-2730 MHz **高于**中带
NLC stack 的 2520-2535：3c 内核组合（无 VV10 fp32、D4、headtail
vmat）仅耗 204-210 W vs NLC stack 317 W，额定加速判据处处干净成立。
**至此审查卡 item-3/4 覆盖完整准入带 48-687 AO**：六命名格
（LC-156 b1-b4 + LC-195/196）+ 带顶（本单元，第三内核组合类）。
诚实界限：单格单跨日块（day-1 = 冻结 2026-10-13 记录）；stack_strict
路线无同格冻结锚（LC-115 未跑 strict，36.26 s/cyc 17 为新鲜记录）；
每道 3 样本工程覆盖范围；无多数票。测量-only；默认与门槛不变；全
knob opt-in。证据
`tasks/issue-9/2026-10-18-lc198-687ao-bandtop-crossday`（24 文件
sha256 锚定）；effects-index 112 数据集（lc198-687ao-bandtop-crossday；
lc156 的 1 条已文档化前勘误摘要保持 LC-197 惯例不重写）。发件箱保持
在 221143c 发射（本段折入下次发射）。

## 84. LC-158 (2026-10-17): 冻结 G2 后备矩阵 —— 12 新分子 × 4 方法 × default+stack = 96 文档 48 格全过门；G2 默认栈面平价居中（cohort 的 TZVP 默认 auto 赢不迁移 = 逐分子依赖）；3c/mtzvpp 小带双栈态皆 fp64 占优或平价（默认中位 1.122）；wb97mv VV10 墙 3.12-5.26x 复现

**起点是 §83 冻结的 LC-157 后备清单**（预注册 11:03:42Z；数据集
`inventory/gpu4pyscf-g2-fallback-20261017-v1`，几何 sha256 + AO 与
冻结筛查逐项交叉验证；两处化学描述勘误：C5H8=螺戊烷、C3H9N=三甲
胺）。运行时 fc6199b（runner 源 50a702c，两 P1 修复在内）；
11:49-12:27 UTC，run_lc158.sh 分离式（LC-156 完成等待门）6 个有界
子单元；96/96 文档 rc=0，192/192 道收敛，48/48 格 in-doc 门全过；
GPU 25-39 C 稳定，最大峰 1040 MiB（预算内）。

| 方法（AO 带） | default frac 范围（中位） | <1.0 | stack frac 范围（中位） | <1.0 |
|---|---|---|---|---|
| b3lyp（123-203 tzvp） | [0.883, 1.197]（1.028） | 5/12 | [1.024, 1.244]（1.106） | 0/12 |
| m062x | [0.870, 1.195]（0.964） | 7/12 | [0.975, 1.193]（1.085） | 2/12 |
| wb97mv | [0.977, 1.174]（1.011） | 3/12 | [0.938, 1.158]（1.058） | 1/12 |
| r2scan3c（87-135 mtzvpp） | [1.034, 1.383]（1.122） | 0/12 | [0.934, 1.352]（1.117） | 2/12 |

**头部读数**：

1. **P1 informative MISS = 主交付物**：预测 G2 带默认 frac"多数
   auto 占优 [0.70,1.02]"，实测 48 格**平价居中**（中位 1.039、
   33/48 高于 1.02、最低 0.870 ≥ 0.65 子声明 HIT）——冻结 cohort 在
   TZVP 117-216 AO 的默认 auto 赢（LC-117/118）**不迁移**到 G2 化
   学：默认栈交叉是**逐分子依赖**（LC-127 的完整栈逐分子性延伸到
   默认栈 + 新化学）。默认期望的工程规则须按化学域限定，不能按
   AO 带一刀切。
2. **3c/mtzvpp 小带（87-135 AO）双栈态皆 fp64 占优或平价**——默认
   12/12 全 ≥1.034（中位 1.122；环丙烷 87 AO 1.383 最 fp64 占优，
   与 LC-136 甲醇-63-AO 1.318 完整栈族一致，该读数现延伸到默认栈）。
3. **完整栈非 NLC 平价簇扩展**：44/48 stack frac 在 [0.975,1.244]、
   b3lyp 0/12 低于平价——LC-133/134/135 的 fp64 占优或平价读数迁
   移到 12 个新分子与两个新化学族（张力环：螺戊烷/环丁烷/环丙烷；
   醚+仲醇）。
4. **wb97mv VV10 墙**：t_scf 3.12-5.26x（12/12，最小 7 原子丙烯
   腈、最大环丁烷）——LC-128 TZVP 墙量级族延伸；非 NLC e2e 增益
   1.30-1.71x（P5 带预测 1.0-1.4 **偏高 MISS**——梯度旋钮在新鲜
   几何上复合更大，helps-more，门全过无精度后果）。
5. **无新失败族**：0 非收敛、0 门败、0 回退；跨栈 fp64 dE ≤6.64e-11
   （3c 6/12 逐位 0）、max|dg| ≤3.41e-6、循环差 ≤1 每格每道。

**判定（vs 11:03:42Z 预注册）**：P1 MISS informative（子声明 HIT）；
P2 HIT（m062x 两格 0.975/0.980 落在 0.98 地板 = 噪声类，不计 auto
占优）；P3 HIT（一格 0.938 噪声类亚平价）；P4 MISS-low 2 格
（0.961/0.934，sub-2.5s 道类）；P5 SPLIT（门全过 + wb97mv 12/12
HIT + 非 NLC 增益带偏高 MISS）；P6 HIT。

**诚实边界**：单会话、1-4 s 道 = 噪声类引用规则（边际 <0.10-0.25
不作符号声明；3c 亚平价两格与 m062x 地板两格为重复会话确认候选，
未贴标签）；复核保留子集（C5H8/C3H9N/cyclobutane/H2CCHCN）同协议
测得但**保留不用于调参**——未来验证单元可盲重跑。

测量-only；默认与门槛不变；全部旋钮 opt-in；无验收声明。证据：
`tasks/issue-9/2026-10-17-lc158-g2-fallback-matrix`（96 文档+raw+
日志+脚本+analysis.json+预注册，487 文件 sha256 锚定）；
effects-index 74 数据集。

**§84 附注（LC-159，2026-10-17 12:35:40Z 预注册；LC-142 纪律
doc2-only）**：上述 4 个噪声类亚平价/地板格已完成**重复会话确认**
（4/4 文档 rc=0，门全过；跨会话 e_tot 在两个 3c 格逐位 0.0、其余
≤2.8e-14，循环全同）。两 Session 对 (s1, s2)：**isobutene/r2scan3c
(0.961, 0.884) 双 < 0.98 → 亚平价两 Session 确认**——G2 集上首个
完整栈 auto 占优格（3c 方法，其噪声族的方向性成员）；CH2NHCH2/
r2scan3c (0.934, 1.115)、CH2NHCH2/m062x (0.980, 1.111)、
C3H6_Cs/wb97mv (0.938, 1.044) 均**跨越平价 → session-1 读数为噪
声，不贴标签**（无多数票）。C5H8/m062x 按 recheck 保留排除（0.005
地板边际无需确认）。P4 修正后的完整栈 G2 面读数：44/48 在
[0.975,1.244] + 1 确认亚平价 + 3 噪声格 + 0 其他确认 auto 占优。
判定 P1-P4 4/4 HIT（P1 每格 s2 距 s1 ≤0.181 < 0.20 带）。首次尝试
驱动引号 bug（--lane 单 token）在 attempt1-quoting-bug/ 原样保留，
未耗 GPU。证据 `tasks/issue-9/2026-10-17-lc159-noisecell-repeat`
（34 文件锚定）；effects-index 75 数据集。

## 85. LC-160（2026-10-17）：冻结 G2 化学上的刚体变换验证 —— LC-154 修复的平移不变性在新化学上泛化（fp32knob 平移不变 ≤4.6e-10 Ha；60/60 精度门全过；旋转仍是双 lane 相等的网格求积类）

**§85（LC-160，2026-10-17 12:44:21Z 预注册；12:45:21-12:55:45 UTC
分离式运行，10.4 分钟墙钟；LC-154 协议逐字复用）**：把 LC-154 的
fp32-VV10 起点减除修复（50a702c；libgdft.so 41190f26 后修版本）的
整分子刚体平移/旋转能量+梯度验证从 4 个 cohort 格扩展到冻结 G2 后备
化学（审查卡队列第 5 项的刚体变换族；next_steps 候选 (b)）。6 个
非保留 G2 分子 × wb97m-v/def2-tzvp DF grids-level-3 conv_tol 1e-10
（isobutene 172 / 环丙烷 129 / 乙甲醚 172 / 异丙醇 172 / 氮丙环 123 /
乙二醛 136 AO；4 个 recheck 保留分子按冻结策略排除设计外；G2 全
闭壳层故无 UKS 格——LC-154 已在 oxygen_triplet 覆盖 UKS）。每
(格, 变换) 两条新鲜子进程 lane——fp64（旋钮未设）vs fp32knob
（GPU4PYSCF_VV10_UWE_KERNEL=fp32），均 precision_mode='fp64'，UWE
内核为唯一差异；变换与 LC-154 完全一致（identity、trans_1e3、
trans_1e6、rot 轴(1,2,3) 角 0.7 支点[100,-50,25]、rot_trans =
rot+1e5 Bohr）；60 条 lane-run 全部收敛（每条 900 s 超时上限，
串行单 GPU，24-32 C）。

**结果（worst-of-6，analysis.json）**：

- **平移不变性（修复的契约）泛化**：fp32knob |dE(T)-dE(0)| 最大
  5.4e-11 Ha @1e3 Bohr、4.6e-10 @1e6 Bohr（预注册带 ≤5e-9）；
  dg 不变性 ≤1.8e-8；两条 lane 平移响应互差 ≤5.0e-11 —— fp32 内核
  在新化学上不贡献任何可测平移误差。
- **精度门 60/60 全过**：|dE| ≤8.9e-11（门 1e-6），max|dg| ≤2.8e-7
  （门 3e-5）——LC-158 的 wb97mv G2 行（已端到端跑过修复后内核）是
  平移安全的；新化学上不存在仅 fp32knob 失败的格（无修复缺陷）。
- **旋转仍是网格求积类（双 lane 相等）**：dE(rot) ∈[4.0e-7,
  2.84e-6]，双 lane 能量响应互差 ≤7.1e-11、旋转梯度互差 ≤2.8e-8；
  isobutene 与乙甲醚的 rot/rot_trans 行在**两条 lane 上**同值失败
  1e-6 不变性门（2.845e-6 / 2.710e-6），全部 6 格的旋转梯度一致性
  失败 3e-5 门（2.8e-3..3.3e-2）——20 条失败如实记录未改写；与
  LC-154 cohort 相比 G2 量级更小（max 2.8e-6 vs inosine 1.5e-5，
  网格更小所致），乙二醛的旋转梯度响应最大。
- rot_trans − rot ≤9.8e-11（平移分量复合归零，旋转主导）。
- 循环在每条 lane 内变换不变（氮丙环 fp32knob 的 11 循环 lane
  属性在三个未旋转几何上恒定；rot 11→10 为唯一 ±1）。

**判定**：P1 实质 HIT + fp64 子带 MISS（1e6 处实测 4.7e-10 vs 预注册
1e-11——预注册带宽设置错误而非物理发现：LC-154 家族本身在 1e6 Bohr
即 e-10 级 [methanol 2.3e-10 … inosine 1.8e-9]，该残差是共享的坐标
表示噪声，由 lane 一致性证明共享）；P2/P3/P4/P5 HIT；P6 PARTIAL
（峰值显存仪表缺口——GPU_SNAP 探针位于运行间隙只读到空闲值，按
LC-158 同分子实测 1040 MiB 上界记录）。按预注册决策规则：P1 实质 +
P2 全过 → **修复的整 SCF 平移不变性泛化到冻结 G2 化学**，
next_steps 候选 (b) 对 G2 集关闭。测量-only；默认与门槛不变；旋钮
保持 opt-in。证据
`tasks/issue-9/2026-10-17-lc160-g2-rigid-transform`（预注册+驱动+
launcher+日志+6 个逐格 validation JSON+聚合+analysis.json+PHASE_RECORD，
14 文件 sha256 锚定）；effects-index 76 数据集。

**§85 附注（LC-161，2026-10-17 13:07:50Z 预注册；13:08:14-13:08:43 UTC
分离式运行，29 s 墙钟；LC-90 纪律——grad_hotspot_profiler 逐字复用，
CELLS 扩展包装器 + 绝对 G2 几何路径）**：审查卡队列第 5 项（从新测量
确定下一项有据实现/优化）——LC-158 的 helps-more 发现为触发，在 4 个
G2 格（每方法一个：isobutene/wb97mv/tzvp 172、C2H6CHOH/m062x/tzvp
172、CH3CH2OCH3/b3lyp/tzvp 172、环丙烷/r2scan3c/mtzvpp 129；recheck
保留分子排除；全闭壳层无 UKS）x 双 lane（fp64 = 旋钮全开的 strict-fp64
lane；auto）x 3 样本，**完整 5 旋钮栈态**（STACK_ENV 进程启动前设置）。
8/8 lane 收敛，profiler 验证 OK，observation-only 全过。

**组合（中位数）**：三杂交格 auto lane 的 `grad.df_jk_ip1` 均为 e2 内部
最大桶（48.0/45.7/50.3% —— 30-65% 带内）；3c 格 **xc_grid 主导**
（50.6/55.7%，df_jk 仅 23-26%，d4_gcp 0.2-0.3% = LC-87 tau 族迁移到
环丙烷）；wb97mv/isobutene 的 `grad.nlc_vv10` 22.3/21.2% **复现 LC-90
的 23.2% "uwe+glue" 族**（预测带 ≤15% 设自错误子桶——带宽错误如实
记录）。**无 ≥5% 新桶族出现** → 按预注册决策规则：**cohort 的小带完整
栈组合泛化到 G2 化学，本带 in-scope 杠杆面保持关闭，不命名新杠杆**
（named-deep 各片在新化学上复证：df_jk 积分构建、xc train、nlc
uwe+glue）。lane 细节与已知族一致：wb97mv auto lane 慢于旋钮全开
fp64 lane（0.443 vs 0.392 s = LC-158 栈 frac 1.158 的组合级呈现，
增长在 xc_grid 22.1% vs 14.2% = auto-lane 梯度旋钮不动族 LC-94）。
判定：P1/P2/P4/P6 HIT；P3 MISS（带宽错误，族确认）；P5 MISS 2/8
lane（0.15-0.45 s 梯度 lane 的亚秒噪声类，份额按中位数可引用）。
测量-only；默认与门槛不变；旋钮全 opt-in。证据
`tasks/issue-9/2026-10-17-lc161-g2-stack-composition`（6 文件锚定）；
effects-index 77 数据集。发件箱保持在 8b2b3f6 发射——本段折入下次发射。

**§85 附注二（LC-162，2026-10-17 13:13:07Z 预注册；13:13:34-13:18:27 UTC
分离式运行，4.9 分钟墙钟）**：LC-154 命名的"未来单元"——旋转求积噪声的
**网格层级定标测量**，在两个旋转失败的 G2 格（isobutene、CH3CH2OCH3，
wb97mv/tzvp）× grids.level {2,3,4} × identity+rot × 双 lane = 24/24 收敛。
**|dE(rot)| 随网格层级单调下降**（isobutene 6.20e-6 → 2.85e-6 → 4.34e-7；
乙甲醚 5.46e-6 → 2.71e-6 → 9.01e-7）；**level-3 逐位复现 LC-160**
（isobutene -2.844626e-6）；**level 4 降 6.56x / 3.01x 并把两格都带到
1e-6 Ha 不变性门以下** → grids.level=4 即旋转敏感工作流的**文档化用户
旋钮**（默认不动——基准 schema 保持 level 3；门槛不动）；lane 相等在每个
层级成立（6.0-9.7e-11）= 类在网格侧与层级无关，fp32-VV10 内核对旋转
响应零贡献；每 (格,层级) 行精度门全过（|dE| ≤1.0e-10，max|dg|
≤1.8e-8）；循环全 10/10。判定 P1-P6 **6/6 HIT**（L2/L3 的 8 条双 lane
不变性失败记录在案未改写——既有类）。诚实边界：2 分子 × 1 方法；
level-4 成本未单独基准（标准层级成本阶梯适用）；旋转坐标系网格
（替代缓解）命名未测。测量-only；默认与门槛不变；全部旋钮 opt-in。
证据 `tasks/issue-9/2026-10-17-lc162-rot-gridlevel`（12 文件锚定）；
effects-index 78 数据集。发件箱保持在 8b2b3f6——本段折入下次发射。

**§84 附注二（LC-163，2026-10-17 13:26:00Z 预注册；13:26:38-13:31:07 UTC
分离式运行，4.5 分钟墙钟）**：LC-158 保留的 **recheck 子集盲复测**（4 个
保留分子 C5H8/C3H9N/cyclobutane/H2CCHCN × 4 方法 = 16 格，complete
5-knob 栈 doc2-only，LC-142/159 双会话纪律）——保留配额作为**验证用途**
支出（LC-158..162 的所有调参决策均按设计排除该子集）。16/16 rc=0 全收敛
全过门（auto |dE| ≤1.75e-11，max|dg| ≤1.21e-6）；**跨会话 e_tot ≤2.84e-13，
32 条 lane 循环数全部相同**；会话-2 frac 范围 [0.983,1.281] 中位 1.122，
**16/16 全部落在各自 lane 宽度类带内**（最大 |Δfrac| 0.182 ≤ sub-2.5s
±0.25）。双会话标签：13/16 fp64-favored-or-parity；2 格跨平价符号移动
（C5H8/wb97mv 1.025→0.983、H2CCHCN/m062x 1.122→0.984，噪声类不标签）；
**C5H8/m062x 的 0.975 地板读数判定为会话-1 噪声**（会话-2 1.101，+0.126）
→ 不是第二个 auto-favored G2 格；isobutene/r2scan3c 仍是唯一。G2
complete-栈表面现带双会话标签 20/48 格，全部与单会话簇 [0.975,1.352]
一致。判定 P1-P5 **5/5 HIT**。诚实边界：单一复测会话；0.8-2.9 s lane =
最宽警示类；保留配额现已支出（16 格此后为普通已测格）。测量-only；
默认与门槛不变；全部旋钮 opt-in。证据
`tasks/issue-9/2026-10-17-lc163-reserved-recheck`（87 文件锚定）；
effects-index 79 数据集。发件箱保持在 8b2b3f6——本段折入下次发射。

**§85 附注三（LC-164，2026-10-17 13:29:10Z 预注册；13:31:37-13:34:03 UTC
分离式运行，2.4 分钟墙钟）**：LC-162 诚实边界"level-4 成本未单独基准"的
**同会话配对定价**——两个旋转失败格 × --grids-level {3,4} × complete
5-knob 栈 × auto/fp64 lane × 3 样本，4/4 rc=0 全过门（auto |dE|
≤2.0e-11，max|dg| ≤4.6e-7；每 doc 的 auto-vs-fp64 比较在同一网格层级
上计算，内部一致）。**缓解很便宜**：e2e L4/L3 = **1.12-1.18x**，t_scf
1.12-1.19x，t_grad 1.05-1.19x，循环 10→10 处处相同；跨层级 fp64
e_tot 移动 -3.05e-6/-2.08e-6 Ha（网格定义求积差，记录不设门）；frac
在噪声类内移动（isobutene 1.055→1.001，乙甲醚 1.061→1.059）。
P2 MISS-low 有信息量：便宜程度超出标准层级成本阶梯的预期——wb97mv
上 NLC/VV10 墙（层级不变）与 df_jk 主导计时，半局域 XC 积分只是较小
份额；**该溢价是成分特定的，是网格更重方法/基组的下界**（非 NLC 的
level-4 成本未测）。~12-18% 的 e2e 溢价买到 LC-162 的 3.0-6.6x 旋转
不变性缩减。判定 P1 HIT、P2 MISS-low（带设定错误记录）、P3/P4 HIT、
P5 PARTIAL（doc 级 peak 仪器为 null，以 LC-158 同分子家族 1040 MiB
为界）。测量-only；基准 schema 保持 level 3；默认与门槛不变。证据
`tasks/issue-9/2026-10-17-lc164-gridlevel-cost`（27 文件锚定）；
effects-index 80 数据集。发件箱保持在 8b2b3f6——本段折入下次发射。

**§84 附注三（LC-166，2026-10-17 13:42:39Z 预注册；13:45:26-13:52:39 UTC
分离式运行，7.2 分钟墙钟）**：剩余 28 格 G2 会话-2 扫描 —— **48 格
complete-栈表面达到全量双会话标签覆盖**（LC-159 的 4 格 + LC-163 的 16
保留格 + 本单元 28 格）。28/28 rc=0 全收敛全过门（auto |dE| ≤3.91e-11，
max|dg| ≤1.28e-6）；跨会话 e_tot ≤2.27e-13，本扫 + LC-163 共 112 条
lane 循环数全部相同；会话-2 frac 范围 [0.966,1.372] 中位 1.062，
**28/28 全部落在各自 lane 宽度类带内**（最大 |Δfrac| 0.175）。双会话
标签：26/28 fp64-favored-or-parity；2 格跨平价符号移动（C3H6_Cs/b3lyp
1.110→0.966、OCHCHO/wb97mv 1.081→0.989，噪声类不标签）；无新增
sub-parity 格。**全矩阵 roll-up：48/48 双会话标签 = 42/48 双会话
fp64-favored-or-parity + 5 符号移动噪声类（不标签）+ 1 sub-parity
（isobutene/r2scan3c，仍是唯一 auto-favored G2 格，全矩阵确认）。**
判定 P1-P5 **5/5 HIT**。诚实边界：每格单一复测会话；0.68-2.66 s lane
= 最宽警示类；本扫为标签覆盖完成（预注册锚定会话-1 读数），非盲测；
default 栈路线未复测（单会话 doc1 读数保留其警示）。测量-only；
默认与门槛不变；全部旋钮 opt-in。证据
`tasks/issue-9/2026-10-17-lc166-g2-remaining-sweep`（147 文件锚定）；
effects-index 82 数据集。发件箱保持在 23cdfb2——本段折入下次发射。

**§85 附注四（LC-165，2026-10-17 13:41:53Z 预注册；13:42:22-13:44:51 UTC
分离式运行，2.5 分钟墙钟）**：grids.level=4 溢价在 **非 NLC 方法**上的
定价（LC-164 命名诚实边界关闭）——同两分子 × b3lyp/tzvp +
r2scan3c/mtzvpp × L3/L4 同会话配对 × complete 栈，8/8 rc=0 全过门
（auto |dE| ≤8.8e-12，max|dg| ≤7.7e-7）。**溢价在方法轴上温和**：
非 NLC e2e L4/L3 = **[1.01,1.40]x**（b3lyp 1.03-1.26；3c 1.01-1.40）vs
wb97mv 的 1.12-1.18x —— "阶梯式"溢价预期**被证伪**（P2 双方法
MISS-low：网格增长被 df_jk/D4+gCP 稀释 + 剪枝使点数增长次线性 +
sub-1.5s 抖动）；最网格重的成分（3c auto lane）付最高溢价 1.25-1.40x
（弱支持成分依赖方向）；t_scf [1.01,1.45]x，循环 10→10 处处；跨层级
fp64 e_tot 移动 +2.79e-6..-6.53e-6 Ha（记录不设门）；P4 1/4 格 MISS：
ether/r2scan3c frac +0.36（0.956→1.316）——auto lane 的 1.40x 溢价
样本一致、fp64 L3 中位数抖动 = 部分真实的成分效应。诚实边界：2 分子
× 2 方法、96-172 AO、sub-1.5 s lane、单一会话、更大尺寸未测。
测量-only；基准 schema 保持 level 3；默认与门槛不变。证据
`tasks/issue-9/2026-10-17-lc165-gridlevel-cost-nonnlc`（47 文件锚定）；
effects-index 81 数据集。发件箱保持在 23cdfb2——本段折入下次发射。

**§86（LC-167+LC-168，2026-10-17 14:04Z 预注册 sha256 edcd458f；14:02:27-16:25:45 UTC
分离式运行，2 h 23 m 墙钟）**：**中等 CHON 扩充以真实公开几何执行**——LC-157
"离线服务器"前提**被纠正**（执行 shell 可达 PubChem PUG-REST，HTTP 200 @
13:57Z）：12 个 CID 钉住的 PubChem 3D 构象几何（aspirin 2244、caffeine 2519、
dopamine 681、epinephrine 5816、glycylglycine 11163、histamine 774、lidocaine
3676、melatonin 896、nicotine 89594、phenylalanine 6140、serotonin 5202、
tryptophan 6305；全 CHON、中性闭壳；sha256 锚定）CPU 准入后**冻结**于
inventory/gpu4pyscf-mediumchon-pubchem-20261017-v1（SVP 157-348 AO = 规则带；
MTZVPP 212-448；混合方法 def2-svp / 3c def2-mtzvpp 按队列中型带惯例；复核保留
caffeine/epinephrine/glycylglycine/lidocaine 不做调参）。矩阵 **96/96 文档
rc=0、192/192 lane 收敛、96/96 全门 PASS**（最差 in-doc |dE| 6.5e-11；最差
max|dg| 2.07e-5 = 门的 69%；峰值 4562 MiB；温度 ≤34 C）。**头条：新中等
N-化学上的 complete 栈表面在全部四方法上平价居中**（中位数 b3lyp 0.995 /
m062x 1.001 / wb97mv 1.010 / 3c 1.056；19/48 < 1.0）——队列/G2 的
fp64-偏好内部簇**不迁移**；auto-偏好尾部集中在 ≥270 AO（tryptophan 270
0.909/0.944/0.975、melatonin 318 0.902/0.998/0.993、lidocaine 348
0.945/0.954/0.970——8/12 ≥270 AO 亚平价，全在 ≥4s 可引用 lane 上），
318-537 auto-偏好带在该化学上**下延至 ~270 AO**。**默认栈表面四方法均
auto-偏倚**（中位数 0.868-0.984；非 NLC 随尺寸单调：melatonin/lidocaine
0.74-0.78）；新交叉点 ~200-250 AO 与 LC-117/118 一致。**inosine 族
wb97mv 默认 auto-win 不迁移**：12/12 默认 wb97mv 平价 [0.970,1.047]——VV10
墙在默认路线**lane 对称**（fp64 路线收益 5.65-7.65x ≈ auto 5.57-7.70x）。
**3c 亚平价族扩至新化学**（aspirin/r2scan3c 0.878，G2-isobutene 族；其余
11 格 1.01-1.24）。**跨栈 fp64 kernel-swap 类首达 1e-6 量级**：epinephrine
默认-vs-栈 fp64 e_tot 移动 -1.93e-6（b3lyp）/-7.16e-7（m062x）/-2.49e-7
（wb97mv），方法系统性、分子特异；两路线 in-doc 门全过（各 doc 的 auto
对自身 fp64 达 1e-12；循环 13/13）；语义按 LC-86/94/122（knob-on fp64
非严格参考，knob-off 文档为严格参考）；其余 45 格 ≤2.8e-10；跨栈
max|dg| 至 1.9e-5（lidocaine/wb97mv，同类）。裁定：P1 SPLIT（wb97mv
大格子 auto-偏好子断言 MISS = 发现三）、P2 MISS（内部地板 0.909 <
预测 0.95 = 头条发现）、P3 HIT（melatonin 0.993 平价级诚实注记）、
P4 MISS（aspirin）、P5 SPLIT（收敛/门 HIT；跨栈包络 MISS = 发现五）、
P6 HIT。诚实边界：单会话（35 lane sub-2.5s 噪声级/54 in 2.5-4s，引用
规则适用）；PubChem 构象几何仅 CPU 筛选（MMFF 级，非 QM 验证；
epinephrine 1.9e-6 的几何相关 SCF 地形贡献不能排除，需复测）；无 UKS
格；无 TZVP 混合点（冻结清单已记录备将来）。测量-only；默认与门槛
不变；全 knob opt-in。证据 `tasks/issue-9/2026-10-17-lc167-mediumchon-pubchem`
（16 文件锚定）+ `tasks/issue-9/2026-10-17-lc168-mediumchon-matrix`
（486 文件锚定）；effects-index 84 数据集。发件箱保持在 23cdfb2——
本段折入下次发射。

**§87（LC-169+LC-170，2026-10-17）**：

**LC-169 —— epinephrine 跨栈 fp64 1.93e-6 Ha 移动的旋钮归因（预注册
16:37Z sha256 748bd6b8；运行 16:33:55-16:35:27 UTC，6 单旋钮 knockout
文档全 fp64 道，6/6 rc=0；attempt-1 `--samples 1` 在参数校验被拒、零
GPU 消耗，保留于 attempt1-samples-schema-reject/；按 runner schema 以
3 样本执行）**：LC-168 发现五（跨栈 fp64 kernel-swap 类首达 1e-6
量级）的归因**完全落在 XC 格点分块旋钮 (b1) 上**——

| knockout | e_tot (Ha) | vs default |
|---|---|---|
| K0 完整栈 | -630.7161547258822 | -1.93e-6（移动量） |
| K1 栈−SCF_VMAT_F32 | -630.7161547258820 | 移动保留 |
| K2 栈−DF_IP1_F32 | -630.7161547258820 | 移动保留 |
| K3 栈−DF_IP1_FUSED | -630.7161547258818 | 移动保留 |
| K4 栈−XC_GRID_BLOCKSIZE | **-630.7161527938963** | **恢复（距 K5 2.0e-13）** |
| K5 默认（全关） | -630.7161527938965 | — |

即：**去掉 blocksize 16384（三个 fp32 旋钮全开）即恢复默认 e_tot 至
2.0e-13**；(b3) ip1-fp32、(b5) vmat-fp32、(b2) 融合旋钮在本格合并
贡献 ≤4e-13（与 LC-105 的 vmat ≤6.8e-11、LC-86 同轨道语义一致）。
1.93e-6 是 **fp64 道上的网格分块效应**（分块粒度求积/筛查差经 SCF
自洽放大；13/13 循环双路线、同支线；两路线 in-doc 门全过；knob-off
文档仍是严格参考）。裁定：P1 HIT；P2 MISS（预测 ip1_f32 主导，实测
blocksize 主导——预测误差即发现）；P3 N/A（单旋钮归因达成）。
**决策相关性（(b1) 直接证据，已录入决策简报数值类别 3）**：45/48
LC-168 格跨栈 fp64 移动 ≤2.8e-10，但 epinephrine 类分子放大至
1.93e-6（b3lyp）/7.16e-7（m062x）/2.49e-7（wb97mv）——不违反门
契约（同 doc 两道共享设置），可默认开提案会在 ~1e-6 量级移动部分
用户的收敛能量。测量-only；无代码/默认/门槛变化。证据
`tasks/issue-9/2026-10-17-lc169-epinephrine-attribution`（47 文件
sha256 锚定）。

**LC-170 —— 冻结 PubChem 中等 CHON 清单的 TZVP 混合方法轴（预注册
16:37:16Z sha256 18e329f7；运行 16:37:27-19:29:51 UTC 分离式 2 h
52 m，12 分子 × b3lyp/m062x/wb97mv @ def2-tzvp 302-659 AO ×
default+complete 栈 = 72 文档，72/72 rc=0、144/144 lane 收敛、144/144
in-doc 门 PASS、峰值 9698 MiB、温度 ≤38 C；3c 排除——其规定基组
def2-mtzvpp 已在 LC-168 测过）**：

- **头条一（P2 MISS = 发现）：per-cell 基组符号稳定性规则不迁移**。
  SVP 边际格（|frac−1| ≥ 0.05）中 **9/12 在 TZVP 符号翻转**
  （histamine/b3lyp 1.109→0.909、glycylglycine/b3lyp 1.106→0.979、
  glycylglycine/wb97mv 1.072→0.990、dopamine/b3lyp 0.925→1.042、
  aspirin/b3lyp 1.100→0.952、nicotine/wb97mv 1.103→0.965、
  serotonin/b3lyp 1.052→0.943、epinephrine/m062x 1.081→0.964、
  tryptophan/m062x 0.944→1.026；另 11 个平价类格允许移动）——
  cohort 的 LC-128/133/134 同分子基组对规则（"complete 栈 lane
  排序不随基组翻转"，测于 109-222 AO cohort 化学）**不迁移**到
  302-659 AO 中等 CHON 化学。**簇级读数迁移**：TZVP complete 栈
  全部 36 格 **[0.909,1.079]**（勘误：原文误写 [0.916,1.080]——0.916 是
  第三低格 lidocaine/m062x，真最小 histamine/b3lyp 0.9091，最大
  glycylglycine/m062x 1.0794；LC-177 审计抓到并更正），比 SVP 杂化格
  [0.902,1.109] 更紧；
  亚平价尾部 20/36（SVP 14/36）——per-cell 标签在该化学上
  **基组条件化**。
- **头条二（P3 MISS 双子项、方向一致）：TZVP 默认栈面强烈 auto
  偏好**。非 NLC 默认 frac 0.60-0.88（中位 ~0.65-0.68 @456-659 AO），
  相对 SVP 移动 −0.05..−0.33（预测带 [−0.20,0]，11/24 格带外、全在
  auto 偏好侧——LC-118 基组驱动交叉点移动**扩展并放大**，~200-250 AO
  SVP 交叉点在 TZVP 全带成立）；wb97mv 默认 [0.896,1.153]
  （caffeine 1.153 带外高、melatonin 0.896 边缘低）。
- **头条三（P5 SPLIT 的信息量）：epinephrine (b1) 跨栈 >1e6 越界
  预期在 TZVP 未出现**（1.14e-12；SVP 为 −1.93e-6；全部 36 格跨栈
  fp64 dE ≤4.55e-10、max|dg| ≤1.52e-5 ≤2.5e-5 带）——LC-169 的
  网格分块 e_tot 效应**基组依赖**（SVP 级网格在 16384 分块下放大，
  TZVP 网格不放大），(b1) 决策证据细化：~1e-6 代价是 SVP 类基组
  特定行为，非分子内禀。
- **P4 HIT**：TZVP VV10 墙迁移（fp64 t_scf 收益 **[4.06,5.06]x**
  全 12 格 wb97mv，预测带 [3.9,5.5]；LC-128 "TZVP 墙小于 SVP 外推"
  族扩展到新化学）；fp64 e2e 路线收益 [4.12,5.00]x。auto 道路线
  收益非 NLC [1.03,1.50]、wb97mv [4.05,5.21]。
- **P1/P6 HIT**：72/72 全过门；峰值 9698 MiB（lidocaine 659 AO，
  687-AO 准入带内）<14 GiB；墙钟 2 h 52 m ≤4 h；串行单 GPU。

裁定汇总：P1 HIT / P2 MISS（发现）/ P3 MISS 双子项（方向一致）/
P4 HIT / P5 SPLIT（门全 HIT + 预期越界条款被驳 = 有信息量）/ P6
HIT。诚实边界：单会话、TZVP 456-659 AO 上 sub-3s 道属噪声类引用
规则；PubChem 构象几何 CPU 筛选级；无 UKS 格；3c 轴未测 TZVP
（规定基组即 mtzvpp）。测量-only；默认与门槛不变；全 knob opt-in。
证据 `tasks/issue-9/2026-10-17-lc170-mediumchon-tzvp`（366 文件
sha256 锚定 + analysis.json）；effects-index 86 数据集。发件箱保持
在 23cdfb2——LC-165..170 段落折入下次发射。

**§87 补（LC-171，2026-10-17 18:03:52Z 预注册 sha256 9348b412；19:30:04-19:41:12
UTC 分离式运行 11 min，16 文档 doc2-only，16/16 rc=0、32/32 lane 收敛、
32/32 in-doc 门 PASS、峰值 4516 MiB、温度 ≤37 C）**：**复核保留 4 分子
（caffeine/epinephrine/glycylglycine/lidocaine）的盲会话-2 确认——干净
全过**。16/16 格全部在道宽类带内（最大移动 0.107 @glycylglycine/m062x
sub-2.5s 带 0.25）；循环处处相同；跨会话 e_tot **8/16 逐位**（全部 3c
格 + 全部 wb97mv 杂化格）、其余 ≤1.14e-12（P3 MISS-narrow：较大 AO 的
跨进程地板是 e-12 类，G2 是 e-13 类——LC-160 同族带设定误差）；无多数
票，两会话标签按 no-vote 纪律指派——**4 个亚平价格确证**（caffeine/b3lyp
0.969/0.957、caffeine/wb97mv 0.966/0.940、lidocaine/b3lyp 0.945/0.961、
lidocaine/m062x 0.954/0.901，两会话均 <0.98，含 ≥4s 可引用的
lidocaine 道）；glycylglycine 四方法全 fp64 占优或平价（预测成立）；
9 格 fp64 占优或平价；3 格噪声类无标签（caffeine/r2scan3c
1.091→0.998、epinephrine/wb97mv 0.989→0.961、lidocaine/wb97mv
0.970→0.983）。**读数**：LC-168 复核子集读数会话稳定；亚平价尾部
下缘在该化学上达 ~246 AO（caffeine 类；LC-168 "≥270 AO" 的诚实细化）。
保留即以验证用途用尽（从未用于调参）。裁定 P1/P2/P4/P5/P6 HIT、
P3 MISS-narrow。测量-only；默认与门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-17-lc171-mediumchon-recheck`（86 文件 sha256
锚定 + analysis.json）；effects-index 87 数据集。发件箱保持在
23cdfb2——LC-165..171 段落折入下次发射。

**§87 续2（LC-172，2026-10-17 19:42:00Z 预注册 sha256 c8870152；19:42:30-20:02:32
UTC 分离式运行 20 min，8 个非保留分子 × 4 方法 × complete 栈 doc2-only =
32 文档，32/32 rc=0、64/64 lane 收敛、64/64 in-doc 门 PASS、峰值 3624
MiB、温度 ≤35 C）**：**剩余-32-格会话-2 扫描——6/6 预测全 HIT，48 格
medium-CHON complete 栈面获得完整两会话标签覆盖**（与 LC-171 相加）。
0/32 带外（最大移动 0.151 @histamine/r2scan3c sub-2.5s 带 0.25）；3c
族 8/8 逐位；杂化格跨会话 e_tot ≤1.14e-12（LC-171 测得的地板带
≤1.2e-12 设定后命中）；循环处处相同。**再确证 4 个亚平价格**
（dopamine/b3lyp 0.925/0.970、serotonin/m062x 0.975/0.977、
tryptophan/b3lyp 0.909/0.957、melatonin/b3lyp 0.902/0.786——后者增强）；
15 格 fp64 占优或平价；13 格噪声类无标签。会话-1 亚平价尾部候选中
tryptophan/b3lyp 与 melatonin/b3lyp 确证；tryptophan/m062x、
tryptophan/wb97mv、melatonin/m062x、melatonin/wb97mv 全跨骑 → 噪声类。
**aspirin/r2scan3c 的 LC-168 亚平价读数（0.878）会话-2 不存活（1.006）
——诚实撤回为噪声类**；G2-isobutene 仍是唯一确证的 3c 亚平价格。
**全矩阵汇总**：8 亚平价确证（方法结构化：b3lyp 5、m062x 2、wb97mv 1、
3c 0；尺寸散布 246-348 AO）+ 24 fp64 占优或平价 + 16 噪声类 = 48 格；
平价居中簇两会话成立（LC-168 头条稳固）。测量-only；默认与门槛不变；
全 knob opt-in。证据 `tasks/issue-9/2026-10-17-lc172-mediumchon-s2-sweep`
（166 文件 sha256 锚定 + analysis.json）；effects-index 88 数据集。发件箱
保持在 23cdfb2——LC-165..172 段落折入下次发射。

**§87 续3（LC-173，2026-10-17 20:04:52Z 预注册 sha256 5cadb1da；20:04:58-21:02:06
UTC 分离式运行 57 min，histamine 157 AO + melatonin 318 AO 双阳离子格
（charge +1, spin 1，既有冻结几何，无新分子）× 4 方法 × default+complete
栈 = 16 文档，16/16 rc=0、32/32 lane 收敛、32/32 in-doc 门 PASS、峰值
5546 MiB、温度 ≤38 C）**：**medium-CHON 面的无 UKS 格诚实边界闭合**。
attempt-1 的预注册（3 方法）与驱动（4 方法）不一致在发射后 ~40 s 被抓
住、以**零结果 JSON** 终止，旧注册与中止日志保留于
attempt1-prereg-mismatch-abort/。结果：complete 栈 frac **[0.863,1.084]**
（预注册带 [0.80,1.30] 内）——**melatonin 318-AO UKS 格双栈皆 auto
偏好（0.81-0.94）= 亚平价带延伸到 UKS 双阳离子**；histamine 157-AO
跨骑（r2scan3c 0.949/0.918，sub-2.5s 单会话噪声类注记）；wb97mv 默认
平价延伸到 UKS（1.029/1.017）；VV10 墙迁移到 UKS 默认道（5.20x fp64
t_scf）；**无 (b1) 分块放大**（跨栈 dE ≤2.79e-10——追加 (b1) 决策证据）；
跨栈 max|dg| ≤9.2e-6；**无 complete 栈 UKS 循环税**（8/8 格栈循环 ≤
默认）；双阳离子循环 17-33 vs 闭壳层 12-15（P5 绝对带 MISS = 带设定
自错误参考类，记录）；无支线/双盆事件。裁定 P1/P2/P3/P4 HIT、P5 SPLIT、
P6 SPLIT（峰值 5546 MiB 超出 <5 GiB 带 ~8%，准入全过——带设定误差）。
诚实边界：2 分子 × 4 方法、单会话、双阳离子在中性构象上（电子问题
良定义，非优化阳离子几何）。测量-only；默认与门槛不变；全 knob
opt-in。证据 `tasks/issue-9/2026-10-17-lc173-mediumchon-uks`（89 文件
sha256 锚定 + analysis.json）；effects-index 89 数据集。发件箱保持在
23cdfb2——LC-165..173 段落折入下次发射。

**§87 续4（LC-174，2026-10-17 21:04:30Z 预注册 sha256 a608c48f；21:05:06-21:25:26
UTC 分离式运行 20 min，LC-160 协议逐字复用于中等带 3 分子 histamine
157 / tryptophan 270 / melatonin 318 AO @ wb97mv/def2-svp，5 变换 × 2 lane
= 30/30 rc=0 收敛；驱动日志保留模板名 run_lc160_driver.log、后改名
run_lc174_driver.log，内容自述 LC-174）**：**LC-160 的具名诚实边界
（变换验证仅至 172 AO）闭合——修复的整 SCF 平移不变性延伸到中等带**。
fp32knob 平移不变性 |dE| ≤1.92e-10 @1e3 / **≤1.14e-9 @1e6 Bohr**（带
5e-9；比 G2 的 4.6e-10 高一个量级于 318-AO 格，仍距门 4 个量级），
**lane 差 ≤1.3e-11 处处**（共享表示噪声证明：两道同动）= 原点减法
内核在 2-4 倍 GLC NLC 对数下**零平移误差**；精度门 30/30（|dE|
≤2.95e-10、max|dg| ≤9.93e-7）；**旋转在两 lane 上同败 1e-6 不变门**
（dE(rot) −1.56e-6/−1.91e-6/−4.42e-6；rot-梯度一致性 0.042-0.045
——勘误 by LC-184 §87 续12：该值为错侧转置比较伪影，更正约定后
knob-off 真值 3.7e-5-2.03e-4，仍全数超 3e-5 梯度门，FAIL 判定不变）
= 常态 lane 相等网格求积类，18 项失败记录不重述（每旋转行 lane 相等
≤1.03e-10；LC-162 的 grids.level=4 仍是文档化缓解）；rot_trans−rot
≤1.1e-10（P4）；循环平移恒等（两例单循环 lane 移动 ±1，记录）；
dg 不变性 ≤2.42e-7（带设定 miss，诊断性）。裁定 P1/P2/P3/P4 HIT、
P5 HIT 单循环例外、P6 PARTIAL（峰值仪器缺口以 LC-168/173 同分子家族
≤5.5 GiB 为界）。**修复在整个准入中等带上平移安全。**测量-only；
默认与门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-17-lc174-mediumchon-rigid-transform`（9 文件
sha256 锚定 + validation.json）；effects-index 90 数据集。发件箱保持在
23cdfb2——LC-165..174 段落折入下次发射（本次会话执行）。

**§87 续5（LC-175，2026-10-17 21:30:10Z 预注册 sha256 7d51ab89；21:30:24-21:31:36
UTC 分离式运行 72 s，serotonin/b3lyp 242 + melatonin/m062x 318 +
tryptophan/wb97mv 270 + aspirin/r2scan3c/mtzvpp × fp64+auto 道 × 3 样本，
8/8 收敛，profiler VALID；观察-only，grad 对基线差：杂化格 ≤1.5e-13，
3c 格 2.7-6.6e-7）**：**新化学
中等带的 complete 栈梯度成分——成分迁移（第三化学：cohort LC-90、G2
LC-161、medium-CHON 本单元）**。df_jk_ip1 为杂化 auto 道最大桶
**56.8-61.2%** t_grad（勘误：原文误写 56.9——wb97mv/tryptophan auto 的
  真值 56.77%，非 serotonin 的 56.93%；LC-177 审计抓到并更正）；
  wb97mv 的 nlc_vv10 **25.4-26.7%**（LC-90 的
uwe+glue 家族延伸到中等带）；3c 格 xc_grid 主导 **44.8-45.5%**、
d4_gcp 0.4%；**无新的 ≥5% 桶族** → 242-318 AO SVP 上 in-scope 杠杆面
在该化学上**保持关闭**，无新杠杆点名（named-deep 再次确认：df_jk
积分构建、xc train、nlc uwe+glue）。（勘误 2026-10-18 LC-178 预检：
serotonin def2-svp AO 数原文误写 221——冻结清单 manifest 的真值
242（C10H12N2O：10×14+12×5+2×14+1×14）；单元覆盖带随之更正为
242-318，成分结论不变。第二处勘误（LC-178 带边单元抓到）：本段
原文写"grad 对基线差 ≤8.9e-14 于每个样本"——8.9e-14 实为首个格
首样本值；杂化格全程 ≤1.5e-13，aspirin/3c 格每个样本读
2.7-6.6e-7（3c 仪表扰动类，远低于 3e-5 门槛，生产路径不受影响；
LC-178 的 histamine/3c 复现同班 1.0-5.7e-7，两分子可复现）。）裁定 P1/P2/P3/P4/P6 HIT、P5 SPLIT
（serotonin 两条 sub-2.5s 道 17.8/20.4% 散布 = 噪声类，预注册允许项）。
测量-only；默认与门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-17-lc175-mediumchon-composition`（6 文件
sha256 锚定）；effects-index 91 数据集。发件箱保持在 6bbf70e 发射
（本段折入下次发射）。

**§87 续6（LC-176，2026-10-17 21:41:55Z 预注册 sha256 32895e17；21:42:33-21:49:40
UTC 分离式运行 7 min，9 个 SVP→TZVP 符号翻转边际格 doc2-only 会话-2 复测，
9/9 rc=0、18/18 lane 收敛、18/18 in-doc 门 PASS、峰值 8842 MiB；attempt-1
的 `--lane "auto fp64"` 引号错误在参数校验即败、零 GPU 消耗，保留于
attempt1-lane-quoting-bug/）**：**7/9 翻转确证为基组真实（预注册 ≥7/9
门槛恰好达成）**——LC-170 的 per-cell 基组符号不稳定发现升级为两会话
结论。全部 9 格在道宽类带内（最大移动 0.126 @histamine/b3lyp sub-2.5s
带）；跨会话 e_tot 2/9 逐位、其余 ≤4.55e-13（TZVP 地板 e-13 类，**比
SVP 的 e-12 更紧**）；循环处处 +0。两个未保持格自身分类：histamine/b3lyp
= sub-2.5s 噪声类格（无基组标签）；glycylglycine/wb97mv = 两会话皆平价类
（0.990/1.017，均低于 SVP 1.072 的 fp64 占优边际——**远离 fp64 占优的
翻转确证**，符号在平价处摆动）。裁定 P1-P6 6/6 HIT（P5 恰在门槛）。
测量-only；默认与门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-17-lc176-tzvp-flip-confirm`（70 文件 sha256 锚定 +
analysis.json）；effects-index 92 数据集。发件箱保持在 6bbf70e 发射（本段
折入下次发射）。

**§87 续7（LC-178，2026-10-17 22:00:18Z 预注册内容 sha256 3d4c2983 于运行前；
22:00:18-22:01:40 UTC 分离式运行 82 s，medium-CHON 带边 5 个非 recheck 格
——histamine × b3lyp/m062x/wb97mv @def2-svp 157 AO（下边）+ histamine/r2scan3c
@def2-mtzvpp 212（3c 下边）+ melatonin/wb97mv @def2-svp 318（非 recheck 上点）
——× fp64+auto 道 × 3 样本，10/10 收敛，profiler VALID；设计性诚实界限：
319-348 上边仅 lidocaine（recheck 保留，只作验证）保持成分未测）**：**新化学
非 recheck 全带 157-318 的 complete 栈梯度成分——成分迁移到带边，杠杆面
保持关闭**。无任何新 ≥5% 桶族（10/10 道，桶集合恰为 LC-90/161/175 族）；
df_jk_ip1 保持杂化 auto 道最大桶 **41.6-57.1%** t_grad。**新读数一：
nlc_vv10 份额沿整个 NLC 对数轴尺寸平坦**——histamine-157 25.6/26.5%，
tryptophan-270 25.4/26.7%（LC-175），melatonin-318 25.0/25.6%——三点两
单元同一带，uwe+glue 家族的**份额**对 pair-count 不敏感。**新读数二：同
方法 df_jk_ip1 尺寸稀释单调**（wb97mv auto 道 47.8%@157 → 56.8%@270 →
57.1%@318）；3c 成分尺寸平坦（histamine-212 xc_grid 49.9/50.8% vs
aspirin-307 44.8-45.5%）。下边固定成本族增长但族内：hcore_e1 157 AO 时
14.1-16.3%（内部带 8.4-11.3% 的约 2 倍），nuc 0.0-0.1% 远低于任何 5% 越
界。裁定：P2/P3/P4/P6 HIT；P5 HIT+许可触发（本单元无 ≥2.5s t_grad 道，
最大 1.6s，首句空真记录；melatonin 道 1.0/5.1%；3 条 histamine 类 sub-2.5s
道 30.1/24.2/16.9% 散布）；**P1 SPLIT（信息性带设定 MISS）**——杂化格
4.6e-14..1.5e-13（LC-175 杂化地板精确复现），histamine/3c 格 1.0-5.7e-7
= **3c 仪表扰动类**（LC-175 aspirin 2.7-6.6e-7 两分子复现；该带承自
LC-175 误报的全局 8.9e-14——本单元跨单元比对抓到并已勘误上文续5；
距 3e-5 门槛约 50 倍，观察-only，生产路径不受影响）。测量-only；默认与
门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-17-lc179-mediumband-rot-gridlevel`（12 文件
sha256 锚定）；effects-index 95 数据集。发件箱保持在 6bbf70e 发射（本段
折入下次发射）。

**§87 续9（LC-180，2026-10-17 22:34:10Z 预注册 sha256 86bd5b37 于运行前；
22:34:53-22:50:22 UTC 分离式运行 15m29s，串行单 GPU；LC-179 的运行器逐字
+ 每格 level 计划：tryptophan-270 完整 {L3,L4,L5} 阶梯（关闭 LC-179 点名的
中点诚实界限）+ histamine-157 与 melatonin-318 的 L5（LC-179 的 L4 残差
格），共 20 lane-run 全 rc=0 收敛；第二个 block-4 门限前 idle-window
单元）**：**主读数：三格全部在 grids.level=5 进入 1e-6 Ha 旋转不变性门内
——histamine +2.281e-7、tryptophan -5.4706e-7、melatonin -9.085e-7（仅
9% 余量）——medium 带旋转缺口在 L5 数值可关**，且代价远低于预期：
**L5/L3 fp64 墙钟溢价仅 ~1.10x（三格 1.099-1.103；预注册期望 1.5-3x =
MISS-low helps-far-more；fp32knob 1.33-1.47x）**。**但收敛是振荡的，
不是逐点单调的**：tryptophan L4 的 +3.56e-8 是一个消除节点（cancellation
node），其 L5 反而变差（0.07x，-5.47e-7）；melatonin 延续缓趋势（1.42x）；
histamine 跨零（-4.42e-6 → -1.16e-6 → +2.28e-7）——**预注册的停带条款
（< 1.5x vs L4）触发，P4 形式 MISS + 实质 SPLIT：不存在给出分子无关
过关保证的固定 level；per-(分子,level) 经验验证仍然必需，rotated-frame
grids 保持已命名的原则性方向**。**P3 MISS-high（信息性）**：tryptophan
L4 降幅 43.78x 远超预注册带 [1.3x, 4.5x]（取自 LC-179 两带边）——
level 响应在 medium 带上分子依赖 ~30 倍（L4 降幅 1.48-43.78x）。
**P2 HIT**：tryptophan L3 逐位复现 LC-174（两道）。**P5 HIT**：每一
level lane 相等（max 1.8e-10，L5 亦 grid-side，fp32-VV10 平移修复全程
免责）。**P1/P6 HIT**（20/20；墙钟 15m29s；GPU 于 22:50:22Z 释放，
block-4 门限前 1h10m）。**LC-179 读数被细化**：level 4 关闭部分 medium
格（tryptophan），不关带边；level 5 关闭全部三格；保证问题（振荡）保持
开放。缓解阶梯获得 level-5 选项（带实测溢价与振荡告诫；melatonin 仅 9%
余量）。诚实界限：3 格 × 1 方法、单会话、无 UKS 格。测量-only；默认与
门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-17-lc180-mediumband-level5-ladder`（13 文件
sha256 锚定）；effects-index 96 数据集。发件箱保持在 6bbf70e 发射（本段
折入下次发射）。

**§87 续10（LC-181，2026-10-17 22:52:51Z 预注册（22:53:58Z 运行前修订补
入 3c 构造细节，无预测改动；修订注中一处自造时间戳 22:56:30Z 经文件
mtime 证明实为 22:53:58Z，当场勘误并记录于 phase record）；22:54:15-
22:59:28 UTC 分离式运行 5m13s，串行单 GPU；LC-179/180 的 L5 关闭结论的
方法轴——histamine-157 × {b3lyp/svp, m062x/svp, r2scan3c/mtzvpp-212}
+ melatonin-318 × {b3lyp/svp, m062x/svp}（melatonin/3c 为墙钟预算剔
除，诚实界限）；levels {3,5} × identity+rot × 仅 fp64 道（grid-side 归因
由 LC-179/180 的 lane 相等承载）；3c 格走生产 D4+gCP 能量路径
（resolve_3c + attach_3c_corrections，density_fit 后挂接按文档顺序）；
20 lane-run 全 rc=0 收敛；第三个 block-4 门限前 idle-window 单元）**：
**P3 信息性 MISS（本单元的交付物）——melatonin/m062x 在 L5 仍高出
1e-6 门 3 倍（-2.97e-6，自 L3 的 -4.66e-5 降 15.68x）：L5 关闭是
METHOD-SCOPED**——在 b3lyp（两尺寸：histamine -1.75e-7、melatonin
-8.24e-7）、r2scan3c（-4.26e-7）、m062x@157（-8.53e-7）上成立，在
318-AO 带顶的 m062x 上不成立。**P2 MISS（信息性）**：L3 旋转响应的
探索带 [0, 5e-6] 被 m062x 双格越出（melatonin -4.66e-5 ≈ 带顶 10 倍、
histamine -5.89e-6）——**方法阶梯 ~60x：b3lyp +0.79e-6（melatonin
L3 已在门内！）< wb97mv -1.9e-6（LC-179）≈ 3c ~-1.5e-6 < m062x
-46.6e-6**；m062x 的 meta 泛函被积函数在同一几何上对旋转求积的权重
~24 倍于 b3lyp。**振荡告诫扩展**：melatonin/b3lyp L3 已过关而其 L5
略差（0.96x）——L3 过关不保证 L5 过关。**P4 HIT**：identity-lane
L5/L3 墙钟溢价 1.203-1.500 全在预注册带 [1.05,1.6]（rot-lane 外点
histamine/m062x 1.81 记录在案）。**P1/P5/P6 HIT**（20/20；跨 level
循环全同；墙钟 5m13s；GPU 22:59:28Z 释放，block-4 门限前 1h）。
**缓解阶梯注释更新**：level=5 实测关闭 medium 带上 b3lyp/r2scan3c/
wb97mv（及 m062x@157）的旋转响应（1.10-1.50x fp64 墙钟溢价）；
**带顶尺寸的 m062x 在 L5 不关闭**（L6/rotated-frame grids 保持已命名
未测候选）。诚实界限：仅 fp64 道、5 方法格、单会话、无 UKS 格、
melatonin/3c 与 L6 未测。测量-only；默认与门槛不变；全 knob opt-in。
证据 `tasks/issue-9/2026-10-17-lc181-l5-method-axis`（18 文件
sha256 锚定）；effects-index 97 数据集。发件箱保持在 6bbf70e 发射（本段
折入下次发射）。

**§87 续11（LC-182 + LC-181 披露，2026-10-17；LC-182 预注册 23:02:01Z
sha256 cf1de2be 于运行前，23:04:28-23:05:48 UTC 分离式运行 80 s，串行
单 GPU，4/4 rc=0，ALL GATES PASS = 无不变性失败记录；第四个 block-4
门限前 idle-window 单元）**：**P2 HIT 首分支（主读数）——melatonin/m062x
在 grids.level=6 进入 1e-6 门内：|dE_rot(L6)| = 8.546e-7（14% 余量），
阶梯 -4.658e-5 @L3 → -2.971e-6 @L5 → -8.546e-7 @L6 单调（L6 对 L5
降 3.48x）；histamine/m062x +3.469e-7 亦过关（2.9x 余量；符号翻转 =
LC-180/181 振荡类持续）**。**medium 带缓解表就此测全：b3lyp/r2scan3c/
wb97mv + m062x@157 在 L5 关闭；m062x@318 需 L6**；L6/L3 identity 溢价
1.232-1.284（**P4 MISS-low**：低于探索带 [1.5, 6.0]，且测得比 L5 溢价
更便宜——剪枝行为，单会话，按实测记录）。**P1/P3/P5/P6 HIT**（循环
L3/L5/L6 全同；墙钟 80 s；GPU 23:05:48Z 释放，block-4 门限前 54 min）。
**LC-181 运行后披露（勘误注记）**：一次后台 tools 测试套件运行
（~600 MiB CUDA 上下文，间歇 SM 负载）与本单元 GPU 窗口重叠
（~22:55:30-22:59:05Z 落在 22:54:15-22:59:28Z 内）——执行纪律失误
（曾假设 tools 测试纯 CPU，实际初始化 CUDA 上下文）。影响：LC-181 的
墙钟读数（P4 溢价与逐运行墙钟）可能带百分之几的并发负载膨胀、时间上
非均匀；方向不受影响（落在 LC-179/180 干净测得的 1.10-1.50 带内）；
dE_rot 值与循环不受影响（独立进程、确定性数值、每 (identity, rot) 对
同代码路径）。LC-182 墙钟干净（其 GPU-idle 门正确等走该并发运行）。
已记录于 LC-181 的 analysis.json + phase record；tools 套件本身通过
（530 OK @ 60628e2，208.760s）。测量-only；默认与门槛不变；全 knob
opt-in。证据 `tasks/issue-9/2026-10-17-lc182-m062x-l6-probe`（9 文件
sha256 锚定）；effects-index 98 数据集。发件箱保持在 6bbf70e 发射（本段
折入下次发射）。

**§87 续8（LC-179，2026-10-17 22:19:47Z 预注册 sha256 f9670e44 于运行前；
22:20:20-22:31:17 UTC 分离式运行 10m57s，串行单 GPU；LC-162 协议逐字
扩展到 LC-174 旋转失败带边 histamine 157 + melatonin 318（wb97mv/
def2-svp，非 recheck 格），levels {3,4} × identity+rot × fp64+fp32knob
= 16 lane-run 全 rc=0 收敛；本单元为 LC-156 block-4 00:00Z 门前的
idle-window 单元，关闭 LC-162 点名的诚实界限「仅 2 分子 × 1 方法、
G2-only」与 LC-174 的未测缓解缺口）**：**P3 信息性 MISS（本单元的
交付物）——level 4 在 medium NLC 带上方向有效但不足以过关**：两格旋转
误差单调下降（histamine 4.4246e-6 → 1.1609e-6 = 3.81x；melatonin
1.9147e-6 → 1.2921e-6 = 1.48x，低于 ≥3x 带）但**两格残差均高于
1e-6 Ha 不变性参考 16-29%**——LC-162 的 G2 规则（level 4 既 ≥3x
降又压到门内：isobutene 6.56x 至 4.34e-7、ether 3.01x 至 9.01e-7）
**不迁移到 medium 带**。后果：(i) 面向用户的缓解说明必须限定作用域
——level=4 在 157-318 AO SVP 上降低但不关闭旋转缺口；(ii) medium
NLC 尺寸旋转工作流的不变性门在 level 4 仍开放（level 3 同样，LC-174）；
(iii) rotated-frame grids 仍为已命名未测的下一候选（LC-162 命名维持）。
**P2 HIT**：level-3 旋转值逐位复现 LC-174（histamine bitwise 0.0；
melatonin -1.14e-12 跨进程地板族）。**P4 HIT**：每一 (格,level)
lane 相等（max |dE_rot_fp32knob − dE_rot_fp64| = 1.04e-10）= 类保持
grid-side，fp32-VV10 平移修复在任何 level 均无贡献（LC-154 修复再度
免责）。**P1 HIT**：精度门每行全过（dE ≤ 2.85e-10，dg ≤ 9.44e-7）。
**P5 MISS by band**：一条 +1 循环移动（melatonin L4 fp32knob rot
15 vs identity 14；LC-160/174 「±1」近阈值末循环族；能量在 1e-12 量
级不受影响；本预注册带宽 ±0 取自 LC-164 的 G2 10→10，诚实 MISS）。
**次要观察**（非 benchmark 文档）：L4/L3 identity-lane 墙钟溢价
histamine 1.037x/1.147x、melatonin 1.050x/1.197x（fp64/fp32knob），
在 LC-164/165 的 G2 溢价带（1.01-1.40x）内、fp64 端偏便宜。诚实界限：
2 分子 × 1 方法、带边格；tryptophan-270 中点 L4 未测；单会话；无 UKS
格。聚合脚本首版把驱动器预期的 invariance-record 退出码 1 误计为子进
程失败（当场抓到修复并记录于 phase record）。测量-only；默认与门槛
不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-17-lc179-mediumband-rot-gridlevel`（12 文件
sha256 锚定）；effects-index 95 数据集。发件箱保持在 6bbf70e 发射（本段
折入下次发射）。

**§87 续12（LC-183/184，2026-10-18；LC-184 预注册 2026-10-18T01:41:27Z
sha256 461bb896 于运行前（驱动 46ef9818、启动器 7f526c16 同锚）；
01:42:19-01:59:35 UTC 分离式运行 17m16s，串行单 GPU，28/28 rc=0 收敛，
温度 ≤35 C，GPU 释放；实现提交 43fc7aa 于运行前）——rotated-frame
grids（LC-162/179/180/181 四度点名的 principled 方向）落地为 opt-in 旋钮
GPU4PYSCF_GRID_MOLFRAME 并在默认 level 3 上整 SCF 验证关闭 medium 带旋
转缺口**：

**设计（LC-183，commit 43fc7aa）**：网格构建中唯一取向敏感件是每符号
的 Lebedev-Laikov 角向表（径向网格球对称、剪枝只依赖半径、Becke 权重
只依赖距离）；旋钮把角向表先旋入分子惯性主轴系再放置
（gen_grid.py::_principal_frame + Grids.build 钩子；近简并惯性
（相对本征值隙 ≤1e-6）回退实验室系网格并警告；对称陀螺/线性/单原子
属回退类）。任何确定性本征向量符号/次序约定都不可能严格旋转协变
（拓扑障碍：绕垂直于本征向量的轴转 π 必翻转符号）；设计改为吸收偏差：
几何与其旋转副本之间的约定偏差必为带符号置换，而 Lebedev 表
（节点,权重）集在全部 48 个带符号置换下不变——对 32 个已初始化尺寸
在 GPU 构建绑定的同一 MakeAngularGrid 调用上逐位验证（worst gap 0.0；
LEBEDEV_NGRID 的 n=1 条目不初始化缓冲区 = 上游怪癖，标准剪枝路径
n≥14 不可达，按权重和有效性检查排除）。**9 个新测试**：表 B3 对称、
帧协变性（mod 带符号对角）、简并回退、网格集旋转协变性（坐标 <1e-9
cKDTree、权重绝对 1e-13）、实验室系缺陷方向（knob-off 非协变 >1e-3）、
knob-off 死路 monkeypatch 守卫、旋钮解析、整 SCF wb97m-v 旋转不变冒烟
（|dE|<1e-11）。套件：grids 7/7、molframe 9/9（3 次稳定）、numint
23/24（test_sparse_index 为已知 pristine-upstream 环境类，净 HEAD 复
现）、rks 16/16。实现期缺陷全部被单测网捕获（stratmann→strattn 手误
破坏 test_grids 2 项→修；n=1 未初始化缓冲导致 B3 测试跨进程不稳→加
有效性过滤；测试助手漏 unit='Bohr' 造成 1.8897x 静默重标度→修——
驱动器 xyz-埃一致不受影响）。

**LC-184 测量（LC-179/181 运行器结构逐字、旋钮轴换级轴）**：
histamine/tryptophan/melatonin × wb97m-v/def2-svp（LC-174 旋转失败带）
× {fp64, fp32knob} + melatonin/m062x/def2-svp（LC-181 L5 失败带顶格）
× fp64；grids.level=3（默认——主读数即默认级关闭）；knob off/on；
28 lane-run。**主读数（P1 实质 HIT）**：knob-on |dE_rot| 全 7 lane-格
≤7.0e-11 Ha（较 knob-off 4.42e-6/1.56e-6/1.91e-6/4.66e-5 降
2.2e4-2.6e6x）；**m062x 带顶格（LC-182 需 L6 且只达 8.55e-7）在 L3
关闭到 3.0e-11**。**梯度（P4 实质 HIT，更正约定后）**：knob-on
dg_rot 2.1e-8-1.7e-7 Eh/Bohr，较 knob-off 3.7e-5-2.03e-4（全部
knob-off 格 FAIL 3e-5 梯度门）降 180-2300x，全部 knob-on 格过门余量
≥180x。**零溢价**：墙钟比 0.984-1.019；点数逐格恒等
（200704/329216/396544）；engagement=1 处处适用；**knob-on fp32knob
道循环旋转稳定而 knob-off 不稳定**（tryptophan [14,15]、melatonin
[15,14] vs on [14,14]/[15,15]）。**裁定**：P2 HIT（knob-off 基线
≤6.8e-13 复现 LC-174/181，tryptophan 逐位 0.0）；P6 HIT（精度门两种
knob 态全过；驱动 rc=1 仅来自错误公式的 dg 门与 P1 带 dE 条目——分
析约定项，原始数据无恙）；**P1/P4 MISS by band**（1e-11/1e-7 预注册带
假定机器地板，实际地板是 SCF conv_tol=1e-10 轨迹噪声（两次独立收敛
越过置换后的累加次序）；6/7 与 4/7 超带，值 1.7e-12-7.0e-11 与
2.1e-8-1.7e-7；网格侧不对称本身已被消除（单测集协变 + 点数恒等）；
与 LC-160 fp64 子带同族）；**P3 SPLIT**（网格重定义位移 +1.87e-7/
-2.20e-6/+4.42e-6/+1.154e-5，m062x 超 [1e-7,1e-5] 顶 15%；LC-164
跨级类，记录不设门；**knob-on 与 knob-off 能量不得混比**）；**P5
MISS 一格**（tryptophan knob-off fp32knob rot +1 循环 = knob-OFF 性质，
LC-160/174 族）。诚实界限：4 格 1 方法族、非 recheck 格；单会话；无
UKS 格；简并惯性分子按设计回退实验室系。**已命名后续（未测）**：
conv_tol=1e-12 探针可把演示地板压到预注册带下；(b6) 类待用户决策——
是否将 MOLFRAME 与 grids.level 并列为旋转敏感工作流的文档化缓解
（opt-in 不变）；简并类 tie-breaker 规范化为具名候选。测量-only；默认
与门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-18-lc184-molframe-rigid-transform`（20 文件
sha256 锚定，含 b3_lebedev_probe 逐位证据）；effects-index 100 数据集。
发件箱保持在 f3e637a 发射（本段折入下次发射）。

**勘误（LC-174 旋转-梯度量级，由 LC-184 更正约定发现）**：§87 续4 的
「rot-梯度一致性 0.042-0.045」与 LC-174 validation.json 的
dg_rotconsistency 用了错侧转置（G_rot − G_id@R）；几何按
x'=U(x−p)+p 旋转时梯度（协向量）应为 G_rot = G_id@U.T，正确比较为
G_rot@U − G_id（等价 G_rot − G_id@U.T）。错侧产生 |G_id| 量级混合伪影
（未弛豫构象 |G|~0.03-0.05）；**更正后 knob-off 值为 3.7e-5-2.03e-4
——仍全部超过 3e-5 梯度门，LC-174 的每条旋转-梯度 FAIL 记录与判定
不变，仅量级虚高 ~1000x**。实证：knob-on（网格集严格协变）只在更正
约定下残差塌缩（2.1e-8-1.7e-7），错侧不塌缩。LC-160/174 的平移 dg
值不受影响（平移无转置歧义）。LC-184 驱动器自身 in-run dg 门犯了同样
错误（运行后分析捕获；原始梯度数据存储无误；聚合以更正约定重算）。

**§87 续13（LC-185，2026-10-18；预注册 2026-10-18T02:16:46Z sha256
43fd3204 于运行前（驱动 09a43137、启动器 aa65ee8e 同锚）；02:17:28-
02:27:38 UTC 分离式运行 10m11s，串行单 GPU，8/8 rc=0 收敛，GPU 释放；
LC-184 的两个具名后续：conv_tol 地板探针 + UKS 阳离子格）**：
**(a) conv_tol=1e-12 地板探针（melatonin/wb97mv/svp，fp64，knob off/
on × identity+rot）——P1 信息性 MISS（预注册的 conv_tol 地板机制被证
伪）**：knob-on |dE_rot| = 2.797e-11 在 conv_tol 1e-12 下**不收紧**
（1e-10 时 -2.842e-11，同量级）而每个单跑跨 conv_tol 复现至 ≤3e-12
（on-identity -764.1290315744423@1e-10 vs -764.1290315744398@1e-12）
= 残差不是 SCF 收敛误差，而是**全管线 fp 累加表示噪声**（764 Ha 总
能量上 ~1e-13 相对；与 LC-174 记录的 fp64-vs-fp32knob「共享表示噪声」
lane 差 ≤1.3e-11 同族）。knob-off -1.9147e-6 跨 conv_tol 与跨会话逐
位稳定（P1b HIT）= 旋转误差确为网格侧、knob 消除的是它本身（单测集
协变 + 点数恒等 + 此处地板归因）。dg_on 1.877e-7 同不收紧（1e-10 时
1.35e-7）= 同族。**LC-184 P1/P4 的带宽 miss 至此机制闭合：预注册带
假定机器地板（1e-12 级），实测地板是管线表示噪声（1e-11..1e-10 级，
分子依赖），网格侧不对称性为零（被消除的量恰是 knob-off 的 1e-6..1e-5
值）**。(b) **UKS 阳离子格（histamine +1/spin1/wb97mv/svp，LC-173 格，
fp64）——P2 HIT：knob-on |dE_rot| = 1.933e-12**（较 knob-off
-3.910e-6 降 2.0e6x；20/20 循环；dg_on 2.34e-8）= **旋钮不变性延伸
到开壳层阳离子双重态，LC-184 的无-UKS 诚实界限关闭**；knob-off 在
预测类带内（P2b）。P3 HIT（循环 17/18 ∈ [15,21]）；P4 HIT（8/8，
温度 ≤35 C，墙钟 10m11s，GPU 释放）。诚实界限：每格单分子单方法
单会话；无 fp32knob 复验（UKS）；表示噪声地板未做逐分量分解（具名
候选）。测量-only；默认与门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-18-lc185-molframe-floor-uks`（12 文件 sha256
锚定）；effects-index 101 数据集。发件箱保持在 f3e637a 发射（本段折入
下次发射）。

**§87 续14（LC-186，2026-10-18；预注册 2026-10-18T02:32:23Z sha256
4bec1420 于运行前；attempt-1 02:32:55Z 因 runner 原子串 NameError 零
GPU 结果中止（identity-only 变体丢失 coords zip）+ 同段复查发现的 env
替换隐患，双修后 attempt-2 02:34:44-02:36:26 UTC 运行 1m42s，串行单
GPU，8/8 rc=0，GPU 释放；attempt-1 保留于 attempt1-runner-defect/）**
：**旋钮与完整 5-knob 栈组合（P1+P2 HIT = 组合成立）**——
melatonin/wb97mv/svp（318 AO，VV10 重）+ tryptophan/m062x/svp（270 AO，
非 NLC 对照）× {fp64, auto} × molframe off/on 于 LC-168 STACK_ENV 逐字
（blocksize 16384 + VV10-fp32 + ip1-fused/ip1-f32/grad-grid-f32 +
vmat-headtail）。**P1 HIT**：两种 knob 态下 in-doc 精度门全过（melatonin
dE 8.4e-12/6.1e-11、dg 1.3e-6/2.0e-6；tryptophan dE ~1e-13、dg
6.7e-7/1.2e-6）。**P2 HIT（组合主读数）**：molframe on-off 能量位移
**lane 相等**（lane 差 5.3e-11 / -6.8e-13）= 网格重定义作用于两道共享
的网格，与精度道及 fp32/fused 旋钮**零交互**。**P3 SPLIT**：
melatonin 位移 +4.4223e-6 与 LC-184 默认栈值逐位相等 = 重定义
**栈无关**；tryptophan/m062x +2.6923e-5 超预注册顶 2.2x（求积定义差，
记录不设门；m062x 格位移最大与其 knob-off 旋转误差最大一致）。**P4
HIT**：墙钟比 0.97-1.02、melatonin auto 14 vs 15 循环（±1 内）、栈加速
如预期（melatonin fp64 17.8s vs 默认栈 ~100s）。**P5 HIT**。读数：
**GPU4PYSCF_GRID_MOLFRAME + 完整推荐栈 = 旋转敏感工作流的有效组合
opt-in 态**。诚实界限：2 格、单会话、仅 identity 几何（旋转不变性本身
由 LC-184/185 建立）。测量-only；默认与门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-18-lc186-molframe-stack`（18 文件 sha256 锚定，
含 attempt-1 保留）；effects-index 102 数据集。发件箱保持在 f3e637a
发射（本段折入下次发射）。

**§87 续15（LC-187，2026-10-18；预注册 02:42:42Z sha256 aa5c659a 于运行
前）**：**molframe 旋钮的 G2 带抽查（命名后续项闭合）**——isobutene +
CH3CH2OCH3（wb97m-v/def2-tzvp，LC-162 旋转失败带；lab-frame L3 dE_rot
-2.8447e-6 / +2.7095e-6）× identity+rot × fp64+fp32knob × knob off/on，
LC-184 协议逐字（def2-svp 除外）。运行 02:43:55-02:46:34Z 串行单 GPU
2m39s，16/16 子进程 rc=0。**P1 HIT**：knob-on dE_rot = 3.13e-12 /
7.38e-11（isobutene）与 -1.14e-13 / -2.71e-11（ether），全部在 1e-10
LC-184-学习带内（fp64 道子带 <=3e-11 双格命中）。**P2 HIT**：knob-off
逐位复现 LC-162 L3 基线（偏差 <=6.8e-11）。**P3 HIT**（精度门全过，
lane 差 <=8e-12）。**P4 HIT（改正约定）**：knob-on dg_rot
1.30e-8..2.04e-8；knob-off 1.40e-5..4.32e-5 在预测 lab-frame 带内。
**P5/P6 HIT**（点数相同、循环 ±1、engaged 全部、峰值 <500 MiB）。网格
重定义位移（记录不设门）：isobutene ~1e-12；ether +2.62e-6（knob-on/off
能量不得混用）。**驱动器内联门缺陷（诚实勘误）**：驱动器 rc=1 完全归因
于逐字继承的 LC-184 驱动器 dg 门错误转置侧（grad_rot @ U.T - grad_id；
LC-174-erratum 类，LC-184 聚合已披露并改正但驱动器本身未打补丁），内联
"失败"值 8.3e-3..9.1e-3 为 |G| 尺度混合伪影；本单元聚合
（aggregate_lc187.py，与 LC-184 聚合同等处理）从原始梯度重算正确协变
关系 G_rot = G_id @ U.T（比较 G_rot @ U - G_id）为权威裁定，全部 6 项预
注册 HIT；后续驱动器（LC-188）已修正内联约定。读数：**旋钮的旋转不
变性扩展到 G2 化学带，knob-off 基线锚定 LC-162**。测量-only；默认与门
槛不变；全 knob opt-in。证据 `tasks/issue-9/2026-10-18-lc187-molframe-g2-spotcheck`
（13 文件 sha256 锚定）；effects-index 103 数据集。发件箱保持在 f3e637a
发射（本段折入下次发射）。

**§87 续16（LC-188，2026-10-18；预注册 03:05:46Z sha256 f556d9fc 于运行
前；attempts 1-2 三个零/低 GPU 驱动器缺陷全部保留：pyscf.data.elem 坏
导入、RUNNER f-string 花括号未转义 KeyError'np'、inplane_cut 质量模板
双括号 2D einsum 错）**：**对称陀螺 tie-breaker —— 4-折类，绕障定理与
验证**。零 GPU 覆盖枚举：G2-12 有 4 个简并类分子（cyclopropane D3h、
trimethylamine C3v、spiropentane/cyclobutane 精确 D2d）、cohort-24 有
4 个（acetonitrile C3v、benzene D6h、methoxy C3v、oxygen_triplet 线
性）、medium-CHON-12 无。**绕障定理（设计被自身测试证伪后推导）**：
首版 k=3..6 复数矩阶梯在 d6h 环的通用旋转下给出 J(60°) —— 分支跳跃非
零测度；推导证明 k=3/5/6 的歧义类含非八面体旋转且选择差异累积旋转的
2π/k 单位绕数，几何的 C_k 对称使面内角仅可观测 mod 2π/k，故**任何确
定性规则对均匀旋转的精确协变概率恰为 1/3（k=3,6）/ 1/5（k=5），与规
则无关**；仅 4-折类 {0,90,180,270°} 全部落入 Lebedev 表不变的 48 符号
置换内 = 网格集层面全旋转协变（含分支割穿越）。**修订设计**：
`_symtop_inplane_axes` 仅以面内四阶复矩 mu_4（|mu_4|/norm > 1e-6）解析
v1（D2d/D4h/C4v + 偶然近简并）；k=3/5/6 保持 lab-frame 回退（诚实文档
化极限：不可预测的 1/3 协变比文档化回退更差）；枚举实证：
cyclopropane/trimethylamine/acetonitrile mu_3~1.0 而 mu_4~1e-7（不可解
析）、benzene mu_6=1.0、两个 k=4 格解析。+9 测试（含 47° 割后旋转的
mod-48 等价、割后网格集协变、cyclobutane SCF 割后不变性 <=1e-11）；
套件 molframe 15/15、grids 7/7、rks 16/16、numint 23/24（已知环境类）。
**GPU 验证**（cyclobutane + C5H8，wb97m-v/def2-tzvp，knob off/on ×
fp64/fp32knob × identity/rot + 自适应割穿越面内旋转
[cyclobutane ψ=46.00°、C5H8 ψ=1.00°]；改正内联 dg 约定）：attempt-2
03:13:54-03:17:13Z 20/20 rc=0。**P1 SPLIT**：knob-on dE_rot
1.53e-12/2.99e-11/1.56e-12/**-1.069e-10**（C5H8 fp32knob 超
preregistered 1e-10 带 7% = LC-184/187 conv_tol 轨迹噪声底族，同类裁
定）；fp64 子带 <=3e-11 HIT。**P2 HIT**：knob-on dg_rot
1.29e-8..2.02e-8；knob-off 1.76e-5..2.35e-5 lab-frame 带。**P3 HIT
（招牌行）**：割穿越 dE = 2.13e-12/8.61e-11/5.68e-14/6.38e-11 全部
<=1e-10 —— C5H8 fp64 行为最强演示：**1.0° 旋转触发整 90° 帧跳变而
能量一致到 5.7e-14 Ha**（k=4 跳变类良性，设计如此）。**P4 HIT**：
knob-off 首测旋转误差 -5.87e-6/-3.67e-6；精度门全过。**P5 HIT**
（engaged、点数、循环 9/9）。网格重定义位移（记录不设门）：
cyclobutane -5.7e-14（文件几何轴对齐：tie-breaker 帧近似 lab 帧的符
号置换）；C5H8 **+1.64e-5**（其帧离 lab 帧 ~45°：对通用对称陀螺取向
tie-breaker 真正重定义网格；knob-on/off 能量不得混用）。读数：**4-折
tie-breaker 端到端验证含分支割良性跳变；k=3/5/6 回退不可能性为推导的
诚实极限（1/3-1/5 协变上界，规则无关）**。测量-only；默认与门槛不
变；全 knob opt-in。证据
`tasks/issue-9/2026-10-18-lc188-symtop-tiebreaker`（21 文件 sha256 锚
定，含 attempt-1 保留）；effects-index 104 数据集。发件箱保持在
f3e637a 发射（本段折入下次发射）。

**§87 续17（LC-189，2026-10-18；预注册 03:28:15Z sha256 1a562440 于运行
前）**：**表示噪声底的分解（命名 LC-185 后续项闭合）**——melatonin/
wb97mv/svp 旋钮 on、fp64、同 minao dm0，identity vs 旋转，max_cycle 截断
于 1/2/3/5/不限制（10/10 rc=0，03:28:58-03:35:06Z 串行单 GPU 6m8s，点
数恒同 396544，收敛循环 15/14）。**逐循环 dE_rot：c1 -3.092e-11 / c2
-4.002e-11 / c3 -2.774e-11 / c5 -2.797e-11 / 收敛 -1.614e-11**。**P1
MISS（信息性、决定性）**：c1 已达全幅 —— 残余在**第一次能量求值**（同
dm0 单次 Fock 构建）即完全出现，非 <= 收敛/10；预注册的轨迹放大假设被
证伪。**P2 MISS（信息性）**：全部循环点位于 [-4.0e-11, -1.6e-11] 平坦带
—— SCF 轨迹放大 ~1x。**P3 MISS-by-band（类复现）**：收敛值 -1.614e-11
vs LC-185 锚 -2.842e-11 同 1e-11 类、超出其会话内 <=3e-12 复现带 = 底的
精确值表示噪声实例依赖（本会话内跨循环点亦变）。**P4/P5 HIT**。**主读
数：表示噪声底由单次求值路径设定、不经 SCF 轨迹放大**——协变网格集作
为集合恒同（LC-184 单元级证明）但其旋转坐标是不同的浮点数，AO 值、
Becke 权重、xc/nlc 求值与 DF 中间量的末位差在 c1 即全幅出现（-3.1e-11
Ha = 4.0e-14 相对）并平坦保持到收敛；LC-185 的 "pipeline fp 表示噪声"
读取现在机制定位于求值通道。测量-only；默认与门槛不变；全 knob
opt-in。证据
`tasks/issue-9/2026-10-18-lc189-repnoise-decomposition`（8 文件 sha256
锚定）；effects-index 105 数据集。发件箱保持在 f3e637a 发射（本段折入
下次发射）。

**§87 续18（LC-190，2026-10-18；预注册 03:43:30Z sha256 84025bdd 于运行
前；惯性预检 03:4xZ 零 GPU 先于预注册）**：**molframe 广度验证 —— 冻结
medium-CHON 12 的其余 9 分子，(b6) 覆盖间隙闭合**。aspirin/caffeine/
dopamine/epinephrine/glycylglycine/lidocaine/nicotine/phenylalanine/
serotonin @ wb97m-v/def2-svp（LC-184 协议逐字；改正内联 dg 约定 G_rot @
U - G_id）；零 GPU 惯性预检：9/9 非对称陀螺（最小相对间隙 1.8e-2，全部
>> 1e-6 回退容差）→ 无回退、无 tie-breaker；72 子进程 lane-run（03:44:38
-04:28:01Z，43m23s，串行单 GPU，峰值 4.1 GiB，温度 <=38 C）。**P2
HIT**：knob-off fp64 identity 能量逐分子复现冻结 LC-168 锚至 <=4.5e-13
（2 格 bitwise 类 0.0）= 跨源码修订（e56c10e→9427e33）可复现性。
**P3 HIT**：精度门全过（max |dE_precision| 4.16e-10；max dg_precision
1.62e-6）。**P5 HIT**：18/18 lane-cell MOLFRAME_GRID=1 且 NLC=1；点数
identity==rot；循环全同（13/13、14/14、15/15）；墙价 0.977-1.043（零
溢价）。**P6 HIT**（72/72、峰值、温度、墙时 43m23s<100min 预算）。
**P1 MISS（信息性）**：LC-184/187 的 1e-10 knob-on |dE_rot| 带在全矩阵
广度不保持 —— epinephrine 7.38e-10（fp64）/6.59e-10（fp32knob）、
lidocaine fp32knob 1.61e-10；15/18 lane-cell <=8.4e-11、10/18 在 3e-11
子带；全部距不变的 1e-6 门 1.35e3x。**P4 MISS（信息性）**：3e-7 dg 带
同样破 —— epinephrine 4.61e-6（两 lane，迄今测得的最大 knob-on 梯度
不变性残差）、lidocaine 5.0e-7/5.2e-7；14/18 lane-cell <=2.4e-7；
全部距 3e-5 门 6.5x；错带格循环全同（13/13）= 非轨迹分叉，为实例
依赖的求值/底效应（LC-189 的底实例依赖；epinephrine 亦带 9 分子中
最大网格重定义位移 -1.38e-5 = 主轴帧离 lab 帧最远 —— 记录为观察，
不作机制断言）。**广度图（(b6) 输入）**：9/9 分子在 lab-frame L3 旋转
敏感 —— |dE_rot| 2.60e-6（nicotine）..2.46e-5（lidocaine）全部破 1e-6；
knob-off dg_rot 至 8.81e-5（aspirin）破 3e-5；旋钮关闭每一格：能量
还原 1.5e4..1.4e6x（aspirin 9.34e-6→2.6e-11、caffeine 1.56e-5→2.4e-11、
dopamine 8.3e-6→2.3e-11、epinephrine 1.09e-5→7.4e-10、glycylglycine
1.26e-5→2.0e-11、lidocaine 2.46e-5→5.4e-11、nicotine 2.6e-6→2.8e-11、
phenylalanine 7.6e-6→5.5e-12、serotonin 5.8e-6→2.4e-11）。**连同
LC-184/185：冻结 medium-CHON 12 全覆盖 —— 12/12 分子 lab-frame L3 旋转
破缺、12/12 被旋钮在默认级关闭、全部不变科学门通过**。网格重定义位移
（记录不设门）：-1.38e-5..+7.3e-8（knob-on/off 能量不得混用）。
读数：**紧 1e-10/3e-7 不变性带是小分子校准带；全矩阵广度下 knob-on
底实例依赖上探至 7.4e-10 Ha / 4.6e-6 Eh/Bohr（仍在科学门内 1.35e3x/
6.5x）；(b6) 的覆盖间隙以冻结矩阵上的零溢价全关关闭**。测量-only；
默认与门槛不变；全 knob opt-in。证据
`tasks/issue-9/2026-10-18-lc190-molframe-mediumchon-breadth`（28 文件
sha256 锚定）；effects-index 106 数据集。发件箱保持在 f3e637a 发射
（本段折入下次发射）。

**§87 续19（LC-191，2026-10-18；预注册 04:45:53Z sha256 bda506d4 于运行
前）**：**epinephrine 实例抬升机制探针（LC-190 后续 next_steps 命名项；
LC-189 协议逐字换分子）**——LC-190 中唯一实例抬升格（dE_rot -7.38e-10 /
dg_rot 4.61e-6，其余 11 分子 <=8.4e-11 带）的机制分解：epinephrine/
wb97mv/svp 旋钮 on、fp64、同 minao dm0，identity vs 旋转，max_cycle 截断
于 1/2/3/5/不限制（10/10 rc=0，04:47:04-04:51:10Z 串行单 GPU 4m06s，点数
恒同 310784 = LC-190，MOLFRAME_GRID=1 全程，收敛 13/13）。**逐循环
dE_rot：c1 -3.353e-8 / c2 +5.208e-7 / c3 -4.231e-9 / c4 +1.296e-9 /
c5 -4.527e-10 / c6 -6.921e-10 / 收敛 -7.382823241641745e-10**。**P1
HIT（决定性）**：抬升在**第一次能量求值**即全幅出现 —— c1（同 dm0 单次
Fock 构建，两表示）-3.35e-8 = 预注册 1e-10 阈的 335x、melatonin LC-189
c1 底（-3.09e-11）的 ~1085x；替代分支（c1 <=5e-11 且收敛放大 >=5x）被
决定性证伪于相反方向：**收敛/c1 = 0.022x —— 轨迹是收缩器不是放大器**。
**P2 MISS（信息性）**：逐循环序列非 c1 幅 3 倍带内平坦 —— c2 一步传播
瞬态 +5.21e-7（c1 幅 15.5x）后 c3 快速收缩（0.126x）、c5 起稳定平坦于
-7e-10 类（2 倍内）；melatonin 平坦模式不作为普适传递（轨迹形状实例
依赖），收敛净效应 = 收缩。**P3 HIT**：收敛 dE_rot 与 LC-190 锚
**bitwise 相同**（偏差 0.0）。**P4 HIT**（点数/循环/MOLFRAME_GRID 全验）。
**P5 PARTIAL**（10/10 rc=0、墙 4m06s、串行验证；本驱动未采样峰值 GPU/
温度 —— 驱动限制如实记录，由 LC-190 同分子同设置记录界定 <5.5 GiB/
<60 C）。**主读数：epinephrine 实例抬升是求值通道性质、被 SCF 轨迹收
缩（非放大）** —— 协变坐标单次 Fock 构建在此实例差 -3.35e-8 Ha（5.3e-11
相对；与 LC-189 定位的一般表示噪声底同机制类、实例幅度 ~1000x），一步
SCF 传播至 +5.2e-7 瞬态后 DIIS 收敛收缩至稳定 -7.4e-10 底、bitwise 复现
LC-190 锚；(b6) 读数不变（旋钮把 epinephrine 旋转隙从 lab 帧 1.09e-5 关
到 7.4e-10 = 1480x，全部科学门以已记录的 1.35e3x/6.5x 裕度通过），抬升
无需新缓解 —— 即一般底的机制带实例抬升幅度。跨进程循环级散布 <=4.5e-13
（内核归约序类），远低于全部判定信号。测量-only；默认与门槛不变；全
knob opt-in。证据
`tasks/issue-9/2026-10-18-lc191-epinephrine-elevation-probe`（8 文件
sha256 锚定）；effects-index 107 数据集。发件箱保持在 f3e637a 发射
（本段折入下次发射）。

**§87 续20（LC-197，2026-10-18；预注册 06:58:54Z sha256 bfa5b4d5 于运行
前）**：**fp32-VV10 修复的整 SCF 刚性变换验证延伸到准入带两端——
审查卡 item-2 的两条具名诚实边界闭合**（LC-174 止于 318 AO、无 UKS 格）：
tamoxifen 537 AO / 679,488 L3 点（NLC 带顶 = fp32 安全括号顶，已测最大
NLC 格）+ ethoxy UKS 67 AO spin-1 双重态（开壳层轴 = 项目 UKS 哨兵）。
LC-160/174 协议逐字（5 变换 × {fp64, fp32knob} 分离子进程，wb97m-v/
def2-svp，DF，conv_tol 1e-10，level 3，precision_mode fp64，UWE 内核
唯一变量）+ 两项申报增量：**更正约定 dg 协向量比较（G_rot@U−G_id，
LC-184 勘误/LC-190 驱动约定）**与 1-Hz 采样器（闭合 LC-174 的 P6 峰值
仪表缺口）。分离式 07:00:49-07:27:15Z = 26m26s，20/20 rc=0 全收敛，
峰值显存 9,260 MiB、25-59 C、串行单 GPU；锚定：tamoxifen identity
fp64 e_tot 复现 LC-196 锚 1.36e-12 @13 循环。**P1 HIT（决定性）**：
全部平移行双格双道过 1e-6 不变门（最差 |dE| 1.10e-9 @tamoxifen 1e6
Bohr = 900x 裕度；ethoxy 1.3e-12/2.8e-10）；fp32knob 带全 HIT
（tamoxifen 7.97e-10 @1e3 / 1.10e-9 @1e6 于 1e-9/5e-9 带内；ethoxy
于 1e-10/5e-10 带内）；子带 MISS（信息性）：预注册 fp64 底 <=1e-10
在带顶不成立（tamoxifen 7.9e-10/1.1e-9；ethoxy 1e6 2.78e-10）——
该底是**双道共享底**而非内核底：平移 lane 差 <=4.5e-12（tamoxifen）/
<=9.1e-13（ethoxy）= LC-174 <=1.3e-11 族延伸；ethoxy 1e6 值
（-2.78e-10）定量匹配 fp64 绝对坐标舍入底（eps×1e6 Bohr ~ 2.2e-10），
1e3（-4.8e-13）匹配其 2.2e-13 底；537 AO 处底转为轨迹主导（~1e-9）。
**P2 HIT 10/10**：最差 |dE_precision| 6.71e-10（1,488x 裕度）、最差
dg_precision 1.56e-7（192x 裕度）；tamoxifen identity 道偏移 +6.67e-10
跨源跨会话复现 LC-121 的 +6.52e-10（50a702c→1f1e16d）。**P3 HIT**：
旋转行双道 FAIL 1e-6 不变门（tamoxifen 3.868e-5 于预测 1e-5..1e-4 带；
ethoxy 3.094e-6 于 1e-6 类）；lane 平权 <=7.4e-10；10 条门失败记录
恰为这些行——保留不重述（文档化缓解不变：LC-162 level-4、LC-183+
molframe）。**P4 SPLIT**：tamoxifen 更正约定 dg_rot 4.15e-5 过 3e-5 门
（已知类，lane 平权）；信息性 MISS——ethoxy dg_rot 2.840e-5 **位于
门内**（5% 裕度）= 首条在默认级过梯度门的旋转行；子带 MISS：tamoxifen
平移 dg_invar 6.6-7.4e-7（2.2-2.5x 超 3e-7 子带、门下 40x）、
rot_trans−rot 1.25e-9（超 1e-9 子带 25%）。**P5 HIT**：全部行循环
恒同（13/13、14/14，含旋转——严于预注册 ±1）；ethoxy UKS 双重态双道
干净收敛无分支事件（每模式道间一致至 1e-11 类）。**P6 HIT**（20/20、
9,260 MiB、59 C、26m26s、串行、采样最大间隙 1.16s）。**主读数：
(1) 修复的整 SCF 平移不变性现已覆盖全部已测尺寸带（48-537 AO）与两
壳层轴（RKS+UKS）于 HEAD，门槛不变；(2) 机制在带顶确认——修复后平移
误差**幅值无关**（tamoxifen 8e-10 @1e3 vs 1.1e-9 @1e6 同类；修复前
methanol 在该区间跨 5 个量级并在 1e6 Bohr 破门）且 **lane 共享**
（fp32 内核不增添平移响应：lane 差 <=4.5e-12）；537 AO 的 ~6.7e-10
道偏移是 identity 处的 fp32 近似常数、不随平移变化 = LC-121 精度族，
门下 3 个量级；(3) 旋转在每一带与壳层轴仍是网格求积类（处处 lane
平权），ethoxy 的 dg_rot 是首条门内旋转行。测量-only；默认与门槛
不变；全 knob opt-in；源未动（全程 HEAD 1f1e16d）。证据
`tasks/issue-9/2026-10-18-lc197-vv10fix-bandtop-uks-rigid-transform`
（12 文件 sha256 锚定，含 1-Hz 采样器 1503 行）；effects-index 111
数据集。发件箱保持在 1f1e16d 发射（本段折入下次发射）。
