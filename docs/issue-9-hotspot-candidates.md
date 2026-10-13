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
