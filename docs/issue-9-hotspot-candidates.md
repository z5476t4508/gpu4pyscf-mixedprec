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
  底，50 cycle 上限烧尽）。且**每 cycle 仅快 1.1–1.3x**（如 b3lyp/svp：
  0.106 vs 0.137 s/cycle）——43–222 AO 的 GEMM 太小，launch/host 开销主导，
  64:1 硬件比远未兑现；端到端 fp32 lane 比 fp64 **慢 4–8x**。混合精度的价值
  只能走 auto lane（fp32 头 + fp64 尾），即 #5–#8 已验证 ≤1e-6 Ha 的那条路。
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
