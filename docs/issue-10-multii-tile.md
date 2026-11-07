# Issue #10 rework — multi-i tiled VV10 UWE kernel（有界 tile 尺寸试验）

**状态**: review_requested（暂用，未验收）· **源码**: `b8d503c` + 本
issue 提交 · **证据**:
`/data/results/gpu4pyscf-agent/tasks/issue-10/20261006-multii-tile/`
（不可变，见本文末 sha256 锚定）

按 control/issue-10-next-round.md（Codex review 2026-10-06 P2 项）：
round-1 的 `warp4` 内核每 block 只服务一个 i、每个 i 都要把整个
j-网格重新搬入 shared memory（全局读放大 128×），而参考内核每 block
服务 128 个 i、j chunk 只搬一次；旧源码注释把参考内核描述成
"每 block 只有 1 个 warp 干活" 的并行度缺陷，掩盖了它流量最优的
事实。本轮按"共享 j staging 的 multi-i tile"路线重做，并修正注释。

## 1. 实现（opt-in，默认关闭不变）

`gpu4pyscf/lib/gdft/vv10.cu`（继承 round-1 的 `warp4` 与选择器框架）：

- 新增模板内核 `vv10_fock_eval_UWE_multii_kernel<I_TILE>`（I_TILE ∈
  {8, 16, 32}）：每 block 服务 I_TILE 个 i 点，128 线程分成
  128/I_TILE 人一组（4..32 线程，组界与 warp 对齐），组内以
  stride=组大小 走 j-循环，块内共享的 j chunk staging 摊薄到
  I_TILE 个 i 上（全局读放大 128/I_TILE = 16×/8×/4×，warp4 为
  128×，参考内核为 1×）。per-pair fp64 表达式与参考内核逐字相同，
  唯一差别是 fp64 求和顺序（组内 shuffle 树归约 vs 串行链）。
- 边界正确性：部分 tile（尾块 i 数 < I_TILE）与尾 chunk
  （j 数 < 128）由统一的 `block_upper_bound` 与"无提前退出"结构
  处理——所有线程到达两个 `__syncthreads()`（死锁/竞态不可能），
  每个活跃 i 恰由组 lane 0 单点写出（写竞争不存在）。静态断言把
  I_TILE 限制在 4..32（归约保持在单 warp 内，`__shfl_down_sync`
  的全 warp mask 均匀合法）。
- 选择器扩展：`GPU4PYSCF_VV10_UWE_KERNEL` 新增 `tile8/tile16/tile32`
  取值；未设/空/未知值仍为 warp1 参考（逐位不变 = 严格 FP64 回退，
  未知值警告并忽略）。**默认路径行为与本 issue 之前逐位相同**。
- 注释修正（review P2 项）：参考内核段落现描述真实设计（每 block
  128 个 i、j chunk 每 block 只搬一次、读流量最优；其局限是
  并行度不足，小网格时欠占用），warp4 段落如实描述其 128× 读放大
  代价与 0.944x–1.152x 的混合实测结果。

## 2. 等价性与调度证据

### 2.1 合成边界扫描（`multii_equivalence.json`）

27 个网格尺寸（覆盖 I_TILE×128 tile 边界 1024/2048/4096 的 ±1、
参考内核 128 线程块边界、尾 chunk ngrids%128 ∈ {1..5,127}、部分尾
tile ngrids%I_TILE、以及一般内部点），输入伪随机、输出预填 NaN
（漏写即失败）。三个 tile 变体 + warp4 对照，全部：

- **all_written = true**（无未写槽位，NaN 探针无命中）；
- dU/dW/dE 最大偏差 ≤ **1.3e-14**（fp64 归约顺序噪声量级，相对
  偏差 ≤ 4.4e-15）；
- CPU numpy 同表达式实现交叉验证参考内核（最大相对偏差 ≤ 4.3e-15，
  排除集体性算术错误）。

### 2.2 实网格重放（同文件 real_replay）

4 个已准入 cell 的 round-1 捕获 `_vv10nlc` 输入（61k–128k 网格点）
重放：tile8/16/32 与 warp4 的 dU ≤ 4.3e-17、dW ≤ 3.7e-16、
dE_grid ≤ 4.4e-16，全部逐位可复现；round-1 warp4 归档输出被逐位
复现（真实输入回放的溯源校验）。

### 2.3 实际调度选择（`multii_dispatch2.json`，本轮新增）

round-1 类比块的 `dispatch_isolation` 有 harness 缺陷（worker 对
warp1 运行会先 pop 环境变量，继承的未知值从未真正到达 C 选择器，
记录的 `stderr_had_warning=false` 是假象）。本轮用父进程设置环境、
子进程只跑内核的方式重测（ngrids=2055，与扫描同种子输入）：

- 回退类（未设/空/`warp1`）输出与默认**逐位相同**；
- 未知值（`tile7`、大小写变体 `TILE8`、近邻 `warp2`）同样回退到
  warp1 **且打印选择器警告**（10/10 case 通过）；
- 选择类（`warp4`/`tile8`/`tile16`/`tile32`）输出与归档扫描输出
  **逐位相同**，且与 warp1 参考逐位**不同**（dU ~3–4e-15）——
  证明选择器真实路由到了 tile 内核而非静默回退。

### 2.4 sanitizer（`sanitizer_report.txt`）

compute-sanitizer racecheck（tile8/16/32，ngrids=2055 部分尾 tile）
与 memcheck（ngrids=4224）全部 **0 hazards / 0 errors**；驱动程序
内核与仓库 vv10.cu 逐语义一致（static_assert 与声明合并差异除外，
已核对）。

## 3. 性能证据

### 3.1 微基准（`microbench_all.log` + `multii_microbench.cu`）

独立 .cu（内核与 vv10.cu 语义一致）：fp64 FMA 链测得算力天花板
（1.11–1.18 TFLOPS）、D2D 拷贝测得 DRAM 带宽（~918 GB/s）、
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` + `cudaFuncGetAttributes`
测得占用（blocksPerSM 9–10、regs 42–52、shared 6–9 KB）、
每次迭代 deviceSynchronize 后计时 3 次取中位。硬件计数器被
ERR_NVGPUCTRPERM 权限阻断（实测记录 `ncu_permission_check.txt`），
故读流量以建模代理（48 B/j 元素 × 128/I_TILE 放大，全 miss 上界）
+ 等时间-不同流量的自然实验（tile32 流量 = warp4/32 但时间≈warp4）
归属。4 个真实网格尺寸（61k/52k/126k/128k）：

| 网格 | warp1 | warp4 | tile8 | tile16 | tile32 |
|---|---|---|---|---|---|
| 61k (methanol) | 134.6 ms | 104.9 | **104.6** | 106.7 | 106.9 |
| 52k (methoxy) | 156.4 ms | 145.8 | **145.3** | 145.4 | 145.5 |
| 126k (benzene svp) | 640.0 ms | 619.1 | **616.1** | 620.2 | 620.9 |
| 128k (benzene tzvp) | 651.3 ms | 638.9 | **634.6** | 639.4 | 649.3 |

内核级规律：tile8 全尺寸最优（warp1 的 0.78×/0.93×/0.96×/0.97×
时间），随网格增大收益收敛到与 warp4/tile 变体几乎重合——大网格
下所有变体都到 fp64 屋顶（pairs_per_s 2.5–2.6e10 一致），小网格
欠占用才是 warp1 的真实瓶颈。tile8 在实测中同时有最低读放大增益
（16×）与最高占用（blocksPerSM=9 但 i 并行度 ×16）。

### 3.2 端到端 SCF 匹配计时（`retest_*.json`，rtx4090-energy-v1）

5 个已准入 ωB97M-V cell × {warp1, tile8} × {fp64, auto}，runner
逐格 3 独立样本 + 弃置预热、deviceSynchronize、端到端含全部开销
（源 commit 同为 b8d503c，设置与 #7 生产运行逐项一致，10/10 文档
schema-VALID）。验收量 = 变体内 auto vs fp64；诊断量 = 变体间差与
中位时间比（`multii_retest_analysis.json`）：

| cell | dE_auto(warp1) | dE_auto(tile8) | \|dE(t8−w1)\| | speedup fp64 | speedup auto |
|---|---|---|---|---|---|
| s0_methanol_svp | 0.0 | -1.1e-13 | 5.7e-14 | **1.076** | **1.125** |
| s1_benzene_svp | 4.5e-13 | 1.7e-13 | 3.4e-13 | 1.061 | 1.049 |
| s1_methoxy_svp_uks | **-1.283e-6** | **-1.283e-6** | 9.1e-12 | **1.217** | 1.041 |
| s2_benzene_tzvp | 2.1e-12 | 6.4e-12 | 4.6e-12 | 0.981 | 1.024 |
| s2_methoxy_tzvp_uks | 2.6e-8 | 2.6e-8 | 2.8e-13 | 1.031 | **1.106** |

（methoxy/svp UKS auto 车道 tile8 收敛 19 循环 vs warp1 11 循环——
SCF 路径噪声，见 §5；其余格循环数差 ≤ 1。）

### 3.3 SCF 级能量与 NLC 分量（`multii_scf_probe.json`）

4 cell × 3 tile 变体 fp64 SCF：|dE_tot| ≤ 1.1e-13 Ha（门内 10 个
数量级余量），NLC 能量分量差 ≤ 2.1e-14 Ha，循环数全部相同；
tile8/tile16/tile32 相互间也无可见差异——内核级 1e-16 量级的
归约顺序噪声在 SCF 收敛后不可见。

## 4. 处置

- **tile8/tile16/tile32 全部保持 opt-in、默认关闭**（默认 = warp1
  参考，逐位不变）。理由：计划要求"全矩阵净加速为正"才可上线；
  实测端到端 0.98×–1.22×（median-of-3），benzene/tzvp fp64 仍为
  0.981（负加速），收益随网格增大收敛到持平——**未达上线线，
  如实记录，不发明加速**。microbench 显示大网格下所有变体都到
  fp64 roofline（2.5–2.6e10 pairs/s 一致），该内核在小网格
  （≤ ~60k 点）欠占用场景有真实收益（tile8 1.29×），benzene 级
  128k 网格收益 ~1.02×。
- **warp4 保留**（round-1 opt-in 路径不动，注释已修正）。
- **使用边界**（供后续 #12/生产参考）：VV10 UWE 内核选择只影响
  NLC 求值内层；小网格体系（S0/methanol 级）tile8 是当前最优
  opt-in；大网格（benzene 级及以上）任何变体收益 ≤ 2%，且
  benzene/tzvp fp64 为负，建议维持默认 warp1。

## 5. 继承失败保持可见（未解决，非本轮范围）

methoxy/def2-svp UKS ωB97M-V auto 车道 -1.283e-6 Ha（issue #7
deferred 家族）在 tile8 下**原样复现**（-1.2826e-6 → -1.2826e-6，
偏移移动 9.3e-12 Ha，`multii_scf_probe.json`
inherited_uks_auto_reproduction 块）——内核等价性不构成 SCF 解选择
问题的修复，该格 verdict 保持 fail、scientific_gate_passed=false。
这是 #7 家族的能量侧问题，按用户连续策略记录后继续，不阻塞
#11/#12 的独立工作。（本轮不重跑已否决的 DIIS reset。）

## 6. 开关文档

| 取值 | 行为 |
|---|---|
| 未设 / 空 / `warp1` / 未知值 | 参考 warp1 内核（默认，逐位历史路径；未知值打印警告并忽略） |
| `warp4` | round-1 多 warp 内核（每 block 一个 i，128× 读放大；0.944–1.152× 混合结果） |
| `tile8` / `tile16` / `tile32` | multi-i tile 内核（读放大 16×/8×/4×；tile8 实测最优，小网格 ~1.29× 内核级、端到端 ≤ 1.22×） |

进程级一次读取（C 侧），Python 不读该变量（单测钉死）。

## 7. 复现

```bash
# 边界/尾块/调度等价（GPU）
/home/qczcode/gpu4090-src/bin/python \
  /data/results/gpu4pyscf-agent/tasks/issue-10/20261006-multii-tile/multii_equivalence.py
/home/qczcode/gpu4090-src/bin/python \
  /data/results/gpu4pyscf-agent/tasks/issue-10/20261006-multii-tile/multii_dispatch2.py
# sanitizer（racecheck + memcheck）
bash -c 'cd /data/tmp/gpu4pyscf-agent/execution/issue10_rework && \
  /usr/local/cuda-12.8/bin/compute-sanitizer --tool racecheck ./sanitize_driver <variant> <ngrids>'
# 微基准
cd /data/tmp/gpu4pyscf-agent/execution/issue10_rework && \
  /usr/local/cuda-12.8/bin/nvcc -O3 -arch=sm_89 multii_microbench.cu -o multii_microbench.bin && ./multii_microbench.bin
# 端到端矩阵 + 分析
bash /data/results/gpu4pyscf-agent/tasks/issue-10/20261006-multii-tile/retest_matrix.sh
/home/qczcode/gpu4090-src/bin/python \
  /data/work/gpu4pyscf-mixedprec/mixedprec/benchmark/rtx4090/multii_retest_analysis.py \
  --out /data/results/gpu4pyscf-agent/tasks/issue-10/20261006-multii-tile/multii_retest_analysis.json
# SCF 探针
/home/qczcode/gpu4090-src/bin/python \
  /data/results/gpu4pyscf-agent/tasks/issue-10/20261006-multii-tile/multii_scf_probe.py
```

仓库内交付物：`gpu4pyscf/lib/gdft/vv10.cu`（内核+注释+选择器）、
`mixedprec/benchmark/rtx4090/multii_retest_analysis.py`（泛化分析，
拒绝路径有单测）、`tools/tests/test_issue10_multii_tile.py`（选择器
契约/结构契约/分析拒绝路径/证据可见性，311 个工作流测试全过两种
政策）、本文件。
