# 混合精度 HF 项目状态 (RTX 5090)

目标 (2026-08-25 更新): **几十万个 xyz 分子的 DF-RHF 单点能量 + AO 基密度矩阵**
批处理吞吐。体系 ~57 原子, 基组 def2-SVP/6-31G*。原始目标 (几何优化混合精度)
已完成阶段一并保留在下面。

**⏸ 暂存点 (2026-08-25): 项目暂停于此, 下一步已论证未实施 ——
搭建批量单点管线 (GPU 纯 FP32 筛选通道 + 可选 CPU 进程池兜底 + h5 分片
落盘 + 断点续跑)。恢复时从「待办/下一步」一节直接开工, 无需重新评估。**

**▶ 2026-09-04 恢复: 批量单点管线 (GPU 通道) 已实现并实测通过 ——
`mixedprec/batch_rhf.py` + `mixedprec/test_batch_rhf.py` (36 测试全过)。
详见下面「批量管线」一节。CPU 兜底通道仍未做。**

## 环境 (已就绪, 2026-08-22)

- venv: `/home/tong/soft/gpu4pyscf/.venv` (python 3.14, `--system-site-packages` 复用系统 pyscf 2.14.0)
- 已装: gpu4pyscf-cuda13x 1.8.1, cupy-cuda13x 14.2.0, cutensor-cu13 2.7.0, nvidia-nccl-cu13, cmake
- cutensor 修复: wheel 的 lib 不在默认搜索路径, 已放 `site-packages/_g4p_preload.py` + `g4p_preload.pth` 自动 ctypes 预加载 (否则会看到 "using cupy as the tensor contraction engine" 警告, 即 cutensor 未生效)
- **坑**: 在 repo 目录里运行 python 会导入源码树 (无 .so) 而不是 wheel —— 所有脚本从 `/tmp` 等其他目录运行:
  `cd /tmp && /home/tong/soft/gpu4pyscf/.venv/bin/python <script>`

## 阶段一结论 (2026-08-25, 全部实测, 数据可信)

### step2: SCF 混合精度 (Tamoxifen 56 原子/1042 AO/def2-TZVP, `step2_result.txt`)

| 配置 | 耗时 | 加速 | 能量误差 vs CPU |
|---|---|---|---|
| FP64 conv 1e-10 | 18.74s (15 轮) | — | 6.9e-11 |
| **混合 conv 1e-10** | **13.59s** | **1.38x** | 5.8e-11 |
| FP64 conv 1e-7 | 14.80s | — | — |
| **混合 conv 1e-7** | **11.01s** | **1.34x** | 3.1e-9 |

- 单轮 SCF: FP32 0.25s vs FP64 0.86s (3.5x); get_jk 单次 0.21s vs 0.82s (3.9x)
- 精度: vj rel err 2.1e-7, vk 1.9e-6, SCF 能量误差 < 1e-9, 零精度损失
- 切换策略: FP32 起步, |dE|<1e-4 单次触发切 FP64 (两次连续太保守: FP32 噪声
  底让 dE 切换后反弹到 4.5e-5, 反而多耗轮次)

### step3: 几何优化端到端 (10 步 geomeTRIC, `step3_result.txt`)

| | FP64 | 混合 | |
|---|---|---|---|
| 总耗时 | 273.4s | 240.2s | **1.14x** |
| SCF (11 次) | 137.7s (50%) | 104.3s (43%) | 每步 ~11.7→~9.3s |
| 梯度 (11 次) | 134.3s | 134.4s | 12.2s/次, FP64 不变 |
| 每步能量差 | — | — | max 1.0e-5, 优化轨迹一致 |

### 每个优化步的成本模型 (1042 AO)

- CDERI 重建: **4.2s** (每几何步, `df.reset()` 清缓存; 其中 `fill_int3c2e`
  CUDA 内核 3.0s = 无法 Python 层优化)
- SCF: FP64 ~11.7s / 混合 ~9.3s (= CDERI 4.2 + 收缩 ~4 + 尾部 FP64 轮 + 杂项)
- 梯度: **12.2s** (全部 FP64; 其中 `sum_ejk_int3c2e_ip1` 导数积分内核 79%)

**SCF 侧混合精度已到顶 (整体 1.14x), 剩余收益都在 CUDA 内核里:**

1. 梯度内核 `sum_ejk_int3c2e_ip1` (9.4s/步, 79% of grad) → FP32 化后若 2-3x,
   梯度降到 ~5s, 整体 ~1.6x
2. CDERI 的 `fill_int3c2e` (3.0s/步) → 同属 gint int3c2e 家族, 可与上面一起做

## 关键实现发现 (fp32_jk.py)

1. **cast 布局陷阱** (本次最大坑): `df.loop()` 产出的 cderi 块是
   (nL,nao,nao) 转置视图 (L stride=8)。在其上 `astype(float32)` 触发非合并
   访问拷贝, 108ms/块 × 11 块 ≈ 0.8s, 比 FP32 GEMM 省下的还多 —— 首次重跑
   出现 0.66x "负加速" 的根因。修复: 在其 C 连续父缓冲 (nao,nao,nL) 上 cast
   (34ms/块) 再转置; cutensor 原生吃跨步张量且跨步 FP32 收缩比连续还快。
2. **callback 局部变量**: gpu4pyscf `_kernel` 的 locals() 没有 `e_delta`,
   只有 `e_tot`/`last_hf_e` (CPU pyscf 同样)。
3. **全局模式不会自复位**: scanner 场景每个几何步的 SCF 都要手动把
   PRECISION_MODE 重置回 fp32 (包一层 mf.kernel), 否则只有第一步是混合精度。
4. **梯度类打点**: density_fit 的梯度是 `gpu4pyscf.df.grad.rhf.Gradients`
   (df_jk.py:225 分发), 不是 `gpu4pyscf.grad.rhf.Gradients`。
5. fill_symmetric 混合 dtype (fp64 入 fp32 出) 会静默产生垃圾值 —— decompress
   内核是 double 硬编码, 不能用这种方式省 cast。
6. 5090 实测: cublas FP32 ~72 TFLOPS vs FP64 ~1.7 TFLOPS; cutensor 3-index
   收缩 FP32 ~17 TFLOPS vs FP64 ~1.5 TFLOPS。

## 目标体系确认 (2026-08-25)

- 用户实际需求: **几十万个分子**的 DF-RHF 计算 (无 DFT), 体系 ~Tamoxifen 尺寸 (57 原子),
  基组 def2-SVP 或 6-31G* (def2-TZVP 只作为压力测试基准)
- 小基组实测 (6 步 geomeTRIC, `/tmp/basis_scale.log`):

| 基组 | AO | FP64 | 混合 | 加速 | SCF/步 | 梯度/步 |
|---|---|---|---|---|---|---|
| def2-TZVP | 1042 | 273.4s | 240.2s | 1.14x | 11.7→9.3s | 12.2s |
| def2-SVP | 537 | 63.3s | 55.8s | 1.14x | 3.8→2.5s | 5.1s |
| 6-31G* | 450 | 43.5s | 38.5s | 1.13x | 2.7→1.8s | 3.4s |

- def2-SVP 分解: 梯度 5.1s 中 `sum_ejk_int3c2e_ip1` 3.8s (75%), CDERI 构建 1.1s/步
- **小基组下梯度占比反而升到 55-64%** —— 阶段二 (内核 FP32 化) 的相对价值更高;
  但绝对收益变小 (每步只省 ~1.5-2s)。6-31G* 无原生 JKFIT, DF 场景推荐 def2-SVP。

## CPU vs GPU 定性评估 (2026-08-25 实测, 57 原子 def2-SVP 单点+AO DM)

机器: Intel Ultra 9 285K (24 核) + RTX 5090。

| 方案 | 单分子耗时 | 整机吞吐 | 10 万分子 |
|---|---|---|---|
| CPU 全 24 线程 1 进程 | 80.0s | 45/mol/h | 92 天 |
| CPU 8 线程 × 3 进程 | 35.0s ×3 并行 | 309/mol/h | 13.5 天 |
| CPU 2 线程 × 12 进程 | 113.2s ×12 并行 | 382/mol/h | 11 天 |
| **GPU FP64** | 4.51s | 800/mol/h | 5.2 天 |
| **GPU 混合** | 2.70s | 1333/mol/h | 3.1 天 |
| **GPU 纯 FP32 筛选** | 1.94s | 1856/mol/h | 2.2 天 |

- CPU 线程扩展性差: 24 线程只有 ~1.7x 于 8 线程 ×3 进程 (DF 的 Cholesky/
  GEMM 内存带宽瓶颈); CPU 整机峰值 ~380-450 mol/h vs GPU 800-1850 mol/h
- **GPU 优势 2-4.7x**, 且 GPU 方案还有第二个隐藏优势: 只占几十 GB 内存中的
  ~3GB, CPU 128GB 内存可以同时喂 24 核 CPU worker + GPU —— 混合模式
  (GPU 为主 + CPU 进程池兜底) 理论可达 ~2000-2200 mol/h
- AO 密度矩阵 (537×537, 2.3MB fp64) 是 SCF 免费副产品, make_rdm1 <1ms;
  存储: 10 万分子 ≈ 230GB (fp64) / 115GB (fp32, 精度足够) —— 存储是
  千亿字节级, 需要规划 (h5 分片或按需只存缩减量)
- 定域化选项 (若未来需要): PM 102s/分子 (不可行), NAO 1.1s (可用),
  Mulliken/Loewdin 电荷 0.5s (可用)

## 批量吞吐 (几十万分子场景, 2026-08-25 实测)

单点 SCF (def2-SVP + jkfit, 57 原子, 单卡 5090):

| 通道 | 精度 | 速度 | 10 万分子耗时 |
|---|---|---|---|
| FP64, conv 1e-7 | 基准 | 4.51s/mol | 5.2 天 |
| 混合 (FP32→FP64@1e-4), conv 1e-7 | <3e-9 | 2.70s/mol (1.67x) | 3.1 天 |
| **纯 FP32, conv 1e-5 (筛选)** | 1e-4 Eh, 排序无损 | **1.94s/mol (2.3x)** | **2.2 天** |

- 纯 FP32@1e-5: 能量绝对误差 ~1e-4 Eh, 但**分子间能隙误差 max 0.044 kcal/mol,
  Spearman 排序相关 = 1.0** —— 对构象排序/初筛完全够用
- 纯 FP32 不能用 conv<1e-5 (FP32 噪声底让 dE 在 4e-5 附近震荡, 收敛假象);
  需要精确能量时用混合通道 (switch@1e-4) 收尾
- **辅助基组陷阱**: def2-universal-jfit (1691) 比 svp-jkfit (2626) 快 1.6x 但误差
  7.6 kcal/mol; def2-svp-ri 11.2 kcal/mol —— RI/universal 拟合基是给 MP2/CC 的,
  J/K 必须用 jkfit。勿贪这个加速。
- 进程级: import+JIT 预热一次性 ~5s, mol.build/mf 创建 ~35ms —— 批处理时每进程
  装载一批分子即可摊平; 无需 CUDA Graph (GPU 占比 >95%)
## 批量管线 (2026-09-04 实现并实测)

`mixedprec/batch_rhf.py` — 目录多 XYZ → GPU DF-RHF 单点 → HDF5 分片, 可断点续跑。

```sh
PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python mixedprec/batch_rhf.py \
    --input-dir mols/ --output-dir results/ --shard-index 0 --num-shards 4
```

- **运行方式变了 (旧「cd /tmp」结论已过期)**: venv 里的 wheel 是发布版 1.8.1,
  **没有** `gpu4pyscf.lib.precision`, 混合精度会被静默降级成 FP64。源码树
  已于 2026-09-03 编译好 `.so`, 所以现在必须用 `PYTHONPATH` 指向源码树。
  驱动启动时 preflight 检查该模块, 缺失就立刻报错退出 (而不是把几十万个
  分子逐个记成 failed)。
- 输入: 每个 `.xyz` 一个分子, 递归扫描, 按相对路径稳定排序; 电荷/自旋从注释行
  `charge=-1 spin=0` 解析, 缺省用 `--charge/--spin`。
- 输出: 每分片一个 h5。DM 用 ragged 布局 (`dm_data` 一维 + `dm_offset`/`nao`),
  float32; 能量 float64。**先写数据 flush, 再标 completed 并 flush** ——
  实测 SIGKILL 打断后, 被中断分子停在 `running` 且无残缺数据, 重开自动
  复位成 pending 重算, 已完成的跳过。
- 续跑校验: schema 版本 + 分子列表指纹 + 全部配置项 (basis/auxbasis/mode/
  conv_tol/...) 必须完全一致, 否则直接报错, 不会把不同设置的结果混进同一文件。
- 单分子失败 (坏 XYZ、SCF 发散) 只记 `failed` + 错误类型/traceback, 不中断整批;
  环境类 ImportError 则视为致命并中止。
- 每个分子算完强制 `precision.set_precision('fp64')` —— `scf/hf.py:_kernel`
  没有 try/finally, SCF 抛异常会把全局 fp32 泄漏给下一个分子。

### conv_tol 重要修正 (实测, 推翻原「conv 1e-5」推荐)

原结论用 `conv_tol=1e-5` 跑纯 FP32 筛选。实测发现该值**落在 FP32 噪声底
(~4e-5) 以下**, 收得更紧买不到任何精度, 只会让迭代数run-to-run 剧烈抖动:

| 分子 | conv_tol | 迭代数范围 | 最慢 | 误差 vs FP64 |
|---|---|---|---|---|
| Azadirachtin (934 AO) | 1e-5 | **9-38** | 18.8s | 3.3e-4 |
| Azadirachtin | **3e-5** | 7-12 | 12.2s | 3.4e-4 |
| Azadirachtin | 1e-4 | 7-11 | 12.0s | 3.9e-4 |
| Tamoxifen (537 AO) | 1e-5 | **11-22** | 2.2s | 1.7e-4 |
| Tamoxifen | **3e-5** | 9-9 | 1.7s | 1.4e-4 |
| Tamoxifen | 1e-4 | 6-6 | 1.5s | 1.8e-4 |

- 误差在三档之间基本不变 —— 精度由 FP32 噪声底决定, 不由 conv_tol 决定。
- 同一进程内跑 4 遍**完全相同**的分子, 能量散布 ~4e-5、迭代数 11/20/11/11:
  FP32 收缩本身不确定 (与 blksize 无关, 实测 blksize 恒为 256, 清显存池也无效)。
- **默认已改为 `--conv-tol 3e-5`**, 严格优于 1e-5 (稳定 + 更快 + 同等精度)。
  10 万分子规模上这能省掉尾部分子最多 ~50% 的墙钟时间。
- 需要更紧收敛时用 `--mode auto` (FP32 起步 + FP64 收尾), 不要压 fp32 的 conv_tol。

## 批量单点的耗时分解 (2026-09-05 实测) —— 优化重心已转移

def2-SVP + jkfit, fp32 通道, 单卡 5090:

| 分子 | AO | mol.build | **CDERI 构建** | h1/S | SCF 迭代 | 总计 |
|---|---|---|---|---|---|---|
| Vitamin C | 208 | 0.01 | 0.32s (38%) | 0.03 | 0.48s (57%) | 0.84s |
| Tamoxifen | 537 | 0.00 | **1.08s (67%)** | 0.04 | 0.50s (31%) | 1.63s |
| Azadirachtin | 934 | 0.00 | **8.55s (69%)** | 0.08 | 3.69s (30%) | 12.31s |

**阶段一优化的 SCF 现在只占 30%** —— 即使 SCF 归零, 整体也只快 30% (Amdahl)。
CDERI 内部: Cholesky 仅 0.09s, 几乎全部时间在 GPU, 其中最大一块是变换
GEMM `j3c · aux_coef` (Tamoxifen 1.3 TFLOP, Azadirachtin 11.1 TFLOP)。

### ❌ fp32 变换 GEMM + Cholesky: 不可行 (2026-09-05, 确认 8 月结论)

朴素 fp32 化 (保留默认的 Cholesky 分解) 精度不可用:

| 分子 | CDERI fp64 | CDERI fp32 | 加速 | 能量误差 |
|---|---|---|---|---|
| Tamoxifen | 1.08s | 0.23s | 4.7x | **3.8 Hartree**, 50 轮不收敛 |
| Azadirachtin | 8.54s | 1.20s | 7.1x | **81 Hartree**, 50 轮不收敛 |

根因 (这次查清了):

- `aux_coef = L^-1 C` 来自二中心度规的 **Cholesky** 分解, L 是三角阵;
  点积中相消极剧烈 —— 实测相消因子 (sum|term|/|result|) **中位数 4387**,
  最大 6.2e6。fp32 eps 6e-8 × 4387 ≈ 2.6e-4, 与实测 **4.7e-4** 吻合,
  也与 8 月记录的 5e-4 精确一致 (同一现象独立复现两次)。
- 曾误判为"当年踩的是 TF32"。**已证伪**: CUPY_TF32 未设, 走的是真 fp32;
  同尺寸随机矩阵 GEMM 只有 3.1e-6 误差 —— **随机矩阵没有相消, 该基准
  不能代表真实情况**, 这是当时误判的来源。
- 对照: fp64 GEMM 建好 CDERI **再 cast 成 fp32**, 误差仅 1.47e-4 —— 问题在
  **累加**, 不在 fp32 存储。
- cuBLAS FP64 仿真 (`cublasSetEmulationStrategy`, cuBLAS 13.2.1 + sm_120):
  符号存在、调用返回成功、Get 能读回设定值, 但**完全不生效** (1.7 TFLOPS
  不变, 误差为零)。消费级 Blackwell 未开放。环境变量
  `CUBLAS_EMULATION_STRATEGY=eager/performant` 同样无效。

### ✅ fp32 CDERI + 特征分解: 可用, 已实现 (opt-in)

**关键**: `aux_coef` 的平方根不唯一。改用**特征分解** (`V·W^{-1/2}`, V 正交)
代替 Cholesky, 同一个 GEMM 的 fp32 误差从 4.7e-4 降到 **2.4e-6 (200 倍)**,
条件数几乎不变 (6.4e4)。eigh 本身很便宜 (naux=2626 只要 0.11s)。

实测 (def2-SVP, fp32 通道, conv 3e-5):

| 分子 | 总计 fp64 CDERI | 总计 fp32 CDERI | 加速 | 绝对误差 |
|---|---|---|---|---|
| Vitamin C (208 AO) | 0.18s | 0.15s | 1.2x | 1.1e-5 → 4.8e-4 |
| Tamoxifen (537 AO) | 1.58s | 0.90s | **1.76x** | 1.3e-4 → 2.4e-3 |
| Azadirachtin (934 AO) | 13.01s | 4.62s | **2.8x** | 2.9e-4 → 1.05e-2 |

CDERI 构建本身: Azadirachtin 8.54s → 1.57s (5.4x)。

**排序质量代价 (10 个扰动构象, 跨度 49.8 kcal/mol)**:

| | 配对能隙误差 max | Spearman |
|---|---|---|
| fp64 CDERI (现状) | **0.090 kcal/mol** | 1.000000 |
| fp32 CDERI (新) | **0.944 kcal/mol** | 1.000000 |

排序保住了, 但配对误差涨 10 倍。**因此设为 opt-in, 默认仍是 fp64**:

- `precision.set_cderi_precision('fp32')`, 或批处理 `--cderi-precision fp32`
- 只允许与 `--mode fp32` 组合; 与 auto/fp64 组合会被拒绝 (fp64 尾轮需要
  fp64 CDERI)
- 构象能量间距若只有 1-3 kcal/mol, **不要用** —— 0.94 kcal/mol 会混淆相近对
- host-memory CDERI 路径不支持 (C 层 transpose_write 是 double 硬编码),
  会自动回退 fp64

**剩余空间**: 补偿式 GEMM (Ozaki) 仍可在不损失精度的前提下拿到类似加速,
但那是独立数值工程项目; 现在有了 opt-in 的快通道, 优先级下降。

### ❌ auto 通道的 fp32 增量尾巴: 已实现并实测, 无可复现收益 (2026-09-05)

auto 的时间分解 (Azadirachtin): CDERI 8.8s (36%) + SCF 15.7s (64%), 其中
**7 轮跑 fp64** (单轮约 fp32 的 4 倍), fp64 尾巴约占总时间 62%。

想法: SCF 后期 `ddm` 很小, fp32 收缩的**绝对**误差按 |ddm| 等比缩小,
所以尾轮可以"一次干净的 fp64 全量重建 + 之后只对增量做 fp32"。

探针完全支持该想法:

| \|ddm\|/\|dm\| | fp32 完整密度 | fp64 基线 + fp32 增量 |
|---|---|---|
| 1e-3 | 2.98e-7 | 9.6e-10 |
| 1e-4 | 2.98e-7 | 9.4e-11 |

**但实现后收益不成立**, 原因链:

1. 先用解析因子化 `[F_new, F_old]·[F_new, -F_old]^T` (秩 2·nocc, 免 SVD,
   实测比 fp64 快 5.4x) —— **精度完全无改善**: 这是两个 O(|dm|) 量相减,
   又把相消引回来了, 误差 4.6e-7 恰等于 fp32 全密度误差, 且逐轮线性累积,
   50 轮不收敛。
2. 改用 SVD 分解增量 (误差确实正比 |ddm|, 稳定在 1.4e-10 不累积), 但增量是
   满秩, 只比 fp64 快 2.7-2.9x。
3. 加 `|ddm|` 门槛把精度拉回 1e-10 后, **速度回落到与关闭尾巴无异**:
   实测 (Azadirachtin, conv 1e-9, min of 4) 关闭尾巴 22.56s (1.63x),
   安全系数 300/1e3/3e3/1e4/3e4 分别 22.32/24.00/24.26/24.00/22.47s ——
   **没有一档比关闭更快**。
4. 门槛放松到 1e-2 能拿 2.45x, 但误差退到 1e-7~1e-8, 且**误差对门槛非单调**
   (3e-5 比 3e-4 更差), 说明主导因素是 fp32 的非确定性路径差异。
5. 离散度对照 (Azadirachtin, conv 1e-9, 6 次): 原始 auto 9.1e-12~5.8e-11;
   带尾巴 5.3e-11~**6.8e-9** —— 最差超过用户要求的 conv_tol。

**结论: 已回退。** auto 通道在不损失精度的前提下没有可复现的剩余空间。
真要再快, 只能回到 CDERI 构建 (Ozaki 补偿式 GEMM)。

### 其他剩余空间

- **小分子 GPU 喂不饱**: 208 AO 只有 1.6x (对比 537 AO 的 3.1x)。多分子
  并发 (CUDA streams) 可能有收益, 未测。
- **int3c2e 积分核**: CDERI 里除 GEMM 外的部分, 仍 fp64, 即原「阶段二」。
- **SCF 初猜**: 目前 8-13 轮, 但只影响那 30%, 上限有限。

## 三通道实测对照 (2026-09-05, def2-SVP + jkfit, 中位数 ×3)

| 分子 | fp64 | auto | fp32 |
|---|---|---|---|
| Vitamin C (208 AO) | 0.29s | 0.29s (1.0x) | 0.18s (1.6x) |
| Tamoxifen (537 AO) | 4.95s | 3.30s (1.5x) | 1.59s (3.1x) |
| Azadirachtin (934 AO) | 37.0s | 24.5s (1.5x) | 11.7s (3.2x) |

误差: auto 1e-10~1e-11 (等同 fp64); fp32 1.1e-5 / 1.4e-4 / 3.8e-4。
加速比随体系增大而增长, 小分子 GPU 喂不饱。按 Tamoxifen 尺寸估算单卡吞吐:
fp32 ~2260 分子/小时 (10 万 ≈ 1.8 天), auto ~1090 分子/小时 (≈ 3.8 天)。

## 待办 / 下一步 (恢复项目时从这里开工)

1. ~~**批量单点管线**~~ — GPU 通道已完成 (见上)。剩余可选项:
   - CPU 进程池兜底通道 (2线程×12进程 ≈ +380 mol/h), 首版刻意未做
   - 同分子构象序列的 DM 链式热启动 (~10% 提速)
   - 大规模实跑前建议先用 `--dry-run` 核对分片清单
2. ~~阶段二 `ejk_int3c2e_ip1.cu` FP32 化~~ — 已完成 (见「阶段二相关数据」),
   梯度 3.31x, 几何优化单步 2.04x。剩余: Hessian 仍只有 1.15x
   (只吃到了共享的网格 helper), 需要先量出 CPHF 求解里 XC 占多少。
3. ~~源码编译两问题~~ **(2026-09-06 更正, 之前的记录是错的)**:
   - **gfortran 装着的** —— `/usr/bin/gfortran` = GNU Fortran 15.2.0。之前
     「缺 gfortran」是误判 (大概是查了 `gfortran-14` 没找到就下结论),
     `build/fake_gfortran` 那个 shim 和构建缓存里的
     `CMAKE_Fortran_COMPILER=/bin/sh` 都是这次误判的产物, 都不需要。
   - 真正的障碍只有一个: glibc 头文件的 `__THROW`/noexcept 和 nvcc 冲突。
     解法是 CUDA flags, 不是换编译器。已验证的干净配置命令:

         cmake -S gpu4pyscf/lib -B <build> \
           -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
           -DCUDA_ARCHITECTURES="120-real" \
           -DCMAKE_CUDA_FLAGS="-U_GNU_SOURCE -U_ISOC23_SOURCE -U_ISOC2X_SOURCE \
             -U_ISOC2Y_SOURCE -DM_PI=3.14159265358979323846 -include stdint.h"

     这样配置全过, Fortran 自动找到 `/usr/bin/gfortran`, 无需任何 shim。
   - **`pbc` 目标编不过, 而且这是个二选一, 不是可以顺手修的 bug**:
     带 `-U_GNU_SOURCE` 时 CUDA 探测才能通过, 但关掉 `_GNU_SOURCE` 后 glibc
     不再暴露 `uselocale` / `__locale_t` / `fwide` / `pthread_mutex_timedlock`,
     而 `pbc` 用到的 libstdc++ 头文件 (`<cwchar>`, `bits/c++locale.h`) 要这些。
     去掉 `-U_GNU_SOURCE` 重配 → 配置阶段直接 13 个错误, 什么都编不了。
     实测过, 别再试。混合精度这条线不需要 `pbc`, 其余目标全部正常。
     真要修得换 gcc/CUDA 版本组合。
   - 现有的 `build/temp.gpu4pyscf` 是能用的 (增量编 `gvhf_rys` 约 6s), 但它的
     缓存里带着上面那条错误的 Fortran 设置。**不要随手重配它**;
     要干净重来就新建目录用上面的命令。

## 阶段二相关数据 (几何优化场景, 已测)

### 梯度 FP32 化 (2026-09-06 完成)

Tamoxifen, 1274 AO, r2SCAN/def2-TZVPP + def2-universal-jkfit, `auxbasis_response=True`:

| 通道 | 梯度耗时 | 加速 | 最大误差 (Eh/Bohr) |
|---|---|---|---|
| fp64 | 8.01s | — | — |
| fp32, 仅 XC 网格 | 4.04s | 1.98x | 9.1e-07 |
| fp32, XC + J 内核 | 2.42s | 3.31x | 1.0e-05 |
| fp32, + 内核累加器修复 | **1.18s** | **6.80x** | 1.1e-05 |

误差比 geomeTRIC 默认收敛阈值 3e-4 Eh/Bohr 低 26 倍。

三处改动:

1. `_j_energy_per_atom` (df/grad/rhf.py) 之前硬写 `sum_ejk_int3c2e_ip1`,
   只有 `_jk_energy_per_atom` 选了 f32 内核。**纯泛函 (r2SCAN/PBE) 走的正是
   J-only 路径**, 所以梯度完全没吃到 fp32 —— 非 XC 部分占 fp32 梯度的 81%。
   现在两条路径一致: 只把交给积分内核的密度 cast 成 float32,
   dm 预收缩和 j2c metric solve 仍留 float64 (metric solve 会把 float32 的
   1e-7 相对误差放大到 1e-3)。
2. `GradientsBase` 新增 `precision_mode` (grad/rhf.py)。之前 `scf.hf._kernel`
   退出时恢复全局模式, 所以 `mf.precision_mode='auto'` 的梯度仍跑 fp64,
   fp32 梯度只有手动 `with precision.fp32()` 才够得着。现在梯度默认继承
   mf 的模式; **梯度没有迭代可收敛, 所以 'auto' 在这里就是 fp32** ——
   1e-5 误差远低于它喂给的收敛阈值。`g.precision_mode` 可覆盖。
   TD 梯度自己重写了 `kernel`, 不受影响 (保守留在 fp64)。
3. **`ejk_int3c2e_ip1_f32.cu` 的 9 个梯度累加器还是 double** —— 但它们喂的
   跨线程归约缓冲区 `reduce` 本身是 `float*`, 额外精度下一条语句就被截断,
   代价却是 `GOUT_WIDTH=54` 最内层循环里每轮 9 次 1/64 速率的 fp64 加法。
   改成 float 后内核 2.42s → 1.18s, 误差 1.00e-5 → 1.14e-5 (基本不变)。

   **发现它的线索**: 内核 fp64→fp32 只拿到 2.03x (3.181s → 1.569s), 而 5090
   的 fp32:fp64 硬件比是 64:1。差这么远就说明内核不是被 fp64 算力卡住的。
   **以后看到 fp32 化收益远低于硬件比, 就往下追。**

   fp64 版内核是 `double *reduce` 配 `double v_ix`, 自洽 —— 这是 f32 移植
   时只改了一半。同时删掉 `datax = (float*)(ROOT_RW_DATA)` 这颗哑弹:
   `ROOT_RW_DATA` 是 `double[]`, 这是位重解释不是数值转换, 读出来是垃圾;
   它被 `(void)datax;` 挂着没人用。
   另测: Rys 根的 Clenshaw 递推改 float **完全没收益** (1.177→1.176s,
   每对壳层只算一次被内层循环摊薄了), 不换速度就不降精度, 已退回 double。

单步端到端 (SCF + 梯度, 同分子):

| | SCF | 梯度 | 单步 |
|---|---|---|---|
| fp64 | 25.67s | 8.04s | 33.72s |
| auto | 14.05s | **1.19s** | **15.24s (2.21x)** |

能量差 8.2e-12 Eh, 梯度差 1.15e-5 Eh/Bohr。此前 auto 单步只有 1.53x。

**成本重心已经回到 SCF**: 梯度现在只占 auto 单步的 7.8%, SCF 占 92%。
再降梯度精度最多再快 8% —— 这条线可以收工了。下一个靶子在 SCF 侧,
而那里卡住的不是精度而是策略: geomeTRIC 的能量判据是 1e-6 Eh 而纯 fp32
SCF 噪声底 4e-5 (高 40 倍, 所以 fp64 尾巴必须留), 但优化早期构型离极小点
还远, SCF 却一直按 conv_tol=1e-9 收敛 —— 这个浪费和精度无关。

### 端到端几何优化 (受体系尺寸门控)

| 分子 / 基组 | AO | fp64 | auto | 加速 |
|---|---|---|---|---|
| Vitamin C, def2-SVP (收敛) | 208 | 67.2s | 78.4s | **0.86x** |
| Tamoxifen, def2-SVP (收敛) | 537 | 347.1s / 41 步 | 224.1s / 43 步 | **1.55x** |
| Tamoxifen, def2-TZVPP (各 7 步) | 1274 | 209.5s | 111.3s | **1.88x** |

**收敛构型对照 (Tamoxifen def2-SVP)** —— 两边最终构型都用 conv_tol=1e-11 的
严格 fp64 重算能量:

    fp64  -1136.682481947
    auto  -1136.682481949     差 2.2e-9 Eh = 1.4e-6 kcal/mol

auto 多走了 2 步 (43 vs 41) 仍快 1.55x。构型坐标最大差 7.05e-3 Bohr ——
比位移阈值 1.8e-3 大, 但极小点附近 PES 平坦, 能量一致到 2e-9 Eh 才是有意义的
判据。**结论: fp32 梯度不改变优化落点的能量。**

Tamoxifen def2-TZVPP 那组两条通道步数完全相同 (7 步), 每步能量差 1e-6~3e-6 Eh,
坐标最大偏离 9.9e-4 Bohr; 但它是 maxsteps 截断的, 不是收敛对比。

**Vitamin C 反而慢**: 208 AO 喂不饱 GPU, 而 auto 的 fp32→fp64 切换多花的
迭代赚不回来。和单点表里 Vitamin C 1.0x 的规律一致。加速比随尺寸单调上升:
208 AO 0.86x → 537 AO 1.55x → 1274 AO 1.88x。
**不要拿单步/梯度加速比当端到端优化的数字报** —— 要看体系尺寸。


## 基准文件

- `mixedprec/fp32_jk.py` — get_jk FP32 化 + install() monkey-patch (含 cast 布局修复)
  (阶段一原型; 现已并入 `gpu4pyscf/df/df_jk.py`, 新代码不要再用这个 patch)
- `mixedprec/batch_rhf.py` — 批量单点驱动 (XYZ → DF-RHF → HDF5 分片 + 续跑)
- `mixedprec/test_batch_rhf.py` — 36 个测试 (解析/分片/存储无需 GPU; GPU 端到端另计)
- `mixedprec/step2_proto.py` — SCF 基准 (FP64/混合 × conv 1e-10/1e-7 + CPU 参照)
- `mixedprec/step3_opt.py` — 几何优化基准 (10 步 FP64 vs 混合, 每步 SCF/梯度计时)
- 结果: `step2_result.txt`, `step3_result.txt`, `instrument_result.txt`
- 小基组对比: `/tmp/basis_scale.log` (6 步优化, def2-SVP / 6-31G*)
