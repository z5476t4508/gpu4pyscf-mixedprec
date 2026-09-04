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

## 待办 / 下一步 (恢复项目时从这里开工)

1. ~~**批量单点管线**~~ — GPU 通道已完成 (见上)。剩余可选项:
   - CPU 进程池兜底通道 (2线程×12进程 ≈ +380 mol/h), 首版刻意未做
   - 同分子构象序列的 DM 链式热启动 (~10% 提速)
   - 大规模实跑前建议先用 `--dry-run` 核对分片清单
2. 阶段二 (仅当回到几何优化场景才值得): `ejk_int3c2e_ip1.cu` FP32 化,
   梯度 75-79%, 周级内核工程, 整体 1.14x→~1.6x。批量单点场景不跑梯度,
   该项对当前需求无用。
3. 源码编译两问题 (阶段二前置): glibc rsqrt noexcept 冲突 (gcc-14 绕过) +
   缺 gfortran

## 阶段二相关数据 (几何优化场景, 已测)

## 基准文件

- `mixedprec/fp32_jk.py` — get_jk FP32 化 + install() monkey-patch (含 cast 布局修复)
  (阶段一原型; 现已并入 `gpu4pyscf/df/df_jk.py`, 新代码不要再用这个 patch)
- `mixedprec/batch_rhf.py` — 批量单点驱动 (XYZ → DF-RHF → HDF5 分片 + 续跑)
- `mixedprec/test_batch_rhf.py` — 36 个测试 (解析/分片/存储无需 GPU; GPU 端到端另计)
- `mixedprec/step2_proto.py` — SCF 基准 (FP64/混合 × conv 1e-10/1e-7 + CPU 参照)
- `mixedprec/step3_opt.py` — 几何优化基准 (10 步 FP64 vs 混合, 每步 SCF/梯度计时)
- 结果: `step2_result.txt`, `step3_result.txt`, `instrument_result.txt`
- 小基组对比: `/tmp/basis_scale.log` (6 步优化, def2-SVP / 6-31G*)
