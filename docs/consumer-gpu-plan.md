# 消费级 GPU (RTX 4090) 分子 DFT 混合精度优化 —— 项目计划与任务矩阵

**2026-10-04 最新执行策略（覆盖后文旧的停机条件）**：用户要求遇到问题记录后继续开发，稍后回看。科学精度门、测试失败、单任务重试耗尽不再终止整个开发队列；失败数据和 scientific_gate_passed=false 原样保留，进入 deferred-issues.json 待审/待修清单。其它方法及后续热点分析、代码优化、显存/批量、梯度开发可暂用未验收成果；缺失关键依赖只延期受影响任务，队列继续。梯度可开展实验性开发，但能量问题未解决时不得宣称梯度已验收。单 GPU、显存预算、有界重试和人工审查保持有效，不改数值阈值、不把失败改成通过。

**文档版本**: 2026-10-03 rev3（G16 实际可用性、测试甄别门、复用已有参考输出、小体系优先） ·
**基准 commit**: `0a4cf33f0db146ef3e04ac97f2ce1f2deeaeb862` (upstream)
**适用仓库**: `gpu4pyscf-mixedprec` · **目标硬件**: NVIDIA RTX 4090 24 GB (sm_89, CUDA 12.8)
**2026-10-03 执行策略更新**：用户授权先持续推进、稍后手动审查。前置任务完成必要检查并处于 review_requested 即可暂用，依赖不再等 accepted；待审查不等于审查通过。科学数值门、失败检测和有界执行保持有效，不自动合并或关闭任务。后文关于等待前置 accepted 的执行屏障以本段为准，状态 accepted 仍仅由明确人工审查产生。

**本文档是 issue #2 的交付物**：只定义计划、schema、规程与工具，不改动物理实现、不跑昂贵基准。

---

## 1. 目标与总验收判据

### 1.1 方法范围（用户确认，覆盖旧的 r2SCAN-only 范围）

| 方法 | 类型 | 必须包含的成分 |
|---|---|---|
| B3LYP | 杂化 GGA | 精确交换 K |
| M06-2X | 全杂化 meta-GGA | 精确交换 K；需检查网格稳定性 |
| omegaB97M-V | RSH 杂化 meta-GGA | **VV10/NLC 非局域相关必须开启**（nlcgrids） |
| r2SCAN-3c | **整体方法，不是单泛函** | r2SCAN + **def2-mTZVPP 专用基组 + D4 (定制参数) + gCP**，缺一即错 |

**能量先行**：全部能量门通过之前不启动梯度（issue #12 前置门）。Hessian、周期体系、MP2/CCSD、多 GPU 不在本期范围。

### 1.2 总验收判据（不可协商）

1. **精度门**：最终 |E_optimized − E_FP64| ≤ **1×10⁻⁶ Hartree**，且对比双方使用**完全相同**的
   basis / auxbasis / DF 开关 / XC 泛函变体 / 网格 (grids+nlcgrids) / screening / 色散与 gCP
   修正 / conv_tol / direct_scf_tol / 初猜。误差分**三级报告，永不混算**（§3.4）：
   - **(1) 精度误差**（唯一的 ≤1e-6 Ha 验收门）：同实现同设置、不同 precision lane，
     对 **GPU4PySCF FP64** 的能量差；
   - **(2) 实现一致性**（诊断项，须报告并归因）：对 **CPU PySCF FP64** 的能量差；
     匹配设置下历史水平 ~1e-10 量级，若显著偏离必须找到具名实现差异
     （screening 默认、libxc 版本等），不得归因于混合精度，也不得静默放行；
   - **(3) 跨程序比较**（G16/ORCA，仅报告）：永不设 1e-6 门；未匹配的方法/设置
     必须标注 unmatched，其差异不得归因于混合精度（§3.6）。
   DF 近似误差另列（§3.3）。
2. **精度验收绝不能仅由 SCF 收敛推断**。每个 pass 必须附能量差数值、收敛旗标、
   精度切换/FP64 收尾证据（切换轮次、触发类型）、以及混合精度入口实际命中计数。
   非收敛、错误电子态（如收敛到不同 SCF 解）、遗漏修正项（D4/gCP/NLC 缺失）、OOM
   一律 fail 或 block，不得重新标注为 pass。**跳过 (skipped) 与 OOM 格必须显式记录
   原因，不得留空、不得杜撰通过结果。**
3. **覆盖面与试点阶梯**：从小/中非平凡分子起步，多档可负担基组，外加一个可负担的
   UKS 开壳层精度哨兵；按 **AO 数 × 基组 × 网格 × 实测峰值显存** 分级放量
   （§3.5 阶梯），不以原子数单独判断。历史 Vitamin C / Tamoxifen 用例是**延期候选**：
   仅当上一级实测峰值显存 + headroom 预算允许时才进入，不是必需的立即运行项。
   调试用小分子（methanol 级）单独不构成验收覆盖。r14 类多解体系只测计时，
   不进精度列（历史结论）。
4. **计时纪律**：预热与**至少 3 次独立初始化的计时样本**分开；每次计时前后
   CUDA `deviceSynchronize()`；报 raw 样本数组、中位数、端到端墙钟（含 CDERI 构建、
   精度切换、FP64 尾巴、回退等**全部成本**——快速 FP64 回退和 FP64 收尾本身不证明
   总加速）；SCF 迭代合计与内核分解计时**两者都报**；峰值显存；SCF 循环数；
   混合精度入口命中数；来源路径 + commit；依赖版本；硬件。**不发明速度目标，
   不跨设置比较计时。**
5. **显存纪律**：每次运行前记录 GPU free/total、运行中记录峰值显存分配与 host RSS
   峰值；执行 headroom 预算（可配置，§4）；预算不足时**先停/升级，不做危险分配**，
   该格记 `skipped`（附实测 free 与预估需求），OOM 如实记 `oom`。
6. **安全回退**：不安全/不支持的路径保留严格 FP64 回退。FP16/BF16/TF32/张量核
   需实测数值论证后才能启用，不得全局开启近似数学，不得绕过错误门换取速度。
7. **负结果保留为证据**：历史否决的路线（§2.3）不重做，除非 RTX 4090 上有新的
   实测证据表明需要重测；重测须记录动机。

### 1.3 与历史 RTX 5090 数据的关系（必须区分）

`mixedprec/STATUS.md` 与 `mixedprec/benchmark/REPORT.md` 中的全部数值来自
**RTX 5090 32 GB（另一台机器）**，本项目中它们只是**假设与先验**，不是证据。
迁移规则：

- **可直接继承（与 GPU 型号无关）**：数值规律（fp32 噪声底、Cholesky 相消、
  二阶导误差不衰减规律）、代码路径知识（精度分支位置、wheel 遮蔽坑、auxbasis
  钉死要求、6d/5d 约定）、已否决路线表（§2.3）、基准方法论（预热/冷启动/
  中位数/哨兵间隔法）。
- **必须在 RTX 4090 重测后才可引用**：一切加速比、耗时、显存峰值、OOM 边界、
  发散域位置。RTX 4090 显存 24 GB（比 5090 少 8 GB），**所有显存墙提前**；
  fp32:fp64 硬件比同为 64:1 但绝对算力不同，热点排序可能不同（issue #9 重归属）。
- 任何报告中引用历史数字必须显式标注 "RTX 5090 历史，未在 RTX 4090 复测"。

---

## 2. 现状盘点（截至本计划）

### 2.1 已有的混合精度实现（位于 upstream 0a4cf33，全部复用，不重复实现）

| 组件 | 位置 | 说明 |
|---|---|---|
| 全局精度开关 + auto SCF 策略 | `gpu4pyscf/lib/precision.py`, `gpu4pyscf/scf/hf.py:235-258` | `precision_mode='auto'`：fp32 起步，\|ΔE\|<1e-4 切 fp64 收尾；含混沌守卫（连续 3 轮 ΔE>1e-2 强制切 fp64） |
| fp32 XC 网格分支 | `gpu4pyscf/dft/numint.py:207,338,586,1021` | `_eval_rho2` 密度侧与 Fock 侧大 GEMM 的 fp32 分支（历史实测 ~25-28x 单点） |
| fp32 梯度内核 | `gpu4pyscf/lib/gvhf-rys/ejk_int3c2e_ip1_f32.cu` + `df/grad/rhf.py` | 纯泛函 J-only 路径也已接 fp32 |
| fp32 CPHF | `df/hessian/rhf.py`（Hessian 阶段，本期不用） | — |
| 3c 驱动 | `gpu4pyscf/drivers/dft_3c_driver.py` | r2SCAN-3c = r2scan + def2-mTZVPP + D4(atm, 定制参数 s8=0.42, α6=5.65, s9=2.0, ga=2.0, gc=1.0) + gCP；注意 to_gpu() 之后才能挂 dispersion（代码内注释有说明） |
| NLC/VV10 | `gpu4pyscf/dft/numint.py:nr_nlc_vxc,_vv10nlc`, `dft/rks.py:50` | omegaB97M-V 的 NLC 路径已存在，但**从未在混合精度车道下验证**（issue #7） |
| 基准 runner | `mixedprec/benchmark/run_benchmark.py` | row-key 合并、E/G ×3 中位、CPU 哨兵、误差列 derive；**为 5090 写**，缺 4090 schema 要求的若干字段（§4） |
| 哨兵方法论 | `mixedprec/step7a_sentinel_adequacy.py`, `vibanalysis.py` | 判据用间隔不用绝对界（vitamin C 间隔 35x）；本期能量门沿用此思想 |

### 2.2 环境

- 参考环境 `/home/ubuntu/pyscf-env`（vda，官方 wheel）**只读保留**，作为"发布版行为"参照；不在其中安装任何东西。
- 工作环境新建于 `/home/qczcode`（vda），源码树 `/data/work/gpu4pyscf-mixedprec`（vdb），编译目录在 `/data`。**必须从源码树加载**（`PYTHONPATH` 指向仓库根）——wheel 没有 `gpu4pyscf.lib.precision`，会**静默降级 fp64**（历史实测中招）。鉴伪特征：auto 与 fp64 的梯度差若只有 ~1e-8 而非 ~1e-5，即运行在 wheel 上。
- RTX 4090 = sm_89；用已装 CUDA 12.8 + 兼容宿主编译器；**不得照搬 5090 的 `-DCUDA_ARCHITECTURES=120-real` 等构建旗标**。

### 2.3 历史否决路线（RTX 5090 实测；不重做，除非 4090 新证据）

| 路线 | 结果 | 教训/可迁移规律 |
|---|---|---|
| fp32 CDERI 变换 GEMM（Cholesky 路线） | 误差 3.8-81 Ha，不可用 | Cholesky 因子相消（中位 4387x）；**存储 fp32 可、累加 fp32 不可** |
| auto 车道 fp32 增量尾巴 | 无可复现收益，已回退 | — |
| XC `_get_vxc_deriv2` fp32 | 5.15x 但频率误差 55x 超界，撤销 | **fp32 误差直接落在二阶导上 = 拒**（无下游衰减，δω≈δH/(2μω) 再放大） |
| JK `ejk_int3c2e_ip2` fp32 内核 | 2.7x 但误差 150-500x 且不可预测，撤销 | 同上；且两分子把两变体排反，n=2 无法预测第三分子 |
| 纯泛函 J-only 分支 fp32（SCF 侧） | r2SCAN 1.03x 换 235x 误差，否 | 不为买不到时间的精度降级付费 |
| 纯 fp32 conv_tol < 3e-5 | 迭代数剧烈抖动 | fp32 噪声底：DF J/K ~4e-5，网格 ~1e-6；要更紧用 auto |
| 杂化泛函 ~1900 AO auto SCF（5090） | fp32 相发散（已加混沌守卫，但发散域内比 fp64 **慢** 2.4-3.1x） | **守卫保正确性不保速度**；4090 上发散域位置未知，须实测；此即"回退成本必须计入总加速"的实证 |
| Direwolf COSX/两阶段网格 | COSX 对纯泛函收益恒为 0；两阶段网格 1.16x 只利单点 | 新范围含杂化泛函，COSX 理论上重新相关，但仍须先归属（#9）再决定 |

---

## 3. 数据与证据规范

### 3.1 目录布局（与 AGENTS.md 一致）

```
/data/work/gpu4pyscf-mixedprec          # 源码（本仓库）
/data/results/gpu4pyscf-agent/issue-N/  # 每个 issue 的原始结果（不可变证据）
    *.json, *.log, *.h5                 # 写入后不再修改；JSON 内记录 sha256
/data/work/gpu4pyscf-agent/state/       # 工作流状态（tasks.json, gpu.lock）— 仓库外
/data/work/gpu4pyscf-agent/outbox/      # review_request.json / patch / 发布清单 — 仓库外
/data/tmp/                              # scratch
```

**不可变证据引用**：结果 JSON 的每个 `evidence` 条目 = 路径 + sha256 + 采集命令。
后续文档引用证据时引用"路径@sha256 前 12 位"，不复述裸数字。长日志留在
`/data/results`，仓库与发布清单只进代码、小 fixture、文档与摘要 JSON。

### 3.2 证据分级

| 级别 | 含义 | 用途 |
|---|---|---|
| A | RTX 4090 实测，schema 合规，sha256 锚定 | 验收、速度声明 |
| B | RTX 5090 历史实测 | 先验、动机；引用须标注 |
| C | 代码阅读/推理（未实测） | 假设；不得单独支撑验收 |

### 3.3 DF 误差与精度误差分离

对每个 (体系, 方法) 组合，另做一组**小规模对照**：同设置 fp64 下
`DF(auxbasis=X)` vs `conventional（无 DF）`，报告为独立行
(`lane: "df-vs-conv"`)，其差值记为 `dE_df_approximation`。
**该差值永不与混合精度误差合并或相抵**；混合精度误差只在"同 DF 设置、
不同 precision lane"之间计算。对外（如对 g16 类参照）比较时必须同理论等级
（历史教训：拿 DF+def2-svp 对常规积分 g16 表面差 0.17-0.78 Ha，全是层次差）。

### 3.4 三级误差分离（精度门 / 实现一致性 / 跨程序）

所有能量差必须归入且仅归入下列一级，报告中分行列出，禁止合并：

| 级 | 名称 | 参照 | 设置要求 | 判据 |
|---|---|---|---|---|
| (1) | **precision error** | GPU4PySCF FP64（同源码、同设置） | 逐字段相同（§1.2 第 1 条清单） | **≤ 1e-6 Ha，硬验收门** |
| (2) | **implementation agreement** | CPU PySCF FP64 | basis/auxbasis/DF/XC 变体/grids/NLC/conv_tol/初猜逐项钉死并记录 | 诊断项：匹配设置下历史 ~1e-10 量级；显著偏离须找到**具名**实现差异并记录，不得归因混合精度，也不得静默放行 |
| (3) | **cross-program comparison** | G16 预计算输出 / ORCA（后者上传后可选） | 尽力匹配（XC 定义尤其 B3LYP 变体、DF/常规、网格、色散、几何单位、版本），无法匹配处显式标 `unmatched` | **仅报告，不设门**；unmatched 方法的差异不得归因于混合精度，不得承诺 1e-6 |

记录契约（进 schema `method` 块与 `accuracy` 块）：XC 定义字符串（如
B3LYP 用哪个变体）、basis/auxbasis、DF 开关、grids 与 nlcgrids、SCF 容差、
charge/spin、几何单位（Å/bohr）、色散/gCP 修正、程序版本。任何一级缺少
这些字段即该行不可作为证据（校验器拒绝）。

### 3.5 试点阶梯（按 AO 数 × 基组 × 网格 × 实测显存放量）

**规划强制的分级放量**。每一级的准入条件是上一级的**实测**数据，不是估算：

| 级 | 规模 | 目的 | 进入条件（看上一级实测） |
|---|---|---|---|
| S0 | 调试分子（methanol 级，~ tens AO） | 冒烟、路径正确性 | 无（首发） |
| S1 | 经甄别的小非平凡体系（起步约 40–150 AO；数字只是候选分区） | 首个精度验收层 | 用例甄别通过 + S0 显存实测与 headroom 预算允许 |
| S2 | 经甄别的中等体系（起步约 150–400 AO，逐例评估） | 精度 + 计时扩展 | S1 通过且实测资源允许；不要求固定分子或基组组合 |
| S3 | 大体系 / 高 AO 历史用例 | 后续容量研究 | 当前暂缓，用户重新调整范围后再评估 |
| S4 | 更大体系 / 更大基组 / host-RAM offload 场景 | 后续研究 | 当前暂缓，不属于自动放量队列 |

规则：
- **不以原子数单独判断**——同原子数不同基组/网格显存差数倍；S 级标签只是
  规划分区，实际准入门是"实测峰值显存 + 配置化 headroom 预算"。
- **不排期预计超 24 GB 的验收用例**；边缘用例先跑显存测绘探针（#11 的早期
  安全容量测量部分），拿到实测峰值再决定进/不进，不进则记 `skipped`（附原因）。
- 每级的 UKS/开壳层哨兵：至少一个**可负担**的开壳层体系进精度列（r14 类
  多解体系仅计时，历史结论）。
- 历史 Vitamin C / Tamoxifen 矩阵是**候选**，不是必需立即运行项（用户规则）。

### 3.6 G16 / ORCA 独立参照：先甄别，再复用

**现状（2026-10-03）**：G16 A.03 已在服务器安装，CPU 冒烟及隔离执行账号访问验证通过；ORCA 尚未确认上传或可用。安装验证不代表所有测试或目标方法已验证。已有私有索引 `/data/datasets/gaussian-test-catalog/`：1147 个正式测试输入、1156 个参考日志，另有辅助 `.com` 文件。输入 route 初筛匹配 B3LYP 106 个、M06-2X 3 个；这是文件统计，不是可用 benchmark 数量。其余目标方法尚无已确认的配套 G16 参照。

**用户要求：可用的测试预计不多，必须甄别；优先复用已有 G16 输出，不常规重新计算 G16。** 主基线仍为同设置 GPU4PySCF/PySCF FP64，不等待 ORCA。

#### 3.6.1 候选用途分级（每条必须有理由，默认未准入）

| 分类 | 可以做什么 | 必须记录 |
|---|---|---|
| `untriaged` | 只列入候选池，不运行或计入覆盖 | 尚缺的甄别项目 |
| `reference_matched` | 复用结构与已有 G16 能量，作跨程序独立参照 | 匹配的具体子任务、完整方法设置与证据 |
| `reference_diagnostic` | 方法基本对应但存在已知实现/设置差异，仅做具名差异诊断 | 不匹配字段、不能归因于精度的原因 |
| `geometry_only` | 只复用结构，目标方法重新计算 FP64 与优化车道；旧 G16 能量不作为该方法参考 | 结构来源及旧结果不适用的原因 |
| `deferred` | 方法/格式/资源/解析复杂度暂不适合，保留待办 | 阻塞项、重新考虑的条件 |
| `rejected` | 当前不适合作 benchmark | 明确排除理由，不能静默丢弃 |

`reference_matched` 也不自动通过 PySCF 一致性或混合精度验收。跨程序容差应基于已确认的理论/数值差异单独制定，不能用拟合容差掩盖方法错误。

#### 3.6.2 甄别清单（#4 的强制前置交付）

1. **定位计算步骤**：输入与输出文件一一核对；多 `Link1` 必须准确拆出对应子任务。确认正常结束、有限能量与目标 SCF 解。优化、MD、扫描、响应性质、特殊 IOp、外场、溶剂、ECP、ghost centers 或检查点依赖等默认人工评估；不把文件最后一个能量或一次正常结束当作整份测试可靠性。
2. **结构与电子态**：从原始输入恢复精确几何、单位、电荷和多重度；若选优化后的结构，明确是哪一步。日志 orientation 坐标有舍入，只用于候选提取，未核实输入精度不能作严格对照。检查 RKS/UKS、占据、稳定解或多解风险；复杂电子态先延期，不为了凑数纳入。
3. **方法匹配**：逐项记录泛函定义（特别 B3LYP 变体）、基组与 AO 数、辅助基组/DF 或常规积分、网格/NLC、SCF 与积分阈值、色散/gCP、程序版本及运行库配置。不支持或不能匹配的项标注 `geometry_only` / `reference_diagnostic` / `deferred`。完整 r2SCAN-3c 不能用一般 r2SCAN 或其他泛函输出冒充。
4. **资源与代表性**：先用小体系；按 AO、网格、DF 张量和实测 CPU/GPU 内存评估准入，预留 headroom。去掉重复结构/同类路径；优先少量可解释且覆盖不同计算特点的体系。UKS 哨兵在可负担条件下独立安排，不随大体系延期。大型或昂贵案例当前暂缓。
5. **独立准入**：提交逐例甄别表及理由，审查通过后才进入正式 cohort；主程序失败、未知设置、匹配不到结构/能量的案例不得悄悄记为通过。候选目录、正常结束标记和历史 G16 结果本身均不构成 accepted。

当前严格自动初筛只留下一个 B3LYP/UKS 小体系候选（11 原子、94 AO），仍未人工准入。一个 MD 测试虽然只有一个 SCF 结果，已因 MD route 排除；这表明必须看计算语义，不能只数 SCF/termination。M06-2X 的三个测试均涉及多子任务或较大基组，需单独拆解；不能声称三例已可用。不承诺测试数量，也不把仅一个小例子的成功外推到四种方法。

#### 3.6.3 交付物与复用已有输出

- 私有 `curation.json` 每行至少包含：case/source hashes、input/log/selected job、分类与原因、准确几何来源和精度核验、电子态、完整 method signature、原始 G16 能量与定位、正常结束与解的检查、AO/网格/内存预算、审核状态、可追溯证据。
- 适配器优先提供 `read_precomputed`：解析已有输出，核对 selected job / method / geometry，不触发 G16 计算。运行外部程序是单独显式的可选动作；只有参考输出损坏、缺失或必须确认关键疑点时才说明原因并安排必要重算，不全量重跑随附测试。
- 每个目标方法先建立少量经过甄别的 cohort；有匹配 G16 结果就复用，无匹配结果就只复用几何或用用户自建小体系，并计算同设置 FP64。ORCA 上传后适用同一甄别流程。
- 混合精度门仍是同设置 GPU4PySCF FP64 的 ≤1e-6 Ha；现成 G16 能量不能替代这一基线。独立程序参考解析不纳入 GPU4PySCF 的计时，历史 G16 日志不能作本机加速比证据。
- 专有输入、日志、提取坐标和程序均保留服务器私有目录，不进入 GitHub；只提交方案、统计、非专有审核摘要与用户自建可公开数据。

---

## 4. 基准结果 Schema（issue #4 实现的合约）

文件：每个基准 run 一个 JSON，`schema_version: "rtx4090-energy-v1"`。
校验器随 #4 交付（schema 不合规的 JSON 不许进验收）。

```jsonc
{
  "schema_version": "rtx4090-energy-v1",
  "meta": {                              // 一次 run 一份
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_total_mem_mib": 24564,
    "driver_version": "595.58.03",
    "cuda_version": "12.8",
    "cpu_model": "...", "cpu_threads": 24, "ram_gb": ...,
    "python": "3.x", "cupy": "...", "pyscf": "...",
    "gpu4pyscf_source_commit": "<full sha>",
    "gpu4pyscf_source_path": "/data/work/gpu4pyscf-mixedprec",
    "libxc": "...",                      // 可取得时
    "mem_headroom_mib": 2048,            // 配置化的安全 headroom 预算 (§1.2 第 5 条)
    "timestamp_utc": "...", "run_id": "<uuid>"
  },
  "rows": [                              // 每个格子一行
    {
      "row_key": {"geom": "...", "basis": "...", "xc": "...", "lane": "..."},
      "pilot_stage": "S1",               // §3.5 阶梯级; 延期候选未跑则不排期不造行
      "system": {"geom_file": "...", "geom_sha256": "...", "natm": 0,
                 "nao": 0, "charge": 0, "spin": 0, "nelec": 0, "cart": false,
                 "geom_units": "angstrom"},
      "method": {"xc": "...", "basis": "...", "auxbasis": "...", "with_df": true,
                 "grids": {"level": 3, "atom_grid": null},
                 "nlcgrids": {"level": null, "atom_grid": [50, 194]},
                 "dispersion": {"model": "D4", "params": {...}},   // r2SCAN-3c
                 "gcp": "r2scan3c",                                 // r2SCAN-3c
                 "conv_tol": 1e-10, "conv_tol_grad": null,
                 "direct_scf_tol": 1e-14, "init_guess": "minao",
                 "max_cycle": 50, "open_shell": false},
      "lane": "fp64 | auto | fp32-screen | cpu-fp64 | df-vs-conv | xref-g16 | xref-orca",
      "status": "ok | skipped | oom | error",   // skipped/oom 行也必须存在并带原因
      "skip_reason": null,              // status != ok 时必填: "headroom-budget" |
                                        //   "oom-observed" | "unsupported" | ...
      "result": {
        "e_tot": 0.0, "converged": true, "scf_cycles": 0,
        "e_terms": {"e1": 0, "coul": 0, "exc": 0, "nlc": 0,
                    "dispersion": 0, "gcp": 0},      // 修正项分列，缺失即 fail
        "timings": {
          "warmup_s": 0.0,             // 独立预热（CUDA 懒加载/JIT），不进样本
          "samples_s": [0, 0, 0],      // >=3 次独立初始化的原始样本
          "median_s": 0.0, "min_s": 0.0, "max_s": 0.0,
          "end_to_end_s": 0.0,         // 含 mol.build/CDERI/SCF/修正项的全程
          "t_cderi_s": 0.0, "t_scf_s": 0.0, "t_xc_grid_s": 0.0,
          "t_jk_s": 0.0, "t_nlc_s": 0.0, "t_disp_s": 0.0,
          "sync": "device-sync before/after each sample"   // 同步证据
        },                             // SCF 迭代合计 vs 内核分解，两者都报
        "mem": {                        // 显存纪律: 每行必填 (skipped 行至少 free_before)
          "gpu_free_before_mib": 0.0,   // 跑前实测 (torch/cupy mem_get_info)
          "gpu_total_mib": 0.0,
          "gpu_peak_alloc_mib": 0.0,    // 运行中峰值分配 (池统计或已用差)
          "host_rss_peak_mib": 0.0,
          "headroom_budget_mib": 0.0,   // 该行适用的预算 (= meta.mem_headroom_mib)
          "within_budget": true         // peak + headroom <= total?
        },
        "fp32_entry_hits": {"df_jk": 0, "eval_rho": 0, "xc_grid": 0,
                            "grad_ip1": 0},   // 实际命中计数，证明混合精度真跑了
        "precision_switch": {"switched": true, "cycle": 0,
                              "trigger": "dE | chaos-guard | none"},
        "fallback_events": [],         // 例如 fp64 回退记录
        "errors": null                 // {"type": "OOM|divergence|wrong-state|...",
      },                               //  "message": "..."}；OOM=硬件边界记录,非 skip
      "accuracy": {                    // derive 阶段填充（同 geom/basis/xc 的行间）
        "dE_vs_gpu_fp64": null,        // (1) 精度误差 — 唯一硬门
        "dE_vs_cpu_fp64": null,        // (2) 实现一致性 — 诊断, 偏离须具名归因
        "impl_agreement_note": null,   // (2) 偏离时的具名差异 (screening/libxc/...)
        "verdict": "pass | fail | no_reference | skipped"  // pass 需 |dE(1)|<=1e-6 且 converged
      },
      "cross_program": [               // (3) 级, 独立于 accuracy, 仅报告永不设门
        {"program": "g16", "method_key": "...", "matched": false,
         "unmatched_fields": ["xc-variant", "df"], "e_tot": 0.0, "dE": 0.0}
      ],
      "evidence": {"raw_log": "path@sha256", "result_json": "path@sha256",
                   "command": "<完整复现命令>"}
    }
  ]
}
```

**强制规则**（校验器逐条检查）：
1. `timings.samples_s.length >= 3` 且 `warmup_s` 独立存在；计时样本必须来自独立
   初始化（新 mol + 新 mf），预热样本不得混入。
2. `status != "ok"` 的行必须带 `skip_reason`/`errors` 且 `accuracy.verdict` 只能是
   `fail`/`no_reference`/`skipped`；`converged == false` 的行 verdict 只能是 `fail`
   （或无参考）。**跳过与 OOM 如实呈现，不得静默缺失、不得杜撰通过**。
3. `fp32_entry_hits` 全零的 auto 行 = 混合精度未生效（wheel 遮蔽或模式未设），
   校验器直接判 `fail` 并标注 `lane-ineffective`。
4. r2SCAN-3c 行必须三个修正项齐全（`e_terms.dispersion`、`e_terms.gcp` 非空，
   basis == def2-mTZVPP）；omegaB97M-V 行必须 `e_terms.nlc` 非空。
5. `dE_df_approximation` 只出现在 `df-vs-conv` 行；`cross_program` 条目只承载
   (3) 级比较且必须带 `matched` 标注——两者都不得写进 (1) 精度误差字段。
6. 每行 `evidence` 必须锚定 `/data/results/...` 下存在且 sha256 匹配的文件。
7. `ok` 行的 `mem` 块必填（free/total/peak/host RSS/headroom/within_budget）；
   `within_budget == false` 的行必须有解释或升级记录，不得静默通过。
8. (2) 级 `dE_vs_cpu_fp64` 显著超出匹配设置历史水平（如 >1e-8 Ha）的行必须
   带 `impl_agreement_note` 具名差异，否则校验器判 `impl-unexplained`。

---

## 5. 任务矩阵（issue #3–#12）

通用规则：每个 issue 一个分支 `zcode/issue-N-<slug>`，一个小提交，先测后改；
状态机由 `tools/consumer_gpu_agent.py` 管理（§6）；状态与证据在仓库外。
所有 issue 的验收都包含"证据 JSON schema 合规 + sha256 锚定 + 复现命令可运行"。

### #3 隔离源码环境（deps: #2）

- **问题**：必须从本 fork 源码编译 sm_89/CUDA12.8 独立环境；wheel 无 precision
  模块会静默降级；不得污染 `/home/ubuntu/pyscf-env`。
- **交付物**：`/home/qczcode/<env>` venv（版本 pin 清单）+ `/data` 上编译目录；
  环境清单 JSON（python/cupy/cutensor/pyscf/libxc/cmake/nvcc 版本、构建旗标、
  源码 commit）；冒烟脚本（不进仓库或作小工具进仓库）。
- **依赖门**：#2 accepted。
- **可运行验收**：
  1. `PYTHONPATH=<repo> python -c "from gpu4pyscf.lib import precision"` 成功；
  2. `import gpu4pyscf; print(gpu4pyscf.__file__)` 指向源码树（非 site-packages）；
  3. 编译产物含 sm_89（`cuobjdump`/构建日志佐证）；
  4. methanol DF-RHF fp64 与 auto 单点都收敛，且 auto 的
     `fp32_entry_hits` 非零、两车道能量差 ≤1e-6 Ha —— 证明混合精度代码路径
     在 4090 上真实生效（这正是历史 wheel 遮蔽事故的回归测试）。
- **必须证据**：环境清单 JSON、编译日志（/data/results，sha256）、冒烟 JSON。

### #4 基准框架（deps: #3）

- **问题**：现有 runner 为 5090 写，缺 §4 schema 字段（预热分离、峰值显存、
  SCF 循环、fp32 命中计数、修正项分列、df-vs-conv 行、evidence 锚定），且无
  显存 headroom 预算执行、skipped/OOM 行语义、三级误差分离字段。
- **交付物**：RTX 4090 runner（**复用** `run_benchmark.py` 的
  cpu_run/gpu_run/timed/row-key 合并骨架，不重写已验证逻辑）+ schema 校验器
  （独立可测模块，含 §4 全部 8 条强制规则）+ 显存纪律执行（跑前测 free/total、
  跑中记 peak 与 host RSS、超预算停/升级并记 `skipped`）+ **可选** reference-adapter
  预计算输出读取接口与逐例甄别表（§3.6；外部程序缺失时不阻塞主基线）+ schema 文档
  （即 §4，随代码落地）。
- **用例准入验收**：先交付 §3.6 甄别表；以多任务、MD/优化、错误结束、未知基组/几何、资源超预算等实际候选证明筛选理由正确。没有准入用例的方法明确标记缺口，禁止用文件计数宣称覆盖。
- **可运行验收**：校验器单测通过（合规样例 pass、逐条违规样例被拒，含
  skipped/OOM 行与 mem 块缺失样例）；methanol 冒烟产出合规 JSON（mem 块齐全）；
  `--help` 完整；CPU 哨兵行可生成。
- **必须证据**：校验器测试输出、methanol 冒烟 JSON（/data/results，sha256）。

### #5–#8 四方法能量基准（deps: 各自 #4；互相独立可并行排期）

**共同的试点阶梯纪律（§3.5）**：每个方法都从 S0 冒烟 + S1 小体系开始，
按上一级实测峰值显存 + headroom 预算决定是否进入 S2；S3/S4 当前暂缓；**不排期预计超
24 GB 的用例**；延期候选（Vitamin C·tzvp / Tamoxifen·tzvp / Azadirachtin 级）
只在显存预算实测评估通过后进入。每一级都要有：GPU-fp64 / GPU-auto / CPU-fp64
三车道 + df-vs-conv 对照（至少 S1 一组）+ 可负担 UKS 哨兵（精度列）。
历史 5090 矩阵规模（Vitamin C 208/395 AO、Tamoxifen 537/1042 AO、Aza ~1900 AO）
只是容量先验（B 级证据），4090 的实际边界以实测为准。

- **#5 B3LYP**：杂化 K 是 fp32 收缩历史发散点（5090 上 ~1877 AO 发散域内
  auto 比 fp64 慢 2.4-3.1x）；4090 上发散域位置、正确性与净速度全部未知。
  交付阶梯化的 {体系 × def2-SVP/def2-TZVP} × {GPU-fp64, GPU-auto, CPU-fp64}
  能量行（DF，auxbasis 两侧钉死 def2-universal-jkfit）+ df-vs-conv 对照行。
  验收：进入的每一级全部 auto 行 `|dE_vs_gpu_fp64| ≤ 1e-6` 且 `converged`；
  (2) 级 dE_vs_cpu_fp64 要么在历史水平、要么带具名差异注记；发散/OOM 行如实
  记 fail/边界而非跳过。
- **#6 M06-2X**：全杂化 meta-GGA，本项目从未测过；meta-GGA 对网格敏感。
  同 #5 阶梯 + **网格稳定性检查**（level 3 vs 4 vs 5 的 fp64 能量差列，确认
  精度对比所用网格下能量已网格收敛，或显式记录网格误差）。
- **#7 omegaB97M-V（含 VV10/NLC）**：RSH + NLC 路径从未在混合精度下验证；
  `nlcgrids` 精度、VV10 内核在 fp32 车道下的行为未知；NLC 项遗漏即整体方法错误。
  同 #5 阶梯（`do_nlc` 开启，nlcgrids 显式记录）+ NLC 项分列校验
  （`e_terms.nlc` 非空且 CPU/GPU 一致到精度门）。
- **#8 完整 r2SCAN-3c**：r2SCAN-3c 是**整体方法**：def2-mTZVPP + r2SCAN +
  D4(定制参数) + gCP，缺一即错。3c 驱动的 to_gpu 顺序坑（dispersion 必须在
  to_gpu 之后挂）与修正项完整性是主要风险；D4/gCP 在 GPU 车道下的调用路径与
  CPU 哨兵必须逐项对齐。同 #5 阶梯 + 修正项分列表（e_xc / dispersion / gcp
  各自 GPU-fp64 vs GPU-auto vs CPU-fp64；gCP/D4 是确定性的，应逐位或近逐位
  一致）+ 3c 专用基组核验（nao 与 def2-mTZVPP 一致）。
  **交付记录（2026-10-04，待人工审查）**：runner 3c 整体方法模式
  （`--xc r2scan3c`：解析别名→纯 r2SCAN + 钉死 def2-mtzvpp + 规定 D4/gCP
  spec 自动填充 method 块 + 运行前 D4/gCP 构造探测；修正项在
  density_fit 之后挂接，顺序坑已复验）+ 校验器 R11（D4/gCP 跨车道一致性，
  实测全部 4 文档逐位一致）。阶梯 S0 甲醇(63 AO)/S1 苯(144 AO)+甲氧
  UKS(58 AO)/S2 维生素 C(298 AO，准入依据 S1 实测峰值 1100 MiB vs 2048
  headroom)，全部 5 车道 × 4 文档 schema-VALID；auto 车道精度门全过
  （3.4e-13/-1.3e-12/8.8e-12/3.9e-12 Ha，含 UKS 哨兵——issue #7 的 UKS
  DIIS 停滞在 r2SCAN-3c 上未复现）；fp32-screen 全部诚实 fail
  （1.7e-6..1.8e-5 Ha，实测 fp32 噪声底）；(2) 级实现一致性 3.6e-12..
  1.7e-11 Ha。scf_summary 语义实测修正：`energy_tot()` 把修正钩子的返回值
  （D4+gCP 之和）写入 `scf_summary['dispersion']`，e_terms 的 D4 分量取自
  专用键 `dispersion_d4`。

### #9 热点归属（deps: #5, #6, #7, #8）

- **问题**：4090 的成本结构未知；规则是"先归属再动手"，不得先选未 profile
  的 CUDA 改法。历史 5090 归属（r2SCAN SCF 的 72.8% 在 XC 网格等）只作先验。
- **交付物**：四方法在已准入的小/中等 cohort 上的
  **阶段 × 函数 × CUDA 内核**归属表：SCF 迭代合计 vs 内核分解（CDERI 构建、
  JK、XC 网格 eval_ao/libxc/GEMM、NLC、D4/gCP、对角化/DIIS），每次计时前后
  设备同步，带 `[unattributed]` 自检行与调用计数。
- **可运行验收**：归属表每行 `[unattributed]` ≤ 10%；fp64/fp32 背靠背对跑
  （防"改动未生效伪装成收益小"）；产出按收益排序的候选清单（含预估上限与
  精度风险分级，套用 §2.3 的"误差落在哪"规律预筛）。
- **必须证据**：归属 JSON + 日志（/data/results，sha256）+ 候选清单文档。

### #10 热点受控混合精度（deps: #9）

- **问题**：把 #9 的候选按"先量精度再动代码"的顺序实现；每个候选先做
  精度探针（小体系实测误差落点），再实现，再全矩阵复测。
- **交付物**：逐候选的 (i) 精度探针证据 → (ii) 实现或否决记录 → (iii) 全矩阵
  精度+速度复测。安全开关默认关（opt-in），严格 FP64 回退保留。
- **可运行验收**：每个上线路径全矩阵 `|dE| ≤ 1e-6` 且端到端净加速为正
  （**含切换开销、FP64 尾巴、回退成本**——发散域内负加速必须如实报告并给出
  使用边界，如"该尺寸域 SCF 直接 fp64"）；未达标候选回退并记录。
- **必须证据**：前后基准 JSON（各 sha256）、否决记录、开关文档。

### #11 显存边界测绘与批量（deps: #10）

**范围拆分（用户规则：host RAM/VRAM 协作是后续研究阶段，在初期小/中体系
正确性与 profiling 门之后）**：

- **本期 #11 只做 (a) 安全容量测绘**：
  - 峰值显存 vs AO 数曲线（按 §3.5 阶梯逐级实测），标注 OOM 边界与
    `skipped`（headroom 预算不足）边界——两者分开记；
  - 24 GB 比 5090 少 8 GB，历史显存墙（5090 上 Hessian ~2500 AO 外推、各种
    固定分块 OOM）在 4090 全部提前，能量/梯度路径的显存边界未测绘；
  - 显存感知分块核查（历史 4 处"固定分块"修复在 4090 复验）；
  - 批量单点吞吐可选通道（复用 `mixedprec/batch_rhf.py` 思路，能量矩阵优先）。
- **(b) 后续研究阶段（本 issue 不实现，显式 blocked-on-research）**：
  tiling、streaming、pinned-host buffers、transfer/compute overlap、
  out-of-core fallback 等 host RAM/VRAM 协作算法。启动条件：#5-#8 能量门
  与 #9/#10 profiling/优化门全部通过。届时**每个候选必须实测传输与完整
  端到端成本及数值误差**——绝不预设 managed memory / CPU offload /
  streaming 会更快；负结果如实保留。
- **可运行验收（只对 (a)）**：能量矩阵已进入的各级零非预期 OOM（预期外
  OOM = fail）；每格峰值显存入 schema `mem` 块；测绘曲线 + 边界结论；
  批量通道（若做）断点续跑与失败记录测试。
- **必须证据**：显存测绘 JSON + 日志（/data/results，sha256）。
- **交付记录（2026-10-04，待人工审查）**：(a) 完成。测绘器
  `vram_mapper.py`（每格独立子进程、deviceSynchronize、MemorySampler、
  预算门，oom 与 skipped 分开记，schema `vram-boundary-v1`）：7 格阶梯
  全跑，能量侧最大实测 GPU 常驻格 1042 AO（峰值 8047 MiB）、梯度侧
  537 AO B3LYP（3311 MiB）、934 AO Azadirachtin/svp（10247 MiB）全部
  within-budget，**零非预期 OOM**；首个超预算格 Aza/tzvp（cderi 上界
  59 GiB > 21.5 GiB 设备预算）记为显式 headroom-skip。CDERI 常驻边界：
  cderi ≤15.1 GiB 上界全部 GPU 常驻；强制既有 host 回退（诊断性，
  非本 (b) 项 offload 开发）在 Tamoxifen/tzvp 上把 GPU 峰值 8047→2371 MiB
  （host RSS 8.4 GiB，能量一致到 ~1e-12 Ha）。分块核查 `chunk_audit.py`：
  历史 4 处修复位全部仍在查显存，但 **H1（grad/rhf.py get_grad_hcore）
  的 5090 时代修复漏了 fp64 itemsize——cap 实际把张量限制在可用显存的
  100% 而非注释声称的 1/8**（H2/H3 都正确除以 8）；已修复（除数补 ×8），
  甲醇梯度回归一致（~7e-12 相对）；新固定分块扫描 0 发现。批量通道
  `batch_channel.py`：能量矩阵 10 格（含 r2scan3c 整体方法与 UKS-级
  NLC 格）10/10 ok、808 格/小时，断点续跑（重调 0 重跑）与失败记录/
  running 复位均有单测（30 个新纯核测试，无 GPU 依赖）。证据见
  `docs/issue-11-vram-boundaries.md` 与 /data/results/.../issue-11
  （sha256 锚定）。(b) 维持 blocked-on-research 未动。
- **返工记录（2026-10-06，Codex 审查 P1，待人工审查）**：原预算门
  以**设备总容量**为准入基准，P1 缺陷复现
  gate_decision(8,24564,1024,2048)→(True,None)（仅 1 GiB free、2 GiB
  reserve 仍放行 8 GiB 请求）。已重写为准入契约
  **current-free-minus-reserve**（`vram_mapper.gate_decision`/
  `gate_record`，mapper probe、batch_channel、能量/梯度 runner 四处
  共用同一契约，每行/每格带完整 gate 记录：resident 请求、reserve、
  理论容量单独报告且永不作为决策依据；非正预算安全跳过；零 OOM
  探测；不编造工作集估计——reserve 覆盖、事后实测峰值）。host-CDERI
  格记录 GPU 工作预算（free−reserve）与 host RAM 上界（张量+4 GiB
  staging ≤ MemAvailable）。批量通道 skipped 行语义改为下次调用重试
  （跳过是瞬态设备状态而非格属性），gate 无法求值时记
  gate-indeterminate 而非伪造放行。derive_boundaries 增加
  admission_gate_audit（缺 gate 记录格暴露 + skip 原因直方图）。
  回归测试先行（P1 场景红→绿）；新增/重写 21 个纯核测试，全套 332
  OK（双策略）。实测证据（issue-11-rework-20261006，sha256 锚定）：
  20.5 GiB 占用进程下 Tamoxifen/svp 探针如实 skip（free 2272 −
  reserve 2048 = 224 MiB 预算 < 2.8 GiB 请求，即 P1 场景本身）；free
  120 MiB < reserve 时小格也安全跳过；8 格阶梯重跑 7 ok + Aza/tzvp
  beyond-device-capacity skip（零非预期 OOM，gate 记录齐全，host 格
  记录 gpu_working_budget 21607.8 MiB / host_available 29.3 GiB）；
  批量通道 3 ok + 1 gate-skip，重调仅重试被跳过格；能量/梯度 runner
  甲醇冒烟行均带 current-free-minus-reserve gate 且 schema-VALID。
  H1 itemsize×8 修复保持不变。历史原始证据（issue-11 目录）原样
  保留。

### #12 梯度（deps: #10, #11；硬前置：#5-#8 能量门全部 accepted）

- **问题**：能量门通过前不做梯度（用户规则）；梯度误差与 SCF 收敛路径差
  混合存在，需要同态分解对照。
- **交付物**：四方法梯度矩阵（max|Δg| vs GPU-fp64 与 CPU-fp64 + 计时 +
  峰值显存），UKS 哨兵行；对优化工作流的建议（阈值对照 geomeTRIC 3e-4）。
- **可运行验收**：能量门状态由工具核验为 accepted 后方可 start；全部梯度行
  max|Δg| 低于优化阈值一个量级（历史水平 ~1e-5 量级，以 4090 实测为准）；
  计时含 SCF+梯度端到端。
- **必须证据**：合规基准 JSON + 同态分解对照记录。
- **交付记录（2026-10-04，待人工审查；实验性）**：按 2026-10-04 用户
  继续策略以 QC_CONTINUE_UNREVIEWED=1 开展（#5-#8 均 review_requested，
  #7 有未解决能量门失败）——本 issue 不是已验收梯度精度。梯度 runner
  （rtx4090-gradient-v1：fp64/auto/fp32-grad/auto-fp64-grad 四 GPU 车道
  同态分解 + cpu-fp64 哨兵；梯度门 max|Δg|≤3e-5 Eh/Bohr）+ 校验器 G1-G5
  （含车道有效性与 G5 provisional 注记强制）。**结构性缺口修复**：
  `gpu4pyscf/df/grad/uhf.py` 原硬编码 fp64 内核，UKS 杂化梯度静默
  忽略 fp32（attempt-1 实测 grad_ip1=0）；已镜像 rhf.py 的 fp32 处理
  （仅积分内核+压缩 pair 张量 fp32，度量求解保持 fp64），+2 回归测试。
  矩阵 9 格×5 车道全 schema-VALID：7/9 auto 过门（1.2e-6..4.9e-6；
  fp32 内核项全部 ≤2.4e-6，含 UKS）；2 个诚实 UKS fail
  （b3lyp/methoxy 1.16e-3、wb97m-v/methoxy 3.69e-2）经同态分解完全
  归因 SCF 路径（auto-fp64-grad 同值），后者精确复现 #7 的
  -1.2826e-6 Ha DIIS 不动点——fp32 梯度内核无罪，根因是继承的 UKS
  能量侧解选择问题（deferred #7 家族）。建议：闭壳层四方法 auto 梯度
  可用；UKS 在 #7 家族解决前走 fp64 梯度。scientific_gate_passed=
  false（2/9 auto 超门，如实保留）。详见 docs/issue-12-gradients.md。

---

## 6. 工作流、工具与人工审查交接

### 6.1 状态机（`tools/consumer_gpu_agent.py`，仓库外状态）

```
pending ──start──> running ──finish(implemented)──> implemented ──handoff──> review_requested
   ^                  │ │                                                   │
   │            fail/ │ └─finish(blocked)──> blocked <──unblock──┐           │review
   │           expire │                                      (外部阻塞)    │
   │                  v                                                    v
   │            (attempt<max: pending;                          accepted / needs_changes
   │             attempt=max: blocked)                              │            │
   │                                                    reopen      │            │rework
   │                                              (新需求,撤回交接)  │            │
   └──────────────────────────────────────────────────────────────┘            │
   ^                                                                            │
   └────────────────────────────────────────────────────────────────────────────┘
```

- **依赖门**：`start` 拒绝任何依赖未 `accepted` 的任务。
- **审查门**：`accepted` 只能由 `review --verdict accepted --reviewer <name>
  --evidence <ref>` 写入（显式人工审查记录：审查者、日期、证据引用）；
  任何其它路径（finish/fail/手改状态字段被校验器拒绝）都到不了 accepted。
- **reopen**：`review_requested → pending`（attempt 不清零），仅在交接后、
  审查前需要**撤回或按新需求修订**时使用；事件日志记录原因。已 accepted
  的任务不可 reopen（那是新的 issue 或 needs_changes 流程的事）。
- **GPU 运行锁**：仓库外锁文件（pid + 心跳），同时只允许一个 GPU 任务；
  死进程/过期锁可被接管并记录接管事件。
- **超时与重试**：任务记录 `timeout_minutes`（默认 240）与 `deadline`；
  `expire` 把超时视为一次失败（attempt+1），attempt ≥ max（默认 3）→ blocked。
- **交接**：`handoff` 生成 `review_request.json`（状态、分支、commit、变更文件、
  测试命令与结果、证据引用、审查要点）+ `issue-N.patch`（仅本 issue 变更，
  相对 base 的 git diff）+ PR 描述 markdown，全部写入仓库外 `--outbox`。
- **发布清单**：`manifest` 按 root-owned publisher 的固定格式生成
  `outbox/issue-N.json`（精确键：issue/base_sha/title/body/files；路径合法性、
  隐藏文件拒绝、大小上限、≤50 文件逐项校验；mode 0644）。工具本身不 push、
  不 merge、不关 issue、不碰任何凭据、不调用外部模型。
- **日报**：`daily-report` 输出机器可读 JSON（§7 字段），服务器定时器负责发布，
  工具不实现定时。

### 6.2 人工审查协议（Codex 显式审查，用户触发）

1. ZCode 完成实现+自测 → `handoff` → 状态 `review_requested`，**停**。
2. 用户让 Codex 审查 → Codex 看代码 + 证据（outbox 的 review_request.json、
   patch、/data/results 的 sha256 锚定文件、复现命令）。
3. 结论写回：`review --verdict accepted|needs_changes --reviewer Codex
   --evidence <证据引用>`；needs_changes 附返工要点 → `rework` → `pending`。
4. 未有审查记录的任务永远不能 accepted；ZCode 不得自行 merge、不得关闭 issue、
   不得自动触发审查。

---

## 7. 日报字段（机器可读，`daily-report` 输出）

```jsonc
{
  "schema": "consumer-gpu-daily-report-v1",
  "generated_at_utc": "...",
  "project": "gpu4pyscf-mixedprec (RTX 4090 consumer DFT)",
  "issues": [ {"issue": 2, "title": "...", "status": "...", "attempt": 0,
               "branch": "...", "updated_at": "...", "evidence_count": 0} ],
  "recent_events": [ {"ts": "...", "issue": 2, "event": "start|finish|fail|..."} ],
  "gpu_lock": {"issue": 0, "pid": 0, "heartbeat_at": "..."} | null,
  "in_review": [2],            // review_requested
  "accepted": [],              // 含审查记录摘要
  "blocked": [ {"issue": 0, "note": "..."} ],
  "next_actions": [ {"issue": 3, "action": "start (deps accepted)"} ],
  "unverified": [              // 诚实清单：没有证据支撑的声明一律在此暴露
    {"issue": 5, "note": "no accuracy evidence recorded yet"}
  ]
}
```

日报正文（服务器模板渲染）只引用本 JSON 与 `/data/results` 的 sha256 锚定数据；
"代理结束"不等于"验收通过"，未运行/未验证的结果必须写明。

---

## 8. 有界执行规程

1. **单 GPU 任务**：任何时刻至多一个 GPU 任务在跑（工具锁强制）；构建并行度
   ≤ 4（`-j 4`）。
2. **超时预算**：跑前估时，`timeout_minutes = 2 × 估时`；命令层面另加
   `timeout`/`setsid`（长 CPU 参照一律磁盘 TMPDIR，历史教训：tmpfs chkfile
   撑爆内存触发看门狗）。
3. **重试上限**：每任务 attempt ≤ 3，超限 blocked 并报告；不无限重试。
4. **中断恢复**：进程死亡/会话中断 → 下次 `expire`/`start` 接管过期锁，
   attempt 照常计数；断点续跑由各 runner 的行级合并保证。
5. **止损**：同一问题连续 2 次同因失败即标 blocked（附日志），换路径或报告，
   不带病硬跑。
6. **禁止**：sudo、凭据访问、直接 GitHub 调用、外部消息、auto-merge、
  自动触发审查、自批、关闭 issue、删除无关数据、改动 publisher 基础设施、
   发布隐藏文件/日志/大二进制/任何凭据类内容。
7. **提交**：每 issue 一个分支、一个小提交、人类可审的文本 diff；工作流状态
   与长日志在仓库外。

---

## 9. 风险登记册

| 风险 | 缓解 |
|---|---|
| wheel 遮蔽（静默 fp64 降级） | #3 验收第 4 条即回归测试；`fp32_entry_hits` 全零 → 校验器 fail |
| CPU/GPU auxbasis 或网格默认值不一致 | method 块逐字段记录 + 校验器要求 lane 间逐项相等 |
| 杂化泛函 auto SCF 发散域（4090 位置未知） | 混沌守卫已存在；#5 起每行记 `precision_switch.trigger`；发散域内如实报负加速 |
| D4/gCP 修正遗漏或 GPU/CPU 参数不一致 | #8 修正项分列 + 参数写入 method 块；缺项即 fail |
| 3c 驱动 to_gpu 顺序坑 | #8 显式复验（能量对 CPU 哨兵） |
| 历史负面路线被无意识重做 | §2.3 表 + #9 候选清单必须引用该表预筛 |
| 显存墙提前（24 GB） | §3.5 阶梯 + headroom 预算 + #11(a) 测绘；固定分块模式在 review 中专门检查 |
| 大用例被盲目排期 | 排期准入只认上一级实测峰值显存 + 预算，不认估算 |
| 计时被热启动污染 | schema 强制独立初始化样本；SCF 冷启动单发记 `t_scf_s` |
| 把 DF 近似误差当成精度误差 | §3.3 分离规则 + 校验器字段隔离 |
| skipped/OOM 格被静默丢失或杜撰通过 | §4 规则 2：status/skip_reason 强制 + 校验器检查 |
| (2) 级实现差异被误归因为混合精度 | §3.4 表 + `impl_agreement_note` 具名要求（校验器规则 8） |
| (3) 级跨程序差异被当成精度门或加速证据 | §3.6：cross_program 独立块、matched 标注、永不设 1e-6 门 |
| 随附测试被误当作可用 benchmark，或旧能量错配方法/步骤 | §3.6 分类、逐例审核、精确几何和 method signature；预计算输出只作独立参照 |
| 假设 offload/streaming 更快 | #11(b) 显式 blocked-on-research；实测传输与端到端成本前不实现 |

---

## 10. 本计划之外（显式排除）

本期 issue #2 只交付：本计划文档、任务矩阵、schema 定义、工作流工具与其测试、
AGENTS.md 更新。**不安装环境、不编译 CUDA、不改物理代码、不跑昂贵基准、
不重复实现已存在的代码**。同样显式排除：

- 正式 G16/ORCA 适配器与 cohort 验收代码（交由 #4）；服务器私有安装与初筛已经完成，但不构成 issue #2 的物理代码或 benchmark 验收；
- host RAM/VRAM 协作算法（tiling、streaming、pinned-host、overlap、out-of-core，
  §#11(b) 显式 blocked-on-research）与大体系/离线实现；
- 任何把历史 RTX 5090 数字当作 RTX 4090 证据的引用。

一切优化声明待 #3 起逐步以 RTX 4090 实测证据建立。
