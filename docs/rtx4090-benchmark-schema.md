# RTX 4090 基准结果 Schema 与校验规则（rtx4090-energy-v1）

**状态**: issue #4 交付物 · 实现为 `mixedprec/benchmark/rtx4090/`（`schema_validator.py` 校验器、`rtx4090_runner.py` 运行器、`g16_curation.py` 甄别适配器）
**上位文档**: `docs/consumer-gpu-plan.md` §4（本文件是其代码化落地）· 日期: 2026-10-03

一个基准 run 产出**两类文件**（都写入 `/data/results/gpu4pyscf-agent/tasks/issue-N/`，不可变、sha256 锚定）：

1. **raw + runlog**（`<stem>.raw.json` / `<stem>.runlog.log`）：原始结果与运行日志；
2. **schema 文档**（`<stem>.json`）：`schema_version: "rtx4090-energy-v1"`，每行 `evidence` 以 `path@sha256` 引用第 1 类文件并附完整复现命令。

校验命令：

```sh
python mixedprec/benchmark/rtx4090/schema_validator.py RESULT.json \
    --evidence-root /data/results
```

退出码 0 = 合规（可作验收证据）；非 0 = 不合规，**不合规的 JSON 不许进验收**。

## 1. 文档结构

```
{
  "schema_version": "rtx4090-energy-v1",
  "meta": { ... 一次 run 一份的机器/环境指纹 ... },
  "rows": [ ... 每个格子一行（含 skipped/OOM 行） ... ]
}
```

`meta` 必含：`gpu_name, gpu_total_mem_mib, driver_version, cuda_version,
cpu_model, cpu_threads, ram_gb, python, cupy, pyscf, gpu4pyscf_source_commit,
gpu4pyscf_source_path, libxc, mem_headroom_mib, timestamp_utc, run_id`。

每行 `row_key = {geom, basis, xc, lane}` 唯一（跨调用按 row_key 合并复跑），
`pilot_stage ∈ S0..S4`，`status ∈ ok|skipped|oom|error`，
`lane ∈ fp64|auto|fp32-screen|cpu-fp64|df-vs-conv|xref-g16|xref-orca`。

`ok` 行必含完整 `system`（geom_file/geom_sha256/natm/nao/charge/spin/nelec/
cart/geom_units）、`method`（xc/basis/xc_definition/auxbasis/with_df/grids/
nlcgrids/dispersion/gcp/conv_tol/conv_tol_grad/direct_scf_tol/init_guess/
max_cycle/open_shell）、`result`（e_tot/converged/scf_cycles/e_terms/timings/
mem/fp32_entry_hits/precision_switch/fallback_events/errors）、`accuracy`、
`evidence`。`skipped/oom/error` 行免除 system/method 结构块，但受规则
R2/R7 约束（原因与显存实测必填）。

`method.xc_definition`（issue #5 起，DFT 行强制）：记录别名实际解析的
XC 定义合约——libxc 泛函分解（id+factor 列表）、hyb_coeff、rsh
(omega/alpha/beta)、nlc_params。例如 pyscf `b3lyp` = libxc 402（VWN3/
VWN_RPA 形式，与 Gaussian 兼容），区别于 `b3lyp5`（VWN5，H2 上差 ~7 mHa）。
`xc='hf'` 无 XC 部分豁免。探测失败的个别字段为 null，但
`libxc_decomposition` 为空的 DFT 行被规则 R4 拒绝（tag
`xc-definition-missing`）。

## 2. 八条强制规则（校验器逐条检查）

| # | 规则 | 校验 tag（拒绝原因） |
|---|---|---|
| R1 | 计时纪律：`samples_s ≥ 3` 次独立初始化样本、`warmup_s` 独立存在、`median/min/max` 与样本一致、`end_to_end_s ≥ 0`、GPU 车道 `sync` 字段须含 device-sync 证据 | `samples-count`, `samples-values`, `sample-statistics`, `warmup-missing`, `end-to-end`, `sync-evidence` |
| R2 | 状态语义：非 ok 行必须带 `skip_reason`/`errors`（OOM 是硬件边界记录，非 skip）；`pass` 仅当 `converged` 且（fp64 参考行或 `|dE_vs_gpu_fp64| ≤ 1e-6 Ha`）；跳过与 OOM 不得静默缺失、不得杜撰通过 | `skip-reason-missing`, `non-ok-unexplained`, `errors-missing`, `pass-on-non-ok`, `pass-not-converged`, `verdict-on-unconverged`, `gate-exceeded`, `pass-without-reference`, `fp64-self-delta` |
| R3 | 车道有效性：ok 的 `auto`/`fp32-screen` 行 `fp32_entry_hits` 全零 = 混合精度未生效（wheel 遮蔽或模式未设），判 `lane-ineffective`，永不 pass | `lane-ineffective`, `lane-ineffective-pass` |
| R4 | 方法完整性：r2SCAN-3c 行必须 `e_terms.dispersion`、`e_terms.gcp` 非空且 basis == `def2-mTZVPP`（缺项=方法错误）；omegaB97M-V 行必须 `e_terms.nlc` 非空；DFT 行必须带非空 `xc_definition.libxc_decomposition`（泛函变体合约，B3LYP 变体等） | `r2scan3c-terms-missing`, `r2scan3c-basis`, `nlc-missing`, `xc-definition-missing` |
| R5 | 误差分级隔离：`dE_df_approximation` 只允许出现在 `df-vs-conv` 行；`cross_program` 条目必须带 `matched` 标注且承载三级比较——两者都不得写进 (1) 级精度误差字段 | `df-approx-misplaced`, `df-approx-missing`, `cross-program` |
| R6 | 证据锚定：每行 `evidence.raw_log/result_json` 为 `path@sha256`，位于 evidence root 下、存在且 sha256 匹配；`command` 为完整复现命令 | `evidence-format`, `evidence-missing`, `evidence-hash-mismatch`, `evidence-outside-root`, `repro-command` |
| R7 | 显存纪律：ok 行 `mem` 块齐全（free/total/peak/host RSS/headroom/within_budget，全部实测）；`within_budget == false` 必须带 `budget_explanation`；skipped 行至少实测 free/total | `mem-block-missing`, `mem-field`, `budget-breach-unexplained`, `mem-block-skipped` |
| R8 | 实现一致性：`|dE_vs_cpu_fp64| > 1e-8 Ha` 且无具名 `impl_agreement_note` 判 `impl-unexplained`（不得归因混合精度、不得静默放行） | `impl-unexplained` |

结构层错误（schema_version、meta 缺字段、row_key 缺字段/重复、枚举越界、
块缺失等）以 `structure/*` tag 报告。

## 3. 误差三级分离（与计划 §3.4 一致）

| 级 | 字段 | 参照 | 判据 |
|---|---|---|---|
| (1) 精度误差（唯一硬门） | `accuracy.dE_vs_gpu_fp64` | GPU4PySCF FP64（同源码、同设置） | **≤ 1e-6 Ha 且 converged** |
| (2) 实现一致性（诊断） | `accuracy.dE_vs_cpu_fp64` + `impl_agreement_note` | CPU PySCF FP64 | 匹配设置下 ~1e-10 量级；> 1e-8 须具名归因 |
| (3) 跨程序（仅报告） | `cross_program[]`（带 `matched`/`unmatched_fields`） | G16/ORCA 预计算输出 | **永不设门**；unmatched 差异不得归因混合精度 |

DF 近似误差独立记录于 `df-vs-conv` 行的 `accuracy.dE_df_approximation`，
永不与 (1) 混算（R5）。

## 4. 运行器行为（rtx4090_runner.py）

- **每车道独立初始化**：warmup 1 次 + `--samples N`（≥3）次全新
  mol+mf 运行；每次含 mol.build、CDERI 构建、SCF 全程（auto 车道的
  fp32 头、精度切换与 fp64 尾都在内）；每相位前后
  `cupy.cuda.runtime.deviceSynchronize()`。
- **fp32 命中计数**：对 `gpu4pyscf.lib.precision.get_precision` 安装
  计数包装，按调用方归因到 `df_jk / eval_rho / xc_grid / grad_ip1`
  （df_jk=`gpu4pyscf.df.df_jk`；eval_rho/xc_grid=`gpu4pyscf.dft.numint`
  的密度侧/Fock 侧分支；grad/hessian 模块=grad_ip1）。这是历史
  wheel 遮蔽事故（静默 fp64 降级）的回归探针（R3）。
- **精度切换证据**：SCF callback 记录每轮 (cycle, E, mode)，从 fp32→fp64
  的模式翻转定位切换轮次，并按最后一轮 fp32 |dE| 推断触发类型
  （<1e-4 → `dE`；>1e-2 → `chaos-guard`；否则 `unknown`）。仅观测，不改物理代码。
- **e_terms**：`scf_summary` 的 e1/coul/exc/dispersion/gcp + 包装
  `nr_nlc_vxc` 单列 NLC 项（RKS 把 enlc 折进 exc，此处拆出）。
- **显存纪律**：跑前 `memGetInfo` 实测 free/total；跑中后台线程轮询
  free 低水位（= 峰值分配，包含 CuPy 池缓存不归还驱动的部分）与池
  used 高水位；host RSS 取进程 ru_maxrss 高水位（同进程多 cell 时为
  累计高水位，见 §5 已知限制）。`--min-free-mib` 触发的跳过记
  `skipped/headroom-budget`；OOM 记 `oom` + errors（R2/R7）。
- **车道**: `fp64`（参考）、`auto`、`fp32-screen`（纯 fp32，conv_tol
  自动下限 3e-5——实测 DF J/K 噪声底 ~4e-5，见
  gpu4pyscf/lib/precision.py）、`cpu-fp64`（PySCF 哨兵）、`df-vs-conv`
  （常规积分 fp64，DF 近似差）。
- **fp32-screen 的 (2) 级注记**：derive 对 fp32-screen 行自动附具名
  `impl_agreement_note`（纯 fp32 车道对 cpu-fp64 的偏差是其自身收缩
  噪声底，非实现差异；实现一致性水平由同行组 fp64 车道单独量化）。
  该行的 (1) 级 verdict 仍按 1e-6 门如实判定（通常为 fail）。
- **来源鉴伪**：启动即检查 `gpu4pyscf.__file__` 指向源码树，wheel 遮蔽
  直接拒绝运行。
- **行合并**：`--merge-from` 载入旧 schema 文档，按 row_key upsert，
  断点续跑不丢行。

## 5. 已知限制（诚实清单）

- `t_xc_grid_s/t_jk_s/t_nlc_s/t_disp_s` 本期为 `null`：内核级分解计时
  属 issue #9 的热点归属表，`t_scf_s`（SCF 合计）与 `t_cderi_s` 已有。
- `host_rss_peak_mib` 是进程 ru_maxrss 高水位：同一进程先前 cell 的
  峰值会被计入（低估单 cell 峰值不会发生；多 cell 合并解读时保守）。
- 切换触发类型是**推断值**（由 callback 序列推出），不是从内核日志读出；
  标注为 `unknown` 时表示序列无法归类。
- 校验器的 `--evidence-root ""` 只做格式检查（单测用）；验收时必须带
  root 做存在性+sha256 检查。

## 6. G16 参照甄别（g16_curation.py，私有）

捆绑的 G16 测试**默认不可用作参照**。机器甄别最多给出 `untriaged`
（临时候选）；`reference_matched / reference_diagnostic / geometry_only`
只能由人工审计产生（计划 §3.6.2 第 5 条）。`read_precomputed` 只读
解析已有日志（route 卡、SCF Done、终止标志、charge/multiplicity、
NBasis），**从不运行 Gaussian**；重算是另行显式安排的例外。分类依据
日志与 route 的原始内容而非索引预消化字段，并对索引-日志不一致、
MD/优化/多任务/多 SCF/错误终止/超 AO 预算等陷阱逐例给出理由
（对抗性测试见 `tools/tests/test_g16_curation.py`）。G16 能量是三级
比较的 (3) 级参照，永不替代 (1) 级的 GPU4PySCF FP64 基线；专有输入、
日志与提取坐标保留服务器私有，不进 GitHub。
