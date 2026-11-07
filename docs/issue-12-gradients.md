# Issue #12: 四方法梯度矩阵（RTX 4090 实测）

**状态**: 待人工审查（review_requested）。本文档只引用
`/data/results/gpu4pyscf-agent/tasks/issue-12/` 下 sha256 锚定的证据；
所有数字为 RTX 4090 实测（A 级证据）。

## 前置门状态（诚实声明）

计划要求"#5–#8 能量门全部 accepted 后方可 start"。实际状态：
`#5–#8` 均 review_requested（未审查），且 **#7 存在一个未解决的能量门失败**
（methoxy/def2-svp/wb97m-v UKS auto 车道 dE = -1.283e-6 Ha > 1e-6 门，
见 deferred-issues.json）。按 2026-10-04 用户策略（记录问题、继续开发），
本 issue 以 `QC_CONTINUE_UNREVIEWED=1` 暂用前置成果，梯度开发为
**实验性**：受影响格全部带 provisional/validation-pending 注记，
绝不宣称已验收梯度精度。**本 issue 的科学门整体判定为
scientific_gate_passed = false**（两格 auto 车道 max|Δg| 超门，
详见下文；根因是继承的能量侧问题，非梯度内核）。

## 交付物

1. **梯度 runner**（`mixedprec/benchmark/rtx4090/gradient_runner.py`）：
   产出 `rtx4090-gradient-v1` 文档。每格四条 GPU 车道 + CPU 哨兵：
   - `fp64`：fp64 SCF + fp64 梯度内核（参照）；
   - `auto`：auto SCF + fp32 梯度内核（生产车道；`grad/rhf.py
     _resolve_precision` 把 'auto' 解析为 fp32——梯度无迭代，无 fp64 尾巴）；
   - `fp32-grad`：fp64 SCF + fp32 梯度内核（**同态分解对照**：单独量出
     梯度内核误差）；
   - `auto-fp64-grad`：auto SCF + fp64 梯度内核（**同态分解对照**：单独
     量出 SCF 收敛路径差）。
   max|Δg(auto)| ≈ 内核误差 + SCF 路径差（加小交互项），这正回答计划
   提出的"梯度误差与 SCF 收敛路径差混合存在，需要同态分解对照"。
   全套计时/显存/entry-hit/evidence 纪律与能量 runner 相同；梯度门
   max|Δg| ≤ 3e-5 Eh/Bohr（低于 geomeTRIC 默认 gmax 3e-4 一个量级）。
2. **schema 校验器梯度契约**（`schema_validator.py`，G1–G5 + R1/R2/R3
   梯度变体）：梯度载荷与范数一致性、门值与参照层级（gate 只对同实现
   gpu-fp64，永不 cross-tier）、车道有效性（声称 fp32 但 grad_ip1=0 →
   直接 fail，即 wheel-shadowing 回归守卫的梯度版）、R8/G4 实现一致性
   具名注记、以及 **G5 能量门对齐**——能量门失败的格子上的梯度 pass
   必须带 provisional 注记，禁止静默通过。
3. **UKS fp32 梯度内核补全**（`gpu4pyscf/df/grad/uhf.py`）：
   `_jk_energy_per_atom` 原先无条件用 fp64 内核 `sum_ejk_int3c2e_ip1`，
   导致 UKS 杂化梯度在 fp32/auto 车道下静默跑 fp64（本 issue 首次实测
   发现的结构性缺口，attempt-1 的 grad_ip1=0 行诚实记录了它）。补齐
   与闭壳层 `df/grad/rhf.py` 完全一致的 fp32 处理：仅 int3c2e_ip1
   积分内核与压缩 pair 张量进 float32，j3c@dm 收缩与 j2c 度规求解保持
   float64（度量求解会把 fp32 相对误差放大进梯度）。回归测试 ×2
   （`gpu4pyscf/df/tests/test_df_mixedprec.py`：fp32 内核选择断言 +
   auto→fp32 解析断言，均含 1e-5 精度检查）。

## 梯度矩阵（9 格 × 5 车道，全部 schema-VALID）

| 格子 | auto max\|Δg\| vs gpu-fp64 | fp32-grad（内核项） | auto-fp64-grad（SCF 路径项） | auto 判定 |
|---|---|---|---|---|
| b3lyp / methanol /svp | 2.30e-6 | 2.45e-6 | 2.9e-8 | pass |
| b3lyp / benzene /svp | 1.96e-6 | 1.97e-6 | 2.1e-7 | pass |
| b3lyp / methoxy UKS /svp | **1.16e-3** | 1.58e-6 | **1.16e-3** | fail |
| m062x / benzene /svp | 1.23e-6 | 1.11e-6 | 2.0e-7 | pass |
| m062x / methoxy UKS /svp | 1.95e-6 | 2.03e-6 | 1.63e-6 | pass |
| wb97m-v / benzene /svp | 1.62e-6 | 1.81e-6 | 1.6e-7 | pass |
| wb97m-v / methoxy UKS /svp | **3.69e-2** | 1.94e-6 | **3.69e-2** | fail |
| r2scan3c / benzene /def2-mtzvpp | 1.16e-6 | 1.17e-6 | 2.0e-7 | pass |
| r2scan3c / methoxy UKS /def2-mtzvpp | 4.92e-6 | 9.7e-7 | 4.82e-6 | pass |

（CPU 哨兵行每格都有；gpu-fp64 参照的 max|Δg| vs cpu-fp64 在
1e-7 量级（实现一致性诊断，G4 具名注记），r2scan3c 的 D4+gCP 梯度
钩子在 GPU/CPU 两侧对称挂接。）

### 结论

1. **fp32 梯度内核本身在全部 9 格（含 UKS）都远在门内**：
   fp32-grad 车道 max|Δg| = 1.0e-6 – 2.4e-6 Eh/Bohr，比 3e-5 门低一个
   量级以上，比 geomeTRIC gmax 3e-4 低两个量级——闭壳层与 UKS 一致
   （UKS 是本 issue 补齐内核后才可测的）。
2. **SCF 路径项（auto-fp64-grad）在 7/9 格为 2e-8 – 4.9e-6**，与内核
   项同量级或更小；**两个失败格的 SCF 路径项独占误差**：
   - b3lyp/methoxy UKS：1.16e-3（能量门实际通过，dE=4.0e-8 Ha——
     轨道/占据的微小 UKS 解差异在能量上极小但被梯度放大）；
   - wb97m-v/methoxy UKS：3.69e-2，且其 auto 能量偏移 -1.2826e-6 Ha
     精确复现 issue-#7 的 DIIS 不动点失败——同一个未解决根因的能量面
     与梯度面表现。
   两格均为诚实 fail（validator 与 runner 双重强制），失败保留为证据，
   根因归入 #7 deferred 家族：fp32 头种子在 UKS 盆地里锁定了不同的
   收敛解。**这是"梯度误差与 SCF 收敛路径差混合"的实证分离**——
   梯度内核无罪，SCF 路径有罪。
3. **对优化工作流的建议**：闭壳层四方法可用 auto 梯度（误差 ~2e-6，
   远低于 geomeTRIC 3e-4）；UKS 体系在 #7 类能量失败解决前，梯度走
   fp64 车道（`g.precision_mode='fp64'` 或直接 fp64 SCF），绕开
   fp32-头种子风险。r2scan3c 的 UKS 格 auto 误差 4.9e-6 也在门内，
   但同样继承未审查状态，标注 provisional。

## 执行记录（有界执行）

- Attempt 1（2026-10-04 13:18 UTC start）：代码开发 + 7 格跑完，被
  60 分钟作业墙钟杀死于 wb97m-v benzene 的 CPU 哨兵车道（CPU VV10
  ~184 s/样本）。已跑行保留（row-key merge 续跑协议）。
- Attempt 2（本 attempt）：`expire` 接管过期锁后 (i) merge 补完
  wb97m-v benzene CPU 车道；(ii) 发现并补齐 UKS fp32 梯度内核缺口；
  (iii) 重跑两个 UKS 格（修 dtype 坑一次、修 xc 别名一致性一次——
  别名不一致会造出重复格，已重跑消除）；(iv) 新增 wb97m-v/methoxy
  与 r2scan3c/methoxy 两个 UKS 格；(v) R8/G4 NLC 具名注记补齐
  （复用能量 runner 的 `_nlc_impl_note`）+ energy-gate 引用来源修正
  （issue-? → issue-7，从证据路径提取）。
- 每格独立 warmup + ≥3 独立初始化样本，phase 级 deviceSynchronize，
  峰值显存 ≤338 MiB（全部格 within_budget，headroom 2048 MiB）。
  尝试 1 的原始 lane-ineffective 行保留在
  `issue-12/attempt1-preserved/`（sha256 见其 SHA256SUMS.txt）。

## 证据（全部 sha256 锚定于 /data/results/gpu4pyscf-agent/tasks/issue-12/）

9 个 `*_grad.json`（schema-VALID）+ 对应 `.raw.json`/`.runlog.log` +
`attempt1-preserved/` + `evidence_sha256.txt` 索引。复现命令记录在
每行 `evidence.command`。
