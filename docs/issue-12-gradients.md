# Issue #12 rework — all-repeat scientific aggregation and trustworthy gradients

**状态**: review_requested（暂用，未验收；本 rework 轮针对 Codex 2026-10-06
审查卡 `control/review-20261006.md` 第 1 条 P1 + issue-12-next-round.md）·
**本轮证据**: `/data/results/gpu4pyscf-agent/tasks/issue-12/2026-10-06-allrepeat-rework/`
（sha256 锚定清单见该目录 `evidence_sha256.txt`）·
**源码**: `b938725` + 本 issue 提交 · **科学门**: scientific_gate_passed 见
§5（继承的 methoxy/svp UKS wb97m-v 能量失败按 2026-10-04 策略如实保留）。
前一轮同作业窗口被 60 分钟墙钟中断（完成 3/9 格）；本轮恢复并完成全部
矩阵（行级独立文档，无重复计算），GPU 锁过期后经受支持的 fail→start
路径接管（attempt 3/3 事件记录在 tasks.json）。

## 1. P1 修复：all-repeat 科学聚合（红→绿回归先行）

**缺陷**（审查确认）：`gradient_runner.py` 的采样循环把每个样本的
`e/converged/cycles/nlc/summary/grad` 写进同一组局部变量——**只有最后一个
repeat 的科学产出进入行记录**。若第 1 个样本不收敛或梯度超门而最后一样本
良好，行记录显示良好，早期失败从门里消失。`rtx4090_runner.py`（能量
runner）有完全相同的覆盖模式（审查卡要求审计）。

**修复契约**（`gradient_runner.py` + `schema_validator.py` 新规则 G6）：

1. **每个 repeat 的科学产出全部保留**在 `result.repeats[]`：能量、收敛旗标、
   SCF 圈数、完整 (natm,3) 梯度、实测内核车道（fp32/fp64）、该 repeat 的
   grad_ip1 命中数、e_terms、precision_switch、per-sample 计时
   （t_mol/t_cderi/t_scf/t_grad/sample_s）、stability_verification 记录。
2. **聚合 = 最差 repeat**（`select_worst_repeat`）：不收敛的 repeat 恒为
   最差；否则对组内 fp64 参照 max|dg| 最大的 repeat；参照/CPU 车道取能量
   离中位数最远的 repeat。并列取最低 index，选择与顺序无关。
3. **判 pass 要求全部 repeat 收敛**（候选车道与 fp64 参照车道都是）：
   **不收敛的 FP64 参照 repeat 不能验证任何候选**——该格全部候选记
   `no_reference`（带 verdict_note），永不 pass。
4. **能量 runner 同契约审计修复**（`select_worst_energy_repeat`）：GPU/CPU
   车道同样保留 `result.repeats[]`（能量/收敛/圈数/e_terms/每样本计时/逐
   repeat precision_switch），聚合取最差 repeat（不收敛优先，其次能量离
   中位数最远——报告离群值而非中位数）。
5. **向后兼容的诚实语义**：无 `result.repeats` 的历史行在 derive 时标记
   `repeat_evidence='legacy-insufficient'` 并原样保留其存储聚合——
   **校验器 G6 拒绝这种行上的 pass**（`repeats-legacy-pass`）：历史记录
   缺 all-repeat 证据就不充分用于新验收，永不静默升级。新行标记
   `'all-repeats'`。
6. **G6 强制规则**：`repeats-missing`（ok 梯度行缺 repeats 且无 legacy
   标记）、`repeats-count`（len(repeats) != len(samples_s)，截断的 repeat
   列表是畸形证据）、`repeat-record`（缺字段/类型错误）、
   `repeat-not-converged`（pass 行里有不收敛 repeat）、
   `repeat-worse-than-aggregate`（某 repeat 对组内 fp64 参照的 max|dg|
   超过存储的聚合值——旧 P1 伪装的直接检测）。

**回归测试先行**（`tools/tests/test_issue12_allrepeat_gradients.py`，22 个，
纯 CPU 无 GPU）：首 repeat 不收敛/超门而末 repeat 良好 → 聚合必须 fail
（含三种排列的顺序无关性）、全好 → pass、参照 repeat 不收敛 → 候选全部
不得 pass、缺参照 → no_reference、畸形 repeat 记录 → ValueError、
legacy 行 → 不得 pass；G6 的 5 个对抗样本（缺 repeats、legacy-pass、计数
不匹配、聚合内不收敛 repeat、repeat 比聚合更差）；能量 runner 的最差
repeat 选择 ×3。红→绿验证：修复前 16 项失败（8 错误 + 8 失败），
修复后 22/22 通过。全套 tools 测试 355 个 OK（含既有梯度 runner/校验器
测试的 fixture 升级到 all-repeat 契约）。

## 2. 四方法梯度矩阵重测（all-repeat 证据，RTX 4090 实测）

设置与 attempt-1 逐项一致（DF/def2-universal-jkfit、conv_tol 1e-10、
minao、grids level 3、S1 试点、每车道 warmup + 3 独立样本、相位级
deviceSynchronize）。本轮每个文档的每行都带 `result.repeats`（3
repeat/行），聚合为最差 repeat：

| 格子 | auto max\|Δg\| (worst repeat) | fp32-grad（内核项） | auto-fp64-grad（SCF 路径项） | auto-verify | auto 判定 |
|---|---|---|---|---|---|
| b3lyp / methanol /svp | 2.30e-6 | 2.45e-6 | 2.9e-8 | — | pass |
| b3lyp / benzene /svp | 1.94e-6 | 2.02e-6 | 2.1e-7 | — | pass |
| b3lyp / methoxy UKS /svp | 1.16e-3 | 1.58e-6 | 1.16e-3 | **1.58e-6 pass** | **fail**（继承） |
| m062x / benzene /svp | 1.32e-6 | 1.09e-6 | 2.0e-7 | — | pass |
| m062x / methoxy UKS /svp | 1.97e-6 | 2.03e-6 | 1.63e-6 | — | pass |
| wb97m-v / benzene /svp | 1.67e-6 | 1.84e-6 | 1.7e-7 | — | pass |
| wb97m-v / methoxy UKS /svp | 3.69e-2 | 1.95e-6 | 3.69e-2 | 3.69e-2 **fail**（支线差本身超门） | **fail**（继承） |
| r2scan3c / benzene /def2-mtzvpp | 1.23e-6 | 1.29e-6 | 2.0e-7 | — | pass |
| r2scan3c / methoxy UKS /def2-mtzvpp | 4.92e-6 | 9.67e-7 | 4.82e-6 | — | pass |

（详表以各 JSON 文档为准；本表数字为各文档 derive 后的存储聚合 =
每格全部 repeat 的最差值。）
结论与 attempt-1 一致且现在由全部 repeat 证据支撑：**fp32 梯度内核项在
全部 9 格 ≤2.5e-6 Eh/Bohr**；两个 methoxy UKS 失败格的误差完全由
auto-fp64-grad 车道复现（SCF 解选择），**不是内核误差**。本轮新增的
auto-verify 复测把 b3lyp 失败格修复到过门（1.58e-6），wb97m-v 格
诚实保留 fail（两支线能量差 1.28e-6 Ha 本身超门）。

wb97mv/benzene 文档为 4 行（fp64/auto/fp32-grad/auto-fp64-grad，无
cpu-fp64：attempt-1 已实测该车道 CPU VV10 ~184 s/样本是 60 分钟作业
墙钟杀手；tier-(2) CPU 梯度对照由同基组/同泛函的 methoxy UKS 格与
b3lyp/m062x/r2scan3c benzene 格的 cpu-fp64 行覆盖，行语义见各文档
accuracy 块）。

## 3. methoxy 失败格复测：issue-7 诊断路线（auto+verify）

新增 `auto-verify` 梯度车道（`LANE_SPEC['auto-verify'] = ('auto+verify',
None)`）：SCF 走 issue-7 rework 的 opt-in 解稳定性验证（auto 策略收敛后
同初猜再跑一条严格 fp64 参照轨迹，采纳更低收敛解，决策记录在
`scf_summary['stability_verification']` 并逐 repeat 保留），梯度内核仍
fp32（验证选择的是解，不是内核车道）。校验器车道枚举/R3 车道有效性规则
同步扩展。

复测结果（`*_autoverify.json` 文档，每 repeat 的
stability_verification 决策记录完整保留在 result.repeats 中）：

- **b3lyp / methoxy /svp UKS：auto-verify 过门**，max|Δg| = 1.58e-6
  Eh/Bohr（对比 auto 车道 1.16e-3）——验证道采纳 reference 支线
  （dE_ref−main = −4.03e-8 Ha，两条支线本身几乎简并，选参考支线即
  消除解选择误差；dE_vs_gpu_fp64 = 0.0）。与 issue-7 实测一致
  （该格验证道选择参考支线）。
- **wb97m-v / methoxy /svp UKS：auto-verify 诚实保留 fail**，
  max|Δg| = 3.69e-2 Eh/Bohr——stability_verification 记录显示两条
  支线能量差 dE_ref−main = +1.2825891e-6 Ha，auto 支线（main，
  19 圈，orbital_grad 6.18e-6）本身就是更低收敛解，验证道按规则采纳
  main；相对 fp64 参照车道的差保持超门。这是 issue-7 §3 的结论复现：
  **两支线能量差本身超门**（解多重性的物理后果，非缺陷）；该格的
  exact-match 路线仍是 fp64 SCF（可加 fp32 梯度内核，实测内核项
  1.95e-6）。G5 provisional 注记在该格所有梯度行上强制。

## 4. 独立有限差分检查（新增，审查卡要求）

`fd_gradient_check.py`（执行目录，私有）：中心差分
`(E(x+h)-E(x-h))/2h` 对每车道解析梯度做**独立实现无关**的交叉验证
（h=0.002 bohr；甲醇冒烟 h=0.01/0.005/0.002/0.001 逐级）。

**闭壳层（FD 有效域）**：
- methanol/b3lyp/fp64（h 逐级）：2.97e-5 → 1.22e-5 → 7.2e-6 → 6.5e-6
  （FD 噪声底 ~6.5e-6 Eh/Bohr，网格级 SCF 噪声经 1/2h 放大）；
- benzene/b3lyp（h=0.002，本轮证据目录 `fd_benzene_b3lyp.json`）：
  fp64 max|解析−FD| = **3.66e-6**、auto = **3.44e-6**（z/y 分量按对称性
  归零，x 分量 4.2e-4 量级上吻合到 4e-6）。
FD 在 ~1e-5 量级验证闭壳层解析梯度的结构正确性（导数正确、无分量
错位/符号错误），与 3e-5 门互补，不替代同实现 max|dg| 门。

**UKS methoxy（FD 失效域，如实记录）**：methoxy/b3lyp 的 FD 出现
max|解析−FD| ~7.2e-3，**这不是解析梯度缺陷**——诊断证据
（`fd_methoxy_b3lyp.json`、`fd_methoxy_b3lyp_catom_diag.json`、
`fd_methoxy_m062x_diag.json`）显示**每个位移几何的能量都低于中心能量**
（O 原子与 C 原子、b3lyp 与 m062x 格全部如此；其中 m062x 格的解析梯度
本身过门 1.97e-6）。Taylor 分析：|g|~1e-2 时每个方向表观曲率均为负
在单一分支上不可能——位移后的 UKS SCF 收敛到比中心运行**更低的解分支**
（issue-7 UKS 解多重性家族现象）：FD 在该格**跨分支混算**，不能验证
解析梯度。鉴别测试：闭壳层 benzene FD 3.7e-6 与甲醇 FD 6.5e-6 证明
FD harness 本身正确。结论：UKS methoxy 格的有效梯度证据是同实现
worst-repeat 门（fp32-grad 内核项 1.6e-6）与 auto-verify 路线
（b3lyp 1.58e-6 过门），不是 FD。
（LC-25 补注 2026-10-07：m062x 格的"所有位移都低于中心"图样获得了
更强的机制解释——LC-25 的全矩阵稳定性分类发现 m062x methoxy UKS 的
基准解本身是鞍点（最软本征值 -3.56e-3 Ha，两道同支线），鞍点中心在
任何位移方向都下坡逃逸；级联跟踪到达低 2.628e-5 Ha 的稳定极小。见
`docs/issue-7-uks-stability.md` §7 与
`/data/results/gpu4pyscf-agent/tasks/issue-7/2026-10-07-lc25-stability-matrix/`。
b3lyp/wb97m-v 格的中心解分类见同一证据目录。）

## 5. 诚实声明与剩余问题

1. **继承失败保留**：wb97m-v/methoxy/svp UKS 的能量门失败（-1.283e-6 Ha）
   与其梯度失败格（auto 与 auto-verify 均 3.69e-2 Eh/Bohr，同态分解
   完全归因 SCF 解选择；两支线能量差本身超门）按 2026-10-04 用户策略
   原样保留，根因归入 deferred 的 #7 家族（解多重性）。b3lyp methoxy
   格的 auto 车道失败（1.16e-3）由 **auto-verify 路线修复到 1.58e-6**
   （过门）；UKS 梯度的可用建议更新为：UKS 格优先 auto-verify 或
   fp64 SCF + fp32 梯度内核，wb97m-v/methoxy 格在 #7 家族解决前走
   fp64。
2. **scientific_gate_passed = false**（继承的能量失败未解决；梯度侧
   8/9 auto 车道过门 + b3lyp 失败格经 auto-verify 路线过门）。
   checks_passed 状态见 issue-12-result.json 的 tests 列表（命令 +
   实测 + 退出码）。
3. **历史 attempt-1 文档**（`/data/results/.../issue-12/*.json`）原样
   保留（无 repeats，legacy）；按 G6 语义它们不再满足新验收的
   all-repeat 证据要求——新矩阵文档（2026-10-06-allrepeat-rework/）是
   当前有效证据。
4. **FD 检查的适用域**（§4）：闭壳层格 FD 有效（benzene 3.7e-6、
   methanol 6.5e-6 噪声底）；UKS methoxy 格 FD 跨分支混算、不能用作
   解析梯度验证（诊断证据已存证据目录）。
5. GPU 锁心跳已知 quirk（issue-7 attempt-3 记录过）：持有者 pid 退出后
   heartbeat 拒绝刷新；本轮通过受支持的 fail→start 重试路径接管过期锁
   （事件记录在 tasks.json）。

## 6. 复现命令

```bash
# 全套纯 CPU 回归（含 22 个新 all-repeat 测试）
PYTHONPATH=. /home/qczcode/gpu4090-src/bin/python -m unittest discover -s tools/tests

# 梯度矩阵（每格一条；GPU 一次一个任务）
PYTHONPATH=. /home/qczcode/gpu4090-src/bin/python \
  mixedprec/benchmark/rtx4090/gradient_runner.py \
  --out /data/results/.../b3lyp_benzene_svp_grad.json \
  --geom benzene.xyz --basis def2-svp --xc b3lyp --pilot-stage S1 --samples 3

# auto-verify 复测道（issue-7 诊断路线）
PYTHONPATH=. /home/qczcode/gpu4090-src/bin/python \
  mixedprec/benchmark/rtx4090/gradient_runner.py \
  --out /data/results/.../wb97mv_methoxy_svp_uks_autoverify.json \
  --geom methoxy.xyz --basis def2-svp --xc wb97m-v --samples 3 \
  --lane fp64 auto-verify cpu-fp64

# 独立有限差分检查
PYTHONPATH=. /home/qczcode/gpu4090-src/bin/python \
  /data/tmp/gpu4pyscf-agent/execution/issue12_rework/fd_gradient_check.py \
  --out /data/results/.../fd_methanol_b3lyp.json \
  --geom methanol.xyz --basis def2-svp --xc b3lyp --lanes fp64 auto \
  --atom-idx 0 --h 0.002 --conv-tol 1e-12
```

## 7. LC-30 补遗 (2026-10-07 长链工作单元): LC-29 修复的梯度后果 —— m062x/methoxy/tzvp 鞍点格

梯度矩阵（§2）只含 svp 格；m062x/methoxy/**tzvp** 的梯度从未测过，
而 LC-28 发现该格 auto 车道收敛在**非定态位移点**（鞍点下方
4.09e-8 Ha，沿不稳定模）。LC-29（commit e2a7c6c）把 auto+verify 的
判据升级为有界 fp64 精化交叉重启后，位移点 vs 定态鞍点的梯度后果
需要实测（证据
`tasks/issue-12/2026-10-07-lc30-autoverify-grad-consequence/`，3 格 ×
3 车道 × 3 独立样本，逐 repeat CUDA 同步，预测先行登记）：

| 格 | auto max\|Δg\| | 内核项 | 解项 | auto+verify（LC-29 后） |
|---|---|---|---|---|
| m062x/methoxy/tzvp | **1.952e-4（超 3e-5 门 6.5×）** | 8.8e-7 | **1.951e-4** | **1.090e-6 pass**（same_solution→采纳鞍点） |
| wb97mv/methoxy/svp | 3.691e-2 | 1.4e-6 | 3.691e-2 | 3.691e-2 诚实 fail 保留（distinct→main） |
| b3lyp/methoxy/svp | 1.161e-3 | 1.8e-6 | 1.161e-3 | 1.575e-6 pass（distinct→reference） |

- **新发现（超出 LC-28 的能量记录）**：auto 策略在 tzvp 鞍点格的
  **梯度超门 6.5×**——1.95e-4 Eh/Bohr 完全由解位移项构成（同密度
  fp64 内核梯度对照 8.8e-7，issue-12 归因约定）。即：fp32 头停在
  不稳定模下坡沿上的 4.09e-8 Ha 能量位移，放大为 1.95e-4 的梯度
  误差。旧 auto+verify（采纳同一位移点，LC-28 三样本确定性记录）
  会交付同一 1.95e-4 误差——**LC-29 修复把该格梯度从 fail 恢复到
  1.09e-6（180×），不只是定态性簿记**。
- **无回归**：wb97mv/svp 保持 3.69e-2 诚实 fail（精化确认真分裂，
  采纳精化 auto 支线——支线物理不变，fail 不被掩盖）；b3lyp/svp
  保持 1.58e-6 pass（与 §3 记录逐位一致；本表 1.575e-6）。
- svp 梯度矩阵两格（b3lyp 1.16e-3、wb97mv 3.69e-2 的 auto 行）与
  §2 记录逐位复现——测量通道自校验。
- G1-G5 全过（预登记）：tzvp auto+verify ≤3e-5、wb97mv 不掩盖、
  b3lyp ≤3e-5、全部车道×样本收敛、auto 车道误差完成内核/解分解。

**对矩阵的含义**：m062x/methoxy/tzvp 是梯度矩阵此前未覆盖的格
（svp-only），本单元以诊断语义补测（auto 道 fail、auto+verify
pass）；不可变的 §2 矩阵判定不动。若未来矩阵扩至 tzvp，该格的
auto 车道应预登记为 fail（解位移项）、auto+verify 为过门路线。

复现：`grad_consequence.py`（增量持久化）、`grad_consequence.json`
（全部 repeat 的梯度数组与决策记录）、`runlog.log`、
`evidence_sha256.txt`。

## 8. LC-33 补遗 (2026-10-07 长链工作单元): 旋转型支线族的梯度后果 —— 能量门通过、梯度门失败的实测不对称

§7 只测了位移族（m062x/tzvp）；LC-31 分类出的**旋转型支线族**
两个能量矩阵格（wb97mv/tzvp、b3lyp/tzvp）的梯度从未测过——它们
的 auto 车道收敛在与 fp64 道不同的**定态支线**（能差仅
2.6e-8/4.4e-8 Ha，occ-SVD beta ~0.9997/0.9996，远小于 1e-6 能量门）。
问题：如此小的支线能差对原子核梯度做什么？（证据
`tasks/issue-12/2026-10-07-lc33-rotation-family-grad/`，3 格 × 3
车道 × 3 独立样本，逐 repeat CUDA 同步，预测先行登记于长链
checkpoint，m062x/tzvp 作 LC-30 复现对照。）

| 格 | auto dE（能量门） | auto max\|Δg\| | 内核项 | 解项 | auto+verify |
|---|---|---|---|---|---|
| wb97mv/methoxy/tzvp | +2.60e-8 **pass** | **1.023e-3（超门 34×）** | 8.4e-7 | **1.023e-3** | **1.020e-6 pass** |
| b3lyp/methoxy/tzvp | +4.40e-8 **pass** | **1.225e-3（超门 41×）** | 9.7e-7 | **1.224e-3** | **8.06e-7 pass** |
| m062x/methoxy/tzvp（对照） | -4.09e-8 pass | 1.952e-4 | 8.8e-7 | 1.951e-4 | 1.087e-6 pass |

- **核心发现（能量/梯度门不对称）**：两个旋转型格的 auto 车道
  能量误差比 1e-6 门小 20-40 倍（2.6e-8/4.4e-8 Ha），梯度误差却
  超过 3e-5 门 **34-41 倍**（1.02e-3/1.22e-3 Eh/Bohr），且完全
  由解项构成（内核项 ≤9.7e-7，issue-12 归因约定）。加上 §7 的
  m062x/tzvp 与 §2 的 b3lyp/svp，**全部 4 个已测 UKS 支线格均
  呈现该不对称**：亚 1e-7 的支线能差可携带 1e-4..1e-3 量级的
  梯度差。能量门不能预测 UKS 支线格的梯度质量。
- **dm→梯度放大率的实测规律**：解项 / dm_max_abs_diff =
  0.168（m062x/tzvp）、0.183（wb97mv/tzvp）、0.188（b3lyp/tzvp）、
  0.105（b3lyp/svp）、0.0998（wb97mv/svp）——tzvp 格聚集在
  0.17-0.19，svp 旋转型在 ~0.10。实用启发式：**支线
  dm_max_abs_diff × 0.2 ≈ 最坏梯度解项**；dm > 1.5e-4 的支线格
  即有超 3e-5 梯度门的风险（1.5e-4 × 0.2 = 3e-5）。未来矩阵
  扩展可用该规则预登记。
- **auto+verify（LC-29 语义）恢复两格**：采纳 reference 支线
  （精化 distinct、1 圈、逐位确认 auto 点已是本盆地定态——与
  m062x/tzvp 的 8 圈爬回形成两类情形对照），梯度 1.020e-6 /
  8.06e-7 过门。唯一保留的 auto+verify 梯度 fail 仍是
  wb97mv/svp 的 3.691e-2（交换型支线，支线能差本身 1.28e-6 超
  能量门——物理性 fail，非位移/精化问题）。
- **对照复现**：m062x/tzvp auto 1.952e-4 / auto+verify 1.087e-6
  （LC-30 记录 1.952e-4 / 1.090e-6，预登记带 [1.9,2.0]e-4 /
  [1.0,1.2]e-6 内）——测量通道自校验。
- G1-G5 全过（预登记）：全部车道×样本×格收敛；两个决定格
  auto+verify ≤3e-5、内核项 ≤3e-5；放大率对 0.1-0.2 规律带如实
  报告（0.183/0.188 在带内上沿）；对照在预登记带内。预测 P2
  （b3lyp/tzvp auto+verify 1-3e-6）略偏保守——实测 8.06e-7 低于
  预测带，门级判断正确，如实记录。

**对矩阵的含义**：至此 6 格 methoxy UKS 能量矩阵 × 梯度后果全部
有实测（4 格 auto 梯度 fail、5 格 auto+verify 梯度 pass、
wb97mv/svp 物理性 fail 保留）；不可变的 §2 矩阵判定不动。若
未来梯度矩阵扩至 tzvp，b3lyp/wb97mv 的 auto 车道应预登记为
fail（解项 ~0.1-0.2 × dm），auto+verify 为过门路线。

复现：`grad_consequence.py`（增量持久化；PYTHONPATH 需指向源码
树）、`grad_consequence.json`（全部 repeat 梯度数组 + 支线度量
dm/occ-SVD/精化记录 + 放大率）、`runlog.log`、
`evidence_sha256.txt`。

## 9. LC-39 补遗 (2026-10-07 长链工作单元): 推荐配置下的混合精度梯度车道文档 —— auto/auto-verify 18 份

§2/§8 后推荐配置（16384 分块）下的混合精度梯度文档缺失（issue-12
原始矩阵测于默认配置，早于 LC-22 布线与 LC-36 缓存）。本单元以生产
gradient_runner 补齐：9 格 × {auto, auto-verify} 车道 × 3 独立新进
程样本，逐样本 CUDA 同步，最差 repeat 聚合，fp64 参考取 §LC-38 文档
（证据 `tasks/issue-12/2026-10-07-lc39-mixedprec-grad-lanes/`，18/18
schema-VALID；预测先行登记）。

| 格 | auto max\|dg\| | auto t_grad | av max\|dg\| | av t_grad | fp64 t |
|---|---|---|---|---|---|
| b3lyp/benzene/svp | 1.7e-6 pass | 0.236 | 1.8e-6 | 0.230 | 0.306 |
| b3lyp/methanol/svp | 1.8e-6 pass | 0.184 | 1.8e-6 | 0.130 | 0.173 |
| b3lyp/methoxy/svp UKS | **1.34e-3 FAIL**（旋转支线） | 0.201 | **1.58e-6 恢复** | 0.161 | 0.172 |
| m062x/benzene/svp | 1.2e-6 pass | 0.265 | 1.2e-6 | 0.211 | 0.349 |
| m062x/methoxy/svp UKS | 1.8e-6 pass | 0.226 | 1.8e-6 | 0.189 | 0.246 |
| wb97mv/benzene/svp | 2.0e-6 pass | 1.008 | 2.0e-6 | 1.018 | 1.182 |
| wb97mv/methoxy/svp UKS | **3.00e-2 FAIL** | 0.423 | **1.94e-6**（见下） | 0.446 | 0.449 |
| r2scan3c/benzene/s1 | 1.5e-6 pass | 0.219 | 1.5e-6 | 0.208 | 0.261 |
| r2scan3c/methoxy/s1 UKS | 1.5e-6 pass | 0.196 | 1.5e-6 | 0.164 | 0.156 |

- **16/18 行与预登记预测一致**（含 b3lyp/methoxy auto 1.34e-3 诚实
  fail → auto-verify 1.58e-6 恢复，复现 LC-30 的 1.575e-6 记录）。
- **预登记偏差（重要，如实记录）**：wb97mv/methoxy 的 auto-verify
  预测为保留 3.69e-2 诚实 fail（默认配置物理），实测 **1.94e-6
  pass**。机制（stability_verification 三 repeat 逐位一致）：16384
  分块把 auto 头的收敛支线移动到交换族第三解（occ-SVD beta 0.4975，
  dm 0.298，dE_ref−main −4.25e-7 —— LC-17 记录过的 auto-16384 支线
  移位）；verify 检出 distinct 分裂后**采纳 fp64 道鞍点**（e_tot 与
  LC-38 fp64 道逐位相同）→ 梯度 1.94e-6。即 LC-29 判据机器在移动了
  的支线上按设计工作：采纳的是定态参考支线。auto 车道本身 3.00e-2
  fail 保留（G3 不掩盖成立——fail 的是 auto，不是 verify）。
- **auto 车道计时与 LC-34 组成事实一致**：GGA/MGGA 格 auto 快于
  fp64（1.1-1.7×），但 UKS 支线格的梯度正确性约束不变（本表两格
  auto fail）。auto-verify 的开销被缓存/分块部分吸收（触发格付参照
  道：wb97mv/methoxy av 0.446 vs auto 0.423 = 1.05×—— 远低于
  LC-31 默认配置记录的 1.97×）。
- 至此推荐配置下 9 格 × {fp64, auto, auto-verify} = 27 份梯度文档
  齐备；不可变的 issue-12 原矩阵判定不动。

复现：`run_matrix.sh`、18 组 `*.json/*.raw.json/*.runlog.log`（runner
自锚定）、`evidence_sha256.txt`。

## 10. LC-57 补遗 (2026-10-07 长链工作单元): auto 道 dg 尺寸增长第三点 —— tamoxifen/687 AO 饱和 + 3c verify 参照道 D4+gCP 遗漏修复

LC-40 标记的唯一可测遗留项（auto 道 max|dg| 随尺寸增长，"须重测、
不得外推"）在下一个尺寸台阶直接测掉：**r2scan3c / 057_Tamoxifen /
def2-mtzvpp（687 AO，闭壳层）**，同一生产 runner、推荐配置
（16384 分块）、{fp64, auto, auto-verify} 3 道 × 3 新进程样本
（证据 `tasks/issue-12/2026-10-07-lc57-dg-growth-next-size-step/`，
预测 P1-P7 先行登记于 checkpoint lc57_preregistration）。

**r2scan3c auto 道尺寸序列（同泛函同基组族、全闭壳层）**：

| 体系 | AO | auto max\|dg\| | 判定 |
|---|---|---|---|
| benzene/mtzvpp | 144 | 1.5e-6 | pass |
| vitaminC/mtzvpp | 298 | 9.9e-6 | pass |
| **tamoxifen/mtzvpp** | **687** | **1.06e-5** | **pass**（门 3e-5，2.8x 余量） |

**核心结论：增长饱和。** 298→687 AO（2.31x 尺寸）max|dg| 仅
9.9e-6→1.06e-5（+7%）；线性外推（~2.3e-5）与幂律外推（~1.0e-4）
双双被实测否证——144→298 的表观陡增并不延续。dE(auto) −1.8e-12，
三 repeat 全收敛，auto-verify 不触发（闭壳层唯一极小，dm-diff
1.78e-6），梯度相 auto/fp64 = 5.19/1.16 s = **4.47x**（序列
3.17x @298 → 4.47x @687，LC-17 机制随 naux 增长的判断再获支持），
整道 auto 16.2 s vs fp64 23.2 s = **1.44x**；峰值显存 6.95 GB
（预算 21.6 GB 内）。P1（单调增长）HIT 但幅度远低于两个外推锚点
（外推否证即本单元目的）；P3/P4/P5 HIT；P6（≥1.5x 梯度相）HIT
（4.47x）；P7 墙钟 5 分钟远低于预算。

**同单元发现并修复的潜伏缺陷（LC-57b，代码 commit 同节）**：
auto+verify 的 fp64 参照道（`stability_verify._build_fp64_sibling`）
不携带实例级色散钩子——r2SCAN-3c 的 D4+gCP 是 runner 在最终对象上
的 `MethodType` 挂载，兄弟对象重建时不被复制。实测后果：**所有
3c verify 记录的 `ref_e_tot` 恰好低 D4+gCP 之和**（benzene/vitaminC/
tamoxifen：−6.55e-4/−7.52e-3/−6.55e-4 Ha 量级——本单元分析
tamoxifen 记录时定位，回溯 vitaminC 同病）。爆炸半径已澄清：全部
已有 3c verify 记录均不触发（distinct=false，能量采纳分支从未进入），
`e_terms` 走专用键不受影响，**无任何已报告文档数字错误**；但任何
未来触发的 3c 格会在采纳分支写出色散缺失的 e_tot。修复：兄弟对象
复制 `get_dispersion`/`do_disp` 实例属性（MethodType 重绑到兄弟自身，
scf_summary 记录对象正确）；修复后 tamoxifen 单格复测 ref−main =
+4.8e-10（修复前 −6.55e-4），新增 2 个回归测试（钩子携带 +
重绑语义），stability 14/14、tools 441、df-ip1 11、grid/loop/int2c2e
37 全绿。修复后记录如实说明：LC-57 基准 3 样本文档测于修复前
commit（44abcae），其科学判定不受影响（不触发路径），fixcheck
文档单列。

复现：`run_gpu_cell.sh`（3 道基准）、`verify_fix_check.sh`（修复
验证单道）、9 个证据文件 sha256 锚定（`evidence_sha256.txt`）。
