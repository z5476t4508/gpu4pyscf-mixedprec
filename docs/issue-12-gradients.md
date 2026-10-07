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
