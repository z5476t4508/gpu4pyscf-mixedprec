# 待用户决策事项清单（决策支持简报）

**版本**: 2026-10-07（LC-58 增补版）· **分支**: zcode/lc1-uks-stability-hessian（fe46813）
**性质**: 决策支持文档。本文**不做任何决策**——只汇总每个待决事项的已测
证据、默认翻转的确切机理、数值类别区分与注意事项，供用户选择。所有数字
可溯源到证据目录或文档章节（来源括注）。

事项清单：(a) 发布通道 65>50 文件阻塞；(b1) SCF/梯度 XC 格点分块 16384
默认开；(b2) ip1 融合启动默认开；(c) UKS 1e-6 能量门的重新表述。
（数字均经 LC-50 跨文档数字审计对 primary 证据复核，LC-57/58 增补数字
为 LC-57/58 会话实测；见各节括注。）

**LC-57/58/60 增补（与决策直接相关的新证据）**：auto 道在程序内最大梯度格
（tamoxifen/mtzvpp 687 AO）梯度相 4.47x、车道端到端 1.44x，且 auto 道
max|dg| 随尺寸**饱和**（687 AO 1.06e-5，较 298 AO 仅 +7%；线性与幂律
外推均被实测否证；3e-5 门 2.8 倍裕量）——(b1)/(b2) 的推荐栈在最大格
仍端到端受益，外推类担忧在 S2 包络内不再成立。auto+verify 的 verify
兄弟对象完整性缺陷两处修复（3c 色散钩子 60de171；原生 mf.disp/nlc
状态 fe46813——修复前 mf.disp 静默错 ref +5.74e-4 Ha、mf.nlc='vv10'
硬崩溃；既有记录全部 no-fire，零已报告数字错），并有四族能量平价回归
测试常驻守护——(c) 中 auto+verify 的判据记录可靠性由此加固。LC-60：
LC-54 根因的 cusolver 跨版本句柄错配已在仓库侧关闭（提交 6a80ac9：
加载器改为配对 cupy 已映射的库；无 shim 下 dftu/gks/EDA/scf FULL 全
绿）——环境 shim 从"必需"降级为"可选"；另修复 cuTensor 预热线程的
罕见退出期竞态（0/100 复现原 2-7%/进程）。

---

## (a) 发布通道阻塞：累计 diff 67 文件 > 出版方 MAX_FILES=50

**现状（LC-60 实测，2026-10-07 @ 6a80ac9）**：`git diff --name-only
0a4cf33..HEAD` = **67 文件**（LC-50 时点 65：LC-58 增补 decision-brief/
ledger 均为已跟踪文件，计数未变；LC-60 的 cusolver 修复带入 2 个新文件
lib/cusolver.py + lib/tests/test_cusolver.py——为关闭 LC-54 根因的实配库
错配（用户可达崩溃面）所必需，非凑数文件）。LC-7 时 53 文件、LC-44 时点
64 文件——每轮新增使计数递增。
所有 issue 清单都是累计 diff upstream-base..HEAD，故任何新清单都超
50 文件界。LC-7 已尝试受支持的清单发射并被拒（`error: 53 files
exceeds the 50-file publication limit`，exit 2；过程记录
`state/long-chain/phases/LC-7-2026-10-06-manifest-blocked.md`）。

**选项（记录未选，出自 LC-7）**：

| 选项 | 内容 | 归属 |
|---|---|---|
| a1 | 提高 MAX_FILES（出版方 + 计划文档 §6.1 修改，用户拥有） | 用户 |
| a2 | 每轮增量清单（base = 上一清单 head；需出版方确认叠层语义） | 用户+出版方 |
| a3 | 仓库文件合并（如三个薄 run_*.sh 包装）——仅在自身合理时做，不为凑界而做 | 可讨论 |

**未阻塞项**：源码、补丁、证据均完整保存（`manifest-history/` + 各
evidence 目录）；issue-12 的 47 文件清单仍是最近可发布工件。开发不
因此停（现行策略）。

## (b1) `GPU4PYSCF_XC_GRID_BLOCKSIZE=16384` 默认开（LC-11/22 落地，默认关）

**默认翻转机理（本会话核实）**：唯一改动点是
`gpu4pyscf/dft/numint.py::_scf_grid_blksize()` 的默认值（MIN_BLK_SIZE
4096 → 16384）。该函数被 `_nr_rks_task`/`_nr_uks_task`（SCF XC）、
`nr_nlc_vxc`（VV10/NLC，numint.py:1830-1832）与梯度侧
`grad/rks.py:168`/`grad/uks.py`（LC-22 布线）共同调用——**一处翻转，
SCF+NLC+梯度三路同时生效**。env 覆盖保留（显式设
`GPU4PYSCF_XC_GRID_BLOCKSIZE` 仍优先生效，含回退到 4096 的能力）。

**实测效应（全部有据）**：

| 路径 | 效应 | 来源 |
|---|---|---|
| nr_rks 格点循环（探针） | 1.26–3.76x（最优/cell 1.57–3.76x；MGGA 最小） | LC-11，issue-9 §9 |
| 整 SCF（小格 43-114 AO） | 1.07–1.16x | LC-11 |
| 冷启动单点作业（合并栈） | 1.26–1.29x | LC-14 |
| 整梯度（4 格） | 1.24-1.27x，同轨道 dg≤1.4e-13 | LC-22 |
| 梯度组成（auto 道） | auto 首次明确快于 fp64（m062x 0.206 vs 0.350s） | LC-34 |

**数值类别（关键区分）**：
1. **同轨道/归约序变化**：~1e-13 Ha 量级（LC-11 实测 dE≤2.3e-13、
   dExcsum≤5e-14；LC-22 dg≤1.4e-13）——与跨进程 fresh-SCF 非确定性
   同类，距 1e-6 门 6 个量级。
2. **UKS 支线选择**（LC-39 实测，如实保留）：16384 使 wb97mv/methoxy
   auto 头收敛到**交换族第三解**（occ-SVD beta 0.4975，dE_ref−main
   −4.25e-7）——auto 道的支线选择**随配置可变**；auto-verify 对该
   位移稳健（检出分裂后采纳 fp64 道鞍点，1.94e-6 pass）。该格 auto
   车道 fail 保留未掩盖。

**注意事项**：
- 峰值显存 +≤60 MiB（LC-11），梯度相在工作预算内（LC-40: 0.69-2.35
  GB vs 21.6 GB 准入预算）。
- 若默认开，基准可比性：与默认配置历史文档的跨配置比较需标注（本仓
  文档已分默认/推荐两轨）。

## (b2) `GPU4PYSCF_DF_IP1_FUSED=1` 默认开（LC-42 落地，默认关）

**默认翻转机理（本会话核实）**：唯一改动点是
`gpu4pyscf/df/grad/rhf.py::_ip1_fused()` 的默认值；`df/grad/uhf.py`
从 rhf 导入同一函数（uhf.py:25）——一处翻转，RHF/RKS 与 UHF/UKS
两路同时生效。仅作用于 J+K（杂化泛函）梯度路径的
`_jk_energy_per_atom`；J-only（纯泛函/r2scan3c）与核吸引路径按设计
不触及。内存守卫（全宽压缩缓冲 < 可用显存 25%）保留，守卫失败逐字
节回退上游。env 覆盖保留。

**实测效应（三证据层级，全部有据）**：

| 层级 | 效应 | 来源 |
|---|---|---|
| 同进程同轨道 A/B（5 样本中位） | fp64 整梯度 **1.08-1.48x**（benzene/svp b3lyp 1.48x、tzvp m062x 1.37x、wb97mv/甲氧基 UKS 最低 1.08x）；auto 1.01–1.15x；dg 4.4e-15..2.9e-13 | LC-42 |
| 文档级同会话配对对照 | b3lyp/svp 0.818→0.705s = 1.16x | LC-43 |
| 中等尺寸配对同会话（222-298 AO） | fp64 梯度相 **1.31–1.55x**（b3lyp 1.53x、m062x 1.55x、wb97mv 1.31x）；**效应随体系增大**（naux 558 vs svp ~204）；J-only 哨兵 vitaminC 1.00–1.02x 无操作确认 | LC-44（本日） |

**数值类别**：融合改变跨块原子交错顺序——同轨道 dg ≤ 2.9e-13
（LC-42 全部 10 车道格实测下限 4.4e-15，原子序噪声级）；跨进程
fresh-SCF 对照 dg ≤ 1.26e-8（LC-44，为 fresh-SCF 自身非确定性，
非旋钮效应——J-only 哨兵同量级）。

**注意事项**：
- r2scan3c 两格走 J-only 路径，旋钮无效应（设计使然，LC-43/44 哨兵
  双重确认）；默认开对它们是纯无操作。
- LC-43 记录了同日跨会话 ~16% 机器状态漂移；上表全部取自**同会话配
  对**对照，不受漂移污染。

## (c) UKS 1e-6 能量门重新表述（issue-7/-12 继承 fail 的处置）

**物理事实（全部实测，多源交叉）**：methoxy/def2-svp UKS wb97m-v 的
auto 道 −1.283e-6 Ha 不是精度不足——是**真实存在的 UKS 解多重性**：
两条确定性轨迹收敛到能量差恰为 1.28e-6 Ha 的不同定态解（fp64 道鞍
点 −1.670e-5 Ha NLC 响应 vs auto 道稳定极小 +1.273e-5，LC-1； occ-SVD
beta 0.2576 单轨道差；交叉重启双向保盆地，issue-7 §2）。梯度后果
（LC-30/33）：能量门 pass（2.6-4.4e-8）不预测梯度质量——同密度
fp64 核梯度差 3.7e-2 Eh/Bohr；auto+verify 恢复 1.0-1.6e-6。

**三个可选表述（记录未排序）**：

| 选项 | 内容 | 已测依据 |
|---|---|---|
| c1 | 维持现状：UKS 精确匹配路线 = fp64 SCF（可选 fp32 梯度内核，实测 ≤2.4e-6 Eh/Bohr）；auto/auto+verify fail 如实保留 | issue-7 §4.1 |
| c2 | 把 UKS 验收门拆成"与所选定态解一致"（如 auto+verify 的判据记录）而非"与 fp64 道一致" | LC-29 机器已提供可审计记录；但"选择更低解"在 -1.28e-6 格上仍超门（除非门本身改写） |
| c3 | 收紧为 auto+verify 默认 + 门内报告解选择记录 | 实测开销 1.05x（推荐配置，LC-39）至 1.97–2.64x（默认配置，LC-31） |

**不可协商项（用户验收判据原文）**：选不同低能 UKS 支线**不满足**匹配
FP64 解的门（AGENTS.md：selecting a different lower-energy UKS branch
does not satisfy matching FP64 solution gates）。任何表述变化都是用户
政策决定，需显式改写验收判据；本仓不自行改门。

---

## 决策请求汇总

| 事项 | 需要用户给出 | 阻塞内容 |
|---|---|---|
| (a) | a1/a2/a3 之一或指示暂缓 | 新清单发射（任何 issue） |
| (b1) | 默认开/保持 opt-in | 无（纯收益类，等政策） |
| (b2) | 默认开/保持 opt-in | 无（纯收益类，等政策） |
| (c) | c1/c2/c3 之一或维持 | issue-7/-12 继承 fail 的最终表述 |

**源码/证据状态**：分支 LC-1..LC-60 全部自检完成；两旋钮默认关、env
可覆盖、可随时翻转；全套件绿（tools 441、df-mixedprec 18、
stability-verify 18（含四族能量平价审计）、dft rks 16 + uks 10、
xc-grid/loop/int2c2e 37、df-ip1 11，LC-58/60 在 HEAD 重验证；无 shim
下 scf FULL 120+3 skip+1 PCM env-fail、EDA 12+1 skip、dftu/gks 6+7 skip）。
LC-1..LC-60 待独立 Codex 审查（不自行批准）。

---

## 附录：本分支已交付的运行时旋钮参考表

四个环境变量旋钮（LC-36 的 Int2c2eOpt 缓存无环境门，恒定启用；本表
每行均在本会话于源码核实）：

| 旋钮 | 默认 | 合法值 | 作用（一处生效点） | 证据 |
|---|---|---|---|---|
| `GPU4PYSCF_XC_GRID_BLOCKSIZE` | 4096（MIN_BLK_SIZE，即上游默认路径） | MIN_BLK_SIZE 的正倍数（非法值告警并回退 4096） | 重划 SCF XC/`nr_nlc_vxc`/梯度 XC 循环的格点分块（`numint._scf_grid_blksize()` 一处，三路共用） | LC-11/22/34/39；§(b1) |
| `GPU4PYSCF_DF_IP1_FUSED` | 关 | `1`（其它非空值告警并忽略） | J+K 梯度路径 ip1 内核融合启动（`df/grad/rhf.py::_ip1_fused()`，uhf 导入共用；J-only/核吸引不触及；内存守卫失败逐字节回退） | LC-42/43/44；§(b2) |
| `GPU4PYSCF_VV10_UWE_KERNEL` | `warp1`（参考内核） | `warp1|warp4|tile8|tile16|tile32`（未知值告警并用 warp1） | VV10 UWE 内核变体选择（`lib/gdft/vv10.cu`，进程级一次读取；SCF 与梯度同入口） | issue-10/LC-3：tile8 仅小格点（≤~60k）受益，≥128k 全变体同顶 |
| `GPU4PYSCF_CUTENSOR_WARM` | **开**（opt-out） | `0/false/False` 关闭 | 导入期守护线程预热 cuTensor 句柄（`lib/cutensor.py` 末尾；锁串行化防双句柄） | LC-13/14 |

组合建议（实测组合即"推荐配置"）：`16384 + DF_IP1_FUSED=1` 为基准
矩阵文档的推荐栈（LC-38..44 全部 27+ 份梯度文档与 4 份中等尺寸能量
文档在此栈测得）；VV10 内核变体在 benzene 尺寸及以上保持默认 warp1
（实测同顶或更差）；cutensor 预热保持默认开。
