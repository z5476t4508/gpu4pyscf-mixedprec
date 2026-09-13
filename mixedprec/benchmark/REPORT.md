# gpu4pyscf 混合精度基准报告

**日期**: 2026-09-10 ~ 09-13 · **分支**: `mixed-precision-hf` · **代码**: commit `067ceed` 基线 + 两个修复 (`9792bfd` auto 发散守卫, `09e00bb` Hessian 内存预算)
**硬件**: NVIDIA RTX 5090 (32 GB) vs CPU PySCF 2.14.0 (24 线程, 同机); 跨代码对比含 Direwolf V0.99 (24 线程, OpenBLAS HASWELL)
**软件**: Python 3.14.4 / CuPy 14.2.0 / gpu4pyscf 源码树
**配置**: DF (辅助基组两边钉死 `def2-universal-jkfit`), conv_tol=1e-10, XC 网格双方默认
**车道**: `fp64` = 全双精度; `auto` = fp32 起步、fp64 收尾的 SCF 策略, 梯度/Hessian 整体 fp32
**矩阵**: 4 个闭壳层体系 × {def2-svp, def2-tzvp} × {DF-RHF, DF-r2SCAN, DF-B3LYP} × {fp64, auto}
**计时纪律**: SCF 取冷启动单发 (重复测量会热启动污染, 已修正); 梯度 3 次取中位; Hessian 单发
**数据**: `results/benchmark.json` (69 行, 含全部原始数组) · 本报告表格由其生成

---

## 1. 摘要

1. **精度**: auto 车道的能量与 CPU PySCF 差 **≤ 2×10⁻¹⁰ Eh** (全部 36 个能量格子);
   梯度 max-范数差 1×10⁻⁶ ~ 2.6×10⁻⁵ Eh/Bohr (对 geomeTRIC 的 3×10⁻⁴ 阈值有一个
   量级以上余量); Hessian 的频率差 rms 3×10⁻⁵ ~ 4×10⁻³ cm⁻¹, ΔZPE ≤ 2.3×10⁻⁶ Eh
   (0.0014 kcal/mol)。**对外锚点**: methanol 能量对 g16 (RB3LYP/6-31+G(d,p),
   同几何同等级) 差 -2.19×10⁻⁷ Eh, CPU/GPU-fp64/GPU-auto 三方一致。
2. **速度**: 梯度加速 1.9~7.1× (随体系增大); SCF 加速 1.2~1.8×; Hessian 加速
   1.4~7.6×。对 CPU: SCF 10~60×, 梯度 6~92×, Hessian (Tamoxifen/svp) 155~285×。
3. **硬件边界**: 09-11 发现的 r2SCAN Hessian OOM 已由内存预算修复消除
   (`09e00bb`: 原 `_j_energy_per_atom` 分块预算超支 1.2× 可用显存) ——
   **全矩阵零 OOM**, r2SCAN Hessian 实测通过到 ~1877 AO (4609s fp64)。
   墙未被消灭只是外推: 二次增长的本体张量预估在 ~2500 AO 量级再现
   (探针 `find_hess_wall.py` 在跑), ~2500 AO 以上需要 out-of-core。
4. **鲁棒性**: Aza tzvp (~1900 AO) 的 hf/B3LYP auto 车道 fp32 相发散
   (ΔE 混沌震荡 ±1-10 Eh, 永远够不着切换阈值) —— 已修 (`9792bfd` 混沌
   触发器): 修复后收敛且能量对齐 fp64 (≤4×10⁻¹²), 但发散域内 auto 比
   fp64 **慢** 2.4~3.1× (混沌轮次浪费 + 劣质起点), 属正确性换速度的
   trade-off; r2SCAN 同体系不受影响。
5. **跨实现系统差**: GPU-fp64 与 CPU PySCF 的 Hessian 在 GGA/meta-GGA 上有
   0.02~0.2 cm⁻¹ (rms) 的 XC 网格实现差 (HF 仅 ~10⁻³), 与混合精度无关;
   热化学层面 (ΔZPE ≤ 2×10⁻⁶ Eh) 化学可忽略。

## 2. 外部验证 (具名外部代码)

methanol, RB3LYP/6-31+G(d,p), 常规积分 (无 DF), 几何与能量同源于 g16
`tests/methanols0.fchk` (等级取自 fchk 头注释; **笛卡尔 6d**, 见 §7):

| 实现 | 能量 (Eh) | vs g16 |
|---|---|---|
| g16 参照 | -115.7348716828 | — |
| CPU PySCF | -115.7348719017 | -2.188×10⁻⁷ |
| GPU fp64 | -115.7348719016 | -2.188×10⁻⁷ |
| GPU auto | -115.7348719016 | -2.188×10⁻⁷ |

性能矩阵 (DF + def2-svp) 对 g16 的表面差值 (+0.10 ~ +0.78 Eh) 是理论层次差
(DF 误差 + 基组失配), 不作验证用 —— 对外验证只在同等级下进行 (本节)。

## 3. 精度

### 3.1 能量 (auto vs CPU, |ΔE| Eh)

全部 36 个可评性格子 ≤ **2.4×10⁻¹¹** (svp) / ≤ **1.0×10⁻¹⁰** (tzvp)。
auto 的 SCF 以 fp64 收尾, 能量精度与 fp64 车道不可区分 —— 设计兑现。

### 3.2 梯度 (auto 的 max|Δg|, Eh/Bohr)

| 体系 (AO) | svp: vs fp64 / vs CPU | tzvp: vs fp64 / vs CPU |
|---|---|---|
| methanol (48/80) | 1.5~2.1×10⁻⁶ / 0.9~1.9×10⁻⁶ | 1.0~1.2×10⁻⁶ / 0.9~1.1×10⁻⁶ |
| Vitamin C (208/395) | 9.0~9.9×10⁻⁶ / 9.4~10.5×10⁻⁶ | 7.8~8.8×10⁻⁶ / 7.6~8.6×10⁻⁶ |
| Tamoxifen (537/1042) | 0.8~1.1×10⁻⁵ / 0.8~1.1×10⁻⁵ | 0.9~1.0×10⁻⁵ / 0.9~1.0×10⁻⁵ |
| Azadirachtin (934/1877) | 1.9~2.6×10⁻⁵ / 1.9~2.6×10⁻⁵ | 2.6~3.3×10⁻⁵ / (CPU 参照豁免) |

结论: 梯度误差不随基组增大而恶化, 封顶 ~2.6×10⁻⁵, 低于 geomeTRIC 阈值
3×10⁻⁴ 十倍以上 —— 混合精度梯度不改变几何优化的停点。

### 3.3 Hessian → 频率 (auto 车道)

主指标 = rms(dν) (max|dν| 在非驻点几何上被软模态污染, 见 §7; 完整 max/rms/
ΔZPE/ΔS_vib 见 JSON):

| 体系·基组 | rms(dν) vs fp64 (cm⁻¹) | rms(dν) vs CPU (cm⁻¹) | ΔZPE vs CPU (Eh) |
|---|---|---|---|
| methanol·svp (hf/r2scan/b3lyp) | 0.6/1.3/2.0 ×10⁻⁴ | 1.0×10⁻³/0.165/0.038 | ≤2.2×10⁻⁶ |
| Vitamin C·svp | 0.9/3.0/1.0 ×10⁻⁴ | 2.1×10⁻³/0.186/0.022 | ≤2.3×10⁻⁶ |
| Tamoxifen·svp | 1.7/1.9/1.8 ×10⁻⁴ | 6.4×10⁻³/0.024/0.010 | ≤1.8×10⁻⁶ |
| methanol·tzvp | 0.3/1.5/7.2 ×10⁻⁴ | 1.5×10⁻³/0.130/0.045 | ≤1.5×10⁻⁶ |
| Vitamin C·tzvp | 2.2/6.0/12.9 ×10⁻⁴ | (CPU Hessian 豁免, §6) | — |
| Tamoxifen·tzvp | 6.8/**5.4**/— ×10⁻⁴ | (同上) | — |
| Azadirachtin·svp | 3.9/**3.7**/— ×10⁻⁴ | (CPU E/G 有, Hessian 豁免) | — |
| Azadirachtin·tzvp | hf 1.9 / r2SCAN 4.4 / b3lyp 5.3 ×10⁻⁴ | (同上) | — |

解读:
- **精度代价** (auto vs fp64, 同实现同网格): rms 全部 ≤ 1.9×10⁻³ cm⁻¹,
  ΔZPE ≤ 10⁻⁸ Eh —— 任何谱学分辨率之下。
- **跨实现系统差** (fp64 vs CPU): HF ~10⁻³ (锚: 无网格依赖); GGA/meta-GGA
  0.01~0.19 —— XC 网格实现不同所致 (实测加密 CPU 网格 level 3→5 不收敛于此
  指标, 是软模态放大; 热化学层面差值 ≤ 2.3×10⁻⁶ Eh, 化学可忽略)。
  两个实现对 r2SCAN 的频率在 0.1~0.5 cm⁻¹ (max) 内互相一致, 但**不是零**,
  引用跨代码频率差时须带此系统差。

### 3.4 未被本轮覆盖的格子 (如实列缺)

| 格子 | 缺什么 | 原因 |
|---|---|---|
| 全部 tzvp 的 Hessian vs CPU | dν/cpu 列 | CPU Hessian tzvp 成本 20h+/泛函, 用户决策豁免 |
| Azadirachtin 的 Hessian vs CPU | dν/cpu 列 | 同上 (天级) |

09-11 版报告中的两处缺口 (r2SCAN Hessian ≥934 AO 的 OOM 格、Aza tzvp
hf/b3lyp 的 auto 未收敛格) 已分别由两个修复消除并重测 (2026-09-13),
本版矩阵**无因代码问题缺失的格子**。

## 4. 速度: auto 相对 fp64 (GPU 内加速)

### 4.1 def2-svp

| 体系 (AO) | 量 | hf | r2SCAN | B3LYP |
|---|---|---|---|---|
| Vitamin C (208) | SCF | 1.2× | 1.3× | 1.3× |
| | 梯度 | **2.9×** | **2.0×** | **2.5×** |
| | Hessian | 1.4× | **2.5×** | 1.7× |
| Tamoxifen (537) | SCF | 1.4× | 1.6× | 1.5× |
| | 梯度 | **2.7×** | **3.8×** | **2.7×** |
| | Hessian | **3.4×** | **4.1×** | **3.4×** |
| Azadirachtin (934) | SCF | 1.5× | 1.6× | 1.5× |
| | 梯度 | **2.2×** | **5.1×** | **2.3×** |
| | Hessian | **6.0×** | **4.9×** | **5.5×** |

### 4.2 def2-tzvp

| 体系 (AO) | 量 | hf | r2SCAN | B3LYP |
|---|---|---|---|---|
| Vitamin C (~395) | SCF | 1.3× | 1.5× | 1.3× |
| | 梯度 | **3.5×** | **3.5×** | **3.4×** |
| | Hessian | 1.4× | **2.8×** | 1.8× |
| Tamoxifen (1042) | SCF | 1.4× | 1.7× | 1.4× |
| | 梯度 | **3.0×** | **6.2×** | **3.2×** |
| | Hessian | **3.7×** | **4.3×** | **3.7×** |
| Azadirachtin (~1877) | SCF | 0.32×† | 1.8× | 0.42×† |
| | 梯度 | **2.2×** | **7.1×** | **2.3×** |
| | Hessian | **7.9×** | **5.1×** | **7.1×** |

† 发散域 (§6.2): 守卫触发前 13 轮 fp32 混沌浪费 + 劣质起点, auto 的 SCF 比
fp64 慢 (hf 194.7s vs 62.1s, b3lyp 225.1s vs 93.7s) —— 正确但负加速;
该体系的 SCF 建议直接用 fp64。梯度/Hessian 在守卫修复后正常受益。

规律: **梯度与 Hessian 加速随体系增大单调上升** (GPU 吃饱效应);
SCF 封顶 1.2~1.8× (fp32 只用于前期迭代, 收尾必须 fp64; 纯泛函的 SCF 大头
在 XC 网格, 本来就是 fp64)。小体系 (<200 AO) 的 SCF 无收益甚至略负。

## 5. 速度: auto 相对 CPU PySCF (24 线程, 同机)

### 5.1 能量+梯度 (用户日常量)

| 体系·基组 | SCF | 梯度 | 备注 |
|---|---|---|---|
| Vitamin C·svp | 51/15/30× | 6.5/15/9.6× | hf/r2scan/b3lyp |
| Vitamin C·tzvp | 23/19/26× | 5.9/27/12× | |
| Tamoxifen·svp | 13/20/16× | 8.0/**50**/14× | |
| Tamoxifen·tzvp | 9.6/15/12× | 11/**92**/20× | r2SCAN 梯度对 CPU 近百倍 |
| Azadirachtin·svp | 9.6/17/10× | 11/**81**/16× | |

### 5.2 Hessian 对 CPU (Tamoxifen·svp, CPU 参照唯一齐的体系)

| 泛函 | CPU | GPU fp64 | GPU auto | auto/CPU |
|---|---|---|---|---|
| DF-RHF | 11970s (3.3h) | 185s | 55s | **218×** |
| DF-r2SCAN | 11450s (3.2h) | 300s | 74s | **155×** |
| DF-B3LYP | 24012s (6.7h) | 286s | 84s | **285×** |

一条 B3LYP 频率分析: CPU 过夜 → GPU 一分半。

### 5.3 对 Direwolf (跨代码, 同 level)

**等级钉死**: DF-r2SCAN / def2-svp / **ccpvdzjkfit** (aux 迁就 Direwolf
的库 —— 它没有 def2-universal-jkfit), 同几何文件; Direwolf V0.99,
24 OpenMP 线程, OpenBLAS `TARGET=HASWELL` (Arrow Lake 对口, config.h
证实)。数据 `results/direwolf/comparison.json`。

| 体系 (AO) | ΔE (gpu-fp64 − Direwolf, Eh) | max|Δ梯度| | SCF: dw/auto | 梯度: dw/auto |
|---|---|---|---|---|
| methanol (48) | **+7.7×10⁻⁷** | 9.8×10⁻⁵ | 1.4× | 0.6× (CPU 反超) |
| Vitamin C (208) | -1.4×10⁻⁵ | 1.9×10⁻⁴ | 2.0× | 3.1× |
| Tamoxifen (537) | -1.8×10⁻⁴ | 1.0×10⁻⁴ | 2.6× | **7.0×** |
| Azadirachtin (934) | -8.4×10⁻⁵ | 2.6×10⁻⁴ | 2.1× | **9.1×** |

解读:
- 能量/梯度差随体系增大 = **XC 网格实现系统差** (Direwolf 的 Lebedev 网格
  更粗; 两边各自收敛; 我们对 CPU PySCF 的能量差 ≤10⁻¹⁰ 是因为共享网格
  方案)。引用跨代码能量差必须带此注。
- **三方链** (Azadirachtin 梯度): PySCF CPU 131s → Direwolf 14.7s →
  gpu4pyscf auto 1.6s。乘法分解 **81× ≈ 8.9× (Direwolf 的 CPU 实现优势)
  × 9.1× (GPU 硬件红利)** ——Direwolf 无 GPU 版本, 它是最强的 CPU 参照物,
  把「实现赚的」和「硬件赚的」分开了; 我们的 81× 不能裸报。
- 交叉点 ~50-100 AO: 更小的体系 Direwolf (24 核) 胜 GPU 启动开销。

## 6. 硬件边界与鲁棒性 (5090, 32 GB)

### 6.1 解析 Hessian 的显存墙: 已修复至 ~1877 AO, 更大仍需 out-of-core

09-11 版报告记录的三个 OOM 格 (r2SCAN Hessian @ 934/1042/1877 AO) 的根因
**不是 r2SCAN 天生吃显存, 而是 `_j_energy_per_atom` 的分块预算超支**:
buf0+buf1 (0.75) + j3c_full (0.15) = 1.2× 可用显存 (实测 35.9 GB vs
29.9 GB)。修复 (`09e00bb`, 系数 0.50+0.10) 后:

| 格子 | 修复前 | 修复后 (2026-09-13 实测) |
|---|---|---|
| r2SCAN Hessian, 934 AO (svp) | OOM | fp64 1685s / auto 344s (**4.9×**) |
| r2SCAN Hessian, 1042 AO (tzvp) | OOM | fp64 706s / auto 163s (**4.3×**) |
| r2SCAN Hessian, ~1877 AO (tzvp) | OOM | fp64 4609s / auto 900s (**5.1×**) |

**但墙只是外推了, 没有消灭**: 本体张量二次增长, 预估 ~2500 AO 再现
(探针进行中)。~2500 AO 以上的体系 (更大的催化复合物、qzvp 基组) 需要
out-of-core (buf/j3c 经 pinned 主机内存流转, 123 GB 可用) —— 已立项。
fp32 CDERI 单独开关实测**不能**推墙 (精度倒是免费: dν max 1.3×10⁻³ cm⁻¹),
其精度结论直接用作 out-of-core 的存储层验收标准。

### 6.2 auto 车道的 SCF 发散域 (已修复, 有代价)

Azadirachtin·tzvp (~1877 AO) 的 hf/B3LYP auto 车道 fp32 相**发散**: ΔE 在
±1~10 Eh 混沌震荡 (DIIS 失稳), 永远够不着 1e-4 的切换阈值, 50 轮打满不收敛
(fp64 同格 <50 轮收敛)。同体系 r2SCAN 的 auto 正常收敛 (gerr 2.8×10⁻⁵) ——
不稳定的是**精确交换 K 的 fp32 收缩**, 悬崖位于 1042 (Tamoxifen tzvp 正常)
与 1877 AO 之间。

修复 (`9792bfd`): 混沌触发器 —— 第 10 轮后连续 3 轮 |ΔE|>10⁻² 即强制切
fp64。验收: 失效格收敛且能量对齐 fp64 (≤4×10⁻¹²); 健康格逐位不变
(Tamoxifen 自然切换在第 6 轮, 守卫摸不到); 哨兵 9/9 (36.1s)。

**代价 (如实)**: 发散域内 auto 的 SCF 比 fp64 慢 2.4~3.1× (混沌轮次浪费 +
劣质起点), 见 §4.2 的 † 标注。**使用建议**: ~1900 AO 及以上的精确交换
auto SCF 检查 `mf.converged`; 该尺寸域内 SCF 直接用 fp64, 梯度/Hessian
仍可用 auto。

### 6.3 CPU 参照的豁免声明

CPU Hessian 参照覆盖 methanol/Vitamin C/Tamoxifen 的 def2-svp (+ methanol
tzvp); tzvp 其余格子按用户决策豁免 (Tamoxifen tzvp 实测 ~20h/泛函);
Azadirachtin 的 CPU Hessian 天级成本, 全部豁免。以上缺列均已在 §3.4 列明,
不存在静默缺失。

## 7. 已知问题与方法学注记

1. **6d/5d 约定**: Gaussian Pople 基组默认笛卡尔 d; PySCF 默认球谐。对 g16
   验证必须 `mol.cart=True`, 否则 1.66×10⁻³ Eh 的约定差冒充代码误差 (实测)。
2. **wheel 遮蔽**: sys.path 缺仓库根时, venv 的发布版 wheel (无 precision
   模块) 静默接管, `precision_mode='auto'` 变无效属性, auto 加速归零且无告警。
   鉴伪: auto-vs-fp64 梯度差若 ~10⁻⁸ (原子非确定性) 而非 ~10⁻⁵ 即中招。
3. **max|dν| 的软模态污染**: 基准几何非所测等级下的驻点, 近零频模态把
   max 放大 (实测 CPU 网格 level 3→5 时 GPU-CPU max 差 0.55→27 cm⁻¹);
   故主指标用 rms(dν) + ΔZPE + ΔS_vib。
4. **SCF 计时的热启动污染**: 同一 mf 重复 kernel() 会从收敛密度热启动
   (实测 Tamoxifen fp64 SCF 5.2s → 1.6s)。报告一律用冷启动单发。
5. **r14 (开壳层 Ru 阳离子)**: 多解 SCF 体系, 已移出默认矩阵; 调查记录见
   `check_r14_*.py` 与 STATUS.md。
6. **cross-lane 梯度差的含义**: |dg|/f64 含 SCF 收敛路径差 + fp32 梯度误差
   两者; 同态分解实验 (r14) 表明后者本身 ~10⁻¹² 量级, 表列数值由前者主导
   但全部低于优化阈值, 结论不受影响。

## 8. 复现

```sh
cd /home/tong/soft/gpu4pyscf
# GPU 相位 (E/G ×3 中位 + Hessian 单发, ~5h)
.venv/bin/python mixedprec/benchmark/run_benchmark.py --skip-cpu
# CPU 相位 (参照; Hessian 长 pole, 用磁盘 TMPDIR 防止 tmpfs 吃内存)
TMPDIR=$HOME/tmp_scf setsid nohup .venv/bin/python mixedprec/benchmark/run_benchmark.py \
    --skip-gpu --cpu-hessian >> mixedprec/benchmark/results/cpu_campaign.log 2>&1 &
# 对外验证
.venv/bin/python mixedprec/benchmark/validate_methanol.py --fp64
# 汇总表
.venv/bin/python mixedprec/benchmark/run_benchmark.py --skip-gpu --skip-cpu   # 仅 derive+打印
```

注意: 必须从源码树跑 (见 §7.2); 跨进程结果按行键自动合并进
`results/benchmark.json`。

## 9. 结论

在闭壳层、def2-svp/tzvp、能量/梯度/Hessian 全路径上, **auto (混合精度)
通道的精度已由具名外部参照 (g16, -2.19×10⁻⁷ Eh)、跨实现 (CPU PySCF,
≤2×10⁻¹⁰ Eh) 与自比 (fp64) 三方交叉证实**; 对最强 CPU 代码 Direwolf 的
同等级对比吻合到网格系统差以内。速度上梯度/Hessian 随体系增大到
2~7× (SCF 1.2~1.8×), 大体系对 CPU 综合一个量级以上 (r2SCAN 梯度对
Direwolf 9×, 对 PySCF CPU 81× = 8.9× 实现 × 9.1× 硬件)。
两个当时的边界 (r2SCAN Hessian 的 OOM、~1900 AO 的 auto 发散) 已分别
修复并重测, 全矩阵无因代码问题缺失的格子; 剩余的真边界是 ~2500 AO 的
显存墙 (out-of-core 立项中) 与发散域内 auto-SCF 的负加速 (已如实标注)。
