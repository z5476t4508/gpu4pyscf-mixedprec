# gpu4pyscf 混合精度基准报告

**日期**: 2026-09-10 ~ 09-11 · **分支**: `mixed-precision-hf` · **代码**: commit `decfbcf`
**硬件**: NVIDIA RTX 5090 (32 GB) vs CPU PySCF 2.14.0 (24 线程, 同机)
**软件**: Python 3.14.4 / CuPy 14.2.0 / gpu4pyscf 源码树 (commit `decfbcf`)
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
3. **硬件边界**: r2SCAN 的解析 Hessian 在 934 AO (svp) / 1042 AO (tzvp) 以上
   超出 32 GB (双车道 OOM); hf/B3LYP 的 Hessian 到 ~1900 AO 仍可通过
   (fp64 单条 2.4~2.7 h)。能量/梯度到 ~1900 AO 无压力。
4. **鲁棒性边界**: Aza tzvp (~1900 AO) 的 hf/B3LYP auto 车道 SCF 50 轮不收敛
   (fp64 收敛) —— 大 tzvp 体系 + fp32 起步的组合需要关注; r2SCAN 同体系正常。
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
| Azadirachtin (934/1877) | 1.9~2.6×10⁻⁵ / 1.9~2.6×10⁻⁵ | r2SCAN 2.8×10⁻⁵ (hf/b3lyp auto 未收敛, 见 §6) |

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
| Tamoxifen·tzvp | 6.8/—/19.0 ×10⁻⁴ | (同上) | — |
| Azadirachtin·svp | 3.9/—/4.0 ×10⁻⁴ | (CPU E/G 有, Hessian 豁免) | — |

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
| Aza tzvp hf/b3lyp 的 auto 全行 | 有效精度/速度 | auto SCF 未收敛 (§6), 数据无效已标记 |
| r2SCAN Hessian @ ≥934 AO | 全部 | OOM (§5), 32 GB 硬件边界 |

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
| | Hessian | **6.0×** | OOM | **5.5×** |

### 4.2 def2-tzvp

| 体系 (AO) | 量 | hf | r2SCAN | B3LYP |
|---|---|---|---|---|
| Vitamin C (~395) | SCF | 1.3× | 1.5× | 1.3× |
| | 梯度 | **3.5×** | **3.5×** | **3.4×** |
| | Hessian | 1.4× | **2.8×** | 1.8× |
| Tamoxifen (1042) | SCF | 1.4× | 1.7× | 1.4× |
| | 梯度 | **3.0×** | **6.2×** | **3.2×** |
| | Hessian | **3.7×** | OOM | **3.7×** |
| Azadirachtin (~1877) | SCF | (未收敛) | 1.8× | (未收敛) |
| | 梯度 | (未收敛) | **7.1×** | (未收敛) |
| | Hessian | (未收敛) | OOM | (未收敛) |

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

## 6. 硬件边界与鲁棒性 (5090, 32 GB)

### 6.1 解析 Hessian 的显存墙 (OOM = 双车道均 cudaErrorMemoryAllocation)

| 格子 | 结果 |
|---|---|
| r2SCAN Hessian, 934 AO (svp) | **OOM** (干净进程复现, 非碎片) |
| r2SCAN Hessian, 1042 AO (tzvp) | **OOM** |
| r2SCAN Hessian, ~395 AO (tzvp) | ✅ 14~39s |
| hf / B3LYP Hessian, 1042 AO (tzvp) | ✅ 552/764s (fp64) |
| hf / B3LYP Hessian, ~1877 AO (tzvp) | ✅ 8762/9830s (fp64, 单条 2.4~2.7h) |

墙的主体是 CDERI (~24 GB @ 934 AO) 叠加 meta-GGA 的网格二阶响应;
hf/B3LYP 的响应张量更小, 能到 ~1900 AO。**含义**: 100+ 原子催化体系的
TS 优化 (梯度驱动) 在 5090 上可行; 唯 r2SCAN 的确认性频率计算要么换
B3LYP/hf, 要么走 out-of-core (已立项, 见 STATUS.md「硬件边界」)。

### 6.2 auto 车道的 SCF 收敛性 (新发现)

Azadirachtin·tzvp 的 hf/B3LYP auto 车道 50 轮内 **未收敛** (fp64 同格收敛;
梯度差 5.5~5.8 Eh/Bohr = 落入不同电子态, 其频率数据无效, 已在 JSON 标记
NOT-CONV 并从上表剔除)。同体系 r2SCAN 的 auto 正常收敛 (gerr 2.8×10⁻⁵)。
**使用建议**: ≥~1900 AO 的 tzvp 大体系跑 auto 时检查 `mf.converged`;
runner 已逐行记录该旗标。

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
通道的精度已由具名外部参照 (g16)、跨实现 (CPU PySCF) 与自比 (fp64) 三方
交叉证实**: 能量无损 (≤10⁻¹⁰ Eh), 梯度误差 ≤2.6×10⁻⁵ (余量十倍以上),
频率误差 rms ≤2×10⁻³ cm⁻¹。速度上梯度/Hessian 随体系增大到 2~7×,
SCF 1.2~1.8×; 对 CPU 综合一个量级以上 (r2SCAN 梯度近百倍)。
边界与例外 (r2SCAN 大体系 Hessian 的 32 GB 墙、~1900 AO tzvp 的 auto
收敛性、跨实现网格系统差) 均已量化并如实标注, 无未解释的误差。
