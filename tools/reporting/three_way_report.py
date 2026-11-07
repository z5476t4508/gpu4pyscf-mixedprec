"""Keep upstream, optimized-baseline and auto effects distinct and hash-pinned."""
import hashlib
import json
from pathlib import Path
import statistics
import effects_report as effects

REFERENCE = Path('/opt/qc-agent/upstream-reference.json')
REQUEST = Path('/data/work/gpu4pyscf-agent/control/upstream-comparison-20261018.md')
ROLES = ('upstream', 'optimized_fp64', 'optimized_auto', 'strict_fp64')
COMPARISONS = (('原版 → 优化基准', 'upstream', 'optimized_fp64'),
               ('优化基准 → auto', 'optimized_fp64', 'optimized_auto'),
               ('原版 → auto', 'upstream', 'optimized_auto'))


class CellFailure(ValueError):
    pass


def load_cell(cell, batch_id):
    docs = {role: effects.load_source(cell['sources'][role]) for role in ROLES}
    certificate = json.loads(REFERENCE.read_text())
    certificate_hash = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    anchor = docs['strict_fp64']['meta']
    required = {'pyscf', 'cupy', 'numpy', 'libxc', 'cutensor'}
    if not required.issubset(anchor['dependencies']):
        raise ValueError('missing runtime versions')
    for role, doc in docs.items():
        meta = doc['meta']
        if len(doc['rows']) == 1 and doc['rows'][0].get('status') in ('oom', 'error', 'failed', 'skipped'):
            raise CellFailure(role + ': ' + doc['rows'][0]['status'])
        if (meta['comparison_role'] != role or meta['comparison_batch_id'] != batch_id
                or meta['timing_protocol'] != 'balanced-blocks-v1'
                or meta['schedule_sha256'] != anchor['schedule_sha256']
                or len(meta['schedule_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in meta['schedule_sha256'])
                or meta['dependencies'] != anchor['dependencies']
                or len(doc['rows']) != 1
                or doc['rows'][0]['row_key']['lane'] != ('auto' if role == 'optimized_auto' else 'fp64')
                or len(doc['rows'][0]['result']['repeats']) != 4):
            raise ValueError('unmatched comparison provenance')
    upstream = docs['upstream']['meta']
    if (upstream['gpu4pyscf_package_root'] != certificate['package_root'] + '/gpu4pyscf'
            or upstream['reference_manifest_sha256'] != certificate_hash):
        raise ValueError('not the pinned official package')
    if any(docs[role]['meta']['gpu4pyscf_package_root'] != anchor['gpu4pyscf_package_root']
           for role in ('optimized_fp64', 'optimized_auto')) or anchor['gpu4pyscf_package_root'] == upstream['gpu4pyscf_package_root']:
        raise ValueError('candidate package provenance mismatch')
    if (anchor['precision_mode'] != 'fp64' or anchor['optimization_env']
            or upstream['optimization_env']
            or docs['optimized_fp64']['meta']['precision_mode'] != 'fp64'
            or docs['optimized_auto']['meta']['precision_mode'] != 'auto'
            or docs['optimized_fp64']['meta']['optimization_env'] != docs['optimized_auto']['meta']['optimization_env']):
        raise ValueError('strict reference or optimized stack mismatch')
    strict = docs['strict_fp64']['rows'][0]['result']
    if (strict['gradient_mode'] != 'fp64'
            or any(effects.finite(v) != 0 for v in strict['fp32_entry_hits'].values())):
        raise ValueError('FP32 execution in strict reference')
    if not any(effects.finite(v) > 0 for v in docs['optimized_auto']['rows'][0]['result']['fp32_entry_hits'].values()):
        raise ValueError('no actual FP32 entry evidence for auto')
    return {role: doc['rows'][0] for role, doc in docs.items()}


def render(index):
    lines = ['', '### 三种对照：整体收益与 auto 额外收益分开', '',
             '- 原版 = 已校验未修改的官方 GPU4PySCF 1.8.1；四条路线使用相同 PySCF/CuPy/libxc 等依赖、几何、方法、基组、网格和收敛设置。',
             '- 原版为官方 wheel，我们为源码编译版本；整体产品收益包含构建差异，不能把全部差值归因于某个内核。',
             '- 优化基准 = 我们优化后的 fp64-SCF 标签路线，允许明确记录的低精度内核；auto = 同一优化栈再启用混合精度 SCF。两者都不冒充全程严格 FP64。',
             '- 第四条严格 FP64 路线仅作精度锚：全部近似开关关闭；|ΔE| ≤ 1e-6 Ha、max|Δ梯度| ≤ 3e-5 Eh/Bohr 不变，每条路线全部正式重复均需收敛。原版与严格锚的实现一致性另行核对。',
             '- 倍数 = 对照耗时 / 候选耗时，大于 1 才更快。计时含构建、DF、SCF、梯度及必要校正/验证/回退；预热分离，配对平衡顺序；重复样本不是独立分子。',
             '- 以下是可复算工程测量，不是独立科学验收；历史文档内 auto/fp64 倍数不能替代原版对照。']
    lines.append('- 配对实验设计方法来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)；仅为方法来源，不是性能证据。')
    try:
        batches = json.loads(Path(index).read_text())['batches']
        if not isinstance(batches, list):
            raise ValueError('invalid comparison index')
        if batches and (not isinstance(batches[-1], dict)
                        or not isinstance(batches[-1].get('id'), str)
                        or not isinstance(batches[-1].get('cells'), list)):
            raise ValueError('invalid comparison batch')
    except (OSError, ValueError, KeyError, TypeError):
        batches = []
    if not batches:
        if REQUEST.exists():
            lines += ['', '- 原版参考已准备，测量要求已接入服务器长链优先任务；等待当前 GPU 批次安全结束后接续。准备/排队不等于原版测量已经开始或通过。']
        lines.append('- 此处等待新的四路配对批次；已有文档内 auto/fp64 结果保留在下面历史实测中，不能据此填写原版对照。')
        lines += ['', '| 对照 | 端到端加速比 | 状态 |', '|---|---:|---|']
        lines += ['| ' + label + ' | — | 待测：尚无完整、匹配且哈希可核对的四路线结果 |'
                  for label, _, _ in COMPARISONS]
        return lines
    # Keep the latest batch in every daily summary, irrespective of legacy-index order.
    batch = batches[-1]
    lines += ['', '#### 配对批次 ' + str(batch['id'])[:100], '']
    cells = batch['cells']
    measured = {label: [] for label, _, _ in COMPARISONS}
    accuracy = {role: [] for role in ROLES}
    unknown, failures = 0, []
    for i, cell in enumerate(cells):
        try:
            rows = load_cell(cell, batch['id'])
            gates = {role: effects.measure(rows[role], rows['strict_fp64']) for role in ROLES}
            cell_results = {}
            for role in ('upstream', 'optimized_fp64', 'optimized_auto'):
                if not gates[role]['passed']:
                    r = gates[role]
                    failures.append(f"- **未通过** 单元 {i+1} {role}：|ΔE|={r['de']:.3e} Ha，max|Δ梯度|={r['dg']:.3e} Eh/Bohr（{'原版实现一致性' if role == 'upstream' else '严格 FP64 精度/收敛/资源门槛'}）。")
            for label, reference, candidate in COMPARISONS:
                r = effects.measure(rows[candidate], rows[reference])
                r['passed'] &= gates[reference]['passed'] and gates[candidate]['passed']
                def scf_time(role):
                    values = [effects.finite(s['t_scf_s']) for s in rows[role]['result']['repeats']]
                    if min(values) <= 0:
                        raise ValueError('nonpositive SCF time')
                    return statistics.median(values)
                r['scf_speed'] = scf_time(reference) / scf_time(candidate)
                cell_results[label] = r
            for label, r in cell_results.items():
                measured[label].append(r)
            for role, gate in gates.items():
                accuracy[role].append(gate)
        except CellFailure as error:
            failures.append(f'- **运行失败/跳过** 单元 {i+1}：{error}；不计通过，不删除计划单元。')
            unknown += 1
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            unknown += 1
    lines += ['| 对照 | 守门通过/计划单元 | 端到端倍数中位数 | SCF 倍数中位数 | 梯度倍数中位数 | 固定通过子集总耗时比 |',
              '|---|---:|---:|---:|---:|---:|']
    notes = []
    for label, _, _ in COMPARISONS:
        passed = [r for r in measured[label] if r['passed']]
        metrics = ['—'] * 4
        if passed:
            metrics = [f"{statistics.median(r[key] for r in passed):.3f}×"
                       for key in ('speed', 'scf_speed', 'grad_speed')]
            metrics.append(f"{sum(r['ref_seconds'] for r in passed) / sum(r['seconds'] for r in passed):.3f}×")
        lines.append(f"| {label} | {len(passed)}/{len(cells)} | " + ' | '.join(metrics) + ' |')
        if passed:
            best = max(passed, key=lambda r: r['speed'])
            worst = min(passed, key=lambda r: r['speed'])
            notes.append(f"- {label}：通过子集端到端 {worst['speed']:.3f}–{best['speed']:.3f}×；最慢 {worst['name']} {worst['ref_seconds']:.3f}s → {worst['seconds']:.3f}s，最快 {best['name']} {best['ref_seconds']:.3f}s → {best['seconds']:.3f}s。")
            notes.append(f"- {label} 固定通过子集各单元中位数之和：{sum(r['ref_seconds'] for r in passed):.3f}s → {sum(r['seconds'] for r in passed):.3f}s；不是含预热的整场实验 wall time。")
            methods = sorted({r['name'].split('/')[0] for r in passed})
            notes.append('- ' + label + ' 按方法中位数：' + '；'.join(
                f"{method} {statistics.median(r['speed'] for r in passed if r['name'].split('/')[0] == method):.3f}×"
                for method in methods) + '。')
    lines += [''] + notes
    for role, label in [('optimized_fp64', '优化基准/R 精度'), ('optimized_auto', 'auto/R 精度'), ('upstream', '原版/R 实现一致性')]:
        checked = accuracy[role]
        if checked:
            lines.append(f"- {label}：通过 {sum(r['passed'] for r in checked)}/{len(cells)}；最大 |ΔE|={max(r['de'] for r in checked):.3e} Ha，max|Δ梯度|={max(r['dg'] for r in checked):.3e} Eh/Bohr。")
    lines += ['', f'- 缺失/不可比单元：{unknown}/{len(cells)}；不得从分母中删去。存在缺失或失败时，上述通过子集不能称为全矩阵整体加速。']
    lines += failures
    return lines
