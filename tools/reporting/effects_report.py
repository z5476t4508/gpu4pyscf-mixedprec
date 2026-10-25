"""Render measured effects from hash-pinned benchmark JSON, never private logs."""
import hashlib
import json
import math
from pathlib import Path
import statistics

TASKS = Path('/data/results/gpu4pyscf-agent/tasks')
INDEX = Path('/data/work/gpu4pyscf-agent/state/long-chain/effects-index.json')


class BenchmarkMetadata(ValueError):
    pass


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('missing/nonfinite measurement')
    return value


def gradient(sample):
    values = sample['gradient']
    if not values or any(len(v) != 3 for v in values):
        raise ValueError('invalid gradient shape')
    return [finite(x) for v in values for x in v]


def measure(row, ref):
    if row['system'] != ref['system'] or row['method'] != ref['method']:
        raise ValueError('unmatched geometry or method settings')
    samples = row['result']['repeats']
    refs = ref['result']['repeats']
    if len(samples) < 3 or len(refs) < 3:
        raise ValueError('fewer than three measured repeats')
    de, dg = 0., 0.
    for a in samples:
        for b in refs:
            de = max(de, abs(finite(a['e_tot']) - finite(b['e_tot'])))
            ga, gb = gradient(a), gradient(b)
            if len(ga) != len(gb):
                raise ValueError('unmatched gradient shape')
            dg = max(dg, max(abs(x-y) for x, y in zip(ga, gb)))
    def timing(items, key):
        values = [finite(s[key]) for s in items]
        if min(values) <= 0:
            raise ValueError('nonpositive timing')
        return statistics.median(values)
    selected = [(s.get('stability_verification') or {}) for s in samples]
    incomplete = any(v.get('verification_incomplete_reason') or v.get('skipped')
                     or v.get('refine_unusable') for v in selected)
    if row['row_key']['lane'] == 'auto-strict':
        incomplete |= any(v.get('policy') != 'auto+strict'
                          or v.get('strict_match_verified') is not True
                          or v.get('strict_match_incomplete_reason') for v in selected)
    # A historical verification can have a correct main output but a WRONG
    # sibling reference (e.g. omitted D4/gCP). Never call that a safe pass.
    bad_reference = any(v and (v.get('ref_converged') is not True or
                        max(abs(finite(v['ref_e_tot']) - finite(b['e_tot']))
                            for b in refs) > 1e-6) for v in selected)
    converged = all(s.get('converged') is True for s in samples + refs)
    passed = (row.get('status') == ref.get('status') == 'ok' and converged
              and not incomplete and not bad_reference and de <= 1e-6 and dg <= 3e-5
              and row['result'].get('mem', {}).get('within_budget') is True
              and ref['result'].get('mem', {}).get('within_budget') is True
              and not row['result'].get('errors') and not ref['result'].get('errors'))
    return {'de': de, 'dg': dg, 'passed': passed, 'n': len(samples),
            'speed': timing(refs, 'sample_s') / timing(samples, 'sample_s'),
            'seconds': timing(samples, 'sample_s'), 'ref_seconds': timing(refs, 'sample_s'),
            'grad_speed': timing(refs, 't_grad_s') / timing(samples, 't_grad_s'),
            'adopted': sum(v.get('selected') in ('reference', 'main_refined') for v in selected),
            'fallbacks': len(row['result'].get('fallback_events') or []),
            'fp32': any(finite(v) > 0 for v in row['result'].get('fp32_entry_hits', {}).values()),
            'reason': '参考验证失效' if bad_reference or incomplete else '精度/收敛/资源门槛',
            'name': row['row_key']['xc'] + '/' + row['row_key']['geom'] + '/' + row['row_key']['basis']}


def read_source(source):
    path = Path(source['path']).resolve()
    if not path.is_relative_to(TASKS.resolve()):
        raise ValueError('source outside approved benchmark evidence')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('evidence hash mismatch')
    return path, raw


def expand_source(source):
    if Path(source['path']).name != 'evidence_sha256.txt':
        return [] if source['path'].endswith('.raw.json') else [source]
    path, raw = read_source(source)
    children = []
    for line in raw.decode().splitlines():
        digest, separator, name = line.partition('  ')
        if not separator or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid evidence manifest')
        child = (path.parent / name).resolve()
        if not child.is_relative_to(path.parent):
            raise ValueError('manifest path traversal')
        if name.endswith('.json') and not name.endswith('.raw.json'):
            children.append({'path': str(child), 'sha256': digest})
    return children


def load_source(source):
    path, raw = read_source(source)
    if path.suffix != '.json':
        raise BenchmarkMetadata('not a benchmark JSON document')
    doc = json.loads(raw)
    if not isinstance(doc, dict) or 'schema_version' not in doc:
        raise BenchmarkMetadata('non-benchmark JSON metadata')
    if doc.get('schema_version') != 'rtx4090-gradient-v1':
        raise ValueError('unsupported benchmark schema')
    return doc


def render(index=INDEX, recent=None):
    lines = ['', '### 实测效果（有日期的快照，不等于今日新增成果）', '',
             '- 对照：同文档、相同几何/方法设置的 lane=fp64；该标签不保证全程严格 FP64。三次及以上重复全部比较，含 SCF、梯度、验证/回退耗时。',
             '- 若两道共享低精度内核，文档内匹配不能独立证明相对严格 FP64 的误差达标，须另核对关闭近似旋钮的参考。',
             '- 门槛：|ΔE| ≤ 1e-6 Ha，max|Δ梯度| ≤ 3e-5 Eh/Bohr。以下为源结果重算，非独立物理验收；数值匹配不证明稳定极小值。',
             '- 倍数 = 文档内对照道时间 / 候选时间；大于 1 才更快。端到端与仅梯度阶段分开。']
    if not index.exists():
        return lines + ['- 尚无可复算的效果索引；效果未知，不以测试数或 PR 数代替。']
    try:
        datasets = json.loads(index.read_text())['datasets']
    except (OSError, ValueError, KeyError, TypeError):
        return lines + ['- 效果索引不可读；效果未知。']
    if recent is not None:
        lines.append(f'- 本摘要展示最近 {min(recent, len(datasets))}/{len(datasets)} 批记录；历史失败不因省略而视为解决，完整历史保留服务器完整日报。')
        datasets = datasets[-recent:]
    for dataset in datasets:
        # Agent-authored labels can contain entire phase narratives. Public
        # effects come from measurements below, not those unverified claims.
        title = str(dataset.get('id') or dataset.get('label', '未命名记录'))[:200]
        sources = dataset.get('sources')
        if not isinstance(sources, list) or not sources:
            lines += ['', '#### ' + title,
                      '', '- 无可复算的 benchmark sources；仅有元数据/审计记录，不计为数值通过。']
            continue
        if any(Path(s.get('path','')).name.endswith('-schedule.json') for s in sources):
            lines += ['', '#### ' + title, '',
                      '- 四路线配对记录：按独立严格R参考判定，性能与失败见上方最新三种对照；不再按旧单文档格式生成空表。']
            continue
        lanes = {'auto': [], 'auto-verify': [], 'auto-strict': []}
        dates, commits, digests = set(), set(), []
        unknown, metadata = 0, 0
        expanded, seen = [], set()
        for source in sources:
            try:
                for child in expand_source(source):
                    identity = (str(Path(child['path']).resolve()), child['sha256'])
                    if identity not in seen:
                        expanded.append(child)
                        seen.add(identity)
            except (OSError, ValueError, KeyError, TypeError):
                unknown += 1
        for source in expanded:
            try:
                doc = load_source(source)
                dates.add(doc['meta']['timestamp_utc'][:10])
                commits.add(doc['meta']['gpu4pyscf_source_commit'][:7])
                digests.append(source['sha256'][:12])
                refs = {tuple(r['row_key'][k] for k in ('geom', 'basis', 'xc')): r
                        for r in doc['rows'] if r['row_key']['lane'] == 'fp64'}
                for row in doc['rows']:
                    lane = row['row_key']['lane']
                    if lane in lanes:
                        key = tuple(row['row_key'][k] for k in ('geom', 'basis', 'xc'))
                        lanes[lane].append(measure(row, refs[key]))
            except BenchmarkMetadata:
                metadata += 1
            except (OSError, ValueError, KeyError, TypeError):
                unknown += 1
        if not any(lanes.values()):
            lines += ['', '#### ' + title, '',
                      '- 此记录没有旧格式的同文档候选/参考配对；机制/审计记录不生成空性能表，四路线性能见上方最新三种对照。']
            if unknown:
                lines.append(f'- 旧格式不可读/缺失/哈希失败来源 {unknown}；覆盖不完整，不能当作精度失败数或通过数。')
            if metadata:
                lines.append(f'- 非 benchmark 元数据：{metadata}；不作为数值通过或计时样本。')
            continue
        lines += ['', '#### ' + title, '',
                  '- 测量日期（UTC）：' + ', '.join(sorted(dates)) + '；源提交：' + ', '.join(sorted(commits)) + '。',
                  '', '| 路线 | 文档内门槛通过/可比较体系 | 最大能量误差 Ha | 最大梯度误差 Eh/Bohr | 端到端倍数范围 | 仅梯度倍数范围 | 采用参考/精炼样本 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        notes = []
        for lane, rows in lanes.items():
            if not rows:
                continue
            n = sum(r['n'] for r in rows)
            speeds = [r['speed'] for r in rows]
            grads = [r['grad_speed'] for r in rows]
            lines.append(f"| {lane} | {sum(r['passed'] for r in rows)}/{len(rows)} | {max(r['de'] for r in rows):.3e} | {max(r['dg'] for r in rows):.3e} | {min(speeds):.3f}–{max(speeds):.3f}× | {min(grads):.2f}–{max(grads):.2f}× | {sum(r['adopted'] for r in rows)}/{n} |")
            notes.append(f"- {lane}：{n} 个计时样本；实际 FP32 入口证据 {sum(r['fp32'] for r in rows)}/{len(rows)} 体系；记录回退事件 {sum(r['fallbacks'] for r in rows)}（非事件率，未记录不表示不存在）。")
            methods = sorted({r['name'].split('/')[0] for r in rows})
            notes.append(f"- {lane} 按方法的端到端倍数中位数：" + '；'.join(
                f"{method} {statistics.median(r['speed'] for r in rows if r['name'].split('/')[0] == method):.3f}×"
                for method in methods) + '（不同分子等权，不是总耗时加速比）。')
            best = max(rows, key=lambda r: r['speed'])
            worst = min(rows, key=lambda r: r['speed'])
            for label, r in [('最快案例', best), ('最慢案例', worst)]:
                notes.append(f"- {lane} {label} {r['name']}：lane=fp64 {r['ref_seconds']:.3f}s → {r['seconds']:.3f}s，{r['speed']:.3f}×，{'通过' if r['passed'] else '未通过'}。")
            for r in rows:
                if not r['passed']:
                    notes.append(f"- **未通过** {lane} {r['name']}（{r['reason']}）：|ΔE|={r['de']:.3e} Ha，max|Δ梯度|={r['dg']:.3e} Eh/Bohr，端到端 {r['speed']:.2f}×。")
        lines += [''] + notes
        if unknown:
            lines.append(f'- 缺失/不匹配/哈希失效来源：{unknown}；不计为通过，覆盖不完整。')
        if metadata:
            lines.append(f'- 非 benchmark 元数据：{metadata}；已核对来源哈希，不作为数值通过或计时样本。')
        lines.append('- 来源摘要 SHA256：' + ', '.join(digests) + '（完整索引及原始 JSON 保留服务器）。')
    lines += ['', '- 方向：优先让 UKS 在原门槛下安全匹配参考，并降低验证代价；只在精度通过且端到端确实更快的体系讨论收益。历史快照需当前源码复测。']
    return lines
