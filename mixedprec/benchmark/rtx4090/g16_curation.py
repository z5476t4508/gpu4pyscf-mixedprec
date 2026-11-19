#!/usr/bin/env python3
"""Curation of the bundled Gaussian 16 test suite (plan section 3.6).

The bundled tests are CANDIDATES, most of them NOT usable as references.
This module implements the machine side of the curation gate:

  * ``parse_g16_log`` / ``read_precomputed``: read-only parsing of EXISTING
    G16 outputs (route cards per Link1 job, SCF Done records, termination
    flags, charge/multiplicity, NBasis).  It never invokes Gaussian and
    never writes to the proprietary tree -- rerunning G16 is a separate,
    explicit, exceptional action that this module cannot perform.
  * ``classify_case``: the admission taxonomy.  Machine gates are
    deliberately conservative: the best a machine can do is
    ``untriaged`` (provisional candidate).  ``reference_matched`` /
    ``reference_diagnostic`` / ``geometry_only`` always require the manual
    audit of plan section 3.6.2 item 5, because printed log geometries are
    rounded and method matching needs the full signature.
  * ``verify_entry_against_log``: cross-checks the catalog index against
    the raw log (terminations, SCF energies, NBasis, route-derived
    methods) so a stale or wrong index can never smuggle a case in.
  * ``curate``: the full pass producing the private curation JSON.

Classification taxonomy (plan section 3.6.1):
  untriaged            machine gates passed; provisional candidate only
  reference_matched    MANUAL ONLY: full method/geometry match verified
  reference_diagnostic MANUAL ONLY: method mostly matches, named diffs
  geometry_only        MANUAL ONLY: reuse structure, recompute energies
  deferred             needs manual evaluation (opt/MD/scan/IOp/solvent/
                       ECP/ghost/checkpoint deps, multi-job, multi-SCF,
                       AO budget, log unavailable)
  rejected             no target method / error-only termination

Adversarial stance: classification derives from the ROUTE CARDS AND LOG
content, never from the index's pre-digested fields; the index is only a
pointer that must itself survive verification.  G16 energies are
cross-program tier-(3) references ONLY -- they never replace the matching
GPU4PySCF FP64 lane for the 1e-6 Ha precision gate.

Usage:
    python g16_curation.py --index /data/datasets/gaussian-test-catalog/index.json \
        --out /data/work/gpu4pyscf-agent/inventory/g16-curation.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

CURATION_SCHEMA = 'g16-curation-v1'

# route-level method tokens (case-insensitive); canonical names are the
# project's four target methods.  A route "matching" a method at token
# level is only the FIRST gate -- never a claim that the full method
# signature (variant, basis, grid, corrections) matches.
TARGET_METHOD_TOKENS = {
    'B3LYP': ('b3lyp',),
    'M06-2X': ('m06-2x', 'm062x', 'm06-2x:'),
    'omegaB97M-V': ('wb97m-v', 'omegab97m-v', 'wb97mv'),
    'r2SCAN-3c': ('r2scan-3c', 'r2scan3c'),
}

# route semantics that default to MANUAL evaluation (plan section 3.6.2
# item 1).  Each entry: (regex, named reason).
DEFERRED_ROUTE_PATTERNS = [
    (r'\bopt\b|opt=', 'optimization job (geometry is step-dependent)'),
    (r'\bmd\b|\bmd=', 'molecular dynamics route (a single SCF inside an MD '
                      'run is not a single-point reference)'),
    (r'\bfreq\b', 'frequency job'),
    (r'\bforce\b', 'force job'),
    (r'\birc\b', 'IRC following'),
    (r'\bscan\b', 'coordinate scan'),
    (r'\bpolar\b', 'response property job'),
    (r'\bstable\b|stable=', 'stability analysis job'),
    (r'\bgeomcheck\b|geom=check', 'geometry checking'),
    (r'\btrack\b', 'state tracking'),
    (r'\btd\b|td=', 'TD-DFT excited-state job'),
    (r'\bnmr\b|nmr=', 'NMR property job'),
    (r'bomd', 'Born-Oppenheimer MD (single SCFs inside a trajectory are '
              'not references)'),
    (r'\bprop\b|prop=', 'property job'),
    (r'\bvolume\b', 'volume job'),
    (r'massage', 'massaged integrals'),
    (r'iop', 'explicit IOp overrides'),
    (r'\bfield\b', 'external field'),
    (r'scrf|solvent', 'continuum solvation'),
    (r'pseudo', 'ECP'),
    (r'ghost', 'ghost centers'),
    (r'guess=read|guess=check|guess=chk', 'checkpoint-dependent guess'),
]

_RE_SCF_DONE = re.compile(r'SCF Done:\s+E\(([^)]+)\)\s*=\s*(-?\d+\.\d+)')
_RE_CHARGE_MULT = re.compile(r'Charge =\s*(-?\d+)\s+Multiplicity =\s*(\d+)')
_RE_NBASIS = re.compile(r'NBasis =\s*(\d+)')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def methods_in_routes(routes):
    '''Target methods named at route level (token match only).'''
    found = []
    text = ' '.join(routes).lower()
    for method, tokens in TARGET_METHOD_TOKENS.items():
        if any(t in text for t in tokens):
            found.append(method)
    return found


def parse_g16_log(path):
    '''Read-only parse of an existing G16 log (the read_precomputed core).

    Returns a dict with routes (one per Link1 job), termination flags,
    SCF Done records, charge/multiplicity pairs and NBasis values; returns
    None when the file cannot be read/parsed at all.
    '''
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except OSError:
        return None
    if 'Gaussian' not in text:
        return None
    routes = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('#') and not stripped.startswith('#N=') \
                and 'Gaussian' not in stripped:
            routes.append(stripped)
    scf_done = [{'label': m.group(1), 'energy': float(m.group(2))}
                for m in _RE_SCF_DONE.finditer(text)]
    return {
        'routes': routes,
        'normal_terminations': text.count('Normal termination'),
        'error_terminations': len(re.findall(r'Error termination', text)),
        'scf_done': scf_done,
        'charge_multiplicity': [[int(m.group(1)), int(m.group(2))]
                                for m in _RE_CHARGE_MULT.finditer(text)],
        'nbasis_values': [int(m.group(1))
                          for m in _RE_NBASIS.finditer(text)],
    }


def read_precomputed(case, expectations=None):
    '''Adapter entry point: read one case's precomputed output.

    ``expectations`` (optional) maps expectation names to required values
    (e.g. {'normal_termination': True, 'n_scf': 1}); a mismatch raises
    ValueError so callers can never mistake a failed read for a reference.
    Never runs Gaussian; never writes anything.
    '''
    log = case.get('log')
    if not log or not os.path.isfile(log):
        raise ValueError(f'case {case.get("case")!r}: log {log!r} unavailable')
    data = parse_g16_log(log)
    if data is None:
        raise ValueError(f'case {case.get("case")!r}: unparseable log {log}')
    if expectations:
        if expectations.get('normal_termination') and \
                data['normal_terminations'] < 1:
            raise ValueError(f'case {case.get("case")!r}: no normal '
                             'termination')
        if 'n_scf' in expectations and \
                len(data['scf_done']) != expectations['n_scf']:
            raise ValueError(
                f'case {case.get("case")!r}: expected '
                f'{expectations["n_scf"]} SCF records, found '
                f'{len(data["scf_done"])}')
    return data


def classify_case(routes, log_data, ao_cap):
    '''The admission gate.  Returns (classification, reasons).

    Order matters: semantic reasons (opt/MD/...) are reported before
    generic ones (multi-SCF) so the table says WHY, not just that.
    '''
    reasons = []
    if not methods_in_routes(routes):
        return 'rejected', ['no target method '
                            f'({"/".join(TARGET_METHOD_TOKENS)}) in any route']
    if log_data is None:
        return 'deferred', ['log unavailable or unparseable; termination '
                            'and SCF verification impossible']
    if log_data['error_terminations'] and \
            not log_data['normal_terminations']:
        return 'rejected', ['error termination only; no normal termination']
    if not log_data['normal_terminations']:
        return 'deferred', ['no normal termination in the log']
    hits = [reason for pattern, reason in DEFERRED_ROUTE_PATTERNS
            if re.search(pattern, ' '.join(routes), re.IGNORECASE)]
    if hits:
        return 'deferred', ['route semantics require manual evaluation: '
                            + '; '.join(hits)]
    if len(log_data['routes']) > 1:
        return 'deferred', [f'multi-job Link1 input ({len(log_data["routes"])}'
                            ' routes); the exact subjob must be matched '
                            'explicitly']
    if not log_data['scf_done']:
        return 'deferred', ['no SCF Done record; cannot pin the target '
                            'solution']
    if len(log_data['scf_done']) > 1:
        return 'deferred', [f'{len(log_data["scf_done"])} SCF results '
                            '(multi-step/multi-solution); must pin the '
                            'exact subjob and check for alternate solutions']
    # NBasis appears only in the standard orientation block; logs that
    # omit it fall back to the (verified-elsewhere) index value
    nbasis = log_data['nbasis_values'][-1] if log_data['nbasis_values'] \
        else None
    if nbasis is not None and ao_cap is not None and nbasis > ao_cap:
        return 'deferred', [f'nbasis {nbasis} exceeds the pilot AO cap '
                            f'{ao_cap} (memory/representativeness budget)']
    return 'untriaged', [
        'machine gates passed; geometry is the printed (rounded) log '
        'orientation -- full-precision input-geometry recovery and '
        'verification still pending',
        'manual admission audit pending (plan section 3.6.2 item 5); '
        'reference_matched/reference_diagnostic/geometry_only are '
        'manual-only classifications']


def verify_entry_against_log(entry, log_data):
    '''Cross-check the catalog index against the raw log.  The log wins.'''
    problems = []
    if log_data is None:
        return ['log unavailable; index claims unverified']
    if entry.get('normal_terminations') is not None and \
            entry['normal_terminations'] != log_data['normal_terminations']:
        problems.append(f'normal_terminations: index '
                        f'{entry["normal_terminations"]} != log '
                        f'{log_data["normal_terminations"]}')
    if entry.get('has_error_termination') is not None and \
            bool(entry['has_error_termination']) != \
            bool(log_data['error_terminations']):
        problems.append(f'has_error_termination: index '
                        f'{entry["has_error_termination"]} != log '
                        f'{bool(log_data["error_terminations"])}')
    idx_scf = entry.get('scf_results') or []
    if len(idx_scf) != len(log_data['scf_done']):
        problems.append(f'scf_results count: index {len(idx_scf)} != log '
                        f'{len(log_data["scf_done"])}')
    else:
        for i, (a, b) in enumerate(zip(idx_scf, log_data['scf_done'])):
            if abs(float(a.get('energy_hartree', 0)) - b['energy']) > 1e-6:
                problems.append(f'scf_results[{i}] energy: index '
                                f'{a.get("energy_hartree")} != log '
                                f'{b["energy"]}')
    idx_methods = set(entry.get('methods') or [])
    route_methods = set(methods_in_routes(log_data['routes']))
    if idx_methods != route_methods:
        problems.append(f'methods: index {sorted(idx_methods)} != '
                        f'routes {sorted(route_methods)}')
    if entry.get('nbasis') is not None and \
            log_data['nbasis_values'] and \
            entry['nbasis'] not in log_data['nbasis_values']:
        problems.append(f'nbasis: index {entry["nbasis"]} not among log '
                        f'values {log_data["nbasis_values"]}')
    return problems


def curate(index_entries, ao_cap=120, verify_hashes=True):
    '''Full curation pass over catalog index entries.'''
    cases, counts = [], {}
    for entry in index_entries:
        idx_routes = entry.get('routes') or []
        log_path = entry.get('log')
        log_data = parse_g16_log(log_path) if log_path else None
        # routes come from the LOG when it parsed (log wins over index)
        routes = log_data['routes'] if log_data and log_data['routes'] \
            else idx_routes
        classification, reasons = classify_case(routes, log_data, ao_cap)
        consistency = verify_entry_against_log(entry, log_data) \
            if methods_in_routes(idx_routes) or log_data else []
        hash_ok = None
        if verify_hashes and log_path and os.path.isfile(log_path):
            hash_ok = sha256_file(log_path) == entry.get('log_sha256')
        case = {
            'case': entry.get('case'),
            'input': entry.get('input'),
            'input_sha256': entry.get('input_sha256'),
            'log': log_path,
            'routes': routes,
            'methods': methods_in_routes(routes),
            'classification': classification,
            'reasons': reasons,
            'normal_terminations': log_data['normal_terminations']
            if log_data else None,
            'error_terminations': log_data['error_terminations']
            if log_data else None,
            'scf_results': log_data['scf_done'] if log_data else None,
            'nbasis': (log_data['nbasis_values'][-1]
                        if log_data and log_data['nbasis_values']
                        else entry.get('nbasis')),
            'charge_multiplicity': log_data['charge_multiplicity'][-1]
            if log_data and log_data['charge_multiplicity'] else None,
            'geometry_source': 'printed_log_orientation_angstrom',
            'index_log_consistency': consistency or 'ok',
            'log_sha256_verified': hash_ok,
            'machine_admission': 'provisional',
            'manual_audit': 'pending',
        }
        if classification == 'untriaged':
            case['g16_energy_hartree'] = log_data['scf_done'][0]['energy']
            case['g16_method_signature'] = routes[0] if routes else None
        cases.append(case)
        counts[classification] = counts.get(classification, 0) + 1
    by_method = {}
    for case in cases:
        if case['methods']:
            key = '+'.join(case['methods'])
            by_method[key] = by_method.get(key, 0) + 1
    doc = {
        'schema': CURATION_SCHEMA,
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                          time.gmtime()),
        'policy': [
            'Bundled G16 tests are candidates, not automatically usable '
            'benchmarks; classification reasons are recorded per case.',
            'Machine admission is provisional (untriaged at best); '
            'reference_matched/reference_diagnostic/geometry_only require '
            'the manual audit of plan section 3.6.2 item 5.',
            'Existing G16 outputs are cross-program tier-(3) references '
            'only; they never replace the matching GPU4PySCF FP64 lane '
            'for the 1e-6 Ha precision gate.',
            'G16 is never rerun by this module; parsing is read-only.',
            'Private: bundled inputs, logs and extracted coordinates never '
            'leave the server.',
        ],
        'ao_cap': ao_cap,
        'counts': {'total': len(cases), 'by_classification': counts,
                   'method_entries': by_method,
                   'provisional_candidates': counts.get('untriaged', 0)},
        'cases': cases,
    }
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='g16_curation',
        description='Classify the bundled G16 tests against the four target '
                    'methods (read-only; never reruns Gaussian). Output is '
                    'PRIVATE inventory (plan section 3.6.3).')
    p.add_argument('--index', required=True,
                   help='catalog index.json (private)')
    p.add_argument('--out', required=True,
                   help='output curation JSON (private inventory)')
    p.add_argument('--ao-cap', type=int, default=120,
                   help='pilot AO budget cap for provisional candidates')
    args = p.parse_args(argv)
    with open(args.index, encoding='utf-8') as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        print('error: index must be a JSON list of case entries',
              file=sys.stderr)
        return 2
    doc = curate(entries, ao_cap=args.ao_cap)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f'wrote {args.out}: {doc["counts"]["total"]} cases, '
          f'provisional candidates (untriaged) = '
          f'{doc["counts"]["provisional_candidates"]}')
    print('by classification:', json.dumps(doc['counts']['by_classification']))
    print('method entries:', json.dumps(doc['counts']['method_entries']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
