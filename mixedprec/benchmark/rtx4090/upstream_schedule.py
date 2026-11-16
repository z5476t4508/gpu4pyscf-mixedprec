#!/usr/bin/env python3
"""Balanced-blocks-v1 schedule generator for the four-arm upstream
comparison (control card 2026-10-18).

Per planned cell the schedule freezes FOUR adjacent four-arm paired blocks
(roles O/B/C/R); the arm order inside the blocks is a Latin square (every
role occupies every position exactly once), randomized per cell from the
card's seed 20261018.  Each (block, arm) run is one independent process
with its own separated warmup and exactly ONE formal sample, so each role
accumulates four formal samples at four different time positions -- the
balanced-blocks answer to the lane-relative session-excursion class
(LC-195/200).

The schedule file is frozen BEFORE the run: its sha256 is the
schedule_sha256 every comparison document must carry, and its cell list
is the batch denominator (planned-but-not-run cells stay in it; the
index never drops them).
"""

import argparse
import hashlib
import json
import os
import random
import sys

SCHEMA = 'upstream-comparison-schedule-v1'
ROLES = ('O', 'B', 'C', 'R')
ROLE_NAMES = {'O': 'upstream', 'B': 'optimized_fp64',
              'C': 'optimized_auto', 'R': 'strict_fp64'}
SEED = 20261018

# benchmark v1 (2026-10-30 resume card / sealed package
# benchmark-v1-20261029): EIGHT paired formal O/B/C/R blocks per cell --
# two seeded independent 4x4 Latin squares over the frozen 64-cell
# PubChem cohort.  The legacy four-block v1 protocol above stays the
# only interpreter of four-block schedules; nothing v2 reinterprets an
# old schedule or its data.
SCHEMA_V2 = 'upstream-comparison-schedule-v2'
SEED_V2 = 20261029
SETTINGS_V2 = {
    'conv_tol': 1e-10,
    'max_cycle': 50,
    'init_guess': 'minao',
    'direct_scf_tol': 1e-14,
    'grids_level': 3,
    # contract.json scientific_settings pins nlcgrids explicitly (level 3
    # equals the library default both stacks record per row; None in v1
    # relied on that default -- v2 pins it so the pin itself is evidence)
    'nlcgrids_level': 3,
    'small_rho_cutoff': 0,
    'headroom_mib': 2048.0,
    'auxbasis': 'def2-universal-jkfit',
    'timeout_s_per_arm': 2700,
    'pilot_stage': 'S2',
}
# the S (auto-strict) paired protocol: same 64 cells, 8 blocks of one
# S/B pair each; S charges its own verification/matching/fallback/
# gradient costs (never C speedups), B is the same optimized_fp64 arm
# the main schedule pairs against -- the pairing B_to_S is the quantity.
SCHEMA_S = 'upstream-comparison-schedule-s-v1'
SEED_S = 20261029
S_ROLES = ('S', 'B')
# the shared run settings, frozen with the schedule: identical numbers for
# all four arms (control card §四路同设置).  These mirror the production
# gradient_runner defaults exactly.
SETTINGS = {
    'conv_tol': 1e-10,
    'max_cycle': 50,
    'init_guess': 'minao',
    'direct_scf_tol': 1e-14,
    'grids_level': 3,
    'nlcgrids_level': None,
    'headroom_mib': 2048.0,
    'auxbasis': 'def2-universal-jkfit',
    'timeout_s_per_arm': 2700,
    'pilot_stage': 'S2',
}
# stage 1: the frozen 12 medium-CHON x 4 methods = 48 cells (LC-202
# surface, manifest order, LC-168 basis assignment), charge 0 spin 0
MEDIUMCHON = ['aspirin', 'caffeine', 'dopamine', 'epinephrine',
              'glycylglycine', 'histamine', 'lidocaine', 'melatonin',
              'nicotine', 'phenylalanine', 'serotonin', 'tryptophan']
MEDIUMCHON_GEOM_DIR = ('/data/work/gpu4pyscf-agent/inventory/'
                       'gpu4pyscf-mediumchon-pubchem-20261017-v1')
LC168_BASES = {'b3lyp': 'def2-svp', 'm062x': 'def2-svp',
               'wb97mv': 'def2-svp', 'r2scan3c': 'def2-mtzvpp'}
METHODS = ('b3lyp', 'm062x', 'wb97mv', 'r2scan3c')


def latin_square(rng):
    '''A random 4x4 Latin square over ROLES: rows = blocks, columns =
    positions.  Every role appears exactly once per row and once per
    column -- the balanced order the card requires.'''
    # start from a cyclic Latin square, then shuffle symbols, rows and
    # columns with the seeded RNG; all three shuffles preserve the
    # Latin property
    symbols = list(ROLES)
    rng.shuffle(symbols)
    square = [[symbols[(j - i) % 4] for j in range(4)] for i in range(4)]
    rows = list(range(4))
    rng.shuffle(rows)
    cols = list(range(4))
    rng.shuffle(cols)
    return [[square[r][c] for c in cols] for r in rows]


def build_cells(cell_specs):
    '''cell_specs: [{geom, basis, xc, charge, spin, tag}] -> indexed cells
    with frozen Latin-square blocks.'''
    rng = random.Random(SEED)
    cells = []
    for index, spec in enumerate(cell_specs):
        cells.append({
            'index': index,
            'geom': spec['geom'],
            'basis': spec['basis'],
            'xc': spec['xc'],
            'charge': spec.get('charge', 0),
            'spin': spec.get('spin', 0),
            'auxbasis': SETTINGS['auxbasis'],
            'tag': spec.get('tag') or
                   f"{os.path.splitext(os.path.basename(spec['geom']))[0]}"
                   f"_{spec['xc']}_{spec['basis']}",
            'blocks': latin_square(rng),
        })
    return cells


def stage1_mediumchon_cells():
    '''The frozen 48-cell stage-1 list: manifest molecule order, LC-168
    basis assignment, method order b3lyp/m062x/wb97mv/r2scan3c.'''
    specs = []
    for mol in MEDIUMCHON:
        for method in METHODS:
            specs.append({
                'geom': os.path.join(MEDIUMCHON_GEOM_DIR, mol + '.xyz'),
                'basis': LC168_BASES[method],
                'xc': method, 'charge': 0, 'spin': 0,
            })
    return specs


# ---------------------------------------------------------------------------
# benchmark v1 (2026-10-30): 8-block two-Latin-square + S-paired protocols
# ---------------------------------------------------------------------------

def blocks_v2(rng):
    '''EIGHT blocks per cell = two independent seeded 4x4 Latin squares
    drawn back-to-back from the one seeded stream: blocks 0-3 are the
    first square, blocks 4-7 the second.  Each square has every role
    exactly once per block (row) and once per position (column), so each
    role accumulates eight formal samples at eight positions.'''
    return latin_square(rng) + latin_square(rng)


def blocks_s(rng):
    '''EIGHT S/B pair blocks: seeded first/second assignment with the
    balance constraint that S leads exactly four blocks and trails four
    (position balance for a two-role design is exactly that).'''
    orders = [['S', 'B']] * 4 + [['B', 'S']] * 4
    rng.shuffle(orders)
    return orders


def load_cohort_cells(cohort_path, geom_dir):
    '''The sealed package's frozen 64 cells, rebased onto the repo copy of
    the public inputs.  cohort.json cell order and every scientific field
    (molecule, xc, basis, charge, spin, geometry_sha256) are carried
    verbatim; the geometry path is rebased so the schedule freezes the
    repo copy the arms will actually read.  Raises on any mismatch
    between the cohort's recorded geometry hash and the file on disk --
    a tampered or re-fetched conformer must fail BEFORE scheduling.'''
    with open(cohort_path, encoding='utf-8') as f:
        cohort = json.load(f)
    if cohort.get('schema') != 'benchmark-cohort-v1':
        raise ValueError(f'cohort schema {cohort.get("schema")!r} != '
                         'benchmark-cohort-v1')
    if cohort.get('status') != 'input_frozen_measurement_not_started':
        raise ValueError(f'cohort status {cohort.get("status")!r} is not '
                         'the frozen package status')
    specs = []
    for cell in cohort['cells']:
        name = os.path.basename(cell['geom'])
        geom = os.path.join(geom_dir, name)
        with open(geom, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != cell['geometry_sha256']:
            raise ValueError(
                f'geometry {geom} sha256 {digest} != cohort-frozen '
                f'{cell["geometry_sha256"]}; refusing to schedule a '
                'non-frozen conformer')
        specs.append({
            'geom': geom,
            'basis': cell['basis'],
            'xc': cell['xc'],
            'charge': cell.get('charge', 0),
            'spin': cell.get('spin', 0),
            'tag': cell['tag'],
            'molecule': cell['molecule'],
            'geometry_sha256': cell['geometry_sha256'],
        })
    return specs


def build_cells_v2(cell_specs, seed):
    '''Indexed cells with frozen eight-block orders (two Latin squares).'''
    rng = random.Random(seed)
    cells = []
    for index, spec in enumerate(cell_specs):
        cells.append({
            'index': index,
            'geom': spec['geom'],
            'basis': spec['basis'],
            'xc': spec['xc'],
            'charge': spec.get('charge', 0),
            'spin': spec.get('spin', 0),
            'auxbasis': SETTINGS_V2['auxbasis'],
            'tag': spec.get('tag') or
                   f"{os.path.splitext(os.path.basename(spec['geom']))[0]}"
                   f"_{spec['xc']}_{spec['basis']}",
            'molecule': spec.get('molecule'),
            'geometry_sha256': spec.get('geometry_sha256'),
            'blocks': blocks_v2(rng),
        })
    return cells


def verify_schedule_v2(schedule, roles=ROLES, two_squares=True):
    '''Structural verification for the 8-block v2/S schedules (pure).  For
    the O/B/C/R protocol each half (blocks 0-3, 4-7) must independently be
    a Latin square: every role once per block and once per position within
    the half.  For the S protocol each block is one S/B pair and S must
    lead exactly four of the eight blocks.  Geometry hashes are re-hashed
    against the frozen value -- a moved/edited conformer fails closed.'''
    two = schedule.get('schema_version') == SCHEMA_V2
    if not two and schedule.get('schema_version') != SCHEMA_S:
        raise ValueError('schema mismatch')
    n_roles = len(roles)
    n_blocks = 8 if two else 8
    if schedule.get('seed') != (SEED_V2 if two else SEED_S):
        raise ValueError('seed must stay the package value')
    for field in ('batch_id', 'source_commit', 'source_root', 'settings',
                  'cells'):
        if not schedule.get(field):
            raise ValueError(f'missing {field}')
    for field in ('conv_tol', 'max_cycle', 'init_guess', 'direct_scf_tol',
                  'grids_level', 'nlcgrids_level', 'small_rho_cutoff',
                  'headroom_mib', 'timeout_s_per_arm'):
        if schedule['settings'].get(field) is None:
            raise ValueError(f'settings missing {field}')
    tags = [cell['tag'] for cell in schedule['cells']]
    if len(set(tags)) != len(tags):
        dup = sorted({t for t in tags if tags.count(t) > 1})
        raise ValueError(f'duplicate cell tags {dup}')
    seen = set()
    for cell in schedule['cells']:
        key = (cell['geom'], cell['basis'], cell['xc'], cell['charge'],
               cell['spin'])
        if key in seen:
            raise ValueError(f'duplicate cell {key}')
        seen.add(key)
        for field in ('index', 'geom', 'basis', 'xc', 'blocks', 'tag',
                      'geometry_sha256'):
            if field not in cell:
                raise ValueError(f'cell {cell.get("index")} missing {field}')
        if not os.path.isfile(cell['geom']):
            raise ValueError(f'cell {cell["index"]}: geometry file '
                             f'{cell["geom"]} does not exist')
        with open(cell['geom'], 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != cell['geometry_sha256']:
            raise ValueError(f'cell {cell["index"]}: geometry sha256 '
                             f'{digest} != frozen {cell["geometry_sha256"]}')
        blocks = cell['blocks']
        if len(blocks) != 8 or any(len(b) != n_roles for b in blocks):
            raise ValueError('eight blocks of %d arms required' % n_roles)
        for row in blocks:
            if sorted(row) != sorted(roles):
                raise ValueError(f'block {row} is not a permutation of '
                                 f'the roles {roles}')
        if two:
            for half in (blocks[:4], blocks[4:]):
                for pos in range(n_roles):
                    column = [b[pos] for b in half]
                    if sorted(column) != sorted(roles):
                        raise ValueError(f'position {pos} not balanced '
                                         f'within a square half: {column}')
        else:
            leads = sum(1 for b in blocks if b[0] == 'S')
            if leads != 4:
                raise ValueError(f'S leads {leads} of 8 blocks; the '
                                 'two-role position balance is exactly 4')
    return True


def verify_dispatch(schedule):
    '''Route a schedule to its verifier by schema: v1 stays v1 (four
    blocks, seed 20261018), v2/S get their own contracts.  A schedule is
    NEVER reinterpreted under another protocol's rules.'''
    schema = schedule.get('schema_version')
    if schema == SCHEMA:
        return verify_schedule(schedule)
    if schema == SCHEMA_V2:
        return verify_schedule_v2(schedule, roles=ROLES)
    if schema == SCHEMA_S:
        return verify_schedule_v2(schedule, roles=S_ROLES)
    raise ValueError(f'unknown schedule schema {schema!r}')


def build_schedule_s(cells_v2, batch_id, source_commit, source_root):
    '''The S-paired schedule reuses the v2 cells verbatim (same cohort,
    same geometry hashes and indices) with 8 S/B pair blocks replacing
    the O/B/C/R squares.  Frozen BEFORE any S measurement.'''
    rng = random.Random(SEED_S)
    cells = []
    for cell in cells_v2:
        c = {k: cell[k] for k in ('index', 'geom', 'basis', 'xc', 'charge',
                                  'spin', 'auxbasis', 'tag', 'molecule',
                                  'geometry_sha256')}
        c['blocks'] = blocks_s(rng)
        cells.append(c)
    return {
        'schema_version': SCHEMA_S,
        'seed': SEED_S,
        'batch_id': batch_id,
        'source_commit': source_commit,
        'source_root': source_root,
        'timing_protocol': 's-paired-v1',
        'settings': SETTINGS_V2,
        'cells': cells,
    }


def verify_schedule(schedule):
    '''Structural verification (pure; unit-testable): Latin property,
settings completeness, cell field completeness.  Raises ValueError.'''
    if schedule.get('schema_version') != SCHEMA:
        raise ValueError('schema mismatch')
    if schedule.get('seed') != SEED:
        raise ValueError('seed must stay the card value 20261018')
    for field in ('batch_id', 'source_commit', 'source_root', 'settings',
                  'cells'):
        if not schedule.get(field):
            raise ValueError(f'missing {field}')
    for field in ('conv_tol', 'max_cycle', 'init_guess', 'direct_scf_tol',
                  'grids_level', 'headroom_mib', 'timeout_s_per_arm'):
        if schedule['settings'].get(field) is None:
            raise ValueError(f'settings missing {field}')
    tags = [cell['tag'] for cell in schedule['cells']]
    if len(set(tags)) != len(tags):
        dup = sorted({t for t in tags if tags.count(t) > 1})
        raise ValueError(f'duplicate cell tags {dup}; sample and document '
                         'paths key on tags')
    seen = set()
    for cell in schedule['cells']:
        key = (cell['geom'], cell['basis'], cell['xc'], cell['charge'],
               cell['spin'])
        if key in seen:
            raise ValueError(f'duplicate cell {key}')
        seen.add(key)
        for field in ('index', 'geom', 'basis', 'xc', 'blocks', 'tag'):
            if field not in cell:
                raise ValueError(f'cell {cell.get("index")} missing {field}')
        if os.path.basename(cell['geom']) == '' or \
                not os.path.isfile(cell['geom']):
            raise ValueError(f'cell {cell["index"]}: geometry file '
                             f'{cell["geom"]} does not exist')
        blocks = cell['blocks']
        if len(blocks) != 4 or any(len(b) != 4 for b in blocks):
            raise ValueError('four blocks of four arms required')
        for row in blocks:
            if sorted(row) != sorted(ROLES):
                raise ValueError(f'block {row} is not a permutation of '
                                 'the four roles')
        for pos in range(4):
            column = [blocks[b][pos] for b in range(4)]
            if sorted(column) != sorted(ROLES):
                raise ValueError(f'position {pos} not balanced across '
                                 f'blocks: {column}')
    return True


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='upstream_schedule',
        description='Freeze a comparison schedule: legacy four-block '
                    'balanced-blocks-v1 (seed 20261018), benchmark-v1 '
                    'eight-block two-Latin-square (seed 20261029) or the '
                    'S-paired variant.')
    p.add_argument('--protocol', choices=['v1', 'v2', 's'], default='v1')
    p.add_argument('--out', required=True)
    p.add_argument('--batch-id', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--cells-json', default=None,
                   help='optional JSON list of extra cell specs '
                        '{geom,basis,xc,charge,spin} appended after the '
                        'frozen stage-1 medium-CHON 48 (staging within '
                        'ONE cumulative batch per the control card)')
    p.add_argument('--cohort-json', default=None,
                   help='v2/s: the sealed package cohort.json (public '
                        'copy); cell list is taken verbatim')
    p.add_argument('--geom-dir', default=None,
                   help='v2/s: directory holding the repo copy of the '
                        'cohort inputs/*.xyz')
    p.add_argument('--emit-s', default=None,
                   help='v2: additionally write the S-paired schedule '
                        'for the same cells to this path')
    args = p.parse_args(argv)

    if args.protocol == 'v1':
        specs = stage1_mediumchon_cells()
        if args.cells_json:
            with open(args.cells_json, encoding='utf-8') as f:
                specs += json.load(f)
        schedule = {
            'schema_version': SCHEMA,
            'seed': SEED,
            'batch_id': args.batch_id,
            'source_commit': args.source_commit,
            'source_root': args.source_root,
            'timing_protocol': 'balanced-blocks-v1',
            'settings': SETTINGS,
            'cells': build_cells(specs),
        }
        verify_schedule(schedule)
        payloads = [(args.out, schedule)]
    else:
        if not args.cohort_json or not args.geom_dir:
            p.error('--cohort-json and --geom-dir are required for '
                    'protocol v2/s')
        specs = load_cohort_cells(args.cohort_json, args.geom_dir)
        schedule = {
            'schema_version': SCHEMA_V2,
            'seed': SEED_V2,
            'batch_id': args.batch_id,
            'source_commit': args.source_commit,
            'source_root': args.source_root,
            'timing_protocol': 'balanced-blocks-v2',
            'settings': SETTINGS_V2,
            'cells': build_cells_v2(specs, SEED_V2),
        }
        verify_schedule_v2(schedule, roles=ROLES)
        payloads = [(args.out, schedule)]
        if args.emit_s:
            payloads.append((args.emit_s, build_schedule_s(
                schedule['cells'], args.batch_id + '-S',
                args.source_commit, args.source_root)))
    for out, sched in payloads:
        verify_dispatch(sched)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(sched, f, indent=1, sort_keys=False)
        digest = hashlib.sha256(open(out, 'rb').read()).hexdigest()
        print(f'wrote {out}: {len(sched["cells"])} cells x '
              f'{len(sched["cells"][0]["blocks"])} blocks, '
              f'sha256 {digest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
