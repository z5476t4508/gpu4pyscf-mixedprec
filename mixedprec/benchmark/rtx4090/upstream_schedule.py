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
        description='Freeze the balanced-blocks-v1 four-arm comparison '
                    'schedule (seed 20261018).')
    p.add_argument('--out', required=True)
    p.add_argument('--batch-id', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--cells-json', default=None,
                   help='optional JSON list of extra cell specs '
                        '{geom,basis,xc,charge,spin} appended after the '
                        'frozen stage-1 medium-CHON 48 (staging within '
                        'ONE cumulative batch per the control card)')
    args = p.parse_args(argv)

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
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(schedule, f, indent=1, sort_keys=False)
    digest = hashlib.sha256(open(args.out, 'rb').read()).hexdigest()
    print(f'wrote {args.out}: {len(schedule["cells"])} cells, '
          f'sha256 {digest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
