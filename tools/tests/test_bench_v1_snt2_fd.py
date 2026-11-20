'''CPU-only tests for the SNT-2 finite-difference sentinel harness
(benchv1_snt2_fd.py, card 2026-10-30-benchv1-sentinels SNT-2): geometry
parse/write, Bohr conversion, connectivity-based atom selection per the
card rules, central-difference arithmetic, the synthetic-proc driver
guard, and a full select->run->compare pipeline against a stub runner
(same document shapes as gradient_runner's auto lane).  No GPU, no
pyscf.'''

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fdm = _load('benchv1_snt2_fd')

# stub gradient_runner: E = g.r + 0.5 k |r - r0|^2 per atom, coordinates
# in Bohr, cfg keyed by cell tag (parsed from the displacement filename);
# writes gradient_runner-shaped documents and appends to a counter file
# so the tests can prove which invocations actually executed.
STUB_RUNNER = r'''
import json, os, sys
args = sys.argv[1:]
def opt(flag):
    return args[args.index(flag) + 1]
geom = opt('--geom'); out = opt('--out')
cfg = json.load(open(os.environ['SNT2_STUB_CFG']))
tag = os.path.basename(geom).split('_a')[0]
c = cfg['by_tag'][tag]
g, base, k = c['gradient'], c['base_bohr'], cfg.get('quadratic_k', 1e-6)
lines = open(geom).read().splitlines()
nat = int(lines[0].split()[0])
xyz = [[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + nat]]
bohr = 1.0 / 0.52917721092
e = 0.0
for gi, (r, r0) in enumerate(zip(xyz, base)):
    rb = [v * bohr for v in r]
    rb0 = [v * bohr for v in r0]
    e += sum(gj * v for gj, v in zip(g[gi], rb))
    e += 0.5 * k * sum((a - b) ** 2 for a, b in zip(rb, rb0))
doc = {'rows': [{'row_key': {'lane': 'auto'}, 'status': 'ok',
                 'result': {'e_tot': e, 'gradient': g,
                            'repeats': [{'e_tot': e} for _ in range(3)]}}]}
json.dump(doc, open(out, 'w'))
with open(cfg['counter'], 'a') as f:
    f.write(geom + '\n')
'''


def toy_cytosine():
    '''N1 (ring N, one H, two C) + amino N with two H + a carbonyl-like O.'''
    syms = ['N', 'C', 'C', 'H', 'N', 'H', 'H', 'O', 'C']
    coords = [[0.0, 0.0, 0.0],      # 0 N1 (ring: 2 C, 1 H)
              [1.40, 0.0, 0.0],     # 1 C
              [-1.40, 0.0, 0.0],    # 2 C
              [0.0, 1.01, 0.0],     # 3 H on N1
              [0.0, 0.0, 4.0],      # 4 amino N
              [1.02, 0.0, 4.0],     # 5 H
              [-1.02, 0.0, 4.0],    # 6 H
              [0.0, -6.0, 0.0],     # 7 O (no H)
              [0.0, -7.1, 0.0]]     # 8 C
    return syms, coords


def toy_acetaminophen():
    '''One hydroxyl O (with H), one carbonyl O (no H), exactly one N.'''
    syms = ['O', 'H', 'O', 'N', 'C', 'C']
    coords = [[0.0, 0.0, 0.0],      # 0 OH oxygen
              [0.0, 0.96, 0.0],     # 1 phenolic H
              [5.0, 0.0, 0.0],      # 2 carbonyl O
              [0.0, 5.0, 0.0],      # 3 the amide N
              [1.4, 0.0, 0.0],      # 4 C
              [0.0, 6.4, 0.0]]      # 5 C
    return syms, coords


class TestGeometry(unittest.TestCase):
    def test_xyz_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'g.xyz')
            fdm.write_xyz(p, 'comment', ['O', 'H'],
                          [[0.0, 0.0, 0.0], [0.0, 0.96, 0.0]])
            syms, coords = fdm.parse_xyz(p)
        self.assertEqual(syms, ['O', 'H'])
        self.assertAlmostEqual(coords[1][1], 0.96)

    def test_displaced_bohr_conversion(self):
        base = [[1.0, 2.0, 3.0]] * 2
        d = fdm.displaced(base, 0, 1, 0.001)
        self.assertAlmostEqual(d[0][1] - base[0][1], 0.001 * fdm.BOHR_ANG,
                               places=15)
        self.assertEqual(d[1], base[1])  # untouched atoms exact

    def test_displaced_negative_sign(self):
        d = fdm.displaced([[0.0, 0.0, 0.0]], 0, 2, -0.002)
        self.assertAlmostEqual(d[0][2], -0.002 * fdm.BOHR_ANG, places=15)


class TestFDMath(unittest.TestCase):
    def test_central_difference_quadratic_exact(self):
        # E = E0 + g.x + 0.5 k x^2 -> central difference returns g exactly
        e_plus = -10.0 + 0.3 * 0.001 + 0.5 * 2.0 * 0.001 ** 2
        e_minus = -10.0 - 0.3 * 0.001 + 0.5 * 2.0 * 0.001 ** 2
        # (cancellation-limited to ~1e-12, not exactly g in floats)
        self.assertAlmostEqual(fdm.central_difference(e_plus, e_minus, 0.001),
                               0.3, delta=1e-10)

    def test_central_difference_cubic_truncation(self):
        # E = x^3 -> (h^3-(-h)^3)/(2h) = h^2 exactly: the O(h^2) truncation
        self.assertAlmostEqual(fdm.central_difference(0.002 ** 3,
                                                      (-0.002) ** 3, 0.002),
                               0.002 ** 2, places=15)


class TestSelectors(unittest.TestCase):
    def test_cytosine_selection(self):
        syms, coords = toy_cytosine()
        grad = [[0.0, 0.0, 0.0] for _ in syms]
        grad[5] = [0.3, 0.4, 0.0]   # norm 0.5
        grad[6] = [2.0, 0.0, 0.0]   # norm 2.0 -> argmax
        atoms = fdm.select_cytosine(syms, coords, grad)
        self.assertEqual(atoms[0]['index'], 3)   # the N1 H
        self.assertEqual(atoms[0]['element'], 'H')
        self.assertEqual(atoms[1]['index'], 6)   # the largest-|g| amino H
        norms = atoms[1]['norms_considered']
        self.assertAlmostEqual(norms['5'], 0.5)
        self.assertAlmostEqual(norms['6'], 2.0)

    def test_cytosine_argmax_follows_gradient(self):
        syms, coords = toy_cytosine()
        grad = [[0.0] * 3 for _ in syms]
        grad[5] = [3.0, 0.0, 0.0]   # now atom 5 wins
        atoms = fdm.select_cytosine(syms, coords, grad)
        self.assertEqual(atoms[1]['index'], 5)

    def test_cytosine_fail_on_two_h_ring_n(self):
        syms, coords = toy_cytosine()
        syms = syms + ['H']
        coords = coords + [[0.0, -1.01, 0.0]]  # second H on N1
        with self.assertRaises(SystemExit):
            fdm.select_cytosine(syms, coords, [[0.0] * 3] * len(syms))

    def test_acetaminophen_selection(self):
        syms, coords = toy_acetaminophen()
        atoms = fdm.select_acetaminophen(syms, coords, [[0.0] * 3] * len(syms))
        self.assertEqual(atoms[0]['index'], 1)   # phenolic H
        self.assertEqual(atoms[0]['element'], 'H')
        self.assertEqual(atoms[1]['index'], 3)   # the single (amide) N
        self.assertEqual(atoms[1]['element'], 'N')

    def test_acetaminophen_fail_on_second_n(self):
        syms, coords = toy_acetaminophen()
        syms = syms + ['N']
        coords = coords + [[9.0, 9.0, 0.0]]
        with self.assertRaises(SystemExit):
            fdm.select_acetaminophen(syms, coords, [[0.0] * 3] * len(syms))

    def test_non_chon_fail_closed(self):
        syms, coords = toy_cytosine()
        syms[8] = 'S'
        with self.assertRaises(SystemExit):
            fdm.connectivity(syms, coords)


class TestRealCohortGeometries(unittest.TestCase):
    '''The REAL sealed-cohort inputs must satisfy the card's atom rules
    (connectivity only; the cytosine ammonium-H argmax falls back to the
    first tied winner under a uniform synthetic gradient).'''

    def test_real_cytosine_selection(self):
        syms, coords = fdm.parse_xyz(
            str(_DIR / 'benchv1_cohort' / 'inputs' / 'cytosine.xyz'))
        self.assertEqual(len(syms), 13)
        grad = [[0.1, 0.1, 0.1] for _ in syms]
        atoms = fdm.select_cytosine(syms, coords, grad)
        self.assertEqual(atoms[0]['element'], 'H')
        self.assertEqual(atoms[1]['element'], 'H')
        bonds = fdm.connectivity(syms, coords)
        # atom 1's H is bonded to a ring N (2 C neighbors); atom 2's H to
        # the NH2 nitrogen
        h1 = atoms[0]['index']
        n1 = next(j for j in bonds[h1] if syms[j] == 'N')
        self.assertGreaterEqual(
            sum(1 for j in bonds[n1] if syms[j] == 'C'), 2)
        h2 = atoms[1]['index']
        n2 = next(j for j in bonds[h2] if syms[j] == 'N')
        self.assertEqual(sum(1 for j in bonds[n2] if syms[j] == 'H'), 2)

    def test_real_acetaminophen_selection(self):
        syms, coords = fdm.parse_xyz(
            str(_DIR / 'benchv1_cohort' / 'inputs' / 'acetaminophen.xyz'))
        self.assertEqual(len(syms), 20)
        self.assertEqual(syms.count('N'), 1)
        atoms = fdm.select_acetaminophen(syms, coords,
                                         [[0.0] * 3] * len(syms))
        self.assertEqual(atoms[0]['element'], 'H')   # phenolic H
        self.assertEqual(atoms[1]['element'], 'N')   # the amide N
        bonds = fdm.connectivity(syms, coords)
        o = next(j for j in bonds[atoms[0]['index']] if syms[j] == 'O')
        self.assertEqual(sum(1 for j in bonds[o] if syms[j] == 'H'), 1)


class TestGuard(unittest.TestCase):
    def _fake_proc(self, entries):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td)
        proc = os.path.join(td, 'proc')
        os.makedirs(os.path.join(proc, str(os.getpid())))
        with open(os.path.join(proc, str(os.getpid()), 'cmdline'),
                  'wb') as f:
            f.write(b'python\0benchv1_snt2_fd.py\0--stage\0run\0')
        for pid, cmd in entries:
            os.makedirs(os.path.join(proc, str(pid)))
            with open(os.path.join(proc, str(pid), 'cmdline'), 'wb') as f:
                f.write(cmd)
        return proc

    def test_guard_refuses_on_driver_marker(self):
        proc = self._fake_proc(
            [(42, b'/usr/bin/python\0run_upstream_batch.py\0--schedule\0')])
        with self.assertRaises(SystemExit) as cm:
            fdm.guard_no_gpu_work(proc_root=proc)
        self.assertIn('GPU workload alive', str(cm.exception))

    def test_guard_refuses_on_arm_marker(self):
        proc = self._fake_proc(
            [(7, b'python\0/snap/upstream_comparison.py\0--role\0x\0')])
        with self.assertRaises(SystemExit):
            fdm.guard_no_gpu_work(proc_root=proc)

    def test_guard_passes_when_clean(self):
        proc = self._fake_proc(
            [(9, b'python\0some_unrelated_tool.py\0')])
        fdm.guard_no_gpu_work(proc_root=proc)  # no raise


class TestPipelineStub(unittest.TestCase):
    '''Full select -> run -> compare against the stub runner.  The stub's
    E = g.r + 0.5 k |r-r0|^2 (Bohr, k=1e-6) makes FD reproduce the
    analytic gradient to ~|r0|*k*h ~ 1e-5 Eh/Bohr, well inside the 3e-4
    diagnostic threshold.'''

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.td)
        py = pathlib.Path(self.td)
        geoms = {'cytosine_b3lyp_def2-svp': toy_cytosine(),
                 'acetaminophen_wb97mv_def2-svp': toy_acetaminophen()}
        self.gradients = {
            tag: [[0.02 * (i + 1), -0.01 * (i + 1), 0.03 * (i + 1)]
                  for i in range(len(syms))]
            for tag, (syms, _) in geoms.items()}
        cells = []
        for idx, (tag, (syms, coords)) in enumerate(geoms.items()):
            gp = py / f'{tag}.xyz'
            fdm.write_xyz(str(gp), 'toy', syms, coords)
            cells.append({'index': idx, 'tag': tag, 'geom': str(gp),
                          'basis': 'def2-svp', 'xc': 'b3lyp', 'charge': 0,
                          'spin': 0, 'auxbasis': 'def2-universal-jkfit',
                          'geometry_sha256': hashlib.sha256(
                              gp.read_bytes()).hexdigest()})
        self.schedule = py / 'schedule.json'
        self.schedule.write_text(json.dumps({
            'schema_version': 'upstream-comparison-schedule-v2',
            'settings': {'conv_tol': 1e-10, 'max_cycle': 50,
                         'init_guess': 'minao', 'direct_scf_tol': 1e-14,
                         'grids_level': 3, 'nlcgrids_level': 3,
                         'auxbasis': 'def2-universal-jkfit'},
            'cells': cells}))
        batch = py / 'batch'
        (batch / 'cells').mkdir(parents=True)
        for tag, g in self.gradients.items():
            (batch / 'cells' / f'{tag}_optimized_auto.json').write_text(
                json.dumps({'rows': [
                    {'row_key': {'lane': 'auto'}, 'status': 'ok',
                     'result': {'e_tot': -10.0, 'gradient': g}}]}))
        self.batch = str(batch)
        self.counter = py / 'counter.txt'
        self.stub_cfg = py / 'stubcfg.json'
        self.stub_cfg.write_text(json.dumps({
            'counter': str(self.counter), 'quadratic_k': 1e-6,
            'by_tag': {t: {'gradient': self.gradients[t],
                           'base_bohr': geoms[t][1]}
                       for t in geoms}}))
        self.stub = py / 'stub_runner.py'
        self.stub.write_text(STUB_RUNNER)
        self.sent = py / 'sent'
        os.environ['SNT2_STUB_CFG'] = str(self.stub_cfg)

    def _argv(self, stage, extra=()):
        return ['--stage', stage, '--schedule', str(self.schedule),
                '--batch-root', self.batch, '--sent-root', str(self.sent),
                '--runner-override', f'{sys.executable}::{self.stub}',
                '--no-guard'] + list(extra)

    def test_full_pipeline_and_skip_existing(self):
        fdm.main(self._argv('select'))
        plan = json.loads((self.sent / 'plan.json').read_text())
        self.assertEqual(len(plan['cells']), 2)
        for cell in plan['cells']:
            self.assertEqual(len(cell['runs']), 24)  # 2 atoms x 3 axes x
            # 2 steps x 2 signs
        # card atom rules on the toy geometries
        self.assertEqual(plan['cells'][0]['atoms'][0]['index'], 3)
        self.assertEqual(plan['cells'][0]['atoms'][1]['index'], 6)
        self.assertEqual(plan['cells'][1]['atoms'][0]['index'], 1)
        self.assertEqual(plan['cells'][1]['atoms'][1]['index'], 3)
        fdm.main(self._argv('run'))
        self.assertEqual(len(self.counter.read_text().splitlines()), 48)
        # second run pass skips every existing ok output (counter frozen)
        fdm.main(self._argv('run'))
        self.assertEqual(len(self.counter.read_text().splitlines()), 48)
        fdm.main(self._argv('compare'))
        res = json.loads((self.sent / 'fd-results.json').read_text())
        self.assertTrue(res['all_within_3x_nominal_floor'])
        self.assertLess(res['max_abs_delta_overall'], 3e-4)
        comp = res['cells'][0]['steps']['0.0010']['components']
        self.assertEqual(len(comp), 6)  # 2 atoms x 3 axes
        for c in comp:
            self.assertAlmostEqual(c['fd'], c['analytic'], delta=2e-5)
        md = (self.sent / 'fd-results.md').read_text()
        self.assertIn('Diagnostic only', md)
        self.assertIn('cytosine_b3lyp_def2-svp', md)

    def test_compare_fail_closed_on_missing_run(self):
        fdm.main(self._argv('select'))
        fdm.main(self._argv('run'))
        runs = sorted((self.sent / 'runs').glob('*.json'))
        runs[0].unlink()
        with self.assertRaises(SystemExit) as cm:
            fdm.main(self._argv('compare'))
        self.assertIn('missing FD run', str(cm.exception))

    def test_compare_fail_closed_on_non_ok_row(self):
        fdm.main(self._argv('select'))
        fdm.main(self._argv('run'))
        runs = sorted((self.sent / 'runs').glob('*.json'))
        doc = json.loads(runs[0].read_text())
        doc['rows'][0]['status'] = 'scf_not_converged'
        runs[0].write_text(json.dumps(doc))
        with self.assertRaises(SystemExit) as cm:
            fdm.main(self._argv('compare'))
        self.assertIn('status=', str(cm.exception))

    def test_select_fail_closed_on_geometry_hash(self):
        sched = json.loads(self.schedule.read_text())
        sched['cells'][0]['geometry_sha256'] = '0' * 64
        self.schedule.write_text(json.dumps(sched))
        with self.assertRaises(SystemExit) as cm:
            fdm.main(self._argv('select'))
        self.assertIn('sha256', str(cm.exception))

    def test_run_refuses_when_guard_trips(self):
        # with the guard enabled, a live batch driver/arm in the real
        # /proc must abort before any runner subprocess; skipped when
        # the batch has already exited (nothing to refuse against)
        fdm.main(self._argv('select'))
        live = 0
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                cmd = open(f'/proc/{pid}/cmdline', 'rb').read()
            except OSError:
                continue
            if b'run_upstream_batch.py' in cmd or \
                    b'upstream_comparison.py' in cmd:
                live += 1
        if not live:
            self.skipTest('no live batch driver in this environment')
        argv = ['--stage', 'run', '--schedule', str(self.schedule),
                '--batch-root', self.batch, '--sent-root', str(self.sent),
                '--runner-override', f'{sys.executable}::{self.stub}']
        with self.assertRaises(SystemExit):
            fdm.main(argv)


if __name__ == '__main__':
    unittest.main()
