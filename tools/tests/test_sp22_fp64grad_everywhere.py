#!/usr/bin/env python3
"""SP-22 regression tests: the plain-auto GRADIENT resolves fp64 at ALL
nao (gpu4pyscf/grad/rhf.py::_resolve_precision; scf-priority-20261025
card item 3; named by SP-21's preregistered JUSTIFIED_SP22 clause).

Measured motivation (SP-21, prereg 7dcd6c23, same-day paired A/B on the
SAME auto head): R = t_grad(fp64grad)/t_grad(fp32 shim) = 0.971 / 1.000
/ 0.942 / 0.952 at 270/318/348/522 AO (median 0.961) -- the fp32
gradient casts NEVER pay and cost 3.2e-7..6.7e-7 Eh/Bohr of isolated
gradient error.  SP-16 had already adopted the same resolution below
_AUTO_HEAD_MIN_AO; SP-22 removes the size condition from the plain-
'auto' branch only.

Companion conditioning (the SP-18 pattern): the lane-effectiveness
rules that treat a zero-grad-hit auto row as an ineffective
mixed-precision lane are conditioned on the by-design semantics --
gradient_runner.FP32_GRAD_LANES loses 'auto' (fp32-grad / auto-verify /
auto-strict keep the strict rule), and schema_validator
_check_grad_lane_effectiveness's zero-grad-hit lane tuple loses 'auto'
identically.

Contracts pinned here:
  * source: the plain-'auto' branch of _resolve_precision sets 'fp64'
    with NO nao condition and no _AUTO_HEAD_MIN_AO import; the
    auto+verify / auto+strict branches are untouched;
  * semantics (CPU construction, no SCF run): plain auto resolves
    'fp64' below the gate AND above it (threshold-lowered); verify /
    strict still 'fp32' at every size; explicit overrides win;
  * lane tables: 'auto' not in gradient_runner.FP32_GRAD_LANES; the
    schema R3 grad clause skips auto rows, still fails a zero-hit
    verdict-pass 'fp32-grad' row and still fails a contaminated
    'auto-fp64-grad' row;
  * bitwise (GPU, in-process same-head pair, threshold trick): with
    the gate threshold lowered so benzene/svp sits ABOVE it, ONE auto
    SCF head is converged and the inherited plain-auto gradient's
    delta must sit in the fp64-vs-fp64 in-process envelope (plus the
    control pair); at 7e6ee9e the inherited gradient runs the
    fp32 dtype-shim there and the delta is the 1e-7-class cast
    signature = the RED.
"""
import os
import re
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
_RTX = os.path.join(REPO, 'mixedprec', 'benchmark', 'rtx4090')
sys.path.insert(0, _RTX)

try:
    import numpy  # noqa: F401
    import cupy as cp  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

GRAD_SRC = os.path.join(REPO, 'gpu4pyscf', 'grad', 'rhf.py')
GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'benzene.xyz')
PY = sys.executable


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\n    def |\nclass |\Z)', src,
                  re.S)
    return m.group(0) if m else None


def _clean_gpu_env():
    return {k: v for k, v in os.environ.items()
            if not k.startswith('GPU4PYSCF_')}


@unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
class TestSourceContract(unittest.TestCase):
    def test_auto_branch_fp64_no_size_condition(self):
        src = _func_src(GRAD_SRC, '_resolve_precision')
        self.assertIsNotNone(src)
        self.assertIn("if mode == 'auto':", src)
        # the SP-22 form: fp64 resolution with no nao condition and no
        # gate import in the plain-auto branch
        auto_branch = re.search(
            r"if mode == 'auto':.*?(?=\n        elif |\n        if mode)",
            src, re.S)
        self.assertIsNotNone(auto_branch, src)
        self.assertNotIn('_AUTO_HEAD_MIN_AO', auto_branch.group(0))
        self.assertIn("mode = 'fp64'", auto_branch.group(0))

    def test_verify_strict_branches_untouched(self):
        src = _func_src(GRAD_SRC, '_resolve_precision')
        self.assertIn("elif mode == 'auto+verify':", src)
        self.assertIn("elif mode == 'auto+strict':", src)
        # no size gate anywhere in the function anymore
        self.assertNotIn('self.mol.nao < _AUTO_HEAD_MIN_AO', src)


@unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
class TestSemantics(unittest.TestCase):
    '''_resolve_precision lane table (CPU construction, no SCF run).
    benzene/def2-svp = 114 AO: below the 250 gate natively, ABOVE it
    with the threshold lowered to 60.'''

    def _resolve(self, base_mode, grad_mode=None, threshold=None):
        import gpu4pyscf.scf.hf as hf
        from pyscf import gto
        from gpu4pyscf.dft import RKS
        saved = getattr(hf, '_AUTO_HEAD_MIN_AO', None)
        try:
            if threshold is not None:
                if saved is None:
                    self.fail('gate constant absent (pristine source)')
                hf._AUTO_HEAD_MIN_AO = threshold
            mol = gto.M(atom=GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='b3lyp')
            if base_mode is not None:
                mf.precision_mode = base_mode
            g = mf.Gradients()
            if grad_mode is not None:
                g.precision_mode = grad_mode
            return g._resolve_precision()
        finally:
            if threshold is not None and saved is not None:
                hf._AUTO_HEAD_MIN_AO = saved

    def test_below_gate_auto_resolves_fp64(self):
        self.assertEqual(self._resolve('auto'), 'fp64')

    def test_above_gate_auto_also_resolves_fp64(self):
        # the SP-22 RED clause: at 7e6ee9e this resolves 'fp32'
        self.assertEqual(self._resolve('auto', threshold=60), 'fp64',
                         'SP-22: plain-auto gradient resolves fp64 at '
                         'every nao (threshold-lowered above-gate form)')

    def test_verify_strict_keep_fp32(self):
        self.assertEqual(self._resolve('auto+verify'), 'fp32')
        self.assertEqual(self._resolve('auto+strict'), 'fp32')

    def test_explicit_overrides_win(self):
        self.assertEqual(self._resolve('auto', grad_mode='fp32'), 'fp32')
        self.assertEqual(self._resolve('auto', grad_mode='fp64'), 'fp64')

    def test_default_none_stays_none(self):
        self.assertIsNone(self._resolve(None))


@unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
class TestLaneEffectivenessConditioning(unittest.TestCase):
    '''The by-design zero-grad-hit auto row must not be failed by the
    lane-effectiveness rules; the fp32-requesting lanes keep them.'''

    ROW_AUTO = {
        'row_key': {'geom': 'x.xyz', 'basis': 'def2-svp', 'xc': 'b3lyp',
                    'lane': 'auto'},
        'status': 'ok',
        'result': {
            'gradient_mode': 'fp64',
            'fp32_entry_hits': {'df_jk': 16, 'eval_rho': 336,
                                'xc_grid': 16, 'grad_ip1': 0,
                                'other': 14}},
        'accuracy': {'verdict': 'pass'},
    }
    ROW_FP32GRAD = {
        'row_key': {'geom': 'x.xyz', 'basis': 'def2-svp', 'xc': 'b3lyp',
                    'lane': 'fp32-grad'},
        'status': 'ok',
        'result': {
            'gradient_mode': 'fp64',
            'fp32_entry_hits': {'df_jk': 0, 'eval_rho': 0,
                                'xc_grid': 0, 'grad_ip1': 0,
                                'other': 0}},
        'accuracy': {'verdict': 'pass'},
    }
    ROW_CONTAMINATED = {
        'row_key': {'geom': 'x.xyz', 'basis': 'def2-svp', 'xc': 'b3lyp',
                    'lane': 'auto-fp64-grad'},
        'status': 'ok',
        'result': {
            'gradient_mode': 'fp32',
            'fp32_entry_hits': {'df_jk': 0, 'eval_rho': 0,
                                'xc_grid': 0, 'grad_ip1': 4,
                                'other': 0}},
        'accuracy': {'verdict': 'pass'},
    }

    def test_gradient_runner_lane_table(self):
        import gradient_runner
        self.assertNotIn('auto', gradient_runner.FP32_GRAD_LANES,
                         "SP-22: 'auto' no longer requests fp32 gradient "
                         'kernels; the by-design fp64 grad must not fail '
                         'the derive rule')
        for lane in ('fp32-grad', 'auto-verify', 'auto-strict'):
            self.assertIn(lane, gradient_runner.FP32_GRAD_LANES)

    def test_r3_skips_by_design_auto_row(self):
        import schema_validator as sv
        issues = sv.Issues()
        sv._check_grad_lane_effectiveness(issues, self.ROW_AUTO, 't')
        tags = {it['tag'] for it in issues.items}
        self.assertNotIn(
            'grad-lane-ineffective-pass', tags,
            'SP-22: a zero-grad-hit verdict-pass auto row with positive '
            'head evidence is the by-design lane, not an ineffective one')

    def test_r3_keeps_fp32grad_strict_rule(self):
        import schema_validator as sv
        issues = sv.Issues()
        sv._check_grad_lane_effectiveness(issues, self.ROW_FP32GRAD, 't')
        tags = {it['tag'] for it in issues.items}
        self.assertIn('grad-lane-ineffective-pass', tags)

    def test_r3_keeps_contamination_rule(self):
        import schema_validator as sv
        issues = sv.Issues()
        sv._check_grad_lane_effectiveness(
            issues, self.ROW_CONTAMINATED, 't')
        tags = {it['tag'] for it in issues.items}
        self.assertIn('grad-lane-contaminated', tags)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestAboveGateSameHeadPair(unittest.TestCase):
    '''The isolated SP-22 contract: with the gate threshold lowered to
    60 in the CHILD, benzene/svp (114 AO) sits ABOVE both the SCF and
    gradient gates; ONE auto SCF head is converged, then THREE
    gradients run on it IN-PROCESS: the inherited plain-auto gradient
    (the branch SP-22 changes), an explicit fp64 override, and a second
    fp64 control.  Post-SP-22 the inherited gradient resolves fp64 and
    its delta must sit in the fp64-vs-fp64 in-process envelope (the
    gradient is not bitwise reproducible even fp64-vs-fp64 per launch,
    ~1e-14, the SP-16/SP-21 class).  At 7e6ee9e the inherited gradient runs
    the fp32 dtype-shim and the delta is the 1e-7-class cast signature
    = RED.  (e_tot auto-vs-fp64-lane bitwise equality is NOT asserted
    here: above the SCF gate the auto head legitimately differs from
    the fp64 head at the 1e-12 class -- the head is untouched by
    SP-22.)'''

    CHILD = r'''
import sys
sys.path.insert(0, @REPO@)
import gpu4pyscf.scf.hf as hf
hf._AUTO_HEAD_MIN_AO = 60
from pyscf import gto
from gpu4pyscf.dft import RKS
import numpy as np
mol = gto.M(atom=@GEOM@, basis='def2-svp', verbose=0)
mf = RKS(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.grids.level = 3
mf.precision_mode = 'auto'
mf.kernel()
g_auto = mf.Gradients().kernel()                     # inherited: SP-22 branch
g_ov = mf.Gradients(); g_ov.precision_mode = 'fp64'  # explicit override
g_o = g_ov.kernel()
g_c = mf.Gradients(); g_c.precision_mode = 'fp64'    # fp64 control
g_ctrl = g_c.kernel()
print('E_TOT %.17g' % mf.e_tot, 'CONV', mf.converged)
for tag, g in (('AUTO', g_auto), ('FP64', g_o), ('CTRL', g_ctrl)):
    print(tag + ' ' + ' '.join('%.17g' % x for x in np.asarray(g).ravel()))
'''

    def test_above_gate_inherited_auto_grad_bitwise_fp64(self):
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.py',
                                         delete=False) as f:
            f.write(self.CHILD
                    .replace('@REPO@', repr(REPO))
                    .replace('@GEOM@', repr(GEOM)))
            script = f.name
        try:
            r = subprocess.run(
                [PY, script], capture_output=True, text=True,
                timeout=600, env=_clean_gpu_env())
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            lines = r.stdout.strip().splitlines()
            self.assertIn('CONV True', lines[-4])
            vals = {}
            for ln in lines[-3:]:
                tag = ln.split()[0]
                vals[tag] = [float(x) for x in ln.split()[1:]]
            self.assertTrue({'AUTO', 'FP64', 'CTRL'} <= set(vals))
        finally:
            os.unlink(script)
        ctrl = max(abs(a - b) for a, b in
                   zip(vals['CTRL'], vals['FP64']))
        d = max(abs(a - b) for a, b in zip(vals['AUTO'], vals['FP64']))
        # the gradient is not bitwise reproducible even fp64-vs-fp64
        # in-process (per-launch reduction order; the SP-16/SP-21
        # measured class ~1e-14), so the contract is ENVELOPE equality:
        # the inherited-auto delta must sit in the fp64-vs-fp64 band;
        # the pre-change dtype-shim signature is 1e-7-class, 4+ orders
        # above it
        self.assertLessEqual(d, max(5 * ctrl, 1e-14),
                             f'SP-22: inherited-auto-vs-fp64 gradient '
                             f'delta {d:.3e} must sit in the '
                             f'fp64-vs-fp64 envelope {ctrl:.3e} (5x '
                             f'band); at 7e6ee9e this is the 1e-7-class '
                             f'fp32 dtype-shim signature')
        self.assertLessEqual(d, 1e-12)


if __name__ == '__main__':
    unittest.main()
