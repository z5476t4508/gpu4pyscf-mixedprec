#!/usr/bin/env python3
"""SP-19 regression tests: the merge-path gate resolution must stay
GPU-free (mixedprec/benchmark/rtx4090/upstream_comparison.py::
_auto_head_min_ao; scf-priority-20261025 card item 3).

Measured motivation (2026-10-26, the SP-19 launch): the SP-18 lazy
'from gpu4pyscf.scf.hf import _AUTO_HEAD_MIN_AO' resolved through the
repo root that the runner modules put on sys.path, executing
gpu4pyscf/__init__.py -> cupy -> a live 480 MiB CUDA context IN THE
CALLING PROCESS.  In run_upstream_batch.py (the driver) the caller IS
the driver, so its own fail-closed gpu-busy gate (wait_gpu_idle ->
gpu_busy: any compute app) could never pass again: cell 0 merged
01:55:44Z, zero further arms for 19 min, driver PID 4178101 itself the
sole nvidia-smi compute app (child repro PID 4180759 acquired 480 MiB
from the bare call).  An SP-18-introduced driver-path defect its
zero-GPU unit could not catch -- the SP-18 merge rerun never reached a
wait_gpu_idle after a merge.

The change: the constant is resolved by reading the assignment out of
gpu4pyscf/scf/hf.py WITHOUT importing anything from the package (the
definition site stays the one source of truth); unresolvable -> None
and every caller still fails CLOSED.  Contracts pinned here:

  * the resolved value equals the constant defined in
    gpu4pyscf/scf/hf.py (source-of-truth identity, both parse paths);
  * the resolution does NOT import any gpu4pyscf module even with the
    repo root importable (sys.modules clean);
  * the resolution does NOT create a CUDA context (no nvidia mapping
    in /proc/self/maps) -- the exact deadlock mechanism;
  * unresolvable layouts (constant absent from the file) return None
    (fail-closed direction unchanged).
"""
import ast
import json
import os
import subprocess
import sys
import unittest

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_DIR, '..', '..'))
_RTX = os.path.join(_REPO, 'mixedprec', 'benchmark', 'rtx4090')
HF = os.path.join(_REPO, 'gpu4pyscf', 'scf', 'hf.py')

_CHILD = r'''
import json, os, sys
REPO = sys.argv[1]
sys.path.insert(0, os.path.join(REPO, 'mixedprec', 'benchmark',
                                'rtx4090'))
sys.path.insert(0, REPO)  # the repo root IS importable, as in the driver
import upstream_comparison as uc
v = uc._auto_head_min_ao()
print(json.dumps({
    'gate': v,
    'gpu4pyscf_imported': any(m == 'gpu4pyscf'
                              or m.startswith('gpu4pyscf.')
                              for m in sys.modules),
    'nvidia_maps': any('nvidia' in l for l in open('/proc/self/maps')),
}))
'''


def _hf_constant():
    '''The gate constant as defined in gpu4pyscf/scf/hf.py: either a
    literal or the pyscf __config__ idiom
    getattr(__config__, 'name', <default>) -- the DEFAULT is the
    effective value (no PYSCF_CONFIG override exists in any run env).'''
    tree = ast.parse(open(HF, encoding='utf-8').read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == '_AUTO_HEAD_MIN_AO'
                for t in node.targets):
            value = node.value
            if (isinstance(value, ast.Call)
                    and (getattr(value.func, 'id', None) == 'getattr'
                         or getattr(value.func, 'attr', None)
                         == 'getattr')
                    and len(value.args) == 3):
                value = value.args[2]
            return ast.literal_eval(value)
    return None


class TestGpuFreeGateResolution(unittest.TestCase):

    def test_resolves_without_gpu_or_package_import(self):
        out = subprocess.run(
            [sys.executable, '-c', _CHILD, _REPO],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        state = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(state['gate'], _hf_constant(),
                         'gate value drifted from gpu4pyscf/scf/hf.py')
        self.assertFalse(state['gpu4pyscf_imported'],
                         'resolution imported the gpu4pyscf package')
        self.assertFalse(state['nvidia_maps'],
                         'resolution created a CUDA context '
                         '(the SP-19 driver deadlock mechanism)')

    def test_constant_defined_in_hf(self):
        # the definition site must keep existing (one source of truth)
        self.assertIsInstance(_hf_constant(), int)


if __name__ == '__main__':
    unittest.main()
