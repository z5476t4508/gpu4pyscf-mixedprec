"""Step 6J: where do B3LYP's 34.8s inside _jk_energy_per_atom actually go?

Step 6H found that the two halves of partial_hess_elec swap places with the
functional: for r2SCAN (pure) the JK half is 8.1s and the XC half 32.6s, but for
B3LYP (hybrid) the JK half is 34.8s -- 41% of its 84.4s Hessian and its single
largest remaining phase. Pure functionals short-circuit to _j_energy_per_atom
(J only); hybrids run the full JK path, and the K term is what costs.

Bracketing that half is a no-op (34.8s -> 34.8s, 0.001 cm^-1), so nothing in it
consults precision and the whole 34.8s is still float64. Before porting any of
it, this says which piece to port. The function's own log.timer_debug1 markers
give only two buckets and the second lumps the dm_tensor GEMMs together with the
ejk_int3c2e_ip2 kernel call, which are completely different jobs:

  GEMM-dominated   -> a host-side cast like df/hessian/rhf.py::_get_jk already
                      does; cheap, and the fill_symmetric hazard is known (it is
                      hardcoded to double and returns garbage on float32 input,
                      so the cast must come after it).
  kernel-dominated -> needs an ejk_int3c2e_ip2_f32.cu, i.e. a ~550-line CUDA
                      port modelled on the existing ejk_int3c2e_ip1_f32.cu.

Every piece is timed directly, by call signature, with the device synchronized.
Nothing here is obtained by subtracting one total from another: that is the
mistake that sent me after three wrong targets earlier in this work.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6j_jk_half_profile.py [--xc B3LYP,r2scan]
"""
import argparse
import collections
import time

import cupy
import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.df.hessian import rhf as df_rhf_hess

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'


def sync():
    cupy.cuda.runtime.deviceSynchronize()


class Probe:
    '''wall time and call count per label, device-synchronized.

    Records only while `active` is set. The probed names (contract in
    particular) are module-level and get called from other phases of the same
    Hessian, so an ungated probe would attribute CPHF work to the JK phase --
    the same attribution error that sent me after three wrong targets earlier.
    '''

    def __init__(self):
        self.t = collections.Counter()
        self.n = collections.Counter()
        self.active = False

    def wrap(self, fn, label_of):
        def inner(*a, **kw):
            if not self.active:
                return fn(*a, **kw)
            label = label_of(*a, **kw)
            # a probed function may call another probed function; only the
            # outermost is timed, so shares never sum past 100%
            self.active = False
            sync()
            t0 = time.time()
            try:
                return fn(*a, **kw)
            finally:
                sync()
                self.t[label] += time.time() - t0
                self.n[label] += 1
                self.active = True
        return inner

    def report(self, total, indent='    '):
        accounted = sum(self.t.values())
        for label, t in self.t.most_common():
            print(f'{indent}{label:34s} {t:7.2f}s  x{self.n[label]:<5d}'
                  f'{100 * t / total:5.1f}% of phase', flush=True)
        print(f'{indent}{"[unattributed]":34s} {total - accounted:7.2f}s'
              f'         {100 * (total - accounted) / total:5.1f}% of phase',
              flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--xc', default='B3LYP,r2scan')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)

    orig_contract = df_rhf_hess.contract
    orig_fill = df_rhf_hess.fill_symmetric
    orig_kern = df_rhf_hess.libvhf_rys.ejk_int3c2e_ip2
    orig_factorize_dm = df_rhf_hess._factorize_dm
    orig_factorize_j2c = df_rhf_hess._factorize_j2c

    for xc in args.xc.split(','):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=args.auxbasis)
        mf.conv_tol = 1e-10
        mf.verbose = 0
        mf.kernel()

        probe = Probe()
        # every piece gets its own label; the einsum subscripts distinguish the
        # j3c_oo build from the dm_tensor build without guessing
        df_rhf_hess.contract = probe.wrap(
            orig_contract, lambda subs, *a, **kw: f'contract {subs}')
        df_rhf_hess.fill_symmetric = probe.wrap(
            orig_fill, lambda *a, **kw: 'fill_symmetric')
        df_rhf_hess.libvhf_rys.ejk_int3c2e_ip2 = probe.wrap(
            orig_kern, lambda *a, **kw: 'ejk_int3c2e_ip2 (CUDA)')
        df_rhf_hess._factorize_dm = probe.wrap(
            orig_factorize_dm, lambda *a, **kw: '_factorize_dm')
        df_rhf_hess._factorize_j2c = probe.wrap(
            orig_factorize_j2c, lambda *a, **kw: '_factorize_j2c (metric)')

        phase = {'t': 0.}
        orig_jk = df_rhf_hess._jk_energy_per_atom
        orig_j = df_rhf_hess._j_energy_per_atom

        def timed_jk(*a, **kw):
            sync()
            t0 = time.time()
            probe.active = True
            try:
                return orig_jk(*a, **kw)
            finally:
                probe.active = False
                sync()
                phase['t'] += time.time() - t0
        df_rhf_hess._jk_energy_per_atom = timed_jk

        try:
            sync()
            t0 = time.time()
            mf.Hessian().kernel()
            sync()
            total = time.time() - t0
        finally:
            df_rhf_hess.contract = orig_contract
            df_rhf_hess.fill_symmetric = orig_fill
            df_rhf_hess.libvhf_rys.ejk_int3c2e_ip2 = orig_kern
            df_rhf_hess._factorize_dm = orig_factorize_dm
            df_rhf_hess._factorize_j2c = orig_factorize_j2c
            df_rhf_hess._jk_energy_per_atom = orig_jk
            df_rhf_hess._j_energy_per_atom = orig_j

        ph = phase['t']
        print(f'{xc}/{args.basis}: Hessian {total:.1f}s, '
              f'_jk_energy_per_atom {ph:.1f}s ({100 * ph / total:.0f}%)',
              flush=True)
        probe.report(ph if ph > 0 else total)
        print(flush=True)


if __name__ == '__main__':
    main()
