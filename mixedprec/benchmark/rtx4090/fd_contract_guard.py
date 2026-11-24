"""Fail-closed derivative-contract guard for finite-difference sentinels.

GRIDRESP-CONTRACT-01 (effects-index entry 265, 2026-11-03) established the
DERIVATIVE_CONTRACT_MISMATCH_CLASS: a finite-difference energy probe rebuilds
the atom-centered integration grids at every displaced geometry (full grid
response implicit), while a default analytic gradient object carries
grid_response=False (pyscf/grad/rks.py; the XC grid-response term omitted).
Comparing the two at tight tolerances mixes derivative contracts -- measured
worst-component scale 1.396e-4 Eh/Bohr with the term omitted vs 1.558e-6 with
it included (uracil/r2SCAN-3c, def2-mtzvpp, grids level 3); a loose FD
diagnostic band (>= 3e-4) silently absorbs the term and stays contract-blind.

Every future FD-vs-analytic comparison in this project must EITHER set
grid_response=True on its analytic gradient object OR declare an explicit
term budget covering the omitted grid-response scale, and must record which
contract it measured. This module enforces that rule at construction time:

  require_fd_contract(...) -> contract record dict (for the run manifest),
  fails closed (ValueError) when neither branch is satisfied.

Adoption contract for future FD sentinel drivers (e.g. successors of
benchv1_snt2_fd.py, whose frozen 2026-10-30 record predates this finding and
is reconciled by the append-only addendum SNT2_GRIDRESP_RECONCILIATION_
ADDENDUM_20261103 in the snt2 evidence dir): build the analytic gradient via

    g = gradient_with_contract(mf, grid_response=..., term_budget_eh_bohr=...)

or call require_fd_contract() before any FD-vs-analytic comparison and embed
the returned record in the output JSON.
"""

import math

FD_CONTRACT_GUARD_SCHEMA = "fd-contract-guard-v1"

#: measured worst-component scale of the omitted grid-response term
#: (GRIDRESP-CONTRACT-01, uracil r2SCAN-3c def2-mtzvpp grids-3); a declared
#: term budget below this class scale is rejected as not covering the known
#: defect class unless the caller pins a measured smaller cell-specific scale
#: via measured_class_scale_eh_bohr.
MEASURED_CLASS_SCALE_EH_BOHR = 1.396027962642203e-04


def require_fd_contract(analytic_grid_response=None,
                        term_budget_eh_bohr=None,
                        source="",
                        measured_class_scale_eh_bohr=None):
    """Return the FD contract record, or raise ValueError (fail-closed).

    Pass grid_response=True when the analytic gradient includes the full
    grid response. Otherwise pass an explicit term_budget_eh_bohr (the
    caller-declared bound on the omitted grid-response scale for the cells
    at hand); it must be a finite positive float no smaller than the
    measured class scale, unless a smaller cell-specific scale was itself
    measured and is passed as measured_class_scale_eh_bohr with its
    evidence noted in the returned record. Non-finite values (NaN, +/-inf)
    are rejected: an infinite budget vacuously covers the contract class
    and a NaN one covers nothing, so neither is a declared bound.
    """
    if not source:
        raise ValueError(
            "fd_contract_guard: source (driver/stage name) is required so "
            "the contract record is traceable to the comparison that "
            "produced it")
    if analytic_grid_response is True:
        if term_budget_eh_bohr is not None:
            raise ValueError(
                "fd_contract_guard: term_budget_eh_bohr is redundant when "
                "analytic_grid_response=True; pass only one branch")
        return {
            "schema": FD_CONTRACT_GUARD_SCHEMA,
            "source": source,
            "contract": "grid_response_true",
            "analytic_grid_response": True,
            "term_budget_eh_bohr": None,
        }
    if analytic_grid_response is not None:
        raise ValueError(
            "fd_contract_guard: analytic_grid_response must be True "
            "(grid response included) or None (term-budget branch); "
            f"got {analytic_grid_response!r}. grid_response=False is the "
            "measured contract-mismatch class (GRIDRESP-CONTRACT-01) and "
            "is never a valid declared contract for an FD comparison")
    if term_budget_eh_bohr is None:
        raise ValueError(
            "fd_contract_guard: neither branch satisfied -- an FD-vs-"
            "analytic comparison must set grid_response=True on the "
            "analytic gradient OR declare term_budget_eh_bohr (GRIDRESP-"
            "CONTRACT-01: FD rebuilds grids per displacement; a default "
            "grid_response=False analytic gradient omits the XC grid-"
            "response term, measured worst 1.396e-4 Eh/Bohr)")
    if isinstance(term_budget_eh_bohr, bool) or \
            not isinstance(term_budget_eh_bohr, float):
        raise ValueError(
            "fd_contract_guard: term_budget_eh_bohr must be a float, got "
            f"{type(term_budget_eh_bohr).__name__} "
            f"({term_budget_eh_bohr!r})")
    if not math.isfinite(term_budget_eh_bohr):
        raise ValueError(
            "fd_contract_guard: term_budget_eh_bohr must be a finite "
            f"float, got {term_budget_eh_bohr!r}; NaN/inf are not declared "
            "bounds (an infinite budget vacuously covers the measured "
            "contract class, a NaN one covers nothing)")
    if not term_budget_eh_bohr > 0.0:
        raise ValueError(
            "fd_contract_guard: term_budget_eh_bohr must be positive, got "
            f"{term_budget_eh_bohr!r}")
    scale = (measured_class_scale_eh_bohr if
             measured_class_scale_eh_bohr is not None
             else MEASURED_CLASS_SCALE_EH_BOHR)
    if measured_class_scale_eh_bohr is not None:
        if isinstance(measured_class_scale_eh_bohr, bool) or \
                not isinstance(measured_class_scale_eh_bohr, float):
            raise ValueError(
                "fd_contract_guard: measured_class_scale_eh_bohr must be "
                "a positive float (a MEASURED cell-specific scale, with "
                "its evidence recorded by the caller), got "
                f"{measured_class_scale_eh_bohr!r}")
        if not math.isfinite(measured_class_scale_eh_bohr):
            raise ValueError(
                "fd_contract_guard: measured_class_scale_eh_bohr must be "
                "a finite float, got "
                f"{measured_class_scale_eh_bohr!r}; a non-finite value is "
                "not a MEASURED cell-specific scale")
        if not measured_class_scale_eh_bohr > 0.0:
            raise ValueError(
                "fd_contract_guard: measured_class_scale_eh_bohr must be "
                "a positive float (a MEASURED cell-specific scale, with "
                "its evidence recorded by the caller), got "
                f"{measured_class_scale_eh_bohr!r}")
        if measured_class_scale_eh_bohr > MEASURED_CLASS_SCALE_EH_BOHR:
            raise ValueError(
                "fd_contract_guard: measured_class_scale_eh_bohr "
                f"({measured_class_scale_eh_bohr!r}) exceeds the measured "
                "class scale; pass the larger value directly as "
                "term_budget_eh_bohr instead")
    if term_budget_eh_bohr < scale:
        raise ValueError(
            "fd_contract_guard: term_budget_eh_bohr "
            f"({term_budget_eh_bohr!r}) is below the measured contract-"
            "class scale ({scale!r} Eh/Bohr); declare a budget that covers "
            "the known defect class, or pin a measured cell-specific scale "
            "via measured_class_scale_eh_bohr")
    return {
        "schema": FD_CONTRACT_GUARD_SCHEMA,
        "source": source,
        "contract": "term_budget",
        "analytic_grid_response": False,
        "term_budget_eh_bohr": term_budget_eh_bohr,
        "measured_class_scale_eh_bohr": scale,
    }


def gradient_with_contract(mf, grid_response=None,
                           term_budget_eh_bohr=None, source=""):
    """Build the analytic gradient object under the declared contract.

    g = gradient_with_contract(mf, grid_response=True) sets
    g.grid_response = True before the caller calls g.kernel().
    The term-budget branch returns the default-gradient object together
    with the contract record the caller must embed in its manifest.
    Returns (gradient_object, contract_record).

    grid_response accepts only True (grid-response branch) or None/omitted
    (term-budget branch, term_budget_eh_bohr required); an explicit
    grid_response=False or any other value is rejected -- silently
    reinterpreting it would let a declared contract diverge from the
    object actually built. After construction the guard verifies the
    runtime object agrees with the recorded contract and fails closed
    otherwise: a config override or previously-configured gradient
    factory that hands back grid_response=True under a term-budget
    declaration is false provenance, and an object that will not accept
    grid_response=True cannot be certified as including the grid response.
    The verified runtime value is embedded as runtime_grid_response in the
    returned record.
    """
    if grid_response is not None and grid_response is not True:
        raise ValueError(
            "fd_contract_guard: gradient_with_contract grid_response "
            "accepts True (include grid response) or None (term-budget "
            f"branch); got {grid_response!r}. An explicit grid_response="
            "False is the measured contract-mismatch class and must not "
            "be silently reinterpreted as the term-budget branch")
    record = require_fd_contract(
        analytic_grid_response=(True if grid_response is True else None),
        term_budget_eh_bohr=term_budget_eh_bohr,
        source=source or "gradient_with_contract")
    g = mf.nuc_grad_method()
    if record["analytic_grid_response"]:
        g.grid_response = True
        runtime = getattr(g, "grid_response", None)
        if runtime is not True:
            raise ValueError(
                "fd_contract_guard: the gradient object returned by "
                "nuc_grad_method() did not end up with grid_response=True "
                f"(actual {runtime!r}); cannot certify the grid-response "
                "contract for an object that drops the assignment")
    else:
        runtime = bool(getattr(g, "grid_response", False))
        if runtime:
            raise ValueError(
                "fd_contract_guard: the gradient object returned by "
                "nuc_grad_method() carries runtime grid_response=True "
                "(config override or previously-configured factory), "
                "which contradicts the declared term-budget contract "
                "(analytic_grid_response=False): the record would claim "
                "the grid-response term is omitted while the object "
                "actually computes it. Declare grid_response=True to "
                "match, or construct the gradient from a clean config")
    record = dict(record, runtime_grid_response=runtime)
    return g, record
