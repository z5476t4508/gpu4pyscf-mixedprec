#!/usr/bin/env python3
"""SNT-2 force finite-difference diagnostics (benchv1 sentinel card 2026-10-30).

Two pre-fixed cells from the sentinel card:

  cytosine / b3lyp / def2-svp        atoms: the N1-ring H, and among the
                                      ammonium-class (amino) H's the one
                                      with the largest |g| atom-norm on the
                                      batch's auto-lane gradient
  acetaminophen / wb97mv / def2-svp  atoms: the phenolic H and the amide N

Protocol: central differences, steps 0.001 and 0.002 Bohr, all three
components per atom; energies from gradient_runner's 'auto' lane (the
production auto SCF + fp32 gradient-kernel lane the card names) at the
schedule's scientific settings verbatim, --samples 3 (schema rule R1).
The analytic reference is the batch's merged auto-lane row for the same
cell, so FD probes the same lane it is compared against.

FD is DIAGNOSTIC ONLY: the criterion is agreement within 3x the card's
stated cancellation floor (~1e-4..1e-3 Eh/Bohr at these steps; the
conservative end 1e-4 sets the 3e-4 flag threshold here). It never
passes or fails the 3e-5 Eh/Bohr mixed-precision gradient gate, and the
band may be re-decided at first view like SNT-1.

Stages (run in order, post-batch only for 'run' -- it launches GPU work):

  select   CPU: verify base geometry hashes vs the schedule pins, read the
           batch auto-lane gradients, select atoms per the card rules,
           write displaced .xyz files + plan.json with the exact runner
           argv (settings verbatim from the schedule).
  run      GPU: refuse while any batch driver/arm process is alive, then
           execute the planned gradient_runner invocations serially
           (one bounded infrastructure retry per run), logging rc's.
  compare  CPU: central differences vs the analytic auto gradient,
           per-run repeat-energy spread as a measured noise floor,
           JSON + Markdown records.  Fail-closed on any missing or
           non-ok run.

The module deliberately imports nothing GPU-side; gradient_runner is
only ever a subprocess, so the CPU tests exercise the full pipeline
against a stub runner.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

BOHR_ANG = 0.52917721092  # CODATA 2018 Bohr radius in Angstrom
NOMINAL_FLOOR = 1e-4      # card's conservative-end cancellation floor, Eh/Bohr
CRIT_FACTOR = 3.0         # card: agreement within 3x the FD floor
AXES = ("x", "y", "z")
STEPS_BOHR = (0.001, 0.002)

# card SNT-2 (fixed cells; atom choice per the rules implemented below)
SNT2_CELLS = [
    {"tag": "cytosine_b3lyp_def2-svp", "molecule": "cytosine"},
    {"tag": "acetaminophen_wb97mv_def2-svp", "molecule": "acetaminophen"},
]

COVALENT_RADII_ANG = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
BOND_TOL = 1.25  # bonded if d <= tol * (r_i + r_j)

GUARD_MARKERS = ("run_upstream_batch.py", "upstream_comparison.py",
                 "gradient_runner.py")


def fail(msg):
    raise SystemExit(f"FAIL-CLOSED: {msg}")


# ---------------------------------------------------------------- geometry

def parse_xyz(path):
    lines = open(path).read().splitlines()
    nat = int(lines[0].split()[0])
    syms, coords = [], []
    for ln in lines[2:2 + nat]:
        parts = ln.split()
        syms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(syms) != nat:
        fail(f"{path}: declared {nat} atoms, found {len(syms)}")
    return syms, coords


def write_xyz(path, comment, syms, coords):
    out = [str(len(syms)), comment]
    for s, (x, y, z) in zip(syms, coords):
        out.append(f"{s} {x:.8f} {y:.8f} {z:.8f}")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def displaced(coords, atom, axis, delta_bohr):
    """Copy of coords with one atom shifted by delta_bohr along axis."""
    new = [list(c) for c in coords]
    new[atom][axis] += delta_bohr * BOHR_ANG
    return new


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def connectivity(syms, coords):
    """Bond lists by covalent-radius threshold; CHON only (fail-closed)."""
    for s in syms:
        if s not in COVALENT_RADII_ANG:
            fail(f"non-CHON element {s!r} outside the sentinel card scope")
    bonds = [[] for _ in syms]
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            d = math.dist(coords[i], coords[j])
            if d <= BOND_TOL * (COVALENT_RADII_ANG[syms[i]]
                                + COVALENT_RADII_ANG[syms[j]]):
                bonds[i].append(j)
                bonds[j].append(i)
    return bonds


def atom_norm(grad, i):
    return math.sqrt(sum(c * c for c in grad[i]))


# ---------------------------------------------------------------- selectors

def select_cytosine(syms, coords, grad):
    """Card: the N1 H (ring N bearing exactly one H), and the ammonium-class
    H (on the NH2 nitrogen) with the largest |g| atom-norm."""
    bonds = connectivity(syms, coords)
    nitrogens = [i for i, s in enumerate(syms) if s == "N"]
    ring_n = [i for i in nitrogens
              if sum(1 for j in bonds[i] if syms[j] == "H") == 1
              and sum(1 for j in bonds[i] if syms[j] == "C") >= 2]
    amino_n = [i for i in nitrogens
               if sum(1 for j in bonds[i] if syms[j] == "H") == 2]
    if len(ring_n) != 1:
        fail(f"cytosine: expected exactly one ring N bearing one H, "
             f"found {len(ring_n)} (indices {ring_n})")
    if len(amino_n) != 1:
        fail(f"cytosine: expected exactly one NH2 nitrogen, "
             f"found {len(amino_n)} (indices {amino_n})")
    h_ring = [j for j in bonds[ring_n[0]] if syms[j] == "H"][0]
    amino_hs = [j for j in bonds[amino_n[0]] if syms[j] == "H"]
    norms = {j: atom_norm(grad, j) for j in amino_hs}
    h_amino = max(amino_hs, key=lambda j: norms[j])
    return [
        {"index": h_ring, "element": "H",
         "rule": "H on the ring N (N1) that bears exactly one H and >=2 C",
         "norms_considered": None},
        {"index": h_amino, "element": "H",
         "rule": "ammonium-class (NH2) H with largest |g| atom-norm on the "
                 "auto-lane gradient",
         "norms_considered": {str(k): v for k, v in norms.items()}},
    ]


def select_acetaminophen(syms, coords, grad):
    """Card: the phenolic H, and the amide N (the single N of the molecule)."""
    bonds = connectivity(syms, coords)
    oh_o = [i for i, s in enumerate(syms) if s == "O"
            and sum(1 for j in bonds[i] if syms[j] == "H") == 1]
    nitrogens = [i for i, s in enumerate(syms) if s == "N"]
    if len(oh_o) != 1:
        fail(f"acetaminophen: expected exactly one hydroxyl O, found {oh_o}")
    if len(nitrogens) != 1:
        fail(f"acetaminophen: expected exactly one (amide) N, found "
             f"{nitrogens}")
    phenol_h = [j for j in bonds[oh_o[0]] if syms[j] == "H"][0]
    return [
        {"index": phenol_h, "element": "H",
         "rule": "H on the single hydroxyl O (phenolic H)",
         "norms_considered": None},
        {"index": nitrogens[0], "element": "N",
         "rule": "the single N of acetaminophen (the amide N)",
         "norms_considered": None},
    ]


SELECTORS = {"cytosine": select_cytosine,
             "acetaminophen": select_acetaminophen}


# ---------------------------------------------------------------- FD math

def central_difference(e_plus, e_minus, h):
    return (e_plus - e_minus) / (2.0 * h)


def runner_argv(python, runner, out, geom, cell, settings):
    """gradient_runner argv for one displaced geometry, settings verbatim."""
    return [python, runner,
            "--out", out,
            "--geom", geom,
            "--basis", cell["basis"],
            "--xc", cell["xc"],
            "--auxbasis", cell.get("auxbasis") or settings["auxbasis"],
            "--charge", str(cell["charge"]),
            "--spin", str(cell["spin"]),
            "--lane", "auto",
            "--samples", "3",
            "--conv-tol", repr(settings["conv_tol"]),
            "--max-cycle", str(settings["max_cycle"]),
            "--init-guess", settings["init_guess"],
            "--direct-scf-tol", repr(settings["direct_scf_tol"]),
            "--grids-level", str(settings["grids_level"]),
            "--nlcgrids-level", str(settings["nlcgrids_level"])]


def load_auto_row(batch_root, tag):
    path = os.path.join(batch_root, "cells", f"{tag}_optimized_auto.json")
    if not os.path.isfile(path):
        fail(f"missing batch merged auto row for {tag}: {path}")
    d = json.load(open(path))
    rows = d.get("rows") or []
    if not rows:
        fail(f"no rows in merged auto row for {tag}: {path}")
    r0 = rows[0]
    if r0.get("status") != "ok":
        fail(f"auto row status={r0.get('status')} for {tag}: {path}")
    res = r0.get("result") or {}
    if res.get("gradient") is None:
        fail(f"auto row for {tag} carries no gradient: {path}")
    return r0, path


# ---------------------------------------------------------------- stages

def stage_select(a):
    schedule = json.load(open(a.schedule))
    settings = schedule["settings"]
    cells = {c["tag"]: c for c in schedule["cells"]}
    disp_root = os.path.join(a.sent_root, "disp")
    runs_root = os.path.join(a.sent_root, "runs")
    os.makedirs(disp_root, exist_ok=True)
    os.makedirs(runs_root, exist_ok=True)
    plan = {"schema": "benchv1-snt2-plan-v1", "created_utc": a.now_utc,
            "schedule": a.schedule, "batch_root": a.batch_root,
            "lane": "auto", "samples_per_run": 3,
            "steps_bohr": list(STEPS_BOHR),
            "nominal_floor": NOMINAL_FLOOR, "crit_factor": CRIT_FACTOR,
            "cells": []}
    for spec in SNT2_CELLS:
        tag, mol = spec["tag"], spec["molecule"]
        cell = cells.get(tag) or fail(f"{tag} not in schedule {a.schedule}")
        got = sha256_file(cell["geom"])
        if got != cell["geometry_sha256"]:
            fail(f"{tag}: geometry sha256 {got} != schedule pin "
                 f"{cell['geometry_sha256']}")
        syms, coords = parse_xyz(cell["geom"])
        row, row_path = load_auto_row(a.batch_root, tag)
        grad = row["result"]["gradient"]
        if len(grad) != len(syms):
            fail(f"{tag}: gradient has {len(grad)} atoms, geometry "
                 f"{len(syms)}")
        atoms = SELECTORS[mol](syms, coords, grad)
        entry = {"tag": tag, "molecule": mol, "geom": cell["geom"],
                 "geometry_sha256": cell["geometry_sha256"],
                 "auto_row": row_path, "analytic_gradient_source_row": row_path,
                 "atoms": atoms, "runs": []}
        for at in atoms:
            for axis in range(3):
                for h in STEPS_BOHR:
                    for sign in (+1, -1):
                        name = (f"{tag}_a{at['index']}{at['element']}_"
                                f"{AXES[axis]}_{h:.4f}bohr_"
                                f"{'plus' if sign > 0 else 'minus'}")
                        geom_out = os.path.join(disp_root, name + ".xyz")
                        write_xyz(geom_out,
                                  f"SNT-2 FD displacement {name} from "
                                  f"{cell['geom']} (schedule pin "
                                  f"{cell['geometry_sha256'][:12]})",
                                  syms, displaced(coords, at["index"], axis,
                                                  sign * h))
                        run_out = os.path.join(runs_root, name + ".json")
                        entry["runs"].append({
                            "name": name, "atom": at["index"],
                            "axis": axis, "step_bohr": h, "sign": sign,
                            "disp_geom": geom_out, "out": run_out,
                            "argv": runner_argv(a.python, a.runner, run_out,
                                                geom_out, cell, settings)})
        plan["cells"].append(entry)
        print(f"[select] {tag}: atoms "
              + ", ".join(f"#{at['index']}({at['element']})" for at in atoms)
              + f"; {len(entry['runs'])} displaced geometries")
    tmp = a.plan + ".tmp"
    with open(tmp, "w") as f:
        json.dump(plan, f, indent=1, ensure_ascii=False)
    json.load(open(tmp))
    os.replace(tmp, a.plan)
    print(f"[select] plan -> {a.plan}")


def guard_no_gpu_work(proc_root="/proc"):
    """Refuse if any batch driver/arm process is alive (fail-closed)."""
    me = os.getpid()
    for pid in sorted(os.listdir(proc_root)):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = open(os.path.join(proc_root, pid, "cmdline"),
                       "rb").read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        hit = next((m for m in GUARD_MARKERS
                    if any(m in part for part in cmd)), None)
        if hit:
            fail(f"GPU workload alive (pid {pid}, marker {hit!r}); the "
                 "SNT-2 runs must not overlap the formal batch or arms")


def stage_run(a):
    plan = json.load(open(a.plan))
    if a.guard:
        guard_no_gpu_work()
    for cell in plan["cells"]:
        for run in cell["runs"]:
            if os.path.isfile(run["out"]) and not a.redo:
                row_ok = _peek_ok(run["out"])
                if row_ok:
                    print(f"[run] skip existing ok: {run['name']}")
                    continue
            log = run["out"].replace(".json", ".runlog.log")
            for attempt in (1, 2):
                with open(log, "a") as lf:
                    lf.write(f"[{a.now_utc}] attempt {attempt} "
                             f"{run['name']}\n")
                    lf.flush()
                    rc = subprocess.call(run["argv"], stdout=lf, stderr=lf)
                print(f"[run] {run['name']} attempt {attempt} rc={rc}")
                if rc == 0 and _peek_ok(run["out"]):
                    break
            else:
                fail(f"{run['name']}: rc={rc} after bounded retry; keeping "
                     f"log {log} and stopping (partial state preserved)")


def _peek_ok(path):
    if not os.path.isfile(path):
        return False
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return False
    for r in d.get("rows") or []:
        if r.get("row_key", {}).get("lane") == "auto":
            return r.get("status") == "ok"
    return False


def stage_compare(a):
    plan = json.load(open(a.plan))
    out = {"schema": "benchv1-snt2-fd-v1", "created_utc": a.now_utc,
           "diagnostic_only": True,
           "energy_source": "gradient_runner auto-lane rows, result.e_tot "
                            "(worst-repeat aggregate); per-repeat spread "
                            "feeds the measured noise floor",
           "criterion": f"max|g_fd - g_analytic| <= {CRIT_FACTOR}x the "
                        f"nominal floor {NOMINAL_FLOOR:.0e} Eh/Bohr "
                        "(card conservative end; band may be re-decided at "
                        "first view); NOT the 3e-5 mixed-precision gate",
           "cells": []}
    worst_overall = 0.0
    for cell in plan["cells"]:
        tag = cell["tag"]
        r0, _ = load_auto_row(a.batch_root, tag)
        grad = r0["result"]["gradient"]
        entry = {"tag": tag, "atoms": cell["atoms"], "steps": {}}
        for h in STEPS_BOHR:
            comps, noises = [], []
            for at in cell["atoms"]:
                for axis in range(3):
                    ep = _auto_energy(a, cell, at, axis, h, +1)
                    em = _auto_energy(a, cell, at, axis, h, -1)
                    fd = central_difference(ep["e"], em["e"], h)
                    ana = grad[at["index"]][axis]
                    comps.append({
                        "atom": at["index"], "element": at["element"],
                        "axis": AXES[axis], "fd": fd, "analytic": ana,
                        "abs_delta": abs(fd - ana)})
                    noises.append(ep["spread"] / (2.0 * h))
                    noises.append(em["spread"] / (2.0 * h))
            worst = max(c["abs_delta"] for c in comps)
            noise_floor = max(noises)
            worst_overall = max(worst_overall, worst)
            entry["steps"][f"{h:.4f}"] = {
                "components": comps, "max_abs_delta": worst,
                "measured_noise_floor": noise_floor,
                "threshold": CRIT_FACTOR * NOMINAL_FLOOR,
                "within_3x_nominal_floor": worst <= CRIT_FACTOR * NOMINAL_FLOOR}
        out["cells"].append(entry)
    out["max_abs_delta_overall"] = worst_overall
    out["all_within_3x_nominal_floor"] = (
        all(s["within_3x_nominal_floor"]
            for c in out["cells"] for s in c["steps"].values()))
    tmp = a.out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    json.load(open(tmp))
    os.replace(tmp, a.out_json)
    _write_md(a, out)
    print(f"[compare] worst max|delta| {worst_overall:.3e} Eh/Bohr; "
          f"within 3x nominal floor: {out['all_within_3x_nominal_floor']}")


def _auto_energy(a, cell, at, axis, h, sign):
    name = next(r["name"] for r in cell["runs"]
                if r["atom"] == at["index"] and r["axis"] == axis
                and r["step_bohr"] == h and r["sign"] == sign)
    path = os.path.join(a.sent_root, "runs", name + ".json")
    if not os.path.isfile(path):
        fail(f"missing FD run output {name}: {path}")
    d = json.load(open(path))
    for r in d.get("rows") or []:
        if r.get("row_key", {}).get("lane") == "auto":
            if r.get("status") != "ok":
                fail(f"FD run {name}: auto row status={r.get('status')}")
            res = r["result"]
            es = [rep.get("e_tot") for rep in res.get("repeats") or []
                  if rep.get("e_tot") is not None]
            if not es:
                fail(f"FD run {name}: no repeat energies")
            return {"e": res["e_tot"], "spread": max(es) - min(es)}
    fail(f"FD run {name}: no auto-lane row")


def _write_md(a, out):
    lines = ["# SNT-2 finite-difference force diagnostics (benchv1 card)",
             "",
             "Diagnostic only: FD-vs-analytic agreement on the auto lane "
             f"criterion {CRIT_FACTOR:.0f}x nominal floor "
             f"{NOMINAL_FLOOR:.0e} Eh/Bohr; never the 3e-5 gate.", ""]
    for cell in out["cells"]:
        lines.append(f"## {cell['tag']}")
        atom_txt = ", ".join(f"#{at['index']} ({at['element']}; "
                             f"{at['rule']})" for at in cell["atoms"])
        lines += ["", f"FD atoms: {atom_txt}", "",
                  "| step (Bohr) | atom | axis | g_fd | g_analytic | "
                  "|delta| |", "|---|---|---|---|---|---|"]
        for hs, st in cell["steps"].items():
            for c in st["components"]:
                lines.append(f"| {hs} | {c['atom']}{c['element']} "
                             f"| {c['axis']} | {c['fd']:+.6e} "
                             f"| {c['analytic']:+.6e} "
                             f"| {c['abs_delta']:.3e} |")
            lines.append(f"| — | — | max | | | "
                         f"{st['max_abs_delta']:.3e} |")
            lines.append("")
            lines.append(f"step {hs}: measured noise floor "
                         f"{st['measured_noise_floor']:.3e}, threshold "
                         f"{st['threshold']:.1e} -> "
                         f"within={st['within_3x_nominal_floor']}")
        lines.append("")
    lines.append(f"Overall worst |delta|: "
                 f"{out['max_abs_delta_overall']:.3e} Eh/Bohr; "
                 f"all within 3x nominal floor: "
                 f"{out['all_within_3x_nominal_floor']}")
    with open(a.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True,
                   choices=["select", "run", "compare"])
    p.add_argument("--schedule", default="/data/results/gpu4pyscf-agent/"
                   "tasks/issue-9/2026-10-30-benchv1-exec/schedule-v2.json")
    p.add_argument("--batch-root", default="/data/results/gpu4pyscf-agent/"
                   "tasks/issue-9/2026-10-30-benchv1-exec/batch-obcr")
    p.add_argument("--sent-root", default="/data/results/gpu4pyscf-agent/"
                   "tasks/issue-9/2026-10-30-benchv1-sentinels/snt2")
    p.add_argument("--plan", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--runner", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gradient_runner.py"))
    p.add_argument("--runner-override", default=None,
                   help="replace the runner argv head [python, runner] "
                        "(stub runner for CPU integration tests)")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--no-guard", action="store_true",
                   help="skip the GPU-workload guard (CPU tests only)")
    a = p.parse_args(argv)
    a.now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    a.sent_root = os.path.abspath(a.sent_root)
    if a.plan is None:
        a.plan = os.path.join(a.sent_root, "plan.json")
    if a.out_json is None:
        a.out_json = os.path.join(a.sent_root, "fd-results.json")
    if a.out_md is None:
        a.out_md = os.path.join(a.sent_root, "fd-results.md")
    if a.runner_override:
        head = a.runner_override.split("::")
        a.python, a.runner = head[0], head[1]
    a.guard = not a.no_guard
    os.makedirs(a.sent_root, exist_ok=True)
    if a.stage == "select":
        stage_select(a)
    elif a.stage == "run":
        stage_run(a)
    else:
        stage_compare(a)


if __name__ == "__main__":
    main()
