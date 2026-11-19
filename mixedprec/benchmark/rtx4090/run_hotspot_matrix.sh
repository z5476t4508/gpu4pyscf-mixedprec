#!/bin/bash
# Issue #9 attribution matrix: four methods x admitted small/moderate cohort,
# fp64/fp32 lanes back-to-back in one process per cell (prevents cross-cell
# JIT contamination; one GPU task at a time).
set -u
export PATH=/home/qczcode/gpu4090-src/bin:$PATH
export PYTHONPATH=/data/work/gpu4pyscf-mixedprec
cd /data/work/gpu4pyscf-mixedprec
OUT=/data/results/gpu4pyscf-agent/tasks/issue-9/hotspot_attribution.json
PY=mixedprec/benchmark/rtx4090/hotspot_profiler.py
COMMON="--out $OUT --samples-instr 1 --samples-uninstr 1"

run() { echo "=== [$(date -u +%H:%M:%S)] $* ===";
  timeout 480 python $PY $COMMON "$@" 2>&1 | grep -v "^$"; }

# hybrid methods: S1 benzene/svp + S2 benzene/tzvp (admitted cohorts of #5-#7)
run --geom benzene.xyz --basis def2-svp --xc b3lyp --pilot-stage S1 --lane fp64 fp32
run --geom benzene.xyz --basis def2-tzvp --xc b3lyp --pilot-stage S2 --lane fp64 fp32
run --geom benzene.xyz --basis def2-svp --xc m062x --pilot-stage S1 --lane fp64 fp32
run --geom benzene.xyz --basis def2-tzvp --xc m062x --pilot-stage S2 --lane fp64 fp32
run --geom benzene.xyz --basis def2-svp --xc wb97m-v --pilot-stage S1 --lane fp64 fp32
run --geom benzene.xyz --basis def2-tzvp --xc wb97m-v --pilot-stage S2 --lane fp64 fp32
# r2SCAN-3c whole method (basis pinned def2-mtzvpp by the profiler)
run --geom benzene.xyz --basis def2-mtzvpp --xc r2scan3c --pilot-stage S1 --lane fp64 fp32
# UKS sentinels (admitted cohorts; open-shell lane composition)
run --geom methoxy.xyz --basis def2-svp --xc b3lyp --pilot-stage S1 --lane fp64 fp32
run --geom methoxy.xyz --basis def2-svp --xc wb97m-v --pilot-stage S1 --lane fp64 fp32
run --geom methoxy.xyz --basis def2-mtzvpp --xc r2scan3c --pilot-stage S1 --lane fp64 fp32
echo "=== ALL DONE $(date -u +%H:%M:%S) ==="
