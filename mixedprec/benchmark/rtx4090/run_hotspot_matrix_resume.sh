#!/bin/bash
# Issue #9 attribution matrix RESUME (attempt 2): attempt 1 was killed by the
# 60-minute job wall clock mid-matrix (benzene/def2-tzvp/wb97m-v fp32 lane).
# The profiler persists the JSON incrementally per lane, so this script runs
# ONLY the cells/lanes missing from hotspot_attribution.json; completed cells
# are never re-run (they stay attempt-1 evidence, sha256-anchored).
set -u
export PATH=/home/qczcode/gpu4090-src/bin:$PATH
export PYTHONPATH=/data/work/gpu4pyscf-mixedprec
cd /data/work/gpu4pyscf-mixedprec
OUT=/data/results/gpu4pyscf-agent/tasks/issue-9/hotspot_attribution.json
LOG=/data/results/gpu4pyscf-agent/tasks/issue-9/hotspot_matrix_resume.runlog.log
PY=mixedprec/benchmark/rtx4090/hotspot_profiler.py
COMMON="--out $OUT --samples-instr 1 --samples-uninstr 1"

run() { echo "=== [$(date -u +%H:%M:%S)] $* ===" | tee -a "$LOG";
  timeout 480 python $PY $COMMON "$@" 2>&1 | grep -v "^$" | tee -a "$LOG"; }

# 1. finish the interrupted cell: fp32 lane only (fp64 lane is attempt-1
#    evidence from a different process; each lane carries its own discarded
#    warmup so JIT state cannot leak between lanes)
run --geom benzene.xyz --basis def2-tzvp --xc wb97m-v --pilot-stage S2 \
    --lane fp32 \
    --evidence-note "fp64 lane measured in attempt-1 process; fp32 lane resumed in attempt-2 process after a wall-clock kill (each lane has its own discarded warmup)"
# 2. r2SCAN-3c whole method (basis pinned def2-mtzvpp by the profiler)
run --geom benzene.xyz --basis def2-mtzvpp --xc r2scan3c --pilot-stage S1 --lane fp64 fp32
# 3. UKS sentinels (admitted cohorts; open-shell lane composition)
run --geom methoxy.xyz --basis def2-svp --xc b3lyp --pilot-stage S1 --lane fp64 fp32
run --geom methoxy.xyz --basis def2-svp --xc wb97m-v --pilot-stage S1 --lane fp64 fp32
run --geom methoxy.xyz --basis def2-mtzvpp --xc r2scan3c --pilot-stage S1 --lane fp64 fp32
echo "=== RESUME DONE $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
