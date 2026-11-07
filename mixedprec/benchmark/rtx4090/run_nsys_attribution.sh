#!/bin/bash
# Issue #9 kernel-level attribution: nsys profile of the instrumented
# profiler (HOTSPOT_NVTX=1 pushes an NVTX range per function bucket), then
# export kernel-by-NVTX-range timings.  One representative cell per method
# family: benzene/def2-svp closed shell on the four methods.
set -u
export PATH=/home/qczcode/gpu4090-src/bin:$PATH
export PYTHONPATH=/data/work/gpu4pyscf-mixedprec
export HOTSPOT_NVTX=1
cd /data/work/gpu4pyscf-mixedprec
OUTDIR=/data/results/gpu4pyscf-agent/tasks/issue-9
PY=mixedprec/benchmark/rtx4090/hotspot_profiler.py
NSYS=/usr/local/cuda-12.8/bin/nsys

profile_cell () {  # tag geom basis xc
  local tag=$1 geom=$2 basis=$3 xc=$4
  echo "=== [$(date -u +%H:%M:%S)] nsys $tag ==="
  rm -f "$OUTDIR/nsys_$tag".{nsys-rep,sqlite,txt}
  timeout 600 $NSYS profile -t cuda,nvtx --force-overwrite=true \
      -o "$OUTDIR/nsys_$tag" \
      python $PY --out "$OUTDIR/hotspot_nsys_aux_$tag.json" \
      --geom "$geom" --basis "$basis" --xc "$xc" --lane fp64 \
      --samples-instr 1 --samples-uninstr 0 2>&1 | tail -4
  timeout 300 $NSYS export --type sqlite --force-overwrite=true \
      -o "$OUTDIR/nsys_$tag.sqlite" "$OUTDIR/nsys_$tag.nsys-rep" 2>&1 | tail -1
}

profile_cell b3lyp_svp   benzene.xyz def2-svp     b3lyp
profile_cell m062x_svp   benzene.xyz def2-svp     m062x
profile_cell wb97mv_svp  benzene.xyz def2-svp     wb97m-v
profile_cell r2scan3c_s1 benzene.xyz def2-mtzvpp  r2scan3c

# kernel-by-range attribution: every CUDA kernel's start time falls inside
# exactly one NVTX bucket range; charge its duration to that bucket.
# (logic lives in nsys_attribution.py -- unit-tested against the two
# pitfalls found live: text-vs-StringIds range names and nested ranges)
PYTHONPATH=/data/work/gpu4pyscf-mixedprec python3 -m nsys_attribution 2>&1 || \
PYTHONPATH=/data/work/gpu4pyscf-mixedprec/mixedprec/benchmark/rtx4090 \
    python3 /data/work/gpu4pyscf-mixedprec/mixedprec/benchmark/rtx4090/nsys_attribution.py
echo "=== NSYS DONE $(date -u +%H:%M:%S) ==="
