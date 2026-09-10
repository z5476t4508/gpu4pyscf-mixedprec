#!/bin/sh
# smoke test + full run helper (classifier outage workaround: short commands)
cd /home/tong/soft/gpu4pyscf || exit 1
export PYTHONPATH=/home/tong/soft/gpu4pyscf
exec .venv/bin/python mixedprec/benchmark/run_benchmark.py "$@"
