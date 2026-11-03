#!/usr/bin/env bash
set -euo pipefail
# Linux/systemd execution wrapper: hard RAM limit and six-core affinity.
GRAPH_R1_PYTHON="${GRAPH_R1_PYTHON:-/home/prime/miniconda3/bin/python3}"
GRAPH_R1_CPUS="$($GRAPH_R1_PYTHON -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:6])))')"
exec systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 -p CPUQuota=600% \
  taskset -c "$GRAPH_R1_CPUS" "$GRAPH_R1_PYTHON" scripts/run_v2.py "$@"
