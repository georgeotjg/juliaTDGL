#!/usr/bin/env bash
# Pin a command to a CPU set, one Julia thread and one BLAS thread. Does not change the kernel or
# the result (bit-identical with or without pinning).
#
#   ./julia/scripts/pin_cpu.sh 0,1 julia --project=julia julia/scripts/run_preset.jl --geom WEDGE ...
#   ./julia/scripts/pin_cpu.sh 4,5 julia --project=julia julia/test/run.jl
set -euo pipefail
cpus="${1:?usage: pin_cpu.sh CPUSET command...}"
shift
export JULIA_CPU_THREADS=1
export OPENBLAS_NUM_THREADS=1   # one core per run: more OpenBLAS threads only oversubscribe
export JULIA_UNBUFFERED=1
exec taskset -c "$cpus" stdbuf -oL -eL "$@"
