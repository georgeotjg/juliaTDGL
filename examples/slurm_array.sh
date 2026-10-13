#!/bin/bash
# Several independent runs as one Slurm job array: here the same notched box at five fields, one task
# per field (each task runs both polarities). This is the natural use of a cluster for this code: the
# runs do not talk to each other, so N tasks give N times the throughput with no change to the solver.
#   sbatch examples/slurm_array.sh
# A task stopped by the time limit resumes by resubmitting that index with RESUME=1:
#   RESUME=1 sbatch --array=3 examples/slurm_array.sh
#SBATCH --job-name=tdgl-field
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FIELDS=(0.015 0.03 0.06 0.09 0.12)                  # in units of H_c2
H=${FIELDS[${SLURM_ARRAY_TASK_ID:-0}]}
julia --project=julia julia/scripts/run_preset.jl --geom WEDGE --kappa 25 --xi-ratio 4 \
      --H $H --J 0.02,0.04,0.06,0.08,0.10,0.20,0.30,0.40 \
      --mesh production --window long --dir runs/field_H$H ${RESUME:+--restart}
