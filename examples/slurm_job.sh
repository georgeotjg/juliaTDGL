#!/bin/bash
# One run as a Slurm batch job (submit from the root of the repository):
#   sbatch examples/slurm_job.sh
# The kernel is threaded over z planes; a job stopped by the time limit resumes
# from its last closed point by resubmitting with --restart and the same --dir.
#SBATCH --job-name=tdgl
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
julia --project=julia julia/scripts/run_preset.jl --geom WEDGE \
      --mesh production --window paper --J 0.0,0.040,0.045 \
      --dir runs/wedge_${SLURM_JOB_ID:-local}
