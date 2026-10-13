#!/bin/bash
# One run as a Slurm batch job. Submit from the root of the repository:
#   sbatch examples/slurm_job.sh                  # RUN=wedge by default
#   RUN=my_run sbatch examples/slurm_job.sh
# A job stopped by the time limit resumes from its last closed current point: submit it again with
# RESUME=1 and the same RUN (the folder must not change between submissions).
#SBATCH --job-name=tdgl
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
# On a cluster, load Julia first (e.g. `module load julia`, or a juliaup install in $HOME).
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # one core per run; more BLAS threads only oversubscribe
RUN=${RUN:-wedge}
julia --project=julia julia/scripts/run_preset.jl --geom WEDGE \
      --mesh production --window paper --J 0.0,0.040,0.045 \
      --dir runs/$RUN ${RESUME:+--restart}
