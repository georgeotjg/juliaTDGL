# Recorded accelerator checks

These are engineering validation data, separate from any research campaign. They establish agreement of the tested
discrete trajectories within their recorded tolerances, not continuum convergence or arbitrary combinations.

The CUDA snapshot used Julia 1.13.2, CUDA.jl 6.5.1, KernelAbstractions 0.10.7 and Adapt 4.8.0 on a Tesla T4.
`t4_environment.json` records uploaded and distributed source hashes. `kernel_body.json` identifies the numerical module
body, unchanged by this audit's host-side cache, strict-mode and execution-counter changes. The current CUDA run tested
those controls as well as the numerical paths. `phases.json` keeps the initial harness failures separate from successful
validation and retries; an initial job exit of 1 is not relabelled as a successful whole job.

- `t4_steps.json`: all 41 configurations, step counts, tolerances, maximum errors and one timing batch per configuration.
- `t4_ladder.json`, `t4_ladder_pcg.json`: short ladders, including PCG; verdicts and cycle counts agree. `FFFF*` leaves the
  onset unresolved. The PCG ladder state error is 5.156e-9 at a 1e-6 tolerance; voltage relative error is 9.587e-11.
- The same CUDA run passed eight analytic assertions and 217 zero-source/warm-history/checkpoint PCG assertions. The
  test sources are in `test/`; these counts are not additional configurations in the 41-case table.
- `t4_resume.json`: five calibrated checkpoint continuations (38 assertions), covering bar/slab/open, noise/KWT and heat,
  including another open-boundary FFT update. Largest state difference: 1.710e-9, at a 1e-7 tolerance.
- `t4_cli.json`: strict CUDA launch, completed resume preserving eleven numerical output files, and a wrapped CPU CLI
  with 68 CUDA steps and zero fallback steps. Strict mode does not remove the documented host responsibilities.

Reproduce validation from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 julia --project=gpu/TDGLGPU -e 'using Pkg; Pkg.test()'
OPENBLAS_NUM_THREADS=1 julia --project=gpu/TDGLGPU gpu/TDGLGPU/test/validate.jl cuda 20
```

The first command explicitly uses KA-CPU, not a GPU. The second needs CUDA installed in that project and a real device.
Use `TDGLGPU_TEST_WORKDIR` to keep each validation in its own output directory.

## Repeated timings

`timings_repeated.json` contains all three samples per path, medians, source/configuration hashes and state errors.
The five exact inputs are in `configs/`. Each batch starts from the configured seed with a new Stepper and cold PCG
history; a discarded batch warms compilation first. All columns below are medians of three batches of twenty steps,
in ms/step. The laptop uses one P core of an i5-1235U, one Julia thread and one BLAS thread.

| Sample mesh / case | Laptop P0 | Colab CPU | T4 batch | T4 cycle | P0/T4 batch |
|---|---:|---:|---:|---:|---:|
| 24 × 12 × 6 | 0.261 | 0.586 | 1.355 | 1.393 | 0.19× |
| 48 × 24 × 12 | 2.142 | 4.408 | 2.268 | 2.405 | 0.94× |
| 96 × 48 × 24 | 22.574 | 37.818 | 7.295 | 4.816 | 3.09× |
| 96 × 48 × 24, PCG | 146.370 | 198.153 | 26.903 | 21.040 | 5.44× |
| 80 × 24 × 24, cable | 171.454 | 270.928 | 58.412 | 52.191 | 2.94× |

The backend batch includes host work inside `gpu_steps!`; the cycle additionally includes upload/download. Both exclude
JIT, Stepper construction, measurements, protocol and checkpoints. Both trajectories are compared with CPU at a 1e-6
tolerance, with no fallback. Mesh labels name the sample cells; the JSON also records padded meshes.

The batch and cycle are separate executions in a fixed order. GPU clocks and thermals were not controlled: the later
cycle can measure faster. Do not subtract the columns to infer transfer cost, or treat these as whole-application
speedups. The comparison is against one laptop core, not the laptop's combined throughput.

Reproduce a recorded input from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 julia --project=gpu/TDGLGPU gpu/TDGLGPU/scripts/benchmark.jl \
  gpu/TDGLGPU/validation/configs/pcg96.toml --steps 20 --repeats 3 --out out/benchmark-pcg
```

Add `--cpu-backend` for an explicitly labelled KA-CPU test. A full campaign needs a separate wall-time measurement.

## Historical timing sample sizes

The older warmed batches below are in `timings.json`. The laptop column is the median of three batches on one P core
of an i5-1235U (one Julia/BLAS thread). The Colab CPU and T4 columns each contain **one** warmed batch, not three independent
repetitions. Installation, compilation, cycle transfers, measurements, checkpoints and protocol work are excluded.
Consequently these numbers are observations, not an end-to-end speedup or a statistical scalability study.

| Mesh / case | Laptop ms/step | Colab CPU ms/step | T4 ms/step | Laptop/T4 |
|---|---:|---:|---:|---:|
| 24 × 12 × 6 | 0.260 | 0.693 | 1.371 | 0.19× |
| 48 × 24 × 12 | 2.027 | 4.584 | 2.336 | 0.87× |
| 96 × 48 × 24 | 21.432 | 66.283 | 7.268 | 2.95× |
| 96 × 48 × 24, PCG | 134.293 | 487.120 | 49.817 | 2.70× |
| 80 × 24 × 24, cable | 108.836 | 314.705 | 53.392 | 2.04× |

These older batches used a different warm-state/history procedure. The new repetitions do not retrospectively enlarge
their sample size, and a change between the two tables is not by itself a code regression.
No A100, other device backend, multi-GPU or cluster performance is inferred here.
