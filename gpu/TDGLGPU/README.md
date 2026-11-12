# TDGLGPU

Optional accelerator for the N-band TDGL 2 solver. It advances the same state and equations as the CPU implementation
with KernelAbstractions kernels and device matrix products. State arrays remain Float64/ComplexF64; the fast insulating
PCG may use a Float32 preconditioner, as on the CPU.

## Installation

From the repository root, with Julia 1.13:

```bash
julia --project=gpu/TDGLGPU -e 'using Pkg; Pkg.instantiate()'
```

The project locates TDGL through its relative `[sources]` entry. Keep the package in the repository layout. For CUDA,
install the optional backend in this environment on a machine with a compatible NVIDIA driver:

```bash
julia --project=gpu/TDGLGPU -e 'using Pkg; Pkg.add("CUDA"); using CUDA; CUDA.versioninfo()'
```

CUDA is not required to use or test the CPU backend. No GPU access is provided by installing this package.

## Run a ladder

```julia
using TDGL, TDGLGPU, CUDA
TDGLGPU.enable!(CUDABackend(); fallback=:error)
try
    run_config("run.toml"; out_dir="out/example")
finally
    TDGLGPU.disable!()
end
```

Or use the command-line wrapper:

```bash
julia --project=gpu/TDGLGPU gpu/TDGLGPU/scripts/run_gpu.jl run.toml --dir out/example --require-gpu
julia --project=gpu/TDGLGPU gpu/TDGLGPU/scripts/run_gpu.jl run.toml --dir out/cpu-check --cpu-backend
julia --project=gpu/TDGLGPU gpu/TDGLGPU/scripts/with_gpu.jl tdgl2/scripts/run.jl run.toml --dir out/example
```

The wrappers accept `--help` and `--require-gpu`; `run_gpu.jl` also accepts `--resume` and `--polarity +|-`. Strict mode
requires functional CUDA and errors on unsupported steppers. It cannot be combined with `--cpu-backend`. It does not
move host responsibilities onto the device. Both wrappers report the backend and accelerated/fallback step counts;
`execution_stats()` exposes these counts in Julia. KA-CPU is named explicitly. There is no `gpu=true` run-file key.
The low-level interface is `GPUStepper`, `device_state`, `gpu_steps!` and `host_state!`.

## Scope

Implemented paths include N bands, KWT/rotating frame, Euler/RKL/RKC stages, explicit or implicit field, insulating PCG,
sculpted vacuum, regional materials, heat, drag, thermal noise (including KWT and regional materials), two/N normal fluids,
and bar/slab/open boundaries. The CPU's combination restrictions still apply: for example N normal fluids with regional
materials and drag with regional materials are rejected by the CPU configuration too.

`supported(st)` reports implementation coverage, not a proof that every backend or combination has been validated.
An unsupported Stepper falls back to CPU with a warning by default; `enable!(backend; fallback=:error)` forbids that fallback.
Re-enabling clears the device cache and counters, including when changing backend. The hook is process-global: concurrent
ladders need separate Julia processes. Only CUDA and the KernelAbstractions CPU backend have validation
evidence; other device backends are not certified by this package.

`enable!` replaces **ladder cycles**. It transfers the state at each cycle and leaves measurements, verdicts and checkpoints
on the host. Direct `TDGL.step!` calls, delay confirmation, `crossing_times`, and tools using such calls remain on CPU. The
open boundary computes its FFT on the host; the seeded random stream is generated there too. This is not an entirely
device-resident application.

`--resume` reads the normal JLD2 checkpoints, including PCG history, the open-boundary ramp, temperature and seeded RNG.
Continuations need not have identical bits across CPU/device arithmetic: link exponentials are reconstructed on CPU when
loading a checkpoint. A short KA-CPU noise/KWT/PCG restart comparison differed by about 4e-13 in Ψ, with identical RNG,
verdicts and cycles. Deterministic bar/slab/open and heat cases are checked separately, including the next open-boundary
FFT update. These short checks do not establish invariance of every critical-current bracket.

## Validation

```bash
# Standard CPU-only entry point: analytic checks, 37 step comparisons, short ladders with/without PCG.
OPENBLAS_NUM_THREADS=1 julia --project=gpu/TDGLGPU -e 'using Pkg; Pkg.test()'

# On an actual CUDA device: the same checks plus four larger step cases (41 step cases in total).
OPENBLAS_NUM_THREADS=1 julia --project=gpu/TDGLGPU gpu/TDGLGPU/test/validate.jl cuda 20
```

The runner writes environment versions, source SHA-256 values, per-case tolerances/errors and timings to
`out/gpu_validation/`. Set `TDGLGPU_TEST_WORKDIR` to choose another destination. The step comparison uses each vector field's
whole magnitude for normalization; the fast PCG cases use a looser, explicitly recorded tolerance (usually 1e-7 versus
1e-10). A passing short ladder tests protocol/checkpoint plumbing, not a resolved critical-current measurement.

The insulating field PCG treats a zero source explicitly, including a warm or extrapolated initial guess. A shared CPU/device regression checks decoupled links, history and checkpoint continuation in four PCG modes.

Analytic checks cover a uniform equilibrium, radial GL/KWT relaxation and a stationary pure lattice gauge. Numerical
CPU/device agreement alone is not an independent validation of the physical model or its continuum limit.

Validation snapshot (12 November 2026): 37 CPU-backend cases and 41 Tesla T4 cases passed, plus eight analytic assertions
and two short ladders on each backend. The zero-source PCG and checkpoint regression passed 217 assertions on CPU, KA-CPU and CUDA. The largest T4 step error was 7.136e-8 at its explicit 1e-7 tolerance; PCG ladder
voltages differed by at most 9.587e-11 relatively, with identical verdicts/cycles. Five CUDA checkpoint continuations passed
38 assertions; the strict CLI and the wrapped CPU CLI both executed CUDA without fallback. A completed strict-CLI resume
preserved all eleven numerical output files. CUDA run: Julia 1.13.2, CUDA.jl 6.5.1,
KernelAbstractions 0.10.7 and Adapt 4.8.0. This records the tested suite and environment, not every possible combination.

The repository CI runs the CPU backend. CUDA validation requires a separate GPU run; a CPU CI badge does not certify CUDA.
Small meshes may run slower on GPU because of launch and transfer overhead. Performance claims must name the device,
mesh, scheme, host CPU and measured workload. No A100 performance is inferred from a T4 run.

The [recorded validation and timing data](validation/README.md) distinguish CUDA measurements from CPU-backend tests and
single timing batches from repeated measurements. A benchmark of one configuration can be run with
`scripts/benchmark.jl RUN.toml --out out/benchmark` (or `--cpu-backend`); see its `--help` for repetitions and batch size.
