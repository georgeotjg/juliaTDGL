# A whole ladder (run_config: protocol, verdicts, checkpoints) with TDGLGPU against the CPU code: the same rungs, the same
# verdicts, the voltages to rounding.
#   julia --project=gpu/TDGLGPU gpu/TDGLGPU/test/compare_ladder.jl [cuda] [pcg]
# pcg: the box with the insulating padding (sigma_vac; the PCG's warm start shared with the CPU solver and its checkpoints).
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Printf, TOML
BLAS.set_num_threads(1)
const CUDA_ON = "cuda" in ARGS
CUDA_ON && @eval using CUDA
backend = CUDA_ON ? CUDABackend() : KernelAbstractions.CPU()
const SMOKE = "smoke" in ARGS
dir = joinpath(get(ENV, "TDGLGPU_TEST_WORKDIR", joinpath(@__DIR__, "work")), "pcg" in ARGS ? "ladder_pcg" : "ladder"); mkpath(dir)
f = joinpath(dir, "run.toml")
write(f, """
[material]
file = "mgb2_model_II.toml"
kappa = 2.0
[mesh]
nx = $(CUDA_ON && !SMOKE ? 48 : 24)
ny = $(CUDA_ON && !SMOKE ? 24 : 12)
nz = $(CUDA_ON && !SMOKE ? 12 : 6)
dx = $(CUDA_ON && !SMOKE ? 0.25 : 0.5)
$("pcg" in ARGS ? "buf = 2" : "")
[geometry]
GEOM_MODE = "WEDGE"
[scheme]
threads = false
$("pcg" in ARGS ? "sigma_vac = 0.01" : "")
[protocol]
t_min_GL = $(SMOKE ? 1.0 : 30.0)
t_min_GL_res = $(SMOKE ? 2.0 : 60.0)
mMax = $(SMOKE ? 60 : 600)
$(SMOKE ? "lMax = 4\niter_min = 4\ndelta_v = 4\nV_win = 6\nV_win_min_res = 12" : "")
stop_after_V = 1
[run]
J = [0.3, 0.6, 0.9, 1.2]
H = 0.03
""")
for d in ("cpu", "gpu"); isdir(joinpath(dir, d)) && mv(joinpath(dir, d), joinpath(dir, d * "_" * string(time_ns())); force=true); end
TDGLGPU.disable!()
tc = @elapsed rc = run_config(f; out_dir=joinpath(dir, "cpu"), polarity=:pos)
TDGLGPU.enable!(backend)
tg = @elapsed rg = run_config(f; out_dir=joinpath(dir, "gpu"), polarity=:pos)
TDGLGPU.disable!()
a, b = rc.pos, rg.pos
same = a.J == b.J && a.conv == b.conv && a.m == b.m
dv = maximum(abs.(a.V .- b.V) ./ max.(abs.(a.V), 1e-300))
for q in eachindex(a.J)
    @printf("J=%+.2f  CPU %s V=%+.6e m=%d   %s %s V=%+.6e m=%d\n", a.J[q], a.conv[q], a.V[q], a.m[q],
            CUDA_ON ? "GPU" : "KA-CPU", b.conv[q], b.V[q], b.m[q])
end
@printf("verdicts and cycles %s; largest relative |ΔV| = %.2e; wall CPU %.1f s, %s %.1f s\n", same ? "the same" : "DIFFER",
        dv, tc, CUDA_ON ? "GPU" : "KA-CPU", tg)
state_rel = maximum(maximum(abs.(x .- y)) / max(maximum(abs, x), 1.0) for
                    (x, y) in zip((a.state.G..., a.state.θx, a.state.θy, a.state.θz),
                                  (b.state.G..., b.state.θx, b.state.θy, b.state.θz)))
abs_v = maximum(abs.(a.V .- b.V))
tol_state = "pcg" in ARGS ? 1e-6 : 1e-9
ok = same && (dv < 1e-6 || abs_v < 1e-10) && state_rel <= tol_state
open(joinpath(dir, "results.toml"), "w") do io
    TOML.print(io, Dict("backend" => CUDA_ON ? "CUDA" : "KA-CPU", "smoke" => SMOKE,
                       "same_verdicts_cycles" => same, "voltage_relative" => dv, "voltage_absolute" => abs_v,
                       "state_relative" => state_rel, "state_tolerance" => tol_state,
                       "cpu_seconds" => tc, "backend_seconds" => tg, "passed" => ok,
                       "verdicts" => String(a.conv)))
end
SMOKE && println("Smoke protocol: tests the ladder plumbing; it is not a resolved scientific onset measurement.")
ok || exit(1)
