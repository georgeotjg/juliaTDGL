# A long comparison in one configuration: RUNDIR's run file, from a kept state, n steps at current J on the CPU and with
# TDGLGPU (CPU backend), the states compared every `every` steps (the ulp of cis growing with the dynamics, or a difference).
#   julia --project=gpu/TDGLGPU gpu/TDGLGPU/test/compare_long.jl RUNDIR STATE.jld2 J n every
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Printf
BLAS.set_num_threads(1)
dir, sf = ARGS[1], ARGS[2]; J = parse(Float64, ARGS[3]); n = parse(Int, ARGS[4]); ev = parse(Int, ARGS[5])
c = load_config(joinpath(dir, "run.toml")); st = Stepper(c.model, c.grid, c.masks; c.scheme...); HJ = TDGL.selffield(st, J)
s0, _ = load_state(sf, c.grid)
backend = KernelAbstractions.CPU()
rel(a, b) = maximum(abs, Array(a) .- Array(b)) / max(maximum(abs, Array(b)), 1e-300)
s = deepcopy(s0); sn = TDGL._pcg_snap(st)
cpu = State[]
for m in 1:n
    step!(s, st; H=c.H, HJ=HJ)
    m % ev == 0 && push!(cpu, deepcopy(s))
end
TDGL._pcg_back!(st, sn); TDGL._pcg_reset!(st)
TDGL._pcg_back!(st, sn)
gs = GPUStepper(st, backend); g = device_state(backend, s0)
gs.sep isa TDGLGPU.GPCG && TDGLGPU._pcg_up!(gs.sep, st.imp)
sg = deepcopy(s0)
for q in 1:length(cpu)
    gpu_steps!(g, gs, ev; H=c.H, HJ=HJ); host_state!(sg, g)
    @printf("step %5d: Ψ1 %.2e  Ψ2 %.2e  hz %.2e  θx %.2e\n", q * ev, rel(sg.G[1], cpu[q].G[1]), rel(sg.G[2], cpu[q].G[2]),
            rel(sg.hz, cpu[q].hz), rel(sg.θx, cpu[q].θx))
end
