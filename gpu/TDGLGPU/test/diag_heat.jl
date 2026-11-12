# The heat model on the CPU backend after 1, 5 and 20 steps of heat_k25.toml: is the difference the ulp of cis growing, or a
# difference of the scheme?  (One step must be at the ulp; then it grows with the dynamics.)
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Printf
BLAS.set_num_threads(1)
backend = KernelAbstractions.CPU()
dir = joinpath(@__DIR__, "work"); mkpath(dir); f = joinpath(dir, "heat.toml")
write(f, replace(read(joinpath(@__DIR__, "..", "..", "..", "tdgl2", "examples", "heat_k25.toml"), String),
                 r"\[protocol\][\s\S]*" => "[run]\nJ = [0.3]\nH = 0.03\n", "[scheme]" => "[scheme]\nthreads = false"))
c = load_config(f); st = Stepper(c.model, c.grid, c.masks; c.scheme...); HJ = TDGL.selffield(st, 0.3)
s0 = initial_state(c.model, c.grid, c.masks, c.H)
for _ in 1:5; step!(s0, st; H=c.H, HJ=HJ); end
θ0 = copy(st.heat.θ)
rel(a, b) = maximum(abs, Array(a) .- Array(b)) / maximum(abs, Array(b))
for n in (1, 5, 20)
    copyto!(st.heat.θ, θ0); s = deepcopy(s0)
    for _ in 1:n; step!(s, st; H=c.H, HJ=HJ); end
    θc = copy(st.heat.θ); copyto!(st.heat.θ, θ0)
    gs = GPUStepper(st, backend); g = device_state(backend, deepcopy(s0))
    gpu_steps!(g, gs, n; H=c.H, HJ=HJ); sg = deepcopy(s0); host_state!(sg, g)
    @printf("%2d steps: θ %.2e  Ψ1 %.2e  Ψ2 %.2e  hz %.2e\n", n, rel(gs.θ, θc), rel(sg.G[1], s.G[1]), rel(sg.G[2], s.G[2]), rel(sg.hz, s.hz))
end
