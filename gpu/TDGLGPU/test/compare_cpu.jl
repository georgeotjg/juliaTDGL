# TDGLGPU against TDGL.step!: n steps of each configuration from the same state, every field of the state compared.
#   julia --project=gpu/TDGLGPU gpu/TDGLGPU/test/compare_cpu.jl [cuda] [n]
# Without `cuda`: the CPU backend of KernelAbstractions (every phase to the bit but the ulp of cis: diag_fields.jl).  With `cuda` (a GPU, and
# CUDA in the environment): the CUDA backend, to rounding, and the time of a step on each side.
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Printf, TOML
BLAS.set_num_threads(1)
const CUDA_ON = "cuda" in ARGS
CUDA_ON && @eval using CUDA
backend = CUDA_ON ? CUDABackend() : KernelAbstractions.CPU()
nst = something(tryparse(Int, get(filter(a -> a != "cuda", ARGS), 1, "")), 20)

base(mesh, extra, shape="") = """
[material]
file = "mgb2_model_II.toml"
kappa = 2.0
[mesh]
$mesh
[geometry]
$(isempty(shape) ? "GEOM_MODE = \"WEDGE\"" : "GEOM_MODE = \"CUSTOM\"\ngeom_file = \"$shape\"")
[scheme]
threads = false
$extra
[run]
J = [0.3]
H = 0.03
"""
small = "nx = 24\nny = 12\nnz = 6\ndx = 0.5"
cases = [
    ("production: RKC, extrapolated links, implicit, bar", small, ""),
    ("Euler substeps (rkl = false)", small, "rkl = false"),
    ("links not extrapolated", small, "extrap = false"),
    ("slab boundary", small, "selffield_bc = \"slab\""),
    ("RKL stages (rkl = true)", small, "rkl = true"),
    ("insulating padding: PCG, pcg_fast + Float32 (defaults)", small * "\nbuf = 2", "sigma_vac = 0.01", 1e-7),
    ("insulating padding: PCG exact (pcg_fast = false)", small * "\nbuf = 2", "sigma_vac = 0.01\npcg_fast = false\npcg_f32 = false"),
    ("insulating padding, explicit field step", small * "\nbuf = 2", "sigma_vac = 0.01\nimplicit = false"),
    ("a cable carved in vacuum (cable_x.toml, PCG)", "nx = 40\nny = 12\nnz = 12\ndx = 0.5\nbuf = 2", "sigma_vac = 0.01", 1e-7, "cable_x.toml"),
]
dir = joinpath(get(ENV, "TDGLGPU_TEST_WORKDIR", joinpath(@__DIR__, "work")), "steps"); mkpath(dir)
const RESULTS = Dict{String,Any}[]
three = """
[material]
name = "three frustrated bands"
kappa = 2.0
sigma_n = 1.0
gamma = [[0.0, -0.1, -0.1], [-0.1, 0.0, -0.1], [-0.1, -0.1, 0.0]]
[[material.band]]
name = "1"
[[material.band]]
name = "2"
[[material.band]]
name = "3"
[mesh]
$small
[geometry]
GEOM_MODE = "WEDGE"
[scheme]
threads = false
[run]
J = [0.3]
H = 0.03
"""

maxrel(a, b) = (d = maximum(abs, Array(a) .- Array(b)); m = maximum(abs, Array(b)); m == 0 ? d : d / m)
function compare(name, text; n=nst, tol=nothing)
    f = joinpath(dir, "case.toml"); write(f, text)
    c = load_config(f)
    st = Stepper(c.model, c.grid, c.masks; c.scheme...)
    # rounding, or the tolerance of the fast PCG when the field takes it (a padding, σn per region)
    tol === nothing && (tol = st.imp isa TDGL.PCGSolver && st.imp.fast !== nothing ? 1e-7 : 1e-10)
    HJ = TDGL.selffield(st, 0.3)
    s0 = initial_state(c.model, c.grid, c.masks, c.H)
    for _ in 1:5; step!(s0, st; H=c.H, HJ=HJ); end                  # off the seed: links and currents nonzero
    sn = TDGL._pcg_snap(st)                                           # the warm start of the padding's PCG, for both
    θ0 = st.heat === nothing ? nothing : copy(st.heat.θ)              # and the temperature field (the Stepper holds it)
    r0 = st.noise === nothing ? nothing : copy(st.noise.rng)          # and the noise stream
    o0 = TDGL._open_snap(st)                                          # and the ramp of the open boundary
    s = deepcopy(s0); sg = deepcopy(s0)
    t_cpu = @elapsed for _ in 1:n; step!(s, st; H=c.H, HJ=HJ); end
    θc = θ0 === nothing ? nothing : copy(st.heat.θ)                   # the CPU run's temperature
    TDGL._pcg_back!(st, sn); θ0 === nothing || copyto!(st.heat.θ, θ0); r0 === nothing || copy!(st.noise.rng, r0); TDGL._open_back!(st, o0)
    gs = GPUStepper(st, backend); g = device_state(backend, sg)
    gpu_steps!(g, gs, 1; H=c.H, HJ=HJ)                                # compile
    device_state!(g, s0); TDGL._pcg_back!(st, sn); θ0 === nothing || copyto!(gs.θ, θ0); r0 === nothing || copy!(st.noise.rng, r0)
    TDGL._open_back!(st, o0); gs.open === nothing || TDGLGPU._open_up!(gs)
    gs.sep isa TDGLGPU.GPCG && TDGLGPU._pcg_up!(gs.sep, st.imp)
    t_gpu = @elapsed gpu_steps!(g, gs, n; H=c.H, HJ=HJ)
    host_state!(sg, g)
    θd = θ0 === nothing ? 0.0 : maxrel(gs.θ, θc)
    names = vcat(["θ (heat)"], ["Ψ$µ" for µ in eachindex(s.G)], ["θx", "θy", "θz", "hx", "hy", "hz", "Jx", "Jy", "Jz"])
    # a vector field against the size of the whole vector (a component nearly zero, hx of a field along z, would inflate
    # its own relative difference)
    vs(qs) = maximum(q -> maximum(abs, Array(getfield(s, q))), qs)
    vrel(q, qs) = (m = vs(qs); d = maximum(abs, Array(getfield(sg, q)) .- Array(getfield(s, q))); m == 0 ? d : d / m)
    ds = vcat([θd], [maxrel(sg.G[µ], s.G[µ]) for µ in eachindex(s.G)],
              [vrel(q, qs) for qs in ((:θx, :θy, :θz), (:hx, :hy, :hz), (:Jx, :Jy, :Jz)) for q in qs])
    d = maximum(ds)
    d > tol && println("    largest in ", names[argmax(ds)], "; the others: ", join((@sprintf("%s %.1e", a, b) for (a, b) in zip(names, ds)), ", "))
    @printf("%-56s N=%d  %d steps: max rel |Δ| = %.3e %s  step CPU %.3f ms, %s %.3f ms\n", name, length(s.G), n, d,
            d <= tol ? "ok  " : "FAIL", 1e3 * t_cpu / n, CUDA_ON ? "GPU" : "KA-CPU", 1e3 * t_gpu / n)
    push!(RESULTS, Dict("name" => name, "bands" => length(s.G), "steps" => n,
                        "max_relative" => d, "tolerance" => tol, "passed" => d <= tol,
                        "cpu_ms" => 1e3 * t_cpu / n, "backend_ms" => 1e3 * t_gpu / n))
    open(joinpath(dir, "results.toml"), "w") do io
        TOML.print(io, Dict("backend" => CUDA_ON ? "CUDA" : "KA-CPU", "cases" => RESULTS))
    end
    flush(stdout)
    d / tol
end

println("TDGLGPU on ", CUDA_ON ? "CUDA ($(CUDA.name(CUDA.device())))" : "the CPU backend of KernelAbstractions")
worst = 0.0
for cs in cases                  # (name, mesh, scheme keys[, tolerance]): the PCG of pcg_fast is exact to its pcg_tol only
    global worst = max(worst, compare(cs[1], base(cs[2], cs[3], get(cs, 5, "")); tol=get(cs, 4, nothing)))
end
worst = max(worst, compare("three frustrated bands", three))
worst = max(worst, compare("three bands with three normal fluids (a matrix of rates)",
                           replace(three, "[scheme]" => "[scheme]\nqp_exchange = [[0.0, 0.5, 0.2], [0.5, 0.0, 0.3], [0.2, 0.3, 0.0]]\nqp_split = [1.0, 1.0, 1.0]",
                                   "name = \"3\"" => "name = \"3\"\na = 0.7")))     # band 3 weaker: the potentials do not cancel
# the heat model (the in-place sweep of θ by planes on the device) and drag: the example heat_k25.toml (both), with KWT too
hk = replace(read(joinpath(@__DIR__, "..", "..", "..", "tdgl2", "examples", "heat_k25.toml"), String),
             r"\[protocol\][\s\S]*" => "[run]\nJ = [0.3]\nH = 0.03\n", "[scheme]" => "[scheme]\nthreads = false")
worst = max(worst, compare("heat (Joule, margin) and drag: heat_k25.toml", hk))
worst = max(worst, compare("heat without drag", replace(hk, r"drag = .*" => "")))
# the open boundary (its field recomputed on the host every `every` steps, its ramp on the device)
worst = max(worst, compare("open boundary (selffield_bc = open), 150 steps: ramps and recomputations", base(small, "selffield_bc = \"open\""); n=150))
worst = max(worst, compare("open boundary with the insulating padding, 150 steps", base(small * "\nbuf = 2", "selffield_bc = \"open\"\nsigma_vac = 0.01"); n=150))
# two normal fluids with a finite interband exchange (qp_exchange): in the box, with the padding, and with KWT (coupled normal currents)
worst = max(worst, compare("normal fluids: qp_exchange = 1", base(small, "qp_exchange = 1.0")))
worst = max(worst, compare("normal fluids: qp_exchange = 0.1, insulating padding", base(small * "\nbuf = 2", "qp_exchange = 0.1\nsigma_vac = 0.01")))
worst = max(worst, compare("normal fluids in a cable carved in vacuum (their CG)", base("nx = 40\nny = 12\nnz = 12\ndx = 0.5\nbuf = 2", "sigma_vac = 0.01\nqp_exchange = 0.5", "cable_x.toml"); tol=1e-7))
worst = max(worst, compare("normal fluids with heat (their Joule heat in the source)", base(small, "qp_exchange = 0.5") * "[heat]\njoule = true\n"))
# thermal noise without KWT: the seeded stream drawn on the host; on the bands, and on the field too
worst = max(worst, compare("noise T̃ = 1e-2 on the bands", base(small, "noise = 1e-2\nnoise_seed = 7")))
worst = max(worst, compare("noise T̃ = 1e-2 on the bands and the field", base(small, "noise = 1e-2\nnoise_seed = 7\nnoise_field = true")))
worst = max(worst, compare("noise on the field with the insulating padding", base(small * "\nbuf = 2", "sigma_vac = 0.01\nnoise = 1e-2\nnoise_seed = 7\nnoise_field = true")))
# materials per region: the examples of the catalogue (a tunnel-coupled bilayer; a one-band metal next to the two-band sample)
for ex in ("bicapa_tunel.toml", "metal_una_banda.toml")
    txt = read(joinpath(@__DIR__, "..", "..", "..", "tdgl2", "examples", ex), String)
    txt = replace(txt, r"\[run\][\s\S]*" => "[run]\nJ = [0.3]\nH = 0.03\n", "qp_exchange = 1.0" => "")   # the stepper; one fluid
    for shp in (m.captures[1] for m in eachmatch(r"geom_file\s*=\s*\"([^\"]+)\"", txt))
        for d in ("shapes", "examples")
            src = joinpath(@__DIR__, "..", "..", "..", "tdgl2", d, shp)
            isfile(src) && cp(src, joinpath(dir, shp); force=true)
        end
    end
    global worst = max(worst, compare("materials: " * ex, txt))
end
# normal fluids with materials: the one-band metal next to the two-band sample with its qp_exchange (a one-fluid region: QPVar),
# and the bilayer with two fluids in both materials (the conversion with ρ and the node's Josephson row)
mub = read(joinpath(@__DIR__, "..", "..", "..", "tdgl2", "examples", "metal_una_banda.toml"), String)
worst = max(worst, compare("normal fluids with materials: metal_una_banda.toml (QPVar)",
                           replace(mub, r"\[run\][\s\S]*" => "[run]\nJ = [0.3]\nH = 0.03\n")))
worst = max(worst, compare("normal fluids with materials: bilayer, qp_exchange = 0.5",
                           replace(read(joinpath(@__DIR__, "..", "..", "..", "tdgl2", "examples", "bicapa_tunel.toml"), String),
                                   r"\[run\][\s\S]*" => "[run]\nJ = [0.3]\nH = 0.03\n") * "[scheme]\nqp_exchange = 0.5\nqp_split = [0.5, 0.5]\n"))
# heat with materials: the bilayer with one D_T (the in-place sweep by planes) and with D_T per material (Jacobi)
bil = replace(read(joinpath(@__DIR__, "..", "..", "..", "tdgl2", "examples", "bicapa_tunel.toml"), String), r"\[run\][\s\S]*" => "[run]\nJ = [0.3]\nH = 0.03\n")
worst = max(worst, compare("heat with materials, one D_T (bilayer)", bil * "[heat]\njoule = true\n"))
worst = max(worst, compare("noise with materials (bilayer, T̃ = 1e-2)", bil * "[scheme]\nnoise = 1e-2\nnoise_seed = 3\n"))
bilk = replace(bil, "Gamma = 2.0\na = 0.8" => "Gamma = 2.0\na = 0.8\ngamma_kwt = 10.0", "Gamma = 2.0\na = 0.5" => "Gamma = 2.0\na = 0.5\ngamma_kwt = 10.0")
occursin("gamma_kwt = 10.0", bilk) || error("the KWT bilayer was not built")
worst = max(worst, compare("noise with materials and KWT (bilayer, γ̃ = 10, RKC)", bilk * "[scheme]\nnoise = 1e-2\nnoise_seed = 3\n"))
worst = max(worst, compare("heat with materials, D_T per material (bilayer)",
                           replace(bil, r"^(\[materials\.[^\]]+\])"m => s"\1\nheat_D_T = 0.1") * "[heat]\njoule = true\n"))
kwt(γ, extra) = """
[material]
name = "two-band notched-box model, kappa = 2, with KWT"
kappa = 2.0
sigma_n = 1.0
T = 0.0
gamma12 = 1e-3
[[material.band]]
name = "sigma"
m = 1.0
Gamma = 1.0
a = 1.0
gamma_kwt = $γ
[[material.band]]
name = "pi"
xi_ratio = 4.0
Gamma = 1.0
a = 0.59
gamma_kwt = $γ
[mesh]
$small
[geometry]
GEOM_MODE = "WEDGE"
[scheme]
threads = false
$extra
[run]
J = [0.3]
H = 0.03
"""
worst = max(worst, compare("KWT γ̃ = 10: rotating frame, RKC", kwt(10.0, "")))
worst = max(worst, compare("KWT γ̃ = 10: rotating frame, Euler substeps", kwt(10.0, "rkl = false")))
worst = max(worst, compare("KWT γ̃ = 30: rotating frame, RKC", kwt(30.0, "")))
worst = max(worst, compare("KWT γ̃ = 10 with heat (the KWT dissipation as its source)", kwt(10.0, "") * "[heat]\njoule = true\n"))
worst = max(worst, compare("normal fluids with KWT γ̃ = 10 (qp_exchange = 0.3)", kwt(10.0, "qp_exchange = 0.3")))
# noise with KWT: the draws before the bands, SK-ROCK's phase kick in the first RKC stage, the log-polar stages, the smoothed
# frame as a rotation, the amplitude after (RKC); and the polar kick after Euler substeps
worst = max(worst, compare("noise with KWT γ̃ = 10, RKC (SK-ROCK), T̃ = 1e-2", kwt(10.0, "noise = 1e-2\nnoise_seed = 5")))
worst = max(worst, compare("noise with KWT γ̃ = 10, Euler substeps", kwt(10.0, "rkl = false\nnoise = 1e-2\nnoise_seed = 5")))
if CUDA_ON     # the same box, finer: where a GPU fills up
    for (m, nn) in (("nx = 48\nny = 24\nnz = 12\ndx = 0.25", 20), ("nx = 96\nny = 48\nnz = 24\ndx = 0.125", 10))
        global worst = max(worst, compare("production, mesh " * replace(m, "\n" => " "), base(m, ""); n=nn))
    end
    # the insulating padding and a carved cable, finer: where the GPU is meant to pay (the PCG's cost there)
    global worst = max(worst, compare("insulating padding, mesh 96x48x24", base("nx = 96\nny = 48\nnz = 24\ndx = 0.125\nbuf = 4",
                                      "sigma_vac = 0.01"); n=10, tol=1e-7))
    global worst = max(worst, compare("cable in vacuum, mesh 80x24x24", base("nx = 80\nny = 24\nnz = 24\ndx = 0.25\nbuf = 4",
                                      "sigma_vac = 0.01", "cable_x.toml"); n=10, tol=1e-7))
end
# On the CPU backend every phase of the step gives the bits except the links, cis(−φ), which the device compiler takes from
# its own libm (one ulp: test/diag_fields.jl); after 20 steps that ulp is ~1e-12 of the state (the fast PCG: its tolerance).
ok = worst <= 1
println(ok ? "every case within its tolerance: PASS" : "a case beyond its tolerance: FAIL")
ok || exit(1)
