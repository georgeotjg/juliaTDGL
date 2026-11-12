# Bounded validation with machine-readable results under out/ (or an explicit work folder).
# julia --project=gpu/TDGLGPU gpu/TDGLGPU/test/validate.jl [cuda] [20]
using TDGL, TDGLGPU, KernelAbstractions, Pkg, TOML, SHA, Dates, LinearAlgebra
BLAS.set_num_threads(1)
work = abspath(get(ENV, "TDGLGPU_TEST_WORKDIR", joinpath(pwd(), "out", "gpu_validation")))
ENV["TDGLGPU_TEST_WORKDIR"] = work
mkpath(work)
versions = Dict(string(p.name) => string(p.version) for p in values(Pkg.dependencies()) if p.version !== nothing)
hashes = Dict{String,String}()
for (label, root) in (("TDGL", pkgdir(TDGL)), ("TDGLGPU", pkgdir(TDGLGPU)))
    for (dir, _, files) in walkdir(joinpath(root, "src")), f in files
        endswith(f, ".jl") || continue
        path = joinpath(dir, f); hashes[label * "/" * relpath(path, root)] = bytes2hex(sha256(read(path)))
    end
end
meta = Dict("started_utc" => string(now(UTC)), "julia" => string(VERSION), "cpu" => Sys.CPU_NAME,
            "backend" => "cuda" in ARGS ? "CUDA" : "KA-CPU", "blas" => string(BLAS.get_config()),
            "versions" => versions, "source_sha256" => hashes)
if "cuda" in ARGS
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional")
    meta["device"] = CUDA.name(CUDA.device())
    open(joinpath(work, "cuda_versioninfo.txt"), "w") do io
        CUDA.versioninfo(io)
    end
end
open(io -> TOML.print(io, meta), joinpath(work, "environment.toml"), "w")

module BackendChecks
include("test_backend_hook.jl")
end
module CheckpointContinuation
include("test_resume.jl")
end
module PCGZeroSource
include("test_pcg_zero_rhs.jl")
end
module AnalyticChecks
include("test_invariants.jl")
end
module StepComparison
include("compare_cpu.jl")
end
push!(ARGS, "smoke")
module LadderComparison
include("compare_ladder.jl")
end
push!(ARGS, "pcg")
module PCGLadderComparison
include("compare_ladder.jl")
end
println("VALIDATION PASS; results: ", work)
