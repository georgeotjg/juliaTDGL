# Runs the suite without Pkg.test:  julia --project=tdgl2 tdgl2/test/run.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "runtests.jl"))
