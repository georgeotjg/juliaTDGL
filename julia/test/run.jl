# Shortcut: runs the suite without Pkg.test (faster, no temporary environment).
#   julia --project=julia julia/test/run.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..") ; io=devnull)
include(joinpath(@__DIR__, "runtests.jl"))
