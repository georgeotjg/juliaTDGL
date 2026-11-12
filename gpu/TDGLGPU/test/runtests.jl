# Pkg.test entry point; CUDA remains optional and is exercised explicitly on a GPU.
using Test
@testset "TDGLGPU CPU backend" begin
    include("validate.jl")
end
