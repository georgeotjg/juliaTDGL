using Test, TDGL

@testset "TDGL 2" begin
    include("test_regression12.jl")
    include("test_geometry.jl")
    include("test_io.jl")
    include("test_protocol.jl")
    include("test_config.jl")
    include("test_physics.jl")
    include("test_openbc.jl")
end
