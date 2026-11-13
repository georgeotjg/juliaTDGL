include("pcg_zero_fixture.jl")
@testset "PCG zero source and checkpoint" begin
    pcg_zero_rhs_checks(st -> s -> TDGL.implicit_curl!(s, st.imp))
end
