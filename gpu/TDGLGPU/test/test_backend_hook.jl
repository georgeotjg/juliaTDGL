using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Test
BLAS.set_num_threads(1)
@testset "backend policy, cache and execution accounting" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(12, 8, 4; dx=0.5)
    st = Stepper(md, gr, box_masks(gr); threads=false)
    s = initial_state(md, gr, st.masks, 0.03)
    ref = deepcopy(s)
    for _ in 1:2; step!(ref, st; H=0.03, HJ=0.0); end
    try
        @test_throws ArgumentError TDGLGPU.enable!(CPU(); fallback=:invalid)
        TDGLGPU.enable!(CPU(); fallback=:error)
        @test_throws ArgumentError TDGLGPU._unsupported("rejected configuration")
        TDGL._cycle_steps!(s, st, 2; H=0.03, HJ=0.0)
        q = execution_stats()
        @test q.backend == "KA-CPU" && q.backend_steps == 2 && q.cpu_fallback_steps == 0
        @test maximum(abs.(s.G[1] .- ref.G[1])) < 1e-12
        @test TDGLGPU._LAST[] !== nothing && TDGLGPU._DEV[] !== nothing
        # Re-enabling must not reuse a stepper/device allocated for another backend.
        TDGLGPU.enable!(CPU())
        @test TDGLGPU._LAST[] === nothing && TDGLGPU._DEV[] === nothing
        @test execution_stats().backend_steps == 0
        @test_logs (:warn, r"rejected configuration") TDGLGPU._unsupported("rejected configuration")
        TDGL._cycle_steps!(s, st, 1; H=0.03, HJ=0.0)
        @test execution_stats().backend_steps == 1
    finally
        TDGLGPU.disable!()
    end
    @test TDGL.STEPS[] === nothing && TDGLGPU._DEV[] === nothing
    @test execution_stats().backend_steps == 1
end
