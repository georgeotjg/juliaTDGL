# Checkpoints: resuming = continuing, to the bit (with band substeps and extrapolated links), and
# the checkpoints of release 1.2 (JLD2) and of the releases up to 1.1 (Octave text) load.
@testset "checkpoints" begin
    par, ini, fin, mk = _load_case("k25_R4_prod")
    st = _stepper_12(par, mk)
    run(s, n) = (for _ in 1:n; step!(s, st; H=par["H"], HJ=par["HJ"]); end; s)
    a = run(_state_from(st.grid, ini), par["N1"])
    b = run(_state_from(st.grid, ini), par["N1"] ÷ 2)
    path = joinpath(mktempdir(), "state_test.jld2")
    save_state(path, b; jp=3, J=par["J"])
    c, w = load_state(path, st.grid)
    @test w["jp"] == 3 && w["nbands"] == 2 && startswith(w["tdgl_version"], "TDGL")
    run(c, par["N1"] - par["N1"] ÷ 2)
    @test _same(c.G[1], a.G[1]) && _same(c.G[2], a.G[2]) && _same(c.θx, a.θx) && _same(c.hz, a.hz)
    @test _same(c.G[2], fin["G2"])
    # an Octave text checkpoint of the reference archive
    ref = joinpath(@__DIR__, "..", "..", "data", "reference_12x8x4", "restart")
    isdir(ref) || (ref = joinpath(@__DIR__, "..", "..", "data", "reference_12x8x4", "restart"))
    f = first(filter(x -> endswith(x, ".mat"), readdir(ref; join=true)))
    s, raw = load_state(f, Grid(12, 8, 4; dx=0.5))
    @test _same(s.G[1], ComplexF64.(raw["G1"])) && _same(s.θz, Float64.(raw["thetaz"]))
end
