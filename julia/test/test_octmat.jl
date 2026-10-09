# ---------------------------------------------------------------------------
#  Reader/writer of the Octave text format.  Without it there is no acceptance
#  test: MAT.jl does not open these files (`is not a MAT file`).
# ---------------------------------------------------------------------------
@testset "octmat: reading and writing Octave text files" begin
    d = read_state(joinpath(FIXT, "restart", "state_Jpos.mat"))
    @test haskey(d, "G1") && haskey(d, "hz") && haskey(d, "jp")
    @test size(d["G1"]) == (15, 11, 7)
    @test eltype(d["G1"]) == ComplexF64
    @test size(d["thetax"]) == (14, 11, 7)
    @test size(d["hx"]) == (15, 10, 6)
    @test d["jp"] == 1.0

    # column order: the buffer nodes (i=1, j=1, k=1) are zero and the first
    # node of the SC block is not.  A row-major reshape would fail here.
    @test d["G1"][1,1,1] == 0
    @test abs(d["G1"][9,2,2]) > 0.5

    # exact round trip: 17 digits are needed and written
    tmp = joinpath(mktempdir(), "rt.mat")
    write_octave_text(tmp, "G1" => d["G1"], "hz" => d["hz"],
                              "jp" => 7.0, "v" => reshape([1.5, -2.25], :, 1))
    e = read_state(tmp)
    @test all(e["G1"] .=== d["G1"])
    @test all(e["hz"] .=== d["hz"])
    @test e["jp"] == 7.0
    @test vec(e["v"]) == [1.5, -2.25]
end
