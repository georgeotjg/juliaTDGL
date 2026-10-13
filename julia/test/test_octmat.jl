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

@testset "JLD2 checkpoints: exact, and the same state as the Octave text" begin
    d = read_state(joinpath(FIXT, "restart", "state_Jpos.mat"))
    dir = mktempdir()
    r = write_jld2(joinpath(dir, "rt.jld2"), "G1" => d["G1"], "hz" => d["hz"], "jp" => 7.0,
                   "v" => reshape([1.5, -2.25], :, 1))
    @test !isfile(r * ".tmp")
    e = read_state(r)
    @test all(e["G1"] .=== d["G1"]) && all(e["hz"] .=== d["hz"])
    @test e["jp"] == 7.0 && vec(e["v"]) == [1.5, -2.25]
    @test e["tdgl_version"] == string(pkgversion(TDGL))
    @test !TDGL._es_mat_binario(joinpath(FIXT, "restart", "state_Jpos.mat"))

    # checkpoint_path: the one that exists, the newer one if both do, the .jld2 name if neither
    @test checkpoint_path(dir, "X") == joinpath(dir, "state_X.jld2")
    touch(joinpath(dir, "state_Y.mat"))
    @test checkpoint_path(dir, "Y") == joinpath(dir, "state_Y.mat")
    sleep(1.1); touch(joinpath(dir, "state_Y.jld2"))
    @test checkpoint_path(dir, "Y") == joinpath(dir, "state_Y.jld2")
    sleep(1.1); touch(joinpath(dir, "state_Y.mat"))
    @test checkpoint_path(dir, "Y") == joinpath(dir, "state_Y.mat")
    @test_throws ErrorException derive_params(cfg_fixture(checkpoint_format="hdf5"))

    # the same short run saved in both formats reloads to the same State, bit for bit
    estados = Dict{String,Any}()
    for fmt in ("jld2", "octave")
        out = mktempdir()
        cfg = cfg_fixture(J_list=[0.0, 0.1], mMax=20, out_dir=out, checkpoint_format=fmt)
        p = derive_params(cfg); g = build_geometry(cfg, p)
        TDGL.sweep(current_grid(cfg), fill(cfg.H_const, 2), "J+", g, p, Strict())
        ruta = checkpoint_path(joinpath(out, "restart"), "Jpos")
        @test endswith(ruta, fmt == "jld2" ? ".jld2" : ".mat") && isfile(ruta)
        estados[fmt] = (p, load_checkpoint(ruta, p, Strict()))
    end
    (_, (sa, ja, Va, Ja)), (_, (sb, jb, Vb, Jb)) = estados["jld2"], estados["octave"]
    for f in (:G1, :G2, :thetax, :thetay, :thetaz, :hx, :hy, :hz, :phix, :phiy, :phiz)
        @test all(getfield(sa, f) .=== getfield(sb, f))
    end
    @test ja == jb && all(Va .=== Vb) && all(Ja .=== Jb)
end
