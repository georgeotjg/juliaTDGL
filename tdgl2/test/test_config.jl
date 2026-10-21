# A run from a file gives the ladder of the same run built by hand (the golden one of 1.2).
@testset "run from a file" begin
    dir = mktempdir()
    f = joinpath(dir, "run.toml")
    write(f, """
    [material]
    file = "mgb2_model_II.toml"
    [mesh]
    nx = 24
    ny = 12
    nz = 4
    dx = 0.5
    [geometry]
    GEOM_MODE = "WEDGE"
    inj_margin = 4
    [scheme]
    threads = false
    selffield_bc = "slab"
    rkl = false
    [protocol]
    t_min_GL = 20.0
    t_min_GL_res = 60.0
    mMax = 4000
    delta_v = 6
    v_margin = 1.0
    f_strict = false
    [run]
    J = [0.30, 0.60, 0.90, 1.20, 1.50]
    H = 0.03
    """)
    c = load_config(f)
    d = JLD2.load(joinpath(@__DIR__, "golden", "ladder_k25.jld2"))
    @test c.model.bands[2].m === d["mr"]
    r = run_config(f; out_dir=joinpath(dir, "out"))
    @test _same(r.pos.V, d["pos"]["V"]) && _same(r.neg.V, d["neg"]["V"])
    @test (r.brackets.pos.lo, r.brackets.pos.hi) == (d["lo_pos"], d["hi_pos"])
    @test isfile(joinpath(dir, "out", "output", "summary.json"))
    @test isfile(joinpath(dir, "out", "output", "rungs_J+.csv")) && isfile(joinpath(dir, "out", "restart", "state_J-.jld2"))
    js = read(joinpath(dir, "out", "output", "summary.json"), String)
    @test occursin("\"bracket_pos\": [1.2, 1.5]", js) && occursin("D_ratio", js) && occursin("bracket_I_pos", js)
    # the files the menu reads: one history line per closed point, the cycle length, the pid, done
    o = joinpath(dir, "out")
    h = readlines(joinpath(o, "j_history.csv"))
    @test h[1] == "time,pol,J,V,conv,it,Nv_s,Nv_p,psi_p2" && length(h) == 1 + length(r.pos.J) + length(r.neg.J)
    @test split(h[2], ',')[2] == "J+" && split(h[end], ',')[2] == "J-"
    @test occursin("t_cycle=", read(joinpath(o, "run_info.txt"), String))
    @test strip(read(joinpath(o, "julia.pid"), String)) == string(getpid())
    @test endswith(strip(read(joinpath(o, "STATUS.txt"), String)), "done")
    # a folder that holds a ladder is not run over; resuming an ended run gives it back without a step
    @test_throws ErrorException run_config(f; out_dir=o)
    r2 = run_config(f; out_dir=o, resume=true)
    @test r2.pos.V == r.pos.V && r2.neg.V == r.neg.V && r2.summary["result"]["bracket_pos"] == [1.2, 1.5]
    # one polarity per process into the same folder: the second to end writes the summary, the same ladders
    p2 = joinpath(dir, "split")
    a = run_config(f; out_dir=p2, polarity=:neg)
    @test a.brackets === nothing && a.summary === nothing && !isfile(joinpath(p2, "output", "summary.json"))
    b = run_config(f; out_dir=p2, polarity=:pos)
    @test b.pos.V == r.pos.V && b.neg.V == r.neg.V && b.pos.conv == r.pos.conv && b.neg.state.G[1] == r.neg.state.G[1]
    @test isfile(joinpath(p2, "output", "summary.json")) && endswith(strip(read(joinpath(p2, "STATUS.txt"), String)), "done")
    # the optional terms from the file: [heat] and drag in [scheme]
    g = joinpath(dir, "heat.toml")
    write(g, replace(read(f, String), "rkl = false" => "rkl = false\ndrag = 0.05") * "[heat]\nD_T = 0.25\neta = 0.05\njoule = true\nmargin = 2\n")
    cg = load_config(g); sg = Stepper(cg.model, cg.grid, cg.masks; cg.scheme...)
    @test sg.heat !== nothing && sg.heat.h.joule && sg.heat.h.margin == 2 && sg.dragν[1][2] == 0.05
    @test sg.dt == Stepper(cg.model, cg.grid, cg.masks; threads=false, selffield_bc=:slab, rkl=false, drag=0.05,
                           heat=Heat(D_T=0.25, eta=0.05, joule=true, margin=2)).dt
end

# The band currents through the mid plane add up to the current the field carries there (silent
# state, no normal current at the mid plane), and at kappa = 25 the pi share is the conversion
# share of the two-band notched box at κ = 25 (0.37).
@testset "band partition" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    st = Stepper(md, gr, mk; threads=false, selffield_bc=:slab)
    L = ladder(st, initial_state(md, gr, mk, 0.03), [0.0, 0.3]; H=0.03,
               pr=Protocol(t_min_GL=40.0, t_min_GL_res=60.0, mMax=4000, delta_v=6))
    Ib = band_currents(L.state, st); It, _ = midplane_current(L.state, st)
    @test L.conv == ['F', 'F']
    @test isapprox(sum(Ib), It; rtol=0.02)
    @test isapprox(Ib[2] / sum(Ib), 0.37; atol=0.02)
end

# The run file kept in the folder: copied when missing, accepted when equal, refused when it differs; several writers at
# once (the two polarities the menu starts together) neither fail nor leave a partial file.
@testset "run.toml into the folder" begin
    d = mktempdir(); src = joinpath(mktempdir(), "a.toml"); write(src, "x = 1\n" ^ 20000)
    @test TDGL._keep_run_toml(src, d) == joinpath(d, "run.toml") && read(joinpath(d, "run.toml"), String) == read(src, String)
    @test TDGL._keep_run_toml(src, d) == joinpath(d, "run.toml")
    other = joinpath(dirname(src), "b.toml"); write(other, "x = 2\n")
    @test_throws ErrorException TDGL._keep_run_toml(other, d)
    @test TDGL._keep_run_toml(joinpath(d, "run.toml"), d) == joinpath(d, "run.toml")
    e = mktempdir(); errs = Threads.Atomic{Int}(0)
    @sync for _ in 1:8
        Threads.@spawn try TDGL._keep_run_toml(src, e) catch; Threads.atomic_add!(errs, 1) end
    end
    @test errs[] == 0 && read(joinpath(e, "run.toml"), String) == read(src, String)
end

# A run with a shape file keeps a copy of it next to run.toml, so the folder alone rebuilds the run (masks, figures).
@testset "the run folder keeps its shape" begin
    dir = mktempdir(); mkpath(joinpath(dir, "formas"))
    write(joinpath(dir, "formas", "disco.toml"), "[[shape]]\ntype = \"circle\"\nx = 3.0\ny = 1.5\nr = 1.0\nz = \"all\"\nf = 0.5\n")
    f = joinpath(dir, "r.toml")
    write(f, """
    [material]
    file = "mgb2_model_II.toml"
    [mesh]
    nx = 12
    ny = 6
    nz = 2
    dx = 0.5
    [geometry]
    GEOM_MODE = "CUSTOM"
    geom_file = "formas/disco.toml"
    [scheme]
    threads = false
    [protocol]
    t_min_GL = 2.0
    t_min_GL_res = 2.0
    mMax = 50
    [run]
    J = [0.05]
    H = 0.0
    """)
    o = joinpath(dir, "out")
    run_config(f; out_dir=o, polarity=:pos)
    @test read(joinpath(o, "formas", "disco.toml"), String) == read(joinpath(dir, "formas", "disco.toml"), String)
    a = load_config(f); b = load_config(joinpath(o, "run.toml"))
    @test b.geom.geom_file == joinpath(o, "formas", "disco.toml") && b.masks.f == a.masks.f && count(<(1), a.masks.f) > 0
end
