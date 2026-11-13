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

# The scheme defaults of 26-oct-2026 (RKC, pcg_tol 1e-8) for new runs; a folder that holds a ladder from before keeps RKL and 1e-11
# (resuming, refining or filming it does not switch the integrator mid-ladder); a folder with scheme_defaults.toml keeps its own.
@testset "the scheme defaults a run folder keeps" begin
    d = mktempdir(); sk = Dict{Symbol,Any}()
    TDGL._scheme_defaults!(sk, d); @test sk[:rkl] == "rkc" && sk[:pcg_tol] == 1e-8          # new folder
    mkpath(joinpath(d, "output")); write(joinpath(d, "output", "rungs_J+.csv"), "x\n")
    sk = Dict{Symbol,Any}(); TDGL._scheme_defaults!(sk, d); @test sk[:rkl] === true && sk[:pcg_tol] == 1e-11   # before
    sk = Dict{Symbol,Any}(:rkl => false); TDGL._scheme_defaults!(sk, d); @test sk[:rkl] === false             # the file wins
    e = mktempdir(); s2 = TDGL._scheme_defaults!(Dict{Symbol,Any}(), e); TDGL._keep_scheme_defaults(e, s2)
    mkpath(joinpath(e, "restart")); sk = Dict{Symbol,Any}(); TDGL._scheme_defaults!(sk, e)
    @test sk[:rkl] == "rkc" && sk[:pcg_tol] == 1e-8                                          # it keeps the ones it started with
    @test sk[:pcg_fast] === true && sk[:pcg_f32] === true
    # a folder of the afternoon of 26-oct (its file has rkl and pcg_tol only): pcg_fast and pcg_f32 were not defaults then
    h = mktempdir(); write(joinpath(h, "scheme_defaults.toml"), "pcg_tol = 1.0e-8\nrkl = \"rkc\"\n"); mkpath(joinpath(h, "restart"))
    sk = Dict{Symbol,Any}(); TDGL._scheme_defaults!(sk, h); @test sk[:rkl] == "rkc" && sk[:pcg_fast] === false && sk[:pcg_f32] === false
    @test Stepper(two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=10.0), Grid(24, 12, 4; dx=0.5),
                  box_masks(Grid(24, 12, 4; dx=0.5)); threads=false).rkc[2] !== nothing      # RKC by default with KWT
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

# start in [run]: a field-cooled film holds vortices at J = 0 that the Meissner seed keeps out (the surface barrier), and a ladder
# from a state file is the ladder from that state by hand
@testset "the start of a run" begin
    dir = mktempdir()
    base = """
    [material]
    kappa = 2.0
    gamma12 = 1e-3
    [[material.band]]
    name = "sigma"
    [[material.band]]
    name = "pi"
    xi_ratio = 4.0
    a = 0.59
    [mesh]
    nx = 32
    ny = 24
    nz = 4
    dx = 0.5
    [geometry]
    GEOM_MODE = "SIM"
    [scheme]
    threads = false
    [protocol]
    t_min_GL = 40.0
    t_min_GL_res = 60.0
    keep_rung_states = true
    [run]
    J = [0.0]
    H = 0.3
    """
    nv(path) = (c = load_config(path); s = first(load_state(joinpath(dir, splitext(basename(path))[1], "restart", "rungs",
                                                                      "state_J+_J+0.0000.jld2"), c.grid));
                k = (c.grid.z1 + c.grid.z2 + 1) ÷ 2; count_vortices(s.G[1], s.Vx, s.Vy, s.hz, c.masks.f, c.grid, k)[1])
    fm = joinpath(dir, "meissner.toml"); write(fm, base)
    ff = joinpath(dir, "cooled.toml"); write(ff, base * "start = \"field_cooled\"\nstart_t_hot = 60.0\n")
    run_config(fm; out_dir=joinpath(dir, "meissner"), polarity=:pos)
    run_config(ff; out_dir=joinpath(dir, "cooled"), polarity=:pos)
    @test nv(fm) == 0
    @test nv(ff) > 0
    # from a state file: the same ladder as by hand from that state
    c = load_config(ff); s = start_state(c, ff)
    save_state(joinpath(dir, "fc.jld2"), s)
    fs = joinpath(dir, "fromfile.toml"); write(fs, replace(base, "J = [0.0]" => "J = [0.1]") * "start = \"fc.jld2\"\n")
    r = run_config(fs; out_dir=joinpath(dir, "fromfile"), polarity=:pos)
    st = Stepper(c.model, c.grid, c.masks; c.scheme...)
    L = ladder(st, first(load_state(joinpath(dir, "fc.jld2"), c.grid)), [0.0, 0.1]; H=c.H, label="J+",
               pr=load_config(fs).protocol)
    @test r.pos.V == L.V
    @test read(joinpath(dir, "fromfile", "fc.jld2")) == read(joinpath(dir, "fc.jld2"))   # the folder keeps its start
end

# Every file of the material catalogue loads as [material] and steps finite on a small wedge; a one-band file is one band
# with the dirty-BCS u = 5.79.
@testset "the material catalogue" begin
    dir = mktempdir()
    cat = sort(filter(f -> endswith(f, ".toml"), readdir(joinpath(@__DIR__, "..", "materials"))))
    @test issubset(["mgb2_model_II.toml", "nb.toml", "nbn.toml", "moge.toml", "three_band_trsb.toml", "fe_spm_model.toml"], cat)
    for m in cat
        f = joinpath(dir, "run_" * m)
        write(f, """
        [material]
        file = "$m"
        [mesh]
        nx = 16
        ny = 10
        nz = 4
        dx = 0.5
        [geometry]
        GEOM_MODE = "WEDGE"
        [run]
        J = [0.2]
        H = 0.03
        """)
        c = load_config(f)
        st = Stepper(c.model, c.grid, c.masks; c.scheme...)
        s = initial_state(c.model, c.grid, c.masks, c.H)
        for _ in 1:20; step!(s, st; H=c.H, HJ=0.2); end
        @test all(g -> all(isfinite, g), s.G) && all(isfinite, s.hz)
        if nbands(c.model) == 1
            b = c.model.bands[1]
            @test b.Γ / (b.m * b.β * c.model.σn) ≈ 5.79 rtol=1e-3
        end
    end
end
