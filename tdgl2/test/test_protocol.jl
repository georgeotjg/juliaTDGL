# A two-polarity ladder against juliaTDGL 1.2 (run_case, :prod, Fast), point by point and to the
# bit: voltages, slopes, cycles, closing flags, slip counts, magnetisation, vortex counts, the
# final states and the brackets.  Golden: test/golden/make_ladder.jl.
@testset "protocol against 1.2" begin
    d = JLD2.load(joinpath(@__DIR__, "golden", "ladder_k25.jld2"))
    md = Model((Band(), Band(m=d["mr"], Γ=1.0, a=0.59)), [0.0 1e-3; 1e-3 0.0]; κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    st = Stepper(md, gr, mk; threads=false, selffield_bc=:slab, rkl=false)
    @test st.dt === d["dt"] && st.nsub == (1, d["n_pi"])
    pr = Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, v_margin=1.0, f_strict=false)
    @test eps_cycle(pr, st) === d["eps_c"]
    r = run_ladders(st, [0.30, 0.60, 0.90, 1.20, 1.50]; H=0.03, pr=pr)
    for (L, g) in ((r.pos, d["pos"]), (r.neg, d["neg"]))
        @test L.J == g["J"]
        @test L.conv == collect(g["conv"])
        @test L.m == g["m"]
        @test _same(L.V, g["V"]) && _same(L.Vb[:, 2], g["V2"]) && _same(L.Vstatic, g["Vst"])
        @test _same(L.Vh[:, 1], g["Vh"]) && _same(L.Vh[:, 2], g["Vh2"])
        @test _same(L.nslip, g["nslip"]) && _same(L.M, g["M"])
        @test L.Nv[:, 1] == g["Nv1"] && L.Nv[:, 2] == g["Nv2"]
        @test _same(L.res[:, 1], g["r1"]) && _same(L.res[:, 2], g["r2"])
        @test _same(L.state.G[1], g["G1"]) && _same(L.state.G[2], g["G2"])
    end
    b = r.brackets
    @test (b.pos.lo, b.pos.hi, b.neg.lo, b.neg.hi) == (d["lo_pos"], d["hi_pos"], d["lo_neg"], d["hi_neg"])
end

# With the default close margin, every rung the ladder closes as resistive passes the verdict rule
# (≥ 10 crossings per line on the slope); at v_margin = 1 (release 1.2) the first one can land at 9.8.
@testset "resistive close passes the verdict rule" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    st = Stepper(md, gr, mk; threads=false)
    L = ladder(st, initial_state(md, gr, mk, 0.03), [0.0, 0.9, 1.2, 1.5]; H=0.03,
               pr=Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6))
    c = abs.(rung_rate(L)) .* rung_time(L) ./ 2pi
    @test any(==('V'), L.conv) && all(c[q] >= 10 for q in eachindex(c) if L.conv[q] == 'V')
    @test all(c[q] < 0.5 for q in eachindex(c) if L.conv[q] == 'F')   # and every fixed point is silent
    # stop_after_V: the same ladder ends after its first resistive point
    L1 = ladder(st, initial_state(md, gr, mk, 0.03), [0.0, 0.9, 1.2, 1.5]; H=0.03,
                pr=Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, stop_after_V=1))
    q = findfirst(==('V'), L.conv)
    @test L1.conv[1:q] == L.conv[1:q] && all(==(' '), L1.conv[q+1:end])
end

# A slow flow whose last slip fell in the first half of the point is not a fixed point: one slip at 30 %
# of the point and a quiet second half (the slope alone reads it as silent).
@testset "fixed-point close sees a slip before the second half" begin
    m = 500; dt_cyc = 0.8
    T1 = [k < 150 ? 0.0 : -2pi for k in 1:m]               # one crossing per line at cycle 150
    Tm1 = fill(-2pi, 3, 2); ok = trues(3, 2)
    @test TDGL._crossings_now(T1, Tm1, ok, m, dt_cyc) < 0.5
    @test !TDGL._silent_now(T1, Tm1, ok, m, dt_cyc, 0.5)
    T0 = [0.3 * (1 - exp(-k / 20)) for k in 1:m]           # the static phase change of a current step
    @test TDGL._silent_now(T0, fill(0.3, 3, 2), ok, m, dt_cyc, 0.5)
end

# The onset by the delay: continuing the 'F' rungs below the first 'V' works on copies, so the ladder is
# the same to the bit (also with the open boundary, whose ramp the continuation borrows and gives back);
# the rungs that cross when continued are 'D', and the one below the first flowing rung was followed to
# t_confirm_GL.  (In this box the first 'V' is 0.8 and 0.7 closes 'F' at 32 τ with 0.13 crossings.)
@testset "onset by the delay (t_confirm_GL)" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    Js = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    for bc in (:bar, :open)
        lad(pr) = ladder(Stepper(md, gr, mk; threads=false, selffield_bc=bc), initial_state(md, gr, mk, 0.03), Js;
                         H=0.03, pr=pr)
        L0 = lad(Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, stop_after_V=1))
        L1 = lad(Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, stop_after_V=1, t_confirm_GL=150.0))
        @test L1.V == L0.V && L1.m == L0.m && L1.state.G[1] == L0.state.G[1] && L1.state.θx == L0.state.θx
        @test all(L1.conv[q] == L0.conv[q] || (L0.conv[q] == 'F' && L1.conv[q] == 'D') for q in eachindex(Js))
        k = findfirst(==('V'), L0.conv)
        @test k !== nothing && k > 2 && L1.t_conf[k-1] > 0
        d = delay_bracket(L1)
        @test d.ok && d.t_silent == 150.0 && d.hi <= abs(Js[k]) && L1.conv[findfirst(==(d.lo), abs.(Js))] == 'F'
        @test all(isnan(L1.delay[q]) == (L1.conv[q] != 'D') for q in eachindex(Js) if L1.t_conf[q] > 0)
        @test all(iszero, L0.t_conf) && !delay_bracket(L0).ok
    end
end

# Resuming: a ladder that ended (or was stopped) after some points and is resumed from its checkpoint is the ladder
# run without a stop, to the bit, also with the delay confirmation (the checkpoint carries its closed rungs) and with
# the open boundary (its ramp).  Here the first 'V' is 0.8: cut at 2 and at 4 (three 'F' rungs waiting to be confirmed).
@testset "resume a ladder from its checkpoint" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    Js = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    pr = Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, stop_after_V=1, t_confirm_GL=150.0)
    for bc in (:bar, :open)
        st() = Stepper(md, gr, mk; threads=false, selffield_bc=bc)
        d0 = mktempdir()
        L0 = ladder(st(), initial_state(md, gr, mk, 0.03), Js; H=0.03, pr=pr, out_dir=d0)
        @test ladder_done(joinpath(d0, "restart", "state_J+.jld2")) && count(==('D'), L0.conv) >= 1
        for cutat in (2, 4)
            d1 = mktempdir()
            ladder(st(), initial_state(md, gr, mk, 0.03), Js[1:cutat]; H=0.03, pr=pr, out_dir=d1)
            L1 = resume_ladder(st(), Js; H=0.03, pr=pr, out_dir=d1)
            @test L1.V == L0.V && L1.m == L0.m && L1.conv == L0.conv && isequal(L1.delay, L0.delay) && L1.t_conf == L0.t_conf
            @test L1.state.G[1] == L0.state.G[1] && L1.state.G[2] == L0.state.G[2] && L1.state.θx == L0.state.θx
            @test isequal(L1.Vh, L0.Vh) && L1.psi2 == L0.psi2 && L1.Nv == L0.Nv
            @test read(joinpath(d1, "output", "rungs_J+.csv"), String) == read(joinpath(d0, "output", "rungs_J+.csv"), String)
            L2 = resume_ladder(st(), Js; H=0.03, pr=pr, out_dir=d1, write=false)     # an ended ladder: read back, no step
            @test L2.V == L0.V && L2.conv == L0.conv && L2.state.G[1] == L0.state.G[1]
        end
        @test_throws ErrorException resume_ladder(st(), [0.0, 0.4]; H=0.03, pr=pr, out_dir=d0)   # other currents
    end
end

# The same with the optional terms on (heat with Joule and a contact margin, and drag): the delay confirmation works on
# copies (each with the temperature its rung closed with) and leaves the ladder and its temperature field to the bit, and a
# resumed ladder is the one without a stop.
@testset "delay confirmation and resume with heat and drag" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    Js = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    st() = Stepper(md, gr, mk; threads=false, drag=0.05, heat=Heat(D_T=0.25, eta=0.05, joule=true, margin=2))
    p0 = Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, stop_after_V=1)
    p1 = Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, stop_after_V=1, t_confirm_GL=150.0)
    s0 = st(); L0 = ladder(s0, initial_state(md, gr, mk, 0.03), Js; H=0.03, pr=p0)
    s1 = st(); d1 = mktempdir(); L1 = ladder(s1, initial_state(md, gr, mk, 0.03), Js; H=0.03, pr=p1, out_dir=d1)
    @test L1.V == L0.V && L1.m == L0.m && L1.state.G[1] == L0.state.G[1] && s1.heat.θ == s0.heat.θ
    @test all(L1.conv[q] == L0.conv[q] || (L0.conv[q] == 'F' && L1.conv[q] == 'D') for q in eachindex(Js))
    @test any(>(0), L1.t_conf) && maximum(s1.heat.θ) > 0
    s2 = st(); d2 = mktempdir()
    ladder(s2, initial_state(md, gr, mk, 0.03), Js[1:3]; H=0.03, pr=p1, out_dir=d2)
    s3 = st(); L3 = resume_ladder(s3, Js; H=0.03, pr=p1, out_dir=d2)
    @test L3.V == L1.V && L3.conv == L1.conv && isequal(L3.delay, L1.delay) && L3.state.G[1] == L1.state.G[1] &&
          s3.heat.θ == s1.heat.θ
end

# 'N': a sample gone normal is a fixed point with no phase to cross, and closed 'F' (with u large the
# onset can be a collapse).  One band with Γ = 48.6, far above depairing: ⟨|Ψ|²⟩ = 0 and 'N', counted as dissipating.
@testset "a sample gone normal closes 'N', not 'F'" begin
    gr = Grid(24, 12, 4; dx=0.5); mk = box_masks(gr)
    md = Model((Band(Γ=48.6),), zeros(1, 1); κ=25.0); st = Stepper(md, gr, mk; threads=false)
    pr(nf) = Protocol(t_min_GL=40.0, t_min_GL_res=60.0, mMax=4000, delta_v=4, n_frac=nf)
    L = ladder(st, initial_state(md, gr, mk, 0.0), [0.0, 0.3, 0.9, 1.5]; H=0.0, pr=pr(0.02))
    L0 = ladder(st, initial_state(md, gr, mk, 0.0), [0.0, 0.3, 0.9, 1.5]; H=0.0, pr=pr(0.0))
    @test L.conv[end] == 'N' && L0.conv[end] == 'F' && L.psi2[end, 1] < 0.01
    @test L.conv[1:2] == ['F', 'F'] && L.V == L0.V && L.m == L0.m      # only the flag changes
    b = bracket_Ic(L.J, rung_rate(L), rung_time(L); conv=L.conv)
    @test b.cls[end] == 'D' && bracket_Ic(L0.J, rung_rate(L0), rung_time(L0)).cls[end] == 'S'
end

# crossing_times: the voltmeter of ladder at fixed current; no crossing far below the onset, a regular train far above, and
# the same verdict as the ladder at those currents (the small box at κ = 25: first 'V' at 0.8, see above).
@testset "crossing_times" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    st = Stepper(md, gr, mk; threads=false)
    s = initial_state(md, gr, mk, 0.03)
    lo = crossing_times(st, s, 0.5; H=0.03, T=150.0, delta_v=6)
    @test isempty(lo.times) && !lo.diverged && abs(lo.t - 150.0) < 0.5
    hi = crossing_times(st, s, 1.4; H=0.03, T=600.0, delta_v=6, stop=4)
    @test length(hi.times) == 4 && issorted(hi.times) && hi.t < 600.0
    @test_throws ErrorException crossing_times(st, s, 0.5; H=0.03, T=10.0, delta_v=20)
end


# A run killed after a point was written (rungs_*.csv, j_history.csv) and before its checkpoint (the delay confirmation
# runs in between) is resumed from the previous checkpoint: the point is redone, and neither table holds it twice.
@testset "resume after a kill between a point's rows and its checkpoint" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    Js = [0.0, 0.5, 0.6, 0.7]
    pr = Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, keep_rung_states=true)
    st() = Stepper(md, gr, mk; threads=false)
    rows(f, l) = count(x -> (w = split(x, ','); length(w) > 1 && w[2] == l), readlines(f))
    d = mktempdir()
    ladder(st(), initial_state(md, gr, mk, 0.03), Js; H=0.03, pr=pr, out_dir=d)
    hist = joinpath(d, "j_history.csv"); rung = joinpath(d, "output", "rungs_J+.csv")
    @test rows(hist, "J+") == 4
    cp(joinpath(d, "restart", "rungs", "state_J+_J+0.6000.jld2"), joinpath(d, "restart", "state_J+.jld2"); force=true)
    write(joinpath(d, "restart", "state_J+.jld2.status"), "running\n")
    resume_ladder(st(), Js; H=0.03, pr=pr, out_dir=d)
    @test count(l -> occursin(r"^\d", l), readlines(rung)) == 4
    @test rows(hist, "J+") == 4
end

# Noise (noise.jl): the same seed gives the same ladder, another seed another; a ladder with noise and the delay
# confirmation, cut and resumed, is the ladder without a stop to the bit (the stream is in the checkpoint and the
# confirmation gives it back).
@testset "noise: seeded, resumable" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    Js = [0.0, 0.6, 0.8, 0.9]
    pr = Protocol(t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6, stop_after_V=1, t_confirm_GL=60.0)
    st(seed=1) = Stepper(md, gr, mk; threads=false, noise=1e-4, noise_seed=seed)
    d0 = mktempdir()
    L0 = ladder(st(), initial_state(md, gr, mk, 0.03), Js; H=0.03, pr=pr, out_dir=d0)
    L0c = ladder(st(2), initial_state(md, gr, mk, 0.03), Js[1:2]; H=0.03, pr=pr)
    d1 = mktempdir()
    ladder(st(), initial_state(md, gr, mk, 0.03), Js[1:2]; H=0.03, pr=pr, out_dir=d1)
    L1 = resume_ladder(st(), Js; H=0.03, pr=pr, out_dir=d1)
    @test L1.V == L0.V && L1.conv == L0.conv && L1.state.G[1] == L0.state.G[1] && L1.state.G[2] == L0.state.G[2]
    L1b = ladder(st(), initial_state(md, gr, mk, 0.03), Js[1:2]; H=0.03, pr=pr)
    @test L1b.state.G[1] != L0c.state.G[1]                 # another seed, another stream
    @test Stepper(two_band(xi_ratio=4.0, κ=25.0, gamma_kwt=10.0), gr, mk; threads=false, noise=1e-4).noise !== nothing   # with KWT too
end
