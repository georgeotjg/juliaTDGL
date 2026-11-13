using Random, Test, TDGL, TOML

@testset "any-band onset" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(12, 8, 4; dx=0.5)
    mk = box_masks(gr)
    currents = [0.1, 0.15, 0.2]
    pr(mode) = Protocol(slip_band_mode=mode, iter_min=7, mMax=40, lMax=1, delta_v=2,
                        t_min_GL=0.0, t_min_GL_res=0.0, V_win=4, V_win_min_res=6)
    stepper() = Stepper(md, gr, mk; threads=false)

    # Manufacture one nonstationary rung with zero crossing rate and another with a
    # continuous phase winding only in band 2. This isolates the protocol from nucleation.
    function phase_cycles!(s, st, n; H, HJ)
        if H == 0.15
            s.hz[1] = 0.01 - s.hz[1]
        elseif H == 0.2
            vi, vo = gr.x1+2, gr.x2-2
            for i in axes(s.G[2], 1)
                s.G[2][i, :, :] .*= cis(0.75pi * (i - vi) / (vo - vi))
            end
            copyto!(s.F[2], s.G[2])
        end
    end
    function run(js, mode, dir="")
        ladder(stepper(), initial_state(md, gr, mk, 0.0), js; H=js, pr=pr(mode), out_dir=dir)
    end

    prior = TDGL.STEPS[]
    TDGL.STEPS[] = phase_cycles!
    try
        old = run(currents, :band1)
        @test old.conv == ['F', '*', '*']
        @test all(iszero, old.V) && abs(old.Vh[end, 2]) > 0
        @test brackets(old, old).pos.cls == ['S', 'S', 'S']
        mktempdir() do dir
            full = joinpath(dir, "full")
            selected = run(currents, :any_band, full)
            @test selected.conv == ['F', '*', 'V']
            @test selected.slip_band == [0, 0, 2]
            @test all(iszero, selected.V) # the band-1 voltage remains available
            @test selected.m[2] == pr(:any_band).mMax
            bracket = brackets(selected, selected).pos
            @test bracket.cls == ['S', 'U', 'D']
            @test (bracket.lo, bracket.hi, bracket.ok) == (0.1, 0.2, false)
            @test bracket.undecided == [0.15]
            rungs = readlines(joinpath(full, "output", "rungs_J+.csv"))
            @test endswith(rungs[2], ",onset_class,slip_band")
            @test endswith(rungs[end-1], ",U,0") && endswith(rungs[end], ",D,2")
            @test endswith(readlines(joinpath(full, "j_history.csv"))[end], ",D")

            interrupted = joinpath(dir, "interrupted")
            run(currents[1:1], :any_band, interrupted)
            @test_throws ArgumentError resume_ladder(stepper(), currents; H=currents, pr=pr(:band1), out_dir=interrupted)
            continued = resume_ladder(stepper(), currents; H=currents, pr=pr(:any_band), out_dir=interrupted)
            @test isequal(continued.Vh, selected.Vh) && continued.slip_band == selected.slip_band
            @test continued.conv == selected.conv && continued.state.G == selected.state.G
            @test readlines(joinpath(interrupted, "output", "rungs_J+.csv")) == rungs

            cfg = TOML.parsefile(joinpath(pkgdir(TDGL), "examples", "wedge_k25.toml"))
            cfg["protocol"]["slip_band_mode"] = "any_band"
            cfg["protocol"]["confirm_replay"] = true
            path = joinpath(dir, "protocol.toml")
            open(io -> TOML.print(io, cfg), path, "w")
            loaded = load_config(path)
            @test loaded.protocol.slip_band_mode === :any_band && loaded.protocol.confirm_replay
            b = brackets(selected, continued)
            summary = summary_dict(stepper(), loaded, (; pos=selected, neg=continued, brackets=b))
            @test summary["ladders"]["J+"]["slip_band"] == [0, 0, 2]
        end
        @test_throws ArgumentError run(currents, :unknown)
    finally
        TDGL.STEPS[] = prior
    end
end

@testset "replay of a buffered rung with PCG and noise" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(12, 8, 4; dx=0.5, buf=2)
    mk = box_masks(gr)
    stepper() = Stepper(md, gr, mk; threads=false, σvac=0.01, pcg_fast=true,
                        noise=1e-7, noise_seed=43)
    pr(replay) = Protocol(confirm_replay=replay, t_confirm_GL=15.0, delta_v=2,
                          lMax=2, iter_min=7, mMax=8, t_min_GL=0.0, t_min_GL_res=0.0)
    Js = [0.0, 0.04, 0.07]
    H = 0.03
    mktempdir() do dir
        observed = stepper()
        full = ladder(observed, initial_state(md, gr, mk, H), Js; H=H, pr=pr(true), out_dir=dir)
        @test full.conv == ['F', 'F', 'F']
        file = joinpath(dir, "restart", "state_J+.jld2")
        _, w = load_state(file, gr)
        @test w["ring_n"] == 2
        @test w["ring1_jp"] == 2 && w["ring1_pcg"][1] && w["ring1_pcg"][3][1]
        @test w["ring1_noise_rng"] != w["noise_rng"]
        @test w["ring1_pcg"] != TDGL._pcg_snap(observed)

        out = (; (k => (k === :conv ? collect(w["out_$k"]) : copy(w["out_$k"])) for k in TDGL._OUT_KEYS)...,
               slip_band=zeros(Int, length(Js)))
        ring = []
        Cooper0 = [copy(w["cooper$µ"]) for µ in 1:2]
        TDGL._restore_ladder!(out, Cooper0, ring, w, Js, fill(H, length(Js)), observed, pr(true))
        old = first(ring)
        restored = TDGL._confirmation_state!(observed, old)
        ref = stepper()
        direct = ladder(ref, initial_state(md, gr, mk, H), Js[1:2]; H=H, pr=pr(true)).state
        @test randn(copy(observed.noise.rng), 16) == randn(copy(ref.noise.rng), 16)
        @test TDGL._pcg_snap(observed) == TDGL._pcg_snap(ref)
        for _ in 1:7
            step!(restored, observed; H=H, HJ=selffield(observed, old.J))
            step!(direct, ref; H=H, HJ=selffield(ref, old.J))
        end
        @test restored.G == direct.G
        @test all(getfield(restored, f) == getfield(direct, f)
                  for f in (:θx, :θy, :θz, :φx, :φy, :φz, :hx, :hy, :hz))
        @test TDGL._pcg_snap(observed) == TDGL._pcg_snap(ref)
        @test randn(copy(observed.noise.rng), 16) == randn(copy(ref.noise.rng), 16)

        legacy = joinpath(dir, "legacy")
        initial = ladder(stepper(), initial_state(md, gr, mk, H), Js[1:2]; H=H, pr=pr(false), out_dir=legacy)
        @test initial.conv == ['F', 'F']
        @test_throws ArgumentError resume_ladder(stepper(), Js; H=H, pr=pr(true), out_dir=legacy)
        @test resume_ladder(stepper(), Js; H=H, pr=pr(false), out_dir=legacy).state.G == full.state.G
        @info "Replay payload per buffered rung" bytes=Base.summarysize((old.pcg, old.noise_rng))
    end
end
