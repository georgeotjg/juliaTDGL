# Shared CPU/device regression: a zero right-hand side must erase a warm
# increment, preserve the actual previous solution and survive a checkpoint.
using TDGL, Test, LinearAlgebra

function pcg_zero_rhs_checks(make_solve; checkpoint_dir=mktempdir())
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(4, 4, 4; dx=0.5, buf=2)
    mk = build_masks(GeomSpec(GEOM_MODE="SIM"), gr)
    modes = ((false, false, true), (true, false, true),
             (true, true, true), (true, true, false))
    for (fast, f32, extrap) in modes
        @testset "fast=$fast f32=$f32 extrap=$extrap" begin
            newstepper() = Stepper(md, gr, mk; threads=false, σvac=0.01,
                                   pcg_fast=fast, pcg_f32=f32, pcg_extrap=extrap)
            st = newstepper(); S = st.imp; sp = S.sep
            s = initial_state(md, gr, mk, 0.0)
            solve! = make_solve(st)
            ranges = (sp.rx, sp.ry, sp.rz)
            theta(s) = (s.θx, s.θy, s.θz)
            phi(s) = (s.φx, s.φy, s.φz)
            # A nonzero background distinguishes Δ=0 from incorrectly setting φ=0.
            function source!(s, q)
                for c in 1:3
                    t, p = theta(s)[c], phi(s)[c]
                    for i in eachindex(t); t[i] = 0.01sin(0.13i + c); end
                    p .= t .+ 0.003              # nonzero source on decoupled links
                    b = view(p, ranges[c]...); th = view(t, ranges[c]...)
                    for (n, i) in enumerate(eachindex(b))
                        b[i] = th[i] + (q == 0 ? 0.0 : 0.001sin(0.17n + c + q))
                    end
                end
            end
            function check_zero!(cold=false)
                previous = deepcopy(S.x)
                expected = map((p, t, d) -> t .+ (p .- t) ./ d, phi(s), theta(s), S.D)
                solve!(s)
                @test all(all(iszero, x) for x in S.x)
                @test S.warm[1] && S.iters[1] == 0
                for c in 1:3
                    @test view(phi(s)[c], ranges[c]...) == view(theta(s)[c], ranges[c]...)
                    view(expected[c], ranges[c]...) .= view(theta(s)[c], ranges[c]...)
                    @test phi(s)[c] == expected[c]   # includes decoupled links
                end
                if fast
                    @test S.fast.have2[1] == (!cold && extrap)
                    if !cold && extrap
                        @test S.fast.xprev == previous
                    end
                end
            end
            source!(s, 0); check_zero!(true)
            for q in 1:3
                source!(s, q); solve!(s)
                @test any(any(!iszero, x) for x in S.x)
            end
            source!(s, 0); check_zero!()

            # Use the same fields and history keys as the ladder checkpoint.
            extra = Pair{Symbol,Any}[Symbol("pcg_x$c") => S.x[c] for c in 1:3]
            if fast && S.fast.have2[1]
                append!(extra, [Symbol("pcg_p$c") => S.fast.xprev[c] for c in 1:3])
            end
            path = joinpath(checkpoint_dir, "zero_$(fast)_$(f32)_$(extrap).jld2")
            save_state(path, s; extra...)
            resumed, raw = load_state(path, gr)
            restored = newstepper()
            prev = haskey(raw, "pcg_p1") ? (true, ntuple(c -> raw["pcg_p$c"], 3)) : nothing
            TDGL._pcg_back!(restored, (true, ntuple(c -> raw["pcg_x$c"], 3), prev))
            continue! = make_solve(restored)
            for q in (0, 4, 5)
                source!(s, q); source!(resumed, q)
                solve!(s); continue!(resumed)
                @test phi(s) == phi(resumed)
                @test TDGL._pcg_snap(st) == TDGL._pcg_snap(restored)
            end
            source!(s, 0); check_zero!()
            source!(s, 0); check_zero!()  # previous solution is now also zero
            TDGL._pcg_reset!(st)
            source!(s, 0); check_zero!(true)
        end
    end
end
