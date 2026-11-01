# Physical checks of the N-band step that do not depend on release 1.2.
import Random

_media(G, gr) = sum(abs2, @view G[gr.x1+1:gr.x2, gr.y1+1:gr.y2, gr.z1+1:gr.z2]) /
                ((gr.x2 - gr.x1) * (gr.y2 - gr.y1) * (gr.z2 - gr.z1))

@testset "one band: amplitude relaxation with Γ ≠ 1 (R0)" begin
    # A uniform amplitude perturbation of a clean box at H = 0 drives no current; per explicit
    # Euler step it decays by 1 − 2ψ₀² dt/Γ (ψ₀ = 1).
    for Γ in (1.0, 3.0)
        md = Model((Band(Γ=Γ),), zeros(1, 1); κ=2.0)
        gr = Grid(8, 6, 4; dx=0.5)
        st = Stepper(md, gr, box_masks(gr); threads=false)
        s = initial_state(md, gr, st.masks, 0.0)
        δ0 = 1e-6
        s.G[1] .*= (1 + δ0)
        i, j, k = 6, 5, 4
        step!(s, st; H=0.0, HJ=0.0)
        d1 = abs(s.G[1][i,j,k]) - 1
        step!(s, st; H=0.0, HJ=0.0)
        d2 = abs(s.G[1][i,j,k]) - 1
        @test isapprox(d2 / d1, 1 - 2 * st.dt / Γ; atol=1e-5)
    end
end

@testset "physical temperature: equilibrium of a clean box" begin
    md = two_band(xi_ratio=4.0, κ=2.0, T=0.3)
    gr = Grid(12, 8, 4; dx=0.5)
    st = Stepper(md, gr, box_masks(gr); threads=false)
    s = initial_state(md, gr, st.masks, 0.0)
    for _ in 1:4000; step!(s, st; H=0.0, HJ=0.0); end
    @test isapprox(_media(s.G[1], gr), 0.7; rtol=0.01)
    @test isapprox(_media(s.G[2], gr), 0.59 - 0.3; rtol=0.02)
    @test two_band(xi_ratio=4.0, κ=2.0, T=0.3).bands[2].m == two_band(xi_ratio=4.0, κ=2.0).bands[2].m
end

@testset "three bands: two identical bands stay identical" begin
    b = two_band(xi_ratio=4.0, κ=25.0).bands
    md = Model((b[1], b[2], b[2]), [0.0 1e-3 1e-3; 1e-3 0.0 0.0; 1e-3 0.0 0.0]; κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5)
    st = Stepper(md, gr, build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr); threads=false)
    @test st.rkl == (0, st.rkl[2], st.rkl[2]) && st.rkl[2] > 1       # RKL stages for the two stiff bands
    s = initial_state(md, gr, st.masks, 0.03)
    HJ = selffield(st, 0.4)
    for _ in 1:400; step!(s, st; H=0.03, HJ=HJ); end
    @test s.G[2] == s.G[3]
    @test maximum(abs, s.G[2]) > 0.1
end

@testset "KWT: the local step and the amplitude rates" begin
    # kwt_update solves w + γ²ψ Re(ψ* w) = √(1+γ²|ψ|²) R/Γ (2×2 real system), also near ψ = 0
    for (ψ, R, γ) in ((0.7 + 0.2im, 0.3 - 0.1im, 10.0), (1e-9 + 0im, 0.2 + 0.4im, 10.0), (0.4 - 0.9im, -0.5 + 0.05im, 2.0))
        dt = 0.01
        w = (TDGL.kwt_update(ψ, dt * R, γ^2) - ψ) / dt
        q = 1 + γ^2 * abs2(ψ)
        M = [1 + γ^2 * real(ψ)^2  γ^2 * real(ψ) * imag(ψ); γ^2 * real(ψ) * imag(ψ)  1 + γ^2 * imag(ψ)^2]
        x = M \ (sqrt(q) .* [real(R), imag(R)])
        @test isapprox(w, complex(x[1], x[2]); rtol=1e-12)
    end
    # a uniform amplitude perturbation decays at 2ψ₀²/(u√(1+γ²ψ₀²)), u = Γ/σn:
    # 2.000, 0.3454, 0.03437 for (u, γ) = (1, 0), (5.79, 0), (5.79, 10)
    for (u, γ, rate) in ((1.0, 0.0, 2.0), (5.79, 0.0, 0.3454), (5.79, 10.0, 0.03437))
        md = Model((Band(Γ=u, γkwt=γ),), zeros(1, 1); κ=2.0)
        gr = Grid(8, 6, 4; dx=0.5)
        st = Stepper(md, gr, box_masks(gr); threads=false, rkl=false)   # the Euler factor per step
        s = initial_state(md, gr, st.masks, 0.0)
        s.G[1] .*= (1 + 1e-6)
        i, j, k = 6, 5, 4
        step!(s, st; H=0.0, HJ=0.0); d1 = abs(s.G[1][i,j,k]) - 1
        step!(s, st; H=0.0, HJ=0.0); d2 = abs(s.G[1][i,j,k]) - 1
        @test isapprox((1 - d2 / d1) / st.dt, rate; rtol=2e-3)   # explicit Euler: factor 1 − λ dt per step
    end
end

@testset "insulating padding: the current goes through the sample" begin
    # silent state at kappa = 2: with a normal-metal padding a third of the current bypasses the
    # sample; with an insulating one and the self-field of the bar on the outer faces, all of it
    # goes through (the rest is the shunt of the weakly conducting vacuum, ∝ σvac)
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(24, 12, 4; dx=0.5, buf=4)
    mk = build_masks(GeomSpec(GEOM_MODE="SIM"), gr)
    frac = Dict{Symbol,Float64}()
    for (k, σv, bc) in ((:metal, nothing, :slab), (:vacuum, 0.01, :bar))
        st = Stepper(md, gr, mk; threads=false, σvac=σv, selffield_bc=bc)
        s = initial_state(md, gr, mk, 0.03); HJ = selffield(st, 0.3)
        for _ in 1:round(Int, 30 / st.dt); step!(s, st; H=0.03, HJ=HJ); end
        It, Is = midplane_current(s, st)
        frac[k] = Is / It
        # the slab condition of 1.x puts the nominal density across the whole domain width;
        # the bar condition carries exactly the nominal current
        bc === :bar && @test isapprox(It / (-0.3 * 12 * 0.5 * 4 * 0.5), 1; atol=2e-3)
    end
    @test frac[:metal] < 0.95
    @test frac[:vacuum] > 0.995
end

# The insulating padding's solve starts from the previous step's solution: fewer iterations, the same state to the
# tolerance of the solve; a ladder stopped and resumed keeps its bits (the checkpoint carries the solution).
@testset "insulating padding: warm start and resume" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(24, 12, 4; dx=0.5, buf=2)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    st = Stepper(md, gr, mk; threads=false, σvac=0.01)
    function run(warm)
        s = initial_state(md, gr, mk, 0.03); HJ = selffield(st, 0.3); TDGL._pcg_reset!(st); its = 0
        for q in 1:round(Int, 5 / st.dt)
            warm || TDGL._pcg_reset!(st)
            step!(s, st; H=0.03, HJ=HJ); q > 10 && (its += st.imp.iters[1])
        end
        s, its
    end
    (a, ia), (b, ib) = run(true), run(false)
    @test ia < 0.7ib && maximum(abs.(a.G[1] .- b.G[1])) < 1e-8 && maximum(abs.(a.θx .- b.θx)) < 1e-8
    pr = Protocol(t_min_GL=10.0, t_min_GL_res=20.0, mMax=400, delta_v=4)
    Js = [0.0, 0.2, 0.4]
    d0 = mktempdir(); d1 = mktempdir()
    L0 = ladder(st, initial_state(md, gr, mk, 0.03), Js; H=0.03, pr=pr, out_dir=d0)
    ladder(st, initial_state(md, gr, mk, 0.03), Js[1:2]; H=0.03, pr=pr, out_dir=d1)
    L1 = resume_ladder(st, Js; H=0.03, pr=pr, out_dir=d1)
    @test L1.V == L0.V && L1.m == L0.m && L1.state.G[1] == L0.state.G[1] && L1.state.θx == L0.state.θx
end

# E4: the conductivity of a dirty BCS material.  One band gives the Kramer–Watts-Tobin u = 5.79; the
# two-band model of the diode studies (Γ = β = 1, g_π = R² a₂ₙ = 9.44) gives σn = 10.44/5.79 = 1.80; a band's share is g Γ/β,
# which does not change when its order parameter is rescaled (g, Γ → s²g, s²Γ; β → s⁴β).
@testset "dirty-limit σn (E4)" begin
    md1 = Model((Band(),), zeros(1, 1); κ=2.0, σn=:dirty)
    @test md1.bands[1].Γ / md1.σn ≈ 5.788 atol=1e-3
    md2 = two_band(xi_ratio=4.0, κ=25.0, σn=:dirty)
    @test md2.σn ≈ (1 + 16 * 0.59) / TDGL.U_DIRTY && isapprox(md2.σn, 1.80; atol=0.01)
    r = regime(md2; L=20.0)
    @test r.sigma_over_dirty ≈ 1 && r.u[1] ≈ 1 / md2.σn
    @test regime(two_band(xi_ratio=4.0, κ=25.0); L=20.0).sigma_over_dirty ≈ 1 / md2.σn
    s = 1.7
    @test sigma_dirty((Band(), Band(g=s^2 * 9.44, Γ=s^2, β=s^4))) ≈ md2.σn
end

# A passive band (ε_π < 0, condensate induced by the coupling), as in MgB₂ near Tc: the damped fixed point of
# _psi2_bound oscillates there and regime() divided 0 by 0 with the uncoupled amplitudes.
@testset "passive band: coupled equilibrium" begin
    for f in ("mgb2_dirty.toml", "mgb2_clean.toml")
        md, _ = material(Dict("file" => f))
        p = equilibrium(md)
        for µ in 1:2
            ν = 3 - µ; b = md.bands[µ]
            @test abs(b.β * p[µ]^3 - eps_band(md, µ) * p[µ] - md.γ[µ][ν] * p[ν]) < 1e-12
        end
        @test eps_band(md, 2) < 0 && 0 < p[2] < 0.3        # induced, Δπ/Δσ ≈ 0.19 at 0.7 Tc
        @test all(TDGL._psi2_bound(md) .≈ p .^ 2)
        r = regime(md; L=20.0)
        @test isfinite(r.ellJ_over_L) && r.ellJ_over_L < 0.1  # strongly coupled: ℓ_J ≪ L
    end
    md = two_band(xi_ratio=4.0, κ=25.0)                      # active bands: the fixed point, unchanged
    @test TDGL._psi2_bound(md)[2] ≈ equilibrium(md)[2]^2 rtol=1e-10
end

# The energy balance of the scheme (B.4).  Without applied field or injection, over one step
# dF/dt = −D + W: D the dissipation in the bands and the normal channel, W the power of the
# supercurrent on the links that join two ghost nodes (τ = 1 there, as in 1.x), which no term of F
# accounts for.  With W the residual is O(dt) (halves with the step); without it, it is W; with the
# ghost links off, W = 0 and F decreases at every step.
@testset "energy balance" begin
    function balance(sc, off)
        md = two_band(xi_ratio=4.0, κ=2.0); gr = Grid(16, 10, 8; dx=0.5); mk = box_masks(gr)
        off && ghost_links_off!(mk, gr)
        st = Stepper(md, gr, mk; threads=false, implicit=false, substeps=(1, 1), dt_scale=sc)
        s = initial_state(md, gr, mk, 0.0); rng = Random.Xoshiro(7)
        for µ in 1:2, c in eachindex(s.G[µ])
            mk.τψ[c] > 0 && (s.G[µ][c] *= (0.6 + 0.4rand(rng)) * cis(1.5randn(rng)))
        end
        for _ in 1:round(Int, 0.5 / st.dt); step!(s, st; H=0.0, HJ=0.0); end
        r = 0.0; rW = 0.0; Wm = 0.0; mono = true
        for _ in 1:round(Int, 0.25 / st.dt)
            s0 = deepcopy(s); F0 = free_energy(s0, st)
            step!(s, st; H=0.0, HJ=0.0); F1 = free_energy(s, st)
            D, W = dissipation(s0, s, st)
            r = max(r, abs(((F1 - F0) / st.dt + D - W) / D)); rW = max(rW, abs(((F1 - F0) / st.dt + D) / D))
            Wm = max(Wm, abs(W) / D); mono &= F1 < F0
        end
        (r, rW, Wm, mono)
    end
    a = balance(0.5, false); b = balance(0.25, false); c = balance(0.5, true)
    @test a[1] < 1e-3 && 0.4 < b[1] / a[1] < 0.6       # O(dt) once W is counted
    @test a[3] > 3e-3 && a[2] > 5a[1]                  # the ghost links are not negligible on this box
    @test c[3] == 0 && c[4] && c[1] < 1e-3             # gradient flow with them off
end

# The balance with KWT: the dissipation of the functional is (Γ/s)[|Ψ̇|² + γ̃²(Re Ψ*Ψ̇)²], s = √(1+γ̃²|Ψ|²) (the band equation
# times Ψ̇*); with it dF/dt = −D to O(dt), where Γ|Ψ̇|² (the heat source and dissipation() before 30-oct) is off by a factor that
# does not shrink with dt (+85 % at γ̃ = 3, ×5.6 at γ̃ = 10).
@testset "energy balance with KWT" begin
    function bal(γ, sc)
        b = two_band(xi_ratio=4.0, κ=2.0).bands
        md = Model((Band(m=b[1].m, γkwt=γ), Band(m=b[2].m, a=0.59, γkwt=γ)), [0.0 1e-3; 1e-3 0.0]; κ=2.0)
        gr = Grid(16, 10, 8; dx=0.5); mk = box_masks(gr); ghost_links_off!(mk, gr)
        st = Stepper(md, gr, mk; threads=false, implicit=false, rkl=false, dt_scale=sc)
        s = initial_state(md, gr, mk, 0.0); rng = Random.Xoshiro(7)
        for µ in 1:2, c in eachindex(s.G[µ])
            mk.τψ[c] > 0 && (s.G[µ][c] *= (0.6 + 0.4rand(rng)) * cis(1.5randn(rng)))
        end
        for _ in 1:round(Int, 0.5 / st.dt); step!(s, st; H=0.0, HJ=0.0); end
        r = 0.0
        for _ in 1:round(Int, 0.25 / st.dt)
            s0 = deepcopy(s); F0 = free_energy(s0, st)
            step!(s, st; H=0.0, HJ=0.0); F1 = free_energy(s, st)
            D, _ = dissipation(s0, s, st)
            r = max(r, abs(((F1 - F0) / st.dt + D) / D))
        end
        r
    end
    a = bal(10.0, 0.5); b = bal(10.0, 0.25)
    @test a < 5e-3 && 0.4 < b / a < 0.6
end

# RKL1 stages for the band update.  The stage count covers each band's bound; with KWT the step is
# the one without it, and the state after a short relaxation is the one of the explicit KWT step
# (ten times smaller) to O(dt).  Off, nothing changes (the regression tests run with rkl = false).
@testset "RKL stages (K1)" begin
    b = two_band(xi_ratio=4.0, κ=25.0).bands
    md = Model((Band(m=b[1].m, γkwt=10.0), Band(m=b[2].m, a=0.59, γkwt=10.0)), [0.0 1e-3; 1e-3 0.0]; κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    se = Stepper(md, gr, mk; threads=false, rkl=false)
    sr = Stepper(md, gr, mk; threads=false)
    s0 = Stepper(two_band(xi_ratio=4.0, κ=25.0), gr, mk; threads=false, rkl=false)
    @test sr.dt == s0.dt && se.dt < sr.dt / 9
    @test sr.rkl[1] >= 3 && sr.rkl[2] >= sr.rkl[1] && sr.nsub == (1, 1)    # π is the stiffer band
    function relax(st, T)
        s = initial_state(md, gr, mk, 0.03); HJ = selffield(st, 0.3)
        for _ in 1:round(Int, T / st.dt); step!(s, st; H=0.03, HJ=HJ); end
        s
    end
    a = relax(se, 20.0); c = relax(sr, 20.0)
    Ia = band_currents(a, se); Ic = band_currents(c, sr)
    @test isapprox(Ic[2] / sum(Ic), Ia[2] / sum(Ia); atol=2e-3)
    @test maximum(abs.(abs2.(c.G[1]) .- abs2.(a.G[1]))) < 2e-2
end

# The band bound with interband coupling: |Ψ_µ|² of the coupled equilibrium is
# above a_µ/β_µ, and a band can be induced (a ≤ 0).  With γ₁₂ = 0.3 the bound of a/β blew up in 23 steps
# (explicit KWT) and a_π = −0.2 gave a DomainError; with the coupled bound both stay at equilibrium.
@testset "band bound with strong interband coupling" begin
    gr = Grid(16, 12, 8; dx=0.5, buf=1); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE"), gr)
    for (a2, rk) in ((0.59, false), (0.59, true), (-0.2, true))
        md = Model((Band(γkwt=10.0), Band(m=0.4, a=a2, γkwt=10.0)), [0.0 0.3; 0.3 0.0]; κ=2.0)
        st = Stepper(md, gr, mk; threads=false, rkl=rk)
        P = sqrt.(TDGL._psi2_bound(md))
        s = initial_state(md, gr, mk, 0.0); rng = Random.Xoshiro(1)
        for µ in 1:2, c in eachindex(s.G[µ])
            s.G[µ][c] = mk.τψ[c] * P[µ] * (1 + 0.01 * (randn(rng) + im * randn(rng)))
        end
        for _ in 1:200; step!(s, st; H=0.0, HJ=0.0); end
        @test all(isfinite, s.G[2]) && maximum(abs, s.G[2]) < 1.1 * P[2]
    end
end

# The rotating frame: in the gauge Φ = 0 the stationary state turns in the conversion layers,
# and the explicit step of that rotation raised max|Ψ1| by ~√(1+γ̃²|Ψ|²)ω²dt/4 (1.084 / 1.044 / 1.005 at dt, dt/2 and
# with Euler substeps, γ̃ = 30, J = 0.6).  Stepped in the turning gauge, the state is the same with any dt and scheme.
@testset "rotating frame: KWT state independent of dt" begin
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    md = two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=30.0)
    function relax(; kw...)
        st = Stepper(md, gr, mk; threads=false, kw...)
        s = initial_state(md, gr, mk, 0.03); HJ = selffield(st, 0.6)
        for _ in 1:round(Int, 40 / st.dt); step!(s, st; H=0.03, HJ=HJ); end
        maximum(abs, s.G[1])
    end
    a = relax(rotframe=true); b = relax(rotframe=true, dt_scale=0.25); c = relax(rotframe=true, rkl=false)
    @test abs(a - b) < 1e-5 && abs(a - c) < 1e-5 && a < 1.0
    @test relax(rotframe=false) - a > 0.05          # without the frame, at the same dt
    @test Stepper(md, gr, mk; threads=false).rotframe                     # on by default with KWT
    @test !Stepper(two_band(xi_ratio=4.0, κ=2.0), gr, mk; threads=false).rotframe
    @test_throws ErrorException Stepper(md, gr, mk; threads=false, rotframe=true, extrap=false)
    # the stage margin: at least the stages of the bound, an error below 1
    @test all(Stepper(md, gr, mk; threads=false, rkl_margin=1.1).rkl .>= Stepper(md, gr, mk; threads=false, rkl_margin=1.0).rkl)
    @test_throws ErrorException Stepper(md, gr, mk; threads=false, rkl_margin=0.9)
    # rkl = "rkc": damped RKC1 stages, fewer for the same bound, the same KWT state
    sc = Stepper(md, gr, mk; threads=false, rkl="rkc"); sl = Stepper(md, gr, mk; threads=false, rkl=true)
    @test sc.dt == sl.dt && all(sc.rkl .<= sl.rkl) && sum(sc.rkl) < sum(sl.rkl) && sl.rkc === nothing
    # where RKC saves no stage the band stays RKL: a run without KWT keeps its bits
    m0 = two_band(xi_ratio=4.0, κ=2.0); s0 = Stepper(m0, gr, mk; threads=false, rkl="rkc")
    @test all(isnothing, s0.rkc) && s0.rkl == Stepper(m0, gr, mk; threads=false).rkl
    @test abs(relax(rkl="rkc") - a) < 1e-5
    @test_throws ErrorException Stepper(md, gr, mk; threads=false, rkl="rk4")
end

# The RKC1 recursion (RKCCoef) on the scalar decay y' = −λy, z = −λτ: s stages bridge 0.762 s² explicit steps with the damping
# ε = 0.5 of the option (0.968 s² with the textbook 0.05; RKL1 s(s+1)/2),
# |R(z)| <= 1 on the whole interval and above 1 just past it, and R(z) = e^z + O(z²): first order.
@testset "RKC stages" begin
    function R(k, s, z)
        a = 1.0; b = k.μ[1] * (1 + z / k.q)
        for j in 2:s
            c = k.μ[j] * b * (1 + z / k.q) + k.ν[j] * a
            a, b = b, c
        end
        b
    end
    for s in 2:20
        k = TDGL.RKCCoef(s); β2 = TDGL._rkc_ratio(s)
        @test 0.761 * s^2 <= β2 <= 0.781 * s^2 && k.c[end] == 1.0 && all(diff(k.c) .> 0)
        @test 0.967 * s^2 <= TDGL._rkc_ratio(s; ε=0.05) <= 0.971 * s^2
        @test maximum(z -> abs(R(k, s, z)), range(0, -2β2, length=2001)) <= 1 + 1e-12
        @test abs(R(k, s, -2.2β2)) > 1                                                  # |R| = 1 again at 2ω₀/ω₁ <= 1.06β
        e1 = R(k, s, -1e-3) - exp(-1e-3); e2 = R(k, s, -5e-4) - exp(-5e-4)
        @test abs(e1) < 1e-6 && 3.9 < e1 / e2 < 4.1                                  # local error O(z²)
    end
end

# The conversion length of regime(): 0.80 ξ for the two-band diode model (in the layer: 0.80 → 2.40 ξ with γ̃ = 10 at the
# layer amplitudes; here at the equilibrium ones), ×√(1+γ̃²|Ψ|²)^(1/2) with KWT, and one band against the law of O2.
@testset "regime(): the conversion length" begin
    r0 = regime(two_band(xi_ratio=4.0, κ=2.0); L=20.0); r10 = regime(two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=10.0); L=20.0)
    @test 0.75 < r0.lambda_c < 0.85 && 2.3 < r10.lambda_c < 2.6 && r10.L_over_lambda_c ≈ 20 / r10.lambda_c
    md1 = Model((Band(Γ=1.0, γkwt=10.0), Band(Γ=1.0, a=1e-9)), zeros(2, 2); κ=25.0)   # a second band with no condensate
    @test regime(md1; L=40.0).lambda_c ≈ sqrt(sqrt(1 + 100) / 1) rtol = 1e-6
    r1 = regime(Model((Band(Γ=1.0, γkwt=10.0),), zeros(1, 1); κ=25.0); L=40.0)       # one band (the pyTDGL case)
    @test r1.lambda_c ≈ sqrt(sqrt(1 + 100)) rtol = 1e-12
    @test r1.u == (1.0,) && r1.L_over_lambda_c ≈ 40 / r1.lambda_c
end


# Three identical bands with equal repulsive couplings (γ < 0) are frustrated: the ground state breaks time reversal, with
# phases 0, ±2π/3, and |Ψ|² = ε + Σ_ν γ cos Δφ = 1 + |γ| in each band (analytic).  From unequal starting phases.
@testset "three bands: frustrated couplings give the TRSB state" begin
    γ = -0.1
    md = Model((Band(), Band(), Band()), [0.0 γ γ; γ 0.0 γ; γ γ 0.0]; κ=25.0)
    gr = Grid(8, 8, 4; dx=0.5); mk = box_masks(gr); st = Stepper(md, gr, mk; threads=false)
    s = initial_state(md, gr, mk, 0.0); s.G[2] .*= cis(1.0); s.G[3] .*= cis(2.5)
    for _ in 1:round(Int, 200 / st.dt); step!(s, st; H=0.0, HJ=0.0); end
    c = ((gr.x1 + gr.x2 + 1) ÷ 2, (gr.y1 + gr.y2 + 1) ÷ 2, (gr.z1 + gr.z2 + 1) ÷ 2)
    ψ = [s.G[µ][c...] for µ in 1:3]
    @test abs(abs(angle(ψ[2] / ψ[1])) - 2pi / 3) < 1e-6 && abs(abs(angle(ψ[3] / ψ[1])) - 2pi / 3) < 1e-6
    @test angle(ψ[2] / ψ[1]) * angle(ψ[3] / ψ[1]) < 0                 # one chirality, not 0 and π
    @test all(abs(abs2(x) - (1 + abs(γ))) < 1e-6 for x in ψ)
end

# Two normal fluids (qp_exchange, exchange.jl): the δ solve is exact; a fast exchange is the step of today to O(1/ν); and in a
# long strip without Josephson the σ share of the converted current follows the 1D two-fluid law (κ = 25, γ̃ = 10, contacts 41 ξ apart: 0.5484 / 0.4567 with ν = 10 / 1; one fluid 0.5664).
@testset "two normal fluids (qp_exchange)" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    H = 0.03; HJ = selffield(Stepper(md, gr, mk; threads=false), 0.5)
    s0 = initial_state(md, gr, mk, H); st0 = Stepper(md, gr, mk; threads=false)
    for _ in 1:300; step!(s0, st0; H=H, HJ=HJ); end
    st1 = Stepper(md, gr, mk; threads=false, qp_exchange=1.0); qx = st1.qpx
    @test qx.s[1] ≈ 1 / (1 + 16 * 0.59) && qx.G ≈ qx.s[1] * 1.0 && st1.dt == st0.dt
    TDGL.qp_potential!(qx, s0, st1)
    # the residual of σn∇²δ − λ0δ = S on the unknown nodes (mirror ghosts on y, z; δ = 0 on the x faces)
    δ = qx.δ; (; Nx, Ny, Nz, dx) = gr; res = 0.0
    S = zeros(size(δ)); g = ntuple(µ -> ntuple(i -> stiffness(md.bands[µ], i), 3), 2)
    for k in gr.z1:gr.z2+1, j in gr.y1:gr.y2+1, i in gr.x1:gr.x2+1
        c = ntuple(µ -> TDGL._conv(s0.G, µ, g[µ], md.γ[µ], nothing, s0.Vx, s0.Vy, s0.Vz, mk.τx, mk.τy, mk.τz, i, j, k,
                                   1/dx^2, 1/dx^2, 1/dx^2), 2)
        S[i,j,k] = c[2] / qx.s[2] - c[1] / qx.s[1]
    end
    for k in 1:Nz+1, j in 1:Ny+1, i in 2:Nx
        jm = j == 1 ? 2 : j-1; jp = j == Ny+1 ? Ny : j+1; km = k == 1 ? 2 : k-1; kp = k == Nz+1 ? Nz : k+1
        lap = (δ[i+1,j,k] + δ[i-1,j,k] + δ[i,jp,k] + δ[i,jm,k] + δ[i,j,kp] + δ[i,j,km] - 6δ[i,j,k]) / dx^2
        res = max(res, abs(qx.σn * lap - qx.λ0 * δ[i,j,k] - S[i,j,k]))
    end
    @test maximum(abs, S) > 0.1 && res < 1e-11
    # a fast exchange: the step of today to O(1/ν)
    sa = initial_state(md, gr, mk, H); sb = initial_state(md, gr, mk, H); stb = Stepper(md, gr, mk; threads=false, qp_exchange=1e6)
    for _ in 1:300; step!(sa, st0; H=H, HJ=HJ); step!(sb, stb; H=H, HJ=HJ); end
    @test maximum(abs.(sa.G[1] .- sb.G[1])) < 1e-4 && maximum(abs.(sa.G[2] .- sb.G[2])) < 1e-4
    # the strip
    mk2 = two_band(xi_ratio=4.0, κ=25.0, gamma12=0.0, gamma_kwt=10.0)
    function strip(ν)
        g2 = Grid(80, 4, 2; dx=0.5); m2 = build_masks(GeomSpec(GEOM_MODE="SIM"), g2)
        st = Stepper(mk2, g2, m2; threads=false, qp_exchange=ν); s = initial_state(mk2, g2, m2, 0.0)
        HJ2 = selffield(st, 0.05)
        for _ in 1:round(Int, 300 / st.dt); step!(s, st; H=0.0, HJ=HJ2); end
        Ib = band_currents(s, st); Ib[1] / (Ib[1] + Ib[2])
    end
    f∞, f10, f1 = strip(nothing), strip(10.0), strip(1.0)
    @test isapprox(f∞, 0.5664; atol=0.003) && isapprox(f10, 0.5484; atol=0.01) && isapprox(f1, 0.4567; atol=0.02)
end

# N normal fluids (QPExchangeN): with two bands it is the two-band model (the same δ = μ_π − μ_σ, the same steps, to rounding);
# with three bands the potentials keep the gauge Σ s_µ μ_µ = 0, a fast exchange is one fluid to O(1/ν), and rates without
# detailed balance are refused.
@testset "N normal fluids (qp_exchange as a matrix)" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    H = 0.03; st0 = Stepper(md, gr, mk; threads=false); HJ = selffield(st0, 0.5)
    s0 = initial_state(md, gr, mk, H); for _ in 1:300; step!(s0, st0; H=H, HJ=HJ); end
    sa = Stepper(md, gr, mk; threads=false, qp_exchange=1.0); q2 = sa.qpx
    χ = [q2.s[µ] * md.σn * md.bands[µ].Γ / stiffness(md.bands[µ]) for µ in 1:2]
    sb = Stepper(md, gr, mk; threads=false, qp_exchange=[0.0 1.0; χ[1]/χ[2] 0.0]); qn = sb.qpx
    @test qn isa TDGL.QPExchangeN && all(qn.s .≈ q2.s)
    TDGL.qp_potential!(q2, s0, sa); TDGL.qp_potential!(qn, s0, sb)
    @test maximum(abs.((qn.μ[2] .- qn.μ[1]) .- q2.δ)) < 1e-12 * max(1, maximum(abs, q2.δ)) && maximum(abs, q2.δ) > 1e-3
    a = deepcopy(s0); b = deepcopy(s0)
    for _ in 1:200; step!(a, sa; H=H, HJ=HJ); step!(b, sb; H=H, HJ=HJ); end
    @test maximum(abs.(a.G[1] .- b.G[1])) < 1e-10 && maximum(abs.(a.G[2] .- b.G[2])) < 1e-10
    # three bands (the frustrated model of the example, notched box): the gauge, a fast exchange, detailed balance
    m3 = Model((Band(), Band(m=2.0, a=0.8), Band(m=0.5, Γ=1.5, a=0.6)),
               [0.0 -0.05 -0.05; -0.05 0.0 -0.05; -0.05 -0.05 0.0]; κ=25.0)
    st3 = Stepper(m3, gr, mk; threads=false); H3 = selffield(st3, 0.3)
    s3 = initial_state(m3, gr, mk, H); for _ in 1:300; step!(s3, st3; H=H, HJ=H3); end
    sh = ntuple(µ -> stiffness(m3.bands[µ]) * m3.bands[µ].Γ / m3.bands[µ].β, 3); sh = sh ./ sum(sh)
    χ3 = [sh[µ] * m3.σn * m3.bands[µ].Γ / stiffness(m3.bands[µ]) for µ in 1:3]
    rates(r) = [µ == n ? 0.0 : r / χ3[µ] for µ in 1:3, n in 1:3]        # G_µν = r for every pair: detailed balance
    stn = Stepper(m3, gr, mk; threads=false, qp_exchange=rates(0.5)); TDGL.qp_potential!(stn.qpx, s3, stn)
    pot = stn.qpx.μ
    @test maximum(abs.(sh[1] .* pot[1] .+ sh[2] .* pot[2] .+ sh[3] .* pot[3])) < 1e-12 && maximum(abs, pot[3]) > 1e-3
    # each μ_µ solves σn∇²μ − S⁻¹Lμ = S⁻¹C − 𝟙ΣC node by node, with a different conductance for each pair (the modes of
    # S⁻¹L against the equation itself; mirror ghosts on y, z, μ = 0 on the x faces)
    Gp = [0.0 0.5 0.1; 0.5 0.0 2.0; 0.1 2.0 0.0]
    stg = Stepper(m3, gr, mk; threads=false, qp_exchange=[µ == n ? 0.0 : Gp[µ, n] / χ3[µ] for µ in 1:3, n in 1:3])
    TDGL.qp_potential!(stg.qpx, s3, stg); M = stg.qpx.μ
    (; Nx, Ny, Nz, dx) = gr; g3 = ntuple(µ -> ntuple(i -> stiffness(m3.bands[µ], i), 3), 3)
    C = [zeros(size(M[1])) for _ in 1:3]
    for k in gr.z1:gr.z2+1, j in gr.y1:gr.y2+1, i in gr.x1:gr.x2+1, µ in 1:3
        C[µ][i,j,k] = TDGL._conv(s3.G, µ, g3[µ], m3.γ[µ], nothing, s3.Vx, s3.Vy, s3.Vz, mk.τx, mk.τy, mk.τz, i, j, k,
                                 1/dx^2, 1/dx^2, 1/dx^2)
    end
    res3 = 0.0
    for k in 1:Nz+1, j in 1:Ny+1, i in 2:Nx, µ in 1:3
        jm = j == 1 ? 2 : j-1; jp = j == Ny+1 ? Ny : j+1; km = k == 1 ? 2 : k-1; kp = k == Nz+1 ? Nz : k+1
        P = M[µ]
        lap = (P[i+1,j,k] + P[i-1,j,k] + P[i,jp,k] + P[i,jm,k] + P[i,j,kp] + P[i,j,km] - 6P[i,j,k]) / dx^2
        Lµ = sum(Gp[µ, n] * (P[i,j,k] - M[n][i,j,k]) for n in 1:3) / sh[µ]
        res3 = max(res3, abs(m3.σn * lap - Lµ - (C[µ][i,j,k] / sh[µ] - (C[1][i,j,k] + C[2][i,j,k] + C[3][i,j,k]))))
    end
    @test maximum(abs, C[3]) > 0.1 && res3 < 1e-10
    sf = Stepper(m3, gr, mk; threads=false, qp_exchange=rates(1e6))
    x = deepcopy(s3); y = deepcopy(s3)
    for _ in 1:200; step!(x, st3; H=H, HJ=H3); step!(y, sf; H=H, HJ=H3); end
    @test all(maximum(abs.(x.G[µ] .- y.G[µ])) < 1e-4 for µ in 1:3)
    @test_throws ErrorException Stepper(m3, gr, mk; qp_exchange=[0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0] .* [1, 2, 3])
    @test_throws ErrorException Stepper(m3, gr, mk; qp_exchange=0.5)            # a number is two bands only
end

# Anisotropic bands (m_axes): an isotropic m_axes is the band of today to the bit; a band with m_x = 4 on dx = 0.5 is the
# isotropic band on dx = 1 (the stretched grid: the same equations per node), here the stationary healing profile next to
# two hard walls (x and y planes); the supercurrent along each axis carries its own stiffness (λ_i ∝ √m_i follows); the
# material file takes m_x, m_y, m_z or xi_ratio_x, ...
@testset "anisotropic bands (m_axes)" begin
    md0 = two_band(xi_ratio=4.0, κ=25.0)
    mdi = Model((Band(), Band(m_axes=ntuple(_ -> md0.bands[2].m, 3), Γ=1.0, a=0.59)), [0.0 1e-3; 1e-3 0.0]; κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    a = Stepper(md0, gr, mk; threads=false); b = Stepper(mdi, gr, mk; threads=false)
    @test a.dt == b.dt && a.bc == b.bc && a.wa == b.wa
    sa = initial_state(md0, gr, mk, 0.03); sb = initial_state(mdi, gr, mk, 0.03); HJ = selffield(a, 0.5)
    for _ in 1:50; step!(sa, a; H=0.03, HJ=HJ); step!(sb, b; H=0.03, HJ=HJ); end
    @test sa.G[1] == sb.G[1] && sa.G[2] == sb.G[2] && sa.θx == sb.θx
    # the stretched grid
    function healed(band, g)
        md = Model((band,), zeros(1, 1); κ=1.0)
        m = box_masks(g); m.f[g.x1 + 6, :, :] .= 0.0; m.f[:, g.y1 + 5, :] .= 0.0
        st = Stepper(md, g, m; threads=false, implicit=false, hard=true); s = initial_state(md, g, m, 0.0)
        for _ in 1:round(Int, 60 / st.dt); step!(s, st; H=0.0, HJ=0.0); end
        abs.(s.G[1])
    end
    pa = healed(Band(m_axes=(4.0, 1.0, 1.0)), Grid(16, 12, 2; dx=0.5))
    pb = healed(Band(), Grid(16, 12, 2; dx=1.0, dy=0.5, dz=0.5))
    @test maximum(abs.(pa .- pb)) < 1e-9 && minimum(pa[4:end-3, 4:end-3, 2]) < 0.1 && maximum(pa) > 0.95
    # supercurrent along each axis: a phase twist q along x and along y
    md = Model((Band(m_axes=(4.0, 1.0, 1.0)),), zeros(1, 1); κ=2.0)
    g = Grid(12, 12, 2; dx=0.5); m = box_masks(g); st = Stepper(md, g, m; threads=false)
    q = 0.2; s = initial_state(md, g, m, 0.0)
    s.G[1] .= [cis(q * (i - 1) * g.dx) for i in 1:g.Nx+1, j in 1:g.Ny+1, k in 1:g.Nz+1]
    TDGL.supercurrent!(s, st); jx = s.Jx[g.x1 + 3, g.y1 + 3, g.z1 + 1]
    s.G[1] .= [cis(q * (j - 1) * g.dy) for i in 1:g.Nx+1, j in 1:g.Ny+1, k in 1:g.Nz+1]
    TDGL.supercurrent!(s, st); jy = s.Jy[g.x1 + 3, g.y1 + 3, g.z1 + 1]
    @test jx ≈ 0.25 * sin(q * g.dx) && jy ≈ sin(q * g.dy)
    # the material file
    mf = material(Dict("kappa" => 2.0, "band" => [Dict("name" => "s", "m_x" => 4.0), Dict("name" => "p", "xi_ratio" => 2.0, "a" => 0.5, "xi_ratio_z" => 1.0)]))[1]
    @test mf.bands[1].ma == (4.0, 1.0, 1.0) && mf.bands[1].m == 4.0
    @test mf.bands[2].ma == (1 / (4 * 0.5), 1 / (4 * 0.5), 1 / 0.5)
end

# Langevin noise against the exact equilibrium of decoupled nodes (g → 0: m = 10⁸): exp(−F/T̃) is Gaussian in n = |Ψ|²,
# with mean ε/β and variance T̃/(dV β); the Euler–Maruyama step biases the variance by 1/(1 − ε dt/Γ).  The time
# correlation of n is exp(−2εt/Γ) (the linear rate of the amplitude), here at lag Γ/(4ε): e^{−1/2}.
@testset "Langevin noise: equilibrium of decoupled nodes" begin
    Γ = 2.0; T = 0.01
    md = Model((Band(m=1e8, Γ=Γ),), zeros(1, 1); κ=2.0)
    gr = Grid(10, 10, 2; dx=0.5); mk = box_masks(gr)
    st = Stepper(md, gr, mk; threads=false, dt_scale=0.02, noise=T, noise_seed=11)
    s = initial_state(md, gr, mk, 0.0)
    dV = gr.dx * gr.dy * gr.dz; lag = round(Int, Γ / 4 / st.dt)
    for _ in 1:round(Int, 20 / st.dt); step!(s, st; H=0.0, HJ=0.0); end     # to equilibrium
    rng = (gr.x1+1:gr.x2, gr.y1+1:gr.y2, gr.z1+1:gr.z2)
    n() = vec(abs2.(s.G[1][rng...]))
    sm = 0.0; s2 = 0.0; c = 0.0; N = 0; prev = n()
    for q in 1:round(Int, 1500 / (lag * st.dt))
        for _ in 1:lag; step!(s, st; H=0.0, HJ=0.0); end
        x = n(); sm += sum(x); s2 += sum(abs2, x); c += sum((x .- 1) .* (prev .- 1)); N += length(x); prev = x
    end
    mean = sm / N; var = s2 / N - mean^2; ρ = (c / N) / var
    @test isapprox(mean, 1.0; atol=0.01)
    @test isapprox(var, T / dV / (1 - st.dt / Γ); rtol=0.04)
    @test isapprox(ρ, exp(-0.5); atol=0.03)
end

# With KWT the mobility depends on Ψ: the noise is scaled apart along Ψ (1/(Γ s)) and across it (s/Γ), with the Itô drift,
# s = √(1 + γ̃²|Ψ|²).  The equilibrium is still exp(−F/T̃): the same mean and variance of n = |Ψ|²; the amplitude relaxes s
# times slower (correlation of n at lag t: exp(−2εt/(Γ s))) and the phase diffuses s times faster
# (⟨cos Δφ⟩ = exp(−T̃ s t/(2Γ dV n))).  Decoupled nodes (m = 10⁸), γ̃ = 3, s = √10.
@testset "Langevin noise with KWT: equilibrium, amplitude and phase rates" begin
    Γ = 2.0; T = 0.01; γk = 3.0; sk = sqrt(1 + γk^2)
    md = Model((Band(m=1e8, Γ=Γ, γkwt=γk),), zeros(1, 1); κ=2.0)
    gr = Grid(10, 10, 2; dx=0.5); mk = box_masks(gr)
    st = Stepper(md, gr, mk; threads=false, dt_scale=0.02, noise=T, noise_seed=5)
    s = initial_state(md, gr, mk, 0.0)
    dV = gr.dx * gr.dy * gr.dz
    for _ in 1:round(Int, 40 / st.dt); step!(s, st; H=0.0, HJ=0.0); end
    rng = (gr.x1+1:gr.x2, gr.y1+1:gr.y2, gr.z1+1:gr.z2)
    lagn = round(Int, Γ * sk / 4 / st.dt)                       # exp(−1/2) for n
    lagp = round(Int, 2Γ * dV / (T * sk) * 0.5 / st.dt)          # exp(−1/2) for the phase
    sm = 0.0; s2 = 0.0; c = 0.0; N = 0; cp = 0.0; Np = 0
    prev = copy(s.G[1][rng...]); since = 0
    for q in 1:round(Int, 3000 / st.dt)
        step!(s, st; H=0.0, HJ=0.0); since += 1
        if since % lagn == 0
            x = abs2.(s.G[1][rng...]); sm += sum(x); s2 += sum(abs2, x); N += length(x)
        end
        if since == lagp
            g = s.G[1][rng...]; cp += sum(real.(g .* conj.(prev)) ./ (abs.(g) .* abs.(prev))); Np += length(g)
            prev = copy(g); since = 0
        end
    end
    mean = sm / N; var = s2 / N - mean^2
    @test isapprox(mean, 1.0; atol=0.01)
    @test isapprox(var, T / dV; rtol=0.06)
    @test isapprox(cp / Np, exp(-0.5); atol=0.03)            # the phase: s times faster than without KWT
    # the amplitude rate: the correlation of n at lag lagn
    s0 = deepcopy(s); xs = Float64[]; ys = Float64[]
    for q in 1:400
        a = abs2.(s.G[1][rng...]); for _ in 1:lagn; step!(s, st; H=0.0, HJ=0.0); end; b = abs2.(s.G[1][rng...])
        append!(xs, vec(a) .- 1); append!(ys, vec(b) .- 1)
    end
    @test isapprox(sum(xs .* ys) / sum(abs2, xs), exp(-0.5); atol=0.04)
end

# The noise with KWT in an extended system (coupled nodes, m = 1) at the production step: the equilibrium does not depend on the
# mobility, so γ̃ = 3 gives the ⟨|Ψ|²⟩ of γ̃ = 0.  It holds with the rotating frame (the default with KWT: within 0.1 % here,
# 0.3–1.7 % over dt = 0.0006–0.04).  Until 26-oct-2026 it failed without the frame (19 % high at this step: the radial error of the
# tangent steps, which the noise feeds everywhere); with noise the KWT band now steps in log-polar form (bands.jl), and it holds
# without the frame too.
@testset "Langevin noise with KWT in an extended system (production step)" begin
    function media(γk; kw...)
        md = Model((Band(m=1.0, Γ=2.0, γkwt=γk),), zeros(1, 1); κ=25.0)
        gr = Grid(10, 10, 2; dx=0.5); mk = box_masks(gr)
        st = Stepper(md, gr, mk; threads=false, noise=0.01, noise_seed=7, kw...)
        s = initial_state(md, gr, mk, 0.0)
        for _ in 1:round(Int, 40 / st.dt); step!(s, st; H=0.0, HJ=0.0); end
        rng = (gr.x1+1:gr.x2, gr.y1+1:gr.y2, gr.z1+1:gr.z2); sm = 0.0; N = 0
        for q in 1:round(Int, 1500 / st.dt)
            step!(s, st; H=0.0, HJ=0.0)
            q % 13 == 0 || continue
            x = abs2.(s.G[1][rng...]); sm += sum(x); N += length(x)
        end
        sm / N, st
    end
    n0, _ = media(0.0); n3, st3 = media(3.0); nf, _ = media(3.0; rotframe=false)
    @test st3.rotframe && isapprox(n3, n0; rtol=0.02)
    @test isapprox(nf, n0; rtol=0.02)        # (until nf > 1.1 n0)
end

# the noise with KWT on the two-band notched box at κ = 2, which gave NaN within 26 steps (γ̃ = 10)
# and 8 (γ̃ = 30) with RKC stages, and whose π band collapsed when finite.  Finite, and ⟨|Ψπ|²⟩ within 5 % of γ̃ = 0.
@testset "Langevin noise with KWT on the two-band notched box" begin
    gr = Grid(24, 12, 6; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    function media(γk)
        md = two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=γk)
        st = Stepper(md, gr, mk; threads=false, noise=1e-3, noise_seed=3)
        s = initial_state(md, gr, mk, 0.03); HJ = selffield(st, 0.0); a = 0.0; c = 0
        for q in 1:1500
            step!(s, st; H=0.03, HJ=HJ)
            q > 500 && q % 10 == 0 && (a += sum(abs2, view(s.G[2], gr.x1+1:gr.x2, gr.y1+1:gr.y2, gr.z1+1:gr.z2)); c += 1)
        end
        all(isfinite, s.G[1]) && all(isfinite, s.G[2]), a / c, st
    end
    ok0, p0, _ = media(0.0); ok, p, st = media(30.0)
    @test ok0 && ok && st.rkl == (5, 13)
    @test isapprox(p, p0; rtol=0.05)
end

# A sculpted sample (a shape with vacuum = true): the free surface is the zero-flux condition, Ψ = 0 in the vacuum, and
# the transport current crosses the section of the sample only.  The cable of tdgl2/shapes/cable_x.toml.
@testset "sculpted sample: free surface and current through the section" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(20, 12, 12; dx=0.5, buf=2)
    (; x1, x2, y1, y2, z1, z2) = gr
    gs = GeomSpec(GEOM_MODE="CUSTOM", geom_file=joinpath(@__DIR__, "..", "shapes", "cable_x.toml"))
    mk = build_masks(gs, gr)
    @test_throws ErrorException Stepper(md, gr, mk; threads=false)                     # vacuum needs σvac
    @test_throws ErrorException Stepper(md, gr, mk; σvac=0.01, selffield_bc=:slab, threads=false)
    @test_throws ErrorException Stepper(md, gr, mk; qp_exchange=1.0, threads=false)        # (vacuum needs σvac, with or without)
    st = Stepper(md, gr, mk; σvac=0.01, threads=false)
    # H = 0, J = 0: the uniform state stays uniform up to the surface (zero flux), Ψ = 0 in the vacuum
    s = initial_state(md, gr, mk, 0.0)
    for _ in 1:100; step!(s, st; H=0.0, HJ=0.0); end
    ref = abs2(s.G[1][x1+10, y1+6, z1+6])
    @test maximum(abs(abs2(s.G[1][I]) - ref) for I in CartesianIndices(mk.vac) if !mk.vac[I] && mk.τψ[I] > 0) < 1e-13
    @test all(iszero, s.G[1][mk.vac]) && all(iszero, s.G[2][mk.vac])
    # the same disc of f = 0 normal metal (release 1.x: no vacuum) suppresses Ψ at the surface by proximity
    p0 = joinpath(mktempdir(), "disco_f0.toml")
    write(p0, replace(read(gs.geom_file, String), "vacuum = true" => "f = 0.0"))
    mk0 = build_masks(GeomSpec(GEOM_MODE="CUSTOM", geom_file=p0), gr)
    st0 = Stepper(md, gr, mk0; σvac=0.01, threads=false); s0 = initial_state(md, gr, mk0, 0.0)
    for _ in 1:100; step!(s0, st0; H=0.0, HJ=0.0); end
    @test maximum(abs(abs2(s0.G[1][I]) - ref) for I in CartesianIndices(mk.vac) if !mk.vac[I] && mk.τψ[I] > 0) > 1e-2
    # J = 0.3: the static current through the middle section is J times the section (as bar_field counts it), all of
    # it through the sample's nodes
    J = 0.3; s = initial_state(md, gr, mk, 0.03); HJ = selffield(st, J)
    for _ in 1:round(Int, 120 / st.dt); step!(s, st; H=0.03, HJ=HJ); end
    im_ = (x1 + x2) ÷ 2
    I_all = 0.0; I_vac = 0.0
    for kk in z1:z2+1, jj in y1:y2+1
        c = md.κ^2 * ((s.hz[im_,jj,kk] - s.hz[im_,jj-1,kk]) / gr.dy - (s.hy[im_,jj,kk] - s.hy[im_,jj,kk-1]) / gr.dz) * gr.dy * gr.dz
        I_all += c; mk.vac[im_,jj,kk] && (I_vac += c)
    end
    sec = .!mk.vac[x1+1, y1:y2+1, z1:z2+1]; A = 0.0; nq = 8
    for b in 1:nq*(z2-z1+1), a in 1:nq*(y2-y1+1)
        yp = (y1-1)*gr.dy + (a-0.5)*gr.dy/nq; zp = (z1-1)*gr.dz + (b-0.5)*gr.dz/nq
        sec[clamp(round(Int, yp/gr.dy)+1, y1, y2+1)-y1+1, clamp(round(Int, zp/gr.dz)+1, z1, z2+1)-z1+1] && (A += (gr.dy/nq)*(gr.dz/nq))
    end
    @test isapprox(abs(I_all), J * A; rtol=2e-3)
    @test abs(I_vac) < 1e-3 * abs(I_all)            # the normal current σvac·E of the vacuum while it relaxes (1e-5 at 300 t_GL)
    @test isapprox(abs(sum(band_currents(s, st; i=im_))), abs(I_all); rtol=1e-2)   # nearly static: nearly all of it is super
end

# Two normal fluids in the conductor only (T8).  With the insulating padding the quasiparticles live in the bar of the sample and
# its contacts (zero flux to the insulator): the strip then follows the 1D two-fluid law with the code's contact (half a ξ of
# metal before the sample: 0.4427 at γ̃ = 10, ν = 1, solved apart) to 0.01, where the metal padding, which conducts and
# exchanges around the strip, is 0.025 off.  A sculpted sample cuts the links to its vacuum (conjugate gradients): the solve
# satisfies the cut-link equation to its tolerance, δ = 0 in the vacuum, and with nothing cut it is the separable solve.
@testset "two normal fluids with the insulating padding and in a sculpted sample (T8)" begin
    m2 = two_band(xi_ratio=4.0, κ=25.0, gamma12=0.0, gamma_kwt=10.0)
    g2 = Grid(80, 4, 2; dx=0.5); k2 = build_masks(GeomSpec(GEOM_MODE="SIM"), g2)
    st = Stepper(m2, g2, k2; threads=false, qp_exchange=1.0, σvac=0.01); s = initial_state(m2, g2, k2, 0.0)
    @test size(st.qpx.S) == (g2.Nx - 1, g2.y2 - g2.y1 + 2, g2.z2 - g2.z1 + 2)
    HJ2 = selffield(st, 0.05)
    for _ in 1:round(Int, 300 / st.dt); step!(s, st; H=0.0, HJ=HJ2); end
    Ib = band_currents(s, st)
    @test isapprox(Ib[1] / (Ib[1] + Ib[2]), 0.4427; atol=0.01)
    @test all(iszero, st.qpx.δ[:, 1:g2.y1-1, :]) && all(iszero, st.qpx.δ[:, :, g2.z2+2:end])     # nothing in the insulator
    # a sculpted cable: the conjugate-gradient solve of δ
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(20, 12, 12; dx=0.5, buf=2)
    mk = build_masks(GeomSpec(GEOM_MODE="CUSTOM", geom_file=joinpath(@__DIR__, "..", "shapes", "cable_x.toml")), gr)
    sc = Stepper(md, gr, mk; σvac=0.01, qp_exchange=1.0, threads=false); s = initial_state(md, gr, mk, 0.03)
    HJ = selffield(sc, 0.3)
    for _ in 1:200; step!(s, sc; H=0.03, HJ=HJ); end
    qx = sc.qpx; ps = qx.ps
    TDGL.qp_potential!(qx, s, sc)
    @test 0 < ps.iters[1] < ps.maxit
    # the source again, and the residual of the cut-link operator on the region
    (; x1, x2, y1, y2, z1, z2, dx) = gr; j0 = ps.j0 - 1; k0 = ps.k0 - 1
    S = zeros(size(qx.S)); g = ntuple(µ -> ntuple(i -> stiffness(md.bands[µ], i), 3), 2)
    for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1
        c = ntuple(µ -> TDGL._conv(s.G, µ, g[µ], md.γ[µ], nothing, s.Vx, s.Vy, s.Vz, mk.τx, mk.τy, mk.τz, i, j, k,
                                   1/dx^2, 1/dx^2, 1/dx^2), 2)
        S[i-1, j-j0, k-k0] = c[2] / qx.s[2] - c[1] / qx.s[1]
    end
    S[.!ps.cond] .= 0
    x = qx.δ[2:gr.Nx, ps.j0:ps.j0+size(S, 2)-1, ps.k0:ps.k0+size(S, 3)-1]
    y = similar(x); TDGL._qp_apply!(y, x, ps, qx.σn, qx.λ0, dx, dx, dx)
    @test maximum(abs, S) > 1e-3 && maximum(abs, y .+ S) < 1e-8 * maximum(abs, S)
    @test all(iszero, x[.!ps.cond])
    # nothing cut: the conjugate gradients give the separable solve
    q0 = TDGL.QPExchange(md, gr; ν=1.0, conductor=true); q1 = TDGL.QPExchange(md, gr; ν=1.0, conductor=true, cond=trues(size(mk.f)))
    TDGL.qp_potential!(q0, s, sc); TDGL.qp_potential!(q1, s, sc)
    @test maximum(abs, q1.δ .- q0.δ) < 1e-8 * maximum(abs, q0.δ)
end

# The Joule heat of two normal fluids: σn|E|² + σn s_σ s_π|∇δ|² + G δ² (T8).  A fast exchange is the heat of one fluid to O(1/ν);
# the extra source is the formula on the node; N fluids with two bands give the same.
@testset "Joule heat of two normal fluids (T8)" begin
    md = two_band(xi_ratio=4.0, κ=25.0)
    gr = Grid(24, 12, 4; dx=0.5); mk = build_masks(GeomSpec(GEOM_MODE="WEDGE", inj_margin=4), gr)
    H = 0.03; hp = Heat(joule=true, eta=0.05)
    function heated(ν)
        st = Stepper(md, gr, mk; threads=false, heat=hp, qp_exchange=ν); s = initial_state(md, gr, mk, H); HJ = selffield(st, 0.9)
        for _ in 1:300; step!(s, st; H=H, HJ=HJ); end
        s, st
    end
    _, st0 = heated(nothing); _, st6 = heated(1e6); s1, st1 = heated(1.0)
    @test maximum(abs.(st6.heat.θ .- st0.heat.θ)) < 1e-5 * maximum(st0.heat.θ)
    @test maximum(st1.heat.θ) > 1.2 * maximum(st0.heat.θ)
    qx = st1.qpx; i, j, k = gr.x1 + 6, gr.y1 + 3, gr.z1 + 2; δ = qx.δ; d = gr.dx
    g2 = sum(0.5 * ((δ[(i,j,k) .+ e...] - δ[i,j,k])^2 + (δ[i,j,k] - δ[(i,j,k) .- e...])^2) for e in ((1,0,0), (0,1,0), (0,0,1))) / d^2
    @test TDGL.qp_joule(qx, i, j, k, gr) ≈ qx.σn * qx.s[1] * qx.s[2] * g2 + qx.G * δ[i,j,k]^2
    χ = [qx.s[µ] * md.σn * md.bands[µ].Γ / stiffness(md.bands[µ]) for µ in 1:2]
    stN = Stepper(md, gr, mk; threads=false, qp_exchange=[0.0 1.0; χ[1]/χ[2] 0.0])
    TDGL.qp_potential!(stN.qpx, s1, stN); TDGL.qp_potential!(qx, s1, st1)
    @test isapprox(TDGL.qp_joule(stN.qpx, i, j, k, gr), TDGL.qp_joule(qx, i, j, k, gr); rtol=1e-9)
end

using LinearAlgebra: Diagonal, Symmetric, eigvals

# Langevin noise on the field (T9): in a normal box (Ψ = 0: f = 0 held) at H = 0 the field energy F_A = κ²dV Σ h² of the free
# plaquettes is quadratic in the link phases, and its mean is Σ_m (T̃/2)/(1 + λ_m dt/2) over its transverse modes (the implicit
# Euler of the field), λ_m = (κ²dx²/σn) the eigenvalues of D^{-1/2} CᵀC D^{-1/2} (C the discrete curl of curl_h!, D = σ_l/σn).
# The gradient modes are free (they cost nothing).  Uniform σn, and the insulating padding (λdt up to 4.5).
@testset "Langevin noise on the field: equipartition of the free modes (T9)" begin
    function equip(κ, σvac, buf, teq, tav)
        md = Model((Band(m=1.0, Γ=1.0),), zeros(1, 1); κ=κ)
        gr = Grid(6, 4, 4; dx=0.5, buf=buf); mk = box_masks(gr); mk.f .= 0.0
        T = 0.02
        st = Stepper(md, gr, mk; threads=false, hard=true, noise=T, noise_seed=3, noise_field=true, σvac=σvac)
        s = initial_state(md, gr, mk, 0.0); s.G[1] .= 0; s.F[1] .= 0
        (; Nx, Ny, Nz, dx) = gr
        L = vcat(vec([(1, i, j, k) for k in 2:Nz, j in 2:Ny, i in 1:Nx]), vec([(2, i, j, k) for k in 2:Nz, j in 1:Ny, i in 2:Nx]),
                 vec([(3, i, j, k) for k in 1:Nz, j in 2:Ny, i in 2:Nx]))
        P = vcat(vec([(1, i, j, k) for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx]), vec([(2, i, j, k) for k in 2:Nz-1, j in 2:Ny, i in 2:Nx-1]),
                 vec([(3, i, j, k) for k in 2:Nz, j in 2:Ny-1, i in 2:Nx-1]))
        hv(s) = [(p[1] == 1 ? s.hx : p[1] == 2 ? s.hy : s.hz)[p[2], p[3], p[4]] for p in P]
        z = deepcopy(s); C = zeros(length(P), length(L))
        for (c, l) in enumerate(L)
            for a in (z.φx, z.φy, z.φz); a .= 0; end
            (l[1] == 1 ? z.φx : l[1] == 2 ? z.φy : z.φz)[l[2], l[3], l[4]] = 1.0
            TDGL.curl_h!(z, gr); C[:, c] = hv(z)
        end
        Dl = st.D === nothing ? ones(length(L)) : [st.D[l[1]][l[2], l[3], l[4]] for l in L]
        ev = eigvals(Symmetric(Diagonal(1 ./ sqrt.(Dl)) * (C' * C) * Diagonal(1 ./ sqrt.(Dl)))); ev = ev[ev .> 1e-9 * maximum(ev)]
        expect = sum((T / 2) ./ (1 .+ (md.κ^2 * dx^2 / md.σn) .* ev .* (st.dt / 2)))
        FA(s) = md.κ^2 * dx^3 * sum(abs2, hv(s))
        for _ in 1:round(Int, teq / st.dt); step!(s, st; H=0.0, HJ=0.0); end
        acc = 0.0; n = 0
        for q in 1:round(Int, tav / st.dt)
            step!(s, st; H=0.0, HJ=0.0); q % 10 == 0 && (acc += FA(s); n += 1)
        end
        acc / n / expect, maximum(abs, s.G[1])
    end
    r, ψ = equip(0.25, nothing, 1, 500.0, 3000.0); @test isapprox(r, 1.0; atol=0.01) && ψ == 0
    r, _ = equip(1.0, nothing, 1, 100.0, 1000.0); @test isapprox(r, 1.0; atol=0.01)        # stiff modes, λdt up to 0.87
    r, _ = equip(0.25, 0.01, 2, 500.0, 3000.0); @test isapprox(r, 1.0; atol=0.01)          # σ per link
    @test_throws ErrorException Stepper(two_band(xi_ratio=4.0, κ=2.0), Grid(6, 4, 4; dx=0.5), box_masks(Grid(6, 4, 4; dx=0.5)); noise_field=true)
end

# Noise with KWT on RKL stages is refused (its equilibrium comes out biased); with RKC (the default) and with the Euler
# substeps it runs; without noise RKL is as before.
@testset "noise: RKL with KWT and the heat model refused" begin
    md = two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=10.0); gr = Grid(12, 8, 4; dx=0.5); mk = box_masks(gr)
    @test_throws ErrorException Stepper(md, gr, mk; threads=false, noise=1e-3, rkl=true)
    @test Stepper(md, gr, mk; threads=false, noise=1e-3).rkc !== nothing
    @test all(==(0), Stepper(md, gr, mk; threads=false, noise=1e-3, rkl=false).rkl)
    @test Stepper(md, gr, mk; threads=false, rkl=true).rkc === nothing
    # noise with the heat model is refused: its source would count the kicks as dissipation
    @test_throws ErrorException Stepper(two_band(xi_ratio=4.0, κ=2.0), gr, mk; threads=false, noise=1e-3, heat=Heat(D_T=0.25))
end
