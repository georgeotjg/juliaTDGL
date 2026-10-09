# ---------------------------------------------------------------------------
#  Optional terms: heat T(r), Joule σn|E|², contact margin and diffused Q.
#  Off, they are not read in the step; on, each does what it says.
#  Small mesh, κ=2: everything runs in seconds.
# ---------------------------------------------------------------------------
const KW3 = (GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4, buf=1,
             dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, kappa_p=2.0, xi_ratio=4.0,
             gamma12=0.001, gamma_ratio=1.0, H_const=0.030,
             selftest=false, verbose=false, use_threads=false, out_dir="",
             use_phi=false)

function _pasos3(cfg, n, J)
    p = derive_params(cfg); g = build_geometry(cfg, p)
    s = initial_state(p, g, cfg.H_const, Strict())
    HJ = p.inject_gain * J * p.slab_d / (2 * p.ks2)
    for _ in 1:n
        if cfg.use_phi
            error("_pasos3 es solo Φ=0")
        end
        substep!(s, p, g, Strict(); H=cfg.H_const, HJ=HJ, J_cur=J)
    end
    s, p, g
end

@testset "terms: flag guards" begin
    @test_throws ErrorException derive_params(Config(; KW3..., use_joule=true))
    @test_throws ErrorException derive_params(Config(; KW3..., use_heat=true, use_joule=true,
                                               use_phi=true, lambda_TF=2.0))
    @test_throws ErrorException derive_params(Config(; KW3..., use_Qss=true))
    @test_throws ErrorException derive_params(Config(; KW3..., use_Qss=true, use_phi=true,
                                               lambda_TF=2.0, tau_Q=0.0))
end

@testset "terms: heat off does not touch θ" begin
    s, p, _ = _pasos3(Config(; KW3...), 50, 0.9)
    @test all(==(p.T_bath), p.Temp)
    @test all(iszero, p.heat_diag)
end

@testset "terms: η_J=0 leaves θ at the bath and Ψ as with heat off" begin
    s0, p0, _ = _pasos3(Config(; KW3...), 200, 0.9)
    s1, p1, _ = _pasos3(Config(; KW3..., use_heat=true, eta_J=0.0), 200, 0.9)
    @test all(==(p1.T_bath), p1.Temp)
    # (a2n−θ)/β2 with θ=0 vs alpha2_n: the same number up to the last bit.
    @test isapprox(sum(abs2, s1.G1), sum(abs2, s0.G1); rtol=1e-13)
    @test isapprox(sum(abs2, s1.G2), sum(abs2, s0.G2); rtol=1e-13)
end

@testset "terms: diffusive dt" begin
    p0 = derive_params(Config(; KW3...))
    D = 50.0
    p1 = derive_params(Config(; KW3..., use_heat=true, D_T=D))
    @test p1.dt <= 0.15 * 0.5^2 / D * (1 + 1e-12)
    @test p1.dt < p0.dt
end

@testset "terms: Joule in the normal state, E² ~ J²" begin
# J=3 ≫ depairing. σn=1: ⟨E²⟩ stays below J² because the vacuum
# conducts in parallel. A misplaced dx in E would give ×4 or ×1/4.
    J = 3.0
    s, p, _ = _pasos3(Config(; KW3..., use_heat=true, use_joule=true,
                             gamma_bath=1e6), 4000, J)
    r = p.heat_diag[3] / J^2
    @test 0.2 < r < 1.2
    @test p.heat_diag[1] >= 0
end

@testset "terms: heat_margin removes the contact in Meissner" begin
# J=0.3 in SIM: Meissner. Next to the faces E≠0 (normal→super conversion).
    kw = (; KW3..., GEOM_MODE="SIM", Nx_sc=24, use_heat=true, use_joule=true,
          gamma_bath=1e6)
    _, p0, _ = _pasos3(Config(; kw..., heat_margin=0), 3000, 0.3)
    _, p8, _ = _pasos3(Config(; kw..., heat_margin=6), 3000, 0.3)
    @test p0.heat_diag[3] > 0
    @test p8.heat_diag[3] < 0.1 * p0.heat_diag[3]
    @test p8.heat_diag[1] <= p0.heat_diag[1]
end

@testset "terms: diffused Q departs from the local Euler" begin
    kw = (; KW3..., GEOM_MODE="SIM", Nx_sc=8, Ny_sc=6,
          use_phi=true, phi_bc=:dirichlet, lambda_TF=2.0, tau_Q=0.5)
    function qmax(cfg)
        p = derive_params(cfg); g = build_geometry(cfg, p)
        s = initial_state(p, g, cfg.H_const, Strict())
        sp = SolverPhi(p; bc=cfg.phi_bc)
        for _ in 1:15
            substep_phi!(s, p, g, Strict(), sp; H=cfg.H_const, HJ=0.0, J_cur=0.2)
        end
        maximum(abs, sp.Qimb)
    end
    q_loc = qmax(Config(; kw..., use_Qss=false))
    q_dif = qmax(Config(; kw..., use_Qss=true, D_Q=1.0))
    @test q_loc > 0
    @test q_dif != q_loc
end

@testset "terms: the heat cap ends the ladder" begin
# J=3 ≫ depairing with large η_J: θ passes 0.45 at the first J > 0. The
# following J are not run.
    kw = (; KW3..., GEOM_MODE="SIM", t_min_GL=1.0, t_min_GL_res=1.0, iter_min=5,
          mMax=60, use_Vconv=true)
    Js = [0.0, 3.0, 2.5, 2.0]
    for calor in (true, false)
        cfg = Config(; kw..., use_heat=calor, eta_J=50.0, gamma_bath=1.0)
        p = derive_params(cfg); g = build_geometry(cfg, p)
        r = sweep(Js, fill(cfg.H_const, length(Js)), "J+", g, p, Strict())
        if calor
            @test r.cflag[2] == 'T'
            @test r.cflag[3] == ' ' && r.cflag[4] == ' '
        else
            @test all(!=(' '), r.cflag)          # without heat the whole ladder runs
            @test !any(==('T'), r.cflag)
        end
    end
end

@testset "terms: T_physical (T_red as bath temperature)" begin
# (1) at T = 0 it is the same step, to the bit (mask of 0 and 1)
    s0, _, _ = _pasos3(Config(; KW3...), 300, 0.9)
    s1, _, _ = _pasos3(Config(; KW3..., T_physical=true), 300, 0.9)
    @test s1.G1 == s0.G1 && s1.G2 == s0.G2 && s1.thetax == s0.thetax
    # (2) uniform equilibrium at T = 0.3, clean box without field:
    #     |Ψσ|² → 1 − T, |Ψπ|² → a2n − T (γ12 = 0.001 shifts it very little)
    kw = (; KW3..., GEOM_MODE="SIM", H_const=0.0, T_red=0.3)
    media(G, p) = sum(abs2, @view G[p.x1+1:p.x2, p.y1+1:p.y2, p.z1+1:p.z2]) /
                  ((p.x2 - p.x1) * (p.y2 - p.y1) * (p.z2 - p.z1))
    s, p, _ = _pasos3(Config(; kw..., T_physical=true), 4000, 0.0)
    @test isapprox(media(s.G1, p), 0.7; rtol=0.01)
    @test isapprox(media(s.G2, p), 0.59 - 0.3; rtol=0.02)
    # (3) without T_physical σ does not feel T_red: it stays at ~1
    s, p, _ = _pasos3(Config(; kw..., T_physical=false), 4000, 0.0)
    @test isapprox(media(s.G1, p), 1.0; rtol=0.01)
    # the π mass does not change with T under T_physical (ξπ/ξσ fixed at T = 0)
    @test derive_params(Config(; kw..., T_physical=true)).mr ≈ derive_params(Config(; KW3..., T_physical=true)).mr
end

@testset "convergence: eps_conv_rate and the :paper preset" begin
    kw = (; GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5,
          kappa_s=2.0, kappa_p=2.0, xi_ratio=1.30, gamma12=0.001, gamma_ratio=1.0,
          H_const=0.50, J_list=[0.0], t_min_GL=0.0, iter_min=1, mMax=200000,
          eps_conv_seed=1.0, tie_seed_tolerance=true,
          selftest=false, snap_every=0, out_dir="", verbose=false)
    # (1) off: the per-cycle tolerance is eps_conv itself, to the bit
    p0 = derive_params(Config(; kw..., eps_conv=1e-5))
    @test p0.eps_conv === 1e-5 && p0.eps_seed === 1e-5
    # (2) on: tolerance per cycle = rate * cycle length; eps_conv is ignored
    p1 = derive_params(Config(; kw..., eps_conv=1e-5, eps_conv_rate=1e-4))
    @test p1.eps_conv ≈ 1e-4 * p1.lMax * p1.dt
    @test_throws ErrorException derive_params(Config(; kw..., eps_conv_rate=-1.0))
    # (3) the point of it: the physical time at which the seed closes does not depend
    #     on the cycle length with a rate, and does with a per-cycle tolerance
    #     (measured: 8.51 vs 6.78 t_GL per cycle; 7.52 vs 7.42 t_GL with the rate)
    tcierre(; k...) = begin
        cfg = Config(; kw..., k...); p = derive_params(cfg); g = build_geometry(cfg, p)
        r = TDGL.sweep([0.0], [0.5], "J+", g, p, Strict())
        r.m_arr[1] * p.lMax * p.dt
    end
    a, b = tcierre(lMax=40, eps_conv_rate=1e-4), tcierre(lMax=10, eps_conv_rate=1e-4)
    @test abs(a - b) / a < 0.03
    c, d = tcierre(lMax=40, eps_conv=1e-5), tcierre(lMax=10, eps_conv=1e-5)
    @test abs(c - d) / c > 0.1
    # (4) the preset: the protocol of the methods paper, overridable, library defaults untouched
    cp = Config(:paper)
    @test cp.eps_conv == 1e-5 && cp.eps_conv_seed == 5e-4
    @test cp.t_min_GL == 1500.0 && cp.t_min_GL_res == 2000.0 && cp.use_Vconv
    @test Config(:paper; t_min_GL=400.0).t_min_GL == 400.0
    @test Config().eps_conv == 1e-4 && Config().eps_conv_rate == 0.0
    @test_throws ErrorException Config(:fast)
end
