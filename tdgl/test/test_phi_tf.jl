# ---------------------------------------------------------------------------
#  Thomas-Fermi screening and charge imbalance.  The gauge-invariance
#  controls of test_phi.jl do NOT apply here: this is new physics.
#  lambda_TF=0 and tau_Q=0 (the default) is the plain Coulomb projection and
#  those tests are unchanged.
# ---------------------------------------------------------------------------

base_tf(; kw...) = Config(:legacy; GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4,
    dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, kappa_p=2.0, xi_ratio=1.30,
    gamma12=0.001, gamma_ratio=1.0, H_const=0.50, J_list=[0.08], delta_v=2,
    t_min_GL=0.0, t_min_GL_res=0.0, iter_min=40, mMax=40, use_Vconv=false,
    eps_conv=1e-14, eps_conv_seed=1e-14, selftest=false, out_dir="",
    verbose=false, use_phi=true, lambda_TF=2.0, kw...)

function _subpaso_chequeo!(s, p, g, mode, sp; H=0.5)
    psi_update!(s.F1, s.F2, s.G1, s.G2, s.Vx, s.Vy, s.Vz, g.fmask, p, mode)
    TDGL.block_B!(s, p, mode)
    TDGL.bloque_C!(s, p, g)
    TDGL.block_C2!(s, p, sp)
    r = residual_div_increment(s, p)
    peor = 0.0
    @inbounds for k in 2:p.Nz, j in 2:p.Ny, i in 2:p.Nx
        # Q does NOT enter the Poisson problem: putting it there makes an
        # unstable Q->Phi->Q loop (Q runs to 1e18).  The TF identity is always
        # D(delta') + k2 Phi = 0; Q only follows the residual with a delay.
        peor = max(peor, abs(r[i,j,k] + sp.k2 * sp.Phi[i,j,k]))
    end
    TDGL.apply_phi_psi!(s, p, sp, mode)
    p.selffield || TDGL.bloque_D!(s, p, g, 0.0)
    TDGL.bloque_E!(s, p, mode)
    TDGL.bloque_F!(s, p)
    TDGL.bloque_G!(s, p, H, 0.0)
    TDGL.bloque_H!(s, p, mode)
    TDGL.bloque_I!(s, p)
    peor
end

@testset "TF: lambda_TF=0 and tau_Q=0 change nothing" begin
    p0 = derive_params(cfg_fixture(use_phi=true))
    @test p0.lambda_TF == 0.0 && p0.tau_Q == 0.0
    sp = SolverPhi(p0)
    @test sp.k2 == 0.0 && sp.tau_Q == 0.0
    @test all(iszero, sp.Qimb)
end

@testset "TF: D(delta') + Phi/lambda_TF^2 = 0 (the control that can fail)" begin
    cfg = base_tf(J_list=[0.0], mMax=2, iter_min=1)
    p = derive_params(cfg); g = build_geometry(cfg, p)
    @test p.lambda_TF == 2.0
    @test abs(p.lambda_TF^(-2) - (1/4)) < 1e-15
    sp = SolverPhi(p)
    @test sp.k2 ≈ 0.25
    s = initial_state(p, g, 0.5, Strict())
    peor = 0.0
    for _ in 1:80
        peor = max(peor, _subpaso_chequeo!(s, p, g, Strict(), sp))
    end
    # theta accumulates O(1); 1e-10 IS machine precision on the residual
    @test peor < 1e-9
    @test all(isfinite, s.G1)
end

@testset "TF: the separable solve IS the LU, also with mass" begin
    p = derive_params(base_tf())
    for bc in (:dirichlet, :neumann)
        a = SolverPhi(p; bc=bc, metodo=:lu)
        b = SolverPhi(p; bc=bc, metodo=:separable)
        r = randn(a.n1, a.n2, a.n3)
        a.rhs .= r; solve!(a)
        b.rhs .= r; solve!(b)
        # with TF Neumann is no longer singular: compared DIRECTLY, mean not removed
        @test maximum(abs, a.rhs .- b.rhs) < 1e-10*max(maximum(abs, a.rhs), 1.0)
    end
end

@testset "TF: TF and Coulomb are not the same state" begin
# The mass term is not a different gauge: it is another equation.  The run
# with lambda_TF>0 need NOT reproduce |Psi| of the Coulomb run, and
# max|div A| is no longer zero to machine precision.
    c0 = run_case(base_tf(lambda_TF=0.0); mode=Strict())
    c1 = run_case(base_tf(lambda_TF=2.0); mode=Strict())
    dpsi = maximum(abs, abs.(c0.pos.estado.G1) .- abs.(c1.pos.estado.G1))
    dA0 = maximum(abs, view(link_divergence(c0.pos.estado, c0.p),
                            3:c0.p.Nx-1, 3:c0.p.Ny-1, 3:c0.p.Nz-1))
    dA1 = maximum(abs, view(link_divergence(c1.pos.estado, c1.p),
                            3:c1.p.Nx-1, 3:c1.p.Ny-1, 3:c1.p.Nz-1))
    @test dA0 < 1e-10
    @test dA1 > 10 * dA0          # TF leaves a residual; Coulomb does not
    # |Psi| does NOT move: subtracting ∇Φ is still a gradient, with or without
    # mass.  TF breaks the Coulomb condition (div A) and the freedom of a
    # constant in Φ, not |Psi|.  Measured: dpsi ~ 1e-15.
    @test dpsi < 1e-12
    @test all(isfinite, c1.pos.estado.G1)
    @info "TF vs Coulomb: max|Δ|Ψ|| = $dpsi  max|div A| = $dA0 -> $dA1"
end

@testset "TF: Dirichlet vs Neumann, still a gradient?" begin
# Subtracting ∇Φ and rotating by Φ leaves |Psi| and B invariant for ANY
# scalar Φ.  The mass term changes WHICH Phi comes out, not the fact that
# it is a gradient.  If |Psi| agrees, what TF "breaks" is that Phi acquires
# a scale (V_Phi, null mode), not that Dirichlet and Neumann give different
# |Psi|.
    d = run_case(base_tf(phi_bc=:dirichlet); mode=Strict())
    n = run_case(base_tf(phi_bc=:neumann); mode=Strict())
    dpsi = maximum(abs, abs.(d.pos.estado.G1) .- abs.(n.pos.estado.G1))
    dV  = abs(d.pos.VPhi_arr[end] - n.pos.VPhi_arr[end])
    @test all(isfinite, d.pos.estado.G1) && all(isfinite, n.pos.estado.G1)
    @test dpsi < 1e-12            # still a gradient: |Psi| invariant
    @test dV > 0.0                # V_Phi DOES depend on the BC, with or without TF
    @info "TF D vs N: max|Δ|Ψ_σ|| = $dpsi  |Δ V_Phi| = $dV"
end

@testset "tau_Q: without TF it does nothing (no source under Coulomb)" begin
    cfg = base_tf(lambda_TF=0.0, tau_Q=5.0, J_list=[0.0], mMax=2, iter_min=1)
    p = derive_params(cfg); g = build_geometry(cfg, p)
    sp = SolverPhi(p)
    s = initial_state(p, g, 0.5, Strict())
    for _ in 1:60
        _subpaso_chequeo!(s, p, g, Strict(), sp)
    end
    @test maximum(abs, sp.Qimb) < 1e-14
end

@testset "tau_Q: with TF it gives finite Q and mu != Phi" begin
    cfg = base_tf(lambda_TF=2.0, tau_Q=5.0, J_list=[0.0], mMax=2, iter_min=1)
    p = derive_params(cfg); g = build_geometry(cfg, p)
    sp = SolverPhi(p)
    s = initial_state(p, g, 0.5, Strict())
    peor = 0.0
    for _ in 1:80
        peor = max(peor, _subpaso_chequeo!(s, p, g, Strict(), sp))
    end
    @test peor < 1e-9
    @test maximum(abs, sp.Qimb) > 1e-12
    @test all(isfinite, s.G1)
    @info "tau_Q: max|Q| = $(maximum(abs, sp.Qimb))  identity = $peor"
end

@testset "tau_Q: Euler step for Q, attractor = residual (not tau_Q/dt)" begin
    # Isolated map of tau_Q dQ/dt + Q = sigma.
    σ, dt, τ = 1e-4, 2.5e-3, 5.0
    Q = 0.0
    n = round(Int, 20τ / dt)
    for _ in 1:n
        Q = (Q + (dt/τ)*σ) / (1 + dt/τ)
    end
    @test Q ≈ σ rtol=1e-8
    Qold = 0.0
    for _ in 1:n
        Qold = (Qold + σ) / (1 + dt/τ)
    end
    @test Qold ≈ (τ/dt)*σ rtol=1e-6

    # Code path: a mesh/H at which a wrong Euler step collapses by
    # t ~ 1.5 t_GL.  4 t_GL is enough to catch it.
    cfg = Config(:legacy; GEOM_MODE="SIM", Nx_sc=12, Ny_sc=8, Nz_sc=4,
        dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, kappa_p=2.0,
        xi_ratio=4.0, gamma12=0.001, gamma_ratio=1.0, H_const=0.030,
        J_list=[0.0], delta_v=2, t_min_GL=0.0, iter_min=1, mMax=2,
        selftest=false, out_dir="", verbose=false,
        use_phi=true, phi_bc=:neumann, lambda_TF=2.0, tau_Q=5.0)
    p = derive_params(cfg); g = build_geometry(cfg, p)
    s = initial_state(p, g, 0.030, Strict())
    sp = SolverPhi(p; bc=:neumann)
    nstep = ceil(Int, 4.0 / p.dt)
    for _ in 1:nstep
        substep_phi!(s, p, g, Strict(), sp; H=0.030, HJ=0.0, J_cur=0.0)
    end
    @test maximum(abs, s.G1) > 0.5
    ps2 = 0.0; nn = 0
    @inbounds for x in s.G2
        ps2 += abs2(x); nn += 1
    end
    @test ps2/nn > 0.05
    r = residual_div_increment(s, p)
    mxQ = 0.0; mxS = 0.0
    @inbounds for k in 2:p.Nz, j in 2:p.Ny, i in 2:p.Nx
        mxQ = max(mxQ, abs(sp.Qimb[i,j,k]))
        mxS = max(mxS, abs(r[i,j,k]))
    end
    # A wrong Euler step leaves max|Q|~0.05 with sigma ~ 0 at J=0 (ratio ~ 1e14).
    @test mxQ < 50 * max(mxS, 1e-16)
    @info "tau_Q Euler: max|Ps|=$(maximum(abs, s.G1))  max|Q|=$mxQ  max|sig|=$mxS"
end

@testset "TF: Config rejects illegal combinations" begin
    @test_throws ErrorException derive_params(cfg_fixture(lambda_TF=1.0))
    @test_throws ErrorException derive_params(cfg_fixture(tau_Q=1.0))
    @test_throws ErrorException derive_params(cfg_fixture(use_phi=true, lambda_TF=-0.1))
    @test_throws ErrorException derive_params(cfg_fixture(use_phi=true, tau_Q=-0.1))
end
