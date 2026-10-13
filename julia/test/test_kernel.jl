# ---------------------------------------------------------------------------
#  The kernel.  Block [A] is checked against psi_update_loop, the slow loop
#  that is the canonical definition of the scheme, not the fast version.
# ---------------------------------------------------------------------------
@testset "kernel [A]: production vs reference loop" begin
    cfg = cfg_fixture(); p = derive_params(cfg); g = build_geometry(cfg, p)
    for mode in (Strict(), Fast())
        peor = run_selftest(g, p, mode)
        @test peor == 0.0            # not "< 1e-12": the same scheme gives the same bit
    end
end

@testset "kernel: pi-band coefficients" begin
    p = derive_params(cfg_fixture())
    kx, ky, kz, cpl1, cpl2 = coef_pi(p)
    gr = p.dt/p.c2
    @test kx === p.rx/(p.mr*gr)
    @test cpl1 === p.dt*p.gamma12
    # `dt/(dt/c2)` is NOT `c2` to the bit; the reference recovers Gamma_r this
    # way on purpose and the code must carry the same rounding.
    @test cpl2 === (p.dt/gr)*p.gamma12
end

@testset "drag: π cross term without an extra 1/mr (reciprocity)" begin
# Energy ν Re[(Dσ)*(Dπ)]. With Gσ=Gπ, Δπ·Γr = Δσ.
# ν·kx2 (= ν·r/(mr Γr)) would give Δπ·Γr = Δσ/mr. At ξ_r=1.30 mr≈1 and it
# would not show; ξ_r=4 ⇒ mr~0.1. Γ_r=1.3 also catches a missing /Γr.
    kw = (GEOM_MODE="SIM", Nx_sc=8, Ny_sc=6, Nz_sc=4, buf=1,
          dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, xi_ratio=4.0,
          gamma12=0.001, gamma_ratio=1.3, H_const=0.03,
          J_list=[0.0], t_min_GL=0.0, mMax=2, selftest=false,
          snap_every=0, vtk_every=0, verbose=false, use_threads=false,
          out_dir="")
    p0 = derive_params(Config(; use_drag=false, nu_drag=0.0, kw...))
    p1 = derive_params(Config(; use_drag=true,  nu_drag=0.05, kw...))
    g  = build_geometry(Config(; use_drag=false, nu_drag=0.0, kw...), p0)
    @test p0.mr < 0.2
    Nx, Ny, Nz = p0.Nx, p0.Ny, p0.Nz
    ii = Float64.(reshape(0:Nx, :, 1, 1))
    jj = Float64.(reshape(0:Ny, 1, :, 1))
    kk = Float64.(reshape(0:Nz, 1, 1, :))
    pert = @. 0.10*sin(0.37*ii + 0.11)*cos(0.23*jj - 0.05)*sin(0.51*kk + 0.29)
    G  = ComplexF64.(@. g.taupsi*(1.0 + pert)*cis(0.20*sin(0.13*ii - 0.31*jj)))
    Vx = ones(ComplexF64, Nx, Ny+1, Nz+1)
    Vy = ones(ComplexF64, Nx+1, Ny, Nz+1)
    Vz = ones(ComplexF64, Nx+1, Ny+1, Nz)
    F1a = copy(G); F2a = copy(G)
    F1b = copy(G); F2b = copy(G)
    psi_update!(F1a, F2a, G, G, Vx, Vy, Vz, g.fmask, p0, Strict())
    psi_update!(F1b, F2b, G, G, Vx, Vy, Vz, g.fmask, p1, Strict())
    d1 = F1b .- F1a
    d2 = F2b .- F2a
    gr = p0.dt / p0.c2
    rec = maximum(abs, d2 .* gr .- d1)
    @test maximum(abs, d1) > 1e-14
    @test rec < 1e-12
    # Reverting to ν·kx2 would give rec ≈ |1 − 1/mr|·|d1| ~ 9·|d1|.
    @test rec < 0.01 * maximum(abs, d1)
end

@testset "kernel: the reference swap in [I] is legitimate" begin
# bloque_I! swaps theta<->phi and V<->U instead of copying.  That is only
# valid if the entries no block writes are constant.  Checked, not assumed.
    cfg = cfg_fixture(); p = derive_params(cfg); g = build_geometry(cfg, p)
    s = initial_state(p, g, 0.5, Strict())
    borde_x = [(i,j,k) for i in 1:p.Nx, j in 1:p.Ny+1, k in 1:p.Nz+1
               if !(2 <= j <= p.Ny && 2 <= k <= p.Nz)]
    ini_phi = [s.phix[i,j,k] for (i,j,k) in borde_x]
    ini_ux  = [s.Ux[i,j,k]   for (i,j,k) in borde_x]
    for _ in 1:200
        substep!(s, p, g, Strict(); H=0.5, HJ=0.0, J_cur=0.0)
    end
    @test [s.thetax[i,j,k] for (i,j,k) in borde_x] == ini_phi
    @test [s.Vx[i,j,k] for (i,j,k) in borde_x] == ini_ux
end

@testset "kernel: threads do not change a single bit" begin
# Each z plane writes distinct cells and reads only G (the previous step),
# so the split among threads cannot alter the result.  A failure here would
# mean a wrong parallelisation, not "noise".
    cfg1 = cfg_fixture(use_threads=false); p1 = derive_params(cfg1)
    g1 = build_geometry(cfg1, p1)
    cfg2 = cfg_fixture(use_threads=true);  p2 = derive_params(cfg2)
    g2 = build_geometry(cfg2, p2)
    s1 = initial_state(p1, g1, 0.5, Strict())
    s2 = initial_state(p2, g2, 0.5, Strict())
    for _ in 1:400
        substep!(s1, p1, g1, Strict(); H=0.5, HJ=0.0, J_cur=0.0)
        substep!(s2, p2, g2, Strict(); H=0.5, HJ=0.0, J_cur=0.0)
    end
    @test all(s1.G1 .=== s2.G1)
    @test all(s1.G2 .=== s2.G2)
    @test all(s1.hz .=== s2.hz)
end

@testset "guards: divergence and wall-clock budget" begin
# (a) The divergence guard must FIRE.  A real divergence is provoked: a dt
# legal for kappa=2 is used at kappa=25, the most common failure (raising
# kappa without recomputing dt: dt2s = dl/(4 kappa^2)).
    cfg = Config(GEOM_MODE="SIM", Nx_sc=8, Ny_sc=6, Nz_sc=4,
                 dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, kappa_p=2.0,
                 H_const=0.5, J_list=[0.0], t_min_GL=0.0, iter_min=5,
                 mMax=500, delta_v=1, selftest=false, out_dir="", verbose=false)
    p2 = derive_params(cfg)
    # same dt but kappa=25 => the Maxwell term is 156x larger
    malo = Config(GEOM_MODE="SIM", Nx_sc=8, Ny_sc=6, Nz_sc=4,
                  dx=0.5, dy=0.5, dz=0.5, kappa_s=25.0, kappa_p=2.0,
                  H_const=0.5, J_list=[0.0], t_min_GL=0.0, iter_min=5,
                  mMax=500, delta_v=1, selftest=false, out_dir="", verbose=false)
    # force the kappa=2 dt on the kappa=25 problem by rebuilding Params
    pm = derive_params(malo)
    campos = [f === :dt   ? p2.dt   :
              f === :rx   ? p2.rx   : f === :ry ? p2.ry : f === :rz ? p2.rz :
              f === :c1   ? p2.c1   : f === :c2 ? p2.c2 : f === :d1 ? p2.d1 :
              f === :sxy  ? p2.sxy*(pm.ks2/p2.ks2) :
              f === :sxz  ? p2.sxz*(pm.ks2/p2.ks2) :
              f === :syx  ? p2.syx*(pm.ks2/p2.ks2) :
              f === :syz  ? p2.syz*(pm.ks2/p2.ks2) :
              f === :szx  ? p2.szx*(pm.ks2/p2.ks2) :
              f === :szy  ? p2.szy*(pm.ks2/p2.ks2) :
              getfield(pm, f) for f in fieldnames(Params)]
    pmal = Params(campos...)
    g = build_geometry(malo, pmal)
    @test_throws ErrorException sweep([0.0], [0.5], "J+", g, pmal, Strict())

    # (b) Without the guard a diverged run must not be reported as converged.
    # Since `NaN > x` is FALSE, a residual computed with comparisons stays at
    # ZERO on a NaN state, the fixed-point criterion is met at the iteration
    # floor and the point is reported with conv='F' and residual 0.  The
    # residual uses `max`, which propagates NaN.
    sin_v = [f === :guard_psi ? 0.0 : getfield(pmal, f) for f in fieldnames(Params)]
    r = sweep([0.0], [0.5], "J+", g, Params(sin_v...), Strict())
    @test !all(isfinite, r.estado.G1)     # the state is NaN...
    @test isnan(r.r1_arr[1])              # ...the residual SAYS so
    @test r.cflag[1] == '*'               # and it is NOT declared converged

    # (c) The wall-clock budget stops the run and leaves the checkpoint written.
    sal = mktempdir()
    cbud = cfg_fixture(out_dir=sal, guard_hours=1e-9, J_list=[0.0, 0.05, 0.10],
                       mMax=40, iter_min=40)
    sde = run_case(cbud; mode=Fast())
    @test isfile(checkpoint_path(joinpath(sal, "restart"), "Jpos"))
    @test count(!=(' '), sde.pos.cflag) < 3     # did not reach the last point
end
