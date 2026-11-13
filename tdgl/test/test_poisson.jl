# ---------------------------------------------------------------------------
#  Scalar potential.  The Poisson solve is checked on its own before it is
#  coupled to the scheme: an equation is interrogated before it is
#  integrated, not after.
# ---------------------------------------------------------------------------
@testset "poisson: the seven-point Laplacian" begin
    A = assemble_laplacian(4, 3, 2, 1.0, 1.0, 1.0)
    @test size(A) == (24, 24)
    @test all(diag(A) .== -6.0)                    # 7 points in 3D
    @test A == A'                                  # symmetric
    @test all(sum(A, dims=2) .<= 1e-14)            # rows sum to <= 0 (Dirichlet)
end

@testset "poisson: second order against the exact solution" begin
    filas = poisson_convergence(mallas=[8, 16, 24, 32], io=devnull)
    ordenes = [f[5] for f in filas[2:end]]
    @test all(o -> 1.2 < o < 2.6, ordenes)         # tends to 2 under refinement
    @test ordenes[end] > ordenes[1]
    @test filas[end][3] < filas[1][3]/5
    @info "poisson: measured orders = $(round.(ordenes, digits=2))"
end

@testset "phi: links are rebuilt with exp(-i theta)" begin
# Not just notation: the kernel writes Ux = exp(-i phix) and then thetax = phix.
# The hard test is to restart from the checkpoint and require that 40 more
# substeps give THE SAME BIT as continuing from memory.  With exp(+i theta)
# this fails completely.
    sal = mktempdir()
    cfg = cfg_fixture(out_dir=sal)
    sde = run_case(cfg; mode=Strict())
    p, g = sde.p, sde.geom
    s_mem = sde.pos.estado
    s_ck, _, _, _ = load_checkpoint(checkpoint_path(joinpath(sal,"restart"), "Jpos"),
                                      p, Strict())
    @test all(s_ck.Vx .=== s_mem.Vx)
    @test maximum(abs, [cis(+t) for t in s_mem.thetax] .- s_mem.Vx) > 1.0
    for _ in 1:40
        substep!(s_mem, p, g, Strict(); H=0.5, HJ=0.0, J_cur=0.0)
        substep!(s_ck,  p, g, Strict(); H=0.5, HJ=0.0, J_cur=0.0)
    end
    @test all(s_mem.G1 .=== s_ck.G1)
    @test all(s_mem.hz .=== s_ck.hz)
end

@testset "phi: div(curl curl A) = 0 EXACTLY, and what follows" begin
# With the staggered operators of the kernel the identity is exact; a
# ratio like 0.993 is an artefact of mixing two different divergences.
    cfg = Config(:legacy; GEOM_MODE="WEDGE", Nx_sc=20, Ny_sc=14, Nz_sc=4,
                 dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, kappa_p=2.0,
                 xi_ratio=1.30, gamma12=0.001, gamma_ratio=1.3, H_const=0.50,
                 J_list=[0.10], delta_v=6, selftest=false, out_dir="",
                 verbose=false)
    p = derive_params(cfg); g = build_geometry(cfg, p)
    ruta = joinpath(RAIZ, "data", "transport_20x14x4", "restart", "state_Jpos.mat")
    s, _, _, _ = load_checkpoint(ruta, p, Strict())
    F = phi_fields(s, p, g)
    nrm(x) = sqrt(sum(abs2, x))

    # (a) the discrete identity
    escala = max(nrm(F.Rx), nrm(F.Ry), nrm(F.Rz))/p.dx
    @test maximum(abs, F.div_rot) < 1e-15
    @test nrm(F.div_rot)/escala < 1e-14
    @test !any(isnan, F.Rx) && !any(isnan, F.Ry) && !any(isnan, F.Rz)

    # (b) div(dA/dt) IS div(J_s), not "close"
    @test nrm(F.div_dAdt .- F.div_Js)/nrm(F.div_Js) < 1e-14

    # (c) d/dt(div A) = div J_s  =>  lap Phi = 0 in the Phi = 0 gauge
    HJ = p.inject_gain*0.10*p.slab_d/(2*p.ks2)
    dd, _, _ = drift_div_A(s, p, g, Strict(); H=0.50, HJ=HJ, J_cur=0.10)
    @test abs(nrm(dd)/nrm(F.div_Js) - 1) < 1e-10
    @info "phi: ||d/dt(div A)|| / ||div J_s|| = $(nrm(dd)/nrm(F.div_Js))"

    # (d) the Coulomb projection does give a bounded, non-trivial Phi
    P, res, _ = solve_poisson(F.div_Js, p.dx)
    @test maximum(abs, res) < 1e-12*max(nrm(F.div_Js), 1.0)
    @test all(isfinite, P)
    @test 0 < maximum(abs, P) < 1e3
end

@testset "phi: the superseded first form is kept and flagged" begin
    w = read_state(joinpath(FIXT, "restart", "state_Jpos.mat"))
    dAdt, divf = vacuum_source(Float64.(w["hy"]), Float64.(w["hz"]), 0.5, 4.0)
    P, res, _ = solve_poisson(-divf, 0.5)
    @test all(isfinite, P)
    @test maximum(abs, P) < 1e3
end
