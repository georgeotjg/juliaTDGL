# ---------------------------------------------------------------------------
#  Derived parameters, and above all what Gamma_r actually does.
# ---------------------------------------------------------------------------
@testset "params: CFL and the meaning of Gamma_r" begin
    p = derive_params(cfg_fixture())
    @test p.Nx == 14 && p.Ny == 10 && p.Nz == 6          # buf = 1
    @test (p.x1, p.x2, p.y1, p.y2, p.z1, p.z2) == (2, 13, 2, 9, 2, 5)
    @test p.dt ≈ 4.95e-3 rtol=1e-12
    @test p.dt_limitante == "dt3"
    @test p.c1 == p.dt
    @test p.c2 ≈ p.dt/1.3 rtol=1e-15

    # xi_ratio sets m_ratio, not the other way round
    c = cfg_fixture(); p2 = derive_params(c)
    @test c.m_ratio ≈ 1/(1.30^2 * 0.59) rtol=1e-12
    @test p2.xi_ratio_real ≈ 1.30 rtol=1e-9

    # [G6] In THIS scheme Gamma_r divides the THREE terms of the pi band, so it
    # is a true relaxation rate and moves nothing static.
    # Hc2_pi = m_r*a2n = 0.5917 for every Gamma_r.
    for gr in (0.25, 0.5, 1.0, 1.3, 2.0, 4.0)
        pg = derive_params(cfg_fixture(gamma_ratio=gr, xi_ratio=1.30))
        @test hc2_pi(pg) ≈ 0.5917 atol=1e-3
        @test xi_ratio_integrated(pg) ≈ 1.300 atol=1e-3
        # and the v13.2 law, needed to read older archived runs: there the
        # same sweep moved the coherence length.
        @test hc2_pi_f21(pg) ≈ 0.5917/gr rtol=2e-3
        @test xi_ratio_f21(pg) ≈ 1.300*sqrt(gr) rtol=2e-3
    end

    # The current grid always includes 0: the baseline V(J=0) that corrects
    # the voltmeter, i.e. the noise floor of the run itself.
    @test 0.0 in current_grid(cfg_fixture(J_list=[0.5, 1.0]))
    @test current_grid(cfg_fixture(J_list=[1.0, 0.5, 1.0])) == [0.0, 0.5, 1.0]
end

@testset "cfl_legacy: the inherited step bounds (gama factor, dt3) can be dropped" begin
    kw = (; GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0,
          kappa_p=2.0, xi_ratio=1.30, gamma12=0.001, gamma_ratio=1.0, H_const=0.5, J_list=[0.0],
          selftest=false, snap_every=0, out_dir="", verbose=false)
    dl = 2 / (3 / 0.5^2)
    p0 = derive_params(Config(; kw...))
    @test Config().cfl_legacy && p0.dt == 0.99 * 0.03 * dl          # dt3 binds at R = 1.3, kappa = 2
    p1 = derive_params(Config(; kw..., cfl_legacy=false))
    @test p1.dt ≈ 0.99 * dl / (4 * 2.0^2)                            # explicit Maxwell binds
    # without the explicit Maxwell bound: diffusion plus reaction of sigma or pi
    s1(p) = 2 / (8 / dl + 2)
    s2(p) = 2 * 1.0 / (8 / (dl * p.mr) + 2 * p.alpha2_n)
    p2 = derive_params(Config(; kw..., cfl_legacy=false, maxwell_implicit=true))
    @test p2.dt ≈ 0.99 * min(s1(p2), s2(p2))
    @test p2.dt < 0.99 * dl / 4 * min(1.0, p2.mr)                    # below the bound of pure diffusion
    p3 = derive_params(Config(; kw..., xi_ratio=4.0, cfl_legacy=false))
    @test p3.dt ≈ 0.99 * s2(p3)                                      # the light pi band binds
end
