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
