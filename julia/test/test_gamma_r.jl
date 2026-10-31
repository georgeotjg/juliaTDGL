# ---------------------------------------------------------------------------
#  Gamma_r is a relaxation rate: it must not move the fixed point (mode F21).
#
#  The same state is integrated at three Gamma_r and the fixed point, a
#  physical quantity, is compared.
#
#      this scheme: Gamma_r divides the THREE pi-band terms
#          => relaxation rate => the three values are EQUAL
#      the faulty scheme (F21): Gamma_r divides only the condensation
#          => Hc2_pi = 0.5917/Gamma_r => at H = 0.90 one gets
#             <|Psi_pi|^2> = 0.4008 / 0.0774 / 0.0000 for Gr = 0.25 / 1 / 4
#
#  The faulty scheme fails here by three orders of magnitude, not by an ulp.
#  This test and the acceptance test cover each other: that one detects any
#  deviation but only in the bank configuration; this one detects ONE
#  specific fault in any configuration.
# ---------------------------------------------------------------------------
@testset "Gamma_r does not move the statics (F21)" begin
    amp = Float64[]
    for gr in (0.25, 1.0, 4.0)
        cfg = Config(:legacy; GEOM_MODE="SIM", Nx_sc=16, Ny_sc=12, Nz_sc=4,
                     dx=0.5, dy=0.5, dz=0.5, kappa_s=2.0, kappa_p=2.0,
                     xi_ratio=1.30, gamma12=0.001, gamma_ratio=gr,
                     H_const=0.90, J_list=[0.0], t_min_GL=0.0, mMax=4000,
                     iter_min=40, eps_conv=1e-14, eps_conv_seed=1e-7,
                     delta_v=2, selftest=false, out_dir="", verbose=false)
        push!(amp, run_case(cfg; mode=Fast()).pos.ps2_arr[1])
    end
    @info "Gamma_r: <|Psi_pi|^2> at Gamma_r = 0.25/1/4 -> $(round.(amp, digits=4))"
    # the three agree to 1 %.  With the faulty scheme the third would be 0.
    @test maximum(amp) - minimum(amp) < 0.01*maximum(amp)
    @test all(a -> a > 0.1, amp)
end
