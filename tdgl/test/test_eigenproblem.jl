# ---------------------------------------------------------------------------
#  EXTERNAL VALIDATION.  The rest of the suite checks the code against the
#  reference implementation or against itself; this checks it against exact
#  answers that exist OUTSIDE the project.
# ---------------------------------------------------------------------------
@testset "eigenproblem: the Landau level against 1 - phi/8" begin
    filas = step1_one_band(io=devnull)
    @test length(filas) == 7
    for f in filas
        # the operator must be EXACTLY Hermitian; otherwise the link convention
        # of the kernel is not the one assumed
        @test f.herm == 0.0
        # and lambda_0/H must follow the continuum law, an EXTERNAL result:
        # 1 - phi/8 + O(phi^2)
        @test abs(f.ratio - f.ley) < 1e-3
    end
    # at the PRODUCTION mesh and field (dx=0.5, H=0.50) => phi = 0.125,
    # i.e. 1.6 % discretisation error in the Landau level.
    H, lam, r, _ = landau_level(40, 40, 0.5, 0.5, 32)
    @test H ≈ 0.5027 atol=1e-3
    @test 0.980 < r < 0.990
    @info "eigenproblem: at H~0.50, lambda_0/H = $(round(r,digits=5)) => $(round(100*(1-r),digits=2)) % mesh error"
end

@testset "eigenproblem: two-band Hc2, and F21" begin
    a2n = 0.59; mr = 1.0029; g12 = 1e-3
    Hp = hc2_block(a2n, mr, g12)
    @test Hp ≈ 1.0 atol=1e-4                    # the two-band Hc2 formula gives Hc2 = 1

    # faulty scheme (F21): Gamma_r ONLY in the condensation => Hc2 DEPENDS on
    # Gamma_r.
    for (gr, esp) in ((0.25, 2.367), (0.50, 1.183), (1.00, 1.000),
                      (1.30, 1.000), (2.00, 1.000), (4.00, 1.000))
        @test hc2_block(a2n, mr, g12; gr=gr) ≈ esp atol=2e-3
    end

    # this scheme: Gamma_r in the THREE terms => Hc2 does NOT depend on
    # Gamma_r and equals the two-band formula EXACTLY.  Not a numerical
    # coincidence: det M = (1/Gr)*[the two-band expression].
    for gr in (0.25, 0.5, 1.0, 1.3, 2.0, 4.0, 10.0)
        @test hc2_scheme(a2n, mr, g12, gr) ≈ Hp rtol=1e-6
    end
    # and at strong coupling too
    for gr in (0.25, 1.0, 4.0)
        @test hc2_scheme(a2n, mr, 0.1, gr) ≈ hc2_block(a2n, mr, 0.1) rtol=1e-6
    end

    # The kinetic scale parameter of the Octave eigen script is INVERTED: `ek`
    # divides, so multiplying the kinetic term by 1/gr requires ek = gr.  With
    # ek = 1/gr the "INTENDED" column does not represent the intended model.
    malo  = hc2_block(a2n, mr, g12; gr=0.25, esc_cinetico=1/0.25, esc_acoplo=1/sqrt(0.25))
    bueno = hc2_block(a2n, mr, g12; gr=0.25, esc_cinetico=0.25,   esc_acoplo=1/sqrt(0.25))
    @test malo > 9.0                             # 9.47: not the Hc2 of anything
    @test bueno ≈ Hp rtol=1e-6                   # corrected, it is the formula
end

@testset "eigenproblem: Saint-James--de Gennes, Hc3/Hc2 = 1.69461" begin
# The OTHER external number.  It depends on no material parameter, so the
# deviation is mesh and boundary error only.  The Neumann condition measured
# is the one of block [H] of the kernel, not an idealisation.
    filas = step3_boundary(dxs=[0.5, 0.25, 0.125], io=devnull)
    exacto = 1.69461
    err = [abs(f.ratio - exacto)/exacto for f in filas]
    @test issorted(err, rev=true)                # converges under refinement
    @test err[end] < 0.01                        # <1 % at dx = 0.125
    r, lam, x0 = hc3_over_hc2(0.0625)
    @test abs(r - exacto)/exacto < 2e-3          # 0.05 % at dx = 0.0625
    @test lam/1.0 ≈ 0.59010 atol=2e-3            # lambda/H at the minimum
    @info "eigenproblem: Hc3/Hc2 = $(round(r,digits=5)) against the exact $exacto ($(round(100*(r-exacto)/exacto,digits=2)) %)"
end
