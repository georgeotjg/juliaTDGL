# Threshold extraction and the diode efficiency, in the convention of the paper:
# eta = (Ic+ - |Ic-|) / (Ic+ + |Ic-|), in per cent.
using Test, TDGL

@testset "observables: efficiency and extract_Ic" begin
    @test TDGL.efficiency(0.0425, -0.1175) ≈ 100 * (0.0425 - 0.1175) / 0.16
    @test TDGL.efficiency(0.0425, 0.1175) ≈ -46.875
    @test TDGL.efficiency(0.1, 0.1) == 0
    @test isnan(TDGL.efficiency(NaN, 0.1))
    J = [0.0, 0.01, 0.02, 0.03, 0.04]
    V = [1e-9, 2e-9, 5e-5, 3e-3, 8e-3]
    @test TDGL.extract_Ic(J, V, 1e-4) ≈ 0.02 + 0.01 * (1e-4 - (5e-5 - 1e-9)) / ((3e-3 - 1e-9) - (5e-5 - 1e-9))
    Vone = [1e-9, 2e-9, 3e-3, 5e-9, 4e-9]          # one isolated rung above threshold
    @test TDGL.extract_Ic(J, Vone, 1e-4) == 0.02
    @test isnan(TDGL.extract_Ic(J, Vone, 1e-4; strict=true))

    # the three guards of the reference code
    Jp = [0.0, 0.02, 0.04, 0.06]; Jn = -Jp
    Vp = [1e-9, 1e-9, 2e-2, 5e-2]; Vn = [1e-9, -1e-9, -1e-9, -3e-2]
    g = ic_guards(Jp, Vp, Jn, Vn, 10, 0.03, 0.05)              # a diode: nothing fires
    @test !g.inert && !g.no_vortex && !g.grid && g.Icp == 0.03 && g.Icn == 0.05
    g = ic_guards(Jp, Vp, Jn, Vp, 10, 0.03, 0.03)              # V even in J: inert drive
    @test g.inert && isnan(g.Icp) && isnan(g.Icn)
    g = ic_guards(Jp, Vp, Jn, Vn, 0, 0.03, 0.05)               # no vortex anywhere
    @test g.no_vortex && isnan(g.Icp)
    g = ic_guards(Jp, Vp, Jn, Vn, 10, 0.015, 0.05)             # Ic below the first grid point
    @test g.grid && isnan(g.Icp) && isnan(g.Icn)
    @test !ic_guards(Jp, zero(Vp), Jn, zero(Vn), 10, NaN, NaN).inert   # H = 0: no voltage at all

    # brackets with the verdict rule: the production ladder of the paper (floors 1500/2000)
    Jp = [0.0, 0.015, 0.030, 0.035, 0.040, 0.045, 0.050]
    Vp = [5e-8, 5.8e-7, 2.1e-5, 1.2e-6, 7.5e-8, 3.012e-2, 6.107e-2]
    tp = [1500, 1500, 1500, 1500, 1500, 4000, 4000.0]          # 0.045 re-measured on a doubled window
    bp = bracket_Ic(Jp, Vp, tp)
    @test bp.lo == 0.040 && bp.hi == 0.045 && bp.ok
    Jn = -[0.0, 0.100, 0.115, 0.120]; Vn = [5e-8, 1.1e-9, 1.2e-10, -5.063e-2]
    bn = bracket_Ic(Jn, Vn, fill(2000.0, 4))
    @test bn.lo == 0.115 && bn.hi == 0.120
    e = eta_bracket(bp, bn)
    @test round(e[1], digits=1) == -46.9 && round(e[2], digits=1) == -50.0 && round(e[3], digits=1) == -43.8
    bu = bracket_Ic(Jp, Vp, [1500, 1500, 1500, 1500, 1500, 2000, 2000.0])  # 0.045 at 9.6 crossings: undecided
    @test bu.undecided == [0.045] && !bu.ok && bu.hi == 0.050
    @test isnan(bracket_Ic([0.0, 0.05], [0.0, 1.0], [0.0, 2000.0]).lo)   # dissipating at the first rung
end
