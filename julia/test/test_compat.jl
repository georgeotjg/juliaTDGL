# ---------------------------------------------------------------------------
#  The layer that makes bit equality possible.  These tests do not compare
#  against Octave (Octave is not required to run the tests): they compare the
#  two formulas with each other and check that the machine can produce the
#  one that reproduces the reference.  scripts/acceptance.jl gives the verdict
#  against the stored reference states.
# ---------------------------------------------------------------------------
@testset "compat: libm primitives" begin
    st = Strict(); ft = Fast()

    # glibc hypot == exact |z| in trivial cases
    @test TDGL.cabs(st, ComplexF64(3.0, 4.0)) == 5.0
    @test TDGL.cabs(st, ComplexF64(-2.5, 0.0)) == 2.5
    @test TDGL.cabs2(st, ComplexF64(3.0, 4.0)) == 25.0

    # |z|^2 via hypot is NOT re^2+im^2 to the bit; that difference is why the
    # Strict mode exists, so the test checks that it is there (if it vanished,
    # one of the two paths would be wrong).
    difieren = 0
    for _ in 1:20000
        z = ComplexF64(randn(), randn())
        TDGL.cabs2(st, z) === TDGL.cabs2(ft, z) || (difieren += 1)
    end
    @test difieren > 0
    @info "cabs2: hypot^2 and re^2+im^2 differ on $difieren of 20000 normal inputs"

    # exp(-i phi): both paths agree to machine precision
    for phi in range(-6.0, 6.0, length=101)
        @test TDGL.emiphi(st, phi) ≈ TDGL.emiphi(ft, phi) atol=1e-15
        @test abs(TDGL.emiphi(st, phi)) ≈ 1.0 atol=1e-15
    end

    # complex product: the contracted and the plain formula agree to machine
    # precision, and the hardware has a true FMA.
    @test TDGL.fma_como_octave()
    dif = 0
    for _ in 1:50000
        x = ComplexF64(randn(), randn()); y = ComplexF64(randn(), randn())
        a = TDGL.cmul(st, x, y); b = TDGL.cmul(ft, x, y)
        @test a ≈ b rtol=1e-15
        a === b || (dif += 1)
    end
    @test dif > 0
    @info "cmul: contracted and plain differ on $dif of 50000 normal inputs"
end
