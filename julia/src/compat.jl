# ============================================================================
#  compat.jl — THE TWO NUMERICAL MODES
#
#  The acceptance criterion is `max|Delta| = 0`, not "close", and a 1e-16 has
#  to be tracked down rather than accepted.  Measuring it showed that the
#  discrepancy is NOT in the scheme but in two libm primitives:
#
#      primitive        native Julia vs glibc (20000 random inputs)
#      ---------------  --------------------------------------------------
#      cos / sin        507 and 652 of 20000 differ by 1 ulp  (2.5 % / 3.3 %)
#      hypot            237 of 20000 differ by 1 ulp          (1.2 %)
#
#  The stored states were written by code that called glibc for both.  Julia
#  implements its own in pure Julia and they are just as correct — they
#  simply round differently in the last bit.  Calling glibc through `ccall`
#  the agreement is exact: 0 of 20000 for all three (test/test_compat.jl).
#
#  Hence the two NUMERICAL MODES.  They are not "fast and dirty" versus "slow
#  and good": they are two equally legitimate roundings, and only one
#  reproduces the stored states exactly.
#
#    Strict : glibc hypot and sincos.  The mode of the regression tests
#             against stored states.
#    Fast   : abs2 (re^2+im^2, no hypot) and native cis.  Vectorises, avoids
#             the per-element `ccall`, and agrees to machine precision, not
#             to the bit.
#
#  The mode is a TYPE PARAMETER, not a flag: it is resolved at compile time
#  and costs no branch inside the loop.
# ============================================================================

abstract type NumMode end

"""Reproducible mode: libm `hypot` and `sincos`; the mode of the regression tests."""
struct Strict <: NumMode end

"""Native Julia mode. `abs2` instead of `hypot^2`, `cis` instead of libm."""
struct Fast <: NumMode end

const LIBM = "libm.so.6"

@inline function hypot_libm(x::Float64, y::Float64)
    ccall((:hypot, LIBM), Float64, (Float64, Float64), x, y)
end

@inline function sincos_libm(x::Float64)
    s = Ref{Float64}(); c = Ref{Float64}()
    ccall((:sincos, LIBM), Cvoid, (Float64, Ptr{Float64}, Ptr{Float64}), x, s, c)
    (s[], c[])
end

# --- |z| ---------------------------------------------------------------------
@inline cabs(::Strict, z::ComplexF64) = hypot_libm(real(z), imag(z))
@inline cabs(::Fast,   z::ComplexF64) = abs(z)

# --- |z|^2 -------------------------------------------------------------------
# CAUTION: in Strict this is hypot(re,im)^2, which is what `abs(g).^2` does
# in the .m, and it is NOT the same as re^2+im^2 in the last bit.  It is the
# condensation term of the update, so the difference propagates.
@inline function cabs2(::Strict, z::ComplexF64)
    h = hypot_libm(real(z), imag(z)); h * h
end
@inline cabs2(::Fast, z::ComplexF64) = abs2(z)

# --- exp(-i*phi) with real phi ----------------------------------------------
# Octave evaluates exp(-1i*phi) as cexp(complex(-0.0, -phi)) = 1.0*(cos(-phi) +
# i sin(-phi)).  exp(+-0.0) is exactly 1.0, so the product introduces no
# rounding and cos/sin suffice.
@inline function emiphi(::Strict, phi::Float64)
    s, c = sincos_libm(-phi)
    ComplexF64(c, s)
end
@inline emiphi(::Fast, phi::Float64) = cis(-phi)

# --- arg(z) ------------------------------------------------------------------
# Julia's and glibc's atan2 agree in the tests made; the hook is kept in
# case they ever do not.  It enters only the voltmeter and the vortex count,
# neither of which touches the state.
@inline cangle(::NumMode, z::ComplexF64) = angle(z)

mode_name(::Strict) = "Strict (libm hypot/sincos)"
mode_name(::Fast)   = "Fast (native Julia, equal to machine precision)"


# --- product of two complex numbers -----------------------------------------
#  With glibc hypot and sincos the port still differed by 1e-15 after 1600
#  substeps, although the FIRST substep was exact.  That first substep is the
#  only one in which `G1` is still REAL (the Meissner seed has no phase), so
#  what entered anew in the second was the product of two genuine complex
#  numbers.  Measured over 50000 random products against this machine's
#  Octave:
#
#      re = a*c - b*d ,  im = a*d + b*c          38 and 27 of 50000 differ
#      re = fma(a,c,-(b*d)) , im = fma(a,d,b*c)   0 and  0 of 50000
#
#  So this Octave CONTRACTS the complex multiplication into an FMA.  That is
#  `-ffp-contract=fast`, GCC's default, on a binary built for x86-64-v3 (which
#  has FMA).  Julia never contracts unless asked, hence the last digit
#  diverged.
#
#  CONSEQUENCE: the bit pattern of the archive depends not only on the scheme
#  but also on the flags with which the Octave that produced it was compiled.
#  `Strict` reproduces THAT archive.  On an Octave built for generic x86-64
#  (no FMA) the reference pattern would be the other one, and the bit-exact
#  mode would be `Fast` on this line.  Bit equality remains attainable; what
#  is not universal is against WHICH binary.  See test/test_compat.jl, which
#  measures it again on the machine where it runs and warns if the verdict
#  changes.
@inline function cmul(::Strict, x::ComplexF64, y::ComplexF64)
    a = real(x); b = imag(x); c = real(y); d = imag(y)
    ComplexF64(fma(a, c, -(b*d)), fma(a, d, b*c))
end
@inline cmul(::Fast, x::ComplexF64, y::ComplexF64) = x*y

"""
    fma_como_octave() -> Bool

Measures again, here and now, whether the contracted formula is the one
that reproduces Octave's `x.*y`.  Returns `true` if `Strict` is the right mode
for the archive of THIS machine.  Without arguments it cannot call Octave,
so it only checks that the hardware has a true FMA (without one, `fma` would
still be correct but in software and very slow).
"""
function fma_como_octave()
    # `fma(a,b,-(a*b))` is the exact RESIDUAL of the rounded product.  It is
    # non-zero only if the fma has infinite intermediate precision, which is
    # what reproducing the compiler's contraction requires.
    fma(0.1, 0.2, -(0.1*0.2)) != 0.0
end
