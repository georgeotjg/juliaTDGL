# ============================================================================
#  noise.jl — thermal noise (Langevin) on the order parameter of each band (off by default)
#
#      Γ_µ ∂ₜΨ_µ = −δF/δΨ_µ* + η_µ,    ⟨η_µ(r,t) η_µ*(r',t')⟩ = 2 Γ_µ T̃ δ(r − r') δ(t − t')
#
#  T̃ the noise temperature in the units of the functional (F per ξ₁³ of the code: F = Σ dV [g|DΨ|² − fε|Ψ|² + β|Ψ|⁴/2 …]),
#  so that the stationary distribution of the discrete system is ∝ exp(−F/T̃) (fluctuation–dissipation).  On a node of
#  volume dV, per step (Euler–Maruyama, after the deterministic band update):
#      ΔΨ = √(T̃ dt/(Γ_µ dV)) (n₁ + i n₂),   n₁, n₂ ~ N(0, 1)
#  from a seeded Xoshiro stream drawn in a fixed order (the same numbers with any number of threads); its state goes into
#  the checkpoint of every closed point (resuming is continuing, to the bit) and the delay confirmation gives it back.
#  The nodes the band update writes; not the hard (f = 0, `hard`) ones nor the vacuum of a sculpted sample.
#
#  On the field (`noise_field = true`, the same T̃ and stream): Ampère is σ_l ∂ₜA = J_s − κ²∇×∇×A = −(1/2) δF/δA per unit
#  volume, so the mobility of a link is 1/(2σ_l dV) and fluctuation–dissipation gives ΔA_l = √(T̃ dt/(σ_l dV)) n; on the link
#  phase φ_l = A_l d_l, before the implicit solve (an explicit source of the step, as the supercurrent): the right side of
#  (D + aK)Δ = φₑ − θ takes D_l·d_l√(T̃ dt/(σ_l dV)) n = d_l√(T̃ dt D_l/(σn dV)) n, on every link the field step evolves (the
#  padding too: it carries normal current; an insulating link gets 1/√(σvac/σn) more after the solve).  The implicit Euler of
#  the field gives a mode of rate λ the variance (T̃/2)/(1 + λdt/2) of its share of F: the stiff modes of the mesh (λdt ≳ 1)
#  are colder; the slow ones, which move Ψ, are right to O(λdt).
#  With KWT the mobility of a node is a 2×2 matrix that depends on Ψ (real and imaginary parts as a vector, s = √(1+γ̃²|Ψ|²)):
#      M = (s/Γ) I − (γ̃²/(Γ s)) ψψᵀ    (along ψ 1/(Γ s), the amplitude; across it s/Γ, the phase),
#  so the kick is √(T̃ dt/dV) √M n, radial and tangential parts scaled apart, and the Itô drift that keeps exp(−F/T̃)
#      (T̃ dt/(2dV)) ∇·M = −(T̃ dt/(2dV)) γ̃²(1 + s²)/(Γ s³) ψ
#  is added; both at the state of the start of the step (Itô).  With γ̃ = 0 it is the kick above (and its bits).
# ============================================================================

using Random: Xoshiro, randn

struct Noise
    T::Float64                     # T̃
    seed::UInt64
    rng::Xoshiro
    field::Bool                    # also on the field (the link phases)
end
Noise(T::Real, seed::Integer; field::Bool=false) =
    (T >= 0 || error("noise: T̃ must be >= 0"); Noise(Float64(T), UInt64(seed), Xoshiro(seed), field))

"""Adds the Langevin kick of one step to every band (s.F, after the band update)."""
function noise_kick!(nz::Noise, s, st)
    gr = st.grid; md = st.model; f = st.masks.f; hard = st.hard; vac = st.masks.vac
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; rng = nz.rng
    st.mt === nothing || return _noise_kick_mat!(nz, s, st)
    for µ in 1:length(s.F)
        Γ = md.bands[µ].Γ; γ2 = md.bands[µ].γkwt^2
        a = sqrt(nz.T * st.dt / (Γ * dV))
        F = s.F[µ]
        if γ2 == 0
            @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
                (hard && f[i,j,k] == 0.0) && continue
                vac === nothing || vac[i,j,k] && continue
                F[i,j,k] += a * complex(randn(rng), randn(rng))
            end
        else
            G = s.G[µ]; c = sqrt(nz.T * st.dt / dV); dr = nz.T * st.dt / (2dV) * γ2 / Γ
            @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
                (hard && f[i,j,k] == 0.0) && continue
                vac === nothing || vac[i,j,k] && continue
                ξ = complex(randn(rng), randn(rng))
                F[i,j,k] = _kwt_kick(F[i,j,k], G[i,j,k], ξ, a, c, dr, γ2, Γ, nz.T * st.dt / (2dV * Γ))
            end
        end
    end
    s
end

# The kick of a KWT node: F (the band update) gets the kick of the mobility at ψ = Ψⁿ (Itô).  In polar form: the phase turned
# by δθ = √(T̃dt s/(Γ dV)) n₂/ρ, the amplitude moved by δρ = √(T̃dt/(Γ s dV)) n₁ and by the drift (T̃dt/(2ΓdV))/(ρ s³), with ρ = |ψ|.
# It is the SDE of the Cartesian kick, ψ + √M n − (T̃dt/(2dV))γ̃²(1+s²)/(Γs³) ψ (which it was until 26-oct-2026): in polar coordinates
# (measure ρ dρ dθ) the drift of ρ is T̃(∂ρ m_ρ + m_ρ/ρ) = (T̃/(2ΓdV))(1/(sρ) − γ̃²ρ/s³) = (T̃/(2ΓdV))/(ρ s³), and the Cartesian kick
# gives the same mean, (T̃/(2ΓdV))(s/ρ − γ̃²(1+s²)ρ/s³) = (T̃/(2ΓdV))/(ρ s³), as the difference of two terms s⁴ times larger: the
# lengthening of ρ by the tangential kick (a tangent step) and the Itô drift.  Once the band step had damped or over-heated the
# tangential noise (stages bridging λdt ≫ 1, the phase modes s times faster with KWT) they no longer cancelled, and ⟨|Ψ|²⟩ of
# the one-band box at γ̃ = 10, T̃ = 1e-2 came out 67 % low.  The polar kick needs no cancellation.  Where |ψ|² or
# |F|² is under _ROT_MIN2 (bands.jl; vortex cores, the normal metal), the Cartesian kick.
@inline function _kwt_kick(F::ComplexF64, ψ::ComplexF64, ξ::ComplexF64, a, c, dr, γ2, Γ, d0)
    r2 = abs2(ψ); sk = sqrt(1 + γ2 * r2)
    r2 > 1e-24 || return F + a * ξ
    u = ψ / sqrt(r2); w = conj(u) * ξ
    if r2 > _ROT_MIN2 && abs2(F) > _ROT_MIN2
        ρ = sqrt(r2); aF = abs(F)
        δρ = c * real(w) / sqrt(Γ * sk) + d0 / (ρ * sk^3)
        δθ = c * imag(w) * sqrt(sk / Γ) / ρ
        return (F / aF) * ((aF + δρ) * cis(δθ))
    end
    F + c * u * complex(real(w) / sqrt(Γ * sk), imag(w) * sqrt(sk / Γ)) - dr * (1 + sk^2) / sk^3 * ψ
end

# SK-ROCK for the phase (Abdulle, Almuslimani & Vilmart, SIAM/ASA J. Uncertainty Quantification 6, 937 (2018)).  A kick
# after the step gives a mode of rate λ the stationary variance 2|z|/(1 − R(z)²) times its share of exp(−F/T̃), z = λdt, R the
# amplification of the step: right for |z| ≪ 1, but RKC stages exist for modes up to |z| ~ 2q, and KWT makes the phase modes s
# times faster: with 3–13 stages the lattice phase modes were over-heated 10–1000×, and their
# disorder lowered ⟨|Ψ|²⟩ (the one-band box at γ̃ = 30, T̃ = 1e-2: 0.08 for 1.00).  SK-ROCK puts the kick Q inside the first RKC
# stage, K₁ = Ψⁿ + (dt/q) f(Ψⁿ + ν₁Q) + κ₁Q (ν₁ = sω₁/2, κ₁ = sω₁/ω₀), and then Ψⁿ⁺¹ = R(z)Ψⁿ + S(z)Q with S(z) =
# U_{s−1}(ω₀+ω₁z)/U_{s−1}(ω₀)(1 + ω₁z/2): every mode at or under its temperature, → 1 as z → 0 (h6_skrock_ou.log).  It is a
# statement on a linear mode, so it is applied to the phase, which the stages carry linearly in log-polar form (kwt_update_rot,
# _combine_log!, the frame as a rotation): the phase kick δθ enters the first stage as a rotation, Ψⁿ e^{iν₁δθ} and e^{i(κ₁−ν₁)δθ}
# (bands.jl).  The amplitude kick and its drift (_kwt_kick) stay after the step: the amplitude modes are s² times slower.
# noise_draw! makes every draw of the step before the bands, in the order of noise_kick!; noise_apply! adds the rest after them.
function noise_draw!(nz::Noise, s, st)
    gr = st.grid; f = st.masks.f; hard = st.hard; vac = st.masks.vac; mt = st.mt
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; rng = nz.rng; c = sqrt(nz.T * st.dt / dV)
    for µ in 1:length(s.F)
        Qξ = st.ws.Qξ[µ]; Qθ = st.ws.Qθ[µ]; G = s.G[µ]; sk = _skrock(st, µ)
        fill!(Qθ, 0.0)
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            (hard && f[i,j,k] == 0.0) && continue
            vac === nothing || vac[i,j,k] && continue
            ξ = complex(randn(rng), randn(rng)); Qξ[i,j,k] = ξ
            sk || continue
            m = mt === nothing ? 1 : Int(mt.idx[i,j,k])
            mt === nothing || !mt.absent[m, µ] || continue
            b = mt === nothing ? st.model.bands[µ] : mt.models[m].bands[µ]
            γ2 = b.γkwt^2; ψ = G[i,j,k]; r2 = abs2(ψ)
            (γ2 > 0 && r2 > _ROT_MIN2) || continue
            sk_ = sqrt(1 + γ2 * r2); ρ = sqrt(r2)
            Qθ[i,j,k] = c * imag(conj(ψ / ρ) * ξ) * sqrt(sk_ / b.Γ) / ρ
        end
    end
    s
end

function noise_apply!(nz::Noise, s, st)
    gr = st.grid; f = st.masks.f; hard = st.hard; vac = st.masks.vac; mt = st.mt
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; c = sqrt(nz.T * st.dt / dV)
    for µ in 1:length(s.F)
        Qξ = st.ws.Qξ[µ]; Qθ = st.ws.Qθ[µ]; G = s.G[µ]; F = s.F[µ]
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            (hard && f[i,j,k] == 0.0) && continue
            vac === nothing || vac[i,j,k] && continue
            m = mt === nothing ? 1 : Int(mt.idx[i,j,k])
            mt === nothing || !mt.absent[m, µ] || continue
            b = mt === nothing ? st.model.bands[µ] : mt.models[m].bands[µ]
            Γ = b.Γ; γ2 = b.γkwt^2; a = sqrt(nz.T * st.dt / (Γ * dV)); ξ = Qξ[i,j,k]
            if γ2 == 0
                F[i,j,k] += a * ξ
            elseif Qθ[i,j,k] != 0.0                     # the phase went in with the stages: the amplitude here
                ψ = G[i,j,k]; ρ = abs(ψ); sk = sqrt(1 + γ2 * ρ^2); u = ψ / ρ
                δρ = c * real(conj(u) * ξ) / sqrt(Γ * sk) + nz.T * st.dt / (2dV * Γ) / (ρ * sk^3)
                aF = abs(F[i,j,k])
                F[i,j,k] = aF > sqrt(_ROT_MIN2) ? F[i,j,k] * ((aF + δρ) / aF) : F[i,j,k] + u * δρ
            else
                F[i,j,k] = _kwt_kick(F[i,j,k], G[i,j,k], ξ, a, c, nz.T * st.dt / (2dV) * γ2 / Γ, γ2, Γ, nz.T * st.dt / (2dV * Γ))
            end
        end
    end
    s
end

# With materials: the kick of each node from its own material's Γ and KWT mobility (and its Itô drift); the same draws in
# the same order as the uniform kick (two per node), so a copy of the material is the uniform noisy run to the bit; a band absent
# from the node's material takes its two draws and no kick.
function _noise_kick_mat!(nz::Noise, s, st)
    gr = st.grid; mt = st.mt; f = st.masks.f; hard = st.hard; vac = st.masks.vac
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; rng = nz.rng
    for µ in 1:length(s.F)
        F = s.F[µ]; G = s.G[µ]
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            (hard && f[i,j,k] == 0.0) && continue
            vac === nothing || vac[i,j,k] && continue
            m = mt.idx[i,j,k]; b = mt.models[m].bands[µ]
            Γ = b.Γ; γ2 = b.γkwt^2
            if γ2 == 0
                a = sqrt(nz.T * st.dt / (Γ * dV))
                ξ = complex(randn(rng), randn(rng))
                mt.absent[m, µ] || (F[i,j,k] += a * ξ)
            else
                c = sqrt(nz.T * st.dt / dV); dr = nz.T * st.dt / (2dV) * γ2 / Γ; a = sqrt(nz.T * st.dt / (Γ * dV))
                ξ = complex(randn(rng), randn(rng))
                mt.absent[m, µ] && continue
                F[i,j,k] = _kwt_kick(F[i,j,k], G[i,j,k], ξ, a, c, dr, γ2, Γ, nz.T * st.dt / (2dV * Γ))
            end
        end
    end
    s
end

"""Adds the Langevin kick of one step to the link phases (s.φ after `field_increment!`, before the implicit solve)."""
function field_kick!(nz::Noise, s, st)
    gr = st.grid; (; Nx, Ny, Nz, dx, dy, dz) = gr
    rng = nz.rng; c = sqrt(nz.T * st.dt / (st.model.σn * dx * dy * dz))
    D = st.D
    ax = c * dx; ay = c * dy; az = c * dz
    @inbounds for k in 2:Nz, j in 2:Ny, i in 1:Nx
        s.φx[i,j,k] += (D === nothing ? ax : ax * sqrt(D[1][i,j,k])) * randn(rng)
    end
    @inbounds for k in 2:Nz, j in 1:Ny, i in 2:Nx
        s.φy[i,j,k] += (D === nothing ? ay : ay * sqrt(D[2][i,j,k])) * randn(rng)
    end
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx
        s.φz[i,j,k] += (D === nothing ? az : az * sqrt(D[3][i,j,k])) * randn(rng)
    end
    s
end

_noise_snap(st) = st.noise === nothing ? nothing : copy(st.noise.rng)
_noise_back!(st, sn) = (sn === nothing || copy!(st.noise.rng, sn); st)
