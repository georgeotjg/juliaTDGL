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
                ψ = G[i,j,k]; r2 = abs2(ψ); sk = sqrt(1 + γ2 * r2)
                kick = if r2 > 1e-24
                    u = ψ / sqrt(r2); w = conj(u) * ξ
                    c * u * complex(real(w) / sqrt(Γ * sk), imag(w) * sqrt(sk / Γ))
                else
                    a * ξ
                end
                F[i,j,k] += kick - dr * (1 + sk^2) / sk^3 * ψ
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
