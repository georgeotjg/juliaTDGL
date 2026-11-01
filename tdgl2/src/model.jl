# ============================================================================
#  model.jl — the material: N bands, their coupling, the field and the normal channel
#
#  Free energy (units of xi_1, t_GL = xi_1^2/D_1, Hc2 of band 1):
#      F = sum_mu [ g_mu |D Psi_mu|^2 - f eps_mu |Psi_mu|^2 + (beta_mu/2) |Psi_mu|^4 ]
#          - sum_{mu<nu} gamma_{mu nu} (Psi_mu^* Psi_nu + c.c.) + kappa^2 |curl A|^2
#  with D = grad - iA, eps_mu = a_mu - T, and the dynamics
#      Gamma_mu d_t Psi_mu = g_mu D^2 Psi_mu + (f eps_mu - beta_mu |Psi_mu|^2) Psi_mu
#                            + sum_nu gamma_{mu nu} Psi_nu
#      sigma_n d_t A       = J_s - kappa^2 curl curl A,   J_s = sum_mu g_mu Im(Psi_mu^* D Psi_mu)
#  (Gurevich-Vinokur's two-band model for N = 2).  u = Gamma_1/sigma_n is the ratio of the
#  relaxation times of the order parameter and of the current.
# ============================================================================

"""
    Band(; m=1, Γ=1, a=1, β=1, γkwt=0, m_axes=nothing)
    Band(; g, ...)

One band.  `m` is the mass ratio, so the stiffness is g = 1/m (pass `g` instead if that is what
the material file gives); `Γ` the relaxation coefficient of the order parameter; `a` the
condensation coefficient at T = 0 (uncoupled equilibrium |Ψ|² = a/β); `β` the quartic one;
`γkwt` the Kramer–Watts-Tobin inelastic factor, referred to |Ψ| in these units (0 = off).
`m_axes = (m_x, m_y, m_z)`: an anisotropic band, the mass along each axis (ξ_i ∝ 1/√m_i, λ_i ∝ √m_i); `m` is then
m_x, the mass along the current (what `regime`, `sigma_dirty` and the summary use).  Without it the band is isotropic.
"""
struct Band
    m::Float64
    Γ::Float64
    a::Float64
    β::Float64
    γkwt::Float64
    ma::NTuple{3,Float64}           # mass along x, y, z (all m for an isotropic band)
end
Band(m::Float64, Γ::Float64, a::Float64, β::Float64, γkwt::Float64) = Band(m, Γ, a, β, γkwt, (m, m, m))

function Band(; m::Real=1.0, g=nothing, Γ::Real=1.0, a::Real=1.0, β::Real=1.0, γkwt::Real=0.0, m_axes=nothing)
    mm = g === nothing ? Float64(m) : 1 / Float64(g)
    ma = m_axes === nothing ? (mm, mm, mm) : Tuple(Float64.(collect(m_axes)))
    length(ma) == 3 || error("Band: m_axes is (m_x, m_y, m_z)")
    m_axes === nothing || (mm = ma[1])
    mm > 0 && all(>(0), ma) && Γ > 0 && β > 0 || error("Band: m (or g, m_axes), Γ and β must be positive")
    γkwt >= 0 || error("Band: γkwt must be >= 0")
    Band(mm, Float64(Γ), Float64(a), Float64(β), Float64(γkwt), ma)
end

stiffness(b::Band) = 1 / b.m
"""The stiffness along axis `i` (1, 2, 3 = x, y, z)."""
stiffness(b::Band, i::Int) = 1 / b.ma[i]
isotropic(b::Band) = b.ma[1] == b.ma[2] == b.ma[3] == b.m

"""u of one dirty BCS band near Tc (Kramer–Watts-Tobin): π⁴/(14ζ(3)) = 5.79."""
const U_DIRTY = π^4 / (14 * 1.2020569031595942)

"""
    sigma_dirty(bands) -> σn

The normal conductivity of a dirty multiband BCS superconductor near Tc, in the units of the code
(band 1 with g = Γ = β = 1): σn = Σ_µ g_µ Γ_µ/β_µ / 5.79.  With Ψ_µ = Δ_µ/s_µ the coefficients go as
g_µ ∝ N_µ D_µ s_µ², Γ_µ ∝ N_µ s_µ², β_µ ∝ N_µ s_µ⁴ (Gurevich, PRB 67, 184515 (2003), Eqs. 15–17), so
g_µΓ_µ/β_µ ∝ N_µ D_µ, the band's share of σn = 2e² Σ N_µ D_µ; one band gives u = Γ/σn = 5.79.
"""
sigma_dirty(bands) = sum(stiffness(b) * b.Γ / b.β for b in bands) / U_DIRTY

"""
    Model(bands, γ; κ, σn=1, T=0)

`bands` is a tuple of `Band`; `γ` the interband Josephson matrix (real, symmetric, zero
diagonal; a complex or non-symmetric coupling would remove the exact C∘M_y null and is not
accepted).  `σn` is the normal conductivity of the sample and the contacts; `σn = :dirty` takes the
one of a dirty BCS material with these bands (`sigma_dirty`).  `T` the bath
temperature (ε_µ = a_µ − T).
"""
struct Model{N}
    bands::NTuple{N,Band}
    γ::NTuple{N,NTuple{N,Float64}}
    κ::Float64
    σn::Float64
    T::Float64
end

function Model(bands::NTuple{N,Band}, γ::AbstractMatrix; κ::Real, σn::Union{Real,Symbol}=1.0, T::Real=0.0) where N
    σn isa Symbol && (σn === :dirty ? (σn = sigma_dirty(bands)) : error("Model: σn is a number or :dirty"))
    size(γ) == (N, N) || error("Model: γ must be $N×$N")
    for µ in 1:N, ν in 1:N
        γ[µ, ν] == γ[ν, µ] || error("Model: γ must be symmetric")
        µ == ν && γ[µ, µ] != 0 && error("Model: the diagonal of γ must be zero")
    end
    κ > 0 && σn > 0 || error("Model: κ and σn must be positive")
    Model{N}(bands, ntuple(µ -> ntuple(ν -> Float64(γ[µ, ν]), N), N), Float64(κ), Float64(σn), Float64(T))
end

nbands(::Model{N}) where N = N

"""ε_µ = a_µ − T: the condensation coefficient of band µ at the bath temperature."""
eps_band(md::Model, µ::Int) = md.bands[µ].a - md.T

"""
    two_band(; xi_ratio, a2n=0.59, gamma12=1e-3, gamma_ratio=1, κ, σn=1, T=0, gamma_kwt=0)

The two-band model of juliaTDGL 1.x: σ with g = Γ = a = 1, π with m_r = 1/(R² a₂ₙ),
Γ = Γr and a = a₂ₙ, both β = 1, Josephson γ₁₂.  `xi_ratio` R = ξπ/ξσ.  `gamma_kwt`: the KWT factor
γ̃ of both bands, or a pair (σ, π); 0 is 1.x.
"""
function two_band(; xi_ratio::Real, a2n::Real=0.59, gamma12::Real=1e-3, gamma_ratio::Real=1.0,
                  κ::Real, σn::Union{Real,Symbol}=1.0, T::Real=0.0, beta2::Real=1.0,
                  gamma_kwt::Union{Real,Tuple{Real,Real}}=0.0)
    mr = 1 / (xi_ratio^2 * (a2n / beta2))
    gk = gamma_kwt isa Real ? (gamma_kwt, gamma_kwt) : gamma_kwt
    Model((Band(γkwt=gk[1]), Band(m=mr, Γ=gamma_ratio, a=a2n, β=beta2, γkwt=gk[2])),
          [0.0 gamma12; gamma12 0.0]; κ=κ, σn=σn, T=T)
end

"""
    equilibrium(md) -> NTuple{N,Float64}

The amplitudes |Ψ_µ| of the homogeneous equilibrium with the interband coupling (same-sign phases, the
larger root): band µ solves β_µ p³ − ε_µ p − Σ_ν |γ_µν| p_ν = 0.  Gauss–Seidel over the bands, Newton from
above on each cubic (convex past its largest root), so it converges also for a passive band (ε_µ < 0),
whose condensate is only the one the coupling induces.
"""
function equilibrium(md::Model{N}) where N
    b = md.bands
    p = [sqrt(max(eps_band(md, µ), 0.0) / b[µ].β) for µ in 1:N]
    for _ in 1:100_000
        dmax = 0.0
        for µ in 1:N
            ε = eps_band(md, µ); β = b[µ].β
            c = sum(ν -> ν == µ ? 0.0 : abs(md.γ[µ][ν]) * p[ν], 1:N)
            x = sqrt(max(ε, 0.0) / β) + cbrt(c / β)      # f(x) ≥ 0 here
            for _ in 1:200
                f = β * x^3 - ε * x - c; df = 3β * x^2 - ε
                (f <= 0 || df <= 0) && break
                dx = f / df; x -= dx
                dx <= 1e-15 * x && break
            end
            dmax = max(dmax, abs(x - p[µ])); p[µ] = x
        end
        dmax < 1e-14 && break
    end
    Tuple(p)
end

"""
    regime(md; L) -> NamedTuple

The four numbers every result of a two-band model reports with it: the ratio of diffusivities
D_µ = g_µ/Γ_µ of the last band to the first, ℓ_J/L (interband Josephson length over the sample
length L, computed with the uncoupled equilibrium amplitudes, or with the coupled ones when a band
is passive, ε_µ ≤ 0, and has no uncoupled condensate), the invariant γ²/(α₁α₂) and
σn/Σg_µ; and with them u_µ = Γ_µ/σn per band and σn over the dirty-BCS value (`sigma_dirty`;
1 when the conductivity is the one a dirty material with these bands would have).  Also the conversion length of the
normal current injected at a contact, λ_c² = σn/Σ_µ Γ̃_µ|Ψ_µ|² with Γ̃_µ = Γ_µ/√(1 + γ̃_µ²|Ψ_µ|²) (in ξ₁, at the same
amplitudes; checked against pyTDGL to 0.2 % with one band, γ̃ = 0–30), and L/λ_c: KWT lengthens it (×3 with γ̃ = 10), and when
L is not many λ_c (a 20 ξ box at κ = 2 with γ̃ = 30: L/λ_c ≈ 5) the middle of the sample still carries unconverted normal
current, which moves the onset by itself.
"""
function regime(md::Model{2}; L::Real)
    b1, b2 = md.bands
    g1, g2 = stiffness(b1), stiffness(b2)
    D = (g2 / b2.Γ) / (g1 / b1.Γ)
    e1, e2 = eps_band(md, 1), eps_band(md, 2)
    p1, p2 = e1 > 0 && e2 > 0 ? (sqrt(e1 / b1.β), sqrt(e2 / b2.β)) : equilibrium(md)
    γ = abs(md.γ[1][2])
    # reduced stiffness of the interband phase: K = g1 g2 p1² p2² / (g1 p1² + g2 p2²)
    K = g1 * g2 * p1^2 * p2^2 / (g1 * p1^2 + g2 * p2^2)
    ℓJ = γ > 0 ? sqrt(K / (γ * p1 * p2)) : Inf
    Γt = (b1.Γ / sqrt(1 + b1.γkwt^2 * p1^2), b2.Γ / sqrt(1 + b2.γkwt^2 * p2^2))
    λc = sqrt(md.σn / (Γt[1] * p1^2 + Γt[2] * p2^2))
    (D_ratio = D, ellJ_over_L = ℓJ / L, coupling_invariant = γ^2 / abs(e1 * e2),
     sigma_over_stiffness = md.σn / (g1 + g2), u = (b1.Γ / md.σn, b2.Γ / md.σn),
     sigma_over_dirty = md.σn / sigma_dirty(md.bands), lambda_c = λc, L_over_lambda_c = L / λc)
end

# Any number of bands: what does not need a pair (the conversion length, u_µ, σn over the stiffness and the dirty value),
# at the coupled equilibrium amplitudes.
function regime(md::Model{N}; L::Real) where N
    b = md.bands
    p = equilibrium(md)
    λc = sqrt(md.σn / sum(b[µ].Γ / sqrt(1 + b[µ].γkwt^2 * p[µ]^2) * p[µ]^2 for µ in 1:N))
    (sigma_over_stiffness = md.σn / sum(stiffness(b[µ]) for µ in 1:N), u = ntuple(µ -> b[µ].Γ / md.σn, N),
     sigma_over_dirty = md.σn / sigma_dirty(b), lambda_c = λc, L_over_lambda_c = L / λc)
end
