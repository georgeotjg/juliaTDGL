# ============================================================================
#  scheme.jl — the time step and the coefficients of the update
#
#  Explicit Euler for every band, optionally with substeps (multirate), and the field equation
#  with its curl-curl term implicit (default) or explicit.  The step is the smallest stability
#  bound times 0.99 times dt_scale; dt_scale = 0.5 keeps every mode of the stiffest band from
#  changing sign each step (accurate in strong flow, release 1.1).
#
#  Bounds (explicit Euler is stable while dt*lambda_max <= 2):
#    band mu        2 Gamma_mu / (8 g_mu/dl + 2 a_mu/beta_mu)   (lattice Laplacian + amplitude mode)
#    field          sigma_n dl/(4 kappa^2)                       (only with the explicit curl-curl)
#    supercurrent   2 sigma_n / sum_mu g_mu a_mu/beta_mu        (longitudinal mode of A, explicit)
#  A band with n substeps relaxes its bound by n.  With substeps = 0 for a band, n is the
#  smallest that keeps it from setting the step.
#
#  The coefficients are computed in a fixed order: with N = 2 and the parameters of release 1.2
#  the step is the one of 1.2 to the bit (test/test_regression12.jl).
# ============================================================================

"""Coefficients of one band for one (sub)step: kinetic, dt/Γ, condensation, quartic, coupling."""
struct BandCoef{N}
    kx::Float64; ky::Float64; kz::Float64
    c::Float64
    ε::Float64
    β::Float64
    cpl::NTuple{N,Float64}
    γ2::Float64                     # γ̃² of KWT (0 = plain TDGL)
end

"""
    _psi2_bound(md) -> NTuple{N,Float64}

|Ψ_µ|² of the homogeneous equilibrium with the interband coupling (same-sign phases taken, the larger
amplitudes): p_µ² = (ε_µ + Σ_ν |γ_µν| p_ν/p_µ)/β_µ by damped fixed point; with γ = 0 it is max(ε_µ, 0)/β_µ.
The fixed point oscillates for a passive band with a strong coupling (ε_π = −9, γ = 1.9: MgB₂ at 0.7 Tc), and
then `equilibrium` is used; the models where it converges keep their bits.
"""
function _psi2_bound(md::Model{N}) where N
    b = md.bands
    p = ntuple(µ -> sqrt(max(eps_band(md, µ), 0.0) / b[µ].β) + 1e-3, N)
    done = false
    for _ in 1:10_000
        q = ntuple(N) do µ
            c = sum(ν -> ν == µ ? 0.0 : abs(md.γ[µ][ν]) * p[ν], 1:N) / p[µ]
            0.5 * p[µ] + 0.5 * sqrt(max(eps_band(md, µ) + c, 0.0) / b[µ].β)
        end
        maximum(abs.(q .- p)) < 1e-13 && (p = q; done = true; break)
        p = q
    end
    done || (p = equilibrium(md))
    ntuple(µ -> p[µ] < 2e-3 && all(ν -> ν == µ || md.γ[µ][ν] == 0, 1:N) ? max(eps_band(md, µ), 0.0) / b[µ].β : p[µ]^2, N)
end

# the section of a sculpted sample at the contact x1 (nodes j in y1:y2+1, k in z1:z2+1 that are not vacuum); the
# current enters and leaves through it, so the far contact should have the same one
function _section(gr::Grid, mk::Masks)
    (; x1, x2, y1, y2, z1, z2) = gr
    a = .!mk.vac[x1+1, y1:y2+1, z1:z2+1]; b = .!mk.vac[x2, y1:y2+1, z1:z2+1]
    a == b || @warn "the sculpted sample has different sections at x1 and x2; the self-field takes the one at x1"
    a
end

_divided(b::BandCoef{N}, n::Real) where N =
    BandCoef{N}(b.kx / n, b.ky / n, b.kz / n, b.c / n, b.ε, b.β, ntuple(ν -> b.cpl[ν] / n, N), b.γ2)

# RKC1, damped (Verwer, Hundsdorfer & Sommeijer, Numer. Math. 57, 157 (1990); Verwer, Appl. Numer. Math. 22, 359 (1996)):
# ω₀ = 1 + ε/s², ω₁ = T_s(ω₀)/T_s'(ω₀); Y₁ = Y₀ + μ̃₁τF(Y₀), Y_j = μ_jY_{j−1} + ν_jY_{j−2} + μ̃_jτF(Y_{j−1}) with
# μ_j = 2ω₀T_{j−1}/T_j, ν_j = −T_{j−2}/T_j, μ̃_j = 2ω₁T_{j−1}/T_j (μ̃₁ = ω₁/ω₀).  μ̃_j/μ_j = ω₁/ω₀ for every j and μ_j + ν_j = 1,
# so a stage is μ_j E(Y_{j−1}) + ν_j Y_{j−2} with E the Euler step of h = τω₁/ω₀, the same pass as RKL1 with other weights:
# the divisor of the step is q = ω₀/ω₁.  Stable for τλ in [0, β], β = (1 + ω₀)/ω₁: s stages bridge a ratio β/2 of explicit steps
# (0.968 s² with ε = 0.05, 0.762 s² with ε = 0.5; RKL1, s(s+1)/2).  Stage j sits at c_j = ω₁T_j'(ω₀)/T_j(ω₀) of the step (c_s = 1).
# The damping is 0.5, not the textbook 0.05: on the KWT notched box (kwt_k2.toml, γ̃ = 10 and 30) ε <= 0.25 diverged within
# 50–400 steps at any J with the same stages, ε = 0.3–0.4 held but not with one stage fewer on the π band, 0.5 holds with one
# fewer, as RKL does.  The weakly damped interior of the stability polynomial
# (|R| up to ~1/cosh√(2ε) at its extrema) leaves modes that the splitting with the field and the other band amplifies.
const RKC_EPS = 0.5
struct RKCCoef
    q::Float64                      # τ/h
    μ::Vector{Float64}; ν::Vector{Float64}; c::Vector{Float64}    # per stage (μ₁ = 1, ν₁ = 0)
    sk::NTuple{2,Float64}           # SK-ROCK's weights of the noise in the first stage (ν₁ = sω₁/2, κ₁ = sω₁/ω₀; noise.jl)
end
function _rkc_cheb(s::Int, ω0::Float64)                 # (T_s(ω₀), T_s'(ω₀))
    s == 0 && return (1.0, 0.0)
    T0, T1, d0, d1 = 1.0, ω0, 0.0, 1.0
    for _ in 2:s
        T0, T1, d0, d1 = T1, 2ω0 * T1 - T0, d1, 2T1 + 2ω0 * d1 - d0
    end
    T1, d1
end
function RKCCoef(s::Int; ε::Float64=RKC_EPS)
    ω0 = 1 + ε / s^2; Ts, dTs = _rkc_cheb(s, ω0); ω1 = Ts / dTs
    μ = ones(s); ν = zeros(s); c = zeros(s)
    for j in 1:s
        Tj, dTj = _rkc_cheb(j, ω0)
        c[j] = ω1 * dTj / Tj
        j >= 2 && (μ[j] = 2ω0 * _rkc_cheb(j - 1, ω0)[1] / Tj; ν[j] = -_rkc_cheb(j - 2, ω0)[1] / Tj)
    end
    c[s] = 1.0
    RKCCoef(ω0 / ω1, μ, ν, c, (s * ω1 / 2, s * ω1 / ω0))
end
function _rkc_ratio(s::Int; ε::Float64=RKC_EPS)        # β/2: the explicit steps that s stages bridge
    ω0 = 1 + ε / s^2; Ts, dTs = _rkc_cheb(s, ω0)
    (1 + ω0) / (2 * Ts / dTs)
end

# The stability bounds of the bands of one model (bb0 without KWT, bbk with it) and the stiffness sum of the supercurrent bound.
function _band_bounds(md::Model{N}, gr::Grid, l::Float64; present::NTuple{N,Bool}=ntuple(_ -> true, N)) where N
    bands = md.bands
    # |Ψ_µ|² is bounded by the coupled homogeneous equilibrium P_µ², not by a_µ/β_µ (interband coupling
    # raises it; a_µ <= 0 is a band induced by the others); the reaction term of the Hessian is at most
    # 3β_µP_µ² − ε_µ (2a_µ/β_µ·β_µ without coupling)
    # (release 1.2 used a_µ/β_µ: kept whenever the coupled bound is within the 1 % of the 0.99 margin,
    # so that the paths of 1.2 stay to the bit)
    P2 = map(_psi2_bound(md), ntuple(µ -> bands[µ].a / bands[µ].β, N)) do p, q
        q > 0 && p <= 1.01q ? q : p
    end
    # (2a/β is 3βP² − ε at P² = a/β only for β = 1; with β ≠ 1 it underestimated the term)
    hr = ntuple(µ -> P2[µ] == bands[µ].a / bands[µ].β && bands[µ].β == 1 ? 2 * (bands[µ].a / bands[µ].β) :
                     max(3 * bands[µ].β * P2[µ] - eps_band(md, µ), 0.0), N)
    # the diffusion part of the bound, 4 Σ_i g_i/dx_i² (8/(l m) for an isotropic band, as always)
    dif = ntuple(N) do µ
        b = bands[µ]
        isotropic(b) ? 8 / (l * b.m) : 4 * (1 / (b.ma[1] * gr.dx^2) + 1 / (b.ma[2] * gr.dy^2) + 1 / (b.ma[3] * gr.dz^2))
    end
    bb0 = ntuple(µ -> 2 * bands[µ].Γ / (dif[µ] + hr[µ]), N)
    # KWT: the phase mode is faster by √(1 + γ̃²|Ψ|²), at most √(1 + γ̃² P²)
    bbk = ntuple(µ -> bands[µ].γkwt > 0 ? bb0[µ] / sqrt(1 + bands[µ].γkwt^2 * P2[µ]) : bb0[µ], N)
    # the superfluid stiffness of the supercurrent bound, Σ g_µ|Ψ_µ|²: a/β for a band that condenses alone (as in 1.2), and for a
    # passive band (a ≤ 0, induced by the others) its coupled bound P², not a/β < 0, which made the sum too small or negative
    # (the clean MgB2 of the catalogue: −15, a negative step)
    ns = ntuple(µ -> bands[µ].a > 0 ? bands[µ].a / bands[µ].β : P2[µ], N)
    λL = sum(ntuple(µ -> (isotropic(bands[µ]) ? stiffness(bands[µ]) : 1 / minimum(bands[µ].ma)) * ns[µ], N))
    all(present) && return bb0, bbk, λL
    # a band absent from this material (materials.jl): no bound, no share of the supercurrent bound
    (ntuple(µ -> present[µ] ? bb0[µ] : Inf, N), ntuple(µ -> present[µ] ? bbk[µ] : Inf, N),
     sum(ntuple(µ -> present[µ] ? (isotropic(bands[µ]) ? stiffness(bands[µ]) : 1 / minimum(bands[µ].ma)) * ns[µ] : 0.0, N)))
end

# The full-step coefficients of band µ of a model; the kinetic ones with the masses mref (the band's own, or the first
# material's for the tables of materials.jl, whose links carry the ratio).
function _bandcoef(md::Model{N}, µ::Int, dt, rx, ry, rz, mref::NTuple{3,Float64}) where N
    b = md.bands[µ]
    c = dt / b.Γ
    Γh = dt / c                                     # Γ as the step sees it (exactly 1 for Γ = 1)
    BandCoef{N}(rx / (mref[1] * Γh), ry / (mref[2] * Γh), rz / (mref[3] * Γh), c, eps_band(md, µ),
                b.β, ntuple(ν -> (dt / Γh) * md.γ[µ][ν], N), b.γkwt^2)
end

struct FieldCoef
    d1::Float64                     # dt/σn, on the supercurrent
    sxy::Float64; sxz::Float64; syx::Float64; syz::Float64; szx::Float64; szy::Float64
    a::Float64                      # (dt/σn) κ²/dx², for the implicit solve
end

"""Work arrays of the band substeps."""
struct Workspace
    W::A3c
    L::NTuple{6,A3c}                # substep links and their rotation per substep
    W2::A3c                         # third stage buffer of RKL (empty without it)
    Ω::A3r                          # rotation rate of the band being stepped (rotframe; empty without it)
    Ωs::A3r                         # rotframe with noise: a copy of Ω for its low-pass (empty otherwise)
    R::NTuple{3,A3r}                # rotframe: the phase rate of each link over a step, link and frame together
    Qθ::Vector{A3r}                 # noise on a KWT band with RKC stages (SK-ROCK, noise.jl): the phase kick of each node and
    Qξ::Vector{A3c}                 #   the draws of every band, made before the bands (empty otherwise)
end

struct Stepper{N,I}
    model::Model{N}
    grid::Grid
    masks::Masks
    dt::Float64
    nsub::NTuple{N,Int}
    rkl::NTuple{N,Int}              # RKL1 stages per band (0 = Euler with nsub substeps)
    bc::NTuple{N,BandCoef{N}}       # full-step coefficients
    bcs::NTuple{N,BandCoef{N}}      # per-substep coefficients (= bc when nsub = 1)
    w::NTuple{N,Float64}            # supercurrent weight g_mu (along x)
    wa::NTuple{3,NTuple{N,Float64}} # the weight along x, y, z (anisotropic bands)
    fc::FieldCoef
    imp::I                          # SepSolver, PCGSolver (conductivity per link), or nothing (explicit)
    D::Union{Nothing,NTuple{3,A3r}} # σ_l/σ_n per link with an insulating padding (nothing = σ_n everywhere)
    bar::Any                        # field of the bar current on the outer faces (nothing = the slab condition of 1.x)
    open::Union{Nothing,OpenBC}     # the open boundary (:open), on top of the bar field
    extrap::Bool                    # substeps rotate the links at the previous step's rate
    rotframe::Bool                  # each band is stepped in the gauge that turns at its own rate (bands.jl)
    hard::Bool                      # f = 0 nodes held at Psi = 0
    threads::Bool
    heat::Union{Nothing,HeatField}
    drag::Union{Nothing,NTuple{N,NTuple{N,NTuple{3,Float64}}}}   # per band µ, the coefficients on band ν
    dragν::Union{Nothing,NTuple{N,NTuple{N,Float64}}}            # ν_µν for the supercurrent
    ws::Workspace
    limit::String                   # which bound sets the step
    dt_scale::Float64               # as given (for the record of a run)
    rkl_margin::Float64             # as resolved (1.1 with KWT by default)
    qpx::Union{Nothing,QPExchange,QPExchangeN}  # normal fluids with a finite interband exchange (exchange.jl); nothing = one fluid
    noise::Union{Nothing,Noise}     # Langevin noise on the bands (noise.jl); nothing = none
    rkc::Union{Nothing,NTuple{N,Union{Nothing,RKCCoef}}}   # rkl = "rkc": per band, damped RKC1 stages where they save one (nothing = RKL1)
    mt::Union{Nothing,MatField{N}}  # material per region (materials.jl); nothing = one material
end

"""
    Stepper(md, gr, mk; implicit=true, substeps=auto, dt_scale=0.5, extrap=true, hard=false,
            threads=Threads.nthreads() > 1, σvac=nothing, pcg_tol=1e-8, rkl="rkc")

The production scheme of release 1.1 is the default: implicit curl-curl, automatic substeps
for every band but the first, half the bound.  `substeps` is a tuple of integers per band
(0 = automatic, 1 = none).  `σvac`: conductivity of the padding outside the bar of the sample and
its contacts (the insulating padding; `nothing` = normal metal σn everywhere, as in 1.x).
`selffield_bc`: `:bar` (default: the field of the current in the bar of the sample on every outer
face, so that the sample carries the nominal current J·W·d) or `:slab` (h_y = ∓HJ on the whole z
faces, as 1.x: Ampère then encloses J·d·(W + 2·buf·dx − dx), 4 % more than nominal on a 40 × 24 × 6
box with one padding cell; the choice to reproduce the runs of release 1.x), or `:open` (the bar field plus the free-space field of every
other current of the box on the y and z faces, recomputed every `open_every` t_GL and ramped:
`openbc.jl`; the field of the screening currents at small κ, which a mirror face suppresses).
`rkl = true`: the band updates by first-order Runge–Kutta–Legendre stages instead of Euler
substeps; the step is then the one of the same model without KWT, and each band takes the stages that
bridge its own bound (with KWT) to it: s stages for a ratio up to s(s+1)/2.  On the ladders of the two-band
notched box at κ = 25 it gives the substeps' brackets and flowing voltages to 1e-4, 8 % faster; with KWT (γ̃ = 10)
4.5 times faster than the explicit step.  `rkl = false`: the Euler substeps of release 1.2.  `rkl = "rkc"`: damped
first-order Runge–Kutta–Chebyshev stages (`RKCCoef`, damping `rkc_damping` = 0.5) in place of RKL1: the same step and the same
passes, 0.76 s² explicit steps bridged by s stages against s(s+1)/2, so fewer stages for a stiff band (γ̃ = 30 on the notched box:
(5, 13) against (6, 15)).  A band takes RKC only where it saves a stage, so a run without KWT keeps the bits of RKL.  The default
since 26-oct-2026 (−6 % per step with γ̃ = 10, −10 % with 30, the same brackets and delays); `rkl = true` gives the bits of the
KWT runs made before.  `pcg_tol` (default 1e-8, 1e-11 before 26-oct-2026): the tolerance of the insulating solve, relative to the
step's right-hand side (1e-8 is −41 % per step on the nanowire, the same brackets, the flowing voltages to 9 digits).
`rotframe`: every band is stepped in the gauge that turns at its own rate at the start of the step and turned back at
the end (bands.jl).  In the gauge Φ = 0 a stationary state rotates wherever the charge-imbalance potential is not zero
(the conversion layers); the explicit step of that rotation leaves a radial error ∝ ω²dt, which KWT multiplies by
√(1 + γ̃²|Ψ|²) (×30 with γ̃ = 30).  With `rotframe` those states are fixed points of the step with any dt
(six digits at dt, dt/2, dt/4 and with Euler substeps, γ̃ = 0, 10, 30).  `rkl_margin`: the RKL stages cover the bound times
this factor.  They are counted at the equilibrium amplitude P, and with γ̃ = 30 the π band of the two-band notched box at κ = 2 gets 14 stages
for a ratio of 104.8 (0.2 % to spare): any |Ψπ| above P (here the rotation error at the contacts) made its fastest phase
mode unstable.  1.1 costs at most a stage per band.  Defaults: both on (rotframe, 1.1) when any band has KWT, off
(false, 1) otherwise, so that every run without KWT keeps its bits.
`qp_exchange = ν` (two bands; with N bands the matrix of rates ν_µν, `QPExchangeN`): the quasiparticles of each band have their own potential and exchange at the rate ν
(σ → π, per t_GL), `qp_split` the share of σn of each band's normal fluid (default: the dirty one, ∝ gΓ/β); `nothing`
(default) is one normal fluid, the exchange infinitely fast: the step of today, to the bit (exchange.jl).  With the insulating
padding the fluids live in the conductor (zero flux to the insulator), in a sculpted sample with the links to its vacuum cut (conjugate
gradients); with the Joule heat the two fluids add σn Σ s_µ|∇μ_µ|² and the exchange Σ G_µν(μ_µ − μ_ν)² to σn|E|².
`noise = T̃`: thermal (Langevin) noise on every band at the noise temperature T̃ in the units of the functional, from the
seeded stream `noise_seed` (noise.jl; with KWT, the Ψ-dependent mobility and its Itô drift); `nothing` (default): none.
With noise, a KWT band steps in log-polar form (its increments, the frame term and the combination of its stages act on the
phase as rotations, bands.jl) and, with RKC stages (which it then takes even where RKL would need as many), gets the phase
kick inside the first stage (SK-ROCK, noise.jl).  Without noise, or without KWT, nothing changes.
`noise_field = true`: the same bath on the field, a kick on every link phase before the implicit solve (noise.jl).
`materials = [m₂, m₃, …]`: the models of the materials 2, 3, … that the regions of the shape name (`mk.mat`; materials.jl), with
the κ, σn and T of `md` and `transparency` the symmetric matrix of the barriers between them (default all 1): per node the band
coefficients of its material, per link the harmonic mean of the stiffnesses; the bounds are the smallest over the materials.
`nothing` (default): one material, the step of today to the bit.
"""
function Stepper(md::Model{N}, gr::Grid, mk::Masks; implicit::Bool=true,
                 substeps::NTuple{N,Int}=ntuple(µ -> µ == 1 ? 1 : 0, N),
                 dt_scale::Real=0.5, extrap::Bool=true, hard::Bool=false,
                 threads::Bool=Threads.nthreads() > 1, σvac::Union{Nothing,Real}=nothing,
                 pcg_tol::Real=1e-8, heat::Union{Nothing,Heat}=nothing,
                 drag::Union{Nothing,Real,AbstractMatrix}=nothing,
                 selffield_bc::Union{Symbol,AbstractString}=:bar, rkl::Union{Bool,AbstractString,Symbol}="rkc",
                 open_every::Real=1.0, rotframe::Union{Nothing,Bool}=nothing,
                 rkl_margin::Union{Nothing,Real}=nothing, qp_exchange::Union{Nothing,Real,AbstractMatrix}=nothing,
                 qp_split=nothing, noise::Union{Nothing,Real}=nothing, noise_seed::Integer=1,
                 noise_field::Bool=false, rkc_damping::Real=RKC_EPS,
                 materials::Union{Nothing,AbstractVector}=nothing, transparency::Union{Nothing,AbstractMatrix}=nothing,
                 absent=nothing, qp_rates::Union{Nothing,AbstractVector}=nothing, pcg_fast::Bool=true, pcg_f32::Bool=true,
                 pcg_extrap::Bool=true, heat_DT::Union{Nothing,AbstractVector}=nothing) where N
    selffield_bc = Symbol(selffield_bc)
    # rkl: true (RKL1 stages), false (Euler substeps), "rkc" (damped RKC1 stages: the same step, fewer stages)
    rkc = rkl isa Bool ? false : Symbol(rkl) === :rkc ? true : Symbol(rkl) === :rkl ? false :
          error("rkl must be true, false, \"rkl\" or \"rkc\", not $(repr(rkl))")
    rkl = rkl isa Bool ? rkl : true
    0 < dt_scale <= 1 || error("dt_scale must be in (0, 1], not $dt_scale")
    # with KWT on any band the rotating frame and a stage margin are on by default (without them the
    # state of the explicit step depends on dt in the conversion layers, and γ̃ = 30 diverged); without KWT both stay off and every path keeps its bits
    kwt = any(b -> b.γkwt > 0, md.bands) || (materials !== nothing && any(m -> any(b -> b.γkwt > 0, m.bands), materials))
    rotframe = something(rotframe, kwt)
    rkl_margin = Float64(something(rkl_margin, kwt ? 1.1 : 1.0))
    rkl_margin >= 1 || error("rkl_margin must be >= 1")
    rotframe && !extrap && error("rotframe needs extrap = true (it turns copies of the links)")
    all(>=(0), substeps) || error("substeps must be >= 0")
    implicit && !(gr.dx == gr.dy == gr.dz) &&
        error("the implicit curl-curl needs dx = dy = dz (separable solve)")
    l = dl(gr)
    bands = md.bands
    mk.mat !== nothing && materials === nothing &&
        error("the shape names materials (material = \"name\"): pass materials (the [materials.<name>] tables of a run file)")
    mats = materials === nothing ? nothing : _check_materials(md, materials, mk, heat, drag, qp_exchange, noise, absent)
    abs_m = mats === nothing ? nothing : _absent_matrix(absent, length(mats), N)
    bb0, bbk, λL = _band_bounds(md, gr, l)
    bsc_m = Inf                                       # the supercurrent bound of the other materials (σn per region)
    if mats !== nothing
        # the bound of each band is the smallest over the materials (the first alone gives today's bits); a band absent from a
        # material (a one-band metal) adds no bound there
        for (q, m) in enumerate(mats[2:end])
            pr = ntuple(µ -> !abs_m[q + 1, µ], N)
            b0, bk, λ = _band_bounds(m, gr, l; present=pr)
            bb0 = min.(bb0, b0); bbk = min.(bbk, bk)
            m.σn == md.σn ? (λL = max(λL, λ)) : λ > 0 && (bsc_m = min(bsc_m, 2 * m.σn / λ))   # (no condensate: no bound)
        end
    end
    σvar = mats !== nothing && any(m -> m.σn != md.σn, mats)   # σn per region: the field solve takes the PCG path
    # with rkl the step is the one of the same model without KWT, and RKL stages bridge every band
    # to it (below); without, the KWT bounds set it
    bb = rkl ? bb0 : bbk
    pad = σvac !== nothing && Float64(σvac) != md.σn
    pad && !(0 < σvac <= md.σn) && error("σvac must be in (0, σn]")
    bfield = implicit ? Inf : (pad ? Float64(σvac) : σvar ? minimum(m -> m.σn, mats) : md.σn) * l / (4 * md.κ^2)
    bsc = σvar ? min(2 * md.σn / λL, bsc_m) : 2 * md.σn / λL
    # automatic substeps: against the bounds that do not substep
    ref = min(bfield, bsc, minimum(ntuple(µ -> substeps[µ] == 0 ? Inf : bb[µ] * max(substeps[µ], 1), N)))
    nsub = ntuple(N) do µ
        substeps[µ] == 0 ? max(1, ceil(Int, dt_scale * ref / bb[µ] - 1e-12)) : substeps[µ]
    end
    bbn = ntuple(µ -> nsub[µ] == 1 && substeps[µ] == 1 ? bb[µ] : bb[µ] * (nsub[µ] / dt_scale), N)
    cands = (bbn..., bfield, bsc)
    names = (ntuple(µ -> "band $µ", N)..., "field (explicit curl-curl)", "explicit supercurrent")
    dt = 0.99 * minimum(cands)
    dt_scale == 1 || (dt *= dt_scale)
    # explicit diffusion of the temperature field
    DTmax = heat === nothing ? 0.0 : heat_DT === nothing ? heat.D_T : max(heat.D_T, maximum(heat_DT))   # (D_T per material)
    if heat !== nothing && DTmax > 0
        dt_dif = 0.15 * min(gr.dx, gr.dy, gr.dz)^2 / DTmax
        dt_dif < dt && (dt = dt_dif; cands = (cands..., dt_dif); names = (names..., "heat diffusion"))
    end
    limit = names[argmin(collect(cands))]

    rx = dt / gr.dx^2; ry = dt / gr.dy^2; rz = dt / gr.dz^2
    bc = ntuple(µ -> _bandcoef(md, µ, dt, rx, ry, rz, bands[µ].ma), N)
    # RKL1 (Meyer, Balsara & Aslam, J. Comput. Phys. 257, 594 (2014)): s stages advance a band dt with
    # Euler sub-steps of dt/q, q = s(s+1)/2, each within its explicit bound (as a substep is).  The
    # linearised band operator is (KWT factor, symmetric positive per node) × (Hessian of F): real
    # spectrum, what the method needs.
    haskwt(µ) = bands[µ].γkwt > 0 || (materials !== nothing && any(m -> m.bands[µ].γkwt > 0, materials))
    stg = ntuple(N) do µ
        rkl || return 0
        r = rkl_margin * dt / (0.99 * bbk[µ])
        r <= 1 + 1e-12 && return 0
        s = 1
        while s * (s + 1) ÷ 2 < r - 1e-12; s += 1; end
        if rkc                                           # RKC only where it saves a stage; elsewhere RKL and its bits
            q = 2
            while _rkc_ratio(q; ε=Float64(rkc_damping)) < r - 1e-12; q += 1; end
            # (the sign marks the band for RKC, undone below); with noise a KWT band takes RKC even at the same count (SK-ROCK)
            (q < s || (noise !== nothing && haskwt(µ))) && (s = -q)
        end
        s
    end
    usec = ntuple(µ -> stg[µ] < 0, N)
    # noise on a KWT band with RKC stages: SK-ROCK (noise.jl); with RKL stages the kick stays after the step: a warning
    skb = ntuple(µ -> noise !== nothing && rkc && usec[µ] && haskwt(µ), N)
    stg = abs.(stg)
    nsub = ntuple(µ -> stg[µ] > 0 ? 1 : nsub[µ], N)
    # noise with KWT: RKL stages keep the kick after the step, which over-heats the stiff phase modes and biases the equilibrium
    # 10–25 % low at T̃ = 1e-2 without a sign: an error (no earlier run depends on it: they diverged); the Euler substeps
    # are biased as much as RKC with SK-ROCK at the production step (π −7 % against +10 % at T̃ = 1e-2) and converge with dt:
    # a warning
    any(µ -> haskwt(µ) && !skb[µ] && stg[µ] > 0, 1:N) && noise !== nothing &&
        error("noise with KWT on RKL stages: the equilibrium comes out biased (10–25 % low at T̃ = 1e-2). Use rkl = \"rkc\" " *
              "(the default: the noise inside the stages, SK-ROCK) or rkl = false (Euler substeps)")
    if noise !== nothing && any(µ -> haskwt(µ) && !skb[µ] && nsub[µ] > 1, 1:N)
        @warn "noise with KWT on Euler substeps: the kick after the step over-heats the stiff phase modes (⟨|Ψπ|²⟩ −7 % at " *
              "T̃ = 1e-2 on the two-band box at the production step; it converges as dt resolves them). rkl = \"rkc\" (the default) " *
              "puts the noise inside the stages (SK-ROCK)." maxlog=1
    end
    rkcc = rkc ? ntuple(µ -> usec[µ] ? RKCCoef(stg[µ]; ε=Float64(rkc_damping)) : nothing, N) : nothing
    divisor(µ) = stg[µ] > 0 ? (rkcc !== nothing && rkcc[µ] !== nothing ? rkcc[µ].q : stg[µ] * (stg[µ] + 1) ÷ 2) : nsub[µ]
    bcs = ntuple(N) do µ
        stg[µ] > 0 ? _divided(bc[µ], divisor(µ)) : nsub[µ] == 1 ? bc[µ] : _divided(bc[µ], nsub[µ])
    end
    # (as Float64: an RKL band divides by an Int and an RKC band by a Float64, and the tuple must be one type)
    mt = mats === nothing ? nothing : MatField(mats, mk, gr, dt, rx, ry, rz, ntuple(µ -> stg[µ] > 0 || nsub[µ] > 1 ? Float64(divisor(µ)) : 1.0, N);
                                              absent=abs_m, heat_DT=heat === nothing ? nothing :
                                                  (heat_DT === nothing ? fill(heat.D_T, length(mats)) : heat_DT),
                                              transparency=transparency)
    w = ntuple(µ -> stiffness(bands[µ]), N)
    wa = ntuple(i -> ntuple(µ -> stiffness(bands[µ], i), N), 3)
    dtσ = dt / md.σn
    ks2 = md.κ^2
    fc = FieldCoef(dtσ, (dtσ * ks2) * (gr.dx / gr.dy), (dtσ * ks2) * (gr.dx / gr.dz),
                   (dtσ * ks2) * (gr.dy / gr.dx), (dtσ * ks2) * (gr.dy / gr.dz),
                   (dtσ * ks2) * (gr.dz / gr.dx), (dtσ * ks2) * (gr.dz / gr.dy),
                   dtσ * ks2 / gr.dx^2)
    sculpted = mk.vac !== nothing
    if sculpted
        pad || error("a sculpted sample (a shape with vacuum) needs the insulating padding: σvac (sigma_vac in [scheme]) below σn")
        heat === nothing || error("a sculpted sample: not with the heat model yet (its diffusion does not see the free surface)")
        drag === nothing || error("a sculpted sample: not with the drag yet (its stencil does not see the free surface)")
        Symbol(selffield_bc) in (:bar, :open) || error("a sculpted sample needs selffield_bc = :bar or :open")
    end
    D = σvar ? material_conductivity(gr, mk, mats, pad ? Float64(σvac) : nothing, md.σn; cond=sculpted ? conducting_nodes(gr, mk) : nothing) :
        pad ? link_conductivity(gr, Float64(σvac), md.σn; cond=sculpted ? conducting_nodes(gr, mk) : nothing) : nothing
    imp = !implicit ? nothing : (pad || σvar) ? PCGSolver(gr, fc, D; tol=Float64(pcg_tol), fast=pcg_fast, f32=pcg_f32,
                                                                        extrap=pcg_extrap) :
          SepSolver(gr, fc.a)
    sz = (gr.Nx + 1, gr.Ny + 1, gr.Nz + 1)
    ws = Workspace(any(>(1), nsub) || any(>(0), stg) ? zeros(ComplexF64, sz) : zeros(ComplexF64, 0, 0, 0),
                   (zeros(ComplexF64, gr.Nx, gr.Ny+1, gr.Nz+1), zeros(ComplexF64, gr.Nx+1, gr.Ny, gr.Nz+1),
                    zeros(ComplexF64, gr.Nx+1, gr.Ny+1, gr.Nz), zeros(ComplexF64, gr.Nx, gr.Ny+1, gr.Nz+1),
                    zeros(ComplexF64, gr.Nx+1, gr.Ny, gr.Nz+1), zeros(ComplexF64, gr.Nx+1, gr.Ny+1, gr.Nz)),
                   any(>(0), stg) ? zeros(ComplexF64, sz) : zeros(ComplexF64, 0, 0, 0),
                   rotframe ? zeros(sz) : zeros(0, 0, 0),
                   rotframe && noise !== nothing ? zeros(sz) : zeros(0, 0, 0),
                   rotframe ? (zeros(gr.Nx, gr.Ny+1, gr.Nz+1), zeros(gr.Nx+1, gr.Ny, gr.Nz+1), zeros(gr.Nx+1, gr.Ny+1, gr.Nz)) :
                              (zeros(0, 0, 0), zeros(0, 0, 0), zeros(0, 0, 0)),
                   any(skb) ? [zeros(sz) for _ in 1:N] : A3r[], any(skb) ? [zeros(ComplexF64, sz) for _ in 1:N] : A3c[])
    hf = heat === nothing ? nothing : HeatField(heat, fill(md.T, sz), md.T, zeros(3))
    νm = drag === nothing ? nothing : drag isa Real ? (N == 2 ? [0.0 drag; drag 0.0] :
         error("drag: a number is for two bands; give the matrix")) : Float64.(drag)
    if νm !== nothing
        size(νm) == (N, N) && νm == transpose(νm) || error("drag must be a symmetric $N×$N matrix")
    end
    dragν = νm === nothing ? nothing : ntuple(µ -> ntuple(ν -> νm[µ, ν], N), N)
    dragc = νm === nothing ? nothing : ntuple(N) do µ
        Γh = dt / (dt / bands[µ].Γ)
        ntuple(ν -> _drag_coef(νm[µ, ν], rx, ry, rz, Γh), N)
    end
    selffield_bc in (:slab, :bar, :open) || error("selffield_bc must be :slab, :bar or :open")
    bar = selffield_bc in (:bar, :open) ? bar_field(gr; section=sculpted ? _section(gr, mk) : nothing) : nothing
    ob = selffield_bc === :open ? OpenBC(gr, md.κ, bar; every=max(1, round(Int, open_every / dt))) : nothing
    # with the insulating padding the quasiparticles live in the bar of the sample and its contacts; a sculpted sample cuts
    # the links to its vacuum (exchange.jl)
    qcond = sculpted ? conducting_nodes(gr, mk) : nothing
    qpx = qp_exchange === nothing ? nothing :
          qp_exchange isa Real ? QPExchange(md, gr; ν=qp_exchange, split=qp_split, conductor=pad, cond=qcond, mt=mt, mk=mk,
                                            rates=qp_rates) :
          QPExchangeN(md, gr; ν=qp_exchange, split=qp_split, conductor=pad, cond=qcond)
    noise_field && noise === nothing && error("noise_field needs noise = T̃ (the same bath for Ψ and A)")
    # the heat source is the dissipation Γ|∂ₜΨ|² + σ|E|² of the step, and with noise it counts the Langevin kicks (∝ T̃/dt): a sample
    # in equilibrium heats by itself, more at a smaller step; a fluctuating heat model
    # consistent with T̃ is another model
    noise !== nothing && heat !== nothing &&
        error("noise with the heat model: the heat source counts the Langevin kicks as dissipation (a sample in equilibrium heats " *
              "by itself, more at a smaller step); not combined")
    nz = noise === nothing ? nothing : Noise(noise, noise_seed; field=noise_field)
    Stepper{N,typeof(imp)}(md, gr, mk, dt, nsub, stg, bc, bcs, w, wa, fc, imp, D, bar, ob, extrap, rotframe, hard, threads,
                           hf, dragc, dragν, ws, limit, Float64(dt_scale), rkl_margin, qpx, nz, rkcc, mt)
end
