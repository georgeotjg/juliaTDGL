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

_divided(b::BandCoef{N}, n::Int) where N =
    BandCoef{N}(b.kx / n, b.ky / n, b.kz / n, b.c / n, b.ε, b.β, ntuple(ν -> b.cpl[ν] / n, N), b.γ2)

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
    R::NTuple{3,A3r}                # rotframe: the phase rate of each link over a step, link and frame together
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
end

"""
    Stepper(md, gr, mk; implicit=true, substeps=auto, dt_scale=0.5, extrap=true, hard=false,
            threads=Threads.nthreads() > 1, σvac=nothing, pcg_tol=1e-11)

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
`rkl` (default): the band updates by first-order Runge–Kutta–Legendre stages instead of Euler
substeps; the step is then the one of the same model without KWT, and each band takes the stages that
bridge its own bound (with KWT) to it: s stages for a ratio up to s(s+1)/2.  On the ladders of the two-band
notched box at κ = 25 it gives the substeps' brackets and flowing voltages to 1e-4, 8 % faster; with KWT (γ̃ = 10)
4.5 times faster than the explicit step.  `rkl = false`: the Euler substeps of release 1.2.
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
`noise_field = true`: the same bath on the field, a kick on every link phase before the implicit solve (noise.jl).
"""
function Stepper(md::Model{N}, gr::Grid, mk::Masks; implicit::Bool=true,
                 substeps::NTuple{N,Int}=ntuple(µ -> µ == 1 ? 1 : 0, N),
                 dt_scale::Real=0.5, extrap::Bool=true, hard::Bool=false,
                 threads::Bool=Threads.nthreads() > 1, σvac::Union{Nothing,Real}=nothing,
                 pcg_tol::Real=1e-11, heat::Union{Nothing,Heat}=nothing,
                 drag::Union{Nothing,Real,AbstractMatrix}=nothing,
                 selffield_bc::Union{Symbol,AbstractString}=:bar, rkl::Bool=true,
                 open_every::Real=1.0, rotframe::Union{Nothing,Bool}=nothing,
                 rkl_margin::Union{Nothing,Real}=nothing, qp_exchange::Union{Nothing,Real,AbstractMatrix}=nothing,
                 qp_split=nothing, noise::Union{Nothing,Real}=nothing, noise_seed::Integer=1,
                 noise_field::Bool=false) where N
    selffield_bc = Symbol(selffield_bc)
    0 < dt_scale <= 1 || error("dt_scale must be in (0, 1], not $dt_scale")
    # with KWT on any band the rotating frame and a stage margin are on by default (without them the
    # state of the explicit step depends on dt in the conversion layers, and γ̃ = 30 diverged); without KWT both stay off and every path keeps its bits
    kwt = any(b -> b.γkwt > 0, md.bands)
    rotframe = something(rotframe, kwt)
    rkl_margin = Float64(something(rkl_margin, kwt ? 1.1 : 1.0))
    rkl_margin >= 1 || error("rkl_margin must be >= 1")
    rotframe && !extrap && error("rotframe needs extrap = true (it turns copies of the links)")
    all(>=(0), substeps) || error("substeps must be >= 0")
    implicit && !(gr.dx == gr.dy == gr.dz) &&
        error("the implicit curl-curl needs dx = dy = dz (separable solve)")
    l = dl(gr)
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
    # with rkl the step is the one of the same model without KWT, and RKL stages bridge every band
    # to it (below); without, the KWT bounds set it
    bb = rkl ? bb0 : bbk
    pad = σvac !== nothing && Float64(σvac) != md.σn
    pad && !(0 < σvac <= md.σn) && error("σvac must be in (0, σn]")
    bfield = implicit ? Inf : (pad ? Float64(σvac) : md.σn) * l / (4 * md.κ^2)
    λL = sum(ntuple(µ -> (isotropic(bands[µ]) ? stiffness(bands[µ]) : 1 / minimum(bands[µ].ma)) * (bands[µ].a / bands[µ].β), N))
    bsc = 2 * md.σn / λL
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
    if heat !== nothing && heat.D_T > 0
        dt_dif = 0.15 * min(gr.dx, gr.dy, gr.dz)^2 / heat.D_T
        dt_dif < dt && (dt = dt_dif; cands = (cands..., dt_dif); names = (names..., "heat diffusion"))
    end
    limit = names[argmin(collect(cands))]

    rx = dt / gr.dx^2; ry = dt / gr.dy^2; rz = dt / gr.dz^2
    bc = ntuple(N) do µ
        b = bands[µ]
        c = dt / b.Γ
        Γh = dt / c                                     # Γ as the step sees it (exactly 1 for Γ = 1)
        BandCoef{N}(rx / (b.ma[1] * Γh), ry / (b.ma[2] * Γh), rz / (b.ma[3] * Γh), c, eps_band(md, µ),
                    b.β, ntuple(ν -> (dt / Γh) * md.γ[µ][ν], N), b.γkwt^2)
    end
    # RKL1 (Meyer, Balsara & Aslam, J. Comput. Phys. 257, 594 (2014)): s stages advance a band dt with
    # Euler sub-steps of dt/q, q = s(s+1)/2, each within its explicit bound (as a substep is).  The
    # linearised band operator is (KWT factor, symmetric positive per node) × (Hessian of F): real
    # spectrum, what the method needs.
    stg = ntuple(N) do µ
        rkl || return 0
        r = rkl_margin * dt / (0.99 * bbk[µ])
        r <= 1 + 1e-12 && return 0
        s = 1
        while s * (s + 1) ÷ 2 < r - 1e-12; s += 1; end
        s
    end
    nsub = ntuple(µ -> stg[µ] > 0 ? 1 : nsub[µ], N)
    bcs = ntuple(N) do µ
        stg[µ] > 0 ? _divided(bc[µ], stg[µ] * (stg[µ] + 1) ÷ 2) : nsub[µ] == 1 ? bc[µ] : _divided(bc[µ], nsub[µ])
    end
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
    D = pad ? link_conductivity(gr, Float64(σvac), md.σn; cond=sculpted ? conducting_nodes(gr, mk) : nothing) : nothing
    imp = !implicit ? nothing : pad ? PCGSolver(gr, fc, D; tol=Float64(pcg_tol)) : SepSolver(gr, fc.a)
    sz = (gr.Nx + 1, gr.Ny + 1, gr.Nz + 1)
    ws = Workspace(any(>(1), nsub) || any(>(0), stg) ? zeros(ComplexF64, sz) : zeros(ComplexF64, 0, 0, 0),
                   (zeros(ComplexF64, gr.Nx, gr.Ny+1, gr.Nz+1), zeros(ComplexF64, gr.Nx+1, gr.Ny, gr.Nz+1),
                    zeros(ComplexF64, gr.Nx+1, gr.Ny+1, gr.Nz), zeros(ComplexF64, gr.Nx, gr.Ny+1, gr.Nz+1),
                    zeros(ComplexF64, gr.Nx+1, gr.Ny, gr.Nz+1), zeros(ComplexF64, gr.Nx+1, gr.Ny+1, gr.Nz)),
                   any(>(0), stg) ? zeros(ComplexF64, sz) : zeros(ComplexF64, 0, 0, 0),
                   rotframe ? zeros(sz) : zeros(0, 0, 0),
                   rotframe ? (zeros(gr.Nx, gr.Ny+1, gr.Nz+1), zeros(gr.Nx+1, gr.Ny, gr.Nz+1), zeros(gr.Nx+1, gr.Ny+1, gr.Nz)) :
                              (zeros(0, 0, 0), zeros(0, 0, 0), zeros(0, 0, 0)))
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
          qp_exchange isa Real ? QPExchange(md, gr; ν=qp_exchange, split=qp_split, conductor=pad, cond=qcond) :
          QPExchangeN(md, gr; ν=qp_exchange, split=qp_split, conductor=pad, cond=qcond)
    noise_field && noise === nothing && error("noise_field needs noise = T̃ (the same bath for Ψ and A)")
    nz = noise === nothing ? nothing : Noise(noise, noise_seed; field=noise_field)
    Stepper{N,typeof(imp)}(md, gr, mk, dt, nsub, stg, bc, bcs, w, wa, fc, imp, D, bar, ob, extrap, rotframe, hard, threads,
                           hf, dragc, dragν, ws, limit, Float64(dt_scale), rkl_margin, qpx, nz)
end
