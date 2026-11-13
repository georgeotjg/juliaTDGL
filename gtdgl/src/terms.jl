# ============================================================================
#  terms.jl — optional terms: heat and interband drag (from release 1.2, off by default)
#
#  Heat: a temperature field theta(r, t) above the bath, eps_mu = a_mu - theta on every node.
#      d_t theta = D_T lap theta + eta [ sum_mu Gamma_mu |d_t Psi_mu|^2 + sigma_n |E|^2 ] - gamma (theta - T)
#  (implicit in the bath term, explicit in the rest).  The source is the dissipation of the
#  functional, not j^2 (which would heat a Meissner state).  `margin` cells next to the injection
#  faces are not counted: there the normal current converts into supercurrent and E != 0 even in
#  the silent state (contact heat, which leads would remove).  The cap theta = 1.5 is a numerical
#  brake: a run that reaches it is not quoted.  `joule = false` reproduces the heat runs of release 1.x
#  made before the Joule term existed.
#
#  Drag: the mixed-gradient term nu_{mu nu} Re[(D Psi_mu)^* . D Psi_nu] of the functional
#  (Silaev-Babaev): band mu feels nu_{mu nu} D^2 Psi_nu, and the supercurrent carries
#  nu_{mu nu} Im(Psi_mu^* D Psi_nu + Psi_nu^* D Psi_mu).
# ============================================================================

"""
    Heat(; D_T=0.25, gamma_bath=0.2, eta=0.05, joule=false, margin=0)

The temperature field (coefficients in t_GL and ξ₁).
"""
Base.@kwdef struct Heat
    D_T::Float64 = 0.25
    gamma_bath::Float64 = 0.2
    eta::Float64 = 0.05
    joule::Bool = false
    margin::Int = 0
    cap::Float64 = 1.5
    cut::Float64 = 0.45          # a ladder point with max θ above this is cut ('T')
end

mutable struct HeatField
    h::Heat
    θ::A3r
    T::Float64                    # bath
    diag::Vector{Float64}         # mean dissipation of Ψ, nodes at the cap, mean |E|²
end

# drag coefficients of band µ on band ν for one step: (ν_µν rx/Γ̂_µ, ν_µν ry/Γ̂_µ, ν_µν rz/Γ̂_µ)
_drag_coef(ν::Float64, rx, ry, rz, Γh) = ((ν * rx) / Γh, (ν * ry) / Γh, (ν * rz) / Γh)

# The power a band dissipates at a node over the step (the dissipation of the functional, so that a stationary state does
# not heat): Γ|Ψ̇|² in TDGL, and with KWT (Γ/s)[|Ψ̇|² + γ̃²(Re Ψ*Ψ̇)²], s = √(1 + γ̃²|Ψ|²): the band equation times Ψ̇*.  The KWT
# operator slows the amplitude by s and leaves the phase at Γ/s, so Γ|Ψ̇|² overcounted the heat of a turning phase by s (×10 at
# γ̃ = 10 in a flowing state).  Ψ at the start of the step; with γ̃ = 0 the expression of before, to the bit.
@inline function _band_heat(Γ::Float64, γk::Float64, g::ComplexF64, f::ComplexF64, invdt2::Float64)
    γk == 0 && return Γ * (abs2(f - g) * invdt2)
    γ2 = γk * γk; d = f - g; r = real(conj(g) * d)
    (Γ / sqrt(1 + γ2 * abs2(g))) * ((abs2(d) + γ2 * (r * r)) * invdt2)
end

"""
    heat_step!(s, st)

One step of the temperature field.  Runs after the band update and the supercurrent: F − G is
the change of Ψ over this step and θ − φ (after the swap of the previous step) the change of the
link phases, so E_x = Δθ_x/(dx dt).
"""
function heat_step!(s::State{N}, st) where N
    hf = st.heat
    hp = hf.h; T = hf.θ; gr = st.grid; dt = st.dt
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dx2 = dx^2; dy2 = dy^2; dz2 = dz^2
    γ = hp.gamma_bath; η = hp.eta; θb = hf.T
    den = 1 + dt * γ
    invdt2 = 1 / (dt * dt)
    st.mt === nothing || return _heat_step_mat!(s, st)
    σn = st.model.σn
    Γ = ntuple(µ -> st.model.bands[µ].Γ, N); γk = ntuple(µ -> st.model.bands[µ].γkwt, N)
    @inbounds for k in z1:z2+1, j in y1:y2+1
        T[x1, j, k] = T[x1+1, j, k]; T[x2+1, j, k] = T[x2, j, k]
    end
    @inbounds for k in z1:z2+1, i in x1:x2+1
        T[i, y1, k] = T[i, y1+1, k]; T[i, y2+1, k] = T[i, y2, k]
    end
    @inbounds for j in y1:y2+1, i in x1:x2+1
        T[i, j, z1] = T[i, j, z1+1]; T[i, j, z2+1] = T[i, j, z2]
    end
    nclamp = 0; sumd = 0.0; sumE = 0.0; cnt = 0
    m = hp.margin
    lo = x1 + m; hi = x2 + 1 - m
    cx = invdt2 / dx2; cy = invdt2 / dy2; cz = invdt2 / dz2
    f = st.masks.f
    @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        q = T[i,j,k]
        lap = (T[i+1,j,k] - 2q + T[i-1,j,k]) / dx2 + (T[i,j+1,k] - 2q + T[i,j-1,k]) / dy2 +
              (T[i,j,k+1] - 2q + T[i,j,k-1]) / dz2
        dentro = lo < i < hi
        src = 0.0
        if dentro
            src = _band_heat(Γ[1], γk[1], s.G[1][i,j,k], s.F[1][i,j,k], invdt2)
            for µ in 2:N
                src += _band_heat(Γ[µ], γk[µ], s.G[µ][i,j,k], s.F[µ][i,j,k], invdt2)
            end
        end
        sumd += src; cnt += 1
        if hp.joule
            e2 = 0.0
            if dentro && f[i,j,k] > 0
                ex = 0.5 * (abs2(s.θx[i-1,j,k] - s.φx[i-1,j,k]) + abs2(s.θx[i,j,k] - s.φx[i,j,k])) * cx
                ey = 0.5 * (abs2(s.θy[i,j-1,k] - s.φy[i,j-1,k]) + abs2(s.θy[i,j,k] - s.φy[i,j,k])) * cy
                ez = 0.5 * (abs2(s.θz[i,j,k-1] - s.φz[i,j,k-1]) + abs2(s.θz[i,j,k] - s.φz[i,j,k])) * cz
                e2 = ex + ey + ez
            end
            sumE += e2
            # two normal fluids: the heat of their potential gradients and of the exchange (exchange.jl, qp_joule)
            qj = (st.qpx !== nothing && dentro && f[i,j,k] > 0) ? qp_joule(st.qpx, i, j, k, gr) : 0.0
            acc = q + dt * (hp.D_T * lap + η * ((src + σn * e2) + qj))
        else
            acc = q + dt * (hp.D_T * lap + η * src)
        end
        θn = (acc + dt * γ * θb) / den
        if θn < 0
            θn = 0.0
        elseif θn > hp.cap
            θn = hp.cap; nclamp += 1
        end
        T[i,j,k] = θn
    end
    hf.diag[1] = cnt == 0 ? 0.0 : sumd / cnt
    hf.diag[2] = Float64(nclamp)
    hf.diag[3] = cnt == 0 ? 0.0 : sumE / cnt
    s
end

# With materials: the dissipation with each node's Γ, the Joule heat with its σn, and D_T per material (the harmonic mean
# on each link: the heat flux crosses an interface in series).  The same operations as heat_step! otherwise.
function _heat_step_mat!(s::State{N}, st) where N
    hf = st.heat; mt = st.mt
    hp = hf.h; T = hf.θ; gr = st.grid; dt = st.dt
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dx2 = dx^2; dy2 = dy^2; dz2 = dz^2
    γ = hp.gamma_bath; η = hp.eta; θb = hf.T
    den = 1 + dt * γ
    invdt2 = 1 / (dt * dt)
    DT = mt.DT; idx = mt.idx
    hm(a, b) = a == b ? a : 2a * b / (a + b)
    @inbounds for k in z1:z2+1, j in y1:y2+1
        T[x1, j, k] = T[x1+1, j, k]; T[x2+1, j, k] = T[x2, j, k]
    end
    @inbounds for k in z1:z2+1, i in x1:x2+1
        T[i, y1, k] = T[i, y1+1, k]; T[i, y2+1, k] = T[i, y2, k]
    end
    @inbounds for j in y1:y2+1, i in x1:x2+1
        T[i, j, z1] = T[i, j, z1+1]; T[i, j, z2+1] = T[i, j, z2]
    end
    nclamp = 0; sumd = 0.0; sumE = 0.0; cnt = 0
    m = hp.margin
    lo = x1 + m; hi = x2 + 1 - m
    cx = invdt2 / dx2; cy = invdt2 / dy2; cz = invdt2 / dz2
    f = st.masks.f
    uni = all(==(hp.D_T), DT)                 # one D_T: today's expression and in-place sweep (and their bits)
    R = uni ? T : copy(T)                     # D_T per material: the neighbours of the step before (Jacobi), so that the
                                              # antisymmetric link fluxes conserve the heat content to rounding
    @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        q = R[i,j,k]
        flux = if uni
            hp.D_T * ((T[i+1,j,k] - 2q + T[i-1,j,k]) / dx2 + (T[i,j+1,k] - 2q + T[i,j-1,k]) / dy2 +
                      (T[i,j,k+1] - 2q + T[i,j,k-1]) / dz2)
        else
            d0 = DT[idx[i,j,k]]
            (hm(d0, DT[idx[i+1,j,k]]) * (R[i+1,j,k] - q) + hm(d0, DT[idx[i-1,j,k]]) * (R[i-1,j,k] - q)) / dx2 +
            (hm(d0, DT[idx[i,j+1,k]]) * (R[i,j+1,k] - q) + hm(d0, DT[idx[i,j-1,k]]) * (R[i,j-1,k] - q)) / dy2 +
            (hm(d0, DT[idx[i,j,k+1]]) * (R[i,j,k+1] - q) + hm(d0, DT[idx[i,j,k-1]]) * (R[i,j,k-1] - q)) / dz2
        end
        md = mt.models[idx[i,j,k]]
        dentro = lo < i < hi
        src = 0.0
        if dentro
            for µ in 1:N
                src += _band_heat(md.bands[µ].Γ, md.bands[µ].γkwt, s.G[µ][i,j,k], s.F[µ][i,j,k], invdt2)
            end
        end
        sumd += src; cnt += 1
        if hp.joule
            e2 = 0.0
            if dentro && f[i,j,k] > 0
                ex = 0.5 * (abs2(s.θx[i-1,j,k] - s.φx[i-1,j,k]) + abs2(s.θx[i,j,k] - s.φx[i,j,k])) * cx
                ey = 0.5 * (abs2(s.θy[i,j-1,k] - s.φy[i,j-1,k]) + abs2(s.θy[i,j,k] - s.φy[i,j,k])) * cy
                ez = 0.5 * (abs2(s.θz[i,j,k-1] - s.φz[i,j,k-1]) + abs2(s.θz[i,j,k] - s.φz[i,j,k])) * cz
                e2 = ex + ey + ez
            end
            sumE += e2
            qj = (st.qpx !== nothing && dentro && f[i,j,k] > 0) ? qp_joule(st.qpx, i, j, k, gr) : 0.0
            acc = q + dt * (flux + η * ((src + md.σn * e2) + qj))
        else
            acc = q + dt * (flux + η * src)
        end
        θn = (acc + dt * γ * θb) / den
        if θn < 0
            θn = 0.0
        elseif θn > hp.cap
            θn = hp.cap; nclamp += 1
        end
        T[i,j,k] = θn                            # in place, as heat_step!
    end
    hf.diag[1] = cnt == 0 ? 0.0 : sumd / cnt
    hf.diag[2] = Float64(nclamp)
    hf.diag[3] = cnt == 0 ? 0.0 : sumE / cnt
    s
end
