# ============================================================================
#  bands.jl — the order-parameter update of every band
#
#  Jacobi: reads G (the current step) and writes F.  Interior nodes only; the faces are set
#  by the covariant Neumann condition (neumann! in field.jl).  Per node and band
#      F = G + k_x(V G[+x] - 2G + V'^* G[-x]) + (y) + (z) + (dt/Γ) G (f ε - β|G|²)
#            + sum_nu (dt/Γ) γ_{mu nu} G_nu
#  in that order of additions.  A band with n substeps is advanced n times with dt/n, the
#  other bands frozen at the start of the step in the coupling, and the links rotated at the
#  rate of the previous step: in the gauge Phi = 0 the links turn wherever E != 0 (the contact
#  layers, even in the silent state), and freezing them lags the band behind that rotation.
#  With rotframe the band itself turns too: it is stepped in the gauge Ψ = e^{iω̂τ}Ψ̃ (τ from the
#  start of the step, ω̂ its rate there) and turned back at the end, so that the rotation of a
#  stationary state is exact instead of an explicit step (see _frame_rate!).
# ============================================================================

# Rows (j, k) of the interior, optionally over threads.  No reductions: same bits with any
# number of threads.
@inline function _rows(f::F, js::UnitRange{Int}, ks::UnitRange{Int}, thr::Bool) where F
    if thr
        nj = length(js); j0 = first(js); k0 = first(ks)
        Threads.@threads :static for q in 0:nj*length(ks)-1
            f(j0 + q % nj, k0 + q ÷ nj)
        end
    else
        for k in ks, j in js
            f(j, k)
        end
    end
    nothing
end

# Condensation coefficient of a node: uniform (ε) or from the temperature field (a − θ).
@inline _epsn(ε::Float64, a::Float64, ::Nothing, i, j, k) = ε
@inline _epsn(ε::Float64, a::Float64, θ::A3r, i, j, k) = @inbounds a - θ[i,j,k]

# Drag of the other bands on this one: Σ_ν c_ν·(lattice D² of Ψ_ν), each axis summed as in 1.2.
@inline _drag(::Nothing, G, µ, Vx, Vy, Vz, vxm, vym, vzm, i, j, k, acc) = acc
@inline function _drag(dr::NTuple{N,NTuple{3,Float64}}, G::NTuple{N,A3c}, µ, Vx, Vy, Vz,
                       vxm, vym, vzm, i, j, k, acc) where N
    @inbounds for ν in 1:N
        ν == µ && continue
        cx, cy, cz = dr[ν]
        P = G[ν]; p = P[i,j,k]
        ex = cx*((Vx[i,j,k]*P[i+1,j,k] - 2*p) + vxm*P[i-1,j,k])
        ey = cy*((Vy[i,j,k]*P[i,j+1,k] - 2*p) + vym*P[i,j-1,k])
        ez = cz*((Vz[i,j,k]*P[i,j,k+1] - 2*p) + vzm*P[i,j,k-1])
        acc += (ex + ey) + ez
    end
    acc
end

# One row of band µ: D <- update of S (this band's source), coupling to the other bands of G.
@inline function _band_row!(D::A3c, S::A3c, G::NTuple{N,A3c}, µ::Int, Vx, Vy, Vz,
                            cf::BandCoef{N}, f::A3r, hard::Bool, x1::Int, x2::Int,
                            j::Int, k::Int, a::Float64, θ, dr) where N
    kx, ky, kz, c, ε, β, cpl, γ2 = cf.kx, cf.ky, cf.kz, cf.c, cf.ε, cf.β, cf.cpl, cf.γ2
    kwt = γ2 > 0
    @inbounds for i in x1+1:x2
        g = S[i,j,k]; fm = f[i,j,k]
        if hard && fm == 0.0
            D[i,j,k] = 0.0 + 0.0im
            continue
        end
        vxm = conj(Vx[i-1,j,k]); vym = conj(Vy[i,j-1,k]); vzm = conj(Vz[i,j,k-1])
        tx = kx*((Vx[i,j,k]*S[i+1,j,k] - 2*g) + vxm*S[i-1,j,k])
        ty = ky*((Vy[i,j,k]*S[i,j+1,k] - 2*g) + vym*S[i,j-1,k])
        tz = kz*((Vz[i,j,k]*S[i,j,k+1] - 2*g) + vzm*S[i,j,k-1])
        nl = (c*g)*(fm*_epsn(ε, a, θ, i, j, k) - β*abs2(g))
        acc = (((g + tx) + ty) + tz) + nl
        for ν in 1:N
            ν == µ && continue
            acc += cpl[ν]*G[ν][i,j,k]
        end
        acc = _drag(dr, G, µ, Vx, Vy, Vz, vxm, vym, vzm, i, j, k, acc)
        D[i,j,k] = kwt ? kwt_update(g, acc - g, γ2) : acc
    end
    nothing
end

# The row of a sculpted sample: a vacuum node stays at Ψ = 0, and each neighbour enters with the
# mask of its link (τ = 0 across the free surface: the zero-flux condition, as the Neumann ghost of a
# face gives).  No drag and no heat field here (the Stepper refuses them with vacuum).
@inline function _band_row_vac!(D::A3c, S::A3c, G::NTuple{N,A3c}, µ::Int, Vx, Vy, Vz,
                                cf::BandCoef{N}, f::A3r, hard::Bool, vac::BitArray{3}, τx::A3r, τy::A3r, τz::A3r,
                                x1::Int, x2::Int, j::Int, k::Int) where N
    kx, ky, kz, c, ε, β, cpl, γ2 = cf.kx, cf.ky, cf.kz, cf.c, cf.ε, cf.β, cf.cpl, cf.γ2
    kwt = γ2 > 0
    @inbounds for i in x1+1:x2
        g = S[i,j,k]; fm = f[i,j,k]
        if vac[i,j,k] || (hard && fm == 0.0)
            D[i,j,k] = 0.0 + 0.0im
            continue
        end
        tx = kx*(τx[i,j,k]*(Vx[i,j,k]*S[i+1,j,k] - g) + τx[i-1,j,k]*(conj(Vx[i-1,j,k])*S[i-1,j,k] - g))
        ty = ky*(τy[i,j,k]*(Vy[i,j,k]*S[i,j+1,k] - g) + τy[i,j-1,k]*(conj(Vy[i,j-1,k])*S[i,j-1,k] - g))
        tz = kz*(τz[i,j,k]*(Vz[i,j,k]*S[i,j,k+1] - g) + τz[i,j,k-1]*(conj(Vz[i,j,k-1])*S[i,j,k-1] - g))
        acc = (((g + tx) + ty) + tz) + (c*g)*(fm*ε - β*abs2(g))
        for ν in 1:N
            ν == µ && continue
            acc += cpl[ν]*G[ν][i,j,k]
        end
        D[i,j,k] = kwt ? kwt_update(g, acc - g, γ2) : acc
    end
    nothing
end

"""
    kwt_update(ψ, Δ, γ2) -> ψ_new

Kramer–Watts-Tobin step of one node: from the plain step Δ = (dt/Γ)R,
Δ′ = √(1+γ̃²|ψ|²) Δ and ψ_new = ψ + Δ′ − γ̃²ψ Re(ψ*Δ′)/(1+γ̃²|ψ|²), the solution of
w + γ̃²ψ Re(ψ* w) = √(1+γ̃²|ψ|²) R/Γ without dividing by ψ (regular in the cores).
"""
@inline function kwt_update(g::ComplexF64, d::ComplexF64, γ2::Float64)
    a2 = abs2(g)
    q = 1 + γ2*a2
    dp = sqrt(q)*d
    g + (dp - (γ2*g)*(real(conj(g)*dp)/q))
end

function _band_pass!(D::A3c, S::A3c, s::State{N}, µ::Int, Vx, Vy, Vz, cf::BandCoef{N},
                     st::Stepper{N}, dr) where N
    gr = st.grid; f = st.masks.f; hard = st.hard; G = s.G
    a = st.model.bands[µ].a
    θ = st.heat === nothing ? nothing : st.heat.θ
    mk = st.masks
    if mk.vac === nothing
        _rows(gr.y1+1:gr.y2, gr.z1+1:gr.z2, st.threads) do j, k
            _band_row!(D, S, G, µ, Vx, Vy, Vz, cf, f, hard, gr.x1, gr.x2, j, k, a, θ, dr)
        end
    else
        _rows(gr.y1+1:gr.y2, gr.z1+1:gr.z2, st.threads) do j, k
            _band_row_vac!(D, S, G, µ, Vx, Vy, Vz, cf, f, hard, mk.vac, mk.τx, mk.τy, mk.τz, gr.x1, gr.x2, j, k)
        end
    end
    D
end

_dragof(st::Stepper, µ::Int, n::Int) = st.drag === nothing ? nothing :
    (n == 1 ? st.drag[µ] : map(t -> (t[1] / n, t[2] / n, t[3] / n), st.drag[µ]))

# ---- rotating frame (rotframe) -----------------------------------------------------------
# In the gauge Φ = 0 a stationary state is not a fixed point of the step: wherever the charge-imbalance
# potential Q is not zero (the conversion layers at the contacts) Ψ turns at ω = Q.  The explicit step of
# that rotation, |1 + iωh| > 1, leaves a radial error ∝ ω²dt, and KWT slows the radial mode by √(1+γ̃²|Ψ|²)
# while it leaves the rotation alone: the amplitude in the layers goes up by ~√(1+γ̃²|Ψ|²)ω²dt/4 (0.08 at
# γ̃ = 30, 6e-5 without KWT, on the small box at J = 0.6).  In the gauge χ = −ω̂τ the band equation gains
# −iω̂Ψ̃ (tangential, so the KWT factor leaves it alone) and the links gain e^{i(ω̂ᵢ₊₁−ω̂ᵢ)τ}; a stationary
# state is then a fixed point of every stage, and Ψⁿ⁺¹ = e^{iω̂dt}Ψ̃ turns it exactly.  ω̂ is the tangential
# rate of the first (sub)stage of the step, so no history is kept; any bounded ω̂ is an exact change of
# gauge, so the cap and the threshold only decide where the frame helps.
const _ROT_MIN2 = 1e-3       # |Ψ|² under which ω̂ = 0 (vortex cores, the normal metal)
const _ROT_MAXDT = 0.5       # |ω̂| dt is capped there

function _frame_rate!(Ω::A3r, C::A3c, Y::A3c, h::Float64, dt::Float64, gr::Grid)
    (; x1, x2, y1, y2, z1, z2) = gr
    wmax = _ROT_MAXDT / dt
    @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        y = Y[i,j,k]; a2 = abs2(y)
        w = a2 > _ROT_MIN2 ? clamp(imag(conj(y) * (C[i,j,k] - y)) / (a2 * h), -wmax, wmax) : 0.0
        Ω[i,j,k] = w
        C[i,j,k] -= im * (w * h) * y
    end
    # the faces take the rate of the node they copy (as the Neumann ghosts, precedence x, y, z)
    @inbounds for k in z1+1:z2, j in y1+1:y2
        Ω[x1,j,k] = Ω[x1+1,j,k]; Ω[x2+1,j,k] = Ω[x2,j,k]
    end
    @inbounds for k in z1+1:z2, i in x1:x2+1
        Ω[i,y1,k] = Ω[i,y1+1,k]; Ω[i,y2+1,k] = Ω[i,y2,k]
    end
    @inbounds for j in y1:y2+1, i in x1:x2+1
        Ω[i,j,z1] = Ω[i,j,z1+1]; Ω[i,j,z2+1] = Ω[i,j,z2]
    end
    Ω
end

# The term of the turning gauge in a later (sub)stage: C ← C − iω̂hS (S the stage's source).
function _frame_term!(C::A3c, S::A3c, Ω::A3r, h::Float64, gr::Grid)
    (; x1, x2, y1, y2, z1, z2) = gr
    @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        C[i,j,k] -= im * (Ω[i,j,k] * h) * S[i,j,k]
    end
    C
end

# e^{ia}: short series while |a| <= 0.05 (error <= 3e-8, as _rot!), exactly above.
@inline _cis(a::Float64) = abs(a) <= 0.05 ? (a2 = a*a; complex(1 - a2/2, a*(1 - a2/6))) : cis(a)

# Links in the turning gauge, τ later: L ← L e^{i(ω̂ᵢ₊₁ − ω̂ᵢ)τ}.
function _frame_links!(Lx, Ly, Lz, Ω::A3r, τ::Float64, gr::Grid)
    @inbounds for I in _rx(gr); i, j, k = Tuple(I); Lx[I] *= _cis((Ω[i+1,j,k] - Ω[i,j,k]) * τ); end
    @inbounds for I in _ry(gr); i, j, k = Tuple(I); Ly[I] *= _cis((Ω[i,j+1,k] - Ω[i,j,k]) * τ); end
    @inbounds for I in _rz(gr); i, j, k = Tuple(I); Lz[I] *= _cis((Ω[i,j,k+1] - Ω[i,j,k]) * τ); end
    nothing
end

# The two rotations of a link at the fraction c of the step are linear in c: its own (the previous step's, −arg q·c with
# q = V conj(U)) and the frame's ((ω̂ᵢ₊₁ − ω̂ᵢ)·c·dt).  Their sum per step, once per band after the first stage, makes
# each later stage one pass with one e^{ia}: L = V e^{iRc}.  (The two passes of _rotfrac! and _frame_links! agree with it
# to the truncation of their short series.)
function _link_rate!(R, V, U, Ω, dt, rng, o)
    @inbounds for I in rng
        q = V[I] * conj(U[I]); sf = -imag(q)
        a = abs(sf) <= 0.05 ? -(sf + sf*sf*sf/6) : angle(q)
        R[I] = a + (Ω[I + o] - Ω[I]) * dt
    end
    R
end
function _rotlinks!(L, V, R, c::Float64, rng)
    @inbounds for I in rng
        L[I] = V[I] * _cis(R[I] * c)
    end
    L
end

# Back to the gauge of the step: Ψⁿ⁺¹ = e^{iω̂dt}Ψ̃.  Exact cis here: the series is short of unit modulus by a⁴/24,
# a bias of the amplitude every step that KWT's slow radial mode would turn into ~1e-5 (the links are copies, redone
# every step, and can take the series).
function _frame_turn!(F::A3c, Ω::A3r, dt::Float64, gr::Grid)
    (; x1, x2, y1, y2, z1, z2) = gr
    @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        F[i,j,k] *= cis(Ω[i,j,k] * dt)
    end
    F
end

"""Advances every band one step: F_µ from G (and the links V)."""
function update_bands!(s::State{N}, st::Stepper{N}) where N
    for µ in 1:N
        n = st.nsub[µ]
        if st.rkl[µ] > 0
            _band_rkl!(s, st, µ, st.rkl[µ])
        elseif n == 1
            _band_pass!(s.F[µ], s.G[µ], s, µ, s.Vx, s.Vy, s.Vz, st.bc[µ], st, _dragof(st, µ, 1))
            if st.rotframe
                _frame_rate!(st.ws.Ω, s.F[µ], s.G[µ], st.dt, st.dt, st.grid)
                _frame_turn!(s.F[µ], st.ws.Ω, st.dt, st.grid)
            end
        else
            _band_substeps!(s, st, µ, n)
        end
    end
    s
end

# Link ranges the band stencil and its ghosts read.
_rx(gr::Grid) = CartesianIndices((gr.x1:gr.x2,   gr.y1:gr.y2+1, gr.z1:gr.z2+1))
_ry(gr::Grid) = CartesianIndices((gr.x1:gr.x2+1, gr.y1:gr.y2,   gr.z1:gr.z2+1))
_rz(gr::Grid) = CartesianIndices((gr.x1:gr.x2+1, gr.y1:gr.y2+1, gr.z1:gr.z2))

# Link of substep m = V Q^((m-1)/n), Q = V conj(U) = exp(-i dθ) the rotation of the previous
# step.  Q^(1/n) from a short series while |sin dθ| <= 0.05 (error <= 3e-8) and exactly above.
function _rot!(L, R, V, U, n::Int, rng)
    @inbounds for I in rng
        v = V[I]; L[I] = v
        q = v * conj(U[I])
        sf = -imag(q)
        if abs(sf) <= 0.05
            a = (sf + sf*sf*sf/6) / n
            a2 = a*a
            R[I] = complex(1 - a2/2, -a*(1 - a2/6))
        else
            R[I] = cis(angle(q) / n)
        end
    end
    L
end

function _advance_links!(Lx, Ly, Lz, Rx, Ry, Rz, gr::Grid)
    @inbounds for I in _rx(gr); Lx[I] *= Rx[I]; end
    @inbounds for I in _ry(gr); Ly[I] *= Ry[I]; end
    @inbounds for I in _rz(gr); Lz[I] *= Rz[I]; end
    nothing
end

function _band_substeps!(s::State{N}, st::Stepper{N}, µ::Int, n::Int) where N
    gr = st.grid
    cf = st.bcs[µ]
    F = s.F[µ]; W = st.ws.W
    if st.extrap
        Lx, Ly, Lz, Rx, Ry, Rz = st.ws.L
        _rot!(Lx, Rx, s.Vx, s.Ux, n, _rx(gr))
        _rot!(Ly, Ry, s.Vy, s.Uy, n, _ry(gr))
        _rot!(Lz, Rz, s.Vz, s.Uz, n, _rz(gr))
    else
        Lx, Ly, Lz = s.Vx, s.Vy, s.Vz
    end
    src = s.G[µ]
    dr = _dragof(st, µ, n)
    rot = st.rotframe; Ω = st.ws.Ω; h = st.dt / n
    for m in 1:n
        dst = iseven(n - m) ? F : W
        _band_pass!(dst, src, s, µ, Lx, Ly, Lz, cf, st, dr)
        rot && (m == 1 ? _frame_rate!(Ω, dst, src, h, st.dt, gr) : _frame_term!(dst, src, Ω, h, gr))
        if m < n
            st.extrap && _advance_links!(Lx, Ly, Lz, Rx, Ry, Rz, gr)
            rot && _frame_links!(Lx, Ly, Lz, Ω, h, gr)
            ghosts!(dst, Lx, Ly, Lz, gr)
        end
        src = dst
    end
    rot && _frame_turn!(F, Ω, st.dt, gr)
    F
end

# Links at the fraction c of the step: V Q^c, Q = V conj(U) (the rotation of the previous step).
function _rotfrac!(L, V, U, c::Float64, rng)
    @inbounds for I in rng
        v = V[I]
        q = v * conj(U[I])
        sf = -imag(q)
        if abs(sf) <= 0.05
            a = (sf + sf*sf*sf/6) * c
            a2 = a*a
            L[I] = v * complex(1 - a2/2, -a*(1 - a2/6))
        else
            L[I] = v * cis(angle(q) * c)
        end
    end
    L
end

function _combine!(C::A3c, A::A3c, μ::Float64, ν::Float64, gr::Grid)
    @inbounds for k in gr.z1+1:gr.z2, j in gr.y1+1:gr.y2, i in gr.x1+1:gr.x2
        C[i,j,k] = μ * C[i,j,k] + ν * A[i,j,k]
    end
    C
end

"""
RKL1 for band µ with `ns` stages: Y₀ = Ψⁿ, Y₁ = E(Y₀), Y_j = μ_j E(Y_{j−1}) + ν_j Y_{j−2}, with
E the Euler step of dt/q (q = ns(ns+1)/2), μ_j = (2j−1)/j, ν_j = (1−j)/j; Y_ns = Ψⁿ⁺¹.  Stage j is
at the fraction c_j = j(j+1)/(ns(ns+1)) of the step: the links and the ghosts of Y_j are those of c_j.
"""
function _band_rkl!(s::State{N}, st::Stepper{N}, µ::Int, ns::Int) where N
    gr = st.grid
    cf = st.bcs[µ]
    q = ns * (ns + 1) ÷ 2
    dr = _dragof(st, µ, q)
    if st.extrap
        Lx, Ly, Lz = st.ws.L[1], st.ws.L[2], st.ws.L[3]
        copyto!(Lx, s.Vx); copyto!(Ly, s.Vy); copyto!(Lz, s.Vz)
    else
        Lx, Ly, Lz = s.Vx, s.Vy, s.Vz
    end
    buf = (st.ws.W, st.ws.W2, s.F[µ])
    rot = st.rotframe; Ω = st.ws.Ω; h = st.dt / q
    A = s.G[µ]; B = s.G[µ]; C = buf[1]
    for j in 1:ns
        C = buf[mod1(j, 3)]
        _band_pass!(C, B, s, µ, Lx, Ly, Lz, cf, st, dr)
        rot && (j == 1 ? _frame_rate!(Ω, C, B, h, st.dt, gr) : _frame_term!(C, B, Ω, h, gr))
        if rot && j == 1 && st.extrap && ns > 1
            Rx, Ry, Rz = st.ws.R
            _link_rate!(Rx, s.Vx, s.Ux, Ω, st.dt, _rx(gr), CartesianIndex(1, 0, 0))
            _link_rate!(Ry, s.Vy, s.Uy, Ω, st.dt, _ry(gr), CartesianIndex(0, 1, 0))
            _link_rate!(Rz, s.Vz, s.Uz, Ω, st.dt, _rz(gr), CartesianIndex(0, 0, 1))
        end
        j >= 2 && _combine!(C, A, (2j - 1) / j, (1 - j) / j, gr)
        if j < ns
            if st.extrap
                c = j * (j + 1) / (ns * (ns + 1))
                if rot
                    Rx, Ry, Rz = st.ws.R
                    _rotlinks!(Lx, s.Vx, Rx, c, _rx(gr)); _rotlinks!(Ly, s.Vy, Ry, c, _ry(gr))
                    _rotlinks!(Lz, s.Vz, Rz, c, _rz(gr))
                else
                    _rotfrac!(Lx, s.Vx, s.Ux, c, _rx(gr)); _rotfrac!(Ly, s.Vy, s.Uy, c, _ry(gr))
                    _rotfrac!(Lz, s.Vz, s.Uz, c, _rz(gr))
                end
            end
            ghosts!(C, Lx, Ly, Lz, gr)
        end
        A = B; B = C
    end
    C === s.F[µ] || copyto!(s.F[µ], C)
    rot && _frame_turn!(s.F[µ], Ω, st.dt, gr)
    s.F[µ]
end

"""Covariant Neumann ghosts of one band on the six faces of the block (precedence x, y, z)."""
function ghosts!(X::A3c, Vx, Vy, Vz, gr::Grid)
    (; x1, x2, y1, y2, z1, z2) = gr
    @inbounds for k in z1:z2+1, j in y1:y2+1
        X[x1,j,k]   = Vx[x1,j,k] * X[x1+1,j,k]
        X[x2+1,j,k] = conj(Vx[x2,j,k]) * X[x2,j,k]
    end
    @inbounds for k in z1:z2+1, i in x1:x2+1
        X[i,y1,k]   = Vy[i,y1,k] * X[i,y1+1,k]
        X[i,y2+1,k] = conj(Vy[i,y2,k]) * X[i,y2,k]
    end
    @inbounds for j in y1:y2+1, i in x1:x2+1
        X[i,j,z1]   = Vz[i,j,z1] * X[i,j,z1+1]
        X[i,j,z2+1] = conj(Vz[i,j,z2]) * X[i,j,z2]
    end
    X
end
