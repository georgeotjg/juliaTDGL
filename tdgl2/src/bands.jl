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

# Material per region (materials.jl): `mt` is nothing (one material: the coefficients of cf) or (mat, MatCoef, ρ) of the band.
@inline _coefs(::Nothing, cf::BandCoef, i, j, k) = (cf.kx, cf.ky, cf.kz, cf.c, cf.ε, cf.β, cf.cpl, cf.γ2)
@inline function _coefs(mt::Tuple, cf::BandCoef, i, j, k)
    @inbounds begin
        m = mt[1][i,j,k]; t = mt[2]
        (t.kx[m], t.ky[m], t.kz[m], t.c[m], t.ε[m], t.β[m], t.cpl[m], t.γ2[m])
    end
end
# The diffusion along one axis: today's expression, and with the link weights ρ₊, ρ₋ its exact extension (ρ = 1 adds 0).
@inline _dif(::Nothing, ρp, ρm, k, vp, sp, vm, sm, g) = k*((vp*sp - 2*g) + vm*sm)
@inline _dif(::Tuple, ρp, ρm, k, vp, sp, vm, sm, g) =
    k*((vp*sp - 2*g) + vm*sm) + k*((ρp - 1)*(vp*sp - g) + (ρm - 1)*(vm*sm - g))
@inline _anode(::Nothing, a, i, j, k) = a                        # the band's a for the temperature field: the node's material
@inline _anode(mt::Tuple, a, i, j, k) = @inbounds mt[4][mt[1][i,j,k]]
@inline _rho(::Nothing, ax, i, j, k) = 1.0
@inline _rho(mt::Tuple, ax, i, j, k) = @inbounds mt[3][ax][i,j,k]

# One row of band µ: D <- update of S (this band's source), coupling to the other bands of G.
@inline function _band_row!(D::A3c, S::A3c, G::NTuple{N,A3c}, µ::Int, Vx, Vy, Vz,
                            cf::BandCoef{N}, f::A3r, hard::Bool, x1::Int, x2::Int,
                            j::Int, k::Int, a::Float64, θ, dr, mt=nothing, rk::Bool=false) where N
    # one material: the coefficients of cf, read once per row (with materials, per node below)
    kx, ky, kz, c, ε, β, cpl, γ2 = cf.kx, cf.ky, cf.kz, cf.c, cf.ε, cf.β, cf.cpl, cf.γ2
    @inbounds for i in x1+1:x2
        g = S[i,j,k]; fm = f[i,j,k]
        if hard && fm == 0.0
            D[i,j,k] = 0.0 + 0.0im
            continue
        end
        mt === nothing || ((kx, ky, kz, c, ε, β, cpl, γ2) = _coefs(mt, cf, i, j, k))
        vxm = conj(Vx[i-1,j,k]); vym = conj(Vy[i,j-1,k]); vzm = conj(Vz[i,j,k-1])
        tx = _dif(mt, _rho(mt, 1, i, j, k), _rho(mt, 1, i-1, j, k), kx, Vx[i,j,k], S[i+1,j,k], vxm, S[i-1,j,k], g)
        ty = _dif(mt, _rho(mt, 2, i, j, k), _rho(mt, 2, i, j-1, k), ky, Vy[i,j,k], S[i,j+1,k], vym, S[i,j-1,k], g)
        tz = _dif(mt, _rho(mt, 3, i, j, k), _rho(mt, 3, i, j, k-1), kz, Vz[i,j,k], S[i,j,k+1], vzm, S[i,j,k-1], g)
        nl = (c*g)*(fm*_epsn(ε, _anode(mt, a, i, j, k), θ, i, j, k) - β*abs2(g))
        acc = (((g + tx) + ty) + tz) + nl
        for ν in 1:N
            ν == µ && continue
            acc += cpl[ν]*G[ν][i,j,k]
        end
        acc = _drag(dr, G, µ, Vx, Vy, Vz, vxm, vym, vzm, i, j, k, acc)
        D[i,j,k] = γ2 > 0 ? (rk ? kwt_update_rot(g, acc - g, γ2) : kwt_update(g, acc - g, γ2)) : acc
    end
    nothing
end

# The row of a sculpted sample: a vacuum node stays at Ψ = 0, and each neighbour enters with the
# mask of its link (τ = 0 across the free surface: the zero-flux condition, as the Neumann ghost of a
# face gives).  No drag and no heat field here (the Stepper refuses them with vacuum).
# With materials, τ is the band's ρ (τ times the weight of the link) and the coefficients are the node's.
@inline function _band_row_vac!(D::A3c, S::A3c, G::NTuple{N,A3c}, µ::Int, Vx, Vy, Vz,
                                cf::BandCoef{N}, f::A3r, hard::Bool, vac::BitArray{3}, τx::A3r, τy::A3r, τz::A3r,
                                x1::Int, x2::Int, j::Int, k::Int, mt=nothing, rk::Bool=false) where N
    # one material: the coefficients of cf, read once per row (with materials, per node below)
    kx, ky, kz, c, ε, β, cpl, γ2 = cf.kx, cf.ky, cf.kz, cf.c, cf.ε, cf.β, cf.cpl, cf.γ2
    @inbounds for i in x1+1:x2
        g = S[i,j,k]; fm = f[i,j,k]
        if vac[i,j,k] || (hard && fm == 0.0)
            D[i,j,k] = 0.0 + 0.0im
            continue
        end
        mt === nothing || ((kx, ky, kz, c, ε, β, cpl, γ2) = _coefs(mt, cf, i, j, k))
        tx = kx*(τx[i,j,k]*(Vx[i,j,k]*S[i+1,j,k] - g) + τx[i-1,j,k]*(conj(Vx[i-1,j,k])*S[i-1,j,k] - g))
        ty = ky*(τy[i,j,k]*(Vy[i,j,k]*S[i,j+1,k] - g) + τy[i,j-1,k]*(conj(Vy[i,j-1,k])*S[i,j-1,k] - g))
        tz = kz*(τz[i,j,k]*(Vz[i,j,k]*S[i,j,k+1] - g) + τz[i,j,k-1]*(conj(Vz[i,j,k-1])*S[i,j,k-1] - g))
        acc = (((g + tx) + ty) + tz) + (c*g)*(fm*ε - β*abs2(g))
        for ν in 1:N
            ν == µ && continue
            acc += cpl[ν]*G[ν][i,j,k]
        end
        D[i,j,k] = γ2 > 0 ? (rk ? kwt_update_rot(g, acc - g, γ2) : kwt_update(g, acc - g, γ2)) : acc
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

"""
    kwt_update_rot(ψ, Δ, γ2) -> ψ_new

As `kwt_update`, with the increment's part across ψ applied as a rotation and its part along ψ as a change of the amplitude:
ψ_new = (|ψ| + r) e^{i t/|ψ|} ψ/|ψ|, with r + i t = ψ̂* Δ′ (Δ′ the KWT increment).  The same to first order; the tangent step
ψ + iεψ lengthens |ψ| by ε² instead.  Used with noise: a noisy state is rough in its phase at the lattice scale, the KWT
factor multiplies the tangential increments by s = √(1+γ̃²|Ψ|²), and the ε² of every stage raised |Ψ| steadily (1.0 → 1.45 in
145 steps of the minimal case) until the stages no longer covered the phase mode (s beyond the bound) and the run gave NaN.
"""
@inline function kwt_update_rot(g::ComplexF64, d::ComplexF64, γ2::Float64)
    a2 = abs2(g)
    q = 1 + γ2*a2
    dp = sqrt(q)*d
    Δ = dp - (γ2*g)*(real(conj(g)*dp)/q)
    a2 > 1e-24 || return g + Δ
    a = sqrt(a2); u = g / a; w = conj(u) * Δ
    u * ((a + real(w)) * cis(imag(w) / a))
end

function _band_pass!(D::A3c, S::A3c, s::State{N}, µ::Int, Vx, Vy, Vz, cf::BandCoef{N},
                     st::Stepper{N}, dr) where N
    gr = st.grid; f = st.masks.f; hard = st.hard; G = s.G
    a = st.model.bands[µ].a
    θ = st.heat === nothing ? nothing : st.heat.θ
    mk = st.masks
    rk = _kwt_rot(st)                             # with noise the KWT increment's tangential part is a rotation
    if st.mt !== nothing
        # the material tables of this (sub)stage: the divided ones when cf is the band's stage coefficients
        mt = (st.mt.idx, cf === st.bcs[µ] ? st.mt.stage[µ] : st.mt.full[µ], st.mt.ρ[µ], st.mt.a[µ])
        if mk.vac === nothing
            _rows(gr.y1+1:gr.y2, gr.z1+1:gr.z2, st.threads) do j, k
                _band_row!(D, S, G, µ, Vx, Vy, Vz, cf, f, hard, gr.x1, gr.x2, j, k, a, θ, dr, mt, rk)
            end
        else
            ρx, ρy, ρz = mt[3]
            _rows(gr.y1+1:gr.y2, gr.z1+1:gr.z2, st.threads) do j, k
                _band_row_vac!(D, S, G, µ, Vx, Vy, Vz, cf, f, hard, mk.vac, ρx, ρy, ρz, gr.x1, gr.x2, j, k, mt, rk)
            end
        end
        # a band absent from a material (a one-band metal) stays at Ψ = 0 there; its links into it carry nothing (ρ = 0)
        @inbounds for c in st.mt.absnodes[µ]; D[c] = 0.0 + 0.0im; end
    elseif mk.vac === nothing
        _rows(gr.y1+1:gr.y2, gr.z1+1:gr.z2, st.threads) do j, k
            _band_row!(D, S, G, µ, Vx, Vy, Vz, cf, f, hard, gr.x1, gr.x2, j, k, a, θ, dr, nothing, rk)
        end
    else
        _rows(gr.y1+1:gr.y2, gr.z1+1:gr.z2, st.threads) do j, k
            _band_row_vac!(D, S, G, µ, Vx, Vy, Vz, cf, f, hard, mk.vac, mk.τx, mk.τy, mk.τz, gr.x1, gr.x2, j, k, nothing, rk)
        end
    end
    D
end

_dragof(st::Stepper, µ::Int, n::Real) = st.drag === nothing ? nothing :
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
_frame_passes(st) = st.noise === nothing ? 0 : 2   # with noise the rate is low-passed
_kwt_rot(st) = st.noise !== nothing                 # with noise the KWT increment is applied as a rotation

function _frame_rate!(Ω::A3r, C::A3c, Y::A3c, h::Float64, dt::Float64, gr::Grid; passes::Int=0, buf::A3r=Ω)
    passes > 0 && return _frame_rate_smooth!(Ω, C, Y, h, dt, gr, passes, buf)
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
# With noise: the rate of the frame low-passed.  The first stage's tangential rate of a noisy state is lattice-scale noise,
# multiplied by the KWT factor (it reached the cap 0.5/dt), and through its 1/h dependence on the state it fed back into the
# phase at the lattice scale: the linearised step about a noisy state had an eigenvalue −1.46 (a lattice-scale phase mode), the
# stages cannot carry it, and the run gave NaN within 35 steps.  Any bounded ω̂ is an exact change of gauge, so the
# frame may follow a smooth ω̂: `passes` sweeps of Ω ← (Ω + mean of the 6 neighbours)/2 (the checkerboard mode to 0, a smooth
# field to O(dx²∇²Ω)) before the frame term of the first stage.  Without noise passes = 0: today's rate and its bits.
function _frame_rate_smooth!(Ω::A3r, C::A3c, Y::A3c, h::Float64, dt::Float64, gr::Grid, passes::Int, T::A3r)
    (; x1, x2, y1, y2, z1, z2) = gr
    wmax = _ROT_MAXDT / dt
    @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        y = Y[i,j,k]; a2 = abs2(y)
        Ω[i,j,k] = a2 > _ROT_MIN2 ? clamp(imag(conj(y) * (C[i,j,k] - y)) / (a2 * h), -wmax, wmax) : 0.0
    end
    faces!(Ω) = begin
        @inbounds for k in z1+1:z2, j in y1+1:y2; Ω[x1,j,k] = Ω[x1+1,j,k]; Ω[x2+1,j,k] = Ω[x2,j,k]; end
        @inbounds for k in z1+1:z2, i in x1:x2+1; Ω[i,y1,k] = Ω[i,y1+1,k]; Ω[i,y2+1,k] = Ω[i,y2,k]; end
        @inbounds for j in y1:y2+1, i in x1:x2+1; Ω[i,j,z1] = Ω[i,j,z1+1]; Ω[i,j,z2+1] = Ω[i,j,z2]; end
        Ω
    end
    faces!(Ω)
    for _ in 1:passes
        copyto!(T, Ω)
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            nb = (T[i+1,j,k] + T[i-1,j,k]) + (T[i,j+1,k] + T[i,j-1,k]) + (T[i,j,k+1] + T[i,j,k-1])
            Ω[i,j,k] = 0.5 * T[i,j,k] + nb / 12
        end
        faces!(Ω)
    end
    # the frame term as the rotation it stands for (with noise the increments are rotations too, kwt_update_rot): C ← C e^{−iω̂h}.
    # (A linear −iω̂hY followed by the exact turn back e^{iω̂dt} leaves a radial error ∝ ω̂h × the increment's own phase step, once
    # the increment itself is a rotation: on the Euler step of the one-band box at γ̃ = 10, T̃ = 1e-2 it lowered ⟨|Ψ|²⟩ by a third.)
    @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
        C[i,j,k] *= cis(-(Ω[i,j,k] * h))
    end
    Ω
end

function _frame_term!(C::A3c, S::A3c, Ω::A3r, h::Float64, gr::Grid; turn::Bool=false)
    (; x1, x2, y1, y2, z1, z2) = gr
    if turn                                   # with noise: as a rotation (see _frame_rate_smooth!)
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            C[i,j,k] *= cis(-(Ω[i,j,k] * h))
        end
        return C
    end
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
                _frame_rate!(st.ws.Ω, s.F[µ], s.G[µ], st.dt, st.dt, st.grid; passes=_frame_passes(st), buf=st.ws.Ωs)
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
        rot && (m == 1 ? _frame_rate!(Ω, dst, src, h, st.dt, gr; passes=_frame_passes(st), buf=st.ws.Ωs) : _frame_term!(dst, src, Ω, h, gr; turn=_kwt_rot(st)))
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

# With noise on a KWT band: the stages combined in log-polar form, Y_j = A·exp(μ log(C/A)) with C = E(Y_{j−1}) and A = Y_{j−2}
# (μ + ν = 1 in RKL1 and in RKC1), which is μC + νA to first order in C − A.  The linear combination of two values of one modulus
# and different phase, with ν < 0, lies outside their circle (|μC + νA|² = |A|²(1 + 2|μν|(1 − cos φ))); on a noisy state the
# KWT factor makes the phase increments of a stage large at the lattice scale, the amplitude is the slow direction (mobility
# 1/(Γs)), and the three-term recursion carries an error of a slow mode on with a factor ~ the stages left: |Ψπ| of the two-band
# box rose from its 0.77 to 2.4 in 19 steps, and faster with more stages.  In log form the phase combines linearly,
# as in the linear polynomial.  Where |A|² or |C|² is under _ROT_MIN2 (vortex cores, the normal metal), the linear combination.
function _combine_log!(C::A3c, A::A3c, μ::Float64, gr::Grid)
    @inbounds for k in gr.z1+1:gr.z2, j in gr.y1+1:gr.y2, i in gr.x1+1:gr.x2
        a = A[i,j,k]; c = C[i,j,k]
        C[i,j,k] = abs2(a) > _ROT_MIN2 && abs2(c) > _ROT_MIN2 ? a * exp(μ * log(c / a)) : μ * c + (1 - μ) * a
    end
    C
end
_skrock(st, µ) = !isempty(st.ws.Qθ) && st.rkc !== nothing && st.rkc[µ] !== nothing && st.rkl[µ] > 0 && _haskwt(st, µ)
function _sk_turn!(D::A3c, X::A3c, Qθ::A3r, c::Float64, gr::Grid)                # D ← X e^{i c δθ}
    @inbounds for k in gr.z1+1:gr.z2, j in gr.y1+1:gr.y2, i in gr.x1+1:gr.x2
        q = Qθ[i,j,k]
        D[i,j,k] = q == 0.0 ? X[i,j,k] : X[i,j,k] * cis(c * q)
    end
    D
end
_haskwt(st, µ) = st.model.bands[µ].γkwt > 0 || (st.mt !== nothing && any(m -> m.bands[µ].γkwt > 0, st.mt.models))
_logcomb(st, µ) = st.noise !== nothing && _haskwt(st, µ)

"""
RKL1 for band µ with `ns` stages: Y₀ = Ψⁿ, Y₁ = E(Y₀), Y_j = μ_j E(Y_{j−1}) + ν_j Y_{j−2}, with
E the Euler step of dt/q (q = ns(ns+1)/2), μ_j = (2j−1)/j, ν_j = (1−j)/j; Y_ns = Ψⁿ⁺¹.  Stage j is
at the fraction c_j = j(j+1)/(ns(ns+1)) of the step: the links and the ghosts of Y_j are those of c_j.
With `st.rkc` (rkl = "rkc") the same passes with the weights of damped RKC1 (`RKCCoef`): E of dt/q, q = ω₀/ω₁, and
μ_j, ν_j, c_j from the Chebyshev recursion.
"""
function _band_rkl!(s::State{N}, st::Stepper{N}, µ::Int, ns::Int) where N
    gr = st.grid
    cf = st.bcs[µ]
    kc = st.rkc === nothing ? nothing : st.rkc[µ]
    q = kc === nothing ? Float64(ns * (ns + 1) ÷ 2) : kc.q     # (one type: the drag divides by it)
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
    lg = _logcomb(st, µ); skq = _skrock(st, µ)
    for j in 1:ns
        C = buf[mod1(j, 3)]
        if j == 1 && skq                     # SK-ROCK for the phase (noise.jl): K₁ = E(Ψⁿ e^{iν₁δθ}) e^{i(κ₁−ν₁)δθ}
            Sp = st.ws.W2; Qθ = st.ws.Qθ[µ]; ν1, κ1 = kc.sk
            _sk_turn!(Sp, B, Qθ, ν1, gr); ghosts!(Sp, Lx, Ly, Lz, gr)
            _band_pass!(C, Sp, s, µ, Lx, Ly, Lz, cf, st, dr)
            rot && _frame_rate!(Ω, C, Sp, h, st.dt, gr; passes=_frame_passes(st), buf=st.ws.Ωs)
            _sk_turn!(C, C, Qθ, κ1 - ν1, gr)
        else
            _band_pass!(C, B, s, µ, Lx, Ly, Lz, cf, st, dr)
            rot && (j == 1 ? _frame_rate!(Ω, C, B, h, st.dt, gr; passes=_frame_passes(st), buf=st.ws.Ωs) : _frame_term!(C, B, Ω, h, gr; turn=_kwt_rot(st)))
        end
        if rot && j == 1 && st.extrap && ns > 1
            Rx, Ry, Rz = st.ws.R
            _link_rate!(Rx, s.Vx, s.Ux, Ω, st.dt, _rx(gr), CartesianIndex(1, 0, 0))
            _link_rate!(Ry, s.Vy, s.Uy, Ω, st.dt, _ry(gr), CartesianIndex(0, 1, 0))
            _link_rate!(Rz, s.Vz, s.Uz, Ω, st.dt, _rz(gr), CartesianIndex(0, 0, 1))
        end
        if j >= 2
            μj, νj = kc === nothing ? ((2j - 1) / j, (1 - j) / j) : (kc.μ[j], kc.ν[j])
            lg ? _combine_log!(C, A, μj, gr) : _combine!(C, A, μj, νj, gr)
        end
        if j < ns
            if st.extrap
                c = kc === nothing ? j * (j + 1) / (ns * (ns + 1)) : kc.c[j]
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
