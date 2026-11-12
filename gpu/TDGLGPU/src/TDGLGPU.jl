# ============================================================================
#  TDGLGPU — TDGL 2.0 kernels for KernelAbstractions (validated on CUDA and CPU)
#
#  This module follows the CPU solver's equations and integration stages. Dense
#  products use device matrices (cuBLAS on CUDA). Backend arithmetic, reductions
#  and the fast PCG can differ in rounding; test/validate.jl records each case's
#  tolerance and error. Float64 state and the optional Float32 preconditioner
#  match the CPU precision choices. The zero-source PCG update is shared.
#
#      gs = GPUStepper(st, backend)          # st: a TDGL.Stepper of a supported configuration (`supported(st)`)
#      g  = device_state(backend, s)         # s: a TDGL.State
#      gpu_steps!(g, gs, n; H, HJ)           # n steps on the device
#      host_state!(s, g)                     # back into s, for the observables and the protocol of the CPU code
#
#  Implemented: N bands, Euler/RKL/RKC, KWT, rotating frame, implicit/explicit field, insulating PCG (including its fast
#  and Float32 preconditioner paths), sculpted vacuum, regional materials, heat, drag, thermal noise (also KWT/SK-ROCK
#  and regional materials), and two or N normal fluids. CPU configuration restrictions still apply (for example N normal
#  fluids with materials and drag with materials). The open boundary recomputes its FFT on the host and ramps on device.
#  The ladder hook accelerates cycles; direct calls to TDGL.step! (including delay confirmation) remain on the CPU.
#  Implementation coverage is distinct from device validation: test/validate.jl records cases, backend and tolerances.
# ============================================================================
module TDGLGPU

using KernelAbstractions, LinearAlgebra
using TDGL
using TDGL: State, Stepper, BandCoef, Grid, SepSolver, PCGSolver
const KA = KernelAbstractions

export GPUStepper, GState, device_state, device_state!, host_state!, gpu_step!, gpu_steps!, supported, enable!, disable!, execution_stats

# ---------------------------------------------------------------------------
#  what this version runs
# ---------------------------------------------------------------------------
"""
    supported(st) -> (ok::Bool, why::String)

Whether `TDGLGPU` runs the step of this Stepper, and if not the first option it does not have yet.
"""
function supported(st::Stepper)
    if st.qpx !== nothing
        st.qpx isa TDGL.QPExchangeN && st.mt !== nothing && return (false, "N normal fluids with materials")
    end
    (true, "")
end

# ---------------------------------------------------------------------------
#  the state on the device: the fields of TDGL.State, as device arrays
# ---------------------------------------------------------------------------
mutable struct GState{N,C<:AbstractArray{ComplexF64,3},R<:AbstractArray{Float64,3}}
    G::NTuple{N,C}; F::NTuple{N,C}
    θx::R; θy::R; θz::R
    φx::R; φy::R; φz::R
    Ux::C; Uy::C; Uz::C
    Vx::C; Vy::C; Vz::C
    Jx::R; Jy::R; Jz::R
    hx::R; hy::R; hz::R
    bc::Vector{Float64}          # (H, HJ) the outer faces of h hold (NaN: not written on the device yet)
end
const _FIELDS = (:θx, :θy, :θz, :φx, :φy, :φz, :Ux, :Uy, :Uz, :Vx, :Vy, :Vz, :Jx, :Jy, :Jz, :hx, :hy, :hz)

_dev(backend, A::AbstractArray) = (B = KA.allocate(backend, eltype(A), size(A)); copyto!(B, A); B)

"""The state `s` copied to the device of `backend`."""
device_state(backend, s::State{N}) where N =
    GState(ntuple(µ -> _dev(backend, s.G[µ]), N), ntuple(µ -> _dev(backend, s.F[µ]), N),
           (_dev(backend, getfield(s, f)) for f in _FIELDS)..., [NaN, NaN])

"""The state `s` copied into the device state `g` (same sizes)."""
function device_state!(g::GState{N}, s::State{N}) where N
    for µ in 1:N; copyto!(g.G[µ], s.G[µ]); copyto!(g.F[µ], s.F[µ]); end
    for f in _FIELDS; copyto!(getfield(g, f), getfield(s, f)); end
    g.bc .= NaN
    g
end

"""The device state `g` copied back into `s` (every field: the observables and the checkpoints of the CPU code read it)."""
function host_state!(s::State{N}, g::GState{N}) where N
    KA.synchronize(KA.get_backend(g.G[1]))
    for µ in 1:N; copyto!(s.G[µ], g.G[µ]); copyto!(s.F[µ], g.F[µ]); end
    for f in _FIELDS; copyto!(getfield(s, f), getfield(g, f)); end
    s
end

# ---------------------------------------------------------------------------
#  the separable implicit solve, with device matrices (implicit.jl: _tx!, _sep_solve!, implicit_curl!)
# ---------------------------------------------------------------------------
struct GSep{M,V,R}
    Vx::M; Ux::M; sx::V
    Vy::M; Uy::M; sy::V
    Vz::M; Uz::M; sz::V
    a::Float64
    rx::NTuple{3,UnitRange{Int}}; ry::NTuple{3,UnitRange{Int}}; rz::NTuple{3,UnitRange{Int}}
    BX::R; BY::R; BZ::R
    fx::NTuple{3,R}; bx::NTuple{3,R}; fy::NTuple{3,R}; by::NTuple{3,R}; fz::NTuple{3,R}; bz::NTuple{3,R}
end
_ax3(backend, S) = (_dev(backend, S.A1), _dev(backend, S.A2), _dev(backend, S.A3))
GSep(backend, S::SepSolver) =
    GSep(_dev(backend, S.Vx), _dev(backend, S.Ux), _dev(backend, S.sx), _dev(backend, S.Vy), _dev(backend, S.Uy),
         _dev(backend, S.sy), _dev(backend, S.Vz), _dev(backend, S.Uz), _dev(backend, S.sz), S.a, S.rx, S.ry, S.rz,
         _dev(backend, S.BX), _dev(backend, S.BY), _dev(backend, S.BZ),
         _ax3(backend, S.fx), _ax3(backend, S.bx), _ax3(backend, S.fy), _ax3(backend, S.by), _ax3(backend, S.fz), _ax3(backend, S.bz))

# A transformed by L along dim 1 (L applied on the left: transpose(M1) of _tx!), M2 along dim 2 and M3 along dim 3: the
# products of _tx!, in its order
function _tx!(T, A, L, M2, M3, gs)
    n1, n2, n3 = size(A); m1, m2, m3 = size(L, 1), size(M2, 2), size(M3, 2)
    A1, A2, A3 = T
    mul!(reshape(A1, m1, n2*n3), L, reshape(A, n1, n2*n3))
    _dim2!(A2, A1, M2, gs)
    mul!(reshape(A3, m1*m2, m3), reshape(A2, m1*m2, n3), M3)
    A3
end
# A2[:, :, k] = A1[:, :, k] M2 for every k: on the CPU backend the products of _tx! (its bits); on a GPU one kernel instead
# of one GEMM per plane (n3 small GEMMs were most of the launches of a step)
@kernel function _dim2_k!(A2, @Const(A1), @Const(M), tr::Bool)
    i, j, k = @index(Global, NTuple)
    acc = zero(eltype(A2))
    @inbounds for l in 1:size(A1, 2)
        acc += A1[i,l,k] * (tr ? M[j,l] : M[l,j])
    end
    @inbounds A2[i,j,k] = acc
end
_unwrap(M::Union{Adjoint,Transpose}) = (parent(M), true)
_unwrap(M) = (M, false)
function _dim2!(A2, A1, M2, gs)
    if gs.backend isa KA.CPU
        for k in 1:size(A1, 3); mul!(view(A2, :, :, k), view(A1, :, :, k), M2); end
    else
        P, tr = _unwrap(M2)
        _launch(gs, _dim2_k!, size(A2), A2, A1, P, tr)
    end
    A2
end

@kernel function _modes!(X, Y, Z, @Const(sx), @Const(sy), @Const(sz), a)
    mx, my, mz = @index(Global, NTuple)
    @inbounds begin
        k1, k2, k3 = sx[mx], sy[my], sz[mz]
        k2s = k1^2 + k2^2 + k3^2
        if k2s != 0
            o = zero(k2s)
            x1 = mx > 1 ? X[mx-1, my, mz] : o
            y1 = my > 1 ? Y[mx, my-1, mz] : o
            z1 = mz > 1 ? Z[mx, my, mz-1] : o
            d = (k1*x1 + k2*y1 + k3*z1) / k2s; f = 1 / (1 + a*k2s)
            mx > 1 && (X[mx-1, my, mz] = k1*d + f*(x1 - k1*d))
            my > 1 && (Y[mx, my-1, mz] = k2*d + f*(y1 - k2*d))
            mz > 1 && (Z[mx, my, mz-1] = k3*d + f*(z1 - k3*d))
        end
    end
end

function _sep_solve!(S::GSep, gs)
    X = _tx!(S.fx, S.BX, transpose(S.Ux), S.Vy, S.Vz, gs)
    Y = _tx!(S.fy, S.BY, transpose(S.Vx), S.Uy, S.Vz, gs)
    Z = _tx!(S.fz, S.BZ, transpose(S.Vx), S.Vy, S.Uz, gs)
    _launch(gs, _modes!, (length(S.sx), length(S.sy), length(S.sz)), X, Y, Z, S.sx, S.sy, S.sz, S.a)
    _hsync(gs)
    # the back transform: transpose(M') = M for the real bases (what transpose(S.Ux') of the CPU code multiplies by)
    S.BX .= _tx!(S.bx, X, S.Ux, S.Vy', S.Vz', gs)
    S.BY .= _tx!(S.by, Y, S.Vx, S.Uy', S.Vz', gs)
    S.BZ .= _tx!(S.bz, Z, S.Vx, S.Vy', S.Uz', gs)
    nothing
end

function _implicit_curl!(g::GState, S::GSep, gs)
    _hsync(gs)
    @views begin
        S.BX .= g.φx[S.rx...] .- g.θx[S.rx...]
        S.BY .= g.φy[S.ry...] .- g.θy[S.ry...]
        S.BZ .= g.φz[S.rz...] .- g.θz[S.rz...]
    end
    _sep_solve!(S, gs)
    @views begin
        g.φx[S.rx...] .= g.θx[S.rx...] .+ S.BX
        g.φy[S.ry...] .= g.θy[S.ry...] .+ S.BY
        g.φz[S.rz...] .= g.θz[S.rz...] .+ S.BZ
    end
    g
end

# ---------------------------------------------------------------------------
#  the insulating padding: PCG with the separable preconditioner (implicit.jl: _apply_A!, _precond!, implicit_curl!,
#  _implicit_curl_fast!, _precond32!).  The flags of the warm start (warm, have2) are the CPU solver's own vectors, so
#  _pcg_reset! and _pcg_back! of the protocol act on both; x and xprev are copied with the state every cycle (_gpu_cycle!),
#  so the checkpoints of the CPU code hold them.
# ---------------------------------------------------------------------------
struct GFast{R,R32,M32,V32}
    xprev::NTuple{3,R}; have2::Vector{Bool}
    zold::NTuple{3,R}
    f32::Bool; extrap::Bool
    V::NTuple{9,M32}; W::NTuple{3,M32}; s32::NTuple{3,V32}
    B32::NTuple{3,R32}; T32::NTuple{6,R32}
end
struct GPCG{S,R,F}
    sep::S
    D::NTuple{3,R}
    s::NTuple{6,Float64}
    dx::Float64; dy::Float64; dz::Float64
    tol::Float64; maxit::Int
    P::NTuple{3,R}; Q::NTuple{3,R}; H::NTuple{3,R}
    x::NTuple{3,R}; r::NTuple{3,R}; z::NTuple{3,R}; b::NTuple{3,R}
    iters::Vector{Int}; warm::Vector{Bool}             # the CPU solver's vectors (shared)
    fast::F
end
_d3(backend, t) = ntuple(c -> _dev(backend, t[c]), length(t))
function GPCG(backend, S::PCGSolver)
    f = S.fast
    fg = f === nothing ? nothing :
         GFast(_d3(backend, f.xprev), f.have2, _d3(backend, f.zold), f.f32, f.extrap, _d3(backend, f.V), _d3(backend, f.W),
               _d3(backend, f.s32), _d3(backend, f.B32), _d3(backend, f.T32))
    GPCG(GSep(backend, S.sep), _d3(backend, S.D), S.s, S.dx, S.dy, S.dz, S.tol, S.maxit, _d3(backend, S.P),
         _d3(backend, S.Q), _d3(backend, S.H), _d3(backend, S.x), _d3(backend, S.r), _d3(backend, S.z), _d3(backend, S.b),
         S.iters, S.warm, fg)
end
# the warm-start state between the host solver and the device one
function _pcg_up!(G::GPCG, S::PCGSolver)
    for c in 1:3; copyto!(G.x[c], S.x[c]); end
    G.fast === nothing || for c in 1:3; copyto!(G.fast.xprev[c], S.fast.xprev[c]); end
end
function _pcg_down!(S::PCGSolver, G::GPCG)
    for c in 1:3; copyto!(S.x[c], G.x[c]); end
    G.fast === nothing || for c in 1:3; copyto!(S.fast.xprev[c], G.fast.xprev[c]); end
end

@kernel function _ahx_k!(hx, @Const(py), @Const(pz), dydz::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds hx[i,j,k] = ((pz[i,j+1,k] - pz[i,j,k]) - py[i,j,k+1] + py[i,j,k]) / dydz
end
@kernel function _ahy_k!(hy, @Const(px), @Const(pz), dzdx::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds hy[i,j,k] = ((px[i,j,k+1] - px[i,j,k]) - pz[i+1,j,k] + pz[i,j,k]) / dzdx
end
@kernel function _ahz_k!(hz, @Const(px), @Const(py), dxdy::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds hz[i,j,k] = ((py[i+1,j,k] - py[i,j,k]) - px[i,j+1,k] + px[i,j,k]) / dxdy
end
@kernel function _aqx_k!(qx, @Const(Dx), @Const(px), @Const(hy), @Const(hz), sxy::Float64, sxz::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds qx[i,j,k] = Dx[i,j,k]*px[i,j,k] + sxy*(hz[i,j,k] - hz[i,j-1,k]) - sxz*(hy[i,j,k] - hy[i,j,k-1])
end
@kernel function _aqy_k!(qy, @Const(Dy), @Const(py), @Const(hx), @Const(hz), syz::Float64, syx::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds qy[i,j,k] = Dy[i,j,k]*py[i,j,k] + syz*(hx[i,j,k] - hx[i,j,k-1]) - syx*(hz[i,j,k] - hz[i-1,j,k])
end
@kernel function _aqz_k!(qz, @Const(Dz), @Const(pz), @Const(hx), @Const(hy), szx::Float64, szy::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds qz[i,j,k] = Dz[i,j,k]*pz[i,j,k] + szx*(hy[i,j,k] - hy[i-1,j,k]) - szy*(hx[i,j,k] - hx[i,j-1,k])
end
function _apply_A!(S::GPCG, gs, Nx, Ny, Nz)
    px, py, pz = S.P; qx, qy, qz = S.Q; hx, hy, hz = S.H
    sxy, sxz, syx, syz, szx, szy = S.s
    Dx, Dy, Dz = S.D
    _launch(gs, _h3_k!, (Nx - 1, Ny - 1, Nz - 1), hx, hy, hz, px, py, pz, Nx, Ny, Nz, S.dy*S.dz, S.dz*S.dx, S.dx*S.dy)
    _launch(gs, _q3_k!, (Nx - 1, Ny - 1, Nz - 1), qx, qy, qz, Dx, Dy, Dz, px, py, pz, hx, hy, hz, Nx, Ny, Nz, S.s)
    _hsync(gs)
    nothing
end

# qx on i 2:Nx-1, j 2:Ny, k 2:Nz; qy on i 2:Nx, j 2:Ny-1, k 2:Nz; qz on i 2:Nx, j 2:Ny, k 2:Nz-1
@kernel function _q3_k!(qx, qy, qz, @Const(Dx), @Const(Dy), @Const(Dz), @Const(px), @Const(py), @Const(pz),
                        @Const(hx), @Const(hy), @Const(hz), Nx::Int, Ny::Int, Nz::Int, s::NTuple{6,Float64})
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    sxy, sxz, syx, syz, szx, szy = s
    @inbounds begin
        i <= Nx - 1 && (qx[i,j,k] = Dx[i,j,k]*px[i,j,k] + sxy*(hz[i,j,k] - hz[i,j-1,k]) - sxz*(hy[i,j,k] - hy[i,j,k-1]))
        j <= Ny - 1 && (qy[i,j,k] = Dy[i,j,k]*py[i,j,k] + syz*(hx[i,j,k] - hx[i,j,k-1]) - syx*(hz[i,j,k] - hz[i-1,j,k]))
        k <= Nz - 1 && (qz[i,j,k] = Dz[i,j,k]*pz[i,j,k] + szx*(hy[i,j,k] - hy[i-1,j,k]) - szy*(hx[i,j,k] - hx[i,j-1,k]))
    end
end

_dot3(a, b) = dot(a[1], b[1]) + dot(a[2], b[2]) + dot(a[3], b[3])

function _precond!(z, r, S::GPCG, gs)
    sp = S.sep
    sp.BX .= r[1]; sp.BY .= r[2]; sp.BZ .= r[3]
    _sep_solve!(sp, gs)
    z[1] .= sp.BX; z[2] .= sp.BY; z[3] .= sp.BZ
    z
end
function _tx32!(T1, T2, B, A, M1, M2, M3, gs)
    n1, n2, n3 = size(A); m1, m2, m3 = size(M1, 2), size(M2, 2), size(M3, 2)
    mul!(reshape(T1, m1, n2*n3), transpose(M1), reshape(A, n1, n2*n3))
    _dim2!(T2, T1, M2, gs)
    mul!(reshape(B, m1*m2, m3), reshape(T2, m1*m2, n3), M3)
    B
end
function _precond32!(z, r, S::GPCG, f::GFast, gs)
    sp = S.sep; a = Float32(sp.a)
    Vx, Ux, Vy, Uy, Vz, Uz, Vxt, Vyt, Vzt = f.V; Uxt, Uyt, Uzt = f.W
    sx, sy, sz = f.s32; BX, BY, BZ = f.B32; T = f.T32
    BX .= r[1]; BY .= r[2]; BZ .= r[3]
    X = _tx32!(T[1], T[2], BX, BX, Ux, Vy, Vz, gs); Y = _tx32!(T[3], T[4], BY, BY, Vx, Uy, Vz, gs)
    Z = _tx32!(T[5], T[6], BZ, BZ, Vx, Vy, Uz, gs)
    _launch(gs, _modes!, (length(sx), length(sy), length(sz)), X, Y, Z, sx, sy, sz, a)
    _hsync(gs)
    z[1] .= _tx32!(T[1], T[2], BX, X, Uxt, Vyt, Vzt, gs); z[2] .= _tx32!(T[3], T[4], BY, Y, Vxt, Uyt, Vzt, gs)
    z[3] .= _tx32!(T[5], T[6], BZ, Z, Vxt, Vyt, Uzt, gs)
    z
end
_precf!(z, r, S, f, gs) = f.f32 ? _precond32!(z, r, S, f, gs) : _precond!(z, r, S, gs)

@kernel function _extrap_k!(xc, pc)                     # x ← 2xₙ − xₙ₋₁, and xₙ kept as the next step's xₙ₋₁
    i = @index(Global, Linear)
    @inbounds begin; v = xc[i]; xc[i] = 2v - pc[i]; pc[i] = v; end
end

function _implicit_curl!(g::GState, S::GPCG, gs)
    f = S.fast
    sp = S.sep
    Nx = size(g.φy, 1) - 1; Ny = size(g.φx, 2) - 1; Nz = size(g.φx, 3) - 1
    Dx, Dy, Dz = S.D
    b = S.b; x = S.x; r = S.r; z = S.z
    _hsync(gs)
    @views begin
        b[1] .= g.φx[sp.rx...] .- g.θx[sp.rx...]
        b[2] .= g.φy[sp.ry...] .- g.θy[sp.ry...]
        b[3] .= g.φz[sp.rz...] .- g.θz[sp.rz...]
    end
    @. g.φx = g.θx + (g.φx - g.θx) / Dx
    @. g.φy = g.θy + (g.φy - g.θy) / Dy
    @. g.φz = g.θz + (g.φz - g.θz) / Dz
    nb = sqrt(_dot3(b, b))
    nb == 0 && return TDGL._pcg_zero_update!(g, S)
    if f === nothing
        S.warm[1] || _precond!(x, b, S, gs)
    elseif !S.warm[1]
        _precond!(x, b, S, gs); f.have2[1] = false
    elseif !f.extrap
    elseif f.have2[1]
        for c in 1:3; _launch(gs, _extrap_k!, (length(x[c]),), x[c], f.xprev[c]); end
        _hsync(gs)
    else
        for c in 1:3; copyto!(f.xprev[c], x[c]); end
        f.have2[1] = true
    end
    S.warm[1] = true
    P = S.P; Q = S.Q
    rngs = (sp.rx, sp.ry, sp.rz)
    gather(full, c) = view(full[c], rngs[c]...)
    for c in 1:3; fill!(P[c], 0.0); view(P[c], rngs[c]...) .= x[c]; end
    _apply_A!(S, gs, Nx, Ny, Nz)
    for c in 1:3; r[c] .= b[c] .- gather(Q, c); end
    it = 0
    if nb > 0 && sqrt(_dot3(r, r)) > S.tol * nb
        f === nothing ? _precond!(z, r, S, gs) : _precf!(z, r, S, f, gs)
        for c in 1:3; fill!(P[c], 0.0); view(P[c], rngs[c]...) .= z[c]; end
        f !== nothing && f.f32 && for c in 1:3; copyto!(f.zold[c], z[c]); end
        rz = _dot3(r, z)
        while it < S.maxit
            it += 1
            _apply_A!(S, gs, Nx, Ny, Nz)
            pAp = 0.0
            for c in 1:3; pAp += dot(gather(P, c), gather(Q, c)); end
            α = rz / pAp
            for c in 1:3
                x[c] .+= α .* gather(P, c)
                r[c] .-= α .* gather(Q, c)
            end
            sqrt(_dot3(r, r)) <= S.tol * nb && break
            f === nothing ? _precond!(z, r, S, gs) : _precf!(z, r, S, f, gs)
            rz2 = _dot3(r, z)
            if f !== nothing && f.f32
                β = (rz2 - _dot3(r, f.zold)) / rz
                for c in 1:3; copyto!(f.zold[c], z[c]); end
            else
                β = rz2 / rz
            end
            rz = rz2
            for c in 1:3; gather(P, c) .= z[c] .+ β .* gather(P, c); end
        end
    end
    S.iters[1] = it
    @views begin
        g.φx[sp.rx...] .= g.θx[sp.rx...] .+ x[1]
        g.φy[sp.ry...] .= g.θy[sp.ry...] .+ x[2]
        g.φz[sp.rz...] .= g.θz[sp.rz...] .+ x[3]
    end
    g
end
# the explicit curl-curl with a conductivity per link (scale_increment!)
function _scale_increment!(g::GState, D, gs)
    _hsync(gs)
    @. g.φx = g.θx + (g.φx - g.θx) / D[1]
    @. g.φy = g.θy + (g.φy - g.θy) / D[2]
    @. g.φz = g.θz + (g.φz - g.θz) / D[3]
    g
end

# ---------------------------------------------------------------------------
#  materials per region (materials.jl): the material of each node, its coefficient tables, the link weights per band
# ---------------------------------------------------------------------------
struct GMat{N,I,T,R,A,V}
    idx::I                                  # material of each node (UInt8)
    full::NTuple{N,T}; stage::NTuple{N,T}   # per band, (kx, ky, kz, c, ε, β, γ2, cpl) of every material
    ρ::NTuple{N,NTuple{3,R}}                # per band and axis, the link weights (τ included)
    abs::NTuple{N,A}                        # per band, the linear indices of the interior nodes where it is absent
    a::NTuple{N,V}                          # per band, the a of each material (ε = a − θ with the heat model)
    Γ::NTuple{N,V}; γk::NTuple{N,V}         # per band, Γ and γ̃ of each material (the heat source)
    σn::V; DT::V                            # per material, σn and D_T (the heat model; DT empty without it)
end
_tables(backend, t) = (kx=_dev(backend, t.kx), ky=_dev(backend, t.ky), kz=_dev(backend, t.kz), c=_dev(backend, t.c),
                       ε=_dev(backend, t.ε), β=_dev(backend, t.β), γ2=_dev(backend, t.γ2), cpl=_dev(backend, t.cpl))
function GMat(backend, mt)
    N = length(mt.full)
    L = LinearIndices(mt.idx)
    GMat(_dev(backend, mt.idx), ntuple(µ -> _tables(backend, mt.full[µ]), N), ntuple(µ -> _tables(backend, mt.stage[µ]), N),
         ntuple(µ -> ntuple(a -> _dev(backend, mt.ρ[µ][a]), 3), N),
         ntuple(µ -> _dev(backend, Int[L[c] for c in mt.absnodes[µ]]), N),
         ntuple(µ -> _dev(backend, mt.a[µ]), N),
         ntuple(µ -> _dev(backend, [m.bands[µ].Γ for m in mt.models]), N),
         ntuple(µ -> _dev(backend, [m.bands[µ].γkwt for m in mt.models]), N),
         _dev(backend, [m.σn for m in mt.models]), _dev(backend, isempty(mt.DT) ? [0.0] : mt.DT))
end

# _band_row! with materials (one row's body, per node): the node's coefficients, and the link weights ρ in the diffusion
# the node's ε, or a − θ of its material with the heat model (_epsn and _anode of bands.jl)
@inline _epsm(ε, A, m, ::Nothing, i, j, k) = ε
@inline _epsm(ε, A, m, θ, i, j, k) = @inbounds A[m] - θ[i,j,k]
@kernel function _band_mat_kernel!(D, @Const(S), G, µ::Int, @Const(Vx), @Const(Vy), @Const(Vz), @Const(idx), T,
                                   @Const(ρx), @Const(ρy), @Const(ρz), @Const(f), hard::Bool, x1::Int, y1::Int, z1::Int, ::Val{N}, A, θ,
                                   rk::Bool) where N
    I = @index(Global, NTuple)
    i = x1 + I[1]; j = y1 + I[2]; k = z1 + I[3]
    @inbounds begin
        g = S[i,j,k]; fm = f[i,j,k]
        if hard && fm == 0.0
            D[i,j,k] = 0.0 + 0.0im
        else
            m = idx[i,j,k]
            kx, ky, kz, c, ε, β, cpl, γ2 = T.kx[m], T.ky[m], T.kz[m], T.c[m], T.ε[m], T.β[m], T.cpl[m], T.γ2[m]
            vxm = conj(Vx[i-1,j,k]); vym = conj(Vy[i,j-1,k]); vzm = conj(Vz[i,j,k-1])
            vp = Vx[i,j,k]; sp = S[i+1,j,k]; sm = S[i-1,j,k]
            tx = kx*((vp*sp - 2*g) + vxm*sm) + kx*((ρx[i,j,k] - 1)*(vp*sp - g) + (ρx[i-1,j,k] - 1)*(vxm*sm - g))
            vp = Vy[i,j,k]; sp = S[i,j+1,k]; sm = S[i,j-1,k]
            ty = ky*((vp*sp - 2*g) + vym*sm) + ky*((ρy[i,j,k] - 1)*(vp*sp - g) + (ρy[i,j-1,k] - 1)*(vym*sm - g))
            vp = Vz[i,j,k]; sp = S[i,j,k+1]; sm = S[i,j,k-1]
            tz = kz*((vp*sp - 2*g) + vzm*sm) + kz*((ρz[i,j,k] - 1)*(vp*sp - g) + (ρz[i,j,k-1] - 1)*(vzm*sm - g))
            nl = (c*g)*(fm*_epsm(ε, A, m, θ, i, j, k) - β*abs2(g))
            acc = (((g + tx) + ty) + tz) + nl
            for ν in 1:N
                ν == µ || (acc += cpl[ν]*G[ν][i,j,k])
            end
            D[i,j,k] = γ2 > 0 ? (rk ? TDGL.kwt_update_rot(g, acc - g, γ2) : TDGL.kwt_update(g, acc - g, γ2)) : acc
        end
    end
end
# _band_row_vac! with materials: ρ in place of τ, the node's coefficients
@kernel function _band_matvac_kernel!(D, @Const(S), G, µ::Int, @Const(Vx), @Const(Vy), @Const(Vz), @Const(idx), T,
                                      @Const(ρx), @Const(ρy), @Const(ρz), @Const(f), hard::Bool, @Const(vac),
                                      x1::Int, y1::Int, z1::Int, ::Val{N}, A, θ, rk::Bool) where N
    I = @index(Global, NTuple)
    i = x1 + I[1]; j = y1 + I[2]; k = z1 + I[3]
    @inbounds begin
        g = S[i,j,k]; fm = f[i,j,k]
        if vac[i,j,k] || (hard && fm == 0.0)
            D[i,j,k] = 0.0 + 0.0im
        else
            m = idx[i,j,k]
            kx, ky, kz, c, ε, β, cpl, γ2 = T.kx[m], T.ky[m], T.kz[m], T.c[m], T.ε[m], T.β[m], T.cpl[m], T.γ2[m]
            tx = kx*(ρx[i,j,k]*(Vx[i,j,k]*S[i+1,j,k] - g) + ρx[i-1,j,k]*(conj(Vx[i-1,j,k])*S[i-1,j,k] - g))
            ty = ky*(ρy[i,j,k]*(Vy[i,j,k]*S[i,j+1,k] - g) + ρy[i,j-1,k]*(conj(Vy[i,j-1,k])*S[i,j-1,k] - g))
            tz = kz*(ρz[i,j,k]*(Vz[i,j,k]*S[i,j,k+1] - g) + ρz[i,j,k-1]*(conj(Vz[i,j,k-1])*S[i,j,k-1] - g))
            acc = (((g + tx) + ty) + tz) + (c*g)*(fm*_epsm(ε, A, m, θ, i, j, k) - β*abs2(g))
            for ν in 1:N
                ν == µ || (acc += cpl[ν]*G[ν][i,j,k])
            end
            D[i,j,k] = γ2 > 0 ? (rk ? TDGL.kwt_update_rot(g, acc - g, γ2) : TDGL.kwt_update(g, acc - g, γ2)) : acc
        end
    end
end
@kernel function _zero_at_k!(D, @Const(L))
    q = @index(Global, Linear)
    @inbounds D[L[q]] = 0.0 + 0.0im
end
# the supercurrent with materials (_supercurrent_mat!): each band's link weighted by its ρ
@inline function _jlink_mat(G::NTuple{N}, w, ρ, ax, v, i, j, k, i2, j2, k2) where N
    @inbounds begin
        acc = (w[1]*ρ[1][ax][i,j,k])*imag((conj(G[1][i,j,k])*v)*G[1][i2,j2,k2])
        for µ in 2:N
            acc += (w[µ]*ρ[µ][ax][i,j,k])*imag((conj(G[µ][i,j,k])*v)*G[µ][i2,j2,k2])
        end
        acc
    end
end
@kernel function _j3mat_k!(Jx, Jy, Jz, G, @Const(Vx), @Const(Vy), @Const(Vz), wx, wy, wz, ρ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        j >= 2 && k >= 2 && (Jx[i,j,k] = _jlink_mat(G, wx, ρ, 1, Vx[i,j,k], i, j, k, i+1, j, k))
        i >= 2 && k >= 2 && (Jy[i,j,k] = _jlink_mat(G, wy, ρ, 2, Vy[i,j,k], i, j, k, i, j+1, k))
        i >= 2 && j >= 2 && (Jz[i,j,k] = _jlink_mat(G, wz, ρ, 3, Vz[i,j,k], i, j, k, i, j, k+1))
    end
end

# ---------------------------------------------------------------------------
#  the heat model (terms.jl, heat_step! with one material).  The CPU code updates θ in place in lexicographic order (a node
#  reads the new θ of i−1, j−1, k−1 and the old one of i+1, j+1, k+1, as release 1.2 does); the same order on a device is a
#  sweep over the planes i + j + k = s, each plane one launch, every node of a plane independent of the others: the bits.
#  hf.diag (means of the sources; read by nothing) is not computed on the device.
# ---------------------------------------------------------------------------
@kernel function _tfx_k!(T, x1::Int, x2::Int, y1::Int, z1::Int)
    a, b = @index(Global, NTuple); j = y1 - 1 + a; k = z1 - 1 + b
    @inbounds begin; T[x1, j, k] = T[x1+1, j, k]; T[x2+1, j, k] = T[x2, j, k]; end
end
@kernel function _tfy_k!(T, x1::Int, y1::Int, y2::Int, z1::Int)
    a, b = @index(Global, NTuple); i = x1 - 1 + a; k = z1 - 1 + b
    @inbounds begin; T[i, y1, k] = T[i, y1+1, k]; T[i, y2+1, k] = T[i, y2, k]; end
end
@kernel function _tfz_k!(T, x1::Int, y1::Int, z1::Int, z2::Int)
    a, b = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b
    @inbounds begin; T[i, j, z1] = T[i, j, z1+1]; T[i, j, z2+1] = T[i, j, z2]; end
end
@kernel function _heat_plane_k!(T, G, F, Γ, γk, @Const(f), @Const(θx), @Const(θy), @Const(θz), @Const(φx), @Const(φy), @Const(φz),
                                s::Int, x1::Int, y1::Int, z1::Int, z2::Int, lo::Int, hi::Int, dx2::Float64, dy2::Float64,
                                dz2::Float64, dt::Float64, invdt2::Float64, cx::Float64, cy::Float64, cz::Float64, DT::Float64,
                                η::Float64, σn::Float64, γb::Float64, θb::Float64, den::Float64, cap::Float64, joule::Bool,
                                ::Val{N}, qj) where N
    a, b = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = s - i - j
    @inbounds if z1 + 1 <= k <= z2
        q = T[i,j,k]
        lap = (T[i+1,j,k] - 2q + T[i-1,j,k]) / dx2 + (T[i,j+1,k] - 2q + T[i,j-1,k]) / dy2 +
              (T[i,j,k+1] - 2q + T[i,j,k-1]) / dz2
        dentro = lo < i < hi
        src = 0.0
        if dentro
            src = TDGL._band_heat(Γ[1], γk[1], G[1][i,j,k], F[1][i,j,k], invdt2)
            for µ in 2:N
                src += TDGL._band_heat(Γ[µ], γk[µ], G[µ][i,j,k], F[µ][i,j,k], invdt2)
            end
        end
        if joule
            e2 = 0.0
            if dentro && f[i,j,k] > 0
                ex = 0.5 * (abs2(θx[i-1,j,k] - φx[i-1,j,k]) + abs2(θx[i,j,k] - φx[i,j,k])) * cx
                ey = 0.5 * (abs2(θy[i,j-1,k] - φy[i,j-1,k]) + abs2(θy[i,j,k] - φy[i,j,k])) * cy
                ez = 0.5 * (abs2(θz[i,j,k-1] - φz[i,j,k-1]) + abs2(θz[i,j,k] - φz[i,j,k])) * cz
                e2 = ex + ey + ez
            end
            qq = (qj !== nothing && dentro && f[i,j,k] > 0) ? _qpjoule(qj, i, j, k, dx2, dy2, dz2) : 0.0
            acc = q + dt * (DT * lap + η * ((src + σn * e2) + qq))
        else
            acc = q + dt * (DT * lap + η * src)
        end
        θn = (acc + dt * γb * θb) / den
        θn = θn < 0 ? 0.0 : θn > cap ? cap : θn
        T[i,j,k] = θn
    end
end
# qp_joule of exchange.jl (the heat of the fluids beyond σn|E|²), two fluids and N fluids; dx2 = dx² as the CPU divides
@inline function _qpjoule(qj::NamedTuple{(:δ, :c, :G)}, i, j, k, dx2, dy2, dz2)
    δ = qj.δ
    @inbounds q = 0.5 * ((δ[i+1,j,k] - δ[i,j,k])^2 + (δ[i,j,k] - δ[i-1,j,k])^2) / dx2 +
                  0.5 * ((δ[i,j+1,k] - δ[i,j,k])^2 + (δ[i,j,k] - δ[i,j-1,k])^2) / dy2 +
                  0.5 * ((δ[i,j,k+1] - δ[i,j,k])^2 + (δ[i,j,k] - δ[i,j,k-1])^2) / dz2
    @inbounds qj.c * q + qj.G * δ[i,j,k]^2
end
@inline function _qpjoule(qj::NamedTuple{(:pot, :s, :σn, :Gm)}, i, j, k, dx2, dy2, dz2)
    q = 0.0; pot = qj.pot; N = length(pot)
    @inbounds for b in 1:N
        m = pot[b]
        q += qj.s[b] * (0.5 * ((m[i+1,j,k] - m[i,j,k])^2 + (m[i,j,k] - m[i-1,j,k])^2) / dx2 +
                        0.5 * ((m[i,j+1,k] - m[i,j,k])^2 + (m[i,j,k] - m[i,j-1,k])^2) / dy2 +
                        0.5 * ((m[i,j,k+1] - m[i,j,k])^2 + (m[i,j,k] - m[i,j,k-1])^2) / dz2)
    end
    q *= qj.σn
    @inbounds for b in 1:N, c in b+1:N
        q += qj.Gm[b][c] * (pot[b][i,j,k] - pot[c][i,j,k])^2
    end
    q
end
_qj(gs) = gs.qp === nothing ? nothing :
          gs.qp isa GQP ? (δ=gs.qp.δ, c=gs.qp.σn * gs.qp.s[1] * gs.qp.s[2], G=gs.qp.G) :
          (pot=gs.qp.μ, s=gs.qp.s, σn=gs.qp.σn, Gm=gs.qp.Gm)
# _heat_step_mat!: the node's Γ, γ̃ and σn; one D_T: the in-place sweep (planes), D_T per material: the neighbours of the step
# before (Jacobi, as the CPU code), one launch, the harmonic mean of D_T on each link
@inline _hm(a, b) = a == b ? a : 2a * b / (a + b)
@kernel function _heat_mat_k!(T, @Const(R), G, F, @Const(idx), Γt, γt, @Const(σt), @Const(DT), uni::Bool, DTu::Float64,
                              @Const(f), @Const(θx), @Const(θy), @Const(θz), @Const(φx), @Const(φy), @Const(φz),
                              s::Int, plane::Bool, x1::Int, y1::Int, z1::Int, z2::Int, lo::Int, hi::Int, dx2::Float64,
                              dy2::Float64, dz2::Float64, dt::Float64, invdt2::Float64, cx::Float64, cy::Float64, cz::Float64,
                              η::Float64, γb::Float64, θb::Float64, den::Float64, cap::Float64, joule::Bool, ::Val{N}, qj) where N
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = plane ? s - i - j : z1 + e
    @inbounds if z1 + 1 <= k <= z2
        q = R[i,j,k]
        flux = if uni
            DTu * ((T[i+1,j,k] - 2q + T[i-1,j,k]) / dx2 + (T[i,j+1,k] - 2q + T[i,j-1,k]) / dy2 +
                   (T[i,j,k+1] - 2q + T[i,j,k-1]) / dz2)
        else
            d0 = DT[idx[i,j,k]]
            (_hm(d0, DT[idx[i+1,j,k]]) * (R[i+1,j,k] - q) + _hm(d0, DT[idx[i-1,j,k]]) * (R[i-1,j,k] - q)) / dx2 +
            (_hm(d0, DT[idx[i,j+1,k]]) * (R[i,j+1,k] - q) + _hm(d0, DT[idx[i,j-1,k]]) * (R[i,j-1,k] - q)) / dy2 +
            (_hm(d0, DT[idx[i,j,k+1]]) * (R[i,j,k+1] - q) + _hm(d0, DT[idx[i,j,k-1]]) * (R[i,j,k-1] - q)) / dz2
        end
        m = idx[i,j,k]
        dentro = lo < i < hi
        src = 0.0
        if dentro
            for µ in 1:N
                src += TDGL._band_heat(Γt[µ][m], γt[µ][m], G[µ][i,j,k], F[µ][i,j,k], invdt2)
            end
        end
        if joule
            e2 = 0.0
            if dentro && f[i,j,k] > 0
                ex = 0.5 * (abs2(θx[i-1,j,k] - φx[i-1,j,k]) + abs2(θx[i,j,k] - φx[i,j,k])) * cx
                ey = 0.5 * (abs2(θy[i,j-1,k] - φy[i,j-1,k]) + abs2(θy[i,j,k] - φy[i,j,k])) * cy
                ez = 0.5 * (abs2(θz[i,j,k-1] - φz[i,j,k-1]) + abs2(θz[i,j,k] - φz[i,j,k])) * cz
                e2 = ex + ey + ez
            end
            qq = (qj !== nothing && dentro && f[i,j,k] > 0) ? _qpjoule(qj, i, j, k, dx2, dy2, dz2) : 0.0
            acc = q + dt * (flux + η * ((src + σt[m] * e2) + qq))
        else
            acc = q + dt * (flux + η * src)
        end
        θn = (acc + dt * γb * θb) / den
        θn = θn < 0 ? 0.0 : θn > cap ? cap : θn
        T[i,j,k] = θn
    end
end
function _heat_step_mat!(g::GState{N}, gs) where N
    st = gs.st; hf = st.heat; hp = hf.h; T = gs.θ; gr = st.grid; dt = st.dt; M = gs.mat; mt = st.mt
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    _launch(gs, _tfx_k!, (y2 - y1 + 2, z2 - z1 + 2), T, x1, x2, y1, z1)
    _launch(gs, _tfy_k!, (x2 - x1 + 2, z2 - z1 + 2), T, x1, y1, y2, z1)
    _launch(gs, _tfz_k!, (x2 - x1 + 2, y2 - y1 + 2), T, x1, y1, z1, z2)
    invdt2 = 1 / (dt * dt); dx2 = dx^2; dy2 = dy^2; dz2 = dz^2
    uni = all(==(hp.D_T), mt.DT)
    m = hp.margin; lo = x1 + m; hi = x2 + 1 - m
    args(R, s, plane) = (T, R, g.G, g.F, M.idx, M.Γ, M.γk, M.σn, M.DT, uni, hp.D_T, gs.f, g.θx, g.θy, g.θz, g.φx, g.φy, g.φz,
                         s, plane, x1, y1, z1, z2, lo, hi, dx2, dy2, dz2, dt, invdt2, invdt2 / dx2, invdt2 / dy2, invdt2 / dz2,
                         hp.eta, hp.gamma_bath, hf.T, 1 + dt * hp.gamma_bath, hp.cap, hp.joule, Val(N), _qj(gs))
    if uni
        for s in (x1 + y1 + z1 + 3):(x2 + y2 + z2)
            _launch(gs, _heat_mat_k!, (x2 - x1, y2 - y1, 1), args(T, s, true)...)
        end
    else
        R = copy(T)
        _launch(gs, _heat_mat_k!, (x2 - x1, y2 - y1, z2 - z1), args(R, 0, false)...)
    end
    g
end

function _heat_step!(g::GState{N}, gs) where N
    gs.mat === nothing || return _heat_step_mat!(g, gs)
    st = gs.st; hf = st.heat; hp = hf.h; T = gs.θ; gr = st.grid; dt = st.dt
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    _launch(gs, _tfx_k!, (y2 - y1 + 2, z2 - z1 + 2), T, x1, x2, y1, z1)
    _launch(gs, _tfy_k!, (x2 - x1 + 2, z2 - z1 + 2), T, x1, y1, y2, z1)
    _launch(gs, _tfz_k!, (x2 - x1 + 2, y2 - y1 + 2), T, x1, y1, z1, z2)
    invdt2 = 1 / (dt * dt); dx2 = dx^2; dy2 = dy^2; dz2 = dz^2
    Γ = ntuple(µ -> st.model.bands[µ].Γ, N); γk = ntuple(µ -> st.model.bands[µ].γkwt, N)
    m = hp.margin; lo = x1 + m; hi = x2 + 1 - m
    for s in (x1 + y1 + z1 + 3):(x2 + y2 + z2)
        _launch(gs, _heat_plane_k!, (x2 - x1, y2 - y1), T, g.G, g.F, Γ, γk, gs.f, g.θx, g.θy, g.θz, g.φx, g.φy, g.φz,
                s, x1, y1, z1, z2, lo, hi, dx2, dy2, dz2, dt, invdt2, invdt2 / dx2, invdt2 / dy2, invdt2 / dz2, hp.D_T,
                hp.eta, st.model.σn, hp.gamma_bath, hf.T, 1 + dt * hp.gamma_bath, hp.cap, hp.joule, Val(N), _qj(gs))
    end
    g
end

# ---------------------------------------------------------------------------
#  thermal noise without KWT (noise.jl: noise_kick!, field_kick!).  The draws are those of the seeded stream of the CPU
#  Stepper (st.noise.rng), taken on the host in the order of the CPU loops, so the stream, its checkpoints and its resumption
#  are those of the CPU code; the device adds them with the same arithmetic.
# ---------------------------------------------------------------------------
struct GNoise{C,R}
    hQ::Vector{Array{ComplexF64,3}}; dQ::Vector{C}        # per band: the draws of the nodes (0 where none)
    hφ::NTuple{3,Array{Float64,3}}; dφ::NTuple{3,R}       # the field: the draws of the links
    dθ::Vector{R}                                         # per band: the phase kick δθ of SK-ROCK (noise with KWT on RKC)
    absent::Vector{Any}                                   # with materials: per band, whether each material lacks it (Bool vector)
end
function GNoise(backend, st)
    N = length(st.model.bands); sz = size(st.masks.f)
    hQ = [zeros(ComplexF64, sz) for _ in 1:N]
    (; Nx, Ny, Nz) = st.grid
    hφ = (zeros(Nx, Ny + 1, Nz + 1), zeros(Nx + 1, Ny, Nz + 1), zeros(Nx + 1, Ny + 1, Nz))
    dQ = [_dev(backend, q) for q in hQ]; dφ = map(x -> _dev(backend, x), hφ)
    dθ = [_dev(backend, zeros(sz)) for _ in 1:N]
    ab = st.mt === nothing ? Any[nothing for _ in 1:N] : Any[_dev(backend, Bool[st.mt.absent[m, µ] for m in 1:length(st.mt.models)]) for µ in 1:N]
    GNoise{eltype(dQ),eltype(dφ)}(hQ, dQ, hφ, dφ, dθ, ab)
end
# with materials: the amplitude of the node's material (sqrt(T̃dt/(Γ_m dV)), as _noise_kick_mat!)
@kernel function _kick_mat_k!(F, @Const(G), @Const(Q), @Const(idx), Γt, γt, Tdt::Float64, dV::Float64, x1::Int, y1::Int, z1::Int)
    p, q, r = @index(Global, NTuple); i = x1 + p; j = y1 + q; k = z1 + r
    @inbounds begin
        ξ = Q[i,j,k]
        if ξ != 0
            m = idx[i,j,k]; Γ = Γt[m]; γ2 = γt[m]^2
            if γ2 == 0
                F[i,j,k] += sqrt(Tdt / (Γ * dV)) * ξ
            else                              # _noise_kick_mat!: the polar kick of the node's material
                c = sqrt(Tdt / dV); dr = Tdt / (2dV) * γ2 / Γ; a = sqrt(Tdt / (Γ * dV))
                F[i,j,k] = TDGL._kwt_kick(F[i,j,k], G[i,j,k], ξ, a, c, dr, γ2, Γ, Tdt / (2dV * Γ))
            end
        end
    end
end
# a KWT band after Euler substeps (noise_kick!, γ̃ > 0): the polar kick of _kwt_kick at Ψⁿ
@kernel function _kick_kwt_k!(F, @Const(G), @Const(Q), a::Float64, c::Float64, dr::Float64, γ2::Float64, Γ::Float64, d0::Float64,
                              x1::Int, y1::Int, z1::Int)
    p, q, r = @index(Global, NTuple); i = x1 + p; j = y1 + q; k = z1 + r
    @inbounds begin
        ξ = Q[i,j,k]
        ξ != 0 && (F[i,j,k] = TDGL._kwt_kick(F[i,j,k], G[i,j,k], ξ, a, c, dr, γ2, Γ, d0))
    end
end
@kernel function _kick_k!(F, @Const(Q), a::Float64, x1::Int, y1::Int, z1::Int)
    p, q, r = @index(Global, NTuple); i = x1 + p; j = y1 + q; k = z1 + r
    @inbounds begin
        ξ = Q[i,j,k]
        ξ != 0 && (F[i,j,k] += a * ξ)
    end
end
function _noise_kick!(g::GState{N}, gs) where N
    st = gs.st; nz = st.noise; gr = st.grid; f = st.masks.f; hard = st.hard; vac = st.masks.vac
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; rng = nz.rng
    mt = st.mt
    for µ in 1:N
        Q = gs.nz.hQ[µ]; fill!(Q, 0.0 + 0.0im)
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            (hard && f[i,j,k] == 0.0) && continue
            vac === nothing || vac[i,j,k] && continue
            ξ = complex(randn(rng), randn(rng))
            # a band absent from the node's material takes its two draws and no kick
            (mt === nothing || !mt.absent[mt.idx[i,j,k], µ]) && (Q[i,j,k] = ξ)
        end
        copyto!(gs.nz.dQ[µ], Q)
        if mt === nothing
            Γ = st.model.bands[µ].Γ; γ2 = st.model.bands[µ].γkwt^2
            a = sqrt(nz.T * st.dt / (Γ * dV))
            if γ2 == 0
                _launch(gs, _kick_k!, (x2 - x1, y2 - y1, z2 - z1), g.F[µ], gs.nz.dQ[µ], a, x1, y1, z1)
            else
                c = sqrt(nz.T * st.dt / dV); dr = nz.T * st.dt / (2dV) * γ2 / Γ
                _launch(gs, _kick_kwt_k!, (x2 - x1, y2 - y1, z2 - z1), g.F[µ], g.G[µ], gs.nz.dQ[µ], a, c, dr, γ2, Γ,
                        nz.T * st.dt / (2dV * Γ), x1, y1, z1)
            end
        else
            _launch(gs, _kick_mat_k!, (x2 - x1, y2 - y1, z2 - z1), g.F[µ], g.G[µ], gs.nz.dQ[µ], gs.mat.idx, gs.mat.Γ[µ],
                    gs.mat.γk[µ], nz.T * st.dt, dV, x1, y1, z1)
        end
    end
    g
end
# noise_draw!: every draw of the step before the bands (host, the CPU order); δθ of each KWT band with RKC on the device
@kernel function _qtheta_k!(Qθ, @Const(Q), @Const(G), c::Float64, γ2::Float64, Γ::Float64, x1::Int, y1::Int, z1::Int)
    p, q, r = @index(Global, NTuple); i = x1 + p; j = y1 + q; k = z1 + r
    @inbounds begin
        ξ = Q[i,j,k]; ψ = G[i,j,k]; r2 = abs2(ψ)
        if ξ != 0 && γ2 > 0 && r2 > TDGL._ROT_MIN2
            sk_ = sqrt(1 + γ2 * r2); ρ = sqrt(r2)
            Qθ[i,j,k] = c * imag(conj(ψ / ρ) * ξ) * sqrt(sk_ / Γ) / ρ
        else
            Qθ[i,j,k] = 0.0
        end
    end
end
function _noise_draw!(g::GState{N}, gs) where N
    st = gs.st; nz = st.noise; gr = st.grid; f = st.masks.f; hard = st.hard; vac = st.masks.vac
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; rng = nz.rng; c = sqrt(nz.T * st.dt / dV)
    for µ in 1:N
        Q = gs.nz.hQ[µ]; fill!(Q, 0.0 + 0.0im)
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            (hard && f[i,j,k] == 0.0) && continue
            vac === nothing || vac[i,j,k] && continue
            Q[i,j,k] = complex(randn(rng), randn(rng))
        end
        copyto!(gs.nz.dQ[µ], Q)
        fill!(gs.nz.dθ[µ], 0.0)
        if TDGL._skrock(st, µ)
            if gs.mat === nothing
                b = st.model.bands[µ]
                _launch(gs, _qtheta_k!, (x2 - x1, y2 - y1, z2 - z1), gs.nz.dθ[µ], gs.nz.dQ[µ], g.G[µ], c, b.γkwt^2, b.Γ, x1, y1, z1)
            else
                _launch(gs, _qtheta_mat_k!, (x2 - x1, y2 - y1, z2 - z1), gs.nz.dθ[µ], gs.nz.dQ[µ], g.G[µ], c, gs.mat.idx,
                        gs.mat.γk[µ], gs.mat.Γ[µ], gs.nz.absent[µ], x1, y1, z1)
            end
        end
    end
    g
end
# noise_apply!: what the stages did not take, after the bands
@kernel function _napply_k!(F, @Const(G), @Const(Q), @Const(Qθ), a::Float64, c::Float64, γ2::Float64, Γ::Float64, Tdt::Float64,
                            dV::Float64, x1::Int, y1::Int, z1::Int)
    p, q, r = @index(Global, NTuple); i = x1 + p; j = y1 + q; k = z1 + r
    @inbounds begin
        ξ = Q[i,j,k]
        if ξ != 0
            if γ2 == 0
                F[i,j,k] += a * ξ
            elseif Qθ[i,j,k] != 0.0
                ψ = G[i,j,k]; ρ = abs(ψ); sk = sqrt(1 + γ2 * ρ^2); u = ψ / ρ
                δρ = c * real(conj(u) * ξ) / sqrt(Γ * sk) + Tdt / (2dV * Γ) / (ρ * sk^3)
                aF = abs(F[i,j,k])
                F[i,j,k] = aF > sqrt(TDGL._ROT_MIN2) ? F[i,j,k] * ((aF + δρ) / aF) : F[i,j,k] + u * δρ
            else
                F[i,j,k] = TDGL._kwt_kick(F[i,j,k], G[i,j,k], ξ, a, c, Tdt / (2dV) * γ2 / Γ, γ2, Γ, Tdt / (2dV * Γ))
            end
        end
    end
end
function _noise_apply!(g::GState{N}, gs) where N
    st = gs.st; nz = st.noise; gr = st.grid
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; c = sqrt(nz.T * st.dt / dV)
    for µ in 1:N
        if gs.mat === nothing
            b = st.model.bands[µ]
            _launch(gs, _napply_k!, (x2 - x1, y2 - y1, z2 - z1), g.F[µ], g.G[µ], gs.nz.dQ[µ], gs.nz.dθ[µ],
                    sqrt(nz.T * st.dt / (b.Γ * dV)), c, b.γkwt^2, b.Γ, nz.T * st.dt, dV, x1, y1, z1)
        else
            _launch(gs, _napply_mat_k!, (x2 - x1, y2 - y1, z2 - z1), g.F[µ], g.G[µ], gs.nz.dQ[µ], gs.nz.dθ[µ], gs.mat.idx,
                    gs.mat.γk[µ], gs.mat.Γ[µ], gs.nz.absent[µ], c, nz.T * st.dt, dV, x1, y1, z1)
        end
    end
    g
end
# with materials: δθ and the rest of the kick with the node's material (an absent band: no kick)
@kernel function _qtheta_mat_k!(Qθ, @Const(Q), @Const(G), c::Float64, @Const(idx), γt, Γt, @Const(ab), x1::Int, y1::Int, z1::Int)
    p, q, r = @index(Global, NTuple); i = x1 + p; j = y1 + q; k = z1 + r
    @inbounds begin
        ξ = Q[i,j,k]; m = idx[i,j,k]; γ2 = γt[m]^2; ψ = G[i,j,k]; r2 = abs2(ψ)
        if ξ != 0 && !ab[m] && γ2 > 0 && r2 > TDGL._ROT_MIN2
            sk_ = sqrt(1 + γ2 * r2); ρ = sqrt(r2)
            Qθ[i,j,k] = c * imag(conj(ψ / ρ) * ξ) * sqrt(sk_ / Γt[m]) / ρ
        else
            Qθ[i,j,k] = 0.0
        end
    end
end
@kernel function _napply_mat_k!(F, @Const(G), @Const(Q), @Const(Qθ), @Const(idx), γt, Γt, @Const(ab), c::Float64, Tdt::Float64,
                                dV::Float64, x1::Int, y1::Int, z1::Int)
    p, q, r = @index(Global, NTuple); i = x1 + p; j = y1 + q; k = z1 + r
    @inbounds begin
        ξ = Q[i,j,k]; m = idx[i,j,k]
        if ξ != 0 && !ab[m]
            Γ = Γt[m]; γ2 = γt[m]^2; a = sqrt(Tdt / (Γ * dV))
            if γ2 == 0
                F[i,j,k] += a * ξ
            elseif Qθ[i,j,k] != 0.0
                ψ = G[i,j,k]; ρ = abs(ψ); sk = sqrt(1 + γ2 * ρ^2); u = ψ / ρ
                δρ = c * real(conj(u) * ξ) / sqrt(Γ * sk) + Tdt / (2dV * Γ) / (ρ * sk^3)
                aF = abs(F[i,j,k])
                F[i,j,k] = aF > sqrt(TDGL._ROT_MIN2) ? F[i,j,k] * ((aF + δρ) / aF) : F[i,j,k] + u * δρ
            else
                F[i,j,k] = TDGL._kwt_kick(F[i,j,k], G[i,j,k], ξ, a, c, Tdt / (2dV) * γ2 / Γ, γ2, Γ, Tdt / (2dV * Γ))
            end
        end
    end
end
@kernel function _fkick_k!(φx, φy, φz, @Const(nx), @Const(ny), @Const(nz), ax::Float64, ay::Float64, az::Float64, D)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        j >= 2 && k >= 2 && (φx[i,j,k] += (D === nothing ? ax : ax * sqrt(D[1][i,j,k])) * nx[i,j,k])
        i >= 2 && k >= 2 && (φy[i,j,k] += (D === nothing ? ay : ay * sqrt(D[2][i,j,k])) * ny[i,j,k])
        i >= 2 && j >= 2 && (φz[i,j,k] += (D === nothing ? az : az * sqrt(D[3][i,j,k])) * nz[i,j,k])
    end
end
function _field_kick!(g::GState, gs)
    st = gs.st; rng = st.noise.rng; (; Nx, Ny, Nz, dx, dy, dz) = st.grid
    hx, hy, hz = gs.nz.hφ
    @inbounds for k in 2:Nz, j in 2:Ny, i in 1:Nx; hx[i,j,k] = randn(rng); end
    @inbounds for k in 2:Nz, j in 1:Ny, i in 2:Nx; hy[i,j,k] = randn(rng); end
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx; hz[i,j,k] = randn(rng); end
    for c in 1:3; copyto!(gs.nz.dφ[c], gs.nz.hφ[c]); end
    c = sqrt(st.noise.T * st.dt / (st.model.σn * dx * dy * dz))
    D = st.D === nothing ? nothing : (gs.sep isa GPCG ? gs.sep.D : gs.D)
    _launch(gs, _fkick_k!, (Nx, Ny, Nz), g.φx, g.φy, g.φz, gs.nz.dφ..., c * dx, c * dy, c * dz, D)
    g
end

# ---------------------------------------------------------------------------
#  two normal fluids (exchange.jl: qp_potential!, qp_turn!) of a two-band model with one material: the conversion of each
#  band on the block (TDGL._conv, its body), the separable Helmholtz solve of δ (the dense products of _tx3!, the division
#  by the modes), δ on the nodes, and each band turned by e^{−iμ_µ dt}
# ---------------------------------------------------------------------------
struct GQP{M,V,R,CG,VR,GT}
    Vx::M; Wx::M; ex::V; Vy::M; Wy::M; ey::V; Vz::M; Wz::M; ez::V
    δ::R; S::R; T1::R; T2::R
    s::NTuple{2,Float64}; σn::Float64; λ0::Float64; j0::Int; k0::Int
    cg::CG                                  # a sculpted sample: the CG of the δ equation (GQPCG), or nothing
    G::Float64                              # the exchange conductance (the Joule heat of the exchange)
    var::VR                                 # materials with σn, λ or a one-fluid region by node (GQPVar), or nothing
    γt::GT                                  # materials: per band, the Josephson row of each material (the conversion), or nothing
end
GQP(backend, q, mt=nothing) = GQP(_dev(backend, q.Vx), _dev(backend, q.Wx), _dev(backend, q.ex), _dev(backend, q.Vy), _dev(backend, q.Wy),
                      _dev(backend, q.ey), _dev(backend, q.Vz), _dev(backend, q.Wz), _dev(backend, q.ez), _dev(backend, q.δ),
                      _dev(backend, q.S), _dev(backend, q.T1), _dev(backend, q.T2), q.s, q.σn, q.λ0, q.ps.j0, q.ps.k0,
                      q.ps.cond === nothing ? nothing : GQPCG(backend, q.ps), q.G,
                      q.var === nothing ? nothing : GQPVar(backend, q.var, q.ps),
                      mt === nothing ? nothing : ntuple(µ -> _dev(backend, [Tuple(m.γ[µ]) for m in mt.models]), 2))
@kernel function _conv_k!(S, G, g1, g2, γ1, γ2, dragν, @Const(Vx), @Const(Vy), @Const(Vz), @Const(τx), @Const(τy), @Const(τz),
                          rx2::Float64, ry2::Float64, rz2::Float64, s1::Float64, s2::Float64, x1::Int, y1::Int, z1::Int,
                          j0::Int, k0::Int)
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        c1 = TDGL._conv(G, 1, g1, γ1, dragν, Vx, Vy, Vz, τx, τy, τz, i, j, k, rx2, ry2, rz2)
        c2 = TDGL._conv(G, 2, g2, γ2, dragν, Vx, Vy, Vz, τx, τy, τz, i, j, k, rx2, ry2, rz2)
        S[i-1, j-j0, k-k0] = c2 / s2 - c1 / s1
    end
end
# _tx3!: B ← A transformed by Mx along x, My along y, Mz along z (the products of exchange.jl)
function _tx3g!(B, A, T, Mx, My, Mz, gs)
    n1, n2, n3 = size(A)
    mul!(reshape(T, n1, n2 * n3), Mx, reshape(A, n1, n2 * n3))
    _dim2!(B, T, transpose(My), gs)
    mul!(reshape(T, n1 * n2, n3), reshape(B, n1 * n2, n3), transpose(Mz))
    copyto!(B, T)
    B
end
@kernel function _qdiv_k!(T2, @Const(ex), @Const(ey), @Const(ez), σn::Float64, λ::Float64)
    i, j, k = @index(Global, NTuple)
    @inbounds T2[i,j,k] /= σn * (ex[i] + ey[j] + ez[k]) - λ
end
@kernel function _qdelta_k!(δ, @Const(S), Nx::Int, j0::Int, k0::Int)
    j, k = @index(Global, NTuple)
    @inbounds begin
        δ[1, j+j0, k+k0] = 0.0; δ[Nx+1, j+j0, k+k0] = 0.0
        for i in 2:Nx; δ[i, j+j0, k+k0] = S[i-1, j, k]; end
    end
end
function _qp_potential!(g::GState, gs)
    st = gs.st; md = st.model; q = gs.qp; gr = st.grid
    (; Nx, x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    fill!(q.S, 0.0)
    gb = ntuple(µ -> ntuple(i -> TDGL.stiffness(md.bands[µ], i), 3), 2)
    if gs.mat === nothing
        _launch(gs, _conv_k!, (x2 - x1 + 2, y2 - y1 + 2, z2 - z1 + 2), q.S, g.G, gb[1], gb[2], md.γ[1], md.γ[2], st.dragν,
                g.Vx, g.Vy, g.Vz, gs.τx, gs.τy, gs.τz, 1 / dx^2, 1 / dy^2, 1 / dz^2, q.s[1], q.s[2], x1, y1, z1, q.j0 - 1, q.k0 - 1)
    else                       # with materials: the link weights ρ of each band and the Josephson row of the node's material
        _launch(gs, _convmat_k!, (x2 - x1 + 2, y2 - y1 + 2, z2 - z1 + 2), q.S, g.G, gb[1], gb[2], q.γt[1], q.γt[2], gs.mat.idx,
                st.dragν, g.Vx, g.Vy, g.Vz, gs.mat.ρ[1], gs.mat.ρ[2], 1 / dx^2, 1 / dy^2, 1 / dz^2, q.s[1], q.s[2], x1, y1, z1,
                q.j0 - 1, q.k0 - 1)
    end
    _hsync(gs)
    if q.var !== nothing                                               # variable coefficients: their CG
        d = q.σn * (2 / gr.dx^2 + 2 / gr.dy^2 + 2 / gr.dz^2) + q.λ0
        _qp_cg_var_dev!(q.S, q.var, d, (Z, R) -> _qsep!(Z, R, q, max(q.λ0, 1e-12 * q.σn), gs), gs)
    elseif q.cg === nothing
        _qsep!(q.S, q.S, q, q.λ0, gs)                                  # _qp_sep!(S, S, ...): to the modes, divided, back
    else
        dx, dy, dz = gr.dx, gr.dy, gr.dz
        _qp_cg_dev!(q.S, q.cg, q.σn, max(q.λ0, 1e-12 * q.σn), dx, dy, dz, (Z, R) -> _qsep!(Z, R, q, q.λ0, gs), gs)
    end
    _launch(gs, _qdelta_k!, (size(q.S, 2), size(q.S, 3)), q.δ, q.S, Nx, q.j0 - 1, q.k0 - 1)
    g
end
@kernel function _qturn_k!(F, @Const(δ), c::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds F[i,j,k] *= cis(-c * δ[i,j,k])
end
function _qp_turn!(g::GState, gs)
    st = gs.st; gr = st.grid; q = gs.qp
    m = (-q.s[2], q.s[1])
    for µ in 1:2
        _launch(gs, _qturn_k!, (gr.x2 - gr.x1, gr.y2 - gr.y1, gr.z2 - gr.z1), g.F[µ], q.δ, m[µ] * st.dt, gr.x1, gr.y1, gr.z1)
    end
    g
end

# _qp_sep!: Z ← (σn∇² − λ)⁻¹ R on the region, separable (the modes of the 1D Laplacians)
function _qsep!(Z, R, q, λ, gs)
    _tx3g!(q.T2, R, q.T1, q.Wx, q.Wy, q.Wz, gs)
    _launch(gs, _qdiv_k!, size(q.T2), q.T2, q.ex, q.ey, q.ez, q.σn, λ)
    _hsync(gs)
    _tx3g!(Z, q.T2, q.T1, q.Vx, q.Vy, q.Vz, gs)
    Z
end
# _qp_cg! (a sculpted sample: the links to the vacuum cut): conjugate gradients preconditioned by the separable solve.  The
# weighted dot products are reductions on the device (another order of additions than the CPU loop): the solution to the
# tolerance of the solve (1e-10), not to the bit.
struct GQPCG{B,R}
    cond::B; wt::R; r::R; z::R; p::R; q::R; x::R
    tol::Float64; maxit::Int; iters::Vector{Int}
end
GQPCG(backend, ps) = GQPCG(_dev(backend, Array{Bool}(ps.cond)), _dev(backend, ps.wt), _dev(backend, ps.r), _dev(backend, ps.z),
                           _dev(backend, ps.p), _dev(backend, ps.q), _dev(backend, ps.x), ps.tol, ps.maxit, ps.iters)
@kernel function _qapply_k!(y, @Const(x), @Const(c), ax::Float64, ay::Float64, az::Float64, λ::Float64, n1::Int, n2::Int, n3::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds if !c[i,j,k]
        y[i,j,k] = (2ax + 2ay + 2az + λ) * x[i,j,k]
    else
        v = x[i,j,k]; acc = λ * v
        acc += ax * (v - (i > 1 ? (c[i-1,j,k] ? x[i-1,j,k] : v) : 0.0))
        acc += ax * (v - (i < n1 ? (c[i+1,j,k] ? x[i+1,j,k] : v) : 0.0))
        if n2 > 1
            if j == 1
                c[i,2,k] && (acc += 2ay * (v - x[i,2,k]))
            elseif j == n2
                c[i,n2-1,k] && (acc += 2ay * (v - x[i,n2-1,k]))
            else
                c[i,j-1,k] && (acc += ay * (v - x[i,j-1,k])); c[i,j+1,k] && (acc += ay * (v - x[i,j+1,k]))
            end
        end
        if n3 > 1
            if k == 1
                c[i,j,2] && (acc += 2az * (v - x[i,j,2]))
            elseif k == n3
                c[i,j,n3-1] && (acc += 2az * (v - x[i,j,n3-1]))
            else
                c[i,j,k-1] && (acc += az * (v - x[i,j,k-1])); c[i,j,k+1] && (acc += az * (v - x[i,j,k+1]))
            end
        end
        y[i,j,k] = acc
    end
end
@kernel function _qproj_k!(z, @Const(c))                     # z ← −z on the conducting unknowns, 0 elsewhere
    I = @index(Global, Linear)
    @inbounds z[I] = c[I] ? -z[I] : 0.0
end
@kernel function _qmask_k!(S, @Const(c))
    I = @index(Global, Linear)
    @inbounds c[I] || (S[I] = 0.0)
end
_wdotg(a, b, w) = sum(w .* a .* b)
function _qp_cg_dev!(S, ps::GQPCG, σn, λ, hx, hy, hz, sep!, gs)
    r, z, p, q, x, w, c = ps.r, ps.z, ps.p, ps.q, ps.x, ps.wt, ps.cond
    n1, n2, n3 = size(S); ax = σn / hx^2; ay = σn / hy^2; az = σn / hz^2
    A!(y, v) = (_launch(gs, _qapply_k!, (n1, n2, n3), y, v, c, ax, ay, az, λ, n1, n2, n3); _hsync(gs); y)
    _launch(gs, _qmask_k!, (length(S),), S, c); _hsync(gs)
    @. r = -S
    bn = sqrt(_wdotg(r, r, w)); bn == 0 && (fill!(S, 0.0); ps.iters[1] = 0; return S)
    prec!(zz, rr) = (sep!(zz, rr); _launch(gs, _qproj_k!, (length(zz),), zz, c); _hsync(gs); zz)
    prec!(z, r)
    copyto!(x, z)
    A!(q, x); @. r = r - q
    prec!(z, r); copyto!(p, z)
    rz = _wdotg(r, z, w); it = 0
    while sqrt(_wdotg(r, r, w)) > ps.tol * bn && it < ps.maxit
        A!(q, p)
        α = rz / _wdotg(p, q, w)
        @. x += α * p; @. r -= α * q
        prec!(z, r)
        rz2 = _wdotg(r, z, w); β = rz2 / rz; rz = rz2
        @. p = z + β * p
        it += 1
    end
    ps.iters[1] = it
    copyto!(S, x)
end

# the conversion with materials (qp_potential!, the mt branch): ρ of each band, the node's Josephson rows
@kernel function _convmat_k!(S, G, g1, g2, γt1, γt2, @Const(idx), dragν, @Const(Vx), @Const(Vy), @Const(Vz), ρ1, ρ2,
                             rx2::Float64, ry2::Float64, rz2::Float64, s1::Float64, s2::Float64, x1::Int, y1::Int, z1::Int,
                             j0::Int, k0::Int)
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        m = idx[i,j,k]
        c1 = TDGL._conv(G, 1, g1, γt1[m], dragν, Vx, Vy, Vz, ρ1[1], ρ1[2], ρ1[3], i, j, k, rx2, ry2, rz2)
        c2 = TDGL._conv(G, 2, g2, γt2[m], dragν, Vx, Vy, Vz, ρ2[1], ρ2[2], ρ2[3], i, j, k, rx2, ry2, rz2)
        S[i-1, j-j0, k-k0] = c2 / s2 - c1 / s1
    end
end
# QPVar on the device and _qp_cg_var! (the weighted dots as reductions: to the tolerance of the solve)
struct GQPVar{I,R}
    typ::I; cx::R; cy::R; cz::R; λ::R
    wt::R; r::R; z::R; p::R; q::R; x::R
    tol::Float64; maxit::Int; iters::Vector{Int}
end
GQPVar(backend, v, ps) = GQPVar(_dev(backend, v.typ), _dev(backend, v.cx), _dev(backend, v.cy), _dev(backend, v.cz), _dev(backend, v.λ),
                                _dev(backend, ps.wt), _dev(backend, ps.r), _dev(backend, ps.z), _dev(backend, ps.p), _dev(backend, ps.q),
                                _dev(backend, ps.x), ps.tol, ps.maxit, ps.iters)
@inline _nbv(t, x, c, a, ii, jj, kk) = @inbounds t[ii,jj,kk] == 1 ? c * (a - x[ii,jj,kk]) : t[ii,jj,kk] == 2 ? c * a : 0.0
@kernel function _qapply_var_k!(y, @Const(x), @Const(t), @Const(cx), @Const(cy), @Const(cz), @Const(λ), d::Float64,
                                n1::Int, n2::Int, n3::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds if t[i,j,k] != 1
        y[i,j,k] = d * x[i,j,k]
    else
        a = x[i,j,k]; acc = λ[i,j,k] * a
        acc += i > 1 ? _nbv(t, x, cx[i,j,k], a, i-1, j, k) : cx[1,j,k] * a
        acc += i < n1 ? _nbv(t, x, cx[i+1,j,k], a, i+1, j, k) : cx[n1+1,j,k] * a
        if n2 > 1
            if j == 1
                acc += 2 * _nbv(t, x, cy[i,1,k], a, i, 2, k)
            elseif j == n2
                acc += 2 * _nbv(t, x, cy[i,n2-1,k], a, i, n2-1, k)
            else
                acc += _nbv(t, x, cy[i,j-1,k], a, i, j-1, k) + _nbv(t, x, cy[i,j,k], a, i, j+1, k)
            end
        end
        if n3 > 1
            if k == 1
                acc += 2 * _nbv(t, x, cz[i,j,1], a, i, j, 2)
            elseif k == n3
                acc += 2 * _nbv(t, x, cz[i,j,n3-1], a, i, j, n3-1)
            else
                acc += _nbv(t, x, cz[i,j,k-1], a, i, j, k-1) + _nbv(t, x, cz[i,j,k], a, i, j, k+1)
            end
        end
        y[i,j,k] = acc
    end
end
@kernel function _vproj_k!(z, @Const(t))
    I = @index(Global, Linear)
    @inbounds z[I] = t[I] == 1 ? -z[I] : 0.0
end
@kernel function _vmask_k!(S, @Const(t))
    I = @index(Global, Linear)
    @inbounds t[I] == 1 || (S[I] = 0.0)
end
function _qp_cg_var_dev!(S, v::GQPVar, d, sep!, gs)
    r, z, p, q, x, w, t = v.r, v.z, v.p, v.q, v.x, v.wt, v.typ
    n1, n2, n3 = size(S)
    A!(y, u) = (_launch(gs, _qapply_var_k!, (n1, n2, n3), y, u, t, v.cx, v.cy, v.cz, v.λ, d, n1, n2, n3); _hsync(gs); y)
    _launch(gs, _vmask_k!, (length(S),), S, t); _hsync(gs)
    @. r = -S
    bn = sqrt(_wdotg(r, r, w)); bn == 0 && (fill!(S, 0.0); v.iters[1] = 0; return S)
    prec!(zz, rr) = (sep!(zz, rr); _launch(gs, _vproj_k!, (length(zz),), zz, t); _hsync(gs); zz)
    prec!(z, r); copyto!(x, z)
    A!(q, x); @. r = r - q
    prec!(z, r); copyto!(p, z)
    rz = _wdotg(r, z, w); it = 0
    while sqrt(_wdotg(r, r, w)) > v.tol * bn && it < v.maxit
        A!(q, p)
        α = rz / _wdotg(p, q, w)
        @. x += α * p; @. r -= α * q
        prec!(z, r)
        rz2 = _wdotg(r, z, w); β = rz2 / rz; rz = rz2
        @. p = z + β * p
        it += 1
    end
    v.iters[1] = it
    copyto!(S, x)
end

# N normal fluids (exchange.jl, QPExchangeN): the conversion of every band, the right sides, the modes of S⁻¹L (each a separable
# Helmholtz solve, as the two-band δ), the potentials back on the nodes, each band turned by e^{−iμ_µ dt}
struct GQPN{N,M,V,R}
    Vx::M; Wx::M; ex::V; Vy::M; Wy::M; ey::V; Vz::M; Wz::M; ez::V
    R::NTuple{N,R}; Mo::NTuple{N,R}; μ::NTuple{N,R}; T1::R; T2::R
    s::NTuple{N,Float64}; P::NTuple{N,NTuple{N,Float64}}; Pinv::NTuple{N,NTuple{N,Float64}}; λ::NTuple{N,Float64}
    σn::Float64; j0::Int; k0::Int
    cg::Any                                 # a sculpted sample: GQPCG, or nothing
    Gm::NTuple{N,NTuple{N,Float64}}         # the exchange conductances (the Joule heat)
end
function GQPN(backend, q::TDGL.QPExchangeN{N}) where N
    d(x) = _dev(backend, x)
    GQPN(d(q.Vx), d(q.Wx), d(q.ex), d(q.Vy), d(q.Wy), d(q.ey), d(q.Vz), d(q.Wz), d(q.ez), map(d, q.R), map(d, q.M), map(d, q.μ),
         d(q.T1), d(q.T2), q.s, ntuple(a -> ntuple(b -> q.P[a, b], N), N), ntuple(a -> ntuple(b -> q.Pinv[a, b], N), N),
         Tuple(q.λ), q.σn, q.ps.j0, q.ps.k0, q.ps.cond === nothing ? nothing : GQPCG(backend, q.ps),
         ntuple(a -> ntuple(b -> q.Gm[a, b], N), N))
end
@kernel function _convN_k!(R, G, gb, γ, dragν, @Const(Vx), @Const(Vy), @Const(Vz), @Const(τx), @Const(τy), @Const(τz),
                           rx2::Float64, ry2::Float64, rz2::Float64, s, x1::Int, y1::Int, z1::Int, j0::Int, k0::Int, ::Val{N}) where N
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        c = ntuple(µ -> TDGL._conv(G, µ, gb[µ], γ[µ], dragν, Vx, Vy, Vz, τx, τy, τz, i, j, k, rx2, ry2, rz2), Val(N))
        C = c[1]
        for µ in 2:N; C += c[µ]; end
        for µ in 1:N; R[µ][i-1, j-j0, k-k0] = c[µ] / s[µ] - C; end
    end
end
@kernel function _tomodes_k!(Mo, R, Pinv, ::Val{N}) where N
    I = @index(Global, Linear)
    @inbounds for kk in 1:N
        acc = 0.0
        for µ in 1:N; acc += Pinv[kk][µ] * R[µ][I]; end
        Mo[kk][I] = acc
    end
end
@kernel function _frommodes_k!(pot, Mo, P, Nx::Int, j0::Int, k0::Int, ::Val{N}) where N   # (pot: µ and μ are one name)
    jy, kz = @index(Global, NTuple)
    @inbounds for b in 1:N, ix in 2:Nx
        acc = 0.0
        for kk in 1:N; acc += P[b][kk] * Mo[kk][ix-1, jy, kz]; end
        pot[b][ix, jy+j0, kz+k0] = acc
    end
end
function _qpN_potential!(g::GState{N}, gs) where N
    st = gs.st; md = st.model; q = gs.qp; gr = st.grid
    (; Nx, x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    for r in q.R; fill!(r, 0.0); end
    gb = ntuple(µ -> ntuple(i -> TDGL.stiffness(md.bands[µ], i), 3), N)
    γ = ntuple(µ -> Tuple(md.γ[µ]), N)
    _launch(gs, _convN_k!, (x2 - x1 + 2, y2 - y1 + 2, z2 - z1 + 2), q.R, g.G, gb, γ, st.dragν, g.Vx, g.Vy, g.Vz,
            gs.τx, gs.τy, gs.τz, 1 / dx^2, 1 / dy^2, 1 / dz^2, q.s, x1, y1, z1, q.j0 - 1, q.k0 - 1, Val(N))
    _launch(gs, _tomodes_k!, (length(q.R[1]),), q.Mo, q.R, q.Pinv, Val(N))
    _hsync(gs)
    for kk in 1:N                                                     # _qp_sep!(M, M, ..., λ_k), or its CG
        λ = q.λ[kk]
        if q.cg === nothing
            _qsep!(q.Mo[kk], q.Mo[kk], q, λ, gs)
        elseif λ > 0 || maximum(abs, q.Mo[kk]) > 0
            _qp_cg_dev!(q.Mo[kk], q.cg, q.σn, max(λ, 1e-12 * q.σn), dx, dy, dz, (Z, R) -> _qsep!(Z, R, q, λ, gs), gs)
        end
    end
    for m in q.μ; fill!(m, 0.0); end
    _launch(gs, _frommodes_k!, (size(q.Mo[1], 2), size(q.Mo[1], 3)), q.μ, q.Mo, q.P, Nx, q.j0 - 1, q.k0 - 1, Val(N))
    g
end
function _qpN_turn!(g::GState{N}, gs) where N
    st = gs.st; gr = st.grid; q = gs.qp
    for µ in 1:N                                                       # F *= cis(−dt μ_µ): _qturn_k! with c = dt
        _launch(gs, _qturn_k!, (gr.x2 - gr.x1, gr.y2 - gr.y1, gr.z2 - gr.z1), g.F[µ], q.μ[µ], st.dt, gr.x1, gr.y1, gr.z1)
    end
    g
end

# ---------------------------------------------------------------------------
#  the open boundary (openbc.jl): the CPU OpenBC keeps the clock and the ramp (old, new: the checkpoints hold them); every
#  `every` steps h is brought to the host and h_δ recomputed there by the CPU code's functions (its FFT and its bits), then
#  old and new go to the device, where every step adds the ramp to the faces after the bar field (_open_add!)
# ---------------------------------------------------------------------------
struct GOpen{R}
    ob::TDGL.OpenBC
    old::NTuple{3,R}; new::NTuple{3,R}       # device copies of the ramp
    h::NTuple{3,Array{Float64,3}}            # host buffers of h for the recomputation
end
GOpen(backend, ob) = GOpen(ob, map(x -> _dev(backend, x), ob.old), map(x -> _dev(backend, x), ob.new),
                           (zeros(0, 0, 0), zeros(0, 0, 0), zeros(0, 0, 0)))
function _open_up!(gs)
    o = gs.open
    for c in 1:3; copyto!(o.old[c], o.ob.old[c]); copyto!(o.new[c], o.ob.new[c]); end
end
# _delta_current! of openbc.jl on host arrays of h (its body)
function _delta_current_h!(ob, hx, hy, hz, gr, κ2::Float64, HJ::Float64)
    (; Nx, Ny, Nz, dx, dy, dz) = gr
    bx, by, bz = ob.buf
    fill!(bx, 0.0); fill!(by, 0.0); fill!(bz, 0.0)
    @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx-1
        bx[i,j,k] = κ2 * ((hz[i,j,k] - hz[i,j-1,k]) / dy - (hy[i,j,k] - hy[i,j,k-1]) / dz) - HJ * ob.Jbar[j,k]
    end
    @inbounds for k in 2:Nz, j in 2:Ny-1, i in 2:Nx
        by[i,j,k] = κ2 * ((hx[i,j,k] - hx[i,j,k-1]) / dz - (hz[i,j,k] - hz[i-1,j,k]) / dx)
    end
    @inbounds for k in 2:Nz-1, j in 2:Ny, i in 2:Nx
        bz[i,j,k] = κ2 * ((hy[i,j,k] - hy[i-1,j,k]) / dx - (hx[i,j,k] - hx[i,j-1,k]) / dy)
    end
    ob
end
# open_update!: the clock, and the recomputation every `every` steps (with h of the device)
function _open_update!(g::GState, gs, HJ::Float64)
    o = gs.open; ob = o.ob; gr = gs.st.grid
    first = ob.count < 0
    ob.count += 1
    if ob.count >= ob.every || first
        for c in 1:3; copyto!(ob.old[c], ob.new[c]); end
        KA.synchronize(gs.backend)
        hx, hy, hz = Array(g.hx), Array(g.hy), Array(g.hz)
        _delta_current_h!(ob, hx, hy, hz, gr, gs.st.model.κ^2, HJ)
        TDGL._convolve!(ob)
        TDGL._face_field!(ob.new, ob, gr)
        first && for c in 1:3; copyto!(ob.old[c], ob.new[c]); end
        ob.count = 0
        _open_up!(gs)
    end
    g
end
# _open_add!: h_δ at the current point of the ramp, on the six faces (its loops as broadcasts, in its order)
function _open_add!(g::GState, gs)
    (; Nx, Ny, Nz) = gs.st.grid
    ob = gs.open.ob; o = gs.open.old; n = gs.open.new
    f = min((ob.count + 1) / ob.every, 1.0)
    hx, hy, hz = g.hx, g.hy, g.hz
    @views begin
        for j in (1, Ny); hz[2:Nx-1, j, :] .+= o[3][2:Nx-1, j, :] .+ f .* (n[3][2:Nx-1, j, :] .- o[3][2:Nx-1, j, :]); end
        for j in (1, Ny); hx[2:Nx, j, 1:Nz] .= o[1][2:Nx, j, 1:Nz] .+ f .* (n[1][2:Nx, j, 1:Nz] .- o[1][2:Nx, j, 1:Nz]); end
        for k in (1, Nz); hy[2:Nx-1, :, k] .+= o[2][2:Nx-1, :, k] .+ f .* (n[2][2:Nx-1, :, k] .- o[2][2:Nx-1, :, k]); end
        for k in (1, Nz); hx[2:Nx, 1:Ny, k] .= o[1][2:Nx, 1:Ny, k] .+ f .* (n[1][2:Nx, 1:Ny, k] .- o[1][2:Nx, 1:Ny, k]); end
        for i in (1, Nx); hz[i, 1:Ny, :] .+= o[3][i, 1:Ny, :] .+ f .* (n[3][i, 1:Ny, :] .- o[3][i, 1:Ny, :]); end
        for i in (1, Nx); hy[i, 1:Ny+1, 1:Nz] .+= o[2][i, 1:Ny+1, 1:Nz] .+ f .* (n[2][i, 1:Ny+1, 1:Nz] .- o[2][i, 1:Ny+1, 1:Nz]); end
    end
    g
end

# ---------------------------------------------------------------------------
#  the stepper: the CPU Stepper (its coefficients) and the device arrays of the step
# ---------------------------------------------------------------------------
struct GPUStepper{N,B,C,R,S,DD,BF,VA,MA,TH,NZ,QP,OB}
    st::Stepper{N}
    backend::B
    f::R                                   # masks.f
    τx::R; τy::R; τz::R                    # link masks of the supercurrent
    W::C; W2::C                            # stage buffers (W2 empty without RKL)
    L::NTuple{6,C}                         # substep links and their rotation
    Ω::R                                   # rotating frame: the rate of the band being stepped
    Ωs::R                                  # with noise: the copy for its low-pass
    Rl::NTuple{3,R}                        # rotating frame: the phase rate of each link over a step
    sep::S                                 # GSep, GPCG (insulating padding) or nothing (explicit field step)
    D::DD                                  # σ_l/σ_n per link for the explicit step with a padding (nothing otherwise)
    bar::BF                                # the bar field on the outer faces, or nothing (slab)
    vac::VA                                # a sculpted sample: its vacuum nodes (Bool), or nothing
    mat::MA                                # materials per region (GMat), or nothing
    θ::TH                                  # the heat model: the temperature field on the device, or nothing
    nz::NZ                                 # thermal noise: host and device buffers of the draws, or nothing
    qp::QP                                 # two normal fluids (GQP), or nothing
    open::OB                               # the open boundary (GOpen), or nothing
end

"""
    GPUStepper(st, backend) -> GPUStepper

The device arrays of the step of `st` on `backend` (`CUDABackend()`, `CPU()`, …).  Errors when `st` has an option this
version does not run (`supported`).
"""
function GPUStepper(st::Stepper{N}, backend) where N
    ok, why = supported(st)
    ok || error("TDGLGPU does not run $why yet (the CPU step does: TDGL.step!)")
    mk = st.masks; ws = st.ws
    sep = st.imp === nothing ? nothing : st.imp isa PCGSolver ? GPCG(backend, st.imp) : GSep(backend, st.imp)
    D = st.imp === nothing && st.D !== nothing ? _d3(backend, st.D) : nothing
    bar = st.bar === nothing ? nothing : map(x -> _dev(backend, x), st.bar)
    vac = mk.vac === nothing ? nothing : _dev(backend, Array{Bool}(mk.vac))
    mat = st.mt === nothing ? nothing : GMat(backend, st.mt)
    θ = st.heat === nothing ? nothing : _dev(backend, st.heat.θ)
    nz = st.noise === nothing ? nothing : GNoise(backend, st)
    qp = st.qpx === nothing ? nothing : st.qpx isa TDGL.QPExchange ? GQP(backend, st.qpx, st.mt) : GQPN(backend, st.qpx)
    ob = st.open === nothing ? nothing : GOpen(backend, st.open)
    GPUStepper{N,typeof(backend),typeof(_dev(backend, ws.W)),typeof(_dev(backend, mk.f)),typeof(sep),typeof(D),typeof(bar),typeof(vac),typeof(mat),typeof(θ),typeof(nz),typeof(qp),typeof(ob)}(
        st, backend, _dev(backend, mk.f), _dev(backend, mk.τx), _dev(backend, mk.τy), _dev(backend, mk.τz),
        _dev(backend, ws.W), _dev(backend, ws.W2), ntuple(q -> _dev(backend, ws.L[q]), 6),
        _dev(backend, ws.Ω), _dev(backend, ws.Ωs), ntuple(q -> _dev(backend, ws.R[q]), 3), sep, D, bar, vac, mat, θ, nz, qp, ob)
end

# a kernel over ndrange on the backend of gs; the CPU backend runs its kernels in order, the host operations (BLAS,
# broadcasts on CPU arrays) wait for them (_hsync); a GPU orders them on its stream
_wg(nd::NTuple{1,Int}) = (256,)
_wg(nd::NTuple{2,Int}) = (32, 8)
_wg(nd::NTuple{3,Int}) = (32, 4, 2)
function _launch(gs, kern, nd, args...)
    all(>(0), nd) || return nothing
    kern(gs.backend, _wg(nd))(args...; ndrange=nd)
    nothing
end
_hsync(gs) = gs.backend isa KA.CPU ? KA.synchronize(gs.backend) : nothing

# ---------------------------------------------------------------------------
#  the bands (bands.jl)
# ---------------------------------------------------------------------------
# _band_row! with one material, no heat field, no drag, no noise: its body
# _drag of bands.jl (the arrays as AbstractArray: on a device they are not Array)
@inline _dragk(::Nothing, G, µ, Vx, Vy, Vz, vxm, vym, vzm, i, j, k, acc) = acc
@inline function _dragk(dr::NTuple{N,NTuple{3,Float64}}, G, µ, Vx, Vy, Vz, vxm, vym, vzm, i, j, k, acc) where N
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
@inline _epsk(ε, a, ::Nothing, i, j, k) = ε
@inline _epsk(ε, a, θ, i, j, k) = @inbounds a - θ[i,j,k]
@kernel function _band_kernel!(D, @Const(S), G, µ::Int, @Const(Vx), @Const(Vy), @Const(Vz), cf::BandCoef{N},
                               @Const(f), hard::Bool, x1::Int, y1::Int, z1::Int, a::Float64, θ, dr, rk::Bool) where N
    I = @index(Global, NTuple)
    i = x1 + I[1]; j = y1 + I[2]; k = z1 + I[3]
    kx, ky, kz, c, ε, β, cpl, γ2 = cf.kx, cf.ky, cf.kz, cf.c, cf.ε, cf.β, cf.cpl, cf.γ2
    @inbounds begin
        g = S[i,j,k]; fm = f[i,j,k]
        if hard && fm == 0.0
            D[i,j,k] = 0.0 + 0.0im
        else
            vxm = conj(Vx[i-1,j,k]); vym = conj(Vy[i,j-1,k]); vzm = conj(Vz[i,j,k-1])
            tx = kx*((Vx[i,j,k]*S[i+1,j,k] - 2*g) + vxm*S[i-1,j,k])
            ty = ky*((Vy[i,j,k]*S[i,j+1,k] - 2*g) + vym*S[i,j-1,k])
            tz = kz*((Vz[i,j,k]*S[i,j,k+1] - 2*g) + vzm*S[i,j,k-1])
            nl = (c*g)*(fm*_epsk(ε, a, θ, i, j, k) - β*abs2(g))
            acc = (((g + tx) + ty) + tz) + nl
            for ν in 1:N
                ν == µ || (acc += cpl[ν]*G[ν][i,j,k])
            end
            acc = _dragk(dr, G, µ, Vx, Vy, Vz, vxm, vym, vzm, i, j, k, acc)
            D[i,j,k] = γ2 > 0 ? (rk ? TDGL.kwt_update_rot(g, acc - g, γ2) : TDGL.kwt_update(g, acc - g, γ2)) : acc
        end
    end
end
# _band_row_vac!: a sculpted sample; a vacuum node stays at Ψ = 0 and each neighbour enters with the mask of its link
@kernel function _band_vac_kernel!(D, @Const(S), G, µ::Int, @Const(Vx), @Const(Vy), @Const(Vz), cf::BandCoef{N},
                                   @Const(f), hard::Bool, @Const(vac), @Const(τx), @Const(τy), @Const(τz),
                                   x1::Int, y1::Int, z1::Int, rk::Bool) where N
    I = @index(Global, NTuple)
    i = x1 + I[1]; j = y1 + I[2]; k = z1 + I[3]
    kx, ky, kz, c, ε, β, cpl, γ2 = cf.kx, cf.ky, cf.kz, cf.c, cf.ε, cf.β, cf.cpl, cf.γ2
    @inbounds begin
        g = S[i,j,k]; fm = f[i,j,k]
        if vac[i,j,k] || (hard && fm == 0.0)
            D[i,j,k] = 0.0 + 0.0im
        else
            tx = kx*(τx[i,j,k]*(Vx[i,j,k]*S[i+1,j,k] - g) + τx[i-1,j,k]*(conj(Vx[i-1,j,k])*S[i-1,j,k] - g))
            ty = ky*(τy[i,j,k]*(Vy[i,j,k]*S[i,j+1,k] - g) + τy[i,j-1,k]*(conj(Vy[i,j-1,k])*S[i,j-1,k] - g))
            tz = kz*(τz[i,j,k]*(Vz[i,j,k]*S[i,j,k+1] - g) + τz[i,j,k-1]*(conj(Vz[i,j,k-1])*S[i,j,k-1] - g))
            acc = (((g + tx) + ty) + tz) + (c*g)*(fm*ε - β*abs2(g))
            for ν in 1:N
                ν == µ || (acc += cpl[ν]*G[ν][i,j,k])
            end
            D[i,j,k] = γ2 > 0 ? (rk ? TDGL.kwt_update_rot(g, acc - g, γ2) : TDGL.kwt_update(g, acc - g, γ2)) : acc
        end
    end
end
function _band_pass!(D, S, g::GState{N}, µ::Int, Vx, Vy, Vz, cf::BandCoef, gs, dr=nothing) where N
    gr = gs.st.grid
    nd = (gr.x2 - gr.x1, gr.y2 - gr.y1, gr.z2 - gr.z1)
    if gs.mat !== nothing
        M = gs.mat
        T = cf === gs.st.bcs[µ] ? M.stage[µ] : M.full[µ]      # the divided tables with the stage coefficients, as _band_pass!
        ρx, ρy, ρz = M.ρ[µ]
        if gs.vac === nothing
            _launch(gs, _band_mat_kernel!, nd, D, S, g.G, µ, Vx, Vy, Vz, M.idx, T, ρx, ρy, ρz, gs.f, gs.st.hard,
                    gr.x1, gr.y1, gr.z1, Val(N), M.a[µ], gs.θ, gs.st.noise !== nothing)
        else
            _launch(gs, _band_matvac_kernel!, nd, D, S, g.G, µ, Vx, Vy, Vz, M.idx, T, ρx, ρy, ρz, gs.f, gs.st.hard, gs.vac,
                    gr.x1, gr.y1, gr.z1, Val(N), M.a[µ], gs.θ, gs.st.noise !== nothing)
        end
        _launch(gs, _zero_at_k!, (length(M.abs[µ]),), D, M.abs[µ])
    elseif gs.vac === nothing
        _launch(gs, _band_kernel!, nd, D, S, g.G, µ, Vx, Vy, Vz, cf, gs.f, gs.st.hard, gr.x1, gr.y1, gr.z1,
                gs.st.model.bands[µ].a, gs.θ, dr, gs.st.noise !== nothing)
    else
        _launch(gs, _band_vac_kernel!, nd, D, S, g.G, µ, Vx, Vy, Vz, cf, gs.f, gs.st.hard, gs.vac, gs.τx, gs.τy, gs.τz,
                gr.x1, gr.y1, gr.z1, gs.st.noise !== nothing)
    end
    D
end

# ghosts!: the covariant Neumann ghosts of one band on the six faces, x, then y, then z (the precedence of the loops)
@kernel function _ghx!(X, @Const(Vx), x1::Int, x2::Int, y1::Int, z1::Int)
    a, b = @index(Global, NTuple); j = y1 - 1 + a; k = z1 - 1 + b
    @inbounds begin
        X[x1,j,k]   = Vx[x1,j,k] * X[x1+1,j,k]
        X[x2+1,j,k] = conj(Vx[x2,j,k]) * X[x2,j,k]
    end
end
@kernel function _ghy!(X, @Const(Vy), x1::Int, y1::Int, y2::Int, z1::Int)
    a, b = @index(Global, NTuple); i = x1 - 1 + a; k = z1 - 1 + b
    @inbounds begin
        X[i,y1,k]   = Vy[i,y1,k] * X[i,y1+1,k]
        X[i,y2+1,k] = conj(Vy[i,y2,k]) * X[i,y2,k]
    end
end
@kernel function _ghz!(X, @Const(Vz), x1::Int, y1::Int, z1::Int, z2::Int)
    a, b = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b
    @inbounds begin
        X[i,j,z1]   = Vz[i,j,z1] * X[i,j,z1+1]
        X[i,j,z2+1] = conj(Vz[i,j,z2]) * X[i,j,z2]
    end
end
function _ghosts!(X, Vx, Vy, Vz, gs)
    (; x1, x2, y1, y2, z1, z2) = gs.st.grid
    _launch(gs, _ghx!, (y2 - y1 + 2, z2 - z1 + 2), X, Vx, x1, x2, y1, z1)
    _launch(gs, _ghy!, (x2 - x1 + 2, z2 - z1 + 2), X, Vy, x1, y1, y2, z1)
    _launch(gs, _ghz!, (x2 - x1 + 2, y2 - y1 + 2), X, Vz, x1, y1, z1, z2)
    X
end

# the link ranges of bands.jl (_rx, _ry, _rz): (origin − 1, lengths)
_rx(gr) = ((gr.x1 - 1, gr.y1 - 1, gr.z1 - 1), (gr.x2 - gr.x1 + 1, gr.y2 - gr.y1 + 2, gr.z2 - gr.z1 + 2))
_ry(gr) = ((gr.x1 - 1, gr.y1 - 1, gr.z1 - 1), (gr.x2 - gr.x1 + 2, gr.y2 - gr.y1 + 1, gr.z2 - gr.z1 + 2))
_rz(gr) = ((gr.x1 - 1, gr.y1 - 1, gr.z1 - 1), (gr.x2 - gr.x1 + 2, gr.y2 - gr.y1 + 2, gr.z2 - gr.z1 + 1))

@kernel function _rotfrac_k!(L, @Const(V), @Const(U), c::Float64, o::NTuple{3,Int})
    a, b, e = @index(Global, NTuple); i = o[1] + a; j = o[2] + b; k = o[3] + e
    @inbounds begin
        v = V[i,j,k]
        q = v * conj(U[i,j,k])
        sf = -imag(q)
        if abs(sf) <= 0.05
            t = (sf + sf*sf*sf/6) * c
            t2 = t*t
            L[i,j,k] = v * complex(1 - t2/2, -t*(1 - t2/6))
        else
            L[i,j,k] = v * cis(angle(q) * c)
        end
    end
end
@kernel function _rot_k!(L, R, @Const(V), @Const(U), n::Int, o::NTuple{3,Int})
    a, b, e = @index(Global, NTuple); i = o[1] + a; j = o[2] + b; k = o[3] + e
    @inbounds begin
        v = V[i,j,k]; L[i,j,k] = v
        q = v * conj(U[i,j,k])
        sf = -imag(q)
        if abs(sf) <= 0.05
            t = (sf + sf*sf*sf/6) / n
            t2 = t*t
            R[i,j,k] = complex(1 - t2/2, -t*(1 - t2/6))
        else
            R[i,j,k] = cis(angle(q) / n)
        end
    end
end
@kernel function _mulby_k!(L, @Const(R), o::NTuple{3,Int})
    a, b, e = @index(Global, NTuple); i = o[1] + a; j = o[2] + b; k = o[3] + e
    @inbounds L[i,j,k] *= R[i,j,k]
end
# _combine_log! (noise with KWT): Y = A exp(μ log(C/A)), the linear combination where |A|² or |C|² is under _ROT_MIN2
@kernel function _combine_log_k!(C, @Const(A), μ::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds begin
        p = A[i,j,k]; c = C[i,j,k]
        C[i,j,k] = abs2(p) > TDGL._ROT_MIN2 && abs2(c) > TDGL._ROT_MIN2 ? p * exp(μ * log(c / p)) : μ * c + (1 - μ) * p
    end
end
# _sk_turn!: D ← X e^{i c δθ}
@kernel function _skturn_k!(D, @Const(X), @Const(Q), c::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds begin
        q = Q[i,j,k]
        D[i,j,k] = q == 0.0 ? X[i,j,k] : X[i,j,k] * cis(c * q)
    end
end
@kernel function _combine_k!(C, @Const(A), μ::Float64, ν::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds C[i,j,k] = μ * C[i,j,k] + ν * A[i,j,k]
end

_rotfrac3!(Lx, Ly, Lz, g::GState, c, gs) = (gr = gs.st.grid;
    _launch(gs, _rotfrac3_k!, _lnd(gr), Lx, Ly, Lz, g.Vx, g.Vy, g.Vz, g.Ux, g.Uy, g.Uz, c, _lb(gr)...))

# ---------------------------------------------------------------------------
#  the rotating frame (bands.jl, rotframe without noise: no low-pass, the frame term linear)
# ---------------------------------------------------------------------------
const _ROT_MIN2 = TDGL._ROT_MIN2
const _ROT_MAXDT = TDGL._ROT_MAXDT
@kernel function _frate_k!(Ω, C, @Const(Y), h::Float64, wmax::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds begin
        y = Y[i,j,k]; a2 = abs2(y)
        w = a2 > _ROT_MIN2 ? clamp(imag(conj(y) * (C[i,j,k] - y)) / (a2 * h), -wmax, wmax) : 0.0
        Ω[i,j,k] = w
        C[i,j,k] -= im * (w * h) * y
    end
end
@kernel function _fx_k!(Ω, x1::Int, x2::Int, y1::Int, z1::Int)
    a, b = @index(Global, NTuple); j = y1 + a; k = z1 + b
    @inbounds begin; Ω[x1,j,k] = Ω[x1+1,j,k]; Ω[x2+1,j,k] = Ω[x2,j,k]; end
end
@kernel function _fy_k!(Ω, x1::Int, y1::Int, y2::Int, z1::Int)
    a, b = @index(Global, NTuple); i = x1 - 1 + a; k = z1 + b
    @inbounds begin; Ω[i,y1,k] = Ω[i,y1+1,k]; Ω[i,y2+1,k] = Ω[i,y2,k]; end
end
@kernel function _fz_k!(Ω, x1::Int, y1::Int, z1::Int, z2::Int)
    a, b = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b
    @inbounds begin; Ω[i,j,z1] = Ω[i,j,z1+1]; Ω[i,j,z2+1] = Ω[i,j,z2]; end
end
@kernel function _frate_only_k!(Ω, @Const(C), @Const(Y), h::Float64, wmax::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds begin
        y = Y[i,j,k]; a2 = abs2(y)
        Ω[i,j,k] = a2 > _ROT_MIN2 ? clamp(imag(conj(y) * (C[i,j,k] - y)) / (a2 * h), -wmax, wmax) : 0.0
    end
end
@kernel function _fsmooth_k!(Ω, @Const(T), x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds begin
        nb = (T[i+1,j,k] + T[i-1,j,k]) + (T[i,j+1,k] + T[i,j-1,k]) + (T[i,j,k+1] + T[i,j,k-1])
        Ω[i,j,k] = 0.5 * T[i,j,k] + nb / 12
    end
end
@kernel function _fturnback_k!(C, @Const(Ω), h::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds C[i,j,k] *= cis(-(Ω[i,j,k] * h))
end
_ffaces!(Ω, gs) = ((; x1, x2, y1, y2, z1, z2) = gs.st.grid;
    _launch(gs, _fx_k!, (y2 - y1, z2 - z1), Ω, x1, x2, y1, z1);
    _launch(gs, _fy_k!, (x2 - x1 + 2, z2 - z1), Ω, x1, y1, y2, z1);
    _launch(gs, _fz_k!, (x2 - x1 + 2, y2 - y1 + 2), Ω, x1, y1, z1, z2))
# _frame_rate_smooth! (with noise): the rate low-passed by 2 sweeps, the frame term as a rotation
function _frame_rate_smooth!(Ω, C, Y, h::Float64, dt::Float64, gs, passes::Int)
    (; x1, x2, y1, y2, z1, z2) = gs.st.grid
    nd = (x2 - x1, y2 - y1, z2 - z1)
    _launch(gs, _frate_only_k!, nd, Ω, C, Y, h, _ROT_MAXDT / dt, x1, y1, z1)
    _ffaces!(Ω, gs)
    for _ in 1:passes
        _hsync(gs); copyto!(gs.Ωs, Ω)
        _launch(gs, _fsmooth_k!, nd, Ω, gs.Ωs, x1, y1, z1)
        _ffaces!(Ω, gs)
    end
    _launch(gs, _fturnback_k!, nd, C, Ω, h, x1, y1, z1)
    Ω
end
function _frame_rate!(Ω, C, Y, h::Float64, dt::Float64, gs)
    gs.st.noise === nothing || return _frame_rate_smooth!(Ω, C, Y, h, dt, gs, 2)
    (; x1, x2, y1, y2, z1, z2) = gs.st.grid
    _launch(gs, _frate_k!, (x2 - x1, y2 - y1, z2 - z1), Ω, C, Y, h, _ROT_MAXDT / dt, x1, y1, z1)
    _launch(gs, _fx_k!, (y2 - y1, z2 - z1), Ω, x1, x2, y1, z1)
    _launch(gs, _fy_k!, (x2 - x1 + 2, z2 - z1), Ω, x1, y1, y2, z1)
    _launch(gs, _fz_k!, (x2 - x1 + 2, y2 - y1 + 2), Ω, x1, y1, z1, z2)
    Ω
end
@kernel function _fterm_k!(C, @Const(S), @Const(Ω), h::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds C[i,j,k] -= im * (Ω[i,j,k] * h) * S[i,j,k]
end
_frame_term!(C, S, Ω, h, gs) = (gr = gs.st.grid; nd = (gr.x2 - gr.x1, gr.y2 - gr.y1, gr.z2 - gr.z1);
    gs.st.noise === nothing ? _launch(gs, _fterm_k!, nd, C, S, Ω, h, gr.x1, gr.y1, gr.z1) :
                              _launch(gs, _fturnback_k!, nd, C, Ω, h, gr.x1, gr.y1, gr.z1); C)    # with noise: a rotation
@kernel function _fturn_k!(F, @Const(Ω), dt::Float64, x1::Int, y1::Int, z1::Int)
    a, b, e = @index(Global, NTuple); i = x1 + a; j = y1 + b; k = z1 + e
    @inbounds F[i,j,k] *= cis(Ω[i,j,k] * dt)
end
_frame_turn!(F, Ω, dt, gs) = (gr = gs.st.grid;
    _launch(gs, _fturn_k!, (gr.x2 - gr.x1, gr.y2 - gr.y1, gr.z2 - gr.z1), F, Ω, dt, gr.x1, gr.y1, gr.z1); F)

@inline _cis(a::Float64) = abs(a) <= 0.05 ? (a2 = a*a; complex(1 - a2/2, a*(1 - a2/6))) : cis(a)
# _frame_links!: L ← L e^{i(ω̂ᵢ₊₁ − ω̂ᵢ)τ}, over the range of the link with the offset d of its far node
@kernel function _flinks_k!(L, @Const(Ω), τ::Float64, o::NTuple{3,Int}, d::NTuple{3,Int})
    a, b, e = @index(Global, NTuple); i = o[1] + a; j = o[2] + b; k = o[3] + e
    @inbounds L[i,j,k] *= _cis((Ω[i+d[1],j+d[2],k+d[3]] - Ω[i,j,k]) * τ)
end
_frame_links!(Lx, Ly, Lz, Ω, τ, gs) = (gr = gs.st.grid; _launch(gs, _flinks3_k!, _lnd(gr), Lx, Ly, Lz, Ω, τ, _lb(gr)...))
@kernel function _lrate_k!(R, @Const(V), @Const(U), @Const(Ω), dt::Float64, o::NTuple{3,Int}, d::NTuple{3,Int})
    a, b, e = @index(Global, NTuple); i = o[1] + a; j = o[2] + b; k = o[3] + e
    @inbounds begin
        q = V[i,j,k] * conj(U[i,j,k]); sf = -imag(q)
        t = abs(sf) <= 0.05 ? -(sf + sf*sf*sf/6) : angle(q)
        R[i,j,k] = t + (Ω[i+d[1],j+d[2],k+d[3]] - Ω[i,j,k]) * dt
    end
end
@kernel function _rotlinks_k!(L, @Const(V), @Const(R), c::Float64, o::NTuple{3,Int})
    a, b, e = @index(Global, NTuple); i = o[1] + a; j = o[2] + b; k = o[3] + e
    @inbounds L[i,j,k] = V[i,j,k] * _cis(R[i,j,k] * c)
end

# ---- the link passes of the three axes in one launch: over x1:x2+1, y1:y2+1, z1:z2+1, the link along x where i ≤ x2,
#      along y where j ≤ y2, along z where k ≤ z2 (the ranges _rx, _ry, _rz); each entry the expression of its pass
@inline function _rotfrac1(V, U, c, i, j, k)
    @inbounds begin
        v = V[i,j,k]; q = v * conj(U[i,j,k]); sf = -imag(q)
        if abs(sf) <= 0.05
            t = (sf + sf*sf*sf/6) * c; t2 = t*t
            v * complex(1 - t2/2, -t*(1 - t2/6))
        else
            v * cis(angle(q) * c)
        end
    end
end
@kernel function _rotfrac3_k!(Lx, Ly, Lz, @Const(Vx), @Const(Vy), @Const(Vz), @Const(Ux), @Const(Uy), @Const(Uz), c::Float64,
                              x1::Int, x2::Int, y1::Int, y2::Int, z1::Int, z2::Int)
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        i <= x2 && (Lx[i,j,k] = _rotfrac1(Vx, Ux, c, i, j, k))
        j <= y2 && (Ly[i,j,k] = _rotfrac1(Vy, Uy, c, i, j, k))
        k <= z2 && (Lz[i,j,k] = _rotfrac1(Vz, Uz, c, i, j, k))
    end
end
@inline function _rot1(V, U, n, i, j, k)
    @inbounds begin
        v = V[i,j,k]; q = v * conj(U[i,j,k]); sf = -imag(q)
        if abs(sf) <= 0.05
            t = (sf + sf*sf*sf/6) / n; t2 = t*t
            (v, complex(1 - t2/2, -t*(1 - t2/6)))
        else
            (v, cis(angle(q) / n))
        end
    end
end
@kernel function _rot3_k!(Lx, Ly, Lz, Rx, Ry, Rz, @Const(Vx), @Const(Vy), @Const(Vz), @Const(Ux), @Const(Uy), @Const(Uz), n::Int,
                          x1::Int, x2::Int, y1::Int, y2::Int, z1::Int, z2::Int)
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        i <= x2 && ((Lx[i,j,k], Rx[i,j,k]) = _rot1(Vx, Ux, n, i, j, k))
        j <= y2 && ((Ly[i,j,k], Ry[i,j,k]) = _rot1(Vy, Uy, n, i, j, k))
        k <= z2 && ((Lz[i,j,k], Rz[i,j,k]) = _rot1(Vz, Uz, n, i, j, k))
    end
end
@kernel function _mulby3_k!(Lx, Ly, Lz, @Const(Rx), @Const(Ry), @Const(Rz), x1::Int, x2::Int, y1::Int, y2::Int, z1::Int, z2::Int)
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        i <= x2 && (Lx[i,j,k] *= Rx[i,j,k])
        j <= y2 && (Ly[i,j,k] *= Ry[i,j,k])
        k <= z2 && (Lz[i,j,k] *= Rz[i,j,k])
    end
end
@kernel function _flinks3_k!(Lx, Ly, Lz, @Const(Ω), τ::Float64, x1::Int, x2::Int, y1::Int, y2::Int, z1::Int, z2::Int)
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        i <= x2 && (Lx[i,j,k] *= _cis((Ω[i+1,j,k] - Ω[i,j,k]) * τ))
        j <= y2 && (Ly[i,j,k] *= _cis((Ω[i,j+1,k] - Ω[i,j,k]) * τ))
        k <= z2 && (Lz[i,j,k] *= _cis((Ω[i,j,k+1] - Ω[i,j,k]) * τ))
    end
end
@inline function _lrate1(V, U, Ω, dt, i, j, k, i2, j2, k2)
    @inbounds begin
        q = V[i,j,k] * conj(U[i,j,k]); sf = -imag(q)
        t = abs(sf) <= 0.05 ? -(sf + sf*sf*sf/6) : angle(q)
        t + (Ω[i2,j2,k2] - Ω[i,j,k]) * dt
    end
end
@kernel function _lrate3_k!(Rx, Ry, Rz, @Const(Vx), @Const(Vy), @Const(Vz), @Const(Ux), @Const(Uy), @Const(Uz), @Const(Ω),
                            dt::Float64, x1::Int, x2::Int, y1::Int, y2::Int, z1::Int, z2::Int)
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        i <= x2 && (Rx[i,j,k] = _lrate1(Vx, Ux, Ω, dt, i, j, k, i+1, j, k))
        j <= y2 && (Ry[i,j,k] = _lrate1(Vy, Uy, Ω, dt, i, j, k, i, j+1, k))
        k <= z2 && (Rz[i,j,k] = _lrate1(Vz, Uz, Ω, dt, i, j, k, i, j, k+1))
    end
end
@kernel function _rotlinks3_k!(Lx, Ly, Lz, @Const(Vx), @Const(Vy), @Const(Vz), @Const(Rx), @Const(Ry), @Const(Rz), c::Float64,
                               x1::Int, x2::Int, y1::Int, y2::Int, z1::Int, z2::Int)
    a, b, e = @index(Global, NTuple); i = x1 - 1 + a; j = y1 - 1 + b; k = z1 - 1 + e
    @inbounds begin
        i <= x2 && (Lx[i,j,k] = Vx[i,j,k] * _cis(Rx[i,j,k] * c))
        j <= y2 && (Ly[i,j,k] = Vy[i,j,k] * _cis(Ry[i,j,k] * c))
        k <= z2 && (Lz[i,j,k] = Vz[i,j,k] * _cis(Rz[i,j,k] * c))
    end
end
_lnd(gr) = (gr.x2 - gr.x1 + 2, gr.y2 - gr.y1 + 2, gr.z2 - gr.z1 + 2)
_lb(gr) = (gr.x1, gr.x2, gr.y1, gr.y2, gr.z1, gr.z2)

# _band_rkl!: RKL1 or damped RKC1 stages (without the rotating frame, SK-ROCK or the log-polar combination)
function _band_rkl!(g::GState{N}, gs::GPUStepper{N}, µ::Int, ns::Int) where N
    st = gs.st; gr = st.grid
    cf = st.bcs[µ]
    kc = st.rkc === nothing ? nothing : st.rkc[µ]
    if st.extrap
        Lx, Ly, Lz = gs.L[1], gs.L[2], gs.L[3]
        copyto!(Lx, g.Vx); copyto!(Ly, g.Vy); copyto!(Lz, g.Vz)
    else
        Lx, Ly, Lz = g.Vx, g.Vy, g.Vz
    end
    buf = (gs.W, gs.W2, g.F[µ])
    q = kc === nothing ? Float64(ns * (ns + 1) ÷ 2) : kc.q
    dr = TDGL._dragof(st, µ, q)
    rot = st.rotframe; Ω = gs.Ω; h = st.dt / q
    lg = TDGL._logcomb(st, µ); skq = TDGL._skrock(st, µ)
    nd = (gr.x2 - gr.x1, gr.y2 - gr.y1, gr.z2 - gr.z1)
    A = g.G[µ]; B = g.G[µ]; C = buf[1]
    for j in 1:ns
        C = buf[mod1(j, 3)]
        if j == 1 && skq                     # SK-ROCK for the phase (noise.jl): K₁ = E(Ψⁿ e^{iν₁δθ}) e^{i(κ₁−ν₁)δθ}
            Sp = gs.W2; Qθ = gs.nz.dθ[µ]; ν1, κ1 = kc.sk
            _launch(gs, _skturn_k!, nd, Sp, B, Qθ, ν1, gr.x1, gr.y1, gr.z1); _ghosts!(Sp, Lx, Ly, Lz, gs)
            _band_pass!(C, Sp, g, µ, Lx, Ly, Lz, cf, gs, dr)
            rot && _frame_rate!(Ω, C, Sp, h, st.dt, gs)
            _launch(gs, _skturn_k!, nd, C, C, Qθ, κ1 - ν1, gr.x1, gr.y1, gr.z1)
        else
            _band_pass!(C, B, g, µ, Lx, Ly, Lz, cf, gs, dr)
            rot && (j == 1 ? _frame_rate!(Ω, C, B, h, st.dt, gs) : _frame_term!(C, B, Ω, h, gs))
        end
        if rot && j == 1 && st.extrap && ns > 1
            Rx, Ry, Rz = gs.Rl
            _launch(gs, _lrate3_k!, _lnd(gr), Rx, Ry, Rz, g.Vx, g.Vy, g.Vz, g.Ux, g.Uy, g.Uz, Ω, st.dt, _lb(gr)...)
        end
        if j >= 2
            μj, νj = kc === nothing ? ((2j - 1) / j, (1 - j) / j) : (kc.μ[j], kc.ν[j])
            lg ? _launch(gs, _combine_log_k!, nd, C, A, μj, gr.x1, gr.y1, gr.z1) :
                 _launch(gs, _combine_k!, nd, C, A, μj, νj, gr.x1, gr.y1, gr.z1)
        end
        if j < ns
            if st.extrap
                c = kc === nothing ? j * (j + 1) / (ns * (ns + 1)) : kc.c[j]
                if rot
                    Rx, Ry, Rz = gs.Rl
                    _launch(gs, _rotlinks3_k!, _lnd(gr), Lx, Ly, Lz, g.Vx, g.Vy, g.Vz, Rx, Ry, Rz, c, _lb(gr)...)
                else
                    _rotfrac3!(Lx, Ly, Lz, g, c, gs)
                end
            end
            _ghosts!(C, Lx, Ly, Lz, gs)
        end
        A = B; B = C
    end
    C === g.F[µ] || copyto!(g.F[µ], C)
    rot && _frame_turn!(g.F[µ], Ω, st.dt, gs)
    g.F[µ]
end

# _band_substeps!: n Euler substeps (without the rotating frame)
function _band_substeps!(g::GState{N}, gs::GPUStepper{N}, µ::Int, n::Int) where N
    st = gs.st; gr = st.grid
    cf = st.bcs[µ]
    F = g.F[µ]; W = gs.W
    if st.extrap
        Lx, Ly, Lz, Rx, Ry, Rz = gs.L
        _launch(gs, _rot3_k!, _lnd(gr), Lx, Ly, Lz, Rx, Ry, Rz, g.Vx, g.Vy, g.Vz, g.Ux, g.Uy, g.Uz, n, _lb(gr)...)
    else
        Lx, Ly, Lz = g.Vx, g.Vy, g.Vz
    end
    src = g.G[µ]
    dr = TDGL._dragof(st, µ, n)
    rot = st.rotframe; Ω = gs.Ω; h = st.dt / n
    for m in 1:n
        dst = iseven(n - m) ? F : W
        _band_pass!(dst, src, g, µ, Lx, Ly, Lz, cf, gs, dr)
        rot && (m == 1 ? _frame_rate!(Ω, dst, src, h, st.dt, gs) : _frame_term!(dst, src, Ω, h, gs))
        if m < n
            if st.extrap
                _launch(gs, _mulby3_k!, _lnd(gr), Lx, Ly, Lz, Rx, Ry, Rz, _lb(gr)...)
            end
            rot && _frame_links!(Lx, Ly, Lz, Ω, h, gs)
            _ghosts!(dst, Lx, Ly, Lz, gs)
        end
        src = dst
    end
    rot && _frame_turn!(F, Ω, st.dt, gs)
    F
end

function _update_bands!(g::GState{N}, gs::GPUStepper{N}) where N
    st = gs.st
    for µ in 1:N
        n = st.nsub[µ]
        if st.rkl[µ] > 0
            _band_rkl!(g, gs, µ, st.rkl[µ])
        elseif n == 1
            _band_pass!(g.F[µ], g.G[µ], g, µ, g.Vx, g.Vy, g.Vz, st.bc[µ], gs, TDGL._dragof(st, µ, 1))
            if st.rotframe
                _frame_rate!(gs.Ω, g.F[µ], g.G[µ], st.dt, st.dt, gs)
                _frame_turn!(g.F[µ], gs.Ω, st.dt, gs)
            end
        else
            _band_substeps!(g, gs, µ, n)
        end
    end
    g
end

# ---------------------------------------------------------------------------
#  the field (field.jl)
# ---------------------------------------------------------------------------
@inline _jdragk(::Nothing, G, v, i, j, k, i2, j2, k2, acc) = acc
@inline function _jdragk(ν::NTuple{N,NTuple{N,Float64}}, G, v, i, j, k, i2, j2, k2, acc) where N
    @inbounds for µ in 1:N, η in µ+1:N
        c = ν[µ][η]
        c == 0.0 && continue
        acc += c*(imag((conj(G[µ][i,j,k])*v)*G[η][i2,j2,k2]) + imag((conj(G[η][i,j,k])*v)*G[µ][i2,j2,k2]))
    end
    acc
end
@inline function _jlink(G::NTuple{N}, w, v, i, j, k, i2, j2, k2, ν=nothing) where N
    @inbounds begin
        acc = w[1]*imag((conj(G[1][i,j,k])*v)*G[1][i2,j2,k2])
        for µ in 2:N
            acc += w[µ]*imag((conj(G[µ][i,j,k])*v)*G[µ][i2,j2,k2])
        end
        _jdragk(ν, G, v, i, j, k, i2, j2, k2, acc)
    end
end
@kernel function _jx_k!(Jx, G, @Const(Vx), w)
    i, a, b = @index(Global, NTuple); j = a + 1; k = b + 1
    @inbounds Jx[i,j,k] = _jlink(G, w, Vx[i,j,k], i, j, k, i+1, j, k)
end
@kernel function _jy_k!(Jy, G, @Const(Vy), w)
    a, j, b = @index(Global, NTuple); i = a + 1; k = b + 1
    @inbounds Jy[i,j,k] = _jlink(G, w, Vy[i,j,k], i, j, k, i, j+1, k)
end
@kernel function _jz_k!(Jz, G, @Const(Vz), w)
    a, b, k = @index(Global, NTuple); i = a + 1; j = b + 1
    @inbounds Jz[i,j,k] = _jlink(G, w, Vz[i,j,k], i, j, k, i, j, k+1)
end
# the three components in one launch over the union of their ranges (each entry the same expression)
@kernel function _j3_k!(Jx, Jy, Jz, G, @Const(Vx), @Const(Vy), @Const(Vz), wx, wy, wz, ν)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        j >= 2 && k >= 2 && (Jx[i,j,k] = _jlink(G, wx, Vx[i,j,k], i, j, k, i+1, j, k, ν))
        i >= 2 && k >= 2 && (Jy[i,j,k] = _jlink(G, wy, Vy[i,j,k], i, j, k, i, j+1, k, ν))
        i >= 2 && j >= 2 && (Jz[i,j,k] = _jlink(G, wz, Vz[i,j,k], i, j, k, i, j, k+1, ν))
    end
end
function _supercurrent!(g::GState, gs)
    (; Nx, Ny, Nz) = gs.st.grid; wx, wy, wz = gs.st.wa
    if gs.mat === nothing
        _launch(gs, _j3_k!, (Nx, Ny, Nz), g.Jx, g.Jy, g.Jz, g.G, g.Vx, g.Vy, g.Vz, wx, wy, wz, gs.st.dragν)
    else
        _launch(gs, _j3mat_k!, (Nx, Ny, Nz), g.Jx, g.Jy, g.Jz, g.G, g.Vx, g.Vy, g.Vz, wx, wy, wz, gs.mat.ρ)
    end
    g
end

@kernel function _phx_k!(φx, @Const(θx), @Const(Jx), @Const(τx), @Const(hy), @Const(hz), d1::Float64, sxy::Float64, sxz::Float64)
    i, a, b = @index(Global, NTuple); j = a + 1; k = b + 1
    @inbounds φx[i,j,k] = ((θx[i,j,k] + (d1*τx[i,j,k])*Jx[i,j,k]) - sxy*(hz[i,j,k] - hz[i,j-1,k])) + sxz*(hy[i,j,k] - hy[i,j,k-1])
end
@kernel function _phy_k!(φy, @Const(θy), @Const(Jy), @Const(τy), @Const(hx), @Const(hz), d1::Float64, syz::Float64, syx::Float64)
    a, j, b = @index(Global, NTuple); i = a + 1; k = b + 1
    @inbounds φy[i,j,k] = ((θy[i,j,k] + (d1*τy[i,j,k])*Jy[i,j,k]) - syz*(hx[i,j,k] - hx[i,j,k-1])) + syx*(hz[i,j,k] - hz[i-1,j,k])
end
@kernel function _phz_k!(φz, @Const(θz), @Const(Jz), @Const(τz), @Const(hx), @Const(hy), d1::Float64, szx::Float64, szy::Float64)
    a, b, k = @index(Global, NTuple); i = a + 1; j = b + 1
    @inbounds φz[i,j,k] = ((θz[i,j,k] + (d1*τz[i,j,k])*Jz[i,j,k]) - szx*(hy[i,j,k] - hy[i-1,j,k])) + szy*(hx[i,j,k] - hx[i,j-1,k])
end
@kernel function _ph3_k!(φx, φy, φz, @Const(θx), @Const(θy), @Const(θz), @Const(Jx), @Const(Jy), @Const(Jz),
                         @Const(τx), @Const(τy), @Const(τz), @Const(hx), @Const(hy), @Const(hz), fc)
    i, j, k = @index(Global, NTuple)
    d1, sxy, sxz, syx, syz, szx, szy = fc.d1, fc.sxy, fc.sxz, fc.syx, fc.syz, fc.szx, fc.szy
    @inbounds begin
        j >= 2 && k >= 2 &&
            (φx[i,j,k] = ((θx[i,j,k] + (d1*τx[i,j,k])*Jx[i,j,k]) - sxy*(hz[i,j,k] - hz[i,j-1,k])) + sxz*(hy[i,j,k] - hy[i,j,k-1]))
        i >= 2 && k >= 2 &&
            (φy[i,j,k] = ((θy[i,j,k] + (d1*τy[i,j,k])*Jy[i,j,k]) - syz*(hx[i,j,k] - hx[i,j,k-1])) + syx*(hz[i,j,k] - hz[i-1,j,k]))
        i >= 2 && j >= 2 &&
            (φz[i,j,k] = ((θz[i,j,k] + (d1*τz[i,j,k])*Jz[i,j,k]) - szx*(hy[i,j,k] - hy[i-1,j,k])) + szy*(hx[i,j,k] - hx[i,j-1,k]))
    end
end
function _field_increment!(g::GState, gs)
    (; Nx, Ny, Nz) = gs.st.grid
    _launch(gs, _ph3_k!, (Nx, Ny, Nz), g.φx, g.φy, g.φz, g.θx, g.θy, g.θz, g.Jx, g.Jy, g.Jz, gs.τx, gs.τy, gs.τz,
            g.hx, g.hy, g.hz, gs.st.fc)
    g
end

@kernel function _ux_k!(Ux, @Const(φx))
    i, a, b = @index(Global, NTuple); j = a + 1; k = b + 1
    @inbounds Ux[i,j,k] = cis(-φx[i,j,k])
end
@kernel function _uy_k!(Uy, @Const(φy))
    a, j, b = @index(Global, NTuple); i = a + 1; k = b + 1
    @inbounds Uy[i,j,k] = cis(-φy[i,j,k])
end
@kernel function _uz_k!(Uz, @Const(φz))
    a, b, k = @index(Global, NTuple); i = a + 1; j = b + 1
    @inbounds Uz[i,j,k] = cis(-φz[i,j,k])
end
@kernel function _u3_k!(Ux, Uy, Uz, @Const(φx), @Const(φy), @Const(φz))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        j >= 2 && k >= 2 && (Ux[i,j,k] = cis(-φx[i,j,k]))
        i >= 2 && k >= 2 && (Uy[i,j,k] = cis(-φy[i,j,k]))
        i >= 2 && j >= 2 && (Uz[i,j,k] = cis(-φz[i,j,k]))
    end
end
function _links!(g::GState, gs)
    (; Nx, Ny, Nz) = gs.st.grid
    _launch(gs, _u3_k!, (Nx, Ny, Nz), g.Ux, g.Uy, g.Uz, g.φx, g.φy, g.φz)
    g
end

@kernel function _hx_k!(hx, @Const(φy), @Const(φz), dydz::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds hx[i,j,k] = ((φz[i,j+1,k] - φz[i,j,k]) - φy[i,j,k+1] + φy[i,j,k]) / dydz
end
@kernel function _hy_k!(hy, @Const(φx), @Const(φz), dzdx::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds hy[i,j,k] = ((φx[i,j,k+1] - φx[i,j,k]) - φz[i+1,j,k] + φz[i,j,k]) / dzdx
end
@kernel function _hz_k!(hz, @Const(φx), @Const(φy), dxdy::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds hz[i,j,k] = ((φy[i+1,j,k] - φy[i,j,k]) - φx[i,j+1,k] + φx[i,j,k]) / dxdy
end
# hx on i 2:Nx, j 2:Ny-1, k 2:Nz-1; hy on i 2:Nx-1, j 2:Ny, k 2:Nz-1; hz on i 2:Nx-1, j 2:Ny-1, k 2:Nz: one launch over
# i, j, k in 2:Nx, 2:Ny, 2:Nz with the guards (the same as the CPU's three loops; also _apply_A! of the PCG)
@kernel function _h3_k!(hx, hy, hz, @Const(φx), @Const(φy), @Const(φz), Nx::Int, Ny::Int, Nz::Int,
                        dydz::Float64, dzdx::Float64, dxdy::Float64)
    a, b, e = @index(Global, NTuple); i = a + 1; j = b + 1; k = e + 1
    @inbounds begin
        j <= Ny - 1 && k <= Nz - 1 && (hx[i,j,k] = ((φz[i,j+1,k] - φz[i,j,k]) - φy[i,j,k+1] + φy[i,j,k]) / dydz)
        i <= Nx - 1 && k <= Nz - 1 && (hy[i,j,k] = ((φx[i,j,k+1] - φx[i,j,k]) - φz[i+1,j,k] + φz[i,j,k]) / dzdx)
        i <= Nx - 1 && j <= Ny - 1 && (hz[i,j,k] = ((φy[i+1,j,k] - φy[i,j,k]) - φx[i,j+1,k] + φx[i,j,k]) / dxdy)
    end
end
function _curl_h!(g::GState, gs)
    (; Nx, Ny, Nz, dx, dy, dz) = gs.st.grid
    _launch(gs, _h3_k!, (Nx - 1, Ny - 1, Nz - 1), g.hx, g.hy, g.hz, g.φx, g.φy, g.φz, Nx, Ny, Nz, dy*dz, dz*dx, dx*dy)
    g
end

# field_bc! as the broadcasts of field.jl, in its order; the loops over the bar's profile as one broadcast each (the same
# set of entries, the same arithmetic)
# The faces it writes are never written by the rest of the step (curl_h! writes the interior of h only), and what it writes
# depends on H and HJ only: when they are those of the last step the faces already hold it, and it is skipped (exact).
function _field_bc!(g::GState, gs, H::Float64, HJ::Float64)
    gs.open === nothing && g.bc[1] == H && g.bc[2] == HJ && return g      # (the open boundary ramps its faces every step)
    g.bc[1] = H; g.bc[2] = HJ
    (; Nx, Ny, Nz) = gs.st.grid
    hx, hy, hz = g.hx, g.hy, g.hz
    _hsync(gs)
    @views begin
        hz[:, :, 1] .= H; hz[:, :, Nz+1] .= H
        hz[1, :, :] .= H; hz[Nx, :, :] .= H
        hz[:, 1, :] .= H; hz[:, Ny, :] .= H
        hx[1, :, :] .= 0.0; hx[Nx+1, :, :] .= 0.0
        hx[:, 1, :] .= 0.0; hx[:, Ny, :] .= 0.0
        hy[:, 1, :] .= 0.0; hy[:, Ny+1, :] .= 0.0
        b = gs.bar
        if b === nothing
            hy[:, :, 1] .= -HJ; hy[:, :, Nz] .= +HJ
        else
            for (c, k) in enumerate((1, Nz))
                hy[:, :, k] .= HJ .* reshape(b.hy_z[:, c], 1, Ny + 1)
            end
            for (c, j) in enumerate((1, Ny+1))
                hy[:, j, 1:Nz] .= HJ .* reshape(b.hy_y[c, :], 1, Nz)
            end
            for (c, k) in enumerate((1, Nz+1))
                hz[:, 1:Ny, k] .= H .+ HJ .* reshape(b.hz_z[:, c], 1, Ny)
            end
            for (c, j) in enumerate((1, Ny))
                hz[:, j, :] .= H .+ HJ .* reshape(b.hz_y[c, :], 1, Nz + 1)
            end
            for i in (1, Nx)
                hz[i, 1:Ny, :] .= H .+ HJ .* b.hz_x
            end
            for i in (1, Nx)
                hy[i, 1:Ny+1, 1:Nz] .= HJ .* b.hy_x
            end
            gs.open === nothing || _open_add!(g, gs)
        end
    end
    g
end

@kernel function _zero_vac_k!(F, @Const(vac))
    i = @index(Global, Linear)
    @inbounds vac[i] && (F[i] = 0.0 + 0.0im)
end
function _neumann!(g::GState{N}, gs) where N
    for µ in 1:N
        _ghosts!(g.F[µ], g.Ux, g.Uy, g.Uz, gs)
    end
    if gs.vac !== nothing          # a sculpted sample: the ghosts of its vacuum on the faces are vacuum too
        for µ in 1:N
            _launch(gs, _zero_vac_k!, (length(g.F[µ]),), g.F[µ], gs.vac)
        end
    end
    g
end

function _advance!(g::GState{N}, gs) where N
    (; x1, x2, y1, y2, z1, z2) = gs.st.grid
    _hsync(gs)
    for µ in 1:N
        @views g.G[µ][x1:x2+1, y1:y2+1, z1:z2+1] .= g.F[µ][x1:x2+1, y1:y2+1, z1:z2+1]
    end
    g.θx, g.φx = g.φx, g.θx
    g.θy, g.φy = g.φy, g.θy
    g.θz, g.φz = g.φz, g.θz
    g.Vx, g.Ux = g.Ux, g.Vx
    g.Vy, g.Uy = g.Uy, g.Vy
    g.Vz, g.Uz = g.Uz, g.Vz
    g
end

# ---------------------------------------------------------------------------
#  the step (step.jl)
# ---------------------------------------------------------------------------
"""
    gpu_step!(g, gs; H, HJ) -> g

One step of `TDGL.step!` on the device: bands, supercurrent, field increment, implicit solve, links, curl, field
boundary, Neumann ghosts, advance.
"""
function gpu_step!(g::GState{N}, gs::GPUStepper{N}; H::Float64, HJ::Float64) where N
    gs.open === nothing || _open_update!(g, gs, HJ)
    gs.qp === nothing || (gs.qp isa GQP ? _qp_potential!(g, gs) : _qpN_potential!(g, gs))
    sk = gs.nz !== nothing && !isempty(gs.st.ws.Qξ)        # noise on a KWT band with RKC stages: the draws before the bands
    sk && _noise_draw!(g, gs)
    _update_bands!(g, gs)
    gs.qp === nothing || (gs.qp isa GQP ? _qp_turn!(g, gs) : _qpN_turn!(g, gs))
    gs.nz === nothing || (sk ? _noise_apply!(g, gs) : _noise_kick!(g, gs))
    _supercurrent!(g, gs)
    gs.θ === nothing || _heat_step!(g, gs)
    _field_increment!(g, gs)
    gs.nz !== nothing && gs.st.noise.field && _field_kick!(g, gs)
    if gs.sep !== nothing
        _implicit_curl!(g, gs.sep, gs)
    elseif gs.D !== nothing
        _scale_increment!(g, gs.D, gs)
    end
    _links!(g, gs)
    _curl_h!(g, gs)
    _field_bc!(g, gs, H, HJ)
    _neumann!(g, gs)
    _advance!(g, gs)
    g
end

"""`n` steps on the device, then a synchronisation."""
function gpu_steps!(g::GState, gs::GPUStepper, n::Integer; H::Float64, HJ::Float64)
    for _ in 1:n
        gpu_step!(g, gs; H=H, HJ=HJ)
    end
    KA.synchronize(gs.backend)
    g
end

# ---------------------------------------------------------------------------
#  a whole run on the device: the cycles of TDGL's ladders (TDGL.STEPS)
# ---------------------------------------------------------------------------
const _LAST = Ref{Any}(nothing)      # (stepper, its GPUStepper or nothing) of the last cycle
const _DEV = Ref{Any}(nothing)       # (host state, its device state)
const _FALLBACK = Ref(:cpu)
const _BACKEND_NAME = Ref("")
const _BACKEND_CYCLES = Ref(0)
const _BACKEND_STEPS = Ref(0)
const _CPU_CYCLES = Ref(0)
const _CPU_STEPS = Ref(0)

"""Counts for the last enabled backend; KA-CPU is named explicitly. Kept after disable!."""
execution_stats() = (backend=_BACKEND_NAME[], backend_cycles=_BACKEND_CYCLES[], backend_steps=_BACKEND_STEPS[],
                     cpu_fallback_cycles=_CPU_CYCLES[], cpu_fallback_steps=_CPU_STEPS[])

function _unsupported(why)
    _FALLBACK[] === :error && throw(ArgumentError("TDGLGPU: $why is unsupported; CPU fallback is disabled"))
    @warn "TDGLGPU: $why is unsupported; this stepper runs on the CPU"
end

"""
    enable!(backend; fallback=:cpu)

Every ladder of TDGL (`ladder`, `run_config`, `scripts/run.jl`, the refinements) steps its cycles on `backend`: the state is
copied to the device, stepped `lMax` times there, and copied back for the measurements, the verdicts and the checkpoints,
which stay those of the CPU code. An unsupported Stepper uses CPU with a warning, or throws with `fallback=:error`.
Enabling a backend starts a fresh device cache and execution counter. The hook is process-global: use separate processes
for concurrent ladders. `disable!()` restores the CPU loop; `execution_stats()` retains the counters.
"""
function enable!(backend; fallback::Symbol=:cpu)
    fallback in (:cpu, :error) || throw(ArgumentError("fallback must be :cpu or :error"))
    _LAST[] = nothing; _DEV[] = nothing
    _FALLBACK[] = fallback
    _BACKEND_NAME[] = backend isa KA.CPU ? "KA-CPU" : string(typeof(backend))
    _BACKEND_CYCLES[] = 0; _BACKEND_STEPS[] = 0; _CPU_CYCLES[] = 0; _CPU_STEPS[] = 0
    TDGL.STEPS[] = (s, st, n; H, HJ) -> _gpu_cycle!(backend, s, st, n; H=H, HJ=HJ)
    nothing
end
disable!() = (TDGL.STEPS[] = nothing; _LAST[] = nothing; _DEV[] = nothing; nothing)

function _gpu_cycle!(backend, s::State, st::Stepper, n::Int; H::Float64, HJ::Float64)
    if _LAST[] === nothing || _LAST[][1] !== st
        ok, why = supported(st)
        ok || _unsupported(why)
        _LAST[] = (st, ok ? GPUStepper(st, backend) : nothing)
        _DEV[] = nothing
    end
    gs = _LAST[][2]
    if gs === nothing
        for _ in 1:n; step!(s, st; H=H, HJ=HJ); end
        _CPU_CYCLES[] += 1; _CPU_STEPS[] += n
        return s
    end
    _DEV[] === nothing || _DEV[][1] !== s || size(_DEV[][2].G[1]) != size(s.G[1]) ? (_DEV[] = (s, device_state(backend, s))) :
                                                                                  device_state!(_DEV[][2], s)
    g = _DEV[][2]
    gs.sep isa GPCG && _pcg_up!(gs.sep, st.imp)
    gs.θ === nothing || copyto!(gs.θ, st.heat.θ)          # the protocol reads θ (the cut 'T') and the checkpoints hold it
    gs.open === nothing || _open_up!(gs)                  # the ramp of the open boundary (the checkpoints hold it on the host)
    gpu_steps!(g, gs, n; H=H, HJ=HJ)
    gs.sep isa GPCG && _pcg_down!(st.imp, gs.sep)
    gs.θ === nothing || copyto!(st.heat.θ, gs.θ)
    if gs.qp isa GQP                                      # the potentials of the last step, for what reads them on the host
        copyto!(st.qpx.δ, gs.qp.δ)
    elseif gs.qp !== nothing
        for µ in eachindex(st.qpx.μ); copyto!(st.qpx.μ[µ], gs.qp.μ[µ]); end
    end
    host_state!(s, g)
    _BACKEND_CYCLES[] += 1; _BACKEND_STEPS[] += n
    s
end

end # module
