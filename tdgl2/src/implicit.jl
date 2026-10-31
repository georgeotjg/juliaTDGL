# ============================================================================
#  implicit.jl — the curl-curl term of the field equation at the new time level
#
#      (I + (dt/sigma_n) kappa^2 K) A^{n+1} = A^n + (dt/sigma_n) J_s^n - (dt/sigma_n) kappa^2 c
#
#  K the Yee curl-curl over the links field_increment! updates, c the boundary fields.  On
#  tensor-product link ranges with dx = dy = dz, K is diagonalised by per-axis bases (the right
#  singular vectors V of the 1D forward difference on nodes, cosine-like, and U = D V / s on
#  edges, sine-like); per wavevector the block is |k|^2 I - k k^T and
#  (I + a K)^{-1} = P_L + P_T / (1 + a |k|^2).  Direct and exact; dense products along each
#  axis, no fill-in.  Only the transverse part of A is implicit: the longitudinal part (the
#  contact conversion layers) still moves with J_s alone.
# ============================================================================

function _sep_bases(nn::Int)
    D = zeros(nn - 1, nn)
    for e in 1:nn-1; D[e, e] = -1.0; D[e, e+1] = 1.0; end
    F = svd(D; full=true)
    V = Matrix(F.V[:, end:-1:1]); s = vcat(0.0, reverse(F.S))
    U = Matrix((D * V[:, 2:end]) ./ s[2:end]')
    V, U, s
end

struct _Ax3
    A1::A3r; A2::A3r; A3::A3r
end
_Ax3(n1, n2, n3, m1, m2, m3) = _Ax3(zeros(m1, n2, n3), zeros(m1, m2, n3), zeros(m1, m2, m3))

# A transformed by M1' along dim 1, M2' along dim 2 and M3' along dim 3
function _tx!(T::_Ax3, A::AbstractArray{Float64,3}, M1::AbstractMatrix, M2::AbstractMatrix,
              M3::AbstractMatrix)
    n1, n2, n3 = size(A); m1, m2, m3 = size(M1, 2), size(M2, 2), size(M3, 2)
    mul!(reshape(T.A1, m1, n2*n3), transpose(M1), reshape(A, n1, n2*n3))
    @views for k in 1:n3; mul!(T.A2[:, :, k], T.A1[:, :, k], M2); end
    mul!(reshape(T.A3, m1*m2, m3), reshape(T.A2, m1*m2, n3), M3)
    T.A3
end

struct SepSolver
    Vx::Matrix{Float64}; Ux::Matrix{Float64}; sx::Vector{Float64}
    Vy::Matrix{Float64}; Uy::Matrix{Float64}; sy::Vector{Float64}
    Vz::Matrix{Float64}; Uz::Matrix{Float64}; sz::Vector{Float64}
    a::Float64
    rx::NTuple{3,UnitRange{Int}}; ry::NTuple{3,UnitRange{Int}}; rz::NTuple{3,UnitRange{Int}}
    BX::A3r; BY::A3r; BZ::A3r
    fx::_Ax3; bx::_Ax3; fy::_Ax3; by::_Ax3; fz::_Ax3; bz::_Ax3
end

function SepSolver(gr::Grid, a::Float64)
    (; Nx, Ny, Nz) = gr
    Vx, Ux, sx = _sep_bases(Nx - 1); Vy, Uy, sy = _sep_bases(Ny - 1); Vz, Uz, sz = _sep_bases(Nz - 1)
    rx = (2:Nx-1, 2:Ny, 2:Nz); ry = (2:Nx, 2:Ny-1, 2:Nz); rz = (2:Nx, 2:Ny, 2:Nz-1)
    n(r) = length.(r)
    SepSolver(Vx, Ux, sx, Vy, Uy, sy, Vz, Uz, sz, a, rx, ry, rz,
              zeros(n(rx)...), zeros(n(ry)...), zeros(n(rz)...),
              _Ax3(n(rx)..., n(rx)...), _Ax3(n(rx)..., n(rx)...),
              _Ax3(n(ry)..., n(ry)...), _Ax3(n(ry)..., n(ry)...),
              _Ax3(n(rz)..., n(rz)...), _Ax3(n(rz)..., n(rz)...))
end

# (I + a K)^{-1} applied in place to S.BX, S.BY, S.BZ
function _sep_solve!(S::SepSolver)
    X = _tx!(S.fx, S.BX, S.Ux, S.Vy, S.Vz)
    Y = _tx!(S.fy, S.BY, S.Vx, S.Uy, S.Vz)
    Z = _tx!(S.fz, S.BZ, S.Vx, S.Vy, S.Uz)
    sx, sy, sz, a = S.sx, S.sy, S.sz, S.a
    @inbounds for mz in eachindex(sz), my in eachindex(sy), mx in eachindex(sx)
        k1, k2, k3 = sx[mx], sy[my], sz[mz]
        k2s = k1^2 + k2^2 + k3^2
        k2s == 0 && continue
        x1 = mx > 1 ? X[mx-1, my, mz] : 0.0
        y1 = my > 1 ? Y[mx, my-1, mz] : 0.0
        z1 = mz > 1 ? Z[mx, my, mz-1] : 0.0
        d = (k1*x1 + k2*y1 + k3*z1) / k2s; f = 1 / (1 + a*k2s)
        mx > 1 && (X[mx-1, my, mz] = k1*d + f*(x1 - k1*d))
        my > 1 && (Y[mx, my-1, mz] = k2*d + f*(y1 - k2*d))
        mz > 1 && (Z[mx, my, mz-1] = k3*d + f*(z1 - k3*d))
    end
    S.BX .= _tx!(S.bx, X, S.Ux', S.Vy', S.Vz')
    S.BY .= _tx!(S.by, Y, S.Vx', S.Uy', S.Vz')
    S.BZ .= _tx!(S.bz, Z, S.Vx', S.Vy', S.Uz')
    nothing
end

"""
    implicit_curl!(s, S)

After field_increment!, φ holds the explicit update φₑ = θ + (dt/σn)J − L θ + c; the implicit one
is φ = θ + (I − L)⁻¹(φₑ − θ).
"""
function implicit_curl!(s::State, S::SepSolver)
    @views begin
        S.BX .= s.φx[S.rx...] .- s.θx[S.rx...]
        S.BY .= s.φy[S.ry...] .- s.θy[S.ry...]
        S.BZ .= s.φz[S.rz...] .- s.θz[S.rz...]
    end
    _sep_solve!(S)
    @views begin
        s.φx[S.rx...] .= s.θx[S.rx...] .+ S.BX
        s.φy[S.ry...] .= s.θy[S.ry...] .+ S.BY
        s.φz[S.rz...] .= s.θz[S.rz...] .+ S.BZ
    end
    s
end

# ============================================================================
#  Insulating padding: a conductivity per link
#
#      sigma_l (A^{n+1} - A^n)/dt = J_s - kappa^2 (K A^{n+1} + c)
#
#  with sigma_l = sigma_n in the sample and in the contacts (the bar along x with the section of
#  the sample) and sigma_vac in the rest of the padding.  In the variables of the step,
#  (D + aK) Δ = φₑ − θ with D = sigma_l/sigma_n and a = (dt/sigma_n) kappa^2/dx^2, which is no
#  longer separable; it is solved by preconditioned conjugate gradients with the separable
#  inverse of (I + aK).  The preconditioned spectrum lies in [sigma_vac/sigma_n, 1], so the
#  iterations grow as (sigma_n/sigma_vac)^(1/2).
# ============================================================================

"""
    link_conductivity(gr, σvac, σn; cond=nothing) -> (Dx, Dy, Dz)

σ_l/σ_n on every link: 1 for links inside the bar y1:y2+1, z1:z2+1 (sample and contacts), σvac/σn
elsewhere.  With `cond` (the conducting nodes, `conducting_nodes`: a sculpted sample) a link is 1
when both its nodes conduct; for the plain bar that is the same set.
"""
function link_conductivity(gr::Grid, σvac::Float64, σn::Float64; cond=nothing)
    (; Nx, Ny, Nz, y1, y2, z1, z2) = gr
    r = σvac / σn
    if cond !== nothing
        c = cond
        return ([c[i,j,k] && c[i+1,j,k] ? 1.0 : r for i in 1:Nx,   j in 1:Ny+1, k in 1:Nz+1],
                [c[i,j,k] && c[i,j+1,k] ? 1.0 : r for i in 1:Nx+1, j in 1:Ny,   k in 1:Nz+1],
                [c[i,j,k] && c[i,j,k+1] ? 1.0 : r for i in 1:Nx+1, j in 1:Ny+1, k in 1:Nz])
    end
    iny(j) = y1 <= j <= y2 + 1; inz(k) = z1 <= k <= z2 + 1
    Dx = [iny(j) && inz(k) ? 1.0 : r for i in 1:Nx,   j in 1:Ny+1, k in 1:Nz+1]
    Dy = [iny(j) && iny(j+1) && inz(k) ? 1.0 : r for i in 1:Nx+1, j in 1:Ny, k in 1:Nz+1]
    Dz = [iny(j) && inz(k) && inz(k+1) ? 1.0 : r for i in 1:Nx+1, j in 1:Ny+1, k in 1:Nz]
    (Dx, Dy, Dz)
end

struct PCGSolver
    sep::SepSolver
    D::NTuple{3,A3r}                          # σ_l/σ_n, full link arrays
    s::NTuple{6,Float64}                      # sxy, sxz, syx, syz, szx, szy
    dx::Float64; dy::Float64; dz::Float64
    tol::Float64; maxit::Int
    P::NTuple{3,A3r}; Q::NTuple{3,A3r}        # full-size work arrays (search direction, A p)
    H::NTuple{3,A3r}                          # field of the search direction
    x::NTuple{3,A3r}; r::NTuple{3,A3r}; z::NTuple{3,A3r}; b::NTuple{3,A3r}   # on the ranges
    iters::Vector{Int}                        # iterations of the last solve
    warm::Vector{Bool}                        # x holds the previous step's solution: start from it
    fast::Union{Nothing,Any}                  # pcg_fast: the fused loops, the extrapolated guess, the Float32 solve
end

# pcg_fast (an option; the default path keeps its bits): the first guess is the linear extrapolation 2xₙ − xₙ₋₁ of the
# last two steps' solutions (4.7 iterations against 12.6 on the nanowire at 1e-8), and with `f32` the separable solve runs in
# Float32 inside the Float64 CG (not exactly symmetric: the flexible, Polak–Ribière β; −12 % more).  Fused loops for the vector
# work were tried and dropped: 7–10 % slower than the view operations.
struct PCGFast
    xprev::NTuple{3,A3r}; have2::Vector{Bool} # the solution of the step before (the extrapolation)
    zold::NTuple{3,A3r}                       # the previous z (flexible β)
    f32::Bool
    extrap::Bool                              # the extrapolated first guess (on with pcg_fast)
    V::NTuple{9,Matrix{Float32}}              # Vx, Ux, Vy, Uy, Vz, Uz and their transposes (Vx', Vy', Vz' with the U ones below)
    W::NTuple{3,Matrix{Float32}}              # Ux', Uy', Uz'
    s32::NTuple{3,Vector{Float32}}
    B32::NTuple{3,Array{Float32,3}}; T32::NTuple{6,Array{Float32,3}}
end

function PCGSolver(gr::Grid, fc::FieldCoef, D::NTuple{3,A3r}; tol::Float64=1e-11, maxit::Int=500, fast::Bool=false,
                   f32::Bool=false, extrap::Bool=true)
    sep = SepSolver(gr, fc.a)
    (; Nx, Ny, Nz) = gr
    full() = (zeros(Nx, Ny+1, Nz+1), zeros(Nx+1, Ny, Nz+1), zeros(Nx+1, Ny+1, Nz))
    rng() = (zeros(size(sep.BX)), zeros(size(sep.BY)), zeros(size(sep.BZ)))
    fs = !(fast || f32) ? nothing : begin
        f(M) = Matrix{Float32}(M)
        n(B) = size(B)
        PCGFast(rng(), [false], rng(), f32, extrap && fast,
                (f(sep.Vx), f(sep.Ux), f(sep.Vy), f(sep.Uy), f(sep.Vz), f(sep.Uz), f(sep.Vx'), f(sep.Vy'), f(sep.Vz')),
                (f(sep.Ux'), f(sep.Uy'), f(sep.Uz')), (Float32.(sep.sx), Float32.(sep.sy), Float32.(sep.sz)),
                (zeros(Float32, n(sep.BX)), zeros(Float32, n(sep.BY)), zeros(Float32, n(sep.BZ))),
                ntuple(c -> zeros(Float32, n((sep.BX, sep.BY, sep.BZ)[(c + 1) ÷ 2])), 6))
    end
    PCGSolver(sep, D, (fc.sxy, fc.sxz, fc.syx, fc.syz, fc.szx, fc.szy), gr.dx, gr.dy, gr.dz, tol, maxit,
              full(), full(), (zeros(Nx+1, Ny, Nz), zeros(Nx, Ny+1, Nz), zeros(Nx, Ny, Nz+1)),
              rng(), rng(), rng(), rng(), [0], [false], fs)
end

# The previous solution as the first guess (warm start): consecutive steps differ by little, so it takes 12
# iterations where the separable guess takes 27 (two-band notched box, σvac = 0.01); the solution is the same to
# the tolerance.  A ladder starts cold, and what continues a copy of the state gives the solver back (_pcg_snap),
# so that a ladder and its resumption keep their bits.
_pcg_reset!(st) = (st.imp isa PCGSolver && (st.imp.warm[1] = false; st.imp.fast === nothing || (st.imp.fast.have2[1] = false)); st)
_pcg_snap(st) = st.imp isa PCGSolver ? (st.imp.warm[1], deepcopy(st.imp.x),
                                       st.imp.fast === nothing ? nothing : (st.imp.fast.have2[1], deepcopy(st.imp.fast.xprev))) : nothing
function _pcg_back!(st, sn)
    sn === nothing && return st
    st.imp.warm[1] = sn[1]
    for c in 1:3; copyto!(st.imp.x[c], sn[2][c]); end
    f = st.imp.fast
    if f !== nothing
        q = length(sn) >= 3 ? sn[3] : nothing
        f.have2[1] = q !== nothing && q[1]
        q !== nothing && q[1] && for c in 1:3; copyto!(f.xprev[c], q[2][c]); end
    end
    st
end

# q = (D + aK) p on the ranges of the separable solve; p, q full-size (p zero outside the ranges)
function _apply_A!(S::PCGSolver, Nx, Ny, Nz)
    px, py, pz = S.P; qx, qy, qz = S.Q; hx, hy, hz = S.H
    sxy, sxz, syx, syz, szx, szy = S.s
    dx, dy, dz = S.dx, S.dy, S.dz
    @inbounds for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx
        hx[i,j,k] = ((pz[i,j+1,k] - pz[i,j,k]) - py[i,j,k+1] + py[i,j,k]) / (dy*dz)
    end
    @inbounds for k in 2:Nz-1, j in 2:Ny, i in 2:Nx-1
        hy[i,j,k] = ((px[i,j,k+1] - px[i,j,k]) - pz[i+1,j,k] + pz[i,j,k]) / (dz*dx)
    end
    @inbounds for k in 2:Nz, j in 2:Ny-1, i in 2:Nx-1
        hz[i,j,k] = ((py[i+1,j,k] - py[i,j,k]) - px[i,j+1,k] + px[i,j,k]) / (dx*dy)
    end
    Dx, Dy, Dz = S.D
    @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx-1
        qx[i,j,k] = Dx[i,j,k]*px[i,j,k] + sxy*(hz[i,j,k] - hz[i,j-1,k]) - sxz*(hy[i,j,k] - hy[i,j,k-1])
    end
    @inbounds for k in 2:Nz, j in 2:Ny-1, i in 2:Nx
        qy[i,j,k] = Dy[i,j,k]*py[i,j,k] + syz*(hx[i,j,k] - hx[i,j,k-1]) - syx*(hz[i,j,k] - hz[i-1,j,k])
    end
    @inbounds for k in 2:Nz-1, j in 2:Ny, i in 2:Nx
        qz[i,j,k] = Dz[i,j,k]*pz[i,j,k] + szx*(hy[i,j,k] - hy[i-1,j,k]) - szy*(hx[i,j,k] - hx[i,j-1,k])
    end
    nothing
end

_dot3(a, b) = dot(a[1], b[1]) + dot(a[2], b[2]) + dot(a[3], b[3])

function _precond!(z, r, S::PCGSolver)
    sp = S.sep
    sp.BX .= r[1]; sp.BY .= r[2]; sp.BZ .= r[3]
    _sep_solve!(sp)
    z[1] .= sp.BX; z[2] .= sp.BY; z[3] .= sp.BZ
    z
end

"""
    implicit_curl!(s, S::PCGSolver)

As the separable version, with a conductivity per link: (D + aK) Δ = φₑ − θ on the coupled links
by PCG; on the links outside the ranges (decoupled from the curl-curl) Δ = (φₑ − θ)/D.
"""
function implicit_curl!(s::State, S::PCGSolver)
    S.fast === nothing || return _implicit_curl_fast!(s, S, S.fast)
    sp = S.sep
    Nx = size(s.φy, 1) - 1; Ny = size(s.φx, 2) - 1; Nz = size(s.φx, 3) - 1
    Dx, Dy, Dz = S.D
    b = S.b; x = S.x; r = S.r; z = S.z
    @views begin
        b[1] .= s.φx[sp.rx...] .- s.θx[sp.rx...]
        b[2] .= s.φy[sp.ry...] .- s.θy[sp.ry...]
        b[3] .= s.φz[sp.rz...] .- s.θz[sp.rz...]
    end
    # the decoupled links (all of them here; the coupled ones are overwritten at the end)
    @. s.φx = s.θx + (s.φx - s.θx) / Dx
    @. s.φy = s.θy + (s.φy - s.θy) / Dy
    @. s.φz = s.θz + (s.φz - s.θz) / Dz
    S.warm[1] || _precond!(x, b, S)
    S.warm[1] = true
    P = S.P; Q = S.Q
    rngs = (sp.rx, sp.ry, sp.rz)
    scatter!(full, v) = (for c in 1:3; fill!(full[c], 0.0); view(full[c], rngs[c]...) .= v[c]; end)
    gather(full, c) = view(full[c], rngs[c]...)
    scatter!(P, x); _apply_A!(S, Nx, Ny, Nz)
    for c in 1:3; r[c] .= b[c] .- gather(Q, c); end
    nb = sqrt(_dot3(b, b)); it = 0
    if nb > 0 && sqrt(_dot3(r, r)) > S.tol * nb
        _precond!(z, r, S)
        scatter!(P, z)
        rz = _dot3(r, z)
        while it < S.maxit
            it += 1
            _apply_A!(S, Nx, Ny, Nz)
            pAp = 0.0
            for c in 1:3; pAp += dot(gather(P, c), gather(Q, c)); end
            α = rz / pAp
            for c in 1:3
                x[c] .+= α .* gather(P, c)
                r[c] .-= α .* gather(Q, c)
            end
            sqrt(_dot3(r, r)) <= S.tol * nb && break
            _precond!(z, r, S)
            rz2 = _dot3(r, z); β = rz2 / rz; rz = rz2
            for c in 1:3
                gather(P, c) .= z[c] .+ β .* gather(P, c)
            end
        end
    end
    S.iters[1] = it
    @views begin
        s.φx[sp.rx...] .= s.θx[sp.rx...] .+ x[1]
        s.φy[sp.ry...] .= s.θy[sp.ry...] .+ x[2]
        s.φz[sp.rz...] .= s.θz[sp.rz...] .+ x[3]
    end
    s
end

"""Explicit curl-curl with a conductivity per link: φ = θ + (φₑ − θ)/D."""
function scale_increment!(s::State, D::NTuple{3,A3r})
    @. s.φx = s.θx + (s.φx - s.θx) / D[1]
    @. s.φy = s.θy + (s.φy - s.θy) / D[2]
    @. s.φz = s.θz + (s.φz - s.θz) / D[3]
    s
end

# ---- pcg_fast ---------------------------------------------------------------------------------------------------
# the separable inverse in Float32 (as _sep_solve!, the dense products in single precision)
function _tx32!(T1, T2, B, A, M1, M2, M3)
    n1, n2, n3 = size(A); m1, m2, m3 = size(M1, 2), size(M2, 2), size(M3, 2)
    mul!(reshape(T1, m1, n2*n3), transpose(M1), reshape(A, n1, n2*n3))
    @views for k in 1:n3; mul!(T2[:, :, k], T1[:, :, k], M2); end
    mul!(reshape(B, m1*m2, m3), reshape(T2, m1*m2, n3), M3)
    B
end
function _precond32!(z, r, S::PCGSolver, f::PCGFast)
    sp = S.sep; a = Float32(sp.a)
    Vx, Ux, Vy, Uy, Vz, Uz, Vxt, Vyt, Vzt = f.V; Uxt, Uyt, Uzt = f.W
    sx, sy, sz = f.s32; BX, BY, BZ = f.B32; T = f.T32
    BX .= r[1]; BY .= r[2]; BZ .= r[3]
    X = _tx32!(T[1], T[2], BX, BX, Ux, Vy, Vz); Y = _tx32!(T[3], T[4], BY, BY, Vx, Uy, Vz)    # in place (A is read first)
    Z = _tx32!(T[5], T[6], BZ, BZ, Vx, Vy, Uz)
    @inbounds for mz in eachindex(sz), my in eachindex(sy), mx in eachindex(sx)
        k1, k2, k3 = sx[mx], sy[my], sz[mz]
        k2s = k1^2 + k2^2 + k3^2
        k2s == 0 && continue
        x1 = mx > 1 ? X[mx-1, my, mz] : 0f0; y1 = my > 1 ? Y[mx, my-1, mz] : 0f0; z1 = mz > 1 ? Z[mx, my, mz-1] : 0f0
        d = (k1*x1 + k2*y1 + k3*z1) / k2s; g = 1 / (1 + a*k2s)
        mx > 1 && (X[mx-1, my, mz] = k1*d + g*(x1 - k1*d))
        my > 1 && (Y[mx, my-1, mz] = k2*d + g*(y1 - k2*d))
        mz > 1 && (Z[mx, my, mz-1] = k3*d + g*(z1 - k3*d))
    end
    z[1] .= _tx32!(T[1], T[2], BX, X, Uxt, Vyt, Vzt); z[2] .= _tx32!(T[3], T[4], BY, Y, Vxt, Uyt, Vzt)
    z[3] .= _tx32!(T[5], T[6], BZ, Z, Vxt, Vyt, Uzt)
    z
end
_precf!(z, r, S, f) = f.f32 ? _precond32!(z, r, S, f) : _precond!(z, r, S)

function _implicit_curl_fast!(s::State, S::PCGSolver, f::PCGFast)
    sp = S.sep
    Nx = size(s.φy, 1) - 1; Ny = size(s.φx, 2) - 1; Nz = size(s.φx, 3) - 1
    Dx, Dy, Dz = S.D
    b = S.b; x = S.x; r = S.r; z = S.z
    @views begin
        b[1] .= s.φx[sp.rx...] .- s.θx[sp.rx...]
        b[2] .= s.φy[sp.ry...] .- s.θy[sp.ry...]
        b[3] .= s.φz[sp.rz...] .- s.θz[sp.rz...]
    end
    @. s.φx = s.θx + (s.φx - s.θx) / Dx
    @. s.φy = s.θy + (s.φy - s.θy) / Dy
    @. s.φz = s.θz + (s.φz - s.θz) / Dz
    if !S.warm[1]
        _precond!(x, b, S); f.have2[1] = false
    elseif !f.extrap
    elseif f.have2[1]
        for c in 1:3                                   # x ← 2xₙ − xₙ₋₁, and xₙ kept as the next step's xₙ₋₁
            xc = x[c]; pc = f.xprev[c]
            @inbounds @simd for i in eachindex(xc); v = xc[i]; xc[i] = 2v - pc[i]; pc[i] = v; end
        end
    else
        for c in 1:3; copyto!(f.xprev[c], x[c]); end
        f.have2[1] = true
    end
    S.warm[1] = true
    P = S.P; Q = S.Q
    rngs = (sp.rx, sp.ry, sp.rz)
    for c in 1:3; fill!(P[c], 0.0); view(P[c], rngs[c]...) .= x[c]; end
    _apply_A!(S, Nx, Ny, Nz)
    for c in 1:3; r[c] .= b[c] .- view(Q[c], rngs[c]...); end
    nb = sqrt(_dot3(b, b)); it = 0
    if nb > 0 && sqrt(_dot3(r, r)) > S.tol * nb
        _precf!(z, r, S, f)
        for c in 1:3; fill!(P[c], 0.0); view(P[c], rngs[c]...) .= z[c]; end
        f.f32 && for c in 1:3; copyto!(f.zold[c], z[c]); end
        rz = _dot3(r, z)
        gather(full, c) = view(full[c], rngs[c]...)
        while it < S.maxit
            it += 1
            _apply_A!(S, Nx, Ny, Nz)
            pAp = 0.0
            for c in 1:3; pAp += dot(gather(P, c), gather(Q, c)); end
            α = rz / pAp
            for c in 1:3
                x[c] .+= α .* gather(P, c)
                r[c] .-= α .* gather(Q, c)
            end
            sqrt(_dot3(r, r)) <= S.tol * nb && break
            _precf!(z, r, S, f)
            rz2 = _dot3(r, z)
            if f.f32                                   # flexible β = r·(z − z_old)/rz_old
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
        s.φx[sp.rx...] .= s.θx[sp.rx...] .+ x[1]
        s.φy[sp.ry...] .= s.θy[sp.ry...] .+ x[2]
        s.φz[sp.rz...] .= s.θz[sp.rz...] .+ x[3]
    end
    s
end
