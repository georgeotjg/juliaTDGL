# ============================================================================
#  maxwell_implicit.jl — OPTIONAL IMPLICIT CURL-CURL IN BLOCK [C]
#
#  Block [C] advances  dA/dt = J_s - kappa^2 curl curl A  explicitly, and the
#  curl-curl term limits the step to dl/(4 kappa^2): at kappa = 25 that is 38
#  times below the next limit (the pi-band diffusion).  With
#  `maxwell_implicit = true` that term is taken at the new time level:
#
#      (I + dt kappa^2 K) A^{n+1} = A^n + dt J_s^n - dt kappa^2 c,
#
#  K the discrete curl^T curl over the links that [C] updates and c the
#  contribution of the boundary fields (applied field, self-field).  K is
#  built from the code of [C] itself, so the implicit operator is the
#  explicit one, link by link.  It acts only on the transverse part of A: the
#  longitudinal part (gradients, the contact conversion layers) still moves
#  with J_s alone, as in the explicit scheme.  K does not depend on time, so
#  the matrix is factorised once per sweep.
#
#  Fixed points are those of the explicit scheme (kappa^2 K A + c = J_s in
#  both).  Off by default; with it off nothing here runs.
# ============================================================================

# ---------------------------------------------------------------------------
#  Separable solver.  K is the Yee curl-curl on tensor-product ranges of links
#  (the boundary rows of [F]/[G] remove whole planes), so with dx = dy = dz it
#  is diagonalised by per-axis bases: the right singular vectors V of the 1D
#  forward difference on nodes (cosine-like) and U = D V / s on edges
#  (sine-like).  Per wavevector k the block is |k|^2 I - k k^T, and
#  (I + a K)^{-1} = P_L + P_T / (1 + a |k|^2).  Cost: dense products along each
#  axis, O(N (n1 + n2 + n3)), no fill-in.
# ---------------------------------------------------------------------------
function _sep_bases(nn::Int)
    D = zeros(nn - 1, nn)
    for e in 1:nn-1; D[e, e] = -1.0; D[e, e+1] = 1.0; end
    F = svd(D; full=true)
    V = Matrix(F.V[:, end:-1:1]); s = vcat(0.0, reverse(F.S))
    U = Matrix((D * V[:, 2:end]) ./ s[2:end]')
    V, U, s
end

struct _Ax3
    A1::Array{Float64,3}; A2::Array{Float64,3}; A3::Array{Float64,3}
end
_Ax3(n1, n2, n3, m1, m2, m3) = _Ax3(zeros(m1, n2, n3), zeros(m1, m2, n3), zeros(m1, m2, m3))

# B = A transformed by M1' along dim 1, M2' along dim 2 and M3' along dim 3
function _tx!(T::_Ax3, A::AbstractArray{Float64,3}, M1::AbstractMatrix, M2::AbstractMatrix,
              M3::AbstractMatrix)
    n1, n2, n3 = size(A); m1, m2, m3 = size(M1, 2), size(M2, 2), size(M3, 2)
    mul!(reshape(T.A1, m1, n2*n3), transpose(M1), reshape(A, n1, n2*n3))
    @views for k in 1:n3; mul!(T.A2[:, :, k], T.A1[:, :, k], M2); end
    mul!(reshape(T.A3, m1*m2, m3), reshape(T.A2, m1*m2, n3), M3)
    T.A3
end

struct _SepSolver
    Vx::Matrix{Float64}; Ux::Matrix{Float64}; sx::Vector{Float64}
    Vy::Matrix{Float64}; Uy::Matrix{Float64}; sy::Vector{Float64}
    Vz::Matrix{Float64}; Uz::Matrix{Float64}; sz::Vector{Float64}
    a::Float64                                   # dt kappa^2 / dx^2
    rx::NTuple{3,UnitRange{Int}}; ry::NTuple{3,UnitRange{Int}}; rz::NTuple{3,UnitRange{Int}}
    BX::Array{Float64,3}; BY::Array{Float64,3}; BZ::Array{Float64,3}
    fx::_Ax3; bx::_Ax3; fy::_Ax3; by::_Ax3; fz::_Ax3; bz::_Ax3
end

function _SepSolver(p::Params)
    (; Nx, Ny, Nz) = p
    Vx, Ux, sx = _sep_bases(Nx - 1); Vy, Uy, sy = _sep_bases(Ny - 1); Vz, Uz, sz = _sep_bases(Nz - 1)
    rx = (2:Nx-1, 2:Ny, 2:Nz); ry = (2:Nx, 2:Ny-1, 2:Nz); rz = (2:Nx, 2:Ny, 2:Nz-1)
    n(r) = length.(r)
    _SepSolver(Vx, Ux, sx, Vy, Uy, sy, Vz, Uz, sz, p.dt * p.ks2 / p.dx^2, rx, ry, rz,
               zeros(n(rx)...), zeros(n(ry)...), zeros(n(rz)...),
               _Ax3(n(rx)..., n(rx)...), _Ax3(n(rx)..., n(rx)...),
               _Ax3(n(ry)..., n(ry)...), _Ax3(n(ry)..., n(ry)...),
               _Ax3(n(rz)..., n(rz)...), _Ax3(n(rz)..., n(rz)...))
end

# (I + a K)^{-1} applied in place to S.BX, S.BY, S.BZ
function _sep_solve!(S::_SepSolver)
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

"""Implicit curl-curl step for block [C]: separable solver, or a sparse LU."""
struct MaxwellImplicit
    ix::Vector{CartesianIndex{3}}; iy::Vector{CartesianIndex{3}}; iz::Vector{CartesianIndex{3}}
    L::SparseMatrixCSC{Float64,Int}      # explicit curl-curl term of [C]: phi += L*theta + c
    sep::Union{Nothing,_SepSolver}
    F::Union{Nothing,SparseArrays.UMFPACK.UmfpackLU{Float64,Int}}   # LU of I - L, if no `sep`
    b::Vector{Float64}; x::Vector{Float64}
end

# The links [C] updates, in the same ranges as bloque_C!.
function _links_C(p::Params)
    (; Nx, Ny, Nz) = p
    ix = vec([CartesianIndex(i, j, k) for i in 1:Nx, j in 2:Ny, k in 2:Nz])
    iy = vec([CartesianIndex(i, j, k) for i in 2:Nx, j in 1:Ny, k in 2:Nz])
    iz = vec([CartesianIndex(i, j, k) for i in 2:Nx, j in 2:Ny, k in 1:Nz])
    ix, iy, iz
end

# The curl-curl term of [C] alone: what bloque_C! adds to theta besides the
# supercurrent, for link phases `s.phi*` and boundary fields H, HJ.
function _curl_term!(ox, oy, oz, s::State, p::Params, H::Float64, HJ::Float64)
    (; Nx, Ny, Nz, sxy, sxz, syx, syz, szx, szy) = p
    bloque_F!(s, p)
    bloque_G!(s, p, H, HJ)
    hx, hy, hz = s.hx, s.hy, s.hz
    @inbounds for k in 2:Nz, j in 2:Ny, i in 1:Nx
        ox[i,j,k] = -sxy*(hz[i,j,k] - hz[i,j-1,k]) + sxz*(hy[i,j,k] - hy[i,j,k-1])
    end
    @inbounds for k in 2:Nz, j in 1:Ny, i in 2:Nx
        oy[i,j,k] = -syz*(hx[i,j,k] - hx[i,j,k-1]) + syx*(hz[i,j,k] - hz[i-1,j,k])
    end
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx
        oz[i,j,k] = -szx*(hy[i,j,k] - hy[i-1,j,k]) + szy*(hx[i,j,k] - hx[i,j-1,k])
    end
    nothing
end

"""
    MaxwellImplicit(p, g; solver=:auto) -> MaxwellImplicit

Builds K by probing the curl-curl term of [C] with unit link phases.  Links
whose indices agree modulo 3 have disjoint stencils (each link couples only
to links within one index), so 3 types x 27 colours = 81 evaluations give
every column.  `solver=:auto` uses the separable solver when dx = dy = dz and
it reproduces (I - L) on a random vector; otherwise, or with `solver=:lu`, a
sparse LU.
"""
function MaxwellImplicit(p::Params, g::Geometry; solver::Symbol=:auto)
    # BLAS threads here, not only in sweep: scripts that step with substep! also get them.  On a
    # 24-cell padded box six OpenBLAS threads take six cores and are slower than one.
    p.blas_threads > 0 && BLAS.set_num_threads(p.blas_threads)
    ix, iy, iz = _links_C(p)
    sets = (ix, iy, iz)
    off = (0, length(ix), length(ix) + length(iy))
    n = off[3] + length(iz)
    s = initial_state(p, g, 0.0, Strict())
    for a in (s.G1, s.G2); fill!(a, 0); end
    num = (zeros(Int, size(s.phix)), zeros(Int, size(s.phiy)), zeros(Int, size(s.phiz)))
    for t in 1:3, (q, I) in enumerate(sets[t]); num[t][I] = off[t] + q; end
    out = (similar(s.phix), similar(s.phiy), similar(s.phiz))
    # affine part (h_tilt, boundary entries [F] does not write): removed from every probe
    base = (zeros(size(s.phix)), zeros(size(s.phiy)), zeros(size(s.phiz)))
    for f in (s.phix, s.phiy, s.phiz); fill!(f, 0.0); end
    _curl_term!(base..., s, p, 0.0, 0.0)
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for a in 1:3, c in CartesianIndices((0:2, 0:2, 0:2))
        phis = (s.phix, s.phiy, s.phiz)
        for f in phis; fill!(f, 0.0); end
        for f in out; fill!(f, 0.0); end
        for I in sets[a]
            mod(I[1], 3) == c[1] && mod(I[2], 3) == c[2] && mod(I[3], 3) == c[3] &&
                (phis[a][I] = 1.0)
        end
        _curl_term!(out..., s, p, 0.0, 0.0)
        for b in 1:3, R in sets[b]
            v = out[b][R] - base[b][R]
            v == 0.0 && continue
            # the unique link of type a and colour c within one index of R
            src = CartesianIndex(ntuple(d -> R[d] + mod(c[d] - R[d] + 1, 3) - 1, 3))
            checkbounds(Bool, num[a], src) || continue
            col = num[a][src]
            col == 0 && continue
            push!(I_, num[b][R]); push!(J_, col); push!(V_, v)
        end
    end
    L = sparse(I_, J_, V_, n, n)
    sep = nothing
    if solver === :auto && p.dx == p.dy == p.dz
        sep = _SepSolver(p)
        # check: (I - L) x = b on a random b, through the same scatter as the step
        b = sin.(1:n); x = zeros(n)
        _sep_apply!(x, b, sep, ix, iy, iz, s)
        r = x .- L * x .- b
        maximum(abs, r) <= 1e-10 * maximum(abs, b) || (sep = nothing)
    end
    F = sep === nothing ? lu(sparse(1.0I, n, n) - L) : nothing
    MaxwellImplicit(ix, iy, iz, L, sep, F, zeros(n), zeros(n))
end

# x = (I - L)^{-1} b for a vector over the link lists, through the separable solver
# (links outside its ranges are decoupled: x = b there)
function _sep_apply!(x, b, S::_SepSolver, ix, iy, iz, s::State)
    fx, fy, fz = s.phix, s.phiy, s.phiz
    q = 0
    for (f, set) in ((fx, ix), (fy, iy), (fz, iz)), I in set; q += 1; f[I] = b[q]; end
    S.BX .= view(fx, S.rx...); S.BY .= view(fy, S.ry...); S.BZ .= view(fz, S.rz...)
    _sep_solve!(S)
    view(fx, S.rx...) .= S.BX; view(fy, S.ry...) .= S.BY; view(fz, S.rz...) .= S.BZ
    q = 0
    for (f, set) in ((fx, ix), (fy, iy), (fz, iz)), I in set; q += 1; x[q] = f[I]; end
    x
end

"""
    implicit_curl!(s, mi)

Called right after bloque_C!: `s.phi*` holds the explicit update
phi_e = theta + dt J_s + L theta + c.  The implicit one solves
(I - L) phi = theta + dt J_s + c, that is  phi = theta + (I - L)^{-1} (phi_e - theta).
"""
function implicit_curl!(s::State, mi::MaxwellImplicit)
    S = mi.sep
    if S !== nothing
        @views begin
            S.BX .= s.phix[S.rx...] .- s.thetax[S.rx...]
            S.BY .= s.phiy[S.ry...] .- s.thetay[S.ry...]
            S.BZ .= s.phiz[S.rz...] .- s.thetaz[S.rz...]
        end
        _sep_solve!(S)
        @views begin
            s.phix[S.rx...] .= s.thetax[S.rx...] .+ S.BX
            s.phiy[S.ry...] .= s.thetay[S.ry...] .+ S.BY
            s.phiz[S.rz...] .= s.thetaz[S.rz...] .+ S.BZ
        end
        return s
    end
    (; ix, iy, iz, F, b, x) = mi
    q = 0
    @inbounds for (th, ph, set) in ((s.thetax, s.phix, ix), (s.thetay, s.phiy, iy),
                                    (s.thetaz, s.phiz, iz))
        for I in set
            q += 1; b[q] = ph[I] - th[I]
        end
    end
    ldiv!(x, F, b)
    q = 0
    @inbounds for (th, ph, set) in ((s.thetax, s.phix, ix), (s.thetay, s.phiy, iy),
                                    (s.thetaz, s.phiz, iz))
        for I in set
            q += 1; ph[I] = th[I] + x[q]
        end
    end
    s
end
