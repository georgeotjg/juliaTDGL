# ============================================================================
#  exchange.jl — two normal fluids: the quasiparticles of each band with their own potential
#
#  With one normal current σn E (the default) the quasiparticles of every band share one electrochemical
#  potential: their interband exchange is infinitely fast.  With `qp_exchange = ν` (two bands) each band has
#  its own normal current σ_µ(−∇μ_µ − ∂ₜA), σ_µ = s_µ σn, and the bands exchange quasiparticles at the rate ν
#  (σ → π, per t_GL; G = χ_σ ν with χ_µ = σ_µ Γ_µ/g_µ = 2e²N_µ, symmetric by detailed balance).  Per band
#  (slaved charge imbalance, as KWT):
#      ∇·J_n,µ = −C_µ − G(μ_µ − μ_ν),     C_µ = Im(Ψ_µ* R_µ) = ∇·J_s,µ + (Josephson)
#  where Γ_µ∂ₜΨ_µ = R_µ is the band equation (C_µ is the conversion of band µ: what its phase equation gives).
#  In the gauge Σ_µ s_µ μ_µ = 0 Ampère is unchanged (σn ∂ₜA = J_s − κ²∇×∇×A) and one field is new,
#  δ = μ_π − μ_σ (μ_σ = −s_π δ, μ_π = s_σ δ), by the difference of the two continuities:
#      σn ∇²δ − G(1/s_σ + 1/s_π) δ = C_π/s_π − C_σ/s_σ
#  on the nodes of the box (sample and metal padding), δ = 0 on the x faces (the contacts: one fluid in the
#  leads) and Neumann on the others; solved exactly by the eigenbases of the three 1D operators (dense
#  products per axis, as the separable curl-curl).  Each band then turns by e^{−iμ_µ dt} (tangential: the
#  KWT factor does not act on it).  ν → ∞: δ → 0, the step of today.  It is the branch (Schmid–Schön) imbalance of each
#  band, slaved (its memory does not move a saddle-node onset), with a finite interband exchange; a strip reduces it to
#  the 1D two-fluid law (test_physics.jl).  N bands: QPExchangeN, below.
#
#  With the insulating padding (σvac) the quasiparticles live only in the conductor: δ is solved on the bar of the sample and
#  its contacts (y1:y2+1, z1:z2+1, every x), with zero flux (Neumann) on its faces to the insulator, still separable and exact.
#  A sculpted sample (vacuum inside the block) cuts the links to the vacuum: no longer separable, the same equation is solved
#  by conjugate gradients with the separable solve of the bar as preconditioner (QPSolve, below), started cold every step (δ is
#  a function of the state: a resumed ladder keeps its bits).  The Joule heat of the two fluids is σn|E|² + σn Σ_µ s_µ|∇μ_µ|²
#  + Σ_{µ<ν} G_µν(μ_µ − μ_ν)² (qp_joule: the gradient and the exchange parts; the cross term vanishes in the gauge Σ s_µμ_µ = 0).
# ============================================================================

# The region of δ and, for a sculpted sample, the conjugate-gradient solve on it.  The second difference with mirror ghosts on
# the Neumann faces is symmetric in the inner product with weight 1/2 on each Neumann face (per axis): CG runs in that product.
struct QPSolve
    j0::Int; k0::Int                       # first y and z node of the region
    cond::Union{Nothing,BitArray{3}}       # conducting unknowns (sculpted sample), nothing = all
    wt::Array{Float64,3}                   # the inner-product weights of the unknowns
    r::Array{Float64,3}; z::Array{Float64,3}; p::Array{Float64,3}; q::Array{Float64,3}; x::Array{Float64,3}
    tol::Float64; maxit::Int
    iters::Vector{Int}                     # iterations of the last solve (per mode)
end

function QPSolve(gr::Grid, jr::UnitRange{Int}, kr::UnitRange{Int}, cond; tol=1e-10, maxit=1000)
    Nx = gr.Nx; nj = length(jr); nk = length(kr); su = (Nx - 1, nj, nk)
    c = cond === nothing ? nothing : BitArray(cond[i+1, j, k] for i in 1:Nx-1, j in jr, k in kr)
    wt = [(j == 1 || j == nj ? 0.5 : 1.0) * (k == 1 || k == nk ? 0.5 : 1.0) for i in 1:Nx-1, j in 1:nj, k in 1:nk]
    QPSolve(first(jr), first(kr), c, wt, zeros(su), zeros(su), zeros(su), zeros(su), zeros(su), Float64(tol), maxit, [0])
end

# y = −(σn∇²_c − λ) x on the unknowns: the links to a non-conducting node (or from one) are cut; x = 0 on the Dirichlet ends;
# a non-conducting unknown is decoupled (y = (6σn/h² + λ) x, its source is 0)
function _qp_apply!(y, x, ps::QPSolve, σn, λ, hx, hy, hz)
    c = ps.cond
    n1, n2, n3 = size(x)
    ax = σn / hx^2; ay = σn / hy^2; az = σn / hz^2
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        if !c[i,j,k]
            y[i,j,k] = (2ax + 2ay + 2az + λ) * x[i,j,k]; continue
        end
        v = x[i,j,k]; acc = λ * v
        # x: Dirichlet ends (δ = 0 beyond the unknowns; the end node conducts when the unknown next to it does)
        acc += ax * (v - (i > 1 ? (c[i-1,j,k] ? x[i-1,j,k] : v) : 0.0))
        acc += ax * (v - (i < n1 ? (c[i+1,j,k] ? x[i+1,j,k] : v) : 0.0))
        # y, z: mirror ghosts on the faces (factor 2 to the only neighbour)
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
    y
end

_wdot(a, b, w) = (s = 0.0; @inbounds for I in eachindex(a); s += w[I] * a[I] * b[I]; end; s)

# solves (σn∇²_c − λ) x = S on the unknowns (S overwritten by x), preconditioned by `sep!(z, r)`, the separable solve of
# (σn∇² − λ) z = r on the whole region
function _qp_cg!(S, ps::QPSolve, σn, λ, hx, hy, hz, sep!)
    r, z, p, q, x, w = ps.r, ps.z, ps.p, ps.q, ps.x, ps.wt
    c = ps.cond
    @inbounds for I in eachindex(S); c[I] || (S[I] = 0.0); end
    @. r = -S                                   # A x = b with A = −(σn∇²_c − λ), b = −S
    bn = sqrt(_wdot(r, r, w)); bn == 0 && (fill!(S, 0.0); ps.iters[1] = 0; return S)
    # z = Π A_sep⁻¹ r, Π the projection on the conducting unknowns (the vacuum ones are decoupled, with no source: x = 0)
    prec!(z, r) = (sep!(z, r); @inbounds(for I in eachindex(z); z[I] = c[I] ? -z[I] : 0.0; end); z)
    prec!(z, r)
    copyto!(x, z)
    _qp_apply!(q, x, ps, σn, λ, hx, hy, hz); @. r = r - q
    prec!(z, r); copyto!(p, z)
    rz = _wdot(r, z, w); it = 0
    while sqrt(_wdot(r, r, w)) > ps.tol * bn && it < ps.maxit
        _qp_apply!(q, p, ps, σn, λ, hx, hy, hz)
        α = rz / _wdot(p, q, w)
        @. x += α * p; @. r -= α * q
        prec!(z, r)
        rz2 = _wdot(r, z, w); β = rz2 / rz; rz = rz2
        @. p = z + β * p
        it += 1
    end
    it >= ps.maxit && @warn "qp_exchange: conjugate gradients did not converge in $(ps.maxit) iterations"
    ps.iters[1] = it
    copyto!(S, x)
end

# Two normal fluids with materials.  With the split s_µ the same in every material, the gauge
# Σ_µ s_µμ_µ = 0 still leaves Ampère unchanged (Σ_µ s_µσ_l∇μ_µ = σ_l∇Σ s_µμ_µ = 0) and the difference of the two continuities is
#     ∇·(σn∇δ) − λ(x) δ = C_π/s_π − C_σ/s_σ,     λ = G(1/s_σ + 1/s_π),  G = s_σσn Γ_σ ν/g_σ of the node's material,
# σn per link the harmonic mean of its nodes (as the field, materials.jl); each band's normal current s_µσ_l(−∇μ_µ − ∂ₜA) is a
# link flux, continuous across an interface.  A material with a one-fluid normal channel (ν = ∞, or a band absent: a one-band
# metal) holds δ = 0 on its nodes (Dirichlet).  Not separable: conjugate gradients preconditioned by the separable solve with the
# first material's σn and λ.  A split that changes between materials would add −σn δ∇s_σ to Ampère and an E·∇s term to the δ
# equation (the common potential is then no longer a gradient the gauge absorbs): refused.
struct QPVar
    typ::Array{Int8,3}             # per unknown: 1 free, 0 cut (vacuum), 2 held at δ = 0 (one fluid)
    cx::Array{Float64,3}; cy::Array{Float64,3}; cz::Array{Float64,3}   # σ_l/h² on the links of the region (cx includes the two x ends)
    λ::Array{Float64,3}
end

struct QPExchange
    ν::Float64                     # quasiparticle exchange rate σ → π [1/t_GL]
    s::NTuple{2,Float64}           # share of σn of each band's normal fluid
    G::Float64                     # exchange conductance per volume, χ_σ ν
    λ0::Float64                    # G (1/s_σ + 1/s_π): σn/ℓ_ex²
    σn::Float64
    Vx::Matrix{Float64}; Wx::Matrix{Float64}; ex::Vector{Float64}   # x: interior nodes 2..Nx (Dirichlet)
    Vy::Matrix{Float64}; Wy::Matrix{Float64}; ey::Vector{Float64}   # y: nodes 1..Ny+1 (Neumann)
    Vz::Matrix{Float64}; Wz::Matrix{Float64}; ez::Vector{Float64}
    δ::Array{Float64,3}            # μ_π − μ_σ on the nodes
    S::Array{Float64,3}            # the source, then the modes
    T1::Array{Float64,3}; T2::Array{Float64,3}
    ps::QPSolve                    # the region of δ (and the CG of a sculpted sample)
    var::Union{Nothing,QPVar}      # materials with σn, λ or a one-fluid region that change by node (nothing: uniform)
end

# 1D second difference on n nodes with spacing h: Dirichlet (the n unknowns between two fixed ends) or Neumann
# (mirror ghosts at both ends); its eigenbasis V (columns), W = V⁻¹ and the eigenvalues.
function _lap1d(n::Int, h::Float64, dirichlet::Bool)
    L = zeros(n, n)
    for a in 1:n
        L[a, a] = -2 / h^2
        a > 1 && (L[a, a-1] = 1 / h^2)
        a < n && (L[a, a+1] = 1 / h^2)
    end
    if !dirichlet && n > 1
        L[1, 2] = 2 / h^2; L[n, n-1] = 2 / h^2
    end
    E = eigen(L)
    V = real.(E.vectors); e = real.(E.values)
    V, inv(V), e
end

"""
    QPExchange(md, gr; ν, split=nothing, conductor=false, cond=nothing)

The two normal fluids of a two-band model on the grid `gr` (see the header of exchange.jl).  `split` is the share of σn
of each band; by default the dirty one, s_µ ∝ g_µΓ_µ/β_µ (as `sigma_dirty`).  `conductor = true` (the insulating padding):
δ on the bar of the sample and its contacts only; `cond` (a sculpted sample: `conducting_nodes`) cuts the links to the vacuum.
"""
function QPExchange(md::Model{N}, gr::Grid; ν::Real, split=nothing, conductor::Bool=false, cond=nothing,
                    mt=nothing, mk=nothing, rates=nothing) where N
    N == 2 || error("qp_exchange: a number is for two bands; with $N bands give the matrix of rates (QPExchangeN)")
    ν >= 0 || error("qp_exchange: the rate must be >= 0")
    b = md.bands
    w = split === nothing ? ntuple(µ -> stiffness(b[µ]) * b[µ].Γ / b[µ].β, 2) : Float64.(Tuple(split))
    length(w) == 2 && all(>(0), w) || error("qp_split: two positive shares")
    s = (w[1] / (w[1] + w[2]), w[2] / (w[1] + w[2]))
    χσ = s[1] * md.σn * b[1].Γ / stiffness(b[1])
    G = χσ * Float64(ν)
    (; Nx, Ny, Nz, dx, dy, dz) = gr
    jr, kr = conductor ? (gr.y1:gr.y2+1, gr.z1:gr.z2+1) : (1:Ny+1, 1:Nz+1)
    Vx, Wx, ex = _lap1d(Nx - 1, dx, true)
    Vy, Wy, ey = _lap1d(length(jr), dy, false)
    Vz, Wz, ez = _lap1d(length(kr), dz, false)
    sz = (Nx + 1, Ny + 1, Nz + 1); su = (Nx - 1, length(jr), length(kr))
    var = mt === nothing ? nothing : _qp_var(md, gr, mt, mk, s, split, Float64(ν), rates, jr, kr, cond)
    QPExchange(Float64(ν), s, G, G * (1 / s[1] + 1 / s[2]), md.σn, Vx, Wx, ex, Vy, Wy, ey, Vz, Wz, ez,
               zeros(sz), zeros(su), zeros(su), zeros(su), QPSolve(gr, jr, kr, cond), var)
end

# The coefficients of the δ equation per node and link with materials; nothing when they are those of the first material
# everywhere (then the separable or sculpted solve of today, and its bits).
function _qp_var(md, gr, mt, mk, s, split, ν, rates, jr, kr, cond)
    mats = mt.models; M = length(mats)
    r = rates === nothing ? fill(ν, M) : Float64.(collect(rates))
    length(r) == M && r[1] == ν && all(>=(0), r) || error("qp_exchange per material: $M rates, the first that of [material] ($ν)")
    one = [r[m] == Inf || any(mt.absent[m, :]) for m in 1:M]          # one normal fluid there (δ = 0)
    one[1] && error("qp_exchange: the first material has two fluids (a finite rate)")
    if split === nothing
        for m in 2:M
            one[m] && continue
            bm = mats[m].bands; wm = ntuple(µ -> stiffness(bm[µ]) * bm[µ].Γ / bm[µ].β, 2); sm = wm[1] / (wm[1] + wm[2])
            abs(sm - s[1]) <= 1e-12 || error("qp_exchange with materials: the dirty split of material $m (s_σ = $(round(sm, sigdigits=10))) " *
                "is not that of [material] ($(round(s[1], sigdigits=10))); a split that changes by region is not supported " *
                ": give qp_split, the same for every material")
        end
    end
    λm = [one[m] ? 0.0 : s[1] * mats[m].σn * mats[m].bands[1].Γ / stiffness(mats[m].bands[1]) * r[m] * (1 / s[1] + 1 / s[2]) for m in 1:M]
    uniform = !any(one) && all(m -> mats[m].σn == md.σn && λm[m] == λm[1], 1:M)
    uniform && return nothing
    (; Nx, x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    mat(i, j, k) = @inbounds mt.idx[clamp(i, x1, x2+1), clamp(j, y1, y2+1), clamp(k, z1, z2+1)]
    conducts(i, j, k) = cond === nothing || @inbounds cond[i, j, k]
    σ(i, j, k) = mats[mat(i, j, k)].σn
    hm(a, b) = a == b ? a : 2a * b / (a + b)
    n1 = Nx - 1; nj = length(jr); nk = length(kr); j0 = first(jr) - 1; k0 = first(kr) - 1
    typ = Array{Int8}(undef, n1, nj, nk); λ = zeros(n1, nj, nk)
    for k in 1:nk, j in 1:nj, i in 1:n1
        I, J, K = i + 1, j + j0, k + k0
        typ[i,j,k] = !conducts(I, J, K) ? 0 : one[mat(I, J, K)] ? 2 : 1
        λ[i,j,k] = λm[mat(I, J, K)]
    end
    link(Ia, Ja, Ka, Ib, Jb, Kb, h) = conducts(Ia, Ja, Ka) && conducts(Ib, Jb, Kb) ? hm(σ(Ia, Ja, Ka), σ(Ib, Jb, Kb)) / h^2 : 0.0
    cx = [link(i, j + j0, k + k0, i + 1, j + j0, k + k0, dx) for i in 1:n1+1, j in 1:nj, k in 1:nk]   # node i to i + 1
    cy = [link(i + 1, j + j0, k + k0, i + 1, j + 1 + j0, k + k0, dy) for i in 1:n1, j in 1:max(nj-1, 0), k in 1:nk]
    cz = [link(i + 1, j + j0, k + k0, i + 1, j + j0, k + 1 + k0, dz) for i in 1:n1, j in 1:nj, k in 1:max(nk-1, 0)]
    QPVar(typ, cx, cy, cz, λ)
end

# y = −(∇·σ∇ − λ) x on the free unknowns, with variable coefficients (QPVar): a neighbour held at 0 (one fluid, or the x ends)
# adds its link to the diagonal, a cut one nothing; the y, z faces as the mirror ghosts of _qp_apply! (factor 2 to the only
# neighbour, weight 1/2 in the inner product); a non-free unknown is decoupled (y = d x, no source)
function _qp_apply_var!(y, x, v::QPVar, d)
    t = v.typ; cx, cy, cz = v.cx, v.cy, v.cz
    n1, n2, n3 = size(x)
    @inbounds for k in 1:n3, j in 1:n2, i in 1:n1
        if t[i,j,k] != 1
            y[i,j,k] = d * x[i,j,k]; continue
        end
        a = x[i,j,k]; acc = v.λ[i,j,k] * a
        nb(c, ii, jj, kk) = t[ii,jj,kk] == 1 ? c * (a - x[ii,jj,kk]) : t[ii,jj,kk] == 2 ? c * a : 0.0
        acc += i > 1 ? nb(cx[i,j,k], i-1, j, k) : cx[1,j,k] * a          # the x ends: δ = 0
        acc += i < n1 ? nb(cx[i+1,j,k], i+1, j, k) : cx[n1+1,j,k] * a
        if n2 > 1
            if j == 1
                acc += 2 * nb(cy[i,1,k], i, 2, k)
            elseif j == n2
                acc += 2 * nb(cy[i,n2-1,k], i, n2-1, k)
            else
                acc += nb(cy[i,j-1,k], i, j-1, k) + nb(cy[i,j,k], i, j+1, k)
            end
        end
        if n3 > 1
            if k == 1
                acc += 2 * nb(cz[i,j,1], i, j, 2)
            elseif k == n3
                acc += 2 * nb(cz[i,j,n3-1], i, j, n3-1)
            else
                acc += nb(cz[i,j,k-1], i, j, k-1) + nb(cz[i,j,k], i, j, k+1)
            end
        end
        y[i,j,k] = acc
    end
    y
end

# solves (∇·σ∇ − λ) x = S on the free unknowns (S overwritten by x), preconditioned by the separable solve of the first
# material projected on them
function _qp_cg_var!(S, ps::QPSolve, v::QPVar, sep!, d)
    r, z, p, q, x, w = ps.r, ps.z, ps.p, ps.q, ps.x, ps.wt
    t = v.typ
    @inbounds for I in eachindex(S); t[I] == 1 || (S[I] = 0.0); end
    @. r = -S
    bn = sqrt(_wdot(r, r, w)); bn == 0 && (fill!(S, 0.0); ps.iters[1] = 0; return S)
    prec!(z, r) = (sep!(z, r); @inbounds(for I in eachindex(z); z[I] = t[I] == 1 ? -z[I] : 0.0; end); z)
    prec!(z, r); copyto!(x, z)
    _qp_apply_var!(q, x, v, d); @. r = r - q
    prec!(z, r); copyto!(p, z)
    rz = _wdot(r, z, w); it = 0
    while sqrt(_wdot(r, r, w)) > ps.tol * bn && it < ps.maxit
        _qp_apply_var!(q, p, v, d)
        α = rz / _wdot(p, q, w)
        @. x += α * p; @. r -= α * q
        prec!(z, r)
        rz2 = _wdot(r, z, w); β = rz2 / rz; rz = rz2
        @. p = z + β * p
        it += 1
    end
    it >= ps.maxit && @warn "qp_exchange: conjugate gradients did not converge in $(ps.maxit) iterations"
    ps.iters[1] = it
    copyto!(S, x)
end

# the separable solve of (σn∇² − λ) Z = R on the region (R, Z of the size of the unknowns; Z may be R)
function _qp_sep!(Z, R, q, T1, T2, λ)
    _tx3!(T2, R, T1, q.Wx, q.Wy, q.Wz)
    @inbounds for k in axes(T2, 3), j in axes(T2, 2), i in axes(T2, 1)
        T2[i,j,k] /= q.σn * (q.ex[i] + q.ey[j] + q.ez[k]) - λ
    end
    _tx3!(Z, T2, T1, q.Vx, q.Vy, q.Vz)
end

# B ← A transformed by Mx along x, My along y, Mz along z (A, B of the size of the unknowns; T a work array)
function _tx3!(B, A, T, Mx, My, Mz)
    n1, n2, n3 = size(A)
    mul!(reshape(T, n1, n2 * n3), Mx, reshape(A, n1, n2 * n3))
    @views for k in 1:n3; mul!(B[:, :, k], T[:, :, k], transpose(My)); end
    mul!(reshape(T, n1 * n2, n3), reshape(B, n1 * n2, n3), transpose(Mz))
    copyto!(B, T)
    B
end

# C_µ of band µ on a node of the block: the divergence of its link supercurrents g Im(Ψ* V Ψ[+e]) over the live links
# (the masks τ, as Ampère sums them; at an interior node it is Im(Ψ* g D²Ψ)), plus the Josephson and drag exchange with
# the other bands: Im(Ψ_µ* R_µ) on the nodes the band update writes, and the same on the face rows (Neumann copies),
# which carry supercurrent too (in a thin strip most of it).
@inline function _conv(G, µ, g, γ, dragν, Vx, Vy, Vz, τx, τy, τz, i, j, k, rx2, ry2, rz2)
    @inbounds begin
        P = G[µ]; p = conj(P[i,j,k]); gx, gy, gz = g          # the stiffness of band µ along x, y, z
        c = ((gx * (τx[i,j,k]*imag(p*Vx[i,j,k]*P[i+1,j,k]) + τx[i-1,j,k]*imag(p*conj(Vx[i-1,j,k])*P[i-1,j,k])) * rx2 +
              gy * (τy[i,j,k]*imag(p*Vy[i,j,k]*P[i,j+1,k]) + τy[i,j-1,k]*imag(p*conj(Vy[i,j-1,k])*P[i,j-1,k])) * ry2) +
             gz * (τz[i,j,k]*imag(p*Vz[i,j,k]*P[i,j,k+1]) + τz[i,j,k-1]*imag(p*conj(Vz[i,j,k-1])*P[i,j,k-1])) * rz2)
        for ν in 1:length(G)
            ν == µ && continue
            Q = G[ν]
            c += γ[ν] * imag(p * Q[i,j,k])
            if dragν !== nothing && dragν[µ][ν] != 0
                d = dragν[µ][ν]
                c += d * (((τx[i,j,k]*imag(p*Vx[i,j,k]*Q[i+1,j,k]) + τx[i-1,j,k]*imag(p*conj(Vx[i-1,j,k])*Q[i-1,j,k])) * rx2 +
                           (τy[i,j,k]*imag(p*Vy[i,j,k]*Q[i,j+1,k]) + τy[i,j-1,k]*imag(p*conj(Vy[i,j-1,k])*Q[i,j-1,k])) * ry2) +
                          (τz[i,j,k]*imag(p*Vz[i,j,k]*Q[i,j,k+1]) + τz[i,j,k-1]*imag(p*conj(Vz[i,j,k-1])*Q[i,j,k-1])) * rz2)
            end
        end
        c
    end
end

"""
    qp_potential!(qx, s, st)

δ = μ_π − μ_σ from the state at the start of the step (the conversion of each band from Ψ and the links of this step,
on every node of the block).
"""
function qp_potential!(qx::QPExchange, s, st)
    gr = st.grid; md = st.model
    (; Nx, x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    S = qx.S; fill!(S, 0.0)
    g = ntuple(µ -> ntuple(i -> stiffness(md.bands[µ], i), 3), 2)
    rx2 = 1 / dx^2; ry2 = 1 / dy^2; rz2 = 1 / dz^2
    τx, τy, τz = st.masks.τx, st.masks.τy, st.masks.τz
    ps = qx.ps; j0 = ps.j0 - 1; k0 = ps.k0 - 1
    mt = st.mt
    if mt === nothing
        @inbounds for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1
            c1 = _conv(s.G, 1, g[1], md.γ[1], st.dragν, s.Vx, s.Vy, s.Vz, τx, τy, τz, i, j, k, rx2, ry2, rz2)
            c2 = _conv(s.G, 2, g[2], md.γ[2], st.dragν, s.Vx, s.Vy, s.Vz, τx, τy, τz, i, j, k, rx2, ry2, rz2)
            S[i-1, j-j0, k-k0] = c2 / qx.s[2] - c1 / qx.s[1]    # S holds the unknowns x = 2..Nx at i − 1 (y, z from the region)
        end
    else
        # with materials: the link weights ρ of each band (τ included) and the Josephson row of the node's material
        ρ1, ρ2 = mt.ρ[1], mt.ρ[2]
        @inbounds for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1
            m = mt.models[mt.idx[i,j,k]]
            c1 = _conv(s.G, 1, g[1], m.γ[1], st.dragν, s.Vx, s.Vy, s.Vz, ρ1[1], ρ1[2], ρ1[3], i, j, k, rx2, ry2, rz2)
            c2 = _conv(s.G, 2, g[2], m.γ[2], st.dragν, s.Vx, s.Vy, s.Vz, ρ2[1], ρ2[2], ρ2[3], i, j, k, rx2, ry2, rz2)
            S[i-1, j-j0, k-k0] = c2 / qx.s[2] - c1 / qx.s[1]
        end
    end
    if qx.var !== nothing
        d = qx.σn * (2 / dx^2 + 2 / dy^2 + 2 / dz^2) + qx.λ0
        _qp_cg_var!(S, ps, qx.var, (Z, R) -> _qp_sep!(Z, R, qx, qx.T1, qx.T2, max(qx.λ0, 1e-12 * qx.σn)), d)
    elseif ps.cond === nothing
        _qp_sep!(S, S, qx, qx.T1, qx.T2, qx.λ0)          # to the modes, divided, back
    else
        _qp_cg!(S, ps, qx.σn, max(qx.λ0, 1e-12 * qx.σn), dx, dy, dz, (Z, R) -> _qp_sep!(Z, R, qx, qx.T1, qx.T2, qx.λ0))
    end
    δ = qx.δ
    @inbounds for k in axes(S, 3), j in axes(S, 2)
        δ[1, j+j0, k+k0] = 0.0; δ[Nx+1, j+j0, k+k0] = 0.0
        for i in 2:Nx; δ[i, j+j0, k+k0] = S[i-1, j, k]; end
    end
    qx
end

"""
    qp_joule(qx, i, j, k, gr) -> Float64

The Joule heat of the two fluids beyond σn|E|² on node (i, j, k): σn s_σ s_π|∇δ|² + G δ² (σn Σ_µ s_µ|∇μ_µ|² + Σ G_µν(μ_µ − μ_ν)²
with N bands), from the potentials of this step; |∇·|² as the heat model takes |E|² (the mean over the two links of each axis).
"""
function qp_joule(qx::QPExchange, i, j, k, gr)
    δ = qx.δ
    @inbounds q = 0.5 * ((δ[i+1,j,k] - δ[i,j,k])^2 + (δ[i,j,k] - δ[i-1,j,k])^2) / gr.dx^2 +
                  0.5 * ((δ[i,j+1,k] - δ[i,j,k])^2 + (δ[i,j,k] - δ[i,j-1,k])^2) / gr.dy^2 +
                  0.5 * ((δ[i,j,k+1] - δ[i,j,k])^2 + (δ[i,j,k] - δ[i,j,k-1])^2) / gr.dz^2
    @inbounds qx.σn * qx.s[1] * qx.s[2] * q + qx.G * δ[i,j,k]^2
end

"""Each band turns by e^{−iμ_µ dt} on the nodes the band update writes (μ_σ = −s_π δ, μ_π = s_σ δ)."""
function qp_turn!(qx::QPExchange, s, st)
    gr = st.grid; dt = st.dt
    (; x1, x2, y1, y2, z1, z2) = gr
    m = (-qx.s[2], qx.s[1])
    for µ in 1:2
        F = s.F[µ]; c = m[µ] * dt
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            F[i,j,k] *= cis(-c * qx.δ[i,j,k])
        end
    end
    s
end

"""The quasiparticle potentials (μ_σ, μ_π) on the nodes, from the last step's δ."""
qp_mu(qx::QPExchange) = (-qx.s[2] .* qx.δ, qx.s[1] .* qx.δ)

# ---- N bands ---------------------------------------------------------------------------------------------------------
#  With N bands each band µ has its normal fluid s_µσn(−∇μ_µ − ∂ₜA) and the bands exchange with the symmetric conductances
#  G_µν = χ_µν_µν (ν_µν the rate µ → ν per t_GL; detailed balance χ_µν_µν = χ_νν_νµ).  With L the Laplacian of the exchange
#  graph (L_µµ = Σ_ν G_µν, L_µν = −G_µν) and S = diag(s), in the gauge Σ_µ s_µμ_µ = 0 (Ampère unchanged):
#      σn ∇²μ − S⁻¹L μ = S⁻¹C − 𝟙 ΣC.
#  S⁻¹L is similar to the symmetric S^{-1/2} L S^{-1/2} = QΛQᵀ: in the modes m = Qᵀ S^{1/2} μ the system is N Helmholtz
#  problems σn∇²m_k − λ_k m_k = r_k, each solved as the two-band one.  The constant mode (λ = 0) has no source (sᵀ of the
#  right side is ΣC − ΣC = 0), so it stays 0.  With N = 2 it is the δ equation above.

struct QPExchangeN{N}
    ν::Matrix{Float64}             # rates µ → ν [1/t_GL]
    s::NTuple{N,Float64}           # shares of σn
    P::Matrix{Float64}             # μ = P m (P = S^{-1/2} Q)
    Pinv::Matrix{Float64}          # m = Pinv μ (Qᵀ S^{1/2})
    λ::Vector{Float64}             # the eigenvalues of S⁻¹L
    σn::Float64
    Vx::Matrix{Float64}; Wx::Matrix{Float64}; ex::Vector{Float64}
    Vy::Matrix{Float64}; Wy::Matrix{Float64}; ey::Vector{Float64}
    Vz::Matrix{Float64}; Wz::Matrix{Float64}; ez::Vector{Float64}
    μ::NTuple{N,Array{Float64,3}}  # the potentials on the nodes
    R::NTuple{N,Array{Float64,3}}  # right sides
    M::NTuple{N,Array{Float64,3}}  # the modes
    T1::Array{Float64,3}; T2::Array{Float64,3}
    Gm::Matrix{Float64}            # the conductances G_µν
    ps::QPSolve
end

"""
    QPExchangeN(md, gr; ν, split=nothing, conductor=false, cond=nothing)

N normal fluids (any N ≥ 2): `ν` the matrix of exchange rates µ → ν (zero diagonal, χ_µν_µν = χ_νν_νµ), `split` the shares of
σn (by default the dirty ones, ∝ g_µΓ_µ/β_µ).  The two-band `QPExchange` is the same model; this one serves N ≥ 3.
`conductor`, `cond`: as `QPExchange` (the insulating padding, a sculpted sample).
"""
function QPExchangeN(md::Model{N}, gr::Grid; ν::AbstractMatrix, split=nothing, conductor::Bool=false, cond=nothing) where N
    size(ν) == (N, N) || error("qp_exchange: an $N×$N matrix of rates for $N bands")
    all(ν[µ, µ] == 0 for µ in 1:N) && all(>=(0), ν) || error("qp_exchange: rates >= 0 with a zero diagonal")
    b = md.bands
    w = split === nothing ? ntuple(µ -> stiffness(b[µ]) * b[µ].Γ / b[µ].β, N) : Float64.(Tuple(split))
    length(w) == N && all(>(0), w) || error("qp_split: $N positive shares")
    s = ntuple(µ -> w[µ] / sum(w), N)
    χ = [s[µ] * md.σn * b[µ].Γ / stiffness(b[µ]) for µ in 1:N]
    G = [χ[µ] * ν[µ, n] for µ in 1:N, n in 1:N]
    isapprox(G, transpose(G); rtol=1e-10, atol=1e-14) ||
        error("qp_exchange: the rates must satisfy detailed balance, χ_µν_µν = χ_νν_νµ (χ = s σn Γ/g: $(round.(χ, sigdigits=4)))")
    G = (G + transpose(G)) / 2
    L = -G; for µ in 1:N; L[µ, µ] = sum(G[µ, n] for n in 1:N if n != µ); end
    rs = sqrt.(collect(s))
    E = eigen(Symmetric(L ./ (rs * rs')))                  # S^{-1/2} L S^{-1/2}
    Q = E.vectors
    (; Nx, Ny, Nz, dx, dy, dz) = gr
    jr, kr = conductor ? (gr.y1:gr.y2+1, gr.z1:gr.z2+1) : (1:Ny+1, 1:Nz+1)
    Vx, Wx, ex = _lap1d(Nx - 1, dx, true)
    Vy, Wy, ey = _lap1d(length(jr), dy, false)
    Vz, Wz, ez = _lap1d(length(kr), dz, false)
    sz = (Nx + 1, Ny + 1, Nz + 1); su = (Nx - 1, length(jr), length(kr))
    QPExchangeN{N}(Float64.(collect(ν)), s, Q ./ rs, transpose(Q) .* rs', max.(E.values, 0.0), md.σn,
                   Vx, Wx, ex, Vy, Wy, ey, Vz, Wz, ez, ntuple(_ -> zeros(sz), N), ntuple(_ -> zeros(su), N),
                   ntuple(_ -> zeros(su), N), zeros(su), zeros(su), G, QPSolve(gr, jr, kr, cond))
end

function qp_potential!(qx::QPExchangeN{N}, s, st) where N
    gr = st.grid; md = st.model
    (; Nx, x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    g = ntuple(µ -> ntuple(i -> stiffness(md.bands[µ], i), 3), N)
    rx2 = 1 / dx^2; ry2 = 1 / dy^2; rz2 = 1 / dz^2
    τx, τy, τz = st.masks.τx, st.masks.τy, st.masks.τz
    for R in qx.R; fill!(R, 0.0); end
    c = zeros(N)
    ps = qx.ps; j0 = ps.j0 - 1; k0 = ps.k0 - 1
    @inbounds for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1
        for µ in 1:N
            c[µ] = _conv(s.G, µ, g[µ], md.γ[µ], st.dragν, s.Vx, s.Vy, s.Vz, τx, τy, τz, i, j, k, rx2, ry2, rz2)
        end
        C = sum(c)
        for µ in 1:N; qx.R[µ][i-1, j-j0, k-k0] = c[µ] / qx.s[µ] - C; end
    end
    # to the modes of S⁻¹L: m_k = Σ_µ Pinv[k,µ] R_µ, then each Helmholtz
    M = qx.M
    for kk in 1:N
        fill!(M[kk], 0.0)
        for µ in 1:N; M[kk] .+= qx.Pinv[kk, µ] .* qx.R[µ]; end
    end
    for kk in 1:N
        λ = qx.λ[kk]
        if ps.cond === nothing
            _qp_sep!(M[kk], M[kk], qx, qx.T1, qx.T2, λ)
        elseif λ > 0 || maximum(abs, M[kk]) > 0
            _qp_cg!(M[kk], ps, qx.σn, max(λ, 1e-12 * qx.σn), dx, dy, dz, (Z, R) -> _qp_sep!(Z, R, qx, qx.T1, qx.T2, λ))
        end
    end
    for µ in 1:N
        pot = qx.μ[µ]; fill!(pot, 0.0)          # (not `μ`: Julia reads the micro sign µ and the Greek μ as one name)
        @inbounds for kz in axes(M[1], 3), jy in axes(M[1], 2), ix in 2:Nx
            acc = 0.0
            for kk in 1:N; acc += qx.P[µ, kk] * M[kk][ix-1, jy, kz]; end
            pot[ix, jy+j0, kz+k0] = acc
        end
    end
    qx
end

function qp_joule(qx::QPExchangeN{N}, i, j, k, gr) where N
    q = 0.0
    @inbounds for µ in 1:N
        m = qx.μ[µ]
        q += qx.s[µ] * (0.5 * ((m[i+1,j,k] - m[i,j,k])^2 + (m[i,j,k] - m[i-1,j,k])^2) / gr.dx^2 +
                        0.5 * ((m[i,j+1,k] - m[i,j,k])^2 + (m[i,j,k] - m[i,j-1,k])^2) / gr.dy^2 +
                        0.5 * ((m[i,j,k+1] - m[i,j,k])^2 + (m[i,j,k] - m[i,j,k-1])^2) / gr.dz^2)
    end
    q *= qx.σn
    @inbounds for µ in 1:N, ν in µ+1:N
        q += qx.Gm[µ, ν] * (qx.μ[µ][i,j,k] - qx.μ[ν][i,j,k])^2
    end
    q
end

function qp_turn!(qx::QPExchangeN{N}, s, st) where N
    gr = st.grid; dt = st.dt
    (; x1, x2, y1, y2, z1, z2) = gr
    for µ in 1:N
        F = s.F[µ]; pot = qx.μ[µ]
        @inbounds for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            F[i,j,k] *= cis(-dt * pot[i,j,k])
        end
    end
    s
end

qp_mu(qx::QPExchangeN) = qx.μ
