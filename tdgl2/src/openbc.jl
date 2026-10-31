# ============================================================================
#  openbc.jl — the open boundary of the field (selffield_bc = :open)
#
#  The outer faces carry the field the currents of the box would have in free space, not the one of
#  a mirror.  The bar field (:bar) is the field of the nominal current spread uniformly over the
#  section of the sample; :open adds h_δ, the free-space field of what the box carries beyond it:
#      δJ = κ² ∇_h × h − J_bar        (J_bar: the uniform current the bar faces give, on the x links)
#      A_δ = G ∗ δJ / κ²,  h_δ = ∇_h × A_δ   on the plaquettes of the y and z faces.
#  G is the Green function of the 7-point lattice Laplacian (−Δ_h G = δ), so that the field imposed
#  on the face is the one of the free continuation of the same discrete scheme; the convolution is
#  Hockney's (zero padding to twice the box, FFT).  The current enters and leaves uniform through the
#  contacts (J_bar on the x faces), so δJ is a closed current inside the box: the links normal to the
#  six faces are left out of δJ, and at the fixed point δJ crosses none of them (the curl of h_δ on a
#  face is the source there, zero).  h_δ goes on all six faces: with the x faces left as mirrors
#  (a first version) the Meissner field of a box 12 ξ long still depended on the padding by 10 %;
#  with them open it does not.
#  h_δ is recomputed every `every` steps (1 t_GL by default) and ramped linearly in between; the
#  iteration contracts (spectral radius ≤ 0.67 measured in 2D at κ = 2) for any lag.
#  Two things it does not do (both inert today):
#  - the components of h normal to a face (hy on j = 1, Ny+1; hz on k = 1, Nz+1) keep the bar field
#    without h_δ: no evolving link and no observable reads them (magnetization uses z1:z2), but a flux
#    through a face computed from them would miss h_δ;
#  - right after a change of current HJ·J_bar is the new one and ∇_h × h still the old, so the first
#    recomputation sees an open uniform δJ of −ΔHJ·J_bar for one interval, which then relaxes.  With the
#    steps of the fine ladders (0.004, 0.6 % of the current) it is invisible; a jump from J = 0 kicks the
#    faces for one interval.
# ============================================================================

# --- lattice Green function of the simple cubic Laplacian (unit spacing) ------------------------
#  G(n) = ∫₀^∞ Π_i e^{−2t} I_{n_i}(2t) dt  (G(0) = 0.252731…, Watson's integral / 2)

function _gauss_legendre(n::Int)
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    E.values, 2 .* E.vectors[1, :] .^ 2
end

# e^{-x} I_n(x) for n = 0..R: (1/π)∫₀^π e^{x(cos θ − 1)} cos nθ dθ by the trapezoid rule
# (spectrally accurate: the integrand is periodic and analytic)
function _scaled_besseli!(b::Vector{Float64}, x::Float64)
    R = length(b) - 1
    M = 32 + ceil(Int, 12 * sqrt(x))
    fill!(b, 0.0)
    for m in 0:M
        θ = π * m / M
        w = (m == 0 || m == M) ? 0.5 : 1.0
        e = w * exp(x * (cos(θ) - 1))
        for n in 0:R
            b[n+1] += e * cos(n * θ)
        end
    end
    b ./= M
end

const _LGF = Dict{Int,Array{Float64,3}}()

"""
    lattice_green(R) -> G[0:R, 0:R, 0:R]

The Green function of −Δ_h on the unit simple-cubic lattice for offsets 0..R per axis (symmetric
in sign and permutation).  Beyond R, `_lgf_far` (1/4πr with its first cubic correction).
"""
function lattice_green(R::Int)
    get!(_LGF, R) do
        x, w = _gauss_legendre(48)
        edges = (0.0, 0.5, 2.0, 8.0, 32.0, 128.0, 512.0)
        T = edges[end]
        G = zeros(R + 1, R + 1, R + 1)
        b = zeros(R + 1)
        for p in 1:length(edges)-1
            a, c = edges[p], edges[p+1]
            for q in eachindex(x)
                t = (a + c) / 2 + (c - a) / 2 * x[q]; wt = (c - a) / 2 * w[q]
                _scaled_besseli!(b, 2t)
                @inbounds for k in 0:R, j in 0:R, i in 0:R
                    G[i+1, j+1, k+1] += wt * b[i+1] * b[j+1] * b[k+1]
                end
            end
        end
        # tail t > T: Π e^{-2t}I_n(2t) ≈ (4πt)^{-3/2} (1 − c/t), c = Σ(4n²−1)/16
        for k in 0:R, j in 0:R, i in 0:R
            c = ((4i^2 - 1) + (4j^2 - 1) + (4k^2 - 1)) / 16
            G[i+1, j+1, k+1] += (4pi)^-1.5 * (2 / sqrt(T) - c * (2 / 3) / T^1.5)
        end
        G
    end
end

"""1/(4πr) with the first correction of the cubic lattice (Duffin): (5Σx⁴/r⁴ − 3)/(8r²)."""
function _lgf_far(i, j, k)
    r2 = Float64(i^2 + j^2 + k^2); r = sqrt(r2)
    (1 + (5 * (i^4 + j^4 + k^4) / r2^2 - 3) / (8r2)) / (4pi * r)
end

_lgf(G, R, i, j, k) = (i = abs(i); j = abs(j); k = abs(k);
                       max(i, j, k) <= R ? G[i+1, j+1, k+1] : _lgf_far(i, j, k))

# --- the open boundary ----------------------------------------------------------------------

mutable struct OpenBC
    every::Int                               # steps between recomputations of h_δ
    count::Int                               # steps since the last one (−1: none yet)
    Ghat::NTuple{3,Array{ComplexF64,3}}      # transform of the padded Green function per link lattice
    buf::NTuple{3,Array{Float64,3}}          # padded δJ, then A_δ
    fwd::NTuple{3,Any}; inv::NTuple{3,Any}   # FFT plans
    θ::NTuple{3,A3r}                         # θ_δ = A_δ·dl on the links of the box
    old::NTuple{3,A3r}; new::NTuple{3,A3r}   # h_δ at the last two recomputations (h-shaped)
    Jbar::Matrix{Float64}                    # κ² curl of the x-face bar field per unit HJ, (j, k) of an x link
end

function OpenBC(gr::Grid, κ::Float64, bar; every::Int, R::Int=8)
    (; Nx, Ny, Nz, dx, dy, dz) = gr
    dx == dy == dz || error(":open needs dx = dy = dz")
    G = lattice_green(R)
    shapes = ((Nx, Ny+1, Nz+1), (Nx+1, Ny, Nz+1), (Nx+1, Ny+1, Nz))
    fac = dx^3 / κ^2
    Ghat = Vector{Array{ComplexF64,3}}(undef, 3); bufs = Vector{Array{Float64,3}}(undef, 3)
    fwd = Vector{Any}(undef, 3); inv = Vector{Any}(undef, 3)
    for (c, sz) in enumerate(shapes)
        M = 2 .* sz
        off(a, n) = a <= n ? a - 1 : a - 1 - 2n          # 1-based padded index -> signed offset
        Gm = [fac * _lgf(G, R, off(a, sz[1]), off(b, sz[2]), off(d, sz[3])) for a in 1:M[1], b in 1:M[2], d in 1:M[3]]
        bufs[c] = zeros(M)
        fwd[c] = plan_rfft(bufs[c]; flags=FFTW.ESTIMATE)
        Ghat[c] = fwd[c] * Gm
        inv[c] = plan_irfft(similar(Ghat[c]), M[1]; flags=FFTW.ESTIMATE)
    end
    hz0 = zeros(Nx, Ny, Nz+1); hy0 = zeros(Nx, Ny+1, Nz); hx0 = zeros(Nx+1, Ny, Nz)
    Jb = zeros(Ny+1, Nz+1)
    for k in 2:Nz, j in 2:Ny
        Jb[j, k] = κ^2 * ((bar.hz_x[j, k] - bar.hz_x[j-1, k]) / dy - (bar.hy_x[j, k] - bar.hy_x[j, k-1]) / dz)
    end
    OpenBC(every, -1, Tuple(Ghat), Tuple(bufs), Tuple(fwd), Tuple(inv),
           (zeros(shapes[1]), zeros(shapes[2]), zeros(shapes[3])),
           (copy(hx0), copy(hy0), copy(hz0)), (copy(hx0), copy(hy0), copy(hz0)), Jb)
end

# δJ = κ² ∇_h × h − J_bar on the links of the box (the links normal to the six faces left out)
function _delta_current!(ob::OpenBC, s::State, gr::Grid, κ2::Float64, HJ::Float64)
    (; Nx, Ny, Nz, dx, dy, dz) = gr
    hx, hy, hz = s.hx, s.hy, s.hz
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

# θ_δ = (dx³/κ²) G ∗ δJ on every link of the box (Hockney)
function _convolve!(ob::OpenBC)
    for c in 1:3
        F = ob.fwd[c] * ob.buf[c]
        F .*= ob.Ghat[c]
        A = ob.inv[c] * F
        θ = ob.θ[c]
        θ .= @view A[1:size(θ, 1), 1:size(θ, 2), 1:size(θ, 3)]
    end
    ob
end

# h_δ = ∇_h × θ_δ / area on the tangential plaquettes of the six faces, into `h`
function _face_field!(h::NTuple{3,A3r}, ob::OpenBC, gr::Grid)
    (; Nx, Ny, Nz, dx, dy, dz) = gr
    φx, φy, φz = ob.θ
    hx, hy, hz = h
    hzv(i, j, k) = ((φy[i+1,j,k] - φy[i,j,k]) - φx[i,j+1,k] + φx[i,j,k]) / (dx*dy)
    hxv(i, j, k) = ((φz[i,j+1,k] - φz[i,j,k]) - φy[i,j,k+1] + φy[i,j,k]) / (dy*dz)
    hyv(i, j, k) = ((φx[i,j,k+1] - φx[i,j,k]) - φz[i+1,j,k] + φz[i,j,k]) / (dz*dx)
    @inbounds for j in (1, Ny), k in 1:Nz+1, i in 1:Nx;  hz[i,j,k] = hzv(i, j, k); end
    @inbounds for j in (1, Ny), k in 1:Nz,   i in 1:Nx+1; hx[i,j,k] = hxv(i, j, k); end
    @inbounds for k in (1, Nz), j in 1:Ny+1, i in 1:Nx;  hy[i,j,k] = hyv(i, j, k); end
    @inbounds for k in (1, Nz), j in 1:Ny,   i in 1:Nx+1; hx[i,j,k] = hxv(i, j, k); end
    @inbounds for i in (1, Nx), k in 1:Nz+1, j in 1:Ny;  hz[i,j,k] = hzv(i, j, k); end
    @inbounds for i in (1, Nx), k in 1:Nz,   j in 1:Ny+1; hy[i,j,k] = hyv(i, j, k); end
    h
end

"""
    open_update!(ob, s, st, HJ)

Advances the clock of the open boundary one step; every `every` steps h_δ is recomputed from the
state (the current ramp value becomes the start of the next ramp).
"""
function open_update!(ob::OpenBC, s::State, gr::Grid, κ2::Float64, HJ::Float64)
    first = ob.count < 0
    ob.count += 1
    if ob.count >= ob.every || first
        for c in 1:3
            copyto!(ob.old[c], ob.new[c])           # the end of the last ramp
        end
        _delta_current!(ob, s, gr, κ2, HJ)
        _convolve!(ob)
        _face_field!(ob.new, ob, gr)
        # the first time (a new run or a resumed one) there is no ramp: the faces start at the field of
        # the state as it is, so a resumed run does not see its boundary drop to the bar field
        first && for c in 1:3; copyto!(ob.old[c], ob.new[c]); end
        ob.count = 0
    end
    ob
end

"""h_δ at the current point of the ramp, added to the six faces by `field_bc!` (after the bar field)."""
function _open_add!(s::State, ob::OpenBC, gr::Grid)
    (; Nx, Ny, Nz) = gr
    f = min((ob.count + 1) / ob.every, 1.0)
    o = ob.old; n = ob.new
    @inbounds begin
        for j in (1, Ny), k in 1:Nz+1, i in 2:Nx-1; s.hz[i,j,k] += o[3][i,j,k] + f * (n[3][i,j,k] - o[3][i,j,k]); end
        for j in (1, Ny), k in 1:Nz,   i in 2:Nx;   s.hx[i,j,k]  = o[1][i,j,k] + f * (n[1][i,j,k] - o[1][i,j,k]); end
        for k in (1, Nz), j in 1:Ny+1, i in 2:Nx-1; s.hy[i,j,k] += o[2][i,j,k] + f * (n[2][i,j,k] - o[2][i,j,k]); end
        for k in (1, Nz), j in 1:Ny,   i in 2:Nx;   s.hx[i,j,k]  = o[1][i,j,k] + f * (n[1][i,j,k] - o[1][i,j,k]); end
        for i in (1, Nx), k in 1:Nz+1, j in 1:Ny;   s.hz[i,j,k] += o[3][i,j,k] + f * (n[3][i,j,k] - o[3][i,j,k]); end
        for i in (1, Nx), k in 1:Nz,   j in 1:Ny+1; s.hy[i,j,k] += o[2][i,j,k] + f * (n[2][i,j,k] - o[2][i,j,k]); end
    end
    s
end
