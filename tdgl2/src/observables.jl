# ============================================================================
#  observables.jl — what is measured.  None of this touches the dynamics.
#
#  Sums are sequential (`sumlin`), as in release 1.2, so that the logs of the two codes can be
#  compared line by line.
# ============================================================================

@inline function sumlin(v)
    s = 0.0
    @inbounds for x in v
        s += x
    end
    s
end
@inline meanlin(v) = sumlin(v) / length(v)

function _median(v::AbstractVector)
    w = sort(collect(v)); n = length(w)
    isodd(n) ? w[(n + 1) ÷ 2] : (w[n ÷ 2] + w[n ÷ 2 + 1]) / 2
end

"""Linear least-squares slope (the Vandermonde system solved with `\\`)."""
function pendiente(t::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    A = hcat(collect(float(t)), ones(length(t)))
    (A \ collect(float(y)))[1]
end

"""
    magnetization(hz, H, gr) -> Float64

M = ⟨B_z⟩ − H over the plaquettes where h_z is the curl of the link phases (x1:x2−1, y1:y2−1,
z1:z2 with one padding cell); the rest is the boundary value.
"""
function magnetization(hz, H::Float64, gr::Grid)
    (; x1, x2, y1, y2, z1, z2) = gr
    meanlin(view(hz, x1:x2-1, y1:y2-1, z1:z2)) - H
end

"""
    count_vortices(psi, Vx, Vy, hz, f, gr, kc) -> (Nv, nmap, Nv_loop, loop_ok)

Vorticity of one band from the phase winding, plaquette by plaquette, in the plane z = kc: the
gauge-invariant link phases arg(Ψᵢ* U Ψⱼ) around the plaquette plus its flux, over 2π.  The mask
is geometric (f > 0), not by amplitude (a core plaquette has small |Ψ| at its four nodes).
`Nv_loop`, the circulation around the perimeter, must equal the sum (a giant vortex inside one
plaquette would alias); `loop_ok` says whether the perimeter runs through material.
"""
function count_vortices(psi, Vx, Vy, hz, f, gr::Grid, kc::Int)
    (; x1, x2, y1, y2, dx, dy) = gr
    P  = view(psi, x1:x2+1, y1:y2+1, kc)
    Ux = view(Vx,  x1:x2,   y1:y2+1, kc)
    Uy = view(Vy,  x1:x2+1, y1:y2,   kc)
    nx, ny = size(P)
    Dx = [angle((conj(P[i,j])*Ux[i,j])*P[i+1,j]) for i in 1:nx-1, j in 1:ny]
    Dy = [angle((conj(P[i,j])*Uy[i,j])*P[i,j+1]) for i in 1:nx,   j in 1:ny-1]
    Fm  = view(f, x1:x2+1, y1:y2+1, kc)
    mat = [Fm[i,j] > 1e-6 && abs(P[i,j]) > 1e-12 for i in 1:nx, j in 1:ny]
    nmap = zeros(Int, nx-1, ny-1)
    for j in 1:ny-1, i in 1:nx-1
        circ = Dx[i,j] + Dy[i+1,j] - Dx[i,j+1] - Dy[i,j]
        Phi_pl = hz[x1-1+i, y1-1+j, kc]*dx*dy
        bueno = mat[i,j] && mat[i+1,j] && mat[i,j+1] && mat[i+1,j+1]
        w = (circ + Phi_pl)/(2pi)
        nmap[i,j] = (bueno && isfinite(w)) ? oround(w) : 0
    end
    Nv = sum(nmap)
    loop = sumlin(@view Dx[:,1]) + sumlin(@view Dy[end,:]) -
           sumlin(@view Dx[:,end]) - sumlin(@view Dy[1,:])
    Phi_enc = sumlin(view(hz, x1:x2, y1:y2, kc))*dx*dy
    wl = (loop + Phi_enc)/(2pi)
    Nv_loop = isfinite(wl) ? oround(wl) : 0
    per = vcat(mat[:,1], mat[end,:], mat[:,end], mat[1,:])
    (Nv, nmap, Nv_loop, all(per))
end

"""
    band_currents(s, st; i=round(Nx/2)) -> NTuple{N,Float64}

Supercurrent of each band through the plane of x links at index `i`, summed over the section
of the sample (units J₀ξ₁²): g_µ Im(Ψ_µ* V Ψ_µ[+x])/dx dy dz (the link product carries J_s dx).
Their shares are the band partition of the current.
"""
function band_currents(s::State{N}, st::Stepper{N}; i::Int=oround(st.grid.Nx / 2)) where N
    gr = st.grid; w = st.w
    ntuple(N) do µ
        G = s.G[µ]; acc = 0.0
        @inbounds for k in gr.z1:gr.z2+1, j in gr.y1:gr.y2+1
            acc += w[µ]*imag((conj(G[i,j,k])*s.Vx[i,j,k])*G[i+1,j,k])
        end
        acc / gr.dx * gr.dy * gr.dz
    end
end

"""Total current and current through the sample across the mid plane, from the curl of h."""
function midplane_current(s::State, st::Stepper)
    gr = st.grid
    (; Nx, Ny, Nz, y1, y2, z1, z2, dy, dz) = gr
    ks2 = st.model.κ^2
    ii = oround(Nx/2)
    Itot = 0.0; Isam = 0.0
    @inbounds for kk in 2:Nz, jj in 2:Ny
        jx = ks2*((s.hz[ii,jj,kk] - s.hz[ii,jj-1,kk])/dy - (s.hy[ii,jj,kk] - s.hy[ii,jj,kk-1])/dz)
        Itot += jx
        (y1 <= jj <= y2+1 && z1 <= kk <= z2+1) && (Isam += jx)
    end
    (Itot*dy*dz, Isam*dy*dz)
end

# ---------------------------------------------------------------------------
#  Critical currents and efficiency
# ---------------------------------------------------------------------------

"""
    bracket_Ic(J, V, tint) -> NamedTuple

The critical current as a bracket, with the verdict rule: a rung integrated for `tint` at
voltage `V` carries |V| tint/2π vortex crossings per measuring line; it dissipates with at least
10, is silent with fewer than 1, undecided in between (re-measure it on a doubled window).  The
bracket is (last silent rung, first dissipating rung) of the ascending ladder.  `ok` when both
ends are measured and no undecided rung lies between them.  With `conv` (the closing flags of the ladder), a rung
closed 'N' (a sample gone normal: no phase to cross, so no crossings) counts as dissipating.
"""
function bracket_Ic(J::AbstractVector, V::AbstractVector, tint::AbstractVector; conv=nothing)
    idx = [q for q in eachindex(J) if abs(J[q]) > 1e-12]
    Ja = [abs(float(J[q])) for q in idx]
    cr = [abs(V[q]) * tint[q] / (2pi) for q in idx]
    cls = [(conv !== nothing && conv[idx[k]] == 'N') || cr[k] >= 10 ? 'D' : cr[k] < 1 ? 'S' : 'U' for k in eachindex(cr)]
    d = findfirst(==('D'), cls)
    hi = d === nothing ? Inf : Ja[d]
    below = d === nothing ? eachindex(Ja) : (1:d-1)
    s = findlast(q -> cls[q] == 'S', collect(below))
    lo = s === nothing ? NaN : Ja[below[s]]
    und = [Ja[q] for q in below if cls[q] == 'U' && (isnan(lo) || Ja[q] > lo)]
    ok = !isnan(lo) && isfinite(hi) && isempty(und)
    (lo = lo, hi = hi, cls = cls, J = Ja, crossings = cr, undecided = und, ok = ok)
end

"""η (per cent) at the bracket centres and at the two extreme combinations of the brackets."""
function eta_bracket(bp, bn)
    (isfinite(bp.hi) && isfinite(bn.hi) && !isnan(bp.lo) && !isnan(bn.lo)) || return (NaN, NaN, NaN)
    e(a, b) = 100 * (a - b) / (a + b)
    (e((bp.lo + bp.hi) / 2, (bn.lo + bn.hi) / 2), e(bp.lo, bn.hi), e(bp.hi, bn.lo))
end

"""η = (Ic⁺ − |Ic⁻|)/(Ic⁺ + |Ic⁻|), per cent."""
function efficiency(Icp, Icn)
    (isnan(Icp) || isnan(Icn)) && return NaN
    a = abs(Icp); b = abs(Icn)
    (a + b) == 0 && return NaN
    100*(a - b)/(a + b)
end

"""
    extract_Ic(J, V, V_thr) -> Float64

Threshold estimate, kept for comparison: the first J with |V − V(0)| > V_thr at two consecutive
points, interpolated.  At low thresholds it fires on the static ramp (mode M8); quote brackets.
"""
function extract_Ic(J::AbstractVector, V::AbstractVector, V_thr::Real)
    length(J) < 3 && return NaN
    Jv = float.(collect(J)); Vv = float.(collect(V))
    iz = argmin(abs.(Jv))
    Vc = Vv .- Vv[iz]
    idx = findall(>(V_thr), abs.(Vc))
    isempty(idx) && return NaN
    for q in 1:length(idx)-1
        if idx[q+1] == idx[q] + 1
            i2 = idx[q]
            i1 = max(1, i2 - 1)
            v1 = abs(Vc[i1]); v2 = abs(Vc[i2])
            v2 == v1 && return Jv[i2]
            return Jv[i1] + (Jv[i2] - Jv[i1])*(V_thr - v1)/(v2 - v1)
        end
    end
    Jv[idx[1]]
end

"""
    symmetry_table(Jp, Vp, Jn, Vn) -> NamedTuple

Pairs the |J| of the two polarities and compares |V − V(0)|.  In the clean box (SIM) this is the
noise floor of the solver.
"""
function symmetry_table(Jp, Vp, Jn, Vn)
    vacio = (J=Float64[], Vp=Float64[], Vn=Float64[], rel=Float64[], relmed=NaN, relmax=NaN, J_atmax=NaN)
    (length(Jp) < 2 || length(Jn) < 2) && return vacio
    izp = argmin(abs.(Jp)); Vpc = abs.(Vp .- Vp[izp])
    izn = argmin(abs.(Jn)); Vnc = abs.(Vn .- Vn[izn])
    J = Float64[]; A = Float64[]; B = Float64[]; R = Float64[]
    for q in eachindex(Jp)
        abs(Jp[q]) < 1e-12 && continue
        dmin, r = findmin(abs.(abs.(Jn) .- abs(Jp[q])))
        if dmin < 1e-9
            a = Vpc[q]; b = Vnc[r]
            if (a + b) > 1e-14
                push!(J, abs(Jp[q])); push!(A, a); push!(B, b); push!(R, 2*abs(a-b)/(a+b))
            end
        end
    end
    isempty(R) && return vacio
    relmax, im = findmax(R)
    (J=J, Vp=A, Vn=B, rel=R, relmed=_median(R), relmax=relmax, J_atmax=J[im])
end

# ---------------------------------------------------------------------------
#  Discrete free energy and dissipation (the energy balance of the scheme)
# ---------------------------------------------------------------------------

"""
    free_energy(s, st) -> F

The discrete functional whose gradient the band equations are, at zero applied field:
F = Σ_int dV [−f ε_µ|Ψ_µ|² + β_µ|Ψ_µ|⁴/2] − Σ_int dV Σ_{µ<ν} 2γ_µν Re(Ψ_µ*Ψ_ν)
  + Σ_{int–int links} dV g_µ |V Ψ_µ' − Ψ_µ|²/dl² + κ² Σ_plaquettes dV h².
The unknowns are the interior nodes (x1+1:x2, y1+1:y2, z1+1:z2); the face nodes are ghosts
(UΨ_int), so an interior–ghost link carries no gradient and no current.  The links that join two
ghost nodes (along a face of the box) carry a supercurrent in Ampère (τ = 1, as in 1.x) but no
energy here: they are the non-variational part of the scheme, and `ghost_work` is their power.
"""
function free_energy(s::State{N}, st::Stepper{N}) where N
    md = st.model; gr = st.grid; f = st.masks.f
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; F = 0.0
    @inbounds for µ in 1:N
        G = s.G[µ]; ε = eps_band(md, µ); β = md.bands[µ].β
        gx, gy, gz = stiffness(md.bands[µ], 1), stiffness(md.bands[µ], 2), stiffness(md.bands[µ], 3)
        for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            a2 = abs2(G[i,j,k])
            F += dV * (-f[i,j,k] * ε * a2 + β * a2^2 / 2)
            for ν in µ+1:N
                F -= dV * 2 * md.γ[µ][ν] * real(conj(G[i,j,k]) * s.G[ν][i,j,k])
            end
            i < x2 && (F += dV * gx * abs2(s.Vx[i,j,k] * G[i+1,j,k] - G[i,j,k]) / dx^2)
            j < y2 && (F += dV * gy * abs2(s.Vy[i,j,k] * G[i,j+1,k] - G[i,j,k]) / dy^2)
            k < z2 && (F += dV * gz * abs2(s.Vz[i,j,k] * G[i,j,k+1] - G[i,j,k]) / dz^2)
        end
    end
    F + md.κ^2 * dV * (sum(abs2, s.hx) + sum(abs2, s.hy) + sum(abs2, s.hz))
end

_isghost(gr, i, j, k) = !(gr.x1 < i <= gr.x2 && gr.y1 < j <= gr.y2 && gr.z1 < k <= gr.z2)
_inbox(gr, i, j, k) = gr.x1 <= i <= gr.x2+1 && gr.y1 <= j <= gr.y2+1 && gr.z1 <= k <= gr.z2+1

"""
    ghost_links(gr) -> (x, y, z)

Boolean masks of the links that join two ghost nodes (both ends on the faces of the box).
"""
function ghost_links(gr::Grid)
    (; Nx, Ny, Nz) = gr
    gl(i, j, k, i2, j2, k2) = _inbox(gr, i, j, k) && _inbox(gr, i2, j2, k2) &&
                              _isghost(gr, i, j, k) && _isghost(gr, i2, j2, k2)
    ([gl(i, j, k, i+1, j, k) for i in 1:Nx,   j in 1:Ny+1, k in 1:Nz+1],
     [gl(i, j, k, i, j+1, k) for i in 1:Nx+1, j in 1:Ny,   k in 1:Nz+1],
     [gl(i, j, k, i, j, k+1) for i in 1:Nx+1, j in 1:Ny+1, k in 1:Nz])
end

"""
    dissipation(s0, s1, st) -> (D, W)

Over one step from the state `s0` (a copy) to `s1`: the dissipated power
D = 2 Σ_int dV Γ_µ|ΔΨ_µ/dt|² + 2 Σ_links dV σ_l (Δθ/dl dt)², and the power of the ghost-link
supercurrents, W = 2 Σ_ghost links (dV/dl²) J_l Δθ/dt (J_l the link current of `s0`, as Ampère uses
it).  Without applied field and injection, the semi-discrete balance is dF/dt = −D + W; with the
ghost links off (τ = 0) W = 0 and F is a Lyapunov function of the scheme.
"""
function dissipation(s0::State{N}, s1::State{N}, st::Stepper{N}) where N
    md = st.model; gr = st.grid; dt = st.dt
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    dV = dx * dy * dz; D = 0.0
    @inbounds for µ in 1:N
        Γ = md.bands[µ].Γ; A = s0.G[µ]; B = s1.G[µ]
        for k in z1+1:z2, j in y1+1:y2, i in x1+1:x2
            D += 2 * dV * Γ * abs2((B[i,j,k] - A[i,j,k]) / dt)
        end
    end
    σl = st.D === nothing ? (nothing, nothing, nothing) : st.D
    gx, gy, gz = ghost_links(gr); W = 0.0
    for (θ0, θ1, J, τ, σ, gm, dl) in ((s0.θx, s1.θx, s0.Jx, st.masks.τx, σl[1], gx, dx),
                                      (s0.θy, s1.θy, s0.Jy, st.masks.τy, σl[2], gy, dy),
                                      (s0.θz, s1.θz, s0.Jz, st.masks.τz, σl[3], gz, dz))
        @inbounds for c in CartesianIndices(θ0)
            r = (θ1[c] - θ0[c]) / dt
            r == 0.0 && continue
            D += 2 * dV * md.σn * (σ === nothing ? 1.0 : σ[c]) * (r / dl)^2
            gm[c] && τ[c] > 0 && (W += 2 * (dV / dl^2) * τ[c] * J[c] * r)
        end
    end
    (D, W)
end

"""
    ghost_links_off!(mk, gr) -> mk

τ = 0 on the links that join two ghost nodes: the scheme becomes the exact gradient flow of
`free_energy` (the superconducting surface moves to the midpoint ghost–interior, dx/2 per face).
An option; release 1.x keeps τ = 1 there.
"""
function ghost_links_off!(mk::Masks, gr::Grid)
    for (τ, g) in zip((mk.τx, mk.τy, mk.τz), ghost_links(gr)), c in CartesianIndices(g)
        g[c] && (τ[c] = 0.0)
    end
    mk
end

# ---------------------------------------------------------------------------
#  A film of a rung: what the ladder only reports as numbers
# ---------------------------------------------------------------------------

"""
    film(st, s, J; H, T, every=1.0, plane=mid) -> NamedTuple

Steps the state `s` at the current density J for a time T and records, every `every` t_GL, on the
z plane `plane` of the sample: |Ψ_µ|² of every band, the vortex map of every band (winding of the
gauge-invariant phase per plaquette, `count_vortices`), and the Josephson voltage of band 1 between
the planes of the voltmeter (mean phase rate over the lines, as the ladder measures it).
Returns `psi2[µ]` (nx, ny, frames), `vort[µ]` (nx−1, ny−1, frames), `t`, `V`, `f` (the mask).
"""
function film(st::Stepper{N}, s::State{N}, J::Real; H::Real, T::Real, every::Real=1.0,
              plane::Int=(st.grid.z1 + st.grid.z2) ÷ 2, delta_v::Int=8) where N
    gr = st.grid; mk = st.masks; HJ = selffield(st, J)
    (; x1, x2, y1, y2, z1, z2) = gr
    ne = max(1, round(Int, every / st.dt)); nf = max(1, round(Int, T / (ne * st.dt)))
    nx, ny = x2 - x1 + 2, y2 - y1 + 2
    psi2 = [zeros(nx, ny, nf) for _ in 1:N]; vort = [zeros(Int, nx - 1, ny - 1, nf) for _ in 1:N]
    t = zeros(nf); V = zeros(nf)
    vi = x1 + delta_v; vo = x2 - delta_v
    # the gauge-invariant phase across the voltmeter along every line (unwrapped line by line, per step)
    lines = [(j, k) for j in y1+1:y2, k in z1+1:z2 if all(mk.τx[i, j, k] > 0.5 for i in vi:vo-1)]
    phases() = [angle(conj(s.G[1][vi, j, k]) * prod(s.Vx[i, j, k] for i in vi:vo-1) * s.G[1][vo, j, k]) for (j, k) in lines]
    pp = phases()
    for q in 1:nf
        unw = 0.0
        for _ in 1:ne
            step!(s, st; H=Float64(H), HJ=HJ)
            p = phases(); d = p .- pp; d .-= 2pi .* round.(d ./ 2pi); unw += sum(d) / length(d); pp = p
        end
        t[q] = q * ne * st.dt; V[q] = -unw / (ne * st.dt)
        for µ in 1:N
            psi2[µ][:, :, q] .= abs2.(@view s.G[µ][x1:x2+1, y1:y2+1, plane])
            vort[µ][:, :, q] .= count_vortices(s.G[µ], s.Vx, s.Vy, s.hz, mk.f, gr, plane)[2]
        end
    end
    (psi2 = psi2, vort = vort, t = t, V = V, f = mk.f[x1:x2+1, y1:y2+1, plane])
end
