# ============================================================================
#  observables.jl — WHAT IS MEASURED.  NONE OF THIS TOUCHES THE DYNAMICS.
#
#  A note on sums.  Julia's `sum` is PAIRWISE (more accurate); Octave's is
#  sequential.  Both are correct and differ by ~1e-18, but a log compared line
#  by line against the .m is more useful if it sums the same way, and none of
#  these quantities enters the evolution.  Hence `sumlin`.  The kernel uses
#  none of this, so the acceptance test does not depend on this choice.
# ============================================================================

"""Sequential sum, as in Octave (Julia's `sum` is pairwise)."""
@inline function sumlin(v)
    s = 0.0
    @inbounds for x in v
        s += x
    end
    s
end
@inline meanlin(v) = sumlin(v) / length(v)

"""
    magnetization(hz, H, p) -> Float64

`M = <B_z>_SC - H`.  Averages hz ONLY where hz is really computed as
curl(phi): block [F] writes `hz(2:Nx-1, 2:Ny-1, 2:Nz)`, which with buf=1 is
exactly `(x1:x2-1, y1:y2-1, z1:z2)`.  Everything else is overwritten by
block [G] with the BC value H, and including it would dilute M towards zero.

`fix_M_window` uses instead a window SYMMETRIC under the mirror;
it changes the value of M, not the dynamics.
"""
function magnetization(hz, H::Float64, p::Params)
    (; x1, x2, y1, y2, z1, z2) = p
    B = p.fix_M_window ? view(hz, x1+1:x2, y1+1:y2, z1:z2+1) :
                         view(hz, x1:x2-1, y1:y2-1, z1:z2)
    meanlin(B) - H
end

"""
    count_vortices(psi, Vx, Vy, hz, fmask, p, kc) -> (Nv, nmap, Nv_loop, loop_ok)

Vorticity from the PHASE WINDING, plaquette by plaquette, in the plane z=kc.

For each plaquette the GAUGE-INVARIANT phase difference of the four links of
the contour is summed,

    D_link = arg( conj(psi_i) * U_link * psi_j )

with the SAME link convention as the kernel supercurrent, so the sign is
consistent with the dynamics.  `angle` already returns the branch (-pi,pi],
which is what fluxoid quantisation requires.

**The flux is ADDED per plaquette.**  The circulation is `2*pi*n - Phi`, not
`2*pi*n`: the product of the contour links carries `exp(-i Phi_enc)`.  Per
plaquette Phi/2pi ~ 3e-3 and `round` gave the right integer anyway, but over
the PERIMETER it is not negligible and leaving it out gave `Nv_loop = -1` in
a pure Meissner state, i.e. a false warning at every point of every run.

**The mask is GEOMETRIC, not by amplitude.**  Masking by `|psi| > 0.20` —as
the old version did— removes exactly what is to be counted: in a GL core
`|psi| ~ tanh(r/xi)`, and with dx=0.25 the four nodes of the core plaquette
are at r=0.177 xi, i.e. |psi|=0.175 < 0.20.  The vortex plaquette was ALWAYS
excluded and Nv came out 0 even with vortices present: every "Nv=0/0" before
v10 was that.
"""
function count_vortices(psi, Vx, Vy, hz, fmask, p::Params, kc::Int)
    (; x1, x2, y1, y2, dx, dy) = p
    P  = view(psi, x1:x2+1, y1:y2+1, kc)
    Ux = view(Vx,  x1:x2,   y1:y2+1, kc)
    Uy = view(Vy,  x1:x2+1, y1:y2,   kc)
    nx, ny = size(P)                     # nodes

    Dx = [angle((conj(P[i,j])*Ux[i,j])*P[i+1,j]) for i in 1:nx-1, j in 1:ny]
    Dy = [angle((conj(P[i,j])*Uy[i,j])*P[i,j+1]) for i in 1:nx,   j in 1:ny-1]

    Fm  = view(fmask, x1:x2+1, y1:y2+1, kc)
    mat = [Fm[i,j] > 1e-6 && abs(P[i,j]) > 1e-12 for i in 1:nx, j in 1:ny]

    nmap = zeros(Int, nx-1, ny-1)
    for j in 1:ny-1, i in 1:nx-1
        circ = Dx[i,j] + Dy[i+1,j] - Dx[i,j+1] - Dy[i,j]
        Phi_pl = hz[x1-1+i, y1-1+j, kc]*dx*dy
        bueno = mat[i,j] && mat[i+1,j] && mat[i,j+1] && mat[i+1,j+1]
        w = (circ + Phi_pl)/(2pi)
        # A diverged state arrives here with NaN, and `oround` would throw
        # InexactError from inside the vortex counter: an opaque error, in the
        # wrong place, about a symptom rather than the cause.  Return 0 and
        # let the sweep guard report what happened.
        nmap[i,j] = (bueno && isfinite(w)) ? oround(w) : 0
    end
    Nv = sum(nmap)

    # --- cross-check: circulation over the full PERIMETER ------------------
    # The sum of plaquette windings over a simply connected region MUST equal
    # the circulation along its boundary.  A difference reveals the only real
    # failure mode of the method: ALIASING.  Each `angle` lives in (-pi,pi],
    # so a plaquette registers only |n|<=1; a GIANT VORTEX whose core fitted
    # in one plaquette would read as 0.  Narrow mesoscopic samples are exactly
    # where such states form.
    loop = sumlin(@view Dx[:,1]) + sumlin(@view Dy[end,:]) -
           sumlin(@view Dx[:,end]) - sumlin(@view Dy[1,:])
    Phi_enc = sumlin(view(hz, x1:x2, y1:y2, kc))*dx*dy
    wl = (loop + Phi_enc)/(2pi)
    Nv_loop = isfinite(wl) ? oround(wl) : 0

    # loop_ok: the check is valid only if the whole perimeter runs through
    # material.  WEDGE cuts its notch into face y1, which the perimeter
    # crosses; there the phase is undefined and Nv_loop is meaningless, not a
    # symptom of aliasing.
    per = vcat(mat[:,1], mat[end,:], mat[:,end], mat[1,:])
    (Nv, nmap, Nv_loop, all(per))
end

"""
    pendiente(t, y) -> Float64

Linear least-squares fit; it is `polyfit(t,y,1)(1)` of the .m.  The same
Vandermonde matrix is built and solved with `\\`, as in Octave.
"""
function pendiente(t::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    A = hcat(collect(float(t)), ones(length(t)))
    (A \ collect(float(y)))[1]
end

"""
    extract_Ic(J, V, V_thr; strict=false) -> Float64

Ic = first J with `|V_corr| > V_thr` at TWO consecutive points, with
linear interpolation between the last below and the first above.
`V_corr = V - V(J~=0)`: the run provides its own baseline, which is the noise
floor of the voltmeter measured in the same run.  At a threshold this low it
fires on the static ramp in the Meissner state (mode M8).
"""
function extract_Ic(J::AbstractVector, V::AbstractVector, V_thr::Real;
                    strict::Bool=false)
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
    strict ? NaN : Jv[idx[1]]
end

"""
    efficiency(Icp, Icn) -> Float64

`eta = (Ic+ - |Ic-|) / (Ic+ + |Ic-|) * 100` in per cent, as in the Octave code
and the paper (up to 3-oct-2026 the port carried a spurious factor 2 here).  It is the
difference of two nearly equal numbers: the voltmeter noise enters directly,
which is why the voltage of each point is averaged over a window rather than
read at one instant.
"""
function efficiency(Icp, Icn)
    (isnan(Icp) || isnan(Icn)) && return NaN
    a = abs(Icp); b = abs(Icn)
    (a + b) == 0 && return NaN
    100*(a - b)/(a + b)
end

"""
    bracket_Ic(J, V, tint) -> NamedTuple

The critical current as a bracket, with the verdict rule of the paper: a rung
integrated for `tint` at voltage `V` carries `|V| tint / 2π` vortex crossings
per measuring line; it is dissipating with at least 10, silent with fewer
than 1, undecided in between (re-measure it on a doubled window).  The
bracket is (last silent rung, first dissipating rung) of the ascending ladder.

Fields: `lo`, `hi` (NaN/Inf when the ladder does not close them), `cls`
(one of 'S', 'U', 'D' per nonzero rung, in ladder order), `undecided` (the
|J| of undecided rungs below `hi`), `crossings`, and `ok`: true when both ends
are measured and no undecided rung lies between them.
"""
function bracket_Ic(J::AbstractVector, V::AbstractVector, tint::AbstractVector)
    idx = [q for q in eachindex(J) if abs(J[q]) > 1e-12]
    Ja = [abs(float(J[q])) for q in idx]
    cr = [abs(V[q]) * tint[q] / (2pi) for q in idx]
    cls = [c >= 10 ? 'D' : c < 1 ? 'S' : 'U' for c in cr]
    d = findfirst(==('D'), cls)
    hi = d === nothing ? Inf : Ja[d]
    below = d === nothing ? eachindex(Ja) : (1:d-1)
    s = findlast(q -> cls[q] == 'S', collect(below))
    lo = s === nothing ? NaN : Ja[below[s]]
    und = [Ja[q] for q in below if cls[q] == 'U' && (isnan(lo) || Ja[q] > lo)]
    ok = !isnan(lo) && isfinite(hi) && isempty(und)
    (lo = lo, hi = hi, cls = cls, J = Ja, crossings = cr, undecided = und, ok = ok)
end

"""
    eta_bracket(bp, bn) -> (centre, low, high)

The efficiency (per cent) at the bracket centres and at the two extreme
combinations of the brackets `bp` (positive branch) and `bn` (negative).
"""
function eta_bracket(bp, bn)
    (isfinite(bp.hi) && isfinite(bn.hi) && !isnan(bp.lo) && !isnan(bn.lo)) || return (NaN, NaN, NaN)
    e(a, b) = 100 * (a - b) / (a + b)
    (e((bp.lo + bp.hi) / 2, (bn.lo + bn.hi) / 2), e(bp.lo, bn.hi), e(bp.hi, bn.lo))
end

"""
    ic_guards(Jp, Vp, Jn, Vn, Nv_total, Icp, Icn) -> NamedTuple

The three checks the reference code applies before it reports Ic and eta;
each one, when it fires, means that the number would not measure a diode.

* `inert` (mode M1): a dissipative voltage is odd in J.  If the odd part of V,
  paired by |J| with its sign, is at rounding against the even part, the
  current is not entering the dynamics.  The even part must be nonzero, so
  that a run at H = 0 (no voltage of either parity) does not fire it.
* `no_vortex`: no vortex at any point of either polarity, in either band.
  Without vortices there is no diode mechanism; eta = 0 is the answer and an
  Ic would be a threshold crossing of a static quantity.
* `grid` (mode M9): an Ic at or below the first nonzero current of the grid
  is an interpolation between J = 0 and that point, not a threshold.

`Icp` and `Icn` are returned as NaN when any of the three fires.
"""
function ic_guards(Jp, Vp, Jn, Vn, Nv_total::Integer, Icp::Real, Icn::Real)
    V_even = 0.0; V_odd = 0.0; npair = 0
    for q in eachindex(Jp)
        abs(Jp[q]) < 1e-12 && continue
        dmin, r = findmin(abs.(abs.(Jn) .- abs(Jp[q])))
        if dmin < 1e-9
            a = Vp[q]; b = Vn[r]                  # b with its sign
            V_odd = max(V_odd, abs(a - b) / 2)
            V_even = max(V_even, abs(a + b) / 2)
            npair += 1
        end
    end
    inert = npair >= 1 && V_even > 1e-12 && V_odd <= 1e-9 * V_even
    no_vortex = Nv_total == 0
    Jnz = sort([abs(j) for j in Jp if abs(j) > 1e-12])
    J_first = isempty(Jnz) ? NaN : Jnz[1]
    grid = !(inert || no_vortex) && !isnan(J_first) &&
           ((!isnan(Icp) && Icp <= J_first) || (!isnan(Icn) && Icn <= J_first))
    ok = !(inert || no_vortex || grid)
    (Icp = ok ? float(Icp) : NaN, Icn = ok ? float(Icn) : NaN, inert, no_vortex, grid,
     V_odd_max = V_odd, V_even_max = V_even, J_first)
end

"""
    symmetry_table(Jp, Vp, Jn, Vn) -> NamedTuple

Pairs |J| points between polarities and compares `|V_corr|`.  In the SIM
control this table IS the noise floor of the solver: if the relative
asymmetry between +J and -J in a box without defects is not negligible, any
eta measured in a geometry with defects is below the noise.
"""
function symmetry_table(Jp, Vp, Jn, Vn)
    vacio = (J=Float64[], Vp=Float64[], Vn=Float64[], rel=Float64[],
             relmed=NaN, relmax=NaN, J_atmax=NaN)
    (length(Jp) < 2 || length(Jn) < 2) && return vacio
    izp = argmin(abs.(Jp)); Vpc = abs.(Vp .- Vp[izp])
    izn = argmin(abs.(Jn)); Vnc = abs.(Vn .- Vn[izn])
    J = Float64[]; A = Float64[]; B = Float64[]; R = Float64[]
    for q in eachindex(Jp)
        abs(Jp[q]) < 1e-12 && continue
        dmin, r = findmin(abs.(Jn .- abs(Jp[q])))
        if dmin < 1e-9
            a = Vpc[q]; b = Vnc[r]
            if (a + b) > 1e-14      # below this, the relative difference is noise/0
                push!(J, abs(Jp[q])); push!(A, a); push!(B, b)
                push!(R, 2*abs(a-b)/(a+b))
            end
        end
    end
    isempty(R) && return vacio
    relmax, im = findmax(R)
    (J=J, Vp=A, Vn=B, rel=R, relmed=median(R), relmax=relmax, J_atmax=J[im])
end
