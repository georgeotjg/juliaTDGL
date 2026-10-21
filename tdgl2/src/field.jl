# ============================================================================
#  field.jl — the field equation and the blocks after the band update
#
#  Order within a step (each block reads what the previous one left):
#    supercurrent J      from G and V (links of the current step)
#    field increment     phi = theta + (dt/sigma_n)(J tau - kappa^2 curl h)   [then implicit solve]
#    links               U = exp(-i phi)
#    field               h = curl(phi)
#    field BCs           applied field and the self-field of the injected current
#    Neumann             covariant ghosts of every band
#    advance             G <- F on the block, swap theta/phi and V/U
#
#  The supercurrent is masked by the link masks and the curl-curl term is not: outside the
#  sample the field obeys sigma_n d_t A = -kappa^2 curl curl A, a normal metal of conductivity
#  sigma_n (the padding of release 1.x), carrying normal current wherever E != 0.
# ============================================================================

"""Supercurrent on every link, J = Σ_µ g_µ Im(Ψ_µ* V Ψ_µ[+e]) (+ the drag terms, if on)."""
function supercurrent!(s::State{N}, st::Stepper{N}) where N
    gr = st.grid; G = s.G; ν = st.dragν
    wx, wy, wz = st.wa                      # the stiffness along each axis (all st.w for isotropic bands)
    (; Nx, Ny, Nz) = gr
    _rows(1:1, 2:Nz, st.threads) do _, k
        @inbounds for j in 2:Ny, i in 1:Nx
            s.Jx[i,j,k] = _jlink(G, wx, ν, s.Vx[i,j,k], i, j, k, i+1, j, k)
        end
        @inbounds for j in 1:Ny, i in 2:Nx
            s.Jy[i,j,k] = _jlink(G, wy, ν, s.Vy[i,j,k], i, j, k, i, j+1, k)
        end
    end
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx
        s.Jz[i,j,k] = _jlink(G, wz, ν, s.Vz[i,j,k], i, j, k, i, j, k+1)
    end
    s
end

@inline function _jlink(G::NTuple{N,A3c}, w, ν, v, i, j, k, i2, j2, k2) where N
    @inbounds begin
        acc = w[1]*imag((conj(G[1][i,j,k])*v)*G[1][i2,j2,k2])
        for µ in 2:N
            acc += w[µ]*imag((conj(G[µ][i,j,k])*v)*G[µ][i2,j2,k2])
        end
        _jdrag(ν, G, v, i, j, k, i2, j2, k2, acc)
    end
end
@inline _jdrag(::Nothing, G, v, i, j, k, i2, j2, k2, acc) = acc
@inline function _jdrag(ν::NTuple{N,NTuple{N,Float64}}, G::NTuple{N,A3c}, v, i, j, k, i2, j2, k2, acc) where N
    @inbounds for µ in 1:N, η in µ+1:N
        c = ν[µ][η]
        c == 0.0 && continue
        acc += c*(imag((conj(G[µ][i,j,k])*v)*G[η][i2,j2,k2]) + imag((conj(G[η][i,j,k])*v)*G[µ][i2,j2,k2]))
    end
    acc
end

"""Explicit increment of the link phases: φ = θ + (dt/σn)(τ J) − (dt κ²/σn) curl h."""
function field_increment!(s::State, st::Stepper)
    (; Nx, Ny, Nz) = st.grid
    (; d1, sxy, sxz, syx, syz, szx, szy) = st.fc
    τx, τy, τz = st.masks.τx, st.masks.τy, st.masks.τz
    hx, hy, hz = s.hx, s.hy, s.hz
    @inbounds for k in 2:Nz, j in 2:Ny, i in 1:Nx
        s.φx[i,j,k] = ((s.θx[i,j,k] + (d1*τx[i,j,k])*s.Jx[i,j,k])
                       - sxy*(hz[i,j,k] - hz[i,j-1,k])) + sxz*(hy[i,j,k] - hy[i,j,k-1])
    end
    @inbounds for k in 2:Nz, j in 1:Ny, i in 2:Nx
        s.φy[i,j,k] = ((s.θy[i,j,k] + (d1*τy[i,j,k])*s.Jy[i,j,k])
                       - syz*(hx[i,j,k] - hx[i,j,k-1])) + syx*(hz[i,j,k] - hz[i-1,j,k])
    end
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx
        s.φz[i,j,k] = ((s.θz[i,j,k] + (d1*τz[i,j,k])*s.Jz[i,j,k])
                       - szx*(hy[i,j,k] - hy[i-1,j,k])) + szy*(hx[i,j,k] - hx[i,j-1,k])
    end
    s
end

"""Links of the new step, U = exp(−iφ)."""
function links!(s::State, st::Stepper)
    (; Nx, Ny, Nz) = st.grid
    _rows(1:1, 1:Nz, st.threads) do _, k
        if 2 <= k <= Nz
            @inbounds for j in 2:Ny, i in 1:Nx
                s.Ux[i,j,k] = emiphi(s.φx[i,j,k])
            end
            @inbounds for j in 1:Ny, i in 2:Nx
                s.Uy[i,j,k] = emiphi(s.φy[i,j,k])
            end
        end
        @inbounds for j in 2:Ny, i in 2:Nx
            s.Uz[i,j,k] = emiphi(s.φz[i,j,k])
        end
    end
    s
end

"""Field on the plaquettes, h = curl(φ)."""
function curl_h!(s::State, gr::Grid)
    (; Nx, Ny, Nz, dx, dy, dz) = gr
    φx, φy, φz = s.φx, s.φy, s.φz
    @inbounds for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx
        s.hx[i,j,k] = ((φz[i,j+1,k] - φz[i,j,k]) - φy[i,j,k+1] + φy[i,j,k]) / (dy*dz)
    end
    @inbounds for k in 2:Nz-1, j in 2:Ny, i in 2:Nx-1
        s.hy[i,j,k] = ((φx[i,j,k+1] - φx[i,j,k]) - φz[i+1,j,k] + φz[i,j,k]) / (dz*dx)
    end
    @inbounds for k in 2:Nz, j in 2:Ny-1, i in 2:Nx-1
        s.hz[i,j,k] = ((φy[i+1,j,k] - φy[i,j,k]) - φx[i,j+1,k] + φx[i,j,k]) / (dx*dy)
    end
    s
end

"""
    bar_field(gr) -> NamedTuple

The field of the injected current on the outer faces when it flows only through the bar of the
sample (and its contacts): the 2D Biot–Savart field of a uniform current density over the section
of the sample, h = (1/κ²)·∫ j/(2π) (−(z−z′), y−y′)/|r−r′|² dA′, per unit HJ (HJ = J d/(2κ²)).
With an insulating padding the uniform h_y = ∓HJ of the slab condition is the field of a current
sheet as wide as the domain, which no curl-free vacuum field can meet: it forces current through
the vacuum.  This one is the field of the actual circuit.  `section` (the conducting nodes of the contact face x1,
over j in y1:y2+1, k in z1:z2+1: a sculpted sample) restricts the current to the dual cells of those
nodes, clipped to the bar; J stays the current density, so the sample carries J times that area.
"""
function bar_field(gr::Grid; nq::Int=8, section=nothing)
    (; Nx, Ny, Nz, y1, y2, z1, z2, dy, dz) = gr
    ya, yb = (y1 - 1) * dy, y2 * dy; za, zb = (z1 - 1) * dz, z2 * dz
    d = (z2 - z1 + 1) * dz
    # per unit HJ: j_code = -J = -2 HJ κ²/d, and h carries 1/κ²: h = (-2/d)/(2π) ∫ (...) dA
    c = (-2 / d) / (2pi)
    ny = nq * (y2 - y1 + 1); nz = nq * (z2 - z1 + 1)
    hq = (yb - ya) / ny; kq = (zb - za) / nz
    # a quadrature point (yp, zp) belongs to the node whose dual cell holds it
    inside(yp, zp) = section === nothing ||
        section[clamp(round(Int, yp / dy) + 1, y1, y2 + 1) - y1 + 1, clamp(round(Int, zp / dz) + 1, z1, z2 + 1) - z1 + 1]
    function h(y, z)
        hy = 0.0; hz = 0.0
        for b in 1:nz, a in 1:ny
            yp = ya + (a - 0.5) * hq; zp = za + (b - 0.5) * kq
            inside(yp, zp) || continue
            r2 = (y - yp)^2 + (z - zp)^2
            hy += -(z - zp) / r2; hz += (y - yp) / r2
        end
        (c * hy * hq * kq, c * hz * hq * kq)
    end
    yc(j) = (j - 0.5) * dy; yn(j) = (j - 1) * dy; zc(k) = (k - 0.5) * dz; zn(k) = (k - 1) * dz
    (hy_z = [h(yn(j), zc(k))[1] for j in 1:Ny+1, k in (1, Nz)],      # hy on the z faces
     hy_y = [h(yn(j), zc(k))[1] for j in (1, Ny+1), k in 1:Nz],      # hy on the y faces
     hz_z = [h(yc(j), zn(k))[2] for j in 1:Ny, k in (1, Nz+1)],      # hz on the z faces
     hz_y = [h(yc(j), zn(k))[2] for j in (1, Ny), k in 1:Nz+1],      # hz on the y faces
     hz_x = [h(yc(j), zn(k))[2] for j in 1:Ny, k in 1:Nz+1],         # hz on the x faces
     hy_x = [h(yn(j), zc(k))[1] for j in 1:Ny+1, k in 1:Nz])         # hy on the x faces

end

"""
    field_bc!(s, gr, H, HJ; tilt=0)

Applied field H ẑ on the outer faces, then the self-field of the injected current: h_y = ∓HJ
on the z faces, HJ = J d/(2κ²) (Ampère for a slab of thickness d with J ∥ x), odd in J.
`tilt` adds a uniform h_y on both z faces (an in-plane field, no transport current).
"""
function field_bc!(s::State, gr::Grid, H::Float64, HJ::Float64; tilt::Float64=0.0, bar=nothing, open=nothing)
    (; Nx, Ny, Nz) = gr
    hx, hy, hz = s.hx, s.hy, s.hz
    hz[:, :, 1] .= H; hz[:, :, Nz+1] .= H
    hz[1, :, :] .= H; hz[Nx, :, :] .= H
    hz[:, 1, :] .= H; hz[:, Ny, :] .= H
    hx[1, :, :] .= 0.0; hx[Nx+1, :, :] .= 0.0
    hx[:, 1, :] .= 0.0; hx[:, Ny, :] .= 0.0
    hy[:, 1, :] .= 0.0; hy[:, Ny+1, :] .= 0.0
    if bar === nothing
        hy[:, :, 1] .= -HJ; hy[:, :, Nz] .= +HJ
    else
        for (c, k) in enumerate((1, Nz)), j in 1:Ny+1
            hy[:, j, k] .= HJ * bar.hy_z[j, c]
        end
        for (c, j) in enumerate((1, Ny+1)), k in 1:Nz
            hy[:, j, k] .= HJ * bar.hy_y[c, k]
        end
        for (c, k) in enumerate((1, Nz+1)), j in 1:Ny
            hz[:, j, k] .= H + HJ * bar.hz_z[j, c]
        end
        for (c, j) in enumerate((1, Ny)), k in 1:Nz+1
            hz[:, j, k] .= H + HJ * bar.hz_y[c, k]
        end
        for i in (1, Nx), k in 1:Nz+1, j in 1:Ny
            hz[i, j, k] = H + HJ * bar.hz_x[j, k]
        end
        # the end faces too: curl_h! never writes hy there (i = 1, Nx), and the field of the bar
        # circulates through them
        for i in (1, Nx), k in 1:Nz, j in 1:Ny+1
            hy[i, j, k] = HJ * bar.hy_x[j, k]
        end
        open === nothing || _open_add!(s, open, gr)
    end
    if tilt != 0
        hy[:, :, 1]  .+= tilt
        hy[:, :, Nz] .+= tilt
    end
    s
end

"""Covariant Neumann condition of every band on the six faces of the block."""
function neumann!(s::State{N}, st::Stepper{N}) where N
    for µ in 1:N
        ghosts!(s.F[µ], s.Ux, s.Uy, s.Uz, st.grid)
    end
    v = st.masks.vac          # a sculpted sample: the ghosts of its vacuum on the faces are vacuum too
    if v !== nothing
        for µ in 1:N
            F = s.F[µ]
            @inbounds for I in eachindex(F)
                v[I] && (F[I] = 0.0 + 0.0im)
            end
        end
    end
    s
end

"""G ← F on the block, and the swap θ ↔ φ, V ↔ U."""
function advance!(s::State{N}, gr::Grid) where N
    (; x1, x2, y1, y2, z1, z2) = gr
    for µ in 1:N
        G = s.G[µ]; F = s.F[µ]
        @inbounds for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1
            G[i,j,k] = F[i,j,k]
        end
    end
    s.θx, s.φx = s.φx, s.θx
    s.θy, s.φy = s.φy, s.θy
    s.θz, s.φz = s.φz, s.θz
    s.Vx, s.Ux = s.Ux, s.Vx
    s.Vy, s.Uy = s.Uy, s.Vy
    s.Vz, s.Uz = s.Uz, s.Vz
    s
end
