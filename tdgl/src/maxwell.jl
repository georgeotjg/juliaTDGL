# ============================================================================
#  maxwell.jl — BLOCKS [B] TO [I] OF THE KERNEL
#
#  The order of the blocks is NOT negotiable, and each reads the state the
#  previous one left:
#
#    [B] supercurrent Q         from G and V (links of the previous step)
#    [C] phi = dA/dt            from theta, Q and h        <- v14 enters here
#    [D] phase clamp            only with inject_mode='phase' (INERT today)
#    [E] links U = exp(-i phi)
#    [F] field h = curl(phi)
#    [G] field BC + self-field injection
#    [H] Neumann BC for Psi on the six SC faces
#    [I] advance dt: G<-F, theta<-phi, V<-U
#
#  --- M17, stated before anyone inherits it -------------------------------
#  In [C] the supercurrent is masked by `taux` and the curl term is NOT.
#  Outside the sample the scheme therefore solves dA/dt = -kappa^2 curl curl A:
#  THE VACUUM IS A DIFFUSIVE MEDIUM, not an insulator, and in vortex flow
#  dA/dt does not vanish there (finding B2, measured).  That is not an
#  oversight of the port: it is what the reference code integrates, and it is
#  exactly the region where Phi lives in v14.
#  Also `d1 = dt*|alpha1| = dt*(1-T)` carries a factor (1-T) on the
#  supercurrent that the printed equation does not have.  At T=0 it equals 1
#  and goes unnoticed.  `f22_supercurrent` and `T_physical` (config.jl)
#  decide it explicitly.
# ============================================================================

"""All kernel fields, preallocated.  No block allocates memory."""
mutable struct State
    G1::Array{ComplexF64,3}; G2::Array{ComplexF64,3}
    F1::Array{ComplexF64,3}; F2::Array{ComplexF64,3}
    thetax::Array{Float64,3}; thetay::Array{Float64,3}; thetaz::Array{Float64,3}
    phix::Array{Float64,3};   phiy::Array{Float64,3};   phiz::Array{Float64,3}
    Ux::Array{ComplexF64,3};  Uy::Array{ComplexF64,3};  Uz::Array{ComplexF64,3}
    Vx::Array{ComplexF64,3};  Vy::Array{ComplexF64,3};  Vz::Array{ComplexF64,3}
    Qx::Array{Float64,3};     Qy::Array{Float64,3};     Qz::Array{Float64,3}
    hx::Array{Float64,3};     hy::Array{Float64,3};     hz::Array{Float64,3}
end

"""
    initial_state(p, g, H, mode) -> State

Meissner seed.  `G1 = taupsi`, `G2 = taupsi*sqrt(a2n)`.

`taupsi` (NODES, x1:x2+1...) is used, not `taux`: the old seed was 1 on
the low-index edges and 0 on the high-index ones, i.e. not invariant under
the reflection that compares +J with -J.
"""
function initial_state(p::Params, g::Geometry, H::Float64, mode::NumMode)
    (; Nx, Ny, Nz) = p
    G1 = ComplexF64.(g.taupsi .* 1.0)
    G2 = ComplexF64.(g.taupsi .* sqrt(p.alpha2_n))
    thetax = zeros(Nx, Ny+1, Nz+1)
    thetay = zeros(Nx+1, Ny, Nz+1)
    thetaz = zeros(Nx+1, Ny+1, Nz)
    hx = zeros(Nx+1, Ny, Nz); hy = zeros(Nx, Ny+1, Nz); hz = zeros(Nx, Ny, Nz+1)
    hz[:, :, 1] .= H; hz[:, :, Nz+1] .= H
    _armar_estado(p, G1, G2, thetax, thetay, thetaz, hx, hy, hz, mode)
end

"""Rebuilds the full `State` from the saved fields."""
function _armar_estado(p::Params, G1, G2, thetax, thetay, thetaz, hx, hy, hz,
                       mode::NumMode)
    (; Nx, Ny, Nz) = p
    phix = copy(thetax); phiy = copy(thetay); phiz = copy(thetaz)
    Ux = emiphi.(Ref(mode), phix); Uy = emiphi.(Ref(mode), phiy)
    Uz = emiphi.(Ref(mode), phiz)
    State(G1, G2, copy(G1), copy(G2),
          thetax, thetay, thetaz, phix, phiy, phiz,
          Ux, Uy, Uz, copy(Ux), copy(Uy), copy(Uz),
          zeros(Nx, Ny+1, Nz+1), zeros(Nx+1, Ny, Nz+1), zeros(Nx+1, Ny+1, Nz),
          hx, hy, hz)
end

# ---------------------------------------------------------------------------
#  [B] SUPERCURRENTS (both bands, 1/mr in pi)
# ---------------------------------------------------------------------------
function block_B!(s::State, p::Params, mode::NumMode)
    drag = p.use_drag && p.nu_drag != 0.0
    if p.hilos
        Threads.@threads for k in 2:p.Nz
            drag ? _B_plano_drag!(s, p, mode, k) : _B_plano!(s, p, mode, k)
        end
    else
        for k in 2:p.Nz
            drag ? _B_plano_drag!(s, p, mode, k) : _B_plano!(s, p, mode, k)
        end
    end
    drag ? _B_qz_drag!(s, p, mode) : _B_qz!(s, p, mode)
    s
end

@inline function _B_plano!(s::State, p::Params, mode::NumMode, k::Int)
    (; Nx, Ny, mr) = p
    imr = 1/mr
    G1, G2, Vx, Vy = s.G1, s.G2, s.Vx, s.Vy
    @inbounds for j in 2:Ny, i in 1:Nx
        s.Qx[i,j,k] = imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vx[i,j,k]), G1[i+1,j,k])) +
                 imr*imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vx[i,j,k]), G2[i+1,j,k]))
    end
    @inbounds for j in 1:Ny, i in 2:Nx
        s.Qy[i,j,k] = imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vy[i,j,k]), G1[i,j+1,k])) +
                 imr*imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vy[i,j,k]), G2[i,j+1,k]))
    end
    nothing
end

@inline function _B_qz!(s::State, p::Params, mode::NumMode)
    (; Nx, Ny, Nz, mr) = p
    imr = 1/mr
    G1, G2, Vz = s.G1, s.G2, s.Vz
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx
        s.Qz[i,j,k] = imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vz[i,j,k]), G1[i,j,k+1])) +
                 imr*imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vz[i,j,k]), G2[i,j,k+1]))
    end
    nothing
end

# σ–π cross term on the link: ν Im[Ψσ* U Ψπ_next + Ψπ* U Ψσ_next].
# 1/mr goes ONLY into the π band's own j_s. The mixed term carries the same
# ν (symmetric functional [[1,ν],[ν,1/mr]]). _B_plano! is untouched (OFF = v14.1).
@inline function _B_plano_drag!(s::State, p::Params, mode::NumMode, k::Int)
    (; Nx, Ny, mr) = p
    imr = 1/mr
    ν = p.nu_drag
    G1, G2, Vx, Vy = s.G1, s.G2, s.Vx, s.Vy
    @inbounds for j in 2:Ny, i in 1:Nx
        s.Qx[i,j,k] = imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vx[i,j,k]), G1[i+1,j,k])) +
                 imr*imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vx[i,j,k]), G2[i+1,j,k])) +
                 ν*(imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vx[i,j,k]), G2[i+1,j,k])) +
                    imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vx[i,j,k]), G1[i+1,j,k])))
    end
    @inbounds for j in 1:Ny, i in 2:Nx
        s.Qy[i,j,k] = imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vy[i,j,k]), G1[i,j+1,k])) +
                 imr*imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vy[i,j,k]), G2[i,j+1,k])) +
                 ν*(imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vy[i,j,k]), G2[i,j+1,k])) +
                    imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vy[i,j,k]), G1[i,j+1,k])))
    end
    nothing
end

@inline function _B_qz_drag!(s::State, p::Params, mode::NumMode)
    (; Nx, Ny, Nz, mr) = p
    imr = 1/mr
    ν = p.nu_drag
    G1, G2, Vz = s.G1, s.G2, s.Vz
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx
        s.Qz[i,j,k] = imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vz[i,j,k]), G1[i,j,k+1])) +
                 imr*imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vz[i,j,k]), G2[i,j,k+1])) +
                 ν*(imag(cmul(mode, cmul(mode, conj(G1[i,j,k]), Vz[i,j,k]), G2[i,j,k+1])) +
                    imag(cmul(mode, cmul(mode, conj(G2[i,j,k]), Vz[i,j,k]), G1[i,j,k+1])))
    end
    nothing
end

# ---------------------------------------------------------------------------
#  [C] VECTOR POTENTIAL  phi = dA/dt
#  v14 does NOT modify this block: it leaves the TENTATIVE increment here and
#  projects it afterwards in `block_C2!` (src/scalar_potential.jl).  The split is not
#  cosmetic: the projection needs the full increment before correcting it,
#  and putting -grad(Phi) inside this loop would require knowing Phi before
#  computing its own source.
# ---------------------------------------------------------------------------
function bloque_C!(s::State, p::Params, g::Geometry)
    (; Nx, Ny, Nz, d1, sxy, sxz, syx, syz, szx, szy) = p
    taux, tauy, tauz = g.taux, g.tauy, g.tauz
    hx, hy, hz = s.hx, s.hy, s.hz
    @inbounds for k in 2:Nz, j in 2:Ny, i in 1:Nx
        v = ((s.thetax[i,j,k] + (d1*taux[i,j,k])*s.Qx[i,j,k])
             - sxy*(hz[i,j,k] - hz[i,j-1,k])) + sxz*(hy[i,j,k] - hy[i,j,k-1])
        s.phix[i,j,k] = v
    end
    @inbounds for k in 2:Nz, j in 1:Ny, i in 2:Nx
        v = ((s.thetay[i,j,k] + (d1*tauy[i,j,k])*s.Qy[i,j,k])
             - syz*(hx[i,j,k] - hx[i,j,k-1])) + syx*(hz[i,j,k] - hz[i-1,j,k])
        s.phiy[i,j,k] = v
    end
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx
        v = ((s.thetaz[i,j,k] + (d1*tauz[i,j,k])*s.Qz[i,j,k])
             - szx*(hy[i,j,k] - hy[i-1,j,k])) + szy*(hx[i,j,k] - hx[i,j-1,k])
        s.phiz[i,j,k] = v
    end
    s
end

# ---------------------------------------------------------------------------
#  [D] Transport-current BC (Dirichlet on dA), 'phase' mode
#  INERT with inject_mode='selffield'.  Kept because it reproduces the
#  v4..v10 runs to the bit (and it is mode M1 of Article I).
# ---------------------------------------------------------------------------
function bloque_D!(s::State, p::Params, g::Geometry, J_cur::Float64)
    (; x1, x2, y1, y2, z1, z2) = p
    if p.axis_y
        for k in z1:z2+1, i in x1:x2+1
            g.tauy[i,y1,k] > 0.5 && (s.phiy[i,y1,k] = J_cur)
            g.tauy[i,y2,k] > 0.5 && (s.phiy[i,y2,k] = J_cur)
        end
    else
        for k in z1:z2+1, j in y1:y2+1
            g.taux[x1,j,k] > 0.5 && (s.phix[x1,j,k] = J_cur)
            g.taux[x2,j,k] > 0.5 && (s.phix[x2,j,k] = J_cur)
        end
    end
    s
end

# ---------------------------------------------------------------------------
#  [E] LINKS  U = exp(-i phi)
# ---------------------------------------------------------------------------
#  MEASURED: this block is 48 % of the cost of a substep on a 40x24x6 mesh,
#  MORE than the Psi update (23 %).  It is counter-intuitive, hence written
#  down: the kernel is bound not by the physics but by ~22000 sincos calls
#  per substep.  There is no cheap shortcut —the link IS exp(-i phi), and
#  updating it multiplicatively would accumulate error and break bit
#  equality—, so what remains is to parallelise it.
function bloque_E!(s::State, p::Params, mode::NumMode)
    if p.hilos
        Threads.@threads for k in 1:p.Nz
            _E_plano!(s, p, mode, k)
        end
    else
        for k in 1:p.Nz
            _E_plano!(s, p, mode, k)
        end
    end
    s
end

@inline function _E_plano!(s::State, p::Params, mode::NumMode, k::Int)
    (; Nx, Ny, Nz) = p
    if 2 <= k <= Nz
        @inbounds for j in 2:Ny, i in 1:Nx
            s.Ux[i,j,k] = emiphi(mode, s.phix[i,j,k])
        end
        @inbounds for j in 1:Ny, i in 2:Nx
            s.Uy[i,j,k] = emiphi(mode, s.phiy[i,j,k])
        end
    end
    @inbounds for j in 2:Ny, i in 2:Nx
        s.Uz[i,j,k] = emiphi(mode, s.phiz[i,j,k])
    end
    nothing
end

# ---------------------------------------------------------------------------
#  [F] FIELD  h = curl(phi)
# ---------------------------------------------------------------------------
function bloque_F!(s::State, p::Params)
    (; Nx, Ny, Nz, dx, dy, dz) = p
    phix, phiy, phiz = s.phix, s.phiy, s.phiz
    @inbounds for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx
        s.hx[i,j,k] = ((phiz[i,j+1,k] - phiz[i,j,k])
                       - phiy[i,j,k+1] + phiy[i,j,k]) / (dy*dz)
    end
    @inbounds for k in 2:Nz-1, j in 2:Ny, i in 2:Nx-1
        s.hy[i,j,k] = ((phix[i,j,k+1] - phix[i,j,k])
                       - phiz[i+1,j,k] + phiz[i,j,k]) / (dz*dx)
    end
    @inbounds for k in 2:Nz, j in 2:Ny-1, i in 2:Nx-1
        s.hz[i,j,k] = ((phiy[i+1,j,k] - phiy[i,j,k])
                       - phix[i,j+1,k] + phix[i,j,k]) / (dx*dy)
    end
    s
end

# ---------------------------------------------------------------------------
#  [G] FIELD BC + SELF-FIELD INJECTION
# ---------------------------------------------------------------------------
"""
    bloque_G!(s, p, H, HJ)

Dirichlet BC of the external field, then the injection.

`HJ = inject_gain * J * d/(2*kappa_s^2)` is Ampere's law for a slab of
thickness d with J||x: the self-field is `h_y = -+ HJ` on the z faces.  The
kappa^2 comes from Maxwell as implemented here
(dA/dt = J_s - kappa^2 curl curl A  =>  curl h = J_s/kappa^2).

Unlike the phase clamp, a field does NOT cancel in the curl —it is the curl
variable— and it is ODD in J by construction.  It goes AFTER the field BCs
so that the ordering does not overwrite it.
"""
function bloque_G!(s::State, p::Params, H::Float64, HJ::Float64)
    (; Nx, Ny, Nz, z1, z2) = p
    hx, hy, hz = s.hx, s.hy, s.hz
    hz[:, :, 1] .= H; hz[:, :, Nz+1] .= H
    hz[1, :, :] .= H; hz[Nx, :, :] .= H
    hz[:, 1, :] .= H; hz[:, Ny, :] .= H
    hx[1, :, :] .= 0.0; hx[Nx+1, :, :] .= 0.0
    hx[:, 1, :] .= 0.0; hx[:, Ny, :] .= 0.0
    hy[:, 1, :] .= 0.0; hy[:, Ny+1, :] .= 0.0
    if p.explicit_h_bc                      # faces that v11 never writes
        hx[:, :, 1] .= 0.0; hx[:, :, Nz] .= 0.0
        hy[1, :, :] .= 0.0; hy[Nx, :, :] .= 0.0
        hy[:, :, 1] .= 0.0; hy[:, :, Nz] .= 0.0
    end
    if p.selffield
        if p.standoff_sample                          # do NOT use with buf>1
            hy[:, :, z1-1] .= -HJ; hy[:, :, z2+1] .= +HJ
        else
            hy[:, :, 1] .= -HJ;    hy[:, :, Nz] .= +HJ
        end
    end
    # Uniform tilt: SAME sign on both faces.
    # d(hy)/dz = 0 => no contribution to Jx = dhz/dy - dhy/dz.  Pure in-plane
    # field, no transport current.
    if p.h_tilt != 0
        hy[:, :, 1]  .+= p.h_tilt
        hy[:, :, Nz] .+= p.h_tilt
    end
    s
end

# ---------------------------------------------------------------------------
#  [H] NEUMANN BC FOR Psi ON THE SIX SC FACES
# ---------------------------------------------------------------------------
"""
    bloque_H!(s, p, mode)

On the WHOLE FACE (edges and corners included), with precedence
x -> y -> z: the last write wins on shared edges.  That precedence is
invariant under R_z(pi) (x faces swap with each other, y faces likewise) and
under the `_REF` mirror.

v8 set only the interior patch of each face and left the edges FROZEN at
their initial value —asymmetric as well: 1 below, 0 above—, feeding spurious
supercurrents.

The order of the ten assignments matters: the y faces read nodes the x faces
have just written.
"""
function bloque_H!(s::State, p::Params, mode::NumMode)
    (; x1, x2, y1, y2, z1, z2) = p
    F1, F2, Ux, Uy, Uz = s.F1, s.F2, s.Ux, s.Uy, s.Uz
    @inbounds for k in z1:z2+1, j in y1:y2+1
        u = Ux[x1,j,k]
        F1[x1,j,k] = cmul(mode, u, F1[x1+1,j,k])
        F2[x1,j,k] = cmul(mode, u, F2[x1+1,j,k])
        uc = conj(Ux[x2,j,k])
        F1[x2+1,j,k] = cmul(mode, uc, F1[x2,j,k])
        F2[x2+1,j,k] = cmul(mode, uc, F2[x2,j,k])
    end
    @inbounds for k in z1:z2+1, i in x1:x2+1
        u = Uy[i,y1,k]
        F1[i,y1,k] = cmul(mode, u, F1[i,y1+1,k])
        F2[i,y1,k] = cmul(mode, u, F2[i,y1+1,k])
        uc = conj(Uy[i,y2,k])
        F1[i,y2+1,k] = cmul(mode, uc, F1[i,y2,k])
        F2[i,y2+1,k] = cmul(mode, uc, F2[i,y2,k])
    end
    @inbounds for j in y1:y2+1, i in x1:x2+1
        u = Uz[i,j,z1]
        F1[i,j,z1] = cmul(mode, u, F1[i,j,z1+1])
        F2[i,j,z1] = cmul(mode, u, F2[i,j,z1+1])
        uc = conj(Uz[i,j,z2])
        F1[i,j,z2+1] = cmul(mode, uc, F1[i,j,z2])
        F2[i,j,z2+1] = cmul(mode, uc, F2[i,j,z2])
    end
    s
end

# ---------------------------------------------------------------------------
#  [I] ADVANCE dt
# ---------------------------------------------------------------------------
"""
    bloque_I!(s, p)

`G <- F` over the SC block, and `theta <- phi`, `V <- U` by SWAPPING
references.

The swap is legitimate because the entries no block writes are constant in
time: the borders of `phi*` are 0 (or the checkpoint value) from
initialisation and nobody touches them, and those of `U*` are
`exp(-i theta_border)`.  `test/test_kernel.jl` checks this rather than
assuming it.  With `copyto!` the result would be the same at the cost of
three copies per substep.
"""
function bloque_I!(s::State, p::Params)
    (; x1, x2, y1, y2, z1, z2) = p
    @inbounds for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1
        s.G1[i,j,k] = s.F1[i,j,k]
        s.G2[i,j,k] = s.F2[i,j,k]
    end
    s.thetax, s.phix = s.phix, s.thetax
    s.thetay, s.phiy = s.phiy, s.thetay
    s.thetaz, s.phiz = s.phiz, s.thetaz
    s.Vx, s.Ux = s.Ux, s.Vx
    s.Vy, s.Uy = s.Uy, s.Vy
    s.Vz, s.Uz = s.Uz, s.Vz
    s
end

# Heat, only with use_heat. Source = |∂t Ψσ|² + Γr |∂t Ψπ|², plus
# σn|E|² with use_joule. It is not j² (that would heat in Meissner).
# The normal channel does exist: block [C] integrates
# ∂tA = j_s − κ²∇×∇×A, i.e. σn = 1, and F decreases as
# 2(|∂tΨσ|² + Γr|∂tΨπ|² + |∂tA|²): the weight of E² is 1.
# Here, after the swap of [I], theta − phi is the ΔA (= A·dx) of the
# previous step, so E_x = Δθx/(dx dt). Only on nodes with f>0:
# the M17 vacuum also relaxes A and is not heated.
# heat_margin excludes from both sources the normal→super conversion
# next to the injection faces (contact heat: E≠0 and the phase rotates
# even in Meissner). 0 = the whole sample, as in Article VII up to 1-oct.
# Up to 1-oct (VII) use_joule did not exist: those runs do not carry it.
# In a stationary Meissner state, away from the contacts, F≈G and
# ΔA≈0: the source switches off. Next to the contacts it does not
# (see heat_margin). The cap θ=1.5 is a numerical brake: a run in which
# a node reaches it is not quoted.
function paso_heat!(s::State, p::Params, g::Geometry)
    p.use_heat || return s
    T = p.Temp
    dx2 = p.dx^2; dy2 = p.dy^2; dz2 = p.dz^2
    dt = p.dt
    γ = p.gamma_bath
    η = p.eta_J
    θb = p.T_bath
    den = 1 + dt * γ
    gr = p.dt / p.c2          # Γr
    invdt2 = 1 / (dt * dt)
    # Neumann: the rim this loop does not write cannot stay at θ=0,
    # because that is a cold wall that swallows the heat.
    @inbounds for k in p.z1:p.z2+1, j in p.y1:p.y2+1
        T[p.x1, j, k] = T[p.x1+1, j, k]
        T[p.x2+1, j, k] = T[p.x2, j, k]
    end
    @inbounds for k in p.z1:p.z2+1, i in p.x1:p.x2+1
        T[i, p.y1, k] = T[i, p.y1+1, k]
        T[i, p.y2+1, k] = T[i, p.y2, k]
    end
    @inbounds for j in p.y1:p.y2+1, i in p.x1:p.x2+1
        T[i, j, p.z1] = T[i, j, p.z1+1]
        T[i, j, p.z2+1] = T[i, j, p.z2]
    end
    nclamp = 0
    sumd = 0.0
    sumE = 0.0
    cnt = 0
    jo = p.use_joule
    fmask = g.fmask
    m = p.heat_margin
    # window of nodes at ≥ m cells from the injection faces
    lo = p.axis_y ? p.y1 + m : p.x1 + m
    hi = p.axis_y ? p.y2 + 1 - m : p.x2 + 1 - m
    cx = invdt2 / dx2; cy = invdt2 / dy2; cz = invdt2 / dz2
    @inbounds for k in p.z1+1:p.z2, j in p.y1+1:p.y2, i in p.x1+1:p.x2
        q = T[i,j,k]
        lap = (T[i+1,j,k] - 2q + T[i-1,j,k]) / dx2 +
              (T[i,j+1,k] - 2q + T[i,j-1,k]) / dy2 +
              (T[i,j,k+1] - 2q + T[i,j,k-1]) / dz2
        d1 = abs2(s.F1[i,j,k] - s.G1[i,j,k]) * invdt2
        d2 = abs2(s.F2[i,j,k] - s.G2[i,j,k]) * invdt2
        ic = p.axis_y ? j : i
        dentro = lo < ic < hi          # always true with heat_margin=0
        if !dentro
            d1 = 0.0; d2 = 0.0
        end
        sumd += d1 + gr * d2
        cnt += 1
        if jo
            # |E|² at the node: mean over the two links of each axis.
            e2 = 0.0
            if dentro && fmask[i,j,k] > 0
                ex = 0.5 * (abs2(s.thetax[i-1,j,k] - s.phix[i-1,j,k]) +
                            abs2(s.thetax[i,j,k]   - s.phix[i,j,k])) * cx
                ey = 0.5 * (abs2(s.thetay[i,j-1,k] - s.phiy[i,j-1,k]) +
                            abs2(s.thetay[i,j,k]   - s.phiy[i,j,k])) * cy
                ez = 0.5 * (abs2(s.thetaz[i,j,k-1] - s.phiz[i,j,k-1]) +
                            abs2(s.thetaz[i,j,k]   - s.phiz[i,j,k])) * cz
                e2 = ex + ey + ez
            end
            sumE += e2
            acc = q + dt * (p.D_T * lap + η * ((d1 + gr * d2) + e2))
        else
            acc = q + dt * (p.D_T * lap + η * (d1 + gr * d2))
        end
        θ = (acc + dt * γ * θb) / den
        if θ < 0
            θ = 0.0
        elseif θ > 1.5
            θ = 1.5
            nclamp += 1
        end
        T[i,j,k] = θ
    end
    p.heat_diag[1] = cnt == 0 ? 0.0 : sumd / cnt
    p.heat_diag[2] = Float64(nclamp)
    p.heat_diag[3] = cnt == 0 ? 0.0 : sumE / cnt
    nclamp > 0 && p.verbose && @printf("heat: %d nodes at the cap θ=1.5\n", nclamp)
    s
end

# ---------------------------------------------------------------------------
#  ONE FULL SUBSTEP
# ---------------------------------------------------------------------------
"""
    substep!(s, p, g, mode; H, HJ, J_cur)

Blocks [A]..[I] in order.  It is the time unit `dt` of the scheme; a CYCLE
is `lMax` of these.
"""
function substep!(s::State, p::Params, g::Geometry, mode::NumMode;
                  H::Float64, HJ::Float64, J_cur::Float64, mi=nothing)
    psi_update!(s.F1, s.F2, s.G1, s.G2, s.Vx, s.Vy, s.Vz, g.fmask, p, mode)
    (p.n_pi > 1 || p.n_sig > 1) && pi_substeps!(s, p, g.fmask, mode)
    block_B!(s, p, mode)
    p.use_heat && paso_heat!(s, p, g)
    bloque_C!(s, p, g)
    mi === nothing || implicit_curl!(s, mi)
    p.selffield || bloque_D!(s, p, g, J_cur)
    bloque_E!(s, p, mode)
    bloque_F!(s, p)
    bloque_G!(s, p, H, HJ)
    bloque_H!(s, p, mode)
    bloque_I!(s, p)
    s
end
