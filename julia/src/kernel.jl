# ============================================================================
#  kernel.jl — BLOCK [A]: THE ORDER-PARAMETER UPDATE
#
#  Jacobi scheme: reads ONLY G (the previous step) and writes F.  The ORDER
#  OF THE SUMS per element is kept as in the .m, because the goal is not
#  "the same to machine precision" but the same bit.  Every grouping below is
#  deliberate; see the note on `dt/(dt/c2)`.
#
#  INTERIOR NODES ONLY (x1+1:x2, ...).  Face and edge nodes are set by the
#  Neumann condition on the whole face.  v8 updated x1:x2: it included the
#  low-index faces (which read vacuum nodes frozen at zero) and excluded the
#  high-index ones, so the update domain was not invariant under R_z(pi) and
#  contributed to the spurious +J/-J asymmetry.
#
#  --- WHAT THIS BLOCK REALLY INTEGRATES, AND WHY IT MATTERS (M16) ---------
#  Dividing by dt, the model THIS code integrates is
#
#      dPsi1/dt = L Psi1 + f Psi1 + g12 Psi2
#      dPsi2/dt = (1/Gr) [ (L/mr) Psi2 + (a2n f - |Psi2|^2) Psi2 + g12 Psi1 ]
#
#  i.e. Gamma_r divides the WHOLE right-hand side of the pi band: the three
#  terms come out of F, so all three must carry it.  Written this way,
#  Gamma_r IS a relaxation rate and CANNOT move any static quantity: Gamma
#  multiplies d/dt and a rate does not move a fixed point.
#
#  UP TO v13.2 THE CODE DIVIDED ONE OF THE THREE.  The kinetic term carried
#  rx/mr and the coupling dt*g12, neither scaled, and only the condensation
#  carried 1/Gamma_r.  That is not Gamma_a dPsi_a/dt = -dF/dPsi_a*: it
#  rescales alpha_2 and beta_2 TOGETHER, which leaves the uniform equilibrium
#  intact (|Psi2|^2_eq = a2n is kept, which is why nothing obvious broke) but
#  changes the balance against the gradient, i.e. the coherence length:
#
#      v13.2:  xi_pi/xi_sigma = 1.300*sqrt(Gamma_r)   Hc2_pi = 0.5917/Gamma_r
#      v13.3:  xi_pi/xi_sigma = 1.300                 Hc2_pi = 0.5917
#
#  Found by the linear eigenproblem and confirmed by run 122.  See
#  docs/HALLAZGO_G6_gamma_r.md and `hc2_pi` in params.jl.
#
#  WHY THE TEST BANK RUNS AT Gamma_r = 1.3.  At Gamma_r = 1 the two versions
#  are IDENTICAL TO THE BIT (c2 = dt), so a port that inherited the v13.2
#  scheme would pass the test anyway.  At 1.3 it does not.  The fixture does
#  not pick 1.3 for physics: it picks it because that is where the fix shows.
# ============================================================================

"""
    coef_pi(p) -> (kx, ky, kz, cpl1, cpl2)

The pi-band coefficients, computed IN THE SAME ORDER as the .m.

`gr = dt/c2` and then `dt/gr`: that is NOT `c2` in floating point, and
writing `c2` instead changes the last bit.  The .m recovers Gamma_r this way
on purpose (so no signature has to change), so the port must carry the same
rounding.
"""
@inline function coef_pi(p::Params)
    gr = p.dt / p.c2
    (p.rx/(p.mr*gr), p.ry/(p.mr*gr), p.rz/(p.mr*gr),
     p.dt*p.gamma12, (p.dt/gr)*p.gamma12)
end

"""
    psi_update!(F1,F2,G1,G2,Vx,Vy,Vz,fmask,p,mode)

Production kernel.  Fused loop over the interior nodes; arithmetic
identical, element by element, to `psi_update_vec` of the .m.
"""
function psi_update!(F1::Array{ComplexF64,3}, F2::Array{ComplexF64,3},
                     G1::Array{ComplexF64,3}, G2::Array{ComplexF64,3},
                     Vx::Array{ComplexF64,3}, Vy::Array{ComplexF64,3},
                     Vz::Array{ComplexF64,3}, fmask::Array{Float64,3},
                     p::Params, mode::NumMode)
    if p.use_heat || p.T_physical      # α per node: θ = T(r) with heating, or a fixed T_red
        if p.use_drag && p.nu_drag != 0.0
            for k in p.z1+1:p.z2
                _A_plano_drag_heat!(F1, F2, G1, G2, Vx, Vy, Vz, fmask, p, mode, k)
            end
        else
            for k in p.z1+1:p.z2
                _A_plano_heat!(F1, F2, G1, G2, Vx, Vy, Vz, fmask, p, mode, k)
            end
        end
    elseif p.use_drag && p.nu_drag != 0.0
        if p.hilos
            Threads.@threads for k in p.z1+1:p.z2
                _A_plano_drag!(F1, F2, G1, G2, Vx, Vy, Vz, fmask, p, mode, k)
            end
        else
            for k in p.z1+1:p.z2
                _A_plano_drag!(F1, F2, G1, G2, Vx, Vy, Vz, fmask, p, mode, k)
            end
        end
    elseif p.hilos
        Threads.@threads for k in p.z1+1:p.z2
            _A_plano!(F1, F2, G1, G2, Vx, Vy, Vz, fmask, p, mode, k)
        end
    else
        for k in p.z1+1:p.z2
            _A_plano!(F1, F2, G1, G2, Vx, Vy, Vz, fmask, p, mode, k)
        end
    end
    (F1, F2)
end

# One z plane of the update.  Each plane writes different cells and reads only
# G (the previous step), so distributing planes over threads gives the SAME
# result to the bit with any number of threads.  That is why it can be
# parallelised without giving up the acceptance test.
@inline function _A_plano!(F1, F2, G1, G2, Vx, Vy, Vz, fmask,
                           p::Params, mode::NumMode, k::Int)
    (; rx, ry, rz, c1, c2, dt, gamma12, mr, alpha2_n, hard) = _kernelvars(p)
    kx2, ky2, kz2, cpl1, cpl2 = coef_pi(p)
    a2n = alpha2_n
    @inbounds for j in p.y1+1:p.y2, i in p.x1+1:p.x2
        g1 = G1[i,j,k]; g2 = G2[i,j,k]; fm = fmask[i,j,k]
        if hard && fm == 0.0
            F1[i,j,k] = 0.0 + 0.0im; F2[i,j,k] = 0.0 + 0.0im
            continue
        end
        vx = Vx[i,j,k]; vy = Vy[i,j,k]; vz = Vz[i,j,k]
        vxm = conj(Vx[i-1,j,k]); vym = conj(Vy[i,j-1,k]); vzm = conj(Vz[i,j,k-1])

        tx1 = rx*((cmul(mode,vx,G1[i+1,j,k]) - 2*g1) + cmul(mode,vxm,G1[i-1,j,k]))
        ty1 = ry*((cmul(mode,vy,G1[i,j+1,k]) - 2*g1) + cmul(mode,vym,G1[i,j-1,k]))
        tz1 = rz*((cmul(mode,vz,G1[i,j,k+1]) - 2*g1) + cmul(mode,vzm,G1[i,j,k-1]))
        nl1 = (c1*g1)*(fm - cabs2(mode, g1))
        F1[i,j,k] = ((((g1 + tx1) + ty1) + tz1) + nl1) + cpl1*g2

        tx2 = kx2*((cmul(mode,vx,G2[i+1,j,k]) - 2*g2) + cmul(mode,vxm,G2[i-1,j,k]))
        ty2 = ky2*((cmul(mode,vy,G2[i,j+1,k]) - 2*g2) + cmul(mode,vym,G2[i,j-1,k]))
        tz2 = kz2*((cmul(mode,vz,G2[i,j,k+1]) - 2*g2) + cmul(mode,vzm,G2[i,j,k-1]))
        nl2 = (c2*g2)*(fm*a2n - cabs2(mode, g2))
        F2[i,j,k] = ((((g2 + tx2) + ty2) + tz2) + nl2) + cpl2*g1
    end
    nothing
end

# Drag path: same stencil as _A_plano! plus the σ–π cross term on the
# links, ν = nu_drag (dimensionless relative to the σ hopping).
# Not used if use_drag=false. It does not touch the production loop.
@inline function _A_plano_drag!(F1, F2, G1, G2, Vx, Vy, Vz, fmask,
                                p::Params, mode::NumMode, k::Int)
    (; rx, ry, rz, c1, c2, dt, gamma12, mr, alpha2_n, hard) = _kernelvars(p)
    kx2, ky2, kz2, cpl1, cpl2 = coef_pi(p)
    a2n = alpha2_n
    ν = p.nu_drag
    # Γ_r = dt/c2 (same rounding as coef_pi). Outside the i,j loop.
    gr = p.dt / p.c2
    @inbounds for j in p.y1+1:p.y2, i in p.x1+1:p.x2
        g1 = G1[i,j,k]; g2 = G2[i,j,k]; fm = fmask[i,j,k]
        if hard && fm == 0.0
            F1[i,j,k] = 0.0 + 0.0im; F2[i,j,k] = 0.0 + 0.0im
            continue
        end
        vx = Vx[i,j,k]; vy = Vy[i,j,k]; vz = Vz[i,j,k]
        vxm = conj(Vx[i-1,j,k]); vym = conj(Vy[i,j-1,k]); vzm = conj(Vz[i,j,k-1])

        tx1 = rx*((cmul(mode,vx,G1[i+1,j,k]) - 2*g1) + cmul(mode,vxm,G1[i-1,j,k]))
        ty1 = ry*((cmul(mode,vy,G1[i,j+1,k]) - 2*g1) + cmul(mode,vym,G1[i,j-1,k]))
        tz1 = rz*((cmul(mode,vz,G1[i,j,k+1]) - 2*g1) + cmul(mode,vzm,G1[i,j,k-1]))
        nl1 = (c1*g1)*(fm - cabs2(mode, g1))
        dx1 = ν*rx*((cmul(mode,vx,G2[i+1,j,k]) - 2*g2) + cmul(mode,vxm,G2[i-1,j,k]))
        dy1 = ν*ry*((cmul(mode,vy,G2[i,j+1,k]) - 2*g2) + cmul(mode,vym,G2[i,j-1,k]))
        dz1 = ν*rz*((cmul(mode,vz,G2[i,j,k+1]) - 2*g2) + cmul(mode,vzm,G2[i,j,k-1]))
        F1[i,j,k] = (((((g1 + tx1) + ty1) + tz1) + nl1) + cpl1*g2) + ((dx1 + dy1) + dz1)

        tx2 = kx2*((cmul(mode,vx,G2[i+1,j,k]) - 2*g2) + cmul(mode,vxm,G2[i-1,j,k]))
        ty2 = ky2*((cmul(mode,vy,G2[i,j+1,k]) - 2*g2) + cmul(mode,vym,G2[i,j-1,k]))
        tz2 = kz2*((cmul(mode,vz,G2[i,j,k+1]) - 2*g2) + cmul(mode,vzm,G2[i,j,k-1]))
        nl2 = (c2*g2)*(fm*a2n - cabs2(mode, g2))
        # Drag in π: ν·(r/Γr) D²Ψσ. NOT ν·kx2 (= ν·r/(mr Γr)): an extra 1/mr.
        # Energy ν Re[(Dσ)*(Dπ)]; F2 already carries 1/Γr. Ampère uses the same ν.
        dx2 = (ν*rx/gr)*((cmul(mode,vx,G1[i+1,j,k]) - 2*g1) + cmul(mode,vxm,G1[i-1,j,k]))
        dy2 = (ν*ry/gr)*((cmul(mode,vy,G1[i,j+1,k]) - 2*g1) + cmul(mode,vym,G1[i,j-1,k]))
        dz2 = (ν*rz/gr)*((cmul(mode,vz,G1[i,j,k+1]) - 2*g1) + cmul(mode,vzm,G1[i,j,k-1]))
        F2[i,j,k] = (((((g2 + tx2) + ty2) + tz2) + nl2) + cpl2*g1) + ((dx2 + dy2) + dz2)
    end
    nothing
end

# T(r): α_σ = (1−θ) f, α_π = (a2n − θ) f. At θ=0 it matches the kernel
# above in arithmetic; this loop runs only with use_heat or T_physical.
@inline function _A_plano_heat!(F1, F2, G1, G2, Vx, Vy, Vz, fmask,
                                p::Params, mode::NumMode, k::Int)
    (; rx, ry, rz, c1, c2, hard) = _kernelvars(p)
    kx2, ky2, kz2, cpl1, cpl2 = coef_pi(p)
    T = p.Temp
    b2 = p.beta2
    a2 = p.a2n_cfg
    @inbounds for j in p.y1+1:p.y2, i in p.x1+1:p.x2
        g1 = G1[i,j,k]; g2 = G2[i,j,k]; fm = fmask[i,j,k]
        if hard && fm == 0.0
            F1[i,j,k] = 0.0 + 0.0im; F2[i,j,k] = 0.0 + 0.0im
            continue
        end
        θ = T[i,j,k]
        vx = Vx[i,j,k]; vy = Vy[i,j,k]; vz = Vz[i,j,k]
        vxm = conj(Vx[i-1,j,k]); vym = conj(Vy[i,j-1,k]); vzm = conj(Vz[i,j,k-1])
        tx1 = rx*((cmul(mode,vx,G1[i+1,j,k]) - 2*g1) + cmul(mode,vxm,G1[i-1,j,k]))
        ty1 = ry*((cmul(mode,vy,G1[i,j+1,k]) - 2*g1) + cmul(mode,vym,G1[i,j-1,k]))
        tz1 = rz*((cmul(mode,vz,G1[i,j,k+1]) - 2*g1) + cmul(mode,vzm,G1[i,j,k-1]))
        nl1 = (c1*g1)*((1-θ)*fm - cabs2(mode, g1))
        F1[i,j,k] = ((((g1 + tx1) + ty1) + tz1) + nl1) + cpl1*g2
        tx2 = kx2*((cmul(mode,vx,G2[i+1,j,k]) - 2*g2) + cmul(mode,vxm,G2[i-1,j,k]))
        ty2 = ky2*((cmul(mode,vy,G2[i,j+1,k]) - 2*g2) + cmul(mode,vym,G2[i,j-1,k]))
        tz2 = kz2*((cmul(mode,vz,G2[i,j,k+1]) - 2*g2) + cmul(mode,vzm,G2[i,j,k-1]))
        nl2 = (c2*g2)*(fm*(a2-θ)/b2 - cabs2(mode, g2))
        F2[i,j,k] = ((((g2 + tx2) + ty2) + tz2) + nl2) + cpl2*g1
    end
    nothing
end

@inline function _A_plano_drag_heat!(F1, F2, G1, G2, Vx, Vy, Vz, fmask,
                                     p::Params, mode::NumMode, k::Int)
    (; rx, ry, rz, c1, c2, hard) = _kernelvars(p)
    kx2, ky2, kz2, cpl1, cpl2 = coef_pi(p)
    ν = p.nu_drag
    gr = p.dt / p.c2
    T = p.Temp
    b2 = p.beta2
    a2 = p.a2n_cfg
    @inbounds for j in p.y1+1:p.y2, i in p.x1+1:p.x2
        g1 = G1[i,j,k]; g2 = G2[i,j,k]; fm = fmask[i,j,k]
        if hard && fm == 0.0
            F1[i,j,k] = 0.0 + 0.0im; F2[i,j,k] = 0.0 + 0.0im
            continue
        end
        θ = T[i,j,k]
        vx = Vx[i,j,k]; vy = Vy[i,j,k]; vz = Vz[i,j,k]
        vxm = conj(Vx[i-1,j,k]); vym = conj(Vy[i,j-1,k]); vzm = conj(Vz[i,j,k-1])
        tx1 = rx*((cmul(mode,vx,G1[i+1,j,k]) - 2*g1) + cmul(mode,vxm,G1[i-1,j,k]))
        ty1 = ry*((cmul(mode,vy,G1[i,j+1,k]) - 2*g1) + cmul(mode,vym,G1[i,j-1,k]))
        tz1 = rz*((cmul(mode,vz,G1[i,j,k+1]) - 2*g1) + cmul(mode,vzm,G1[i,j,k-1]))
        nl1 = (c1*g1)*((1-θ)*fm - cabs2(mode, g1))
        dx1 = ν*rx*((cmul(mode,vx,G2[i+1,j,k]) - 2*g2) + cmul(mode,vxm,G2[i-1,j,k]))
        dy1 = ν*ry*((cmul(mode,vy,G2[i,j+1,k]) - 2*g2) + cmul(mode,vym,G2[i,j-1,k]))
        dz1 = ν*rz*((cmul(mode,vz,G2[i,j,k+1]) - 2*g2) + cmul(mode,vzm,G2[i,j,k-1]))
        F1[i,j,k] = (((((g1 + tx1) + ty1) + tz1) + nl1) + cpl1*g2) + ((dx1 + dy1) + dz1)
        tx2 = kx2*((cmul(mode,vx,G2[i+1,j,k]) - 2*g2) + cmul(mode,vxm,G2[i-1,j,k]))
        ty2 = ky2*((cmul(mode,vy,G2[i,j+1,k]) - 2*g2) + cmul(mode,vym,G2[i,j-1,k]))
        tz2 = kz2*((cmul(mode,vz,G2[i,j,k+1]) - 2*g2) + cmul(mode,vzm,G2[i,j,k-1]))
        nl2 = (c2*g2)*(fm*(a2-θ)/b2 - cabs2(mode, g2))
        dx2 = (ν*rx/gr)*((cmul(mode,vx,G1[i+1,j,k]) - 2*g1) + cmul(mode,vxm,G1[i-1,j,k]))
        dy2 = (ν*ry/gr)*((cmul(mode,vy,G1[i,j+1,k]) - 2*g1) + cmul(mode,vym,G1[i,j-1,k]))
        dz2 = (ν*rz/gr)*((cmul(mode,vz,G1[i,j,k+1]) - 2*g1) + cmul(mode,vzm,G1[i,j,k-1]))
        F2[i,j,k] = (((((g2 + tx2) + ty2) + tz2) + nl2) + cpl2*g1) + ((dx2 + dy2) + dz2)
    end
    nothing
end

"""
    psi_update_ref!(...)

REFERENCE: literal transcription of `psi_update_loop` of the .m, itself the
historical triple loop (v3_final..v6).  Used only by the selftest.

DO NOT REMOVE.  It is the canonical definition of the scheme: if the fast
kernel and this one ever disagree, the fast one is wrong.  The .m keeps it
for the same reason.
"""
function psi_update_ref!(F1, F2, G1, G2, Vx, Vy, Vz, fmask, p::Params, mode::NumMode)
    rx, ry, rz = p.rx, p.ry, p.rz
    c1, c2, dt = p.c1, p.c2, p.dt
    g12, mr, a2n = p.gamma12, p.mr, p.alpha2_n
    hard = p.hard
    for k in p.z1+1:p.z2
        for j in p.y1+1:p.y2
            for i in p.x1+1:p.x2
                f_loc = fmask[i,j,k]
                if hard && f_loc == 0.0
                    F1[i,j,k] = 0.0im; F2[i,j,k] = 0.0im
                    continue
                end
                g1 = G1[i,j,k]; g2 = G2[i,j,k]
                tx1 = rx*((cmul(mode,Vx[i,j,k],G1[i+1,j,k]) - 2*g1) + cmul(mode,conj(Vx[i-1,j,k]),G1[i-1,j,k]))
                ty1 = ry*((cmul(mode,Vy[i,j,k],G1[i,j+1,k]) - 2*g1) + cmul(mode,conj(Vy[i,j-1,k]),G1[i,j-1,k]))
                tz1 = rz*((cmul(mode,Vz[i,j,k],G1[i,j,k+1]) - 2*g1) + cmul(mode,conj(Vz[i,j,k-1]),G1[i,j,k-1]))
                nl1 = (c1*g1)*(f_loc - cabs2(mode, g1))
                F1[i,j,k] = ((((g1 + tx1) + ty1) + tz1) + nl1) + (dt*g12)*g2

                grl = dt/c2      # see the note in coef_pi
                tx2 = (rx/(mr*grl))*((cmul(mode,Vx[i,j,k],G2[i+1,j,k]) - 2*g2) + cmul(mode,conj(Vx[i-1,j,k]),G2[i-1,j,k]))
                ty2 = (ry/(mr*grl))*((cmul(mode,Vy[i,j,k],G2[i,j+1,k]) - 2*g2) + cmul(mode,conj(Vy[i,j-1,k]),G2[i,j-1,k]))
                tz2 = (rz/(mr*grl))*((cmul(mode,Vz[i,j,k],G2[i,j,k+1]) - 2*g2) + cmul(mode,conj(Vz[i,j,k-1]),G2[i,j,k-1]))
                nl2 = (c2*g2)*(f_loc*a2n - cabs2(mode, g2))
                F2[i,j,k] = ((((g2 + tx2) + ty2) + tz2) + nl2) + ((dt/grl)*g12)*g1
            end
        end
    end
    (F1, F2)
end

@inline _kernelvars(p::Params) = (rx=p.rx, ry=p.ry, rz=p.rz, c1=p.c1, c2=p.c2,
                                  dt=p.dt, gamma12=p.gamma12, mr=p.mr,
                                  alpha2_n=p.alpha2_n, hard=p.hard)

"""
    run_selftest(g, p, mode) -> Float64

Checks that the production kernel and the reference loop agree on a
non-trivial state (deterministic perturbation + non-trivial links), three
iterations.  Returns the worst difference; aborts above 1e-12.

It is the SAME test as `run_selftest` of the .m, with the SAME deterministic
seed, so the two implementations can be compared with each other and with
Octave.
"""
function run_selftest(g::Geometry, p::Params, mode::NumMode=Strict())
    p.verbose && print("[selftest] production kernel vs reference loop... ")
    Nx, Ny, Nz = p.Nx, p.Ny, p.Nz
    ii = Float64.(reshape(0:Nx, :, 1, 1))
    jj = Float64.(reshape(0:Ny, 1, :, 1))
    kk = Float64.(reshape(0:Nz, 1, 1, :))
    pert = @. 0.10*sin(0.37*ii + 0.11)*cos(0.23*jj - 0.05)*sin(0.51*kk + 0.29)
    G1 = @. g.taupsi*(1.0 + pert)*cis(0.20*sin(0.13*ii - 0.31*jj))
    G2 = @. g.taupsi*(sqrt(p.alpha2_n) + 0.5*pert)*cis(-0.15*cos(0.17*jj + 0.19*kk))
    G1 = ComplexF64.(G1); G2 = ComplexF64.(G2)

    phix = [0.05*sin(0.29*(i-1) + 0.07*(j-1)) for i in 1:Nx,   j in 1:Ny+1, k in 1:Nz+1]
    phiy = [0.05*cos(0.31*(j-1) - 0.09*(k-1)) for i in 1:Nx+1, j in 1:Ny,   k in 1:Nz+1]
    phiz = [0.05*sin(0.27*(k-1) + 0.13*(i-1)) for i in 1:Nx+1, j in 1:Ny+1, k in 1:Nz]
    Vx = emiphi.(Ref(mode), phix); Vy = emiphi.(Ref(mode), phiy); Vz = emiphi.(Ref(mode), phiz)

    F1a = copy(G1); F2a = copy(G2); F1b = copy(G1); F2b = copy(G2)
    G1a = copy(G1); G2a = copy(G2); G1b = copy(G1); G2b = copy(G2)
    peor = 0.0
    for _ in 1:3
        psi_update!(F1a, F2a, G1a, G2a, Vx, Vy, Vz, g.fmask, p, mode)
        psi_update_ref!(F1b, F2b, G1b, G2b, Vx, Vy, Vz, g.fmask, p, mode)
        peor = max(peor, maximum(abs, F1a .- F1b), maximum(abs, F2a .- F2b))
        copyto!(G1a, F1a); copyto!(G2a, F2a); copyto!(G1b, F1b); copyto!(G2b, F2b)
    end
    peor > 1e-12 && error("SELFTEST FAILED: max diff = $peor (>1e-12). Do NOT run.")
    p.verbose && @printf("OK (max diff = %.2e).\n", peor)
    peor
end
