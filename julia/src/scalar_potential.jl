# ============================================================================
#  scalar_potential.jl — THE SCALAR POTENTIAL, COUPLED.
#
#  All of v13 runs in the gauge Phi = 0.  This lifts that restriction.
#
#  --- WHY IT COULD BE WRITTEN -----------------------------------------------
#  Step 2 (poisson.jl) measured
#
#      d/dt(div A) = div J_s     to 100.00 %,   div(curl curl A) = 5.6e-17
#
#  so  lap Phi = div J_s - d/dt(div A) = 0  IDENTICALLY.  That is: the kernel's
#  equation for A ALREADY IS charge conservation, and taking its divergence
#  adds no information.  **Phi is fixed by no equation: it is gauge freedom.**
#
#  That says WHAT the Phi step is.  It is not "one more field solved and
#  added": it is the PROJECTION that keeps the chosen gauge.  In the Coulomb
#  gauge (div A = 0):
#
#      1. compute the TENTATIVE link increment, without Phi:
#             delta = [C](theta, Q, h) - theta
#      2. solve           D(G Phi) = D(delta)
#      3. correct         delta' = delta - G Phi     =>   D(delta') = 0
#
#  with G the discrete gradient from a nodal scalar to the links
#  (G Phi)_x[i] = Phi[i+1] - Phi[i], and D its adjoint from links to nodes.
#  With THESE two operators, D o G is EXACTLY the seven-point Laplacian
#  already in poisson.jl, so  D(delta') = 0  holds to machine precision, not
#  approximately.  It is the pressure step of an incompressible solver, with
#  the same structure and the same cost.
#
#  --- THE THREE PLACES ---------------------------------------------------------
#
#  [v14-A]  The Psi update gains  -i Phi Psi  in BOTH bands, with `dt` and NOT
#           `dt/Gamma_r`.  The reason: the covariant time derivative is
#           Gamma_a (d_t + i Phi) Psi_a = -dF/dPsi_a*, and dividing by
#           Gamma_a takes the Phi term with it because it is on the SAME
#           side.  Confusing the two is exactly M16.
#
#  [v14-B]  The A update gains  -grad Phi.  And M17 appears: `d1 = dt*(1-T)`
#           multiplies the supercurrent, and the correct equation
#           sigma(d_t A + grad Phi) = J_s - kappa^2 curl curl A  does NOT
#           carry that factor.  At T = 0 it equals 1 and goes unnoticed.  It is
#           exposed as `cfg.f22_supercurrent` (and `cfg.T_physical`).
#
#  [v14-C]  The new voltmeter: V_Phi = <Phi>_in - <Phi>_out over the SAME
#           planes and masks as the Josephson voltmeter, so the comparison is
#           over the same path.
#
#  --- THE BOUNDARY CONDITION OF Phi: WHAT CHANGES AND WHAT DOES NOT --------
#  Changing the BC of Phi changes the GAUGE, not the physics.  The invariant
#  observables —|Psi|, h, and the Josephson voltage on the gauge-invariant
#  phase— must come out equal with either.  What DOES change is the value of
#  Phi, and hence what "V_Phi" means.
#
#  So the two voltage definitions are NOT two rival candidates for the same
#  quantity.  The physical voltage is invariant and equal in both gauges; what
#  changes is HOW IT SPLITS between the drop of Phi and the drift of the
#  phase.  With Phi = 0 all of it is in the phase.  In Coulomb, part is in
#  Phi.  The useful question is not "do they agree?" but "how does it split?",
#  which has a measurable answer that depends on the BC.
#
#  Both BCs are implemented; the choice is `cfg.phi_bc`.
#  Default :dirichlet = regression (Octave, bank 3).  The PHYSICAL gauge of
#  this setup is :neumann: there are no electrodes, the current enters
#  through the self-field.
#
#  --- v14.1: WHAT BREAKS THE EQUIVALENCE (Article IV) ----------------------
#  With lambda_TF=0 and tau_Q=0, Phi is gauge and the above holds.
#
#  Thomas-Fermi (lambda_TF>0): the operator becomes D∘G - I/lambda_TF^2.
#  Same cost, only the spectrum changes.  Then D(delta') + Phi/lambda_TF^2 = 0,
#  not D(delta')=0: the mass term fixes WHICH Phi comes out (it gives Phi a
#  scale, removes the null mode and leaves div A nonzero).  With tau_Q=0 the
#  step is still a lattice gauge transformation, whatever equation fixes Phi:
#  theta loses grad(Phi) and Psi rotates by the same Phi, so |Psi|, B and the
#  Josephson voltage are unchanged and only V_Phi moves.  test_phi_tf.jl
#  checks both: div A no longer vanishes, |Psi| agrees to 1e-12.  (Up to
#  6-oct-2026 this comment said the |Psi| control "MUST fail" with TF; the
#  test and the equations say otherwise.)  What does change the dynamics is
#  tau_Q>0: Psi then rotates by Phi+Q while A is corrected by Phi alone.
#
#  tau_Q>0: charge-imbalance field Q.  Implicit Euler for
#      tau_Q dQ/dt + Q = D(delta')
#      Q <- (Q + (dt/tau_Q) D(delta')) / (1 + dt/tau_Q)
#  Attractor: Q -> D(delta'), in the same units as Phi_projection (which
#  already carries dt).  The old formula Q <- (Q+sigma)/(1+dt/tau_Q) had the
#  attractor (tau_Q/dt)*sigma ~ 2000 sigma; that rotation killed the
#  condensate at J=0 (measured).
#  The rotation of Psi uses mu = Phi + Q, not Phi.  Q does NOT feed back into
#  the Poisson problem of Phi: putting it there (rhs = D(delta) - Q) was an
#  unstable loop, Q ~ 2^n.  WITHOUT lambda_TF the source D(delta') vanishes
#  (exact Coulomb) and Q->0: tau_Q is physics only together with TF.
#  sigma_n(r) is not implemented.
# ============================================================================

"""
    SolverPhi

The projection step, with the factorisation done ONCE.

The matrix is CONSTANT —geometry and mesh do not change in time— so it is
prepared at construction and each substep only applies it.

**TWO METHODS; the second exists because the first was measured.**

`:lu` — sparse factorisation (SuiteSparse), as in the Octave scripts.
Correct and direct.  Measured: at the production mesh the Phi step is
**80 %** of a v14 substep, and v14 as a whole costs 4.6x v13.

`:separable` — the seven-point Laplacian on a uniform box is SEPARABLE:

    A = Lx (+) Ly (+) Lz

so it is diagonalised by three SMALL 1-D eigenproblems (of size Nx-1, Ny-1,
Nz-1), done once, and each solve is three dense matrix products —BLAS-3— plus
a division by `lam_x + lam_y + lam_z`.  **It is an exact direct method, not
an approximation**: it gives the same solution as the LU up to rounding, and
`test_phi.jl` checks it.

It needs no new dependency: `LinearAlgebra` suffices.  FFTW was considered
—a DST diagonalises this just as exactly— and rejected so as not to add a
binary dependency to a package that has none.
"""
struct SolverPhi
    metodo::Symbol                        # :lu | :separable
    fact                                  # LU factorisation (method :lu)
    Qx::Matrix{Float64}; Qy::Matrix{Float64}; Qz::Matrix{Float64}
    inv_lam::Array{Float64,3}             # 1/(lam_x+lam_y+lam_z), 0 on the null mode
    b1::Array{Float64,3}; b2::Array{Float64,3}   # transform buffers
    Phi::Array{Float64,3}                 # potential on ALL nodes
    rhs::Array{Float64,3}                 # right-hand side on the interior block
    bc::Symbol                            # :dirichlet | :neumann
    n1::Int; n2::Int; n3::Int             # size of the interior block
    lambda_TF::Float64                    # 0 = pure Coulomb
    k2::Float64                           # 1/lambda_TF^2, or 0
    tau_Q::Float64                        # 0 = no imbalance
    Qimb::Array{Float64,3}                # charge imbalance (nodes)
end

"""
    SolverPhi(p; bc=:dirichlet) -> SolverPhi

Builds and factorises the Laplacian `D∘G` on the interior nodes `2:Nx, 2:Ny,
2:Nz`.

`bc = :dirichlet` sets `Phi = 0` on the six faces.  It is what the Octave
scripts did and leaves the system non-singular.

`bc = :neumann` imposes `dPhi/dn = 0`, natural for a domain from which no
normal current leaves —and in this code the current enters through the
self-field, not through electrodes—.  It is the PHYSICAL gauge of this setup.
It leaves `Phi` defined up to a constant, so one node is anchored (only if
lambda_TF=0: with Thomas-Fermi the null mode disappears).  A free constant is
NOT a problem: a constant `Phi` cannot change any observable, and that is a
TEST (see `test_phi.jl`), not an assumption.
"""
function SolverPhi(p::Params; bc::Symbol=:dirichlet, metodo::Symbol=:separable)
    bc in (:dirichlet, :neumann) ||
        error("unknown phi_bc: $bc (use :dirichlet or :neumann)")
    metodo in (:lu, :separable) ||
        error("unknown method: $metodo (use :lu or :separable)")
    n1, n2, n3 = p.Nx-1, p.Ny-1, p.Nz-1
    ceros3(a,b,c) = zeros(a,b,c)
    k2 = p.lambda_TF > 0 ? 1 / p.lambda_TF^2 : 0.0
    Phi0 = zeros(p.Nx+1, p.Ny+1, p.Nz+1)
    Q0   = zeros(p.Nx+1, p.Ny+1, p.Nz+1)

    # --- the 1-D operator, with the requested BC ---------------------------
    # Homogeneous Neumann: the row of the border node loses its outer
    # neighbour, so its diagonal goes from -2 to -1.  It is the SAME
    # construction `hc3_over_hc2` uses for the Neumann condition of block
    # [H], for the same reason: the boundary is imposed as the kernel does.
    function L1d(n, h)
        M = zeros(n, n)
        for i in 1:n
            M[i,i] = -2/h^2
            i > 1 && (M[i,i-1] = 1/h^2)
            i < n && (M[i,i+1] = 1/h^2)
        end
        if bc === :neumann
            M[1,1] = -1/h^2; M[n,n] = -1/h^2
        end
        M
    end
    Lx = L1d(n1, p.dx); Ly = L1d(n2, p.dy); Lz = L1d(n3, p.dz)

    if metodo === :lu
        A = kron(sparse(I,n3,n3), kron(sparse(I,n2,n2), sparse(Lx))) +
            kron(sparse(I,n3,n3), kron(sparse(Ly), sparse(I,n1,n1))) +
            kron(sparse(Lz),      kron(sparse(I,n2,n2), sparse(I,n1,n1)))
        if k2 != 0.0
            A = A - k2 * sparse(I, n1*n2*n3, n1*n2*n3)
        end
        if bc === :neumann && k2 == 0.0
            # singular up to a constant: the first node is anchored.  The
            # anchored constant is pure gauge.  With TF the null mode goes
            # away (l = -1/lambda_TF^2) and no anchoring is needed.
            A = Matrix(A); A[1,:] .= 0.0; A[1,1] = 1.0; A = sparse(A)
        end
        z = zeros(0,0)
        return SolverPhi(:lu, lu(A), z, z, z, ceros3(0,0,0),
                         ceros3(0,0,0), ceros3(0,0,0),
                         Phi0, ceros3(n1,n2,n3),
                         bc, n1, n2, n3,
                         p.lambda_TF, k2, p.tau_Q, Q0)
    end

    # --- separable: three 1-D eigenproblems, once ------------------------
    ex = eigen(Symmetric(Lx)); ey = eigen(Symmetric(Ly)); ez = eigen(Symmetric(Lz))
    inv_lam = zeros(n1, n2, n3)
    for k in 1:n3, j in 1:n2, i in 1:n1
        l0 = ex.values[i] + ey.values[j] + ez.values[k]
        # Branch k2==0: IDENTICAL to the original v14, including the 1e-10
        # threshold of the Neumann null mode.  Untouched: v14 acceptance lives here.
        if k2 == 0.0
            inv_lam[i,j,k] = abs(l0) < 1e-10 ? 0.0 : 1/l0
        else
            inv_lam[i,j,k] = 1 / (l0 - k2)
        end
    end
    SolverPhi(:separable, nothing, Matrix(ex.vectors), Matrix(ey.vectors),
              Matrix(ez.vectors), inv_lam, ceros3(n1,n2,n3), ceros3(n1,n2,n3),
              Phi0, ceros3(n1,n2,n3), bc, n1, n2, n3,
              p.lambda_TF, k2, p.tau_Q, Q0)
end

"""
    solve!(sp) -> sp

Solves `A Phi = sp.rhs` with the chosen method and leaves the result in
`sp.rhs` (in place).
"""
function solve!(sp::SolverPhi)
    if sp.metodo === :lu
        v = vec(sp.rhs)
        # anchor node 1 ONLY if the system is singular (Neumann, k2=0).
        # With Thomas-Fermi the null mode goes away and overwriting v[1]
        # would solve ANOTHER equation.
        sp.bc === :neumann && sp.k2 == 0.0 && (v[1] = 0.0)
        copyto!(v, sp.fact \ v)
        return sp
    end
    n1, n2, n3 = sp.n1, sp.n2, sp.n3
    # forward transform: Q' applied along each direction (BLAS-3)
    mul!(reshape(sp.b1, n1, n2*n3), transpose(sp.Qx), reshape(sp.rhs, n1, n2*n3))
    @inbounds for k in 1:n3
        mul!(view(sp.b2,:,:,k), view(sp.b1,:,:,k), sp.Qy)
    end
    mul!(reshape(sp.b1, n1*n2, n3), reshape(sp.b2, n1*n2, n3), sp.Qz)
    # division by the eigenvalue
    @inbounds @simd for i in eachindex(sp.b1)
        sp.b1[i] *= sp.inv_lam[i]
    end
    # inverse transform: Q
    mul!(reshape(sp.b2, n1*n2, n3), reshape(sp.b1, n1*n2, n3), transpose(sp.Qz))
    @inbounds for k in 1:n3
        mul!(view(sp.b1,:,:,k), view(sp.b2,:,:,k), transpose(sp.Qy))
    end
    mul!(reshape(sp.rhs, n1, n2*n3), sp.Qx, reshape(sp.b1, n1, n2*n3))
    sp
end

"""
    block_C2!(s, p, sp) -> s

**[v14-B], the projection.**  Called AFTER `bloque_C!` and BEFORE
`bloque_E!`: `bloque_C!` leaves the tentative increment in `phi*`, and this
projects it onto the divergence-free subspace and stores `Phi`.

The ranges are EXACTLY those `bloque_C!` writes.  Going outside them would
correct links the scheme does not evolve, the kind of off-by-one behind mode
M5.
"""
function block_C2!(s::State, p::Params, sp::SolverPhi)
    (; Nx, Ny, Nz, dx, dy, dz) = p
    Phi = sp.Phi; rhs = sp.rhs

    # rhs = D(delta), with delta = phi - theta the tentative link increment.
    # D is the adjoint of G: from links to nodes, divided by h^2.
    @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx
        dxm = (s.phix[i,j,k]   - s.thetax[i,j,k])   -
              (s.phix[i-1,j,k] - s.thetax[i-1,j,k])
        dym = (s.phiy[i,j,k]   - s.thetay[i,j,k])   -
              (s.phiy[i,j-1,k] - s.thetay[i,j-1,k])
        dzm = (s.phiz[i,j,k]   - s.thetaz[i,j,k])   -
              (s.phiz[i,j,k-1] - s.thetaz[i,j,k-1])
        dval = dxm/dx^2 + dym/dy^2 + dzm/dz^2
        rhs[i-1,j-1,k-1] = dval
    end

    solve!(sp)
    fill!(Phi, 0.0)
    @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx
        Phi[i,j,k] = rhs[i-1,j-1,k-1]
    end
    if sp.bc === :neumann
        # extend to the faces by Neumann: dPhi/dn = 0 => copy the neighbour.
        # Needed so that G Phi is defined on the border links.
        @inbounds for k in 2:Nz, j in 2:Ny
            Phi[1,j,k] = Phi[2,j,k];  Phi[Nx+1,j,k] = Phi[Nx,j,k]
        end
        @inbounds for k in 2:Nz, i in 1:Nx+1
            Phi[i,1,k] = Phi[i,2,k];  Phi[i,Ny+1,k] = Phi[i,Ny,k]
        end
        @inbounds for j in 1:Ny+1, i in 1:Nx+1
            Phi[i,j,1] = Phi[i,j,2];  Phi[i,j,Nz+1] = Phi[i,j,Nz]
        end
    end

    # delta' = delta - G Phi.  G from a nodal scalar to the links is the
    # PLAIN DIFFERENCE, not divided by h: the link already carries the length.
    @inbounds for k in 2:Nz, j in 2:Ny, i in 1:Nx
        s.phix[i,j,k] -= Phi[i+1,j,k] - Phi[i,j,k]
    end
    @inbounds for k in 2:Nz, j in 1:Ny, i in 2:Nx
        s.phiy[i,j,k] -= Phi[i,j+1,k] - Phi[i,j,k]
    end
    @inbounds for k in 1:Nz, j in 2:Ny, i in 2:Nx
        s.phiz[i,j,k] -= Phi[i,j,k+1] - Phi[i,j,k]
    end

    # Imbalance: IMPLICIT Euler for  tau_Q dQ/dt + Q = sigma,
    # sigma = D(delta') after the correction (same units as Phi_projection).
    # With TF, sigma = -k2 Phi; with pure Coulomb, sigma=0 and Q goes to zero.
    # The factor dt/tau_Q in the source makes the attractor Q->sigma.
    # Without it, the attractor was (tau_Q/dt)*sigma and Psi collapsed at J=0.
    # Stable for tau_Q as small as desired, including tau_Q << dt
    # (then Q_new ~ sigma in one step).
    if sp.tau_Q > 0
        α = p.dt / sp.tau_Q
        den = 1 + α
        if p.use_Qss
            # τ_Q (∂t Q − D_Q ∇²Q) + Q = σ.  Implicit relaxation (as in
            # v14) plus explicit diffusion of the old Q. This block runs only
            # with use_Qss.
            Qold = copy(sp.Qimb)
            β = p.dt * p.D_Q
            @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx
                dxm = (s.phix[i,j,k]   - s.thetax[i,j,k])   -
                      (s.phix[i-1,j,k] - s.thetax[i-1,j,k])
                dym = (s.phiy[i,j,k]   - s.thetay[i,j,k])   -
                      (s.phiy[i,j-1,k] - s.thetay[i,j-1,k])
                dzm = (s.phiz[i,j,k]   - s.thetaz[i,j,k])   -
                      (s.phiz[i,j,k-1] - s.thetaz[i,j,k-1])
                σ = dxm/dx^2 + dym/dy^2 + dzm/dz^2
                q = Qold[i,j,k]
                lap = (Qold[i+1,j,k] - 2q + Qold[i-1,j,k]) / dx^2 +
                      (Qold[i,j+1,k] - 2q + Qold[i,j-1,k]) / dy^2 +
                      (Qold[i,j,k+1] - 2q + Qold[i,j,k-1]) / dz^2
                sp.Qimb[i,j,k] = (q + α * σ) / den + β * lap
            end
        else
            @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx
                dxm = (s.phix[i,j,k]   - s.thetax[i,j,k])   -
                      (s.phix[i-1,j,k] - s.thetax[i-1,j,k])
                dym = (s.phiy[i,j,k]   - s.thetay[i,j,k])   -
                      (s.phiy[i,j-1,k] - s.thetay[i,j-1,k])
                dzm = (s.phiz[i,j,k]   - s.thetaz[i,j,k])   -
                      (s.phiz[i,j,k-1] - s.thetaz[i,j,k-1])
                σ = dxm/dx^2 + dym/dy^2 + dzm/dz^2
                sp.Qimb[i,j,k] = (sp.Qimb[i,j,k] + α * σ) / den
            end
        end
    end
    s
end

"""
    apply_phi_psi!(s, p, sp) -> s

**[v14-A].**  Adds `-i dt Phi Psi` to both bands, on the INTERIOR nodes,
which are the ones `psi_update!` evolves.

**The coefficient is IDENTICAL in both bands**.  The covariant time
derivative is

    Gamma_a (d_t + i Phi) Psi_a = -dF/dPsi_a*

so dividing by `Gamma_a` takes the `Phi` term with it: it is on the SAME side
of the equation.  Treating it as if it came from the right-hand side —and
hence giving it `1/Gamma_r` in the pi band— is exactly the M16 error,
committed on a new term.

**And WITHOUT a `dt` factor.**  "It carries dt, not dt/Gamma_r" is true of
the PHYSICAL `Phi`.  But the `Phi` that comes out of the projection is not
the physical one: since the link INCREMENT is projected —which is already
`dt * d_t A * dx`— the Lagrange multiplier comes out with the `dt` inside:

    Phi_proyeccion = dt * Phi_fisico

Consistency is fixed by the gauge transformation on the LATTICE, which
leaves no room for interpretation: with `psi_i -> e^{i chi_i} psi_i`, the
invariance of `psi_i^* U_ij psi_j` requires `theta_ij -> theta_ij + (chi_j -
chi_i)`.  Here `Phi[i+1] - Phi[i]` is subtracted from theta, i.e.
`chi = -Phi`, and then the rotation of psi is `e^{-i Phi}` — **with the same
`Phi`, without an extra `dt`**.

Using `dt*Phi` here breaks gauge invariance by O(dt) every substep.
Measured before the fix: `|Psi|` differed by 2.8e-2 between the gauge
`Phi=0` and Coulomb, where the difference must be ZERO.  Control 3 of
`test_phi.jl` caught it.

**And with the EXACT `exp(-i Phi)`, not the linearisation `1 - i Phi`.**  The
lattice gauge transformation is a PHASE, not an increment: `psi ->
e^{i chi} psi`.  Linearising it adds an O(Phi^2) error per substep that
accumulates to O(dt) over the run.  Measured: with `1 - i Phi` gauge
invariance held to 2.6e-6; with `cis(-Phi)` to machine precision.

**And AFTER the projection, not before.**  The correction of `theta` and the
rotation of `psi` are the TWO HALVES of the same gauge transformation and
must use the SAME `Phi`.  Applying it in `[A]` —which runs before `[C]`— uses
the `Phi` of the previous substep and mismatches the two halves by one step.
Measured: that cost a factor 10^4 in the invariance.

The kernel is explicit and almost everything is read from the previous step,
so placing this in `[A]` is the natural reflex.  It does not apply here,
because this is not one more term of the equation: it is half of a gauge
transformation.
"""
function apply_phi_psi!(s::State, p::Params, sp::SolverPhi, mode::NumMode)
    Phi = sp.Phi
    @inbounds for k in p.z1+1:p.z2, j in p.y1+1:p.y2, i in p.x1+1:p.x2
        χ = Phi[i,j,k]
        # tau_Q=0: do not add 0.0 (it would change the bit).  With imbalance
        # the condensate sees the ELECTROCHEMICAL potential mu = Phi + Q, not
        # Phi.  The two halves of the "gauge transformation" are no longer the
        # same, and that is the new physics: Maxwell sees Phi, Psi sees mu.
        sp.tau_Q > 0 && (χ += sp.Qimb[i,j,k])
        u = emiphi(mode, χ)               # exp(-i chi), WITHOUT dt: see the docstring
        s.F1[i,j,k] = cmul(mode, u, s.F1[i,j,k])
        s.F2[i,j,k] = cmul(mode, u, s.F2[i,j,k])
    end
    s
end

"""
    substep_phi!(s, p, g, mode, sp; H, HJ, J_cur)

The v14 substep: the same nine blocks plus `[v14-A]` and `[v14-B]` as the
projection block `[C2]`.

    [A]  Psi update, without Phi
    [B]  supercurrent
    [C]  tentative dA/dt
    [C2] PROJECTION      = [v14-B]  solves Phi and subtracts G Phi from theta
    [A2] rotation of Psi = [v14-A]  multiplies by exp(-i Phi), the SAME Phi
    [E] [F] [G] [H] [I]  as in v13

The order `[C2]` before `[A2]` is not negotiable: they are the two halves of
one gauge transformation and must use the same `Phi`.

With `use_phi = false` this is not called and the port is v13.3 bit for bit.
"""
function substep_phi!(s::State, p::Params, g::Geometry, mode::NumMode,
                      sp::SolverPhi; H::Float64, HJ::Float64, J_cur::Float64)
    psi_update!(s.F1, s.F2, s.G1, s.G2, s.Vx, s.Vy, s.Vz, g.fmask, p, mode)
    block_B!(s, p, mode)
    p.use_heat && paso_heat!(s, p, g)
    bloque_C!(s, p, g)
    block_C2!(s, p, sp)                             # [v14-B] solves Phi
    apply_phi_psi!(s, p, sp, mode)                 # [v14-A] SAME Phi as [C2]
    p.selffield || bloque_D!(s, p, g, J_cur)
    bloque_E!(s, p, mode)
    bloque_F!(s, p)
    bloque_G!(s, p, H, HJ)
    bloque_H!(s, p, mode)
    bloque_I!(s, p)
    s
end

"""
    phi_voltage(sp, p, vi, vo, Min, Mou, axis_y) -> Float64

Returns the PHYSICAL potential (`Phi_projection / dt`), not the raw one.

**[v14-C].**  `V_Phi = <Phi>_in - <Phi>_out` over the SAME planes and masks
as the Josephson voltmeter, so both measurements are over the same path and
the comparison means something.

**What this measurement is NOT**: the physical voltage is gauge invariant and
equal in `Phi = 0` and in Coulomb.  What changes between gauges is how it
SPLITS between `V_Phi` and the drift of the phase.  `V_Phi` alone, without the
phase part, is not the voltage: it is its electrostatic half in a given gauge
with a given BC.
"""
function phi_voltage(sp::SolverPhi, p::Params, vi::Int, vo::Int, Min, Mou,
                     axis_y::Bool)
    (; x1, x2, y1, y2, z1, z2) = p
    Phi = sp.Phi
    si = 0.0; so = 0.0; ni = 0; no = 0
    if axis_y
        @inbounds for (kk, k) in enumerate(z1:z2+1), (ii, i) in enumerate(x1:x2+1)
            if Min[ii,kk]; si += Phi[i,vi,k]; ni += 1; end
            if Mou[ii,kk]; so += Phi[i,vo,k]; no += 1; end
        end
    else
        @inbounds for (kk, k) in enumerate(z1:z2+1), (jj, j) in enumerate(y1:y2+1)
            if Min[jj,kk]; si += Phi[vi,j,k]; ni += 1; end
            if Mou[jj,kk]; so += Phi[vo,j,k]; no += 1; end
        end
    end
    # The `Phi` of the projection carries the `dt` inside (see
    # `apply_phi_psi!`), so the PHYSICAL potential is Phi/dt.  Returning the
    # raw one would give a voltage 200 times smaller than the Josephson one by
    # mere choice of variable.
    (si/max(ni,1) - so/max(no,1)) / p.dt
end

"""
    residual_div_increment(s, p) -> Array

`D(phi - theta)`, the same operator that builds the rhs of the projection,
on the increment of the current substep.  Thomas-Fermi control: with
`lambda_TF>0`, `residual + Phi/lambda_TF^2 = 0` must hold to machine
precision in the interior, not "small".
"""
function residual_div_increment(s::State, p::Params)
    (; Nx, Ny, Nz, dx, dy, dz) = p
    d = zeros(Nx+1, Ny+1, Nz+1)
    @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx
        dxm = (s.phix[i,j,k]   - s.thetax[i,j,k])   -
              (s.phix[i-1,j,k] - s.thetax[i-1,j,k])
        dym = (s.phiy[i,j,k]   - s.thetay[i,j,k])   -
              (s.phiy[i,j-1,k] - s.thetay[i,j-1,k])
        dzm = (s.phiz[i,j,k]   - s.thetaz[i,j,k])   -
              (s.phiz[i,j,k-1] - s.thetaz[i,j,k-1])
        d[i,j,k] = dxm/dx^2 + dym/dy^2 + dzm/dz^2
    end
    d
end

"""
    link_divergence(s, p) -> divA

`D(theta)`, the discrete divergence of the vector potential in link
variables, with **the same operator D the projection uses**.  It is the
quantity the Coulomb gauge keeps at zero, hence the direct check that the
projection works: it must stay at its initial value (zero if A starts at
zero) to machine precision, not "small".
"""
function link_divergence(s::State, p::Params)
    (; Nx, Ny, Nz, dx, dy, dz) = p
    d = zeros(Nx+1, Ny+1, Nz+1)
    @inbounds for k in 2:Nz, j in 2:Ny, i in 2:Nx
        d[i,j,k] = (s.thetax[i,j,k] - s.thetax[i-1,j,k])/dx^2 +
                   (s.thetay[i,j,k] - s.thetay[i,j-1,k])/dy^2 +
                   (s.thetaz[i,j,k] - s.thetaz[i,j,k-1])/dz^2
    end
    d
end
