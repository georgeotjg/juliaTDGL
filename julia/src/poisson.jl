# ============================================================================
#  poisson.jl — v14: THE SCALAR POTENTIAL, ISOLATED.  STEPS 0, 1 and 2.
#
#  All of v13 runs in the gauge Phi = 0.  Article IV lifts that restriction.
#  The coupled solver is in scalar_potential.jl; this file keeps the isolated Poisson
#  problem and the measurements that decided how Phi enters.
#
#  --- THE EQUATION, DERIVED -------------------------------------------------
#      sigma (dA/dt + grad Phi) = J_s - kappa^2 curl curl A
#      div:  sigma d/dt(div A) + sigma lap Phi = div J_s     [div curl = 0]
#      =>    lap Phi = [ div J_s - sigma d/dt(div A) ] / sigma
#
#  and in the COULOMB GAUGE (div A = 0 maintained):
#
#      lap Phi = div J_s / sigma          <-- ONE term
#
#  An earlier planning note had `lap Phi = -div J_s - div(dA/dt)`, two terms
#  with the same sign, never derived.  Measuring the two terms showed they are
#  equal (ratio 0.993 on an archived state, with the .m operator).
#
#  --- WITH STAGGERED OPERATORS THE IDENTITY IS EXACT ------------------------
#  With the STAGGERED operators consistent with the kernel's curl,
#  `div(curl curl A)` is 5.6e-17 —machine zero— and then
#
#      div(dA/dt) = div(J_s)          to 3.6e-16 relative, ratio 1.0000000000
#
#  The 0.993 is an OPERATOR ARTEFACT: the .m applies a CENTRED difference in x
#  to a field that lives on the x links and leaves out the y and z
#  components.  With that mixture div(curl) does not vanish and the ratio
#  differs from state to state (0.993 in run 67, 1.225 in this port's
#  transport bank), the signature that the operator is the problem.  The
#  planning-note equation was EXACTLY -2 div J_s.
#
#  --- AND THEN THE RELATION DOES NOT DETERMINE Phi -------------------------
#  Measuring d/dt(div A) —one real substep, not an estimate— gives div J_s to
#  100.00 %, exactly what the equation for A requires.  Substituting,
#  `lap Phi = 0`: the relation is an IDENTITY OF THE SCHEME, not an equation
#  for Phi, and with Phi = 0 (the v13 gauge) it is satisfied by itself.
#
#  --- THE DESIGN DECISION ----------------------------------------------------
#  The kernel does NOT impose the Coulomb gauge: it evolves A freely.  The
#  two options are NOT symmetric:
#
#    (b) not imposing Coulomb and carrying d/dt(div A) is EMPTY: it gives
#        Phi = 0 always.  It is the gauge the code is already in.
#    (a) imposing div A = 0 is the only one that produces a non-trivial Phi,
#        and then Phi is not an extra field: IT IS THE PROJECTION.  Solving
#        lap Phi = div J_s and subtracting grad Phi from the A update is,
#        literally, the pressure step of an incompressible solver.
#
#  That reframes the cost (a projection method, with its literature) and the
#  risk (the BC of Phi becomes part of the definition of the problem).  See
#  `full_potential`, which measures the three things on the archive.
#
#  --- STATUS ----------------------------------------------------------------
#    [x] step 0  isolated Poisson against an exact solution.  Order 2.
#    [x] step 1  the problem in the VACUUM from an archived state.  Superseded
#                by step 2 (its equation was the planning-note one); kept
#                because its number is one of the three that expose the error.
#    [x] step 2  J_s inside the sample, the three equations solved and
#                compared.
#    [x] step 3  coupled in scalar_potential.jl: projection [C2] + rotation [A2].
#    [x] step 4  control: at stationary J=0, Phi constant (test_phi.jl).
#                With use_phi=false the kernel is still v13.3 bit for bit.
#
#  poisson.jl does not advance Psi or A.
#
#  --- ON THE LINEAR METHOD -------------------------------------------------
#  The matrix is CONSTANT: geometry and mesh do not change in time, so it is
#  factorised ONCE and each substep is only back-substitution.  Measured: at
#  7175 unknowns the stored LU takes 2.8 ms per solve, and a PCG with ichol
#  started from the previous Phi takes 25 ms, TEN TIMES MORE.  Do not start
#  with multigrid.
# ============================================================================

"""
    assemble_laplacian(nx, ny, nz, dx, dy, dz) -> SparseMatrixCSC

Seven-point Laplacian on `nx*ny*nz` INTERIOR nodes, with implicit zero
Dirichlet on the faces.  Kronecker product, as in the .m; the unknowns are in
column order (i fastest), which is the order of `reshape` in both Julia and
Octave.
"""
function assemble_laplacian(nx::Int, ny::Int, nz::Int,
                              dx::Float64, dy::Float64, dz::Float64)
    d2(n, h) = spdiagm(-1 => ones(n-1), 0 => fill(-2.0, n), 1 => ones(n-1)) ./ h^2
    Lx = d2(nx, dx); Ly = d2(ny, dy); Lz = d2(nz, dz)
    Ix = sparse(I, nx, nx); Iy = sparse(I, ny, ny); Iz = sparse(I, nz, nz)
    kron(Iz, kron(Iy, Lx)) + kron(Iz, kron(Ly, Ix)) + kron(Lz, kron(Iy, Ix))
end

# ---------------------------------------------------------------------------
#  STEP 0 — the isolated solve against an exact solution
# ---------------------------------------------------------------------------
"""
    poisson_convergence(; mallas=[8,16,24,32], L=20.0, io=stdout) -> Vector

`Phi = sin(ax)sin(by)sin(cz)` satisfies `lap Phi = -(a^2+b^2+c^2) Phi` with
zero Dirichlet on the six faces.  The Laplacian the kernel would use is
built, given that right-hand side and compared with the exact solution.  A
second-order scheme must give an error ~ dx^2.

The equation is checked BEFORE integrating it, not after.

Returns `[(mesh, dx, errmax, errrms, order, t_solve), ...]`.
"""
function poisson_convergence(; mallas=[8, 16, 24, 32], L::Float64=20.0, io=stdout)
    println(io, "\n  POISSON IN 3D, SEVEN POINTS, AGAINST sin(ax)sin(by)sin(cz)\n")
    @printf(io, "  %-18s %-9s %-12s %-12s %-10s %s\n",
            "mesh", "dx", "err max", "err rms", "order", "solve time")
    println(io, "  ", "-"^76)
    filas = []
    err_ant = NaN; dx_ant = NaN
    for N in mallas
        Nx = N; Ny = max(oround(N*0.6), 4); Nz = max(oround(N*0.3), 3)
        dx = L/Nx; dy = (L*0.6)/Ny; dz = (L*0.3)/Nz
        nx, ny, nz = Nx-1, Ny-1, Nz-1
        a = pi/L; b = pi/(L*0.6); c = pi/(L*0.3)
        Phi_ex = [sin(a*i*dx)*sin(b*j*dy)*sin(c*k*dz) for i in 1:nx, j in 1:ny, k in 1:nz]
        rhs = -(a^2 + b^2 + c^2) .* Phi_ex
        A = assemble_laplacian(nx, ny, nz, dx, dy, dz)
        t0 = time(); Phi = A \ vec(rhs); t1 = time() - t0
        e = Phi .- vec(Phi_ex)
        emax = maximum(abs, e); erms = sqrt(sum(abs2, e)/length(e))
        ord = isnan(err_ant) ? NaN : log(err_ant/emax)/log(dx_ant/dx)
        @printf(io, "  %-18s %-9.4f %-12.3e %-12.3e %-10s %.3f s\n",
                "$(Nx)x$(Ny)x$(Nz)", dx, emax, erms,
                isnan(ord) ? "   --  " : @sprintf("%6.2f", ord), t1)
        push!(filas, ("$(Nx)x$(Ny)x$(Nz)", dx, emax, erms, ord, t1))
        err_ant = emax; dx_ant = dx
    end
    filas
end

"""
    poisson_cost(Nx, Ny, Nz, dx; nrep=20, io=stdout) -> (n, tfac, tsol)

The number that decides whether v14 runs on a given machine: the cost of ONE
solve with the LU already factorised, against the cost of a production
point.  Measured on the target laptop (i5-1235U) with Octave:

    buffer   domain        unknowns     LU (once)      per solve   Poisson/point
      1     42x26x8            7175        0.18 s        0.0028 s      0.47 h
      4     48x32x14          18941        1.30 s        0.0075 s      1.26 h
      8     56x40x22          45045       11.47 s        0.0329 s      5.54 h

A production point costs ~0.4 h, so at the production buffer v14 costs a
little more than TWICE v13, not an order of magnitude.  At buffer 8 it no
longer does, and that is the real frontier: what costs is the vacuum, not
the scalar potential.
"""
function poisson_cost(Nx::Int, Ny::Int, Nz::Int, dx::Float64;
                       nrep::Int=20, io=stdout)
    nx, ny, nz = Nx-1, Ny-1, Nz-1
    A = assemble_laplacian(nx, ny, nz, dx, dx, dx)
    n = size(A, 1)
    rhs = randn(n)
    t0 = time(); F = lu(A); tfac = time() - t0
    Phi = F \ rhs                                    # warm-up
    t0 = time(); for _ in 1:nrep; Phi = F \ rhs; end; tsol = (time()-t0)/nrep
    @printf(io, "    unknowns                       : %d\n", n)
    @printf(io, "    LU factorisation (once)        : %.3f s\n", tfac)
    @printf(io, "    one solve with the stored LU   : %.4f s\n", tsol)
    @printf(io, "    at 15000 cycles x 40 substeps  : %.1f h of Poisson ALONE\n",
            tsol*15000*40/3600)
    (n, tfac, tsol)
end

# ---------------------------------------------------------------------------
#  PIECES SHARED BY STEPS 1 AND 2
# ---------------------------------------------------------------------------
"""
    links_from_theta(thetax) -> Vx

Rebuilds the links from the saved phase.  **It is `exp(-i theta)`, with a
minus sign.**

Not a matter of notation.  The kernel writes `Ux = exp(-i phix)` and then
`thetax = phix`, so that is the only reconstruction that returns the links
the solver had.  Checked in `test_poisson.jl` the hard way: restarting from
the checkpoint and verifying that 40 more substeps give the SAME BIT as
continuing from the state in memory.

With `exp(+i theta)` one gets `conj(Vx)`, and then `Q_x` comes out with the
opposite sign: `div J_s` flips sign and `V_Phi` with it.  The magnitude of
the step 2 result does not depend on this (norms do not see the sign), but
the SIGN of the Article IV observable does.
"""
links_from_theta(thetax::AbstractArray{Float64,3}) = @. cis(-thetax)

"""
    supercurrent_x(G1, G2, thetax, mr) -> Qx

The x component of the two-band supercurrent, **copied from block [B] of
the kernel, not rewritten**:

    Q_x = Im[conj(G1) Vx G1(+x)] + (1/mr) Im[conj(G2) Vx G2(+x)]

WHY COPIED.  The supercurrent of this code is not `Im(psi* grad psi)`: it is
the LINK-VARIABLE version, which makes the discretisation gauge invariant.
Writing it differently would give a SIMILAR and wrong number, and that kind
of substitution is exactly what produced two of the catalogue modes.

The ranges are those of block [B]: `1:Nx, 2:Ny, 2:Nz`.  An off-by-one would
be the same node-versus-cell error behind mode M5.
"""
function supercurrent_x(G1::AbstractArray{ComplexF64,3},
                          G2::AbstractArray{ComplexF64,3},
                          thetax::AbstractArray{Float64,3}, mr::Float64)
    Nx, Ny1, Nz1 = size(thetax)
    Ny = Ny1 - 1; Nz = Nz1 - 1
    Vx = links_from_theta(thetax)
    Qx = zeros(Nx, Ny, Nz)
    imr = 1/mr
    for k in 2:Nz, j in 2:Ny, i in 1:Nx
        Qx[i,j,k] = imag((conj(G1[i,j,k])*Vx[i,j,k])*G1[i+1,j,k]) +
               imr*imag((conj(G2[i,j,k])*Vx[i,j,k])*G2[i+1,j,k])
    end
    Qx
end

"""
    curl_x(hy, hz, dx) -> rot

`(curl curl A)_x = d_y h_z - d_z h_y`, with the ranges of block [C] of the
kernel.  It is the piece multiplied by `-kappa^2` in `dA/dt`.
"""
function curl_x(hy::AbstractArray{Float64,3}, hz::AbstractArray{Float64,3},
                      dx::Float64)
    Nx, Ny = size(hz, 1), size(hz, 2); Nz = size(hz, 3) - 1
    rot = zeros(Nx, Ny, Nz)
    for k in 2:Nz, j in 2:Ny, i in 1:Nx
        rot[i,j,k] = (hz[i,j,k] - hz[i,j-1,k])/dx - (hy[i,j,k] - hy[i,j,k-1])/dx
    end
    rot
end

"""
    divergence_x(F, dx) -> div

Centred difference in x, with zero on both faces.  It is the component the
transport moves; the other two enter the full problem, not this section,
which compares terms along the current axis.
"""
function divergence_x(F::AbstractArray{Float64,3}, dx::Float64)
    Nx, Ny, Nz = size(F)
    d = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 2:Nx-1
        d[i,j,k] = (F[i+1,j,k] - F[i-1,j,k])/(2dx)
    end
    d
end

"""
    vacuum_source(hy, hz, dx, ks2) -> (dAdt, divf)

STEP 1, kept.  `dA/dt|_x` in the VACUUM, where `J_s = 0` and
`dA/dt|_x = -kappa^2 (d_y h_z - d_z h_y)`, rebuilt from the `hy` and `hz`
that the saved states carry.

**Its equation (`lap Phi = -div dA/dt`) is NOT the correct one**; see the
header.  Kept because its number is one of the three that, put together by
step 2, exposed the error.
"""
function vacuum_source(hy::Array{Float64,3}, hz::Array{Float64,3},
                      dx::Float64, ks2::Float64)
    dAdt = -ks2 .* curl_x(hy, hz, dx)
    (dAdt, divergence_x(dAdt, dx))
end

"""
    solve_poisson(rhs, dx) -> (P, res, tsolve)

Solves `lap Phi = rhs` in the interior with zero Dirichlet on the faces and
returns the potential embedded in the full domain, the assembly residual
and the time.

The right-hand side enters COMPLETE and with its sign: there is no hidden
convention here.

What to look at, in this order:
  (a) the residual `max|lap Phi - rhs|` at machine precision; otherwise the
      assembly is wrong;
  (b) that Phi comes out BOUNDED and smooth — a huge Phi is the signature of
      a right-hand side incompatible with the boundary.
"""
function solve_poisson(rhs::Array{Float64,3}, dx::Float64)
    Nx, Ny, Nz = size(rhs)
    nx, ny, nz = Nx-2, Ny-2, Nz-2
    A = assemble_laplacian(nx, ny, nz, dx, dx, dx)
    b = vec(rhs[2:Nx-1, 2:Ny-1, 2:Nz-1])
    t0 = time(); Phi = A \ b; tsolve = time() - t0
    P = zeros(Nx, Ny, Nz)
    P[2:Nx-1, 2:Ny-1, 2:Nz-1] .= reshape(Phi, nx, ny, nz)
    (P, A*Phi .- b, tsolve)
end

"""
    divergence_A(thetax, thetay, thetaz, dx, dy, dz) -> divA

`div A` on the cells, with the same backward differences with which the
kernel defines the links.

It is what had to be measured to decide between options (a) and (b) of the
header: if `d/dt(div A)` is negligible against `div J_s` on real states,
option (b) is free and the Coulomb projection is not needed.  It is a
measurement on the archive, without running anything.
"""
function divergence_A(thetax::Array{Float64,3}, thetay::Array{Float64,3},
                       thetaz::Array{Float64,3}, dx::Float64, dy::Float64,
                       dz::Float64)
    Nx = size(thetax, 1); Ny = size(thetay, 2); Nz = size(thetaz, 3)
    dA = zeros(Nx+1, Ny+1, Nz+1)
    for k in 2:Nz, j in 2:Ny, i in 2:Nx
        dA[i,j,k] = (thetax[i,j,k] - thetax[i-1,j,k])/dx +
                    (thetay[i,j,k] - thetay[i,j-1,k])/dy +
                    (thetaz[i,j,k] - thetaz[i,j,k-1])/dz
    end
    dA
end

# ---------------------------------------------------------------------------
#  STEP 2 — THE POTENTIAL WITH THE SUPERCURRENT INSIDE
#
#  1. THE IDENTITY IS EXACT.  With the STAGGERED operators consistent with
#     the kernel's curl, `div(curl curl A)` is 5.6e-17 —machine zero— hence
#
#         div(dA/dt) = div(J_s)          to 3.6e-16 relative
#
#     The 0.993 (and the 1.225 of another state) is an ARTEFACT of mixing two
#     different divergences: the .m applies a CENTRED difference in x to a
#     field that lives on the x links and leaves out the y and z components.
#
#  2. THE RELATION DOES NOT DETERMINE Phi.  Measuring `d/dt(div A)` —one
#     kernel substep, not an estimate— gives
#
#         d/dt(div A) = div J_s          to 100.00 %
#
#     which is exactly what the kernel's equation for A requires:
#     dA/dt = J_s - kappa^2 curl curl A, and taking the divergence removes
#     the second term.  Substituting into
#
#         lap Phi = div J_s - sigma d/dt(div A)
#
#     gives `lap Phi = 0`.  THAT RELATION IS AN IDENTITY OF THE SCHEME, not an
#     equation for Phi.  With Phi = 0 —the gauge of v13— it holds by itself,
#     which is why v13 is consistent without solving anything.
#
#  3. WHAT IT MEANS FOR v14.  Phi is determined only by CHOOSING A GAUGE:
#
#       (b) "not imposing Coulomb and carrying d/dt(div A)" is EMPTY: it gives
#           Phi = 0 always.  It is the gauge the code is already in.
#       (a) imposing div A = 0 is the only one that produces a non-trivial
#           Phi.  Then Phi is not a field solved and then added: IT IS THE
#           PROJECTION.  Solving lap Phi = div J_s and subtracting grad Phi
#           from the A update is the pressure step of an incompressible
#           solver.
#
#     The BC of Phi then becomes part of the definition of the problem (see
#     the warning of `full_potential`).
#
#  WHAT THIS IS NOT.  Phi solved ONCE on a state integrated with Phi = 0 gives
#  the order of magnitude and the sign, not the number: in the coupled v14
#  Phi feeds back and the state differs.  And the PHYSICAL voltage of this
#  work is the Josephson one on the gauge-invariant phase, which does not
#  depend on the gauge; Phi alone, without the phase, is not invariant.
# ---------------------------------------------------------------------------

"""
    phi_fields(s, p, g) -> NamedTuple

Everything step 2 needs from a state, with the KERNEL operators: the three
components of the supercurrent (block [B]), `curl curl A` (block [C]),
`dA/dt`, and the staggered divergences of the last two.

`dA/dt` is rebuilt as block [C] writes it, with `tau` masking the
supercurrent and NOT the curl.  That asymmetric split is finding B2 (the
vacuum of this scheme is not an insulator) and it is respected here, not
"fixed": the aim is to measure what the code does.
"""
function phi_fields(s::State, p::Params, g::Geometry)
    Qx, Qy, Qz = supercurrent(s.G1, s.G2, s.thetax, s.thetay, s.thetaz, p.mr)
    Rx, Ry, Rz = curl_curl_A(s.hx, s.hy, s.hz, p)
    c = p.d1 / p.dt                      # = |alpha1| = (1-T).  This is M17.
    dAx = c .* (g.taux .* Qx) .- p.ks2 .* Rx
    dAy = c .* (g.tauy .* Qy) .- p.ks2 .* Ry
    dAz = c .* (g.tauz .* Qz) .- p.ks2 .* Rz
    (Qx=Qx, Qy=Qy, Qz=Qz, Rx=Rx, Ry=Ry, Rz=Rz, dAx=dAx, dAy=dAy, dAz=dAz,
     div_Js   = divergence(Qx, Qy, Qz, p.dx, p.dy, p.dz),
     div_dAdt = divergence(dAx, dAy, dAz, p.dx, p.dy, p.dz),
     div_rot  = divergence(Rx, Ry, Rz, p.dx, p.dy, p.dz))
end

"""
    full_potential(ruta, cfg; V_josephson=NaN, io=stdout) -> NamedTuple

Full STEP 2 on an archived state.  `cfg` must be the SAME configuration
that produced the state: `mr`, `kappa`, `dx` and the `tau` masks come from it,
and using others would give a similar and wrong number.

Prints, in this order:

  (a) the identity `div(curl curl A) = 0`, which must give machine zero;
  (b) `div(dA/dt)` against `div J_s`, which by (a) are the SAME number;
  (c) `d/dt(div A)` measured with a real substep, which by the equation for A
      is again `div J_s`  =>  `lap Phi = 0` in the v13 gauge;
  (d) the Phi of the Coulomb PROJECTION, `lap Phi = div J_s`, the only
      non-trivial one, and its drop along the current axis;
  (e) the same number with the `.m` operator (d/dx only, centred), for
      traceability and to see how much the choice of operator moves it.

**ON THE BOUNDARY CONDITIONS.**  Zero Dirichlet `Phi = 0` on the six faces is
used here, as in the Octave scripts.  For a projection on a domain with
current entering through the z faces, Neumann (`dPhi/dn = 0`) on the
insulating faces would be natural.  That choice CHANGES the potential drop;
scalar_potential.jl exposes it as `phi_bc`.
"""
function full_potential(ruta::AbstractString, cfg::Config;
                            V_josephson::Float64=NaN, io=stdout)
    p = derive_params(cfg); g = build_geometry(cfg, p); mode = Strict()
    s, _, _, _ = load_checkpoint(ruta, p, mode)
    F = phi_fields(s, p, g)
    nrm(x) = sqrt(sumlin(abs2(v) for v in x))

    @printf(io, "\n  state: %s\n  SC mesh %dx%dx%d, dx = %.2f, m_r = %.4f, kappa = %.2f\n",
            basename(ruta), cfg.Nx_sc, cfg.Ny_sc, cfg.Nz_sc, p.dx, p.mr, cfg.kappa_s)

    escala = max(nrm(F.Rx), nrm(F.Ry), nrm(F.Rz)) / p.dx
    @printf(io, "\n  (a) DISCRETE IDENTITY  div(curl curl A) = 0\n")
    @printf(io, "      max|div(curl curl A)| = %.3e   ||.||/scale = %.3e\n",
            maximum(abs, F.div_rot), nrm(F.div_rot)/max(escala, eps()))
    println(io, "      Machine zero.  With the staggered operators of the kernel the")
    println(io, "      identity is EXACT on the lattice, not only in the continuum.")

    nJ = nrm(F.div_Js); nA = nrm(F.div_dAdt)
    @printf(io, "\n  (b) div(dA/dt) AGAINST div J_s\n")
    @printf(io, "      ||div J_s||    = %.6e\n", nJ)
    @printf(io, "      ||div dA/dt||  = %.6e\n", nA)
    @printf(io, "      ratio          = %.10f\n", nJ/max(nA, eps()))
    @printf(io, "      ||dif||/||div J_s|| = %.3e\n",
            nrm(F.div_dAdt .- F.div_Js)/max(nJ, eps()))
    println(io, "      They are THE SAME TERM.  The equation")
    println(io, "      lap Phi = -div J_s - div(dA/dt) is exactly -2 div J_s.")

    HJ = p.inject_gain * cfg.J_list[end] * p.slab_d / (2*p.ks2)
    dd, _, _ = drift_div_A(s, p, g, mode; H=cfg.H_const, HJ=HJ,
                            J_cur=cfg.J_list[end])
    @printf(io, "\n  (c) d/dt(div A) MEASURED (one real substep of the kernel)\n")
    @printf(io, "      ||d/dt(div A)|| = %.6e   (%.2f %% of ||div J_s||)\n",
            nrm(dd), 100*nrm(dd)/max(nJ, eps()))
    println(io, "      => lap Phi = div J_s - d/dt(div A) = 0 in the Phi = 0 gauge.")
    println(io, "      The relation is an IDENTITY of the scheme, not an equation")
    println(io, "      for Phi.  Phi is fixed only by choosing a gauge; not imposing")
    println(io, "      Coulomb gives Phi = 0.")

    P, res, ts = solve_poisson(F.div_Js, p.dx)
    N1, N2, N3 = size(P); jc = oround(N2/2); kc = oround(N3/2)
    Vk = P[2,jc,kc] - P[N1-1,jc,kc]
    @printf(io, "\n  (d) THE COULOMB PROJECTION, lap Phi = div J_s\n")
    @printf(io, "      assembly residual = %.3e   cost %.4f s, %d unknowns\n",
            maximum(abs, res), ts, (N1-2)*(N2-2)*(N3-2))
    @printf(io, "      Phi bounded?  min %+.4e  max %+.4e\n", minimum(P), maximum(P))
    @printf(io, "      drop along x, mid plane: V_Phi = %+.5e\n", Vk)
    println(io, "      Phi is NOT an extra field that is added: it IS the projection")
    println(io, "      that keeps div A = 0, the pressure step of an incompressible")
    println(io, "      solver, with the same structure and the same cost.")
    println(io, "      BC used: Dirichlet Phi=0 on the six faces (as in the Octave code).")
    println(io, "      Neumann on the insulating faces would change this number; it is")
    println(io, "      a physical choice, not an implementation detail.")

    Qxm = supercurrent_x(s.G1, s.G2, s.thetax, p.mr)
    dJx = divergence_x(Qxm, p.dx)
    P2, _, _ = solve_poisson(dJx, p.dx)
    M1, M2, M3 = size(P2); j2 = oround(M2/2); k2 = oround(M3/2)
    Vm = P2[2,j2,k2] - P2[M1-1,j2,k2]
    @printf(io, "\n  (e) with the x-only operator (d/dx, centred), for comparison\n")
    @printf(io, "      ||div J_s||_(x only) = %.4e   against %.4e with the full one\n",
            nrm(dJx), nJ)
    @printf(io, "      V_Phi = %+.5e   against %+.5e\n", Vm, Vk)
    println(io, "      The difference is not noise: they are two different operators,")
    println(io, "      and only one of them annihilates div(curl).")

    if !isnan(V_josephson)
        @printf(io, "\n  (f) against the Josephson voltage of the SAME state\n")
        @printf(io, "      V_Josephson = %+.5e   V_Phi/V_Josephson = %+.4f\n",
                V_josephson, Vk/V_josephson)
        println(io, "      Phi solved once on a state integrated with Phi = 0 gives the")
        println(io, "      order of magnitude and the sign only.  The physical voltage is")
        println(io, "      the Josephson one on the gauge-invariant phase, which does not")
        println(io, "      depend on the gauge; Phi alone, without the phase, does.")
    end
    println(io)

    (V_coulomb=Vk, V_operador_m=Vm, norm_divJs=nJ, norm_divdAdt=nA,
     norm_ddivA=nrm(dd), max_div_rot=maximum(abs, F.div_rot),
     Phi=P, campos=F, t_solve=ts)
end


"""
    vacuum_from_file(ruta; dx=0.5, ks2=4.0, io=stdout)

STEP 1, kept as it was so that its number can be reproduced.  **Its equation
is not the correct one**; use `full_potential` for the right number.
"""
function vacuum_from_file(ruta::AbstractString; dx::Float64=0.5,
                             ks2::Float64=4.0, io=stdout)
    w = read_state(ruta)
    hy = Float64.(w["hy"]); hz = Float64.(w["hz"])
    Nx, Ny = size(hz, 1), size(hz, 2); Nz = size(hz, 3) - 1
    @printf(io, "\n  state: %s, cells %dx%dx%d, dx = %.2f\n",
            basename(ruta), Nx, Ny, Nz, dx)
    dAdt, divf = vacuum_source(hy, hz, dx, ks2)
    P, res, tsolve = solve_poisson(-divf, dx)
    @printf(io, "  (a) assembly residual : max|lap Phi - rhs| = %.3e\n",
            maximum(abs, res))
    @printf(io, "  (b) Phi bounded?  min = %+.4e   max = %+.4e\n",
            minimum(P), maximum(P))
    @printf(io, "      max |dA/dt| in the domain = %.4e\n", maximum(abs, dAdt))
    kc = oround(Nz/2); jc = oround(Ny/2)
    @printf(io, "  (c) drop along x, mid plane = %+.5e\n",
            P[2,jc,kc] - P[Nx-1,jc,kc])
    println(io, "      NOTE: SUPERSEDED equation (written down, not derived).")
    @printf(io, "  cost: %.4f s with %d unknowns\n\n", tsolve, (Nx-2)*(Ny-2)*(Nz-2))

    (P, dAdt, res)
end

# ---------------------------------------------------------------------------
#  THE FULL DIVERGENCE
# ---------------------------------------------------------------------------
"""
    supercurrent(G1, G2, thetax, thetay, thetaz, mr) -> (Qx, Qy, Qz)

The THREE components of block [B], not only x.  Same link convention as
the kernel (`V = exp(-i theta)`, see `links_from_theta`).
"""
function supercurrent(G1, G2, thetax, thetay, thetaz, mr::Float64)
    Nx = size(thetax, 1); Ny = size(thetay, 2); Nz = size(thetaz, 3)
    Vx = links_from_theta(thetax)
    Vy = links_from_theta(thetay)
    Vz = links_from_theta(thetaz)
    # `zeros`, not `similar`: entries no loop writes MUST be zero.  With
    # `similar` they are uninitialised memory, and although the divergence
    # reads only the written ones, any norm over the full array is
    # contaminated (seen: a NaN in Ry).
    Qx = zeros(Nx+1, Ny+1, Nz+1); Qy = zeros(Nx+1, Ny+1, Nz+1)
    Qz = zeros(Nx+1, Ny+1, Nz+1)
    imr = 1/mr
    for k in 2:Nz, j in 2:Ny, i in 1:Nx
        Qx[i,j,k] = imag((conj(G1[i,j,k])*Vx[i,j,k])*G1[i+1,j,k]) +
               imr*imag((conj(G2[i,j,k])*Vx[i,j,k])*G2[i+1,j,k])
    end
    for k in 2:Nz, j in 1:Ny, i in 2:Nx
        Qy[i,j,k] = imag((conj(G1[i,j,k])*Vy[i,j,k])*G1[i,j+1,k]) +
               imr*imag((conj(G2[i,j,k])*Vy[i,j,k])*G2[i,j+1,k])
    end
    for k in 1:Nz, j in 2:Ny, i in 2:Nx
        Qz[i,j,k] = imag((conj(G1[i,j,k])*Vz[i,j,k])*G1[i,j,k+1]) +
               imr*imag((conj(G2[i,j,k])*Vz[i,j,k])*G2[i,j,k+1])
    end
    (Qx, Qy, Qz)
end

"""
    divergence(Fx, Fy, Fz, dx, dy, dz) -> div

The natural STAGGERED divergence: each component lives on its link and is
differenced backwards to the node.

    div F|_(i,j,k) = (Fx[i]-Fx[i-1])/dx + (Fy[j]-Fy[j-1])/dy + (Fz[k]-Fz[k-1])/dz

Not a matter of taste.  With THESE operators —the same the kernel uses for
the curl— the identity `div(curl V) = 0` is EXACT on the lattice, not
approximate.  See `div_curl_identity`.
"""
function divergence(Fx, Fy, Fz, dx::Float64, dy::Float64, dz::Float64)
    n1, n2, n3 = size(Fx)
    d = zeros(n1, n2, n3)
    for k in 2:n3-1, j in 2:n2-1, i in 2:n1-1
        d[i,j,k] = (Fx[i,j,k] - Fx[i-1,j,k])/dx +
                   (Fy[i,j,k] - Fy[i,j-1,k])/dy +
                   (Fz[i,j,k] - Fz[i,j,k-1])/dz
    end
    d
end

"""
    curl_curl_A(hx, hy, hz, p) -> (Rx, Ry, Rz)

The `curl h` term of block [C], separated from the rest: it is what
multiplies `-kappa^2` in `dA/dt`.  Written with the SAME indices as the
kernel, reordering nothing, because the exactness of the identity below
depends on it.
"""
function curl_curl_A(hx, hy, hz, p::Params)
    (; Nx, Ny, Nz, dx, dy, dz) = p
    Rx = zeros(Nx+1, Ny+1, Nz+1); Ry = zeros(Nx+1, Ny+1, Nz+1)
    Rz = zeros(Nx+1, Ny+1, Nz+1)
    for k in 2:Nz, j in 2:Ny, i in 1:Nx
        Rx[i,j,k] = (hz[i,j,k] - hz[i,j-1,k])/dy - (hy[i,j,k] - hy[i,j,k-1])/dz
    end
    for k in 2:Nz, j in 1:Ny, i in 2:Nx
        Ry[i,j,k] = (hx[i,j,k] - hx[i,j,k-1])/dz - (hz[i,j,k] - hz[i-1,j,k])/dx
    end
    for k in 1:Nz, j in 2:Ny, i in 2:Nx
        Rz[i,j,k] = (hy[i,j,k] - hy[i-1,j,k])/dx - (hx[i,j,k] - hx[i,j-1,k])/dy
    end
    (Rx, Ry, Rz)
end

"""
    div_curl_identity(hx, hy, hz, p) -> (max_abs, rel)

Measures `div(curl h)` on a real state.  **It must come out zero to machine
precision**, the strong form of the step 2 result.

With the correct staggered divergence it is exact on the lattice, not only
in the continuum.  Hence

    div(dA/dt) = div(J_s) - kappa^2 div(curl curl A) = div(J_s)

IDENTICALLY inside the sample, and the planning-note equation
`lap Phi = -div J_s - div(dA/dt)` was exactly `-2 div J_s`, the correct term
multiplied by -2.
"""
function div_curl_identity(hx, hy, hz, p::Params)
    Rx, Ry, Rz = curl_curl_A(hx, hy, hz, p)
    d = divergence(Rx, Ry, Rz, p.dx, p.dy, p.dz)
    escala = max(maximum(abs, Rx), maximum(abs, Ry), maximum(abs, Rz))/p.dx
    (maximum(abs, d), maximum(abs, d)/max(escala, eps()))
end

"""
    drift_div_A(s, p, g, mode; H, HJ, J_cur) -> (ddivA, divA0, divA1)

`d/dt(div A)` measured: take `div A` of the state, advance ONE substep and
take it again.

**It is the number that decides between options (a) and (b)** of the header.
The exact equation is

    lap Phi = div J_s - sigma d/dt(div A)          (sigma = 1 in this code)

so if `||d/dt(div A)||` is negligible against `||div J_s||`, option (b) —not
imposing Coulomb— is free and `lap Phi = div J_s` holds as is.  Otherwise the
projection must be paid for, or the term carried.
"""
function drift_div_A(s::State, p::Params, g::Geometry, mode::NumMode;
                      H::Float64, HJ::Float64, J_cur::Float64)
    dA0 = divergence_A(s.thetax, s.thetay, s.thetaz, p.dx, p.dy, p.dz)
    substep!(s, p, g, mode; H=H, HJ=HJ, J_cur=J_cur)
    dA1 = divergence_A(s.thetax, s.thetay, s.thetaz, p.dx, p.dy, p.dz)
    ((dA1 .- dA0) ./ p.dt, dA0, dA1)
end
