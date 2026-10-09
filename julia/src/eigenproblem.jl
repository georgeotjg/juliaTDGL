# ============================================================================
#  eigenproblem.jl — THE EXTERNAL VALIDATION OF THE PROJECT
#
#  WHAT IT IS AND IS NOT.  It is not a simulation and does not touch the
#  solver.  It is linear algebra: the SAME discrete operator the kernel uses
#  —the same link variables, the same second-order difference, the same
#  Neumann condition— is built and its eigenvalues are computed.  Seconds,
#  not hours.
#
#  WHY IT MATTERS.  Everything else that accompanies the port is INTERNAL
#  validation: the acceptance test measures the port against Octave, the
#  selftest the fast kernel against the slow one, the SIM control the mesh
#  against its mirror.  This checks the STRUCTURE of the equations against
#  exact results that exist OUTSIDE the project:
#
#    step 1  the lowest Landau level on the lattice:  lambda_0/H -> 1
#            with the discretisation law  lambda_0/H = 1 - phi/8 + O(phi^2)
#    step 2  two-band Hc2 against Eq.(49) of the paper
#    step 3  the Saint-James--de Gennes constant  Hc3/Hc2 = 1.6946
#
#  Steps 1 and 3 are the only two numbers of the whole work checked against
#  an exact external result without fitting anything.
#
#  Step 1 also runs the test the solver itself cannot: the canonical test
#  Hc2 = 1 needs periodic boundaries, and every face of the solver is a
#  superconductor-vacuum interface.  This is not the solver: here periodic
#  boundaries cost three lines.
# ============================================================================

# ---------------------------------------------------------------------------
#  STEP 1 — the covariant Laplacian, one band, periodic lattice
# ---------------------------------------------------------------------------
"""
    covariant_laplacian(Nx, Ny, dx, dy, phi) -> SparseMatrixCSC

Discrete covariant Laplacian on a PERIODIC `Nx x Ny` lattice with flux `phi`
per plaquette.  `-L` is positive semidefinite and its lowest eigenvalue is
the lowest Landau level of the lattice.

**The link convention is that of block [A] of the kernel**, copied, not
reinvented:

    (L psi)(i) = [ Vx(i) psi(i+1) - 2 psi(i) + conj(Vx(i-1)) psi(i-1) ] / dx^2

with `Vx = exp(-i theta_x)`.  If this departed from the solver, step 2 would
prove nothing about the solver.

**The gauge** is Landau's, with the twist that closes the torus:

    theta_y(i,j) = phi*(i-1)     for all i
    theta_x(i,j) = 0             except at i = Nx, where it is -phi*Nx*(j-1)

The phase per plaquette is `phi` on ALL of them, including the one across the
border.  Flux quantisation forces `phi*Nx*Ny = 2*pi*m`, so H is not free on a
periodic lattice: the allowed sequence must be taken.
"""
function covariant_laplacian(Nx::Int, Ny::Int, dx::Float64, dy::Float64,
                               phi::Float64)
    idx(i, j) = i + (j-1)*Nx                  # column order
    envol(k, N) = mod(k-1, N) + 1
    thx = zeros(Nx, Ny); thy = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        thy[i,j] = phi*(i-1)
        i == Nx && (thx[i,j] = -phi*Nx*(j-1))
    end
    Vx = @. cis(-thx); Vy = @. cis(-thy)

    I = Int[]; J = Int[]; S = ComplexF64[]
    for j in 1:Ny, i in 1:Nx
        n = idx(i, j)
        push!(I, n); push!(J, n);                          push!(S, -2/dx^2 - 2/dy^2)
        push!(I, n); push!(J, idx(envol(i+1,Nx), j));       push!(S, Vx[i,j]/dx^2)
        ip = envol(i-1, Nx)
        push!(I, n); push!(J, idx(ip, j));                  push!(S, conj(Vx[ip,j])/dx^2)
        push!(I, n); push!(J, idx(i, envol(j+1,Ny)));       push!(S, Vy[i,j]/dy^2)
        jp = envol(j-1, Ny)
        push!(I, n); push!(J, idx(i, jp));                  push!(S, conj(Vy[i,jp])/dy^2)
    end
    sparse(I, J, S, Nx*Ny, Nx*Ny)
end

"""
    landau_level(Nx, Ny, dx, dy, m) -> (H, lambda0, lambda0_sobre_H, hermiticidad)

The lowest Landau level of the lattice for the `m`-th allowed flux.

In code units the lowest level of `-(grad - iA)^2` equals H, so the
instability arrives at H = 1: **that** is the statement Hc2 = 1.  The
deviation of `lambda_0/H` from 1 is the DISCRETISATION ERROR of the
covariant Laplacian at that mesh, measured rather than assumed.
"""
function landau_level(Nx::Int, Ny::Int, dx::Float64, dy::Float64, m::Int)
    phi = 2pi*m/(Nx*Ny)
    H = phi/(dx*dy)
    L = covariant_laplacian(Nx, Ny, dx, dy, phi)
    herm = norm(Matrix(L - L'), 1)
    A = -(L + L')/2                            # symmetrised for safety
    lam = minimum(real, eigvals(Hermitian(Matrix(A))))
    (H, lam, lam/H, herm)
end

"""
    step1_one_band(; Nx=40, Ny=40, dx=0.5, dy=0.5, ms=[10,20,40,60,64,80,100], io=stdout)

Full STEP 1.  Returns the table and checks it against the discretisation law
of the continuum

    lambda_0/H = 1 - phi/8 + O(phi^2),      phi = H*dx*dy

which is an EXTERNAL result: it comes from no fit of this project.
"""
function step1_one_band(; Nx::Int=40, Ny::Int=40, dx::Float64=0.5,
                         dy::Float64=0.5, ms=[10,20,40,60,64,80,100], io=stdout)
    @printf(io, "\n  LATTICE %dx%d, dx=dy=%.3f xi_sigma.  Quantised flux: H = 2*pi*m/(dx*dy*Nx*Ny)\n\n",
            Nx, Ny, dx)
    @printf(io, "    %-5s %-11s %-12s %-13s %-13s %s\n",
            "m", "H (Hc2)", "lambda_0", "lambda_0/H", "1 - phi/8", "hermitian?")
    println(io, "   ", "-"^76)
    filas = []
    for m in ms
        H, lam, r, herm = landau_level(Nx, Ny, dx, dy, m)
        phi = H*dx*dy
        @printf(io, "    %-5d %-11.5f %-12.6f %-13.7f %-13.7f %.2e\n",
                m, H, lam, r, 1 - phi/8, herm)
        push!(filas, (m=m, H=H, lambda0=lam, ratio=r, ley=1-phi/8, herm=herm))
    end
    println(io, """
  Reading: lambda_0/H = 1 is the exact lowest Landau level, i.e. Hc2 = 1 in
  these units.  The deviation is the MESH error, and the column '1 - phi/8'
  is the continuum law it must agree with.""")
    filas
end

# ---------------------------------------------------------------------------
#  STEP 2 — two bands, Hc2 against Eq.(49)
# ---------------------------------------------------------------------------
"""
    hc2_block(a2n, mr, g12; gr=1.0, esc_cinetico=1.0, esc_acoplo=1.0) -> H

Largest root of the instability condition of the 2x2 block at the lowest
Landau level (`L -> -H`), by bisection:

    [ 1 - H                    g12*ea          ]
    [ g12*ea        -H/(mr*ek) + a2n/gr         ]

`ek` and `ea` apply the Gamma_r factor also to the kinetic and coupling
terms.  **MIND THE CONVENTION:** to MULTIPLY the kinetic term by `1/gr`, pass
`ek = gr`, because it enters as a divisor.  See `step2_two_bands`.
"""
function hc2_block(a2n::Float64, mr::Float64, g12::Float64; gr::Float64=1.0,
                    esc_cinetico::Float64=1.0, esc_acoplo::Float64=1.0)
    f(H) = maximum(real, eigvals([1 - H                                 g12*esc_acoplo;
                                  g12*esc_acoplo   -H/(mr*esc_cinetico) + a2n/gr]))
    lo = 0.01; hi = 20.0
    for _ in 1:200
        mid = (lo + hi)/2
        f(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    (lo + hi)/2
end

"""
    hc2_scheme(a2n, mr, g12, gr) -> H

The two-band Hc2 integrated by **v13.3**, where Gamma_r divides the THREE
terms of the pi band.  The linearised matrix is

    [ 1 - H              g12          ]
    [ g12/Gr     (a2n - H/mr)/Gr      ]

and its determinant is

    det = (1/Gr) [ (1-H)(a2n - H/mr) - g12^2 ]

**so the root does NOT depend on Gamma_r and agrees EXACTLY with Eq.(49) of
the paper, for every Gamma_r.**  That is not a numerical coincidence: a global
factor `1/Gr` comes out of the determinant.
"""
function hc2_scheme(a2n::Float64, mr::Float64, g12::Float64, gr::Float64)
    f(H) = maximum(real, eigvals([1 - H       g12;
                                  g12/gr     (a2n - H/mr)/gr]))
    lo = 0.01; hi = 20.0
    for _ in 1:200
        mid = (lo + hi)/2
        f(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    (lo + hi)/2
end

"""
    step2_two_bands(; a2n=0.59, mr=1.0029, g12s=[1e-3,1e-2,1e-1],
                       grs=[0.25,0.5,1.0,1.3,2.0,4.0], io=stdout)

STEP 2.  Compares four things on the same block:

| columna | modelo |
|---|---|
| `Ec.(49)` | `(1-H)(a2n - H/mr) = g12^2`, no Gamma_r |
| `v13.2` | Gamma_r ONLY in the condensation — mode M16 |
| `v13.3` | Gamma_r in all three terms — the scheme this port reproduces |
| `script ek=1/gr` | the «INTENDED» column of the `.m`, for traceability |

**TWO POINTS WHERE THIS PORT DIFFERS FROM THE OCTAVE SCRIPT.**

1. The «INTENDED model» column of the `.m` passes `escala_cinetico = 1/gr`,
   but `ek` enters as a DIVISOR (`-H/(mr*ek)`), so to multiply the kinetic
   term by `1/gr` one must pass `ek = gr`.  With `1/gr` the column gives 9.47
   at Gamma_r = 0.25 instead of 1.000002: **it did not represent the model it
   meant to represent**.

2. With that fixed, the clean result appears: under v13.3 Eq.(49) is **exact
   for every Gamma_r**, not only at Gamma_r = 1.  "Only at Gamma_r = 1" is
   true of the v13.2 scheme; with the v13.3 fix it holds always.
"""
function step2_two_bands(; a2n::Float64=0.59, mr::Float64=1.0029,
                          g12s=[1e-3, 1e-2, 1e-1],
                          grs=[0.25, 0.5, 1.0, 1.3, 2.0, 4.0], io=stdout)
    @printf(io, "\n  a2n=%.3f  m_r=%.4f.  Hc2 in units of the nominal Hc2^sigma.\n", a2n, mr)
    filas = []
    for g12 in g12s
        @printf(io, "\n  gamma12 = %g\n", g12)
        @printf(io, "    %-9s %-14s %-14s %-14s %-16s %s\n",
                "Gamma_r", "two-band", "F21 scheme", "this scheme",
                "script ek=1/gr", "F21/two-band")
        println(io, "    ", "-"^92)
        Hp = hc2_block(a2n, mr, g12)
        for gr in grs
            H2 = hc2_block(a2n, mr, g12; gr=gr)                       # v13.2
            H3 = hc2_scheme(a2n, mr, g12, gr)                            # v13.3
            Hs = hc2_block(a2n, mr, g12; gr=gr, esc_cinetico=1/gr,
                            esc_acoplo=1/sqrt(gr))                     # the .m
            @printf(io, "    %-9.2f %-14.6f %-14.6f %-14.6f %-16.6f %.4f\n",
                    gr, Hp, H2, H3, Hs, H2/Hp)
            push!(filas, (g12=g12, gr=gr, paper=Hp, v132=H2, v133=H3, script=Hs))
        end
    end
    println(io, """
  Reading.  The two-band column does NOT depend on Gamma_r: it cannot, since
  Gamma multiplies d/dt and Hc2 is static.  The M16 column DOES depend on it
  (that is mode M16), and the column of this scheme agrees with the two-band
  formula again, exactly and for every Gamma_r, because
  det M = (1/Gr)*[the two-band expression].  The last column reproduces the
  Octave eigen script as it is, whose kinetic scale parameter is inverted
  (see the docstring).""")
    filas
end

# ---------------------------------------------------------------------------
#  STEP 3 — the boundary:  Hc3/Hc2 = 1.6946
# ---------------------------------------------------------------------------
"""
    hc3_over_hc2(dx; H=1.0, L=12.0, paso_x0=nothing) -> (Hc3_Hc2, lambda_min, x0)

The Saint-James--de Gennes constant, measured on **the boundary as the solver
implements it**.

Every face of this solver is a superconductor-vacuum interface with Neumann
on Psi.  That condition has a textbook consequence: surface
superconductivity survives above Hc2, up to

    Hc3 / Hc2 = 1.6946...      (Saint-James -- de Gennes, 1963)

the root of a one-dimensional eigenproblem that **depends on no material
parameter**.  With the Landau level of step 1, it is one of the only two
numbers of this work checked against an exact external result.

SETUP.  Semi-infinite strip, Neumann at x=0, uniform field.  In the Landau
gauge the problem separates and for each momentum k it reads

    -psi'' + H^2 (x - k/H)^2 psi = lambda psi ,     psi'(0) = 0

a harmonic oscillator centred at `x0 = k/H` and cut at x=0.  With `x0` deep
inside, the Landau level is recovered (lambda = H); with `x0` near the surface
lambda DROPS, and the minimum over `x0` gives Hc3.

**The Neumann condition is that of block [H] of the kernel**: `psi(face) =
psi(interior node)`, so the first row of the Laplacian loses its outer
neighbour and has `-1` instead of `-2`.  This measures the IMPLEMENTED
boundary, not an idealisation of it.
"""
function hc3_over_hc2(dx::Float64; H::Float64=1.0, L::Float64=12.0,
                       paso_x0=nothing)
    N = oround(L/dx)
    x = collect(0:N-1) .* dx                 # x=0 is the surface
    h = paso_x0 === nothing ? dx/4 : paso_x0
    mejor = Inf; x0mejor = NaN
    e = ones(N-1)
    T = spdiagm(-1 => -e, 0 => fill(2.0, N), 1 => -e) ./ dx^2
    T[1,1] = 1/dx^2                          # kernel block [H]: psi(0) = psi(1)
    for x0 in 0.0:h:3.0
        A = T + spdiagm(0 => H^2 .* (x .- x0).^2)
        lam = minimum(real, eigvals(Hermitian(Matrix((A + A')/2))))
        if lam < mejor
            mejor = lam; x0mejor = x0
        end
    end
    (H/mejor, mejor, x0mejor)
end

"""
    step3_boundary(; dxs=[0.5,0.25,0.125,0.0625], io=stdout)

Full STEP 3, refining the mesh.  The exact value is
`Hc3/Hc2 = 1.69461`, or `lambda/H = 0.59010` at the minimum.
"""
function step3_boundary(; dxs=[0.5, 0.25, 0.125, 0.0625], io=stdout)
    const_exacta = 1.69461
    @printf(io, "\n  BOUNDARY: Hc3/Hc2 with the Neumann condition of block [H]\n\n")
    @printf(io, "  %-9s %-6s %-14s %-10s %-13s %s\n",
            "dx", "N", "min lambda", "at x0", "Hc3/Hc2", "error vs 1.69461")
    println(io, "  ", "-"^76)
    filas = []
    for dx in dxs
        r, lam, x0 = hc3_over_hc2(dx)
        @printf(io, "  %-9.4f %-6d %-14.6f %-10.3f %-13.4f %+.2f %%\n",
                dx, oround(12.0/dx), lam, x0, r, 100*(r - const_exacta)/const_exacta)
        push!(filas, (dx=dx, lambda=lam, x0=x0, ratio=r))
    end
    println(io, """
  Exact Saint-James -- de Gennes value:  Hc3/Hc2 = 1.69461
  (lambda/H = 0.59010 at the minimum).  It depends on no material parameter,
  so the deviation is mesh and boundary error only.""")
    filas
end

"""
    eigenproblem(; io=stdout)

The three steps in a row.  It is the EXTERNAL validation of the port: steps 1
and 3 check against exact results that do not come from this project.
"""
function eigenproblem(; io=stdout)
    println(io, "\n", "="^78)
    println(io, " LINEAR EIGENPROBLEM — validation against exact results")

    println(io, "="^78)
    r1 = step1_one_band(io=io)
    r2 = step2_two_bands(io=io)
    r3 = step3_boundary(io=io)
    (paso1=r1, paso2=r2, paso3=r3)
end
