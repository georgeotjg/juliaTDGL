# The open boundary (:open, openbc.jl).

@testset "lattice Green function" begin
    G = TDGL.lattice_green(8)
    @test G[1, 1, 1] ≈ 0.2527310098 rtol=1e-8                 # Watson's integral / 2
    @test G[2, 1, 1] ≈ G[1, 1, 1] - 1/6 rtol=1e-8              # −Δ_h G = δ at the origin
    @test G[3, 2, 1] ≈ G[2, 3, 1] && G[3, 2, 1] ≈ G[1, 2, 3]   # cubic symmetry
    # −Δ_h G = 0 off the origin, and the far form matches at the edge of the table
    @test abs(6G[3, 2, 2] - (G[4, 2, 2] + G[2, 2, 2] + 2G[3, 3, 2] + 2G[3, 1, 2])) < 1e-8
    @test isapprox(G[9, 9, 1], TDGL._lgf_far(8, 8, 0); rtol=2e-4)
end

# No superconductivity: once the applied field has diffused in through the normal metal (σL²/κ² ~ 4 τ),
# nothing is screened, h_δ → 0 and the field is H everywhere.
@testset ":open without currents" begin
    md = Model((Band(a=0.0),), zeros(1, 1); κ=2.0, T=0.5)      # ε < 0: normal metal
    gr = Grid(8, 6, 4; dx=0.5)
    st = Stepper(md, gr, box_masks(gr); threads=false, selffield_bc=:open, open_every=0.1)
    s = initial_state(md, gr, st.masks, 0.03)
    for _ in 1:round(Int, 80 / st.dt); step!(s, st; H=0.03, HJ=0.0); end
    @test maximum(abs, st.open.new[3]) < 1e-6 && maximum(abs, s.hz .- 0.03) < 1e-6
end

# With a current: the source is closed (no δJ on the face-normal links, divergence free inside), the
# total current through the mid plane is the nominal, and the iteration settles.
@testset ":open with a current" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(16, 8, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="SIM"), gr)
    st = Stepper(md, gr, mk; threads=false, selffield_bc=:open)
    s = initial_state(md, gr, mk, 0.03); HJ = selffield(st, 0.3)
    for _ in 1:round(Int, 40 / st.dt); step!(s, st; H=0.03, HJ=HJ); end
    It, Is = midplane_current(s, st)
    @test isapprox(abs(It), 0.3 * slab_thickness(gr) * (gr.y2 - gr.y1 + 1) * gr.dy; rtol=2e-3)
    ob = st.open; bx, by, bz = ob.buf; (; Nx, Ny, Nz, dx) = gr
    dmax = maximum(abs((bx[i,j,k] - bx[i-1,j,k]) + (by[i,j,k] - by[i,j-1,k]) + (bz[i,j,k] - bz[i,j,k-1]))
                   for i in 3:Nx-1, j in 3:Ny-1, k in 3:Nz-1)
    @test dmax < 1e-8 * maximum(abs, bx)
    old = copy(ob.new[1]); for _ in 1:round(Int, 2 / st.dt); step!(s, st; H=0.03, HJ=HJ); end
    @test maximum(abs, ob.new[1] .- old) < 1e-3 * maximum(abs, old)
end

# At H = 0 the two polarities are conjugate (Ψ → Ψ*, θ → −θ, h → −h, J → −J) on any geometry, so the
# diode of the notch cannot come from the boundary: the lag and the ramp of :open must keep it exactly.
@testset ":open keeps the J → −J symmetry at H = 0" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(16, 8, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE"), gr)
    run(J) = (st = Stepper(md, gr, mk; threads=false, selffield_bc=:open, open_every=0.5);
              s = initial_state(md, gr, mk, 0.0); HJ = selffield(st, J);
              for _ in 1:round(Int, 10 / st.dt); step!(s, st; H=0.0, HJ=HJ); end; s)
    a = run(0.3); b = run(-0.3)
    @test all(a.G[µ] == conj.(b.G[µ]) for µ in 1:2)
    @test a.θx == -b.θx && a.θy == -b.θy && a.θz == -b.θz && a.hz == -b.hz
    @test maximum(abs, a.hz) > 1e-3                              # the current did make a field
end

# Resuming a :open run with its ramp (open_extra / restore_open!) is continuing it, to the bit.
@testset ":open resumes to the bit" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(16, 8, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="WEDGE"), gr)
    mkst() = Stepper(md, gr, mk; threads=false, selffield_bc=:open, open_every=0.5)
    n = round(Int, 6 / mkst().dt) + 7                       # the save falls inside a ramp
    st = mkst(); HJ = selffield(st, 0.3)
    run(s, st, k) = (for _ in 1:k; step!(s, st; H=0.03, HJ=HJ); end; s)
    a = run(initial_state(md, gr, mk, 0.03), st, 2n)
    st = mkst(); b = run(initial_state(md, gr, mk, 0.03), st, n)
    path = joinpath(mktempdir(), "open.jld2"); save_state(path, b; open_extra(st)...)
    st2 = mkst(); c, w = load_state(path, gr); restore_open!(st2, w)
    run(c, st2, n)
    @test c.G[1] == a.G[1] && c.G[2] == a.G[2] && c.hz == a.hz && c.θx == a.θx
end

# A sample with the mirror y → −y (SIM) at H ≠ 0: the state at +J is the mirror of the state at −J (C∘M_y), so η = 0
# to rounding, with each self-field boundary.
@testset "mirror-symmetric sample: +J is the mirror of −J" begin
    md = two_band(xi_ratio=4.0, κ=2.0)
    gr = Grid(16, 8, 4; dx=0.5)
    mk = build_masks(GeomSpec(GEOM_MODE="SIM"), gr)
    @test check_my_symmetry(mk, gr) == 0
    for bc in (:slab, :bar, :open)
        run(J) = (st = Stepper(md, gr, mk; threads=false, selffield_bc=bc, open_every=0.5);
                  s = initial_state(md, gr, mk, 0.05); HJ = selffield(st, J);
                  for _ in 1:round(Int, 10 / st.dt); step!(s, st; H=0.05, HJ=HJ); end; s)
        a = run(0.3); b = run(-0.3); ny = size(a.G[1], 2)
        @test all(maximum(abs.(abs.(a.G[µ]) .- abs.(b.G[µ][:, ny:-1:1, :]))) < 1e-12 for µ in 1:2)
        @test maximum(abs.(abs.(a.G[1]) .- abs.(b.G[1]))) > 1e-4      # without the mirror the two differ
    end
end
