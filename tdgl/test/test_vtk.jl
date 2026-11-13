# ---------------------------------------------------------------------------
#  VTK legacy ASCII.  Not compared against ParaView: against the CONTRACT of
#  the Octave writer (header, fields, crop to the sample, i-j-k order).
# ---------------------------------------------------------------------------

@testset "vtk: header and fields" begin
    cfg = cfg_fixture(out_dir="", verbose=false)
    p = derive_params(cfg); g = build_geometry(cfg, p)
    s = initial_state(p, g, 0.5, Strict())
    fn = joinpath(mktempdir(), "estado.vtk")
    write_vtk(fn, s, p; fmask=g.fmask)
    txt = read(fn, String)
    @test startswith(txt, "# vtk DataFile Version 3.0\n")
    @test occursin("ASCII\nDATASET STRUCTURED_POINTS\n", txt)
    nx = p.x2 - p.x1 + 2; ny = p.y2 - p.y1 + 2; nz = p.z2 - p.z1 + 2
    @test occursin("DIMENSIONS $nx $ny $nz", txt)
    @test occursin("POINT_DATA $(nx*ny*nz)", txt)
    @test occursin("SPACING 1 1 1", txt)
    for nom in ("psi_s", "psi_p", "rho_s", "dphi", "Bmag", "fmask")
        @test occursin("SCALARS $nom float 1", txt)
    end
    @test occursin("VECTORS B float", txt)
    @test !occursin("SCALARS Phi", txt)   # no Phi unless requested
end

@testset "vtk: spacing=:xi and optional Phi" begin
    cfg = cfg_fixture(out_dir="", verbose=false, use_phi=true)
    p = derive_params(cfg); g = build_geometry(cfg, p)
    s = initial_state(p, g, 0.5, Strict())
    sp = SolverPhi(p)
    fn = joinpath(mktempdir(), "xi.vtk")
    write_vtk(fn, s, p; spacing=:xi, Phi=sp.Phi)
    txt = read(fn, String)
    @test occursin("SPACING 0.5 0.5 0.5", txt)
    @test occursin("SCALARS Phi float 1", txt)
end

@testset "snap_every writes a PGM of the z plane when the point closes" begin
    sal = mktempdir()
    cfg = cfg_fixture(out_dir=sal, snap_every=1, vtk_every=0, mMax=3, iter_min=1,
                      t_min_GL=0.0, run_tag="snapprueba")
    run_case(cfg; mode=Fast())
    sd = joinpath(sal, "output", "snapshots")
    @test isdir(sd)
    files = readdir(sd)
    @test any(endswith(".pgm"), files)
    @test any(endswith(".csv"), files)
    @test isfile(joinpath(sd, "INDEX.txt"))
    pgm = read(joinpath(sd, first(f for f in files if endswith(f, ".pgm"))), String)
    @test startswith(pgm, "P2")
end

@testset "vtk: vtk_every writes when the point closes" begin
    sal = mktempdir()
    cfg = cfg_fixture(out_dir=sal, vtk_every=1, mMax=3, iter_min=1,
                      t_min_GL=0.0, run_tag="vtkprueba")
    run_case(cfg; mode=Fast())
    vtkdir = joinpath(sal, "output", "vtk")
    @test isdir(vtkdir)
    files = readdir(vtkdir)
    @test any(endswith(".vtk"), files)
    @test any(contains("J+"), files)
    @test any(contains("J-"), files)
    txt = read(joinpath(vtkdir, first(files)), String)
    @test occursin("SCALARS psi_s float 1", txt)
    @test occursin("SCALARS fmask float 1", txt)
end

@testset "vtk: from a stored checkpoint, without simulating" begin
    fout = joinpath(mktempdir(), "desde_archivo.vtk")
    fmat = joinpath(FIXT, "restart", "state_Jpos.mat")
    vtk_from_checkpoint(fmat, fout; buf=1, mr=1/(1.30^2 * 0.59), dx=0.5, dy=0.5, dz=0.5)
    txt = read(fout, String)
    @test startswith(txt, "# vtk DataFile Version 3.0\n")
    # bank 12x8x4, buf=1 => sample 13 x 9 x 5 nodes
    @test occursin("DIMENSIONS 13 9 5", txt)
    @test occursin("POINT_DATA 585", txt)
end

@testset "vtk: the crop does not interpolate (zeros where the staggered grid does not reach)" begin
# hx lives on (Nx+1, Ny, Nz); the sample window of nodes is larger in y,z.
# What is missing must be ZERO, not an invented value.
    cfg = cfg_fixture()
    p = derive_params(cfg)
    hx = ones(p.Nx+1, p.Ny, p.Nz)
    B = TDGL._recorta_B(hx, p.x2-p.x1+2, p.y2-p.y1+2, p.z2-p.z1+2,
                        p.x1, p.y1, p.z1)
    @test all(x -> x == 0.0 || x == 1.0, B)
    @test B[1, 1, 1] == 1.0
end

@testset "params: phi_bc / phi_method / vtk_every are validated early" begin
    @test_throws ErrorException derive_params(cfg_fixture(phi_bc=:robin))
    @test_throws ErrorException derive_params(cfg_fixture(phi_method=:fft))
    @test_throws ErrorException derive_params(cfg_fixture(vtk_every=-1))
end
