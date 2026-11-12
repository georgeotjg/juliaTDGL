# ---------------------------------------------------------------------------
#  Geometry from a file (shapes.jl): a catalogue shape written as a file gives
#  the same mask; the mirror is the nodal one; the palette respects z; the
#  same shape on another mesh removes the same physical volume.
# ---------------------------------------------------------------------------
const FORMAS = joinpath(@__DIR__, "..", "shapes")
const KWF = (Nx_sc=40, Ny_sc=24, Nz_sc=6, buf=1, dx=0.5, dy=0.5, dz=0.5,
             kappa_s=25.0, kappa_p=25.0, xi_ratio=4.0, H_const=0.030,
             selftest=false, verbose=false, use_threads=false, out_dir="")

_fmask(; kw...) = (cfg = Config(:legacy; KWF..., kw...); p = derive_params(cfg);
                   (build_geometry(cfg, p).fmask, p))

@testset "shape: catalogue written as a file, to the bit" begin
    for (modo, archivo) in (("ESFERA", "sphere_as_ESFERA.toml"),
                            ("RECT_Y1", "rect_as_RECT_Y1.toml"),
                            ("TRI_Y1", "tri_as_TRI_Y1.toml"))
        f0, _ = _fmask(GEOM_MODE=modo)
        f1, _ = _fmask(GEOM_MODE="CUSTOM", geom_file=joinpath(FORMAS, archivo))
        @test f1 == f0
        @test count(<(1), f1) > 0
    end
end

@testset "shape: CUSTOM_REF is the nodal mirror" begin
    arch = joinpath(FORMAS, "example_pixels.toml")
    f, p = _fmask(GEOM_MODE="CUSTOM", geom_file=arch)
    fr, _ = _fmask(GEOM_MODE="CUSTOM_REF", geom_file=arch)
    (; x1, x2, y1, y2, z1, z2) = p
    for k in z1:z2, j in y1+1:y2, i in x1:x2
        @test fr[i, j, k] == f[i, (y1 + y2 + 1) - j, k]
    end
    g = Geometry(f, f, f, f, f, Defect[])
    @test check_my_symmetry(g, p) > 0          # the drawing breaks My
end

@testset "shape: the palette respects the z range" begin
    f, p = _fmask(GEOM_MODE="CUSTOM", geom_file=joinpath(FORMAS, "example_pixels.toml"))
    (; x1, y1, z1, z2, dx, dz) = p
    Lz = (z2 - z1) * dz
    # column x = 20 (index 21 of the row), y = 0: 'b' (lower half)
    i = x1 + 20; j = y1
    for k in z1:z2
        Z = (k - z1) * dz
        @test f[i, j, k] == (Z <= Lz / 2 + 1e-9 ? 0.0 : 1.0)
    end
    # 'x = 14, y = 0': '#' through the whole thickness
    @test all(==(0.0), f[x1 + 14, y1, z1:z2])
    # '5': f = 0.5
    @test f[x1 + 16, y1 + 20, z1] == 0.5
end

@testset "shape: another mesh, same volume" begin
    arch = joinpath(FORMAS, "rect_as_RECT_Y1.toml")
    fa, pa = _fmask(GEOM_MODE="CUSTOM", geom_file=arch)
    fb, pb = _fmask(GEOM_MODE="CUSTOM", geom_file=arch,
                    Nx_sc=80, Ny_sc=48, Nz_sc=12, dx=0.25, dy=0.25, dz=0.25)
    va = count(<(1), fa) * pa.dx^3
    vb = count(<(1), fb) * pb.dx^3
    @test isapprox(va, vb; rtol=0.35)          # nodes ≠ volume: order only
    @test vb < va                              # the edge weighs less on the fine mesh
end

@testset "shape: templates cat_*.toml = catalogue to the bit" begin
    cats = sort(filter(n -> startswith(n, "cat_") && endswith(n, ".toml"), readdir(FORMAS)))
    @test length(cats) >= 15
    for n in cats
        modo = n[5:end-5]
        f0, _ = _fmask(GEOM_MODE=modo)
        f1, _ = _fmask(GEOM_MODE="CUSTOM", geom_file=joinpath(FORMAS, n))
        @test f1 == f0
        # CUSTOM_REF is the exact nodal mirror (F18) of the template
        f0r, p = _fmask(GEOM_MODE=modo * "_REF")
        f1r, _ = _fmask(GEOM_MODE="CUSTOM_REF", geom_file=joinpath(FORMAS, n))
        (; x1, x2, y1, y2, z1, z2) = p
        I = x1+1:x2; Jj = y1+1:y2; K = z1:z2
        esp = [f0[i, y1 + y2 + 1 - j, k] for i in I, j in Jj, k in K]
        @test f1r[I, Jj, K] == esp
        # and equals the catalogue _REF.  Measured exception: the catalogue
        # FGRAD_REF is not the exact mirror (504 interior nodes), because
        # mirror_defects! reflects ya/yb/dir and the discrete phase of the ramp is
        # not symmetric.  The catalogue is kept as is.
        modo == "FGRAD" || @test f1r[I, Jj, K] == f0r[I, Jj, K]
    end
end

@testset "shape: voxels, any 3D pattern, round trip" begin
# 3D masks (drawing with partial z, f>1 of FGRAD, the wedge) → voxels → equal
    for (modo, arch) in (("CUSTOM", joinpath(FORMAS, "example_pixels.toml")),
                         ("FGRAD", ""), ("WEDGE", ""))
        f0, p = isempty(arch) ? _fmask(GEOM_MODE=modo) : _fmask(GEOM_MODE=modo, geom_file=arch)
        dir = mktempdir(joinpath(@__DIR__, ".."))     # inside julia/, not in /tmp
        try
            v = write_voxels(joinpath(dir, "v.toml"), f0, p; comentario="prueba")
            f1, _ = _fmask(GEOM_MODE="CUSTOM", geom_file=v)
            @test f1 == f0
        finally
            rm(dir; recursive=true)                    # temporary file of the test
        end
    end
end

@testset "shape: files in the first (Spanish) format are still read" begin
    # Runs made before the format was in English keep their copy forma.toml.
    viejo = """
    [[forma]]
    tipo = "pixeles"
    dx = 0.5
    filas = \"\"\"
    ......bb..
    ....###5..\"\"\"

    [forma.paleta]
    "#" = { z = "todo", f = 0.0 }
    "b" = { z = "abajo", f = 0.0 }
    "5" = { z = "todo", f = 0.5 }

    [[forma]]
    tipo = "triangulo"
    x = 12.0
    y = 0.0
    w = 4.0
    h = 3.0
    sesgo = 1.0
    z = "arriba"
    f = 0.0
    """
    nuevo = replace(viejo, "[[forma]]" => "[[shape]]", "[forma.paleta]" => "[shape.palette]",
                    "tipo = \"pixeles\"" => "type = \"pixels\"", "filas =" => "rows =",
                    "tipo = \"triangulo\"" => "type = \"triangle\"", "sesgo =" => "skew =",
                    "\"todo\"" => "\"all\"", "\"abajo\"" => "\"bottom\"", "\"arriba\"" => "\"top\"")
    dir = mktempdir(joinpath(@__DIR__, ".."))
    try
        a = joinpath(dir, "old.toml"); b = joinpath(dir, "new.toml")
        write(a, viejo); write(b, nuevo)
        fa, _ = _fmask(GEOM_MODE="CUSTOM", geom_file=a)
        fb, _ = _fmask(GEOM_MODE="CUSTOM", geom_file=b)
        @test fa == fb
        @test count(<(1), fa) > 0
    finally
        rm(dir; recursive=true)
    end
end

@testset "shape: a figure on another face (axis), kept (outside); vacuum is release 2" begin
    dir = mktempdir(joinpath(@__DIR__, ".."))
    try
        a = joinpath(dir, "axis_x.toml")
        write(a, "[[shape]]\ntype = \"circle\"\naxis = \"x\"\nu = 6.0\nv = 1.25\nr = 1.0\noutside = true\nf = 0.0\n")
        f, p = _fmask(GEOM_MODE="CUSTOM", geom_file=a)
        (; x1, x2, y1, y2, z1, z2, dx, dz) = p
        esp = [((j - y1) * dx - 6.0)^2 + ((k - z1) * dz - 1.25)^2 <= 1.0 + 1e-9 ? 1.0 : 0.0 for i in x1:x2, j in y1:y2, k in z1:z2]
        @test f[x1:x2, y1:y2, z1:z2] == esp
        b = joinpath(dir, "vac.toml")
        write(b, "[[shape]]\ntype = \"circle\"\nx = 10.0\ny = 6.0\nr = 2.0\nvacuum = true\n")
        @test_throws ErrorException _fmask(GEOM_MODE="CUSTOM", geom_file=b)
    finally
        rm(dir; recursive=true)
    end
end
