# Masks against juliaTDGL 1.2, to the bit: the whole catalogue, its mirrors, z modes, shifts and
# every shape file.  Golden masks: test/golden/make_masks.jl.
using JLD2

const SHAPES4 = joinpath(@__DIR__, "..", "shapes")

@testset "geometry against 1.2" begin
    casos = JLD2.load(joinpath(@__DIR__, "golden", "masks.jld2"), "casos")
    nbad = 0
    for (nombre, c) in casos
        startswith(nombre, "_") && continue
        m = c["malla"]
        kw = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in c["geo"])
        haskey(kw, :geom_file) && (kw[:geom_file] = joinpath(SHAPES4, kw[:geom_file]))
        gr = Grid(m["Nx_sc"], m["Ny_sc"], m["Nz_sc"]; dx=m["dx"], buf=m["buf"])
        mk = build_masks(GeomSpec(; kw...), gr)
        ok = mk.f == c["fmask"]
        ok || (nbad += 1; @info "mask differs" nombre)
        @test ok
    end
    @test nbad == 0
    b = casos["_box_40x24x6"]
    mk = box_masks(Grid(40, 24, 6; dx=0.5))
    @test mk.τx == b["taux"] && mk.τy == b["tauy"] && mk.τz == b["tauz"] && mk.τψ == b["taupsi"]
end

@testset "geometry: symmetries of the catalogue" begin
    gr = Grid(40, 24, 6; dx=0.5)
    @test check_my_symmetry(build_masks(GeomSpec(GEOM_MODE="SIM"), gr), gr) == 0
    @test check_my_symmetry(build_masks(GeomSpec(GEOM_MODE="ESFERA"), gr), gr) == 0
    @test check_my_symmetry(build_masks(GeomSpec(GEOM_MODE="WEDGE"), gr), gr) > 0
end

# Figures on any face (axis), the figure kept (outside), and vacuum (a sculpted sample).
@testset "shapes: axis, outside and vacuum" begin
    gr = Grid(20, 12, 12; dx=0.5, buf=2)
    (; x1, x2, y1, y2, z1, z2, dx) = gr
    dir = mktempdir()
    forma(txt) = (p = joinpath(dir, "f$(hash(txt)).toml"); write(p, txt); GeomSpec(GEOM_MODE="CUSTOM", geom_file=p))
    # a disc on the y–z plane through the whole sample, f = 0.3 inside
    mk = build_masks(forma("[[shape]]\ntype = \"circle\"\naxis = \"x\"\nu = 3.0\nv = 2.5\nr = 1.5\nf = 0.3\n"), gr)
    esp = [((j - y1) * dx - 3.0)^2 + ((k - z1) * dx - 2.5)^2 <= 1.5^2 + 1e-9 ? 0.3 : 1.0 for i in x1:x2, j in y1:y2, k in z1:z2]
    @test mk.f[x1:x2, y1:y2, z1:z2] == esp
    @test mk.vac === nothing
    # outside: the same disc kept, the rest at f = 0.3
    mk2 = build_masks(forma("[[shape]]\ntype = \"circle\"\naxis = \"x\"\nu = 3.0\nv = 2.5\nr = 1.5\nf = 0.3\noutside = true\n"), gr)
    @test mk2.f[x1:x2, y1:y2, z1:z2] == [e == 0.3 ? 1.0 : 0.3 for e in esp]
    # a rectangle on the x–z plane (axis y), only through the first 2 ξσ along y
    mk3 = build_masks(forma("[[shape]]\ntype = \"rect\"\naxis = \"y\"\nu = 5.0\nv = 3.0\nw = 2.0\nh = 2.0\nalong = [0.0, 2.0]\n"), gr)
    esp3 = [abs((i - x1) * dx - 5.0) <= 1 + 1e-9 && abs((k - z1) * dx - 3.0) <= 1 + 1e-9 && (j - y1) * dx <= 2 + 1e-9 ? 0.0 : 1.0
            for i in x1:x2, j in y1:y2, k in z1:z2]
    @test mk3.f[x1:x2, y1:y2, z1:z2] == esp3
    # vacuum: the nodes leave the block (τ = 0 on every link that touches one), the ghost planes copy their plane
    mk4 = build_masks(GeomSpec(GEOM_MODE="CUSTOM", geom_file=joinpath(SHAPES4, "cable_x.toml")), gr)
    v = mk4.vac
    @test v !== nothing && any(v) && !all(v[x1+1:x2, y1+1:y2, z1+1:z2])
    @test v[x1, :, :] == v[x1+1, :, :] && v[:, y2+1, :] == v[:, y2, :] && v[:, :, z2+1] == v[:, :, z2]
    @test all(mk4.τψ[v] .== 0) && all(mk4.f[v] .== 0)
    @test all(mk4.τx[i,j,k] == 0 for i in x1:x2, j in y1:y2+1, k in z1:z2+1 if v[i,j,k] || v[i+1,j,k])
    @test all(mk4.τx[i,j,k] == 1 for i in x1:x2, j in y1:y2+1, k in z1:z2+1 if !v[i,j,k] && !v[i+1,j,k])
    # the conducting nodes: the bar minus the vacuum, the contacts with the section of the face
    c = conducting_nodes(gr, mk4)
    @test c[1, :, :] == (.!v[x1, :, :] .& c[x1, :, :]) && c[end, :, :] == c[x2+1, :, :]
    @test conducting_nodes(gr, build_masks(GeomSpec(GEOM_MODE="WEDGE"), gr)) == conducting_nodes(gr, box_masks(gr))
    # a plain bar: the link conductivity with the conducting nodes is the one of the bar, to the bit
    @test link_conductivity(gr, 0.01, 1.0; cond=conducting_nodes(gr, box_masks(gr))) == link_conductivity(gr, 0.01, 1.0)
end
