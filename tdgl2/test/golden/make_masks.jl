# Golden masks for the geometry tests of 2.0, written by juliaTDGL 1.2 (julia/).
#
#   julia --project=julia tdgl2/test/golden/make_masks.jl
#
# Every catalogue mode and its _REF mirror on two meshes, the z modes, the shifts, the prism
# angle, and every shape file of julia/shapes (CUSTOM and CUSTOM_REF).  The masks depend only on
# the geometry keywords, which the GeomSpec of 2.0 carries with the same names.
using TDGL, JLD2

const OUT = @__DIR__
const SHAPES = normpath(joinpath(@__DIR__, "..", "..", "..", "julia", "shapes"))

modos = ["SIM", "ESFERA", "ARRAY", "WEDGE", "WEDGE_BOTH", "ESF_CUNA", "FGRAD", "SAWROW", "SPHROW",
         "SAWSPH", "STAGGER", "FUNNEL", "FUNNEL_IN", "TRI_Y1", "TRI_Y1_INV", "RECT_Y1", "HEX", "HEX_STAG"]
mallas = [(40, 24, 6, 0.5, 1), (30, 20, 8, 0.5, 2)]

casos = Tuple{String,NamedTuple,NamedTuple}[]
for (nx, ny, nz, dx, b) in mallas, m in modos, ref in ("", "_REF")
    push!(casos, ("$(m)$(ref)_$(nx)x$(ny)x$(nz)", (Nx_sc=nx, Ny_sc=ny, Nz_sc=nz, dx=dx, buf=b),
                  (GEOM_MODE=m * ref,)))
end
for zm in ("half", "buried"), m in ("WEDGE", "ESFERA", "TRI_Y1")
    push!(casos, ("$(m)_z$(zm)", (Nx_sc=40, Ny_sc=24, Nz_sc=8, dx=0.5, buf=1), (GEOM_MODE=m, z_mode=zm)))
end
push!(casos, ("WEDGE_shift", (Nx_sc=40, Ny_sc=24, Nz_sc=6, dx=0.5, buf=1),
              (GEOM_MODE="WEDGE", x_shift=0.05, y_shift=-0.04)))
push!(casos, ("WEDGE_deep", (Nx_sc=40, Ny_sc=24, Nz_sc=6, dx=0.5, buf=1),
              (GEOM_MODE="WEDGE_BOTH", wedge_depth=0.45, wedge_half=0.15, wedge_depth2=0.2)))
push!(casos, ("FUNNEL_phi", (Nx_sc=40, Ny_sc=24, Nz_sc=6, dx=0.5, buf=1), (GEOM_MODE="FUNNEL", prism_phi=0.3)))
push!(casos, ("FGRAD_dir", (Nx_sc=40, Ny_sc=24, Nz_sc=6, dx=0.5, buf=1),
              (GEOM_MODE="FGRAD", fgrad_flo=0.8, fgrad_fhi=1.3, fgrad_nper=3, fgrad_dir=-1)))
for f in sort(readdir(SHAPES)), m in ("CUSTOM", "CUSTOM_REF")
    endswith(f, ".toml") || continue
    push!(casos, ("$(m)_$(splitext(f)[1])", (Nx_sc=40, Ny_sc=24, Nz_sc=6, dx=0.5, buf=1),
                  (GEOM_MODE=m, geom_file=joinpath(SHAPES, f))))
end

out = Dict{String,Any}()
for (nombre, malla, geo) in casos
    cfg = Config(:legacy; malla..., geo..., dy=malla.dx, dz=malla.dx, verbose=false, selftest=false)
    p = derive_params(cfg)
    g = build_geometry(cfg, p)
    geo_rel = haskey(geo, :geom_file) ? merge(geo, (geom_file=basename(geo.geom_file),)) : geo
    out[nombre] = Dict("malla" => Dict(string(k) => v for (k, v) in pairs(malla)),
                       "geo" => Dict(string(k) => v for (k, v) in pairs(geo_rel)),
                       "fmask" => g.fmask)
    # the link and node masks are those of the box; kept for one case only
    nombre == "WEDGE_40x24x6" && (out["_box_40x24x6"] = Dict("taux" => g.taux, "tauy" => g.tauy,
                                                              "tauz" => g.tauz, "taupsi" => g.taupsi))
end
jldsave(joinpath(OUT, "masks.jld2"), true; casos=out)
println(length(out), " masks written")
