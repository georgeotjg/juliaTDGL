# Writes the catalogue modes as pixel shapes, to open and modify them in the
# editor. Production box 40×24×6, dx = 0.5, z_mode through.
#   julia --project=julia julia/scripts/catalogue_to_shapes.jl
# Output: julia/shapes/cat_<MODE>.toml. Each column pattern in z (contiguous
# range + f, also f>1) is one palette character. A mode with a column that
# is not a contiguous range with a single f is skipped with a warning.
# On that mesh the template gives the catalogue mask to the bit (test_shapes.jl).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
using TDGL, Printf

const MODOS = ["SIM", "WEDGE", "WEDGE_BOTH", "TRI_Y1", "TRI_Y1_INV", "RECT_Y1",
               "FUNNEL", "FUNNEL_IN", "ESFERA", "ARRAY", "ESF_CUNA", "STAGGER",
               "SAWROW", "SAWSPH", "SPHROW", "HEX", "HEX_STAG", "FGRAD"]
const CHARS = collect("#abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
const BASE = (Nx_sc=40, Ny_sc=24, Nz_sc=6, buf=1, dx=0.5, dy=0.5, dz=0.5,
              kappa_s=25.0, kappa_p=25.0, xi_ratio=4.0, H_const=0.030,
              selftest=false, verbose=false, use_threads=false, out_dir="")

fnum(v) = (s = @sprintf("%.6f", v); s = rstrip(rstrip(s, '0'), '.'); occursin('.', s) ? s : s * ".0")

function plantilla(modo; base=BASE)
    cfg = Config(; base..., GEOM_MODE=modo)
    p = derive_params(cfg)
    f = build_geometry(cfg, p).fmask
    (; x1, x2, y1, y2, z1, z2, dx, dz) = p
    paleta = Dict{Tuple{Int,Int,Float64},Char}()
    filas = String[]
    for j in y2:-1:y1                      # first row = largest y
        fila = Char[]
        for i in x1:x2
            ks = [k for k in z1:z2 if f[i, j, k] != 1]   # f>1 = stronger condensation (FGRAD)
            if isempty(ks)
                push!(fila, '.')
                continue
            end
            vals = unique(f[i, j, ks])
            (length(vals) == 1 && ks == collect(first(ks):last(ks))) || return nothing
            clave = (first(ks), last(ks), vals[1])
            if !haskey(paleta, clave)
                length(paleta) < length(CHARS) || return nothing
                paleta[clave] = CHARS[length(paleta) + 1]
            end
            push!(fila, paleta[clave])
        end
        push!(filas, String(fila))
    end
    io = IOBuffer()
    println(io, "# Catalogue template: GEOM_MODE=\"$modo\" written as pixels.")
    println(io, "# Box 40x24x6, dx = 0.5. On that mesh it gives the catalogue mask to the bit;")
    println(io, "# on another mesh the pixels are resampled to the nearest node.")
    println(io)
    if !isempty(paleta)
        println(io, "[[shape]]")
        println(io, "type = \"pixels\"")
        println(io, "dx = ", fnum(dx))
        println(io, "rows = \"\"\"")
        print(io, join(filas, "\n"))
        println(io, "\"\"\"")
        println(io)
        println(io, "[shape.palette]")
        for ((ka, kb, v), c) in sort(collect(paleta); by=last)
            z = (ka == z1 && kb == z2) ? "\"all\"" :
                "[" * fnum((ka - z1) * dz) * ", " * fnum((kb - z1) * dz) * "]"
            println(io, "\"", c, "\" = { z = ", z, ", f = ", repr(v), " }")   # exact
        end
    else
        # SIM: the whole box is superconducting; a shape that touches nothing.
        println(io, "[[shape]]")
        println(io, "type = \"rect\"")
        println(io, "x = 0.0\ny = 0.0\nw = 0.0\nh = 0.0\nz = \"all\"\nf = 1.0")
    end
    String(take!(io))
end

if abspath(PROGRAM_FILE) == @__FILE__
    dest = joinpath(@__DIR__, "..", "shapes")
    for m in MODOS
        t = plantilla(m)
        if t === nothing
            println("skip $m: some column is not a contiguous range with a single f")
            continue
        end
        write(joinpath(dest, "cat_$m.toml"), t)
        println("cat_$m.toml")
    end
end
