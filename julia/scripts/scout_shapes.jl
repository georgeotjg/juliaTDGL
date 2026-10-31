# Quick ranking of shapes for the geometric diode: minutes per shape, a ranking and not a measurement.
#
#   julia --project=julia julia/scripts/scout_shapes.jl a.toml b.toml ... [options]
#
# Options (defaults in brackets):
#   --buf N        buffer (normal-metal padding) cells on every face [1]. It moves |η|:
#                  −47 % with 1 cell, −21 % with 8 in the methods paper. Rank with one value.
#   --H h          applied field in Hc2 [0.50]. |η| grows towards the entry field (−57 % at 0.45).
#   --J a:s:b      current grid in J0 [0.02:0.02:0.16]
#   --tmin t       floor of a static point, t_GL [200]
#   --tres t       floor of a flowing point, t_GL [600]
#   --mirror       also run the mirror (CUSTOM_REF) of every shape: η must change sign
#
# Fixed: production box 40×24×6, dx = 0.5, κ = 2, ξπ/ξσ = 1.30, γ12 = 1e-3, self-field injection,
# the default scheme and mode (prod, Fast). Each shape writes to julia/runs/scout/<name>/; the ranking goes to julia/runs/scout/RANKING.txt.
#
# What the numbers are. On these short windows a slow onset carries few crossings, so two readings
# are printed:
#   η1   brackets with a one-crossing threshold: first rung with ≥ 1 crossing per line;
#   η10  brackets with the verdict rule of the paper (silent < 1, dissipating ≥ 10). It is often
#        open here; that is expected.
# The winner then needs the full protocol: windows 1500/2000 t_GL, doubled at the deciding rungs,
# the mirror (η flips sign), and 4–8 buffer cells.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
using TDGL, Printf, Dates

function opts(args)
    o = Dict{String,Any}("buf" => 1, "H" => 0.50, "J" => collect(0.02:0.02:0.16),
                         "tmin" => 200.0, "tres" => 600.0, "mirror" => false, "shapes" => String[])
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--buf";      i += 1; o["buf"] = parse(Int, args[i])
        elseif a == "--H";    i += 1; o["H"] = parse(Float64, args[i])
        elseif a == "--J";    i += 1; p = parse.(Float64, split(args[i], ":")); o["J"] = collect(p[1]:p[2]:p[3])
        elseif a == "--tmin"; i += 1; o["tmin"] = parse(Float64, args[i])
        elseif a == "--tres"; i += 1; o["tres"] = parse(Float64, args[i])
        elseif a == "--mirror"; o["mirror"] = true
        elseif startswith(a, "-"); error("unknown option $a")
        else push!(o["shapes"], a)
        end
        i += 1
    end
    isempty(o["shapes"]) && error("give one or more .toml shapes")
    o
end

# first |J| with at least n crossings per line, and the last |J| below it with fewer than one
function onset(J, V, t, n)
    q = [k for k in eachindex(J) if abs(J[k]) > 1e-12]
    Ja = abs.(J[q]); cr = abs.(V[q]) .* t[q] ./ (2pi)
    d = findfirst(>=(n), cr)
    d === nothing && return (NaN, Inf)
    s = findlast(<(1), cr[1:d-1])
    (s === nothing ? 0.0 : Ja[s], Ja[d])
end
eta(a, b) = 100 * (a - b) / (a + b)
function eta_txt(bp, bn)
    (isfinite(bp[2]) && isfinite(bn[2])) || return "   open    "
    c = eta((bp[1] + bp[2]) / 2, (bn[1] + bn[2]) / 2)
    @sprintf("%+5.0f [%+4.0f,%+4.0f]", c, eta(bp[1], bn[2]), eta(bp[2], bn[1]))
end

o = opts(ARGS)
root = joinpath(@__DIR__, "..", "runs", "scout"); mkpath(root)
rows = String[]
for shp in o["shapes"], geom in (o["mirror"] ? ("CUSTOM", "CUSTOM_REF") : ("CUSTOM",))
    name = splitext(basename(shp))[1] * (geom == "CUSTOM_REF" ? "_REF" : "")
    cfg = Config(GEOM_MODE=geom, geom_file=abspath(shp), Nx_sc=40, Ny_sc=24, Nz_sc=6,
                 dx=0.5, dy=0.5, dz=0.5, buf=o["buf"], kappa_s=2.0, kappa_p=2.0, xi_ratio=1.30,
                 gamma12=0.001, H_const=o["H"], inject_mode="selffield", inject_gain=1.0,
                 J_list=o["J"], t_min_GL=o["tmin"], t_min_GL_res=o["tres"], mMax=20000,
                 eps_conv=1e-5, selftest=false, snap_every=1, verbose=false, use_threads=false,
                 out_dir=joinpath(root, name), run_tag=name)
    t0 = time()
    sde = run_case(cfg)
    tc = sde.p.dt * sde.p.lMax
    tp = sde.pos.m_arr .* tc; tn = sde.neg.m_arr .* tc
    # the rule needs the second-half rate of every point (V_arr of a fixed point is its last 40 cycles)
    vr(r) = [isnan(r.Vh_arr[q]) ? r.V_arr[q] : r.Vh_arr[q] for q in eachindex(r.V_arr)]
    vp, vn = vr(sde.pos), vr(sde.neg)
    b1 = (onset(sde.pos.J_arr, vp, tp, 1), onset(sde.neg.J_arr, vn, tn, 1))
    b10 = (onset(sde.pos.J_arr, vp, tp, 10), onset(sde.neg.J_arr, vn, tn, 10))
    nv = maximum(vcat(sde.pos.Nv1_arr, sde.neg.Nv1_arr))
    r = @sprintf("%-28s η1 %s   η10 %s   Ic+ (1 cr) %s   |Ic-| (1 cr) %s   Nv≤%d   %.1f min",
                 name, eta_txt(b1...), eta_txt(b10...),
                 isfinite(b1[1][2]) ? @sprintf("(%.3f,%.3f)", b1[1]...) : "open",
                 isfinite(b1[2][2]) ? @sprintf("(%.3f,%.3f)", b1[2]...) : "open", nv, (time() - t0) / 60)
    println(r); flush(stdout); push!(rows, r)
end
open(joinpath(root, "RANKING.txt"), "a") do io
    println(io, "# ", Dates.now(), "  buf=", o["buf"], " H=", o["H"], " J=", o["J"][1], ":", o["J"][end],
            " floors ", o["tmin"], "/", o["tres"], " t_GL  (ranking only; see the header of scout_shapes.jl)")
    foreach(r -> println(io, r), rows)
end
