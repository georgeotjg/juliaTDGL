# Launched from the terminal menu (or by hand). Writes to julia/runs/<name>/.
#
#   julia --project=julia julia/scripts/run_preset.jl WEDGE 2 0.10 0
#   julia --project=julia julia/scripts/run_preset.jl \
#       --geom WEDGE --kappa 2 --gamma12 0.001 --J 0.0,0.10 --drag 0 \
#       --mesh small --window smoke --z through --dir PATH --restart
#   meshes:  small 20x14x4 | production 40x24x6 | dx2 80x48x12 (dx 0.25)
#   windows: smoke 20/40 | short 20/200 | long 200/1200 | hard 1200/1200 | paper 1500/2000
#            (paper = the protocol of the methods paper: floors 1500 and 2000 t_GL)
#            (minimum t_GL per current point: static / flowing)
# Optional terms (one at a time; the menu does not combine them):
#       --drag 1 --nu 0.05          mixed-gradient (drag) coupling
#       --heat 0.08 [--eta 0.05] [--joule 1] [--margin 8]   θ(r,t) with bath γ
#       --qss 1                     diffused Q (turns on Φ, λ_TF=2, τ_Q=1, D_Q=0.25)
#       --shape path.toml           geometry from the editor
#   (--forma, --calor and --margen are accepted as older names of the same flags)
# Scalar potential (independent of the terms above):
#       --phi off|coulomb|tf [--lambda-tf 2]   off = gauge Φ = 0 (default);
#       coulomb = Coulomb projection (same |Ψ|, B and V, only the gauge changes);
#       tf = Thomas–Fermi screening λ_TF (changes the physics). --qss implies tf.
# Custom mesh and window (override the preset): --Nx 40 --Ny 24 --Nz 6 --dx 0.5 --buf 4
#       --tmin 200 --tminres 1200   (minimum t_GL per static / flowing point)
# Bath temperature (not a term; combines with any of them):
#       --T 0.3                     T_red = 0.3 with T_physical = true (both bands
#                                   feel T). Without --T: T_red = 0.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
using TDGL, Printf, Dates

const CATALOGO = (
    "SIM", "WEDGE", "WEDGE_REF", "WEDGE_BOTH", "ESFERA", "ARRAY",
    "ESF_CUNA", "FGRAD", "SAWROW", "SPHROW", "SAWSPH", "STAGGER",
    "FUNNEL", "FUNNEL_IN", "TRI_Y1", "TRI_Y1_INV", "RECT_Y1", "HEX", "HEX_STAG",
)
const WIDE = Set(["FUNNEL", "FUNNEL_IN", "TRI_Y1", "TRI_Y1_INV", "RECT_Y1", "STAGGER", "SAWROW", "SAWSPH", "ARRAY",
                  "ESF_CUNA", "HEX", "HEX_STAG", "SPHROW", "WEDGE_BOTH", "FGRAD"])

function _json_val(v::Bool); v ? "true" : "false"; end
_json_val(v::Integer) = string(v)
_json_val(v::AbstractFloat) = isfinite(v) ? string(v) : "null"
_json_val(v::AbstractString) = "\"" * replace(v, "\\" => "\\\\", "\"" => "\\\"") * "\""
_json_val(v::AbstractVector) = "[" * join(_json_val.(v), ", ") * "]"

function _write_job(path, obj::Dict)
    open(path, "w") do io
        println(io, "{")
        ks = sort(collect(keys(obj)))
        for (i, k) in enumerate(ks)
            print(io, "  ", _json_val(string(k)), ": ", _json_val(obj[k]))
            println(io, i == length(ks) ? "" : ",")
        end
        println(io, "}")
    end
end

function _parse_J(s)
    parts = split(s, r"[, ]+"; keepempty=false)
    isempty(parts) && return Float64[0.10]
    parse.(Float64, parts)
end

function parse_cli(args)
    o = Dict{String,Any}(
        "geom" => "WEDGE", "kappa" => 2.0, "gamma12" => 0.001,
        "J" => Float64[0.10], "drag" => false, "nu" => 0.05,
        "mesh" => "small", "window" => "smoke", "z" => "through",
        "dir" => "", "restart" => false, "cpus" => "8,9",
        "H" => NaN, "xi" => NaN, "name" => "",
        "Nx" => 0, "Ny" => 0, "Nz" => 0, "buf" => 0,
        "tmin" => NaN, "tminres" => NaN, "mMax" => 0,
        "rotate" => NaN, "forma" => "",
        "calor" => NaN, "eta" => 0.05, "joule" => false, "margen" => 0, "qss" => false,
        "T" => 0.0, "dx" => NaN, "phi" => "off", "lambda_tf" => 2.0,
    )
    # positional back-compat: GEOM KAPPA J DRAG
    i = 1
    if !isempty(args) && !startswith(args[1], "-")
        o["geom"] = uppercase(args[1]); i = 2
        if length(args) >= 2 && !startswith(args[2], "-")
            o["kappa"] = parse(Float64, args[2]); i = 3
        end
        if length(args) >= 3 && !startswith(args[3], "-")
            o["J"] = _parse_J(args[3]); i = 4
        end
        if length(args) >= 4 && !startswith(args[4], "-")
            o["drag"] = parse(Int, args[4]) != 0; i = 5
        end
    end
    while i <= length(args)
        a = args[i]
        if a == "--restart"; o["restart"] = true
        elseif a == "--geom" && i < length(args); i += 1; o["geom"] = uppercase(args[i])
        elseif a == "--kappa" && i < length(args); i += 1; o["kappa"] = parse(Float64, args[i])
        elseif a == "--gamma12" && i < length(args); i += 1; o["gamma12"] = parse(Float64, args[i])
        elseif a == "--J" && i < length(args); i += 1; o["J"] = _parse_J(args[i])
        elseif a == "--drag" && i < length(args); i += 1; o["drag"] = parse(Int, args[i]) != 0
        elseif a == "--nu" && i < length(args); i += 1; o["nu"] = parse(Float64, args[i])
        elseif a == "--mesh" && i < length(args); i += 1; o["mesh"] = lowercase(args[i])
        elseif a == "--window" && i < length(args); i += 1; o["window"] = lowercase(args[i])
        elseif a == "--z" && i < length(args); i += 1; o["z"] = lowercase(args[i])
        elseif a == "--dir" && i < length(args); i += 1; o["dir"] = args[i]
        elseif a == "--cpus" && i < length(args); i += 1; o["cpus"] = args[i]
        elseif a == "--H" && i < length(args); i += 1; o["H"] = parse(Float64, args[i])
        elseif a == "--xi-ratio" && i < length(args); i += 1; o["xi"] = parse(Float64, args[i])
        elseif a == "--name" && i < length(args); i += 1; o["name"] = args[i]
        elseif a == "--mMax" && i < length(args); i += 1; o["mMax"] = parse(Int, args[i])
        elseif a == "--tmin" && i < length(args); i += 1; o["tmin"] = parse(Float64, args[i])
        elseif a == "--tminres" && i < length(args); i += 1; o["tminres"] = parse(Float64, args[i])
        elseif a == "--rotate-deg" && i < length(args); i += 1; o["rotate"] = parse(Float64, args[i])
        elseif a in ("--shape", "--forma") && i < length(args); i += 1; o["forma"] = args[i]
        elseif a in ("--heat", "--calor") && i < length(args); i += 1; o["calor"] = parse(Float64, args[i])
        elseif a == "--eta" && i < length(args); i += 1; o["eta"] = parse(Float64, args[i])
        elseif a == "--joule" && i < length(args); i += 1; o["joule"] = parse(Int, args[i]) != 0
        elseif a in ("--margin", "--margen") && i < length(args); i += 1; o["margen"] = parse(Int, args[i])
        elseif a == "--qss" && i < length(args); i += 1; o["qss"] = parse(Int, args[i]) != 0
        elseif a == "--phi" && i < length(args); i += 1; o["phi"] = lowercase(args[i])
        elseif a == "--lambda-tf" && i < length(args); i += 1; o["lambda_tf"] = parse(Float64, args[i])
        elseif a == "--T" && i < length(args); i += 1; o["T"] = parse(Float64, args[i])
        elseif a == "--Nx" && i < length(args); i += 1; o["Nx"] = parse(Int, args[i])
        elseif a == "--Ny" && i < length(args); i += 1; o["Ny"] = parse(Int, args[i])
        elseif a == "--Nz" && i < length(args); i += 1; o["Nz"] = parse(Int, args[i])
        elseif a == "--buf" && i < length(args); i += 1; o["buf"] = parse(Int, args[i])
        elseif a == "--dx" && i < length(args); i += 1; o["dx"] = parse(Float64, args[i])
        else
            error("unknown flag: $a")
        end
        i += 1
    end
    o
end

opt = parse_cli(ARGS)
geom = opt["geom"]
# --shape: geometry from an editor file (src/shapes.jl). It is copied into the
# run folder and the run reads that copy.
forma = opt["forma"]
if !isempty(forma)
    forma = abspath(forma)
    isfile(forma) || error("--shape: $forma does not exist")
    geom in ("CUSTOM", "CUSTOM_REF") || (geom = "CUSTOM")
end
geom in CATALOGO || geom in ("CUSTOM", "CUSTOM_REF") ||
    error("geom=$geom. Catalogue: $(join(CATALOGO, " ")) or --shape file.toml")
kappa = opt["kappa"]
g12 = opt["gamma12"]
drag = opt["drag"]
calor = isfinite(opt["calor"])
qss = opt["qss"]
nterm = count((drag, calor, qss))
nterm > 1 && println("WARNING: $(nterm) optional terms on at once (drag, heat, Q): ",
                     "it will not be possible to tell which one moved the result.")
(opt["joule"] || opt["margen"] > 0) && !calor && error("--joule/--margin need --heat γ")
phi = qss ? "tf" : opt["phi"]
phi in ("off", "coulomb", "tf") || error("--phi off|coulomb|tf")
phi == "tf" && !(opt["lambda_tf"] > 0) && error("--lambda-tf must be > 0")
lTF = phi == "tf" ? opt["lambda_tf"] : 0.0
opt["joule"] && phi != "off" && error("--joule needs --phi off (with Φ, E carries −∇Φ; not verified)")
zmode = opt["z"]
zmode in ("through", "half", "buried") || error("z=through|half|buried")
Tb = opt["T"]
0.0 <= Tb < 1.0 || error("--T: 0 ≤ T < 1 (in units of Tc)")
Tb > 0 && println("bath T = $Tb with T_physical (α_σ=(1−T)f, α_π=(a2n−T)f).")

mesh = get(Dict("production" => "paper"), opt["mesh"], opt["mesh"])
if mesh == "small"
    Nx, Ny, Nz, dx, buf = 20, 14, 4, 0.5, 1
elseif mesh == "paper"
    Nx, Ny, Nz, dx, buf = 40, 24, 6, 0.5, 1
elseif mesh == "dx2"
    Nx, Ny, Nz, dx, buf = 80, 48, 12, 0.25, 2
else
    error("mesh=small|production|dx2")
end
# Custom mesh (--Nx --Ny --Nz --dx): overrides the preset.
if opt["Nx"] > 0 || opt["Ny"] > 0 || opt["Nz"] > 0 || isfinite(opt["dx"])
    opt["Nx"] > 0 && (Nx = opt["Nx"]); opt["Ny"] > 0 && (Ny = opt["Ny"])
    opt["Nz"] > 0 && (Nz = opt["Nz"]); isfinite(opt["dx"]) && (dx = opt["dx"])
    mesh = "custom"
end
# --buf N: buffer (normal-metal padding) of N cells on every face; it moves |η| (methods paper)
opt["buf"] > 0 && (buf = opt["buf"])
g0 = replace(geom, "_REF" => "")
if g0 in WIDE && mesh == "small"
    Nx, Ny, Nz = 40, 24, 6
    mesh = "paper"
    println("WARNING: $geom does not fit well in 20×14; mesh raised to 40×24×6")
end
if !isempty(forma) && mesh == "small"
    Nx, Ny, Nz = 40, 24, 6
    mesh = "paper"
    println("WARNING: the editor draws on the 40×24×6 box; mesh raised to 40×24×6")
end

win = get(Dict("short" => "sana", "long" => "larga"), opt["window"], opt["window"])
if win == "smoke"
    tmin, tres, mMax = 20.0, 40.0, 3000
elseif win == "sana"
    tmin, tres, mMax = 20.0, 200.0, 4000
elseif win == "larga"
    tmin, tres, mMax = 200.0, 1200.0, 8000
elseif win == "hard"
    tmin, tres, mMax = 1200.0, 1200.0, 12000
elseif win == "paper"
    tmin, tres, mMax = 1500.0, 2000.0, 20000
else
    error("window=smoke|short|long|hard|paper")
end
# Custom window: mMax is only a ceiling; with an explicit t_min it is raised so the point is not cut short.
(isfinite(opt["tmin"]) || isfinite(opt["tminres"])) && opt["mMax"] == 0 && (mMax = 500000)
isfinite(opt["tmin"]) && (tmin = opt["tmin"])
isfinite(opt["tminres"]) && (tres = opt["tminres"])
opt["mMax"] > 0 && (mMax = opt["mMax"])

H = isfinite(opt["H"]) ? opt["H"] : (kappa >= 20 ? 0.030 : 0.50)
xi = isfinite(opt["xi"]) ? opt["xi"] : (kappa >= 20 ? 4.0 : 1.30)
Jlist = Vector{Float64}(opt["J"])
nJ = length(sort(unique(vcat(0.0, abs.(Jlist)))))
n_expected = 2 * nJ   # +J and −J; J=0 is always included

if !isempty(opt["dir"])
    sal = abspath(opt["dir"])
else
    tagj = join((replace(@sprintf("%.3f", abs(j)), "." => "") for j in Jlist[1:min(end, 3)]), "-")
    gtag = isempty(forma) ? geom :
           (geom == "CUSTOM_REF" ? "F_REF_" : "F_") * splitext(basename(forma))[1]
    ttag = calor ? "_calor" * replace(@sprintf("%g", opt["calor"]), "." => "p") *
                   (opt["joule"] ? "J" : "") : qss ? "_qss" : ""
    Tb > 0 && (ttag *= "_T" * replace(@sprintf("%g", Tb), "." => "p"))
    phi != "off" && !qss && (ttag *= phi == "tf" ? "_TF" : "_phiC")
    auto = "TUI$(ttag)_$(gtag)_k$(Int(round(kappa)))_g$(replace(string(g12), '.' => 'p'))_J$(tagj)_$(drag ? "ON" : "OFF")"
    name = isempty(opt["name"]) ? auto : opt["name"]
    sal = joinpath(@__DIR__, "..", "runs", name)
end
mkpath(sal)
token = basename(sal)
copia_forma = ""
if !isempty(forma)
    copia_forma = joinpath(sal, "forma.toml")
    if isfile(copia_forma)
        opt["restart"] || read(copia_forma) == read(forma) ||
            error("$copia_forma exists and is a different shape; use another folder or --restart")
    else
        cp(forma, copia_forma)
    end
end
statusf = joinpath(sal, "STATUS.txt")
logf = joinpath(sal, "log.txt")
stopf = joinpath(sal, "STOP.txt")
opt["restart"] && isfile(stopf) && rm(stopf)

open(joinpath(sal, "julia.pid"), "w") do io
    println(io, getpid())
end

function marcar(msg)
    open(statusf, "w") do io
        println(io, Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "  ", msg)
    end
    println("STATUS: ", msg); flush(stdout)
    open(logf, "a") do io
        println(io, Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "  ", msg)
    end
end

dir = joinpath(sal, geom, "A")
mkpath(dir)
_write_job(joinpath(sal, "job.json"), Dict{String,Any}(
    "geom" => geom, "kappa" => kappa, "gamma12" => g12,
    "J_list" => Jlist, "drag" => drag, "nu_drag" => (drag ? opt["nu"] : 0.0),
    "Nx" => Nx, "Ny" => Ny, "Nz" => Nz, "dx" => dx, "buf" => buf,
    "t_min_GL" => tmin, "t_min_GL_res" => tres, "mMax" => mMax,
    "z_mode" => zmode, "H" => H, "xi_ratio" => xi,
    "cpus" => opt["cpus"], "n_expected" => n_expected,
    "mesh" => mesh, "window" => win, "restart" => opt["restart"],
    "project" => "julia",
    "rotate_deg" => (isfinite(opt["rotate"]) ? opt["rotate"] : NaN),
    "geom_file" => (isempty(copia_forma) ? "" : "forma.toml"), "forma_origen" => forma,
    "calor" => (calor ? opt["calor"] : NaN), "eta" => opt["eta"], "joule" => opt["joule"],
    "margen" => opt["margen"], "qss" => qss,
    "T_red" => Tb, "T_physical" => Tb > 0, "phi" => phi, "lambda_TF" => lTF,
))
marcar(opt["restart"] ? "resume $geom/A" : "running $geom/A")

cfg = Config(
    GEOM_MODE=geom, geom_file=copia_forma, Nx_sc=Nx, Ny_sc=Ny, Nz_sc=Nz, buf=buf,
    dx=dx, dy=dx, dz=dx, kappa_s=kappa, kappa_p=kappa,
    xi_ratio=xi, gamma12=g12, H_const=H,
    J_list=Jlist, t_min_GL=tmin, t_min_GL_res=tres, mMax=mMax,
    use_Vconv=true, selftest=false, snap_every=1, vtk_every=0,
    verbose=true, use_threads=Threads.nthreads() > 1, guard_hours=win in ("hard", "paper") ? 36.0 : 12.0,
    use_drag=drag, nu_drag=drag ? opt["nu"] : 0.0,
    use_heat=calor, gamma_bath=(calor ? opt["calor"] : 0.2), eta_J=opt["eta"],
    use_joule=opt["joule"], heat_margin=opt["margen"],
    use_phi=(phi != "off"), lambda_TF=lTF, tau_Q=(qss ? 1.0 : 0.0), use_Qss=qss,
    T_red=Tb, T_physical=Tb > 0,
    z_mode=zmode,
    prism_phi = isfinite(opt["rotate"]) ? deg2rad(opt["rotate"]) : NaN,
    wedge_depth2 = (g0 == "WEDGE_BOTH" ? 0.20 : 0.0),
    restart_pos=opt["restart"], restart_neg=opt["restart"],
    out_dir=dir, run_tag=token,
)
# Geometry preview before the first current point: output/geometry_mask.txt (f on the sample
# nodes) and, if python3 with matplotlib is there, figures/geometry.png drawn in the background.
try
    pg = TDGL.derive_params(cfg); gg = TDGL.build_geometry(cfg, pg)
    od = joinpath(dir, "output"); mkpath(od)
    open(joinpath(od, "geometry_mask.txt"), "w") do io
        println(io, "$(pg.x2 - pg.x1 + 1) $(pg.y2 - pg.y1 + 1) $(pg.z2 - pg.z1 + 1) $(pg.dx) $(pg.dz)")
        for k in pg.z1:pg.z2, j in pg.y1:pg.y2
            println(io, join((@sprintf("%.4g", gg.fmask[i, j, k]) for i in pg.x1:pg.x2), " "))
        end
    end
    py = joinpath(@__DIR__, "plot_geometry.py")
    Sys.which("python3") === nothing ||
        run(pipeline(`python3 $py $dir`; stdout=devnull, stderr=devnull); wait=false)
catch err
    println("geometry preview skipped: ", sprint(showerror, err))
end
t1 = time()
sde = run_case(cfg; mode=Strict())
open(logf, "a") do io
    println(io, "# TUI $geom $(isempty(forma) ? "" : basename(forma)) k=$kappa γ12=$g12 J=$Jlist drag=$drag ",
            calor ? "calor γ=$(opt["calor"]) η=$(opt["eta"]) joule=$(opt["joule"]) margen=$(opt["margen"]) " : "",
            qss ? "qss " : "", phi == "off" ? "" : "phi=$phi ", Tb > 0 ? "T=$Tb " : "", "z=$zmode mesh=$mesh win=$win")
    @printf(io, "eta=%s\n", isnan(sde.eta_pct) ? "NaN" : @sprintf("%.4f", sde.eta_pct))
    for (sgn, r) in (("+", sde.pos), ("-", sde.neg))
        r === nothing && continue
        for q in eachindex(r.J_arr)
            abs(r.J_arr[q]) < 1e-15 && r.cflag[q] == ' ' && continue
            @printf(io, "    %s J=%+.3f V=%+.4e conv=%s it=%d Nv=%d/%d\n",
                    sgn, r.J_arr[q], r.V_arr[q], string(r.cflag[q]),
                    r.m_arr[q], r.Nv1_arr[q], r.Nv2_arr[q])
        end
    end
    @printf(io, "# wall %.1f min\n%s done\n", (time()-t1)/60, token)
end
stopped = isfile(stopf)
marcar(token * (stopped ? " stopped (STOP.txt; u in the menu resumes)" : " done"))
println(token, stopped ? " stopped" : " done"); flush(stdout)
