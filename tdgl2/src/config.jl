# ============================================================================
#  config.jl — a run from a file
#
#  One TOML file sets a run: the material (inline or from materials/), the mesh, the shape, the
#  scheme, the protocol and the currents.  The run writes into its folder the file it was given,
#  the resolved parameters, the rung tables and a JSON summary with units, code version and the
#  regime numbers, so a run is reproduced by one file and one command.
#
#      [material]   file = "mgb2_model_II.toml"        (or the keys below inline)
#                   kappa, sigma_n, T, gamma12 | gamma = [[...], ...]
#                   [[material.band]]  m | g | xi_ratio, Gamma, a, beta, gamma_kwt, name; an anisotropic band:
#                                      m_x, m_y, m_z or xi_ratio_x, xi_ratio_y, xi_ratio_z
#      [mesh]       nx, ny, nz, dx, buf (bufx, bufy, bufz)
#      [geometry]   the GeomSpec keys (GEOM_MODE, geom_file, wedge_depth, ...)
#      [scheme]     implicit, dt_scale, substeps = [1, 0], extrap, hard, rkl, rotframe, rkl_margin,
#                   selffield_bc ("bar" | "open" | "slab"), open_every, sigma_vac, drag (a number for two bands, or the matrix),
#                   qp_exchange (ν for two bands; the matrix of rates for N), qp_split = [s_1, …], noise (T̃), noise_seed, noise_field
#      [heat]       the temperature field: D_T, gamma_bath, eta, joule, margin (as Heat)
#      [protocol]   the Protocol keys (t_min_GL, t_min_GL_res, eps_conv, delta_v, ...)
#      [run]        J = [...], H, out_dir
# ============================================================================

const MATERIALS = normpath(joinpath(@__DIR__, "..", "materials"))

_sym(d::AbstractDict) = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d)

function _band(d::AbstractDict, a1::Float64)
    k = _sym(d)
    if haskey(k, :xi_ratio)
        # ξ_µ/ξ_1 = 1/√(m_µ a_µ/β_µ) (with band 1 the unit): m_µ = 1/(R² a_µ/β_µ)
        a = Float64(get(k, :a, 1.0)); β = Float64(get(k, :beta, 1.0))
        k[:m] = 1 / (Float64(k[:xi_ratio])^2 * (a / β))
    end
    # an anisotropic band: m_x, m_y, m_z, or xi_ratio_x, ... (the axes not given take the isotropic m)
    a = Float64(get(k, :a, 1.0)); β = Float64(get(k, :beta, 1.0))
    m0 = haskey(k, :g) ? 1 / Float64(k[:g]) : Float64(get(k, :m, 1.0))
    ax = ntuple(3) do i
        c = ("x", "y", "z")[i]
        haskey(k, Symbol("m_$c")) ? Float64(k[Symbol("m_$c")]) :
        haskey(k, Symbol("xi_ratio_$c")) ? 1 / (Float64(k[Symbol("xi_ratio_$c")])^2 * (a / β)) : nothing
    end
    mx = all(isnothing, ax) ? nothing : ntuple(i -> something(ax[i], m0), 3)
    Band(; m=Float64(get(k, :m, 1.0)), g=haskey(k, :g) ? Float64(k[:g]) : nothing,
         Γ=Float64(get(k, :Gamma, 1.0)), a=a, β=β, γkwt=Float64(get(k, :gamma_kwt, 0.0)), m_axes=mx)
end

"""
    material(d) -> (Model, info)

A model from a `[material]` table (or a material file).  `info` keeps the name, the source
notes and the per-parameter labels of the file, which the run summary repeats.
"""
function material(d::AbstractDict)
    if haskey(d, "file")
        f = d["file"]
        path = isabspath(f) ? f : isfile(f) ? f : joinpath(MATERIALS, f)
        base = TOML.parsefile(path)
        over = Dict(k => v for (k, v) in d if k != "file")
        d = merge(base, over)
    end
    bands = d["band"]
    N = length(bands)
    bs = ntuple(µ -> _band(bands[µ], 1.0), N)
    γ = zeros(N, N)
    if haskey(d, "gamma")
        γ .= reduce(vcat, permutedims.(Float64.(r) for r in d["gamma"]))
    elseif haskey(d, "gamma12")
        N == 2 || error("material: gamma12 is for two bands; give the matrix `gamma`")
        γ[1, 2] = γ[2, 1] = Float64(d["gamma12"])
    end
    sn = get(d, "sigma_n", 1.0)
    sn = sn == "dirty" ? :dirty : Float64(sn)          # "dirty": the conductivity of a dirty BCS material
    md = Model(bs, γ; κ=Float64(d["kappa"]), σn=sn, T=Float64(get(d, "T", 0.0)))
    info = Dict{String,Any}("name" => get(d, "name", ""), "regime" => get(d, "regime", ""),
                            "sources" => get(d, "sources", Dict()),
                            "band_names" => [get(b, "name", "band $µ") for (µ, b) in enumerate(bands)])
    md, info
end

"""
    load_config(path) -> NamedTuple

Reads a run file: `(model, info, grid, masks, geom, scheme, protocol, J, H, out_dir, raw)`.
"""
function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    md, info = material(raw["material"])
    ms = raw["mesh"]
    gr = Grid(ms["nx"], ms["ny"], ms["nz"]; dx=Float64(ms["dx"]), buf=get(ms, "buf", 1),
              bufx=get(ms, "bufx", get(ms, "buf", 1)), bufy=get(ms, "bufy", get(ms, "buf", 1)),
              bufz=get(ms, "bufz", get(ms, "buf", 1)))
    gk = _sym(get(raw, "geometry", Dict()))
    if haskey(gk, :geom_file) && !isabspath(gk[:geom_file])
        f = gk[:geom_file]
        gk[:geom_file] = isfile(joinpath(dirname(path), f)) ? joinpath(dirname(path), f) :
                         joinpath(@__DIR__, "..", "shapes", f)
    end
    geom = GeomSpec(; gk...)
    mk = build_masks(geom, gr)
    sk = _sym(get(raw, "scheme", Dict()))
    haskey(sk, :substeps) && (sk[:substeps] = Tuple(Int.(sk[:substeps])))
    haskey(sk, :dt_scale) && (sk[:dt_scale] = Float64(sk[:dt_scale]))
    haskey(sk, :sigma_vac) && (sk[:σvac] = Float64(pop!(sk, :sigma_vac)))    # TOML keys are ASCII unless quoted
    # [heat]: the temperature field (keys of Heat: D_T, gamma_bath, eta, joule, margin, cap, cut); drag goes in [scheme]
    if haskey(raw, "heat")
        hk = _sym(raw["heat"])
        for (k, v) in hk; v isa Integer && fieldtype(Heat, k) == Float64 && (hk[k] = Float64(v)); end
        sk[:heat] = Heat(; hk...)
    end
    haskey(sk, :qp_split) && (sk[:qp_split] = Tuple(Float64.(sk[:qp_split])))
    haskey(sk, :qp_exchange) && sk[:qp_exchange] isa Vector &&
        (sk[:qp_exchange] = reduce(vcat, [permutedims(Float64.(r)) for r in sk[:qp_exchange]]))   # N bands: the matrix
    haskey(sk, :drag) && sk[:drag] isa Vector && (sk[:drag] = reduce(vcat, [permutedims(Float64.(r)) for r in sk[:drag]]))
    pk = _sym(get(raw, "protocol", Dict()))
    for (k, v) in pk
        v isa Integer && fieldtype(Protocol, k) == Float64 && (pk[k] = Float64(v))
    end
    pr = Protocol(; pk...)
    rn = raw["run"]
    J = Float64.(rn["J"])
    H = Float64(get(rn, "H", 0.0))
    out_dir = get(rn, "out_dir", "")
    (model=md, info=info, grid=gr, masks=mk, geom=geom, scheme=sk, protocol=pr, J=J, H=H,
     out_dir=out_dir, raw=raw)
end

# --- a minimal JSON writer (numbers, strings, booleans, arrays, dictionaries) ----
_json(io, x::AbstractString) = print(io, '"', escape_string(x), '"')
_json(io, x::Bool) = print(io, x ? "true" : "false")
_json(io, x::Integer) = print(io, x)
_json(io, x::Real) = isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
_json(io, ::Nothing) = print(io, "null")
_json(io, x::Char) = _json(io, string(x))
_json(io, x::Symbol) = _json(io, string(x))
function _json(io, x::Union{AbstractVector,Tuple})
    print(io, '['); for (q, v) in enumerate(x); q > 1 && print(io, ", "); _json(io, v); end; print(io, ']')
end
function _json(io, d::AbstractDict, ind::Int=0)
    print(io, "{\n")
    ks = collect(keys(d))
    for (q, k) in enumerate(ks)
        print(io, " "^(ind + 2)); _json(io, string(k)); print(io, ": ")
        v = d[k]
        v isa AbstractDict ? _json(io, v, ind + 2) : _json(io, v)
        q < length(ks) && print(io, ",")
        print(io, "\n")
    end
    print(io, " "^ind, "}")
end
_json(io, x::NamedTuple, ind::Int=0) = _json(io, Dict(string(k) => v for (k, v) in pairs(x)), ind)

function _git_rev()
    dir = dirname(@__DIR__)
    try
        h = readchomp(pipeline(`git -C $dir rev-parse --short HEAD`, stderr=devnull))
        d = readchomp(pipeline(`git -C $dir status --porcelain -- src`, stderr=devnull))
        isempty(d) ? h : h * "+dirty"
    catch
        "unknown"
    end
end

# The run file into the folder (run.toml), or the check that the one there is the same.  The menu starts the two
# polarities together (two processes, the same file): written to a temporary and renamed, so that neither reads a
# half-written file nor fails on the other's copy (cp refuses an existing target).
function _keep_run_toml(path, od)
    rt = joinpath(od, "run.toml")
    abspath(rt) == abspath(path) && return rt
    txt = read(path, String)
    if !isfile(rt)
        tmp = tempname(od; cleanup=false)
        write(tmp, txt); mv(tmp, rt; force=true)
    end
    read(rt, String) == txt || error("$rt differs from $path: resume with the file the run started with")
    rt
end

# The shape file of a CUSTOM geometry next to run.toml, at the relative path the run file names (so that the folder alone
# rebuilds the run: `load_config(joinpath(od, "run.toml"))` finds it there).
function _keep_shape(c, od)
    g = get(c.raw, "geometry", Dict()); haskey(g, "geom_file") || return nothing
    rel = g["geom_file"]; (isabspath(rel) || startswith(normpath(rel), "..")) && return nothing   # only inside the folder
    src = c.geom.geom_file; dst = joinpath(od, rel)
    abspath(dst) == abspath(src) && return dst
    txt = read(src, String)
    if !isfile(dst)
        mkpath(dirname(dst)); tmp = tempname(dirname(dst); cleanup=false)
        write(tmp, txt); mv(tmp, dst; force=true)
    end
    read(dst, String) == txt || error("$dst differs from $src: the run folder holds another shape under that name")
    dst
end

"""
    run_config(path; out_dir=nothing, resume=false, polarity=:both) -> (pos, neg, brackets, summary)

Runs both polarities of the file at `path`.  Writes into the run folder: `run.toml` (the file),
`output/rungs_J+.csv`, `output/rungs_J-.csv`, `output/summary.json`, and the checkpoints (`restart/state_J±.jld2`,
one per closed point, with all the ladder carries).

- `resume = true` continues each polarity from its checkpoint (`STOP.txt` in the folder stops a run cleanly after
  the current point; a killed run loses only that point): the result is the one of the run without a stop, to the bit.
  Without it, a folder that already holds a ladder is refused.
- `polarity = :pos` or `:neg` runs one polarity (one process each, both into the same folder); the one that ends
  second, with the other ended, writes the summary.
"""
function run_config(path::AbstractString; out_dir::Union{Nothing,String}=nothing, resume::Bool=false,
                    polarity::Symbol=:both)
    c = load_config(path)
    od = out_dir === nothing ? c.out_dir : out_dir
    st = Stepper(c.model, c.grid, c.masks; c.scheme...)
    labels = polarity === :both ? ("J+", "J-") : polarity === :pos ? ("J+",) : ("J-",)
    if !isempty(od)
        mine = [joinpath(od, "output", "rungs_$l.csv") for l in labels]
        !resume && any(isfile, mine) &&
            error("$od already holds a ladder ($(join(filter(isfile, mine), ", "))): resume = true continues it, or use another folder")
        mkpath(joinpath(od, "output"))
        _keep_run_toml(path, od); _keep_shape(c, od)
    end
    split = polarity !== :both && !isempty(od)
    split && write(joinpath(od, "julia_$(labels[1]).pid"), string(getpid(), "\n"))
    t0 = time()
    r = run_ladders(st, c.J; H=c.H, pr=c.protocol, out_dir=od, resume=resume, polarity=polarity)
    if split
        # one polarity: with the other ended, read it back (its checkpoint is complete) and write the summary
        other = polarity === :pos ? "J-" : "J+"
        if ladder_done(joinpath(od, "restart", "state_$(other).jld2"))
            Jp = sort(unique(vcat(0.0, abs.(c.J))))
            o = resume_ladder(st, other == "J+" ? Jp : -Jp; H=c.H, label=other, pr=c.protocol, out_dir=od, write=false)
            r = polarity === :pos ? (pos=r.pos, neg=o, brackets=brackets(r.pos, o)) :
                                    (pos=o, neg=r.neg, brackets=brackets(o, r.neg))
        else
            # the menu follows julia.pid: hand it to the polarity still running
            pf = joinpath(od, "julia_$(other).pid")
            p = isfile(pf) ? tryparse(Int, strip(read(pf, String))) : nothing
            p !== nothing && ccall(:kill, Cint, (Cint, Cint), p, 0) == 0 && write(joinpath(od, "julia.pid"), string(p, "\n"))
        end
    end
    if r.brackets === nothing && !isempty(od)
        tui_status(od, ladder_done(joinpath(od, "restart", "state_$(labels[1]).jld2")) ?
                       "$(labels[1]) ended; waiting for the other polarity" : "$(labels[1]) stopped (STOP.txt): resume continues it")
    end
    r.brackets === nothing && return (pos=r.pos, neg=r.neg, brackets=nothing, summary=nothing)
    summ = summary_dict(st, c, r; wall_min=(time() - t0) / 60)
    if !isempty(od)
        open(io -> (_json(io, summ); println(io)), joinpath(od, "output", "summary.json"), "w")
        done = all(ladder_done(joinpath(od, "restart", "state_$l.jld2")) for l in ("J+", "J-"))
        tui_status(od, done ? "both polarities done" : "stopped (STOP.txt): resume continues it")
    end
    (pos=r.pos, neg=r.neg, brackets=r.brackets, summary=summ)
end

function _Ibracket(b, L, gr)
    f = filter(isfinite, L.Itot)
    k = (isempty(f) ? 1.0 : sum(f) / length(f)) * (gr.y2 - gr.y1 + 1) * gr.dy * slab_thickness(gr)
    [b.lo * k, b.hi * k]
end

"""The machine-readable summary of a run: what was run, with what, and what came out."""
function summary_dict(st::Stepper{N}, c, r; wall_min=NaN) where N
    md = st.model; gr = st.grid; b = r.brackets
    L = (gr.x2 - gr.x1 + 1) * gr.dx
    reg = regime(md; L=L)
    lad(Lr) = Dict("J" => Lr.J, "V" => Lr.V, "rate" => rung_rate(Lr), "t_int" => rung_time(Lr),
                   "crossings_per_line" => abs.(rung_rate(Lr)) .* rung_time(Lr) ./ (2pi),
                   "conv" => String(Lr.conv), "Nv" => [collect(Lr.Nv[:, µ]) for µ in 1:N],
                   "band_current" => [collect(Lr.Iband[:, µ]) for µ in 1:N])
    Dict{String,Any}(
        "code" => Dict("version" => CODE_VERSION, "commit" => _git_rev(), "julia" => string(VERSION)),
        "units" => "lengths xi_1, time t_GL = xi_1^2/D_1, field Hc2 of band 1, current J0",
        "material" => Dict("name" => c.info["name"], "regime_note" => c.info["regime"],
                           "kappa" => md.κ, "sigma_n" => md.σn, "T" => md.T,
                           "bands" => [Dict("name" => c.info["band_names"][µ], "m" => md.bands[µ].m, "m_axes" => collect(md.bands[µ].ma),
                                            "Gamma" => md.bands[µ].Γ, "a" => md.bands[µ].a,
                                            "beta" => md.bands[µ].β, "gamma_kwt" => md.bands[µ].γkwt)
                                       for µ in 1:N],
                           "gamma" => [collect(md.γ[µ]) for µ in 1:N]),
        "regime" => reg === nothing ? Dict() : Dict(string(k) => v for (k, v) in pairs(reg)),
        "mesh" => Dict("sample_cells" => [gr.x2 - gr.x1 + 1, gr.y2 - gr.y1 + 1, gr.z2 - gr.z1 + 1],
                       "dx" => gr.dx, "padding" => [gr.x1 - 1, gr.y1 - 1, gr.z1 - 1]),
        "geometry" => c.geom.GEOM_MODE,
        "scheme" => Dict("dt" => st.dt, "limit" => st.limit, "substeps" => collect(st.nsub), "rkl_stages" => collect(st.rkl),
                         "implicit_curl_curl" => st.imp !== nothing, "dt_scale" => st.dt_scale,
                         "rotframe" => st.rotframe, "rkl_margin" => st.rkl_margin,
                         "qp_exchange" => st.qpx === nothing ? nothing : Dict("nu" => st.qpx.ν isa Matrix ? [collect(r) for r in eachrow(st.qpx.ν)] : st.qpx.ν,
                                                                  "split" => collect(st.qpx.s)),
                         "noise" => st.noise === nothing ? nothing : Dict("T" => st.noise.T, "seed" => Int(st.noise.seed), "field" => st.noise.field),
                         "heat" => st.heat === nothing ? nothing : Dict(string(k) => getfield(st.heat.h, k) for k in fieldnames(Heat)),
                         "drag" => st.dragν === nothing ? nothing : [collect(r) for r in st.dragν],
                         "selffield_bc" => st.bar === nothing ? "slab" : st.open === nothing ? "bar" : "open"),
        "protocol" => Dict(string(k) => getfield(c.protocol, k) for k in fieldnames(Protocol)),
        "H" => c.H,
        "result" => Dict("bracket_pos" => [b.pos.lo, b.pos.hi], "bracket_neg" => [b.neg.lo, b.neg.hi],
                         "eta_percent" => collect(b.eta), "closed" => b.pos.ok && b.neg.ok,
                         # with t_confirm_GL: the brackets of the delay (the silent end followed that long)
                         "delay_bracket_pos" => [b.delay.pos.lo, b.delay.pos.hi],
                         "delay_bracket_neg" => [b.delay.neg.lo, b.delay.neg.hi],
                         "delay_eta_percent" => collect(b.delay.eta),
                         "delay_closed" => b.delay.pos.ok && b.delay.neg.ok,
                         "undecided" => vcat(b.pos.undecided, -b.neg.undecided),
                         # the onsets as total currents through the sample, I = J·W·d × (measured total over
                         # nominal): free of the half cell that makes J ambiguous at O(dx/W) with the slab boundary of 1.x
                         "bracket_I_pos" => _Ibracket(b.pos, r.pos, gr), "bracket_I_neg" => _Ibracket(b.neg, r.neg, gr),
                         "W_d_nominal" => [(gr.y2 - gr.y1 + 1) * gr.dy, slab_thickness(gr)]),
        "ladders" => Dict("J+" => lad(r.pos), "J-" => lad(r.neg)),
        "wall_min" => wall_min)
end
