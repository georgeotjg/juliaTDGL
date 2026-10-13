# Terminal menu for TDGL runs. Closing it NEVER stops Julia. Standard library only: it does not load
# the solver. A package of its own so that its compiled code is cached (precompile workload at the end):
# the first start compiles it once, every later one takes a fraction of a second.
#
#   julia julia/scripts/tdgl_tui.jl              interactive (live runs; a = all)
#   julia julia/scripts/tdgl_tui.jl --once       print the table once (--all: every run)
#   julia julia/scripts/tdgl_tui.jl RUNDIR       details of one run directory
#   julia julia/scripts/tdgl_tui.jl --add-dir [PATH]   watch a folder of runs from now on
#   julia julia/scripts/tdgl_tui.jl --rm-dir PATH      stop watching it;  --dirs lists them
#   julia julia/scripts/tdgl_tui.jl --help
module TDGLTUI
using Dates, Printf, Sockets, TOML
include("json_min.jl"); using .JSONMin

const BASE = normpath(joinpath(@__DIR__, "..", "..", ".."))     # julia/tui/src -> project root
const JULIA = joinpath(BASE, "julia")
const LANZAR = joinpath(JULIA, "scripts", "pin_cpu.sh")
const FORMAS = joinpath(JULIA, "shapes")
const EDITOR = joinpath(JULIA, "editor", "server.jl")
const PRESET = joinpath(JULIA, "scripts", "run_preset.jl")
const RUNS = joinpath(JULIA, "runs")
const CARPETAS = joinpath(get(ENV, "XDG_CONFIG_HOME", joinpath(homedir(), ".config")), "tdgl_tui", "carpetas.txt")

# --- small path helpers (Base has no glob) -----------------------------------------------------
_subdirs(d) = isdir(d) ? [joinpath(d, x) for x in readdir(d) if isdir(joinpath(d, x))] : String[]
_oculta(rel) = any(x -> startswith(x, ".") || startswith(x, "trash_"), splitpath(rel))
"""Files named `name` under `d`, at any depth (`**/name`), skipping hidden and trash_ folders."""
function _rbuscar(d, name)
    out = String[]
    isdir(d) || return out
    for (raiz, dirs, files) in walkdir(d; onerror=_ -> nothing)
        filter!(x -> !startswith(x, ".") && !startswith(x, "trash_"), dirs)
        name in files && push!(out, joinpath(raiz, name))
    end
    sort!(out)
end
"""`d/name`, `d/*/name` and `d/*/*/name` that exist."""
_hasta2(d, name) = filter(isfile, vcat(joinpath(d, name), [joinpath(x, name) for x in _subdirs(d)],
                                        [joinpath(y, name) for x in _subdirs(d) for y in _subdirs(x)]))
_mtime(p) = try mtime(p) catch; 0.0 end
_leer(p) = try read(p, String) catch; "" end
_dentro(p, d) = startswith(abspath(p) * "/", rstrip(abspath(d), '/') * "/") && abspath(p) != abspath(d)

# julia/runs first (runs launched from this menu), then any other runs/ folder of the project up to
# two levels down, listed read-only. Folders added with --add-dir are searched whole; a run found under
# two roots is listed once.
const ROOTS_PROYECTO = String[]       # filled at start (not at precompile time): runs/ folders appear
function _raices_proyecto()
    r = [joinpath(JULIA, "runs")]
    cands = vcat(joinpath(BASE, "runs"),                      # where the examples write
                 [joinpath(x, "runs") for x in _subdirs(BASE)],
                 [joinpath(y, "runs") for x in _subdirs(BASE) for y in _subdirs(x)])
    for d in sort(cands)
        isdir(d) && d != r[1] && !_oculta(relpath(d, BASE)) && push!(r, d)
    end
    empty!(ROOTS_PROYECTO); append!(ROOTS_PROYECTO, r)
end
const EXTRA = String[]
const ROOTS = String[]

function _carpetas_guardadas()
    isfile(CARPETAS) || return String[]
    [strip(x) for x in eachline(CARPETAS) if !isempty(strip(x)) && !startswith(lstrip(x), "#")]
end

function _recargar_carpetas()
    isempty(ROOTS_PROYECTO) && _raices_proyecto()
    extra = vcat(split(get(ENV, "TDGL_TUI_EXTRA", ""), ":"), _carpetas_guardadas())
    extra = unique([abspath(expanduser(p)) for p in extra if !isempty(p)])
    empty!(EXTRA); append!(EXTRA, filter(isdir, extra))
    empty!(ROOTS); append!(ROOTS, ROOTS_PROYECTO); append!(ROOTS, [d for d in EXTRA if !(d in ROOTS_PROYECTO)])
end

"""A path typed, pasted or dragged from a file manager ('...', file:///..., ~/...)."""
function _ruta_pegada(t)
    t = strip(strip(t), ['\'', '"'])
    if startswith(t, "file://")
        t = replace(t[8:end], r"%([0-9A-Fa-f]{2})" => m -> string(Char(parse(UInt8, m[2:3]; base=16))))
    end
    abspath(expanduser(isempty(t) ? "." : t))
end

function _cambiar_carpeta(d, quitar::Bool)
    d = rstrip(d, '/')
    guardadas = [rstrip(abspath(expanduser(x)), '/') for x in _carpetas_guardadas()]
    !quitar && !isdir(d) && return "not a folder: $d"
    quitar && !(d in guardadas) && return "not in the list: $d"
    quitar ? filter!(!=(d), guardadas) : (d in guardadas || push!(guardadas, d))
    mkpath(dirname(CARPETAS)); write(CARPETAS, join(guardadas .* "\n"))
    _recargar_carpetas()
    (quitar ? "no longer watching " : "watching ") * d
end

const CATALOG = [
    ("SIM", "clean box: control, η must be 0"),
    ("WEDGE", "sawtooth notch on face y1 (geometric diode)"),
    ("WEDGE_REF", "WEDGE mirrored in y (sign control)"),
    ("WEDGE_BOTH", "notches on y1 and y2"),
    ("ESFERA", "through cylinder of reduced f (pins, does not rectify)"),
    ("ARRAY", "three cylinders of growing radius"),
    ("ESF_CUNA", "cylinder + notch"),
    ("FGRAD", "sawtooth in f(y), no hole"),
    ("SAWROW", "row of five notches"),
    ("SPHROW", "three cylinders (R=3,6,4)"),
    ("SAWSPH", "notches + cylinders"),
    ("STAGGER", "staggered triangles"),
    ("FUNNEL", "two rows of triangles (funnels); one apex touches y2"),
    ("FUNNEL_IN", "FUNNEL with the apex away from y2"),
    ("TRI_Y1", "one triangle with its base on face y1"),
    ("TRI_Y1_INV", "triangle with its apex on y1"),
    ("RECT_Y1", "11×5 rectangle on y1"),
    ("HEX", "hexagonal hole"),
    ("HEX_STAG", "hexagonal hole + triangles around it"),
]
const USADAS = ("SIM", "WEDGE", "WEDGE_REF", "STAGGER", "FUNNEL")
const CATALOG_VISIBLE = [c for c in CATALOG if c[1] in USADAS]
const CATALOG_OTRAS = [c for c in CATALOG if !(c[1] in USADAS)]
const WIDE = Set(["FUNNEL", "FUNNEL_IN", "TRI_Y1", "TRI_Y1_INV", "RECT_Y1", "STAGGER", "SAWROW", "SAWSPH",
                  "ARRAY", "ESF_CUNA", "HEX", "HEX_STAG", "SPHROW", "WEDGE_BOTH", "FGRAD"])

# --- processes --------------------------------------------------------------------------------
"""True only if a Julia process of THIS run is alive. STATUS 'done' means off (the .pid file stays
and Linux reuses the number)."""
function _alive(pid::AbstractString, status::AbstractString="")
    st = lowercase(status)
    (occursin(" done", st) || endswith(rstrip(st), "done")) && return false
    all(isdigit, pid) && !isempty(pid) || return false
    proc = "/proc/$pid"
    isdir(proc) || return false
    occursin("julia", lowercase(_leer("$proc/comm")))
end

function _ps(pid)
    try
        o = split(read(`ps -o etime=,pcpu= -p $pid`, String))
        length(o) >= 2 && return (String(o[1]), o[2] * "%")
    catch
    end
    ("—", "—")
end

"""True when the script loads the TDGL package (using/import in its first 16 kB)."""
function _carga_tdgl(script)
    cab = try String(read(open(script), 16384)) catch; return false end
    occursin(r"^\s*(using|import)\s+TDGL\b"m, cab)
end

"""Julia processes running a .jl of this project that the menu did not launch (campaign or
verification scripts): one row each, so they show as live."""
function _julia_externos(conocidos::Set{String})
    filas = Dict{String,Any}[]
    for pid in readdir("/proc")
        (all(isdigit, pid) && !(pid in conocidos) && pid != string(getpid())) || continue
        proc = "/proc/$pid"
        occursin("julia", lowercase(_leer("$proc/comm"))) || continue
        args = filter(!isempty, split(_leer("$proc/cmdline"), '\0'))
        i = findfirst(a -> endswith(a, ".jl"), args)
        i === nothing && continue
        jl = args[i]
        cwd = try readlink("$proc/cwd") catch; BASE end
        script = isabspath(jl) ? jl : normpath(joinpath(cwd, jl))
        basename(dirname(script)) == "test" && continue
        # the menu and the editor server themselves are not runs (another open menu would show up live)
        (basename(script) in ("tdgl_tui.jl", "server.jl") || occursin("/tui/", script)) && continue
        (_dentro(script, BASE) || any(d -> _dentro(script, d), EXTRA) || _carga_tdgl(script)) || continue
        rundir = dirname(script)
        et, pc = _ps(pid)
        last, js, n_exp, caso, conv = "—", String[], 0, 0, ""
        outs = try [readlink("$proc/fd/1")] catch; sort(filter(f -> startswith(basename(f), "stdout") && endswith(f, ".txt"), readdir(rundir; join=true))) end
        # only regular files: reading a terminal or a pipe would wait forever
        filter!(isfile, outs)
        for out in outs
            texto = _leer(out); isempty(texto) && continue
            partes = split(texto, r"^=== TDGL two-band solver.*$"m)
            caso = length(partes) - 1
            actual = partes[end]
            hits = collect(eachmatch(r"\[(J[+-])\] J=([+-][0-9.]+).*?conv=(\S)", actual))
            m = match(r"\[J_list\][^\n]*?(\d+) points", actual)
            if !isempty(hits) || m !== nothing
                js = ["J" * h[2] for h in hits]
                n_exp = m === nothing ? 0 : 2 * parse(Int, m[1])
                isempty(hits) || ((last, conv) = ("J" * hits[end][2], hits[end][3]))
            end
        end
        extra = join(args[i+1:end], " ")
        kx = findfirst(d -> _dentro(rundir, d), EXTRA)
        base = _dentro(rundir, BASE) ? BASE : kx === nothing ? nothing : EXTRA[kx]
        sitio = base === nothing ? replace(rundir, homedir() => "~"; count=1) :
                relpath(rundir, base == BASE ? BASE : dirname(base))
        tag = sitio * (isempty(extra) ? "" : " [$extra]")
        push!(filas, Dict{String,Any}("tag" => tag * (caso > 1 ? " · case $caso" : ""), "on" => true,
            "etime" => et, "pcpu" => pc, "last" => last, "age_min" => nothing, "dir" => rundir,
            "status" => "running (script)", "pid" => pid, "n_j" => length(js), "js" => js,
            "job" => Dict{String,Any}("n_expected" => n_exp), "hist" => isempty(conv) ? Dict{String,String}[] : [Dict("conv" => conv)],
            "stdout" => isempty(outs) ? "" : outs[1], "frozen" => true, "mtime" => time()))
    end
    filas
end

# --- quick results: the verdict rule of the papers, read from j_history.csv --------------------
"""Physical time of one cycle (dt*lMax): run_info.txt next to the history, else the run summary,
else the dt printed in a stdout of the folder or its parent."""
function _tciclo(d)
    for ri in (joinpath(d, "run_info.txt"), joinpath(d, "output", "run_info.txt"))
        m = match(r"t_cycle=([0-9.eE+-]+)", _leer(ri)); m === nothing || return parse(Float64, m[1])
    end
    v = _tciclo_resumen(d); v === nothing || return v
    for s in vcat(_lista(d, r"^stdout.*\.txt$"), _lista(dirname(d), Regex("^stdout.*" * basename(d) * ".*\\.txt\$")),
                  _lista(dirname(d), r"^stdout\.txt$"))
        dts = collect(eachmatch(r"CFL: dt=([0-9.eE+-]+)", _leer(s)))
        isempty(dts) || return parse(Float64, dts[end][1]) * 40
    end
    nothing
end
_lista(d, re) = isdir(d) ? sort([joinpath(d, f) for f in readdir(d) if occursin(re, f)]) : String[]

function _tciclo_resumen(d)
    for s in reverse(sort(vcat(_lista(d, r"^RUN_SUMMARY_.*\.txt$"), _lista(joinpath(d, "output"), r"^RUN_SUMMARY_.*\.txt$"),
                               _lista(dirname(d), r"^RUN_SUMMARY_.*\.txt$"))))
        t = _leer(s); a = match(r"dt=([0-9.eE+-]+) t_GL", t); b = match(r"lMax=(\d+)", t)
        (a === nothing || b === nothing) || return parse(Float64, a[1]) * parse(Int, b[1])
    end
    nothing
end

"""rows = (|J|, V, cycles) of one polarity, ascending. Same rule as TDGL.bracket_Ic: crossings per
line |V| t / 2π, <1 silent, >=10 dissipating."""
function _bracket(rows, tc)
    cls = Tuple{Float64,String,Float64,Float64}[]
    for (j, v, it) in rows
        c = abs(v) * it * tc / 2π
        push!(cls, (j, c >= 10 ? "D" : c < 1 ? "S" : "U", c, v))
    end
    d = findfirst(x -> x[2] == "D", cls)
    hi = d === nothing ? Inf : cls[d][1]
    below = d === nothing ? cls : cls[1:d-1]
    s = [x[1] for x in below if x[2] == "S"]
    lo = isempty(s) ? NaN : s[end]
    und = [x[1] for x in below if x[2] == "U" && (isnan(lo) || x[1] > lo)]
    (lo=lo, hi=hi, und=und, cls=cls)
end

function _csv(p)
    lineas = split(_leer(p), '\n'; keepempty=false)
    isempty(lineas) && return Dict{String,String}[]
    hdr = strip.(split(lineas[1], ','))
    [Dict(string(i <= length(hdr) ? hdr[i] : i) => String(strip(x)) for (i, x) in enumerate(split(l, ',')))
     for l in lineas[2:end] if length(split(l, ',')) >= 5]
end

"""One line per case (j_history.csv) under d: brackets, η with its range, undecided rungs and σ_V
where both polarities dissipate."""
function quick_result(d)
    out = String[]
    for csvp in unique(_hasta2(d, "j_history.csv"))
        basename(dirname(csvp)) == "output" && continue
        cd_ = dirname(csvp)
        tc = something(_tciclo(cd_), _tciclo(joinpath(cd_, "output")), Some(nothing))
        last = Dict{Tuple{String,Float64},Tuple{Float64,Int}}()
        for r in _csv(csvp)
            try
                j = abs(parse(Float64, r["J"]))
                j > 1e-12 && (last[(r["pol"], round(j; digits=6))] = (parse(Float64, r["V"]), round(Int, parse(Float64, get(r, "it", "0")))))
            catch
            end
        end
        caso = cd_ == d ? basename(d) : relpath(cd_, d)
        if tc === nothing || isempty(last)
            push!(out, "$caso: (no summary with dt yet)"); continue
        end
        B = Dict(pol => _bracket(sort([(j, v, it) for ((p_, j), (v, it)) in last if p_ == pol]), tc) for pol in ("J+", "J-"))
        f(x) = isnan(x) ? "?" : isinf(x) ? "∞" : _g4(x)
        bp, bn = B["J+"], B["J-"]
        br(b) = isempty(b.cls) ? "— (not started)" : "($(f(b.lo)), $(f(b.hi)))"
        line = "$caso: Ic+ ∈ $(br(bp))  |Ic−| ∈ $(br(bn))"
        if all(isfinite, (bp.lo, bp.hi, bn.lo, bn.hi))
            e(a, b) = 100 * (a - b) / (a + b)
            line *= @sprintf("  η = %+.1f %% [%+.1f, %+.1f]", e((bp.lo + bp.hi) / 2, (bn.lo + bn.hi) / 2), e(bp.lo, bn.hi), e(bp.hi, bn.lo))
        end
        und = vcat(["+" * f(x) for x in bp.und], ["−" * f(x) for x in bn.und])
        isempty(und) || (line *= "  undecided: " * join(und, " "))
        dp = Dict(x[1] => x[4] for x in bp.cls if x[2] == "D"); dn = Dict(x[1] => x[4] for x in bn.cls if x[2] == "D")
        com = sort(collect(intersect(keys(dp), keys(dn))))
        if !isempty(com)
            j = com[1]; a, b = abs(dp[j]), abs(dn[j])
            line *= @sprintf("  σ_V@%s = %+.1f %%", f(j), 100 * (a - b) / (a + b))
        end
        push!(out, line)
    end
    out
end
_g4(x) = (s = @sprintf("%.4g", x); occursin('e', s) ? s : s)

"""Breaks a result line before 'undecided', 'σ_V' or 'η =' when it does not fit."""
function _envolver(lineas, ancho)
    out = String[]
    for ln in lineas
        if _vis(ln) <= ancho
            push!(out, ln)
        else
            partes = split(ln, r"  (?=undecided:|σ_V@|η =)")
            push!(out, partes[1]); append!(out, ["      " * x for x in partes[2:end]])
        end
    end
    out
end

_notificar(t, c) = Sys.which("notify-send") === nothing ||
    run(pipeline(`notify-send -a TDGL $t $c`; stdout=devnull, stderr=devnull); wait=false)

const TERMINADAS = String[]
const _VIGIA = Ref(true)          # off only in the precompile workload

"""Background watcher: when a live run disappears, notify with its result."""
function _vigia(intervalo=30.0)
    vivos = Dict{String,Tuple{String,String}}()
    while true
        try
            ahora = Dict(r["pid"] => (r["tag"], r["dir"]) for r in collect_runs() if r["on"] && !isempty(r["pid"]))
            for (pid, (tag, d)) in vivos
                haskey(ahora, pid) && continue
                res = quick_result(d)
                push!(TERMINADAS, Dates.format(now(), "HH:MM") * " $tag: " * (isempty(res) ? "finished" : res[end]))
                _notificar("TDGL: finished $tag", isempty(res) ? d : join(res[max(1, end - 2):end], "\n"))
            end
            vivos = ahora
        catch
        end
        sleep(intervalo)
    end
end

function _resultados(rows)
    println()
    lin, vistos = String[], Set{String}()
    for r in rows
        res = [x for x in quick_result(r["dir"]) if !(x in vistos)]
        union!(vistos, res)
        isempty(res) && continue
        push!(lin, _col("1", r["tag"]) * (r["on"] ? "  " * _dim("(live)") : ""))
        append!(lin, ["  " * x for x in res])
    end
    caja("results — verdict rule: crossings per line <1 silent, ≥10 dissipating",
         isempty(lin) ? ["no j_history.csv yet"] : _envolver(lin, _ancho() - 6))
    _volver()
end

_jtag(name) = (m = match(r"(J[+-][0-9.]+)", name); m === nothing ? name : m[1])

function _job_json(rundir)
    p = joinpath(rundir, "job.json")
    isfile(p) || return Dict{String,Any}()
    try
        v = json_parse(read(p, String)); v isa Dict ? v : Dict{String,Any}()
    catch
        Dict{String,Any}()
    end
end

function _j_history(rundir)
    for cand in vcat(joinpath(rundir, "j_history.csv"), _rbuscar(rundir, "j_history.csv"))
        isfile(cand) || continue
        rows = _csv(cand)
        isempty(rows) || return rows
    end
    Dict{String,String}[]
end

function _j_points(rundir, hist)
    if !isempty(hist)
        tags = String[]
        for rec in hist
            t = try @sprintf("J%+.3f", parse(Float64, rec["J"])) catch; get(rec, "pol", "") * get(rec, "J", "") end
            t in tags || push!(tags, t)
        end
        return tags
    end
    pts = Tuple{Float64,String}[]; vistos = Set{String}()
    for (raiz, _, files) in walkdir(rundir; onerror=_ -> nothing), f in files
        endswith(f, "psi_s.pgm") || continue
        p = joinpath(raiz, f); rel = splitpath(relpath(p, rundir))
        tag = strip((length(rel) > 1 ? rel[1] : "") * " " * _jtag(f))
        tag in vistos && continue
        push!(vistos, tag); push!(pts, (_mtime(p), tag))
    end
    [t for (_, t) in sort(pts)]
end

function _log_js(rundir)
    out = String[]
    for cand in (joinpath(rundir, "log.txt"), joinpath(rundir, "stdout.txt"))
        for line in split(_leer(cand), '\n')
            occursin(r"J=[+-]", line) && occursin("V=", line) || continue
            (occursin("it=0", line) && occursin("conv= ", line)) && continue
            push!(out, strip(line))
        end
    end
    unique(out)
end

function collect_runs()
    ahora = time(); rows = Dict{String,Any}[]; vistos = Set{String}()
    for root in ROOTS
        isdir(root) || continue
        for status in _rbuscar(root, "STATUS.txt")
            rundir = dirname(status)
            (basename(rundir) in ("output", "restart", "figures", "snapshots") || rundir in vistos) && continue
            push!(vistos, rundir)
            lineas = split(strip(_leer(status)), '\n')
            st = isempty(lineas) ? "" : String(lineas[end])
            pidf = joinpath(rundir, "julia.pid")
            pid = isfile(pidf) ? strip(_leer(pidf)) : ""
            on = _alive(pid, st)
            et, pc = on ? _ps(pid) : ("—", "—")
            last, age_min = "—", nothing
            pgms = String[]
            for (raiz, _, files) in walkdir(rundir; onerror=_ -> nothing), f in files
                endswith(f, "psi_s.pgm") && push!(pgms, joinpath(raiz, f))
            end
            if !isempty(pgms)
                p = pgms[argmax(_mtime.(pgms))]
                last = _jtag(basename(p)); age_min = floor(Int, (ahora - _mtime(p)) / 60)
            end
            hist = _j_history(rundir)
            if !isempty(hist)
                try last = @sprintf("J%+.3f", parse(Float64, hist[end]["J"])) catch end
            end
            partes = splitpath(relpath(rundir, root))
            tag = join(partes[1:min(2, end)], "/")
            pts = _j_points(rundir, hist)
            mt = maximum(_mtime.(vcat(status, _hasta2(rundir, "j_history.csv"))))
            push!(rows, Dict{String,Any}("tag" => tag, "on" => on, "etime" => et, "pcpu" => pc, "last" => last,
                "age_min" => age_min, "dir" => rundir, "status" => st, "pid" => pid, "n_j" => length(pts),
                "js" => pts, "job" => _job_json(rundir), "hist" => hist, "frozen" => root != ROOTS[1], "mtime" => mt))
        end
    end
    append!(rows, _julia_externos(Set{String}(String(r["pid"]) for r in rows if r["on"])))
    sort!(rows; by=r -> (!r["on"], -get(r, "mtime", ahora)))     # live first, then the most recent
    rows
end

"""Run without job.json: 'J=+0.56,+0.64' → (those |J| ∪ {0}) × 2 polarities."""
function _n_exp_from_status(st)
    m = match(r"J=([0-9eE+.,\-]+)", st); m === nothing && return 0
    js = Float64[]
    for p in split(m[1], ','); x = tryparse(Float64, p); x === nothing || push!(js, abs(x)); end
    isempty(js) ? 0 : 2 * length(unique(vcat(0.0, js)))
end

function _pct(row)
    job = row["job"]
    n_exp = try Int(round(Float64(something(get(job, "n_expected", 0), 0)))) catch; 0 end
    n = row["n_j"]
    n_exp <= 0 && occursin("HARD1200", row["tag"]) && (n_exp = 4)
    n_exp <= 0 && occursin(",", row["status"]) && (n_exp = _n_exp_from_status(row["status"]))
    (n_exp <= 0 || n > n_exp) && return n > 0 ? "$n pts" : "—"
    frac = n / n_exp
    (row["on"] && n < n_exp && row["age_min"] !== nothing) && (frac = min(0.99, frac + 0.05))
    @sprintf("%.0f%% %d/%d", 100frac, n, n_exp)
end

# --- drawing ------------------------------------------------------------------------------------
_tty() = stdout isa Base.TTY
_col(code, s) = _tty() ? "\e[$(code)m$(s)\e[0m" : s
_dim(s) = _col("2", s)
const _ANSI = r"\e\[[0-9;]*m"
_vis(s) = length(replace(s, _ANSI => ""))
_ancho() = clamp(_tty() ? displaysize(stdout)[2] : something(tryparse(Int, get(ENV, "COLUMNS", "100")), 100), 76, 130)
_corta(s, n) = _vis(s) <= n ? s : first(replace(s, _ANSI => ""), max(0, n - 1)) * "…"

"""Cut a run name to n characters. For "folder/run" the folder goes first (it is shared by many
runs), down to a few characters; then the run is cut in the middle, since names differ at the end
and often in the middle too (field, sign)."""
function _corta_medio(s, n)
    length(s) <= n && return s
    if occursin('/', s)
        car, run = rsplit(s, '/'; limit=2)
        resto = n - length(run) - 1
        resto >= 4 && return length(car) > resto ? first(car, resto - 1) * "…/" * run : car * "/" * run
        s = first(car, 3) * "…/" * run
        length(s) <= n && return s
    end
    cab = (n - 1) ÷ 2
    first(s, cab) * "…" * last(s, n - 1 - cab)
end

"""A box as wide as the terminal, title on the left."""
function caja(titulo, lineas; derecha="", color="36")
    W = _ancho(); t = " $titulo "; d = isempty(derecha) ? "" : " $derecha "
    println(_dim("┌─") * _col("1;" * color, t) * _dim("─"^max(0, W - 3 - _vis(t) - _vis(d)) * d * "┐"))
    for ln in lineas
        ln = _corta(ln, W - 4)
        println(_dim("│ ") * ln * " "^max(0, W - 4 - _vis(ln)) * _dim(" │"))
    end
    println(_dim("└" * "─"^(W - 2) * "┘"))
end

_K(l) = _col("1;36", "[$l]")
const _ENTRADA = Ref{IO}(devnull)   # the keyboard; set in main (a buffer of keys in the workload)
function _input(prompt)
    print(prompt); flush(stdout)
    io = _ENTRADA[]
    eof(io) && throw(EOFError())
    String(strip(readline(io)))
end
_volver() = _input("$(_K("Enter")) back ")
_clear() = _tty() && print("\e[H\e[2J")

function _motor(r)
    r["status"] == "running (script)" && return "script"
    r["frozen"] && return "archive"
    t = _term_de_job(r["job"])
    isempty(t) ? "base" : haskey(t, "drag") ? "drag" : haskey(t, "calor") ? "heat" : "Q diff."
end

function _estado(r)
    st = lowercase(r["status"])
    r["on"] && return _col("1;32", "● live")
    (occursin("done", st) || occursin("terminad", st)) && return _dim("✓ done")
    occursin("stop", st) && return _col("33", "■ stopped")
    _dim("· idle")
end

function _barra(r)
    pc = _pct(r); m = match(r"(\d+)/(\d+)", pc)
    m === nothing && return _dim(endswith(pc, "pts") ? pc : "—")
    a, b = parse(Int, m[1]), parse(Int, m[2])
    k = b > 0 ? min(8, round(Int, 8a / b)) : 0
    _col("36", "█"^k) * _dim("░"^(8 - k)) * " $a/$b"
end

function _ultimo(r)
    h = r["hist"]; conv = isempty(h) ? "" : get(h[end], "conv", "")
    r["last"] * (isempty(conv) ? "" : " " * _col(get(Dict("V" => "33", "T" => "31", "F" => "2"), conv, "0"), conv))
end

_pad(s, n) = s * " "^max(0, n - _vis(s))

function table(rows; only_on::Bool)
    show = only_on ? filter(r -> r["on"], rows) : rows
    W = _ancho(); wjob = max(18, W - 4 - 62)
    cab = rpad("#", 3) * " " * rpad("run", wjob) * " " * rpad("term", 8) * " " * rpad("state", 9) * " " *
          rpad("progress", 14) * " " * rpad("last J", 13) * " " * lpad("%CPU", 6)
    filas = [_dim("Closing this does not stop Julia."), "", _col("1", cab)]
    isempty(show) && push!(filas, _dim(only_on ? "  (no live runs: [a] shows all)" : "  (empty)"))
    for (i, r) in enumerate(show)
        push!(filas, join([rpad(string(i), 3), _pad(_corta_medio(r["tag"], wjob), wjob), rpad(_motor(r), 8),
                           _pad(_estado(r), 9), _pad(_barra(r), 14), _pad(_ultimo(r), 13), lpad(r["pcpu"], 6)], " "))
    end
    caja("TDGL · runs", filas; derecha=Dates.format(now(), "yyyy-mm-dd HH:MM") * " · " * (only_on ? "live only" : "all"))
end

function _ascii_curve(hist, pol; width=52, height=8)
    pts = Tuple{Float64,Float64}[]
    for rec in hist
        j = tryparse(Float64, get(rec, "J", "")); v = tryparse(Float64, get(rec, "V", ""))
        (j === nothing || v === nothing) && continue
        p = get(rec, "pol", "")
        pol == "+" && (j < -1e-12 || endswith(p, "-")) && continue
        pol == "-" && (j > 1e-12 || endswith(p, "+")) && continue
        push!(pts, (abs(j), v))
    end
    length(pts) < 2 && return
    sort!(pts); ys = last.(pts); lo, hi = extrema(ys); hi - lo < 1e-18 && (hi = lo + 1e-12)
    @printf("  V(J) polarity %s   %.2f → %.2f\n", pol, pts[1][1], pts[end][1])
    grid = fill(' ', height, width)
    for (j, v) in pts
        x = clamp(floor(Int, (j - pts[1][1]) / max(pts[end][1] - pts[1][1], 1e-12) * (width - 1)), 0, width - 1)
        y = clamp(floor(Int, (hi - v) / (hi - lo) * (height - 1)), 0, height - 1)
        grid[y+1, x+1] = '•'
    end
    for k in 1:height; println("  |", String(grid[k, :])); end
    println("  +", "-"^width)
end

_fv(v) = (x = tryparse(Float64, v); x === nothing ? first(v, 12) : @sprintf("%+.4e", x))

function detail(r)
    d = r["dir"]; job = r["job"]
    println()
    res = quick_result(d)
    isempty(res) || caja("result (verdict rule)", _envolver(res, _ancho() - 6); color="32")
    caja(r["tag"], ["folder   $d", "state    $(_estado(r))   $(_dim(r["status"]))",
                    "term     $(_motor(r))   pid $(isempty(r["pid"]) ? "—" : r["pid"])   progress $(_barra(r))",
                    "         $(_term_texto(_term_de_job(job)))"])
    if !_vacio(get(job, "geom", nothing))
        g(k) = string(something(get(job, k, ""), ""))
        bits = ["geom=$(g("geom"))", "κ=$(g("kappa"))", "H=$(g("H"))", "γ12=$(g("gamma12"))"]
        _vacio(get(job, "Nx", nothing)) || push!(bits, "mesh=$(g("Nx"))x$(g("Ny"))x$(g("Nz")) dx=$(g("dx"))")
        _vacio(get(job, "t_min_GL", nothing)) || push!(bits, "t=$(g("t_min_GL"))/$(g("t_min_GL_res"))")
        something(get(job, "T_red", 0), 0) != 0 && push!(bits, "T=$(g("T_red"))")
        println("params ", join(bits, " "))
        jl = get(job, "J_list", nothing)
        (jl === nothing || isempty(jl)) || println("J list $(jl isa AbstractVector ? "[" * join(string.(jl), ", ") * "]" : jl)   expected $(g("n_expected")) closed points (both polarities)")
    end
    csvd = d
    if r["status"] == "running (script)"
        cands = filter(c -> !("output" in splitpath(relpath(c, d))), _rbuscar(d, "j_history.csv"))
        hist = Dict{String,String}[]
        if !isempty(cands)
            so = splitext(basename(get(r, "stdout", "")))[1]
            mio = filter(c -> !isempty(basename(dirname(c))) && occursin(basename(dirname(c)), so), cands)
            pool = isempty(mio) ? cands : mio
            csvd = dirname(pool[argmax(_mtime.(pool))]); hist = _j_history(csvd)
            println("case   $(csvd == d ? basename(d) : relpath(csvd, d))  (the one running now)")
        end
    else
        hist = isempty(r["hist"]) ? _j_history(d) : r["hist"]
    end
    if !isempty(hist)
        println("--- history (one line per closed current point; j_history.csv) ---")
        println("  ", rpad("time", 20), " ", rpad("pol", 4), " ", lpad("J", 8), " ", lpad("V", 12), " ", rpad("class", 5), " Nv  |Ψπ|²")
        for rec in hist
            println("  ", rpad(get(rec, "time", ""), 20), " ", rpad(get(rec, "pol", ""), 4), " ",
                    lpad(first(get(rec, "J", ""), 8), 8), " ", lpad(_fv(get(rec, "V", "")), 12), " ",
                    rpad(get(rec, "conv", ""), 5), " ", get(rec, "Nv_s", ""), "/", get(rec, "Nv_p", ""), "  ", get(rec, "psi_p2", ""))
        end
        c = joinpath(csvd, "j_history.csv")
        println("  csv  ", isfile(c) ? c : joinpath(d, "**", "j_history.csv"))
        _ascii_curve(hist, "+"); _ascii_curve(hist, "-")
    else
        js = r["js"]
        println("closed points ($(length(js))): ", isempty(js) ? "(none yet)" : join(js, " → "))
        logs = _log_js(d)
        if isempty(logs)
            println("(no V yet: results are written when each current point CLOSES)")
        else
            println("--- V in log.txt ---"); foreach(l -> println("  ", l), logs[max(1, end - 19):end])
        end
    end
    println("Checkpoint = last closed point. Stopping in the middle of a point loses that point.")
    println("Classes: F fixed point (no flow), V limit cycle (flux flow), T heat cap reached, * cycle ceiling.")
    if r["status"] == "running (script)" || r["frozen"]
        println(r["frozen"] && r["status"] != "running (script)" ? "Article run (archive): read only." :
                "Launched by a script, outside the menu: read only here.")
        println(); println("$(_K("Enter")) or $(_K("b")) back   $(_K("o")) folder   $(_K("e")) csv   $(_K("f")) figures")
    else
        println(); println("$(_K("Enter")) or $(_K("b")) back   $(_K("s")) stop   $(_K("u")) resume   $(_K("o")) folder   " *
                           "$(_K("d")) to trash_   $(_K("e")) csv   $(_K("f")) figures")
    end
    k = lowercase(_input("> "))
    if k == "o"; _open_dir(d)
    elseif k == "s" && !r["frozen"]; _stop_job(r)
    elseif k == "u" && !r["frozen"]; _resume_job(r)
    elseif k == "d"; _delete_job(r)
    elseif k == "e"
        c = joinpath(csvd, "j_history.csv"); println(isfile(c) ? c : "(no j_history.csv yet)"); _volver()
    elseif k == "f"; _figuras(r)
    end
end

_open_dir(d) = (println(d); _abrir(d))
"""xdg-open without waiting: the viewer does not freeze the menu."""
_abrir(cosa) = Sys.which("xdg-open") === nothing ? println("(no xdg-open: open $cosa by hand)") :
    run(pipeline(_suelto(["xdg-open", cosa]); stdout=devnull, stderr=devnull); wait=false)
"""A command in its own session (setsid), so closing the menu or its terminal does not stop it."""
_suelto(v; dir=BASE) = Cmd(Cmd(Sys.which("setsid") === nothing ? v : vcat("setsid", v)); dir=dir)

function _puerto_vivo(p)
    try
        close(connect(ip"127.0.0.1", p)); true
    catch
        false
    end
end

"""Starts the editor server if it is not running and opens the browser."""
function _editor_url(cargar="")
    if !_puerto_vivo(8765)
        run(pipeline(_suelto([Base.julia_cmd().exec[1], "--startup-file=no", EDITOR, "--no-browser"]); stdout=devnull, stderr=devnull); wait=false)
        for _ in 1:40; _puerto_vivo(8765) && break; sleep(0.2); end
    end
    url = "http://127.0.0.1:8765/" * (isempty(cargar) ? "" : "?load=$cargar")
    _abrir(url); url
end

"""Moves julia/runs/<run> to trash_/tui_jobs/. Never article runs, never a delete."""
function _delete_job(r)
    d = abspath(r["dir"]); root = abspath(RUNS); trash = joinpath(BASE, "trash_", "tui_jobs")
    println()
    r["on"] && (println("it is running. Stop it first (s)."); return _volver())
    r["frozen"] && (println("article run: not touched from here."); return _volver())
    _dentro(d, root) || (println("outside julia/runs/, not touched: $d"); return _volver())
    dest = joinpath(trash, r["tag"])
    ispath(dest) && (dest = joinpath(trash, r["tag"] * "_" * Dates.format(now(), "yyyymmdd_HHMMSS")))
    println("move  $d\n  →   $dest\nnothing is deleted; trash_ is emptied by hand.")
    conf = _input("type exactly:  TRASH $(r["tag"])\n> ")
    conf == "TRASH $(r["tag"])" || (println("cancelled"); return _volver())
    mkpath(dirname(dest)); mv(d, dest)
    println("moved to $dest"); _volver()
end

"""Shape editor in the browser (local server on 127.0.0.1:8765)."""
function _abrir_editor(cargar="")
    isfile(EDITOR) || (println("editor not found: $EDITOR"); return _volver())
    println("editor: $(_editor_url(cargar))   (saves to julia/shapes/)")
    println(_dim("a small local Julia server; it stops by itself 30 min after the editor tab is closed"))
    _volver()
end

"""Returns (geom, path) for a shape in julia/shapes/, or nothing (back)."""
function _elegir_forma()
    fs = isdir(FORMAS) ? sort(filter(f -> endswith(f, ".toml"), readdir(FORMAS; join=true))) : String[]
    isempty(fs) && (println("no shapes in $FORMAS — draw one with g → e"); return nothing)
    stem(f) = splitext(basename(f))[1]
    mias = filter(f -> !startswith(stem(f), "cat_"), fs); cats = filter(f -> startswith(stem(f), "cat_"), fs)
    orden = vcat(mias, cats)
    if !isempty(mias)
        println(_col("1", "  your shapes"))
        for (n, f) in enumerate(mias); println("  ", _col("1;36", lpad(n, 2)), "  ", stem(f)); end
    end
    if !isempty(cats)
        println(_col("1", "  catalogue templates (box 40×24×6)"))
        println("  ", join(("$(_col("1;36", string(length(mias) + n)))=$(stem(f)[5:end])" for (n, f) in enumerate(cats)), "  "))
    end
    nuevo = isempty(mias) ? 1 : argmax(_mtime.(mias))
    isempty(mias) || println(_dim("  default = the most recently saved: $(stem(mias[nuevo]))"))
    k = _ask("shape (number)", string(nuevo))
    n = k === nothing ? nothing : tryparse(Int, k)
    (n === nothing || !(1 <= n <= length(orden))) && return nothing
    ref = _ask("mirror in y (_REF)? y/N", "N"); ref === nothing && return nothing
    (lowercase(ref) in ("y", "s", "si") ? "CUSTOM_REF" : "CUSTOM", orden[n])
end

function _geom_menu()
    lin = [_dim("The sample is the box; a geometry is where the condensation strength f is reduced."),
           _dim("J runs along x and H along z, so vortices move along y: a diode needs a shape that"),
           _dim("is not symmetric under the mirror in y."), "", _col("1", "most used")]
    for (i, (n, nota)) in enumerate(CATALOG_VISIBLE); push!(lin, "  $(_K(string(i))) $(rpad(n, 10)) $(_dim(nota))"); end
    append!(lin, ["", "$(_K("e")) shape editor (draw, rotate, cut planes, voxels)",
                  "$(_K("p")) templates: the other $(length(CATALOG_OTRAS)) catalogue shapes",
                  _dim("number = open that shape in the editor  ·  Enter or b = back")])
    caja("geometries", lin)
    k = lowercase(_input("geom> "))
    if k == "e"; _abrir_editor()
    elseif k == "p"
        l2 = ["  $(_K(string(i))) $(rpad(n, 10)) $(_dim(nota))" for (i, (n, nota)) in enumerate(CATALOG_OTRAS)]
        push!(l2, _dim("number = open in the editor  ·  Enter or b = back")); caja("templates", l2)
        n = tryparse(Int, _input("template> "))
        (n !== nothing && 1 <= n <= length(CATALOG_OTRAS)) && _abrir_editor("cat_" * CATALOG_OTRAS[n][1])
    else
        n = tryparse(Int, k)
        (n !== nothing && 1 <= n <= length(CATALOG_VISIBLE)) && _abrir_editor("cat_" * CATALOG_VISIBLE[n][1])
    end
end

"""Input with a default; returns nothing when the user types b (back)."""
function _ask(prompt, default; back=true)
    s = _input("$prompt [$default]$(back ? "  b=back" : "")> ")
    back && lowercase(s) in ("b", "-", "q") && return nothing
    isempty(s) ? default : s
end

function _term_texto(t)
    isempty(t) && return "none (two-band TDGL only)"
    haskey(t, "drag") && return "drag ν=$(get(t, "nu", "0.05"))"
    haskey(t, "calor") && return "heat γ=$(t["calor"])" * (get(t, "joule", "0") == "1" ? " + Joule" : "") * ", margin $(get(t, "margen", "0"))"
    haskey(t, "qss") && return "diffused residual (Φ on, λ_TF=2, τ_Q=1, D_Q=0.25)"
    "?"
end

_vacio(x) = x === nothing || x === false || x == "" || x == "null" || x == 0
_nulo(x) = x === nothing || x == "" || x == "null"
"""The optional term a run used, read from its job.json (run_preset names or Config names)."""
function _term_de_job(job)
    (!_vacio(get(job, "drag", nothing)) || !_vacio(get(job, "use_drag", nothing))) &&
        return Dict("drag" => "1", "nu" => string(get(job, "nu_drag", 0.05)))
    calor = get(job, "calor", nothing)
    (_nulo(calor) && !_vacio(get(job, "use_heat", nothing))) && (calor = get(job, "gamma_bath", 0.2))
    if !_nulo(calor)
        return Dict("calor" => string(calor),
                    "joule" => (!_vacio(get(job, "joule", nothing)) || !_vacio(get(job, "use_joule", nothing))) ? "1" : "0",
                    "margen" => string(get(job, "margen", get(job, "heat_margin", get(job, "calor_margen", 0)))))
    end
    (!_vacio(get(job, "qss", nothing)) || !_vacio(get(job, "use_Qss", nothing))) && return Dict("qss" => "1")
    Dict{String,String}()
end

_phi_texto(phi) = startswith(phi, "tf") ? "Thomas–Fermi, λ_TF = $(occursin(',', phi) ? split(phi, ',')[2] : "2")" :
    get(Dict("off" => "off (gauge Φ = 0)", "coulomb" => "Coulomb projection"), phi, phi)

"""One term at a time: with two on, a change cannot be attributed. nothing = back."""
function _elegir_termino(previo, phi)
    previo = something(previo, Dict{String,String}())
    println("  $(_K("0")) none           two-band TDGL only (with Φ as chosen in step 4)")
    println("  $(_K("1")) drag ν         mixed-gradient coupling between the bands")
    println("  $(_K("2")) heat θ(r,t)    dissipation heats the sample; bath γ; Joule and contact margin")
    println("  $(_K("3")) diffused Q     diffusion of the Coulomb-projection residual (turns Φ on)")
    actual = haskey(previo, "drag") ? "1" : haskey(previo, "calor") ? "2" : haskey(previo, "qss") ? "3" : "0"
    k = _ask("term", actual); k === nothing && return nothing
    k == "0" && return Dict{String,String}()
    if k == "1"
        nu = _ask("ν, coupling strength (0 to ~0.3)", get(previo, "nu", "0.05"))
        return nu === nothing ? nothing : Dict("drag" => "1", "nu" => nu)
    elseif k == "2"
        g = _ask("bath γ, how fast the sample cools (0.2 strong · 0.04 weak · 0 none)", get(previo, "calor", "0.08"))
        g === nothing && return nothing
        if phi != "off"
            println(_dim("  Joule heating needs Φ off (with Φ, E carries −∇Φ; not verified): left out")); j = "n"
        else
            j = _ask("include Joule heating |E|²? Y/n", "Y")
        end
        j === nothing && return nothing
        m = _ask("contact margin: cells next to the injection faces left out of the heat source", "8")
        m === nothing && return nothing
        return Dict("calor" => g, "joule" => lowercase(j) in ("n", "no") ? "0" : "1", "margen" => m)
    elseif k == "3"
        return Dict("qss" => "1")
    end
    println("unknown term"); nothing
end

# --- run parameters: key, name, units, what it is, validation type -----------------------------
const PARAMS = [
    ("kappa", "κ", "", "λ/ξ; 2 = soft vortices, many of them · 25 = strongly type II, few vortices", "num>0"),
    ("H", "H", "H_c2", "applied field along z", "num>=0"),
    ("xi", "ξπ/ξσ", "", "coherence length of the π band relative to σ (sets its mass)", "num>0"),
    ("g12", "γ12", "", "interband Josephson coupling (0.001 nearly free · 0.1 locked)", "num>=0"),
    ("T", "bath T", "T_c", "0 = no temperature; >0 is felt by both bands", "0<=num<1"),
    ("J", "current J", "J₀", "from, to, step (J=0 is always added; both polarities run)", "rango"),
    ("malla", "mesh", "cells, ξσ", "Nx×Ny×Nz of the sample and dx (current along x, field along z)", "malla"),
    ("t", "time per J", "t_GL", "minimum if the point is static / if it flows (voltage window)", "tiempos"),
    ("z", "defect in z", "", "through = full thickness · half = lower half · buried = inside", "z"),
    ("cpus", "cores", "", "CPUs to pin the run to (e.g. 0,1); one thread per run", "texto"),
    ("scheme", "scheme", "", "prod = fast (1.1) · legacy = explicit, as in 1.0.0", "scheme"),
]
const PRESETS = [
    ("quick test", "κ=2, box 20×14×4, one current; minutes: check that it runs",
     Dict("kappa" => "2", "H" => "0.5", "xi" => "1.3", "g12" => "0.001", "T" => "0", "J" => "0.10,0.10,0.10",
          "malla" => "20x14x4,0.5", "t" => "20,40", "z" => "through", "cpus" => "8,9", "scheme" => "prod")),
    ("production box, κ=2", "40×24×6, vortex flow; minutes per current (prod)",
     Dict("kappa" => "2", "H" => "0.03", "xi" => "4", "g12" => "0.001", "T" => "0", "J" => "0.40,0.80,0.40",
          "malla" => "40x24x6,0.5", "t" => "20,200", "z" => "through", "cpus" => "8,9", "scheme" => "prod")),
    ("production box, κ=25", "40×24×6, onset near J≈0.7; minutes per current (prod; hours with legacy)",
     Dict("kappa" => "25", "H" => "0.03", "xi" => "4", "g12" => "0.001", "T" => "0", "J" => "0.68,0.76,0.02",
          "malla" => "40x24x6,0.5", "t" => "200,1200", "z" => "through", "cpus" => "0,1", "scheme" => "prod")),
    ("custom", "your own mesh, κ, field, currents and times (asked in step 3)",
     Dict("kappa" => "2", "H" => "0.03", "xi" => "4", "g12" => "0.001", "T" => "0", "J" => "0.40,0.80,0.40",
          "malla" => "40x24x6,0.5", "t" => "20,200", "z" => "through", "cpus" => "8,9", "scheme" => "prod")),
]

function _lista_J(rango)
    a, b, d = parse.(Float64, split(rango, ','))
    (d <= 0 || b < a) && throw(ArgumentError("J range: to >= from and step > 0"))
    n = round(Int, (b - a) / d) + 1
    n > 60 && throw(ArgumentError("more than 60 current values"))
    [round(a + i * d; digits=6) for i in 0:n-1]
end
_gfmt(x) = (s = @sprintf("%g", x); s)

"""Returns the normalised value or throws ArgumentError with the reason."""
function _valida(clave, v)
    tipo = Dict(p[1] => p[5] for p in PARAMS)[clave]
    v = replace(strip(v), " " => "")
    if startswith(tipo, "num") || startswith(tipo, "0<=")
        x = parse(Float64, v)
        tipo == "num>0" && x <= 0 && throw(ArgumentError("must be > 0"))
        tipo == "num>=0" && x < 0 && throw(ArgumentError("must be >= 0"))
        tipo == "0<=num<1" && !(0 <= x < 1) && throw(ArgumentError("between 0 and 1"))
        return v
    elseif tipo == "rango"
        p = split(v, ','); length(p) == 1 && (p = [p[1], p[1], "0.1"])
        v = join(p, ","); _lista_J(v); return v
    elseif tipo == "malla"
        tam, dx = split(replace(lowercase(v), "×" => "x"), ',')
        nx, ny, nz = parse.(Int, split(tam, 'x'))
        (min(nx, ny, nz) < 4 || parse(Float64, dx) <= 0) && throw(ArgumentError("each side >= 4 cells and dx > 0"))
        return "$(nx)x$(ny)x$(nz),$(_gfmt(parse(Float64, dx)))"
    elseif tipo == "tiempos"
        a, b = parse.(Float64, split(v, ','))
        (a <= 0 || b < a) && throw(ArgumentError("static > 0 and flowing >= static"))
        return "$(_gfmt(a)),$(_gfmt(b))"
    elseif tipo == "scheme"
        v in ("prod", "legacy") || throw(ArgumentError("prod or legacy")); return v
    elseif tipo == "z"
        v in ("through", "half", "buried") || throw(ArgumentError("through, half or buried")); return v
    end
    v
end

function _mostrar_valor(clave, v)
    if clave == "J"
        js = _lista_J(v); return "$(_gfmt(js[1])) … $(_gfmt(js[end]))" * (length(js) > 1 ? "  ($(length(js)) values)" : "")
    elseif clave == "malla"
        tam, dx = split(v, ','); nx, ny, nz = parse.(Int, split(tam, 'x')); h = parse(Float64, dx)
        return "$(nx)×$(ny)×$(nz), dx=$dx  ($(_gfmt(nx*h))×$(_gfmt(ny*h))×$(_gfmt(nz*h)) ξσ)"
    elseif clave == "t"
        a, b = split(v, ','); return "$a / $b"
    end
    v
end

"""Editable table. true = continue, false = back."""
function _tabla_parametros(par)
    while true
        lin = String[]
        for (i, (k, nombre, unid, que, _)) in enumerate(PARAMS)
            u = isempty(unid) ? "" : " [$unid]"; sp = " "^(length(string(length(PARAMS))) - length(string(i)) + 1)
            v = _col("1", _mostrar_valor(k, par[k]))     # as the original: the colour codes count in the 28
            push!(lin, "$(_K(string(i)))$sp$(rpad(nombre * u, 20)) $(_pad(v, 28 - (length(v) - _vis(v))))  $(_dim(que))")
        end
        append!(lin, ["", _dim("number = change that parameter  ·  Enter = continue  ·  b = previous step")])
        caja("parameters", lin; color="36")
        k = lowercase(_input("> "))
        k == "" && return true
        k in ("b", "q", "-") && return false
        n = tryparse(Int, k); (n === nothing || !(1 <= n <= length(PARAMS))) && continue
        clave, nombre, unid, que, tipo = PARAMS[n]
        ej = get(Dict("rango" => "from,to,step  (e.g. 0.50,0.80,0.02)", "malla" => "NxxNyxNz,dx  (e.g. 40x24x6,0.5)",
                      "tiempos" => "static,flowing  (e.g. 200,1200)"), tipo, "")
        println(_dim("  $nombre: $que" * (isempty(ej) ? "" : "  · format $ej")))
        v = _input("  $nombre [$(par[clave])]> "); isempty(v) && continue
        try
            par[clave] = _valida(clave, v)
        catch e
            println(_col("31", "  not valid: " * (e isa ArgumentError ? e.msg : sprint(showerror, e)))); sleep(1.2)
        end
    end
end

_paso(n, total, titulo) = (println(); println(_col("1;36", "── Launch · step $n/$total · $titulo ") * _dim("── b = previous step")))

# --- launch wizard: a sequence of steps; b goes back one step ---------------------------------
function _w_inicio(s)
    _paso(1, 6, "starting point")
    for (i, (nombre, desc, _)) in enumerate(PRESETS); println("  $(_K(string(i))) $(rpad(nombre, 24)) $(_dim(desc))"); end
    println(_dim("  It only fills the initial values: mesh (Nx×Ny×Nz, dx), κ, H, currents and times"))
    println(_dim("  are all editable in step 3."))
    k = _ask("starting point", get(s, "preset", "1")); k === nothing && return :back
    n = tryparse(Int, k); (n === nothing || !(1 <= n <= length(PRESETS))) && return :again
    (k != get(s, "preset", "") || !haskey(s, "par")) && (s["par"] = copy(PRESETS[n][3]))
    s["preset"] = k; :next
end

function _w_geometria(s)
    _paso(2, 6, "geometry")
    for (i, (n, nota)) in enumerate(CATALOG_VISIBLE); println("  $(_K(string(i))) $(rpad(n, 10)) $(_dim(nota))"); end
    println("  $(_K("e")) draw a new shape in the editor (opens the browser), then use it")
    println("  $(_K("f")) a saved shape from julia/shapes/ (yours, or a catalogue template)")
    println(_dim("  Any other catalogue name can be typed (g → p lists them)."))
    gsel = _ask("geometry", get(s, "gsel", "2")); gsel === nothing && return :back
    forma = ""
    if lowercase(gsel) == "e"
        isfile(EDITOR) || (println("editor not found: $EDITOR"); return :again)
        println("  editor: $(_editor_url())")
        println(_dim("  Draw the shape and press 'save' in the browser (it goes to julia/shapes/)."))
        _ask("Enter when saved", "") === nothing && return :again
        r = _elegir_forma(); r === nothing && return :again
        geom, forma = r
    elseif lowercase(gsel) == "f"
        r = _elegir_forma(); r === nothing && return :again
        geom, forma = r
    elseif (n = tryparse(Int, gsel); n !== nothing && 1 <= n <= length(CATALOG_VISIBLE))
        geom = CATALOG_VISIBLE[n][1]
    else
        geom = uppercase(gsel)
    end
    if isempty(forma) && !any(c -> geom == c[1] || geom == c[1] * "_REF", CATALOG)
        println("unknown geometry"); return :again
    end
    par = s["par"]
    if (!isempty(forma) || replace(geom, "_REF" => "") in WIDE) && startswith(par["malla"], "20x14x4")
        par["malla"] = "40x24x6,0.5"; println(_dim("  this shape needs the 40×24×6 box: mesh adjusted"))
    end
    rot = ""
    if replace(geom, "_REF" => "") in ("FUNNEL", "STAGGER", "HEX", "HEX_STAG")
        rot = _ask("rotation of the polygon in degrees (Enter = catalogue value)", get(s, "rot", ""))
        rot === nothing && return :again
    end
    merge!(s, Dict("gsel" => gsel, "geom" => geom, "forma" => forma, "rot" => rot)); :next
end

_w_parametros(s) = (_paso(3, 6, "parameters"); _tabla_parametros(s["par"]) ? :next : :back)

const PHI = [("off", "gauge Φ = 0 (default, fastest). |Ψ|, B and the Josephson voltage are gauge invariant,"),
             ("coulomb", "Coulomb projection: solves Poisson for Φ each step. Same |Ψ|, B and V as off; it also"),
             ("tf", "Thomas–Fermi screening length λ_TF: Φ is screened; this changes the physics")]
const PHI_2 = ["so for them nothing is lost", "gives Φ and the potential drop V_Φ (gauge-dependent)", ""]

function _w_phi(s)
    _paso(4, 6, "scalar potential Φ")
    for (i, ((n, d), d2)) in enumerate(zip(PHI, PHI_2))
        println("  $(_K(string(i - 1))) $(rpad(n, 8)) $(_dim(d))")
        isempty(d2) || println("  ", " "^12, _dim(d2))
    end
    previo = get(s["par"], "phi", "off")
    actual = string(findfirst(p -> p[1] == split(previo, ',')[1], PHI) - 1)
    k = _ask("Φ", actual); k === nothing && return :back
    n = tryparse(Int, k); (n === nothing || !(0 <= n < length(PHI))) && return :again
    modo = PHI[n+1][1]
    if modo == "tf"
        l = _ask("λ_TF in ξσ (Debye-like screening length)", occursin(',', previo) ? split(previo, ',')[2] : "2")
        l === nothing && return :again
        x = tryparse(Float64, l)
        (x === nothing || x <= 0) && (println(_col("31", "  λ_TF must be a number > 0")); return :again)
        modo = "tf,$l"
    end
    s["par"]["phi"] = modo; :next
end

function _w_termino(s)
    _paso(5, 6, "optional term (one at a time)")
    t = _elegir_termino(get(s, "term", nothing), get(s["par"], "phi", "off")); t === nothing && return :back
    if haskey(t, "qss") && !startswith(get(s["par"], "phi", "off"), "tf")
        s["par"]["phi"] = "tf,2"; println(_dim("  diffused Q lives in the Φ step: Φ set to Thomas–Fermi, λ_TF = 2"))
    end
    s["term"] = t; :next
end

function _w_resumen(s)
    _paso(6, 6, "summary")
    geom, forma, rot, par = s["geom"], s["forma"], s["rot"], s["par"]
    lin = ["geometry    $geom" * (isempty(forma) ? "" : "   ← $(basename(forma))") * (isempty(rot) ? "" : "   rot $(rot)°")]
    for (k, nombre, unid, _, _) in PARAMS
        push!(lin, rpad(nombre, 12) * _mostrar_valor(k, par[k]) * (!isempty(unid) && !(k in ("malla", "J")) ? "  $unid" : ""))
    end
    t = s["term"]
    append!(lin, ["Φ           $(_phi_texto(get(par, "phi", "off")))", "term        $(_term_texto(t))", "writes to   julia/runs/"])
    if get(par, "scheme", "") == "prod" && (get(par, "phi", "off") != "off" || haskey(t, "qss"))
        push!(lin, _dim("prod has no scalar potential; with Φ or Q this run uses legacy (1.0.0)"))
    elseif get(par, "scheme", "") == "prod" && (something(tryparse(Float64, get(par, "T", "0")), 0.0) > 0 || haskey(t, "drag") || haskey(t, "calor"))
        push!(lin, _dim("with heat, drag or T the faster scheme is used: prod at large κ, legacy otherwise"))
    end
    caja("about to launch", lin; color="32")
    k = _ask("launch? y/N", "N"); k === nothing && return :back
    lowercase(k) in ("y", "s", "si", "yes") || (println("cancelled"); return :cancel)
    _spawn(geom, par; rotate=rot, forma=forma, terminos=t)
    _volver(); :next
end

function _launch_wizard()
    pasos = (_w_inicio, _w_geometria, _w_parametros, _w_phi, _w_termino, _w_resumen)
    s = Dict{String,Any}(); i = 1
    while 1 <= i <= length(pasos)
        r = pasos[i](s)
        if r == :next
            i += 1
        elseif r == :back
            i -= 1
        elseif r == :cancel
            return
        end
    end
end

function _spawn(geom, par; rotate="", forma="", terminos=Dict{String,String}(), restart=false, rundir=nothing, name="", mMax="")
    mkpath(RUNS)
    tam, dx = split(par["malla"], ','); nx, ny, nz = split(tam, 'x'); tmin, tres = split(par["t"], ',')
    js = get(par, "J_lista", join(_gfmt.(_lista_J(par["J"])), ","))
    cmd = String[LANZAR, par["cpus"], Base.julia_cmd().exec[1], "--project=$JULIA", PRESET,
                 "--geom", geom, "--kappa", par["kappa"], "--H", par["H"], "--xi-ratio", par["xi"],
                 "--gamma12", par["g12"], "--J", js, "--Nx", nx, "--Ny", ny, "--Nz", nz, "--dx", dx,
                 "--tmin", tmin, "--tminres", tres, "--z", par["z"], "--cpus", par["cpus"],
                 "--scheme", get(par, "scheme", "legacy")]
    isempty(mMax) || append!(cmd, ["--mMax", mMax])
    something(tryparse(Float64, get(par, "T", "0")), 0.0) > 0 && append!(cmd, ["--T", par["T"]])
    phi = get(par, "phi", "off")
    if phi != "off"
        append!(cmd, ["--phi", split(phi, ',')[1]]); occursin(',', phi) && append!(cmd, ["--lambda-tf", split(phi, ',')[2]])
    end
    isempty(rotate) || append!(cmd, ["--rotate-deg", rotate])
    isempty(forma) || append!(cmd, ["--shape", forma])
    t = terminos
    haskey(t, "drag") && append!(cmd, ["--drag", "1", "--nu", get(t, "nu", "0.05")])
    haskey(t, "calor") && append!(cmd, ["--heat", t["calor"], "--joule", get(t, "joule", "0"), "--margin", get(t, "margen", "0")])
    haskey(t, "qss") && append!(cmd, ["--qss", "1"])
    rundir === nothing || append!(cmd, ["--dir", rundir])
    isempty(name) || append!(cmd, ["--name", name])
    restart && push!(cmd, "--restart")
    log = joinpath(rundir === nothing ? RUNS : rundir, "TUI_launch.log"); mkpath(dirname(log))
    # setsid: the run belongs to its own session, so closing the terminal of the menu does not stop it
    run(pipeline(_suelto(cmd); stdout=log, stderr=log, append=true); wait=false)
    println("launched. Closing the menu does not stop the run. log: $log")
    log
end

"""Wizard parameters rebuilt from a run_preset job.json."""
function _par_de_job(job)
    jl = get(job, "J_list", [0.10]); jl = jl isa AbstractVector ? Float64.(jl) : [Float64(jl)]
    js = sort([abs(x) for x in jl if abs(x) > 0]); isempty(js) && (js = [0.10])
    paso = length(js) > 1 ? minimum(diff(js)) : 0.1; paso == 0 && (paso = 0.1)
    s(k, d) = string(something(get(job, k, d), d))
    Dict("kappa" => s("kappa", 2), "H" => s("H", 0.5), "xi" => s("xi_ratio", 1.3), "g12" => s("gamma12", 0.001),
         "T" => (v = get(job, "T_red", 0); _vacio(v) ? "0" : string(v)),
         "J" => "$(_gfmt(js[1])),$(_gfmt(js[end])),$(_gfmt(paso))",
         "malla" => "$(s("Nx", 20))x$(s("Ny", 14))x$(s("Nz", 4)),$(s("dx", 0.5))",
         "t" => "$(s("t_min_GL", 20)),$(s("t_min_GL_res", 40))",
         "z" => s("z_mode", "through"), "cpus" => s("cpus", "8,9"), "scheme" => s("scheme", "legacy"),
         "J_lista" => join(_gfmt.(jl), ","),
         "phi" => get(job, "phi", "") == "tf" ? "tf,$(s("lambda_TF", 2))" : (_vacio(get(job, "qss", nothing)) ? s("phi", "off") : "off"))
end

"""Numbered list in a box; Enter or b goes back without choosing."""
function _pick(rows, prompt)
    titulo = let t = strip(split(prompt, '#')[1]); isempty(t) ? "choose" : t end
    isempty(rows) && (caja(titulo, [_dim("no runs for this")]; color="33"); _volver(); return nothing)
    lin = ["$(_K(string(i))) $(r["tag"])   $(_estado(r))   $(_ultimo(r))" for (i, r) in enumerate(rows)]
    push!(lin, _dim("number = choose  ·  Enter or b = back")); caja(titulo, lin; color="33")
    n = tryparse(Int, _input("$titulo # > "))
    (n === nothing || !(1 <= n <= length(rows))) ? nothing : rows[n]
end

function _stop_job(r)
    d = r["dir"]
    println("\nStop a run. Checkpoint = last CLOSED point.")
    println("  1  clean (STOP.txt): finishes the current point, saves and exits  ← preferred")
    println("  2  immediate (SIGTERM): loses the current point")
    println("  N  cancel")
    r["frozen"] && println("  Article run: only option 2 applies, and it should not be needed.")
    k = lowercase(_input("stop> "))
    if k == "1"
        r["frozen"] && (println("STOP.txt is not written into an article run."); return _volver())
        write(joinpath(d, "STOP.txt"), "stop requested by tdgl_tui\n")
        println("STOP.txt written in $d\nwhen the point closes, STATUS shows the last point and [u] resumes.")
    elseif k == "2"
        if !r["on"] || !all(isdigit, r["pid"])
            println("no live pid")
        else
            x = _ask("SIGTERM to pid $(r["pid"])? y/N", "N")
            (x !== nothing && lowercase(x) in ("y", "s", "si", "yes")) && (run(`kill -TERM $(r["pid"])`); println("SIGTERM sent"))
        end
    end
    _volver()
end

function _resume_job(r)
    job = isempty(r["job"]) ? _job_json(r["dir"]) : r["job"]
    if isempty(job) || !(get(job, "project", "") in ("julia", "juliatdgl_v3", "juliatdgl_v2")) || r["frozen"]
        println("Only runs launched from this menu (with their job.json) are resumed here.")
        println("Others: same script, restart_pos/neg=true, same out_dir."); return _volver()
    end
    r["on"] && (println("still running; not relaunching on top of it."); return _volver())
    ck = filter(f -> occursin(r"^state_J.*\.(jld2|mat)$", basename(f)) && basename(dirname(f)) == "restart",
                [joinpath(a, f) for (a, _, fs) in walkdir(r["dir"]; onerror=_ -> nothing) for f in fs])
    println("resume $(r["tag"])  checkpoints=$(length(ck))  last J=$(r["last"])")
    k = _ask("resume? y/N", "N")
    (k === nothing || !(lowercase(k) in ("y", "s", "si", "yes"))) && return
    rot = get(job, "rotate_deg", nothing)
    _spawn(get(job, "geom", "WEDGE"), _par_de_job(job); restart=true, rundir=r["dir"],
           rotate=rot === nothing || rot == "" ? "" : string(rot),
           forma=_nulo(get(job, "geom_file", nothing)) ? "" : joinpath(r["dir"], job["geom_file"]),
           terminos=_term_de_job(job), mMax=string(something(get(job, "mMax", ""), "")))
    _volver()
end

function _help()
    caja("help", [
        "$(_K("l")) launch: starting point → geometry → parameters → Φ → optional term → summary",
        "$(_K("s")) stop a live run: STOP.txt (closes the current point) or SIGTERM (loses it)",
        "$(_K("u")) resume from the last closed point (checkpoint)",
        "$(_K("d")) move a finished run to trash_/tui_jobs/ (nothing is deleted)",
        "$(_K("a")) live only / all      $(_K("r")) refresh      $(_K("o")) open folder",
        "$(_K("f")) 2D figures (V(J), |Ψ|² maps; matplotlib) in RUN/figures/; 3D: vtk_from_checkpoint → ParaView",
        "$(_K("g")) geometries: catalogue and shape editor in the browser",
        "$(_K("v")) results: Ic brackets with the verdict rule (crossings per line <1 silent, ≥10",
        "  dissipating), η with its range, undecided rungs, σ_V where both polarities flow",
        _dim("While this menu is open a desktop notification is sent when a run finishes;"),
        _dim("without the menu: julia julia/scripts/tdgl_tui.jl --watch   (or --results)"),
        "$(_dim("number")) details of a run: current history, V(J) and actions", "",
        _dim("In every menu: Enter = default value, b = previous step, Ctrl+C = back to this list.")]; color="33")
    _volver()
end

"""Ctrl+C inside a menu returns to the list; an error is shown and the menu stays up."""
function _seguro(f, args...)
    try
        f(args...)
    catch e
        if e isa InterruptException
            println("\n", _dim("cancelled: back to the list")); sleep(0.6)
        elseif e isa EOFError
            rethrow()
        else
            caja("error (the menu keeps running; Julia was not touched)", [sprint(showerror, e)]; color="31")
            try _volver() catch end
        end
    end
end

function _figuras(r)
    script = joinpath(JULIA, "scripts", "plot_summary.py")
    if Sys.which("python3") === nothing
        println("2D figures need Python 3 with matplotlib (optional); 3D: vtk_from_checkpoint.jl → ParaView")
    else
        println("2D figures with matplotlib (V(J) and |Ψ|² maps) in ", joinpath(r["dir"], "figures"))
        try run(`python3 $script $(r["dir"])`) catch e; println(sprint(showerror, e)) end
    end
    _volver()
end

"""Watched folders: add one (typed, pasted or dragged from a file manager) or remove one."""
function _carpetas_menu()
    while true
        _clear()
        guardadas = [rstrip(abspath(expanduser(x)), '/') for x in _carpetas_guardadas()]
        lin = [_dim("project (always): ") * join((relpath(d, BASE) for d in ROOTS_PROYECTO), ", ")]
        for d in EXTRA; d in guardadas || push!(lin, _dim("fixed: ") * d); end
        for (i, d) in enumerate(guardadas); push!(lin, "$(_K(string(i))) $d" * (isdir(d) ? "" : _dim("  (missing)"))); end
        append!(lin, ["", _dim("A Julia script that loads TDGL shows as live from any folder, listed or not."),
                      _dim("Listing a folder adds its finished runs to the table and to the results.")])
        caja("watched folders", lin; color="36")
        println("$(_K("+")) add: paste or drag the folder here (Enter alone = the current one)   $(_K("-")) number: remove   Enter: back")
        r = _input("> ")
        isempty(r) && return
        if startswith(r, "-")
            n = tryparse(Int, strip(r[2:end]))
            (n !== nothing && 1 <= n <= length(guardadas)) && (println(_cambiar_carpeta(guardadas[n], true)); sleep(0.8))
            continue
        end
        startswith(r, "+") && (r = strip(r[2:end]); isempty(r) && (r = _input("folder > ")))
        println(_cambiar_carpeta(_ruta_pegada(r), false)); sleep(0.8)
    end
end

"""One key of the main screen. Runs inside _seguro."""
function _accion(key, rows, show, only_on)
    k = lowercase(key)
    if k == "g"; _geom_menu()
    elseif k == "l"; _launch_wizard()
    elseif k in ("h", "?"); _help()
    elseif k == "s"; r = _pick(filter(r -> r["on"], rows), "stop # > "); r === nothing || _stop_job(r)
    elseif k == "u"; r = _pick(filter(r -> !r["on"] && isfile(joinpath(r["dir"], "job.json")), rows), "resume # > "); r === nothing || _resume_job(r)
    elseif k == "d"; r = _pick(filter(r -> !r["on"] && !r["frozen"], only_on ? collect_runs() : rows), "to trash_ # > "); r === nothing || _delete_job(r)
    elseif k == "o"; r = _pick(isempty(show) ? rows : show, "open # > "); r === nothing || _open_dir(r["dir"])
    elseif k == "f"; r = _pick(isempty(show) ? collect_runs() : show, "figures # > "); r === nothing || _figuras(r)
    elseif k == "v"; _resultados(isempty(show) ? collect_runs() : show)
    elseif k == "c"; _carpetas_menu()
    else
        n = tryparse(Int, k); (n !== nothing && 1 <= n <= length(show)) && detail(show[n])
    end
end

function interactive()
    ccall(:jl_generating_output, Cint, ()) == 1 || Base.exit_on_sigint(false)
    _tty() && print("\e]0;TDGL\a")             # the terminal tab says TDGL, not Julia
    only_on = true
    _VIGIA[] && @async _vigia()
    while true
        _clear()
        isempty(TERMINADAS) || caja("finished while the menu was open", TERMINADAS[max(1, end - 4):end]; color="32")
        rows = try
            r = collect_runs(); table(r; only_on=only_on); r
        catch e
            caja("error reading the runs", [sprint(showerror, e)]; color="31"); Dict{String,Any}[]
        end
        show = only_on ? filter(r -> r["on"], rows) : rows
        caja("keys", [
            "$(_K("l")) launch   $(_K("s")) stop   $(_K("u")) resume   $(_K("d")) to trash_   " * _dim("number = details"),
            "$(_K("a")) live/all   $(_K("r")) refresh   $(_K("o")) folder   $(_K("f")) figures   $(_K("v")) results",
            "$(_K("c")) watched folders   $(_K("g")) geometries and shape editor   $(_K("h")) help   $(_K("q")) quit (runs continue)",
            _dim("In every menu: Enter = default, b = previous step, Ctrl+C = back here.")]; color="33")
        key = try
            _input("> ")
        catch e
            (e isa EOFError || e isa InterruptException) && (println(); return)
            rethrow()
        end
        key in ("q", "Q") && return
        if key in ("a", "A")
            only_on = !only_on
        elseif !(key in ("r", "R", ""))
            try
                _seguro(_accion, key, rows, show, only_on)
            catch e
                e isa EOFError && return
                rethrow()
            end
        end
    end
end

const AYUDA = """
usage: tdgl-tui [--help] [--once [--all]] [--watch] [--results] [--add-dir [PATH]] [--rm-dir PATH]
                [--dirs] [--notify] [--editor] [rundir]

Terminal menu for TDGL runs: launch, watch, stop, resume.

  rundir            show the details of one run and exit
  --once            print the table of live runs once and exit
  --all             with --once: every run, not only the live ones
  --watch           no menu: notify on the desktop when a run finishes
  --results         print the quick results of every run and exit
  --add-dir [PATH]  watch PATH (default: the current folder) from now on; saved in $CARPETAS
  --rm-dir PATH     stop watching PATH
  --dirs            list the watched folders
  --notify          with --add-dir/--rm-dir: also a desktop notification
  --editor          start the shape editor server instead (--port N, --no-browser, --idle MIN)
"""

function (@main)(args)
    ccall(:jl_generating_output, Cint, ()) == 1 || (_ENTRADA[] = stdin)
    _recargar_carpetas()
    flag(f) = f in args
    if flag("-h") || flag("--help")
        print(AYUDA); return
    end
    if flag("--editor")
        return editor_main(filter(!=("--editor"), args))
    end
    i = findfirst(==("--add-dir"), args); j = findfirst(==("--rm-dir"), args)
    if i !== nothing || j !== nothing
        quitar = j !== nothing; k = quitar ? j : i
        ruta = k < length(args) && !startswith(args[k+1], "--") ? args[k+1] : (quitar ? error("--rm-dir needs a PATH") : ".")
        msg = _cambiar_carpeta(_ruta_pegada(ruta), quitar); println(msg)
        flag("--notify") && _notificar("TDGL", msg)
        return
    end
    if flag("--dirs")
        foreach(println, ROOTS); println("(list: $CARPETAS; a Julia script that loads TDGL shows as live from any folder)"); return
    end
    pos = filter(a -> !startswith(a, "--"), args)
    if !isempty(pos)
        p = isabspath(pos[1]) ? pos[1] : joinpath(BASE, pos[1])
        rows = filter(r -> abspath(r["dir"]) == rstrip(abspath(p), '/') || endswith(r["dir"], rstrip(pos[1], '/')), collect_runs())
        isempty(rows) ? println("no run found at $p") : (try detail(rows[1]) catch e; e isa EOFError || rethrow() end)
        return
    end
    flag("--once") && return table(collect_runs(); only_on=!flag("--all"))
    if flag("--results")
        vistos = Set{String}()
        for r in collect_runs()
            res = [x for x in quick_result(r["dir"]) if !(x in vistos)]; union!(vistos, res)
            isempty(res) && continue
            println(r["tag"], r["on"] ? "  (live)" : ""); foreach(x -> println("  ", x), res)
        end
        return
    end
    if flag("--watch")
        println("watching the runs; a desktop notification is sent when one finishes (Ctrl+C to stop)")
        try _vigia() catch e; e isa InterruptException || rethrow() end
        return
    end
    interactive()
end

# --- shape editor server: 127.0.0.1 only, standard library only ---------------------------------
#   GET  /                  the editor
#   GET  /shapes            list of julia/shapes/*.toml (JSON)
#   GET  /shape?name=x      shape x as JSON (TOML parsed) plus its text
#   POST /save?name=x       writes shapes/x.toml; an existing one is MOVED to trash_/shapes/ (never deleted)
#   GET  /ping, /quit       keep-alive from the page, and its "close server" button
const EDITOR_HTML = joinpath(JULIA, "editor", "shape_editor.html")
const _NOMBRE_OK = r"^[A-Za-z0-9_\-]{1,64}$"
const _ULTIMA = Ref(time())

_json(code, x) = (code, "application/json; charset=utf-8", Vector{UInt8}(json_string(x)))
_pct_dec(t) = replace(replace(t, "+" => " "), r"%([0-9A-Fa-f]{2})" => m -> string(Char(parse(UInt8, m[2:3]; base=16))))
function _query(ruta)
    q = Dict{String,String}()
    i = findfirst('?', ruta); i === nothing && return (ruta, q)
    for par in split(ruta[i+1:end], '&'; keepempty=false)
        k, v = occursin('=', par) ? split(par, '='; limit=2) : (par, "")
        q[_pct_dec(k)] = _pct_dec(v)
    end
    (ruta[1:i-1], q)
end

"""The answer of the editor server to one request: (status, content type, body)."""
function _responder(metodo, ruta, cuerpo::Vector{UInt8})
    _ULTIMA[] = time()
    camino, q = _query(ruta)
    nombre = get(q, "name", "")
    if metodo == "GET"
        camino in ("/", "/shape_editor.html") && return (200, "text/html; charset=utf-8", read(EDITOR_HTML))
        camino == "/ping" && return _json(200, Dict("ok" => true))
        camino == "/quit" && return _json(200, Dict("ok" => true, "quit" => true))
        if camino == "/shapes"
            mkpath(FORMAS)
            return _json(200, sort([splitext(f)[1] for f in readdir(FORMAS) if endswith(f, ".toml")]))
        elseif camino == "/shape"
            occursin(_NOMBRE_OK, nombre) || return _json(400, Dict("error" => "name: letters, digits, _ and - (max. 64)"))
            p = joinpath(FORMAS, nombre * ".toml")
            isfile(p) || return _json(404, Dict("error" => "$(basename(p)) does not exist"))
            texto = read(p, String)
            datos = try TOML.parse(texto) catch e; return _json(422, Dict("error" => "invalid TOML: " * sprint(showerror, e))) end
            return _json(200, Dict("toml" => datos, "text" => texto))
        end
    elseif metodo == "POST" && camino == "/save"
        occursin(_NOMBRE_OK, nombre) || return _json(400, Dict("error" => "name: letters, digits, _ and - (max. 64)"))
        (isempty(cuerpo) || length(cuerpo) > 2_000_000) && return _json(400, Dict("error" => "empty or too large body"))
        texto = String(copy(cuerpo))
        try TOML.parse(texto) catch e; return _json(422, Dict("error" => "invalid TOML: " * sprint(showerror, e))) end
        mkpath(FORMAS); dest = joinpath(FORMAS, nombre * ".toml"); movido = nothing
        if isfile(dest)
            papelera = joinpath(BASE, "trash_", "shapes"); mkpath(papelera)
            movido = joinpath(papelera, nombre * "_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ".toml"); mv(dest, movido)
        end
        write(dest, texto)
        return _json(200, Dict("ok" => true, "path" => relpath(dest, BASE), "previous" => movido === nothing ? nothing : relpath(movido, BASE)))
    end
    _json(404, Dict("error" => "not found"))
end

const _RAZON = Dict(200 => "OK", 400 => "Bad Request", 404 => "Not Found", 422 => "Unprocessable Entity", 500 => "Internal Server Error")

function _atender(sock, srv)
    try
        linea = readline(sock); isempty(linea) && return
        partes = split(linea, ' '); length(partes) >= 2 || return
        metodo, ruta = partes[1], partes[2]
        cab = Dict{String,String}()
        while true
            l = strip(readline(sock)); isempty(l) && break
            occursin(':', l) || continue
            k, v = split(l, ':'; limit=2); cab[lowercase(strip(k))] = strip(v)
        end
        n = something(tryparse(Int, get(cab, "content-length", "0")), 0)
        cuerpo = metodo == "POST" && 0 < n <= 2_000_000 ? read(sock, n) : UInt8[]
        code, tipo, datos = try
            _responder(metodo, ruta, cuerpo)
        catch e
            _json(500, Dict("error" => sprint(showerror, e)))
        end
        write(sock, "HTTP/1.1 $code $(get(_RAZON, code, ""))\r\nContent-Type: $tipo\r\nContent-Length: $(length(datos))\r\n" *
                    "Cache-Control: no-store\r\nConnection: close\r\n\r\n", datos)
        if startswith(ruta, "/quit")
            println("closed from the page: stopping the editor server"); close(srv)
        end
    catch
    finally
        close(sock)
    end
end

"""The shape editor server. `editor_main(["--port", "8765", "--no-browser", "--idle", "30"])`."""
function editor_main(args)
    opt(k, d) = (i = findfirst(==(k), args); i === nothing || i == length(args) ? d : args[i+1])
    puerto = parse(Int, opt("--port", "8765")); idle = parse(Float64, opt("--idle", "30"))
    srv = listen(ip"127.0.0.1", puerto)
    url = "http://127.0.0.1:$puerto/"
    println("shape editor: $url   (Ctrl+C to close)\nsaves to: $FORMAS")
    "--no-browser" in args || _abrir(url)
    if idle > 0
        @async begin
            while isopen(srv) && time() - _ULTIMA[] < 60idle
                sleep(min(10.0, 15idle))
            end
            isopen(srv) && (println("idle: stopping the editor server"); close(srv))
        end
    end
    while isopen(srv)
        sock = try accept(srv) catch; break end
        @async _atender(sock, srv)
    end
end

# --- precompile workload: a fake run, listed, summarised and shown, so the code is cached ----------
if ccall(:jl_generating_output, Cint, ()) == 1
    let d = mktempdir(), r = joinpath(d, "runs", "demo")
        mkpath(r)
        write(joinpath(r, "STATUS.txt"), "done J=+0.1\n")
        write(joinpath(r, "run_info.txt"), "dt=0.02\nlMax=40\nt_cycle=0.8\n")
        write(joinpath(r, "job.json"), "{\"geom\": \"WEDGE\", \"kappa\": 2.0, \"J_list\": [0.1, 0.2], \"calor\": null, " *
              "\"n_expected\": 4, \"Nx\": 40, \"Ny\": 24, \"Nz\": 6, \"dx\": 0.5, \"project\": \"julia\"}")
        write(joinpath(r, "j_history.csv"), "time,pol,J,V,conv,it,Nv_s,Nv_p,psi_p2\n" *
              "2026-01-01 00:00:00,J+,0.1,1e-9,F,500,0,0,0.3\n2026-01-01 00:01:00,J+,0.2,0.5,V,2000,1,1,0.3\n" *
              "2026-01-01 00:02:00,J-,-0.1,-1e-9,F,500,0,0,0.3\n2026-01-01 00:03:00,J-,-0.2,-0.4,V,2000,1,1,0.3\n")
        empty!(ROOTS); push!(ROOTS, joinpath(d, "runs"))
        redirect_stdout(devnull) do
            rows = collect_runs()
            table(rows; only_on=false); table(rows; only_on=true)
            quick_result(r); _envolver(quick_result(r), 80)
            _ascii_curve(rows[1]["hist"], "+"); _par_de_job(rows[1]["job"]); _term_texto(_term_de_job(rows[1]["job"]))
            for (k, v) in (("kappa", "2"), ("J", "0.1,0.3,0.1"), ("malla", "40x24x6,0.5"), ("t", "20,200"), ("z", "through"))
                _mostrar_valor(k, _valida(k, v))
            end
            json_string(Dict("a" => [1, 2], "b" => nothing)); _corta_medio("carpeta/run_largo_H0p09_neg", 20)
            for a in (["--help"], ["--dirs"], ["--once"], ["--once", "--all"], ["--results"])
                main(a)
            end
            # a session: all runs, details of one, results, help, the launch wizard up to "launch? N", quit
            _VIGIA[] = false
            _ENTRADA[] = IOBuffer("a\n1\n\nv\n\nh\n\nl\n1\n2\n\n0\n0\nN\nq\n")
            empty!(ROOTS); push!(ROOTS, joinpath(d, "runs")); interactive()
            _ENTRADA[] = devnull; _VIGIA[] = true
            for (m, r) in (("GET", "/ping"), ("GET", "/shapes"), ("GET", "/shape?name=cat_WEDGE"), ("GET", "/shape?name=b%20ad"),
                           ("POST", "/save?name=b%20ad"), ("GET", "/nada"))
                _responder(m, r, UInt8[])
            end
        end
        empty!(ROOTS); empty!(ROOTS_PROYECTO); empty!(EXTRA)
    end
end

end # module
