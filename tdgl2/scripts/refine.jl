# Refine the onset of a run, or carry a ladder that never flowed further up, from the states it kept (does not touch the run):
#   OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/refine.jl RUNDIR + | - [--step S] [--up S --to JMAX] [--confirm T]
# It reads the rungs of RUNDIR and of every refinement already made (RUNDIR/refine_*/), and the states they kept
# (keep_rung_states = true in [protocol]).  With a flowing rung: the bracket (lo, hi] of |J| (the highest silent rung with a
# kept state below the lowest flowing one), and a ladder from the state at lo in steps of S (default: a fifth of the bracket,
# rounded to 1, 2 or 5 × 10^k) up to hi.  With no flowing rung: from the highest silent rung, steps of --up (default: the last
# step of the ladder) up to --to (default: twice that J).  The new rungs go into RUNDIR/refine_<polarity>_<S>/, a run folder
# itself (run.toml and the shape file copied, its states kept), so that scripts/figures.jl and run_report.py read it and the
# next refine.jl call on RUNDIR finds it.  With --confirm T: the bracket confirmed by the delay of the first slip, as the
# ladders of the papers (t_confirm_GL in [protocol]): from the state of the second silent rung below the flowing end, the rungs up to its
# flowing end again, the silent ones below the first flowing one followed T t_GL in all; one that crosses becomes 'D' and the
# next one down is followed (RUNDIR/confirm_<polarity>_<T>/).  At the end it prints the bracket from all the rungs, and η when
# the other polarity has one too.  Run the two polarities as two processes.
using TDGL, LinearAlgebra, Printf
BLAS.set_num_threads(1)
if length(ARGS) < 2 || ARGS[1] in ("-h", "--help")
    println("usage: julia --project=tdgl2 tdgl2/scripts/refine.jl RUNDIR + | - [--step S] [--up S --to JMAX] [--confirm T]"); exit(0)
end
dir = rstrip(ARGS[1], '/'); pol = ARGS[2] == "-" ? -1 : 1; lab = pol > 0 ? "J+" : "J-"
opt(k) = (i = findfirst(==(k), ARGS); i === nothing ? nothing : parse(Float64, ARGS[i + 1]))
isfile(joinpath(dir, "run.toml")) || error("$dir has no run.toml: not the folder of a release-2 run")

# every rung of this polarity in the run and its refinements: (|J|, verdict, kept state or "")
function rungs_of(label)
    out = Tuple{Float64,Char,String}[]
    folders = vcat(dir, [joinpath(dir, f) for f in readdir(dir) if (startswith(f, "refine_") || startswith(f, "confirm_")) &&
                                                                   isdir(joinpath(dir, f))])
    for d in folders
        p = joinpath(d, "output", "rungs_$label.csv"); isfile(p) || continue
        L = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(p))
        h = split(L[1], ","); iJ = findfirst(==("J"), h); ic = findfirst(==("conv"), h)
        # the verdicts the delay confirmation gave ('D': it crossed when followed), which the rows of the ladder do not carry
        pc = joinpath(d, "output", "confirm_$label.csv"); cf = Dict{Float64,Char}()
        if isfile(pc)
            for r in split.(filter(l -> !isempty(strip(l)), readlines(pc))[2:end], ",")
                cf[abs(parse(Float64, r[2]))] = r[end][1]
            end
        end
        for r in split.(L[2:end], ",")
            isempty(r[ic]) && continue
            J = parse(Float64, r[iJ])
            sp = joinpath(d, "restart", "rungs", @sprintf("state_%s_J%+.4f.jld2", label, J))
            push!(out, (abs(J), get(cf, abs(J), r[ic][1]), isfile(sp) ? sp : ""))
        end
    end
    sort!(out; by=first)
end
flows(c) = c in ('V', 'N', 'D')
function bracket(rs)
    iV = findfirst(r -> flows(r[2]), rs); iV === nothing && return nothing
    iF = findlast(r -> r[2] == 'F' && r[1] < rs[iV][1], rs)
    (iF === nothing ? nothing : rs[iF], rs[iV])
end
nice(x) = (e = floor(log10(x)); m = x / 10^e; (m >= 5 ? 5 : m >= 2 ? 2 : 1) * 10^e)

rs = rungs_of(lab)
isempty(rs) && error("no closed rung of $lab in $dir")
c = load_config(joinpath(dir, "run.toml"))
b = bracket(rs); conf = opt("--confirm")
if conf !== nothing
    b === nothing && error("$lab: --confirm needs a flowing rung")
    lo, hi = b
    lo === nothing && error("$lab: the first rung already flows, nothing to confirm")
    below = filter(r -> r[2] == 'F' && !isempty(r[3]) && r[1] < lo[1], rs)
    isempty(below) && error("$lab: --confirm needs a silent rung with a kept state below $(lo[1]), to start from")
    start = below[max(1, end - 1)]                  # two silent rungs below the flowing end, if there are: either can become 'D'
    Jl = sort(unique([r[1] for r in rs if start[1] < r[1] <= hi[1]]))
    S = conf
    what = @sprintf("confirm (%.4f, %.4f] by the delay, %g t_GL, from %.4f", lo[1], hi[1], conf, start[1])
elseif b !== nothing
    lo, hi = b
    lo === nothing && error("$lab: the first rung already flows, no silent rung below it: extend the ladder down first")
    isempty(lo[3]) && error("$lab: the silent rung $(lo[1]) kept no state (keep_rung_states = true in [protocol])")
    S = something(opt("--step"), nice((hi[1] - lo[1]) / 5))
    S < hi[1] - lo[1] || error("$lab: the bracket ($(lo[1]), $(hi[1])] is already as fine as --step $S")
    Jl = collect(lo[1]+S:S:hi[1]-S/2); start = lo
    what = @sprintf("refine (%.4f, %.4f] in steps of %g", lo[1], hi[1], S)
else
    iF = findlast(r -> r[2] == 'F' && !isempty(r[3]), rs)
    iF === nothing && error("$lab: no silent rung with a kept state to continue from")
    start = rs[iF]
    S = something(opt("--up"), length(rs) > 1 ? rs[end][1] - rs[end-1][1] : 0.1)
    Jmax = something(opt("--to"), 2 * start[1])
    Jl = collect(start[1]+S:S:Jmax+S/2)
    what = @sprintf("no flowing rung: up from %.4f in steps of %g to %.4f", start[1], S, Jmax)
end
od = joinpath(dir, @sprintf("%s_%s_%g", conf === nothing ? "refine" : "confirm", lab, S)); mkpath(od)
isfile(joinpath(od, "output", "rungs_$lab.csv")) && error("$od already holds a ladder of $lab")
cp(joinpath(dir, "run.toml"), joinpath(od, "run.toml"); force=true)
TDGL._keep_shape(c, od)
pr = c.protocol
pr = Protocol(; (k => getfield(pr, k) for k in fieldnames(Protocol))..., keep_rung_states=true,
             t_confirm_GL=conf === nothing ? pr.t_confirm_GL : conf)
st = TDGL.Stepper(c.model, c.grid, c.masks; c.scheme...)
s, _ = load_state(start[3], c.grid)
@printf("%s %s: %s: %s\n", basename(abspath(dir)), lab, what, string(round.(Jl; digits=5)))
L = ladder(st, s, pol .* Jl; H=c.H, label=lab, pr=pr, out_dir=od)
for (a, v, w) in zip(L.J, L.V, L.conv); @printf("%s %s J=%+.5f V=%+.4e %s\n", basename(abspath(dir)), lab, a, v, w); end

# the bracket from all the rungs, and η if the other polarity has one
fmt(b) = b === nothing ? "not reached" : @sprintf("(%s, %.5g]", b[1] === nothing ? "?" : @sprintf("%.5g", b[1][1]), b[2][1])
bp = bracket(rungs_of(lab)); other = lab == "J+" ? "J-" : "J+"; bo = bracket(rungs_of(other))
println("$lab bracket from every rung: ", fmt(bp))
if bp !== nothing && bo !== nothing && bp[1] !== nothing && bo[1] !== nothing
    (P, N) = lab == "J+" ? (bp, bo) : (bo, bp)
    cen(b) = (b[1][1] + b[2][1]) / 2; e(a, b) = 100 * (a - b) / (a + b)
    @printf("η = %.2f %% [%.2f, %.2f] (J+ %s, J− %s)\n", e(cen(P), cen(N)), e(P[1][1], N[2][1]), e(P[2][1], N[1][1]), fmt(P), fmt(N))
end
