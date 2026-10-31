# The down branch of a run: from the state its ladder ended in, the currents walked back down (does not touch the run):
#   OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/hysteresis.jl RUNDIR + | - [--step S] [--to JMIN] [--resume]
# It starts from the checkpoint of the polarity (RUNDIR/restart/state_J±.jld2: the state of the last closed rung, with the
# open boundary and the temperature field), which must have flowed.  The currents: those of the up ladder below the last
# one, in decreasing order, down to --to (default 0); with --step S, from the last current down in steps of S.  The rungs go
# into RUNDIR/down_<polarity>[_<S>]/, a run folder itself (run.toml, the shape file and the scheme defaults copied), so
# that scripts/figures.jl and run_report.py read it.  At the end: the retrapping bracket (Ir in (lo, hi]: the highest
# silent rung going down, the lowest flowing one above it), the onset bracket of the up branch, and the width of the
# hysteresis Ic − Ir (at the bracket centres, with its interval), and the voltage of both branches at the currents they
# share.  STOP.txt in the subfolder stops it after the current rung; the same command with --resume continues it from its
# checkpoint, to the bit.  Run the two polarities as two processes.
using TDGL, LinearAlgebra, Printf
BLAS.set_num_threads(1)
if length(ARGS) < 2 || ARGS[1] in ("-h", "--help")
    println("usage: julia --project=tdgl2 tdgl2/scripts/hysteresis.jl RUNDIR + | - [--step S] [--to JMIN] [--resume]"); exit(0)
end
dir = rstrip(ARGS[1], '/'); pol = ARGS[2] == "-" ? -1 : 1; lab = pol > 0 ? "J+" : "J-"
opt(k) = (i = findfirst(==(k), ARGS); i === nothing ? nothing : parse(Float64, ARGS[i + 1]))
isfile(joinpath(dir, "run.toml")) || error("$dir has no run.toml: not the folder of a release-2 run")
ck = joinpath(dir, "restart", "state_$lab.jld2")
isfile(ck) || error("$dir has no checkpoint of $lab ($ck)")

# the up ladder: (|J|, verdict) of its closed rungs, in the order walked
function rungs(d, label)
    p = joinpath(d, "output", "rungs_$label.csv"); isfile(p) || error("no $p")
    L = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(p))
    h = split(L[1], ","); iJ = findfirst(==("J"), h); ic = findfirst(==("conv"), h); iV = findfirst(==("V"), h)
    [(abs(parse(Float64, r[iJ])), r[ic][1], parse(Float64, r[iV])) for r in split.(L[2:end], ",") if !isempty(r[ic])]
end
flows(c) = c in ('V', 'N', 'D', 'T')
up = rungs(dir, lab)
isempty(up) && error("$lab: no closed rung in $dir")
Jtop, ctop, _ = up[end]
flows(ctop) || error("$lab: the ladder ended silent at $(Jtop) ('$ctop'): there is no flowing state to come down from")
S = opt("--step"); Jmin = something(opt("--to"), 0.0)
Jl = S === nothing ? sort(unique([r[1] for r in up if Jmin - 1e-12 <= r[1] < Jtop - 1e-12]); rev=true) :
                     round.(collect(Jtop-S:-S:Jmin-S/2); digits=10)   # 0.6, not the 0.6000000000000001 of the range
isempty(Jl) && error("$lab: no current between $Jmin and $Jtop to walk down")

c = load_config(joinpath(dir, "run.toml"))
od = joinpath(dir, S === nothing ? "down_$lab" : @sprintf("down_%s_%g", lab, S)); mkpath(od)
resume = "--resume" in ARGS && isfile(joinpath(od, "restart", "state_$lab.jld2"))
!resume && isfile(joinpath(od, "output", "rungs_$lab.csv")) && error("$od already holds a down branch of $lab (--resume continues it)")
cp(joinpath(dir, "run.toml"), joinpath(od, "run.toml"); force=true)
isfile(joinpath(dir, "scheme_defaults.toml")) && cp(joinpath(dir, "scheme_defaults.toml"), joinpath(od, "scheme_defaults.toml"); force=true)
TDGL._keep_shape(c, od)
st = TDGL.Stepper(c.model, c.grid, c.masks; c.scheme...)
pr = c.protocol
# every rung is walked (no stop after 'V', no delay confirmation), and the first one closes with the tolerance of the
# others: it is a rung of the branch, not the J = 0 seed
pr = Protocol(; (k => getfield(pr, k) for k in fieldnames(Protocol))..., keep_rung_states=true, stop_after_V=0,
             t_confirm_GL=0.0, eps_seed=TDGL.eps_cycle(pr, st))
@printf("%s %s: down from %.4f ('%s'): %s%s\n", basename(abspath(dir)), lab, Jtop, ctop, string(round.(Jl; digits=5)),
        resume ? " (resumed)" : "")
L = if resume
    resume_ladder(st, pol .* Jl; H=c.H, label=lab, pr=pr, out_dir=od)
else
    s, w = load_state(ck, c.grid)
    TDGL.restore_open!(st, w)
    st.heat === nothing || copyto!(st.heat.θ, w["Temp"])
    ladder(st, s, pol .* Jl; H=c.H, label=lab, pr=pr, out_dir=od)
end
TDGL.ladder_done(joinpath(od, "restart", "state_$lab.jld2")) || (tui_status(od, "$lab down branch stopped (STOP.txt): --resume continues it"); exit(0))
tui_status(od, "$lab down branch done")
for (a, v, x, n) in zip(L.J, L.V, L.conv, eachrow(L.Nv))
    @printf("%s %s down J=%+.5f V=%+.4e %s Nv=%s\n", basename(abspath(dir)), lab, a, v, x, join(n, "/"))
end

# retrapping: walking down, the first silent rung and the flowing one before it
fmt(b) = b === nothing ? "not reached" : @sprintf("(%s, %.5g]", b[1] === nothing ? "?" : @sprintf("%.5g", b[1]), b[2])
dn = [(abs(a), x) for (a, x) in zip(L.J, L.conv)]
iF = findfirst(r -> r[2] == 'F', dn)
br = iF === nothing ? nothing : (dn[iF][1], iF == 1 ? Jtop : dn[iF-1][1])
iV = findfirst(r -> flows(r[2]), up)
kF = iV === nothing ? nothing : findlast(r -> r[2] == 'F' && r[1] < up[iV][1], up)
bu = iV === nothing ? nothing : (kF === nothing ? nothing : up[kF][1], up[iV][1])
println("$lab onset going up: Ic in ", fmt(bu))
println("$lab retrapping going down: Ir in ", fmt(br), iF === nothing ? @sprintf(" (still flowing at %.5g)", dn[end][1]) : "")
# the two branches at the currents they share: a lower voltage going up than going down at the same current is the
# bistability itself, which a bracket as coarse as the ladder may not show
for (a, x, v) in zip(abs.(L.J), L.conv, L.V)
    k = findfirst(r -> abs(r[1] - a) < 1e-9, up); k === nothing && continue
    @printf("%s |J|=%.5f  up: %s V=%+.4e   down: %s V=%+.4e\n", lab, a, up[k][2], up[k][3], x, v)
end
if br !== nothing && bu !== nothing && bu[1] !== nothing
    z(x) = abs(x) < 1e-12 ? 0.0 : x                  # the same bracket: 0, not a rounding residue
    @printf("%s hysteresis Ic − Ir = %.4g [%.4g, %.4g]\n", lab, z((bu[1] + bu[2]) / 2 - (br[1] + br[2]) / 2),
            z(bu[1] - br[2]), z(bu[2] - br[1]))
end
