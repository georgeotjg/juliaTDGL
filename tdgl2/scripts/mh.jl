# The magnetisation loop M(H) of a run file at a fixed current (default 0):
#   OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/mh.jl RUN.toml --Hmax HMAX --dH DH [--J J] [--dir OUT] [--no-down] [--resume]
# One ladder at the current J whose points are the fields 0, DH, ..., HMAX and back down to 0 (without --no-down), each
# from the state of the one before, from the start state of the file ([run] start; Meissner at H = 0 by default).  Each
# point closes as a rung of the ladder does (a fixed point, or a flow if J drives one), so the field a vortex enters at
# going up and the one it leaves at going down are read from the same rule.  M = ⟨B_z⟩ − H on the sample block [Hc2 of
# band 1], Nv the vortex count of each band on its plane.  Writes OUT/ (default: out_dir of [run] + "_MH_H<HMAX>_d<DH>[_J<J>]") as a run folder
# (run.toml, the shape file, output/rungs_MH.csv, the state of every point in restart/rungs/), output/mh.csv (H, branch,
# M, Nv and ⟨|Ψ|²⟩ of each band, verdict) and, with python3 + matplotlib, mh.png (tools/mh_plot.py).  STOP.txt in OUT stops it
# after the current point; the same command with --resume continues it from its checkpoint, to the bit.
using TDGL, LinearAlgebra, Printf
BLAS.set_num_threads(1)
if isempty(ARGS) || ARGS[1] in ("-h", "--help")
    println("usage: julia --project=tdgl2 tdgl2/scripts/mh.jl RUN.toml --Hmax HMAX --dH DH [--J J] [--dir OUT] [--no-down] [--resume]"); exit(0)
end
path = ARGS[1]
arg(k) = (i = findfirst(==(k), ARGS); i === nothing ? nothing : ARGS[i + 1])
num(k) = (a = arg(k); a === nothing ? nothing : parse(Float64, a))
Hmax = num("--Hmax"); dH = num("--dH")
(Hmax === nothing || dH === nothing) && error("--Hmax and --dH are needed")
0 < dH <= Hmax || error("0 < --dH <= --Hmax")
J = something(num("--J"), 0.0)
Hup = collect(0.0:dH:Hmax + dH / 2)
Hs = "--no-down" in ARGS ? Hup : vcat(Hup, reverse(Hup)[2:end])
branch = [k <= length(Hup) ? "up" : "down" for k in eachindex(Hs)]

c = load_config(path)
od = arg("--dir")
# by default a folder per loop, named by its fields and current, next to out_dir of [run]
od === nothing && (isempty(c.out_dir) ? error("[run] has no out_dir: give --dir") :
                   (od = rstrip(c.out_dir, '/') * @sprintf("_MH_H%g_d%g", Hmax, dH) * (J == 0 ? "" : @sprintf("_J%g", J))))
resume = "--resume" in ARGS && isfile(joinpath(od, "restart", "state_MH.jld2"))
!resume && isfile(joinpath(od, "output", "rungs_MH.csv")) && error("$od already holds an M(H) loop (--resume continues it)")
if resume
    w = TDGL.read_state(joinpath(od, "restart", "state_MH.jld2"))
    j0 = Int(w["jp"])
    (j0 <= length(Hs) && w["Hs"] == Hs[1:j0] && all(==(J), w["J"])) ||
        error("$od holds a loop with other fields or another current (H = $(w["Hs"]), J = $(w["J"][1])): give another --dir")
end
mkpath(joinpath(od, "output"))
TDGL._scheme_defaults!(c.scheme, od)
TDGL._keep_run_toml(path, od); TDGL._keep_shape(c, od); TDGL._keep_start(c, path, od); TDGL._keep_scheme_defaults(od, c.scheme)
st = TDGL.Stepper(c.model, c.grid, c.masks; c.scheme...)
pr = c.protocol
pr = Protocol(; (k => getfield(pr, k) for k in fieldnames(Protocol))..., keep_rung_states=true, stop_after_V=0, t_confirm_GL=0.0)
@printf("%s: M(H) at J = %g, H = 0 → %g%s in steps of %g (%d points)%s\n", basename(abspath(od)), J, Hmax,
        "--no-down" in ARGS ? "" : " → 0", dH, length(Hs), resume ? " (resumed)" : "")
L = if resume
    resume_ladder(st, fill(J, length(Hs)); H=Hs, label="MH", pr=pr, out_dir=od)
else
    s0 = TDGL.start_state(c, path)
    s = s0 === nothing ? initial_state(c.model, c.grid, c.masks, Hs[1]; materials=st.mt === nothing ? nothing : st.mt.models) : s0
    ladder(st, s, fill(J, length(Hs)); H=Hs, label="MH", pr=pr, out_dir=od)
end
TDGL.ladder_done(joinpath(od, "restart", "state_MH.jld2")) || (tui_status(od, "M(H) stopped (STOP.txt): --resume continues it"); exit(0))
tui_status(od, "M(H) done")

N = size(L.Nv, 2)
open(joinpath(od, "output", "mh.csv"), "w") do io
    println(io, "# M = <B_z> - H on the sample block [Hc2 of band 1]; J = $J [J0]; Nv: vortex count of each band on its plane")
    println(io, "point,H,branch,M,", join(("Nv_$µ" for µ in 1:N), ","), ",", join(("psi2_$µ" for µ in 1:N), ","), ",conv,t_int")
    for k in eachindex(L.H)
        @printf(io, "%d,%.6f,%s,%.8e,%s,%s,%s,%.6e\n", k, L.H[k], branch[k], L.M[k], join(L.Nv[k, :], ","),
                join((@sprintf("%.6e", x) for x in L.psi2[k, :]), ","), L.conv[k], L.m[k] * L.t_cycle)
    end
end
for k in eachindex(L.H)
    @printf("%-4s H=%.4f M=%+.5e Nv=%s %s\n", branch[k], L.H[k], L.M[k], join(L.Nv[k, :], "/"), L.conv[k])
end
# the fields the vortex count changed at, up and down: where they differ, the loop is open
for b in ("up", "down")
    ks = findall(==(b), branch); isempty(ks) && continue
    ch = [@sprintf("%.4g→%.4g: %d→%d", L.H[ks[q-1]], L.H[ks[q]], L.Nv[ks[q-1], 1], L.Nv[ks[q], 1])
          for q in 2:length(ks) if L.Nv[ks[q], 1] != L.Nv[ks[q-1], 1]]
    println("Nv_1 changes going $b: ", isempty(ch) ? "none" : join(ch, ", "))
end
tool = joinpath(@__DIR__, "..", "tools", "mh_plot.py")
if Sys.which("python3") !== nothing && isfile(tool)
    success(pipeline(`python3 $tool $(joinpath(od, "output", "mh.csv")) -o $(joinpath(od, "mh.png"))`; stdout=stdout, stderr=stderr)) ||
        println("mh_plot.py failed (matplotlib?): the data are in output/mh.csv")
end
