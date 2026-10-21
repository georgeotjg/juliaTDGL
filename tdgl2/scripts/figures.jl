# The standard figures of a run, from its folder (does not touch the run; it can be running):
#   OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/figures.jl RUNDIR
# Writes RUNDIR/figures/ the data of each figure (.npy, meta.json) and, with python3 + matplotlib, the figures, with
# README.txt saying what each one is:
#   ladder.png          the sheet of the ladder (tools/ladder_sheet.py): crossings, |V|, rung times, current, the brackets
#   psi_<state>.png     |Ψµ|² of every band on the three mid cuts (x–y, x–z, y–z), vortices and antivortices on x–y
#   share_<state>.png   the share of the supercurrent each band carries, plane by plane along x (tools/state_figures.py)
#   REPORT.md           one page with what was run, every rung, the brackets and η, and the figures (tools/run_report.py)
# <state>: the last silent rung ('F') and the first flowing one of each polarity when the run kept its rung states
# (keep_rung_states = true in [protocol]); otherwise the last state of each polarity.
using TDGL, Printf, LinearAlgebra
BLAS.set_num_threads(1)
isempty(ARGS) && (println("usage: julia --project=tdgl2 tdgl2/scripts/figures.jl RUNDIR"); exit(0))
dir = ARGS[1]
isfile(joinpath(dir, "run.toml")) || error("$dir has no run.toml: not the folder of a release-2 run")
c = load_config(joinpath(dir, "run.toml"))
st = TDGL.Stepper(c.model, c.grid, c.masks; c.scheme...)
gr = c.grid; mk = c.masks; N = nbands(c.model)
(; x1, x2, y1, y2, z1, z2) = gr
out = joinpath(dir, "figures"); mkpath(out)

# the rung history of a polarity: J and the verdict of each closed point
function rungs(label)
    p = joinpath(dir, "output", "rungs_$label.csv"); isfile(p) || return (J=Float64[], conv=Char[])
    L = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(p))
    h = split(L[1], ","); iJ = findfirst(==("J"), h); ic = findfirst(==("conv"), h)
    rows = split.(L[2:end], ",")
    (J = [parse(Float64, r[iJ]) for r in rows], conv = [r[ic][1] for r in rows])
end

# which states: the last 'F' and the first flowing rung of each polarity (from restart/rungs), else the last state
states = Tuple{String,String}[]
for label in ("J+", "J-")
    r = rungs(label); dr = joinpath(dir, "restart", "rungs")
    kept(J) = joinpath(dr, @sprintf("state_%s_J%+.4f.jld2", label, J))
    iF = findlast(==('F'), r.conv); iV = findfirst(k -> r.conv[k] in ('V', 'N', 'D') && (iF === nothing || k > iF), eachindex(r.conv))
    picked = false
    for (k, what) in ((iF, "last silent"), (iV, "first flowing"))
        k === nothing && continue
        p = kept(r.J[k]); isfile(p) || continue
        push!(states, (p, @sprintf("%s_J%+.4f", label, r.J[k]) * " | $label, $what rung, J = $(r.J[k]) ('$(r.conv[k])')")); picked = true
    end
    if !picked
        p = joinpath(dir, "restart", "state_$label.jld2")
        isfile(p) && push!(states, (p, "$(label)_last | $label, the last state of the ladder"))
    end
end

wnpy(name, A) = write_npy(joinpath(out, name), A)
vac = mk.vac === nothing ? falses(size(mk.f)) : mk.vac
blk(A) = A[x1:x2+1, y1:y2+1, z1:z2+1]
V = blk(Float64.(vac))
# each cut on the middle plane, unless it falls in sculpted vacuum (the gap between stacked slabs, a film on a substrate of
# vacuum): then on the centre of the run of full planes (at least half the sample of the fullest) nearest the middle, ties upward
function cut(d, mid)
    n = size(V, d); m = [count(<(0.5), selectdim(V, d, k)) for k in 1:n]; ok = m .>= maximum(m) / 2
    ok[mid] && return mid
    k = argmin(k -> (ok[k] ? 0 : 10^9) + 2abs(k - mid) - (k > mid), 1:n)
    a = k; b = k
    while a > 1 && ok[a-1]; a -= 1; end
    while b < n && ok[b+1]; b += 1; end
    (a + b + 1) ÷ 2
end
ic = x1 - 1 + cut(1, (x1 + x2) ÷ 2 + 2 - x1); jc = y1 - 1 + cut(2, (y1 + y2) ÷ 2 + 2 - y1); kc = z1 - 1 + cut(3, (z1 + z2) ÷ 2 + 2 - z1)
wnpy("vac_xy.npy", V[:, :, kc - z1 + 1]); wnpy("vac_xz.npy", V[:, jc - y1 + 1, :]); wnpy("vac_yz.npy", V[ic - x1 + 1, :, :])
meta = Dict{String,Any}("bands" => c.info["band_names"], "dx" => gr.dx, "dy" => gr.dy, "dz" => gr.dz,
                        "cuts" => Dict("z" => (kc - z1) * gr.dz, "y" => (jc - y1) * gr.dy, "x" => (ic - x1) * gr.dx),
                        "H" => c.H, "states" => Any[])
for (p, desc) in states
    tag, what = strip.(split(desc, "|"))
    s, _ = load_state(p, gr)
    for µ in 1:N
        P = blk(abs2.(s.G[µ]))
        wnpy("psi$(µ)_xy_$tag.npy", P[:, :, kc - z1 + 1]); wnpy("psi$(µ)_xz_$tag.npy", P[:, jc - y1 + 1, :])
        wnpy("psi$(µ)_yz_$tag.npy", P[ic - x1 + 1, :, :])
        _, nmap, _, _ = count_vortices(s.G[µ], s.Vx, s.Vy, s.hz, mk.f, gr, kc)
        wnpy("vort$(µ)_xy_$tag.npy", Int64.(nmap))
    end
    # the supercurrent of each band through every plane of x links between the contacts
    I = [band_currents(s, st; i=i)[µ] for i in x1:x2, µ in 1:N]
    wnpy("share_$tag.npy", I)
    push!(meta["states"], Dict("tag" => tag, "what" => what))
end
open(io -> TDGL._json(io, meta), joinpath(out, "meta.json"), "w")
println("data for $(length(states)) state(s) in $out")
tools = joinpath(dirname(@__DIR__), "tools")
if Sys.which("python3") !== nothing
    write(joinpath(out, "README.txt"), "The standard figures of $(basename(abspath(dir))) (tdgl2/scripts/figures.jl): lengths in ξ of " *
          "band 1, J in J0, H = $(c.H) Hc2.\n" *
          "ladder.png: the sheet of the ladders (tools/ladder_sheet.py): crossings per line, |V|, the time of each rung, the " *
          "current through the mid plane; open markers silent ('F'), filled flowing ('V'); the brackets in the legend.\n")
    sheet = success(pipeline(`python3 $(joinpath(tools, "ladder_sheet.py")) $dir -o $(joinpath(out, "ladder.png"))`;
                             stdout=devnull, stderr=stderr))
    sheet || println("ladder_sheet.py failed")
    states_ok = success(pipeline(`python3 $(joinpath(tools, "state_figures.py")) $dir`; stdout=stdout, stderr=stderr))
    states_ok || println("state_figures.py failed: the data are in $out")
    success(pipeline(`python3 $(joinpath(tools, "run_report.py")) $dir`; stdout=stdout, stderr=stderr)) ||
        println("run_report.py failed")
else
    println("no python3: the data are in $out (tools/ladder_sheet.py and tools/state_figures.py draw them)")
end
