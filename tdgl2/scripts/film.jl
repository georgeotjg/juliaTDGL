# A 3D film of a run, only around the moments that matter (does not touch the run; it can be running):
#   OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/film.jl RUNDIR [+|-] [--J J] [--every DT] [--events N] [--T TMAX]
#                                                                [--state FILE]
# From the kept state of the first flowing rung of a polarity (+ by default; keep_rung_states = true in [protocol]), or of the
# last silent one with --J above it, or from any state file of the run's mesh (--state, save_state; at --J, 0 by default: the
# cooling of a sample seeded by start_state, the relaxation of a lattice), the state is stepped at its J; every DT t_GL (0.05) the vortex map of every band is taken on
# every interior z plane (the winding of the gauge-invariant phase per x–y plaquette, count_vortices: an integer, exact on the lattice).
# An event starts when the map changes: the 5 samples before it (held in memory) and every sample after it are kept until the map
# is back to what it was before for 5 samples (the vortex left) or has not changed for 40 (a new arrangement that stays): the
# entry, the crossing and the exit of each vortex, not the waiting between them.  It stops after N events (2) or TMAX t_GL (400).
# Writes RUNDIR/figures/film_<polarity>_J<J>/:
#   film.pvd (frame_NNNN.vti)    the data of every frame on the nodes, as VTK files (any VTK reader: ParaView, VisIt, or Python):
#                                |Ψµ|² of every band, the supercurrent of all bands (vector, with the sign of the applied
#                                current), the field B (vector, Hc2), the vacuum mask and the condensation mask f
#   vortices.pvd (vort_NNNN.vtp) the vortex plaquettes as points, with their band and sign
#   film.html                    by tools/film_html.py: the same film in 3D in any browser, turned and zoomed with the mouse and
#                                played with a slider (plotly.js inside the page when the Python package plotly is installed,
#                                from its CDN otherwise)
#   film.gif, film_frame.png     by tools/film_gif.py (numpy + matplotlib): the sample, the mid plane with |Ψ1|² and the
#                                supercurrent, each vortex as a line through its plaquettes, in 3D; the mid plane from above
#   vortices.npy (t, x, y, z, band, sign; one row of sign 0 per kept time), mid.npz-like npy files of the mid z plane
#   (psi1_mid, jx_mid, jy_mid: frames × nx × ny), counts.npy (every sample: t and the plaquettes of each band), meta.json
using TDGL, Printf, LinearAlgebra
BLAS.set_num_threads(1)
if isempty(ARGS) || ARGS[1] in ("-h", "--help")
    println("usage: julia --project=tdgl2 tdgl2/scripts/film.jl RUNDIR [+|-] [--J J] [--every DT] [--events N] [--T TMAX] [--state FILE]"); exit(0)
end
dir = ARGS[1]
opt(k, d) = (i = findfirst(==(k), ARGS); i === nothing ? d : parse(Float64, ARGS[i + 1]))
label = "-" in ARGS[2:end] ? "J-" : "J+"
every = opt("--every", 0.05); nev = round(Int, opt("--events", 2.0)); Tmax = opt("--T", 400.0)
isfile(joinpath(dir, "run.toml")) || error("$dir has no run.toml: not the folder of a release-2 run")
c = load_config(joinpath(dir, "run.toml"))
st = TDGL.Stepper(c.model, c.grid, c.masks; c.scheme...)
gr = c.grid; mk = c.masks; N = nbands(c.model)
(; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr

# the state: the first flowing rung of the polarity at its own J, or the last silent one stepped at --J
p = joinpath(dir, "output", "rungs_$label.csv"); isfile(p) || error("$p not found")
L = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(p))
h = split(L[1], ","); iJ = findfirst(==("J"), h); ic = findfirst(==("conv"), h)
rows = split.(L[2:end], ","); Js = [parse(Float64, r[iJ]) for r in rows]; cv = [isempty(r[ic]) ? ' ' : r[ic][1] for r in rows]
kept(J) = joinpath(dir, "restart", "rungs", @sprintf("state_%s_J%+.4f.jld2", label, J))
Jarg = opt("--J", NaN)
si = findfirst(==("--state"), ARGS)
if si !== nothing
    sp = ARGS[si + 1]; isfile(sp) || error("--state: $sp not found")
    J = (label == "J-" ? -1 : 1) * abs(isnan(Jarg) ? 0.0 : Jarg)
    s, _ = load_state(sp, gr); from = "the state " * basename(sp)
elseif isnan(Jarg)
    k = findfirst(q -> cv[q] in ('V', 'N', 'D') && isfile(kept(Js[q])), eachindex(cv))
    k === nothing && error("no kept state of a flowing rung in $dir ($label): run the ladder with keep_rung_states = true, or give --J")
    J = Js[k]; s, _ = load_state(kept(J), gr); from = @sprintf("the flowing rung %+.4f", J)
else
    J = (label == "J-" ? -1 : 1) * abs(Jarg)
    k = findlast(q -> cv[q] == 'F' && abs(Js[q]) < abs(J) && isfile(kept(Js[q])), eachindex(cv))
    k === nothing && error("no kept silent state below |J| = $(abs(J)) in $dir ($label)")
    s, _ = load_state(kept(Js[k]), gr); from = @sprintf("the silent rung %+.4f", Js[k])
end
out = joinpath(dir, "figures", @sprintf("film_%s_J%+.4f", label, J)); mkpath(out)
HJ = selffield(st, J)
nx, ny, nz = x2 - x1 + 2, y2 - y1 + 2, z2 - z1 + 2
vac = mk.vac === nothing ? falses(size(mk.f)) : mk.vac

# one frame: the fields on the nodes of the sample and the vortex plaquettes of every band on every z plane (a function of its
# arguments only: the globals of a script are untyped and would make these loops slow)
function frame(s, md, mk, gr::Grid, N::Int)
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = gr
    nx, ny, nz = x2 - x1 + 2, y2 - y1 + 2, z2 - z1 + 2
    P = [Float32.(abs2.(@view s.G[µ][x1:x2+1, y1:y2+1, z1:z2+1])) for µ in 1:N]
    Jv = zeros(Float32, 3, nx, ny, nz); Bv = zeros(Float32, 3, nx, ny, nz)
    for µ in 1:N
        G = s.G[µ]; wx, wy, wz = (stiffness(md.bands[µ], a) for a in 1:3)
        lx(i, j, k) = -wx * imag(conj(G[i,j,k]) * s.Vx[i,j,k] * G[i+1,j,k]) / dx
        ly(i, j, k) = -wy * imag(conj(G[i,j,k]) * s.Vy[i,j,k] * G[i,j+1,k]) / dy
        lz(i, j, k) = -wz * imag(conj(G[i,j,k]) * s.Vz[i,j,k] * G[i,j,k+1]) / dz
        # the links to the nodes: the mean of the link that leaves the node and the one that arrives, one at the faces
        nodal(f, q, q1, q2, up, dn) = q == q1 ? f(up...) : q == q2 + 1 ? f(dn...) : (f(up...) + f(dn...)) / 2
        for (c3, k) in enumerate(z1:z2+1), (b, j) in enumerate(y1:y2+1), (a, i) in enumerate(x1:x2+1)
            Jv[1, a, b, c3] += nodal(lx, i, x1, x2, (i, j, k), (i - 1, j, k))
            Jv[2, a, b, c3] += nodal(ly, j, y1, y2, (i, j, k), (i, j - 1, k))
            Jv[3, a, b, c3] += nodal(lz, k, z1, z2, (i, j, k), (i, j, k - 1))
        end
    end
    for (c3, k) in enumerate(z1:z2+1), (b, j) in enumerate(y1:y2+1), (a, i) in enumerate(x1:x2+1)
        # B on the nodes: the mean of the faces that touch the node, in its plane
        Bv[1, a, b, c3] = (s.hx[i, j-1, k-1] + s.hx[i, j, k-1] + s.hx[i, j-1, k] + s.hx[i, j, k]) / 4
        Bv[2, a, b, c3] = (s.hy[i-1, j, k-1] + s.hy[i, j, k-1] + s.hy[i-1, j, k] + s.hy[i, j, k]) / 4
        Bv[3, a, b, c3] = (s.hz[i-1, j-1, k] + s.hz[i, j-1, k] + s.hz[i-1, j, k] + s.hz[i, j, k]) / 4
    end
    V = Float64[]
    # the vortices on the interior planes, the ones with an equation: the face planes are copies of their neighbours, and the
    # mask f of a notch is not written on the top one (a count there would see the windings inside the notch)
    for µ in 1:N, k in z1+1:z2
        _, n, _, _ = count_vortices(s.G[µ], s.Vx, s.Vy, s.hz, mk.f, gr, k)
        for I in findall(!=(0), n)
            append!(V, ((I[1] - 0.5) * dx, (I[2] - 0.5) * dy, (k - z1) * dz, µ, n[I]))
        end
    end
    (P = P, J = Jv, B = Bv, V = reshape(V, 5, :))
end
vmap(F) = sortslices(F.V; dims=2)

# --- the VTK XML files (raw appended binary, little endian) ---
function write_vti(fn, F)
    arrays = Any[("psi2_" * c.info["band_names"][µ], F.P[µ], 1) for µ in 1:N]
    push!(arrays, ("supercurrent", F.J, 3), ("B", F.B, 3), ("vacuum", Float32.(vac[x1:x2+1, y1:y2+1, z1:z2+1]), 1),
          ("f", Float32.(mk.f[x1:x2+1, y1:y2+1, z1:z2+1]), 1))
    open(fn, "w") do io
        print(io, """<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n""",
              """<ImageData WholeExtent="0 $(nx-1) 0 $(ny-1) 0 $(nz-1)" Origin="0 0 0" Spacing="$dx $dy $dz">\n""",
              """<Piece Extent="0 $(nx-1) 0 $(ny-1) 0 $(nz-1)">\n<PointData>\n""")
        off = 0
        for (name, A, nc) in arrays
            print(io, """<DataArray type="Float32" Name="$name" NumberOfComponents="$nc" format="appended" offset="$off"/>\n""")
            off += 8 + sizeof(A)
        end
        print(io, "</PointData>\n</Piece>\n</ImageData>\n<AppendedData encoding=\"raw\">\n_")
        for (_, A, _) in arrays
            write(io, htol(UInt64(sizeof(A)))); write(io, htol.(vec(A)))
        end
        print(io, "\n</AppendedData>\n</VTKFile>\n")
    end
end
function write_vtp(fn, V)
    n = size(V, 2); pts = Float32.(V[1:3, :]); band = Float32.(V[4, :]); sg = Float32.(V[5, :])
    conn = Int64.(0:n-1); offs = Int64.(1:n)
    arrays = (("Points", pts, 3, "Float32"), ("band", band, 1, "Float32"), ("sign", sg, 1, "Float32"),
              ("connectivity", conn, 1, "Int64"), ("offsets", offs, 1, "Int64"))
    open(fn, "w") do io
        print(io, """<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n""",
              """<PolyData>\n<Piece NumberOfPoints="$n" NumberOfVerts="$n" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n""")
        off = 0
        tag(name, nc, T) = """<DataArray type="$T" Name="$name" NumberOfComponents="$nc" format="appended" offset="$off"/>\n"""
        sz(A) = 8 + sizeof(A)
        print(io, "<Points>\n", tag("Points", 3, "Float32")); off += sz(pts); print(io, "</Points>\n<PointData>\n")
        print(io, tag("band", 1, "Float32")); off += sz(band); print(io, tag("sign", 1, "Float32")); off += sz(sg)
        print(io, "</PointData>\n<Verts>\n", tag("connectivity", 1, "Int64")); off += sz(conn)
        print(io, tag("offsets", 1, "Int64")); off += sz(offs); print(io, "</Verts>\n</Piece>\n</PolyData>\n")
        print(io, "<AppendedData encoding=\"raw\">\n_")
        for (_, A, _, _) in arrays
            write(io, htol(UInt64(sizeof(A)))); write(io, htol.(vec(A)))
        end
        print(io, "\n</AppendedData>\n</VTKFile>\n")
    end
end

# --- the run: a ring of the last 5 samples, kept when the vortex map changes ---
function film_run(s, st, c, mk, gr, N, HJ, out, every, nev, Tmax)
    ne = max(1, round(Int, every / st.dt)); nsteps = round(Int, Tmax / st.dt)
    ring = Tuple{Float64,Any}[]; kept = Tuple{Float64,Any}[]; counts = Float64[]
    prev = nothing; base = nothing; events = 0; active = false; still = 0; back = 0; t = 0.0; nchg = 0; nsmp = 0
    function keep!(tf)
        push!(kept, tf); q = length(kept)
        write_vti(joinpath(out, @sprintf("frame_%04d.vti", q)), tf[2]); write_vtp(joinpath(out, @sprintf("vort_%04d.vtp", q)), tf[2].V)
    end
    for q in 1:nsteps
        step!(s, st; H=c.H, HJ=HJ); t += st.dt
        q % ne == 0 || continue
        F = frame(s, c.model, mk, gr, N); m = vmap(F)
        push!(counts, t); for µ in 1:N; push!(counts, count(==(µ), F.V[4, :])); end
        changed = prev !== nothing && m != prev
        nsmp += 1; nchg += changed
        if !active && changed
            # an event starts: the 5 samples before it, then every sample until the map is back to what it was before
            # (the vortex left) for 5 samples, or has not changed for 40 (a new arrangement that stays)
            base = prev; active = true; still = 0; back = 0
            foreach(keep!, ring); empty!(ring); keep!((t, F))
        elseif active
            keep!((t, F))
            still = changed ? 0 : still + 1; back = m == base ? back + 1 : 0
            if back >= 5 || still >= 40
                events += 1; active = false
                @printf("event %d ended at t = %.2f t_GL\n", events, t); flush(stdout)
                events >= nev && break
            end
        else
            push!(ring, (t, F)); length(ring) > 5 && popfirst!(ring)
        end
        prev = m
        length(kept) >= 2000 && (println("2000 frames: stopping"); break)
    end
    kept, counts, events, t, nchg / max(nsmp - 1, 1)
end
kept_frames, counts, events, t, fchg = film_run(s, st, c, mk, gr, N, HJ, out, every, nev, Tmax)

for (name, pre, ext) in (("film.pvd", "frame", "vti"), ("vortices.pvd", "vort", "vtp"))
    open(joinpath(out, name), "w") do io
        print(io, """<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0">\n<Collection>\n""")
        for (q, (tq, _)) in enumerate(kept_frames)
            @printf(io, "<DataSet timestep=\"%.6f\" file=\"%s_%04d.%s\"/>\n", tq, pre, q, ext)
        end
        print(io, "</Collection>\n</VTKFile>\n")
    end
end
rowsV = Float64[]
for (tq, F) in kept_frames
    append!(rowsV, (tq, NaN, NaN, NaN, 0, 0))
    for r in eachcol(F.V); append!(rowsV, (tq, r...)); end
end
write_npy(joinpath(out, "vortices.npy"), permutedims(reshape(rowsV, 6, :)))
write_npy(joinpath(out, "counts.npy"), permutedims(reshape(counts, N + 1, :)))
kc = (z1 + z2 + 1) ÷ 2 - z1 + 1                     # the middle node plane of the sample
if !isempty(kept_frames)
    write_npy(joinpath(out, "psi1_mid.npy"), Float64.(cat((F.P[1][:, :, kc] for (_, F) in kept_frames)...; dims=3)))
    write_npy(joinpath(out, "jx_mid.npy"), Float64.(cat((F.J[1, :, :, kc] for (_, F) in kept_frames)...; dims=3)))
    write_npy(joinpath(out, "jy_mid.npy"), Float64.(cat((F.J[2, :, :, kc] for (_, F) in kept_frames)...; dims=3)))
end
write_npy(joinpath(out, "vac.npy"), Float64.(vac[x1:x2+1, y1:y2+1, z1:z2+1]))
write_npy(joinpath(out, "f.npy"), Float64.(mk.f[x1:x2+1, y1:y2+1, z1:z2+1]))   # the condensation mask (0 in a notch of f = 0)
# material per region (release 2): the material of each node (1 = [material], 2… = [materials.<name>] in sorted order)
mk.mat === nothing || write_npy(joinpath(out, "mat.npy"), Float64.(mk.mat[x1:x2+1, y1:y2+1, z1:z2+1]))
meta = Dict{String,Any}("run" => basename(abspath(dir)), "polarity" => label, "J" => J, "from" => from, "H" => c.H,
                        "every" => max(1, round(Int, every / st.dt)) * st.dt, "events" => events, "T" => t, "frames" => length(kept_frames),
                        "times" => [tq for (tq, _) in kept_frames], "bands" => c.info["band_names"],
                        "dx" => dx, "dy" => dy, "dz" => dz, "mid_z" => (kc - 1) * dz,
                        "changed_fraction" => fchg,
                        "materials" => mk.mat === nothing ? String[] : vcat("material", c.geom.materials))
open(io -> TDGL._json(io, meta), joinpath(out, "meta.json"), "w")
@printf("%s %s, J = %+.4f from %s: %d event(s) in %.1f t_GL, %d frames in %s\n", basename(abspath(dir)), label, J, from,
        events, t, length(kept_frames), out)
# a continuous flow (the map changes at almost every sample) is not a train of events: if the sampling is not much finer than
# its period (2π/V, V from the rungs), the frames alias like a stroboscope (a ν = 1 film at 2 t_GL against a period of 1.05)
fchg > 0.5 && @printf("note: the vortex map changed at %.0f %% of the samples, a continuous flow: check that --every (%.3g) is well below its period 2π/V\n",
                      100fchg, every)
tools = joinpath(dirname(@__DIR__), "tools")
if Sys.which("python3") !== nothing && !isempty(kept_frames)
    success(pipeline(`python3 $(joinpath(tools, "film_html.py")) $out`; stdout=stdout, stderr=stderr)) ||
        println("film_html.py failed: the data are in $out")
    success(pipeline(`python3 $(joinpath(tools, "film_gif.py")) $out`; stdout=stdout, stderr=stderr)) ||
        println("film_gif.py failed: the data are in $out")
end

