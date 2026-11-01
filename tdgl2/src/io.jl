# ============================================================================
#  io.jl — checkpoints
#
#  A checkpoint is the full state of a step: every band, the link phases of this step and of the
#  previous one (with them the band substeps of the first step after a resume rotate the links
#  as if the run had not stopped: resuming = continuing, to the bit) and the field.  Format:
#  JLD2, compressed, with the code version.  For N = 2 the names are those of release 1.2
#  (G1, G2, thetax, ..., thetax_prev, hx, ...), so 1.2 checkpoints load directly, and so do the
#  Octave text `.mat` of the releases up to 1.1 (read only).
# ============================================================================

"""
    read_octave_text(ruta) -> Dict{String,Any}

Reads the text format of Octave's `save`.  Supports scalar, matrix,
complex scalar, complex matrix, bool and string; with `# rows/# columns`
(2-D) or `# ndims` (N-D).  Data are in column order (Fortran), as in Julia,
so `reshape` is enough.
"""
function read_octave_text(ruta::AbstractString)
    lineas = readlines(ruta)
    out = Dict{String,Any}()
    i = 1
    n = length(lineas)
    while i <= n
        l = lineas[i]
        if !startswith(l, "# name: ")
            i += 1
            continue
        end
        nombre = strip(l[9:end])
        i += 1
        i <= n || error("octmat: '$nombre' without '# type:'")
        startswith(lineas[i], "# type: ") || error("octmat: '$nombre' without '# type:'")
        tipo = strip(lineas[i][9:end]); i += 1

        if tipo == "scalar"
            out[nombre] = parse(Float64, strip(lineas[i])); i += 1
        elseif tipo == "bool"
            out[nombre] = parse(Float64, strip(lineas[i])) != 0; i += 1
        elseif tipo == "complex scalar"
            out[nombre] = _parse_complejo(strip(lineas[i])); i += 1
        elseif tipo == "string"
            # '# elements: k' and then k pairs (# length / text)
            i += 1                                  # skip '# elements:'
            len = parse(Int, strip(split(lineas[i], ':')[2])); i += 1
            out[nombre] = lineas[i][1:min(len, length(lineas[i]))]; i += 1
        elseif tipo == "matrix" || tipo == "complex matrix" ||
               tipo == "bool matrix"
            dims, i = _leer_dims(lineas, i, nombre)
            ntot = prod(dims)
            escomplejo = startswith(tipo, "complex")
            datos = escomplejo ? Vector{ComplexF64}(undef, ntot) :
                                 Vector{Float64}(undef, ntot)
            k = 0
            while k < ntot
                s = strip(lineas[i]); i += 1
                isempty(s) && continue
                k += 1
                datos[k] = escomplejo ? _parse_complejo(s) : parse(Float64, s)
            end
            out[nombre] = ntot == 0 ? reshape(datos, dims...) :
                                      reshape(datos, dims...)
        else
            error("octmat: unsupported type '$tipo' in '$nombre'. " *
                  "Save it again with save('-v7',...) and use MAT.jl.")
        end
    end
    return out
end

function _leer_dims(lineas, i, nombre)
    if startswith(lineas[i], "# ndims:")
        i += 1
        dims = parse.(Int, split(strip(lineas[i])))
        return (dims, i + 1)
    elseif startswith(lineas[i], "# rows:")
        r = parse(Int, strip(split(lineas[i], ':')[2])); i += 1
        startswith(lineas[i], "# columns:") ||
            error("octmat: '$nombre' has '# rows:' but no '# columns:'")
        c = parse(Int, strip(split(lineas[i], ':')[2])); i += 1
        return ([r, c], i)
    end
    error("octmat: '$nombre' without dimensions")
end

@inline function _parse_complejo(s::AbstractString)
    t = strip(s)
    (startswith(t, '(') && endswith(t, ')')) ||
        error("octmat: '$t' is not an Octave complex number")
    j = findfirst(==(','), t)
    ComplexF64(parse(Float64, t[2:j-1]), parse(Float64, t[j+1:end-1]))
end

"""
    read_state(path) -> Dict{String,Any}

Opens a checkpoint: JLD2 (`.jld2`) or Octave text (`.mat` of releases up to 1.1).  Binary
MATLAB files are read by release 1.2 (`using MAT`).
"""
function read_state(path::AbstractString)
    isfile(path) || error("does not exist: $path")
    endswith(lowercase(path), ".jld2") && return read_jld2(path)
    open(io -> String(read(io, 6)) == "MATLAB", path) &&
        error("$path is a binary MATLAB file: read it with release 1.2 (`using MAT`) and save it as JLD2")
    read_octave_text(path)
end

read_jld2(path::AbstractString) = JLD2.jldopen(f -> Dict{String,Any}(k => f[k] for k in keys(f)), path, "r")

"""
    checkpoint_path(dir, label) -> String

The checkpoint `state_<label>` in `dir`: the `.jld2` or the `.mat` that exists (the newer one if
both do); if neither, the `.jld2` name.
"""
function checkpoint_path(dir::AbstractString, label::AbstractString)
    j = joinpath(dir, "state_$label.jld2"); m = joinpath(dir, "state_$label.mat")
    isfile(j) && isfile(m) && return mtime(j) >= mtime(m) ? j : m
    isfile(m) ? m : j
end

const CODE_VERSION = "TDGL " * string(something(pkgversion(@__MODULE__), "dev"))

"""
    save_state(path, s; extra...) -> path

Writes the state (atomically: a temporary file, then a rename).  `extra` are stored with it
(current point, ladder so far, parameters).
"""
function save_state(path::AbstractString, s::State{N}; extra...) where N
    mkpath(dirname(abspath(path)))
    tmp = path * ".tmp"
    JLD2.jldopen(tmp, "w"; compress=true) do f
        f["tdgl_version"] = CODE_VERSION
        f["nbands"] = N
        for µ in 1:N
            f["G$µ"] = s.G[µ]
        end
        f["thetax"] = s.θx; f["thetay"] = s.θy; f["thetaz"] = s.θz
        # after the swap of advance! phi holds the link phases of the previous step
        f["thetax_prev"] = s.φx; f["thetay_prev"] = s.φy; f["thetaz_prev"] = s.φz
        f["hx"] = s.hx; f["hy"] = s.hy; f["hz"] = s.hz
        for (k, v) in extra
            f[string(k)] = v
        end
    end
    mv(tmp, path; force=true)
    path
end

"""
    open_extra(st) -> NamedTuple

The ramp of the open boundary (empty for the other boundaries), for `save_state(path, s; open_extra(st)...)`.
With `restore_open!(st, w)` on the loaded dictionary, resuming a `:open` run is continuing it to the bit;
without it the boundary starts from the field of the state, with no ramp (close, not equal).
"""
open_extra(st::Stepper) = st.open === nothing ? (;) :
    (open_count=st.open.count, open_old_x=st.open.old[1], open_old_y=st.open.old[2], open_old_z=st.open.old[3],
     open_new_x=st.open.new[1], open_new_y=st.open.new[2], open_new_z=st.open.new[3])

function restore_open!(st::Stepper, w)
    (st.open === nothing || !haskey(w, "open_count")) && return st
    ob = st.open
    ob.count = Int(w["open_count"])
    for (c, x) in enumerate(("x", "y", "z"))
        copyto!(ob.old[c], w["open_old_$x"]); copyto!(ob.new[c], w["open_new_$x"])
    end
    st
end

"""
    load_state(path, gr) -> (State, Dict)

A state from a 2.0 or 1.2 checkpoint (the second value is everything the file holds).  Without
the previous link phases (Octave text, old JLD2) they are taken equal to the current ones: the
first step after the load then freezes the links in the band substeps, as 1.2 did.
"""
function load_state(path::AbstractString, gr::Grid)
    w = read_state(path)
    N = haskey(w, "nbands") ? Int(w["nbands"]) : 2
    G = ntuple(µ -> ComplexF64.(w["G$µ"]), N)
    sz = (gr.Nx + 1, gr.Ny + 1, gr.Nz + 1)
    all(size(g) == sz for g in G) || error("$path: Ψ is $(size(G[1])), the grid needs $sz")
    θ = (Float64.(w["thetax"]), Float64.(w["thetay"]), Float64.(w["thetaz"]))
    θp = haskey(w, "thetax_prev") ?
         (Float64.(w["thetax_prev"]), Float64.(w["thetay_prev"]), Float64.(w["thetaz_prev"])) : θ
    h = (Float64.(w["hx"]), Float64.(w["hy"]), Float64.(w["hz"]))
    State(gr, G, θ, θp, h), w
end

"""
    write_npy(path, A)

A real or integer array as a NumPy `.npy` file (version 1.0, little endian, Fortran order), so that the
figures read the output of a run without an HDF5 reader.
"""
function write_npy(path::AbstractString, A::AbstractArray{T}) where T<:Union{Float64,Float32,Int64,Int32}
    d = T === Float64 ? "<f8" : T === Float32 ? "<f4" : T === Int64 ? "<i8" : "<i4"
    shp = join(size(A), ", ") * (ndims(A) == 1 ? "," : "")
    h = "{'descr': '$d', 'fortran_order': True, 'shape': ($shp), }"
    pad = 64 - (10 + length(h) + 1) % 64
    h = h * " "^pad * "\n"
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        write(io, UInt8(0x93), "NUMPY", UInt8(1), UInt8(0), UInt16(length(h)), h)
        write(io, collect(A))
    end
    path
end
