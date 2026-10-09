# ============================================================================
#  octmat.jl — READING AND WRITING THE ARCHIVE
#
#  The archive's `restart/state_Jpos.mat` cannot be read with MAT.jl: Octave's
#  `save`, without `-v7`, writes its own TEXT format, and MAT.jl answers
#  `is not a MAT file`.  The whole project archive is in that format.
#
#  Converting with Octave (`save -v7`) would tie the port to Octave in exactly
#  the test meant to make it independent.  So the reader is written here:
#  ~120 lines, no bit lost (Octave prints 17 significant digits, an exact
#  round trip for Float64), and any `state_J*.mat` of the archive opens from
#  Julia with nothing installed.
#
#  `read_state` tries MAT.jl first (in case the file IS binary v7) and falls
#  back to the text reader, so one function name serves both halves of the
#  archive.
#
#  The WRITER emits the same text format, so a checkpoint written by Julia is
#  opened by Octave with `load` without conversion.  That is what allows the
#  two codes to alternate in the middle of a sweep.
# ============================================================================

# ---------------------------------------------------------------------------
#  READING
# ---------------------------------------------------------------------------

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
    read_state(ruta) -> Dict{String,Any}

Opens an archived `.mat`, binary (MAT.jl) or Octave text.
It is the entry point of the acceptance test.
"""
function read_state(ruta::AbstractString)
    isfile(ruta) || error("does not exist: $ruta")
    try
        return MAT.matread(ruta)
    catch
        return read_octave_text(ruta)
    end
end

# ---------------------------------------------------------------------------
#  WRITING
# ---------------------------------------------------------------------------

"""
    write_octave_text(ruta, pares...)

Writes the Octave text format with 17 significant digits, an exact round
trip for Float64.  `pares` are `"name" => value`.
"""
function write_octave_text(ruta::AbstractString, pares::Pair{String,<:Any}...)
    mkpath(dirname(ruta))
    open(ruta, "w") do io
        println(io, "# Created by TDGL.jl (Octave text format), "
,
                Dates.format(Dates.now(), "e u dd HH:MM:SS yyyy"))
        for (nombre, valor) in pares
            _escribir_var(io, nombre, valor)
        end
    end
    return ruta
end

_fmt(x::Float64) = x == 0.0 ? (signbit(x) ? "-0" : "0") : @sprintf("%.17g", x)

function _escribir_var(io, nombre, v::Real)
    println(io, "# name: ", nombre)
    println(io, "# type: scalar")
    println(io, _fmt(Float64(v)))
end

function _escribir_var(io, nombre, v::AbstractArray{<:Real})
    println(io, "# name: ", nombre)
    println(io, "# type: matrix")
    _dims(io, v)
    for x in v
        println(io, " ", _fmt(Float64(x)))
    end
end

function _escribir_var(io, nombre, v::AbstractArray{<:Complex})
    println(io, "# name: ", nombre)
    println(io, "# type: complex matrix")
    _dims(io, v)
    for z in v
        println(io, " (", _fmt(Float64(real(z))), ",", _fmt(Float64(imag(z))), ")")
    end
end

function _dims(io, v::AbstractArray)
    if ndims(v) <= 2
        println(io, "# rows: ", size(v, 1))
        println(io, "# columns: ", ndims(v) == 1 ? 1 : size(v, 2))
    else
        println(io, "# ndims: ", ndims(v))
        println(io, " ", join(size(v), " "))
    end
end
