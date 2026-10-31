# ============================================================================
#  shapes.jl — geometry from a shape file (GEOM_MODE = "CUSTOM" | "CUSTOM_REF")
#
#  The sample is still the box. A shape file lists, in order, regions of the
#  box whose condensation strength f is overwritten. Positions and sizes are
#  in units of ξσ, so the same file rasterises on any mesh. Nodes are tested
#  at their physical position X = (i - x1) dx, over the same node range the
#  catalogue painter uses (x1:x2, y1:y2, z1:z2), so a catalogue shape written
#  as a file gives the same mask bit for bit (test_shapes.jl).
#
#  CUSTOM_REF evaluates the shapes at the mirrored point Y -> (y2 + 1 - j) dy,
#  which is node j -> (y1 + y2 + 1) - j: the exact mirror of the interior node
#  set (mode M5), the same one mirror_defects! uses.
#
#  File format (TOML), all optional except [[shape]]:
#
#      [[shape]]
#      type = "circle"    # circle | rect | triangle | ngon | polygon | sphere | pixels | voxels
#      x = 10.0           # centre (ξσ). Origin: corner (x1, y1, z1) of the sample
#      y = 1.0
#      r = 2.0
#      rot = 15.0         # degrees, about z, counter-clockwise
#      z = "all"          # all | bottom | top | buried | [z0, z1] (ξσ)
#      f = 0.0            # value written inside (1 restores material)
#
#  rect: w, h. triangle: w (base), h (height), skew (apex offset along the
#  base, 0 = isosceles, ±w/2 = right triangle), anchored at the base midpoint.
#  ngon: n, r. polygon: points = [[x, y], ...]. sphere: x, y, zc, r (ignores z).
#  pixels: dx (pixel size, ξσ), rows (one string per row, first row = largest
#  y, first column = x = 0), palette = { "#" = { z = "all", f = 0.0 }, ... };
#  "." and " " leave the node untouched.
#  voxels: dx, dz, layers = one block of rows per z plane (first block = z = 0),
#  rows as in pixels; palette = { "#" = { f = 0.0 }, ... }. Any 3D pattern.
#
#  A figure on another face: axis = "x" (drawn on the y–z plane, extruded along x) or "y"
#  (on the x–z plane, along y); "z" is the default and the keys above.  With axis x or y the
#  centre is u, v (axis x: u = y, v = z; axis y: u = x, v = z), the polygon points are (u, v),
#  rot turns about the axis, and along = "all" | "bottom" | "top" | "buried" | [a0, a1] is
#  the extent along the axis (ξσ, from the face x1 or y1), as z is for axis z.
#  outside = true writes the value outside the figure instead of inside: the figure is
#  what is kept (a cable: a circle with axis = "x", outside = true).
#  vacuum = true (instead of f) removes the material: no superconductor and no normal metal,
#  the insulating outside of the sample (release 2 only: the scheme needs sigma_vac).  A voxel
#  palette entry takes it too: { vacuum = true }.  f = 0 is a normal metal, not vacuum.
#
#  Files written before the format was in English ([[forma]], tipo = "circulo",
#  z = "todo", ...) are still read: _normalise maps the old names.
# ============================================================================

const _Z_WORDS = ("all", "bottom", "top", "buried")
const _TYPES = ("circle", "rect", "triangle", "ngon", "polygon", "sphere", "pixels", "voxels")

# names of the first (Spanish) version of the format
const _OLD_KEYS = Dict("tipo" => "type", "sesgo" => "skew", "puntos" => "points",
                       "filas" => "rows", "paleta" => "palette", "capas" => "layers")
const _OLD_TYPES = Dict("circulo" => "circle", "triangulo" => "triangle", "ngono" => "ngon",
                        "poligono" => "polygon", "esfera" => "sphere", "pixeles" => "pixels",
                        "voxeles" => "voxels")
const _OLD_Z = Dict("todo" => "all", "abajo" => "bottom", "arriba" => "top", "enterrado" => "buried")

_normalise_z(z) = z isa AbstractString ? get(_OLD_Z, z, z) : z

function _normalise(e::AbstractDict)
    o = Dict{String,Any}(get(_OLD_KEYS, k, k) => v for (k, v) in e)
    haskey(o, "type") && (o["type"] = get(_OLD_TYPES, o["type"], o["type"]))
    haskey(o, "z") && (o["z"] = _normalise_z(o["z"]))
    if haskey(o, "palette")
        o["palette"] = Dict{String,Any}(ch => Dict{String,Any}(k => (k == "z" ? _normalise_z(v) : v)
                                                               for (k, v) in q)
                                        for (ch, q) in o["palette"])
    end
    o
end

"""Reads a shape file (TOML) and returns the list of elements."""
function read_shape(path::AbstractString)
    isfile(path) || error("geom_file does not exist: $path")
    d = TOML.parsefile(path)
    raw = get(d, "shape", get(d, "forma", nothing))
    raw === nothing && error("$path: no [[shape]] block")
    elems = [_normalise(e) for e in raw]
    for (n, e) in enumerate(elems)
        haskey(e, "type") || error("$path: shape #$n has no 'type'")
        e["type"] in _TYPES || error("$path: shape #$n has unknown type '$(e["type"])'")
    end
    elems
end

# z range in ξσ of a z specification; Lz = (z2 - z1) dz is the top node plane
# the painter reaches.
function _rango_z(spec, Lz)
    spec === nothing && return (-Inf, Inf)
    if spec isa AbstractString
        spec == "all"    && return (-Inf, Inf)
        spec == "bottom" && return (-Inf, Lz / 2)
        spec == "top"    && return (Lz / 2, Inf)
        spec == "buried" && return (Lz / 4, 3Lz / 4)
        error("z = '$spec' is not one of $(_Z_WORDS) nor [z0, z1]")
    end
    length(spec) == 2 || error("z = $spec: expected [z0, z1]")
    (Float64(spec[1]), Float64(spec[2]))
end

const _TOL = 1e-9
const _AXES = ("x", "y", "z")

# point (u, v) in the shape's local frame (centre at origin, rotation undone)
function _local(e, X, Y)
    cx = Float64(get(e, "x", 0.0)); cy = Float64(get(e, "y", 0.0))
    th = deg2rad(Float64(get(e, "rot", 0.0)))
    dx = X - cx; dy = Y - cy
    th == 0.0 && return (dx, dy)
    c = cos(th); s = sin(th)
    (c * dx + s * dy, -s * dx + c * dy)
end

function _en_poligono(px, py, vx, vy)
    # crossing number; points on an edge count as inside
    n = length(vx); dentro = false
    j = n
    for i in 1:n
        xi, yi, xj, yj = vx[i], vy[i], vx[j], vy[j]
        # on the segment?
        cr = (xj - xi) * (py - yi) - (yj - yi) * (px - xi)
        if abs(cr) <= _TOL * max(1.0, hypot(xj - xi, yj - yi)) &&
           min(xi, xj) - _TOL <= px <= max(xi, xj) + _TOL &&
           min(yi, yj) - _TOL <= py <= max(yi, yj) + _TOL
            return true
        end
        if (yi > py) != (yj > py)
            xint = xi + (py - yi) * (xj - xi) / (yj - yi)
            px < xint && (dentro = !dentro)
        end
        j = i
    end
    dentro
end

"""Is the physical point (X, Y, Z) inside element `e`? (not for pixel layers)"""
function _dentro(e, X, Y, Z, L::NTuple{3,Float64})
    t = e["type"]
    if t == "sphere"
        cx = Float64(e["x"]); cy = Float64(e["y"]); cz = Float64(e["zc"])
        r = Float64(e["r"])
        return (X - cx)^2 + (Y - cy)^2 + (Z - cz)^2 <= r^2 + _TOL
    end
    ax = get(e, "axis", "z")
    ax in _AXES || error("axis = '$ax' is not one of $(_AXES)")
    if ax != "z"
        # the figure in its own plane: (u, v, along) = (y, z, x) or (x, z, y), with the centre (u, v)
        A, la, U, V = ax == "x" ? (X, L[1], Y, Z) : (Y, L[2], X, Z)
        a0, a1 = _rango_z(get(e, "along", "all"), la)
        (a0 - _TOL <= A <= a1 + _TOL) || return false
        q = Dict{String,Any}(k => v for (k, v) in e if !(k in ("x", "y", "z", "u", "v", "axis", "along")))
        q["x"] = get(e, "u", 0.0); q["y"] = get(e, "v", 0.0)
        return _dentro(q, U, V, 0.0, L)
    end
    z0, z1 = _rango_z(get(e, "z", "all"), L[3])
    (z0 - _TOL <= Z <= z1 + _TOL) || return false
    if t == "circle"
        cx = Float64(e["x"]); cy = Float64(e["y"]); r = Float64(e["r"])
        return (X - cx)^2 + (Y - cy)^2 <= r^2 + _TOL
    elseif t == "rect"
        u, v = _local(e, X, Y)
        return abs(u) <= Float64(e["w"]) / 2 + _TOL && abs(v) <= Float64(e["h"]) / 2 + _TOL
    elseif t == "triangle"
        u, v = _local(e, X, Y)
        w = Float64(e["w"]); h = Float64(e["h"]); s = Float64(get(e, "skew", 0.0))
        return _en_poligono(u, v, (-w / 2, w / 2, s), (0.0, 0.0, h))
    elseif t == "ngon"
        u, v = _local(e, X, Y)
        n = Int(e["n"]); r = Float64(e["r"])
        vx = ntuple(k -> r * cos(2pi * (k - 1) / n), n)
        vy = ntuple(k -> r * sin(2pi * (k - 1) / n), n)
        return _en_poligono(u, v, vx, vy)
    elseif t == "polygon"
        pts = e["points"]
        length(pts) >= 3 || error("polygon with fewer than 3 points")
        u, v = _local(e, X, Y)   # x, y default 0: points are absolute unless offset
        return _en_poligono(u, v, Float64.(first.(pts)), Float64.(last.(pts)))
    end
    error("type '$t' not handled here")
end

# value the pixel layer writes at (X, Y, Z), or nothing
function _pixel(e, X, Y, Z, Lz)
    dxp = Float64(get(e, "dx", 0.5))
    filas = e["rows"]
    filas isa AbstractString && (filas = split(chomp(filas), '\n'))
    nr = length(filas)
    c = round(Int, X / dxp) + 1
    r = nr - round(Int, Y / dxp)
    (1 <= r <= nr) || return nothing
    fila = collect(filas[r])
    (1 <= c <= length(fila)) || return nothing
    ch = string(fila[c])
    (ch == "." || ch == " ") && return nothing
    pal = e["palette"]
    haskey(pal, ch) || error("pixels: character '$ch' is not in the palette")
    q = pal[ch]
    z0, z1 = _rango_z(get(q, "z", "all"), Lz)
    (z0 - _TOL <= Z <= z1 + _TOL) || return nothing
    _valor(q)
end

# what an element (or a palette entry) writes: its f, or NaN for vacuum
_valor(q) = get(q, "vacuum", false) === true ? NaN : Float64(get(q, "f", 0.0))

_filas(t) = t isa AbstractString ? split(chomp(t), '\n') : t

# value the voxel layer writes at (X, Y, Z), or nothing
function _voxel(e, X, Y, Z)
    dxv = Float64(get(e, "dx", 0.5)); dzv = Float64(get(e, "dz", dxv))
    capas = e["layers"]
    l = round(Int, Z / dzv) + 1
    (1 <= l <= length(capas)) || return nothing
    filas = _filas(capas[l])
    nr = length(filas)
    r = nr - round(Int, Y / dxv)
    c = round(Int, X / dxv) + 1
    (1 <= r <= nr) || return nothing
    fila = collect(filas[r])
    (1 <= c <= length(fila)) || return nothing
    ch = string(fila[c])
    (ch == "." || ch == " ") && return nothing
    pal = e["palette"]
    haskey(pal, ch) || error("voxels: character '$ch' is not in the palette")
    _valor(pal[ch])
end

"""
    rasterize_shape!(fmask, elems, p; espejo=false, vac=nothing) -> fmask

Writes the f of each element, in order, on the nodes x1:x2, y1:y2, z1:z2.  A vacuum element
(`vacuum = true`) marks its nodes in `vac` (and f = 0 there); a later element that writes an
f gives the material back.  Without `vac` a vacuum element is an error.
"""
function rasterize_shape!(fmask, elems, p::Params; espejo::Bool=false, vac=nothing)
    (; x1, x2, y1, y2, z1, z2, dx, dy, dz) = p
    L = ((x2 - x1) * dx, (y2 - y1) * dy, (z2 - z1) * dz)
    Lz = L[3]
    for k in z1:z2, j in y1:y2, i in x1:x2
        X = (i - x1) * dx
        jj = espejo ? (y1 + y2 + 1) - j : j
        Y = (jj - y1) * dy
        Z = (k - z1) * dz
        for e in elems
            if e["type"] == "pixels"
                v = _pixel(e, X, Y, Z, Lz)
            elseif e["type"] == "voxels"
                v = _voxel(e, X, Y, Z)
            else
                v = _dentro(e, X, Y, Z, L) != (get(e, "outside", false) === true) ? _valor(e) : nothing
            end
            v === nothing && continue
            if isnan(v)
                vac === nothing && error("vacuum = true needs release 2 and an insulating padding (sigma_vac)")
                vac[i, j, k] = true; fmask[i, j, k] = 0.0
            else
                vac === nothing || (vac[i, j, k] = false)
                fmask[i, j, k] = v
            end
        end
    end
    fmask
end

"""
    write_voxels(path, fmask, p; comentario="")

Writes the mask (nodes x1:x2, y1:y2, z1:z2) as a voxel shape: any 3D
pattern, one character per value of f other than 1.
"""
function write_voxels(path::AbstractString, fmask, p::Params; comentario::AbstractString="")
    (; x1, x2, y1, y2, z1, z2, dx, dz) = p
    chars = collect("#abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    pal = Dict{Float64,Char}()
    capas = String[]
    for k in z1:z2
        filas = String[]
        for j in y2:-1:y1
            fila = Char[]
            for i in x1:x2
                v = fmask[i, j, k]
                if v == 1
                    push!(fila, '.')
                else
                    if !haskey(pal, v)
                        length(pal) < length(chars) || error("more than $(length(chars)) values of f")
                        pal[v] = chars[length(pal) + 1]
                    end
                    push!(fila, pal[v])
                end
            end
            push!(filas, String(fila))
        end
        push!(capas, join(filas, "\n"))
    end
    open(path, "w") do io
        isempty(comentario) || println(io, "# ", comentario)
        println(io, "[[shape]]\ntype = \"voxels\"\ndx = ", repr(dx), "\ndz = ", repr(dz))
        println(io, "layers = [")
        for (n, c) in enumerate(capas)
            println(io, "\"\"\"\n", c, "\"\"\"", n < length(capas) ? "," : "")
        end
        println(io, "]\n\n[shape.palette]")
        for (v, c) in sort(collect(pal); by=last)
            println(io, "\"", c, "\" = { f = ", repr(v), " }")
        end
    end
    path
end

"""Nodes with f<1 closer than `inj_margin` cells to the injection faces."""
function margen_forma(fmask, cfg::Config, p::Params)
    (; x1, x2, y1, y2, z1, z2) = p
    mg = cfg.inj_margin
    n = 0
    if lowercase(cfg.current_axis) == "x"
        for k in z1:z2, j in y1:y2, i in x1:x2
            (i < x1 + mg || i > x2 - mg) && fmask[i, j, k] < 1 && (n += 1)
        end
    else
        for k in z1:z2, j in y1:y2, i in x1:x2
            (j < y1 + mg || j > y2 - mg) && fmask[i, j, k] < 1 && (n += 1)
        end
    end
    n > 0 && cfg.verbose &&
        println("  WARNING [margin] the shape touches $n node(s) closer than $mg cells to ",
                "the injection faces (F1).")

    n
end
