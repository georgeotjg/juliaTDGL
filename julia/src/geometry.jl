# ============================================================================
#  geometry.jl — MASK f(r), MATERIAL MASKS AND DEFECT CATALOGUE
#
#  The material exists in the WHOLE SC block: a defect is a reduced f, not a
#  vacuum hole.  Only the outer buffer is vacuum.  That avoids internal
#  vacuum-SC surfaces and the Dirichlet/Neumann ambiguity of the old code.
#
#  *** STAGGERED MASKS PER LINK TYPE (v9) ***
#  The Psi NODES of the SC block run x1..x2+1; the x links run x1..x2 and
#  live on the transverse NODE rows y1..y2+1, z1..z2+1.  Using ONE mask for
#  the three link types (as v8 did) excludes the high-index rows and includes
#  the low-index ones => the Maxwell source term, the injection BC and the
#  voltmeter are biased towards one edge and the R_z(pi) equivalence between
#  +J and -J breaks.  That was the spurious asymmetry that grew to ~14 % in
#  the SIM control.
# ============================================================================

"""A catalogue defect.  Fields unused by a type are `nothing`."""
Base.@kwdef mutable struct Defect
    type::String
    xc::Union{Int,Nothing} = nothing
    yc::Union{Int,Nothing} = nothing
    zc::Union{Int,Nothing} = nothing
    R::Union{Float64,Nothing} = nothing
    n::Union{Int,Nothing} = nothing
    phi::Union{Float64,Nothing} = nothing
    hx::Union{Int,Nothing} = nothing
    hy::Union{Int,Nothing} = nothing
    hz::Union{Int,Nothing} = nothing
    half::Union{Int,Nothing} = nothing
    depth::Union{Int,Nothing} = nothing
    f::Float64 = 1.0
    xa::Union{Int,Nothing} = nothing
    xb::Union{Int,Nothing} = nothing
    ya::Union{Int,Nothing} = nothing
    yb::Union{Int,Nothing} = nothing
    flo::Union{Float64,Nothing} = nothing
    fhi::Union{Float64,Nothing} = nothing
    nper::Union{Int,Nothing} = nothing
    dir::Union{Int,Nothing} = nothing
    zmode::Union{String,Nothing} = nothing
    zlo::Union{Int,Nothing} = nothing
    zhi::Union{Int,Nothing} = nothing
end

"""The five masks the kernel reads."""
struct Geometry
    fmask::Array{Float64,3}
    taux::Array{Float64,3}
    tauy::Array{Float64,3}
    tauz::Array{Float64,3}
    taupsi::Array{Float64,3}
    defects::Vector{Defect}
end

# ---------------------------------------------------------------------------
#  RASTERISER — writes d.f in the region of the defect
# ---------------------------------------------------------------------------
"""
    in_ngon(px,py,cx,cy,R,n,phi) -> Bool

Point inside a regular n-gon of circumradius `R` and rotation `phi`.
Apothem a = R cos(pi/n); edge radius R_b = a/cos(mod(th,2pi/n) - pi/n).
Large `n` tends to the circle.
"""
function in_ngon(px, py, cx, cy, R, n, phi)
    dx = px - cx; dy = py - cy; r = hypot(dx, dy)
    a = R*cos(pi/n)
    # +1e-9: cos(π/n)*R lands 1 ulp above and the base row was lost
    # (triangle n=3 R=6: base of 10 cells instead of 11). Since v2; it
    # affects only prisms (FUNNEL, STAGGER, HEX…), not WEDGE or SIM (IV, V).
    r <= a + 1e-9 && return true
    r >  R + 1e-9 && return false
    th  = atan(dy, dx) - phi
    seg = mod(th, 2pi/n) - pi/n
    r <= a/cos(seg) + 1e-9
end

function paint_defect!(fmask, d::Defect, x1, x2, y1, y2, z1, z2)
    zA = d.zlo === nothing ? z1 : max(z1, d.zlo)
    zB = d.zhi === nothing ? z2 : min(z2, d.zhi)

    if d.type == "sphere"
        for k in zA:zB, j in y1:y2, i in x1:x2
            if (i-d.xc)^2 + (j-d.yc)^2 + (k-d.zc)^2 <= d.R^2
                fmask[i,j,k] = d.f
            end
        end
    elseif d.type == "cylinder"
        for k in zA:zB, j in y1:y2, i in x1:x2
            if (i-d.xc)^2 + (j-d.yc)^2 <= d.R^2
                fmask[i,j,k] = d.f
            end
        end
    elseif d.type == "box"
        for k in zA:zB, j in y1:y2, i in x1:x2
            if abs(i-d.xc) <= d.hx && abs(j-d.yc) <= d.hy && abs(k-d.zc) <= d.hz
                fmask[i,j,k] = d.f
            end
        end
    elseif d.type == "prism"
        for k in zA:zB, j in y1:y2, i in x1:x2
            if in_ngon(i, j, d.xc, d.yc, d.R, d.n, d.phi)
                fmask[i,j,k] = d.f
            end
        end
    elseif d.type == "wedge"
        # 'dir' lets the notch grow in +y (from face y1) or in -y (from y2).
        # Without it the y mirror is impossible, which is why v7 had removed
        # it from the catalogue.  With dir the mirror is exact:
        # yc -> (y1+y2+1)-yc  and  dir -> -dir.
        dd = d.dir === nothing ? 1 : d.dir
        for k in zA:zB, j in y1:y2, i in x1:x2
            layer = dd*(j - d.yc)
            if layer >= 0 && layer < d.depth
                frac     = 1 - layer/d.depth
                w_steep  = max(1, oround(2*frac))     # steep wall
                w_gentle = oround(d.half*frac)        # gentle ramp
                dxr = i - d.xc
                if dxr >= -w_steep && dxr <= w_gentle
                    fmask[i,j,k] = d.f
                end
            end
        end
    elseif d.type == "fgrad"
        # Continuous sawtooth in f(y): t runs over [0,1] along [ya,yb]
        # (reversed if dir<0), phase = mod(t*nper,1), and f ramps down from fhi
        # to flo and jumps back to fhi at the start of the next tooth.
        # fgrad WRITES f: do not combine it with other defects in the same region.
        Ly = max(1, d.yb - d.ya)
        for j in max(y1, d.ya):min(y2, d.yb)
            t = (j - d.ya)/Ly
            d.dir < 0 && (t = 1 - t)
            fase = mod(t*d.nper, 1)
            fj = d.fhi + (d.flo - d.fhi)*fase
            for k in zA:zB
                fmask[max(x1,d.xa):min(x2,d.xb), j, k] .= fj
            end
        end
    else
        error("unknown defect type: $(d.type)")
    end
    fmask
end

# ---------------------------------------------------------------------------
#  TRANSFORMATIONS OF THE ARRAY
# ---------------------------------------------------------------------------
"""Shifts all defects by `dcell` cells along `ax` ('x'|'y')."""
function shift_defects!(defects, ax::AbstractString, dcell::Int)
    for d in defects
        if lowercase(ax) == "x"
            d.xc !== nothing && (d.xc += dcell)
            d.xa !== nothing && (d.xa += dcell; d.xb += dcell)
        else
            d.yc !== nothing && (d.yc += dcell)
            d.ya !== nothing && (d.ya += dcell; d.yb += dcell)
        end
    end
    defects
end

"""
    mirror_defects!(defects, cfg, x1,x2,y1,y2)

EXACT mirror reflection along the AXIS OF VORTEX MOTION.
With J||x vortices travel along y, so the mirror is in y:

    yc -> (y1+y2+1)-yc ,  phi -> -phi ,  wedge/fgrad: dir -> -dir

The mirror is `(y1+y2+1)-j`, NOT `(y1+y2)-j`: the physical nodes of the
block run y1..y2+1 and that is the only reflection that maps the set of
interior nodes onto itself.  The old mirror reflected and also SHIFTED by
one cell.

The `_REF` control is built HERE and only here.  Flipping again by hand
inside a `case` produces a DOUBLE mirror: that is what left SAWROW_REF with
0 painted cells, i.e. a clean box posing as a control.
"""
function mirror_defects!(defects, cfg::Config, x1, x2, y1, y2)
    mirror_y = lowercase(cfg.current_axis) == "x"
    for d in defects
        if d.type == "wedge" && !mirror_y
            error("mirror_defects: mirroring 'wedge' in x is not defined " *
                  "(its wall/ramp asymmetry is wired along x). Use J||x.")
        end
        if mirror_y
            d.yc  !== nothing && (d.yc = (y1+y2+1) - d.yc)
            d.phi !== nothing && (d.phi = -d.phi)
            if d.type == "fgrad"
                ya0, yb0 = d.ya, d.yb
                d.ya = (y1+y2+1) - yb0
                d.yb = (y1+y2+1) - ya0
                d.dir = -d.dir
            end
            d.type == "wedge" && (d.dir = -d.dir)
        else
            d.xc  !== nothing && (d.xc = (x1+x2+1) - d.xc)
            d.phi !== nothing && (d.phi = pi - d.phi)
            d.type == "fgrad" && error("mirror_defects: fgrad is only defined for J||x.")
        end
    end
    defects
end

"""Sets the z range of each defect from `cfg.z_mode` (overrides the default)."""
function apply_zmode!(defects, cfg::Config, z1, z2)
    zc = oround((z1+z2)/2)
    for d in defects
        zm = lowercase(d.zmode === nothing ? cfg.z_mode : d.zmode)
        if zm == "through"
            d.zlo = z1;  d.zhi = z2
        elseif zm == "half"
            d.zlo = z1;  d.zhi = zc
        elseif zm == "buried"
            q = max(1, oround((z2-z1)/4))
            d.zlo = z1+q; d.zhi = z2-q
        else
            error("unknown z_mode: $zm (use through|half|buried)")
        end
    end
    defects
end

# ---------------------------------------------------------------------------
#  CATALOGUE
# ---------------------------------------------------------------------------
"""
    build_defect_list(cfg, x1,x2,y1,y2,z1,z2) -> Vector{Defect}

*** GEOMETRY CATALOGUE (J||x) ***

Lessons that precede any new geometry:

 1. No defect may touch the INJECTION faces (x1,x2 with J||x).  Current over
    normal material => SPURIOUS eta (a false -12 % was measured).
 2. The ratchet asymmetry must point ALONG THE VORTEX MOTION: with J||x that
    is y.  An asymmetric shape with its ramp along x does NOT rectify with J||x.
 3. In the collective ratchet regime the barrier must CLOSE the path in y
    along (almost) all of x: a free parallel channel sets the bulk Ic and
    eta -> 0.  At κ=25 with Nv≤2 the gate is nucleation: WEDGE rectifies
    with ~70 % of the columns free.  It is not a veto of FUNNEL.
 4. Symmetric objects (spheres, columns) pin but do NOT rectify, even when
    the array is asymmetric.
 5. The theorem that kills η is My or Rz(π) (order 2). A C3 (three blades)
    does NOT contain Rz(π): it is not forbidden. On this mesh 120° is
    irrational and Nv≤2 gives no collective ratchet; that, not the theorem,
    is why it is not run.

The `_REF` modes are not written by hand: they come from `mirror_defects!`.
"""
_phi_cat(cfg::Config, default::Float64) =
    isnan(cfg.prism_phi) ? default : cfg.prism_phi

function build_defect_list(cfg::Config, x1, x2, y1, y2, z1, z2)
    yc = oround((y1+y2)/2); zc = oround((z1+z2)/2); xc = oround((x1+x2)/2)
    Nx_sc = x2-x1+1; Ny_sc = y2-y1+1
    D = Defect[]

    mode = uppercase(cfg.GEOM_MODE)
    is_ref = false
    if length(mode) > 4 && mode[end-3:end] == "_REF"
        is_ref = true; mode = mode[1:end-4]
    end

    mg = cfg.inj_margin
    xsafe_lo, xsafe_hi = lowercase(cfg.current_axis) == "x" ?
                         (x1 + mg, x2 - mg) : (x1, x2)

    if mode == "SIM"
        # Clean box, no defects.  CONTROL: eta MUST be ~0.

    elseif mode == "ESFERA"
        push!(D, Defect(type="sphere", xc=xc, yc=yc, zc=zc, R=6.0, f=0.0))

    elseif mode == "ARRAY"
        # Three defects of growing radius along x.  Known result: eta ~ 0
        # (graded symmetric pinning does not rectify).  With z_mode='through'
        # they are through-going CYLINDERS of circular section, not spheres:
        # in a slab of 10 cells a sphere of R=8 is truncated.
        xs = oround.([0.30, 0.50, 0.70] .* Nx_sc) .+ x1
        for (s, R) in zip(xs, (4.0, 6.0, 8.0))
            push!(D, Defect(type="sphere", xc=s, yc=yc, zc=zc, R=R, f=0.0))
        end

    elseif mode == "WEDGE"
        # Notch (sawtooth) cut into face y1.  BENCHMARK.
        # The injection faces are x1/x2 and the notch is on a y face, so it
        # does not touch them.  A notch on y1 + a flat y2 face breaks the
        # inversion in y, which is THE AXIS OF VORTEX MOTION with J||x:
        # vortices pushed towards y1 exit through the notch, those pushed
        # towards y2 meet the flat face.  Different barriers => DIODE.
        push!(D, Defect(type="wedge", xc=xc, yc=y1, dir=+1,
                        half=oround(cfg.wedge_half*Nx_sc),
                        depth=oround(cfg.wedge_depth*Ny_sc), f=0.0))

    elseif mode == "WEDGE_BOTH"
        # Production notch on y1 PLUS a second one, of independent
        # depth, on face y2.  Tests the falsifiable prediction that a
        # parameter acting on y2 moves |Ic-| and leaves Ic+ unchanged.
        # The second goes at yc = y2+1, NOT y2: the exact node mirror
        # is yc -> (y1+y2+1)-yc.  With yc=y2 it was one cell deeper, and a
        # single cell out of 23 produced |V(+J)|/|V(-J)| = 1.094.
        push!(D, Defect(type="wedge", xc=xc, yc=y1, dir=+1,
                        half=oround(cfg.wedge_half*Nx_sc),
                        depth=oround(cfg.wedge_depth*Ny_sc), f=0.0))
        if cfg.wedge_depth2 > 0
            push!(D, Defect(type="wedge", xc=xc, yc=y2+1, dir=-1,
                            half=oround(cfg.wedge_half*Nx_sc),
                            depth=oround(cfg.wedge_depth2*Ny_sc), f=0.0))
        end

    elseif mode == "ESF_CUNA"
        push!(D, Defect(type="sphere", xc=xc, yc=yc, zc=zc, R=6.0, f=0.0))
        push!(D, Defect(type="wedge", xc=xc + oround(0.18*Nx_sc), yc=y1, dir=+1,
                        half=oround(0.10*Nx_sc), depth=oround(0.22*Ny_sc), f=0.0))

    elseif mode == "FGRAD"
        # Continuous barrier across all of x within the margin; in y, f(y)
        # is a sawtooth.  No f=0 anywhere: no injection artefact and no
        # impassable holes.  The vortex ALWAYS crosses the barrier, at a cost
        # that depends on the direction.
        push!(D, Defect(type="fgrad", xa=xsafe_lo, xb=xsafe_hi, ya=y1, yb=y2,
                        flo=cfg.fgrad_flo, fhi=cfg.fgrad_fhi,
                        nper=cfg.fgrad_nper, dir=cfg.fgrad_dir, f=NaN))

    elseif mode == "SAWROW" || mode == "SAWSPH"
        nteeth = 5
        dep = oround(0.55*Ny_sc)
        xlo = x1 + mg + 2; xhi = x2 - mg - 2
        half = max(3, Int(floor((xhi-xlo)/(2nteeth))) - 1)
        xs = oround.(range(xlo+half, xhi-half, length=nteeth))
        for s in xs
            push!(D, Defect(type="wedge", xc=s, yc=y1, dir=+1,
                            half=half, depth=dep, f=0.0))
        end
        if mode == "SAWSPH"
            zc0 = oround((z1+z2)/2); ysph = y2 - oround(0.25*Ny_sc)
            xsph = oround.([0.30,0.50,0.70] .* (x2-x1)) .+ x1
            for (s, R) in zip(xsph, (3.0, 5.0, 4.0))
                push!(D, Defect(type="sphere", xc=s, yc=ysph, zc=zc0, R=R, f=0.0))
            end
        end

    elseif mode == "SPHROW"
        zc0 = oround((z1+z2)/2); yc0 = oround((y1+y2)/2)
        xs = oround.([0.30,0.50,0.70] .* (x2-x1)) .+ x1
        for (s, R) in zip(xs, (3.0, 6.0, 4.0))
            push!(D, Defect(type="sphere", xc=s, yc=yc0, zc=zc0, R=R, f=0.0))
        end

    elseif mode == "STAGGER"
        Rtri = 5.0
        xs  = oround.(range(xsafe_lo+Rtri, xsafe_hi-Rtri, length=4))
        dyr = oround(0.22*Ny_sc)
        for (s_, xv) in enumerate(xs)
            ys = isodd(s_) ? [yc-dyr, yc+dyr] : [yc-2dyr, yc, yc+2dyr]
            for yv in ys
                if yv > y1+2 && yv < y2-2
                    push!(D, Defect(type="prism", xc=xv, yc=oround(yv), zc=zc,
                                    R=Rtri, n=3, phi=_phi_cat(cfg, 0.0), f=0.0))
                end
            end
        end

    elseif mode == "FUNNEL" || mode == "FUNNEL_IN"
        # Double row of triangles pointing to +y (funnels).  The vortex goes
        # through the gaps: towards +y the converging funnel guides it,
        # towards -y it meets the flat bases.
        # FUNNEL: T1 (larger yc) puts its apex on y2 (1 cell). VI scout at
        # J=0.72: both arms V. FUNNEL_IN: moves the apex away from y2.
        # The FUNNEL of I–III is the one touching the wall.
        Rtri = 6.0; dyrow = 10; per = 16
        inset = mode == "FUNNEL_IN" ? 4 : 0
        xa = xsafe_lo + Rtri; xb = xsafe_hi - Rtri
        xs = collect(xa:per/2:xb)
        isempty(xs) && error("FUNNEL: the mesh is too short in x to place " *
                             "triangles inside the margin.")
        for (s, xv) in enumerate(xs)
            yrow = isodd(s) ? yc - dyrow/2 : yc + dyrow/2
            if inset > 0
                # phi=π/2: apex towards +y at yc+R; base at yc−R/2.
                ymax = y2 - Rtri - inset
                ymin = y1 + Rtri * cos(pi / 3) + inset
                yrow = clamp(Float64(yrow), ymin, ymax)
            end
            push!(D, Defect(type="prism", xc=oround(xv), yc=oround(yrow), zc=zc,
                            R=Rtri, n=3, phi=_phi_cat(cfg, pi/2), f=0.0))
        end

    elseif mode == "TRI_Y1"
        # One n=3 triangle, base on y1 (same face as WEDGE). Compares a
        # "triangle on the wall" with the steep/gentle notch. Not FUNNEL.
        Rtri = 6.0
        yc_t = y1 + oround(Rtri * cos(pi / 3))
        push!(D, Defect(type="prism", xc=xc, yc=yc_t, zc=zc,
                        R=Rtri, n=3, phi=_phi_cat(cfg, pi/2), f=0.0))

    elseif mode == "TRI_Y1_INV"
        # Apex on y1, flat base towards +y (T1 mirrored onto the WEDGE face).
        # Candidate of 20-sep. Not run.
        Rtri = 6.0
        yc_t = y1 + oround(Rtri)
        push!(D, Defect(type="prism", xc=xc, yc=yc_t, zc=zc,
                        R=Rtri, n=3, phi=_phi_cat(cfg, -pi/2), f=0.0))

    elseif mode == "RECT_Y1"
        # 11×5 box on y1 (~55 cells, the WEDGE area). Flat surface +
        # corners towards +y. type=box already existed. Not run.
        hx, hy = 5, 2
        push!(D, Defect(type="box", xc=xc, yc=y1 + hy, zc=zc,
                        hx=hx, hy=hy, hz=max(z2 - z1, 8), f=0.0))

    elseif mode == "HEX"
        # Through-going hexagonal prism (n=6 via in_ngon).  The sample is
        # still the Cartesian MgB2 box: the hexagon is an f=0 defect, not a
        # hexagonal sample nor CAD.  Not used in IV or V.
        Rhex = max(3.0, min(0.22 * min(Float64(Nx_sc), Float64(Ny_sc)),
                            0.5 * Nx_sc - mg - 2.0))
        push!(D, Defect(type="prism", xc=xc, yc=yc, zc=zc,
                        R=Rhex, n=6, phi=_phi_cat(cfg, 0.0), f=0.0))

    elseif mode == "HEX_STAG"
        # Hexagonal hole + n=3 triangles in the REMAINING SC (outside the
        # hex). Inside the hole f=0 paints nothing more. The sample is still
        # the box: not a hexagonal crystal.
        Rhex = max(3.0, min(0.20 * min(Float64(Nx_sc), Float64(Ny_sc)),
                            0.5 * Nx_sc - mg - 2.0))
        push!(D, Defect(type="prism", xc=xc, yc=yc, zc=zc,
                        R=Rhex, n=6, phi=_phi_cat(cfg, 0.0), f=0.0))
        Rtri = 4.0
        phi_t = _phi_cat(cfg, 0.0)
        xs = oround.(range(xsafe_lo + Rtri, xsafe_hi - Rtri, length=4))
        dyr = oround(0.22 * Ny_sc)
        for (s_, xv) in enumerate(xs)
            ys = isodd(s_) ? [yc - dyr, yc + dyr] : [yc - 2dyr, yc, yc + 2dyr]
            for yv in ys
                (yv > y1 + 2 && yv < y2 - 2) || continue
                hypot(Float64(xv - xc), Float64(yv - yc)) > Rhex + Rtri + 1 || continue
                push!(D, Defect(type="prism", xc=xv, yc=oround(yv), zc=zc,
                                R=Rtri, n=3, phi=phi_t, f=0.0))
            end
        end

    else
        error("""unknown GEOM_MODE: $(cfg.GEOM_MODE).
              Catalogue: SIM | ESFERA | ARRAY | WEDGE | WEDGE_BOTH | ESF_CUNA |
              FGRAD | SAWROW | SPHROW | SAWSPH | STAGGER | FUNNEL | FUNNEL_IN |
              TRI_Y1 | TRI_Y1_INV | RECT_Y1 | HEX | HEX_STAG,
              any of them with the suffix _REF.""")
    end

    # ---- robustness tests: shift the whole array ----
    if cfg.x_shift != 0
        dxs = oround(cfg.x_shift*Nx_sc); shift_defects!(D, "x", dxs)
        cfg.verbose && @printf("[x_shift] Defects shifted by %+d cells in x.\n", dxs)
    end
    if cfg.y_shift != 0
        dys = oround(cfg.y_shift*Ny_sc); shift_defects!(D, "y", dys)
        cfg.verbose && @printf("[y_shift] Defects shifted by %+d cells in y.\n", dys)
    end

    is_ref && mirror_defects!(D, cfg, x1, x2, y1, y2)
    apply_zmode!(D, cfg, z1, z2)
    check_injection_margin(D, cfg, x1, x2, y1, y2)
    D
end

# ---------------------------------------------------------------------------
#  CONSTRUCTION AND GUARDS
# ---------------------------------------------------------------------------
function build_geometry(cfg::Config, p::Params)
    (; Nx, Ny, Nz, x1, x2, y1, y2, z1, z2) = p
    fmask  = ones(Float64, Nx+1, Ny+1, Nz+1)
    taux   = zeros(Float64, Nx+1, Ny+1, Nz+1)
    tauy   = zeros(Float64, Nx+1, Ny+1, Nz+1)
    tauz   = zeros(Float64, Nx+1, Ny+1, Nz+1)
    taupsi = zeros(Float64, Nx+1, Ny+1, Nz+1)
    taux[x1:x2  , y1:y2+1, z1:z2+1] .= 1.0   # x links of the SC block
    tauy[x1:x2+1, y1:y2  , z1:z2+1] .= 1.0   # y links
    tauz[x1:x2+1, y1:y2+1, z1:z2  ] .= 1.0   # z links
    taupsi[x1:x2+1, y1:y2+1, z1:z2+1] .= 1.0 # NODES (initial state of Psi)

    # Shape from a file (shapes.jl). The catalogue below is untouched.
    modo = uppercase(cfg.GEOM_MODE)
    if modo in ("CUSTOM", "CUSTOM_REF")
        isempty(cfg.geom_file) && error("GEOM_MODE=$modo needs geom_file")
        rasterize_shape!(fmask, read_shape(cfg.geom_file), p;
                          espejo = modo == "CUSTOM_REF")
        margen_forma(fmask, cfg, p)
        return Geometry(fmask, taux, tauy, tauz, taupsi, Defect[])
    end

    defects = build_defect_list(cfg, x1, x2, y1, y2, z1, z2)
    for d in defects
        paint_defect!(fmask, d, x1, x2, y1, y2, z1, z2)
    end
    Geometry(fmask, taux, tauy, tauz, taupsi, defects)
end

"""
    check_injection_margin(defects, cfg, x1,x2,y1,y2)

No defect may touch the injection faces.  With `inject_mode='selffield'`
the current enters through the self-field on the z faces, not through a
clamp on the electrodes, so a defect near x1/x2 no longer opens any window:
the guard is downgraded from ABORT to WARNING.
"""
function check_injection_margin(defects, cfg::Config, x1, x2, y1, y2)
    estricto = cfg.strict_margin && lowercase(cfg.inject_mode) != "selffield"
    lowercase(cfg.current_axis) == "x" || return nothing
    mg = cfg.inj_margin
    for d in defects
        ext = _extension_x(d)
        ext === nothing && continue
        lo, hi = ext
        if lo < x1 + mg || hi > x2 - mg
            msg = "[margin] defect '$(d.type)' spans x=$lo..$hi; the safe zone " *
                  "is $(x1+mg)..$(x2-mg) (inj_margin=$mg)."
            estricto ? error(msg) : (cfg.verbose && println("  WARNING ", msg))

        end
    end
    nothing
end

function _extension_x(d::Defect)
    if d.type in ("sphere", "cylinder", "prism")
        return (d.xc - Int(ceil(d.R)), d.xc + Int(ceil(d.R)))
    elseif d.type == "box"
        return (d.xc - d.hx, d.xc + d.hx)
    elseif d.type == "wedge"
        return (d.xc - max(1, 2), d.xc + d.half)
    elseif d.type == "fgrad"
        return (d.xa, d.xb)
    end
    nothing
end

"""
    check_f_shunt(g, p) -> (frac_normal, hay_camino)

Diagnostic of the mask f: fraction of the block with f < 0.5 and whether a
path of normal material crosses the sample along the current axis (a normal
short would give a spurious Ic).
"""
function check_f_shunt(g::Geometry, p::Params)
    (; x1, x2, y1, y2, z1, z2) = p
    f = @view g.fmask[x1:x2, y1:y2, z1:z2]
    frac = count(<(0.5), f) / length(f)
    # continuous normal path along x at some (j,k)
    hay = false
    for k in axes(f, 3), j in axes(f, 2)
        all(<(0.5), @view f[:, j, k]) && (hay = true; break)
    end
    (frac, hay)
end

"""
    check_my_symmetry(g, p) -> Int

Counts the cells in which `fmask`, restricted to the INTERIOR NODES
(x1+1:x2, y1+1:y2), differs from its reflection `j -> (y1+y2+1)-j`.  It is
zero if the geometry is M_y-symmetric.  The restriction to interior nodes is
deliberate: it is the region `psi_update!` actually reads, and the raw count
over y1:y2 gives a COSMETIC discrepancy (155 vs 128 cells for WEDGE) that
does not affect the dynamics.
"""
function check_my_symmetry(g::Geometry, p::Params)
    (; x1, x2, y1, y2, z1, z2) = p
    f = g.fmask
    n = 0
    for k in z1:z2, j in y1+1:y2, i in x1+1:x2
        jm = (y1 + y2 + 1) - j
        (y1+1 <= jm <= y2) || continue
        f[i,j,k] != f[i,jm,k] && (n += 1)
    end
    n
end

function check_mx_symmetry(g::Geometry, p::Params)
    (; x1, x2, y1, y2, z1, z2) = p
    f = g.fmask
    n = 0
    for k in z1:z2, j in y1+1:y2, i in x1+1:x2
        im = (x1 + x2 + 1) - i
        (x1+1 <= im <= x2) || continue
        f[i,j,k] != f[im,j,k] && (n += 1)
    end
    n
end

function check_rz_symmetry(g::Geometry, p::Params)
    (; x1, x2, y1, y2, z1, z2) = p
    f = g.fmask
    n = 0
    for k in z1:z2, j in y1+1:y2, i in x1+1:x2
        im = (x1 + x2 + 1) - i
        jm = (y1 + y2 + 1) - j
        (x1+1 <= im <= x2 && y1+1 <= jm <= y2) || continue
        f[i,j,k] != f[im,jm,k] && (n += 1)
    end
    n
end
