# ============================================================================
#  vtk.jl — LEGACY ASCII VTK DUMP FOR PARAVIEW
#
#  Literal port of `write_vtk` of v13.1 (tdgl_mgb2_3d_v13_3.m).  Legacy
#  ASCII VTK: no dependencies, no plugins, readable with an editor.  The port
#  does not plot; this only writes the state for ParaView (or any VTK reader).
#
#  Cropped to the SAMPLE (x1:x2+1, y1:y2+1, z1:z2+1), not to the domain with
#  vacuum, because the vacuum has no condensate and |Psi|^2 isosurfaces would
#  be cut by the box instead of by the sample.
#
#  The default SPACING is in CELLS (1,1,1), not in xi_sigma: at dx=0.5 the
#  real thickness is 3 xi_sigma over 6 cells, so ParaView draws it with twice
#  the relative height.  For LOOKING it makes no difference and counting cells
#  is easier; for a paper figure pass `spacing=:xi`.
#
#  Fields, in the same order as the .m:
#      psi_s, psi_p, rho_s, dphi, Bmag, VECTORS B
#  Extra, only on request (the .m did not have them):
#      fmask  — the condensation mask, to see the notch
#      Phi    — the v14 scalar potential, in the gauge of that run
#      Qimb   — charge imbalance (tau_Q)
#
#  snap_every (Config): XY maps of the central z plane, PGM+CSV in
#  output/snapshots/, written when a current point closes.
# ============================================================================

"""Relative phase wrapped to (-π, π], as Octave's `mod(Δ+π, 2π)-π`."""
@inline function _dphi_wrap(z1::ComplexF64, z2::ComplexF64)
    δ = angle(z2) - angle(z1)
    mod(δ + π, 2π) - π
end

"""
    _recorta_B(h, nx, ny, nz, x1, y1, z1) -> Array{Float64,3}

`hx`, `hy` and `hz` live on staggered faces and each has a different size.
For LOOKING it is enough to crop the sample window and pad with zero what
the staggering leaves out; there is no interpolation, because a poor
interpolation here would be an invented number in a figure.
"""
function _recorta_B(h::AbstractArray, nx::Int, ny::Int, nz::Int,
                    x1::Int, y1::Int, z1::Int)
    B = zeros(nx, ny, nz)
    ax = min(nx, size(h, 1) - x1 + 1)
    ay = min(ny, size(h, 2) - y1 + 1)
    az = min(nz, size(h, 3) - z1 + 1)
    if ax > 0 && ay > 0 && az > 0
        B[1:ax, 1:ay, 1:az] = h[x1:x1+ax-1, y1:y1+ay-1, z1:z1+az-1]
    end
    B
end

function _vtk_escalar(io::IO, nom::AbstractString, A::AbstractArray)
    nx, ny, nz = size(A)
    @printf(io, "SCALARS %s float 1\nLOOKUP_TABLE default\n", nom)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        @printf(io, "%.6g\n", Float64(A[i, j, k]))
    end
end

"""
    write_vtk(fn, F1, F2, hx, hy, hz, mr, x1, x2, y1, y2, z1, z2;
                spacing=:celdas, dx=1, dy=1, dz=1, fmask=nothing, Phi=nothing)

Same signature as in the `.m`.  `spacing = :celdas` reproduces the .m
(SPACING 1 1 1).  `spacing = :xi` uses `dx, dy, dz` in units of ξ_σ.

`F1`/`F2` are cropped to the sample; after a full cycle `G` and `F` agree
there, so either can be passed.
"""
function write_vtk(fn::AbstractString,
                     F1::AbstractArray{<:Complex}, F2::AbstractArray{<:Complex},
                     hx::AbstractArray, hy::AbstractArray, hz::AbstractArray,
                     mr::Real, x1::Int, x2::Int, y1::Int, y2::Int, z1::Int, z2::Int;
                     spacing::Symbol=:celdas,
                     dx::Real=1.0, dy::Real=1.0, dz::Real=1.0,
                     fmask::Union{AbstractArray,Nothing}=nothing,
                     Phi::Union{AbstractArray,Nothing}=nothing,
                     Qimb::Union{AbstractArray,Nothing}=nothing)
    A1 = F1[x1:x2+1, y1:y2+1, z1:z2+1]
    A2 = F2[x1:x2+1, y1:y2+1, z1:z2+1]
    nx, ny, nz = size(A1)
    ps = abs2.(A1)
    pp = abs2.(A2)
    rh = ps .+ pp ./ mr
    dp = [_dphi_wrap(ComplexF64(A1[i, j, k]), ComplexF64(A2[i, j, k]))
          for i in 1:nx, j in 1:ny, k in 1:nz]
    Bx = _recorta_B(hx, nx, ny, nz, x1, y1, z1)
    By = _recorta_B(hy, nx, ny, nz, x1, y1, z1)
    Bz = _recorta_B(hz, nx, ny, nz, x1, y1, z1)
    Bm = sqrt.(Bx.^2 .+ By.^2 .+ Bz.^2)

    sx, sy, sz = spacing === :xi ? (Float64(dx), Float64(dy), Float64(dz)) :
                                   (1.0, 1.0, 1.0)
    spacing in (:celdas, :xi) ||
        error("unknown spacing: $spacing (use :celdas or :xi)")

    mkpath(dirname(fn))
    open(fn, "w") do io
        print(io, "# vtk DataFile Version 3.0\n")
        print(io, "TDGL dos bandas, estado convergido\n")
        print(io, "ASCII\nDATASET STRUCTURED_POINTS\n")
        @printf(io, "DIMENSIONS %d %d %d\n", nx, ny, nz)
        @printf(io, "ORIGIN 0 0 0\nSPACING %.6g %.6g %.6g\n", sx, sy, sz)
        @printf(io, "POINT_DATA %d\n", nx * ny * nz)
        _vtk_escalar(io, "psi_s", ps)
        _vtk_escalar(io, "psi_p", pp)
        _vtk_escalar(io, "rho_s", rh)
        _vtk_escalar(io, "dphi",  dp)
        _vtk_escalar(io, "Bmag",  Bm)
        print(io, "VECTORS B float\n")
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            @printf(io, "%.6g %.6g %.6g\n", Bx[i, j, k], By[i, j, k], Bz[i, j, k])
        end
        if fmask !== nothing
            _vtk_escalar(io, "fmask", fmask[x1:x2+1, y1:y2+1, z1:z2+1])
        end
        if Phi !== nothing
            _vtk_escalar(io, "Phi", Phi[x1:x2+1, y1:y2+1, z1:z2+1])
        end
        if Qimb !== nothing
            _vtk_escalar(io, "Qimb", Qimb[x1:x2+1, y1:y2+1, z1:z2+1])
        end
    end
    fn
end

"""
    write_vtk(fn, s, p; spacing=:celdas, fmask=nothing, Phi=nothing)

Convenience method on a `State` and its `Params`.  Uses `G1`/`G2` (what the
checkpoint holds).  After a cycle they agree with `F` on the sample.
"""
function write_vtk(fn::AbstractString, s::State, p::Params;
                     spacing::Symbol=:celdas,
                     fmask::Union{AbstractArray,Nothing}=nothing,
                     Phi::Union{AbstractArray,Nothing}=nothing,
                     Qimb::Union{AbstractArray,Nothing}=nothing)
    write_vtk(fn, s.G1, s.G2, s.hx, s.hy, s.hz, p.mr,
                p.x1, p.x2, p.y1, p.y2, p.z1, p.z2;
                spacing=spacing, dx=p.dx, dy=p.dy, dz=p.dz,
                fmask=fmask, Phi=Phi, Qimb=Qimb)
end

"""
    write_vtk(fn, w, p; kwargs...)

From a `Dict` of `read_state` (a Julia or Octave checkpoint).
"""
function write_vtk(fn::AbstractString, w::AbstractDict, p::Params; kwargs...)
    write_vtk(fn, ComplexF64.(w["G1"]), ComplexF64.(w["G2"]),
                w["hx"], w["hy"], w["hz"], p.mr,
                p.x1, p.x2, p.y1, p.y2, p.z1, p.z2; kwargs...)
end

"""
    write_snapshot_xy(dir, s, p, label, J; tag="") -> String

Central z plane of `|Ψ_σ|²` and `|Ψ_π|²` cropped to the sample, as ASCII
PGM (opens with eog/feh/GIMP) and CSV.  It is the `snap_every` of the `.m`
(XY maps per band every N current points).  Called when a point CLOSES, like
the VTK dump: it does not touch the dynamics.
"""
function write_snapshot_xy(dir::AbstractString, s::State, p::Params,
                             label::AbstractString, J::Float64;
                             tag::AbstractString="")
    mkpath(dir)
    kc = round(Int, (p.z1 + p.z2) / 2)
    xs, xe, ys, ye = p.x1, p.x2, p.y1, p.y2
    p1 = abs2.(s.G1[xs:xe, ys:ye, kc])
    p2 = abs2.(s.G2[xs:xe, ys:ye, kc])
    stem = @sprintf("%s_%s_J%+.3f", isempty(tag) ? "snap" : tag, label, J)
    f1 = joinpath(dir, stem * "_psi_s.pgm")
    f2 = joinpath(dir, stem * "_psi_p.pgm")
    mkpath(dirname(f1))
    _escribe_pgm(f1, p1, "psi_s |Psi|^2  $label J=$J  z=$kc")
    _escribe_pgm(f2, p2, "psi_p |Psi|^2  $label J=$J  z=$kc")
    _escribe_csv2(joinpath(dir, stem * "_psi_s.csv"), p1)
    _escribe_csv2(joinpath(dir, stem * "_psi_p.csv"), p2)
    idx = joinpath(dir, "INDEX.txt")
    open(idx, "a") do io
        @printf(io, "%s  max_s=%.4f mean_s=%.4f  max_p=%.4f mean_p=%.4f  %s %s\n",
                stem, maximum(p1), sum(p1)/length(p1),
                maximum(p2), sum(p2)/length(p2), f1, f2)
    end
    f1
end

function _escribe_pgm(fn::AbstractString, A::AbstractMatrix, comentario::AbstractString)
    nx, ny = size(A)
    open(fn, "w") do io
        println(io, "P2")
        println(io, "# ", comentario)
        @printf(io, "%d %d\n255\n", nx, ny)
        # j decreasing: y1 at the bottom, as on a Cartesian axis.
        for j in ny:-1:1
            for i in 1:nx
                v = A[i, j]
                g = v < 0 ? 0 : (v > 1 ? 255 : round(Int, 255 * v))
                print(io, g, (i == nx ? '\n' : ' '))
            end
        end
    end
    fn
end

function _escribe_csv2(fn::AbstractString, A::AbstractMatrix)
    nx, ny = size(A)
    open(fn, "w") do io
        for j in 1:ny
            for i in 1:nx
                @printf(io, "%.6g", A[i, j])
                print(io, i == nx ? '\n' : ',')
            end
        end
    end
    fn
end

"""
    vtk_from_checkpoint(fmat, fout; buf=1, mr=nothing, spacing=:celdas)

From the archive to `.vtk` without simulating.  Infers the sample window
from `buf` (default 1, the solver's).  `mr` is taken from the checkpoint if
present; otherwise it must be passed: it enters `rho_s = |Ψ_σ|² + |Ψ_π|²/m_r`.

    vtk_from_checkpoint("runs/.../restart/state_Jpos.mat", "salida.vtk")
"""
function vtk_from_checkpoint(fmat::AbstractString, fout::AbstractString;
                              buf::Int=1, mr::Union{Real,Nothing}=nothing,
                              spacing::Symbol=:celdas,
                              dx::Real=1.0, dy::Real=1.0, dz::Real=1.0)
    w = read_state(fmat)
    G1 = w["G1"]
    nx, ny, nz = size(G1)                 # domain nodes = (Nx+1, Ny+1, Nz+1)
    Nx, Ny, Nz = nx - 1, ny - 1, nz - 1
    x1 = buf + 1; x2 = Nx - buf
    y1 = buf + 1; y2 = Ny - buf
    z1 = buf + 1; z2 = Nz - buf
    x2 >= x1 && y2 >= y1 && z2 >= z1 ||
        error("vtk_from_checkpoint: buf=$buf does not fit in the domain $(Nx)x$(Ny)x$(Nz)")
    mr_ = mr === nothing ? (haskey(w, "mr") ? Float64(w["mr"]) : 1.0) : Float64(mr)
    write_vtk(fout, ComplexF64.(G1), ComplexF64.(w["G2"]),
                w["hx"], w["hy"], w["hz"], mr_,
                x1, x2, y1, y2, z1, z2;
                spacing=spacing, dx=dx, dy=dy, dz=dz)
end
