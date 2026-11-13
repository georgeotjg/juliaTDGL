# ============================================================================
#  grid.jl — the staggered mesh and the masks
#
#  Nodes (Psi) are 1..N+1 along each axis; x links live on (i, i+1), and so on.  The
#  superconducting block is x1:x2+1 (nodes) inside a padding of bx, by, bz cells.  The mask f
#  is the local condensation; the link masks say which links carry supercurrent.
# ============================================================================

struct Grid
    Nx::Int; Ny::Int; Nz::Int          # cells, padding included
    x1::Int; x2::Int; y1::Int; y2::Int; z1::Int; z2::Int
    dx::Float64; dy::Float64; dz::Float64
end

"""
    Grid(nx, ny, nz; dx, dy=dx, dz=dx, buf=1, bufx=buf, bufy=buf, bufz=buf)

A sample of nx×ny×nz cells with `buf` padding cells on each face (per axis if given).
"""
function Grid(nx::Int, ny::Int, nz::Int; dx::Real, dy::Real=dx, dz::Real=dx,
              buf::Int=1, bufx::Int=buf, bufy::Int=buf, bufz::Int=buf)
    min(nx, ny, nz) >= 2 || error("Grid: at least 2 cells per axis")
    min(bufx, bufy, bufz) >= 1 || error("Grid: at least one padding cell per face")
    Grid(nx + 2bufx, ny + 2bufy, nz + 2bufz,
         bufx + 1, bufx + nx, bufy + 1, bufy + ny, bufz + 1, bufz + nz,
         Float64(dx), Float64(dy), Float64(dz))
end

"""Sample thickness along z [ξ₁]: sets the self-field of the injected current."""
slab_thickness(gr::Grid) = (gr.z2 - gr.z1 + 1) * gr.dz

"""2/(dx⁻² + dy⁻² + dz⁻²): the length² that sets the diffusion bounds."""
dl(gr::Grid) = 2.0 / (1/gr.dx^2 + 1/gr.dy^2 + 1/gr.dz^2)

struct Masks
    f::Array{Float64,3}      # condensation (1 bulk, 0 < f < 1 weakened, 0 cut)
    τx::Array{Float64,3}     # x links of the superconducting block
    τy::Array{Float64,3}
    τz::Array{Float64,3}
    τψ::Array{Float64,3}     # nodes of the block (initial state)
    vac::Union{Nothing,BitArray{3}}   # vacuum nodes of a sculpted sample (shape with vacuum = true); nothing = none
    mat::Union{Nothing,Array{UInt8,3}}  # material of each node (1 = [material], 2… = [materials.<name>]); nothing = one material
end
Masks(f, τx, τy, τz, τψ) = Masks(f, τx, τy, τz, τψ, nothing, nothing)
Masks(f, τx, τy, τz, τψ, vac) = Masks(f, τx, τy, τz, τψ, vac, nothing)

"""Masks of a plain box: f = 1 and every link of the block live."""
function box_masks(gr::Grid)
    (; Nx, Ny, Nz, x1, x2, y1, y2, z1, z2) = gr
    f  = ones(Nx+1, Ny+1, Nz+1)
    τx = zeros(Nx+1, Ny+1, Nz+1); τy = zeros(Nx+1, Ny+1, Nz+1); τz = zeros(Nx+1, Ny+1, Nz+1)
    τψ = zeros(Nx+1, Ny+1, Nz+1)
    τx[x1:x2,   y1:y2+1, z1:z2+1] .= 1.0
    τy[x1:x2+1, y1:y2,   z1:z2+1] .= 1.0
    τz[x1:x2+1, y1:y2+1, z1:z2  ] .= 1.0
    τψ[x1:x2+1, y1:y2+1, z1:z2+1] .= 1.0
    Masks(f, τx, τy, τz, τψ)
end
