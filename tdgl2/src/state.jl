# ============================================================================
#  state.jl — every field of the step, preallocated
#
#  Psi_mu on nodes, one contiguous array per band.  Link phases theta = ∫A·dl at the current
#  step and phi at the next (the two are swapped at the end of a step), links V = exp(-i theta)
#  and U = exp(-i phi); after the swap U holds the links of the previous step, which the band
#  substeps use to extrapolate the rotation.  J: supercurrent on links.  h: field on plaquettes.
# ============================================================================

const A3c = Array{ComplexF64,3}
const A3r = Array{Float64,3}

mutable struct State{N}
    G::NTuple{N,A3c}             # Psi at the current step
    F::NTuple{N,A3c}             # Psi at the next step (work)
    θx::A3r; θy::A3r; θz::A3r
    φx::A3r; φy::A3r; φz::A3r
    Ux::A3c; Uy::A3c; Uz::A3c
    Vx::A3c; Vy::A3c; Vz::A3c
    Jx::A3r; Jy::A3r; Jz::A3r
    hx::A3r; hy::A3r; hz::A3r
end

@inline emiphi(φ::Float64) = cis(-φ)

"""
    State(gr, G; θ, θprev=θ, h) -> State

From the band fields and the link phases (θ of this step and of the previous one).
"""
function State(gr::Grid, G::NTuple{N,A3c}, θ::NTuple{3,A3r}, θprev::NTuple{3,A3r},
               h::NTuple{3,A3r}) where N
    (; Nx, Ny, Nz) = gr
    θx, θy, θz = copy.(θ)
    φx, φy, φz = copy.(θprev)
    Vx, Vy, Vz = emiphi.(θx), emiphi.(θy), emiphi.(θz)
    Ux, Uy, Uz = emiphi.(φx), emiphi.(φy), emiphi.(φz)
    State{N}(copy.(G), copy.(G), θx, θy, θz, φx, φy, φz, Ux, Uy, Uz, Vx, Vy, Vz,
             zeros(Nx, Ny+1, Nz+1), zeros(Nx+1, Ny, Nz+1), zeros(Nx+1, Ny+1, Nz),
             copy(h[1]), copy(h[2]), copy(h[3]))
end

"""
    initial_state(md, gr, mk, H) -> State

Meissner seed: Ψ_µ = √(ε_µ/β_µ) on the nodes of the block, A = 0, B_z = H on the z faces.
"""
function initial_state(md::Model{N}, gr::Grid, mk::Masks, H::Real) where N
    (; Nx, Ny, Nz) = gr
    G = ntuple(µ -> ComplexF64.(mk.τψ .* sqrt(max(eps_band(md, µ), 0.0) / md.bands[µ].β)), N)
    θ = (zeros(Nx, Ny+1, Nz+1), zeros(Nx+1, Ny, Nz+1), zeros(Nx+1, Ny+1, Nz))
    hx = zeros(Nx+1, Ny, Nz); hy = zeros(Nx, Ny+1, Nz); hz = zeros(Nx, Ny, Nz+1)
    hz[:, :, 1] .= H; hz[:, :, Nz+1] .= H
    State(gr, G, θ, θ, (hx, hy, hz))
end
