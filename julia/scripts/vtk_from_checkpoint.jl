# ============================================================================
#  scripts/vtk_from_checkpoint.jl — checkpoint to ParaView, no simulation
#
#  Reads a state_J*.mat (Octave text or binary v7) and writes legacy ASCII
#  VTK.  Works for Octave checkpoints as well as Julia ones.
#
#  Usage:
#    julia --project=julia julia/scripts/vtk_from_checkpoint.jl \
#          data/reference_12x8x4/restart/state_Jpos.mat out.vtk
#
#  Optional: spacing in xi_sigma (default: cells):
#    julia --project=julia julia/scripts/vtk_from_checkpoint.jl \
#          state_Jpos.mat out.vtk --xi 0.5
# ============================================================================
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
using TDGL

length(ARGS) >= 2 || error("usage: vtk_from_checkpoint.jl INPUT.mat OUTPUT.vtk [--xi DX]")
fmat = ARGS[1]
fout = ARGS[2]
xi = nothing
if length(ARGS) >= 4 && ARGS[3] == "--xi"
    xi = parse(Float64, ARGS[4])
end
kwargs = xi === nothing ? (;) : (; spacing=:xi, dx=xi, dy=xi, dz=xi)
vtk_from_checkpoint(fmat, fout; buf=1, kwargs...)
println("written: ", fout)
