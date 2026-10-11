using Test, TDGL, Printf

using LinearAlgebra
const RAIZ = normpath(joinpath(@__DIR__, "..", ".."))
const FIXT = joinpath(RAIZ, "data", "reference_12x8x4")

"""Configuration of the reference bank.  Gamma_r = 1.3 is deliberate: at
Gamma_r = 1 the F21 fix is inactive and a code with that fault would pass
anyway."""
cfg_fixture(; kw...) = Config(; GEOM_MODE="WEDGE",
    Nx_sc=12, Ny_sc=8, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5,
    kappa_s=2.0, kappa_p=2.0, xi_ratio=1.30, gamma12=0.001,
    gamma_ratio=1.3, H_const=0.50, J_list=[0.0],
    t_min_GL=0.0, mMax=300, eps_conv=1e-14,
    selftest=false, snap_every=0, out_dir="", verbose=false, kw...)

@testset "TDGL" begin
    include("test_compat.jl")
    include("test_octmat.jl")
    include("test_params.jl")
    include("test_geometry.jl")
    include("test_kernel.jl")
    include("test_gamma_r.jl")
    include("test_acceptance.jl")
    include("test_poisson.jl")
    include("test_phi.jl")
    include("test_phi_tf.jl")
    include("test_eigenproblem.jl")
    include("test_vtk.jl")
    include("test_optional_terms.jl")
    include("test_maxwell_implicit.jl")
    include("test_pi_substeps.jl")
    include("test_restart_dt.jl")
    include("test_shapes.jl")
    include("test_observables.jl")
end
