# ============================================================================
#  TDGL.jl — three-dimensional two-band time-dependent Ginzburg-Landau solver.
#
#  WHAT IT IS
#    * A two-band TDGL solver on a staggered Cartesian mesh with link
#      variables (exact lattice gauge invariance), a self-consistent field
#      and transport current injected through its Ampere self-field.
#    * Optional scalar potential by Coulomb projection, with a Thomas-Fermi
#      length and a local relaxation of the residual (scalar_potential.jl,
#      poisson.jl).
#    * Optional terms, all off by default: interband drag, a temperature
#      field with Joule heating, a diffused residual, a physical bath
#      temperature, and shapes read from files.
#    * Regression: with everything off, `Strict` mode reproduces the stored
#      states in data/ (test/test_acceptance.jl).
#
#  WHAT IT IS NOT
#    * It does not use `DifferentialEquations.jl`: the scheme is explicit and
#      the step is set by the CFL condition, not by an adaptive integrator.
#
#  MINIMAL USE
#      using TDGL
#      cfg = Config(GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4,
#                   dx=0.5, dy=0.5, dz=0.5, xi_ratio=1.30, gamma_ratio=1.3,
#                   H_const=0.50, J_list=[0.0], t_min_GL=0, mMax=300,
#                   eps_conv=1e-14, out_dir="ofix")
#      sde = run_case(cfg)
# ============================================================================
module TDGL

using LinearAlgebra, SparseArrays, Printf, Dates, Statistics
import MAT
import TOML
using PrecompileTools

export Config, Params, Geometry, State, SDE, SweepResult, Defect
export Strict, Fast, NumMode, mode_name
export derive_params, current_grid, hc2_pi, xi_ratio_integrated, hc2_pi_f21, xi_ratio_f21
export build_geometry, build_defect_list, paint_defect!, in_ngon
export check_f_shunt, check_my_symmetry, check_mx_symmetry, check_rz_symmetry
export read_shape, rasterize_shape!, write_voxels
export mirror_defects!, shift_defects!, apply_zmode!
export psi_update!, psi_update_ref!, run_selftest, coef_pi
export initial_state, substep!, block_B!
export sweep, run_case, write_summary, ic_guards, bracket_Ic, eta_bracket, brackets
export save_checkpoint, load_checkpoint
export read_state, read_octave_text, write_octave_text
export magnetization, count_vortices, extract_Ic, efficiency, symmetry_table
export write_vtk, vtk_from_checkpoint, write_snapshot_xy
export assemble_laplacian, poisson_convergence, poisson_cost
export vacuum_source, solve_poisson, vacuum_from_file, full_potential
export supercurrent_x, curl_x, divergence_x, divergence_A
export supercurrent, divergence, curl_curl_A, div_curl_identity, drift_div_A
export phi_fields
export MaxwellImplicit, implicit_curl!
export SolverPhi, substep_phi!, block_C2!, apply_phi_psi!, phi_voltage
export link_divergence, residual_div_increment, solve!
export eigenproblem, step1_one_band, step2_two_bands, step3_boundary
export covariant_laplacian, landau_level, hc2_block, hc2_scheme, hc3_over_hc2
export links_from_theta

include("compat.jl")
include("octmat.jl")
include("config.jl")
include("params.jl")
include("geometry.jl")
include("shapes.jl")
include("kernel.jl")
include("maxwell.jl")
include("maxwell_implicit.jl")
include("observables.jl")
include("vtk.jl")
include("sweep.jl")
include("driver.jl")
include("poisson.jl")
include("scalar_potential.jl")
include("eigenproblem.jl")

# ============================================================================
#  PRECOMPILATION.  MEASURED: the first call to `run_case` in a session
#  costs ~44 s of JIT compilation and the second 0.17 s.  For the real work
#  pattern —many short runs launched from the command line— those 44 s per
#  process dominate, not the computation.
#
#  This block runs a tiny case while PRECOMPILING the package, so the
#  machine code is stored in the precompile file and every later session
#  starts compiled.  It is paid once per Julia and package version.
#
#  Both numerical modes are exercised on purpose: `Strict` and `Fast` are
#  different types and generate different specialisations of the kernel.
@setup_workload begin
    cfgs = [(Strict(), 1), (Fast(), 1)]
    @compile_workload begin
        for (modo, _) in cfgs
            c = Config(GEOM_MODE="WEDGE", Nx_sc=6, Ny_sc=4, Nz_sc=2,
                       dx=0.5, dy=0.5, dz=0.5, xi_ratio=1.30, gamma_ratio=1.3,
                       H_const=0.5, J_list=[0.0], delta_v=1, t_min_GL=0.0,
                       iter_min=1, mMax=2, selftest=false, out_dir="",
                       verbose=false, use_threads=false)
            run_case(c; mode=modo)
        end
        assemble_laplacian(3, 3, 3, 1.0, 1.0, 1.0)
        # the v14 path (projection + rotation of Psi) is another set of
        # specialisations; without this, the first run with use_phi=true
        # pays compilation again.
        c14 = Config(GEOM_MODE="WEDGE", Nx_sc=6, Ny_sc=4, Nz_sc=2,
                     dx=0.5, dy=0.5, dz=0.5, xi_ratio=1.30, gamma_ratio=1.0,
                     H_const=0.5, J_list=[0.0], delta_v=1, t_min_GL=0.0,
                     iter_min=1, mMax=2, selftest=false, out_dir="",
                     verbose=false, use_threads=false,
                     use_phi=true, phi_method=:separable)
        run_case(c14; mode=Strict())
    end
end

end # module
