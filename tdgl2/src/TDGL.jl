# ============================================================================
#  TDGL.jl — juliaTDGL 2 (development): N-band time-dependent Ginzburg–Landau in 3D
#
#  N order parameters with a real symmetric Josephson coupling, a self-consistent vector
#  potential with a normal channel of conductivity sigma_n, current injected through its
#  Ampère self-field, staggered mesh with link variables (exact lattice gauge invariance),
#  gauge Phi = 0.  Explicit Euler for the bands with per-band substeps, implicit curl-curl for
#  the field.
# ============================================================================
module TDGL

using LinearAlgebra, Printf
import JLD2
using FFTW: plan_rfft, plan_irfft
import FFTW
import TOML

export Band, Model, two_band, regime, equilibrium, nbands, stiffness, eps_band, sigma_dirty
export Grid, Masks, box_masks, slab_thickness
export State, initial_state
export Stepper, step!, selffield, link_conductivity, Heat
export GeomSpec, build_masks, conducting_nodes, Defect, build_defect_list, paint_defect!, in_ngon
export mirror_defects!, shift_defects!, apply_zmode!
export check_f_shunt, check_my_symmetry, check_mx_symmetry, check_rz_symmetry
export read_shape, rasterize_shape!, write_voxels
export save_state, load_state, read_state, checkpoint_path, write_npy, film, open_extra, restore_open!
export magnetization, count_vortices, band_currents, midplane_current, free_energy, dissipation, ghost_links, ghost_links_off!
export bracket_Ic, eta_bracket, efficiency, extract_Ic, symmetry_table
export material, load_config, run_config, summary_dict, start_state
export Protocol, Ladder, ladder, resume_ladder, ladder_done, run_ladders, brackets, delay_bracket, crossing_times, tui_status, tui_point, rung_rate, rung_time, eps_cycle, dt_release10

include("model.jl")
include("grid.jl")
include("geometry.jl")
include("shapes.jl")
include("state.jl")
include("terms.jl")
include("openbc.jl")
include("exchange.jl")
include("noise.jl")
include("materials.jl")
include("scheme.jl")
include("implicit.jl")
include("bands.jl")
include("field.jl")
include("step.jl")
include("io.jl")
include("observables.jl")
include("protocol.jl")
include("config.jl")

end # module
