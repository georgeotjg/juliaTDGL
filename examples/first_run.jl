# The first run of the README: a small notched box, both polarities, short
# windows; about a minute. Output in runs/first/ (summary, history, snapshots).
#   julia --project=julia examples/first_run.jl
using TDGL
sde = run_case(Config(GEOM_MODE="WEDGE", Nx_sc=24, Ny_sc=16, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5,
                      kappa_s=2.0, kappa_p=2.0, xi_ratio=1.3, H_const=0.5,
                      J_list=[0.05, 0.15], t_min_GL=100.0, t_min_GL_res=150.0,
                      out_dir="runs/first", run_tag="first"))
println("V(+J) = ", sde.pos.V_arr)
println("V(-J) = ", sde.neg.V_arr)
