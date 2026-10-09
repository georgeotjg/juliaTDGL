# The run behind the figures of the README: the same notched box, J from 0.02
# to 0.24, windows of 300/400 t_GL, about ten minutes. Then
#   python3 julia/scripts/plot_summary.py runs/demo
#   julia --project=julia examples/notched_box_demo.jl
using TDGL
run_case(Config(GEOM_MODE="WEDGE", Nx_sc=24, Ny_sc=16, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5,
                kappa_s=2.0, kappa_p=2.0, xi_ratio=1.3, H_const=0.5,
                J_list=collect(0.02:0.02:0.24), t_min_GL=300.0, t_min_GL_res=400.0,
                out_dir="runs/demo", run_tag="demo"); mode=Fast())
