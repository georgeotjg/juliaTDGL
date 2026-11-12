# Golden ladder for the protocol tests of 2.0: a full two-polarity run of juliaTDGL 1.2 (run_case,
# Config(:prod), Fast mode) on a small notched box at kappa = 25, with short windows.
#
#   OPENBLAS_NUM_THREADS=1 julia --project=julia tdgl2/test/golden/make_ladder.jl
using TDGL, JLD2, LinearAlgebra
BLAS.set_num_threads(1)
cfg = Config(:prod; GEOM_MODE="WEDGE", Nx_sc=24, Ny_sc=12, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5, buf=1,
             kappa_s=25.0, kappa_p=25.0, xi_ratio=4.0, gamma_ratio=1.0, gamma12=1e-3, H_const=0.03,
             J_list=[0.30, 0.60, 0.90, 1.20, 1.50], t_min_GL=20.0, t_min_GL_res=60.0, mMax=4000, delta_v=6,
             inj_margin=4, selftest=false, verbose=false, use_threads=false, out_dir="")
sde = run_case(cfg; mode=Fast())
p = sde.p
lad(r) = Dict("J" => r.J_arr, "V" => r.V_arr, "V2" => r.V2_arr, "Vh" => r.Vh_arr, "Vh2" => r.Vh2_arr,
              "Vst" => r.Vst_arr, "m" => r.m_arr, "conv" => String(r.cflag), "nslip" => r.nslip_arr,
              "M" => r.M_arr, "Nv1" => r.Nv1_arr, "Nv2" => r.Nv2_arr, "r1" => r.r1_arr, "r2" => r.r2_arr,
              "G1" => r.estado.G1, "G2" => r.estado.G2)
b = brackets(sde)
jldsave(joinpath(@__DIR__, "ladder_k25.jld2"), true;
        pos=lad(sde.pos), neg=lad(sde.neg), eps_c=p.eps_conv, dt=p.dt, n_pi=p.n_pi, mr=p.mr,
        lo_pos=b.pos.lo, hi_pos=b.pos.hi, lo_neg=b.neg.lo, hi_neg=b.neg.hi)
println("J+ ", sde.pos.J_arr, " ", String(sde.pos.cflag), " ", sde.pos.V_arr)
println("J- ", sde.neg.J_arr, " ", String(sde.neg.cflag), " ", sde.neg.V_arr)
println("brackets ", (b.pos.lo, b.pos.hi), " ", (b.neg.lo, b.neg.hi), " eta ", b.eta)
