# Golden states for the regression tests of 2.0, written by juliaTDGL 1.2 (julia/) in Fast mode.
#
#   OPENBLAS_NUM_THREADS=1 julia --project=julia tdgl2/test/golden/make_golden.jl
#
# Each case: the 1.2 run is advanced N0 steps from its initial state (so that the contact layers
# rotate the links and the band substeps extrapolate them), the full State is saved ("start"),
# N1 more steps are taken and the State is saved again ("end"). 2.0 loads "start", takes N1 steps
# and must reproduce "end" to the bit. The masks and the step 1.2 chose are saved with them.
using TDGL, JLD2, LinearAlgebra
BLAS.set_num_threads(1)

const OUT = @__DIR__
const N0 = 200
const N1 = 40

base = (GEOM_MODE="WEDGE", Nx_sc=16, Ny_sc=10, Nz_sc=4, dx=0.5, dy=0.5, dz=0.5, buf=1,
        gamma12=1e-3, T_red=0.0, J_list=[0.0], selftest=false, verbose=false,
        use_threads=false, out_dir="")

casos = [
    # explicit scheme with the stability bounds only (no implicit curl-curl, no substeps)
    ("k2_R13_explicit", Config(:legacy; base..., kappa_s=2.0, kappa_p=2.0, xi_ratio=1.30, gamma_ratio=1.3,
                               H_const=0.50, cfl_legacy=false, maxwell_implicit=false, pi_substeps=1), 0.30),
    # production scheme, light pi band with substeps, Gamma_r != 1
    ("k2_R4_prod", Config(:prod; base..., kappa_s=2.0, kappa_p=2.0, xi_ratio=4.0, gamma_ratio=0.7,
                          H_const=0.03), 0.50),
    # production scheme at kappa = 25
    ("k25_R4_prod", Config(:prod; base..., kappa_s=25.0, kappa_p=25.0, xi_ratio=4.0, gamma_ratio=1.0,
                           H_const=0.03), 0.40),
    # heat with Joule and a contact margin (1.2 takes one pi substep with terms on)
    ("k2_R4_heat", Config(:prod; base..., kappa_s=2.0, kappa_p=2.0, xi_ratio=4.0, gamma_ratio=1.0,
                          H_const=0.03, use_heat=true, D_T=0.25, gamma_bath=0.2, eta_J=0.5,
                          use_joule=true, heat_margin=3), 0.60),
    # interband drag
    ("k2_R4_drag", Config(:prod; base..., kappa_s=2.0, kappa_p=2.0, xi_ratio=4.0, gamma_ratio=1.3,
                          H_const=0.03, use_drag=true, nu_drag=0.3), 0.50),
]

campos(s) = Dict("G1" => copy(s.G1), "G2" => copy(s.G2), "F1" => copy(s.F1), "F2" => copy(s.F2),
                 "thetax" => copy(s.thetax), "thetay" => copy(s.thetay), "thetaz" => copy(s.thetaz),
                 "phix" => copy(s.phix), "phiy" => copy(s.phiy), "phiz" => copy(s.phiz),
                 "Ux" => copy(s.Ux), "Uy" => copy(s.Uy), "Uz" => copy(s.Uz),
                 "Vx" => copy(s.Vx), "Vy" => copy(s.Vy), "Vz" => copy(s.Vz),
                 "hx" => copy(s.hx), "hy" => copy(s.hy), "hz" => copy(s.hz))

for (nombre, cfg, J) in casos
    p = derive_params(cfg)
    g = build_geometry(cfg, p)
    mode = Fast()
    H = cfg.H_const
    s = initial_state(p, g, H, mode)
    mi = p.maxwell_implicit ? MaxwellImplicit(p, g) : nothing
    HJ = p.inject_gain * J * p.slab_d / (2 * p.ks2)
    for _ in 1:N0
        substep!(s, p, g, mode; H=H, HJ=HJ, J_cur=J, mi=mi)
    end
    ini = campos(s)
    p.use_heat && (ini["Temp"] = copy(p.Temp))
    for _ in 1:N1
        substep!(s, p, g, mode; H=H, HJ=HJ, J_cur=J, mi=mi)
    end
    fin = campos(s)
    p.use_heat && (fin["Temp"] = copy(p.Temp))
    par = Dict("dt" => p.dt, "n_pi" => p.n_pi, "Nx_sc" => cfg.Nx_sc, "Ny_sc" => cfg.Ny_sc,
               "Nz_sc" => cfg.Nz_sc, "buf" => cfg.buf, "dx" => cfg.dx, "kappa" => cfg.kappa_s,
               "m_ratio" => p.mr, "a2n" => cfg.a2n, "beta2" => cfg.beta2, "gamma12" => cfg.gamma12,
               "gamma_ratio" => cfg.gamma_ratio, "H" => H, "J" => J, "HJ" => HJ,
               "implicit" => p.maxwell_implicit, "pi_substeps_cfg" => cfg.pi_substeps,
               "dt_scale" => cfg.dt_scale, "extrap" => p.pi_extrap, "N1" => N1,
               "limit" => p.dt_limitante, "code" => "juliaTDGL 1.2.0, Fast mode",
               "heat" => p.use_heat, "D_T" => cfg.D_T, "gamma_bath" => cfg.gamma_bath, "eta_J" => cfg.eta_J,
               "joule" => cfg.use_joule, "heat_margin" => cfg.heat_margin,
               "drag" => cfg.use_drag ? cfg.nu_drag : 0.0)
    jldsave(joinpath(OUT, "$nombre.jld2"); par, ini, fin,
            fmask=g.fmask, taux=g.taux, tauy=g.tauy, tauz=g.tauz, taupsi=g.taupsi)
    println(rpad(nombre, 18), " dt = ", p.dt, "  n_pi = ", p.n_pi, "  (", p.dt_limitante, ")")
end
