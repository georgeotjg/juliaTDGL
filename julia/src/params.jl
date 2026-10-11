# ============================================================================
#  params.jl — DERIVED PARAMETERS (SECTIONS 3 and 4 of the .m)
#
#  dt is chosen here from the CFL condition and the scheme coefficients are
#  built.  It is plain code, and it is where M16 lives: `c2 = dt/Gamma_r`.
#  Up to v13.2 it scaled only the pi condensation term and moved the STATICS
#  of the pi band; since v13.3 it divides the whole pi equation and is a
#  true relaxation rate (see hc2_pi and hc2_pi_f21 below).
# ============================================================================

struct Params
    # mesh
    Nx::Int; Ny::Int; Nz::Int
    x1::Int; x2::Int; y1::Int; y2::Int; z1::Int; z2::Int
    dx::Float64; dy::Float64; dz::Float64
    # time
    dt::Float64; lMax::Int; mMax::Int
    # coefficients of the Psi update
    rx::Float64; ry::Float64; rz::Float64
    c1::Float64; c2::Float64; d1::Float64
    gamma12::Float64; mr::Float64; alpha2_n::Float64
    # Maxwell
    ks2::Float64
    sxy::Float64; sxz::Float64; syx::Float64; syz::Float64
    szx::Float64; szy::Float64
    # injection
    slab_d::Float64; inject_gain::Float64; inject_mode::String
    inject_standoff::String; h_tilt::Float64
    selffield::Bool; standoff_sample::Bool
    # convergence / voltmeter
    eps_conv::Float64; eps_seed::Float64; iter_min::Int
    t_min_GL::Float64; t_min_GL_res::Float64
    delta_v::Int; v_thr::Float64
    use_Vconv::Bool; V_win::Int; V_win_min_res::Int
    tol_V::Float64; tol_res::Float64; eps_relax::Float64
    floor_k::Float64; floor_mult::Float64
    # geometry / protocol
    current_axis::String; defect_model::String; hard::Bool; axis_y::Bool
    H_const::Float64; kappa_s::Float64
    # flags
    fix_M_window::Bool; explicit_h_bc::Bool; v13_diag::Bool
    dump_split::Bool; dump_band_lines::Bool
    early_exit::Bool; snap_every::Int
    vtrace_every::Int; vtrace_Jmin::Float64
    vtk_every::Int
    use_phi::Bool; phi_bc::Symbol; phi_method::Symbol; f22_supercurrent::Bool
    lambda_TF::Float64; tau_Q::Float64
    hilos::Bool
    guard_psi::Float64; guard_hours::Float64
    out_dir::String; run_tag::String; verbose::Bool
    # diagnostic of the dt choice
    dt_limitante::String
    xi_ratio_real::Float64
    use_drag::Bool
    nu_drag::Float64
    # Optional terms. When off they are not read in the step (the bit is kept).
    use_Qss::Bool
    D_Q::Float64
    use_heat::Bool
    D_T::Float64
    gamma_bath::Float64
    eta_J::Float64
    T_bath::Float64
    a2n_cfg::Float64
    beta2::Float64
    Temp::Array{Float64,3}
    # [1] = mean |∂tΨ|² of the last heat step. [2] = nodes at the cap.
    # [3] = mean |E|² (normal Joule) over the nodes with f>0.
    heat_diag::Vector{Float64}
    use_joule::Bool
    heat_margin::Int
    T_physical::Bool
    unwrap_every::Int
    keep_rung_states::Bool
    maxwell_implicit::Bool
    # pi substeps per step (1 = off) and their work array (empty when off)
    n_pi::Int
    n_sig::Int
    conv_by_time::Bool
    blas_threads::Int
    Wpi::Array{ComplexF64,3}             # band substep work array
    pi_extrap::Bool
    Lpi::Vector{Array{ComplexF64,3}}     # substep links and rotations, allocated on first use
    # charge residual read from a checkpoint, copied into the Phi solver when a
    # sweep starts (empty unless use_phi and tau_Q > 0)
    Q_restart::Array{Float64,3}
end

"""
    derive_params(cfg) -> Params

SECTION 3 + SECTION 4 of the .m.  Mutates `cfg.m_ratio` if `cfg.xi_ratio > 0`,
as the original does (it reassigns and prints it).
"""
function derive_params(cfg::Config)
    if cfg.use_Qss && !cfg.use_phi
        error("use_Qss needs use_phi=true: Q lives in the Phi step")
    end
    if cfg.use_Qss && cfg.tau_Q <= 0
        error("use_Qss needs tau_Q>0")
    end
    if cfg.use_heat && cfg.D_T < 0
        error("D_T must be >= 0")
    end
    if cfg.use_joule && !cfg.use_heat
        error("use_joule needs use_heat=true")
    end
    if cfg.use_joule && cfg.use_phi
        error("use_joule needs use_phi=false: with Φ, E carries −∇Φ and that is not verified")
    end
    bx = cfg.bufx < 0 ? cfg.buf : cfg.bufx
    by = cfg.bufy < 0 ? cfg.buf : cfg.bufy
    bz = cfg.bufz < 0 ? cfg.buf : cfg.bufz
    Nx = cfg.Nx_sc + 2bx; Ny = cfg.Ny_sc + 2by; Nz = cfg.Nz_sc + 2bz
    x1 = bx + 1; x2 = bx + cfg.Nx_sc
    y1 = by + 1; y2 = by + cfg.Ny_sc
    z1 = bz + 1; z2 = bz + cfg.Nz_sc

    alpha1   = -(1.00 - cfg.T_red)
    alpha2   = -(cfg.a2n - cfg.T_red)
    alpha2_n = abs(alpha2) / cfg.beta2

    if cfg.xi_ratio > 0
        mr_old = cfg.m_ratio
        # T_physical: ξπ/ξσ is fixed at T = 0 (the mass does not change with T)
        a2_ref = cfg.T_physical ? cfg.a2n / cfg.beta2 : alpha2_n
        cfg.m_ratio = 1 / (cfg.xi_ratio^2 * a2_ref)
        cfg.verbose && @printf("[xi_ratio] set xi_pi/xi_sigma = %.2f  =>  m_ratio %.4f -> %.4f\n",
                               cfg.xi_ratio, mr_old, cfg.m_ratio)
    end
    xi_ratio_real = 1 / sqrt(cfg.m_ratio * alpha2_n)
    cfg.verbose && @printf("BANDS: m_ratio=%.4f a2n=%.3f  =>  xi_pi/xi_sigma = %.3f\n",
                           cfg.m_ratio, alpha2_n, xi_ratio_real)

    # --- CFL ---------------------------------------------------------------
    # `gama` is a remnant of a generalised TDGL that is not implemented; it only
    # scales dt1 by 1/sqrt(1+gama^2). It also acts as the safety factor of the
    # explicit diffusion step: dt1 = 0.144*dl, against a stability bound dl/4
    # (dl/4 * Gamma_r*m_r for pi). In the MgB2 runs dt1p binds at kappa=2 and
    # dt2s at kappa=25; with the defaults (m_r=1.5) dt3 binds at kappa=2.
    # Kept as is so that dt does not change in published runs.
    eta_cfl = 5.79; bt_cfl = 1.00; nu_cfl = 0.03; zeta_cfl = 0.50; gama = 10.0
    dl   = 2.0 / (1/cfg.dx^2 + 1/cfg.dy^2 + 1/cfg.dz^2)
    dt1  = eta_cfl * dl / (4 * sqrt(1 + gama^2))
    gr_  = cfg.gamma_ratio
    # the pi branch of the CFL scales with m_r AND with Gamma_r.
    # Up to v13.2 the update did not apply it and the guard was detached;
    # since v13.3 it matches.
    dt1p = dt1 * cfg.m_ratio * gr_
    dt2s = bt_cfl * dl / (4 * cfg.kappa_s^2)
    dt2p = bt_cfl * dl / (4 * cfg.kappa_p^2)
    dt3  = nu_cfl * dl / (4 * zeta_cfl^2)
    dt_leg = 0.99 * min(dt1, dt1p, dt2s, dt2p, dt3)    # the 1.0.0 step, for eps_conv_legacy
    if !cfg.cfl_legacy
        # Explicit Euler is stable when dt*lambda_max <= 2, with lambda_max the largest
        # eigenvalue of the linearised right-hand side: the lattice Laplacian (8/dl)
        # plus the reaction term of the amplitude mode, 3|Psi|^2 - alpha <= 2 alpha in
        # equilibrium (alpha = 1 for sigma, a2n for pi).  dl/4 alone is the bound of
        # pure diffusion and is a few per cent too large.
        dt1  = 2 / (8 / dl + 2)                                      # sigma
        dt1p = 2 * gr_ / (8 / (dl * cfg.m_ratio) + 2 * alpha2_n)     # pi: (1/Gamma_r)[(1/m_r) lap + ...]
        dt3  = Inf
    end
    if cfg.maxwell_implicit
        cfg.use_phi && error("maxwell_implicit with use_phi is not implemented")
        dt2s = Inf; dt2p = Inf             # the implicit curl-curl has no step bound
    end
    cfg.blas_threads >= 0 || error("blas_threads must be >= 0, not $(cfg.blas_threads)")
    cfg.pi_substeps >= 0 || error("pi_substeps must be >= 0, not $(cfg.pi_substeps)")
    cfg.sigma_substeps >= 1 || error("sigma_substeps must be >= 1, not $(cfg.sigma_substeps)")
    n_sig = cfg.sigma_substeps
    if n_sig > 1
        (cfg.use_heat || cfg.T_physical || cfg.use_phi || (cfg.use_drag && cfg.nu_drag != 0.0)) &&
            error("sigma_substeps > 1 is implemented only for the plain kernel (no heat, T_physical, Phi or drag)")
        dt1 *= n_sig
    end
    n_pi = 1
    terms = cfg.use_heat || cfg.T_physical || cfg.use_phi || (cfg.use_drag && cfg.nu_drag != 0.0)
    # automatic substeps (pi_substeps = 0, as in :prod) fall back to one substep when a term is on:
    # the implicit curl-curl alone is validated with heat and drag (CHANGELOG 1.1)
    if cfg.pi_substeps == 0 && terms
        cfg.verbose && println("pi band: no substeps with heat, T_physical, Phi or drag (one per step)")
    elseif cfg.pi_substeps != 1
        terms &&
            error("pi_substeps > 1 is implemented only for the plain kernel (no heat, T_physical, Phi or drag)")
        # automatic: as many substeps as the step the other bounds give, after dt_scale
        n_pi = cfg.pi_substeps == 0 ?
               max(1, ceil(Int, cfg.dt_scale * min(dt1, dt2s, dt2p, dt3) / dt1p - 1e-12)) : cfg.pi_substeps
        dt1p *= n_pi / cfg.dt_scale        # each substep keeps dt/n_pi <= 0.99 of the pi bound
    end
    cands = (dt1, dt1p, dt2s, dt2p, dt3)
    dt = 0.99 * minimum(cands)
    0 < cfg.dt_scale <= 1 || error("dt_scale must be in (0, 1], not $(cfg.dt_scale)")
    cfg.dt_scale == 1 || (dt *= cfg.dt_scale)
    # Explicit diffusion of Q or T. Only if the flag is ON: otherwise dt is
    # unchanged and the bit does not move.
    if cfg.use_heat || cfg.use_Qss
        Ddif = 0.0
        cfg.use_heat && (Ddif = max(Ddif, cfg.D_T))
        cfg.use_Qss  && (Ddif = max(Ddif, cfg.D_Q))
        if Ddif > 0
            dl2 = min(cfg.dx, cfg.dy, cfg.dz)^2
            dt_dif = 0.15 * dl2 / Ddif
            dt_dif < dt && (dt = dt_dif)
        end
    end
    nombres = ("dt1 (sigma diffusion)", "dt1p (pi diffusion, m_r<1)",
               "dt2s (Maxwell kappa_s)", "dt2p (Maxwell kappa_p)", "dt3")
    ilim = argmin(collect(cands))
    cfg.verbose && @printf("CFL: dt=%.3e t_GL (limited by %s)\n", dt, nombres[ilim])
    cfg.verbose && n_pi > 1 && @printf("     pi band: %d substeps of %.3e\n", n_pi, dt / n_pi)
    # The supercurrent enters block [C] explicitly.  Linearised in the link
    # phase, its longitudinal mode (which the implicit curl-curl does not touch)
    # decays at |Psi_s|^2 + |Psi_p|^2/m_r: explicit Euler needs dt below 2 over
    # that, ~0.3 at m_r = 0.106.  No bound used today reaches it.
    lam_L = abs(alpha1) + alpha2_n / cfg.m_ratio
    dt > 2 / lam_L && @warn @sprintf("dt = %.3g exceeds the explicit supercurrent bound 2/(|a1| + a2n/m_r) = %.3g", dt, 2 / lam_L)
    # Stable is not accurate.  Above half its bound the stiffest sigma mode changes
    # sign every step and barely decays (factor 1 - dt*lambda_max -> -0.98 at 0.99 of
    # the bound); in strong flow at kappa = 25 that gives V 7 % low.  At or below
    # half the bound no mode changes sign.
    cfg.verbose && !cfg.cfl_legacy && dt > 0.5 * dt1 &&
        @warn @sprintf("dt = %.3g is %.0f %% of the sigma stability bound; use dt_scale <= 0.5 for accurate flowing states",
                       dt, 100 * dt / dt1)

    rx = dt/cfg.dx^2; ry = dt/cfg.dy^2; rz = dt/cfg.dz^2
    c1 = dt
    # c2 = dt/Gamma_r.  In THIS scheme (v13.3) Gamma_r divides the
    # WHOLE pi-band equation (kinetic, condensation and coupling), so it is a
    # relaxation rate and does NOT move xi_pi or Hc2_pi.  Up to v13.2 it
    # scaled only the condensation term and then xi_pi/xi_sigma =
    # 1.300*sqrt(Gamma_r).  See hc2_pi vs hc2_pi_f21.
    c2 = dt / gr_
    # `d1` multiplies the supercurrent in block [C].  |alpha1| = (1-T)
    # should NOT be there: the equation is d_t A = J_s - kappa^2 rot rot A.
    # At T = 0 it equals 1 and both branches agree to the bit.
    d1 = (cfg.f22_supercurrent && !cfg.T_physical) ? dt * abs(alpha1) : dt
    ks2 = cfg.kappa_s^2
    sxy = (dt*ks2)*(cfg.dx/cfg.dy); sxz = (dt*ks2)*(cfg.dx/cfg.dz)
    syx = (dt*ks2)*(cfg.dy/cfg.dx); syz = (dt*ks2)*(cfg.dy/cfg.dz)
    szx = (dt*ks2)*(cfg.dz/cfg.dx); szy = (dt*ks2)*(cfg.dz/cfg.dy)

    cfg.eps_conv_rate >= 0 || error("eps_conv_rate must be >= 0, not $(cfg.eps_conv_rate)")
    # Fixed-point tolerance per cycle: eps_conv, or a rate times the cycle length
    # (eps_conv_legacy: the rate of eps_conv per cycle of the 1.0.0 step).
    eps_c = cfg.eps_conv_rate > 0 ? cfg.eps_conv_rate * (cfg.lMax * dt) :
            cfg.eps_conv_legacy  ? cfg.eps_conv * (dt / dt_leg) : cfg.eps_conv
    eps_seed = cfg.tie_seed_tolerance ? min(cfg.eps_conv_seed, eps_c) :
                                        cfg.eps_conv_seed

    cfg.phi_bc in (:dirichlet, :neumann) ||
        error("unknown phi_bc: $(cfg.phi_bc) (use :dirichlet or :neumann)")
    cfg.phi_method in (:separable, :lu) ||
        error("unknown phi_method: $(cfg.phi_method) (use :separable or :lu)")
    cfg.vtk_every >= 0 || error("vtk_every must be >= 0, not $(cfg.vtk_every)")
    cfg.lambda_TF >= 0 || error("lambda_TF must be >= 0, not $(cfg.lambda_TF)")
    cfg.tau_Q >= 0 || error("tau_Q must be >= 0, not $(cfg.tau_Q)")
    if (cfg.lambda_TF > 0 || cfg.tau_Q > 0) && !cfg.use_phi
        error("lambda_TF/tau_Q need use_phi=true (otherwise there is no Phi)")
    end
    if cfg.use_phi && cfg.T_red != 0 && cfg.f22_supercurrent && cfg.verbose
        println("WARNING [F22]: use_phi=true at T!=0 with f22_supercurrent=true. ",
                "The correct equation has no (1-T). Set f22_supercurrent=false.")
    end
    if cfg.tau_Q > 0 && cfg.lambda_TF == 0 && cfg.verbose
        println("WARNING: tau_Q>0 with lambda_TF=0. Under exact Coulomb ",
                "D(delta')=0 and the source of Q vanishes: the imbalance does ",
                "nothing. Turn on lambda_TF to give the charge a scale.")

    end

    Params(Nx, Ny, Nz, x1, x2, y1, y2, z1, z2,
           cfg.dx, cfg.dy, cfg.dz,
           dt, cfg.lMax, cfg.mMax,
           rx, ry, rz, c1, c2, d1,
           cfg.gamma12, cfg.m_ratio, alpha2_n,
           ks2, sxy, sxz, syx, syz, szx, szy,
           (z2 - z1 + 1) * cfg.dz, cfg.inject_gain, cfg.inject_mode,
           cfg.inject_standoff, cfg.h_tilt,
           lowercase(cfg.inject_mode) == "selffield",
           lowercase(cfg.inject_standoff) == "sample",
           eps_c, eps_seed, cfg.iter_min,
           cfg.t_min_GL, cfg.t_min_GL_res,
           cfg.delta_v, cfg.V_threshold,
           cfg.use_Vconv, cfg.V_win, cfg.V_win_min_res,
           cfg.tol_V, cfg.tol_res, cfg.eps_relax,
           cfg.floor_k, cfg.floor_mult,
           cfg.current_axis, cfg.defect_model,
           lowercase(cfg.defect_model) == "hard",
           lowercase(cfg.current_axis) == "y",
           cfg.H_const, cfg.kappa_s,
           cfg.fix_M_window, cfg.explicit_h_bc, cfg.v13_diag,
           cfg.dump_split, cfg.dump_band_lines,
           cfg.early_exit, cfg.snap_every,
           cfg.vtrace_every, cfg.vtrace_Jmin, cfg.vtk_every,
           cfg.use_phi, cfg.phi_bc, cfg.phi_method, cfg.f22_supercurrent,
           cfg.lambda_TF, cfg.tau_Q,
           cfg.use_threads && Threads.nthreads() > 1,
           cfg.guard_psi, cfg.guard_hours,
           cfg.out_dir, cfg.run_tag, cfg.verbose,
           nombres[ilim], xi_ratio_real,
           cfg.use_drag, cfg.nu_drag,
           cfg.use_Qss, cfg.D_Q,
           cfg.use_heat, cfg.D_T, cfg.gamma_bath, cfg.eta_J,
           cfg.T_red, cfg.a2n, cfg.beta2,
           fill(cfg.T_red, Nx+1, Ny+1, Nz+1),
           zeros(3), cfg.use_joule, cfg.heat_margin, cfg.T_physical,
           cfg.unwrap_every, cfg.keep_rung_states, cfg.maxwell_implicit,
           n_pi, n_sig, cfg.conv_by_time, cfg.blas_threads, (n_pi > 1 || n_sig > 1) ? zeros(ComplexF64, Nx+1, Ny+1, Nz+1) : zeros(ComplexF64, 0, 0, 0),
           cfg.pi_link_extrapolation, Array{ComplexF64,3}[],
           (cfg.use_phi && cfg.tau_Q > 0) ? zeros(Nx+1, Ny+1, Nz+1) : zeros(0, 0, 0))
end

"""
    current_grid(cfg) -> Vector{Float64}

SECTION 5.  Explicit `J_list` (J=0 always included, which gives the
baseline V(J=0)) or the historical fine/coarse grid.
"""
function current_grid(cfg::Config)
    if !isempty(cfg.J_list)
        Jp = sort(unique(vcat(0.0, abs.(cfg.J_list))))
        cfg.verbose && @printf("[J_list] explicit grid: %d points, J_max=%.3f\n",
                               length(Jp), maximum(Jp))
        return Jp
    end
    vcat(collect(0:cfg.dJ_fine:cfg.J_tr),
         collect(cfg.J_tr+cfg.dJ_coarse:cfg.dJ_coarse:cfg.J_max))
end

"""
    hc2_pi(p) -> Float64
    xi_ratio_integrated(p) -> Float64

The upper critical field and the coherence length of the pi band AS THIS
KERNEL INTEGRATES THEM:

    Hc2_pi = m_r * a2n              xi_pi/xi_sigma = 1/sqrt(m_r*a2n)

**Neither depends on Gamma_r**, and that independence is a PROPERTY OF THE
v13.3 SCHEME, not a coincidence: there Gamma_r divides the three right-hand
terms of the pi band, so it is a true relaxation rate and cannot move a
fixed point.  `test_params.jl` checks it by integrating the same state at
Gamma_r = 0.25, 1 and 4 and requiring <|Psi_pi|^2> to agree.

TO READ THE ARCHIVE PRIOR TO v13.3, see `hc2_pi_f21`.
"""
hc2_pi(p::Params) = p.mr * p.alpha2_n
xi_ratio_integrated(p::Params) = 1 / sqrt(p.mr * p.alpha2_n)

"""
    hc2_pi_f21(p) -> Float64
    xi_ratio_f21(p) -> Float64

The same quantities under the v13.2 and earlier scheme, where Gamma_r
scaled ONLY the condensation term:

    Hc2_pi = m_r*a2n/Gamma_r        xi_pi/xi_sigma = 1.300*sqrt(Gamma_r)

The solver does not use them.  They are here because **any archived run
prior to v13.3 that swept `gamma_ratio` must be read with these**: that
sweep did not move a relaxation rate, it moved the coherence length of the
pi band.
"""
hc2_pi_f21(p::Params)     = p.mr * p.alpha2_n * (p.c2 / p.dt)
xi_ratio_f21(p::Params)   = sqrt((p.dt / p.c2) / (p.mr * p.alpha2_n))
