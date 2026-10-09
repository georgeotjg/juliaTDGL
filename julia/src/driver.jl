# ============================================================================
#  driver.jl — THE FULL RUN: geometry, preflight, two polarities, Ic
#              extraction and summary.
#
#  The project protocol, which this file implements and does not invent:
#  an eta is reported only after (1) a clean injection margin, (2) the `_REF`
#  control, which must INVERT the sign while keeping |Ic|, and (3) robustness
#  against x_shift / y_shift.  The summary repeats this on every run, next to
#  the number, because a bare eta gets quoted wrongly.
# ============================================================================

"""
    SDE

What a full run returns: both polarities, the Ic, eta and everything needed
to redo the figures without simulating again.
"""
struct SDE
    cfg::Config
    p::Params
    geom::Geometry
    pos::Union{SweepResult,Nothing}
    neg::Union{SweepResult,Nothing}
    Ic_plus::Float64
    Ic_minus::Float64
    eta_pct::Float64
    simetria::NamedTuple
    wall_min::Float64
    mode::NumMode
end

"""
    run_case(cfg; mode=Strict()) -> SDE

Full pipeline of ONE run.

`mode` selects the rounding (see compat.jl):
  * `Strict()` uses libm hypot/sincos; the mode of the regression tests.
  * `Fast()`   is the same scheme with Julia's native arithmetic, faster.
"""
function run_case(cfg::Config; mode::NumMode=Strict())
    t0 = time()
    p = derive_params(cfg)
    if cfg.verbose
        println("\n=== TDGL two-band solver ===")
        @printf("SC mesh %dx%dx%d | dx=%.3f xi | J||%s | H=%.2f Hc2 | g12=%.3f | %s/%s\n",
                cfg.Nx_sc, cfg.Ny_sc, cfg.Nz_sc, cfg.dx, cfg.current_axis,
                cfg.H_const, cfg.gamma12, cfg.GEOM_MODE, cfg.defect_model)
        println("Numerical mode: ", mode_name(mode))
    end

    g = build_geometry(cfg, p)
    if cfg.verbose
        frac, shunt = check_f_shunt(g, p)
        @printf("[MASK f] min=%.3f  max=%.3f  |  f<0.5 on %.1f %% of the block\n",
                minimum(view(g.fmask, p.x1:p.x2, p.y1:p.y2, p.z1:p.z2)),
                maximum(view(g.fmask, p.x1:p.x2, p.y1:p.y2, p.z1:p.z2)), 100frac)
        shunt && println("  WARNING [f-shunt]: there is a continuous path of normal material " *
                         "along the current axis. Any Ic will be spurious.")
        nasym = check_my_symmetry(g, p)
        nrx   = check_rz_symmetry(g, p)
        nmx   = check_mx_symmetry(g, p)
        @printf("Geometry: %d defect(s). Interior asymmetry: My=%d Rz=%d Mx=%d cells.\n",
                length(g.defects), nasym, nrx, nmx)
        flush(stdout)
    end

    if cfg.preflight
        _preflight(cfg, p)
        return SDE(cfg, p, g, nothing, nothing, NaN, NaN, NaN,
                   symmetry_table(Float64[], Float64[], Float64[], Float64[]),
                   0.0, mode)
    end

    cfg.selftest && run_selftest(g, p, mode)
    _preparar_salidas(cfg)

    if uppercase(cfg.run_mode) == "MH"
        return _run_MH(cfg, p, g, mode, t0)
    end

    J_pos = current_grid(cfg)
    J_neg = -J_pos
    Hc = fill(cfg.H_const, length(J_pos))

    est_p, jpp, Va_p, Ja_p = _quiza_restart(cfg, p, mode, "Jpos", cfg.restart_pos)
    res_p = sweep(J_pos, Hc, "J+", g, p, mode;
                    estado=est_p, jp_start=jpp, J_arr0=Ja_p, V_arr0=Va_p)

    est_n, jpn, Va_n, Ja_n = _quiza_restart(cfg, p, mode, "Jneg", cfg.restart_neg)
    # θ lives in Params, not in the checkpoint. Without this the − sign
    # inherits the heat of the + sign (fixed 22-sep, 4e71482).
    if p.use_heat && est_n === nothing
        fill!(p.Temp, p.T_bath)
    end
    res_n = sweep(J_neg, Hc, "J-", g, p, mode;
                    estado=est_n, jp_start=jpn, J_arr0=Ja_n, V_arr0=Va_n)

    Icp = extract_Ic(res_p.J_arr, res_p.V_arr, cfg.V_threshold; strict=cfg.strict_two_rung)
    Icn = extract_Ic(abs.(res_n.J_arr), res_n.V_arr, cfg.V_threshold; strict=cfg.strict_two_rung)
    Nv_total = sum(res_p.Nv1_arr) + sum(res_p.Nv2_arr) + sum(res_n.Nv1_arr) + sum(res_n.Nv2_arr)
    gd = ic_guards(res_p.J_arr, res_p.V_arr, res_n.J_arr, res_n.V_arr, Nv_total, Icp, Icn)
    if cfg.verbose
        gd.inert && @printf("\n[parity] V is even in J: odd part %.3e against even part %.3e.\n%s\n",
            gd.V_odd_max, gd.V_even_max,
            "  The current is not entering the dynamics (mode M1): Ic and eta are not reported.")
        gd.no_vortex && println("\n[Nv] no vortex at any point, in either band: there is no diode ",
            "mechanism and eta = 0 is the answer; Ic and eta are not reported.")
        gd.grid && @printf("\n[grid] Ic+ = %.5f, Ic- = %.5f at or below the first current of the grid (%.4f):\n%s\n",
            Icp, Icn, gd.J_first, "  the grid does not resolve Ic (mode M9). Start the grid lower; Ic and eta are not reported.")
    end
    Icp, Icn = gd.Icp, gd.Icn
    eta = efficiency(Icp, Icn)
    sym = symmetry_table(res_p.J_arr, res_p.V_arr, abs.(res_n.J_arr), res_n.V_arr)

    sde = SDE(cfg, p, g, res_p, res_n, Icp, Icn, eta, sym, (time()-t0)/60, mode)
    cfg.verbose && _resultado(sde)
    isempty(cfg.out_dir) || write_summary(sde)
    sde
end

function _run_MH(cfg::Config, p::Params, g::Geometry, mode::NumMode, t0)
    H_range = cfg.MH_updown ? vcat(cfg.H_list, reverse(cfg.H_list)[2:end]) :
                              collect(cfg.H_list)
    J_range = zeros(length(H_range))
    res = sweep(J_range, H_range, "MH", g, p, mode)
    sde = SDE(cfg, p, g, res, nothing, NaN, NaN, NaN,
              symmetry_table(Float64[], Float64[], Float64[], Float64[]),
              (time()-t0)/60, mode)
    if cfg.verbose
        println("\n=== M(H): Meissner branch, first jump = EFFECTIVE Hc1 of THIS sample ===")
        println("  (not the bulk Hc1: in a slab ~10 xi wide the surface barrier shifts it")
        println("   considerably)")
    end
    isempty(cfg.out_dir) || write_summary(sde)
    sde
end

function _quiza_restart(cfg::Config, p::Params, mode::NumMode, ckl::String, activo::Bool)
    activo || return (nothing, 1, nothing, nothing)
    ruta = joinpath(cfg.out_dir, "restart", "state_$(ckl).mat")
    isfile(ruta) || (cfg.verbose && println("[restart] $ruta does not exist; starting from scratch");
                     return (nothing, 1, nothing, nothing))
    s, jp, V, J = load_checkpoint(ruta, p, mode)
    cfg.verbose && println("[restart] $ruta resumed at jp=$(jp+1)")
    (s, jp+1, V, J)
end

function _preparar_salidas(cfg::Config)
    isempty(cfg.out_dir) && return
    for d in ("output", "restart", "figures")
        mkpath(joinpath(cfg.out_dir, d))
    end
end

# ---------------------------------------------------------------------------
#  PREFLIGHT — cost estimate without simulating
# ---------------------------------------------------------------------------
"""
    _preflight(cfg, p)

Builds the geometry and prints what the run will cost BEFORE launching it.
The number that matters is not "how many cycles" but how much PHYSICAL TIME
is integrated: `t = mMax * lMax * dt` in units of t_GL.  A sweep with a high
mMax and a small dt can fall short in physical time and give a voltage that
looks converged and is not.
"""
function _preflight(cfg::Config, p::Params)
    dt_cyc = p.lMax*p.dt
    npts = length(current_grid(cfg))
    it_min = max(p.iter_min, p.t_min_GL > 0 ? ceil(Int, p.t_min_GL/dt_cyc) : 0)
    n = (p.Nx+1)*(p.Ny+1)*(p.Nz+1)
    seg_ciclo = cfg.sec_per_step*p.lMax*n/(121*41*11)
    @printf("\n[PREFLIGHT]\n")
    @printf("  nodes                     : %d  (%dx%dx%d)\n", n, p.Nx+1, p.Ny+1, p.Nz+1)
    @printf("  dt                        : %.4e t_GL  (limited by %s)\n", p.dt, p.dt_limitante)
    @printf("  physical time per cycle   : %.4f t_GL\n", dt_cyc)
    @printf("  EFFECTIVE floor in cycles : %d  (t_min_GL=%.1f)\n", it_min, p.t_min_GL)
    @printf("  ceiling mMax              : %d cycles = %.1f t_GL\n", p.mMax, p.mMax*dt_cyc)
    @printf("  fixed-point tolerance     : %.2e per cycle = %.2e per t_GL%s\n",
            p.eps_conv, p.eps_conv/dt_cyc, cfg.eps_conv_rate > 0 ? " (eps_conv_rate)" : "")
    floor_GL = max(p.t_min_GL, cfg.use_Vconv ? p.t_min_GL_res : 0.0)
    if p.mMax*dt_cyc < floor_GL
        @printf("  WARNING: the ceiling (%.1f t_GL) is below the floor (%.1f t_GL): every point\n",
                p.mMax*dt_cyc, floor_GL)
        println("           will close at mMax, not at the floor. Raise mMax.")
    end
    @printf("  points x polarities       : %d x 2\n", npts)
    @printf("  => total range            : %.2f h (all at the floor) ... %.2f h (all at mMax)\n",
            2npts*it_min*seg_ciclo/3600, 2npts*p.mMax*seg_ciclo/3600)
    @printf("  integrated xi_pi/xi_sigma : %.4f    Hc2_pi = %.4f Hc2_sigma\n",
            xi_ratio_integrated(p), hc2_pi(p))
    @printf("  (with Gamma_r in the condensation only (mode M16) they would be %.4f and %.4f:\n",
            xi_ratio_f21(p), hc2_pi_f21(p))
    println("   the values of runs made with that scheme and gamma_ratio != 1)")
end

# ---------------------------------------------------------------------------
#  OUTPUT
# ---------------------------------------------------------------------------
"""
    brackets(sde) -> (pos, neg, eta)

Both critical currents as brackets with the verdict rule of the paper (see
`bracket_Ic`), and the efficiency at the bracket centres with its range.
This is the estimate to quote; the threshold Ic of `extract_Ic` is kept for
comparison with the reference code.
"""
function brackets(sde::SDE)
    (sde.pos === nothing || sde.neg === nothing) && return nothing
    tcyc = sde.p.dt * sde.p.lMax
    # the rule needs a rate over physical time: the second-half slope (V_arr of a
    # fixed point is the slope of its last V_win cycles)
    vr(r) = [isnan(r.Vh_arr[q]) ? r.V_arr[q] : r.Vh_arr[q] for q in eachindex(r.V_arr)]
    bp = bracket_Ic(sde.pos.J_arr, vr(sde.pos), sde.pos.m_arr .* tcyc)
    bn = bracket_Ic(sde.neg.J_arr, vr(sde.neg), sde.neg.m_arr .* tcyc)
    # sigma_V with the same rate as the brackets (a dissipating rung can close as a fixed point)
    Vabs(r, j) = abs(vr(r)[findfirst(x -> abs(abs(x) - j) < 1e-12, r.J_arr)])
    sv(j) = (a = Vabs(sde.pos, j); c = Vabs(sde.neg, j); 100 * (a - c) / (a + c))
    (pos = bp, neg = bn, eta = eta_bracket(bp, bn), sv = sv)
end

function _txt_brackets(b)
    b === nothing && return String[]
    f(x) = isnan(x) ? "?" : isinf(x) ? "∞" : @sprintf("%.4g", x)
    L = [@sprintf("  Ic+  in (%s, %s) J0   |Ic-| in (%s, %s) J0", f(b.pos.lo), f(b.pos.hi), f(b.neg.lo), f(b.neg.hi))]
    e = b.eta
    push!(L, isnan(e[1]) ? "  eta  : brackets not closed on both branches" :
              @sprintf("  eta  = %.1f %%  [%.1f, %.1f] at the bracket centres and ends%s", e[1], e[2], e[3],
                       (b.pos.ok && b.neg.ok) ? "" : "  (provisional: undecided rungs inside a bracket)"))
    # voltage non-reciprocity at the first |J| where both polarities dissipate
    dp = Dict(b.pos.J[q] => b.pos.crossings[q] for q in eachindex(b.pos.J) if b.pos.cls[q] == 'D')
    com = sort([j for j in b.neg.J if haskey(dp, j) && b.neg.cls[findfirst(==(j), b.neg.J)] == 'D'])
    if !isempty(com)
        j = com[1]; s = b.sv(j)
        push!(L, @sprintf("  sigma_V = (|V+| - |V-|)/(|V+| + |V-|) = %+.1f %% at |J| = %s, both dissipating", s, f(j)))
    end
    und = vcat(["+$(f(x))" for x in b.pos.undecided], ["-$(f(x))" for x in b.neg.undecided])
    isempty(und) || push!(L, "  undecided rungs (1 to 10 crossings per line): " * join(und, ", ") *
                             " -> re-measure them on a doubled window")
    L
end
function _resultado(sde::SDE)
    c = sde.cfg
    println("\n====================== RESULT =======================")
    b = brackets(sde)
    if b !== nothing
        println(" brackets, verdict rule (crossings per line: <1 silent, >=10 dissipating):")
        foreach(println, _txt_brackets(b))
        println(" threshold estimate (|V - V(0)| > V_threshold, interpolated):")
    end
    @printf(" Geom=%s | model=%s | J||%s\n", c.GEOM_MODE, c.defect_model, c.current_axis)
    @printf(" ks=%g | g12=%g | H=%.2f Hc2 | mr=%.2f | a2n=%.2f\n",
            c.kappa_s, c.gamma12, c.H_const, c.m_ratio, c.a2n)
    @printf(" Ic+ = %s J0 | Ic- = %s J0 | eta = %s %%\n",
            _num(sde.Ic_plus), _num(sde.Ic_minus), _num(sde.eta_pct))
    if !isnan(sde.simetria.relmed)
        @printf(" symmetry +J/-J: median %.3e | max %.3e at J=%.4f\n",
                sde.simetria.relmed, sde.simetria.relmax, sde.simetria.J_atmax)
    end
    println("=====================================================")
    @printf(" Wall time: %.1f min\n", sde.wall_min)
end

_num(x) = isnan(x) ? "NaN" : @sprintf("%.4f", x)

# Commit of the source tree the run used ("+dirty" if src/ has uncommitted changes).
function _code_rev()
    dir = dirname(@__DIR__)
    try
        h = readchomp(pipeline(`git -C $dir rev-parse --short HEAD`, stderr=devnull))
        d = readchomp(pipeline(`git -C $dir status --porcelain -- src`, stderr=devnull))
        isempty(d) ? h : h * "+dirty"
    catch
        "unknown (not a git checkout)"
    end
end

"""
    write_summary(sde) -> String

Writes `output/RUN_SUMMARY_<geom>_<tag>_<date>.txt` and appends a line to
`output/runs_log.csv`, with the same columns as the .m so that the existing
sweep checkers keep working.
"""
function write_summary(sde::SDE)
    c = sde.cfg; p = sde.p
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    fecha = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    od = joinpath(c.out_dir, "output")
    mkpath(od)
    fn = joinpath(od, "RUN_SUMMARY_$(c.GEOM_MODE)_$(c.run_tag)_$(stamp).txt")
    open(fn, "w") do io
        println(io, "="^62)
        println(io, " TDGL two-band solver (Julia) — RUN SUMMARY")
        println(io, "="^62)
        println(io, "[RUN]")
        @printf(io, "  date         : %s\n", fecha)
        @printf(io, "  geometry     : %s\n", c.GEOM_MODE)
        @printf(io, "  tag          : %s\n", c.run_tag)
        @printf(io, "  wall time    : %.1f min (%.2f h)\n", sde.wall_min, sde.wall_min/60)
        @printf(io, "  julia        : %s | threads %d\n", VERSION, Threads.nthreads())
        @printf(io, "  code         : %s\n", _code_rev())
        println(io, "\n[RESULT]")
        println(io, "  brackets, verdict rule (crossings per line: <1 silent, >=10 dissipating):")
        foreach(l -> println(io, l), _txt_brackets(brackets(sde)))
        println(io, "  threshold estimate (|V - V(0)| > V_threshold, interpolated):")
        @printf(io, "  Ic+  = %s J0\n  Ic-  = %s J0\n  eta  = %s %%\n",
                _num(sde.Ic_plus), _num(sde.Ic_minus), _num(sde.eta_pct))
        println(io, "  (protocol: clean margin + _REF reverses the sign keeping |Ic|")
        println(io, "   + robust under x_shift/y_shift, BEFORE reporting any eta as physical)")
        println(io, "\n[MESH]")
        @printf(io, "  Nx x Ny x Nz : %d x %d x %d (SC) | dx=dy=dz=%.3f xi\n",
                c.Nx_sc, c.Ny_sc, c.Nz_sc, c.dx)
        @printf(io, "  sample       : %.1f x %.1f x %.1f xi\n",
                c.Nx_sc*c.dx, c.Ny_sc*c.dy, c.Nz_sc*c.dz)
        println(io, "\n[PHYSICS]")
        @printf(io, "  kappa_s=%.2f | kappa_p=%.2f | T_red=%.3f\n", c.kappa_s, c.kappa_p, c.T_red)
        @printf(io, "  xi_ratio: requested=%g | REALISED=%.3f | Hc2_pi/Hc2_sigma=%.4f\n",
                c.xi_ratio, p.xi_ratio_real, hc2_pi(p))
        @printf(io, "  a2n=%.3f | m_ratio=%.3f | gamma12=%.4f\n", c.a2n, c.m_ratio, c.gamma12)
        @printf(io, "  gamma_ratio = %.4f  =>  INTEGRATED xi_pi/xi_sigma = %.4f (independent of Gamma_r)\n",
                c.gamma_ratio, xi_ratio_integrated(p))
        @printf(io, "  (Gamma_r divides the THREE pi-band terms, so it is a relaxation rate and\n")
        @printf(io, "   does not move the statics. Dividing only the condensation (F21) would\n")
        @printf(io, "   give xi_pi/xi_sigma=%.4f and Hc2_pi=%.4f.)\n",
                xi_ratio_f21(p), hc2_pi_f21(p))
        @printf(io, "  H=%.3f Hc2 | J||%s | defect_model=%s | z_mode=%s\n",
                c.H_const, c.current_axis, c.defect_model, c.z_mode)
        @printf(io, "  INJECTION: inject_mode=%s | inject_gain=%g\n", c.inject_mode, c.inject_gain)
        println(io, "  VOLTAGE: Josephson relation, V = d(gauge-invariant phase)/dt")
        println(io, "\n[NUMERICS]")
        @printf(io, "  dt=%.4e t_GL (limited by %s)\n", p.dt, p.dt_limitante)
        @printf(io, "  eps_conv=%.1e (seed J=0: %.1e) | iter_min=%d | lMax=%d | mMax=%d | V_thr=%.1e\n",
                p.eps_conv, p.eps_seed, c.iter_min, c.lMax, c.mMax, c.V_threshold)
        c.eps_conv_rate > 0 &&
            @printf(io, "  eps_conv is per cycle, from eps_conv_rate=%.2e per t_GL\n", c.eps_conv_rate)
        @printf(io, "  numerical mode: %s\n", mode_name(sde.mode))
        @printf(io, "  threads      : %s\n", p.hilos ? "yes" : "no")
        if c.use_phi
            println(io, "\n[SCALAR POTENTIAL]")
            @printf(io, "  use_phi=true | phi_bc=%s | phi_method=%s | f22_supercurrent=%s\n",
                    string(c.phi_bc), string(c.phi_method),
                    string(c.f22_supercurrent))
            @printf(io, "  lambda_TF=%g  tau_Q=%g\n", c.lambda_TF, c.tau_Q)
            if c.lambda_TF == 0 && c.tau_Q == 0
                println(io, "  Phi is a gauge choice. Default Dirichlet = the regression states in")
                println(io, "  data/. The PHYSICAL gauge of this setup is Neumann: there are")
                println(io, "  no electrodes, the current enters through its own field.")
                println(io, "  V_Phi depends on the gauge (factor ~15). The physical voltage is")
                println(io, "  V_Josephson, which is invariant.")
            elseif c.tau_Q > 0
                println(io, "  gauge equivalence BROKEN by tau_Q: Psi rotates by Phi + Q while A is")
                println(io, "  corrected by Phi alone, so the residual enters the dynamics.")
            else
                println(io, "  Thomas-Fermi gives Phi a scale (no null mode, div A != 0), but with")
                println(io, "  tau_Q = 0 the step is still a gauge transformation: |Psi|, B and")
                println(io, "  V_Josephson are those of Phi = 0 (test_phi_tf.jl); only V_Phi moves.")
            end
        end
        println(io, "\n[GEOMETRY] $(length(sde.geom.defects)) defect(s)")
        for (q, d) in enumerate(sde.geom.defects)
            @printf(io, "  #%02d %-9s %s\n", q, d.type, _desc(d))
        end
        println(io, "\n[CONFIG] (every field, to reproduce the run)")
        for f in fieldnames(typeof(c))
            v = repr(getfield(c, f))
            length(v) > 200 && (v = first(v, 200) * " ...")
            @printf(io, "  %-20s = %s\n", f, v)
        end
        for (lbl, r) in (("J+", sde.pos), ("J-", sde.neg))
            r === nothing && continue
            println(io, "\n[CONVERGENCE $lbl]  (iters = cycles m; res = final residual;")
            println(io, "                   '*' = reached mMax without going below eps => NOT converged)")

            @printf(io, "  %12s %14s %8s %12s %12s %6s\n",
                    "J", "V", "iters", "res_Psi1", "res_Psi2", "conv")
            for q in eachindex(r.J_arr)
                @printf(io, "  %+12.4f %14.5e %8d %12.3e %12.3e %6s\n",
                        r.J_arr[q], r.V_arr[q], r.m_arr[q], r.r1_arr[q], r.r2_arr[q],
                        string(r.cflag[q]))
            end
        end
    end

    csv = joinpath(od, "runs_log.csv")
    nuevo = !isfile(csv)
    open(csv, "a") do io
        nuevo && println(io, "fecha,geom,tag,axis,mesh,kappa_s,H,gamma12,m_ratio,a2n,T_red," *
                             "z_mode,defect_model,x_shift,y_shift,J_max,dt,Ic_plus,Ic_minus," *
                             "eta_pct,wall_min")
        @printf(io, "%s,%s,%s,%s,%dx%dx%d,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%s,%s,%.3f,%.3f,%.3f,%.4e,%s,%s,%s,%.1f\n",
                fecha, c.GEOM_MODE, c.run_tag, c.current_axis,
                c.Nx_sc, c.Ny_sc, c.Nz_sc, c.kappa_s, c.H_const, c.gamma12,
                c.m_ratio, c.a2n, c.T_red, c.z_mode, c.defect_model,
                c.x_shift, c.y_shift, c.J_max, p.dt,
                _num(sde.Ic_plus), _num(sde.Ic_minus), _num(sde.eta_pct), sde.wall_min)
    end
    c.verbose && println("Summary written: ", fn)
    fn
end

function _desc(d::Defect)
    campos = String[]
    for f in (:xc,:yc,:zc,:R,:n,:phi,:half,:depth,:hx,:hy,:hz,:xa,:xb,:ya,:yb,
              :flo,:fhi,:nper,:dir,:zlo,:zhi)
        v = getfield(d, f)
        v === nothing || push!(campos, "$f=$v")
    end
    push!(campos, "f=$(d.f)")
    join(campos, " ")
end
