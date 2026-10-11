# ============================================================================
#  sweep.jl — THE SWEEP.  Two-band TDGL kernel, gauge Phi = 0.
#
#  Walks over (J,H) PAIRS, which covers both protocols with one loop:
#      V-I   : constant H_range,  J_range = current ramp
#      M(H)  : J_range = zeros,   H_range = field ramp
#  Seeding is SEQUENTIAL —each point starts from the previous state— and that
#  is not an optimisation: it produces the hysteresis of the V-I curve and the
#  STEPS of M(H).  Each jump is a vortex entering a metastable state, and
#  breaking the sequence erases them (and makes a point depend on the points
#  before it: mode M13).
# ============================================================================

"""Result of a sweep of one polarity."""
struct SweepResult
    J_arr::Vector{Float64};  V_arr::Vector{Float64};  H_arr::Vector{Float64}
    Vst_arr::Vector{Float64}; V2_arr::Vector{Float64}
    m_arr::Vector{Int};      r1_arr::Vector{Float64}; r2_arr::Vector{Float64}
    cflag::Vector{Char}
    M_arr::Vector{Float64};  Nv1_arr::Vector{Int};    Nv2_arr::Vector{Int}
    nslip_arr::Vector{Float64}
    ps1_arr::Vector{Float64};  ps2_arr::Vector{Float64}
    rhos_arr::Vector{Float64}; lamef_arr::Vector{Float64}; rpi_arr::Vector{Float64}
    Itot_arr::Vector{Float64}; Isam_arr::Vector{Float64}
    nslipw_arr::Vector{Float64}
    thmax_arr::Vector{Float64}; dphcyc_arr::Vector{Float64}
    VPhi_arr::Vector{Float64}   # [v14-C] drop of Phi; NaN if use_phi=false
    dmax_arr::Vector{Float64}   # largest per-line phase increment in one cycle (rad)
    Vf1_arr::Vector{Float64}    # voltage over the lines whose path has f = 1 throughout
    VA_arr::Vector{Float64}     # rate of the line integral of A over the window (not gauge invariant; diagnostic)
    nbig_arr::Vector{Int}       # per-line increments above 2 rad (possible misattributed slips)
    Vh_arr::Vector{Float64}     # slope of <Theta> over the second half of EVERY point (F or V);
                                # V_arr of a fixed point is the slope of the last V_win cycles
    Vh2_arr::Vector{Float64}    # same for the pi band; V2_arr of a fixed point is a last-V_win slope too
    vtrace::Vector{NamedTuple}
    estado::State
end

function sweep(J_range::Vector{Float64}, H_range::Vector{Float64},
                 label::String, g::Geometry, p::Params, mode::NumMode;
                 estado::Union{State,Nothing}=nothing, jp_start::Int=1,
                 J_arr0=nothing, V_arr0=nothing)

    (; Nx, Ny, Nz, x1, x2, y1, y2, z1, z2, dx, dy, dz, dt, lMax, mMax) = p
    Npts = length(J_range)
    axis_y = p.axis_y
    dt_cyc = lMax*dt                      # physical time per cycle [t_GL]

    # The EFFECTIVE floor is the larger of the floor in cycles and the
    # one set by t_min_GL.  The criterion then no longer depends on dt/kappa/lMax.
    iter_min = p.iter_min
    p.t_min_GL > 0 && (iter_min = max(iter_min, ceil(Int, p.t_min_GL/dt_cyc)))

    s = estado === nothing ? initial_state(p, g, H_range[1], mode) : estado
    J_arr = zeros(Npts)
    V_arr = zeros(Npts)
    if J_arr0 !== nothing
        n0 = min(length(J_arr0), Npts)
        J_arr[1:n0] .= J_arr0[1:n0]
    end
    if V_arr0 !== nothing
        n0 = min(length(V_arr0), Npts)
        V_arr[1:n0] .= V_arr0[1:n0]
    end

    m_arr = zeros(Int, Npts); r1_arr = zeros(Npts); r2_arr = zeros(Npts)
    cflag = fill(' ', Npts)
    Vst_arr = zeros(Npts); V2_arr = zeros(Npts)
    M_arr = zeros(Npts); Nv1_arr = zeros(Int, Npts); Nv2_arr = zeros(Int, Npts)
    nslip_arr = zeros(Npts)
    ps1_arr = zeros(Npts); ps2_arr = zeros(Npts); rhos_arr = zeros(Npts)
    lamef_arr = zeros(Npts); rpi_arr = zeros(Npts)
    Itot_arr = zeros(Npts); Isam_arr = zeros(Npts); nslipw_arr = zeros(Npts)
    thmax_arr = zeros(Npts); dphcyc_arr = zeros(Npts)
    VPhi_arr = fill(NaN, Npts)
    dmax_arr = zeros(Npts); Vf1_arr = fill(NaN, Npts)
    VA_arr = fill(NaN, Npts); nbig_arr = zeros(Int, Npts); Vh_arr = fill(NaN, Npts)
    Vh2_arr = fill(NaN, Npts)
    vtrace = NamedTuple[]

    # [v14] The Phi solver is built —and FACTORISED— once for the whole
    # sweep: the matrix depends neither on time nor on the current point.
    sp = p.use_phi ? SolverPhi(p; bc=p.phi_bc, metodo=p.phi_method) : nothing
    if sp !== nothing && sp.tau_Q > 0 && length(p.Q_restart) == length(sp.Qimb) && any(!iszero, p.Q_restart)
        sp.Qimb .= p.Q_restart; fill!(p.Q_restart, 0.0)     # residual of a resumed checkpoint
    end
    p.blas_threads > 0 && BLAS.set_num_threads(p.blas_threads)
    mi = p.maxwell_implicit ? MaxwellImplicit(p, g) : nothing
    if sp !== nothing && p.verbose
        @printf("[phi] scalar potential ON | gauge=%s | method=%s | F22 (1-T) on J_s: %s\n",
                p.phi_bc, p.phi_method, p.f22_supercurrent ? "KEPT" : "removed")
        @printf("      Phi: %d unknowns, factorised once\n", sp.n1*sp.n2*sp.n3)
        p.lambda_TF > 0 && @printf("      Thomas-Fermi lambda_TF=%.4g xi  (breaks gauge equivalence)\n", p.lambda_TF)
        p.tau_Q > 0 && @printf("      charge imbalance tau_Q=%.4g t_GL  (mu = Phi + Q)\n", p.tau_Q)
        p.phi_bc === :dirichlet && p.lambda_TF == 0 && p.tau_Q == 0 &&
            println("      gauge: Dirichlet is the regression BC; the physical one for this",
                    " setup is Neumann (no electrodes, current enters through its own field).")
    end

    Cooper0_F1 = [cabs(mode, s.G1[i,j,k]) for i in x1:x2+1, j in y1:y2+1, k in z1:z2+1]
    Cooper0_F2 = [cabs(mode, s.G2[i,j,k]) for i in x1:x2+1, j in y1:y2+1, k in z1:z2+1]
    fieldOld = copy(s.hz)

    # --- VOLTMETER: PRECOMPUTED masks and counters --------------------------
    # The planes and the set of links change neither with J nor with m, so
    # they are built once.  The voltmeter is evaluated EVERY cycle to feed
    # the series V(m), so its cost matters.
    dv = p.delta_v
    if axis_y
        vi = y1 + dv; vo = y2 - dv
        Min = [g.tauy[i,vi,k] > 0.5 for i in x1:x2+1, k in z1:z2+1]
        Mou = [g.tauy[i,vo,k] > 0.5 for i in x1:x2+1, k in z1:z2+1]
    else
        vi = x1 + dv; vo = x2 - dv
        Min = [g.taux[vi,j,k] > 0.5 for j in y1:y2+1, k in z1:z2+1]
        Mou = [g.taux[vo,j,k] > 0.5 for j in y1:y2+1, k in z1:z2+1]
    end
    nin = count(Min); nou = count(Mou)
    (nin == 0 || nou == 0) && error(
        "sweep: the voltmeter found no live links on the measuring planes " *
        "delta_v=$dv. Check the geometry or lower cfg.delta_v.")

    # PATH for the line integral of the link phase between the two
    # measuring planes.  Only FULLY live transverse lines: if the notch cuts
    # the line, that path does not exist.
    # `line_ok` is built from tau (link connectivity), NOT from f.  That
    # is why lines crossing the f=0 region survive: tau is still live there.
    # The consequence is documented in the .m: the observable is
    # D = arg(psi_i^* U_ij psi_j), and where |psi| ~ 1e-5 inside the notch
    # arg(psi) is amplified rounding noise.
    line_ok = if axis_y
        [all(g.tauy[i,jj,k] > 0.5 for jj in vi:vo-1) for i in x1:x2+1, k in z1:z2+1]
    else
        [all(g.taux[ii,j,k] > 0.5 for ii in vi:vo-1) for j in y1:y2+1, k in z1:z2+1]
    end
    nlin = count(line_ok)
    # Lines whose whole path lies where f = 1: the voltage on them alone tells
    # whether slips counted inside the notch (|psi| small, arg ill-conditioned)
    # contribute to V.  Diagnostic only.
    line_f1 = if axis_y
        [line_ok[a, b] && all(g.fmask[x1+a-1, jj, z1+b-1] >= 1 - 1e-12 for jj in vi:vo)
         for a in axes(line_ok, 1), b in axes(line_ok, 2)]
    else
        [line_ok[a, b] && all(g.fmask[ii, y1+a-1, z1+b-1] >= 1 - 1e-12 for ii in vi:vo)
         for a in axes(line_ok, 1), b in axes(line_ok, 2)]
    end
    nlin_f1 = count(line_f1)
    nlin == 0 && error("sweep: no complete transverse line between the " *
                       "measuring planes. Lower cfg.delta_v or check the geometry.")

    nwin = max(2, p.V_win)
    # five per-cycle series of mMax + 1 entries each (40 bytes per cycle of ceiling),
    # two more with v13_diag
    nbytes = (p.v13_diag ? 56 : 40) * (mMax + 1)
    nbytes > 1e9 && @warn @sprintf("sweep: mMax = %d reserves %.1f GB for the per-cycle series; use the ceiling you need", mMax, nbytes / 1e9)
    Vser  = zeros(mMax + 1)     # V series per cycle, within one current point
    Tser  = zeros(mMax + 1)     # unwrapped gauge-invariant phase (sigma)
    Tser2 = zeros(mMax + 1)     # same, pi band
    Tser_f1 = zeros(mMax + 1)   # same, f = 1 lines only (diagnostic)
    dmax_cur = 0.0              # largest per-line increment of this point
    nbig_cur = 0                # per-line increments above 2 rad
    # total and sample current through the mid plane, per cycle (v13_diag): the
    # share of the sample fluctuates by a few per cent as vortices cross, so the
    # point reports its mean over the second half, not one instant
    ItSer = zeros(p.v13_diag ? mMax + 1 : 0); IsSer = zeros(p.v13_diag ? mMax + 1 : 0)
    TAser = zeros(mMax + 1)     # Theta_A: link phases of A summed along the path, line mean

    nl_1, nl_2 = size(line_ok)
    Tmat  = zeros(nl_1, nl_2); Tmat2 = zeros(nl_1, nl_2)
    Dprev = zeros(nl_1, nl_2);  Dprev2 = zeros(nl_1, nl_2)
    Dl    = zeros(nl_1, nl_2);  Dl2    = zeros(nl_1, nl_2)

    t_ini = time()
    for jp in jp_start:Npts
        if _stop_pedido(p)
            p.verbose && @printf("[STOP] before jp=%d. Checkpoint of the previous J intact.\n", jp)
            break
        end
        J_cur = J_range[jp]; H = H_range[jp]
        m = 1; conv_mode = '*'
        HJ = p.inject_gain * J_cur * p.slab_d / (2*p.ks2)
        res_prev = 0.0
        it_res_min = max(iter_min, ceil(Int, p.t_min_GL_res/dt_cyc))

        sub = p.unwrap_every                      # 0 = once per cycle
        if sub > 0                                # baseline for in-cycle unwrapping
            _fase_camino!(Dprev,  s.G1, s, p, vi, vo, axis_y)
            _fase_camino!(Dprev2, s.G2, s, p, vi, vo, axis_y)
            fill!(Tmat, 0.0); fill!(Tmat2, 0.0); dmax_cur = 0.0; nbig_cur = 0
        end
        while m <= mMax
            for l in 1:lMax
                if sp === nothing
                    substep!(s, p, g, mode; H=H, HJ=HJ, J_cur=J_cur, mi=mi)
                else
                    substep_phi!(s, p, g, mode, sp; H=H, HJ=HJ, J_cur=J_cur)
                end
                if sub > 0 && (l % sub == 0 || l == lMax)
                    _fase_camino!(Dl,  s.G1, s, p, vi, vo, axis_y)
                    _fase_camino!(Dl2, s.G2, s, p, vi, vo, axis_y)
                    @inbounds for i in eachindex(Tmat)
                        line_ok[i] || continue
                        d = Dl[i] - Dprev[i]
                        d -= 2pi*oround_f(d/(2pi))
                        dmax_cur = max(dmax_cur, abs(d))
                        abs(d) > 2.0 && (nbig_cur += 1)
                    end
                    _desenvolver!(Tmat, Dl, Dprev)
                    _desenvolver!(Tmat2, Dl2, Dprev2)
                end
            end

            # Soft cap: if θ exceeds 0.45 the condensate is no longer the
            # diode of Article V. This J is cut. With heating off, this branch
            # does not run.
            if p.use_heat && maximum(p.Temp) > 0.45
                conv_mode = 'T'
                p.verbose && @printf("heat: θmax=%.3f > 0.45, cutting J=%+.3f\n",
                                     maximum(p.Temp), J_cur)
                flush(stdout)
                break
            end

            # --- [J1] INSTANTANEOUS VOLTMETER -----------------------------
            # <dA/dt> along the current axis, EVERY cycle.
            # The planes vi and vo ARE exact mirror images (link i maps
            # to x1+x2-i); the bias was in the transverse set of nodes, which
            # is now complete and symmetric.
            Vser[m] = _promedio_plano(s, p, vo, Mou, axis_y)/nou -
                      _promedio_plano(s, p, vi, Min, axis_y)/nin
            TAser[m] = _theta_A(s, p, vi, vo, axis_y, line_ok)

            # GAUGE-INVARIANT PHASE DIFFERENCE along the path.
            # int A.dl is NOT used: that integral is NOT gauge invariant, and a
            # pure gauge drift (leaving |psi| and B intact) moves it; -5.1e-4
            # was measured in a static state where it should be 0.  The right
            # quantity is the one in the Josephson relation,
            #     D_link = arg( psi_i^* U_link psi_j ),
            # which IS invariant.  Its time derivative is the voltage.
            sub == 0 && _fase_camino!(Dl,  s.G1, s, p, vi, vo, axis_y)
            sub == 0 && _fase_camino!(Dl2, s.G2, s, p, vi, vo, axis_y)
            if sub > 0
                Tser[m]  = sumlin(Tmat[i]  for i in eachindex(Tmat)  if line_ok[i])/nlin
                Tser2[m] = sumlin(Tmat2[i] for i in eachindex(Tmat2) if line_ok[i])/nlin
                nlin_f1 > 0 && (Tser_f1[m] = sumlin(Tmat[i] for i in eachindex(Tmat) if line_f1[i])/nlin_f1)
            elseif m == 1
                copyto!(Dprev, Dl); fill!(Tmat, 0.0);  Tser[1] = 0.0
                copyto!(Dprev2, Dl2); fill!(Tmat2, 0.0); Tser2[1] = 0.0
                Tser_f1[1] = 0.0; dmax_cur = 0.0; nbig_cur = 0
            else
                # UNWRAP PER LINE, NOT THE AVERAGE.  A vortex crossing
                # ONE line makes it jump by 2*pi but moves the average only by
                # 2*pi/nlin: with nlin ~ 175, "accumulate 2*pi" needed ~175
                # crossings and the gate declared V=0 in runs that DID flow.
                # the unwrapping is faithful while every per-line increment is
                # below pi; the largest one is recorded (diagnostic only)
                @inbounds for i in eachindex(Tmat)
                    line_ok[i] || continue
                    d = Dl[i] - Dprev[i]
                    d -= 2pi*oround_f(d/(2pi))
                    dmax_cur = max(dmax_cur, abs(d))
                    abs(d) > 2.0 && (nbig_cur += 1)
                end
                _desenvolver!(Tmat, Dl, Dprev)
                _desenvolver!(Tmat2, Dl2, Dprev2)
                nlin_f1 > 0 && (Tser_f1[m] = sumlin(Tmat[i] for i in eachindex(Tmat) if line_f1[i])/nlin_f1)
                Tser[m]  = sumlin(Tmat[i]  for i in eachindex(Tmat)  if line_ok[i])/nlin
                Tser2[m] = sumlin(Tmat2[i] for i in eachindex(Tmat2) if line_ok[i])/nlin
            end

            p.v13_diag && ((ItSer[m], IsSer[m]) = _midplane_current(s, p))

            # --- [J2] CONVERGENCE.  Two routes; converges by EITHER --------
            #  (a) FIXED POINT in |psi| and hz -> Meissner or pinned vortices.
            #  (b) STATIONARY <V> -> resistive limit cycle.  There (a) is
            #      UNREACHABLE by construction: each passing vortex modulates
            #      |psi| periodically and the residual saturates at the
            #      amplitude of that oscillation, never below eps.
            # The FIRST point (J=0) uses its own tolerance, eps_seed: that state
            # is the seed carried through the whole sweep.  By default it is
            # 5e-4, LOOSER than the 1e-5 of production runs (it was set when
            # eps_conv was 1e-3); tie_seed_tolerance=true takes the smaller.
            eps_here = jp == 1 ? p.eps_seed : p.eps_conv
            diff_F1 = 0.0; diff_F2 = 0.0; amax = 0.0
            @inbounds for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1
                a = cabs(mode, s.F1[i,j,k]); b = cabs(mode, s.F2[i,j,k])
                ii = i-x1+1; jj = j-y1+1; kk = k-z1+1
                # `max`, not `>`: see the note on `amax` below.  Here it is
                # WORSE: with the comparison, a NaN state left diff_F1 = 0, the
                # fixed-point criterion was satisfied and the point was
                # reported as conv='F' with ZERO residual.  A diverged run
                # presented itself as converged.
                diff_F1 = max(diff_F1, abs(a - Cooper0_F1[ii,jj,kk]))
                diff_F2 = max(diff_F2, abs(b - Cooper0_F2[ii,jj,kk]))
                # `max`, not `a > amax`: NaN > x is FALSE, so with the
                # comparison the maximum never became NaN and the guard did not
                # see the divergence in front of it.  Julia's `max` DOES
                # propagate NaN.  (Its own test found this bug.)
                amax = max(amax, a, b)
                Cooper0_F1[ii,jj,kk] = a; Cooper0_F2[ii,jj,kk] = b
            end
            # [GUARD] Divergence.  Checked in the same pass, at no cost.
            # Without it a run with the CFL violated goes on to mMax producing
            # NaN and is found broken hours later.  The message names the
            # three real causes, by frequency.
            if p.guard_psi > 0 && (!isfinite(amax) || amax > p.guard_psi)
                save_checkpoint(p, s, label, jp, V_arr, J_arr;
                                Qimb = (sp !== nothing && sp.tau_Q > 0) ? sp.Qimb : nothing)
                error("""
                [GUARD] The scheme DIVERGED in $label, point jp=$jp (J=$J_cur), cycle m=$m.
                        max|Psi| = $amax  (equilibrium = 1, threshold cfg.guard_psi = $(p.guard_psi))
                        dt = $(p.dt), limited by $(p.dt_limitante).
                  Causes, by frequency:
                    1. kappa raised without recomputing dt -> dt2s = dl/(4 kappa^2)
                    2. m_r < 1 (large xi_ratio) without dt1p -> the pi band diffuses 1/m_r faster
                    3. dx lowered without lowering dt      -> dl ~ dx^2
                  The state was saved in restart/ for inspection.""")
            end
            diffh = 0.0
            @inbounds for i in eachindex(s.hz)
                diffh = max(diffh, abs(s.hz[i] - fieldOld[i]))
            end
            copyto!(fieldOld, s.hz)
            r1_arr[jp] = diff_F1; r2_arr[jp] = diff_F2

            if m >= iter_min && max(diff_F1, diff_F2) < eps_here && diffh < eps_here
                # conv_by_time: not while the second half of the point carries one or
                # more phase crossings per line (a flowing state can look fixed for one
                # short cycle)
                if !p.conv_by_time || m < 2 ||
                   max(abs(Tser[m] - Tser[max(1, m ÷ 2)]), abs(Tser2[m] - Tser2[max(1, m ÷ 2)])) < 2pi
                    conv_mode = 'F'; break
                end
            end
            # Route (b) requires J != 0.  At J=0 the state MUST be a
            # true fixed point; and since the voltmeter carries a STATIC
            # offset (mode M1) above V_threshold, the stationarity test was
            # satisfied TRIVIALLY.  In M(H) mode ALL points have J=0, so
            # 'jp>1' was not enough.
            # conv_by_time: also on the first point of a sweep (a ladder resumed from a
            # rung state, or one rung per call, starts at jp = 1); J = 0 stays excluded.
            if p.use_Vconv && (jp > 1 || p.conv_by_time) && abs(J_cur) > 0 && m >= max(iter_min, 2nwin)
                # NO EARLY STOP IN THE RESISTIVE STATE.  Agreement
                # between windows is the wrong test —with 34 % scatter per
                # window it fires only when the noise cancels by chance, and
                # those chances are BIASED: it returned 6.5e-4 where the
                # long-run value was 1.18e-2—.  The right test is to INTEGRATE
                # LONG ENOUGH, and stop when the residual stalls, which is the
                # signature of the limit cycle.
                res_now = max(diff_F1, diff_F2)
                if p.conv_by_time
                    # the verdict rule decides: ten or more phase crossings per line (the
                    # second-half winding times two, as the crossings/line of the RES
                    # lines).  A stall of the residual needs windows longer than the
                    # crossing period (~100 t_GL near onset) and never fired there.
                    res_plateau = abs(Tser[m] - Tser[max(1, m ÷ 2)]) * 2 >= 10 * 2pi
                else
                    res_plateau = res_prev > 0 && abs(res_now - res_prev) <= p.tol_res*res_now
                end
                res_prev = res_now
                if res_plateau && m >= it_res_min
                    conv_mode = 'V'; break
                end
            end
            m += 1
        end
        mlast = min(m, mMax)   # `m` can be mMax+1 if the ceiling was reached

        # --- [K] V OF THE POINT ---------------------------------------------
        # The DC voltage *is* the averaged <dA/dt>; reading one instant carries
        # the phase noise of the limit cycle, and eta is the difference of two
        # nearly equal Ic, so that noise would go straight into eta.
        nav = min(nwin, mlast)
        V_static = meanlin(view(Vser, mlast-nav+1:mlast))
        # If the point closed by the resistive route, the slope is
        # taken over the SECOND HALF, not over a fixed window: the scatter
        # goes from 34 % (400 cycles) to 1.6 % (3000).  The first half is
        # discarded because it contains the transient of the limit cycle.
        conv_mode == 'V' && (nav = min(max(p.V_win_min_res, mlast ÷ 2), mlast))
        wq = mlast-nav+1:mlast
        nslip = 0.0; V_cur = 0.0; V2_cur = 0.0
        if length(wq) >= 3
            tq = (collect(wq) .- wq[1]) .* dt_cyc
            V2_cur = -pendiente(tq, view(Tser2, wq))
            V_cur  = -pendiente(tq, view(Tser,  wq))
            nlin_f1 > 0 && (Vf1_arr[jp] = -pendiente(tq, view(Tser_f1, wq)))
            # Rate of the line integral of A over the window, from its two ends,
            # in the sign convention of the Josephson voltmeter.  A is smooth, so
            # no core can be misattributed here.  It is the ELECTRIC potential
            # (int E.dl, Phi = 0); the Josephson voltmeter is the ELECTROCHEMICAL
            # one, and in TDGL they differ by the conversion potential around
            # moving cores (M18): 5-26 % in flux flow on a 24x16x4 box.  At J = 0
            # any drift is a transient of the longitudinal A (gradient flow).
            # Not gauge invariant.  Diagnostic only.
            VA_arr[jp] = (TAser[wq[end]] - TAser[wq[1]]) / ((wq[end] - wq[1]) * dt_cyc)
            # PHASE-SLIP GATE.  V != 0 requires a vortex to CROSS
            # the measuring path, and each crossing is worth 2*pi.  If over
            # the whole window the phase does not complete 2*pi there was no
            # crossing and the voltage is ZERO by definition, whatever slope
            # the fit returns: that slope is a transient, or the oscillation
            # of a pinned vortex that approaches and recedes without passing.
            # SCOPE: sum|Tmat| adds ABSOLUTE VALUES and random-walks upwards,
            # so in practice the threshold does not fire and covers only the
            # degenerate case of a very short run.  The working discriminator
            # is |V| against V(J=0) of the SAME run, and the window crossing
            # count (nslipw).
            nslip = sumlin(abs(Tmat[i]) for i in eachindex(Tmat) if line_ok[i])/(2pi)
            nslip < 1 && (V_cur = 0.0)
        end
        # The rate the verdict rule needs, for every point. A point that closes as a
        # fixed point reports in V_cur the slope of its last V_win cycles: 8 t_GL at
        # kappa = 2 but 0.1 t_GL at kappa = 25, where a still-slipping state then
        # reads like noise. The slope over the second half of the point is the
        # same quantity as V_cur for a resistive point and a physical-time rate for
        # a fixed one. Diagnostic: it does not enter the dynamics.
        wh = (mlast - mlast ÷ 2 + 1):mlast
        if length(wh) >= 3
            th = (collect(wh) .- wh[1]) .* dt_cyc
            Vh_arr[jp] = nslip < 1 ? 0.0 : -pendiente(th, view(Tser, wh))
            # pi band: same window and the same gate, on its own phase
            nslip2 = sumlin(abs(Tmat2[i]) for i in eachindex(Tmat2) if line_ok[i])/(2pi)
            Vh2_arr[jp] = nslip2 < 1 ? 0.0 : -pendiente(th, view(Tser2, wh))
        end

        # [v14-C] The drop of Phi over the SAME planes as the Josephson
        # voltmeter.  The two do NOT measure the same thing: the physical
        # voltage is gauge invariant and equal in both gauges; what changes
        # between them is how it splits between Phi and the phase drift.
        sp === nothing || (VPhi_arr[jp] = phi_voltage(sp, p, vi, vo, Min, Mou, axis_y))

        nslip_arr[jp] = nslip
        J_arr[jp] = J_cur; V_arr[jp] = V_cur; Vst_arr[jp] = V_static
        dmax_arr[jp] = dmax_cur
        nbig_arr[jp] = nbig_cur
        V2_arr[jp] = V2_cur; m_arr[jp] = m; cflag[jp] = conv_mode

        # --- magnetic diagnostics ---------------------------------------
        kc = oround((z1+z2)/2)
        M_arr[jp] = magnetization(s.hz, H, p)
        Nv1, _, nl1v, lok = count_vortices(s.F1, s.Vx, s.Vy, s.hz, g.fmask, p, kc)
        Nv2, _, nl2v, _   = count_vortices(s.F2, s.Vx, s.Vy, s.hz, g.fmask, p, kc)
        Nv1_arr[jp] = Nv1; Nv2_arr[jp] = Nv2

        if p.v13_diag
            _diagnosticos_v13!(jp, s, p, g, H, J_cur, V_cur, dt_cyc, wq, Tser,
                               ps1_arr, ps2_arr, rhos_arr, lamef_arr, rpi_arr,
                               Itot_arr, Isam_arr, nslipw_arr, thmax_arr, dphcyc_arr;
                               ItSer=ItSer, IsSer=IsSer, mlast=mlast)
        end

        if lok && (nl1v != Nv1 || nl2v != Nv2)
            p.verbose && @printf("  [Nv] plaquettes and perimeter disagree: sigma %d/%d, pi %d/%d. Possible giant vortex or perimeter over a region without condensate.\n",
                                 Nv1, nl1v, Nv2, nl2v)
        end

        if p.vtrace_every > 0 && abs(J_cur) >= p.vtrace_Jmin &&
           (jp % p.vtrace_every == 0 || jp == Npts)
            push!(vtrace, (J=J_cur, V=Vser[1:mlast], T=Tser[1:mlast],
                           T2=Tser2[1:mlast], Tmat=copy(Tmat), Tmat2=copy(Tmat2),
                           line_ok=copy(line_ok), dt_cyc=dt_cyc, conv=conv_mode))
        end

        if p.verbose
            @printf("[%s][%s] J=%+.3f H=%.3f | Vdc=%+.3e (slips=%.2f) | Vest=%+.3e | dF1=%.2e | it=%d | conv=%s | M=%+.3e | Nv=%d/%d | Vpi=%+.3e\n",
                    Dates.format(Dates.now(), "HH:MM:SS"), label, J_cur, H,
                    V_cur, nslip, V_static, r1_arr[jp], m, conv_mode,
                    M_arr[jp], Nv1, Nv2, V2_cur)
            sp === nothing || @printf("   [v14] J=%+.3f | V_Phi=%+.4e | V_Josephson=%+.4e | V_Phi/V_J=%+.4f | max|div A|=%.2e\n",
                    J_cur, VPhi_arr[jp], V_cur,
                    VPhi_arr[jp]/(abs(V_cur) < 1e-30 ? NaN : V_cur),
                    maximum(abs, link_divergence(s, p)))
            @printf("   [voltmeter] max per-line increment=%.3f rad (%d above 2 rad) | V(f=1 lines)=%+.4e on %d of %d lines | V_A=%+.4e | V(2nd half)=%+.4e, %.2f crossings/line | Vpi(2nd half)=%+.4e\n",
                    dmax_cur, nbig_cur, Vf1_arr[jp], nlin_f1, nlin, VA_arr[jp],
                    Vh_arr[jp], abs(Vh_arr[jp]) * mlast * dt_cyc / (2pi), Vh2_arr[jp])
            p.v13_diag && @printf("   [v13] J=%+.3f | <|Ps|^2>=%.4f <|Pp|^2>=%.4f | rho_s=%.4f lam_ef=%.4f rho_p/rho_s=%.4f | I/Inom=%.4f frac_sample=%.3f | crossings_window=%.2f | max|th|=%.3g dphi_cycle=%.4f\n",
                    J_cur, ps1_arr[jp], ps2_arr[jp], rhos_arr[jp], lamef_arr[jp],
                    rpi_arr[jp], Itot_arr[jp], Isam_arr[jp], nslipw_arr[jp],
                    thmax_arr[jp], dphcyc_arr[jp])
            flush(stdout)
        end

        # If |V| falls within a decade of V_threshold, the point is
        # DECIDING Ic and is close to the criterion, which is exactly where
        # <V> needs more time to settle.
        # the warning names the parameter that governs AT THIS POINT.
        # t_min_GL_res acts only on resistive points; a point that closes as
        # a fixed point —the important case is the J=0 seed, which sets the
        # floor— never touches it, so obeying a generic warning returned the
        # SAME value bit for bit and the user concluded it was converged.
        # A GENERIC WARNING PRODUCED A FALSE CONFIRMATION.
        if p.verbose && 0.1*p.v_thr < abs(V_cur) < 10*p.v_thr
            par = conv_mode == 'V' ? "t_min_GL_res" : "t_min_GL"
            @printf("    [M8] WARNING: |V|=%.2e is within a decade of V_threshold=%.1e.\n      It closed with conv=%s, so the parameter that governs HERE is %s:\n      repeat this J with it doubled and check that V(J) comes out MONOTONIC.\n",
                    abs(V_cur), p.v_thr, conv_mode, par)
        end

        ck = save_checkpoint(p, s, label, jp, V_arr, J_arr;
                           VPhi_arr = sp === nothing ? nothing : VPhi_arr,
                           Qimb = (sp !== nothing && sp.tau_Q > 0) ? sp.Qimb : nothing)
        if p.keep_rung_states && ck !== nothing
            dr = joinpath(dirname(ck), "rungs"); mkpath(dr)
            cp(ck, joinpath(dr, @sprintf("state_%s_J%+.4f.mat", label, J_cur)); force=true)
        end
        _append_j_history(p, label, J_cur, V_cur, conv_mode, m, Nv1, Nv2,
                          p.v13_diag ? ps2_arr[jp] : NaN)

        if p.vtk_every > 0 && !isempty(p.out_dir) && mod(jp - 1, p.vtk_every) == 0
            vd = joinpath(p.out_dir, "output", "vtk")
            mkpath(vd)
            fn = joinpath(vd, @sprintf("estado_%s_%s_J%+.3f.vtk",
                                       p.run_tag, label, J_cur))
            write_vtk(fn, s, p; spacing=:xi, fmask=g.fmask,
                        Phi = sp === nothing ? nothing : sp.Phi,
                        Qimb = (sp !== nothing && sp.tau_Q > 0) ? sp.Qimb : nothing)
            p.verbose && println("   [vtk] ", fn)
        end
        if p.snap_every > 0 && !isempty(p.out_dir) &&
           (jp == 1 || jp == Npts || mod(jp, p.snap_every) == 0)
            sd = joinpath(p.out_dir, "output", "snapshots")
            pgm = write_snapshot_xy(sd, s, p, label, J_cur; tag=p.run_tag)
            p.verbose && println("   [snap] ", pgm)
        end

        # [GUARD] Wall-clock budget.  The current point has closed and its
        # checkpoint is written, so stopping here is safe: the run resumes with
        # restart_pos/restart_neg and continues at jp+1.  Meant for cluster
        # queues with a time limit, where being killed without a checkpoint
        # costs the whole run.
        if p.guard_hours > 0 && (time() - t_ini)/3600 >= p.guard_hours
            p.verbose && @printf("[GUARD] Budget of %.2f h used up after point jp=%d (J=%+.4f).\n         Checkpoint written. Resume with restart_%s=true.\n",
                                 p.guard_hours, jp, J_cur,
                                 label == "J+" ? "pos" : "neg")
            break
        end
        # STOP.txt (TUI key s): closes this J, saves, exits. The current J
        # is not lost. Resume = restart_pos/neg. It is not a mid-cycle
        # checkpoint.
        if _stop_pedido(p)
            p.verbose && @printf("[STOP] STOP.txt after jp=%d (J=%+.4f). Checkpoint written.\n",
                                 jp, J_cur)
            break
        end
        # Heat cap (conv='T'): this J was cut with θmax > 0.45 and θ is still
        # high, so the next J would be cut in its first cycle, and so on for
        # the whole ladder, with no time to cool. The ladder of this polarity
        # ends here. The V of a 'T' point is not quoted.
        if conv_mode == 'T'
            p.verbose && @printf("[HEAT] θmax > 0.45 at jp=%d (J=%+.4f): end of the %s ladder.\n",
                                 jp, J_cur, label)
            break
        end

        # [L] Ohmic early exit (optional; OFF by default)
        if p.early_exit && jp >= 8 && abs(V_arr[jp]) > 15*p.v_thr
            Jr = J_arr[jp-3:jp]; Vr = V_arr[jp-3:jp]
            if all(!isnan, Vr) && (Jr[end] - Jr[1]) != 0
                dVs = diff(Vr) ./ diff(Jr)
                if length(dVs) >= 2 && std(dVs)/(abs(mean(dVs)) + 1e-15) < 0.02
                    slp = mean(dVs)
                    for je in jp+1:Npts
                        J_arr[je] = J_range[je]
                        V_arr[je] = V_arr[jp] + slp*(J_range[je] - J_range[jp])
                    end
                    p.verbose && @printf("[%s] OHMIC at J=%.4f: %d points extrapolated.\n",
                                         label, J_cur, Npts - jp)
                    break
                end
            end
        end
    end

    SweepResult(J_arr, V_arr, copy(H_range), Vst_arr, V2_arr, m_arr, r1_arr,
              r2_arr, cflag, M_arr, Nv1_arr, Nv2_arr, nslip_arr,
              ps1_arr, ps2_arr, rhos_arr, lamef_arr, rpi_arr,
              Itot_arr, Isam_arr, nslipw_arr, thmax_arr, dphcyc_arr,
              VPhi_arr, dmax_arr, Vf1_arr, VA_arr, nbig_arr, Vh_arr, Vh2_arr, vtrace, s)
end

# ---------------------------------------------------------------------------
#  VOLTMETER HELPERS
# ---------------------------------------------------------------------------
function _promedio_plano(s::State, p::Params, plano::Int, M, axis_y::Bool)
    (; x1, x2, y1, y2, z1, z2) = p
    tot = 0.0
    if axis_y
        @inbounds for (kk, k) in enumerate(z1:z2+1), (ii, i) in enumerate(x1:x2+1)
            M[ii,kk] && (tot += s.phiy[i,plano,k])
        end
    else
        @inbounds for (kk, k) in enumerate(z1:z2+1), (jj, j) in enumerate(y1:y2+1)
            M[jj,kk] && (tot += s.phix[plano,j,k])
        end
    end
    tot
end

"""Mean over the measuring lines of the link phases of A summed along vi->vo."""
function _theta_A(s::State, p::Params, vi::Int, vo::Int, axis_y::Bool, line_ok)
    (; x1, x2, y1, y2, z1, z2) = p
    tot = 0.0; n = 0
    if axis_y
        @inbounds for (kk, k) in enumerate(z1:z2+1), (ii, i) in enumerate(x1:x2+1)
            line_ok[ii,kk] || continue
            for jj in vi:vo-1; tot += s.phiy[i,jj,k]; end
            n += 1
        end
    else
        @inbounds for (kk, k) in enumerate(z1:z2+1), (jj, j) in enumerate(y1:y2+1)
            line_ok[jj,kk] || continue
            for ii in vi:vo-1; tot += s.phix[ii,j,k]; end
            n += 1
        end
    end
    n > 0 ? tot/n : 0.0
end

"""Sum of `arg(psi_i^* U psi_j)` along the path vi->vo, per line."""
function _fase_camino!(D, G, s::State, p::Params, vi::Int, vo::Int, axis_y::Bool)
    (; x1, x2, y1, y2, z1, z2) = p
    fill!(D, 0.0)
    if axis_y
        @inbounds for (kk, k) in enumerate(z1:z2+1), (ii, i) in enumerate(x1:x2+1)
            acc = 0.0
            for jj in vi:vo-1
                acc += angle((conj(G[i,jj,k])*s.Vy[i,jj,k])*G[i,jj+1,k])
            end
            D[ii,kk] = acc
        end
    else
        @inbounds for (kk, k) in enumerate(z1:z2+1), (jj, j) in enumerate(y1:y2+1)
            acc = 0.0
            for ii in vi:vo-1
                acc += angle((conj(G[ii,j,k])*s.Vx[ii,j,k])*G[ii+1,j,k])
            end
            D[jj,kk] = acc
        end
    end
    D
end

"""Accumulates the WRAPPED increment per line: 2pi jumps count themselves."""
function _desenvolver!(Tmat, Dl, Dprev)
    @inbounds for i in eachindex(Tmat)
        d = Dl[i] - Dprev[i]
        d -= 2pi*oround_f(d/(2pi))
        Tmat[i] += d
        Dprev[i] = Dl[i]
    end
    Tmat
end

# ---------------------------------------------------------------------------
#  DIAGNOSTICS.  All ADDITIVE: none enters the dynamics.
# ---------------------------------------------------------------------------
#  Why they exist (review of 24-Aug-2026): three real findings survived for
#  months ONLY because these quantities were not saved:
#
#   A8  <|Psi_a|^2> per band.  rho_pi/rho_sigma <= 0.05 had been inferred
#       ASSUMING |Psi_sigma|^2 = 1 (its zero-field value).  Measured it is
#       0.64 and the true ratio is 0.46: a factor of NINE.
#   A1  Split of the current between sample and vacuum.  The self-field is a
#       BC written on the WHOLE z face, so the enclosed current scales with
#       the width of the DOMAIN, not of the sample.  The fraction crossing the
#       sample swings from -20 % to +45 % in a snapshot: the AVERAGE is needed.
#   A5  N_cross and V were measured over DIFFERENT WINDOWS —the count over the
#       whole record, the slope over the second half—.  Both counts are saved.
#   B6  Unwrapping wraps the INCREMENT, so what matters is
#       |V|*dt_cycle < pi.  Measured: margin x527 at the rung that decides
#       Ic+, but x1.3 up the depairing ramp.
# ---------------------------------------------------------------------------
# |Ψ|² and θ per z plane. Only with heating. The faces (k=z1, k=z2)
# against the centre tell whether π switches off through the thickness.
function _calor_por_z(s::State, p::Params, J_cur)
    isempty(p.out_dir) && return
    path = joinpath(p.out_dir, "calor_z.csv")
    nuevo = !isfile(path)
    open(path, "a") do io
        nuevo && println(io, "J,k,ps1,ps2,Tmed,Tmax,Tclamped")
        for k in p.z1:p.z2+1
            n = 0
            a1 = 0.0; a2 = 0.0; aT = 0.0; tM = 0.0
            for j in p.y1:p.y2+1, i in p.x1:p.x2+1
                a1 += abs2(s.F1[i,j,k])
                a2 += abs2(s.F2[i,j,k])
                tk = p.Temp[i,j,k]
                aT += tk
                tM = max(tM, tk)
                n += 1
            end
            n == 0 && continue
            @printf(io, "%.6f,%d,%.6e,%.6e,%.6e,%.6e,%.0f\n",
                    J_cur, k, a1/n, a2/n, aT/n, tM, p.heat_diag[2])
        end
    end
    # A single point: the hottest node, in case the average hides a hot spot.
    hi = hi_j = hi_k = 0
    tH = -1.0
    for k in p.z1:p.z2+1, j in p.y1:p.y2+1, i in p.x1:p.x2+1
        tk = p.Temp[i,j,k]
        if tk > tH
            tH = tk; hi = i; hi_j = j; hi_k = k
        end
    end
    hot = joinpath(p.out_dir, "calor_hot.csv")
    nuevo_h = !isfile(hot)
    open(hot, "a") do io
        nuevo_h && println(io, "J,i,j,k,T,ps1,ps2")
        if hi > 0
            @printf(io, "%.6f,%d,%d,%d,%.6e,%.6e,%.6e\n",
                    J_cur, hi, hi_j, hi_k, tH,
                    abs2(s.F1[hi,hi_j,hi_k]), abs2(s.F2[hi,hi_j,hi_k]))
        end
    end
end

# Total current and current through the sample across the mid plane x = Nx/2.
# j_x = kappa^2 [ (hz(j)-hz(j-1))/dy - (hy(k)-hy(k-1))/dz ] at node (j,k),
# where phix lives.  The solver evaluates it only for j=2..Ny, k=2..Nz: THAT is
# the range that carries current.
function _midplane_current(s::State, p::Params)
    (; Nx, Ny, Nz, y1, y2, z1, z2, dy, dz) = p
    ii = oround(Nx/2)
    Itot = 0.0; Isam = 0.0
    @inbounds for kk in 2:Nz, jj in 2:Ny
        jx = p.ks2*((s.hz[ii,jj,kk] - s.hz[ii,jj-1,kk])/dy -
                    (s.hy[ii,jj,kk] - s.hy[ii,jj,kk-1])/dz)
        Itot += jx
        (y1 <= jj <= y2+1 && z1 <= kk <= z2+1) && (Isam += jx)
    end
    (Itot*dy*dz, Isam*dy*dz)
end

function _diagnosticos_v13!(jp, s::State, p::Params, g::Geometry, H, J_cur,
                            V_cur, dt_cyc, wq, Tser, ps1_arr, ps2_arr,
                            rhos_arr, lamef_arr, rpi_arr, Itot_arr, Isam_arr,
                            nslipw_arr, thmax_arr, dphcyc_arr;
                            ItSer=Float64[], IsSer=Float64[], mlast=0)
    (; Nx, Ny, Nz, x1, x2, y1, y2, z1, z2, dx, dy, dz, mr) = p

    ps1 = meanlin(abs2(s.F1[i,j,k]) for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1)
    ps2 = meanlin(abs2(s.F2[i,j,k]) for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1)
    ps1_arr[jp] = ps1; ps2_arr[jp] = ps2
    p.use_heat && _calor_por_z(s, p, J_cur)
    rhos = ps1 + ps2/mr                      # MEASURED rho_s, not assumed
    rhos_arr[jp] = rhos
    lamef_arr[jp] = p.kappa_s/sqrt(max(rhos, eps()))
    rpi_arr[jp] = (ps2/mr)/max(ps1, eps())

    # mean over the second half of the point (one instant if there is no series)
    if mlast >= 2 && length(ItSer) >= mlast
        r = (mlast ÷ 2 + 1):mlast
        Itot = sum(@view ItSer[r]) / length(r); Isam = sum(@view IsSer[r]) / length(r)
    else
        Itot, Isam = _midplane_current(s, p)
    end
    Inom = J_cur*((z2-z1+1)*dz)*((y2-y1+1)*dy)
    if abs(Inom) > 0
        Itot_arr[jp] = abs(Itot)/abs(Inom)
        Isam_arr[jp] = abs(Isam)/max(abs(Itot), eps())
    end

    length(wq) >= 3 && (nslipw_arr[jp] = abs(Tser[wq[end]] - Tser[wq[1]])/(2pi))

    thmax_arr[jp] = max(maximum(abs, s.thetax), maximum(abs, s.thetay),
                        maximum(abs, s.thetaz))
    dphcyc_arr[jp] = abs(V_cur)*dt_cyc
    if p.verbose && dphcyc_arr[jp] > 0.5pi
        @printf("    [B6] WARNING: |V|*dt_cycle = %.3f rad, more than half pi.\n      Phase unwrapping wraps increments and stops being reliable\n      near pi=%.3f: the crossing count of THIS point may be\n      undersampled. Lower lMax (not dt).\n"
,
                dphcyc_arr[jp], pi)
    end
    nothing
end

# ---------------------------------------------------------------------------
#  TUI / clean stop / current history
# ---------------------------------------------------------------------------
function _job_root(out_dir::AbstractString)
    isempty(out_dir) && return ""
    d = out_dir
    for _ in 1:8
        (isfile(joinpath(d, "STATUS.txt")) || isfile(joinpath(d, "job.json"))) && return d
        parent = dirname(d)
        parent == d && break
        d = parent
    end
    out_dir
end

function _stop_pedido(p::Params)
    isempty(p.out_dir) && return false
    d = p.out_dir
    for _ in 1:8
        isfile(joinpath(d, "STOP.txt")) && return true
        parent = dirname(d)
        parent == d && break
        d = parent
    end
    false
end

function _append_j_history(p::Params, label, J, V, conv, it, Nv1, Nv2, ps2)
    isempty(p.out_dir) && return
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    line = @sprintf("%s,%s,%.6f,%.8e,%s,%d,%d,%d,%.6f\n",
                    stamp, label, J, V, string(conv), it, Nv1, Nv2, ps2)
    hdr = "time,pol,J,V,conv,it,Nv_s,Nv_p,psi_p2\n"
    paths = String[joinpath(p.out_dir, "output", "j_history.csv")]
    root = _job_root(p.out_dir)
    isempty(root) || push!(paths, joinpath(root, "j_history.csv"))
    for fn in unique(paths)
        mkpath(dirname(fn))
        # the cycle length next to the history, so that crossings per line
        # (|V| it dt lMax / 2π) can be counted while the run is still going
        info = joinpath(dirname(fn), "run_info.txt")
        isfile(info) || open(info, "w") do io
            @printf(io, "dt=%.10e\nlMax=%d\nt_cycle=%.10e\n", p.dt, p.lMax, p.dt * p.lMax)
        end
        nuevo = !isfile(fn)
        open(fn, "a") do io
            nuevo && write(io, hdr)
            write(io, line)
        end
    end
    if !isempty(root)
        open(joinpath(root, "STATUS.txt"), "w") do io
            @printf(io, "%s  %s J=%+.3f V=%+.4e conv=%s Nv=%d/%d\n",
                    stamp, label, J, V, string(conv), Nv1, Nv2)
        end
    end
    nothing
end

# ---------------------------------------------------------------------------
#  CHECKPOINT
# ---------------------------------------------------------------------------
"""
    save_checkpoint(p, s, label, jp, V_arr, J_arr; VPhi_arr, Qimb)

Writes `restart/state_Jpos.mat` or `state_Jneg.mat` (text `.mat`): Ψ of both
bands, the link phases, the fields, the point index and the V(J) so far.
With heat on it also stores the temperature field θ, and with a relaxing
residual (tau_Q > 0) the residual Q, so that a resumed run continues with
them instead of restarting them from the bath and from zero.
"""
function save_checkpoint(p::Params, s::State, label::String, jp::Int,
                            V_arr, J_arr; VPhi_arr=nothing, Qimb=nothing)
    isempty(p.out_dir) && return nothing
    ckl = replace(replace(label, "+" => "pos"), "-" => "neg")
    ruta = joinpath(p.out_dir, "restart", "state_$(ckl).mat")
    campos = Pair{String,Any}["G1" => s.G1, "G2" => s.G2,
        "thetax" => s.thetax, "thetay" => s.thetay, "thetaz" => s.thetaz,
        # link phases of the previous step (after the swap of [I] they are in phi):
        # with them a resumed run rotates the links of its first step as if it had
        # not stopped (pi_substeps).  Optional on load.
        "thetax_prev" => s.phix, "thetay_prev" => s.phiy, "thetaz_prev" => s.phiz,
        "hx" => s.hx, "hy" => s.hy, "hz" => s.hz,
        "jp" => Float64(jp), "V_arr" => reshape(V_arr, :, 1),
        "J_arr" => reshape(J_arr, :, 1)]
    # VPhi is a diagnostic: Phi is recomputed every substep and is not needed to restart
    VPhi_arr === nothing || push!(campos, "VPhi_arr" => reshape(VPhi_arr, :, 1))
    p.use_heat && push!(campos, "Temp" => p.Temp)
    Qimb === nothing || push!(campos, "Qimb" => Qimb)
    write_octave_text(ruta, campos...)
    ruta
end

"""
    load_checkpoint(ruta, p, mode) -> (State, jp, V_arr, J_arr)

Rebuilds a full `State` from a saved `.mat` (text or binary v7), and restores
θ and Q when the checkpoint has them and the run uses them.
"""
function load_checkpoint(ruta::AbstractString, p::Params, mode::NumMode)
    w = read_state(ruta)
    s = _armar_estado(p, ComplexF64.(w["G1"]), ComplexF64.(w["G2"]),
                      Float64.(w["thetax"]), Float64.(w["thetay"]),
                      Float64.(w["thetaz"]), Float64.(w["hx"]),
                      Float64.(w["hy"]), Float64.(w["hz"]), mode)
    # temperature and charge residual, when the checkpoint has them and the run uses them
    p.use_heat && haskey(w, "Temp") && size(w["Temp"]) == size(p.Temp) && (p.Temp .= w["Temp"])
    haskey(w, "Qimb") && size(w["Qimb"]) == size(p.Q_restart) && (p.Q_restart .= w["Qimb"])
    # previous-step links (older checkpoints do not have them: then U = V, frozen first step)
    for (k, ph, U) in (("thetax_prev", s.phix, s.Ux), ("thetay_prev", s.phiy, s.Uy), ("thetaz_prev", s.phiz, s.Uz))
        if haskey(w, k) && size(w[k]) == size(ph)
            ph .= w[k]; U .= emiphi.(Ref(mode), ph)
        end
    end
    jp = Int(w["jp"])
    (s, jp, vec(Float64.(w["V_arr"])), vec(Float64.(w["J_arr"])))
end
