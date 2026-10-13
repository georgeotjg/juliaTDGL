# ============================================================================
#  config.jl — EVERY KNOB IN ONE PLACE
#
#  Each field carries a short note on what it does and, where it matters,
#  why its default is what it is.
#
#  Not here, on purpose:
#    plotting              : figures are made from the saved state
#                            (scripts/plot_summary.py, vtk_from_checkpoint.jl).
#    parallel polarities   : the useful parallelism is over z planes
#                            (`use_threads`) and over independent processes.
#    kappa_list            : an outer sweep, i.e. a loop over `run_case`.
# ============================================================================

Base.@kwdef mutable struct Config
    # --- mesh ----------------------------------------------------------------
    Nx_sc::Int = 120;  Ny_sc::Int = 40;  Nz_sc::Int = 10
    dx::Float64 = 0.25; dy::Float64 = 0.25; dz::Float64 = 0.25
    buf::Int = 1                      # vacuum cells per face

    # --- physics -------------------------------------------------------------
    kappa_s::Float64 = 2.0            # BENCHMARK point of I–III and IV, not MgB2
    kappa_p::Float64 = 2.0            # enters only the CFL condition
    T_red::Float64 = 0.0
    beta1::Float64 = 1.0; beta2::Float64 = 1.0
    a2n::Float64 = 0.59               # |Psi2_eq|^2 = Tc_pi/Tc_sigma = 23/39
    m_ratio::Float64 = 1.50           # m*_pi/m*_sigma (only in the pi kinetic term)
    xi_ratio::Float64 = 0.0           # >0 fixes xi_pi/xi_sigma and RECOMPUTES m_ratio
    gamma12::Float64 = 0.001          # interband Josephson coupling
    # Gamma_pi/Gamma_sigma.  In THIS scheme (v13.3) it divides the three
    # terms of the pi band, so it is a true relaxation rate and moves no
    # static quantity.  CAUTION when reading the archive: up to v13.2 it
    # divided only the condensation term and then DID move xi_pi and Hc2_pi
    # (1.300*sqrt(Gr) and 0.5917/Gr).  See `hc2_pi` and `hc2_pi_f21` in
    # params.jl, and failure mode M16 of the methods paper (Zenodo, see README).
    gamma_ratio::Float64 = 1.0
    H_const::Float64 = 0.30           # applied H [Hc2] in VI mode

    # --- current -------------------------------------------------------------
    J_list::Vector{Float64} = [0.10:0.10:0.80; 0.85:0.025:1.35; 1.45; 1.60]
    J_max::Float64 = 1.6
    dJ_fine::Float64 = 0.005; dJ_coarse::Float64 = 0.02; J_tr::Float64 = 0.40
    current_axis::String = "x"
    inject_mode::String = "selffield"  # 'selffield' (Ampere) | 'phase' (clamp v4..v10)
    inject_gain::Float64 = 1.0
    inject_standoff::String = "domain" # [C1] do NOT use 'sample' with buf > 1
    h_tilt::Float64 = 0.0

    # --- convergence and voltmeter -------------------------------------------
    V_threshold::Float64 = 1e-4
    eps_conv::Float64 = 1e-4
    eps_conv_seed::Float64 = 5e-4      # only for the first point (J=0)
    tie_seed_tolerance::Bool = false   # true: the J=0 seed uses min(eps_conv_seed, eps_conv)
    # Fixed-point tolerance as a RATE [1/t_GL].  eps_conv is per cycle, so the
    # same value is a different physical criterion whenever dt or lMax change
    # (x38 looser at kappa = 25 than at kappa = 2 with the defaults).  If > 0,
    # the tolerance per cycle is eps_conv_rate*lMax*dt and eps_conv is ignored.
    # 0 = per cycle, as always.
    eps_conv_rate::Float64 = 0.0
    # true: eps_conv is read per cycle of the 1.0.0 step (legacy bounds, explicit
    # curl-curl) and turned into a rate, so a larger step keeps the fixed-point
    # criterion of the explicit scheme in physical time.  Ignored if
    # eps_conv_rate > 0.  Config(:prod) sets it.
    eps_conv_legacy::Bool = false
    # Curl-curl term of the field equation at the new time level (see
    # src/maxwell_implicit.jl).  Removes the Maxwell bound dl/(4 kappa^2) from
    # the step; the other bounds stay.  Off = the explicit scheme, to the bit.
    maxwell_implicit::Bool = false
    # Multiplies the time step chosen by the stability bounds (0 < dt_scale <= 1),
    # for convergence tests in dt.  1 = the step as always.
    dt_scale::Float64 = 1.0
    # true: the historical step (dt1 with the gama factor, dt3 = 0.03 dl); both
    # come from the generalised TDGL the code descends from and bound no
    # equation integrated here.  false: only the stability bounds of the
    # equations this code integrates (diffusion of sigma and pi, Maxwell when
    # explicit, heat or Q when on).  Accuracy is then a matter of dt_scale.
    cfl_legacy::Bool = true
    # Substeps of the pi band per step (multirate).  pi is advanced pi_substeps
    # times with dt/pi_substeps, with sigma frozen in the coupling and the links
    # rotated at the previous step's rate (pi_link_extrapolation), and the pi
    # diffusion bound on dt is relaxed by that factor.  The step is then set by
    # sigma: use dt_scale <= 0.5 (see the warning in params.jl).
    # 0 = automatic: as many as the other bounds need.  1 = the scheme as
    # always, to the bit.  Substeps need the plain kernel: with heat, T_physical
    # or drag, 0 means one substep, and n > 1 is an error.
    pi_substeps::Int = 1
    # With pi substeps: rotate the links inside the step at the rate of the
    # previous step (see pi_substeps! in src/kernel.jl) instead of freezing them.
    pi_link_extrapolation::Bool = true
    # BLAS threads set when a sweep starts (0 = leave as is).  The separable
    # field solve is the only BLAS user; with many processes, one thread each
    # avoids oversubscribing the cores.  Results do not depend on it.
    blas_threads::Int = 0
    # Substeps of the sigma band per step, same scheme (1 = off, to the bit).
    # The sigma bound is relaxed by that factor; with dt_scale = 0.5 each sigma
    # substep stays at half its bound.  For accuracy studies of larger steps.
    sigma_substeps::Int = 1
    iter_min::Int = 40                 # floor in CYCLES
    lMax::Int = 40                     # substeps per cycle
    mMax::Int = 4000
    t_min_GL::Float64 = 20.0           # [t_GL] physical-time floor per point
    t_min_GL_res::Float64 = 1200.0     # floor for a RESISTIVE point
    delta_v::Int = 8                   # voltmeter offset into the bulk
    use_Vconv::Bool = true
    V_win::Int = 40
    V_win_min_res::Int = 400
    tol_V::Float64 = 2e-3
    tol_res::Float64 = 0.15            # STALLED residual => limit cycle
    # Closing tests by physical content, not per cycle (off = as always).  The
    # per-cycle tests depend on the cycle length in t_GL: a prod cycle is 8 times
    # longer than an explicit one, its resistive points never "stalled" and ran to
    # the ceiling, while the explicit ones closed near onset by a spurious 'F'.
    # On: a resistive point closes after t_min_GL_res once it has ten or more
    # phase crossings per line (the verdict rule), and a fixed point ('F') is not
    # accepted while the second half carries one or more crossings per line.
    conv_by_time::Bool = false
    eps_relax::Float64 = 1e-3
    floor_k::Float64 = 160.0
    floor_mult::Float64 = 5.0

    # --- protocol ------------------------------------------------------------
    run_mode::String = "VI"            # 'VI' | 'MH'
    H_list::Vector{Float64} = collect(0:0.02:1.00)
    MH_updown::Bool = false

    # --- geometry ------------------------------------------------------------
    GEOM_MODE::String = "WEDGE"
    # GEOM_MODE="CUSTOM" | "CUSTOM_REF": shape read from this file (src/shapes.jl)
    geom_file::String = ""
    defect_model::String = "soft"      # 'soft' = mask f | 'hard' = Psi=0
    wedge_depth::Float64 = 0.35        # fraction of Ny_sc
    wedge_half::Float64 = 0.20         # fraction of Nx_sc
    wedge_depth2::Float64 = 0.00       # second notch, WEDGE_BOTH only
    fgrad_flo::Float64 = 1.00; fgrad_fhi::Float64 = 1.50
    fgrad_nper::Int = 2; fgrad_dir::Int = +1
    inj_margin::Int = 6
    strict_margin::Bool = true
    x_shift::Float64 = 0.0; y_shift::Float64 = 0.0
    z_mode::String = "through"         # 'through' | 'half' | 'buried'
    # Rotation of n-gons (HEX / STAGGER / FUNNEL / HEX_STAG) [rad].
    # NaN = the catalogue angle (HEX/STAGGER 0, FUNNEL π/2).
    prism_phi::Float64 = NaN

    # --- audit flags (the defaults reproduce v11) ----------------------------
    fix_M_window::Bool = false         # magnetisation window symmetric under the mirror (mode M4)
    explicit_h_bc::Bool = false        # also write h on the faces the reference code leaves untouched
    strict_two_rung::Bool = false      # extract_Ic: NaN, not the first rung above V_threshold, when no two consecutive rungs are above it
    v13_diag::Bool = true              # additive v13 diagnostics
    dump_split::Bool = false           # sample/vacuum split of dA/dt
    dump_band_lines::Bool = false      # per-line instrumentation

    # --- execution -----------------------------------------------------------
    early_exit::Bool = false
    # Every N current points, writes the central z plane of |Psi_s|^2 and
    # |Psi_p|^2 to output/snapshots/ (ASCII PGM + CSV).  0 = off.
    # The .m kept it in snaps.mat; here they are separate files that can be
    # opened while the run goes on.  Always jp=1 and the last point.
    snap_every::Int = 8
    restart_pos::Bool = false; restart_neg::Bool = false
    vtrace_every::Int = 0; vtrace_Jmin::Float64 = 0.0
    selftest::Bool = true              # vectorised kernel vs reference loop
    preflight::Bool = false
    force_long::Bool = false
    sec_per_step::Float64 = 0.035
    run_tag::String = ""
    out_dir::String = ""
    verbose::Bool = true
    # Parallelises the three blocks worth it ([A], [B], [E]) over z planes.
    # Each plane writes different memory and there is no reduction, so the
    # result is BIT FOR BIT the same with any number of threads: the
    # acceptance test still holds.  Acts only if julia started with -t N.
    # On a hybrid laptop processor (most cores are efficiency cores) it scales
    # poorly; for a campaign, one single-threaded process per core is better.
    use_threads::Bool = true
    # VTK dump for ParaView.  0 = none.  n>0 writes
    # output/vtk/estado_<tag>_<pol>_J<...>.vtk every n current points,
    # cropped to the sample.  Does not touch the dynamics.  See src/vtk.jl.
    vtk_every::Int = 0

    # --- GUARDS for long unattended runs -------------------------------------
    # The .m has none: if the scheme diverges (CFL violated, kappa raised
    # without lowering dt, m_r < 1 without the dt1p guard) the loop runs to
    # mMax producing NaN, and the run is found broken hours later.  These
    # two stop in time and say why.
    guard_psi::Float64 = 5.0   # abort if max|Psi| exceeds it. 0 = off.
                               # Equilibrium is |Psi|=1; 5 is already divergence.
    guard_hours::Float64 = 0.0 # wall-clock budget in hours. 0 = no limit.
                               # When spent it closes the current point, SAVES
                               # THE CHECKPOINT and stops: the run resumes
                               # with restart_pos/restart_neg.  Meant for
                               # cluster queues with a time limit.

    # --- v14: scalar potential ----------------------------------------------
    # With use_phi = false the kernel is BIT FOR BIT v13.3.  That equality is
    # the two-sided certification that lets the whole archive carry over.
    use_phi::Bool = false
    # Gauge of the Phi step when lambda_TF=0 and tau_Q=0 (Phi pure gauge).
    # Changing it changes the GAUGE, not the physics: invariant observables
    # come out equal with both, and that is a test (test_phi.jl).
    #
    # Default :dirichlet = the Octave scripts and bank 3 (regression).
    # The PHYSICAL gauge of this setup is :neumann: there are no electrodes,
    # the current enters through the Ampere self-field.  Dirichlet Phi=0 on
    # the box is an arbitrary ground and moves V_Phi by a factor ~15.  See
    # the guide (docs/solver_guide.pdf, scalar potential).  The default is kept
    # so as not to break the numerical anchor.
    phi_bc::Symbol = :dirichlet        # :dirichlet | :neumann
    # Method of the Phi solve.  `:separable` diagonalises the Laplacian into
    # three small 1-D eigenproblems and solves with BLAS-3; it is DIRECT AND
    # EXACT, not an approximation, and gives the same solution as `:lu` up to
    # rounding.  It exists because the Phi step was 80 % of a v14 substep.
    phi_method::Symbol = :separable    # :separable | :lu
    # The (1-T) factor multiplying the supercurrent in block [C].
    # The correct equation does NOT carry it.  At T = 0 it equals 1 and goes
    # unnoticed, which is why it survived.  true = v13 behaviour (and bit for
    # bit with the archive); **v14 at T != 0 must set it to false**.  That is
    # the decision the [v14-B] warning asked to take on purpose rather than inherit.
    f22_supercurrent::Bool = true
    # [Physical T, 2-oct-2026] T_red as a uniform, fixed bath temperature:
    # α_σ = (1-T) f and α_π = (a2n-T) f (the heat path with θ ≡ T_red),
    # j_s without the (1-T) of M17, and m_ratio fixed with ξπ/ξσ at T = 0 so
    # that a sweep in T does not change the π mass. false = the code of
    # IV/V, where σ does not feel T_red. At T_red = 0 both paths agree.
    T_physical::Bool = false
    # --- v14.1: physics that breaks gauge equivalence (Article IV) -----------
    # 0 = off.  With both at zero, v14 is the Coulomb projection
    # (Phi = gauge) and the controls of test_phi.jl still hold to the bit.
    #
    # lambda_TF > 0 [xi_sigma]: Thomas-Fermi screening.  The operator
    # becomes  D∘G - I/lambda_TF^2.  Same cost (only the spectrum changes).
    # Residual: D(delta') + Phi/lambda_TF^2 = 0.  Dirichlet and Neumann are
    # no longer equivalent gauges.
    lambda_TF::Float64 = 0.0
    # tau_Q > 0 [t_GL]: charge imbalance.  Field Q on the nodes.
    # Implicit Euler for tau_Q dQ/dt + Q = D(delta'):
    #     Q <- (Q + (dt/tau_Q) D(delta')) / (1 + dt/tau_Q)
    # Attractor Q -> D(delta'), same units as Phi_projection.
    # The rotation of Psi uses mu = Phi + Q.  Q does not enter the Poisson
    # problem for Phi (that was an unstable loop).  WITHOUT lambda_TF the
    # source D(delta') vanishes (exact Coulomb) and tau_Q does nothing: there
    # is a warning.  The two together are Article IV; sigma_n(r) is left out.
    tau_Q::Float64 = 0.0
    # Mixed-gradient (Silaev–Babaev) drag between the bands. Default OFF.
    use_drag::Bool = false
    nu_drag::Float64 = 0.0
    # Diffused Q. If false, the Q of v14 is the local Euler update.
    # Requires use_phi and tau_Q>0. D_Q in ξσ²/t_GL.
    use_Qss::Bool = false
    D_Q::Float64 = 0.25
    # Temperature field θ(r). false: scalar T_red. true: heating from the
    # order-parameter dissipation + diffusion + α(θ) per node.
    # Coefficients in t_GL, ξσ.
    use_heat::Bool = false
    D_T::Float64 = 0.25
    gamma_bath::Float64 = 0.2
    eta_J::Float64 = 0.05
    # Normal Joule term σn|E|² in the heat source. σn = 1 is that of block
    # [C] (∂tA = j_s − κ²∇×∇×A). Requires use_heat and use_phi=false.
    # Default OFF: the Article VII runs up to 1-oct do not carry it.
    use_joule::Bool = false
    # Cells from each injection face where heat (|∂tΨ|² and, with
    # use_joule, E²) is not counted. There the current enters as normal
    # current and converts into supercurrent within ~ξ: E≠0 and the phase
    # rotates even in the Meissner state (measured 1-oct, κ=25, J=0.40:
    # ⟨|∂tΨ|²⟩=4.5e-3, ⟨E²⟩=4.4e-3). That heat is contact heat; in a device
    # the leads remove it. Same criterion as the voltmeter (delta_v).
    # 0 = the whole sample, as in the Article VII runs up to 1-oct.
    heat_margin::Int = 0

    # --- measurement and padding options (defaults reproduce the reference) ---
    # Phase unwrapping every `unwrap_every` substeps instead of once per cycle
    # (0 = once per cycle). A per-line increment above pi in one cycle is
    # aliased; unwrapping more often removes that risk at a small cost.
    unwrap_every::Int = 0
    # Keep the state of every closed current point in restart/rungs/, not only
    # the last one of each polarity, for analysis after the run (edge fields,
    # current maps).  Off by default: one checkpoint per rung.
    keep_rung_states::Bool = false
    # Checkpoint format: "jld2" (compressed binary, exact, about a quarter of the text size, 5-10 times
    # faster) or "octave" (the text .mat of the releases up to 1.1, exact). Either is read back by
    # `read_state` and resumed by `run_case`; `checkpoint_path(dir, label)` finds whichever exists.
    checkpoint_format::String = "jld2"
    # Vacuum padding per axis; -1 = use `buf`. Allows separating the padding
    # across the faces y (edge field) from that across the faces z.
    bufx::Int = -1; bufy::Int = -1; bufz::Int = -1
end

"""
    Config(:paper; kwargs...)

The measuring protocol of the methods paper (Robles-Calderon 2026, Sec.
"Measurement protocol"): fixed point at 1e-5 per cycle, seed at 5e-4,
physical-time floors of 1500 t_GL (fixed point) and 2000 t_GL (resistive),
convergence by stationarity of V allowed, V_threshold 1e-4 for comparison,
and a ceiling of 600000 cycles.  Any keyword overrides the preset.  The
library defaults are not changed.

    Config(:prod; kwargs...)

The production scheme of release 1.1, recommended for new work: implicit
curl-curl, pi substeps, half the sigma bound, closing by the verdict rule in
physical time, one BLAS thread, and the fixed-point criterion of the explicit
scheme in physical time.  Same onset brackets as 1.0.0; x3 per t_GL at
kappa = 2, ~x100 at kappa = 25.  With heat, T_physical or drag the pi band
takes one substep and only the curl-curl is implicit (~x30 at kappa = 25).  No
Phi.  Combine with :paper by passing its keywords.

    Config(:legacy; kwargs...)

The explicit scheme of release 1.0.0 and of the published runs: the library
defaults, named.
"""
function Config(preset::Symbol; kw...)
    if preset === :paper
        return Config(; eps_conv=1e-5, eps_conv_seed=5e-4, t_min_GL=1500.0,
                      t_min_GL_res=2000.0, use_Vconv=true, V_threshold=1e-4,
                      mMax=600_000, kw...)
    elseif preset === :prod
        # 1.1 production scheme: implicit curl-curl, stability-only step bounds,
        # pi substeps, half the sigma bound (accurate in strong flow), closing
        # tests in physical time, one BLAS thread per process, and the fixed-point
        # tolerance of the explicit scheme in physical time (eps_conv per cycle of
        # the 1.0.0 step: 40 x 6.6e-5 t_GL at kappa = 25, 40 x 2.518e-3 at kappa = 2
        # in the production box).  Validated against the explicit scheme: same onset
        # brackets at kappa = 2 and 25, flowing voltages within 0.16 % (CHANGELOG 1.1).
        return Config(; maxwell_implicit=true, cfl_legacy=false, pi_substeps=0,
                      dt_scale=0.5, conv_by_time=true, blas_threads=1,
                      eps_conv_legacy=true, kw...)
    elseif preset === :legacy
        # the explicit scheme of release 1.0.0 and of the published runs: the library defaults
        return Config(; kw...)
    end
    error("unknown preset: $preset (use :paper, :prod or :legacy)")
end

"""
    Config(; kwargs...)

Built like the override `struct(...)` of the .m:

    cfg = Config(GEOM_MODE="WEDGE", Nx_sc=12, Ny_sc=8, Nz_sc=4, H_const=0.5)
"""
Config

# --- Octave-compatible rounding ---------------------------------------------
# Octave's `round` breaks ties AWAY from zero; Julia's breaks ties to EVEN.
# round(0.5) gives 1 in Octave and 0 in Julia.  All the geometry is indexed
# with `round`, so the difference moves defects by one cell.  This is not a
# subtlety: a single cell out of 23 produced 9.4 % in the voltage (see the
# note of the WEDGE_BOTH case in the .m).
@inline oround(x::Real) = round(Int, x, RoundNearestTiesAway)
@inline oround_f(x::Real) = round(x, RoundNearestTiesAway)
