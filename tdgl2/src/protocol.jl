# ============================================================================
#  protocol.jl — a current ladder and the verdict rule
#
#  Each point starts from the state the previous one left (sequential seeding: it is what gives
#  the hysteresis of the V-I curve; breaking it makes a point depend on how it was reached, mode
#  M13).  A point is integrated in cycles of lMax steps until it closes:
#    'F'  fixed point: |Psi_mu| and h change by less than eps per cycle, after the floor t_min_GL,
#         and the second half of the point carries less than one phase crossing per line in every
#         band (a flowing state can look fixed for one short cycle);
#    'V'  resistive: after the floor t_min_GL_res, the second half carries ten or more crossings
#         per line (the verdict rule) in band 1;
#    '*'  the ceiling mMax.
#  The voltage is the Josephson one: the time derivative of the gauge-invariant phase
#  arg(Psi_i^* U Psi_j) summed along each measuring line between two planes delta_v cells inside
#  the contacts, unwrapped line by line (a vortex crossing one line jumps it by 2 pi), averaged
#  over the lines, per band.  A rung is silent with fewer than one crossing per line over its
#  integration, dissipating with ten or more, undecided in between (bracket_Ic).
# ============================================================================

"""
    Protocol(; ...)

The measuring protocol.  Defaults: the windows of the κ = 25 campaigns of release 1.x (fixed-point
floor 400 τ, resistive floor 1584 τ), the fixed-point tolerance of release 1.1 production
(1e-4 per cycle of the release-1.0 step, i.e. a rate in physical time).
"""
Base.@kwdef struct Protocol
    lMax::Int = 40                   # steps per cycle
    iter_min::Int = 40               # floor in cycles
    mMax::Int = 500_000              # ceiling in cycles
    t_min_GL::Float64 = 400.0        # physical-time floor of a point [t_GL]
    t_min_GL_res::Float64 = 1584.0   # floor of a resistive point [t_GL]
    eps_conv::Float64 = 1e-4         # fixed-point tolerance per cycle of the release-1.0 step
    eps_rate::Float64 = 0.0          # > 0: the tolerance per t_GL instead (per cycle: eps_rate lMax dt)
    eps_seed::Float64 = 5e-4         # tolerance of the first point (the J = 0 seed), per cycle
    delta_v::Int = 8                 # voltmeter planes, cells inside the contacts
    V_win::Int = 40                  # cycles of the instantaneous-voltage mean
    V_win_min_res::Int = 400         # least window of a resistive slope, cycles
    use_Vconv::Bool = true           # allow the resistive route
    v_margin::Float64 = 1.1          # close 'V' at v_margin × 10 crossings (end-point estimate); 1.0 = 1.2.
                                     # At 1.0 the slope the verdict uses lands at 10 ± noise (9.8–10.1)
    f_strict::Bool = true            # closes as the verdict counts: 'F' only below f_margin crossings and
                                     # 'V' only at ≥ 10 by the slope (a long-period flow near onset passed
                                     # the end-point 11 with 9.75 by the slope); for 'F', with margin
                                     # (slope crossings < f_margin); false = 1.2, where a slow flow near
                                     # onset could close 'F' at 1–2.5 crossings
    f_margin::Float64 = 0.5          # at 1 the close lands at 0.99 by construction (as 'V' did at 10)
    guard_psi::Float64 = 5.0         # abort above this max|Psi| (0 = off)
    keep_rung_states::Bool = false   # keep the state of every rung in restart/rungs/
    stop_after_V::Int = 0            # > 0: the ladder ends after this many 'V' points in a row (0 = walk all J)
    t_confirm_GL::Float64 = 0.0      # > 0: the onset by the delay.  At the first 'V' after an 'F', the 'F' rungs
                                     # below it are continued at their current, from copies of their closed
                                     # states, to this time in all: one that crosses becomes 'D' (a flow that
                                     # starts after a delay) and the next one down is continued, until one stays
                                     # silent (delay_bracket).  Near a saddle-node onset the delay diverges and a
                                     # rung just above it closes 'F' on any finite window (κ = 2: 1600–2600 τ
                                     # against windows of 400).  The ladder itself is not changed.
    confirm_max::Int = 4             # at most this many rungs are continued
    n_frac::Float64 = 0.02           # 'N': a rung that would close 'F' with ⟨|Ψ₁|²⟩ on the block below n_frac of the largest
                                     # of the 'F' rungs before it is a sample gone normal, not a silent one (the normal state is a fixed point
                                     # with no phase to cross: with u or γ̃ large the onset can be a collapse).
                                     # It counts as dissipating in brackets, delay_bracket and stop_after_V. 0 = off.  Not 0.5:
                                     # a silent state with the π band normal and σ at |Ψ|² = 0.26 (the golden ladder of 1.2: the
                                     # two-band notched box at κ = 25, J = 1.2) is a legitimate 'F'.
    verbose::Bool = false
end

"""
    dt_release10(md, gr) -> Float64

The step release 1.0 would take (its inherited bounds).  Used only to translate a tolerance
given "per cycle of the 1.0 step" into one in physical time; 2.0 does not step with it.
"""
function dt_release10(md::Model{N}, gr::Grid) where N
    l = dl(gr)
    dt1 = 5.79 * l / (4 * sqrt(1 + 10.0^2))
    bands = ntuple(µ -> µ == 1 ? dt1 : dt1 * md.bands[µ].m * md.bands[µ].Γ, N)
    dt2 = 1.00 * l / (4 * md.κ^2)
    dt3 = 0.03 * l / (4 * 0.50^2)
    0.99 * min(bands..., dt2, dt2, dt3)
end

eps_cycle(pr::Protocol, st::Stepper) =
    pr.eps_rate > 0 ? pr.eps_rate * (pr.lMax * st.dt) : pr.eps_conv * (st.dt / dt_release10(st.model, st.grid))

"""What a ladder of one polarity returns.  Per-band quantities are matrices (point × band)."""
struct Ladder
    label::String
    J::Vector{Float64}
    H::Vector{Float64}
    V::Vector{Float64}             # band-1 Josephson voltage, gated by the slip count
    Vb::Matrix{Float64}            # per-band slope over the closing window (ungated)
    Vh::Matrix{Float64}            # per-band slope over the second half of the point, gated per band
    Vstatic::Vector{Float64}       # mean of the instantaneous <dA/dt> over the last V_win cycles
    m::Vector{Int}                 # cycles
    conv::Vector{Char}
    res::Matrix{Float64}           # last residual of |Psi_mu|
    nslip::Vector{Float64}
    M::Vector{Float64}
    Nv::Matrix{Int}
    psi2::Matrix{Float64}          # <|Psi_mu|^2> on the block
    Iband::Matrix{Float64}         # supercurrent of each band through the mid plane (end of point)
    Itot::Vector{Float64}          # total current / nominal, mean over the second half
    Isam::Vector{Float64}          # share of it through the sample
    dmax::Vector{Float64}          # largest per-line phase increment in one cycle
    nbig::Vector{Int}
    Vf1::Vector{Float64}           # voltage over the lines with f = 1 throughout
    VA::Vector{Float64}            # rate of ∫A·dl (not gauge invariant; diagnostic)
    delay::Vector{Float64}         # t_confirm_GL: time of the first crossing of a continued rung (NaN: none)
    t_conf::Vector{Float64}        # t_confirm_GL: time the rung was followed in all (0: not continued)
    t_cycle::Float64
    state::State
end

"""Crossings per line of each rung, with the rate the verdict needs (second-half slope)."""
rung_rate(L::Ladder) = [isnan(L.Vh[q, 1]) ? L.V[q] : L.Vh[q, 1] for q in eachindex(L.V)]
rung_time(L::Ladder) = L.m .* L.t_cycle

# --- voltmeter helpers --------------------------------------------------------
function _plane_sum(s::State, gr::Grid, plano::Int, M)
    (; y1, y2, z1, z2) = gr
    tot = 0.0
    @inbounds for (kk, k) in enumerate(z1:z2+1), (jj, j) in enumerate(y1:y2+1)
        M[jj,kk] && (tot += s.φx[plano,j,k])
    end
    tot
end

function _theta_A(s::State, gr::Grid, vi::Int, vo::Int, line_ok)
    (; y1, y2, z1, z2) = gr
    tot = 0.0; n = 0
    @inbounds for (kk, k) in enumerate(z1:z2+1), (jj, j) in enumerate(y1:y2+1)
        line_ok[jj,kk] || continue
        for ii in vi:vo-1; tot += s.φx[ii,j,k]; end
        n += 1
    end
    n > 0 ? tot/n : 0.0
end

function _path_phase!(D, G, s::State, gr::Grid, vi::Int, vo::Int)
    (; y1, y2, z1, z2) = gr
    fill!(D, 0.0)
    @inbounds for (kk, k) in enumerate(z1:z2+1), (jj, j) in enumerate(y1:y2+1)
        acc = 0.0
        for ii in vi:vo-1
            acc += angle((conj(G[ii,j,k])*s.Vx[ii,j,k])*G[ii+1,j,k])
        end
        D[jj,kk] = acc
    end
    D
end

function _unwrap!(T, Dl, Dprev)
    @inbounds for i in eachindex(T)
        d = Dl[i] - Dprev[i]
        d -= 2pi*oround_f(d/(2pi))
        T[i] += d
        Dprev[i] = Dl[i]
    end
    T
end

# The verdict of a rung that would close now: fewer than one crossing per line over the integration,
# counted as the rung lines are (second-half slope × time, zero without a whole slip).
function _crossings_now(T1, Tm1, line_ok, m::Int, dt_cyc)
    sumlin(abs(Tm1[i]) for i in eachindex(Tm1) if line_ok[i]) / (2pi) < 1 && return 0.0
    wh = (m - m ÷ 2 + 1):m
    length(wh) < 3 && return 0.0
    th = (collect(wh) .- wh[1]) .* dt_cyc
    abs(pendiente(th, view(T1, wh))) * m * dt_cyc / (2pi)
end
# A fixed point also needs no slip since the first quarter of the point: the second-half slope misses
# a flow whose period exceeds half the elapsed time when its last slip fell in the first half (κ = 25,
# 0.368 after a flowing 0.364: slips every 230–350 τ, closed 'F' at 438 τ with the second slip at 450).
# The first quarter is left out so that the static phase change of the current step does not count.
function _late_crossings(T1, m::Int)
    q = max(1, m ÷ 4)
    m > q ? abs(T1[m] - T1[q]) * m / (m - q) / (2pi) : 0.0
end
_silent_now(T1, Tm1, line_ok, m::Int, dt_cyc, lim) =
    max(_crossings_now(T1, Tm1, line_ok, m, dt_cyc), _late_crossings(T1, m)) < lim

# --- the onset by the delay (Protocol t_confirm_GL) --------------------------------------------
_open_snap(st::Stepper) = st.open === nothing ? nothing :
    (st.open.count, deepcopy(st.open.θ), deepcopy(st.open.old), deepcopy(st.open.new))
function _open_back!(st::Stepper, sn)
    sn === nothing && return st
    ob = st.open; ob.count = sn[1]
    for c in 1:3
        copyto!(ob.θ[c], sn[2][c]); copyto!(ob.old[c], sn[3][c]); copyto!(ob.new[c], sn[4][c])
    end
    st
end

# Continues a copy of the closed rung c at its current for T t_GL; the time of its first crossing from
# the start of the rung, or NaN: the line-averaged band-1 path phase across the voltmeter passing 2π.
function _continue_rung(st::Stepper, c, T, lMax, line_ok, vi, vo, D, Dp)
    gr = st.grid
    s = deepcopy(c.s); _open_back!(st, c.open)
    st.heat === nothing || copyto!(st.heat.θ, c.θ)          # the temperature the rung closed with
    HJ = selffield(st, c.J); dtc = lMax * st.dt
    _path_phase!(Dp, s.G[1], s, gr, vi, vo)
    acc = zeros(size(D)); n = count(line_ok)
    for q in 1:ceil(Int, T / dtc)
        for _ in 1:lMax
            step!(s, st; H=c.H, HJ=HJ)
        end
        _path_phase!(D, s.G[1], s, gr, vi, vo)
        _unwrap!(acc, D, Dp)
        abs(sumlin(acc[i] for i in eachindex(acc) if line_ok[i]) / n) >= 2pi && return c.t + q * dtc
    end
    NaN
end

"""
    crossing_times(st, s, J; H, T, stop=3, delta_v=8, every=10) -> (times, t, diverged)

Integrates the state `s` (in place) at fixed current `J` and field `H` for up to `T` t_GL and returns the times at which
the line-averaged band-1 phase across the voltmeter (the planes `delta_v` cells inside the contacts, the lines that are
complete between them: the voltmeter of `ladder`) passes a further multiple of 2π, sampled every `every` steps; it stops
after `stop` crossings (0: never).  `t` is the time integrated, `diverged` whether the phase became NaN.  The delay before
the first crossing near a saddle-node onset grows as a/√(J − Jc): with the delays at two or more currents above the onset,
from the same silent state, `tools/delay_fit.py` gives Jc.  With `:open` the ramp of
the boundary continues from wherever `st` left it (`st.open.count = -1` restarts it from the state).
"""
function crossing_times(st::Stepper, s::State, J::Real; H::Real, T::Real, stop::Integer=3, delta_v::Integer=8,
                        every::Integer=10)
    gr = st.grid; mk = st.masks
    (; x1, x2, y1, y2, z1, z2) = gr
    vi = x1 + delta_v; vo = x2 - delta_v
    vi < vo || error("crossing_times: delta_v = $delta_v leaves no voltmeter")
    line_ok = [all(mk.τx[ii, j, k] > 0.5 for ii in vi:vo-1) for j in y1:y2+1, k in z1:z2+1]
    n = count(line_ok); n == 0 && error("crossing_times: no complete line between the voltmeter planes")
    D = zeros(size(line_ok)); Dp = zeros(size(line_ok)); acc = zeros(size(line_ok))
    HJ = selffield(st, J)
    _path_phase!(Dp, s.G[1], s, gr, vi, vo)
    times = Float64[]; next = 1; nst = round(Int, T / st.dt); q = 0
    while q < nst
        for _ in 1:every
            step!(s, st; H=Float64(H), HJ=HJ)
        end
        q += every
        _path_phase!(D, s.G[1], s, gr, vi, vo)
        _unwrap!(acc, D, Dp)
        ph = sumlin(acc[i] for i in eachindex(acc) if line_ok[i]) / n
        isfinite(ph) || return (times = times, t = q * st.dt, diverged = true)
        abs(ph) >= next * 2pi && (push!(times, q * st.dt); next += 1)
        stop > 0 && length(times) >= stop && break
    end
    (times = times, t = q * st.dt, diverged = false)
end

_stop_requested(dir::AbstractString) = !isempty(dir) && any(isfile(joinpath(d, "STOP.txt"))
    for d in unique([dir, dirname(dir), dirname(dirname(dir))]))

"""
    ladder(st, s, J; H, label="J+", pr=Protocol(), out_dir="") -> Ladder

Walks the currents `J` (applied field `H`, scalar or one per point) from the state `s`, which it
advances.  With `out_dir` it writes the checkpoint of every closed point (`restart/`), the rung
history (`output/rungs_<label>.csv`) and obeys `STOP.txt`.
"""
function ladder(st::Stepper{N}, s::State{N}, Jl::AbstractVector{<:Real}; H::Union{Real,AbstractVector},
                label::String="J+", pr::Protocol=Protocol(), out_dir::String="",
                resume::Union{Nothing,AbstractDict}=nothing) where N
    gr = st.grid; mk = st.masks
    (; x1, x2, y1, y2, z1, z2) = gr
    Js = Float64.(collect(Jl)); np = length(Js)
    Hs = H isa Real ? fill(Float64(H), np) : Float64.(collect(H))
    length(Hs) == np || error("ladder: one H per current point")
    lMax = pr.lMax; mMax = pr.mMax; dt = st.dt
    dt_cyc = lMax * dt
    iter_min = pr.iter_min
    pr.t_min_GL > 0 && (iter_min = max(iter_min, ceil(Int, pr.t_min_GL / dt_cyc)))
    eps_c = eps_cycle(pr, st)

    # voltmeter: planes, live links, complete lines
    vi = x1 + pr.delta_v; vo = x2 - pr.delta_v
    Min = [mk.τx[vi,j,k] > 0.5 for j in y1:y2+1, k in z1:z2+1]
    Mou = [mk.τx[vo,j,k] > 0.5 for j in y1:y2+1, k in z1:z2+1]
    nin = count(Min); nou = count(Mou)
    (nin == 0 || nou == 0) && error("ladder: no live links on the voltmeter planes (delta_v = $(pr.delta_v))")
    line_ok = [all(mk.τx[ii,j,k] > 0.5 for ii in vi:vo-1) for j in y1:y2+1, k in z1:z2+1]
    nlin = count(line_ok)
    nlin == 0 && error("ladder: no complete line between the voltmeter planes")
    line_f1 = [line_ok[a, b] && all(mk.f[ii, y1+a-1, z1+b-1] >= 1 - 1e-12 for ii in vi:vo)
               for a in axes(line_ok, 1), b in axes(line_ok, 2)]
    nlin_f1 = count(line_f1)
    nwin = max(2, pr.V_win)

    Vser = zeros(mMax + 1); TAser = zeros(mMax + 1); Tf1 = zeros(mMax + 1)
    Tser = [zeros(mMax + 1) for _ in 1:N]
    ItSer = zeros(mMax + 1); IsSer = zeros(mMax + 1)
    nl1, nl2 = size(line_ok)
    Tmat = [zeros(nl1, nl2) for _ in 1:N]; Dprev = [zeros(nl1, nl2) for _ in 1:N]
    Dl = [zeros(nl1, nl2) for _ in 1:N]

    out = (V = zeros(np), Vb = fill(NaN, np, N), Vh = fill(NaN, np, N), Vst = zeros(np),
           m = zeros(Int, np), conv = fill(' ', np), res = zeros(np, N), nslip = zeros(np),
           M = zeros(np), Nv = zeros(Int, np, N), psi2 = zeros(np, N), Ib = zeros(np, N),
           Itot = fill(NaN, np), Isam = fill(NaN, np), dmax = zeros(np), nbig = zeros(Int, np),
           Vf1 = fill(NaN, np), VA = fill(NaN, np), delay = fill(NaN, np), tconf = zeros(np))
    ring = Any[]; confirmed = false              # closed 'F' rungs kept for the delay confirmation
    lastF = 0.0                                  # the largest ⟨|Ψ₁|²⟩ of the 'F' rungs so far, for the 'N' verdict

    nb = (x2 - x1 + 2, y2 - y1 + 2, z2 - z1 + 2)
    Cooper0 = [[abs(s.G[µ][i,j,k]) for i in x1:x2+1, j in y1:y2+1, k in z1:z2+1] for µ in 1:N]
    fieldOld = copy(s.hz)
    kc = oround((z1 + z2) / 2)
    hist = isempty(out_dir) ? "" : joinpath(out_dir, "output", "rungs_$(label).csv")
    nV = 0; j0 = 0; cut = false
    _pcg_reset!(st)
    # each ladder draws its own noise stream (seed and label), so that the two polarities in one process or in two agree
    st.noise === nothing || copy!(st.noise.rng, Xoshiro(hash(label, st.noise.seed)))
    if resume !== nothing
        haskey(resume, "pcg_x1") && _pcg_back!(st, (true, (resume["pcg_x1"], resume["pcg_x2"], resume["pcg_x3"])))
        st.noise !== nothing && haskey(resume, "noise_rng") && copy!(st.noise.rng, resume["noise_rng"])
        j0, lastF, nV, confirmed, cut = _restore_ladder!(out, Cooper0, ring, resume, Js, Hs, gr, N)
        copyto!(fieldOld, s.hz)
        isempty(hist) || _truncate_rungs(hist, j0)
        isempty(out_dir) || _truncate_history(joinpath(out_dir, "j_history.csv"), label, j0)
    end
    isempty(hist) || _rung_header(hist, N, st, pr)
    isempty(out_dir) || tui_status(out_dir, @sprintf("%s %s, %d points", label,
                                                     j0 == 0 ? "starting" : "resumed after point $j0", np); pid=true)

    stopped = false
    for jp in j0+1:np
        cut && break
        if _stop_requested(out_dir)
            pr.verbose && @printf("[STOP] before point %d\n", jp)
            stopped = true
            break
        end
        if pr.stop_after_V > 0 && nV >= pr.stop_after_V
            pr.verbose && @printf("[%d V in a row] before point %d\n", nV, jp)
            break
        end
        J = Js[jp]; Hc = Hs[jp]
        HJ = selffield(st, J)
        m = 1; conv = '*'; dmax_cur = 0.0; nbig_cur = 0
        it_res_min = max(iter_min, ceil(Int, pr.t_min_GL_res / dt_cyc))
        while m <= mMax
            for _ in 1:lMax
                step!(s, st; H=Hc, HJ=HJ)
            end
            # heat that leaves the condensate of the diode: this current is cut ('T')
            if st.heat !== nothing && maximum(st.heat.θ) > st.heat.h.cut
                conv = 'T'
                pr.verbose && @printf("heat: θmax = %.3f > %.2f, cutting J = %+.3f\n", maximum(st.heat.θ), st.heat.h.cut, J)
                break
            end
            Vser[m] = _plane_sum(s, gr, vo, Mou)/nou - _plane_sum(s, gr, vi, Min)/nin
            TAser[m] = _theta_A(s, gr, vi, vo, line_ok)
            for µ in 1:N
                _path_phase!(Dl[µ], s.G[µ], s, gr, vi, vo)
            end
            if m == 1
                for µ in 1:N
                    copyto!(Dprev[µ], Dl[µ]); fill!(Tmat[µ], 0.0); Tser[µ][1] = 0.0
                end
                Tf1[1] = 0.0; dmax_cur = 0.0; nbig_cur = 0
            else
                @inbounds for i in eachindex(Tmat[1])
                    line_ok[i] || continue
                    d = Dl[1][i] - Dprev[1][i]
                    d -= 2pi*oround_f(d/(2pi))
                    dmax_cur = max(dmax_cur, abs(d))
                    abs(d) > 2.0 && (nbig_cur += 1)
                end
                for µ in 1:N
                    _unwrap!(Tmat[µ], Dl[µ], Dprev[µ])
                end
                nlin_f1 > 0 && (Tf1[m] = sumlin(Tmat[1][i] for i in eachindex(Tmat[1]) if line_f1[i])/nlin_f1)
                for µ in 1:N
                    Tser[µ][m] = sumlin(Tmat[µ][i] for i in eachindex(Tmat[µ]) if line_ok[i])/nlin
                end
            end
            (ItSer[m], IsSer[m]) = midplane_current(s, st)

            eps_here = jp == 1 ? pr.eps_seed : eps_c
            dif = zeros(N); amax = 0.0
            @inbounds for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1
                ii = i-x1+1; jj = j-y1+1; kk = k-z1+1
                for µ in 1:N
                    a = abs(s.F[µ][i,j,k])
                    dif[µ] = max(dif[µ], abs(a - Cooper0[µ][ii,jj,kk]))
                    amax = max(amax, a)
                    Cooper0[µ][ii,jj,kk] = a
                end
            end
            if pr.guard_psi > 0 && (!isfinite(amax) || amax > pr.guard_psi)
                isempty(out_dir) || save_state(joinpath(out_dir, "restart", "state_$(label)_diverged.jld2"), s)
                error("ladder $label: the scheme diverged at point $jp (J = $J), cycle $m: max|Ψ| = $amax; dt = $(st.dt) ($(st.limit))")
            end
            diffh = 0.0
            @inbounds for i in eachindex(s.hz)
                diffh = max(diffh, abs(s.hz[i] - fieldOld[i]))
            end
            copyto!(fieldOld, s.hz)
            out.res[jp, :] .= dif

            if m >= iter_min && maximum(dif) < eps_here && diffh < eps_here
                if m < 2 || maximum(abs(Tser[µ][m] - Tser[µ][max(1, m ÷ 2)]) for µ in 1:N) < 2pi
                    if !pr.f_strict || _silent_now(Tser[1], Tmat[1], line_ok, m, dt_cyc, pr.f_margin)
                        conv = 'F'; break
                    end
                end
            end
            if pr.use_Vconv && abs(J) > 0 && m >= max(iter_min, 2nwin)
                plateau = abs(Tser[1][m] - Tser[1][max(1, m ÷ 2)]) * 2 >= pr.v_margin * 10 * 2pi
                if plateau && m >= it_res_min &&
                   (!pr.f_strict || _crossings_now(Tser[1], Tmat[1], line_ok, m, dt_cyc) >= 10)
                    conv = 'V'; break
                end
            end
            m += 1
        end
        mlast = min(m, mMax)

        nav = min(nwin, mlast)
        out.Vst[jp] = meanlin(view(Vser, mlast-nav+1:mlast))
        conv == 'V' && (nav = min(max(pr.V_win_min_res, mlast ÷ 2), mlast))
        wq = mlast-nav+1:mlast
        nslip = 0.0; Vc = 0.0
        if length(wq) >= 3
            tq = (collect(wq) .- wq[1]) .* dt_cyc
            for µ in 1:N
                out.Vb[jp, µ] = -pendiente(tq, view(Tser[µ], wq))
            end
            Vc = out.Vb[jp, 1]
            nlin_f1 > 0 && (out.Vf1[jp] = -pendiente(tq, view(Tf1, wq)))
            out.VA[jp] = (TAser[wq[end]] - TAser[wq[1]]) / ((wq[end] - wq[1]) * dt_cyc)
            nslip = sumlin(abs(Tmat[1][i]) for i in eachindex(Tmat[1]) if line_ok[i])/(2pi)
            nslip < 1 && (Vc = 0.0)
        end
        wh = (mlast - mlast ÷ 2 + 1):mlast
        if length(wh) >= 3
            th = (collect(wh) .- wh[1]) .* dt_cyc
            for µ in 1:N
                ns = sumlin(abs(Tmat[µ][i]) for i in eachindex(Tmat[µ]) if line_ok[i])/(2pi)
                out.Vh[jp, µ] = ns < 1 ? 0.0 : -pendiente(th, view(Tser[µ], wh))
            end
        end
        out.V[jp] = Vc; out.nslip[jp] = nslip; out.m[jp] = m; out.conv[jp] = conv
        out.dmax[jp] = dmax_cur; out.nbig[jp] = nbig_cur
        out.M[jp] = magnetization(s.hz, Hc, gr)
        for µ in 1:N
            out.Nv[jp, µ] = count_vortices(s.F[µ], s.Vx, s.Vy, s.hz, mk.f, gr, kc)[1]
            out.psi2[jp, µ] = mk.vac === nothing ?
                meanlin(abs2(s.F[µ][i,j,k]) for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1) :
                sumlin(abs2(s.F[µ][i,j,k]) for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1 if !mk.vac[i,j,k]) /
                count(!mk.vac[i,j,k] for k in z1:z2+1, j in y1:y2+1, i in x1:x2+1)   # over the sample, not its vacuum
        end
        if conv == 'F' && pr.n_frac > 0
            if abs(J) > 0 && lastF > 0 && out.psi2[jp, 1] < pr.n_frac * lastF
                conv = 'N'; out.conv[jp] = 'N'
                pr.verbose && @printf("[%s] J=%+.4f: ⟨|Ψ₁|²⟩ %.3g < %.2f × %.3g of the largest 'F': normal ('N')\n", label, J,
                                      out.psi2[jp, 1], pr.n_frac, lastF)
            else
                lastF = max(lastF, out.psi2[jp, 1])   # the largest of the 'F' rungs: a collapse spread over several rungs
            end                                       # cannot walk the reference down with it
        end
        out.Ib[jp, :] .= band_currents(s, st)
        if mlast >= 2
            r = (mlast ÷ 2 + 1):mlast
            It = sum(@view ItSer[r]) / length(r); Is = sum(@view IsSer[r]) / length(r)
            Inom = J * slab_thickness(gr) * ((y2 - y1 + 1) * gr.dy)
            if abs(Inom) > 0
                out.Itot[jp] = abs(It) / abs(Inom); out.Isam[jp] = abs(Is) / max(abs(It), eps())
            end
        end
        pr.verbose && @printf("[%s] J=%+.4f H=%.4f | V=%+.4e (slips %.2f) | Vh=%s | m=%d conv=%s | Nv=%s\n",
                              label, J, Hc, Vc, nslip, join((@sprintf("%+.3e", out.Vh[jp, µ]) for µ in 1:N), "/"),
                              m, conv, join(out.Nv[jp, :], "/"))
        if !isempty(out_dir)
            _rung_line(hist, jp, J, Hc, out, mlast * dt_cyc, N)
            _tui_point(out_dir, label, J, isnan(out.Vh[jp, 1]) ? out.V[jp] : out.Vh[jp, 1], out.conv[jp],
                       out.m[jp], out.Nv[jp, 1], out.Nv[jp, N], out.psi2[jp, N], dt_cyc)
        end
        if pr.t_confirm_GL > 0 && abs(J) > 0
            if conv == 'F'
                push!(ring, (jp=jp, J=J, H=Hc, t=mlast * dt_cyc, s=deepcopy(s), open=_open_snap(st),
                             θ=st.heat === nothing ? nothing : copy(st.heat.θ)))
                length(ring) > pr.confirm_max && popfirst!(ring)
            elseif conv in ('V', 'N') && !confirmed && !isempty(ring) && ring[end].jp == jp - 1
                confirmed = true
                sn = _open_snap(st); sp = _pcg_snap(st); sr = _noise_snap(st); sθ = st.heat === nothing ? nothing : copy(st.heat.θ)
                D = similar(Dl[1]); Dp = similar(Dl[1]); want = jp - 1
                for c in reverse(ring)
                    c.jp == want || break
                    want -= 1
                    T = pr.t_confirm_GL - c.t
                    T <= 0 && (out.tconf[c.jp] = c.t; break)
                    d = _continue_rung(st, c, T, lMax, line_ok, vi, vo, D, Dp)
                    out.delay[c.jp] = d; out.tconf[c.jp] = isnan(d) ? pr.t_confirm_GL : d
                    pr.verbose && @printf("[%s] confirm J=%+.4f: %s\n", label, c.J,
                                          isnan(d) ? @sprintf("silent for %.0f", pr.t_confirm_GL) : @sprintf("crosses at %.0f", d))
                    isnan(d) && break
                    out.conv[c.jp] = 'D'
                end
                _open_back!(st, sn); _pcg_back!(st, sp); _noise_back!(st, sr); sθ === nothing || copyto!(st.heat.θ, sθ); empty!(ring)
                if !isempty(out_dir)
                    fn = joinpath(out_dir, "output", "confirm_$(label).csv")
                    open(fn, "w") do io
                        println(io, "point,J,t_closed,t_followed,first_crossing,conv")
                        for q in 1:jp
                            out.tconf[q] > 0 && @printf(io, "%d,%.6f,%.3f,%.3f,%.3f,%s\n", q, Js[q], out.m[q] * dt_cyc,
                                                        out.tconf[q], out.delay[q], string(out.conv[q]))
                        end
                    end
                end
            end
        end
        nV = conv in ('V', 'N') ? nV + 1 : 0
        # the checkpoint of the closed point: the state and all the ladder carries to the next point, in one
        # file written at once, so that resuming from it (resume) is continuing the ladder to the bit
        if !isempty(out_dir)
            ck = _save_ladder(joinpath(out_dir, "restart", "state_$(label).jld2"), s, st, jp, Js, Hs, out, Cooper0,
                              lastF, nV, confirmed, conv == 'T', ring)
            if pr.keep_rung_states
                dr = joinpath(out_dir, "restart", "rungs"); mkpath(dr)
                cp(ck, joinpath(dr, @sprintf("state_%s_J%+.4f.jld2", label, J)); force=true)
            end
        end
        # after a heat cut the next current would be cut in its first cycle: the ladder ends here
        conv == 'T' && break
    end
    isempty(out_dir) || _mark_done(joinpath(out_dir, "restart", "state_$(label).jld2"), !stopped)
    Ladder(label, Js, Hs, out.V, out.Vb, out.Vh, out.Vst, out.m, out.conv, out.res, out.nslip,
           out.M, out.Nv, out.psi2, out.Ib, out.Itot, out.Isam, out.dmax, out.nbig, out.Vf1, out.VA,
           out.delay, out.tconf, dt_cyc, s)
end

# --- resuming a ladder (the checkpoint of every closed point holds all it carries) --------------------------
const _OUT_KEYS = (:V, :Vb, :Vh, :Vst, :m, :conv, :res, :nslip, :M, :Nv, :psi2, :Ib, :Itot, :Isam, :dmax, :nbig,
                   :Vf1, :VA, :delay, :tconf)

_state_pairs(p, s::State{N}) where N =
    (ntuple(µ -> p * "G$µ" => s.G[µ], N)..., p * "thetax" => s.θx, p * "thetay" => s.θy, p * "thetaz" => s.θz,
     p * "thetax_prev" => s.φx, p * "thetay_prev" => s.φy, p * "thetaz_prev" => s.φz,
     p * "hx" => s.hx, p * "hy" => s.hy, p * "hz" => s.hz)
_state_of(p, w, gr, N) = State(gr, ntuple(µ -> w[p * "G$µ"], N), (w[p * "thetax"], w[p * "thetay"], w[p * "thetaz"]),
                               (w[p * "thetax_prev"], w[p * "thetay_prev"], w[p * "thetaz_prev"]),
                               (w[p * "hx"], w[p * "hy"], w[p * "hz"]))

function _save_ladder(path, s, st, jp, Js, Hs, out, Cooper0, lastF, nV, confirmed, cut, ring)
    ex = Pair{String,Any}["jp" => jp, "J" => Js[1:jp], "V" => out.V[1:jp], "H" => Hs[jp], "Hs" => Hs[1:jp],
                          "lastF" => lastF, "nV" => nV, "confirmed" => confirmed, "cut" => cut,
                          "ring_n" => length(ring)]
    for k in _OUT_KEYS
        x = getfield(out, k)
        push!(ex, "out_$k" => (k === :conv ? String(x[1:jp]) : copy(selectdim(x, 1, 1:jp))))
    end
    for (µ, c) in enumerate(Cooper0); push!(ex, "cooper$µ" => c); end
    for (q, c) in enumerate(ring)
        append!(ex, ["ring$(q)_jp" => c.jp, "ring$(q)_J" => c.J, "ring$(q)_H" => c.H, "ring$(q)_t" => c.t,
                     "ring$(q)_open" => c.open, "ring$(q)_theta" => c.θ])
        append!(ex, collect(Pair{String,Any}, _state_pairs("ring$(q)_", c.s)))
    end
    st.heat === nothing || push!(ex, "Temp" => st.heat.θ)
    for (k, v) in pairs(open_extra(st)); push!(ex, string(k) => v); end
    st.imp isa PCGSolver && st.imp.warm[1] && append!(ex, ["pcg_x$c" => st.imp.x[c] for c in 1:3])
    st.noise === nothing || push!(ex, "noise_rng" => copy(st.noise.rng))
    save_state(path, s; (Symbol(k) => v for (k, v) in ex)...)
    write(path * ".status", "running\n")
    path
end

# done: the ladder ended by itself (all points, stop_after_V or a heat cut); stopped: by STOP.txt
_mark_done(path, done::Bool) = isfile(path) && write(path * ".status", done ? "done\n" : "stopped\n")
ladder_done(path) = isfile(path * ".status") && strip(read(path * ".status", String)) == "done"

function _restore_ladder!(out, Cooper0, ring, w, Js, Hs, gr, N)
    j0 = Int(w["jp"])
    (j0 <= length(Js) && w["J"] == Js[1:j0] && w["Hs"] == Hs[1:j0]) ||
        error("resume: the checkpoint's currents $(w["J"]) are not the first $j0 of this ladder")
    for k in _OUT_KEYS
        x = getfield(out, k); v = w["out_$k"]
        k === :conv ? (x[1:j0] .= collect(v)) : (selectdim(x, 1, 1:j0) .= v)
    end
    for µ in 1:N; copyto!(Cooper0[µ], w["cooper$µ"]); end
    for q in 1:Int(w["ring_n"])
        push!(ring, (jp=Int(w["ring$(q)_jp"]), J=w["ring$(q)_J"], H=w["ring$(q)_H"], t=w["ring$(q)_t"],
                     s=_state_of("ring$(q)_", w, gr, N), open=w["ring$(q)_open"], θ=w["ring$(q)_theta"]))
    end
    (j0, Float64(w["lastF"]), Int(w["nV"]), Bool(w["confirmed"]), Bool(w["cut"]))
end

# the rung table without the rows after the checkpoint (a point closed and written, then the run was killed
# before its checkpoint)
function _truncate_rungs(path, j0)
    isfile(path) || return
    L = readlines(path)
    keep = filter(l -> startswith(l, "#") || startswith(l, "point") || (p = tryparse(Int, split(l, ',')[1]);
                       p !== nothing && p <= j0), L)
    length(keep) == length(L) || write(path, join(keep .* "\n"))
end

# the same for the menu's history (j_history.csv): one row per closed point of each polarity, in order, the rows of
# the other polarity possibly interleaved (two processes into one folder); this ladder keeps its first j0
function _truncate_history(path, label, j0)
    isfile(path) || return
    L = readlines(path); n = 0
    keep = filter(L) do l
        w = split(l, ',')
        (length(w) < 2 || w[2] != label) && return true
        n += 1
        n <= j0
    end
    length(keep) == length(L) || write(path, join(keep .* "\n"))
end

"""
    resume_ladder(st, J; H, label, pr, out_dir) -> Ladder

Continues the ladder of `label` from its checkpoint in `out_dir` (`restart/state_<label>.jld2`): the state, the
open boundary, the temperature field and everything the ladder carries from one point to the next, so that the result
is the one of the ladder run without a stop, to the bit.  `J` must start with the currents already done (it may add
more).  A ladder that had ended returns at once.  `write = false` reads it back without touching the folder.
"""
function resume_ladder(st::Stepper{N}, Jl::AbstractVector{<:Real}; H::Union{Real,AbstractVector}, label::String="J+",
                       pr::Protocol=Protocol(), out_dir::String, write::Bool=true) where N
    path = joinpath(out_dir, "restart", "state_$(label).jld2")
    s, w = load_state(path, st.grid)
    restore_open!(st, w)
    st.heat === nothing || copyto!(st.heat.θ, w["Temp"])
    ladder(st, s, Jl; H=H, label=label, pr=pr, out_dir=write ? out_dir : "", resume=w)
end

# --- the run folder as the menu (julia/tui) reads it: julia.pid, STATUS.txt, j_history.csv and
#     run_info.txt at the folder root, the same files and columns release 1.2 writes ---------------
_stamp() = Libc.strftime("%Y-%m-%d %H:%M:%S", time())

"""
    tui_status(dir, msg; pid=false)

Writes `STATUS.txt` (one line, `msg` with a time stamp) in the run folder `dir`, and with `pid` the
`julia.pid` the menu uses to tell a live run.  `msg` ending in "done" marks the run off.
"""
function tui_status(dir::AbstractString, msg::AbstractString; pid::Bool=false)
    isempty(dir) && return nothing
    mkpath(dir)
    pid && write(joinpath(dir, "julia.pid"), string(getpid(), "\n"))
    write(joinpath(dir, "STATUS.txt"), string(_stamp(), "  ", msg, "\n"))
    nothing
end

function _tui_point(dir, label, J, V, conv, it, Nv1, NvN, psiN, tcyc)
    info = joinpath(dir, "run_info.txt")
    isfile(info) || write(info, @sprintf("t_cycle=%.10e\n", tcyc))
    fn = joinpath(dir, "j_history.csv")
    nuevo = !isfile(fn)
    open(fn, "a") do io
        nuevo && write(io, "time,pol,J,V,conv,it,Nv_s,Nv_p,psi_p2\n")
        @printf(io, "%s,%s,%.6f,%.8e,%s,%d,%d,%d,%.6f\n", _stamp(), label, J, V, string(conv), it, Nv1, NvN, psiN)
    end
    tui_status(dir, @sprintf("%s J=%+.3f V=%+.4e conv=%s Nv=%d/%d", label, J, V, string(conv), Nv1, NvN))
end

"""
    tui_point(dir, label, J, crossings, T; dt)

One closed current point of a run that is not a `ladder` (a fixed-current window, a jump), in the files the menu
reads (`j_history.csv`, `run_info.txt`, `STATUS.txt`): the rate 2π·crossings/T as V, 'V' with a crossing and 'F'
without, and T/dt steps of `dt` as the cycles, so that the menu's crossing count is the one of the run.
"""
tui_point(dir::AbstractString, label::AbstractString, J::Real, crossings::Integer, T::Real; dt::Real) =
    (mkpath(dir); _tui_point(dir, label, Float64(J), crossings > 0 ? 2pi * crossings / T : 0.0, crossings > 0 ? 'V' : 'F',
               round(Int, T / dt), 0, 0, 0.0, Float64(dt)))

function _rung_header(path, N, st, pr)
    mkpath(dirname(path))
    isfile(path) && return
    open(path, "w") do io
        println(io, "# ", CODE_VERSION, " | dt=", st.dt, " t_GL | t_cycle=", pr.lMax * st.dt,
                " t_GL | units: J [J0], V [Phi0-free Josephson rate, J0/sigma units], t [t_GL]")
        b = join(("Vh_$µ,Nv_$µ,psi2_$µ,Iband_$µ" for µ in 1:N), ",")
        println(io, "point,J,H,V,t_int,crossings_per_line,conv,cycles,", b, ",Itot,Isam")
    end
end

function _rung_line(path, jp, J, H, out, tint, N)
    v = isnan(out.Vh[jp, 1]) ? out.V[jp] : out.Vh[jp, 1]
    open(path, "a") do io
        b = join((@sprintf("%.8e,%d,%.6e,%.6e", out.Vh[jp, µ], out.Nv[jp, µ], out.psi2[jp, µ], out.Ib[jp, µ]) for µ in 1:N), ",")
        @printf(io, "%d,%.6f,%.6f,%.8e,%.6e,%.4f,%s,%d,%s,%.6e,%.6e\n", jp, J, H, out.V[jp], tint,
                abs(v) * tint / (2pi), string(out.conv[jp]), out.m[jp], b, out.Itot[jp], out.Isam[jp])
    end
end

"""
    brackets(pos, neg) -> (pos, neg, eta)

Both critical currents as brackets with the verdict rule, and η at the bracket centres with its
range.  This is the estimate to quote.
"""
function brackets(pos::Ladder, neg::Ladder)
    bp = bracket_Ic(pos.J, rung_rate(pos), rung_time(pos); conv=pos.conv)
    bn = bracket_Ic(neg.J, rung_rate(neg), rung_time(neg); conv=neg.conv)
    dp = delay_bracket(pos); dn = delay_bracket(neg)
    (pos = bp, neg = bn, eta = eta_bracket(bp, bn), delay = (pos = dp, neg = dn, eta = eta_bracket(dp, dn)))
end

"""
    delay_bracket(L) -> (lo, hi, t_silent, ok)

The onset as the delay brackets it (`Protocol(t_confirm_GL=...)`): `hi` the first rung that flows ('V',
or 'D': it crossed when continued), `lo` the rung below it and `t_silent` how long that one was followed
without a crossing.  `ok` when `lo` was continued to `t_confirm_GL` (without the option, `ok` is false
and the bracket is the one of the windows).
"""
function delay_bracket(L::Ladder)
    Ja = abs.(L.J)
    k = findfirst(q -> Ja[q] > 0 && L.conv[q] in ('V', 'D', 'N'), eachindex(Ja))
    (k === nothing || k == 1) && return (lo = NaN, hi = k === nothing ? Inf : Ja[1], t_silent = NaN, ok = false)
    q = k - 1
    (lo = Ja[q], hi = Ja[k], t_silent = L.t_conf[q] > 0 ? L.t_conf[q] : L.m[q] * L.t_cycle,
     ok = L.conv[q] == 'F' && L.t_conf[q] > 0)
end

"""
    run_ladders(st, J; H, pr=Protocol(), out_dir="") -> (pos, neg, brackets)

Both polarities from the same Meissner seed: the currents 0, |J₁|, |J₂|, ... and their
negatives (the J = 0 seed is relaxed with its own tolerance).
"""
function run_ladders(st::Stepper, Jl::AbstractVector{<:Real}; H::Real, pr::Protocol=Protocol(),
                     out_dir::String="", resume::Bool=false, polarity::Symbol=:both)
    polarity in (:both, :pos, :neg) || error("run_ladders: polarity is :both, :pos or :neg")
    Jp = sort(unique(vcat(0.0, abs.(Float64.(collect(Jl))))))
    function one(label, Js)
        ck = joinpath(out_dir, "restart", "state_$(label).jld2")
        resume && !isempty(out_dir) && isfile(ck) && return resume_ladder(st, Js; H=H, label=label, pr=pr, out_dir=out_dir)
        # each polarity starts from the Meissner seed and the bath, so the ramp of an open boundary starts afresh too
        st.heat === nothing || fill!(st.heat.θ, st.heat.T)
        st.open === nothing || (st.open.count = -1)
        ladder(st, initial_state(st.model, st.grid, st.masks, H), Js; H=H, label=label, pr=pr, out_dir=out_dir)
    end
    pos = polarity === :neg ? nothing : one("J+", Jp)
    neg = polarity === :pos ? nothing : one("J-", -Jp)
    b = pos === nothing || neg === nothing ? nothing : brackets(pos, neg)
    (pos = pos, neg = neg, brackets = b)
end
