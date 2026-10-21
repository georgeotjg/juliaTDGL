# Changelog

## 2.0.0

Release 2.0 grows until it is complete; the tag `v2.0.0` follows it. The sculpted sample (vacuum) was the last item of its
frozen scope; apart from it, a GPU port. Not planned (see the README's limitations): the memory of the band-resolved (Schmid–Schön)
charge imbalance (its slaved form and the interband exchange are in) and the scalar potential of release 1.x (with the
charge imbalance slaved it is a choice of gauge; it stays in release 1.2).

A new package, `TDGL` 2.0.0 in `tdgl2/`, next to release 1.2.0 (`julia/`), which stays to reproduce the published work. With
the new terms off and `selffield_bc = :slab`, 2.0.0 reproduces release 1.2.0 (regression tests compare full states: to the
bit on OpenBLAS's Haswell kernels, to 1e-10 on other machines). Julia 1.13 or later.

**Model**
- N bands (`Model`, `Band`), a symmetric Josephson matrix, passive bands (coupled equilibrium, `equilibrium`).
- Kramer–Watts-Tobin relaxation per band (`γkwt`; `two_band(...; gamma_kwt)`).
- σ_n given or that of a dirty material (`σn = :dirty`, `sigma_dirty`).
- Self-field boundaries `:bar` (default: the sample carries the nominal current) and `:open` (the free-space field of the
  currents of the box, lattice Green function and FFT); `:slab` is the boundary of 1.x.
- Insulating buffer outside the bar of the sample and its contacts (`σvac` in Julia, `sigma_vac` in a run file; preconditioned
  conjugate gradients).
- Heat, drag and a bath temperature, as in 1.2.
- Normal fluids per band (`qp_exchange = ν` with two bands, a matrix of rates with more): the quasiparticles of each band with their own potential and a finite
  interband exchange rate ν; one field δ = μ_π − μ_σ per step (a Helmholtz solve, exact). Off (the default) is one normal
  fluid, the exchange infinitely fast, as before. In a long strip the σ share of the converted current follows the 1D
  two-fluid law (0.63 → 0.18 from ν = ∞ to 0 without KWT). With the insulating buffer the fluids live in the conductor (zero
  flux to the insulator), in a sculpted sample with the links to its vacuum cut (conjugate gradients preconditioned by the
  separable solve, started cold every step: resuming keeps the bits), and with the Joule heat they add the heat of their
  potential gradients and of the exchange.
- Anisotropic bands (`Band(; m_axes)`, `m_x`, `m_y`, `m_z` or `xi_ratio_x`, … in a material file): a mass per axis in the
  band update, the supercurrent and the step bound. An isotropic band keeps its bits.
- Thermal (Langevin) noise on the bands (`noise = T̃`, `noise_seed`): fluctuation–dissipation at the noise temperature,
  a seeded stream per ladder, saved in the checkpoints (resuming keeps the bits); with KWT, by the Ψ-dependent mobility.
  `noise_field = true`: the same bath on the field (ΔA_l = √(T̃ dt/(σ_l dV)) n on every link before the implicit solve); in a
  normal box the field energy of the free modes is that of the implicit Euler of the field at T̃, to 0.03 %.

**Scheme**
- Runge–Kutta–Legendre stages for the bands that KWT makes stiff (`rkl`, default).
- The rotating frame (`rotframe`) and a stage margin (`rkl_margin`), on by default with KWT: the state no longer depends on
  dt in the conversion layers, and γ̃ = 30 no longer diverges. The rotation of the links and that of the frame go in one
  pass per stage, so the frame costs nothing over the stages (KWT with γ̃ = 10: ×2.2 per t_GL over the model without it).
- The insulating solve starts from the previous step's solution (12 iterations instead of 27).

**Protocol and output**
- The onset by the delay of the first slip (`Protocol(t_confirm_GL)`, `delay_bracket`, closing flag 'D'); `crossing_times`.
- A sample gone normal closes 'N', not 'F' (`n_frac`, against the largest ⟨|Ψ₁|²⟩ of the silent rungs before it).
- Strict fixed-point close ('F' only without slow flow; `f_strict`, `f_margin`) and a margin on the resistive close (`v_margin`).
- Runs from TOML files (`run_config`, `scripts/run.jl`), materials in `materials/`, `output/summary.json` with units,
  version, commit, the scheme as resolved (`dt_scale`, `rotframe`, `rkl_margin`, stages) and the regime numbers, including
  the conversion length λ_c (`regime`, for any number of bands).
- Stop and resume to the bit: the checkpoint of every closed point holds the state and all the ladder carries
  (`resume_ladder`, `run_config(...; resume=true)`, `scripts/run.jl --resume`); `--polarity +|-` runs one polarity per
  process into the same folder. A folder that holds a ladder is not run over.
- Checkpoints in JLD2 (reads those of 1.2); `tui_status`/`tui_point` for the terminal menu of 1.2, which launches,
  stops and resumes runs of 2.0.
- The optional terms from the run file: `[heat]`, `drag` in `[scheme]`.
- Discrete energy balance (`free_energy`, `dissipation`).

**Validation**
- Against release 1.2 (full states) and against pyTDGL 0.9 for one band in 2D: depairing, the conversion length with KWT
  (γ̃ = 0–30, 0.2 % between the codes) and the onset of a notched strip.
- In the suite: the discrete energy balance, dt convergence with the rotating frame, the analytic three-band state that breaks
  time reversal, and a mirror-symmetric sample (the state at +J is the mirror of the one at −J to rounding, with each boundary).

**Tools**
- `tools/ladder_sheet.py` (the standard sheet of a ladder) and `tools/delay_fit.py` (the onset from the delay law).
- `scripts/figures.jl` (the standard figures of a run from its folder: the ladder sheet, |Ψ|² of every band on the three
  mid cuts with the vortices (a cut that falls in sculpted vacuum moves to the centre of the nearest full run of planes), the band shares of the current along the sample; `tools/state_figures.py` draws the states).
- The run folder keeps a copy of the shape file it names, at the same relative path.
- `tools/collect_runs.py` (every run under some folders as one CSV: options, brackets, η, version; ladders run from a
  script, without summary.json, from their rungs). `tools/run_report.py` (one Markdown page per run: what was run, every rung,
  the brackets and η, the figures; `figures.jl` writes it).

**Updated 21-oct-2026**
- The onset by the delay (`t_confirm_GL`) works with the heat model: each rung kept for the confirmation carries the
  temperature it closed with, and the ladder gets its own back (it was an error). Tested to the bit with heat, Joule, a
  contact margin and drag, also resumed.
- `sigma_vac` for the insulating buffer in a run file (TOML keys are ASCII unless quoted; `σvac` stays in Julia).
- Run files for every option in `examples/`: the open boundary, the insulating buffer, heat with drag, three frustrated
  bands.
- README: what each release has, the cost of each option, the limitations (isotropic bands, dx = dy = dz, no GPU).
- The launch wizard of the terminal menu launches the same run with 2.0 (it writes the run file).
- `dt_scale = 1` validated with KWT for onsets (the silent state to six digits, the same delays).
- Thermal noise with KWT: the kick scaled apart along Ψ and across it by the Ψ-dependent mobility, with its Itô drift (the
  equilibrium exp(−F/T̃) checked; without the drift the mean |Ψ|² comes out 39 % high).
- Normal fluids per band with any number of bands (`qp_exchange` as a matrix of rates with detailed balance): the modes of the
  exchange graph make N Helmholtz solves of the two-band kind; with two bands it gives the two-band result to rounding.
- A sculpted sample: a shape element with `vacuum = true` removes the material (no superconductor, no normal metal). Its nodes
  leave the block (the links that touch them carry no supercurrent and the band update reads no neighbour across them: the
  zero-flux condition of a free surface), the insulating padding follows the sample and its leads take its section
  (`conducting_nodes`, `bar_field(gr; section)`); J stays the current density. A uniform state stays uniform up to the
  surface to 1e-13, and the static current through a cable is J times its section to 2e-3, none of it through the vacuum.
  Needs `sigma_vac`; not with heat, drag or `qp_exchange`. Example: `examples/cable_k2.toml`, `shapes/cable_x.toml`.
- Shape files: a figure on another face (`axis = "x"` or `"y"`, centre `u`, `v`, extent `along`), and `outside = true`, which
  writes outside the figure: the figure is what is kept. The shape editor draws figures on its three views.
