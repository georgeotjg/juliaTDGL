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
- A material per region (`material = "name"` on a shape region or a palette entry, `[materials.<name>]` tables in the run file,
  `materials` and `transparency` in Julia): per node the band coefficients of its material, per link the harmonic mean of the
  stiffnesses times the transparency of the barrier between the two materials; the supercurrent, the free energy and the
  dissipation take the same weights. κ, σ_n and T are the sample's. Two (or three) regions of the same material are the uniform
  run to the bit; the discrete continuity holds to 1e-15 across an interface; the proximity length matches the lattice solution
  to 4e-5. Example: `examples/bicapa_tunel.toml`. A `[materials.<name>]` table must give every key that `[material]` gives
  (Γ, β, γ̃, the interband coupling): one left out is an error, not the package default. Not yet with `qp_exchange`, heat,
  drag, noise, anisotropic bands, or a σ_n per region (each an explicit error).

**Scheme**
- Runge–Kutta–Legendre stages for the bands that KWT makes stiff (`rkl`, default).
- The rotating frame (`rotframe`) and a stage margin (`rkl_margin`), on by default with KWT: the state no longer depends on
  dt in the conversion layers, and γ̃ = 30 no longer diverges. The rotation of the links and that of the frame go in one
  pass per stage, so the frame costs nothing over the stages (KWT with γ̃ = 10: ×2.2 per t_GL over the model without it).
- The insulating solve starts from the previous step's solution (12 iterations instead of 27).
- Damped Runge–Kutta–Chebyshev stages, the default since 26-oct-2026 (`rkl = "rkc"`, damping `rkc_damping = 0.5`; `rkl = true` gives
  the RKL stages of the runs before): 0.76 s² explicit steps
  bridged by s stages against s(s+1)/2, taken only by a band where it saves a stage, so a run without KWT keeps the bits of
  RKL. On a notched two-band box with KWT −5 % per step at γ̃ = 10 and −10 % at γ̃ = 30, with the same brackets, the same delay
  of the first slip (0.05 %) and the same energy balance (on the laptop, pinned to one core: −6.1 % and −10.4 %). The textbook
  damping 0.05 diverges there within 100 steps.
- The tolerance of the insulating solve is 1e-8 by default since 26-oct-2026 (`pcg_tol`; 1e-11 before): −41 % per step on the
  nanowire, −45 % on a ladder of the notched nanowire, the same bracket and the flowing voltages to 9 digits. The solve's error
  is then three orders below the scheme's own first-order error in time.
- A run folder keeps the scheme defaults it started with (`scheme_defaults.toml`); a folder that holds a ladder without that
  file started before 26-oct-2026 and keeps `rkl = true` and `pcg_tol = 1e-11`, so resuming, refining or filming it does not
  switch the integrator mid-ladder. A key given in `[scheme]` always wins.
- `summary.json` writes the integrator of each band (`euler`, `rkl`, `rkc`) and the tolerance of the insulating solve.
- Materials, completed: σ_n per region (`sigma_n` in a `[materials.<name>]` table: the
  conductivity of a link is the harmonic mean of its nodes, and the field solve takes the conjugate gradients of the
  insulating buffer); `qp_exchange` with materials (a rate per material, the same split of σ_n between the bands in every
  material: a different split needs the scalar potential, another model); a band absent from a material (`absent = [2]`: a
  one-band metal next to a two-band superconductor, `examples/metal_una_banda.toml`); noise, anisotropic bands and heat
  (`heat_D_T` per material) with materials. Drag with materials stays refused. A material of the catalogue in a
  `[materials.<name>]` table (`file = "<name>.toml"`, the keys given next to it win), as `[material]` takes it (29-oct).
- The insulating solve starts from the extrapolation of the last two steps and runs its separable preconditioner in Float32
  inside a flexible conjugate gradient (`pcg_fast`, `pcg_f32`, the defaults since 26-oct-2026, evening): 14 → 6.5 iterations
  and −55 % per step on the nanowire (on the laptop, pinned to one core), the state within 2e-9 of the solve without them, the
  same bracket and flowing voltages on the notched nanowire, and a ladder resumes to the bit. A run folder from before keeps
  them off (`scheme_defaults.toml`; a key its file lacks takes the value of before). A multigrid of the full auxiliary space
  would bring 8–9 iterations to 4–6 at 3–5 times the cost of one: it does not pay.
- A ladder with noise closes its silent rungs: before, 'F' asked for a fixed point, which the kicks never let the state reach,
  so at T̃ ≳ 1e-3 a silent rung ran to the ceiling of 500 000 cycles. With noise a rung is now silent when it holds
  `t_min_GL_res` without a phase slip on any band. Without noise nothing changes.
- Noise with the heat model is refused: the heat source (the dissipation of the step) counted the Langevin kicks, so a sample
  in equilibrium heated by itself, more at a smaller step (at J = 0, H = 0 the temperature reached its cap). A fluctuating heat
  model consistent with T̃ is another model. Every other pair of options was run together.
- **Thermal noise with KWT fixed**. It diverged (NaN within 4–1000 steps) on the two-band box at
  κ = 2, and even when finite its equilibrium was wrong (⟨|Ψ|²⟩ down to −67 %, the π band collapsed). The cause is a phase mode at
  the lattice scale, s = √(1 + γ̃²|Ψ|²) times faster with KWT and fed by the noise through the frame rate, the tangent step, the
  three-term recursion of the stages and the kick after the step. With noise a KWT band now steps in log-polar form (increments,
  frame term and stage combination as rotations; a smoothed frame rate), the phase kick goes inside the first RKC stage (SK-ROCK,
  Abdulle, Almuslimani and Vilmart 2018) and the amplitude kick is polar. Without noise, or without KWT, the bits are those of
  before. ⟨|Ψ|²⟩ with KWT is within 1.2 % of γ̃ = 0 at T̃ = 1e-3; at T̃ = 1e-2 the stiff π band comes out +5–10 % (RKC) or −7 %
  (Euler substeps) at the production step, an error of the integrator on its stiffest modes that goes away when the step resolves
  them. Noise with KWT on RKL stages is an error (biased 10–25 %); `rkl = "rkc"`, the default, is the one to use. Noise runs without
  KWT are not affected.
- The band kernel reads the coefficients per node only with materials: the uniform path is as fast as before them (to 0.5 %)
  and keeps its bits.
- `tools/run_report.py` names each material per region (its bands and barrier), `ghost_links`, and the scheme defaults the run
  folder started with. A run file with materials stops and resumes to the bit (in the suite).
- `ghost_links = false` in `[mesh]`: τ = 0 on the links that join two ghost nodes (`ghost_links_off!`), the exact gradient
  flow. With materials, for a current that crosses an interface reaching the faces of the box: the face links of 1.x copy
  each side's |Ψ| and are not variational, and there they make a spurious charge imbalance.

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
  mid cuts with the vortices and, on x–y, the streamlines of the supercurrent (a cut that falls in sculpted vacuum moves to the centre of the nearest full run of planes), the band shares of the current along the sample; `tools/state_figures.py` draws the states).
- The run folder keeps a copy of the shape file it names, at the same relative path.
- `tools/collect_runs.py` (every run under some folders as one CSV: options, brackets, η, version; ladders run from a
  script, without summary.json, from their rungs). `tools/run_report.py` (one Markdown page per run: what was run, every rung,
  the brackets and η, the figures; `figures.jl` writes it).
- `scripts/refine.jl` (25-oct): the onset of a run refined from its kept states in one command, or a ladder that never flowed
  carried further up; it gathers the rungs of the run and of every refinement (subfolders that are run folders) and prints the
  bracket and η; `--confirm T`, the bracket confirmed by the delay in one command. `film.jl --state FILE`: the film of any
  state file of the run's mesh (the cooling of a sample seeded in the field: a lattice forming). `start` in `[run]` (`"meissner"`,
  `"field_cooled"`, or a state file, which the run folder keeps a copy of; `start_state`), with its tests. `TUTORIAL.md`.
- `scripts/hysteresis.jl` (30-oct): the down branch of a run, from the state its ladder ended in (the checkpoint, with the
  open boundary and the temperature field), into a subfolder that is a run folder; the retrapping bracket, the onset of the up
  branch and the width Ic − Ir. `scripts/mh.jl` (30-oct): the magnetisation loop M(H) at a fixed current, one ladder whose
  points are the fields 0 → H → 0, with the vortex count of each band (`output/mh.csv`, `tools/mh_plot.py`).
- `hysteresis.jl` and `mh.jl` stop with `STOP.txt` and continue with `--resume`, to the bit (checked: a down branch and a loop
  stopped mid-way and resumed give the rows of the runs without a stop); the menu launches them from the details of a run (keys
  `h` and `l`); a loop goes to its own folder (`<out_dir>_MH_H<Hmax>_d<dH>`), and resuming one with other fields is refused.
  `figures.jl` picks the states of a down branch by |J| (the last silent and the first flowing rung), as of an up ladder.
- The catalogue: one-band Nb, NbN and a-MoGe (`materials/nb.toml`, `nbn.toml`, `moge.toml`) and a three-band frustrated model
  (`three_band_trsb.toml`, a model: s + is) and an iron-based s± model (`fe_spm_model.toml`: γ₁₂ < 0, κ ~ 100 from measured λ
  and ξ; it relaxes to the relative phase π), with their sources; a ladder of each through run.jl, its summary and its figures; a test that
  every file of the catalogue loads and steps finite (30-oct).
- Fixed (30-oct): the explicit supercurrent bound of the step took the superfluid density of a passive band (a ≤ 0, condensed by
  the coupling) as a/β < 0, so the stiffness sum came out too small, or negative: the clean MgB₂ of the catalogue got a negative
  step and NaN at once. A passive band now enters with its coupled bound P², as its diffusion bound already did. The bands that
  condense alone keep a/β, so every other model keeps its bits (the dirty MgB₂ kept its step: there the band bound is the
  smaller one).
- Fixed (30-oct, evening): with KWT, the heat source and `dissipation` took a band's dissipation as Γ|Ψ̇|², the TDGL one. The
  dissipation of the functional with KWT is (Γ/s)[|Ψ̇|² + γ̃²(Re Ψ*Ψ̇)²], s = √(1 + γ̃²|Ψ|²) (the band equation times Ψ̇*): the old
  form counted the heat of a turning phase s times over (the energy balance was off by +85 % at γ̃ = 3 and ×5.6 at γ̃ = 10, not
  shrinking with dt; now 1e-3, O(dt): a test). Without KWT nothing changes (the bits). Runs with heat and KWT heated too fast.
- The shape editor suggests the material catalogue and drafts the run file of a shape (`[materials.<name>]` from the
  catalogue, the mesh, the shape, the buffer), checked in release 2 and in a headless browser (30-oct).
- The film tools draw the face planes y and z of the box (ghosts) as what they copy: the neighbour in a box, outside in a sculpted
  sample, so that a cylinder of a few cells draws round and a notch reads the same on both faces; `figures.jl` cuts on an interior
  plane.
- The film tools draw the sample from a smoothed indicator and outline it on cuts across the current (`tools/film_geom.py`), and
  name in the legend only the vortices the film shows; the renderer for ParaView was dropped (the VTK data stay).
- `scripts/film.jl` (25-oct): the 3D film of a run from a kept rung state, only around the vortex events (a ring of the last
  samples in memory, kept when the vortex map changes, until it is back for 5 samples or still for 40): an interactive 3D page
  (`tools/film_html.py`; plotly.js inside the page when the Python package is installed, from its CDN otherwise), a GIF with
  matplotlib (`tools/film_gif.py`), and the data as VTK files (`film.pvd`, `vortices.pvd`, XML with raw binary data). A frame
  costs ~5 ms against ~0.1 s per step on the nanowire box: it does not slow the run. The solver is not touched. Since 26-oct
  it says when the vortex map changed at most samples (a continuous flow, where `--every` must be well below the period 2π/V or
  the frames alias like a stroboscope) and writes that fraction in `meta.json` (`changed_fraction`).

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
