# juliaTDGL 2 (`TDGL` 2.0)

A three-dimensional generalised time-dependent Ginzburg–Landau (gTDGL) solver in Julia for multiband superconductors
under applied field and transport current: any number of bands coupled by a Josephson matrix, each with its own
stiffness, relaxation rate and Kramer–Watts-Tobin (KWT) parameter, on a mesoscopic sample of any shape. It was written for
the geometric superconducting diode in two-gap materials (MgB₂ motivated it; its parameters are in `materials/`), and it
is not tied to them: a material is a file. It grew out of release 1.2 (two bands, standard TDGL), which it reproduces when
the new terms are off, and adds KWT per band, an open boundary for the self-field, an insulating buffer, and a protocol
that brackets onsets by the delay of the first phase slip.

**Author:** Jorge Alfredo Robles Calderón — Departamento de Física, Universidad Nacional de Colombia, Bogotá.

The model, the numerics, the protocol and the checks are described in [`docs/solver_guide_2.pdf`](../docs/solver_guide_2.pdf).

**Quick start** (Julia 1.13 or later):

```bash
julia --project=tdgl2 -e 'using Pkg; Pkg.instantiate()'
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/test/run.jl                                  # the suite (a few minutes)
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/run.jl tdgl2/examples/wedge_k25.toml    # a notched box at κ = 25
```

## What it solves

N order parameters Ψ_µ with the vector potential **A** on a box with a buffer of a few cells (lengths in ξ₁, time in
t_GL = ξ₁²/D₁, fields in H_c2 of band 1, currents in J₀). For each band

```
(Γ_µ/√(1 + γ̃_µ²|Ψ_µ|²)) [∂t Ψ_µ + γ̃_µ² Ψ_µ Re(Ψ_µ* ∂t Ψ_µ)] = g_µ (∇ − iA)² Ψ_µ + (ε_µ f − β_µ|Ψ_µ|²) Ψ_µ + Σ_ν γ_µν Ψ_ν
σ_n ∂t A = Σ_µ J_µ − κ² ∇×∇×A,     J_µ = g_µ Im[Ψ_µ*(∇ − iA)Ψ_µ]
```

in the gauge Φ = 0, where the charge imbalance of the contacts (Gurevich–Vinokur) is the rotation of Ψ in the conversion
layers. With γ̃_µ = 0 this is the TDGL of release 1.2; `two_band(...)` builds the two-band model of the earlier papers.

- **Bands and materials.** Any number of bands (`Model`, `Band`), a symmetric Josephson matrix γ_µν (its sign gives s±),
  passive bands (ε_µ ≤ 0, condensate induced by the coupling), σ_n given or that of a dirty material (`σn = :dirty`).
  Materials and runs are TOML files (`materials/`, `examples/`; a new material starts from `materials/template.toml`); the optional terms of 1.2 (heat with Joule and a contact
  margin, interband drag, a bath temperature) go in the same file (`[heat]`, `drag` in `[scheme]`, `T` in `[material]`).
- **Self-field boundary** (`selffield_bc`): `:bar` (default; the field of the nominal current spread over the section of
  the sample on every outer face, so the sample carries J·W·d), `:slab` (release 1.x), `:open` (the bar field plus the
  free-space field of every other current of the box on all six faces, from the lattice Green function by FFT; the field
  of the screening currents that a mirror face suppresses at small κ).
- **Buffer.** Normal metal (σ_n) by default; insulating outside the bar of the sample and its contacts with `σvac` (`sigma_vac` in a run file)
  (preconditioned conjugate gradients started from the previous step's solution; ×8 the cost of a metal buffer).
- **Geometry.** The shapes of release 1.2 (catalogue, TOML files, the browser editor), unchanged.

## How it solves it

- Staggered mesh with gauge-invariant link variables, as in 1.2; the curl-curl term implicit (one separable solve per step).
- Bands explicit; with KWT, Runge–Kutta–Legendre stages (`rkl`, default) bridge each band's bound to the step of the model
  without KWT (on the 40 × 24 × 6 box at κ = 2, KWT with γ̃ = 10 costs ×2.2 per t_GL with the defaults; the explicit step
  at the KWT bound would be ten times shorter). With KWT the bands are stepped in the gauge that turns with them
  (`rotframe`, on by default with KWT): a stationary state with charge imbalance is then a fixed point of the step for any
  dt (without it the amplitude in the conversion layers depended on dt at first order, ×√(1 + γ̃²|Ψ|²)).
- **Protocol** (`ladder`, `run_ladders`): each current point runs for a minimum time and closes as a fixed point ('F'), a
  flow ('V'), a sample gone normal ('N') or a heat cut ('T'); the verdict counts vortex crossings per measuring line
  (silent below one, dissipating at ten) in band 1, so band 1 must carry a condensate of its own (not a passive band). Near a saddle-node onset the delay of the first slip diverges as a/√(J − Jc), so
  a fixed window biases the onset up: `Protocol(t_confirm_GL=...)` follows the silent rungs below the first flowing one,
  and `delay_bracket` gives the onset by the delay. `crossing_times(st, s, J; H, T)` measures delays directly and
  `tools/delay_fit.py` fits the law.
- **Stop and resume.** The checkpoint of every closed point holds the state and all the ladder carries to the next point;
  `STOP.txt` in the run folder stops a run after the current point, and `scripts/run.jl ... --resume` continues it, to the
  bit (a killed run loses only the point it was on). `--polarity +` and `--polarity -` run the two polarities as two
  processes into the same folder; the second to end writes the summary.
- **Output.** `run_config` writes the run file, the rung tables, checkpoints (JLD2) and `output/summary.json` with units,
  code version, commit, the scheme as resolved and the regime numbers (`regime`: D_π/D_σ, ℓ_J/L, γ²/(α₁α₂), σ_n/Σg_µ, u_µ, and the conversion
  length λ_c with L/λ_c). A shape file the run names is copied next to `run.toml`, so the folder alone rebuilds the run.
- **Figures of a run.** `scripts/figures.jl RUNDIR` draws, from the folder of a run, the standard set into `RUNDIR/figures/`
  with a `README.txt` that says what each one is: the sheet of the ladder (`tools/ladder_sheet.py`), |Ψµ|² of every band on
  the three mid cuts with its vortices (a cut that would fall in sculpted vacuum, such as the gap between stacked slabs,
  moves to the centre of the nearest run of full planes), and the share of the supercurrent each band carries plane by plane
  (`tools/state_figures.py`), for the last silent and the first flowing rung of each polarity when the run kept its rung
  states (`keep_rung_states = true`), otherwise for its last state, and `REPORT.md` (`tools/run_report.py`): one page with
  what was run, every rung, the brackets and η, and what each figure shows, for a person or a language model to read without
  the code. It does not touch the run.
  `tools/collect_runs.py DIR...` writes one CSV with a row per run under those folders (options, brackets, η, the delay
  onsets, version and commit): a campaign as one table.

## A first run

```bash
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/run.jl tdgl2/examples/wedge_k25.toml   # κ = 25, minutes
```

Every option has a run file in `examples/` to start from (one command each, as above; times on one core of a laptop):

| file | what it shows | time |
|---|---|---|
| `wedge_k25.toml` | the two-band notched box at κ = 25 | minutes |
| `kwt_k2.toml` | KWT with γ̃ = 10 at κ = 2 and the onset by the delay: (0.73, 0.74] and (0.75, 0.76], η = −1.3 % | ~1 h (35 min with `--polarity` in two processes) |
| `open_k2.toml` | the open boundary of the self-field | 13 min |
| `insulating_k2.toml` | the insulating buffer (`sigma_vac`) | 35 min |
| `heat_k25.toml` | heat with Joule and a contact margin, and drag: (0.3, 0.4] and (0.5, 0.6] | 16 min |
| `three_bands.toml` | three bands with frustrated couplings (the time-reversal-breaking state) | 6 min |

From Julia:

```julia
using TDGL
md = two_band(xi_ratio=4.0, κ=2.0, gamma_kwt=10.0)
gr = Grid(40, 24, 6; dx=0.5)
mk = build_masks(GeomSpec(GEOM_MODE="WEDGE"), gr)
st = Stepper(md, gr, mk)                                   # production scheme; rotframe on (KWT)
L  = ladder(st, initial_state(md, gr, mk, 0.03), [0.0, 0.5, 0.7, 0.74]; H=0.03,
            pr=Protocol(t_confirm_GL=3000.0, stop_after_V=1))
delay_bracket(L)
```

The terminal menu of release 1.2 (`julia/scripts/tdgl-tui.sh`) lists the runs of 2.0 that have an `out_dir`, launches
one from a run file (key `n`; two CPUs run the polarities in parallel), stops it after the current point (`s`) and resumes
it (`u`). Its launch wizard (`l`: starting point, geometry, parameters, optional term) ends with the choice of release:
`2` writes the run file of release 2 (the two-band model of 1.2, KWT off) into the run folder and launches it.

## Which release, which option

| | release 1.2 (`julia/`) | release 2.0 (`tdgl2/`) |
|---|---|---|
| bands | 2 | N, a Josephson matrix, passive bands |
| KWT relaxation | — | per band |
| scheme | implicit curl-curl by default; `Config(:legacy)` the explicit scheme of 1.0.0 | implicit curl-curl, the default |
| stiff bands | Euler substeps | Runge–Kutta–Legendre stages and the rotating frame |
| self-field boundary | the slab of 1.x | `:bar` (default), `:open`, `:slab` |
| buffer | normal metal | normal metal or insulating |
| scalar potential Φ (Coulomb projection, Thomas–Fermi), diffused residual Q | yes (the run switches to the explicit scheme by itself) | not yet |
| heat (Joule, contact margin), drag, bath temperature | yes (in `:prod`, without π substeps) | yes, with every other option |
| interband quasiparticle exchange (normal fluids per band) | — (one fluid) | `qp_exchange = ν` (two bands) or a matrix of rates (N bands); off = one fluid |
| anisotropic bands (a mass per axis) | — | `m_axes`, or `m_x`, `m_y`, `m_z` in a material file |
| thermal (Langevin) noise | — | `noise = T̃`, seeded, resumable (with KWT too); `noise_field = true` on A too |
| numerical modes | `Fast` (default) and `Strict` (digit for digit with earlier releases) | native arithmetic |
| onset by the delay, the 'N' verdict | — | yes |
| a run is | `Config`, `run_preset.jl` flags or the menu wizard | a TOML file, or the menu (key `n`) |
| stop and resume | yes | yes, to the bit |
| shapes, catalogue and editor | yes (figures on any face, kept or removed) | the same, and vacuum: a sculpted sample (a cable, a film with a hole) with leads of its own section |

Use 1.2 for the scalar potential, for the `Strict` mode, or to reproduce the published work of release 1.x; 2.0 for the rest.

## Performance

On one core, the two-band notched box (40 × 24 × 6 cells, dx = 0.5 ξ, κ = 2):

| run | s per t_GL | relative |
|---|---|---|
| defaults, no KWT | 0.078 | 1 |
| KWT with γ̃ = 10 (defaults) | 0.17 | 2.2 |
| `:open` boundary | 0.092 | 1.2 (1.03 with KWT) |
| insulating buffer, two cells, σvac = 0.01 | 0.79 | 8 against a metal buffer of two cells (0.101) |
| `dt_scale = 1` | half | 0.5: for onsets (below) |
| insulating buffer with `pcg_tol = 1e-7` (default 1e-11) | | 0.65 of the default buffer step (12 iterations of its solver per step instead of 30) |

A rung of 1600 t_GL is about two minutes without KWT. What it costs in accuracy:
- `pcg_tol = 1e-7` with the insulating buffer: on the notched cable at J = 0.6, after 400 steps the states differ by 2e-8 in
  Ψ and 6e-9 in h from those at the default tolerance, far below the fixed-point tolerance of a rung (1e-4 per cycle); the
  default stays 1e-11.
- `dt_scale = 1` for onsets: on this box the delay of the first slip changed by 0.2 % without KWT, and with KWT (γ̃ = 10)
  the silent state agreed to six digits and the delays (148.1 and 90.7 t_GL at 0.74 and 0.75) were the same. Flowing
  voltages were not checked: quote them at the default step.
- The step is set by stability, and the accuracy ceiling is close above it: in strong flow the voltage is a few per cent
  off at dt ≈ 0.04 (release 1.2). Quote voltages at the default step.
- Campaigns: one single-threaded process per physical core (`OPENBLAS_NUM_THREADS=1`); on a laptop the total gain
  saturates near 2.5 times one process (memory bandwidth). Two polarities as two processes halve the time of one run.
- Threads within one run (`threads = true`) give the same bits but no speed on a laptop (×0.8–1.06 with 2–8 threads,
  measured on this box and on the 80 × 48 × 12 one at dx = 0.25 ξ, which costs about 3.5 s per t_GL).

## Checks

- **Release 1.2 reproduced** (regression tests on the full state): the step, the masks, the checkpoints and a two-polarity
  ladder point by point, with the new terms off and `selffield_bc = :slab`. To the bit with OpenBLAS's Haswell kernels and
  code for that CPU (`TDGL_TEST_BITS=1 OPENBLAS_CORETYPE=Haswell julia -C haswell ...`, any x86 machine with AVX2); to
  10⁻¹⁰ on others, where the dense products and the SIMD sums round differently. Verdicts and counts are exact either way.
- **Energy balance**: the discrete dF/dt = −D + W, including the work of the boundary links.
- **dt**: with `rotframe` the KWT state is the same to six digits at dt, dt/2, dt/4 and with explicit substeps.
- **pyTDGL** (one band, 2D, version 0.9; the one-band equation of this code with Γ = u, σ_n = 1, κ large):
  - the depairing current of a narrow strip, 0.385 (pyTDGL's current unit is 4 times the GL unit);
  - the conversion length with KWT, γ̃ = 0–30: 0.2 % between the codes and 0.1 % from λ_c² = √(1 + γ̃²|Ψ|²)/(u|Ψ|²);
  - the onset of a notched strip, independent of γ̃ in both codes; with the mesh it approaches pyTDGL's value (8 % below
    it at dx = 0.5 ξ, 4 % at 0.25 ξ: the O(dx) error at the notch tip).
- **Three bands**: equal repulsive couplings frustrate the phases; the ground state breaks time reversal with phases 0,
  ±2π/3 and |Ψ|² = 1 + |γ| (analytic), to 10⁻⁶.
- **Symmetry**: a mirror-symmetric sample gives η = 0 to machine precision (also with `:open`).
- `:open` resumes exactly from a checkpoint and keeps the J → −J symmetry at H = 0.
- **Normal fluids**: in a long strip the σ share of the converted current follows the 1D two-fluid law (with the insulating
  buffer to 0.004); a sculpted cable gives the square bar's share; the conjugate-gradient solve of a sculpted sample satisfies
  its cut-link equation to 10⁻⁸ and is the separable solve when nothing is cut; with two bands the N-fluid model is the two-fluid
  one to rounding, and its Joule heat too.
- **Noise**: decoupled nodes reach the mean, variance and correlation of the linear theory (with KWT too); on the field, the
  energy of the free modes of a normal box is Σ (T̃/2)/(1 + λdt/2) to 0.03 % (σn and the insulating buffer).

## Known limitations

- **Absolute onsets carry an O(dx) error** of 5–12 % at dx = 0.5 ξ, at any κ: the tip of a notch is resolved at the scale
  of a cell, and at small κ the rows on the faces of the sample, Neumann copies without an equation of their own, carry a
  large share of the screening supercurrent (51 % at κ = 2, dx = 0.5; 30 % at 0.25). Ratios between polarities (η) and
  trends do not feel it; quote absolute onsets with their mesh.
- **With KWT the conversion at the contacts is long** (×3 with γ̃ = 10): keep `regime(md; L).L_over_lambda_c` ≳ 10, or the
  middle of the sample carries unconverted normal current that moves the onset by itself.
- **The charge imbalance is slaved.** KWT per band is the band-resolved branch (Schmid–Schön) imbalance in its slaved
  form, in the gauge Φ = 0, and `qp_exchange` adds a finite interband exchange. Its memory (which acts on the resistive branch,
  not on a saddle-node onset) and the scalar potential of release 1.2 (Coulomb projection, Thomas–Fermi) are not in.
- **Normal fluids per band** (`qp_exchange`: a rate for two bands, a matrix of rates with detailed balance for N). With the
  insulating buffer they live in the conductor (zero flux to the insulator; a strip then follows the 1D two-fluid law with the
  code's contact to 0.004, where the metal buffer, which conducts and exchanges around it, is 0.025 off); in a sculpted sample
  the links to its vacuum are cut and δ is solved by conjugate gradients (20–60 iterations a step, about 3 times the cost of
  the buffer alone); with the Joule heat the fluids add σn Σ s_µ|∇μ_µ|² and the exchange Σ G_µν(μ_µ − μ_ν)² (ν = 1 heats the
  notched box 55 % more than one fluid). Validated against the 1D two-fluid law, not yet against experiment; the result depends on the share of σn between the
  bands (`qp_split`, by default the dirty one, ∝ gΓ/β), a model choice to quote with it.
- **Anisotropy**: a mass per axis and band (`m_axes`); σn and drag stay isotropic. **Noise**: on the bands (with KWT, its Ψ-dependent mobility and the Itô drift), and with
  `noise_field = true` on A, the same bath (a kick on every link phase before the implicit solve; in a normal box the field
  energy of the free modes is Σ (T̃/2)/(1 + λdt/2) to 0.03 %, with σn and with the insulating buffer: the implicit Euler of the
  field leaves the stiff mesh modes colder, the slow ones right to O(λdt)).
  With KWT keep the rotating frame (the default): in a box of coupled nodes ⟨|Ψ|²⟩ is that of γ̃ = 0 within 0.3 % at small
  steps and 0.1–1.7 % at the production step; without the frame the radial error of the rotation, fed by the noise, leaves it
  19 % high at the production step.
- **A sculpted sample** (a shape with `vacuum = true`: a cable, a hole through the film) needs the insulating padding
  (`sigma_vac`) and costs what it costs (about 8 times a metal buffer); its surface is free (zero flux), its leads take its
  section, and J is the current density (the sample carries J times its meshed section). Not with heat or drag.
  f = 0 is a normal metal, not vacuum.
- **The implicit solve needs dx = dy = dz.**
- **The insulating buffer is the slow option** (about 8 times a metal buffer; see Performance): its preconditioner does not
  see the contrast between the bar and the buffer.
- **No GPU.** The loops are written per node and per link with no reductions; the band kernel ported to
  KernelAbstractions.jl gives the CPU loop's bits (a development check, not in the package). The full port and its speed
  need a GPU.

## Citing

The solver of release 1.x, the protocol and its failure modes:

> Robles Calderón, J. A. (2026). *Silent failure modes in three-dimensional two-band time-dependent Ginzburg–Landau
> simulations of geometric superconducting diodes* (Version 1.0.0) [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.23163207

The results of 2.0 (KWT and the conversion partition in a two-band diode) are in preparation.

## License

MIT. The code started from a short one-band TDGL script provided by J. Barba-Ortega (Universidad Nacional de Colombia).
