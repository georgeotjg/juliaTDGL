# Changelog

## 1.2.0 (2026-10-13)

Same solver and the same results as 1.1.0: the test suite reproduces its reference states to the
bit. What changed is around it.

- **Checkpoints are compressed JLD2** (`restart/state_J*.jld2`): about a quarter of the size of the
  text `.mat`, 5–10 times faster to write and read, exact, stamped with the code version.
  `checkpoint_format = "octave"` keeps the text `.mat` of the earlier releases. `read_state` and
  `load_checkpoint` read both, and every earlier run resumes. A new test resumes a stopped ladder for
  every kind of state (heat with Joule, Φ with τ_Q, drag, bath temperature, shape files, `:prod`) in
  both formats and checks it against the same ladder run without a stop, to the bit. Scripts that
  build the name `state_*.mat` by hand should use `checkpoint_path(dir, label)`.
- **A resumed run with Φ keeps the V_Φ of the points before the stop** (it was NaN).
- **MAT.jl is no longer a dependency** (54 packages instead of 83): it is a package extension, loaded
  only to read binary MATLAB files.
- **The terminal menu and the shape editor server are Julia** (`julia/tui/`, a small package with the
  standard library only, Julia 1.11 or later): same screens and results as the Python versions, which
  are retired. Python is needed only for the optional matplotlib figures. It is also a Pkg app (Julia
  1.12 or later): `Pkg.Apps.develop(path="julia/tui")` installs the command `tdgl-tui`
  (`--editor` opens the editor).
- **Terminal menu:** live runs first, then the most recently active (it was alphabetical); folders
  added with `--add-dir` are searched whole; the `runs/` folder at the root, where the examples write,
  is listed; a long name is cut in the folder first, so the run name stays readable.
- **Cluster:** `examples/slurm_array.sh`, a job array (one task per field); `slurm_job.sh` fixed (one
  BLAS thread, a folder that does not change between submissions, so `RESUME=1` resumes).
- Tested with Julia 1.12 and 1.13.

## 1.1.0 (2026-10-10)

- `maxwell_implicit` (off by default): the curl-curl term of the field equation at the new time
  level. It removes the Maxwell bound on dt; the step is then set by the π-band diffusion. Separable
  solver for dx = dy = dz, sparse LU otherwise. Validated at κ = 25 against the explicit scheme:
  onset brackets, flowing voltages to 0.3 %, fixed points to 10⁻⁸; about 20 times faster per unit
  time at κ = 25, no gain at κ = 2.
- `dt_scale` (default 1): multiplies the time step, for convergence tests in dt.
- `cfl_legacy` (default true): the historical step keeps two bounds inherited from a generalised
  TDGL that this code does not integrate (the factor 1/√(1+γ²) on the diffusion bound and a fixed
  0.03 dl). With `cfl_legacy=false` the step is set only by the stability of the equations integrated
  here: explicit Euler on diffusion plus reaction, Maxwell when explicit, heat or the residual when
  on. Measured on a flowing state: twice the step and V within 0.01 % in the explicit scheme, eight
  times the step and V within 0.25 % with `maxwell_implicit`; the error is first order in dt.
- `pi_substeps` (default 1, off; 0 = automatic): the π band, whose diffusion bound is the smallest
  at ξπ/ξσ = 4, advances with n substeps per step, σ frozen in the coupling and the links rotated at
  the previous step's rate. With `maxwell_implicit`, `cfl_legacy=false` and `dt_scale=0.5` the
  step is set by σ. Validated at ξπ/ξσ = 4 against one substep: onset brackets identical at κ = 2 and
  κ = 25, flowing V to 0.12 %, π current share at the contacts to 0.02 %, η = 0 by symmetry to
  10⁻¹². Against 1.0.0, 3 times faster per unit time at κ = 2 and about 100 times at κ = 25. Not
  available with heat, `T_physical`, Φ or drag.
- The share of the current that crosses the sample (`Isam_arr`, and `Itot_arr`) is the mean over
  the second half of each point, not its last instant: in flow it fluctuates by one to five per
  cent as vortices cross. Dynamics unchanged.
- `Config(:prod)`: the production scheme in one word (implicit curl-curl, stability-only bounds,
  π substeps, half the σ bound, `conv_by_time`, one BLAS thread, `eps_conv_legacy`). Validated
  against the explicit scheme: onset brackets identical at κ = 2 and κ = 25, flowing voltages within
  0.16 %. Recommended for new work; `Config(:legacy)` names the 1.0.0 scheme, which stays the
  library default so that existing scripts reproduce the published numbers.
- `eps_conv_legacy` (off by default; on in `:prod`): `eps_conv` is read per cycle of the 1.0.0
  step and turned into a rate, so a larger step keeps the fixed-point criterion of the explicit
  scheme in physical time.
- `run_preset.jl --scheme prod|legacy` and a `scheme` row in the menu's launch wizard: prod by
  default; runs with Φ or Q fall back to legacy, and runs with heat, drag or a bath temperature use
  prod only where its step is at least twice the explicit one (large κ); a resumed run keeps
  the scheme it started with (legacy for runs that predate the option), and `--restart` with a
  different `--scheme` is refused.
- `Config(:prod)` with heat, a bath temperature or drag: the π band takes one substep (automatic
  substeps fall back to one; n > 1 is still an error) and the curl-curl term stays implicit. Against
  the explicit runs at κ = 25, J = 0.72: a heated flowing state (γ = 0.08) and a collapsed one
  (γ = 0.04) within 0.24 % and 0.01 % in V, with the same maximum temperature and the π band
  extinct in the collapse; drag (ν = 0.05) within 0.24 % in both polarities; a bath temperature
  T = 0.3 (small box, J = ±0.45) within 0.05 %; 20–40 times faster. At κ = 2 with contact heat the
  deciding reverse rung gives the same crossings (3.43 per line) and voltage to four digits, but
  there the explicit scheme is faster.
- `blas_threads` is applied when the implicit solver is built, so scripts that step with `substep!`
  get it too, and `pin_cpu.sh` sets one OpenBLAS thread. On a box with thick padding six OpenBLAS
  threads took six cores and ran slower than one.
- The terminal menu watches only the project's run folders and the ones the user adds
  (`TDGL_TUI_EXTRA`, the `c` key, `--add-dir`); `--help` lists its options and an unknown option is
  an error.
- Printed labels in English (the step limiter, the `v13_diag` line).
- The README and the guide state that a thick buffer carries normal current in parallel with a
  silent sample (no common electrodes), so it is not a closer model of a film in vacuum.
- `conv_by_time` (off by default; on in `:prod`): a resistive point closes once it carries ten
  phase crossings per line after `t_min_GL_res` (the verdict rule), the resistive test is active from
  the first point of a sweep, and a fixed point is not accepted while the second half carries
  crossings. The per-cycle tests depended on the length of a cycle in time.
- Checkpoints carry the link phases of the previous step: a resumed run with substeps is
  bit-identical to the uninterrupted one. Older checkpoints still load.
- `sigma_substeps` (default 1, off): the same substeps for σ, for accuracy studies of larger steps.
  They showed that the production step cannot grow much: in strong flow at κ = 25 the voltage is
  0.16 % off at dt = 0.03 and 3 % off at dt = 0.04, with σ and π each at half their bound. The
  error comes from the field and the links being updated once per step during a phase slip; Strang
  splitting does not reduce it.
- With `cfl_legacy=false`, a warning when σ runs above half its stability bound: there explicit
  Euler is stable but its stiffest mode changes sign every step, and in strong flow at κ = 25 the
  voltage came out 7 % low at the full step.
- `maxwell_implicit` is first order in dt: in a slowly evolving driven state the error at the full
  step is a few per cent in M and halves with `dt_scale=0.5`.
- Checkpoints store the temperature field and the charge residual when they are in use, and a resumed
  run continues with them.
- A sweep warns when `mMax` reserves more than 1 GB for the per-cycle series.
- The geometry preview also draws the cut cells in 3D (`figures/geometry3d.png`).

## 1.0.0 (2026-10-09)

First public release.

**Model.** Two order parameters Ψσ, Ψπ with Josephson coupling, a self-consistent vector potential
and a normal-metal buffer, on a three-dimensional box with a condensation mask f(r).
- The transport current enters through its own Ampère field.
- Γr divides the three terms of the π equation, so it is a ratio of relaxation times and does not
  move the static state.
- Optional, all off by default:
  - a scalar potential by Coulomb projection, with a Thomas–Fermi length and a relaxing residual;
  - interband drag;
  - a temperature field with Joule heating;
  - a bath temperature.

**Numerics.** Link variables on a staggered mesh, gauge Φ = 0 unless the scalar potential is on,
explicit time stepping.
- The kernel is threaded over z planes, with the same result for any thread count.
- `Strict` fixes the order of floating-point operations; `Fast` is about 1.3 times faster.

**Measurement.**
- **Josephson voltmeter:** the gauge-invariant phase along complete transverse lines, unwrapped line
  by line.
- **Convergence:** each point closes as a fixed point or as a stationary limit cycle after a floor
  in physical time.
- **Brackets:** the critical currents come from the crossing-count verdict rule of the methods paper.
- **`Vh_arr`, `Vh2_arr`:** the slopes of the σ and π phases over the second half of each point.
- **`eps_conv_rate`:** a tolerance per unit time instead of per cycle.
- **`Config(:paper)`:** the protocol of the methods paper.
- **The run summary** records the commit, the threads and every `Config` field.

**Tools.**
- Shapes from TOML files and a browser editor.
- A terminal menu that launches and follows runs.
- A shape scout.
- VTK output, and figures of every run.
- Checkpoints after every current point.
