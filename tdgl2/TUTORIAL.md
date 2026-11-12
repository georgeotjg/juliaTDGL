# The first hour with juliaTDGL 2

From a clean checkout to an onset, its efficiency, the figures and a film of the vortex that carries the voltage. Every command
runs on one core of a laptop; the times are those of an i5-1235U. Julia is needed; Python 3 with NumPy and matplotlib only for
the figures (plotly, if installed, makes the 3D page work offline).

## 1. Install

```bash
julia --project=tdgl2 -e 'using Pkg; Pkg.instantiate()'
```

## 2. A run from a file

Copy an example and keep the state of every rung, so that the tools of steps 4–6 can start from them:

```bash
cp tdgl2/examples/wedge_k25.toml mine.toml
# in mine.toml: under [protocol] add   keep_rung_states = true
#               under [run]      set   out_dir = "runs/mine"
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/run.jl mine.toml --polarity + &
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/run.jl mine.toml --polarity - &
wait
```

Two processes run the two polarities in parallel into the same folder; the second to end writes
`runs/mine/output/summary.json`, with the onset bracket of each polarity and the efficiency η. The terminal menu
(`julia julia/scripts/tdgl_tui.jl`) lists the run while it goes: key `n` launches a run file from the menu itself.

## 3. Look at it

```bash
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/figures.jl runs/mine
```

`runs/mine/figures/REPORT.md` says what was run, every rung, the brackets and η, and what each figure shows: the sheet of the
ladders, |Ψ|² of each band on three cuts with its vortices and the supercurrent lines, and the share of the current each band
carries along the sample.

## 4. A finer onset

The coarse ladder gives a bracket of its step. Refine it from the state of the last silent rung:

```bash
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/refine.jl runs/mine + --step 0.01 &
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/refine.jl runs/mine - --step 0.01 &
wait
```

Each prints the bracket from all the rungs so far, and the second to end prints η. Run it again, with a smaller step, for a finer
bracket; with a ladder that never flowed it carries on upwards instead (`--up`, `--to`). Near an onset the first slip can take
much longer than a rung (its delay diverges), so a bracket for a paper is confirmed by following the silent rungs below it:

```bash
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/refine.jl runs/mine + --confirm 1500 &
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/refine.jl runs/mine - --confirm 1500 &
wait
```

A silent rung that slips within the 1500 t_GL becomes `D` (a flow after a delay), and the bracket moves down.

## 5. A film of the vortex

```bash
OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/film.jl runs/mine + --J 0.40 --events 1 --every 0.2
```

It steps the kept silent state at the given current and keeps the frames around the first vortex event: its entry at the tip
of the notch, its crossing and its exit, about 130 frames: `film.html` (open it in a browser and turn it with the mouse),
`film.gif`, and the data as VTK files, in `runs/mine/figures/film_J+_J+0.4000/`. A sample every 0.2 t_GL (`--every`): the
vortex rests at the tip for a few t_GL before it crosses, and with the default 0.05 that rest already ends the event. From the
menu: `m` in the details of the run, `w` to open the page.

## 6. Change the physics

- **Another shape:** draw it in the editor (`julia/editor/shape_editor.html`, or key `g` of the menu) and name it in
  `[geometry] geom_file`. A shape with vacuum needs `sigma_vac` in `[scheme]`: 0.01; 0.1 runs 1.5 times faster and gave the same onsets in the
  notched nanowire, but a resistive voltage 16 % lower.
- **Another material:** the bands in `[material]` (`[[material.band]]` tables: `xi_ratio`, `a`, `Gamma`, `gamma_kwt`), or a file
  of `tdgl2/materials/`.
- **The options of release 2:** each has its example in `tdgl2/examples/` (KWT, the open boundary, the insulating buffer, heat, three
  bands, a sculpted nanowire), listed in the README.
- **A vortex lattice:** a smooth film, a field of a few tenths of H_c2 and `start = "field_cooled"` in `[run]`. From the Meissner
  state a smooth sample keeps the field out up to its entry field.

## 7. Many runs

Make one run file per case, run them (one process per core), then

```bash
python3 tdgl2/tools/collect_runs.py runs/
```

writes one CSV with a row per run: its options, brackets, η and code version.

## Where to read more

- `README.md`: what the code solves, the options and the checks.
- `docs/solver_guide_2.pdf`: the model, the discretisation, the protocol of measurement and the limitations.
- `tdgl2/tools/README.md`: every tool.
