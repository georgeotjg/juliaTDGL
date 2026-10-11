# julia

The solver package (`TDGL`). The optional terms below are all off by default and meant to be
switched on one at a time.

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/test/run.jl
```

| flag (Config) | default | what it adds |
|---|---|---|
| `use_drag`, `nu_drag` | false, 0 | Mixed-gradient coupling ν Re[(DΨσ)*·(DΨπ)] in both order-parameter equations and in the supercurrent. The π-side term carries ν·r/Γr (no extra 1/mr), which keeps the coupling variational. |
| `use_heat` | false | Temperature field θ(r,t): ∂tθ = D_T∇²θ + η_J s − γ(θ − θ_bath), with α_σ = (1−θ)f and α_π = (a2n−θ)f. Source s = \|∂tΨσ\|² + Γr\|∂tΨπ\|². Parameters `D_T`, `eta_J`, `gamma_bath`. |
| `use_joule` | false | Adds the normal-current Joule term \|E\|² to s (σn = 1, the normal channel of Ampère's law). Requires `use_heat` and `use_phi=false`. |
| `heat_margin` | 0 | Number of cells next to each injection face excluded from s. The injected normal current converts into supercurrent within about one coherence length, with a field and a rotating phase even in the Meissner state; that is contact heat. |
| `T_physical` | false | Treats `T_red` as a uniform bath temperature in both bands: α_σ = (1−T)f, α_π = (a2n−T)f, no extra (1−T) on the supercurrent, and ξπ/ξσ fixed at T = 0. With it off, the σ band does not feel `T_red`, so the default is physical only at T = 0. Identical at T = 0. |
| `use_Qss`, `D_Q` | false | Diffuses the Coulomb-projection residual Q: τ_Q(∂tQ − D_Q∇²Q) + Q = σ. Requires `use_phi` and `tau_Q > 0`. This is not a Schmid–Schön charge-imbalance mode. |

Diffusion can lower the time step: dt ≤ 0.15 dx²/D for the largest active D.

## One run from the command line

```bash
./julia/scripts/pin_cpu.sh 8,9 julia --project=julia julia/scripts/run_preset.jl \
    --geom WEDGE --kappa 25 --mesh production --window short --J 0.72 --heat 0.08 --joule 1 --margin 8
```

`pin_cpu.sh 8,9` (optional) pins the run to two cores. This writes `julia/runs/<name>/` (`job.json`,
`j_history.csv`, snapshots, checkpoints for `--restart`). Meshes: `small` 20×14×4, `production`
40×24×6, `dx2` 80×48×12; windows (minimum t_GL per static / flowing point): `smoke` 20/40, `short`
20/200, `long` 200/1200, `hard` 1200/1200, `paper` 1500/2000 (the protocol of the methods paper). Optional terms: `--drag 1 --nu ν`, `--heat γ [--joule 1] [--margin m]`, `--qss 1`;
scalar potential: `--phi off|coulomb|tf [--lambda-tf 2]`; geometry from a file: `--shape shape.toml`;
bath temperature: `--T 0.3`, which switches on `T_physical`; custom mesh and times: `--Nx --Ny --Nz --dx`,
`--tmin --tminres`; scheme: `--scheme prod|legacy` (prod by default; with Φ or Q legacy, and with
heat, drag or T the faster of the two, which is prod at large κ). The terminal menu
`julia/scripts/tdgl_tui.py` builds the same command, one term at a time.

## Geometry from a file

`GEOM_MODE="CUSTOM"` (or `"CUSTOM_REF"` for the y-mirror) with `geom_file="shape.toml"` reads a
list of regions of the box, in units of ξσ: circle, rectangle, triangle (with an apex offset, for
teeth), regular polygon, free polygon, sphere, a pixel layer whose characters each carry a z
range and a value of f, and a voxel layer for any three-dimensional pattern. Because positions are physical, the same file rasterises on any mesh.
Catalogue shapes written as files reproduce the catalogue mask exactly (`test/test_shapes.jl`).

A browser editor draws these files and checks, while you draw, the mirror symmetry that would force
η = 0, contact with the injection faces, and normal shunts:

```bash
python3 julia/editor/server.py        # http://127.0.0.1:8765/
```

Use one term at a time against a control run with the same protocol. With several terms on, a
change in a number cannot be attributed to any one of them. The coefficients above are model
values chosen to switch a term on, not material constants of MgB₂.

`test/test_optional_terms.jl` checks the guards, that an inactive term leaves the step untouched, the
diffusive time step, the size of the Joule term in a forced normal state, the contact margin, that
the diffused residual departs from the local one, and that `T_physical` is bit-identical at T = 0 and
gives the uniform equilibrium |Ψσ|² = 1 − T, |Ψπ|² ≈ a2n − T at T = 0.3.
