# Changelog

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
