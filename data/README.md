# Regression states

Stored states that the regression tests compare against, array by array. They guard against changes
of the code that are not meant to change the results.

| folder | what it is | what it exercises |
|---|---|---|
| `reference_12x8x4/` | 12×8×4 notched box, two bands, Γr = 1.3, H = 0.50, J = 0, 300 cycles | the kernel, the field update, the mask; Meissner state (no vortex, no voltage) |
| `transport_20x14x4/` | 20×14×4 notched box, J = ±0.10, 400 cycles per point | the voltmeter: 105 measuring lines, one vortex, phase slips, the resistive branch and per-line unwrapping; V(+J) = 1.5e-1 against V(−J) = 5.3e-5 |
| `v14_16x10x4/` | 16×10×4, scalar potential on (Coulomb projection, Dirichlet Φ = 0), J = 0.08, 120 cycles | the scalar-potential step |

The first two were produced by an earlier implementation of the same scheme, the third by this
package. Each point runs a fixed number of cycles (`eps_conv = 1e-14`, `iter_min = mMax`), so the
comparison does not depend on a stopping rule. The states are text `.mat` files, read by `read_state`
(`julia/src/octmat.jl`).
