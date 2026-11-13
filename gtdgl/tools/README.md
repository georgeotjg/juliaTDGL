# Tools

- `ladder_sheet.py DIR [DIR ...] [-o out.png]`: the standard sheet of one or more ladders (any run folder with
  `output/rungs_*.csv`): crossings per line, |V|, time per rung and the current through the sample against |J|.
- `delay_fit.py FILE.csv`: the onset from the saddle-node delay law, D(J) = a/√(J − Jc) + b, with the delays measured from
  one silent state (`crossing_times` of the package). Columns `J, delay` (empty: no crossing).

- `state_figures.py DIR`: |Ψµ|² of every band on the three mid cuts (vortices and the supercurrent streamlines on x–y) and the
  share of the supercurrent each band carries along x, from the data `scripts/figures.jl` exports.
- `run_report.py DIR`: `DIR/figures/REPORT.md`, one page with what was run, every rung, the brackets and η, and the figures.
- `collect_runs.py DIR [DIR ...]`: one CSV with a row per run under those folders (options, brackets, η, the delay onsets).

- `film_html.py FILMDIR`: a film of `scripts/film.jl` as one HTML page: 3D, turned and zoomed with the mouse, played with a slider
  (the sample, the vortex cores |Ψ1|² = 0.3, the vortex plaquettes, the mid plane with |Ψ1|² and the supercurrent). With the
  Python package plotly installed the page carries plotly.js inside and opens offline; without it, it loads a pinned version from
  its CDN. The face planes y and z of the box are ghosts (copies of their neighbours) and are drawn as what they copy: in a box,
  the neighbour, so a notch reads the same on both faces; in a sculpted sample, outside, as vacuum, so a cylinder of a few cells
  draws round instead of bulging where it meets the box (`film_geom.solid`).
- `film_gif.py FILMDIR`: the GIF of a film of `scripts/film.jl`, with numpy and matplotlib only: in 3D the sample, the mid plane
  coloured by |Ψ1|² with the supercurrent on it and each vortex as a line through its plaquettes; the mid plane from above; the
  plaquettes against time.


- `film_geom.py`: the shape of the sample for the two film tools (a smoothed indicator, its outline on cuts).

`scripts/figures.jl` calls the first three; `scripts/film.jl` the two film tools. They need Python 3 with NumPy and matplotlib (`run_report.py` and
`collect_runs.py` only the standard library; the run options need Python ≥ 3.11).
