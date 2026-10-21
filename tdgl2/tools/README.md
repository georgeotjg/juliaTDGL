# Tools

- `ladder_sheet.py DIR [DIR ...] [-o out.png]`: the standard sheet of one or more ladders (any run folder with
  `output/rungs_*.csv`): crossings per line, |V|, time per rung and the current through the sample against |J|.
- `delay_fit.py FILE.csv`: the onset from the saddle-node delay law, D(J) = a/√(J − Jc) + b, with the delays measured from
  one silent state (`crossing_times` of the package). Columns `J, delay` (empty: no crossing).

Both need Python 3 with NumPy; `ladder_sheet.py` also matplotlib.
