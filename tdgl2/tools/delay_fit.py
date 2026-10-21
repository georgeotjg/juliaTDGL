#!/usr/bin/env python3
"""The onset from delays.  Near a saddle-node onset the delay before the first phase slip, after a jump from a silent state,
grows as D(J) = a/sqrt(J - Jc) + b; a window of fixed length biases the onset up (a rung just above it closes silent).
With the delays at two or more currents above the onset, all from the same silent state (`crossing_times` in TDGL), this
gives Jc: with three or more points the three parameters (b by least squares at each Jc; Jc by bisection on the middle
residual with three points, by least squares otherwise); with two, b = 0.  On the two-band notched diode at kappa = 2 the two forms
agree to 2e-4, and the same Jc comes from a base 17 % below the onset as from the last silent rung (+-0.001).

    python3 delay_fit.py FILE.csv          columns: J, delay   (delay empty or nan: no crossing; listed, not fitted)
"""
import csv, math, sys
import numpy as np


def fit(J, D):
    J = np.asarray(J, float); D = np.asarray(D, float)
    if len(J) == 2:
        r = (D[0] / D[1]) ** 2
        Jc = (r * J[0] - J[1]) / (r - 1)
        return Jc, D[0] * math.sqrt(J[0] - Jc), 0.0

    def ls(Jc):
        x = 1 / np.sqrt(J - Jc); A = np.vstack([x, np.ones_like(x)]).T
        c = np.linalg.lstsq(A, D, rcond=None)[0]
        return c, D - A @ c
    lo, hi = J.min() - 0.5 * (J.max() - J.min()) - 0.1, J.min() - 1e-9
    if len(J) == 3:
        mid = np.argsort(J)[1]
        f = lambda Jc: ls(Jc)[1][mid]
        flo = f(lo)
        for _ in range(200):
            m = (lo + hi) / 2
            if np.sign(f(m)) == np.sign(flo):
                lo, flo = m, f(m)
            else:
                hi = m
        Jc = (lo + hi) / 2
    else:
        grid = np.linspace(lo, hi, 4000)
        Jc = grid[np.argmin([np.sum(ls(x)[1] ** 2) for x in grid])]
    c, _ = ls(Jc)
    return Jc, c[0], c[1]


if __name__ == "__main__":
    rows = list(csv.DictReader(open(sys.argv[1])))
    pts = sorted((float(r["J"]), float(r["delay"]) if r["delay"].strip() not in ("", "nan", "NaN") else None) for r in rows)
    for J, D in pts:
        print(f"J = {J:.4f}   delay = {'none' if D is None else f'{D:.0f}'}")
    ok = [(J, D) for J, D in pts if D is not None]
    if len(ok) < 2:
        sys.exit("need two or more currents with a delay")
    Jc, a, b = fit(*zip(*ok))
    print(f"Jc = {Jc:.4f}   (a = {a:.1f}, b = {b:.1f}; {len(ok)} points)")
    Js = [J for J, _ in ok]
    lo = min(Js) - 0.5 * (max(Js) - min(Js)) - 0.1
    if len(ok) >= 3 and Jc - lo < 1e-3 * (min(Js) - lo):
        print(f"warning: Jc is at the edge of the search range ({lo:.4f}); the delays do not fix it (measure closer to the onset)")
