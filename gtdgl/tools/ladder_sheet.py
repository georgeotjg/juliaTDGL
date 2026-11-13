#!/usr/bin/env python3
"""The standard sheet of one or more ladders of TDGL 2 (any run folder with output/rungs_*.csv).

    python3 ladder_sheet.py DIR [DIR ...] [-o out.png] [--title TEXT]

(a) crossings per line over the rung (slope of the second half × rung time, as the verdict counts) against
    |J|, with the verdict thresholds: < 1 silent, >= 10 flowing, in between undecided (shaded);
(b) |V| (second-half slope) against |J|;
(c) the time each rung was integrated (the silent window and the resistive floor show up here);
(d) the current through the mid plane over the nominal, and the share of it through the sample.
Open markers: closed 'F' (fixed point); filled: 'V'; crosses: anything else.  The legend gives each
bracket (lo, hi] by the verdict rule and the undecided rungs inside it."""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("-o", "--out", default="")
ap.add_argument("--title", default="")
a = ap.parse_args()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})


def rungs(f):
    rows = [r for r in csv.DictReader(L for L in open(f) if not L.startswith("#"))]
    o = []
    for r in rows:
        if r["conv"].strip() == "":
            continue
        o.append(dict(J=abs(float(r["J"])), V=abs(float(r["V"])), T=float(r["t_int"]),
                      c=float(r["crossings_per_line"]), conv=r["conv"].strip(),
                      onset=r.get("onset_class"),
                      It=float(r.get("Itot", "nan") or "nan"), Is=float(r.get("Isam", "nan") or "nan")))
    o.sort(key=lambda x: x["J"])
    return o


def bracket(d):
    if any(x["onset"] for x in d):
        verdicts = [(x["J"], x["onset"]) for x in d if x["J"] > 1e-12]
    else:
        verdicts = [(x["J"], "D" if x["c"] >= 10 or x["conv"] == "N" else
                     "S" if x["c"] < 1 else "U" if x["c"] < 10 else "") for x in d]
    hi = next((j for j, flag in verdicts if flag == "D"), None)
    if hi is None:
        return "no flowing rung"
    lo = max((j for j, flag in verdicts if j < hi and flag == "S"), default=None)
    und = [j for j, flag in verdicts if j < hi and flag == "U" and (lo is None or j > lo)]
    back = [j for j, flag in verdicts if j > hi and flag == "S"]
    s = f"({lo:.3f}, {hi:.3f}]" if lo is not None else f"(?, {hi:.3f}]"
    und and (s := s + " undecided " + ", ".join(f"{j:.3f}" for j in und))
    back and (s := s + " | silent above: " + ", ".join(f"{j:.3f}" for j in back))
    return s


series = []
for d in a.dirs:
    p = Path(d)
    for f in sorted(p.glob("**/output/rungs_*.csv")):
        rel = f.parent.parent.relative_to(p.parent) if f.parent.parent != p else p.name
        series.append((f"{rel} {f.stem.replace('rungs_', '')}", rungs(f)))
if not series:
    raise SystemExit("no output/rungs_*.csv under " + " ".join(a.dirs))

fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.6 + 0.16 * len(series)))
for q, (name, d) in enumerate(series):
    col = f"C{q % 10}"
    J = [x["J"] for x in d]
    for A, key, floor in ((ax[0, 0], "c", 1e-4), (ax[0, 1], "V", 1e-8)):
        A.semilogy(J, [max(x[key], floor) for x in d], "-", color=col, lw=0.7, alpha=0.6)
        for x in d:
            flag = x["onset"] or x["conv"]
            mk, fc = ("o", "white") if flag in ("F", "S") else (("o", col) if flag in ("V", "D") else ("x", col))
            A.semilogy(x["J"], max(x[key], floor), mk, color=col, mfc=fc, ms=4)
    ax[0, 0].plot([], [], "-o", color=col, ms=3, label=f"{name}: {bracket(d)}")
    ax[1, 0].plot(J, [x["T"] for x in d], "-o", color=col, ms=3, lw=0.7)
    ax[1, 1].plot(J, [x["It"] for x in d], "-o", color=col, ms=3, lw=0.7)
    ax[1, 1].plot(J, [x["Is"] for x in d], "--", color=col, lw=0.7)
A = ax[0, 0]
A.axhspan(1, 10, color="0.9", zorder=0); A.axhline(1, color="0.5", lw=0.5); A.axhline(10, color="0.5", lw=0.5)
A.set_ylabel("crossings per line")
fig.legend(*A.get_legend_handles_labels(), fontsize=6, frameon=False, loc="lower center", ncol=1)
ax[0, 1].set_ylabel(r"$|V|$ (band 1, closing window)")
ax[1, 0].set_ylabel(r"rung time [$\tau$]")
ax[1, 1].set_ylabel("total/nominal (solid), share in sample (dashed)"); ax[1, 1].ticklabel_format(axis="y", useOffset=False)
for A, l in zip(ax.flat, "abcd"):
    A.set_xlabel(r"$|J|$ [$J_0$]")
    A.text(0.98, 0.04, f"({l})", transform=A.transAxes, fontsize=8, ha="right")
a.title and fig.suptitle(a.title, fontsize=8)
fig.tight_layout(rect=(0, 0.03 * len(series) / (1 + 0.03 * len(series)) + 0.02, 1, 1))
out = Path(a.out) if a.out else Path(a.dirs[0]) / "ladder_sheet.png"
fig.savefig(out, dpi=160)
print(out)
for name, d in series:
    print(f"{name}: {bracket(d)}")
