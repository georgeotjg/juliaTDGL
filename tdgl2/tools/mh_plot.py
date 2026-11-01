#!/usr/bin/env python3
"""The magnetisation loop of scripts/mh.jl (output/mh.csv of its folder).

    python3 mh_plot.py mh.csv [mh.csv ...] [-o mh.png] [--title TEXT]

(a) -M against H, the up branch with filled markers and the down branch with open ones (an open loop: the fields a
    vortex enters at and leaves at differ);
(b) the vortex count of each band against H, the same markers.
Crosses: a point that did not close as a fixed point ('V', 'N', ...)."""
import argparse, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("-o", "--out", default="mh.png")
ap.add_argument("--title", default="")
a = ap.parse_args()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
for n, f in enumerate(a.files):
    rows = list(csv.DictReader(L for L in open(f) if not L.startswith("#")))
    nb = sum(1 for k in rows[0] if k.startswith("Nv_"))
    col = f"C{n}"
    for b, fill in (("up", True), ("down", False)):
        r = [x for x in rows if x["branch"] == b]
        if not r:
            continue
        H = [float(x["H"]) for x in r]
        ok = [x["conv"] == "F" for x in r]
        mk = dict(mfc=col if fill else "none", mec=col, ms=4)
        ax.plot(H, [-float(x["M"]) for x in r], "-", color=col, lw=0.8, alpha=0.6)
        ax.plot([h for h, o in zip(H, ok) if o], [-float(x["M"]) for x, o in zip(r, ok) if o], "o", **mk,
                label=f"{b}" + (f" ({f})" if len(a.files) > 1 else ""))
        ax.plot([h for h, o in zip(H, ok) if not o], [-float(x["M"]) for x, o in zip(r, ok) if not o], "x", color=col)
        for m in range(1, nb + 1):
            bx.plot(H, [int(x[f"Nv_{m}"]) for x in r], "o-" if m == 1 else "s--", lw=0.8, **mk,
                    label=f"band {m}, {b}" if n == 0 else None)
ax.set_xlabel(r"$H$ [$H_{c2}$ of band 1]"); ax.set_ylabel(r"$-M = H - \langle B_z\rangle$")
ax.axhline(0, color="0.6", lw=0.5); ax.legend(frameon=False, fontsize=7); ax.set_title("(a)", loc="left")
bx.set_xlabel(r"$H$ [$H_{c2}$ of band 1]"); bx.set_ylabel(r"$N_v$"); bx.legend(frameon=False, fontsize=6)
bx.set_title("(b)", loc="left")
if a.title:
    fig.suptitle(a.title, fontsize=9)
fig.savefig(a.out, dpi=160)
print("wrote", a.out)
