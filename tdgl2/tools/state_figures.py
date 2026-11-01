#!/usr/bin/env python3
"""The figures of the states of a release-2 run, from the data that scripts/figures.jl exported.

  python3 tdgl2/tools/state_figures.py RUNDIR        (scripts/figures.jl calls it, after ladder_sheet.py)

Writes into RUNDIR/figures/ psi_<state>.png and share_<state>.png, and adds to README.txt what each one is.
Colours: J+ blue, J- orange (the polarity); |Psi|^2 on one sequential hue, vacuum grey.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib and numpy are needed (python3 -m pip install matplotlib numpy)")
    sys.exit(1)

POL = {"J+": "#2a78d6", "J-": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = ["-", "--", ":", "-."]
GREEK = {"sigma": "σ", "pi": "π"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def cut(out: Path, name: str):
    p = out / name
    return np.load(p) if p.is_file() else None


def psi(out: Path, meta: dict, st: dict, notes: list[str]):
    tag, bands = st["tag"], [GREEK.get(b, b) for b in meta["bands"]]
    N = len(bands)
    cuts = [("xy", "x", "y", f"z = {meta['cuts']['z']:.2f}"), ("xz", "x", "z", f"y = {meta['cuts']['y']:.2f}"),
            ("yz", "y", "z", f"x = {meta['cuts']['x']:.2f}")]
    step = {"x": meta["dx"], "y": meta["dy"], "z": meta["dz"]}
    w = [cut(out, f"vac_{c}.npy").shape[0] * step[u] for c, u, _, _ in cuts]
    fig, axs = plt.subplots(N, 3, figsize=(10.0, 1.0 + 2.1 * N), squeeze=False, constrained_layout=True,
                            gridspec_kw={"width_ratios": w})
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#d9d8d4")
    for mu in range(N):
        for col, (c, u, v, where) in enumerate(cuts):
            ax = axs[mu][col]
            P = cut(out, f"psi{mu + 1}_{c}_{tag}.npy"); vac = cut(out, f"vac_{c}.npy") > 0.5
            P = np.ma.masked_where(vac, P)
            nu, nv = P.shape
            im = ax.imshow(P.T, origin="lower", cmap=cmap, vmin=0, vmax=max(1e-12, float(P.max())), interpolation="nearest",
                           extent=(-step[u] / 2, (nu - 0.5) * step[u], -step[v] / 2, (nv - 0.5) * step[v]))
            ax.grid(False); ax.set_xlabel(f"{u}  [ξ]"); ax.set_ylabel(f"{v}  [ξ]")
            ax.set_title(f"|Ψ{bands[mu]}|², {where}", fontsize=8, color=INK)
            if c == "xy":
                jx, jy = cut(out, f"jx_xy_{tag}.npy"), cut(out, f"jy_xy_{tag}.npy")
                if jx is not None and jy is not None and np.hypot(jx, jy).max() > 0:
                    xs = np.arange(nu) * step["x"]; ys = np.arange(nv) * step["y"]
                    ax.streamplot(xs, ys, np.ma.masked_where(vac, jx).T, np.ma.masked_where(vac, jy).T, color="white",
                                  density=(1.1 * nu / max(nu, nv), 1.1 * nv / max(nu, nv)), linewidth=0.55,
                                  arrowsize=0.5, broken_streamlines=False)
                    ax.set_xlim(-step[u] / 2, (nu - 0.5) * step[u]); ax.set_ylim(-step[v] / 2, (nv - 0.5) * step[v])
                n = cut(out, f"vort{mu + 1}_xy_{tag}.npy")
                if n is not None:
                    for sgn, mk, fc in ((1, "o", INK), (-1, "o", "white")):
                        ii, jj = np.nonzero(n == sgn)
                        if ii.size:
                            ax.scatter((ii + 0.5) * step["x"], (jj + 0.5) * step["y"], s=22, marker=mk, facecolor=fc,
                                       edgecolor=INK, lw=0.8, label="vortex" if sgn > 0 else "antivortex")
            if col == 2:
                fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"{st['what']}: |Ψ|² of each band on the three mid cuts\non x–y, the supercurrent of all bands "
                 "(white streamlines), vortices (filled) and antivortices (open); grey = vacuum", color=INK, fontsize=9)
    fig.savefig(out / f"psi_{tag}.png", dpi=160); plt.close(fig)
    notes.append(f"psi_{tag}.png: {st['what']}. |Ψ|² of every band (rows) on the mid cuts x–y, x–z and y–z (columns), "
                 "on x–y the streamlines of the supercurrent of all bands (white) and the vortices of each band (filled +1, open -1); "
                 "grey is vacuum.")


def share(out: Path, meta: dict, st: dict, notes: list[str]):
    tag, bands = st["tag"], [GREEK.get(b, b) for b in meta["bands"]]
    I = cut(out, f"share_{tag}.npy")
    if I is None:
        return
    tot = I.sum(axis=1)
    ok = np.abs(tot) > 1e-12 * max(1.0, np.abs(tot).max())
    x = (np.arange(I.shape[0]) + 0.5) * meta["dx"]
    lab = tag.split("_")[0]
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    for mu in range(I.shape[1]):
        fr = np.where(ok, I[:, mu] / np.where(ok, tot, 1.0), np.nan)
        ax.plot(x, fr, color=POL.get(lab, INK), ls=STYLE[mu % 4], lw=2, label=bands[mu])
        j = len(fr) // 2                       # the label in the middle, where the share is flat
        if np.isfinite(fr[j]):
            ax.annotate(f"{bands[mu]}  {fr[j]:.3f}", (x[j], fr[j]), xytext=(0, 7 if fr[j] >= 0.5 else -11),
                        textcoords="offset points", color=INK, fontsize=8, ha="center")
    ax.set_xlabel("x  [ξ]  (contact x1 at 0)"); ax.set_ylabel("share of the supercurrent")
    ax.set_ylim(-0.05, 1.05); ax.legend(loc="upper center", ncol=len(bands))
    ax.set_title(f"{st['what']}: the share each band carries, plane by plane", fontsize=9, color=INK)
    fig.savefig(out / f"share_{tag}.png", dpi=160); plt.close(fig)
    notes.append(f"share_{tag}.png: {st['what']}. The share of the supercurrent each band carries through every x plane "
                 "(it is set within 1-2 ξ of the contacts, where the normal current converts, and is flat between them "
                 "when the interband Josephson coupling is weak).")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    run = Path(sys.argv[1]); out = run / "figures"
    meta = json.loads((out / "meta.json").read_text())
    notes = []
    for st in meta["states"]:
        psi(out, meta, st, notes)
        share(out, meta, st, notes)
    with (out / "README.txt").open("a") as f:
        f.write("\n".join(notes) + "\n")
    print(f"{len(notes)} state figure(s) in {out}")


if __name__ == "__main__":
    main()
