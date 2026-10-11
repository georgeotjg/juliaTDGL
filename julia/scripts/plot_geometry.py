#!/usr/bin/env python3
"""Geometry preview of a run, drawn when the run starts.

    python3 julia/scripts/plot_geometry.py RUNDIR

Reads RUNDIR/output/geometry_mask.txt (written by run_preset.jl before the
first current point) and writes RUNDIR/figures/geometry.png: the six outer
faces of the sample unfolded around the top view, the mid-thickness plane, and
a map of how many z planes each (x, y) column has with f != 1, so a cut that
does not cross the thickness is visible at a glance; and RUNDIR/figures/geometry3d.png:
the cells with f < 1 drawn as voxels inside the box, the contact faces x1, x2 in orange.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # the run does not need the preview
    sys.exit(0)


def read_mask(fn: Path):
    with open(fn) as fh:
        Nx, Ny, Nz, dx, dz = fh.readline().split()
        Nx, Ny, Nz, dx, dz = int(Nx), int(Ny), int(Nz), float(dx), float(dz)
        vals = np.array(fh.read().split(), dtype=float)
    return vals.reshape(Nz, Ny, Nx), dx, dz        # f[k, j, i], k = 0 bottom


def main(rundir: str) -> None:
    run = Path(rundir)
    f, dx, dz = read_mask(run / "output" / "geometry_mask.txt")
    Nz, Ny, Nx = f.shape
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    kw = dict(cmap=cmap, vmin=0.0, vmax=max(1.0, float(f.max())), interpolation="nearest", origin="lower")
    faces = {
        "top (z top)": (f[-1, :, :], dx, dx, "x", "y"),
        "bottom (z = 0)": (f[0, :, :], dx, dx, "x", "y"),
        "front (y1)": (f[:, 0, :], dx, dz, "x", "z"),
        "back (y2)": (f[:, -1, :], dx, dz, "x", "z"),
        "left (x1)": (f[:, :, 0].T, dz, dx, "z", "y"),
        "right (x2)": (f[:, :, -1].T, dz, dx, "z", "y"),
    }
    removed = (f != 1.0).sum(axis=0)                      # planes per column

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 3, 1], width_ratios=[1, 4, 1, 4],
                          hspace=0.55, wspace=0.35)

    def show(ax, A, sx, sy, xl, yl, title):
        h, w = A.shape
        ax.imshow(A, extent=[0, w * sx, 0, h * sy], aspect="equal", **kw)
        touched = int((A != 1.0).sum())
        ax.set_title(f"{title}\n{touched} node(s) with f ≠ 1", fontsize=8)
        ax.set_xlabel(f"{xl} (ξσ)", fontsize=7); ax.set_ylabel(f"{yl} (ξσ)", fontsize=7)
        ax.tick_params(labelsize=6)

    show(fig.add_subplot(gs[0, 1]), *faces["back (y2)"], "back (y2)")
    show(fig.add_subplot(gs[1, 0]), *faces["left (x1)"], "left (x1)")
    show(fig.add_subplot(gs[1, 1]), *faces["top (z top)"], "top (z top)")
    show(fig.add_subplot(gs[1, 2]), *faces["right (x2)"], "right (x2)")
    show(fig.add_subplot(gs[2, 1]), *faces["front (y1)"], "front (y1)")
    show(fig.add_subplot(gs[0, 3]), *faces["bottom (z = 0)"], "bottom (z = 0)")
    kc = Nz // 2
    show(fig.add_subplot(gs[1, 3]), f[kc], dx, dx, "x", "y", f"mid plane k = {kc} (z = {kc * dz:g})")
    ax = fig.add_subplot(gs[2, 3])
    im = ax.imshow(removed, extent=[0, Nx * dx, 0, Ny * dx], origin="lower", aspect="equal",
                   cmap="magma_r", vmin=0, vmax=Nz, interpolation="nearest")
    ax.set_title(f"z planes with f ≠ 1 per column (of {Nz})", fontsize=8)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"{'/'.join(run.parts[-3:])}: geometry, {Nx}×{Ny}×{Nz} nodes, dx = {dx:g}, dz = {dz:g} ξσ "
                 f"(colour: f, 0 = removed, 1 = superconductor)", fontsize=9)
    fig.savefig(out / "geometry.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(out / "geometry.png")
    plot_3d(f, dx, dz, out / "geometry3d.png", "/".join(run.parts[-3:]))


def plot_3d(f, dx, dz, fn: Path, title: str) -> None:
    """Voxels of the cells with f < 1 inside the box; contacts x1, x2 shaded."""
    Nz, Ny, Nx = f.shape
    cut = np.transpose(f < 0.999, (2, 1, 0))                 # [i, j, k]
    if not cut.any():
        return
    val = np.transpose(f, (2, 1, 0))
    cols = plt.get_cmap("viridis")(np.clip(val, 0, 1))
    cols[..., 3] = 0.9
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    ax.voxels(cut, facecolors=cols, edgecolor=(0.1, 0.1, 0.1, 0.25), linewidth=0.2)
    for x in (0, Nx):                                          # contact faces
        ax.plot_surface(np.full((2, 2), x), np.array([[0, 0], [Ny, Ny]]), np.array([[0, Nz], [0, Nz]]),
                        color="#e59866", alpha=0.35, linewidth=0)
    for a, b in [((0,0,0),(Nx,0,0)), ((0,Ny,0),(Nx,Ny,0)), ((0,0,Nz),(Nx,0,Nz)), ((0,Ny,Nz),(Nx,Ny,Nz)),
                 ((0,0,0),(0,Ny,0)), ((Nx,0,0),(Nx,Ny,0)), ((0,0,Nz),(0,Ny,Nz)), ((Nx,0,Nz),(Nx,Ny,Nz)),
                 ((0,0,0),(0,0,Nz)), ((Nx,0,0),(Nx,0,Nz)), ((0,Ny,0),(0,Ny,Nz)), ((Nx,Ny,0),(Nx,Ny,Nz))]:
        ax.plot(*zip(a, b), color="0.2", lw=0.8)
    ax.set_box_aspect((Nx * dx, Ny * dx, Nz * dz))
    ax.set_xlabel("x (cells)", fontsize=7); ax.set_ylabel("y (cells)", fontsize=7); ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=28, azim=-60)
    ax.set_title(f"{title}: cells with f < 1 (colour: f), contacts x1, x2 shaded", fontsize=8)
    fig.savefig(fn, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(fn)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
