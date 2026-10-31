#!/usr/bin/env python3
"""The GIF of a film of scripts/film.jl: what happens around each vortex event, for slides and papers (film.html turns it in 3D).

    python3 tdgl2/tools/film_gif.py FILMDIR        (scripts/film.jl calls it; writes FILMDIR/film.gif and film_frame.png)

Left, in 3D: the sample (its cells, translucent, and its outline on cuts across x every 4 ξ), the mid z plane coloured by |Ψ1|² with the supercurrent of all bands on it
(white lines), and each vortex as a line through its plaquettes plane by plane (band 1: blue, +1; red, −1; the other bands grey).
Right, above: the same mid plane seen from above, with the vortices of that plane (filled +1, open −1).  Right, below: the number
of vortex plaquettes of each band at every sample (not only the kept ones); the dot is the frame shown.  Only numpy and matplotlib.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from film_geom import solid, outline_rings, outline_2d
from matplotlib.animation import FuncAnimation, PillowWriter

INK, MUTED, SURF, VX, AV = "#1d232b", "#66707c", "#c9ced6", "#1f5fbf", "#d0452f"
BAND_COLORS = ("#1f5fbf", "#8a8f98", "#b0b4bb", "#c4c7cc")
MAT_COLORS = ((0.90, 0.49, 0.13), (0.15, 0.68, 0.38), (0.56, 0.27, 0.68), (0.16, 0.50, 0.73))   # the other materials, as the editor
GREEK = {"sigma": "σ", "pi": "π"}


def _lines(pts, dmax):
    """The plaquettes of one band and sign joined into lines: from plane to plane, each point to the nearest free one of the next
    plane within dmax (a vortex line along z; a point that finds no partner starts a new line)."""
    if len(pts) == 0:
        return []
    zs = np.unique(pts[:, 2]); lines = []; open_ = []
    for zk in zs:
        here = [p for p in pts[pts[:, 2] == zk]]; used = [False] * len(here); new_open = []
        for ln in open_:
            last = ln[-1]
            best, bd = None, dmax
            for i, p in enumerate(here):
                d_ = np.hypot(p[0] - last[0], p[1] - last[1])
                if not used[i] and d_ <= bd:
                    best, bd = i, d_
            if best is None:
                lines.append(ln)
            else:
                used[best] = True; ln.append(here[best]); new_open.append(ln)
        new_open += [[p] for i, p in enumerate(here) if not used[i]]
        open_ = new_open
    lines += open_
    return [np.array(l) for l in lines]


def _streamlines(xs, ys, jx, jy, vac2, nx, ny):
    """The supercurrent lines of the mid plane, as segments in the plane (matplotlib's streamplot on a figure that is not shown)."""
    if not np.hypot(jx, jy).max() > 0:
        return []
    f, a = plt.subplots()
    sp = a.streamplot(xs, ys, np.ma.masked_where(vac2, jx).T, np.ma.masked_where(vac2, jy).T,
                      density=(0.8 * nx / max(nx, ny), 0.8 * ny / max(nx, ny)), broken_streamlines=False)
    segs = [np.array(sg) for sg in sp.lines.get_segments()]
    plt.close(f)
    # join the short segments of each line back into polylines (consecutive segments share an end)
    out, cur = [], None
    for sg in segs:
        if cur is not None and np.allclose(cur[-1], sg[0]):
            cur = np.vstack([cur, sg[1:]])
        else:
            if cur is not None:
                out.append(cur)
            cur = sg
    if cur is not None:
        out.append(cur)
    return out


def main(d: Path):
    m = json.loads((d / "meta.json").read_text())
    V = np.load(d / "vortices.npy"); t, x, y, z, band, sg = V.T
    times = np.array(m["times"]); C = np.load(d / "counts.npy")
    P = np.load(d / "psi1_mid.npy"); JX = np.load(d / "jx_mid.npy"); JY = np.load(d / "jy_mid.npy")
    # outside the superconductor: the vacuum and the nodes where the condensation mask f switches it off (a notch of f = 0)
    vac = ~solid(np.load(d / "vac.npy") > 0.5, np.load(d / "f.npy") if (d / "f.npy").is_file() else None)   # and the ghosts
    dx, dy, dz = m["dx"], m["dy"], m["dz"]; nx, ny, nz = vac.shape
    # the cells of the sample (a cell is conductor when most of its 8 nodes are), drawn once, translucent
    sol = ~vac
    cell = (sol[:-1, :-1, :-1].astype(int) + sol[1:, :-1, :-1] + sol[:-1, 1:, :-1] + sol[:-1, :-1, 1:] + sol[1:, 1:, :-1]
            + sol[1:, :-1, 1:] + sol[:-1, 1:, 1:] + sol[1:, 1:, 1:]) >= 5
    kc = int(round(m["mid_z"] / dz)); vmid = vac[:, :, kc]

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_axes([0.0, 0.05, 0.5, 0.88], projection="3d", computed_zorder=False)
    a2 = fig.add_axes([0.6, 0.42, 0.38, 0.46]); a3 = fig.add_axes([0.6, 0.1, 0.38, 0.2])
    L = ((nx - 1) * dx, (ny - 1) * dy, (nz - 1) * dz)
    ax.set_box_aspect(L); ax.view_init(elev=26, azim=-58)
    gx, gy, gz = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy, np.arange(nz) * dz, indexing="ij")
    ax.voxels(gx, gy, gz, cell, facecolors=(0.80, 0.83, 0.88, 0.07), edgecolors=(0.55, 0.6, 0.68, 0.06), linewidth=0.2)
    # material per region (release 2): the cells of each other material tinted, and its interfaces on the mid plane
    mat = np.rint(np.load(d / "mat.npy")).astype(int) if (d / "mat.npy").is_file() else None
    if mat is not None:
        for q in range(2, mat.max() + 1):
            s8 = (mat == q) & sol
            mc = (s8[:-1, :-1, :-1].astype(int) + s8[1:, :-1, :-1] + s8[:-1, 1:, :-1] + s8[:-1, :-1, 1:] + s8[1:, 1:, :-1]
                  + s8[1:, :-1, 1:] + s8[:-1, 1:, 1:] + s8[1:, 1:, 1:]) >= 5
            ax.voxels(gx, gy, gz, mc, facecolors=MAT_COLORS[(q - 2) % len(MAT_COLORS)] + (0.10,), edgecolors=(0, 0, 0, 0), linewidth=0)
    for ring in outline_rings(sol, dx, dy, dz):             # the outline of the sample on cuts across x
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#59616b", lw=0.9)
    ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_zlim(0, L[2])
    ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3 if L[2] < L[0] / 5 else 6))   # a thin sample: few ticks
    ax.set_xlabel("x (ξ), current", fontsize=8); ax.set_ylabel("y (ξ)", fontsize=8); ax.set_zlabel("z (ξ), field", fontsize=8)
    ax.tick_params(labelsize=7, colors=MUTED)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False; a.pane.set_edgecolor("#e3e6ea")
    ax.grid(False)
    nb = len(m["bands"]); names = [GREEK.get(b, b) for b in m["bands"]]
    present = {(int(b), int(k)) for b, k in zip(band[sg != 0], sg[sg != 0])}   # the legend names only what the film shows
    for b in range(1, nb + 1):
        col = (VX, AV) if b == 1 else (BAND_COLORS[min(b - 1, 3)],) * 2
        if (b, 1) in present:
            ax.plot([], [], [], color=col[0], lw=3, label=f"{names[b-1]} vortex")
        if (b, -1) in present:
            ax.plot([], [], [], color=col[1], lw=3, ls=(0, (1.5, 1)), label=f"{names[b-1]} antivortex")
    dyn = []                                             # the artists of one frame, removed before the next
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    title = fig.suptitle("", fontsize=9, color=INK)
    ext = (-dx / 2, (nx - 0.5) * dx, -dy / 2, (ny - 0.5) * dy)
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#d9d8d4")
    vmax = float(np.nanmax(np.where(vmid[..., None], np.nan, P))) or 1.0
    xs = np.arange(nx) * dx; ys = np.arange(ny) * dy
    tc = C[:, 0]
    for b in range(1, nb + 1):
        a3.plot(tc, C[:, b], lw=1.0, color=BAND_COLORS[min(b - 1, 3)] if b > 1 else VX, label=names[b - 1])
    dot, = a3.plot([], [], "o", color=INK, ms=4)
    a3.set_xlabel("t (t_GL)", fontsize=8); a3.set_ylabel("plaquettes", fontsize=8); a3.tick_params(labelsize=7)
    a3.legend(fontsize=7, frameon=False, loc="upper right")
    for s_ in ("top", "right"):
        a3.spines[s_].set_visible(False)

    zm = m["mid_z"]; XX, YY = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy, indexing="ij")

    def frame(q):
        tq = times[q]; sel = t == tq
        for a in dyn:
            a.remove()
        dyn.clear()
        # the mid plane coloured by |Ψ1|² (vacuum transparent), and the supercurrent lines on it
        rgba = cmap(np.clip(P[:, :, q] / vmax, 0, 1)); rgba[vmid] = (0, 0, 0, 0); rgba[..., 3] = np.where(vmid, 0, 0.92)
        dyn.append(ax.plot_surface(XX, YY, np.full_like(XX, zm), facecolors=rgba, rstride=1, cstride=1, shade=False,
                                   linewidth=0, antialiased=False))
        for seg in _streamlines(xs, ys, JX[:, :, q], JY[:, :, q], vmid, nx, ny):
            dyn.append(ax.plot(seg[:, 0], seg[:, 1], np.full(len(seg), zm + 0.02), color="white", lw=0.6)[0])
        # each vortex as a line through its plaquettes, plane by plane
        for b in range(1, nb + 1):
            for k, col, ls in ((1, VX if b == 1 else BAND_COLORS[min(b - 1, 3)], "-"),
                               (-1, AV if b == 1 else BAND_COLORS[min(b - 1, 3)], (0, (1.5, 1)))):
                mm = sel & (band == b) & (sg == k)
                for line in _lines(np.c_[x[mm], y[mm], z[mm]], 1.6 * max(dx, dy)):
                    if len(line) == 1:
                        dyn.append(ax.plot(line[:, 0], line[:, 1], line[:, 2], "o", color=col, ms=4)[0])
                    else:
                        dyn.append(ax.plot(line[:, 0], line[:, 1], line[:, 2], color=col, lw=3, ls=ls,
                                           solid_capstyle="round")[0])
        a2.clear()
        Pm = np.ma.masked_where(vmid, P[:, :, q])
        a2.imshow(Pm.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
        jx, jy = JX[:, :, q], JY[:, :, q]
        if np.hypot(jx, jy).max() > 0:
            a2.streamplot(xs, ys, np.ma.masked_where(vmid, jx).T, np.ma.masked_where(vmid, jy).T, color="white",
                          density=(1.0 * nx / max(nx, ny), 1.0 * ny / max(nx, ny)), linewidth=0.5, arrowsize=0.5,
                          broken_streamlines=False)
        if mat is not None and mat[:, :, kc].max() > mat[:, :, kc].min():
            a2.contour(xs, ys, mat[:, :, kc].T.astype(float), levels=np.arange(1.5, mat.max()), colors="#e67e22", linewidths=1.0)
        a2.set_xlim(ext[:2]); a2.set_ylim(ext[2:])
        mm = sel & (band == 1) & (np.abs(z - m["mid_z"]) < dz / 2)
        for k, fc in ((1, INK), (-1, "white")):
            w = mm & (sg == k)
            a2.scatter(x[w], y[w], s=20, facecolor=fc, edgecolor=INK, lw=0.8)
        a2.set_title(f"|Ψ{names[0]}|² at z = {m['mid_z']:.2f}, supercurrent (white)", fontsize=8, color=INK)
        a2.tick_params(labelsize=7); a2.set_xlabel("x (ξ)", fontsize=8); a2.set_ylabel("y (ξ)", fontsize=8)
        dot.set_data([tq], [np.interp(tq, tc, C[:, 1])])
        nv = int(np.count_nonzero(sel & (sg != 0)))
        title.set_text(f"{m['run']}, {m['polarity']}, J = {m['J']:+.4f}, H = {m['H']}: t = {tq:.2f} t_GL"
                       + ("" if nv else "  (no vortex)"))
        return ()

    nf = len(times)
    best = int(np.argmax([np.count_nonzero((t == tq) & (sg != 0)) for tq in times]))
    frame(best); fig.savefig(d / "film_frame.png", dpi=140)
    FuncAnimation(fig, frame, frames=nf, blit=False).save(d / "film.gif", writer=PillowWriter(fps=8), dpi=80)
    print(f"{d / 'film.gif'}: {nf} frames")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    main(Path(sys.argv[1]))
