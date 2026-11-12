"""The shape of the sample for the film tools (film_html.py, film_gif.py): a smoothed indicator of the conductor, so that the
stepped mask of the lattice draws as the surface it stands for (a cylinder looks round), and its outline on a few cuts across
the current, as lines.  numpy and matplotlib only."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def solid(vac, f=None):
    """The nodes drawn as superconductor: not vacuum and not switched off by the condensation mask (f < 0.5, a notch).  The face
    planes y and z of the box are copies of their neighbours (ghosts, without an equation), and are drawn as what they copy: in a
    box, its neighbour (the faces of the box are its surface, and a notch reads the same on both faces, though the mask f of 1.2
    is written on the bottom planes only); in a sculpted sample (with vacuum), as outside, because a link to a ghost carries no
    current as a link to vacuum does, so the surface lies half a cell inside the box there too (a disc drawn with its ghosts
    would bulge by a cell where it meets the box: a "cross").  The x faces, at the contacts, are kept: the ends of the bar."""
    s = ~vac if f is None else ~(vac | (f < 0.5))
    s = s.copy()
    sculpted = vac.any()
    for ax in (1, 2):
        if s.shape[ax] <= 2:
            continue
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        for end, nb in ((0, 1), (-1, -2)):
            face, near = [slice(None)] * 3, [slice(None)] * 3
            face[ax], near[ax] = end, nb
            s[tuple(face)] = False if sculpted else s[tuple(near)]
    return s


def smooth(sol, passes=2, closed=False):
    """The conductor indicator (1 inside, 0 in vacuum) averaged over each node's 3 × 3 × 3 neighbourhood, `passes` times.  The box
    faces are extended (a sample that touches a face does not close there), or with `closed` taken as vacuum (it closes there)."""
    f = sol.astype(float)
    for _ in range(passes):
        p = np.pad(f, 1, mode="constant") if closed else np.pad(f, 1, mode="edge")
        f = sum(p[1 + a:p.shape[0] - 1 + a, 1 + b:p.shape[1] - 1 + b, 1 + c:p.shape[2] - 1 + c]
                for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)) / 27.0
    return f


def outline_rings(sol, dx, dy, dz, spacing=4.0):
    """The outline of the sample (`solid`) on cuts x = const every `spacing` (in ξ) and at both ends: a list of (n, 3) arrays
    (x, y, z), closed by a frame of vacuum around each cut.  A cut through vacuum only gives none."""
    nx, ny, nz = sol.shape
    f = smooth(sol, passes=2)
    xs = sorted(set([0, nx - 1] + list(range(0, nx, max(1, int(round(spacing / dx)))))))
    # each cut with a frame of vacuum around it, so that the outline closes along the faces of the box the sample touches
    yy, zz = np.arange(-1, ny + 1) * dy, np.arange(-1, nz + 1) * dz
    out = []
    fig, ax = plt.subplots()
    for i in xs:
        sl = np.pad(f[i], 1, mode="constant")
        if sl.max() < 0.5:
            continue
        cs = ax.contour(yy, zz, sl.T, levels=[0.5])
        for seg in cs.allsegs[0]:
            if len(seg) > 2:
                out.append(np.c_[np.full(len(seg), i * dx), seg[:, 0], seg[:, 1]])
    plt.close(fig)
    return out


def outline_2d(vac2, dx, dy):
    """The outline of the sample on a 2D cut (vacuum mask of the cut): a list of (n, 2) arrays."""
    f = np.pad(smooth(~vac2[:, :, None], passes=1)[:, :, 0], 1, mode="constant")
    fig, ax = plt.subplots()
    cs = ax.contour(np.arange(-1, vac2.shape[0] + 1) * dx, np.arange(-1, vac2.shape[1] + 1) * dy, f.T, levels=[0.5])
    segs = [s for s in cs.allsegs[0] if len(s) > 2]
    plt.close(fig)
    return segs
