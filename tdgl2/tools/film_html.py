#!/usr/bin/env python3
"""An interactive 3D page of a film of scripts/film.jl: one HTML file that opens in any browser, turned and zoomed with the mouse,
played with a slider.

    python3 tdgl2/tools/film_html.py FILMDIR        (scripts/film.jl calls it; writes FILMDIR/film.html)

It needs numpy and matplotlib here (for the supercurrent lines).  With the Python package plotly installed, the page carries
plotly.js inside and opens without a network (+5 MB); without it, the page loads plotly.js (a pinned version) from its CDN.  What it shows, frame by frame (the frames of film.pvd):
- the sample (grey, transparent): the surface of its smoothed indicator, so that a stepped cylinder draws round, and its outline
  on cuts across x every 4 ξ and on the mid plane (dark grey lines);
- the vortex cores of band 1 (blue): the contour |Ψ1|² = 0.3 with the vacuum lifted to 1 (a free surface does not lower |Ψ|, so
  what is left are cores and normal regions);
- the vortex plaquettes of every z plane (band 1: blue +1, red −1; other bands grey);
- the mid z plane coloured by |Ψ1|², with the supercurrent of all bands on it (white lines).
"""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from film_geom import solid, smooth, outline_rings, outline_2d

PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
GREEK = {"sigma": "σ", "pi": "π"}
MAT_COLORS = ("#e67e22", "#27ae60", "#8e44ad", "#2980b9")   # the other materials, as the editor and film_gif.py
# matplotlib's Blues: white at 0, dark blue at 1 (plotly's own "Blues" runs the other way)
BLUES = [[0, "#f7fbff"], [0.25, "#c6dbef"], [0.5, "#6baed6"], [0.75, "#2171b5"], [1, "#08306b"]]


def read_vti(p: Path):
    """The point arrays of a .vti written by film.jl (raw appended data, UInt64 headers, Float32, little endian), as (nx, ny, nz[, 3])."""
    raw = p.read_bytes()
    head, _, data = raw.partition(b'<AppendedData encoding="raw">\n_')
    h = head.decode()
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', h).group(1).split()]
    n = (ext[1] + 1, ext[3] + 1, ext[5] + 1)
    out = {}
    for name, nc, off in re.findall(r'Name="([^"]+)" NumberOfComponents="(\d+)" format="appended" offset="(\d+)"', h):
        off, nc = int(off), int(nc)
        nb = int(np.frombuffer(data[off:off + 8], "<u8")[0])
        a = np.frombuffer(data[off + 8:off + 8 + nb], "<f4")
        # Julia writes column-major: the component index runs fastest, then x, y, z
        out[name] = a.reshape((n[2], n[1], n[0], nc)).transpose(2, 1, 0, 3) if nc > 1 else a.reshape((n[2], n[1], n[0])).transpose(2, 1, 0)
    return out


def lines_of(xs, ys, jx, jy, vac2):
    if not np.hypot(jx, jy).max() > 0:
        return []
    f, a = plt.subplots()
    nx, ny = len(xs), len(ys)
    sp = a.streamplot(xs, ys, np.ma.masked_where(vac2, jx).T, np.ma.masked_where(vac2, jy).T,
                      density=(0.8 * nx / max(nx, ny), 0.8 * ny / max(nx, ny)), broken_streamlines=False)
    segs = [np.array(s) for s in sp.lines.get_segments()]
    plt.close(f)
    out, cur = [], None
    for s in segs:
        if cur is not None and np.allclose(cur[-1], s[0]):
            cur = np.vstack([cur, s[1:]])
        else:
            if cur is not None:
                out.append(cur)
            cur = s
    if cur is not None:
        out.append(cur)
    return out


r3 = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()


def main(d: Path):
    m = json.loads((d / "meta.json").read_text())
    V = np.load(d / "vortices.npy"); t, x, y, z, band, sg = V.T
    times = m["times"]; nb = len(m["bands"]); b1 = "psi2_" + m["bands"][0]; names = [GREEK.get(b, b) for b in m["bands"]]
    dx, dy, dz = m["dx"], m["dy"], m["dz"]
    # outside the superconductor: the vacuum and the nodes where the condensation mask f switches it off (a notch of f = 0)
    f0 = read_vti(d / "frame_0001.vti"); vac = ~solid(f0["vacuum"] > 0.5, f0.get("f"))     # and the face planes, ghosts
    nx, ny, nz = vac.shape
    X, Y, Z = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy, np.arange(nz) * dz, indexing="ij")
    flat = lambda a: r3(a.ravel())
    kc = int(round(m["mid_z"] / dz)); vmid = vac[:, :, kc]
    xs, ys = np.arange(nx) * dx, np.arange(ny) * dy
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    # the sample: the surface of its smoothed indicator (a stepped cylinder draws round) and its outline on cuts across x
    sm = smooth(~vac, closed=True)
    sample = dict(type="isosurface", x=flat(X), y=flat(Y), z=flat(Z), value=r3((1 - sm).ravel()), isomin=0.5, isomax=0.5,
                  surface=dict(count=1), caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
                  colorscale=[[0, "#b9bec6"], [1, "#b9bec6"]], showscale=False, opacity=0.18, name="sample", showlegend=True,
                  hoverinfo="skip")
    ox, oy, oz = [], [], []
    for L in outline_rings(~vac, dx, dy, dz):
        ox += r3(L[:, 0]) + [None]; oy += r3(L[:, 1]) + [None]; oz += r3(L[:, 2]) + [None]
    for L in outline_2d(vmid, dx, dy):
        ox += r3(L[:, 0]) + [None]; oy += r3(L[:, 1]) + [None]; oz += [m["mid_z"] + 0.03] * len(L) + [None]
    rings = dict(type="scatter3d", mode="lines", x=ox, y=oy, z=oz, line=dict(color="#59616b", width=3),
                 name="outline of the sample", hoverinfo="skip")
    vmax = 0.0
    frames = []
    for q, tq in enumerate(times, start=1):
        F = read_vti(d / f"frame_{q:04d}.vti")
        P = F[b1]; vmax = max(vmax, float(P[~vac].max()))
        core = dict(type="isosurface", x=flat(X), y=flat(Y), z=flat(Z), value=flat(P + vac), isomin=0.3, isomax=0.3,
                    surface=dict(count=1), caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
                    colorscale=[[0, "#1f5fbf"], [1, "#1f5fbf"]], showscale=False, opacity=0.75, name=f"cores |Ψ{names[0]}|² = 0.3",
                    showlegend=True)
        # the plane: white where |Ψ1|² = 0, dark blue at its maximum (as the figures of the run); a hole over the vacuum
        zp = np.where(vmid, np.nan, m["mid_z"])
        plane = dict(type="surface", x=r3(XX), y=r3(YY), z=[[None if np.isnan(v) else v for v in row] for row in zp.tolist()],
                     surfacecolor=r3(np.where(vmid, 0, P[:, :, kc])), colorscale=BLUES, cmin=0, cmax=1, showscale=True,
                     colorbar=dict(title=dict(text=f"|Ψ{names[0]}|²"), len=0.5, x=1.0), opacity=0.95, name=f"|Ψ{names[0]}|² at z = {m['mid_z']:.2f}",
                     showlegend=True, hoverinfo="skip")
        J = F["supercurrent"]
        lx, ly, lz = [], [], []
        for L in lines_of(xs, ys, J[:, :, kc, 0], J[:, :, kc, 1], vmid):
            lx += r3(L[:, 0]) + [None]; ly += r3(L[:, 1]) + [None]; lz += [m["mid_z"] + 0.02] * len(L) + [None]
        stream = dict(type="scatter3d", mode="lines", x=lx, y=ly, z=lz, line=dict(color="white", width=2), name="supercurrent",
                      hoverinfo="skip")
        pts = []
        for b in range(1, nb + 1):
            for k, col, lab in ((1, "#1f5fbf" if b == 1 else "#8a8f98", "vortex"), (-1, "#d0452f" if b == 1 else "#b0b4bb", "antivortex")):
                mm = (t == tq) & (band == b) & (sg == k)
                pts.append(dict(type="scatter3d", mode="markers", x=r3(x[mm]), y=r3(y[mm]), z=r3(z[mm]),
                                marker=dict(size=3 if b == 1 else 2, color=col), name=f"{names[b-1]} {lab}",
                                showlegend=bool(((V[:, 4] == b) & (V[:, 5] == k)).any())))
        full = [core, plane, stream] + pts
        if q == 1:
            first = full
        # a frame carries only what changes (plotly merges it into the traces): the values of the cores, the colours of the plane,
        # the points of the lines and of the vortices
        frames.append(dict(name=f"{tq:.2f}", data=[dict(value=core["value"]), dict(surfacecolor=plane["surfacecolor"]),
                                                   dict(x=lx, y=ly, z=lz)] + [dict(x=p_["x"], y=p_["y"], z=p_["z"]) for p_ in pts]))
    # every frame keeps the same trace order; traces 0 and 1 (the sample and its outline) do not change
    nd = len(frames[0]["data"])
    for f in frames:
        f["traces"] = list(range(2, nd + 2))
    # material per region (release 2): each other material as a translucent surface (after the animated traces: their indices stay)
    mats = []
    if (d / "mat.npy").is_file():
        mat = np.rint(np.load(d / "mat.npy")).astype(int); nombres = m.get("materials", [])
        for q in range(2, mat.max() + 1):
            sq = smooth((mat == q) & ~vac, closed=True)
            mats.append(dict(type="isosurface", x=flat(X), y=flat(Y), z=flat(Z), value=r3((1 - sq).ravel()), isomin=0.5, isomax=0.5,
                             surface=dict(count=1), caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
                             colorscale=[[0, MAT_COLORS[(q - 2) % 4]], [1, MAT_COLORS[(q - 2) % 4]]], showscale=False, opacity=0.22,
                             name="material " + (nombres[q - 1] if q - 1 < len(nombres) else str(q)), showlegend=True, hoverinfo="skip"))
    data = [sample, rings] + first + mats
    L = ((nx - 1) * dx, (ny - 1) * dy, (nz - 1) * dz); s = max(L)
    layout = dict(
        title=dict(text=f"{m['run']}, {m['polarity']}, J = {m['J']:+.4f}, H = {m['H']}: the vortex events (drag to turn, scroll to zoom)",
                   font=dict(size=13)),
        scene=dict(xaxis=dict(title=dict(text="x (ξ), current")), yaxis=dict(title=dict(text="y (ξ)")),
                   zaxis=dict(title=dict(text="z (ξ), field")),
                   aspectmode="manual", aspectratio=dict(x=L[0] / s * 2, y=L[1] / s * 2, z=L[2] / s * 2),
                   camera=dict(eye=dict(x=-0.9, y=-1.6, z=0.9))),
        margin=dict(l=0, r=0, t=40, b=0), showlegend=True, legend=dict(x=0, y=1),
        updatemenus=[dict(type="buttons", x=0.0, y=0.02, xanchor="left", buttons=[
            dict(label="play", method="animate", args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
            dict(label="pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(x=0.12, len=0.86, y=0.02, currentvalue=dict(prefix="t = ", suffix=" t_GL"),
                      steps=[dict(label=f["name"], method="animate",
                                  args=[[f["name"]], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for f in frames])])
    try:                                   # plotly.js inside the page when the Python package is there: it opens offline
        from plotly.offline import get_plotlyjs
        script = "<script>" + get_plotlyjs() + "</script>"
    except ImportError:
        script = f'<script src="{PLOTLY}"></script>'
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{m['run']} film</title>
{script}
<style>html,body{{margin:0;height:100%;background:#fff;font-family:sans-serif}}#f{{width:100%;height:100%}}</style></head>
<body><div id="f"></div>
<script>
const data = {json.dumps(data)};
const layout = {json.dumps(layout)};
const frames = {json.dumps(frames)};
Plotly.newPlot("f", data, layout, {{responsive: true}}).then(() => Plotly.addFrames("f", frames));
</script></body></html>
"""
    (d / "film.html").write_text(page)
    print(f"{d / 'film.html'}: {len(frames)} frames, {len(page) / 1e6:.1f} MB" + ("" if script.startswith("<script>") else
          " (plotly.js from its CDN: the page needs a network)"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    main(Path(sys.argv[1]))
