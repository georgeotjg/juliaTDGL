# Shape editor

```
python3 julia/editor/server.py      # or, in the terminal menu: g → e, or l → geometry → e
```

Opens `http://127.0.0.1:8765/`. Saves to `julia/shapes/<name>.toml`; if the
name exists, the previous version is moved to `trash_/shapes/`.

- The box is the sample (it starts full). Each voxel is a mesh node.
- Three editable views: top (a z plane), front (x–z cut) and side (y–z cut).
  Right click on a view moves the cuts there; `[` `]` change the z plane.
- Left panel, pinned while the views scroll: tabs Draw (work face, tools and
  only the options of the active tool), Palette (characters and their f; the
  legacy z-extent mode) and Box (mesh, field, ξπ/ξσ).
- Work face: pick one of the six faces (top, bottom, front y1, back y2, left
  x1, right x2) on the unfolded cube of the left panel or by clicking it on the
  3D model; the 3D turns to it and its view is outlined (the views never move). The pencil
  and the eraser then fill, at each clicked cell, `depth` cells inward from that
  face, starting `from` cells deep ("through" = the whole sample). `from` > 0
  gives cuts inside the sample, not only on its surface. Counts are in cells of
  the mesh you set in Box.
- Pencil and eraser (legacy mode, "pencil fills: the character's z extent"):
  free or straight stroke, any width in cells, on any view. On the
  top view a click fills the z extent of the palette character under the
  cursor: all (the whole column), this plane only, bottom, top, buried or
  [z0, z1]; on the front and side cuts it paints one voxel. The default palette
  already has `#` (all), `p` (this plane), `b` (bottom) and `a` (top): one slab
  removed = `p` on the z plane chosen with `[` `]`; a channel that does not
  cross the thickness = `b`, `a` or a new character with buried or [z0, z1].
  ✕ on a palette entry deletes the character and what it painted. Saved as
  `type = "voxels"` (any 3D pattern).
- Gradient tool: drag a rectangle on the top view. "f" mode: f goes from f0
  (where the drag starts) to f1 along the longer side in n bands, each with the
  pencil's z extent. "depth" modes: a staircase of n steps, from one plane
  removed (where the drag starts) to the whole thickness, from the top or from
  the bottom, with f = f1 in the removed part. Each step is a palette character.
- Figures (circle, rectangle, triangle, n-gon, polygon) are drawn on the top
  view. Their z extent is, by default, "from the work face": the `from` and
  `depth` cells of the top or bottom face, as for the pencil, so a disc three
  cells deep from the top or a buried one from 2 to 3 cells needs no
  arithmetic. The other choices are all (through), bottom, top or buried half,
  and a range [z0, z1] in ξσ, where plane k sits at z = k dz and the top plane at
  (Nz − 1) dz (2.5 ξσ for Nz = 6, dz = 0.5). The list of figures shows the
  planes each one covers and edits its z0, z1; "z ← work face" gives a
  figure the current from/depth.
- select/move moves a figure, or, clicking on a pencil voxel, the whole pencil
  layer (all strokes together, by whole cells).
- ◐ (top right) switches the theme: system, light or dark; "cells" picks the colour of the
  superconducting cells; both are remembered by the browser.
- "close server" (top right) stops the local server at once; otherwise it stops
  after 30 min without the page open.
- Ctrl+Z / Ctrl+Y undo and redo everything. 3D view: drag rotates, the wheel zooms.
- Figures: circle, rectangle, triangle (with skew), n-gon, free polygon and
  sphere, with a rotation in degrees. They are painted in list order; the
  pencil layer is one more entry of that list.
- Physics at a glance: the 3D view draws the current J (x1 → x2), the field H
  (along z) and the face where vortices enter for each sign of J (+J through
  y1, −J through y2); the top view says the same at its edges. Each face of the
  cube in the left panel shows how many cells with f ≠ 1 it has; x1 and x2 in
  red, because a defect on a contact injects through it (mode M2). "mirror
  (_REF)" outlines the shape reflected in y, the control that must invert η.
- Live checks: My and Rz (a symmetric shape gives η = 0 exactly), injection
  strips, normal path across the sample, where π or σ become normal.

Starting from the catalogue: under "load…", group "catalogue", are WEDGE,
TRI_Y1, ESFERA, … as pixel layers (`shapes/cat_*.toml`, written by
`scripts/catalogue_to_shapes.jl`). On the 40×24×6 box they give the catalogue
mask to the bit; when one is opened the proposed name is `<mode>_mod`, so the
template is not overwritten. `http://127.0.0.1:8765/?load=<name>` opens a shape.

Colours: blue = superconductor; hatched = removed (f = 0); blend = 0 < f < 1;
green = f > 1 (stronger condensation). Translucent red = strips next to x1 and
x2, where the current enters. The checks warn, they do not block.

Running a shape: terminal menu → l → geometry `e` or `f`, or

```
julia --project=julia julia/scripts/run_preset.jl --shape julia/shapes/x.toml --kappa 25 --J 0.72
```

The run keeps its own copy in `<run>/forma.toml`.

The file is in units of ξσ: Julia rasterises it with the dx of the run
(`src/shapes.jl`). Checked:
- the editor preview and Julia give the same nodes, also with voxels painted
  on the three views;
- every catalogue template passed through the editor (load → save) gives the
  catalogue mask to the bit, and its `CUSTOM_REF` is the exact nodal mirror
  (`test/test_shapes.jl`);
- the My and Rz counts of the editor equal Julia's `check_my_symmetry` /
  `check_rz_symmetry` for the 18 modes;
- files in the first format (`[[forma]]`, `tipo = "circulo"`, `z = "todo"`)
  are still read.

Note: the catalogue `FGRAD_REF` is not the exact mirror of FGRAD (504 interior
nodes); the catalogue is kept as it is.
