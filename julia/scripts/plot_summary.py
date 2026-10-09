#!/usr/bin/env python3
"""Standard figures AFTER a run (does not touch the solver).

  python3 julia/scripts/plot_summary.py RUNDIR

Writes RUNDIR/figures/ :
  iv.png             V(J) for both polarities: linear, and log|V| with V_th
  dvdj.png           differential resistance
  psi_s_last.png     last |Ψσ|² slice (from PGM); psi_p_last.png for Ψπ
  psi_s_mosaic.png   closed current points, one row per polarity (up to 8 each)

3D: the solver writes legacy ASCII VTK (vtk.jl). For ParaView:
  julia --project=julia julia/scripts/vtk_from_checkpoint.jl \\
        RUNDIR/.../restart/state_Jpos.mat out.vtk
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib is needed (python3 -m pip install matplotlib)")
    sys.exit(1)


def _j_hist(rundir: Path) -> list[dict]:
    rows = []
    for cand in [rundir / "j_history.csv", *rundir.glob("**/j_history.csv")]:
        if not cand.is_file():
            continue
        with cand.open(errors="replace") as f:
            r = csv.DictReader(f)
            rows = list(r)
        if rows:
            return rows
    # log.txt: "J=+0.100 V=+1.23e-2 conv=V"
    log = rundir / "log.txt"
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(
                r"J=\s*([+-]?[0-9.]+).*\bV=\s*([+-]?[0-9.eE+-]+)", line
            )
            if not m:
                continue
            rows.append({"J": m.group(1), "V": m.group(2), "pol": "J+" if float(m.group(1)) >= 0 else "J-"})
    return rows


def _read_pgm(path: Path):
    raw = path.read_text(errors="replace").splitlines()
    toks: list[str] = []
    for line in raw:
        line = line.split("#", 1)[0].strip()
        if line:
            toks.extend(line.split())
    if len(toks) < 4 or toks[0] not in ("P2", "P5"):
        return None
    w, h, mv = int(toks[1]), int(toks[2]), int(toks[3])
    pix = np.array([int(x) for x in toks[4 : 4 + w * h]], dtype=float)
    if pix.size < w * h:
        return None
    return pix.reshape(h, w) / max(mv, 1)


def fig_iv(rows: list[dict], out: Path) -> None:
    jp, vp, jm, vm = [], [], [], []
    for rec in rows:
        try:
            j, v = float(rec["J"]), float(rec["V"])
        except (KeyError, ValueError):
            continue
        pol = rec.get("pol", "")
        if j < 0 or str(pol).endswith("-"):
            jm.append(abs(j))
            vm.append(v)
        else:
            jp.append(abs(j))
            vp.append(v)
    if len(jp) + len(jm) < 2:
        return
    fig, (ax, axl) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for a, f in ((ax, lambda v: v), (axl, lambda v: np.abs(v))):
        if jp:
            o = np.argsort(jp)
            a.plot(np.array(jp)[o], f(np.array(vp)[o]), "o-", color="#1f4e79", ms=5, label=r"$+J$")
        if jm:
            o = np.argsort(jm)
            a.plot(np.array(jm)[o], f(np.array(vm)[o]), "s-", color="#c45911", ms=5,
                   mfc="white", label=r"$-J$")
        a.set_xlabel(r"$|J|$")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_ylabel(r"$V$")
    ax.legend(frameon=False)
    ax.set_title("Josephson voltmeter (value when each point closes)", fontsize=9)
    axl.set_yscale("log")
    axl.axhline(1e-4, color="0.5", ls=":", lw=0.9)
    axl.text(0.01, 1.4e-4, r"$V_{\rm th}$", transform=axl.get_yaxis_transform(), color="0.4")
    axl.set_ylabel(r"$|V|$")
    axl.set_title("where the thresholds are read", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close()
    print(" ", out)
    if len(jp) >= 3 or len(jm) >= 3:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        if len(jp) >= 3:
            o = np.argsort(jp)
            js, vs = np.array(jp)[o], np.array(vp)[o]
            ax.plot(js[1:], np.diff(vs) / np.diff(js), "o-", color="#1f4e79",
                    ms=4, label=r"$dV/dJ$ $+J$")
        if len(jm) >= 3:
            o = np.argsort(jm)
            js, vs = np.array(jm)[o], np.array(vm)[o]
            ax.plot(js[1:], np.diff(vs) / np.diff(js), "s--", color="#c45911",
                    ms=4, mfc="white", label=r"$dV/dJ$ $-J$")
        ax.set_xlabel(r"$|J|$")
        ax.set_ylabel(r"$dV/dJ$")
        ax.legend(frameon=False)
        ax.set_title("differential resistance")
        fig.tight_layout()
        dvp = out.with_name("dvdj.png")
        fig.savefig(dvp, dpi=140)
        plt.close()
        print(" ", dvp)


def fig_pgm(pgms: list[Path], out: Path, mosaic: bool) -> None:
    if not pgms:
        return
    if not mosaic:
        A = _read_pgm(pgms[-1])
        if A is None:
            return
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        im = ax.imshow(A, origin="upper", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(pgms[-1].name)
        ax.set_xticks([])
        ax.set_yticks([])
        lab = r"$|\Psi_\pi|^2$" if "psi_p" in pgms[-1].name else r"$|\Psi_\sigma|^2$"
        fig.colorbar(im, ax=ax, fraction=0.046, label=lab)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close()
        print(" ", out)
        return
    def jval(path: Path) -> float:
        m = re.search(r"_J([+-][0-9.]+)_psi", path.name)
        return float(m.group(1)) if m else 0.0

    filas = [sorted([q for q in pgms if jval(q) >= 0 and "_J-_" not in q.name], key=jval),
             sorted([q for q in pgms if jval(q) < 0 or "_J-_" in q.name], key=lambda q: abs(jval(q)))]
    filas = [f[-8:] for f in filas if f]
    cols = max(len(f) for f in filas)
    fig, axes = plt.subplots(len(filas), cols, figsize=(2.6 * cols, 2.3 * len(filas)), squeeze=False)
    for r, fila in enumerate(filas):
        for c in range(cols):
            a = axes[r][c]
            a.set_xticks([])
            a.set_yticks([])
            if c >= len(fila):
                a.axis("off")
                continue
            A = _read_pgm(fila[c])
            if A is not None:
                a.imshow(A, origin="upper", cmap="viridis", vmin=0, vmax=1)
            a.set_title(f"J = {jval(fila[c]):+.4g}", fontsize=8)
    fig.suptitle(r"mid-plane $|\Psi_\sigma|^2$ (PGM when each point closes)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close()
    print(" ", out)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: plot_summary.py RUNDIR")
        sys.exit(2)
    rundir = Path(sys.argv[1]).resolve()
    if not rundir.is_dir():
        print("not a directory:", rundir)
        sys.exit(1)
    out = rundir / "figures"
    out.mkdir(exist_ok=True)
    rows = _j_hist(rundir)
    if rows:
        fig_iv(rows, out / "iv.png")
    else:
        print("  (no j_history.csv and no V= in log; no I–V)")
    pgms = sorted(rundir.glob("**/*psi_s.pgm"))
    if pgms:
        fig_pgm(pgms, out / "psi_s_last.png", mosaic=False)
        if len(pgms) > 1:
            fig_pgm(pgms, out / "psi_s_mosaic.png", mosaic=True)
    pgmp = sorted(rundir.glob("**/*psi_p.pgm"))
    if pgmp:
        fig_pgm(pgmp, out / "psi_p_last.png", mosaic=False)
    if not pgms:
        print("  (no PGM; a snapshot is written when each point closes)")
    print("done", out)
    print("3D: vtk_from_checkpoint.jl on restart/state_J*.mat → ParaView")


if __name__ == "__main__":
    main()
