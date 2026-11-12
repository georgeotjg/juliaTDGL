#!/usr/bin/env python3
"""One table of every release-2 run under one or more folders: one row per run (each folder with output/summary.json).

    python3 collect_runs.py DIR [DIR ...] [-o runs.csv]

Columns: the folder (relative to the DIR it was found under), geometry, κ, H, the mesh, the options that change the physics (KWT per band,
qp_exchange, selffield_bc, σvac, noise, heat, drag), the onset brackets of both polarities, η with its range, the onsets and η by
the delay when the run had the confirmation, whether the brackets closed, the wall time, and the code version and commit.
Ladders run from a script (output/rungs_J±.csv without summary.json) get a row with their brackets and η from the rungs.  For
comparing runs, sorting, plotting, or handing the whole campaign to someone (or a script) as one file.
"""
import argparse, csv, json, sys
from pathlib import Path


def fila(base: Path, sj: Path) -> dict:
    d = json.loads(sj.read_text())
    r, sc, m, me = d.get("result", {}), d.get("scheme", {}), d.get("material", {}), d.get("mesh", {})
    b = lambda k: (r.get(k) or [None, None])
    e = r.get("eta_percent") or [None, None, None]; ed = r.get("delay_eta_percent") or [None, None, None]
    bands = m.get("bands", [])
    try:
        rel = str(sj.parent.parent.relative_to(base))
    except ValueError:
        rel = str(sj.parent.parent)
    return {
        "run": rel, "geometry": d.get("geometry"), "kappa": m.get("kappa"), "H": d.get("H"),
        "dx": me.get("dx"), "sample_cells": "x".join(map(str, me.get("sample_cells", []))) if me.get("sample_cells") else None,
        "padding": me.get("padding"), "nbands": len(bands),
        "gamma_kwt": "/".join(str(x.get("gamma_kwt")) for x in bands) if bands else None,
        "qp_exchange": json.dumps(sc.get("qp_exchange")) if sc.get("qp_exchange") is not None else None,
        "selffield_bc": sc.get("selffield_bc"), "sigma_vac": sc.get("sigma_vac"), "noise": sc.get("noise"),
        "heat": None if sc.get("heat") is None else "on", "drag": sc.get("drag"), "dt": sc.get("dt"),
        "pos_lo": b("bracket_pos")[0], "pos_hi": b("bracket_pos")[1], "neg_lo": b("bracket_neg")[0], "neg_hi": b("bracket_neg")[1],
        "eta_pct": e[0], "eta_lo": e[1], "eta_hi": e[2], "closed": r.get("closed"),
        "delay_pos_lo": b("delay_bracket_pos")[0], "delay_pos_hi": b("delay_bracket_pos")[1],
        "delay_neg_lo": b("delay_bracket_neg")[0], "delay_neg_hi": b("delay_bracket_neg")[1],
        "delay_eta_pct": ed[0], "undecided": len(r.get("undecided") or []),
        "wall_min": round(d["wall_min"], 1) if isinstance(d.get("wall_min"), (int, float)) else None,
        "version": d.get("code", {}).get("version"), "commit": d.get("code", {}).get("commit"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+"); ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    rows = [fila(Path(d).resolve(), sj.resolve()) for d in a.dirs for sj in sorted(Path(d).rglob("output/summary.json"))]
    # ladders run from a script (no summary.json): the brackets and η from their rungs (run_report.py)
    from run_report import summary_of
    for d in a.dirs:
        for rc in sorted(Path(d).rglob("output/rungs_J*.csv")):
            run = rc.parent.parent
            if (run / "output" / "summary.json").is_file() or any(r.get("_dir") == str(run) for r in rows):
                continue
            s = summary_of(run); bp, bn, e = s["bp"] or (None, None), s["bn"] or (None, None), s["eta"] or (None, None, None)
            try:
                rel = str(run.resolve().relative_to(Path(d).resolve()))
            except ValueError:
                rel = str(run)
            r = {k: None for k in rows[0]} if rows else {}
            r.update({"run": rel, "pos_lo": bp[0], "pos_hi": bp[1], "neg_lo": bn[0], "neg_hi": bn[1],
                      "eta_pct": e[0], "eta_lo": e[1], "eta_hi": e[2], "_dir": str(run)})
            rows.append(r)
    for r in rows:
        r.pop("_dir", None)
    if not rows:
        print("no release-2 runs (output/summary.json) under", ", ".join(a.dirs)); sys.exit(1)
    out = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if a.out:
        out.close(); print(f"{len(rows)} run(s) in {a.out}")


if __name__ == "__main__":
    main()
