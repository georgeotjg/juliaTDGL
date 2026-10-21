#!/usr/bin/env python3
"""One Markdown page per release-2 run, for a person or a language model to read without opening the code.

    python3 run_report.py RUNDIR [-o REPORT.md]        (scripts/figures.jl calls it; default RUNDIR/figures/REPORT.md)

From the folder of a run: what was run (material, mesh, the options that change the physics, from run.toml), every rung of
each ladder (J, V, the verdict, crossings per line, time held), the onset brackets and η (from output/summary.json when the
run wrote one, otherwise from the rungs), and the figures of figures/ with what each one shows (figures/README.txt).
Numbers are copied, not recomputed, except the brackets of a ladder without summary.json: the last silent rung below the first
flowing one, and η = (I+ − |I−|)/(I+ + |I−|) at the bracket centres with its range from the bracket ends.
"""
import argparse, csv, json, sys
from pathlib import Path

try:
    import tomllib
except ImportError:                                  # Python < 3.11
    tomllib = None

FLOW = ("V", "N", "D")


def rungs(dir: Path, label: str):
    p = dir / "output" / f"rungs_{label}.csv"
    if not p.is_file():
        return []
    rows = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return list(csv.DictReader(rows))


def bracket(rows):
    """(lo, hi) of |J|: the last silent rung below the first flowing one; lo None when the ladder starts flowing."""
    pts = sorted(((abs(float(r["J"])), r["conv"]) for r in rows), key=lambda t: t[0])
    iv = next((i for i, (_, c) in enumerate(pts) if c in FLOW), None)
    if iv is None:
        return None
    lo = next((pts[i][0] for i in range(iv - 1, -1, -1) if pts[i][1] == "F"), None)
    return (lo, pts[iv][0])


def eta(bp, bn):
    if not bp or not bn or bp[0] is None or bn[0] is None:
        return None
    c = lambda b: (b[0] + b[1]) / 2
    f = lambda a, b: 100 * (a - b) / (a + b)
    return (f(c(bp), c(bn)), f(bp[0], bn[1]), f(bp[1], bn[0]))


def fmt_b(b):
    if not b:
        return "not reached"
    return f"({'?' if b[0] is None else f'{b[0]:.4g}'}, {b[1]:.4g}]"


def summary_of(dir: Path):
    """The brackets and η of a run: from summary.json if there is one, else from its rungs."""
    sj = dir / "output" / "summary.json"
    if sj.is_file():
        r = json.loads(sj.read_text()).get("result", {})
        bp, bn = r.get("bracket_pos"), r.get("bracket_neg")
        return dict(source="output/summary.json", bp=tuple(bp) if bp else None, bn=tuple(bn) if bn else None,
                    eta=tuple(r["eta_percent"]) if r.get("eta_percent") else None,
                    dbp=r.get("delay_bracket_pos"), dbn=r.get("delay_bracket_neg"),
                    deta=r.get("delay_eta_percent"))
    bp, bn = bracket(rungs(dir, "J+")), bracket(rungs(dir, "J-"))
    return dict(source="the rungs (no summary.json)", bp=bp, bn=bn, eta=eta(bp, bn), dbp=None, dbn=None, deta=None)


def options(dir: Path):
    p = dir / "run.toml"
    if tomllib is None or not p.is_file():
        return None
    c = tomllib.loads(p.read_text())
    m, me, sc, g, r = (c.get(k, {}) for k in ("material", "mesh", "scheme", "geometry", "run"))
    bands = m.get("band", [])
    lines = [f"- material: {m.get('file', 'inline')}, κ = {m.get('kappa')}, " + (f"{len(bands)} band(s)" if bands else "the bands of the file")
             + (", ".join([""] + [f"{b.get('name', '?')}: KWT γ̃ = {b.get('gamma_kwt', 0)}" for b in bands]) if bands else ""),
             f"- mesh: {me.get('nx')} × {me.get('ny')} × {me.get('nz')} cells, dx = {me.get('dx')}, buffer {me.get('buf', 1)}",
             f"- geometry: {g.get('GEOM_MODE', '?')}" + (f" ({g['geom_file']})" if g.get("geom_file") else ""),
             f"- field H = {r.get('H')} Hc2; currents J = {r.get('J')}"]
    keys = [k for k in ("selffield_bc", "sigma_vac", "qp_exchange", "qp_split", "noise", "noise_field", "drag", "dt_scale")
            if k in sc]
    lines.append("- scheme: " + (", ".join(f"{k} = {sc[k]}" for k in keys) if keys else "the defaults (bar boundary, metal buffer)"))
    if "heat" in c:
        lines.append("- heat: " + ", ".join(f"{k} = {v}" for k, v in c["heat"].items()))
    if "protocol" in c:
        lines.append("- protocol: " + ", ".join(f"{k} = {v}" for k, v in c["protocol"].items()))
    return lines


def report(dir: Path) -> str:
    out = [f"# {dir.resolve().name}", ""]
    o = options(dir)
    out += ["## What was run", ""] + (o if o else ["(no run.toml in the folder, or Python < 3.11)"]) + [""]
    s = summary_of(dir)
    out += ["## Onsets and efficiency", "", f"From {s['source']}; J in J0, η in %.", "",
            "| | bracket | by the delay |", "|---|---|---|",
            f"| J+ | {fmt_b(s['bp'])} | {fmt_b(tuple(s['dbp'])) if s['dbp'] else '—'} |",
            f"| J− | {fmt_b(s['bn'])} | {fmt_b(tuple(s['dbn'])) if s['dbn'] else '—'} |", ""]
    e = s["eta"]
    line = (f"η = {e[0]:.2f} % [{e[1]:.2f}, {e[2]:.2f}]" if e else "η: both brackets are needed (a '?' means the silent "
            "rung below is not in this folder: a fine ladder started from another run's state).")
    if s["deta"]:
        line += f"; by the delay {s['deta'][0]:.2f} % [{s['deta'][1]:.2f}, {s['deta'][2]:.2f}]"
    out += [line, ""]
    for lab in ("J+", "J-"):
        rows = rungs(dir, lab)
        if not rows:
            continue
        out += [f"## Ladder {lab.replace('-', '−')}", "", "| J | V | verdict | crossings per line | time held (t_GL) |",
                "|---|---|---|---|---|"]
        for r in rows:
            out.append(f"| {float(r['J']):+.4f} | {float(r['V']):+.3e} | {r['conv'] or '(not run)'} | "
                       f"{float(r.get('crossings_per_line') or 0):.3g} | {float(r.get('t_int') or 0):.0f} |")
        out.append("")
    out.append("Verdicts: F silent (fixed point), V flowing, N a band went normal, D flowing found by the delay confirmation; "
               "an empty verdict is a rung the ladder did not run (it stopped after the flowing ones).")
    rd = dir / "figures" / "README.txt"
    if rd.is_file():
        out += ["", "## Figures", ""]
        for l in rd.read_text().splitlines()[1:]:
            name, _, what = l.partition(": ")
            out.append(f"- `{name}`: {what}" if what else f"- {l}")
    out += ["", "Written by tdgl2/tools/run_report.py from the files of the folder; it does not touch the run."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir"); ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    d = Path(a.dir)
    if not (d / "output").is_dir():
        print(f"{d} has no output/: not the folder of a release-2 run"); sys.exit(1)
    o = Path(a.out) if a.out else d / "figures" / "REPORT.md"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(report(d))
    print(f"report in {o}")


if __name__ == "__main__":
    main()
