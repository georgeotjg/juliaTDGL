#!/usr/bin/env python3
"""Terminal menu for TDGL runs. Closing it NEVER stops Julia.

  tdgl_tui.py              interactive (live runs; a = all)
  tdgl_tui.py --once       print the table once
  tdgl_tui.py RUNDIR       details of one run directory
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import math
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
# julia/runs first, then every other <folder>/runs of the project (articles, etc.).
# julia/runs first (runs launched from this menu), then any other runs/ folder of
# the project up to two levels down (articles, campaigns), listed read-only.
ROOTS = [BASE / "julia" / "runs"] + sorted(
    d for patron in ("*/runs", "*/*/runs") for d in BASE.glob(patron)
    if d.is_dir() and d != BASE / "julia" / "runs"
    and not any(x.startswith((".", "trash_")) for x in d.relative_to(BASE).parts))
# One solver, julia/: the two-band kernel plus optional terms, all off by default.
JULIA = BASE / "julia"
LANZAR = JULIA / "scripts" / "pin_cpu.sh"
FORMAS = JULIA / "shapes"
EDITOR = JULIA / "editor" / "server.py"
PRESET = JULIA / "scripts" / "run_preset.jl"

CATALOG = [
    ("SIM", "clean box: control, η must be 0"),
    ("WEDGE", "sawtooth notch on face y1 (geometric diode)"),
    ("WEDGE_REF", "WEDGE mirrored in y (sign control)"),
    ("WEDGE_BOTH", "notches on y1 and y2"),
    ("ESFERA", "through cylinder of reduced f (pins, does not rectify)"),
    ("ARRAY", "three cylinders of growing radius"),
    ("ESF_CUNA", "cylinder + notch"),
    ("FGRAD", "sawtooth in f(y), no hole"),
    ("SAWROW", "row of five notches"),
    ("SPHROW", "three cylinders (R=3,6,4)"),
    ("SAWSPH", "notches + cylinders"),
    ("STAGGER", "staggered triangles"),
    ("FUNNEL", "two rows of triangles (funnels); one apex touches y2"),
    ("FUNNEL_IN", "FUNNEL with the apex away from y2"),
    ("TRI_Y1", "one triangle with its base on face y1"),
    ("TRI_Y1_INV", "triangle with its apex on y1"),
    ("RECT_Y1", "11×5 rectangle on y1"),
    ("HEX", "hexagonal hole"),
    ("HEX_STAG", "hexagonal hole + triangles around it"),
]
# Shown first; the rest stay in the solver and as templates julia/shapes/cat_*.toml.
USADAS = ("SIM", "WEDGE", "WEDGE_REF", "STAGGER", "FUNNEL")
CATALOG_VISIBLE = [(n, nota) for n, nota in CATALOG if n in USADAS]
CATALOG_OTRAS = [(n, nota) for n, nota in CATALOG if n not in USADAS]
WIDE = {
    "FUNNEL", "FUNNEL_IN", "TRI_Y1", "TRI_Y1_INV", "RECT_Y1", "STAGGER", "SAWROW", "SAWSPH", "ARRAY", "ESF_CUNA",
    "HEX", "HEX_STAG", "SPHROW", "WEDGE_BOTH", "FGRAD",
}
RUNS = JULIA / "runs"
DEFAULTS = {
    "geom": "WEDGE",
    "kappa": "2",
    "gamma12": "0.001",
    "J": "0.10",
    "drag": "0",
    "mesh": "small",
    "window": "smoke",
    "z": "through",
    "cpus": "8,9",
}


def _alive(pid: str, rundir: Path | None = None, status: str = "") -> bool:
    """True only if a Julia process of THIS run is alive. STATUS 'done' means off
    (the .pid file stays and Linux reuses the number)."""
    st = status.lower()
    if " done" in st or st.rstrip().endswith("done"):
        return False
    if not pid.isdigit():
        return False
    proc = Path("/proc") / pid
    if not proc.exists():
        return False
    try:
        comm = (proc / "comm").read_text().strip().lower()
    except OSError:
        return False
    if "julia" not in comm:
        return False
    if rundir is None:
        return True
    try:
        cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except OSError:
        cmd = ""
    try:
        cwd = os.readlink(proc / "cwd")
    except OSError:
        cwd = ""
    needle = str(rundir.resolve())
    tag = rundir.name
    if needle in cmd or needle in cwd or tag in cmd or tag in cwd:
        return True
    # older campaign scripts do not put the run directory in argv; the pid
    # file lives in THIS directory and STATUS does not say done.
    return True


def _ps(pid: str) -> tuple[str, str]:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "etime=,pcpu=", "-p", pid], text=True
        ).split()
        if len(out) >= 2:
            return out[0], out[1] + "%"
    except Exception:
        pass
    return "—", "—"


def _julia_externos(conocidos: set[str]) -> list[dict]:
    """Julia processes running a .jl of this project that the menu did not launch
    (campaign or verification scripts): one row each, so they show as live."""
    filas = []
    for proc in Path("/proc").iterdir():
        pid = proc.name
        if not pid.isdigit() or pid in conocidos:
            continue
        try:
            if "julia" not in (proc / "comm").read_text().lower():
                continue
            cmd = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [c.decode(errors="replace") for c in cmd if c]
        jl = next((a for a in args if a.endswith(".jl")), None)
        if jl is None:
            continue
        try:
            cwd = Path(os.readlink(proc / "cwd"))
        except OSError:
            cwd = BASE
        script = (cwd / jl).resolve() if not Path(jl).is_absolute() else Path(jl)
        if BASE not in script.parents or script.parent.name == "test":
            continue
        rundir = script.parent
        et, pc = _ps(pid)
        last, js, n_exp, caso, conv = "—", [], 0, 0, ""
        # the process's own stdout (several runs may share one folder)
        try:
            outs = [Path(os.readlink(proc / "fd" / "1"))]
        except OSError:
            outs = sorted(rundir.glob("stdout*.txt"))
        for out in outs:
            try:
                texto = out.read_text(errors="replace")
            except OSError:
                continue
            # one script may run several cases: progress is that of the current one
            partes = re.split(r"^=== TDGL two-band solver.*$", texto, flags=re.M)
            caso = len(partes) - 1
            actual = partes[-1]
            hits = re.findall(r"\[(J[+-])\] J=([+-][0-9.]+).*?conv=(\S)", actual)
            m = re.search(r"\[J_list\][^\n]*?(\d+) points", actual)
            if hits or m:
                js = [f"J{h[1]}" for h in hits]
                n_exp = 2 * int(m.group(1)) if m else 0
                if hits:
                    last, conv = f"J{hits[-1][1]}", hits[-1][2]
        extra = " ".join(a for a in args[args.index(jl) + 1:])
        tag = str(rundir.relative_to(BASE)) + (f" [{extra}]" if extra else "")
        filas.append({
            "tag": tag + (f" · case {caso}" if caso > 1 else ""),
            "on": True, "etime": et, "pcpu": pc, "last": last, "age_min": None,
            "dir": rundir, "status": "running (script)", "pid": pid, "n_j": len(js),
            "js": js, "job": {"n_expected": n_exp}, "hist": [{"conv": conv}] if conv else [],
            "stdout": str(outs[0]) if outs else "",
            "frozen": True,
        })
    return filas


# ---------------------------------------------------------------------------
#  Quick results: the verdict rule of the paper, read from j_history.csv
# ---------------------------------------------------------------------------
def _tciclo(d: Path) -> float | None:
    """Physical time of one cycle (dt*lMax): run_info.txt written by the solver
    next to the history, else the run summary, else the dt printed in a stdout
    of the folder or its parent (runs started before run_info.txt existed)."""
    for ri in (d / "run_info.txt", d / "output" / "run_info.txt"):
        if ri.is_file():
            m = re.search(r"t_cycle=([0-9.eE+-]+)", ri.read_text(errors="replace"))
            if m:
                return float(m.group(1))
    v = _tciclo_resumen(d)
    if v is not None:
        return v
    outs = sorted(d.glob("stdout*.txt")) + sorted(d.parent.glob(f"stdout*{d.name}*.txt")) + sorted(d.parent.glob("stdout.txt"))
    for s in outs:
        dts = re.findall(r"CFL: dt=([0-9.eE+-]+)", s.read_text(errors="replace"))
        if dts:
            return float(dts[-1]) * 40      # lMax = 40, the default, for runs that do not print it
    return None


def _tciclo_resumen(d: Path) -> float | None:
    for s in sorted(list(d.glob("RUN_SUMMARY_*.txt")) + list(d.glob("output/RUN_SUMMARY_*.txt")) +
                    list(d.parent.glob("RUN_SUMMARY_*.txt")))[::-1]:
        txt = s.read_text(errors="replace")
        a = re.search(r"dt=([0-9.eE+-]+) t_GL", txt)
        b = re.search(r"lMax=(\d+)", txt)
        if a and b:
            return float(a.group(1)) * int(b.group(1))
    return None


def _bracket(rows: list[tuple[float, float, int]], tc: float) -> dict:
    """rows = (|J|, V, cycles) of one polarity, ascending.  Same rule as
    TDGL.bracket_Ic: crossings per line |V| t / 2π, <1 silent, >=10 dissipating."""
    cls = []
    for j, v, it in rows:
        c = abs(v) * it * tc / (2 * math.pi)
        cls.append((j, "D" if c >= 10 else "S" if c < 1 else "U", c, v))
    d = next((q for q, x in enumerate(cls) if x[1] == "D"), None)
    hi = cls[d][0] if d is not None else math.inf
    below = cls[:d] if d is not None else cls
    s = [x[0] for x in below if x[1] == "S"]
    lo = s[-1] if s else math.nan
    und = [x[0] for x in below if x[1] == "U" and (math.isnan(lo) or x[0] > lo)]
    return {"lo": lo, "hi": hi, "und": und, "cls": cls}


def quick_result(d: Path) -> list[str]:
    """One line per case (j_history.csv) under d: brackets, η with its range,
    undecided rungs and σ_V where both polarities dissipate."""
    out = []
    for csvp in sorted(set(d.glob("j_history.csv")) | set(d.glob("*/j_history.csv")) | set(d.glob("*/*/j_history.csv"))):
        if "output" in csvp.parts[-2:-1]:
            continue
        tc = _tciclo(csvp.parent) or _tciclo(csvp.parent / "output")
        last: dict[tuple[str, float], tuple[float, int]] = {}
        try:
            with csvp.open(errors="replace") as f:
                for r in csv.DictReader(f):
                    j = abs(float(r["J"]))
                    if j > 1e-12:
                        last[(r["pol"], round(j, 6))] = (float(r["V"]), int(float(r.get("it") or 0)))
        except (OSError, KeyError, ValueError):
            continue
        caso = str(csvp.parent.relative_to(d)) if csvp.parent != d else d.name
        if tc is None or not last:
            out.append(f"{caso}: (no summary with dt yet)")
            continue
        B = {}
        for pol in ("J+", "J-"):
            rows = sorted((j, v, it) for (p_, j), (v, it) in last.items() if p_ == pol)
            B[pol] = _bracket(rows, tc)
        f = lambda x: "?" if math.isnan(x) else "∞" if math.isinf(x) else f"{x:.4g}"
        bp, bn = B["J+"], B["J-"]
        br = lambda b: f"({f(b['lo'])}, {f(b['hi'])})" if b["cls"] else "— (not started)"
        line = f"{caso}: Ic+ ∈ {br(bp)}  |Ic−| ∈ {br(bn)}"
        if all(math.isfinite(x) for x in (bp["lo"], bp["hi"], bn["lo"], bn["hi"])):
            e = lambda a, b: 100 * (a - b) / (a + b)
            line += (f"  η = {e((bp['lo']+bp['hi'])/2, (bn['lo']+bn['hi'])/2):+.1f} %"
                     f" [{e(bp['lo'], bn['hi']):+.1f}, {e(bp['hi'], bn['lo']):+.1f}]")
        und = [f"+{f(x)}" for x in bp["und"]] + [f"−{f(x)}" for x in bn["und"]]
        if und:
            line += "  undecided: " + " ".join(und)
        dp = {x[0]: x[3] for x in bp["cls"] if x[1] == "D"}
        dn = {x[0]: x[3] for x in bn["cls"] if x[1] == "D"}
        com = sorted(set(dp) & set(dn))
        if com:
            j = com[0]
            a, b = abs(dp[j]), abs(dn[j])
            line += f"  σ_V@{f(j)} = {100 * (a - b) / (a + b):+.1f} %"
        out.append(line)
    return out


def _envolver(lineas: list[str], ancho: int) -> list[str]:
    """Breaks a result line before 'undecided' or 'σ_V' when it does not fit."""
    out = []
    for ln in lineas:
        if _vis(ln) <= ancho:
            out.append(ln)
            continue
        partes = re.split(r"  (?=undecided:|σ_V@|η =)", ln)
        out.append(partes[0])
        out += ["      " + x for x in partes[1:]]
    return out


def _notificar(titulo: str, cuerpo: str) -> None:
    if shutil.which("notify-send"):
        subprocess.Popen(["notify-send", "-a", "TDGL", titulo, cuerpo],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


TERMINADAS: list[str] = []


def _vigia(intervalo: float = 30.0) -> None:
    """Background watcher: when a live run disappears, notify with its result."""
    vivos: dict[str, tuple[str, Path]] = {}
    while True:
        try:
            ahora = {r["pid"]: (r["tag"], r["dir"]) for r in collect() if r["on"] and r["pid"]}
            for pid, (tag, d) in vivos.items():
                if pid not in ahora:
                    res = quick_result(d)
                    cuerpo = "\n".join(res[-3:]) if res else str(d)
                    TERMINADAS.append(f"{time.strftime('%H:%M')} {tag}: " + (res[-1] if res else "finished"))
                    _notificar(f"TDGL: finished {tag}", cuerpo)
            vivos = ahora
        except Exception:   # noqa: BLE001 — the watcher must never take the menu down
            pass
        time.sleep(intervalo)


def _resultados(rows: list[dict]) -> None:
    print()
    lin, vistos = [], set()
    for r in rows:
        res = [x for x in quick_result(r["dir"]) if x not in vistos]
        vistos.update(res)
        if res:
            lin.append(_col("1", r["tag"]) + ("  " + _dim("(live)") if r["on"] else ""))
            lin += ["  " + x for x in res]
    caja("results — verdict rule: crossings per line <1 silent, ≥10 dissipating",
         _envolver(lin, _ancho() - 6) or ["no j_history.csv yet"])
    _volver()


def _jtag(name: str) -> str:
    m = re.search(r"(J[+-][0-9.]+)", name)
    return m.group(1) if m else name


def _job_json(rundir: Path) -> dict:
    p = rundir / "job.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _j_history(rundir: Path) -> list[dict]:
    rows = []
    for cand in (rundir / "j_history.csv", *rundir.glob("**/j_history.csv")):
        if not cand.exists():
            continue
        try:
            lines = cand.read_text(errors="replace").splitlines()
        except Exception:
            continue
        if not lines:
            continue
        hdr = [h.strip() for h in lines[0].split(",")]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            rec = {hdr[i] if i < len(hdr) else str(i): parts[i].strip()
                   for i in range(len(parts))}
            rows.append(rec)
        if rows:
            return rows
    return rows


def _j_points(rundir: Path) -> list[tuple[float, str]]:
    hist = _j_history(rundir)
    if hist:
        out = []
        for rec in hist:
            pol = rec.get("pol", "")
            j = rec.get("J", "")
            try:
                tag = f"{pol}{float(j):+.3f}".replace("J+", "J+").replace("J-", "J-")
            except Exception:
                tag = f"{pol}{j}"
            if not tag.startswith("J"):
                tag = ("J+" if "+" in pol or (pol.endswith("+")) else "J-") + (
                    f"{float(j):.3f}" if j else ""
                )
            try:
                jf = float(j)
                tag = ("J+" if (pol.endswith("+") or pol == "+") else "J-") + f"{abs(jf):.3f}"
                if jf < 0 or str(pol).endswith("-"):
                    tag = f"J{jf:+.3f}"
                else:
                    tag = f"J{jf:+.3f}"
            except Exception:
                tag = f"{pol}{j}"
            out.append((0.0, tag))
        # unique chronological
        seen = []
        for _, t in out:
            if t not in seen:
                seen.append(t)
        return [(i, t) for i, t in enumerate(seen)]
    pts: list[tuple[float, str]] = []
    seen = set()
    for p in rundir.glob("**/*psi_s.pgm"):
        try:
            geom = p.relative_to(rundir).parts[0]
        except Exception:
            geom = ""
        tag = f"{geom} {_jtag(p.name)}".strip()
        if tag in seen:
            continue
        seen.add(tag)
        pts.append((p.stat().st_mtime, tag))
    pts.sort()
    return pts


def _log_js(rundir: Path) -> list[str]:
    out = []
    for cand in (rundir / "log.txt", rundir / "stdout.txt"):
        if not cand.exists():
            continue
        try:
            for line in cand.read_text(errors="replace").splitlines():
                if re.search(r"J=[+-]", line) and "V=" in line:
                    if "it=0" in line and "conv= " in line:
                        continue
                    out.append(line.strip())
        except Exception:
            pass
    parent = rundir.parent
    if parent.name == "runs":
        for p in parent.glob(f"TUI_{rundir.name.split('_')[0] if '_' in rundir.name else ''}*_stdout.txt"):
            try:
                for line in p.read_text(errors="replace").splitlines():
                    if re.search(r"J=[+-]", line) and "V=" in line:
                        out.append(line.strip())
            except Exception:
                pass
    # unique preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def collect() -> list[dict]:
    now = time.time()
    rows = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        for status in sorted(root.rglob("STATUS.txt")):
            rundir = status.parent
            if rundir.name in ("output", "restart", "figures", "snapshots"):
                continue
            lines = status.read_text(errors="replace").strip().splitlines()
            st = lines[-1] if lines else ""
            pidf = rundir / "julia.pid"
            pid = pidf.read_text().strip() if pidf.exists() else ""
            on = _alive(pid, rundir, st)
            et, pc = _ps(pid) if on else ("—", "—")
            pgms = list(rundir.glob("**/*psi_s.pgm"))
            last, age_min = "—", None
            if pgms:
                p = max(pgms, key=lambda x: x.stat().st_mtime)
                last = _jtag(p.name)
                age_min = int((now - p.stat().st_mtime) / 60)
            hist = _j_history(rundir)
            if hist:
                rec = hist[-1]
                try:
                    last = f"J{float(rec['J']):+.3f}"
                except Exception:
                    pass
            try:
                partes = rundir.relative_to(root).parts
                # a run with cases in subfolders: show run/case
                tag = "/".join(partes[:2])
            except (ValueError, IndexError):
                tag = rundir.name
            pts = _j_points(rundir)
            job = _job_json(rundir)
            rows.append(
                {
                    "tag": tag,
                    "on": on,
                    "etime": et,
                    "pcpu": pc,
                    "last": last,
                    "age_min": age_min,
                    "dir": rundir,
                    "status": st,
                    "pid": pid,
                    "n_j": len(pts),
                    "js": [t for _, t in pts],
                    "job": job,
                    "hist": hist,
                    "frozen": root != ROOTS[0],
                }
            )
    rows += _julia_externos({r["pid"] for r in rows if r["on"]})
    rows.sort(key=lambda r: (not r["on"], r["tag"]))
    return rows


def _n_exp_from_status(st: str) -> int:
    """Run without job.json: 'J=+0.56,+0.64' → (those |J| ∪ {0}) × 2 polarities."""
    m = re.search(r"J=([0-9eE+.,\-]+)", st or "")
    if not m:
        return 0
    js = []
    for p in m.group(1).split(","):
        try:
            js.append(abs(float(p)))
        except ValueError:
            continue
    if not js:
        return 0
    return 2 * len(set([0.0] + js))


def _pct(row: dict) -> str:
    job = row.get("job") or {}
    n_exp = int(job.get("n_expected") or 0)
    n_closed = row["n_j"]
    tag = row["tag"]
    if n_exp <= 0 and "HARD1200" in tag:
        n_exp = 4
    if n_exp <= 0 and "," in (row.get("status") or ""):
        n_exp = _n_exp_from_status(row.get("status") or "")
    if n_exp <= 0 or n_closed > n_exp:
        return f"{n_closed} pts" if n_closed else "—"
    frac = n_closed / float(n_exp)
    if row["on"] and n_closed < n_exp and row["age_min"] is not None:
        frac = min(0.99, frac + 0.05)
    return f"{100 * frac:.0f}% {n_closed}/{n_exp}"


def _tty() -> bool:
    return sys.stdout.isatty()


def _col(code: str, s: str) -> str:
    if not _tty():
        return s
    return f"\033[{code}m{s}\033[0m"


def _dim(s: str) -> str:
    return _col("2", s)


def _key(letter: str, rest: str) -> str:
    return _col("1;36", letter) + _dim(" " + rest)



_ANSI = re.compile(r"\033\[[0-9;]*m")


def _vis(s: str) -> int:
    """Visible columns (without colour codes)."""
    return len(_ANSI.sub("", s))


def _ancho() -> int:
    return max(76, min(130, shutil.get_terminal_size((100, 30)).columns))


def _corta(s: str, n: int) -> str:
    if _vis(s) <= n:
        return s
    return _ANSI.sub("", s)[: max(0, n - 1)] + "…"


def _corta_medio(s: str, n: int) -> str:
    """Cut in the middle: run names differ at the end."""
    if len(s) <= n:
        return s
    cab = (n - 1) // 2
    return s[:cab] + "…" + s[len(s) - (n - 1 - cab):]


def caja(titulo: str, lineas: list[str], derecha: str = "", color: str = "36") -> None:
    """A box as wide as the terminal, title on the left."""
    W = _ancho()
    t, d = f" {titulo} ", (f" {derecha} " if derecha else "")
    print(_dim("┌─") + _col("1;" + color, t) + _dim("─" * max(0, W - 3 - _vis(t) - _vis(d)) + d + "┐"))
    for ln in lineas:
        ln = _corta(ln, W - 4)
        print(_dim("│ ") + ln + " " * (W - 4 - _vis(ln)) + _dim(" │"))
    print(_dim("└" + "─" * (W - 2) + "┘"))


def _K(letra: str) -> str:
    return _col("1;36", f"[{letra}]")


def _volver() -> None:
    """Pause at the foot of each screen."""
    input(f"{_K('Enter')} back ")


def _motor(r: dict) -> str:
    """archive = an article run (read only); otherwise the optional term it used."""
    if r.get("status") == "running (script)":
        return "script"
    if r.get("frozen"):
        return "archive"
    t = _term_de_job(r.get("job") or {})
    return "base" if not t else ("drag" if t.get("drag") else "heat" if t.get("calor") else "Q diff.")


def _estado(r: dict) -> str:
    st = (r.get("status") or "").lower()
    if r["on"]:
        return _col("1;32", "● live")
    if "done" in st or "terminad" in st:
        return _dim("✓ done")
    if "stop" in st:
        return _col("33", "■ stopped")
    return _dim("· idle")


def _barra(r: dict) -> str:
    pc = _pct(r)
    m = re.search(r"(\d+)/(\d+)", pc)
    if not m:
        return _dim(pc if pc.endswith("pts") else "—")
    a, b = int(m.group(1)), int(m.group(2))
    k = min(8, round(8 * a / b)) if b else 0
    return _col("36", "█" * k) + _dim("░" * (8 - k)) + f" {a}/{b}"


def _ultimo(r: dict) -> str:
    hist = r.get("hist") or []
    conv = hist[-1].get("conv", "") if hist else ""
    color = {"V": "33", "T": "31", "F": "2"}.get(conv, "0")
    return r["last"] + (" " + _col(color, conv) if conv else "")



def table(rows: list[dict], *, only_on: bool) -> None:
    show = [r for r in rows if r["on"]] if only_on else rows
    W = _ancho()
    wjob = max(18, W - 4 - 62)
    cab = (f"{'#':<3} {'run':<{wjob}} {'term':<8} {'state':<9} "
           f"{'progress':<14} {'last J':<13} {'%CPU':>6}")
    filas = [_dim("Closing this does not stop Julia."), "", _col("1", cab)]
    if not show:
        filas.append(_dim("  (no live runs: [a] shows all)" if only_on else "  (empty)"))
    for i, r in enumerate(show, 1):
        tag = _corta_medio(r["tag"], wjob)
        celdas = [f"{i:<3}", tag + " " * (wjob - _vis(tag)), f"{_motor(r):<8}",
                  _estado(r) + " " * max(0, 9 - _vis(_estado(r))),
                  _barra(r) + " " * max(0, 14 - _vis(_barra(r))),
                  _ultimo(r) + " " * max(0, 13 - _vis(_ultimo(r))), f"{r['pcpu']:>6}"]
        filas.append(" ".join(celdas))
    caja("TDGL · runs", filas,
         derecha=time.strftime("%Y-%m-%d %H:%M") + " · " + ("live only" if only_on else "all"))


def _ascii_curve(hist: list[dict], pol: str, width: int = 52, height: int = 8) -> None:
    pts = []
    for rec in hist:
        try:
            j = float(rec["J"])
            v = float(rec["V"])
        except Exception:
            continue
        p = rec.get("pol", "")
        if pol == "+" and (j < -1e-12 or str(p).endswith("-")):
            continue
        if pol == "-" and (j > 1e-12 or str(p).endswith("+")):
            continue
        pts.append((abs(j), v))
    if len(pts) < 2:
        return
    pts.sort()
    ys = [v for _, v in pts]
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-18:
        hi = lo + 1e-12
    print(f"  V(J) polarity {pol}   {pts[0][0]:.2f} → {pts[-1][0]:.2f}")
    grid = [[" "] * width for _ in range(height)]
    for j, v in pts:
        x = int((j - pts[0][0]) / max(pts[-1][0] - pts[0][0], 1e-12) * (width - 1))
        y = int((hi - v) / (hi - lo) * (height - 1))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        grid[y][x] = "•"
    for row in grid:
        print("  |" + "".join(row))
    print("  +" + "-" * width)


def detail(r: dict) -> None:
    d = r["dir"]
    job = r.get("job") or {}
    print()
    res = quick_result(d)
    if res:
        caja("result (verdict rule)", _envolver(res, _ancho() - 6), color="32")
    caja(r["tag"], [
        f"folder   {d}",
        f"state    {_estado(r)}   {_dim(r['status'])}",
        f"term     {_motor(r)}   pid {r['pid'] or '—'}   progress {_barra(r)}",
        f"         {_term_texto(_term_de_job(job))}",
    ])
    if job.get("geom"):
        bits = [f"geom={job.get('geom')}", f"κ={job.get('kappa')}", f"H={job.get('H')}",
                f"γ12={job.get('gamma12')}"]
        if job.get("Nx"):
            bits.append(f"mesh={job.get('Nx')}x{job.get('Ny')}x{job.get('Nz')} dx={job.get('dx')}")
        if job.get("t_min_GL"):
            bits.append(f"t={job.get('t_min_GL')}/{job.get('t_min_GL_res')}")
        if job.get("T_red"):
            bits.append(f"T={job.get('T_red')}")
        print("params " + " ".join(bits))
        jl = job.get("J_list")
        if jl:
            print(f"J list {jl}   expected {job.get('n_expected')} closed points (both polarities)")
    csvd = d
    if r.get("status") == "running (script)":
        # a script may run several cases: show the one being written now
        cands = [c for c in d.glob("**/j_history.csv") if "output" not in c.parts[len(d.parts):]]
        hist = []
        if cands:
            # the case of THIS process: its stdout is often named after it (stdout_<case>.txt)
            so = Path(r.get("stdout") or "").stem
            mio = [c for c in cands if c.parent.name and c.parent.name in so]
            csvd = max(mio or cands, key=lambda c: c.stat().st_mtime).parent
            hist = _j_history(csvd)
            print(f"case   {csvd.relative_to(d) if csvd != d else d.name}  (the one running now)")
    else:
        hist = r.get("hist") or _j_history(d)
    if hist:
        print("--- history (one line per closed current point; j_history.csv) ---")
        print(f"  {'time':<20} {'pol':<4} {'J':>8} {'V':>12} {'class':<5} Nv  |Ψπ|²")
        for rec in hist:
            print(
                f"  {rec.get('time',''):<20} {rec.get('pol',''):<4} "
                f"{rec.get('J','')[:8]:>8} {_fv(rec.get('V','')):>12} "
                f"{rec.get('conv',''):<5} {rec.get('Nv_s','')}/{rec.get('Nv_p','')}  "
                f"{rec.get('psi_p2','')}"
            )
        csv = csvd / "j_history.csv"
        print(f"  csv  {csv if csv.exists() else d / '**/j_history.csv'}")
        _ascii_curve(hist, "+")
        _ascii_curve(hist, "-")
    else:
        js = r["js"]
        print(f"closed points ({len(js)}): " + (" → ".join(js) if js else "(none yet)"))
        logs = _log_js(d)
        if logs:
            print("--- V in log.txt ---")
            for line in logs[-20:]:
                print(" ", line)
        else:
            print("(no V yet: results are written when each current point CLOSES)")
    print("Checkpoint = last closed point. Stopping in the middle of a point loses that point.")
    print("Classes: F fixed point (no flow), V limit cycle (flux flow), T heat cap reached, * cycle ceiling.")
    if r.get("status") == "running (script)":
        print("Launched by a script, outside the menu: read only here.")
        print()
        print(f"{_K('Enter')} or {_K('b')} back   {_K('o')} folder   {_K('e')} csv   {_K('f')} figures")
    elif r.get("frozen"):
        print("Article run (archive): read only.")
        print()
        print(f"{_K('Enter')} or {_K('b')} back   {_K('o')} folder   {_K('e')} csv   {_K('f')} figures")
    else:
        print()
        print(f"{_K('Enter')} or {_K('b')} back   {_K('s')} stop   {_K('u')} resume   {_K('o')} folder   "
              f"{_K('d')} to trash_   {_K('e')} csv   {_K('f')} figures")
    k = input("> ").strip().lower()
    if k == "o":
        _open_dir(d)
    elif k == "s" and not r.get("frozen"):
        _stop_job(r)
    elif k == "u" and not r.get("frozen"):
        _resume_job(r)
    elif k == "d":
        _delete_job(r)
    elif k == "e":
        csvp = csvd / "j_history.csv"
        print(csvp if csvp.exists() else "(no j_history.csv yet)")
        _volver()
    elif k == "f":
        _figuras(r)


def _fv(v: str) -> str:
    try:
        return f"{float(v):+.4e}"
    except ValueError:
        return v[:12]


def _open_dir(d: Path) -> None:
    print(d)
    _abrir(str(d))


def _abrir(cosa: str) -> None:
    """xdg-open without waiting: the viewer does not freeze the menu."""
    subprocess.Popen(["xdg-open", cosa], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def _editor_url(cargar: str = "") -> str:
    """Starts the editor server if it is not running and opens the browser."""
    import socket
    with socket.socket() as so:
        so.settimeout(0.3)
        vivo = so.connect_ex(("127.0.0.1", 8765)) == 0
    if not vivo:
        subprocess.Popen([sys.executable, str(EDITOR), "--no-browser"], cwd=str(BASE),
                         start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8)
    url = "http://127.0.0.1:8765/" + (f"?load={cargar}" if cargar else "")
    _abrir(url)
    return url


def _delete_job(r: dict) -> None:
    """Moves julia/runs/<run> to trash_/tui_jobs/. Never article runs, never rmtree."""
    d = r["dir"].resolve()
    root = RUNS.resolve()
    trash = (BASE / "trash_" / "tui_jobs").resolve()
    print()
    if r["on"]:
        print("it is running. Stop it first (s).")
        _volver()
        return
    if r.get("frozen"):
        print("article run: not touched from here.")
        _volver()
        return
    try:
        rel = d.relative_to(root)
    except ValueError:
        print("outside julia/runs/, not touched:", d)
        _volver()
        return
    if d == root or rel == Path(".") or ".." in rel.parts:
        print("refusing this path.")
        _volver()
        return
    dest = trash / r["tag"]
    if dest.exists():
        dest = trash / f"{r['tag']}_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"move  {d}")
    print(f"  →   {dest}")
    print("nothing is deleted; trash_ is emptied by hand.")
    conf = input(f"type exactly:  TRASH {r['tag']}\n> ").strip()
    if conf != f"TRASH {r['tag']}":
        print("cancelled")
        _volver()
        return
    trash.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))
    print("moved to", dest)
    _volver()


def _abrir_editor(cargar: str = "") -> None:
    """Shape editor in the browser (local server on 127.0.0.1:8765)."""
    if not EDITOR.exists():
        print("editor not found:", EDITOR)
        _volver()
        return
    url = _editor_url(cargar)
    print(f"editor: {url}   (saves to julia/shapes/)")
    print(_dim("a small local Python server; it stops by itself 30 min after the editor tab is closed"))
    _volver()


def _elegir_forma() -> tuple[str, str] | None:
    """Returns (geom, path) for a shape in julia/shapes/, or None (back)."""
    fs = sorted(FORMAS.glob("*.toml")) if FORMAS.is_dir() else []
    if not fs:
        print("no shapes in", FORMAS, "— draw one with g → e")
        return None
    mias = [f for f in fs if not f.stem.startswith("cat_")]
    cats = [f for f in fs if f.stem.startswith("cat_")]
    orden = mias + cats
    if mias:
        print(_col("1", "  your shapes"))
        for n, f in enumerate(mias, 1):
            print(f"  {_col('1;36', f'{n:2}')}  {f.stem}")
    if cats:
        print(_col("1", "  catalogue templates (box 40×24×6)"))
        print("  " + "  ".join(f"{_col('1;36', str(len(mias) + n))}={f.stem[4:]}" for n, f in enumerate(cats, 1)))
    nuevo = max(range(len(mias)), key=lambda i: mias[i].stat().st_mtime) + 1 if mias else 1
    if mias:
        print(_dim(f"  default = the most recently saved: {mias[nuevo - 1].stem}"))
    k = _ask("shape (number)", str(nuevo))
    if k is None or not k.isdigit() or not 1 <= int(k) <= len(orden):
        return None
    ref = _ask("mirror in y (_REF)? y/N", "N")
    if ref is None:
        return None
    return ("CUSTOM_REF" if ref.lower() in ("y", "s", "si") else "CUSTOM", str(orden[int(k) - 1]))


def _geom_menu() -> None:
    lineas = [_dim("The sample is the box; a geometry is where the condensation strength f is reduced."),
              _dim("J runs along x and H along z, so vortices move along y: a diode needs a shape that"),
              _dim("is not symmetric under the mirror in y."),
              "", _col("1", "most used")]
    for i, (n, nota) in enumerate(CATALOG_VISIBLE, 1):
        lineas.append(f"  {_K(str(i))} {n:<10} {_dim(nota)}")
    lineas += ["",
               f"{_K('e')} shape editor (draw, rotate, cut planes, voxels)",
               f"{_K('p')} templates: the other {len(CATALOG_OTRAS)} catalogue shapes",
               _dim("number = open that shape in the editor  ·  Enter or b = back")]
    caja("geometries", lineas)
    k = (input("geom> ") or "").strip().lower()
    if k == "e":
        _abrir_editor()
    elif k == "p":
        lin = [f"  {_K(str(i))} {n:<10} {_dim(nota)}" for i, (n, nota) in enumerate(CATALOG_OTRAS, 1)]
        lin.append(_dim("number = open in the editor  ·  Enter or b = back"))
        caja("templates", lin)
        k2 = (input("template> ") or "").strip()
        if k2.isdigit() and 1 <= int(k2) <= len(CATALOG_OTRAS):
            _abrir_editor("cat_" + CATALOG_OTRAS[int(k2) - 1][0])
    elif k.isdigit() and 1 <= int(k) <= len(CATALOG_VISIBLE):
        _abrir_editor("cat_" + CATALOG_VISIBLE[int(k) - 1][0])


def _ask(prompt: str, default: str, *, back: bool = True) -> str | None:
    """Input with a default; returns None when the user types b (back)."""
    extra = "  b=back" if back else ""
    s = input(f"{prompt} [{default}]{extra}> ").strip()
    if back and s.lower() in ("b", "-", "q"):
        return None
    return s if s else default


def _term_texto(t: dict) -> str:
    if not t:
        return "none (two-band TDGL only)"
    if t.get("drag"):
        return f"drag ν={t.get('nu', '0.05')}"
    if t.get("calor"):
        return (f"heat γ={t['calor']}" + (" + Joule" if t.get("joule") == "1" else "")
                + f", margin {t.get('margen', '0')}")
    if t.get("qss"):
        return "diffused residual (Φ on, λ_TF=2, τ_Q=1, D_Q=0.25)"
    return "?"


def _term_de_job(job: dict) -> dict:
    """The optional term a run used, read from its job.json."""
    # run_preset writes drag/calor/joule/margen/qss; other scripts use the Config
    # names (use_drag, use_heat, gamma_bath, use_joule, …).
    if job.get("drag") or job.get("use_drag"):
        return {"drag": "1", "nu": str(job.get("nu_drag", 0.05))}
    calor = job.get("calor")
    if calor in (None, "", "null") and job.get("use_heat"):
        calor = job.get("gamma_bath", 0.2)
    if calor not in (None, "", "null"):
        return {"calor": str(calor), "joule": "1" if (job.get("joule") or job.get("use_joule")) else "0",
                "margen": str(job.get("margen", job.get("heat_margin", job.get("calor_margen", 0))))}
    if job.get("qss") or job.get("use_Qss"):
        return {"qss": "1"}
    return {}


def _phi_texto(phi: str) -> str:
    if phi.startswith("tf"):
        return f"Thomas–Fermi, λ_TF = {phi.split(',')[1] if ',' in phi else '2'}"
    return {"off": "off (gauge Φ = 0)", "coulomb": "Coulomb projection"}.get(phi, phi)


def _elegir_termino(previo: dict | None = None, phi: str = "off") -> dict | None:
    """One term at a time: with two on, a change cannot be attributed. None = back."""
    print(f"  {_K('0')} none           two-band TDGL only (with Φ as chosen in step 4)")
    print(f"  {_K('1')} drag ν         mixed-gradient coupling between the bands")
    print(f"  {_K('2')} heat θ(r,t)    dissipation heats the sample; bath γ; Joule and contact margin")
    print(f"  {_K('3')} diffused Q     diffusion of the Coulomb-projection residual (turns Φ on)")
    actual = "1" if (previo or {}).get("drag") else "2" if (previo or {}).get("calor") else \
             "3" if (previo or {}).get("qss") else "0"
    k = _ask("term", actual)
    if k is None:
        return None
    if k == "0":
        return {}
    if k == "1":
        nu = _ask("ν, coupling strength (0 to ~0.3)", (previo or {}).get("nu", "0.05"))
        return None if nu is None else {"drag": "1", "nu": nu}
    if k == "2":
        g = _ask("bath γ, how fast the sample cools (0.2 strong · 0.04 weak · 0 none)", (previo or {}).get("calor", "0.08"))
        if g is None:
            return None
        if phi != "off":
            print(_dim("  Joule heating needs Φ off (with Φ, E carries −∇Φ; not verified): left out"))
            j = "n"
        else:
            j = _ask("include Joule heating |E|²? Y/n", "Y")
        if j is None:
            return None
        m = _ask("contact margin: cells next to the injection faces left out of the heat source", "8")
        if m is None:
            return None
        return {"calor": g, "joule": "0" if j.lower() in ("n", "no") else "1", "margen": m}
    if k == "3":
        return {"qss": "1"}
    print("unknown term")
    return None


# --- Run parameters -----------------------------------------------------------
# Each row: key, name, units, what it is (one line), validation type.
PARAMS = [
    ("kappa", "κ", "", "λ/ξ; 2 = soft vortices, many of them · 25 = strongly type II, few vortices", "num>0"),
    ("H", "H", "H_c2", "applied field along z", "num>=0"),
    ("xi", "ξπ/ξσ", "", "coherence length of the π band relative to σ (sets its mass)", "num>0"),
    ("g12", "γ12", "", "interband Josephson coupling (0.001 nearly free · 0.1 locked)", "num>=0"),
    ("T", "bath T", "T_c", "0 = no temperature; >0 is felt by both bands", "0<=num<1"),
    ("J", "current J", "J₀", "from, to, step (J=0 is always added; both polarities run)", "rango"),
    ("malla", "mesh", "cells, ξσ", "Nx×Ny×Nz of the sample and dx (current along x, field along z)", "malla"),
    ("t", "time per J", "t_GL", "minimum if the point is static / if it flows (voltage window)", "tiempos"),
    ("z", "defect in z", "", "through = full thickness · half = lower half · buried = inside", "z"),
    ("cpus", "cores", "", "CPUs to pin the run to (e.g. 0,1); one thread per run", "texto"),
]

PRESETS = [
    ("quick test", "κ=2, box 20×14×4, one current; minutes: check that it runs",
     {"kappa": "2", "H": "0.5", "xi": "1.3", "g12": "0.001", "T": "0", "J": "0.10,0.10,0.10",
      "malla": "20x14x4,0.5", "t": "20,40", "z": "through", "cpus": "8,9"}),
    ("production box, κ=2", "40×24×6, vortex flow; hours",
     {"kappa": "2", "H": "0.03", "xi": "4", "g12": "0.001", "T": "0", "J": "0.40,0.80,0.40",
      "malla": "40x24x6,0.5", "t": "20,200", "z": "through", "cpus": "8,9"}),
    ("production box, κ=25", "40×24×6, onset near J≈0.7; hours to a day",
     {"kappa": "25", "H": "0.03", "xi": "4", "g12": "0.001", "T": "0", "J": "0.68,0.76,0.02",
      "malla": "40x24x6,0.5", "t": "200,1200", "z": "through", "cpus": "0,1"}),
    ("custom", "your own mesh, κ, field, currents and times (asked in step 3)",
     {"kappa": "2", "H": "0.03", "xi": "4", "g12": "0.001", "T": "0", "J": "0.40,0.80,0.40",
      "malla": "40x24x6,0.5", "t": "20,200", "z": "through", "cpus": "8,9"}),
]


def _lista_J(rango: str) -> list[float]:
    a, b, d = (float(x) for x in rango.split(","))
    if d <= 0 or b < a:
        raise ValueError("J range: to >= from and step > 0")
    n = int(round((b - a) / d)) + 1
    if n > 60:
        raise ValueError("more than 60 current values")
    return [round(a + i * d, 6) for i in range(n)]


def _valida(clave: str, v: str) -> str:
    """Returns the normalised value or raises ValueError with the reason."""
    tipo = dict((k, t) for k, _, _, _, t in PARAMS)[clave]
    v = v.strip().replace(" ", "")
    if tipo.startswith("num") or tipo.startswith("0<="):
        x = float(v)
        if tipo == "num>0" and x <= 0:
            raise ValueError("must be > 0")
        if tipo == "num>=0" and x < 0:
            raise ValueError("must be >= 0")
        if tipo == "0<=num<1" and not 0 <= x < 1:
            raise ValueError("between 0 and 1")
        return v
    if tipo == "rango":
        partes = v.split(",")
        if len(partes) == 1:
            partes = [partes[0]] * 2 + ["0.1"]
        v = ",".join(partes)
        _lista_J(v)
        return v
    if tipo == "malla":
        tam, dx = v.lower().replace("×", "x").split(",")
        nx, ny, nz = (int(s) for s in tam.split("x"))
        if min(nx, ny, nz) < 4 or float(dx) <= 0:
            raise ValueError("each side >= 4 cells and dx > 0")
        return f"{nx}x{ny}x{nz},{float(dx):g}"
    if tipo == "tiempos":
        a, b = (float(s) for s in v.split(","))
        if a <= 0 or b < a:
            raise ValueError("static > 0 and flowing >= static")
        return f"{a:g},{b:g}"
    if tipo == "z":
        if v not in ("through", "half", "buried"):
            raise ValueError("through, half or buried")
        return v
    return v


def _mostrar_valor(clave: str, v: str) -> str:
    if clave == "J":
        js = _lista_J(v)
        return f"{js[0]:g} … {js[-1]:g}" + (f"  ({len(js)} values)" if len(js) > 1 else "")
    if clave == "malla":
        tam, dx = v.split(",")
        nx, ny, nz = (int(s) for s in tam.split("x"))
        return f"{nx}×{ny}×{nz}, dx={dx}  ({nx*float(dx):g}×{ny*float(dx):g}×{nz*float(dx):g} ξσ)"
    if clave == "t":
        a, b = v.split(",")
        return f"{a} / {b}"
    return v


def _tabla_parametros(par: dict) -> bool:
    """Editable table. True = continue, False = back."""
    while True:
        lin = []
        for i, (k, nombre, unid, que, _) in enumerate(PARAMS, 1):
            u = f" [{unid}]" if unid else ""
            lin.append(f"{_K(str(i))} {nombre + u:<20} {_col('1', _mostrar_valor(k, par[k])):<28}  {_dim(que)}")
        lin.append("")
        lin.append(_dim("number = change that parameter  ·  Enter = continue  ·  b = previous step"))
        caja("parameters", lin, color="36")
        k = input("> ").strip().lower()
        if k == "":
            return True
        if k in ("b", "q", "-"):
            return False
        if not (k.isdigit() and 1 <= int(k) <= len(PARAMS)):
            continue
        clave, nombre, unid, que, tipo = PARAMS[int(k) - 1]
        ejemplo = {"rango": "from,to,step  (e.g. 0.50,0.80,0.02)",
                   "malla": "NxxNyxNz,dx  (e.g. 40x24x6,0.5)",
                   "tiempos": "static,flowing  (e.g. 200,1200)"}.get(tipo, "")
        print(_dim(f"  {nombre}: {que}" + (f"  · format {ejemplo}" if ejemplo else "")))
        v = input(f"  {nombre} [{par[clave]}]> ").strip()
        if not v:
            continue
        try:
            par[clave] = _valida(clave, v)
        except (ValueError, KeyError) as e:
            print(_col("31", f"  not valid: {e}"))
            time.sleep(1.2)


def _paso(n: int, total: int, titulo: str) -> None:
    print()
    print(_col("1;36", f"── Launch · step {n}/{total} · {titulo} ") + _dim("── b = previous step"))


# --- Launch wizard: a sequence of steps; b goes back one step -------------------
def _w_inicio(s: dict) -> str:
    _paso(1, 6, "starting point")
    for i, (nombre, desc, _) in enumerate(PRESETS, 1):
        print(f"  {_K(str(i))} {nombre:<24} {_dim(desc)}")
    print(_dim("  It only fills the initial values: mesh (Nx×Ny×Nz, dx), κ, H, currents and times"))
    print(_dim("  are all editable in step 3."))
    k = _ask("starting point", s.get("preset", "1"))
    if k is None:
        return "back"
    if not (k.isdigit() and 1 <= int(k) <= len(PRESETS)):
        return "again"
    if k != s.get("preset") or "par" not in s:
        s["par"] = dict(PRESETS[int(k) - 1][2])
    s["preset"] = k
    return "next"


def _w_geometria(s: dict) -> str:
    _paso(2, 6, "geometry")
    for i, (n, nota) in enumerate(CATALOG_VISIBLE, 1):
        print(f"  {_K(str(i))} {n:<10} {_dim(nota)}")
    print(f"  {_K('e')} draw a new shape in the editor (opens the browser), then use it")
    print(f"  {_K('f')} a saved shape from julia/shapes/ (yours, or a catalogue template)")
    print(_dim("  Any other catalogue name can be typed (g → p lists them)."))
    gsel = _ask("geometry", s.get("gsel", "2"))
    if gsel is None:
        return "back"
    forma = ""
    if gsel.lower() == "e":
        if not EDITOR.exists():
            print("editor not found:", EDITOR)
            return "again"
        print(f"  editor: {_editor_url()}")
        print(_dim("  Draw the shape and press 'save' in the browser (it goes to julia/shapes/)."))
        if _ask("Enter when saved", "") is None:
            return "again"
        r = _elegir_forma()
        if r is None:
            return "again"
        geom, forma = r
    elif gsel.lower() == "f":
        r = _elegir_forma()
        if r is None:
            return "again"
        geom, forma = r
    elif gsel.isdigit() and 1 <= int(gsel) <= len(CATALOG_VISIBLE):
        geom = CATALOG_VISIBLE[int(gsel) - 1][0]
    else:
        geom = gsel.upper()
    if not forma and not any(geom == n or geom == n + "_REF" for n, _ in CATALOG):
        print("unknown geometry")
        return "again"
    par = s["par"]
    if (forma or geom.replace("_REF", "") in WIDE) and par["malla"].startswith("20x14x4"):
        par["malla"] = "40x24x6,0.5"
        print(_dim("  this shape needs the 40×24×6 box: mesh adjusted"))
    rot = ""
    if geom.replace("_REF", "") in ("FUNNEL", "STAGGER", "HEX", "HEX_STAG"):
        rot = _ask("rotation of the polygon in degrees (Enter = catalogue value)", s.get("rot", ""))
        if rot is None:
            return "again"
    s.update(gsel=gsel, geom=geom, forma=forma, rot=rot)
    return "next"


def _w_parametros(s: dict) -> str:
    _paso(3, 6, "parameters")
    return "next" if _tabla_parametros(s["par"]) else "back"


PHI = [
    ("off", "gauge Φ = 0 (default, fastest). |Ψ|, B and the Josephson voltage are gauge invariant,"),
    ("coulomb", "Coulomb projection: solves Poisson for Φ each step. Same |Ψ|, B and V as off; it also"),
    ("tf", "Thomas–Fermi screening length λ_TF: Φ is screened; this changes the physics"),
]
PHI_2 = ["so for them nothing is lost", "gives Φ and the potential drop V_Φ (gauge-dependent)", ""]


def _w_phi(s: dict) -> str:
    _paso(4, 6, "scalar potential Φ")
    for i, ((n, d), d2) in enumerate(zip(PHI, PHI_2)):
        print(f"  {_K(str(i))} {n:<8} {_dim(d)}")
        if d2:
            print(f"  {'':<12}{_dim(d2)}")
    previo = s["par"].get("phi", "off")
    actual = str([n for n, _ in PHI].index(previo.split(",")[0]))
    k = _ask("Φ", actual)
    if k is None:
        return "back"
    if not (k.isdigit() and 0 <= int(k) < len(PHI)):
        return "again"
    modo = PHI[int(k)][0]
    if modo == "tf":
        l = _ask("λ_TF in ξσ (Debye-like screening length)", previo.split(",")[1] if "," in previo else "2")
        if l is None:
            return "again"
        try:
            if float(l) <= 0:
                raise ValueError
        except ValueError:
            print(_col("31", "  λ_TF must be a number > 0"))
            return "again"
        modo = f"tf,{l}"
    s["par"]["phi"] = modo
    return "next"


def _w_termino(s: dict) -> str:
    _paso(5, 6, "optional term (one at a time)")
    t = _elegir_termino(s.get("term"), phi=s["par"].get("phi", "off"))
    if t is None:
        return "back"
    if t.get("qss") and not s["par"].get("phi", "off").startswith("tf"):
        s["par"]["phi"] = "tf,2"
        print(_dim("  diffused Q lives in the Φ step: Φ set to Thomas–Fermi, λ_TF = 2"))
    s["term"] = t
    return "next"


def _w_resumen(s: dict) -> str:
    _paso(6, 6, "summary")
    geom, forma, rot, par = s["geom"], s["forma"], s["rot"], s["par"]
    lin = [f"geometry    {geom}" + (f"   ← {Path(forma).name}" if forma else "") + (f"   rot {rot}°" if rot else "")]
    for k, nombre, unid, _, _ in PARAMS:
        lin.append(f"{nombre:<12}{_mostrar_valor(k, par[k])}" + (f"  {unid}" if unid and k not in ('malla', 'J') else ""))
    lin += [f"Φ           {_phi_texto(par.get('phi', 'off'))}",
            f"term        {_term_texto(s['term'])}", "writes to   julia/runs/"]
    caja("about to launch", lin, color="32")
    k = _ask("launch? y/N", "N")
    if k is None:
        return "back"
    if k.lower() not in ("y", "s", "si", "yes"):
        print("cancelled")
        return "cancel"
    _spawn(geom, par, rotate=rot, forma=forma, terminos=s["term"], restart=False, rundir=None)
    _volver()
    return "next"


def _launch_wizard() -> None:
    pasos = [_w_inicio, _w_geometria, _w_parametros, _w_phi, _w_termino, _w_resumen]
    s: dict = {}
    i = 0
    while 0 <= i < len(pasos):
        r = pasos[i](s)
        if r == "next":
            i += 1
        elif r == "back":
            i -= 1
        elif r == "cancel":
            return


def _spawn(geom: str, par: dict, *, rotate: str = "", forma: str = "", terminos: dict | None = None,
           restart: bool = False, rundir: Path | None = None, name: str = "", mMax: str = "") -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    tam, dx = par["malla"].split(",")
    nx, ny, nz = tam.split("x")
    tmin, tres = par["t"].split(",")
    # J_lista (resume) = the exact list of the job.json; otherwise the wizard range.
    js = par.get("J_lista") or ",".join(f"{j:g}" for j in _lista_J(par["J"]))
    cmd = [
        str(LANZAR), str(par["cpus"]),
        "julia", f"--project={JULIA}", str(PRESET),
        "--geom", geom, "--kappa", str(par["kappa"]), "--H", str(par["H"]),
        "--xi-ratio", str(par["xi"]), "--gamma12", str(par["g12"]), "--J", js,
        "--Nx", nx, "--Ny", ny, "--Nz", nz, "--dx", dx,
        "--tmin", tmin, "--tminres", tres, "--z", par["z"], "--cpus", str(par["cpus"]),
    ]
    if mMax:
        cmd += ["--mMax", str(mMax)]
    if float(par.get("T", 0) or 0) > 0:
        cmd += ["--T", str(par["T"])]
    phi = par.get("phi", "off")
    if phi != "off":
        cmd += ["--phi", phi.split(",")[0]]
        if "," in phi:
            cmd += ["--lambda-tf", phi.split(",")[1]]
    if rotate:
        cmd += ["--rotate-deg", str(rotate)]
    if forma:
        cmd += ["--shape", str(forma)]
    t = terminos or {}
    if t.get("drag"):
        cmd += ["--drag", "1", "--nu", str(t.get("nu", "0.05"))]
    if t.get("calor"):
        cmd += ["--heat", str(t["calor"]), "--joule", str(t.get("joule", "0")),
                "--margin", str(t.get("margen", "0"))]
    if t.get("qss"):
        cmd += ["--qss", "1"]
    if rundir is not None:
        cmd += ["--dir", str(rundir)]
    if name:
        cmd += ["--name", name]
    if restart:
        cmd.append("--restart")
    log = (rundir if rundir is not None else RUNS) / "TUI_launch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(BASE))
    print("launched. Closing the menu does not stop the run. log:", log)
    return log


def _par_de_job(job: dict) -> dict:
    """Wizard parameters rebuilt from a run_preset job.json."""
    jl = job.get("J_list", [0.10])
    jl = jl if isinstance(jl, list) else [float(jl)]
    js = sorted(abs(float(x)) for x in jl if abs(float(x)) > 0) or [0.10]
    paso = min((b - a for a, b in zip(js, js[1:])), default=0.1) or 0.1
    return {
        "kappa": str(job.get("kappa", 2)), "H": str(job.get("H", 0.5)),
        "xi": str(job.get("xi_ratio", 1.3)), "g12": str(job.get("gamma12", 0.001)),
        "T": str(job.get("T_red", 0) or 0),
        "J": f"{js[0]:g},{js[-1]:g},{paso:g}",
        "malla": f"{job.get('Nx', 20)}x{job.get('Ny', 14)}x{job.get('Nz', 4)},{job.get('dx', 0.5)}",
        "t": f"{job.get('t_min_GL', 20)},{job.get('t_min_GL_res', 40)}",
        "z": str(job.get("z_mode", "through")), "cpus": str(job.get("cpus", "8,9")),
        "J_lista": ",".join(f"{float(x):g}" for x in jl),
        "phi": (f"tf,{job.get('lambda_TF', 2)}" if job.get("phi") == "tf"
                else str(job.get("phi", "off")) if not job.get("qss") else "off"),
    }


def _pick(rows: list[dict], prompt: str) -> dict | None:
    """Numbered list in a box; Enter or b goes back without choosing."""
    titulo = prompt.split("#")[0].strip() or "choose"
    if not rows:
        caja(titulo, [_dim("no runs for this")], color="33")
        _volver()
        return None
    lin = [f"{_K(str(i))} {r['tag']}   {_estado(r)}   {_ultimo(r)}" for i, r in enumerate(rows, 1)]
    lin.append(_dim("number = choose  ·  Enter or b = back"))
    caja(titulo, lin, color="33")
    k = input(f"{titulo} # > ").strip()
    if not k.isdigit() or not (1 <= int(k) <= len(rows)):
        return None
    return rows[int(k) - 1]


def _stop_job(r: dict) -> None:
    d = r["dir"]
    print()
    print("Stop a run. Checkpoint = last CLOSED point.")
    print("  1  clean (STOP.txt): finishes the current point, saves and exits  ← preferred")
    print("  2  immediate (SIGTERM): loses the current point")
    print("  N  cancel")
    if r.get("frozen"):
        print("  Article run: only option 2 applies, and it should not be needed.")
    k = input("stop> ").strip().lower()
    if k == "1":
        if r.get("frozen"):
            print("STOP.txt is not written into an article run.")
            _volver()
            return
        (d / "STOP.txt").write_text("stop requested by tdgl_tui\n")
        print("STOP.txt written in", d)
        print("when the point closes, STATUS shows the last point and [u] resumes.")
    elif k == "2":
        if not r["on"] or not r["pid"].isdigit():
            print("no live pid")
        else:
            extra = _ask("SIGTERM to pid %s? y/N" % r["pid"], "N")
            if extra and extra.lower() in ("y", "s", "si", "yes"):
                os.kill(int(r["pid"]), signal.SIGTERM)
                print("SIGTERM sent")
    _volver()


def _resume_job(r: dict) -> None:
    job = r.get("job") or _job_json(r["dir"])
    if not job or job.get("project") not in ("julia", "juliatdgl_v3", "juliatdgl_v2") or r.get("frozen"):
        print("Only runs launched from this menu (with their job.json) are resumed here.")
        print("Others: same script, restart_pos/neg=true, same out_dir.")
        _volver()
        return
    if r["on"]:
        print("still running; not relaunching on top of it.")
        _volver()
        return
    ck = list(r["dir"].glob("**/restart/state_J*.mat"))
    print(f"resume {r['tag']}  checkpoints={len(ck)}  last J={r['last']}")
    k = _ask("resume? y/N", "N")
    if not k or k.lower() not in ("y", "s", "si", "yes"):
        return
    _spawn(job.get("geom", "WEDGE"), _par_de_job(job), restart=True, rundir=r["dir"],
           rotate="" if job.get("rotate_deg") in (None, "", "null") else str(job.get("rotate_deg")),
           forma=str(r["dir"] / job["geom_file"]) if job.get("geom_file") else "",
           terminos=_term_de_job(job), mMax=str(job.get("mMax", "")))
    _volver()


def _help() -> None:
    caja("help", [
        f"{_K('l')} launch: starting point → geometry → parameters → Φ → optional term → summary",
        f"{_K('s')} stop a live run: STOP.txt (closes the current point) or SIGTERM (loses it)",
        f"{_K('u')} resume from the last closed point (checkpoint)",
        f"{_K('d')} move a finished run to trash_/tui_jobs/ (nothing is deleted)",
        f"{_K('a')} live only / all      {_K('r')} refresh      {_K('o')} open folder",
        f"{_K('f')} 2D figures (V(J) and |Ψ|² maps) in RUN/figures/; 3D: vtk_from_checkpoint → ParaView",
        f"{_K('g')} geometries: catalogue and shape editor in the browser",
        f"{_K('v')} results: Ic brackets with the verdict rule (crossings per line <1 silent, ≥10",
        "  dissipating), η with its range, undecided rungs, σ_V where both polarities flow",
        _dim("While this menu is open a desktop notification is sent when a run finishes;"),
        _dim("without the menu: python3 julia/scripts/tdgl_tui.py --watch   (or --results)"),
        f"{_dim('number')} details of a run: current history, V(J) and actions",
        "",
        _dim("In every menu: Enter = default value, b = previous step, Ctrl+C = back to this list."),
    ], color="33")
    _volver()


def _seguro(fn, *args) -> None:
    """Ctrl+C inside a menu returns to the list; an error is shown and the menu stays up."""
    try:
        fn(*args)
    except KeyboardInterrupt:
        print("\n" + _dim("cancelled: back to the list"))
        time.sleep(0.6)
    except EOFError:
        pass
    except Exception as e:      # noqa: BLE001 — the menu must not die on an odd run
        caja("error (the menu keeps running; Julia was not touched)", [f"{type(e).__name__}: {e}"], color="31")
        try:
            _volver()
        except (KeyboardInterrupt, EOFError):
            pass


def _figuras(r: dict) -> None:
    script = BASE / "julia" / "scripts" / "plot_summary.py"
    print("2D figures with matplotlib (V(J) and |Ψ|² maps) in", r["dir"] / "figures")
    subprocess.call(["python3", str(script), str(r["dir"])])
    _volver()


def _accion(key: str, rows: list[dict], show: list[dict], only_on: bool) -> None:
    """One key of the main screen. Runs inside _seguro."""
    if key in ("g", "G"):
        _geom_menu()
    elif key in ("l", "L"):
        _launch_wizard()
    elif key in ("h", "H", "?"):
        _help()
    elif key in ("s", "S"):
        r = _pick([r for r in rows if r["on"]], "stop # > ")
        r and _stop_job(r)
    elif key in ("u", "U"):
        r = _pick([r for r in rows if not r["on"] and (r["dir"] / "job.json").exists()], "resume # > ")
        r and _resume_job(r)
    elif key in ("d", "D"):
        cand = [r for r in (collect() if only_on else rows) if not r["on"] and not r.get("frozen")]
        r = _pick(cand, "to trash_ # > ")
        r and _delete_job(r)
    elif key in ("o", "O"):
        r = _pick(show if show else rows, "open # > ")
        r and _open_dir(r["dir"])
    elif key in ("f", "F"):
        r = _pick(show if show else collect(), "figures # > ")
        r and _figuras(r)
    elif key in ("v", "V"):
        _resultados(show if show else collect())
    elif key.isdigit() and 1 <= int(key) <= len(show):
        detail(show[int(key) - 1])


def interactive() -> None:
    only_on = True
    threading.Thread(target=_vigia, daemon=True).start()
    while True:
        os.system("clear")
        if TERMINADAS:
            caja("finished while the menu was open", TERMINADAS[-5:], color="32")
        try:
            rows = collect()
            table(rows, only_on=only_on)
        except Exception as e:   # noqa: BLE001 — a broken job.json must not take the menu down
            rows = []
            caja("error reading the runs", [f"{type(e).__name__}: {e}"], color="31")
        show = [r for r in rows if r["on"]] if only_on else rows
        caja("keys", [
            f"{_K('l')} launch   {_K('s')} stop   {_K('u')} resume   {_K('d')} to trash_   "
            + _dim("number = details"),
            f"{_K('a')} live/all   {_K('r')} refresh   {_K('o')} folder   {_K('f')} figures   {_K('v')} results",
            f"{_K('g')} geometries and shape editor   {_K('h')} help   {_K('q')} quit (runs continue)",
            _dim("In every menu: Enter = default, b = previous step, Ctrl+C = back here."),
        ], color="33")
        try:
            key = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if key in ("q", "Q"):
            return
        if key in ("a", "A"):
            only_on = not only_on
        elif key not in ("r", "R", ""):
            _seguro(_accion, key, rows, show, only_on)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true", help="no menu: notify on the desktop when a run finishes")
    ap.add_argument("--results", action="store_true", help="print the quick results of every run and exit")
    ap.add_argument("--on", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("rundir", nargs="?")
    args, _ = ap.parse_known_args()
    if args.rundir:
        p = Path(args.rundir)
        if not p.is_absolute():
            p = BASE / p
        rows = [r for r in collect() if r["dir"] == p.resolve() or str(r["dir"]).endswith(args.rundir)]
        if not rows:
            print("no run found at", p)
            return
        detail(rows[0])
        return
    if args.once:
        table(collect(), only_on=not args.all)
        return
    if args.results:
        vistos = set()
        for r in collect():
            res = [x for x in quick_result(r["dir"]) if x not in vistos]
            vistos.update(res)
            if res:
                print(r["tag"] + ("  (live)" if r["on"] else ""))
                for x in res:
                    print("  " + x)
        return
    if args.watch:
        print("watching the runs; a desktop notification is sent when one finishes (Ctrl+C to stop)")
        try:
            _vigia()
        except KeyboardInterrupt:
            pass
        return
    interactive()


if __name__ == "__main__":
    main()
