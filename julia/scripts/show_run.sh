#!/usr/bin/env bash
# Look at a run WITHOUT touching it. Safe to close and reopen (Julia keeps going).
#   ./julia/scripts/show_run.sh julia/runs/my_run
#   watch -n 15 ./julia/scripts/show_run.sh julia/runs/my_run
# Without arguments: one-shot list of all runs (the terminal menu's job table).
set -euo pipefail
if [[ $# -lt 1 ]]; then
  exec python3 "$(cd "$(dirname "$0")" && pwd)/tdgl_tui.py" --once --on
fi
ROOT="$1"
echo "=== $ROOT  $(date '+%H:%M:%S') ==="
if [[ -f "$ROOT/STATUS.txt" ]]; then
  echo -n "STATUS: "; cat "$ROOT/STATUS.txt"
else
  echo "STATUS: (no STATUS.txt)"
fi
if [[ -f "$ROOT/julia.pid" ]]; then
  pid=$(cat "$ROOT/julia.pid")
  if ps -p "$pid" >/dev/null 2>&1; then
    ps -o pid,etime,pcpu,psr= -p "$pid" | sed 's/^/  /'
  else
    echo "  pid $pid is gone (finished or died)"
  fi
fi
snap=$(find "$ROOT" -name '*psi_s.pgm' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1)
if [[ -n "${snap:-}" ]]; then
  echo -n "latest PGM: "
  date -d "@${snap%% *}" '+%H:%M:%S' 2>/dev/null || echo "${snap%% *}"
  echo "  ${snap#* }"
  echo "  (a PGM is written when a current point CLOSES, not every cycle)"
else
  echo "no PGM yet; with a long minimum time the first point can take hours"
fi
if [[ -f "$ROOT/log.txt" ]]; then
  echo "--- log (tail) ---"
  tail -n 8 "$ROOT/log.txt"
fi
echo "reopen: watch -n 15 $0 $ROOT"
echo "this does not stop the run; Ctrl-C only leaves watch."
