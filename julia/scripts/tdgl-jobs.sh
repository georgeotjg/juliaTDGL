#!/usr/bin/env bash
JL=$(command -v julia || echo "$HOME/.juliaup/bin/julia")
exec "$JL" --startup-file=no "$(cd "$(dirname "$0")" && pwd)/tdgl_tui.jl" --once "$@"
