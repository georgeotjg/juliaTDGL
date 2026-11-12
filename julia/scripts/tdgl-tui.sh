#!/usr/bin/env bash
JL=$(command -v julia || echo "$HOME/.juliaup/bin/julia")
# the menu takes a second to start (several after a Julia update, while it compiles): say so on an empty window
[ $# -eq 0 ] && [ -t 1 ] && printf '\e]0;TDGL\a\e[2mTDGL: loading the menu…\e[0m\n'
exec "$JL" --startup-file=no "$(cd "$(dirname "$0")" && pwd)/tdgl_tui.jl" "$@"
