#!/usr/bin/env bash
exec python3 "$(cd "$(dirname "$0")" && pwd)/tdgl_tui.py" --once --on "$@"
