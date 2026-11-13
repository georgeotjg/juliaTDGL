# Terminal menu for TDGL runs (tdgl/tui, standard library only). Closing it never stops Julia.
#   julia tdgl/scripts/tdgl_tui.jl [--help | --once [--all] | --results | --watch | --dirs | RUNDIR]
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "tui"))
using TDGLTUI
TDGLTUI.main(ARGS)
