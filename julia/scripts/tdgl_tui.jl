# Terminal menu for TDGL runs (julia/tui, standard library only). Closing it never stops Julia.
#   julia julia/scripts/tdgl_tui.jl [--help | --once [--all] | --results | --watch | --dirs | RUNDIR]
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "tui"))
using TDGLTUI
TDGLTUI.main(ARGS)
