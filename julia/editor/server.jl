# Local server for the shape editor (127.0.0.1 only, standard library only; code in julia/tui).
#   julia julia/editor/server.jl                 # opens the browser
#   julia julia/editor/server.jl --port 8765 --no-browser --idle 30   (minutes without requests; 0 = never)
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "tui"))
using TDGLTUI
TDGLTUI.editor_main(ARGS)
