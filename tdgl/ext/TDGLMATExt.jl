# Optional reader for binary MATLAB v7 files. The checkpoints of this code are JLD2 (or Octave text, which
# TDGL reads without help); only old binary .mat files need MAT.jl. Loaded when MAT is installed and imported.
module TDGLMATExt
using TDGL, MAT
TDGL._read_mat_binary(ruta::AbstractString) = MAT.matread(ruta)
end
