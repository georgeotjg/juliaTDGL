# One run from a file:  OPENBLAS_NUM_THREADS=1 julia --project=tdgl2 tdgl2/scripts/run.jl RUN.toml [--dir OUT] [--resume] [--polarity +|-]
using TDGL, LinearAlgebra, Printf
BLAS.set_num_threads(1)
if isempty(ARGS) || ARGS[1] in ("-h", "--help")
    println("usage: julia --project=tdgl2 tdgl2/scripts/run.jl RUN.toml [--dir OUT] [--resume] [--polarity +|-]\n",
            "Writes run.toml, output/rungs_J+.csv, output/rungs_J-.csv, output/summary.json and restart/.\n",
            "  --resume       continue from the checkpoints in the folder (STOP.txt there stops a run after the current point)\n",
            "  --polarity +   one polarity only (two processes into the same folder run both in parallel; the second to end\n",
            "                 writes the summary)")
    exit(0)
end
i = findfirst(==("--dir"), ARGS)
out = i === nothing ? nothing : ARGS[i + 1]
k = findfirst(==("--polarity"), ARGS)
pol = k === nothing ? :both : ARGS[k + 1] in ("+", "pos", "J+") ? :pos : ARGS[k + 1] in ("-", "neg", "J-") ? :neg :
      error("--polarity is + or -")
r = run_config(ARGS[1]; out_dir=out, resume="--resume" in ARGS, polarity=pol)
if r.brackets === nothing
    println("this polarity ended; the other is still running (or stopped): its process writes the summary")
    exit(0)
end
b = r.brackets
f(x) = isnan(x) ? "?" : isinf(x) ? "∞" : @sprintf("%.4g", x)
@printf("Ic+ in (%s, %s]   |Ic-| in (%s, %s]   eta = %s %%  [%s, %s]%s\n", f(b.pos.lo), f(b.pos.hi),
        f(b.neg.lo), f(b.neg.hi), f(b.eta[1]), f(b.eta[2]), f(b.eta[3]),
        (b.pos.ok && b.neg.ok) ? "" : "  (undecided rungs inside a bracket: re-measure them)")
# with t_confirm_GL the onset quoted is the one of the delay (the silent end followed that long)
if r.summary["protocol"]["t_confirm_GL"] > 0
    d = b.delay
    @printf("by the delay: Ic+ in (%s, %s]   |Ic-| in (%s, %s]   eta = %s %%  [%s, %s]\n", f(d.pos.lo), f(d.pos.hi),
            f(d.neg.lo), f(d.neg.hi), f(d.eta[1]), f(d.eta[2]), f(d.eta[3]))
end
