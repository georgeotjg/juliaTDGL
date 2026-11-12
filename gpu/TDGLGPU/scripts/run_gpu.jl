# Run TDGL 2.0 with an explicit accelerator backend; measurements/checkpoints stay on the host.
using TDGL, TDGLGPU, KernelAbstractions, LinearAlgebra, Printf
BLAS.set_num_threads(1)
const USAGE = "usage: julia --project=gpu/TDGLGPU gpu/TDGLGPU/scripts/run_gpu.jl RUN.toml [--dir OUT] [--resume] [--polarity +|-] [--cpu-backend] [--require-gpu]"
if isempty(ARGS) || ARGS[1] in ("-h", "--help")
    println(USAGE)
    println("--require-gpu requires functional CUDA and rejects unsupported configurations instead of CPU fallback.")
    exit(0)
end
function parse_options(args)
    opts = Dict{String,String}(); flags = Set{String}()
    i = 2
    while i <= length(args)
        flag = args[i]
        if flag in ("--dir", "--polarity")
            i < length(args) && !startswith(args[i+1], "--") || error("$flag requires a value")
            opts[flag] = args[i+1]; i += 2
        elseif flag in ("--resume", "--cpu-backend", "--require-gpu")
            push!(flags, flag); i += 1
        else
            error("unknown option: $flag")
        end
    end
    opts, flags
end
opts, flags = parse_options(ARGS)
require_gpu = "--require-gpu" in flags
require_gpu && "--cpu-backend" in flags && error("--require-gpu cannot be combined with --cpu-backend")
polarg = get(opts, "--polarity", "both")
polarg in ("both", "+", "pos", "J+", "-", "neg", "J-") || error("invalid --polarity: $polarg")
pol = polarg == "both" ? :both : polarg in ("+", "pos", "J+") ? :pos : :neg
if "--cpu-backend" in flags
    backend = KernelAbstractions.CPU()
else
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional; use --cpu-backend for an explicit KA-CPU check")
    backend = CUDABackend()
    println("CUDA device: ", CUDA.name(CUDA.device()))
end
TDGLGPU.enable!(backend; fallback=require_gpu ? :error : :cpu)
t0 = time()
r = try
    run_config(ARGS[1]; out_dir=get(opts, "--dir", nothing), resume="--resume" in flags, polarity=pol)
finally
    TDGLGPU.disable!()
end
stats = execution_stats()
@printf("wall %.1f min; backend=%s, backend_steps=%d, cpu_fallback_steps=%d\n",
        (time() - t0) / 60, stats.backend, stats.backend_steps, stats.cpu_fallback_steps)
if r.brackets === nothing
    println("this polarity ended; the other is still running (or stopped): its process writes the summary")
    exit(0)
end
b = r.brackets
f(x) = isnan(x) ? "?" : isinf(x) ? "∞" : @sprintf("%.4g", x)
@printf("Ic+ in (%s, %s]   |Ic-| in (%s, %s]   eta = %s %%  [%s, %s]%s\n", f(b.pos.lo), f(b.pos.hi),
        f(b.neg.lo), f(b.neg.hi), f(b.eta[1]), f(b.eta[2]), f(b.eta[3]),
        (b.pos.ok && b.neg.ok) ? "" : "  (undecided rungs inside a bracket: re-measure them)")
if r.summary["protocol"]["t_confirm_GL"] > 0
    d = b.delay
    @printf("by the delay: Ic+ in (%s, %s]   |Ic-| in (%s, %s]   eta = %s %%  [%s, %s]\n", f(d.pos.lo), f(d.pos.hi),
            f(d.neg.lo), f(d.neg.hi), f(d.eta[1]), f(d.eta[2]), f(d.eta[3]))
end
